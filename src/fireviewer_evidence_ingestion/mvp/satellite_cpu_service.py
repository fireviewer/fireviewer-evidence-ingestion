"""Authenticated Azure CPU service for incident-day satellite preparation."""

from __future__ import annotations

import json
import os
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

from pydantic import Field, SecretStr, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, StrictModel
from fireviewer_evidence_ingestion.mvp.satellite_cpu import (
    AzureFederatedSageMakerAsyncProvider,
    AzureManagedIdentityTokenProvider,
    BackendSatelliteBandFetcher,
    CanonicalPrithviRasterBuilder,
    SageMakerAsyncConfig,
    SatelliteCpuError,
    SatelliteCpuWorker,
)
from fireviewer_evidence_ingestion.mvp.satellite_observations import (
    CdseObservationS3Config,
    CdseS3ObservationAssetReader,
    SatelliteObservationCpuWorker,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    AzureBackendEventEvidenceConfig,
    AzureBackendIncidentDayEvidenceAdapter,
    BackendEventEvidenceError,
    BackendIncidentDaySatelliteAnalysisPublisher,
    BackendIncidentDaySatelliteObservationPublisher,
)

_MAX_REQUEST_BYTES = 4 * 1_024


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


class SatelliteCpuRequest(StrictModel):
    analysis_id: SafeIdentifierV2


class SatelliteObservationCpuRequest(StrictModel):
    analysis_id: SafeIdentifierV2
    artifact_revision_id: SafeIdentifierV2


class SatelliteCpuServiceSettings(StrictModel):
    host: str = Field(default="0.0.0.0", min_length=1, max_length=255)  # noqa: S104
    port: int = Field(default=8080, ge=1, le=65_535)
    worker_token: SecretStr = Field(min_length=32, max_length=4_096)
    paid_invocation_enabled: bool = False
    managed_identity_client_id: str | None = Field(default=None, min_length=36, max_length=36)
    aws_oidc_audience: str | None = Field(default=None, min_length=8, max_length=512)
    sagemaker: SageMakerAsyncConfig | None = None
    maximum_raster_pixels: int = Field(default=100_000_000, ge=256, le=250_000_000)
    cdse: CdseObservationS3Config | None = None
    backend: AzureBackendEventEvidenceConfig

    @model_validator(mode="after")
    def validate_paid_provider(self) -> SatelliteCpuServiceSettings:
        if self.paid_invocation_enabled and (
            self.managed_identity_client_id is None
            or self.aws_oidc_audience is None
            or self.sagemaker is None
        ):
            raise ValueError("paid SageMaker invocation configuration is incomplete")
        return self

    @classmethod
    def from_env(cls) -> SatelliteCpuServiceSettings:
        enabled = _env_bool("FIREVIEWER_SAGEMAKER_GEO_INVOCATION_ENABLED", False)
        sagemaker = None
        if enabled:
            sagemaker = SageMakerAsyncConfig(
                region_name=os.environ["AWS_REGION"],
                role_arn=os.environ["FIREVIEWER_SAGEMAKER_GEO_ROLE_ARN"],
                endpoint_name=os.environ["FIREVIEWER_SAGEMAKER_GEO_ENDPOINT_NAME"],
                bucket_name=os.environ["FIREVIEWER_SAGEMAKER_GEO_BUCKET"],
                input_prefix=os.getenv(
                    "FIREVIEWER_SAGEMAKER_GEO_INPUT_PREFIX",
                    "async/input/production",
                ),
                poll_seconds=float(os.getenv("FIREVIEWER_SAGEMAKER_GEO_POLL_SECONDS", "10")),
                maximum_wait_seconds=int(
                    os.getenv("FIREVIEWER_SAGEMAKER_GEO_MAX_WAIT_SECONDS", "180")
                ),
            )
        cdse_access_key = os.getenv("FIREVIEWER_CDSE_S3_ACCESS_KEY")
        cdse_secret_key = os.getenv("FIREVIEWER_CDSE_S3_SECRET_KEY")
        openeo_enabled = _env_bool("FIREVIEWER_CDSE_OPENEO_INVOCATION_ENABLED", False)
        openeo_access_token = os.getenv("FIREVIEWER_CDSE_OPENEO_ACCESS_TOKEN")
        openeo_credit_ceiling = float(
            os.getenv("FIREVIEWER_CDSE_OPENEO_MAXIMUM_AUTHORIZED_CREDITS", "0")
        )
        if bool(cdse_access_key) != bool(cdse_secret_key):
            raise ValueError("CDSE S3 credentials must be configured together")
        if openeo_enabled and (not cdse_access_key or not cdse_secret_key):
            raise ValueError("openEO observations require the configured CDSE satellite reader")
        cdse = (
            CdseObservationS3Config(
                access_key=SecretStr(cdse_access_key),
                secret_key=SecretStr(cdse_secret_key),
                maximum_window_pixels=int(
                    os.getenv(
                        "FIREVIEWER_SATELLITE_MAX_OBSERVATION_WINDOW_PIXELS",
                        "4000000",
                    )
                ),
                sentinel2_maximum_download_bytes=int(
                    os.getenv(
                        "FIREVIEWER_SENTINEL2_MAXIMUM_DOWNLOAD_BYTES",
                        str(512 * 1_024 * 1_024),
                    )
                ),
                openeo_invocation_enabled=openeo_enabled,
                openeo_access_token=(
                    SecretStr(openeo_access_token) if openeo_access_token else None
                ),
                openeo_maximum_authorized_credits=openeo_credit_ceiling,
                openeo_timeout_seconds=float(
                    os.getenv("FIREVIEWER_CDSE_OPENEO_TIMEOUT_SECONDS", "120")
                ),
            )
            if cdse_access_key is not None and cdse_secret_key is not None
            else None
        )
        return cls(
            host=os.getenv("FIREVIEWER_SATELLITE_CPU_HOST", "0.0.0.0"),  # noqa: S104
            port=int(os.getenv("PORT", "8080")),
            worker_token=SecretStr(os.environ["FIREVIEWER_SATELLITE_CPU_WORKER_TOKEN"]),
            paid_invocation_enabled=enabled,
            managed_identity_client_id=os.getenv("AZURE_CLIENT_ID") or None,
            aws_oidc_audience=os.getenv("FIREVIEWER_AWS_OIDC_AUDIENCE") or None,
            sagemaker=sagemaker,
            maximum_raster_pixels=int(
                os.getenv("FIREVIEWER_SATELLITE_MAX_RASTER_PIXELS", "100000000")
            ),
            cdse=cdse,
            backend=AzureBackendEventEvidenceConfig(
                base_url=os.environ["FIREVIEWER_BACKEND_BASE_URL"],
                bearer_token=SecretStr(os.environ["FIREVIEWER_BACKEND_TOKEN"]),
                timeout_seconds=float(os.getenv("FIREVIEWER_BACKEND_TIMEOUT_SECONDS", "30")),
                max_response_bytes=20 * 1024 * 1024,
            ),
        )


class SatelliteCpuHandler(BaseHTTPRequestHandler):
    settings: SatelliteCpuServiceSettings
    observation_lock = Lock()

    def log_message(self, format: str, *args: object) -> None:
        print(json.dumps({"message": format % args}, separators=(",", ":")), flush=True)

    def _write(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._write(
            HTTPStatus.OK,
            {
                "status": "ready",
                "cpu_preparation_ready": True,
                "deterministic_satellite_observations_ready": self.settings.cdse is not None,
                "sentinel1_openeo_enabled": bool(
                    self.settings.cdse is not None
                    and self.settings.cdse.openeo_invocation_enabled
                ),
                "paid_invocation_enabled": self.settings.paid_invocation_enabled,
                "gpu_invoked": False,
            },
        )

    def do_POST(self) -> None:
        if self.path not in {
            "/v1/incident-day/satellite",
            "/v1/incident-day/satellite-observations",
        }:
            self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        supplied = self.headers.get("authorization", "")
        expected = f"Bearer {self.settings.worker_token.get_secret_value()}"
        if not compare_digest(supplied, expected):
            self._write(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if self.path == "/v1/incident-day/satellite-observations":
            self._run_satellite_observation()
            return
        if not self.settings.paid_invocation_enabled:
            self._write(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "paid_sagemaker_invocation_disabled"},
            )
            return
        try:
            size = int(self.headers.get("content-length", "0"))
        except ValueError:
            size = 0
        if size <= 0 or size > _MAX_REQUEST_BYTES:
            self._write(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid_size"})
            return
        try:
            request = SatelliteCpuRequest.model_validate_json(self.rfile.read(size))
            sagemaker = self.settings.sagemaker
            identity_client_id = self.settings.managed_identity_client_id
            audience = self.settings.aws_oidc_audience
            if sagemaker is None or identity_client_id is None or audience is None:
                raise ValueError("validated SageMaker settings are missing")
            provider = AzureFederatedSageMakerAsyncProvider(
                sagemaker,
                web_token_provider=AzureManagedIdentityTokenProvider(
                    managed_identity_client_id=identity_client_id,
                    audience=audience,
                ),
            )
            fetcher = BackendSatelliteBandFetcher(self.settings.backend)
            try:
                receipt = SatelliteCpuWorker(
                    repository=AzureBackendIncidentDayEvidenceAdapter(self.settings.backend),
                    band_fetcher=fetcher,
                    raster_builder=CanonicalPrithviRasterBuilder(
                        maximum_pixels=self.settings.maximum_raster_pixels
                    ),
                    provider=provider,
                    publisher=BackendIncidentDaySatelliteAnalysisPublisher(self.settings.backend),
                ).run(request.analysis_id)
            finally:
                fetcher.close()
        except SatelliteCpuError as exc:
            self._write(
                HTTPStatus.SERVICE_UNAVAILABLE if exc.retryable else HTTPStatus.CONFLICT,
                {"error": exc.code, "retryable": exc.retryable},
            )
            return
        except (BackendEventEvidenceError, ValueError):
            self._write(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        self._write(
            HTTPStatus.OK,
            {
                "analysis_id": receipt.analysis_id,
                "processed": receipt.processed,
                "remaining": receipt.remaining,
                "statuses": list(receipt.statuses),
            },
        )

    def _run_satellite_observation(self) -> None:
        if self.settings.cdse is None:
            self._write(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "cdse_s3_credentials_unavailable", "retryable": True},
            )
            return
        try:
            size = int(self.headers.get("content-length", "0"))
        except ValueError:
            size = 0
        if size <= 0 or size > _MAX_REQUEST_BYTES:
            self._write(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid_size"})
            return
        if not self.observation_lock.acquire(blocking=False):
            self._write(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "satellite_observation_worker_busy", "retryable": True},
            )
            return
        asset_reader: CdseS3ObservationAssetReader | None = None
        try:
            request = SatelliteObservationCpuRequest.model_validate_json(self.rfile.read(size))
            asset_reader = CdseS3ObservationAssetReader(self.settings.cdse)
            receipt = SatelliteObservationCpuWorker(
                repository=AzureBackendIncidentDayEvidenceAdapter(self.settings.backend),
                asset_reader=asset_reader,
                publisher=BackendIncidentDaySatelliteObservationPublisher(self.settings.backend),
                openeo_maximum_authorized_credits=(
                    self.settings.cdse.openeo_maximum_authorized_credits
                ),
            ).run(request.analysis_id, request.artifact_revision_id)
        except SatelliteCpuError as exc:
            self._write(
                HTTPStatus.SERVICE_UNAVAILABLE if exc.retryable else HTTPStatus.CONFLICT,
                {"error": exc.code, "retryable": exc.retryable},
            )
            return
        except (BackendEventEvidenceError, ValueError):
            self._write(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        finally:
            if asset_reader is not None:
                asset_reader.close()
            self.observation_lock.release()
        self._write(
            HTTPStatus.OK,
            {
                "analysis_id": receipt.analysis_id,
                "artifact_revision_id": receipt.artifact_revision_id,
                "processed": receipt.processed,
                "remaining": receipt.remaining,
                "status": receipt.status,
            },
        )


def main() -> None:
    SatelliteCpuHandler.settings = SatelliteCpuServiceSettings.from_env()
    server = ThreadingHTTPServer(
        (SatelliteCpuHandler.settings.host, SatelliteCpuHandler.settings.port),
        SatelliteCpuHandler,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()


__all__ = [
    "SatelliteCpuHandler",
    "SatelliteCpuRequest",
    "SatelliteCpuServiceSettings",
    "SatelliteObservationCpuRequest",
    "main",
]
