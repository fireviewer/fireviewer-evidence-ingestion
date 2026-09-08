"""Authenticated Azure CPU endpoint for incident-day public-media processing."""

from __future__ import annotations

import json
import os
import shutil
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pydantic import Field, SecretStr, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, StrictModel
from fireviewer_evidence_ingestion.mvp.research.multimodal_evidence import (
    AzureFederatedBedrockClient,
    AzureManagedIdentityWebTokenProvider,
    BedrockPixtralConfig,
    BedrockPixtralMultimodalProvider,
)
from fireviewer_evidence_ingestion.mvp.research.public_media_cpu import (
    HttpTransientYoloDetector,
    PublicMediaAnalysisRunReceipt,
    PublicMediaCpuConfig,
    PublicMediaCpuWorker,
)
from fireviewer_evidence_ingestion.mvp.research.transcription import (
    AzureSpeechFastTranscriptionProvider,
    FfmpegAudioTrackExtractor,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    AzureBackendEventEvidenceConfig,
    AzureBackendIncidentDayEvidenceAdapter,
    BackendIncidentDayMediaAnalysisPublisher,
)
from fireviewer_evidence_ingestion.research_broker import ResearchBroker

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


class PublicMediaAnalysisRequest(StrictModel):
    analysis_id: SafeIdentifierV2


class PublicMediaCpuServiceSettings(StrictModel):
    host: str = Field(default="0.0.0.0", min_length=1, max_length=255)  # noqa: S104
    port: int = Field(default=8080, ge=1, le=65_535)
    worker_token: SecretStr = Field(min_length=32, max_length=4_096)
    broker_control_token: SecretStr = Field(min_length=32, max_length=4_096)
    yolo_endpoint: str = Field(min_length=12, max_length=2_048)
    yolo_token: SecretStr = Field(min_length=32, max_length=4_096)
    transcription_enabled: bool = False
    speech_endpoint: str | None = Field(default=None, min_length=12, max_length=2_048)
    multimodal_enabled: bool = False
    managed_identity_client_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
    )
    aws_role_arn: str | None = Field(
        default=None,
        pattern=r"^arn:aws:iam::\d{12}:role/[A-Za-z0-9+=,.@_/-]+$",
    )
    aws_oidc_audience: str | None = Field(default=None, min_length=8, max_length=512)
    aws_region: str = Field(default="eu-west-3", pattern=r"^[a-z]{2}-[a-z]+-\d$")
    bedrock_model_id: str = Field(
        default="eu.mistral.pixtral-large-2502-v1:0",
        min_length=3,
        max_length=256,
    )
    maximum_media_per_run: int = Field(default=1, ge=1, le=8)
    maximum_media_bytes: int = Field(
        default=512 * 1_024 * 1_024,
        ge=1_024 * 1_024,
        le=512 * 1_024 * 1_024,
    )
    backend: AzureBackendEventEvidenceConfig

    @model_validator(mode="after")
    def validate_provider_configuration(self) -> PublicMediaCpuServiceSettings:
        if self.transcription_enabled and (
            self.speech_endpoint is None or self.managed_identity_client_id is None
        ):
            raise ValueError("managed Azure Speech configuration is incomplete")
        if self.multimodal_enabled and (
            self.managed_identity_client_id is None
            or self.aws_role_arn is None
            or self.aws_oidc_audience is None
        ):
            raise ValueError("Azure-federated Bedrock configuration is incomplete")
        return self

    @classmethod
    def from_env(cls) -> PublicMediaCpuServiceSettings:
        return cls(
            host=os.getenv("FIREVIEWER_PUBLIC_MEDIA_HOST", "0.0.0.0"),  # noqa: S104
            port=int(os.getenv("PORT", "8080")),
            worker_token=SecretStr(os.environ["FIREVIEWER_PUBLIC_MEDIA_WORKER_TOKEN"]),
            broker_control_token=SecretStr(
                os.environ["FIREVIEWER_PUBLIC_MEDIA_BROKER_CONTROL_TOKEN"]
            ),
            yolo_endpoint=os.environ["FIREVIEWER_YOLO_BASE_URL"],
            yolo_token=SecretStr(os.environ["FW_YOLO_AUTH_TOKEN"]),
            transcription_enabled=_env_bool(
                "FIREVIEWER_TRANSCRIPTION_ENABLED",
                False,
            ),
            speech_endpoint=os.getenv("FIREVIEWER_AZURE_SPEECH_ENDPOINT") or None,
            multimodal_enabled=_env_bool("FIREVIEWER_MULTIMODAL_ENABLED", False),
            managed_identity_client_id=os.getenv("AZURE_CLIENT_ID") or None,
            aws_role_arn=os.getenv("FIREVIEWER_BEDROCK_ROLE_ARN") or None,
            aws_oidc_audience=os.getenv("FIREVIEWER_AWS_OIDC_AUDIENCE") or None,
            aws_region=os.getenv("AWS_REGION", "eu-west-3"),
            bedrock_model_id=os.getenv(
                "FIREVIEWER_BEDROCK_MODEL_ID",
                "eu.mistral.pixtral-large-2502-v1:0",
            ),
            maximum_media_per_run=int(
                os.getenv("FIREVIEWER_PUBLIC_MEDIA_MAX_PER_RUN", "1")
            ),
            maximum_media_bytes=int(
                os.getenv(
                    "FIREVIEWER_PUBLIC_MEDIA_MAX_BYTES",
                    str(512 * 1_024 * 1_024),
                )
            ),
            backend=AzureBackendEventEvidenceConfig(
                base_url=os.environ["FIREVIEWER_BACKEND_BASE_URL"],
                bearer_token=SecretStr(os.environ["FIREVIEWER_BACKEND_TOKEN"]),
                timeout_seconds=float(
                    os.getenv("FIREVIEWER_BACKEND_TIMEOUT_SECONDS", "20")
                ),
            ),
        )


class PublicMediaCpuService:
    def __init__(
        self,
        *,
        settings: PublicMediaCpuServiceSettings,
        worker: PublicMediaCpuWorker | None = None,
    ) -> None:
        self.settings = settings
        if worker is None:
            speech = None
            if settings.transcription_enabled:
                if (
                    settings.speech_endpoint is None
                    or settings.managed_identity_client_id is None
                ):
                    raise ValueError("validated Azure Speech settings are missing")
                speech = AzureSpeechFastTranscriptionProvider(
                    endpoint=settings.speech_endpoint,
                    token_provider=AzureManagedIdentityWebTokenProvider(
                        audience="https://cognitiveservices.azure.com/",
                        managed_identity_client_id=settings.managed_identity_client_id,
                    ),
                )
            evidence_provider = None
            if settings.multimodal_enabled:
                if (
                    settings.aws_role_arn is None
                    or settings.aws_oidc_audience is None
                    or settings.managed_identity_client_id is None
                ):
                    raise ValueError("validated Bedrock settings are missing")
                evidence_provider = BedrockPixtralMultimodalProvider(
                    BedrockPixtralConfig(
                        region_name=settings.aws_region,
                        model_id=settings.bedrock_model_id,
                    ),
                    client=AzureFederatedBedrockClient(
                        role_arn=settings.aws_role_arn,
                        region_name=settings.aws_region,
                        web_token_provider=AzureManagedIdentityWebTokenProvider(
                            audience=settings.aws_oidc_audience,
                            managed_identity_client_id=(
                                settings.managed_identity_client_id
                            ),
                        ),
                        role_session_name="fireviewer-public-media",
                    ),
                )
            broker_token = settings.broker_control_token.get_secret_value()
            worker = PublicMediaCpuWorker(
                repository=AzureBackendIncidentDayEvidenceAdapter(settings.backend),
                publisher=BackendIncidentDayMediaAnalysisPublisher(settings.backend),
                broker=ResearchBroker(control_token=broker_token),
                broker_control_token=broker_token,
                detector=HttpTransientYoloDetector(
                    endpoint=settings.yolo_endpoint,
                    token=settings.yolo_token,
                ),
                transcription_provider=speech,
                evidence_provider=evidence_provider,
                audio_extractor=FfmpegAudioTrackExtractor(),
                config=PublicMediaCpuConfig(
                    maximum_media_per_run=settings.maximum_media_per_run,
                    maximum_media_bytes=settings.maximum_media_bytes,
                ),
            )
        self.worker = worker

    @property
    def enabled(self) -> bool:
        return True

    def authorize(self, header: str | None) -> bool:
        expected = f"Bearer {self.settings.worker_token.get_secret_value()}"
        return header is not None and compare_digest(header, expected)

    def run(self, payload: object) -> PublicMediaAnalysisRunReceipt:
        request = PublicMediaAnalysisRequest.model_validate(payload)
        return self.worker.run_analysis(request.analysis_id)


def _handler_for(service: PublicMediaCpuService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "FireViewerPublicMediaCpu/1.0"

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/healthz":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "runtime": "azure-cpu",
                    "opencv": True,
                    "ffmpeg": shutil.which("ffmpeg") is not None,
                    "yolo_sink_connected": True,
                    "transcription_enabled": service.settings.transcription_enabled,
                    "multimodal_enabled": service.settings.multimodal_enabled,
                    "missing_optional_providers_are_journaled": True,
                    "raw_public_media_stored": False,
                    "raw_keyframes_stored": False,
                    "transcripts_stored": False,
                },
            )

        def do_POST(self) -> None:
            if self.path != "/v1/incident-day/public-media":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            if not service.authorize(self.headers.get("Authorization")):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= _MAX_REQUEST_BYTES:
                    raise ValueError("request_size_invalid")
                payload = json.loads(self.rfile.read(length))
                receipt = service.run(payload)
            except Exception as exc:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": type(exc).__name__, "detail": str(exc)[:1_000]},
                )
                return
            self._json(HTTPStatus.OK, receipt.model_dump(mode="json"))

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def create_public_media_cpu_server(
    service: PublicMediaCpuService,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(
        (service.settings.host, service.settings.port),
        _handler_for(service),
    )


def main() -> None:
    settings = PublicMediaCpuServiceSettings.from_env()
    create_public_media_cpu_server(PublicMediaCpuService(settings=settings)).serve_forever()


if __name__ == "__main__":
    main()


__all__ = [
    "PublicMediaAnalysisRequest",
    "PublicMediaCpuService",
    "PublicMediaCpuServiceSettings",
    "create_public_media_cpu_server",
]
