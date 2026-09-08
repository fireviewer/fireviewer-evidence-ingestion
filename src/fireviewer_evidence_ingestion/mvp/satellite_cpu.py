"""CPU preparation and cost-gated SageMaker Async bridge for incident-day satellite data."""

from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast
from urllib.parse import quote, urlsplit

import boto3
import httpx
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import Field

from fireviewer_contracts.contracts import StrictModel, WorkerInputV2
from fireviewer_contracts.geo_gpu import (
    EncodedPayload,
    GeoGpuRequest,
    GeoGpuResponse,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    AzureBackendEventEvidenceConfig,
    BackendIncidentDaySatelliteAnalysisPublisher,
    BackendIncidentDaySatelliteArtifact,
    DurableEventEvidence,
    EventEvidenceRepository,
)

_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")
_DESCRIPTIONS = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SatelliteCpuError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class AzureManagedIdentityTokenProvider:
    """Read one Azure managed-identity token from the Container Apps broker."""

    def __init__(self, *, audience: str, managed_identity_client_id: str) -> None:
        self._audience = audience
        self._managed_identity_client_id = managed_identity_client_id

    def __call__(self) -> str:
        endpoint = os.environ.get("IDENTITY_ENDPOINT", "").strip()
        identity_header = os.environ.get("IDENTITY_HEADER", "").strip()
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or not identity_header
        ):
            raise SatelliteCpuError(
                "azure_managed_identity_endpoint_unavailable", retryable=True
            )
        try:
            response = httpx.get(
                endpoint,
                params={
                    "resource": self._audience,
                    "api-version": "2019-08-01",
                    "client_id": self._managed_identity_client_id,
                },
                headers={"X-IDENTITY-HEADER": identity_header},
                timeout=5.0,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SatelliteCpuError(
                "azure_managed_identity_token_failed", retryable=True
            ) from exc
        token = payload.get("access_token") if isinstance(payload, Mapping) else None
        if not isinstance(token, str) or len(token) < 100:
            raise SatelliteCpuError(
                "azure_managed_identity_token_invalid", retryable=True
            )
        return token


class SatelliteBandFetcher(Protocol):
    def fetch(
        self,
        *,
        artifact: BackendIncidentDaySatelliteArtifact,
        directory: Path,
    ) -> dict[str, Path]: ...


class BackendSatelliteBandFetcher:
    """Stream immutable JP2 bands into one ephemeral directory with SHA verification."""

    def __init__(
        self,
        config: AzureBackendEventEvidenceConfig,
        *,
        client: httpx.Client | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        self._config = config
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(connect=10, read=120, write=30, pool=10),
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": "FireViewer-Satellite-CPU/1.0"},
        )
        self._owns_client = client is None
        self._chunk_size = chunk_size

    def fetch(
        self,
        *,
        artifact: BackendIncidentDaySatelliteArtifact,
        directory: Path,
    ) -> dict[str, Path]:
        if artifact.materialization_state != "materialized":
            raise SatelliteCpuError("satellite_materialization_incomplete", retryable=True)
        paths: dict[str, Path] = {}
        for receipt in artifact.prithvi_bands:
            path = directory / f"{receipt.canonical_band}.jp2"
            digest = sha256()
            size = 0
            url = self._config.base_url + receipt.content_path
            try:
                with self._client.stream(
                    "GET",
                    url,
                    headers={
                        "Accept": "image/jp2",
                        "Authorization": (
                            f"Bearer {self._config.bearer_token.get_secret_value()}"
                        ),
                    },
                ) as response:
                    if response.status_code != 200:
                        raise SatelliteCpuError(
                            f"satellite_band_http_{response.status_code}",
                            retryable=response.status_code >= 500,
                        )
                    if response.headers.get("content-type", "").split(";", 1)[0] != "image/jp2":
                        raise SatelliteCpuError(
                            "satellite_band_content_type_invalid", retryable=False
                        )
                    with path.open("wb") as output:
                        for chunk in response.iter_bytes(self._chunk_size):
                            size += len(chunk)
                            if size > receipt.size_bytes:
                                raise SatelliteCpuError(
                                    "satellite_band_size_mismatch", retryable=False
                                )
                            digest.update(chunk)
                            output.write(chunk)
            except httpx.HTTPError as exc:
                path.unlink(missing_ok=True)
                raise SatelliteCpuError(
                    "satellite_band_download_failed", retryable=True
                ) from exc
            if size != receipt.size_bytes or digest.hexdigest() != receipt.content_sha256:
                path.unlink(missing_ok=True)
                raise SatelliteCpuError("satellite_band_digest_mismatch", retryable=False)
            paths[receipt.canonical_band] = path
        if tuple(paths) != _BANDS:
            raise SatelliteCpuError("satellite_band_set_incomplete", retryable=False)
        return paths

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


@dataclass(frozen=True, slots=True)
class PreparedSatelliteRaster:
    path: Path
    sha256: str
    size_bytes: int
    crs: str
    width: int
    height: int
    geotransform: tuple[float, float, float, float, float, float]
    bbox_wgs84: tuple[float, float, float, float]
    resolution_m: float


class CanonicalPrithviRasterBuilder:
    """Align the six Sentinel-2 bands on the 20 m B11 grid and crop to the incident."""

    def __init__(self, *, maximum_pixels: int = 100_000_000) -> None:
        if maximum_pixels < 256 or maximum_pixels > 250_000_000:
            raise ValueError("maximum_pixels is outside the safe raster budget")
        self.maximum_pixels = maximum_pixels

    def build(
        self,
        *,
        band_paths: Mapping[str, Path],
        incident_bbox: tuple[float, float, float, float],
        output_path: Path,
    ) -> PreparedSatelliteRaster:
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject, transform_bounds
        from rasterio.windows import Window, from_bounds

        if tuple(band_paths) != _BANDS:
            raise SatelliteCpuError("satellite_band_order_invalid", retryable=False)
        with ExitStack() as stack:
            datasets = {
                band: stack.enter_context(rasterio.open(band_paths[band]))
                for band in _BANDS
            }
            target = datasets["B11"]
            if target.crs is None:
                raise SatelliteCpuError("satellite_target_crs_missing", retryable=False)
            target_crs = target.crs
            target_bounds = transform_bounds(
                "EPSG:4326",
                target_crs,
                *incident_bbox,
                densify_pts=21,
            )
            requested = from_bounds(*target_bounds, transform=target.transform)
            requested = requested.round_offsets().round_lengths()
            full = Window(0, 0, target.width, target.height)
            try:
                window = requested.intersection(full)
            except rasterio.errors.WindowError as exc:
                raise SatelliteCpuError(
                    "incident_outside_satellite_product", retryable=False
                ) from exc
            width = int(window.width)
            height = int(window.height)
            if width <= 0 or height <= 0 or width * height > self.maximum_pixels:
                raise SatelliteCpuError("satellite_crop_size_invalid", retryable=False)
            transform = target.window_transform(window)
            profile = {
                "driver": "GTiff",
                "width": width,
                "height": height,
                "count": 6,
                "dtype": "uint16",
                "crs": target_crs,
                "transform": transform,
                "nodata": 0,
                "compress": "deflate",
                "predictor": 2,
                "BIGTIFF": "IF_SAFER",
            }
            with rasterio.open(output_path, "w", **profile) as output:
                for index, (band, description) in enumerate(
                    zip(_BANDS, _DESCRIPTIONS, strict=True), start=1
                ):
                    source = datasets[band]
                    if source.crs is None:
                        raise SatelliteCpuError(
                            "satellite_source_crs_missing", retryable=False
                        )
                    destination = np.zeros((height, width), dtype=np.uint16)
                    reproject(
                        source=rasterio.band(source, 1),
                        destination=destination,
                        src_transform=source.transform,
                        src_crs=source.crs,
                        src_nodata=source.nodata,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        dst_nodata=0,
                        resampling=Resampling.bilinear,
                    )
                    output.write(destination, index)
                    output.set_band_description(index, description)
            bounds_wgs84 = transform_bounds(
                target_crs,
                "EPSG:4326",
                *rasterio.transform.array_bounds(height, width, transform),
                densify_pts=21,
            )
        return PreparedSatelliteRaster(
            path=output_path,
            sha256=_file_sha256(output_path),
            size_bytes=output_path.stat().st_size,
            crs=str(target_crs),
            width=width,
            height=height,
            geotransform=cast(
                tuple[float, float, float, float, float, float],
                tuple(float(value) for value in transform.to_gdal()),
            ),
            bbox_wgs84=cast(
                tuple[float, float, float, float],
                tuple(float(value) for value in bounds_wgs84),
            ),
            resolution_m=max(abs(float(transform.a)), abs(float(transform.e))),
        )


class SatelliteGeoProvider(Protocol):
    model_id: str
    model_revision: str

    def invoke(self, request: GeoGpuRequest) -> GeoGpuResponse: ...


class SageMakerAsyncConfig(StrictModel):
    region_name: str = Field(pattern=r"^[a-z]{2}-[a-z]+-\d$")
    role_arn: str = Field(pattern=r"^arn:aws:iam::\d{12}:role/[A-Za-z0-9+=,.@_/-]+$")
    endpoint_name: str = Field(pattern=r"^fireviewer-geo-async-[0-9a-f]{16}$")
    bucket_name: str = Field(pattern=r"^fireviewer-geo-ai-\d{12}-[a-z0-9-]+$")
    input_prefix: str = Field(
        default="async/input/production",
        pattern=r"^async/input/[a-z0-9/-]+$",
    )
    poll_seconds: float = Field(default=10, ge=1, le=60)
    # Azure Container Apps HTTP ingress has a fixed 240 second request timeout.
    # Short polling windows are resumed from the immutable S3 reservation.
    maximum_wait_seconds: int = Field(default=180, ge=30, le=210)


def _s3_location(value: str, *, bucket: str, prefixes: tuple[str, ...]) -> tuple[str, str]:
    parsed = urlsplit(value)
    if parsed.scheme != "s3" or parsed.netloc != bucket:
        raise SatelliteCpuError("sagemaker_output_location_invalid", retryable=False)
    key = parsed.path.lstrip("/")
    if not any(key.startswith(prefix) for prefix in prefixes):
        raise SatelliteCpuError("sagemaker_output_prefix_invalid", retryable=False)
    return bucket, key


class AzureFederatedSageMakerAsyncProvider:
    """Invoke one immutable Async endpoint; S3 reservation prevents paid duplicate calls."""

    model_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
    model_revision = "a3f2c410e45b8ac7417976614528a872f024d831"

    def __init__(
        self,
        config: SageMakerAsyncConfig,
        *,
        web_token_provider: Callable[[], str],
        sts_client: Any | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._web_token_provider = web_token_provider
        self._sts = sts_client or boto3.client(
            "sts",
            region_name=config.region_name,
            config=BotocoreConfig(
                retries={"mode": "adaptive", "total_max_attempts": 4},
                connect_timeout=5,
                read_timeout=10,
            ),
        )
        self._clock = clock
        self._sleeper = sleeper
        self._expires_at: datetime | None = None
        self._s3: Any | None = None
        self._runtime: Any | None = None

    def _clients(self) -> tuple[Any, Any]:
        now = self._clock()
        if (
            self._s3 is not None
            and self._runtime is not None
            and self._expires_at is not None
            and self._expires_at - now > timedelta(minutes=5)
        ):
            return self._s3, self._runtime
        try:
            web_token = self._web_token_provider()
        except SatelliteCpuError:
            raise
        except Exception as exc:
            raise SatelliteCpuError(
                "azure_managed_identity_token_failed", retryable=True
            ) from exc
        try:
            assumed = self._sts.assume_role_with_web_identity(
                RoleArn=self.config.role_arn,
                RoleSessionName="fireviewer-satellite-cpu",
                WebIdentityToken=web_token,
                DurationSeconds=3600,
            )
            credentials = assumed["Credentials"]
            expires_at = credentials["Expiration"]
            if not isinstance(expires_at, datetime):
                raise TypeError("AWS STS expiration is invalid")
            session = boto3.Session(
                aws_access_key_id=str(credentials["AccessKeyId"]),
                aws_secret_access_key=str(credentials["SecretAccessKey"]),
                aws_session_token=str(credentials["SessionToken"]),
                region_name=self.config.region_name,
            )
            common = BotocoreConfig(
                retries={"mode": "adaptive", "total_max_attempts": 5},
                connect_timeout=5,
                read_timeout=30,
            )
            self._s3 = session.client("s3", config=common)
            self._runtime = session.client("sagemaker-runtime", config=common)
            self._expires_at = expires_at.astimezone(UTC)
        except (
            BotoCoreError,
            ClientError,
            KeyError,
            TypeError,
        ) as exc:
            raise SatelliteCpuError("aws_sagemaker_federation_failed", retryable=True) from exc
        return self._s3, self._runtime

    def _read_json(self, s3: Any, *, bucket: str, key: str, limit: int) -> dict[str, Any]:
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            body = response["Body"].read(limit + 1)
        except (BotoCoreError, ClientError, KeyError) as exc:
            raise SatelliteCpuError("sagemaker_result_read_failed", retryable=True) from exc
        if len(body) > limit:
            raise SatelliteCpuError("sagemaker_result_too_large", retryable=False)
        try:
            decoded = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SatelliteCpuError("sagemaker_result_invalid_json", retryable=False) from exc
        if not isinstance(decoded, dict):
            raise SatelliteCpuError("sagemaker_result_invalid_shape", retryable=False)
        return decoded

    def invoke(self, request: GeoGpuRequest) -> GeoGpuResponse:
        s3, runtime = self._clients()
        body = _canonical_json_bytes(request.model_dump(mode="json", by_alias=True))
        body_sha = sha256(body).hexdigest()
        analysis_id = (
            request.worker_input.analysis_window.analysis_id
            if request.worker_input
            else "none"
        )
        key_root = (
            f"{self.config.input_prefix.rstrip('/')}/{quote(analysis_id, safe='')}/"
            f"{body_sha}"
        )
        request_key = f"{key_root}/request.json"
        reservation_key = f"{key_root}/reservation.json"
        reserved = _canonical_json_bytes(
            {
                "schema": "fireviewer.sagemaker-async-reservation.v1",
                "state": "reserved",
                "request_sha256": body_sha,
            }
        )
        output_location: str | None = None
        failure_location = ""
        reservation_etag: str | None = None
        try:
            reservation = s3.put_object(
                Bucket=self.config.bucket_name,
                Key=reservation_key,
                Body=reserved,
                ContentType="application/json",
                IfNoneMatch="*",
            )
            reservation_etag = str(reservation["ETag"])
        except ClientError as exc:
            if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 412:
                existing = self._read_json(
                    s3,
                    bucket=self.config.bucket_name,
                    key=reservation_key,
                    limit=16 * 1024,
                )
                if (
                    existing.get("schema")
                    != "fireviewer.sagemaker-async-reservation.v1"
                    or existing.get("request_sha256") != body_sha
                ):
                    raise SatelliteCpuError(
                        "sagemaker_reservation_mismatch", retryable=False
                    ) from exc
                if existing.get("state") != "submitted":
                    raise SatelliteCpuError(
                        "sagemaker_submission_state_ambiguous", retryable=False
                    ) from exc
                raw_output = existing.get("output_location")
                raw_failure = existing.get("failure_location")
                if not isinstance(raw_output, str):
                    raise SatelliteCpuError(
                        "sagemaker_reservation_output_missing", retryable=False
                    ) from exc
                if raw_failure is not None and not isinstance(raw_failure, str):
                    raise SatelliteCpuError(
                        "sagemaker_reservation_failure_invalid", retryable=False
                    ) from exc
                output_location = raw_output
                failure_location = raw_failure or ""
            else:
                raise SatelliteCpuError("sagemaker_reservation_failed", retryable=True) from exc
        if output_location is None:
            if reservation_etag is None:
                raise SatelliteCpuError("sagemaker_reservation_etag_missing", retryable=False)
            try:
                s3.put_object(
                    Bucket=self.config.bucket_name,
                    Key=request_key,
                    Body=body,
                    ContentType="application/json",
                    Metadata={"sha256": body_sha},
                    IfNoneMatch="*",
                )
                invocation = runtime.invoke_endpoint_async(
                    EndpointName=self.config.endpoint_name,
                    InputLocation=f"s3://{self.config.bucket_name}/{request_key}",
                    ContentType="application/json",
                    InferenceId=request.request_id,
                )
                output_location = str(invocation["OutputLocation"])
                failure_location = str(invocation.get("FailureLocation") or "")
                submitted = _canonical_json_bytes(
                    {
                        "schema": "fireviewer.sagemaker-async-reservation.v1",
                        "state": "submitted",
                        "request_sha256": body_sha,
                        "output_location": output_location,
                        "failure_location": failure_location or None,
                    }
                )
                s3.put_object(
                    Bucket=self.config.bucket_name,
                    Key=reservation_key,
                    Body=submitted,
                    ContentType="application/json",
                    IfMatch=reservation_etag,
                )
            except (BotoCoreError, ClientError, KeyError) as exc:
                raise SatelliteCpuError(
                    "sagemaker_async_submission_ambiguous", retryable=False
                ) from exc
        output_bucket, output_key = _s3_location(
            output_location,
            bucket=self.config.bucket_name,
            prefixes=("async/output/",),
        )
        failure_target = (
            _s3_location(
                failure_location,
                bucket=self.config.bucket_name,
                prefixes=("async/failure/",),
            )
            if failure_location
            else None
        )
        deadline = self._clock() + timedelta(seconds=self.config.maximum_wait_seconds)
        while self._clock() < deadline:
            try:
                s3.head_object(Bucket=output_bucket, Key=output_key)
            except ClientError as exc:
                status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if status not in {403, 404}:
                    raise SatelliteCpuError(
                        "sagemaker_result_poll_failed", retryable=True
                    ) from exc
            else:
                payload = self._read_json(
                    s3, bucket=output_bucket, key=output_key, limit=8 * 1024 * 1024
                )
                response = GeoGpuResponse.model_validate(payload)
                if response.request_id != request.request_id:
                    raise SatelliteCpuError(
                        "sagemaker_result_request_mismatch", retryable=False
                    )
                return response
            if failure_target is not None:
                try:
                    s3.head_object(Bucket=failure_target[0], Key=failure_target[1])
                except ClientError:
                    pass
                else:
                    raise SatelliteCpuError("sagemaker_inference_failed", retryable=False)
            self._sleeper(self.config.poll_seconds)
        raise SatelliteCpuError("sagemaker_async_timeout", retryable=True)


def _cloud_cover(artifact: BackendIncidentDaySatelliteArtifact) -> float | None:
    for key in ("eo:cloud_cover", "cloud_cover", "cloudCover"):
        value = artifact.quality_flags.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return min(100.0, max(0.0, float(value)))
    return None


def _safe_source_key(value: str) -> str:
    if value and len(value) <= 128 and all(
        character.isalnum() or character in "._:-" for character in value
    ):
        return value
    return f"SRC-{sha256(value.encode()).hexdigest()[:24]}"


def build_prithvi_request(
    *,
    durable: DurableEventEvidence,
    artifact: BackendIncidentDaySatelliteArtifact,
    raster: PreparedSatelliteRaster,
    request_id: str,
) -> GeoGpuRequest:
    if (
        durable.incident_id is None
        or durable.incident_day_episode_id is None
        or durable.incident_day_local_date is None
        or durable.incident_day_timezone is None
        or artifact.materialization_bundle_id is None
        or artifact.materialization_manifest_sha256 is None
        or artifact.acquisition_start_at is None
        or durable.event.time_window.from_at is None
        or durable.event.time_window.to_at is None
    ):
        raise SatelliteCpuError("satellite_incident_context_incomplete", retryable=False)
    content = raster.path.read_bytes()
    if sha256(content).hexdigest() != raster.sha256:
        raise SatelliteCpuError("satellite_prithvi_input_changed", retryable=False)
    input_id = artifact.artifact_revision_id
    payload_url = f"https://payload.fireviewer.invalid/{raster.sha256}.tif"
    reference_url = (
        "https://backend.fireviewer.invalid/api/v1/internal/satellite-materializations/"
        f"{artifact.materialization_bundle_id}"
    )
    worker_input = WorkerInputV2.model_validate(
        {
            "schema_version": "2.0",
            "batch_id": f"SATB-{raster.sha256[:24]}",
            "batch_type": "satellite_media",
            "priority": "scheduled",
            "analysis_window": {
                "analysis_id": durable.event.event_id,
                "fire_id": durable.incident_id,
                "episode_id": durable.incident_day_episode_id,
                "window_start_at": durable.event.time_window.from_at.isoformat(),
                "window_end_at": durable.event.time_window.to_at.isoformat(),
                "local_date": durable.incident_day_local_date.isoformat(),
                "timezone": durable.incident_day_timezone,
            },
            "reference_bundle": {
                "reference_id": f"REF-{artifact.materialization_manifest_sha256[:24]}",
                "manifest_sha256": artifact.materialization_manifest_sha256,
                "assets": [
                    {
                        "kind": "source_manifest",
                        "working_file_url": reference_url,
                        "sha256": artifact.materialization_manifest_sha256,
                        "crs": raster.crs,
                        "resolution_m": raster.resolution_m,
                    }
                ],
            },
            "items": [
                {
                    "input_id": input_id,
                    "media_type": "satellite_image",
                    "working_file_url": payload_url,
                    "captured_at": artifact.acquisition_start_at.isoformat(),
                    "provenance": {
                        "source_key": _safe_source_key(artifact.provider_key),
                        "source_reference_url": artifact.source_url,
                        "license_identifier": (
                            f"LIC-{sha256(artifact.license.encode()).hexdigest()[:24]}"
                        ),
                        "attribution": artifact.attribution[:500],
                        "trust": "institutional",
                        "source_registry_version": "incident-day-cdse-v1",
                        "source_kind": "satellite",
                        "source_confidence": "A+",
                        "publication_policy": "dataset_license_required",
                        "claim_types": ["burned_area"],
                    },
                    "satellite": {
                        "product_id": (
                            "SAT-"
                            + sha256(artifact.external_product_id.encode()).hexdigest()[:24]
                        ),
                        "provider": artifact.provider_key,
                        "acquired_at": artifact.acquisition_start_at.isoformat(),
                        "crs": raster.crs,
                        "raster_width_px": raster.width,
                        "raster_height_px": raster.height,
                        "geotransform": raster.geotransform,
                        "bbox_wgs84": raster.bbox_wgs84,
                        "resolution_m": raster.resolution_m,
                        "bands": _DESCRIPTIONS,
                        "cloud_cover_percent": _cloud_cover(artifact),
                    },
                }
            ],
        }
    )
    return GeoGpuRequest(
        request_id=request_id,
        operation="prithvi.burned_area",
        payloads=(
            EncodedPayload(
                input_id=input_id,
                content_type="image/tiff",
                content_sha256=raster.sha256,
                content_base64=base64.b64encode(content).decode("ascii"),
            ),
        ),
        worker_input=worker_input,
    )


@dataclass(frozen=True, slots=True)
class SatelliteCpuRunReceipt:
    analysis_id: str
    processed: int
    remaining: int
    statuses: tuple[str, ...]


class SatelliteCpuWorker:
    def __init__(
        self,
        *,
        repository: EventEvidenceRepository,
        band_fetcher: SatelliteBandFetcher,
        raster_builder: CanonicalPrithviRasterBuilder,
        provider: SatelliteGeoProvider,
        publisher: BackendIncidentDaySatelliteAnalysisPublisher,
    ) -> None:
        self.repository = repository
        self.band_fetcher = band_fetcher
        self.raster_builder = raster_builder
        self.provider = provider
        self.publisher = publisher

    def run(self, analysis_id: str) -> SatelliteCpuRunReceipt:
        durable = self.repository.read(analysis_id)
        if durable.research_target_kind != "incident_day" or durable.incident_day_bbox is None:
            raise SatelliteCpuError("satellite_incident_day_required", retryable=False)
        completed = {
            (item.materialization_bundle_id, item.materialization_manifest_sha256)
            for item in durable.satellite_analysis_batches
            if item.status in {"completed", "abstained"}
        }
        pending = [
            item
            for item in durable.satellite_artifact_tickets
            if item.materialization_state == "materialized"
            and item.materialization_bundle_id is not None
            and item.materialization_manifest_sha256 is not None
            and (item.materialization_bundle_id, item.materialization_manifest_sha256)
            not in completed
        ]
        if not pending:
            return SatelliteCpuRunReceipt(
                analysis_id=analysis_id,
                processed=0,
                remaining=0,
                statuses=(),
            )
        artifact = pending[0]
        with TemporaryDirectory(prefix="fireviewer-satellite-cpu-") as directory:
            root = Path(directory)
            bands = self.band_fetcher.fetch(artifact=artifact, directory=root)
            raster = self.raster_builder.build(
                band_paths=bands,
                incident_bbox=durable.incident_day_bbox,
                output_path=root / "prithvi-input.tif",
            )
            request_digest = sha256(
                (analysis_id + artifact.artifact_revision_id + raster.sha256).encode()
            ).hexdigest()
            request_id = f"GEO-{request_digest[:24]}"
            request = build_prithvi_request(
                durable=durable,
                artifact=artifact,
                raster=raster,
                request_id=request_id,
            )
            response = self.provider.invoke(request)
            if response.operation != "prithvi.burned_area":
                raise SatelliteCpuError("satellite_provider_operation_mismatch", retryable=False)
            annotations_by_input = response.result.get("annotations", {})
            proposals_by_input = response.result.get("spatial_proposals", {})
            annotations = (
                annotations_by_input.get(artifact.artifact_revision_id, [])
                if isinstance(annotations_by_input, Mapping)
                else []
            )
            proposals = (
                proposals_by_input.get(artifact.artifact_revision_id, [])
                if isinstance(proposals_by_input, Mapping)
                else []
            )
            status = "completed" if proposals else "abstained"
            request_sha256 = sha256(
                _canonical_json_bytes(request.model_dump(mode="json", by_alias=True))
            ).hexdigest()
            if (
                artifact.materialization_bundle_id is None
                or artifact.materialization_manifest_sha256 is None
            ):
                raise SatelliteCpuError("satellite_materialization_incomplete", retryable=False)
            self.publisher.publish(
                candidate_id=analysis_id,
                payload={
                    "schema_version": "incident-day-satellite-analysis-1.0",
                    "analysis_id": analysis_id,
                    "source_revision_sha256": durable.source_revision_sha256,
                    "artifact_revision_id": artifact.artifact_revision_id,
                    "materialization_bundle_id": artifact.materialization_bundle_id,
                    "materialization_manifest_sha256": artifact.materialization_manifest_sha256,
                    "prithvi_input_sha256": raster.sha256,
                    "request_id": request_id,
                    "request_sha256": request_sha256,
                    "status": status,
                    "model_id": response.model_id,
                    "model_revision": response.model_revision,
                    "annotations": annotations,
                    "spatial_proposals": proposals,
                    "reason_codes": (
                        list(response.reason_codes)
                        if status == "abstained"
                        else []
                    ),
                },
            )
        return SatelliteCpuRunReceipt(
            analysis_id=analysis_id,
            processed=1,
            remaining=max(0, len(pending) - 1),
            statuses=(status,),
        )


__all__ = [
    "AzureFederatedSageMakerAsyncProvider",
    "AzureManagedIdentityTokenProvider",
    "BackendSatelliteBandFetcher",
    "CanonicalPrithviRasterBuilder",
    "PreparedSatelliteRaster",
    "SageMakerAsyncConfig",
    "SatelliteCpuError",
    "SatelliteCpuRunReceipt",
    "SatelliteCpuWorker",
    "build_prithvi_request",
]
