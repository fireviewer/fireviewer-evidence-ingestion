"""Deterministic CPU extraction of daily Copernicus wildfire observations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlsplit

import boto3
import httpx
import numpy as np
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import Field, SecretStr, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, StrictModel
from fireviewer_evidence_ingestion.mvp.satellite_cpu import SatelliteCpuError
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    BackendIncidentDaySatelliteArtifact,
    BackendIncidentDaySatelliteObservationPublisher,
    DurableEventEvidence,
    EventEvidenceRepository,
)

_CLMS_PROCESSOR = "clms_burned_area_daily_v1"
_CLMS_REVISION = "fireviewer-clms-burned-area-cpu-1.1.0"
_CLMS_COLLECTION = "clms_ba_global_300m_daily_v4_cog"
_CLMS_ASSETS = ("ba300_dob_nrt", "ba300_cp_nrt", "ba300_bf_nrt")
_S2_PROCESSOR = "sentinel2_nbr_change_v1"
_S2_REVISION = "fireviewer-sentinel2-nbr-change-cpu-1.3.0"
_S2_COLLECTION = "sentinel-2-l2a"
_S2_ASSETS = ("B04_20m", "B8A_20m", "B11_20m", "B12_20m", "SCL_20m")
_S1_PROCESSOR = "sentinel1_vvvh_change_v1"
_S1_REVISION = "fireviewer-sentinel1-vvvh-change-openeo-1.1.0"
_S1_COLLECTION = "sentinel-1-grd"
_S1_ASSETS = ("openeo_vv_vh",)
_FRP_PROCESSOR = "sentinel3_frp_v1"
_FRP_REVISION = "fireviewer-sentinel3-frp-cpu-1.1.0"
_PROBABILITY_BUCKET_WIDTH = 0.05
_FRP_COLLECTIONS = {"sentinel-3-sl-2-frp-nrt", "sentinel-3-sl-2-frp-ntc"}
_FRP_ASSETS_BY_COLLECTION = {
    "sentinel-3-sl-2-frp-nrt": ("FRP_MWIR1km_STANDARD",),
    "sentinel-3-sl-2-frp-ntc": ("FRP_in",),
}
_PROCESSOR_REVISIONS = {
    _CLMS_PROCESSOR: _CLMS_REVISION,
    _S1_PROCESSOR: _S1_REVISION,
    _S2_PROCESSOR: _S2_REVISION,
    _FRP_PROCESSOR: _FRP_REVISION,
}
_MAX_FRP_SAMPLES = 500_000
_DEFAULT_S2_MAXIMUM_DOWNLOAD_BYTES = 512 * 1_024 * 1_024


class SatelliteObservationAsset(StrictModel):
    asset_name: SafeIdentifierV2
    object_uri: str = Field(pattern=r"^s3://eodata/[A-Za-z0-9._~!$&'()*+,;=:@%/-]+$")
    media_type: str = Field(min_length=3, max_length=255)
    file_size_bytes: int = Field(gt=0, le=2_147_483_648)
    file_checksum: str = Field(pattern=r"^[0-9a-f]{32,128}$")
    proj_code: str | None = Field(default=None, min_length=3, max_length=128)
    proj_shape: tuple[int, int] | None = None
    proj_transform: tuple[float, float, float, float, float, float] | None = None
    nodata: float | None = Field(default=None, allow_inf_nan=False)
    data_type: str | None = Field(default=None, min_length=2, max_length=32)
    raster_scale: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_grid(self) -> SatelliteObservationAsset:
        grid_values = (
            self.proj_code,
            self.proj_shape,
            self.proj_transform,
            self.nodata,
            self.data_type,
            self.raster_scale,
        )
        if self.asset_name in set(_CLMS_ASSETS) | set(_S2_ASSETS):
            if any(value is None for value in grid_values):
                raise ValueError("raster processing asset has no complete grid")
            if self.asset_name in _CLMS_ASSETS and self.proj_code != "EPSG:4326":
                raise ValueError("CLMS processing asset has no complete EPSG:4326 grid")
            if self.asset_name in _S2_ASSETS and not self.object_uri.startswith(
                "s3://eodata/Sentinel-2/MSI/L2A/"
            ):
                raise ValueError("Sentinel-2 processing asset has an invalid object path")
        elif any(value is not None for value in grid_values):
            raise ValueError("non-raster satellite asset unexpectedly exposes a grid")
        if self.asset_name in {"FRP_in", "FRP_MWIR1km_STANDARD"} and (
            self.file_size_bytes > 256 * 1_024 * 1_024
        ):
            raise ValueError("Sentinel-3 FRP processing asset exceeds 256 MiB")
        return self

    @property
    def s3_key(self) -> str:
        parsed = urlsplit(self.object_uri)
        if parsed.scheme != "s3" or parsed.netloc != "eodata":
            raise SatelliteCpuError("cdse_satellite_asset_uri_invalid", retryable=False)
        return parsed.path.lstrip("/")


class CdseObservationS3Config(StrictModel):
    endpoint_url: Literal["https://eodata.dataspace.copernicus.eu"] = (
        "https://eodata.dataspace.copernicus.eu"
    )
    access_key: SecretStr = Field(min_length=8, max_length=512)
    secret_key: SecretStr = Field(min_length=16, max_length=512)
    region_name: str = Field(default="eu-central-1", min_length=3, max_length=64)
    maximum_window_pixels: int = Field(default=4_000_000, ge=256, le=25_000_000)
    sentinel2_maximum_download_bytes: int = Field(
        default=_DEFAULT_S2_MAXIMUM_DOWNLOAD_BYTES,
        ge=1_024,
        le=2 * 1_024 * 1_024 * 1_024,
    )
    openeo_invocation_enabled: bool = False
    openeo_access_token: SecretStr | None = Field(default=None, min_length=32, max_length=8_192)
    openeo_maximum_authorized_credits: float = Field(default=0, ge=0, le=100)
    openeo_timeout_seconds: float = Field(default=120, ge=10, le=300)
    openeo_maximum_response_bytes: int = Field(
        default=128 * 1_024 * 1_024,
        ge=1_024,
        le=512 * 1_024 * 1_024,
    )

    @model_validator(mode="after")
    def validate_openeo_gate(self) -> CdseObservationS3Config:
        if self.openeo_invocation_enabled and (
            self.openeo_access_token is None or self.openeo_maximum_authorized_credits <= 0
        ):
            raise ValueError("openEO invocation requires a token and a positive credit ceiling")
        if not self.openeo_invocation_enabled and self.openeo_maximum_authorized_credits != 0:
            raise ValueError("disabled openEO invocation cannot authorize credits")
        return self


@dataclass(frozen=True, slots=True)
class SatelliteAssetReceipt:
    asset_name: str
    source_checksum: str
    derived_content_sha256: str
    bytes_read: int

    def as_payload(self, *, source_artifact_revision_id: str | None = None) -> dict[str, Any]:
        payload = {
            "asset_name": self.asset_name,
            "source_checksum": self.source_checksum,
            "derived_content_sha256": self.derived_content_sha256,
            "bytes_read": self.bytes_read,
        }
        if source_artifact_revision_id is not None:
            payload["source_artifact_revision_id"] = source_artifact_revision_id
        return payload


@dataclass(frozen=True, slots=True)
class ClmsRasterWindow:
    day_of_burn: np.ndarray[Any, Any]
    burn_probability: np.ndarray[Any, Any]
    burn_fraction: np.ndarray[Any, Any]
    valid_masks: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
    transform: Any
    receipts: tuple[SatelliteAssetReceipt, ...]


@dataclass(frozen=True, slots=True)
class Sentinel2ChangeWindow:
    pre: Mapping[str, np.ndarray[Any, Any]]
    post: Mapping[str, np.ndarray[Any, Any]]
    transform: Any
    crs: str
    receipts_pre: tuple[SatelliteAssetReceipt, ...]
    receipts_post: tuple[SatelliteAssetReceipt, ...]


@dataclass(frozen=True, slots=True)
class Sentinel1ChangeWindow:
    pre: Mapping[str, np.ndarray[Any, Any]]
    post: Mapping[str, np.ndarray[Any, Any]]
    transform: Any
    crs: str
    receipts_pre: tuple[SatelliteAssetReceipt, ...]
    receipts_post: tuple[SatelliteAssetReceipt, ...]


@dataclass(frozen=True, slots=True)
class Sentinel2ObservationOutcome:
    observations: tuple[dict[str, Any], ...]
    valid_coverage_geojson: dict[str, Any] | None
    coverage_metrics: dict[str, float | int]


@dataclass(frozen=True, slots=True)
class Sentinel1ObservationOutcome:
    observations: tuple[dict[str, Any], ...]
    valid_coverage_geojson: dict[str, Any] | None
    coverage_metrics: dict[str, float | int]


class SatelliteObservationAssetReader(Protocol):
    def read_clms_window(
        self,
        *,
        assets: tuple[SatelliteObservationAsset, ...],
        bbox: tuple[float, float, float, float],
    ) -> ClmsRasterWindow: ...

    def fetch_frp_file(
        self,
        *,
        asset: SatelliteObservationAsset,
        output_path: Path,
    ) -> SatelliteAssetReceipt: ...

    def read_sentinel2_change_window(
        self,
        *,
        reference_assets: tuple[SatelliteObservationAsset, ...],
        observation_assets: tuple[SatelliteObservationAsset, ...],
        bbox: tuple[float, float, float, float],
    ) -> Sentinel2ChangeWindow: ...

    def read_sentinel1_change_window(
        self,
        *,
        reference_artifact: BackendIncidentDaySatelliteArtifact,
        observation_artifact: BackendIncidentDaySatelliteArtifact,
        bbox: tuple[float, float, float, float],
    ) -> Sentinel1ChangeWindow: ...


def _window_digest(
    *, asset: SatelliteObservationAsset, array: np.ndarray[Any, Any], transform: Any
) -> str:
    digest = sha256()
    digest.update(asset.asset_name.encode())
    digest.update(asset.file_checksum.encode())
    digest.update(str(tuple(float(value) for value in transform.to_gdal())).encode())
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validate_clms_window_size(*, width: float, height: float, maximum_pixels: int) -> None:
    pixels = math.ceil(width) * math.ceil(height)
    if pixels <= 0:
        raise SatelliteCpuError("clms_satellite_window_empty", retryable=False)
    if pixels > maximum_pixels:
        raise SatelliteCpuError("clms_satellite_window_too_large", retryable=False)


class CdseS3ObservationAssetReader:
    """Read bounded COG windows and one small FRP NetCDF from official CDSE S3."""

    def __init__(
        self,
        config: CdseObservationS3Config,
        *,
        s3_client: Any | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        session = boto3.Session(
            aws_access_key_id=config.access_key.get_secret_value(),
            aws_secret_access_key=config.secret_key.get_secret_value(),
            region_name=config.region_name,
        )
        self._session = session
        self._s3 = s3_client or session.client(
            "s3",
            endpoint_url=config.endpoint_url,
            config=BotocoreConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                retries={"mode": "adaptive", "total_max_attempts": 5},
                connect_timeout=10,
                read_timeout=120,
            ),
        )
        self._owns_http_client = http_client is None
        self._http = http_client or httpx.Client()

    def close(self) -> None:
        """Release the internally owned HTTP client used for openEO calls."""

        if self._owns_http_client:
            self._http.close()

    @staticmethod
    def _openeo_process_graph(
        *,
        bbox: tuple[float, float, float, float],
        temporal_extent: tuple[datetime, datetime],
        width: int,
        height: int,
    ) -> dict[str, Any]:
        west, south, east, north = bbox
        start_at, end_at = temporal_extent
        return {
            "process_graph": {
                "load": {
                    "process_id": "load_collection",
                    "arguments": {
                        "id": "sentinel-1-grd",
                        "spatial_extent": {
                            "west": west,
                            "south": south,
                            "east": east,
                            "north": north,
                            "width": width,
                            "height": height,
                        },
                        "temporal_extent": [
                            start_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                            end_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                        ],
                        "bands": ["VV", "VH"],
                    },
                },
                "backscatter": {
                    "process_id": "sar_backscatter",
                    "arguments": {
                        "data": {"from_node": "load"},
                        "coefficient": "sigma0-ellipsoid",
                        "elevation_model": "COPERNICUS_30",
                        "local_incidence_angle": False,
                    },
                },
                "median": {
                    "process_id": "reduce_dimension",
                    "arguments": {
                        "data": {"from_node": "backscatter"},
                        "dimension": "t",
                        "reducer": {
                            "process_graph": {
                                "median": {
                                    "process_id": "median",
                                    "arguments": {"data": {"from_parameter": "data"}},
                                    "result": True,
                                }
                            }
                        },
                    },
                },
                "save": {
                    "process_id": "save_result",
                    "arguments": {
                        "data": {"from_node": "median"},
                        "format": "GTiff",
                    },
                    "result": True,
                },
            }
        }

    def _openeo_raster(
        self,
        *,
        artifact: BackendIncidentDaySatelliteArtifact,
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], Any, str, SatelliteAssetReceipt]:
        from rasterio.io import MemoryFile

        if not self.config.openeo_invocation_enabled or self.config.openeo_access_token is None:
            raise SatelliteCpuError("cdse_openeo_invocation_disabled", retryable=True)
        start_at = artifact.acquisition_start_at
        if start_at is None:
            raise SatelliteCpuError("sentinel1_acquisition_time_missing", retryable=False)
        end_at = artifact.acquisition_end_at or (start_at + timedelta(minutes=10))
        if end_at <= start_at or end_at - start_at > timedelta(days=1):
            raise SatelliteCpuError("sentinel1_acquisition_window_invalid", retryable=False)
        payload = {
            "process": self._openeo_process_graph(
                bbox=bbox,
                temporal_extent=(start_at, end_at),
                width=width,
                height=height,
            )
        }
        try:
            response = self._http.post(
                "https://openeosh.dataspace.copernicus.eu/1.2/result",
                json=payload,
                headers={
                    "Accept": "image/tiff",
                    "Accept-Encoding": "identity",
                    "Authorization": (
                        "Bearer " + self.config.openeo_access_token.get_secret_value()
                    ),
                    "User-Agent": "FireViewer-satellite-cpu/1",
                },
                timeout=self.config.openeo_timeout_seconds,
                follow_redirects=False,
            )
        except httpx.TimeoutException as exc:
            raise SatelliteCpuError("cdse_openeo_timeout", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise SatelliteCpuError("cdse_openeo_network_error", retryable=True) from exc
        if response.status_code in {401, 403}:
            raise SatelliteCpuError("cdse_openeo_authentication_failed", retryable=True)
        if response.status_code == 429:
            raise SatelliteCpuError("cdse_openeo_rate_limited", retryable=True)
        if not 200 <= response.status_code < 300:
            raise SatelliteCpuError("cdse_openeo_processing_failed", retryable=True)
        media_type = response.headers.get("content-type", "").partition(";")[0].casefold()
        if media_type not in {"image/tiff", "image/geotiff"}:
            raise SatelliteCpuError("cdse_openeo_content_type_invalid", retryable=False)
        content = response.content
        if not content or len(content) > self.config.openeo_maximum_response_bytes:
            raise SatelliteCpuError("cdse_openeo_response_size_invalid", retryable=False)
        try:
            with MemoryFile(content) as memory, memory.open() as dataset:
                if (
                    dataset.count != 2
                    or dataset.width != width
                    or dataset.height != height
                    or dataset.crs is None
                ):
                    raise SatelliteCpuError("cdse_openeo_raster_contract_invalid", retryable=False)
                arrays = {
                    "VV": np.asarray(dataset.read(1), dtype=np.float64),
                    "VH": np.asarray(dataset.read(2), dtype=np.float64),
                }
                transform = dataset.transform
                crs = str(dataset.crs)
        except SatelliteCpuError:
            raise
        except Exception as exc:
            raise SatelliteCpuError("cdse_openeo_raster_invalid", retryable=False) from exc
        return (
            arrays,
            transform,
            crs,
            SatelliteAssetReceipt(
                asset_name="openeo_vv_vh",
                source_checksum=artifact.content_hash,
                derived_content_sha256=sha256(content).hexdigest(),
                bytes_read=len(content),
            ),
        )

    def read_sentinel1_change_window(
        self,
        *,
        reference_artifact: BackendIncidentDaySatelliteArtifact,
        observation_artifact: BackendIncidentDaySatelliteArtifact,
        bbox: tuple[float, float, float, float],
    ) -> Sentinel1ChangeWindow:
        west, south, east, north = bbox
        mean_latitude = (south + north) / 2
        width = max(
            1,
            math.ceil((east - west) * 111_320 * math.cos(math.radians(mean_latitude)) / 20),
        )
        height = max(1, math.ceil((north - south) * 110_540 / 20))
        if width > 2_500 or height > 2_500 or width * height > self.config.maximum_window_pixels:
            raise SatelliteCpuError("cdse_openeo_window_too_large", retryable=False)
        pre, pre_transform, pre_crs, pre_receipt = self._openeo_raster(
            artifact=reference_artifact,
            bbox=bbox,
            width=width,
            height=height,
        )
        post, post_transform, post_crs, post_receipt = self._openeo_raster(
            artifact=observation_artifact,
            bbox=bbox,
            width=width,
            height=height,
        )
        if pre_transform != post_transform or pre_crs != post_crs:
            raise SatelliteCpuError("cdse_openeo_grid_mismatch", retryable=False)
        return Sentinel1ChangeWindow(
            pre=pre,
            post=post,
            transform=pre_transform,
            crs=pre_crs,
            receipts_pre=(pre_receipt,),
            receipts_post=(post_receipt,),
        )

    def read_clms_window(
        self,
        *,
        assets: tuple[SatelliteObservationAsset, ...],
        bbox: tuple[float, float, float, float],
    ) -> ClmsRasterWindow:
        import rasterio
        from rasterio.session import AWSSession
        from rasterio.windows import Window, from_bounds

        if tuple(item.asset_name for item in assets) != _CLMS_ASSETS:
            raise SatelliteCpuError("clms_satellite_asset_set_invalid", retryable=False)
        arrays: list[np.ndarray[Any, Any]] = []
        masks: list[np.ndarray[Any, Any]] = []
        receipts: list[SatelliteAssetReceipt] = []
        common_transform: Any | None = None
        common_shape: tuple[int, int] | None = None
        aws_session = AWSSession(session=self._session, requester_pays=False)
        endpoint_host = urlsplit(self.config.endpoint_url).hostname
        try:
            with rasterio.Env(
                aws_session,
                AWS_S3_ENDPOINT=endpoint_host,
                AWS_HTTPS="YES",
                AWS_VIRTUAL_HOSTING="FALSE",
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            ):
                for asset in assets:
                    with rasterio.open(f"/vsis3/eodata/{asset.s3_key}") as dataset:
                        if str(dataset.crs) != asset.proj_code or dataset.count != 1:
                            raise SatelliteCpuError("clms_satellite_grid_mismatch", retryable=False)
                        requested = from_bounds(*bbox, transform=dataset.transform)
                        requested = requested.round_offsets().round_lengths()
                        try:
                            window = requested.intersection(
                                Window(0, 0, dataset.width, dataset.height)
                            )
                        except rasterio.errors.WindowError as exc:
                            raise SatelliteCpuError(
                                "incident_outside_clms_product", retryable=False
                            ) from exc
                        _validate_clms_window_size(
                            width=window.width,
                            height=window.height,
                            maximum_pixels=self.config.maximum_window_pixels,
                        )
                        masked = dataset.read(1, window=window, masked=True)
                        transform = dataset.window_transform(window)
                        if masked.size == 0:
                            raise SatelliteCpuError("clms_satellite_window_empty", retryable=False)
                        if common_shape is None:
                            common_shape = cast(tuple[int, int], masked.shape)
                            common_transform = transform
                        elif masked.shape != common_shape or transform != common_transform:
                            raise SatelliteCpuError("clms_satellite_grid_mismatch", retryable=False)
                        scale = float(asset.raster_scale or 1.0)
                        values = np.asarray(masked.filled(0), dtype=np.float64) * scale
                        valid = ~np.ma.getmaskarray(masked)
                        arrays.append(values)
                        masks.append(valid)
                        receipts.append(
                            SatelliteAssetReceipt(
                                asset_name=asset.asset_name,
                                source_checksum=asset.file_checksum,
                                derived_content_sha256=_window_digest(
                                    asset=asset, array=np.asarray(masked.data), transform=transform
                                ),
                                bytes_read=int(masked.data.nbytes),
                            )
                        )
        except SatelliteCpuError:
            raise
        except (BotoCoreError, ClientError, OSError, rasterio.errors.RasterioError) as exc:
            raise SatelliteCpuError("cdse_clms_read_failed", retryable=True) from exc
        if common_transform is None or len(arrays) != 3:
            raise SatelliteCpuError("clms_satellite_window_incomplete", retryable=False)
        return ClmsRasterWindow(
            day_of_burn=arrays[0],
            burn_probability=arrays[1],
            burn_fraction=arrays[2],
            valid_masks=cast(
                tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]],
                tuple(masks),
            ),
            transform=common_transform,
            receipts=tuple(receipts),
        )

    def fetch_frp_file(
        self,
        *,
        asset: SatelliteObservationAsset,
        output_path: Path,
    ) -> SatelliteAssetReceipt:
        digest = sha256()
        size = 0
        try:
            response = self._s3.get_object(Bucket="eodata", Key=asset.s3_key)
            # Close the response even when a size guard interrupts the download.
            with response["Body"] as body, output_path.open("wb") as stream:
                while True:
                    chunk = body.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > asset.file_size_bytes:
                        raise SatelliteCpuError("sentinel3_frp_size_mismatch", retryable=False)
                    digest.update(chunk)
                    stream.write(chunk)
        except SatelliteCpuError:
            output_path.unlink(missing_ok=True)
            raise
        except (BotoCoreError, ClientError, KeyError, OSError) as exc:
            output_path.unlink(missing_ok=True)
            raise SatelliteCpuError("cdse_sentinel3_frp_read_failed", retryable=True) from exc
        if size != asset.file_size_bytes:
            output_path.unlink(missing_ok=True)
            raise SatelliteCpuError("sentinel3_frp_size_mismatch", retryable=False)
        return SatelliteAssetReceipt(
            asset_name=asset.asset_name,
            source_checksum=asset.file_checksum,
            derived_content_sha256=digest.hexdigest(),
            bytes_read=size,
        )

    def _materialize_sentinel2_asset(
        self,
        *,
        asset: SatelliteObservationAsset,
        output_path: Path,
    ) -> None:
        body: Any | None = None
        size = 0
        try:
            response = self._s3.get_object(Bucket="eodata", Key=asset.s3_key)
            body = response["Body"]
            with output_path.open("xb") as stream:
                while True:
                    chunk = body.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > asset.file_size_bytes:
                        raise SatelliteCpuError(
                            "sentinel2_asset_size_exceeds_manifest",
                            retryable=False,
                        )
                    stream.write(chunk)
        except SatelliteCpuError:
            output_path.unlink(missing_ok=True)
            raise
        except (BotoCoreError, ClientError, KeyError, OSError) as exc:
            output_path.unlink(missing_ok=True)
            raise SatelliteCpuError("cdse_sentinel2_read_failed", retryable=True) from exc
        finally:
            if body is not None:
                body.close()
        if size != asset.file_size_bytes:
            output_path.unlink(missing_ok=True)
            raise SatelliteCpuError("sentinel2_asset_download_incomplete", retryable=True)

    def read_sentinel2_change_window(
        self,
        *,
        reference_assets: tuple[SatelliteObservationAsset, ...],
        observation_assets: tuple[SatelliteObservationAsset, ...],
        bbox: tuple[float, float, float, float],
    ) -> Sentinel2ChangeWindow:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import transform_bounds
        from rasterio.windows import Window, from_bounds

        if (
            tuple(item.asset_name for item in reference_assets) != _S2_ASSETS
            or tuple(item.asset_name for item in observation_assets) != _S2_ASSETS
        ):
            raise SatelliteCpuError("sentinel2_change_asset_set_invalid", retryable=False)
        requested_download_bytes = sum(
            item.file_size_bytes for item in (*reference_assets, *observation_assets)
        )
        if requested_download_bytes > self.config.sentinel2_maximum_download_bytes:
            raise SatelliteCpuError("sentinel2_download_too_large", retryable=False)
        try:
            with TemporaryDirectory(prefix="fireviewer-sentinel2-assets-") as temp_value:
                directory = Path(temp_value)
                paths: dict[str, dict[str, Path]] = {"pre": {}, "post": {}}
                for temporal_role, assets in (
                    ("pre", reference_assets),
                    ("post", observation_assets),
                ):
                    for index, asset in enumerate(assets):
                        output_path = directory / f"{temporal_role}-{index}-{asset.asset_name}.jp2"
                        self._materialize_sentinel2_asset(asset=asset, output_path=output_path)
                        paths[temporal_role][asset.asset_name] = output_path

                with ExitStack() as stack:
                    reference = {
                        asset_name: stack.enter_context(rasterio.open(path))
                        for asset_name, path in paths["pre"].items()
                    }
                    observation = {
                        asset_name: stack.enter_context(rasterio.open(path))
                        for asset_name, path in paths["post"].items()
                    }
                    target = observation["B11_20m"]
                    if target.crs is None:
                        raise SatelliteCpuError("sentinel2_target_crs_missing", retryable=False)
                    target_bounds = transform_bounds("EPSG:4326", target.crs, *bbox, densify_pts=21)
                    requested = from_bounds(*target_bounds, transform=target.transform)
                    requested = requested.round_offsets().round_lengths()
                    try:
                        window = requested.intersection(Window(0, 0, target.width, target.height))
                    except rasterio.errors.WindowError as exc:
                        raise SatelliteCpuError(
                            "incident_outside_sentinel2_product", retryable=False
                        ) from exc
                    _validate_clms_window_size(
                        width=window.width,
                        height=window.height,
                        maximum_pixels=self.config.maximum_window_pixels,
                    )
                    width, height = int(window.width), int(window.height)
                    transform = target.window_transform(window)
                    target_crs = target.crs

                    def aligned(
                        datasets: Mapping[str, Any],
                        assets: tuple[SatelliteObservationAsset, ...],
                    ) -> tuple[dict[str, np.ndarray[Any, Any]], tuple[SatelliteAssetReceipt, ...]]:
                        arrays: dict[str, np.ndarray[Any, Any]] = {}
                        receipts: list[SatelliteAssetReceipt] = []
                        assets_by_name = {item.asset_name: item for item in assets}
                        for asset_name in _S2_ASSETS:
                            dataset = datasets[asset_name]
                            asset = assets_by_name[asset_name]
                            if (
                                dataset.crs is None
                                or str(dataset.crs) != asset.proj_code
                                or dataset.count != 1
                                or dataset.crs != target_crs
                                or dataset.transform != target.transform
                                or dataset.width != target.width
                                or dataset.height != target.height
                            ):
                                raise SatelliteCpuError(
                                    "sentinel2_change_grid_mismatch", retryable=False
                                )
                            destination = dataset.read(
                                1,
                                window=window,
                                out_shape=(height, width),
                                out_dtype="float32",
                                resampling=(
                                    Resampling.nearest
                                    if asset_name == "SCL_20m"
                                    else Resampling.bilinear
                                ),
                            )
                            if not np.isfinite(destination).all():
                                raise SatelliteCpuError(
                                    "sentinel2_window_contains_non_finite_values",
                                    retryable=False,
                                )
                            arrays[asset_name] = destination
                            receipts.append(
                                SatelliteAssetReceipt(
                                    asset_name=asset_name,
                                    source_checksum=asset.file_checksum,
                                    derived_content_sha256=_window_digest(
                                        asset=asset, array=destination, transform=transform
                                    ),
                                    bytes_read=int(destination.nbytes),
                                )
                            )
                        return arrays, tuple(receipts)

                    pre, receipts_pre = aligned(reference, reference_assets)
                    post, receipts_post = aligned(observation, observation_assets)
        except SatelliteCpuError:
            raise
        except (BotoCoreError, ClientError, OSError, rasterio.errors.RasterioError) as exc:
            raise SatelliteCpuError("cdse_sentinel2_read_failed", retryable=True) from exc
        return Sentinel2ChangeWindow(
            pre=pre,
            post=post,
            transform=transform,
            crs=str(target_crs),
            receipts_pre=receipts_pre,
            receipts_post=receipts_post,
        )


def _artifact_assets(
    artifact: BackendIncidentDaySatelliteArtifact,
) -> tuple[str, tuple[SatelliteObservationAsset, ...]]:
    processor = artifact.quality_flags.get("satellite_observation_processor")
    raw_assets = artifact.quality_flags.get("satellite_observation_assets")
    if processor not in {
        _CLMS_PROCESSOR,
        _S1_PROCESSOR,
        _S2_PROCESSOR,
        _FRP_PROCESSOR,
    } or not isinstance(raw_assets, list):
        raise SatelliteCpuError("satellite_observation_manifest_missing", retryable=False)
    try:
        assets = tuple(SatelliteObservationAsset.model_validate(item) for item in raw_assets)
    except ValueError as exc:
        raise SatelliteCpuError("satellite_observation_manifest_invalid", retryable=False) from exc
    expected: tuple[str, ...]
    if processor == _CLMS_PROCESSOR:
        expected = _CLMS_ASSETS
    elif processor == _S1_PROCESSOR:
        expected = _S1_ASSETS
    elif processor == _S2_PROCESSOR:
        expected = _S2_ASSETS
    else:
        expected = _FRP_ASSETS_BY_COLLECTION.get(artifact.collection_key, ())
    if tuple(item.asset_name for item in assets) != expected:
        raise SatelliteCpuError("satellite_observation_asset_set_invalid", retryable=False)
    return str(processor), assets


def _processing_context_sha256(
    *,
    durable: DurableEventEvidence,
    artifact_id: str,
    reference_artifact_id: str | None,
    processor: str,
    processor_revision: str,
) -> str:
    payload = {
        "schema": "fireviewer.satellite-observation-processing-context.v1",
        "analysis_id": durable.event.event_id,
        "local_date": (
            durable.incident_day_local_date.isoformat()
            if durable.incident_day_local_date is not None
            else None
        ),
        "incident_bbox": (
            [float(value) for value in durable.incident_day_bbox]
            if durable.incident_day_bbox is not None
            else None
        ),
        "artifact_revision_id": artifact_id,
        "reference_artifact_revision_id": reference_artifact_id,
        "processor": processor,
        "processor_revision": processor_revision,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _result_id(
    analysis_id: str,
    artifact_id: str,
    processor: str,
    processor_revision: str,
    processing_context_sha256: str,
) -> str:
    digest = sha256(
        (
            f"{analysis_id}:{artifact_id}:{processor}:"
            f"{processor_revision}:{processing_context_sha256}"
        ).encode()
    ).hexdigest()
    return f"SATOBS-{digest[:24]}"


def _current_completed_artifact_ids(
    durable: DurableEventEvidence,
    eligible: list[BackendIncidentDaySatelliteArtifact],
) -> set[str]:
    completed: set[str] = set()
    for artifact in eligible:
        processor = artifact.quality_flags.get("satellite_observation_processor")
        if not isinstance(processor, str):
            continue
        revision = _PROCESSOR_REVISIONS.get(processor)
        if revision is None:
            continue
        reference_id: str | None = None
        if processor in {_S1_PROCESSOR, _S2_PROCESSOR}:
            references = [
                item
                for item in durable.satellite_artifact_tickets
                if item.quality_flags.get("satellite_observation_processor") == processor
                and item.quality_flags.get("temporal_role") == "pre_fire_reference"
            ]
            if len(references) != 1:
                continue
            reference_id = references[0].artifact_revision_id
        context_sha256 = _processing_context_sha256(
            durable=durable,
            artifact_id=artifact.artifact_revision_id,
            reference_artifact_id=reference_id,
            processor=processor,
            processor_revision=revision,
        )
        if any(
            batch.artifact_revision_id == artifact.artifact_revision_id
            and batch.reference_artifact_revision_id == reference_id
            and batch.processor == processor
            and batch.processor_revision == revision
            and batch.processing_context_sha256 == context_sha256
            for batch in durable.satellite_observation_batches
        ):
            completed.add(artifact.artifact_revision_id)
    return completed


def _observation_time(
    durable: DurableEventEvidence, artifact: BackendIncidentDaySatelliteArtifact
) -> datetime:
    start = durable.event.time_window.from_at
    end = durable.event.time_window.to_at
    if start is None or end is None:
        raise SatelliteCpuError("satellite_incident_day_time_missing", retryable=False)
    acquired = artifact.acquisition_start_at
    if acquired is not None and start <= acquired < end:
        return acquired
    return start + ((end - start) / 2)


def _probability_bucket_geometries(
    *,
    probability: np.ndarray[Any, Any],
    selected: np.ndarray[Any, Any],
    transform: Any,
    crs: str,
    incident_bbox: tuple[float, float, float, float],
    bucket_width: float = _PROBABILITY_BUCKET_WIDTH,
) -> tuple[tuple[int, float, int, Any], ...]:
    """Preserve spatial probability variation with a bounded polygon set."""

    from rasterio.features import shapes
    from rasterio.warp import transform_geom
    from shapely.geometry import box, shape
    from shapely.ops import unary_union

    if not 0 < bucket_width <= 0.25:
        raise ValueError("probability bucket width is outside the reviewed range")
    if probability.shape != selected.shape or not np.isfinite(probability[selected]).all():
        raise SatelliteCpuError("satellite_probability_grid_invalid", retryable=False)
    bucket_count = round(1.0 / bucket_width)
    bucket_index = np.zeros(probability.shape, dtype=np.uint8)
    bucket_index[selected] = np.clip(
        np.floor(probability[selected] / bucket_width).astype(np.int32) + 1,
        1,
        bucket_count,
    ).astype(np.uint8)
    incident_extent = box(*incident_bbox)
    results: list[tuple[int, float, int, Any]] = []
    for index in sorted(int(item) for item in np.unique(bucket_index[selected])):
        bucket_mask = selected & (bucket_index == index)
        polygons = []
        for raw_geometry, value in shapes(
            bucket_mask.astype(np.uint8), mask=bucket_mask, transform=transform
        ):
            if int(value) != 1:
                continue
            projected = transform_geom(crs, "EPSG:4326", raw_geometry, precision=7)
            clipped = shape(projected).intersection(incident_extent)
            if not clipped.is_empty:
                polygons.append(clipped)
        if not polygons:
            continue
        geometry = unary_union(polygons)
        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise SatelliteCpuError("satellite_probability_geometry_invalid", retryable=False)
        results.append(
            (
                index,
                float(np.mean(probability[bucket_mask])),
                int(np.count_nonzero(bucket_mask)),
                geometry,
            )
        )
    return tuple(results)


def _sentinel1_vvvh_observations(
    *,
    durable: DurableEventEvidence,
    artifact: BackendIncidentDaySatelliteArtifact,
    window: Sentinel1ChangeWindow,
    vv_change_threshold_db: float,
    vh_change_threshold_db: float,
) -> Sentinel1ObservationOutcome:
    if durable.incident_day_bbox is None:
        raise SatelliteCpuError("satellite_incident_day_required", retryable=False)
    return sentinel1_vvvh_from_window(
        incident_bbox=durable.incident_day_bbox,
        observed_at=_observation_time(durable, artifact),
        source_revision_id=artifact.artifact_revision_id,
        resolution_m=float(artifact.resolution_m or 40),
        window=window,
        vv_change_threshold_db=vv_change_threshold_db,
        vh_change_threshold_db=vh_change_threshold_db,
    )


def sentinel1_vvvh_from_window(
    *,
    incident_bbox: tuple[float, float, float, float],
    observed_at: datetime,
    source_revision_id: str,
    resolution_m: float,
    window: Sentinel1ChangeWindow,
    vv_change_threshold_db: float = 1.5,
    vh_change_threshold_db: float = 1.5,
) -> Sentinel1ObservationOutcome:
    """Shared, uncalibrated SAR support; an absence of change is not a negative."""
    from rasterio.features import shapes
    from rasterio.warp import transform_geom
    from shapely.geometry import box, mapping, shape
    from shapely.ops import unary_union

    pre_vv = np.asarray(window.pre["VV"], dtype=np.float64)
    pre_vh = np.asarray(window.pre["VH"], dtype=np.float64)
    post_vv = np.asarray(window.post["VV"], dtype=np.float64)
    post_vh = np.asarray(window.post["VH"], dtype=np.float64)
    if not (pre_vv.shape == pre_vh.shape == post_vv.shape == post_vh.shape):
        raise SatelliteCpuError("sentinel1_change_shape_mismatch", retryable=False)
    valid = (
        np.isfinite(pre_vv)
        & np.isfinite(pre_vh)
        & np.isfinite(post_vv)
        & np.isfinite(post_vh)
        & (pre_vv > 0)
        & (pre_vh > 0)
        & (post_vv > 0)
        & (post_vh > 0)
    )
    valid_pixel_count = int(np.count_nonzero(valid))
    if valid_pixel_count == 0:
        return Sentinel1ObservationOutcome(
            observations=(),
            valid_coverage_geojson=None,
            coverage_metrics={"valid_pixel_count": 0, "invalid_fraction": 1.0},
        )
    incident_extent = box(*incident_bbox)
    coverage_polygons = []
    for raw_geometry, value in shapes(
        valid.astype(np.uint8), mask=valid, transform=window.transform
    ):
        if int(value) != 1:
            continue
        projected = transform_geom(window.crs, "EPSG:4326", raw_geometry, precision=7)
        clipped = shape(projected).intersection(incident_extent)
        if not clipped.is_empty:
            coverage_polygons.append(clipped)
    valid_coverage = unary_union(coverage_polygons) if coverage_polygons else None
    if valid_coverage is not None and valid_coverage.geom_type not in {"Polygon", "MultiPolygon"}:
        raise SatelliteCpuError("sentinel1_coverage_geometry_invalid", retryable=False)
    coverage_geojson = mapping(valid_coverage) if valid_coverage is not None else None
    coverage_metrics: dict[str, float | int] = {
        "valid_pixel_count": valid_pixel_count,
        "invalid_fraction": float(1 - (valid_pixel_count / valid.size)),
    }
    epsilon = np.finfo(np.float64).tiny
    pre_vv_db = 10 * np.log10(np.maximum(pre_vv, epsilon))
    pre_vh_db = 10 * np.log10(np.maximum(pre_vh, epsilon))
    post_vv_db = 10 * np.log10(np.maximum(post_vv, epsilon))
    post_vh_db = 10 * np.log10(np.maximum(post_vh, epsilon))
    vv_change = np.abs(post_vv_db - pre_vv_db)
    vh_change = np.abs(post_vh_db - pre_vh_db)
    magnitude = np.hypot(vv_change, vh_change)
    selected = valid & (vv_change >= vv_change_threshold_db) & (vh_change >= vh_change_threshold_db)
    changed_pixel_count = int(np.count_nonzero(selected))
    if changed_pixel_count == 0:
        return Sentinel1ObservationOutcome(
            observations=(),
            valid_coverage_geojson=coverage_geojson,
            coverage_metrics={**coverage_metrics, "changed_pixel_count": 0},
        )
    probability = np.clip(
        0.25
        + 0.10
        * (
            (vv_change / max(vv_change_threshold_db, 0.1))
            + (vh_change / max(vh_change_threshold_db, 0.1))
        ),
        0,
        0.45,
    )
    buckets = _probability_bucket_geometries(
        probability=probability,
        selected=selected,
        transform=window.transform,
        crs=window.crs,
        incident_bbox=incident_bbox,
    )
    if not buckets:
        return Sentinel1ObservationOutcome(
            observations=(),
            valid_coverage_geojson=coverage_geojson,
            coverage_metrics={**coverage_metrics, "changed_pixel_count": changed_pixel_count},
        )
    observations = []
    for bucket_index, bucket_probability, bucket_pixels, geometry in buckets:
        # Reduction roundoff must not lift the experimental 0.45 ceiling.
        bucket_probability = min(0.45, bucket_probability)
        observation_digest = sha256(
            (
                source_revision_id
                + geometry.wkb_hex
                + f"{vv_change_threshold_db:.6f}:{vh_change_threshold_db:.6f}:"
                + f"{bucket_index}:{bucket_probability:.6f}"
            ).encode()
        ).hexdigest()
        observations.append(
            {
                "observation_id": f"S1-VVVH-{observation_digest[:24]}",
                "observed_at": observed_at.isoformat(),
                "geometry_geojson": mapping(geometry),
                "coverage_geojson": coverage_geojson,
                "horizontal_accuracy_m": max(40.0, resolution_m),
                "confidence": bucket_probability,
                "metrics": {
                    **coverage_metrics,
                    "changed_pixel_count": changed_pixel_count,
                    "probability_bucket_pixel_count": bucket_pixels,
                    "change_probability_mean": bucket_probability,
                    "change_magnitude_mean_db": float(np.mean(magnitude[selected])),
                    "pre_vv_db_mean": float(np.mean(pre_vv_db[selected])),
                    "post_vv_db_mean": float(np.mean(post_vv_db[selected])),
                    "pre_vh_db_mean": float(np.mean(pre_vh_db[selected])),
                    "post_vh_db_mean": float(np.mean(post_vh_db[selected])),
                    "vv_change_threshold_db": vv_change_threshold_db,
                    "vh_change_threshold_db": vh_change_threshold_db,
                },
            }
        )
    return Sentinel1ObservationOutcome(
        observations=tuple(observations),
        valid_coverage_geojson=coverage_geojson,
        coverage_metrics={**coverage_metrics, "changed_pixel_count": changed_pixel_count},
    )


def _sentinel2_nbr_observations(
    *,
    durable: DurableEventEvidence,
    artifact: BackendIncidentDaySatelliteArtifact,
    window: Sentinel2ChangeWindow,
    dnbr_threshold: float,
    minimum_probability: float,
) -> Sentinel2ObservationOutcome:
    if durable.incident_day_bbox is None:
        raise SatelliteCpuError("satellite_incident_day_required", retryable=False)
    return sentinel2_nbr_from_window(
        incident_bbox=durable.incident_day_bbox,
        observed_at=_observation_time(durable, artifact),
        source_revision_id=artifact.artifact_revision_id,
        resolution_m=float(artifact.resolution_m or 20),
        window=window,
        dnbr_threshold=dnbr_threshold,
        minimum_probability=minimum_probability,
    )


def sentinel2_nbr_from_window(
    *,
    incident_bbox: tuple[float, float, float, float],
    observed_at: datetime,
    source_revision_id: str,
    resolution_m: float,
    window: Sentinel2ChangeWindow,
    dnbr_threshold: float,
    minimum_probability: float,
) -> Sentinel2ObservationOutcome:
    from rasterio.features import shapes
    from rasterio.warp import transform_geom
    from shapely.geometry import box, mapping, shape
    from shapely.ops import unary_union

    pre_scl = np.rint(window.pre["SCL_20m"]).astype(np.int16)
    post_scl = np.rint(window.post["SCL_20m"]).astype(np.int16)
    invalid_scl = np.array([0, 1, 3, 8, 9, 10, 11], dtype=np.int16)
    scl_valid = ~np.isin(pre_scl, invalid_scl) & ~np.isin(post_scl, invalid_scl)
    pre_nir = np.asarray(window.pre["B8A_20m"], dtype=np.float64)
    pre_swir = np.asarray(window.pre["B12_20m"], dtype=np.float64)
    post_nir = np.asarray(window.post["B8A_20m"], dtype=np.float64)
    post_swir = np.asarray(window.post["B12_20m"], dtype=np.float64)
    pre_denominator = pre_nir + pre_swir
    post_denominator = post_nir + post_swir
    spectral_valid = (
        np.isfinite(pre_nir)
        & np.isfinite(pre_swir)
        & np.isfinite(post_nir)
        & np.isfinite(post_swir)
        & (pre_nir >= 0)
        & (pre_swir >= 0)
        & (post_nir >= 0)
        & (post_swir >= 0)
        & (pre_denominator > 0)
        & (post_denominator > 0)
    )
    # Negative BOA values may exist, but cannot form bounded NBR evidence. Do
    # not clip a physically invalid ratio into a maximum-confidence burn mask.
    valid = scl_valid & spectral_valid
    valid_pixel_count = int(np.count_nonzero(valid))
    coverage_metrics: dict[str, float | int] = {
        "valid_pixel_count": valid_pixel_count,
        "valid_fraction": float(valid_pixel_count / valid.size),
        "cloud_fraction": float(
            np.mean(np.isin(pre_scl, [8, 9, 10]) | np.isin(post_scl, [8, 9, 10]))
        ),
        "no_data_fraction": float(np.mean((pre_scl == 0) | (post_scl == 0))),
        "scl_invalid_fraction": float(np.mean(~scl_valid)),
        "spectral_invalid_pixel_count": int(np.count_nonzero(scl_valid & ~spectral_valid)),
    }
    if valid_pixel_count == 0:
        return Sentinel2ObservationOutcome(
            observations=(),
            valid_coverage_geojson=None,
            coverage_metrics=coverage_metrics,
        )
    incident_extent = box(*incident_bbox)
    coverage_polygons = []
    for raw_geometry, value in shapes(
        valid.astype(np.uint8), mask=valid, transform=window.transform
    ):
        if int(value) != 1:
            continue
        projected = transform_geom(window.crs, "EPSG:4326", raw_geometry, precision=7)
        clipped = shape(projected).intersection(incident_extent)
        if not clipped.is_empty:
            coverage_polygons.append(clipped)
    valid_coverage = unary_union(coverage_polygons) if coverage_polygons else None
    if valid_coverage is not None and valid_coverage.geom_type not in {"Polygon", "MultiPolygon"}:
        raise SatelliteCpuError("sentinel2_coverage_geometry_invalid", retryable=False)
    coverage_geojson = mapping(valid_coverage) if valid_coverage is not None else None
    pre_nbr = np.zeros_like(pre_nir)
    post_nbr = np.zeros_like(post_nir)
    np.divide(pre_nir - pre_swir, pre_denominator, out=pre_nbr, where=valid)
    np.divide(post_nir - post_swir, post_denominator, out=post_nbr, where=valid)
    dnbr = pre_nbr - post_nbr
    probability = np.clip(
        0.5 + (dnbr - dnbr_threshold) / max(0.2, 2 * (0.66 - dnbr_threshold)),
        0,
        1,
    )
    # SCL 6 is water, not a burned land surface. Keep its valid coverage, but
    # exclude positive change over water in either acquisition. This is not a
    # valid_negative observation and must not erase a previous burned state.
    water = (pre_scl == 6) | (post_scl == 6)
    spectral_positive = valid & (dnbr >= dnbr_threshold) & (probability >= minimum_probability)
    selected = spectral_positive & ~water
    coverage_metrics["water_pixel_count"] = int(np.count_nonzero(valid & water))
    coverage_metrics["water_positive_excluded_count"] = int(
        np.count_nonzero(spectral_positive & water)
    )
    burned_pixel_count = int(np.count_nonzero(selected))
    if burned_pixel_count == 0:
        return Sentinel2ObservationOutcome(
            observations=(),
            valid_coverage_geojson=coverage_geojson,
            coverage_metrics={**coverage_metrics, "burned_pixel_count": 0},
        )
    buckets = _probability_bucket_geometries(
        probability=probability,
        selected=selected,
        transform=window.transform,
        crs=window.crs,
        incident_bbox=incident_bbox,
    )
    if not buckets:
        return Sentinel2ObservationOutcome(
            observations=(),
            valid_coverage_geojson=coverage_geojson,
            coverage_metrics={**coverage_metrics, "burned_pixel_count": burned_pixel_count},
        )
    observations = []
    for bucket_index, bucket_probability, bucket_pixels, geometry in buckets:
        bucket_mask = selected & (
            np.clip(
                np.floor(probability / _PROBABILITY_BUCKET_WIDTH).astype(np.int32) + 1,
                1,
                20,
            )
            == bucket_index
        )
        observation_digest = sha256(
            (
                source_revision_id
                + geometry.wkb_hex
                + f"{dnbr_threshold:.6f}:{bucket_index}:{bucket_probability:.6f}"
            ).encode()
        ).hexdigest()
        observations.append(
            {
                "observation_id": f"S2-DNBR-{observation_digest[:24]}",
                "observed_at": observed_at.isoformat(),
                "geometry_geojson": mapping(geometry),
                "coverage_geojson": coverage_geojson,
                "horizontal_accuracy_m": max(20.0, float(resolution_m)),
                "confidence": bucket_probability,
                "metrics": {
                    **coverage_metrics,
                    "burned_pixel_count": burned_pixel_count,
                    "probability_bucket_pixel_count": bucket_pixels,
                    "dnbr_mean": float(np.mean(dnbr[bucket_mask])),
                    "dnbr_max": float(np.max(dnbr[bucket_mask])),
                    "pre_nbr_mean": float(np.mean(pre_nbr[bucket_mask])),
                    "post_nbr_mean": float(np.mean(post_nbr[bucket_mask])),
                    "dnbr_threshold": dnbr_threshold,
                    "minimum_probability": minimum_probability,
                },
            }
        )
    return Sentinel2ObservationOutcome(
        observations=tuple(observations),
        valid_coverage_geojson=coverage_geojson,
        coverage_metrics={**coverage_metrics, "burned_pixel_count": burned_pixel_count},
    )


def _clms_observations(
    *,
    durable: DurableEventEvidence,
    artifact: BackendIncidentDaySatelliteArtifact,
    window: ClmsRasterWindow,
    probability_threshold: float,
    fraction_threshold: float,
) -> list[dict[str, Any]]:

    if durable.incident_day_local_date is None or durable.incident_day_bbox is None:
        raise SatelliteCpuError("satellite_incident_day_required", retryable=False)
    return clms_observations_from_window(
        local_date=durable.incident_day_local_date,
        incident_bbox=durable.incident_day_bbox,
        observed_at=_observation_time(durable, artifact),
        source_revision_id=artifact.artifact_revision_id,
        resolution_m=float(artifact.resolution_m or 300),
        window=window,
        probability_threshold=probability_threshold,
        fraction_threshold=fraction_threshold,
    )


def clms_observations_from_window(
    *,
    local_date: date,
    incident_bbox: tuple[float, float, float, float],
    observed_at: datetime,
    source_revision_id: str,
    resolution_m: float,
    window: ClmsRasterWindow,
    probability_threshold: float,
    fraction_threshold: float,
) -> list[dict[str, Any]]:
    """Shared CLMS calculation; a daily product is never dated as its publication day."""
    from shapely.geometry import mapping

    target_day = local_date.timetuple().tm_yday
    valid = window.valid_masks[0] & window.valid_masks[1] & window.valid_masks[2]
    selected = (
        valid
        & (np.rint(window.day_of_burn).astype(np.int32) == target_day)
        & (window.burn_probability >= probability_threshold)
        & (window.burn_fraction >= fraction_threshold)
    )
    pixel_count = int(np.count_nonzero(selected))
    if pixel_count == 0:
        return []
    buckets = _probability_bucket_geometries(
        probability=window.burn_probability,
        selected=selected,
        transform=window.transform,
        crs="EPSG:4326",
        incident_bbox=incident_bbox,
    )
    observations = []
    for bucket_index, bucket_probability, bucket_pixels, geometry in buckets:
        bucket_mask = selected & (
            np.clip(
                np.floor(window.burn_probability / _PROBABILITY_BUCKET_WIDTH).astype(np.int32) + 1,
                1,
                20,
            )
            == bucket_index
        )
        fraction_mean = float(np.mean(window.burn_fraction[bucket_mask]))
        observation_digest = sha256(
            (
                source_revision_id
                + str(target_day)
                + geometry.wkb_hex
                + f":{bucket_index}:{bucket_probability:.6f}"
            ).encode()
        ).hexdigest()
        observations.append(
            {
                "observation_id": f"CLMS-BA-{observation_digest[:24]}",
                "observed_at": observed_at.isoformat(),
                "geometry_geojson": mapping(geometry),
                "horizontal_accuracy_m": max(300.0, resolution_m),
                "confidence": min(1.0, max(0.0, bucket_probability)),
                "metrics": {
                    "target_day_of_year": target_day,
                    "pixel_count": pixel_count,
                    "probability_bucket_pixel_count": bucket_pixels,
                    "burn_probability_mean": bucket_probability,
                    "burn_fraction_mean": fraction_mean,
                    "probability_threshold": probability_threshold,
                    "fraction_threshold": fraction_threshold,
                },
            }
        )
    return observations


def _netcdf_variables(dataset: Any) -> dict[str, Any]:
    variables: dict[str, Any] = {}

    def visit(name: str, value: Any) -> None:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            variables.setdefault(name.rsplit("/", 1)[-1], value)

    dataset.visititems(visit)
    return variables


def _variable(variables: Mapping[str, Any], *names: str, required: bool = False) -> Any | None:
    folded = {name.casefold(): variable for name, variable in variables.items()}
    for name in names:
        if name.casefold() in folded:
            return folded[name.casefold()]
    if required:
        raise SatelliteCpuError("sentinel3_frp_variable_missing", retryable=False)
    return None


def _flat_values(
    variable: Any | None, *, length: int, default: float = math.nan
) -> np.ndarray[Any, Any]:
    if variable is None:
        return np.full(length, default, dtype=np.float64)
    values = np.ma.asarray(variable[:]).reshape(-1)
    if len(values) != length:
        raise SatelliteCpuError("sentinel3_frp_variable_shape_mismatch", retryable=False)
    return np.asarray(values.filled(default))


def _frp_sample_count(variable: Any) -> int:
    shape = getattr(variable, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 1:
        raise SatelliteCpuError("sentinel3_frp_variable_shape_mismatch", retryable=False)
    length = int(shape[0])
    if length < 0 or length > _MAX_FRP_SAMPLES:
        raise SatelliteCpuError("sentinel3_frp_sample_limit_exceeded", retryable=False)
    return length


def _frp_observation_times(
    variable: Any | None,
    *,
    length: int,
    fallback: datetime,
) -> tuple[datetime, ...]:
    if variable is None:
        return (fallback,) * length
    values = np.ma.asarray(variable[:]).reshape(-1)
    if len(values) != length:
        raise SatelliteCpuError("sentinel3_frp_variable_shape_mismatch", retryable=False)
    units = variable.attrs.get("units", "")
    if isinstance(units, bytes):
        units = units.decode("ascii", errors="strict")
    normalized_units = str(units).strip().casefold()
    if normalized_units and (
        "since" not in normalized_units or "2000-01-01" not in normalized_units
    ):
        raise SatelliteCpuError("sentinel3_frp_time_units_invalid", retryable=False)
    if normalized_units.startswith(("second", "s since")):
        multiplier = 1_000_000.0
    elif normalized_units.startswith(("millisecond", "ms since")):
        multiplier = 1_000.0
    else:
        multiplier = 1.0
    origin = datetime(2000, 1, 1, tzinfo=UTC)
    try:
        return tuple(origin + timedelta(microseconds=float(value) * multiplier) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SatelliteCpuError("sentinel3_frp_time_invalid", retryable=False) from exc


def _frp_observations(
    *,
    durable: DurableEventEvidence,
    artifact: BackendIncidentDaySatelliteArtifact,
    path: Path,
    minimum_confidence: float,
) -> list[dict[str, Any]]:
    if durable.incident_day_bbox is None:
        raise SatelliteCpuError("satellite_incident_day_required", retryable=False)
    start = durable.event.time_window.from_at
    end = durable.event.time_window.to_at
    if start is None or end is None:
        raise SatelliteCpuError("satellite_incident_day_time_missing", retryable=False)
    return frp_observations_from_file(
        incident_bbox=durable.incident_day_bbox,
        start=start,
        end=end,
        fallback_time=_observation_time(durable, artifact),
        source_revision_id=artifact.artifact_revision_id,
        resolution_m=float(artifact.resolution_m or 1000),
        path=path,
        minimum_confidence=minimum_confidence,
    )


def frp_observations_from_file(
    *,
    incident_bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    fallback_time: datetime,
    source_revision_id: str,
    resolution_m: float,
    path: Path,
    minimum_confidence: float,
) -> list[dict[str, Any]]:
    """Shared bounded FRP decoder; returns thermal points, never fire boundaries."""
    import h5py

    try:
        with h5py.File(path, "r") as dataset:
            variables = _netcdf_variables(dataset)
            latitude_variable = cast(Any, _variable(variables, "latitude", "lat", required=True))
            longitude_variable = cast(Any, _variable(variables, "longitude", "lon", required=True))
            frp_variable = cast(Any, _variable(variables, "FRP_MWIR", "frp_mwir", required=True))
            length = _frp_sample_count(latitude_variable)
            if (
                _frp_sample_count(longitude_variable) != length
                or _frp_sample_count(frp_variable) != length
            ):
                raise SatelliteCpuError("sentinel3_frp_variable_shape_mismatch", retryable=False)
            latitude = np.ma.asarray(latitude_variable[:]).reshape(-1)
            longitude = np.ma.asarray(longitude_variable[:]).reshape(-1)
            frp = np.ma.asarray(frp_variable[:]).reshape(-1)
            if len(longitude) != length or len(frp) != length:
                raise SatelliteCpuError("sentinel3_frp_variable_shape_mismatch", retryable=False)
            uncertainty = _flat_values(
                _variable(variables, "FRP_uncertainty", "frp_uncertainty_mwir"),
                length=length,
            )
            ifov_area = _flat_values(
                _variable(variables, "IFOV_area", "ifov_area_m2"), length=length
            )
            provider_confidence = _flat_values(
                _variable(
                    variables,
                    "confidence",
                    "confidence_level",
                    "fire_confidence",
                ),
                length=length,
            )
            confidence_class = _flat_values(_variable(variables, "confidence_class"), length=length)
            classification = _flat_values(
                _variable(
                    variables,
                    "classification",
                    "fire_classification",
                    required=True,
                ),
                length=length,
            )
            observation_times = _frp_observation_times(
                _variable(variables, "time"),
                length=length,
                fallback=fallback_time,
            )
    except SatelliteCpuError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise SatelliteCpuError("sentinel3_frp_decode_failed", retryable=False) from exc

    min_lon, min_lat, max_lon, max_lat = incident_bbox
    accuracy = max(1_000.0, resolution_m)
    observations: list[dict[str, Any]] = []
    for index in range(length):
        lon = float(longitude[index])
        lat = float(latitude[index])
        power = float(frp[index])
        confidence_value = float(provider_confidence[index])
        class_confidence = float(confidence_class[index])
        if not math.isfinite(confidence_value) and math.isfinite(class_confidence):
            confidence_value = {0: 15.0, 1: 55.0, 2: 90.0}.get(int(class_confidence), math.nan)
        normalized_confidence = (
            confidence_value / 100.0
            if math.isfinite(confidence_value) and confidence_value > 1
            else confidence_value
        )
        if not math.isfinite(normalized_confidence):
            normalized_confidence = 0.5
        normalized_confidence = min(1.0, max(0.0, normalized_confidence))
        class_value = float(classification[index])
        vegetation_fire = math.isfinite(class_value) and (int(class_value) & 1) == 1
        observed_at = observation_times[index]
        if (
            not all(math.isfinite(value) for value in (lon, lat, power))
            or not min_lon <= lon <= max_lon
            or not min_lat <= lat <= max_lat
            or power < 0
            or normalized_confidence < minimum_confidence
            or not vegetation_fire
            or not start <= observed_at < end
        ):
            continue
        metrics: dict[str, Any] = {"frp_mwir_mw": power}
        uncertainty_value = float(uncertainty[index])
        if math.isfinite(uncertainty_value) and uncertainty_value >= 0:
            metrics["frp_uncertainty_mw"] = uncertainty_value
        ifov_value = float(ifov_area[index])
        if math.isfinite(ifov_value) and ifov_value >= 0:
            metrics["ifov_area_m2"] = ifov_value
        if math.isfinite(class_value):
            metrics["classification"] = int(class_value)
        if math.isfinite(confidence_value):
            metrics["provider_confidence"] = normalized_confidence
        digest = sha256(
            f"{source_revision_id}:{index}:{lon:.8f}:{lat:.8f}:{power:.8f}".encode()
        ).hexdigest()
        observations.append(
            {
                "observation_id": f"S3-FRP-{digest[:24]}",
                "observed_at": observed_at.isoformat(),
                "geometry_geojson": {"type": "Point", "coordinates": [lon, lat]},
                "horizontal_accuracy_m": accuracy,
                "confidence": normalized_confidence,
                "metrics": metrics,
            }
        )
    if len(observations) > 2_048:
        raise SatelliteCpuError("sentinel3_frp_observation_limit_exceeded", retryable=False)
    return observations


@dataclass(frozen=True, slots=True)
class SatelliteObservationCpuRunReceipt:
    analysis_id: str
    artifact_revision_id: str
    processed: int
    remaining: int
    status: Literal["completed", "no_observation", "unavailable", "replayed"]


class SatelliteObservationCpuWorker:
    def __init__(
        self,
        *,
        repository: EventEvidenceRepository,
        asset_reader: SatelliteObservationAssetReader,
        publisher: BackendIncidentDaySatelliteObservationPublisher,
        probability_threshold: float = 0.5,
        fraction_threshold: float = 0.1,
        dnbr_threshold: float = 0.1,
        minimum_burn_probability: float = 0.5,
        minimum_frp_confidence: float = 0.3,
        sentinel1_vv_change_threshold_db: float = 1.5,
        sentinel1_vh_change_threshold_db: float = 1.5,
        openeo_maximum_authorized_credits: float = 0,
    ) -> None:
        for value in (
            probability_threshold,
            fraction_threshold,
            dnbr_threshold,
            minimum_burn_probability,
            minimum_frp_confidence,
        ):
            if not 0 <= value <= 1:
                raise ValueError("satellite observation threshold is outside [0, 1]")
        for value in (sentinel1_vv_change_threshold_db, sentinel1_vh_change_threshold_db):
            if not 0.1 <= value <= 50:
                raise ValueError("Sentinel-1 dB change threshold is outside [0.1, 50]")
        if not 0 <= openeo_maximum_authorized_credits <= 100:
            raise ValueError("openEO credit ceiling is outside [0, 100]")
        self.repository = repository
        self.asset_reader = asset_reader
        self.publisher = publisher
        self.probability_threshold = probability_threshold
        self.fraction_threshold = fraction_threshold
        self.dnbr_threshold = dnbr_threshold
        self.minimum_burn_probability = minimum_burn_probability
        self.minimum_frp_confidence = minimum_frp_confidence
        self.sentinel1_vv_change_threshold_db = sentinel1_vv_change_threshold_db
        self.sentinel1_vh_change_threshold_db = sentinel1_vh_change_threshold_db
        self.openeo_maximum_authorized_credits = openeo_maximum_authorized_credits

    def run(self, analysis_id: str, artifact_revision_id: str) -> SatelliteObservationCpuRunReceipt:
        durable = self.repository.read(analysis_id)
        if (
            durable.research_target_kind != "incident_day"
            or durable.incident_day_bbox is None
            or durable.incident_day_local_date is None
        ):
            raise SatelliteCpuError("satellite_incident_day_required", retryable=False)
        eligible = []
        for item in durable.satellite_artifact_tickets:
            processor = item.quality_flags.get("satellite_observation_processor")
            if processor in {
                _CLMS_PROCESSOR,
                _S1_PROCESSOR,
                _S2_PROCESSOR,
                _FRP_PROCESSOR,
            } and (item.quality_flags.get("temporal_role") != "pre_fire_reference"):
                eligible.append(item)
        completed_ids = _current_completed_artifact_ids(durable, eligible)
        artifact = next(
            (item for item in eligible if item.artifact_revision_id == artifact_revision_id),
            None,
        )
        if artifact is None:
            raise SatelliteCpuError("satellite_observation_artifact_unknown", retryable=False)
        if artifact_revision_id in completed_ids:
            return SatelliteObservationCpuRunReceipt(
                analysis_id=analysis_id,
                artifact_revision_id=artifact_revision_id,
                processed=0,
                remaining=len(
                    [item for item in eligible if item.artifact_revision_id not in completed_ids]
                ),
                status="replayed",
            )
        processor, assets = _artifact_assets(artifact)
        if processor == _CLMS_PROCESSOR and artifact.collection_key != _CLMS_COLLECTION:
            raise SatelliteCpuError("clms_satellite_collection_mismatch", retryable=False)
        if processor == _FRP_PROCESSOR and artifact.collection_key not in _FRP_COLLECTIONS:
            raise SatelliteCpuError("sentinel3_frp_collection_mismatch", retryable=False)
        if processor == _S1_PROCESSOR and artifact.collection_key != _S1_COLLECTION:
            raise SatelliteCpuError("sentinel1_change_collection_mismatch", retryable=False)
        if processor == _S2_PROCESSOR and artifact.collection_key != _S2_COLLECTION:
            raise SatelliteCpuError("sentinel2_change_collection_mismatch", retryable=False)
        reference_artifact = None
        reference_assets: tuple[SatelliteObservationAsset, ...] = ()
        if processor in {_S1_PROCESSOR, _S2_PROCESSOR}:
            change_collection = _S1_COLLECTION if processor == _S1_PROCESSOR else _S2_COLLECTION
            references = [
                item
                for item in durable.satellite_artifact_tickets
                if item.collection_key == change_collection
                and item.quality_flags.get("satellite_observation_processor") == processor
                and item.quality_flags.get("temporal_role") == "pre_fire_reference"
            ]
            if len(references) != 1:
                raise SatelliteCpuError(
                    "satellite_change_prefire_reference_unavailable", retryable=False
                )
            reference_artifact = references[0]
            _reference_processor, reference_assets = _artifact_assets(reference_artifact)
        processor_revision = _PROCESSOR_REVISIONS[processor]
        processing_context_sha256 = _processing_context_sha256(
            durable=durable,
            artifact_id=artifact_revision_id,
            reference_artifact_id=(
                reference_artifact.artifact_revision_id if reference_artifact is not None else None
            ),
            processor=processor,
            processor_revision=processor_revision,
        )
        if processor == _S1_PROCESSOR and self.openeo_maximum_authorized_credits <= 0:
            assert reference_artifact is not None
            self.publisher.publish(
                candidate_id=analysis_id,
                payload={
                    "schema_version": "incident-day-satellite-observation-1.1",
                    "analysis_id": analysis_id,
                    "source_revision_sha256": durable.source_revision_sha256,
                    "artifact_revision_id": artifact_revision_id,
                    "reference_artifact_revision_id": reference_artifact.artifact_revision_id,
                    "result_id": _result_id(
                        analysis_id,
                        artifact_revision_id,
                        processor,
                        processor_revision,
                        processing_context_sha256,
                    ),
                    "processing_context_sha256": processing_context_sha256,
                    "processor": processor,
                    "processor_revision": _S1_REVISION,
                    "status": "unavailable",
                    "unavailable_reason": "cdse_openeo_not_authorized",
                    "observations": [],
                    "valid_coverage_geojson": None,
                    "coverage_metrics": {},
                    "asset_receipts": [],
                    "processing_parameters": {},
                    "raw_satellite_content_stored": False,
                },
            )
            return SatelliteObservationCpuRunReceipt(
                analysis_id=analysis_id,
                artifact_revision_id=artifact_revision_id,
                processed=1,
                remaining=max(
                    0,
                    len(
                        [
                            item
                            for item in eligible
                            if item.artifact_revision_id not in completed_ids
                        ]
                    )
                    - 1,
                ),
                status="unavailable",
            )
        valid_coverage_geojson: dict[str, Any] | None = None
        coverage_metrics: dict[str, float | int] = {}
        with TemporaryDirectory(prefix="fireviewer-satellite-observation-") as directory:
            if processor == _CLMS_PROCESSOR:
                clms_window = self.asset_reader.read_clms_window(
                    assets=assets, bbox=durable.incident_day_bbox
                )
                observations = _clms_observations(
                    durable=durable,
                    artifact=artifact,
                    window=clms_window,
                    probability_threshold=self.probability_threshold,
                    fraction_threshold=self.fraction_threshold,
                )
                receipts = clms_window.receipts
                revision = _CLMS_REVISION
                parameters = {
                    "probability_threshold": self.probability_threshold,
                    "fraction_threshold": self.fraction_threshold,
                    "probability_bucket_width": _PROBABILITY_BUCKET_WIDTH,
                }
                receipt_payloads = [
                    item.as_payload(source_artifact_revision_id=artifact_revision_id)
                    for item in receipts
                ]
            elif processor == _S1_PROCESSOR:
                assert reference_artifact is not None
                if self.openeo_maximum_authorized_credits <= 0:
                    raise SatelliteCpuError("cdse_openeo_credit_ceiling_missing", retryable=False)
                sentinel1_window = self.asset_reader.read_sentinel1_change_window(
                    reference_artifact=reference_artifact,
                    observation_artifact=artifact,
                    bbox=durable.incident_day_bbox,
                )
                sentinel1_outcome = _sentinel1_vvvh_observations(
                    durable=durable,
                    artifact=artifact,
                    window=sentinel1_window,
                    vv_change_threshold_db=self.sentinel1_vv_change_threshold_db,
                    vh_change_threshold_db=self.sentinel1_vh_change_threshold_db,
                )
                observations = list(sentinel1_outcome.observations)
                valid_coverage_geojson = sentinel1_outcome.valid_coverage_geojson
                coverage_metrics = sentinel1_outcome.coverage_metrics
                receipts = (*sentinel1_window.receipts_pre, *sentinel1_window.receipts_post)
                receipt_payloads = [
                    item.as_payload(
                        source_artifact_revision_id=reference_artifact.artifact_revision_id
                    )
                    for item in sentinel1_window.receipts_pre
                ] + [
                    item.as_payload(source_artifact_revision_id=artifact_revision_id)
                    for item in sentinel1_window.receipts_post
                ]
                revision = _S1_REVISION
                parameters = {
                    "vv_change_threshold_db": self.sentinel1_vv_change_threshold_db,
                    "vh_change_threshold_db": self.sentinel1_vh_change_threshold_db,
                    "maximum_authorized_credits": self.openeo_maximum_authorized_credits,
                    "probability_bucket_width": _PROBABILITY_BUCKET_WIDTH,
                }
            elif processor == _S2_PROCESSOR:
                assert reference_artifact is not None
                sentinel2_window = self.asset_reader.read_sentinel2_change_window(
                    reference_assets=reference_assets,
                    observation_assets=assets,
                    bbox=durable.incident_day_bbox,
                )
                sentinel2_outcome = _sentinel2_nbr_observations(
                    durable=durable,
                    artifact=artifact,
                    window=sentinel2_window,
                    dnbr_threshold=self.dnbr_threshold,
                    minimum_probability=self.minimum_burn_probability,
                )
                observations = list(sentinel2_outcome.observations)
                valid_coverage_geojson = sentinel2_outcome.valid_coverage_geojson
                coverage_metrics = sentinel2_outcome.coverage_metrics
                receipts = (*sentinel2_window.receipts_pre, *sentinel2_window.receipts_post)
                receipt_payloads = [
                    item.as_payload(
                        source_artifact_revision_id=reference_artifact.artifact_revision_id
                    )
                    for item in sentinel2_window.receipts_pre
                ] + [
                    item.as_payload(source_artifact_revision_id=artifact_revision_id)
                    for item in sentinel2_window.receipts_post
                ]
                revision = _S2_REVISION
                parameters = {
                    "dnbr_threshold": self.dnbr_threshold,
                    "minimum_probability": self.minimum_burn_probability,
                    "probability_bucket_width": _PROBABILITY_BUCKET_WIDTH,
                }
            else:
                output_path = Path(directory) / "sentinel3-frp.nc"
                receipt = self.asset_reader.fetch_frp_file(asset=assets[0], output_path=output_path)
                observations = _frp_observations(
                    durable=durable,
                    artifact=artifact,
                    path=output_path,
                    minimum_confidence=self.minimum_frp_confidence,
                )
                receipts = (receipt,)
                revision = _FRP_REVISION
                parameters = {"minimum_confidence": self.minimum_frp_confidence}
                receipt_payloads = [
                    item.as_payload(source_artifact_revision_id=artifact_revision_id)
                    for item in receipts
                ]
            status: Literal["completed", "no_observation"] = (
                "completed" if observations else "no_observation"
            )
            self.publisher.publish(
                candidate_id=analysis_id,
                payload={
                    "schema_version": "incident-day-satellite-observation-1.1",
                    "analysis_id": analysis_id,
                    "source_revision_sha256": durable.source_revision_sha256,
                    "artifact_revision_id": artifact_revision_id,
                    "reference_artifact_revision_id": (
                        reference_artifact.artifact_revision_id
                        if reference_artifact is not None
                        else None
                    ),
                    "result_id": _result_id(
                        analysis_id,
                        artifact_revision_id,
                        processor,
                        processor_revision,
                        processing_context_sha256,
                    ),
                    "processing_context_sha256": processing_context_sha256,
                    "processor": processor,
                    "processor_revision": revision,
                    "status": status,
                    "observations": observations,
                    "valid_coverage_geojson": valid_coverage_geojson,
                    "coverage_metrics": coverage_metrics,
                    "asset_receipts": receipt_payloads,
                    "processing_parameters": parameters,
                    "raw_satellite_content_stored": False,
                },
            )
        return SatelliteObservationCpuRunReceipt(
            analysis_id=analysis_id,
            artifact_revision_id=artifact_revision_id,
            processed=1,
            remaining=max(
                0,
                len([item for item in eligible if item.artifact_revision_id not in completed_ids])
                - 1,
            ),
            status=status,
        )


__all__ = [
    "CdseObservationS3Config",
    "CdseS3ObservationAssetReader",
    "ClmsRasterWindow",
    "SatelliteAssetReceipt",
    "SatelliteObservationAsset",
    "SatelliteObservationCpuRunReceipt",
    "SatelliteObservationCpuWorker",
    "Sentinel1ChangeWindow",
    "Sentinel1ObservationOutcome",
    "Sentinel2ChangeWindow",
    "Sentinel2ObservationOutcome",
]
