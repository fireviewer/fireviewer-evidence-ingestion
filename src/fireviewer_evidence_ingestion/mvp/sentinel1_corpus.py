"""Reference-isolated SAR pairs and bounded CDSE openEO corpus processing.

The general CDSE backend uses credits, not Sentinel Hub processing units. No
invocation is possible without an explicitly verified free allowance. A failed
or unpriced request consumes its full reservation; it is never retried here.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np
import rasterio
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds
from shapely.geometry import box, shape

from fireviewer_evidence_ingestion.mvp.satellite_corpus import SatelliteCorpusRequest, _timestamp
from fireviewer_evidence_ingestion.mvp.satellite_cpu import SatelliteCpuError
from fireviewer_evidence_ingestion.mvp.satellite_observations import (
    SatelliteAssetReceipt,
    Sentinel1ChangeWindow,
    sentinel1_vvvh_from_window,
)

OPENEO_URL = "https://openeo.dataspace.copernicus.eu/openeo/1.2"
REVISION = "fireviewer-sentinel1-corpus-cpu-1.0.0"


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def orbit_key(item: dict[str, Any]) -> tuple[str, int, str]:
    p = item["properties"]
    if (
        p.get("sar:instrument_mode") != "IW"
        or set(p.get("sar:polarizations", [])) != {"VV", "VH"}
        or p.get("platform") not in {"sentinel-1a", "sentinel-1b", "sentinel-1c"}
        or p.get("sat:orbit_state") not in {"ascending", "descending"}
        or not isinstance(p.get("sat:relative_orbit"), int)
        or not 1 <= p["sat:relative_orbit"] <= 175
    ):
        raise ValueError("sentinel1_orbit_or_polarization_unresolved")
    return p["platform"], p["sat:relative_orbit"], p["sat:orbit_state"]


@dataclass(frozen=True)
class Sentinel1Pair:
    pre: dict[str, Any]
    post: dict[str, Any]

    @property
    def identity(self) -> str:
        return _digest([self.pre, self.post])


def select_pairs(
    items: list[dict[str, Any]], request: SatelliteCorpusRequest
) -> tuple[list[Sentinel1Pair], dict[str, int]]:
    """Select the latest comparable pair per track, retaining independent granules."""
    eligible = []
    rejected = {"incompatible_orbit": 0, "reference_excluded": 0, "unpaired_post": 0}
    extent = box(*request.bbox)
    for item in items:
        p = item["properties"]
        # The CEMS source table sometimes identifies only a sensor and date.
        stamp = _timestamp(p["datetime"]).strftime("%Y%m%d")
        if any(
            ref.startswith(("CEMS_SENTINEL1_", "CEMS_SENTINEL1A_", "CEMS_SENTINEL1B_"))
            and stamp in ref
            for ref in request.forbidden_input_refs
        ):
            rejected["reference_excluded"] += 1
            continue
        try:
            orbit_key(item)
            geometry = shape(item["geometry"])
            if not geometry.is_valid or not geometry.intersects(extent):
                raise ValueError("outside incident")
        except (KeyError, TypeError, ValueError):
            rejected["incompatible_orbit"] += 1
            continue
        eligible.append(item)
    pre = [
        i
        for i in eligible
        if _timestamp(i["properties"].get("end_datetime") or i["properties"]["datetime"])
        < request.event_started_at
    ]
    post = sorted(
        (
            i
            for i in eligible
            if _timestamp(i["properties"]["datetime"]) >= request.event_started_at
        ),
        key=lambda i: (i["properties"]["datetime"], i["id"]),
        reverse=True,
    )
    pairs: list[Sentinel1Pair] = []
    selected_tracks: set[tuple[str, int, str]] = set()
    for item in post:
        key = orbit_key(item)
        if key in selected_tracks:
            continue
        candidates = [
            i
            for i in pre
            if orbit_key(i) == key
            and shape(i["geometry"]).intersection(shape(item["geometry"])).intersects(extent)
        ]
        if not candidates:
            rejected["unpaired_post"] += 1
            continue
        reference = max(candidates, key=lambda i: (i["properties"]["datetime"], i["id"]))
        pairs.append(Sentinel1Pair(reference, item))
        selected_tracks.add(key)
        if len(pairs) == request.maximum_pairs:
            break
    return pairs, rejected


class OpenEOAllowance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    token: SecretStr | None = None
    free_credits_verified: bool = False
    maximum_total_credits: float = Field(default=0, ge=0, le=100, allow_inf_nan=False)
    maximum_request_credits: float = Field(default=1, gt=0, le=10, allow_inf_nan=False)

    @model_validator(mode="after")
    def guard(self) -> OpenEOAllowance:
        if self.maximum_total_credits and (
            not self.free_credits_verified
            or self.token is None
            or len(self.token.get_secret_value()) < 32
        ):
            raise ValueError("openEO requires verified free credits and an access token")
        return self

    @classmethod
    def from_environment(cls) -> OpenEOAllowance:
        token = os.environ.get("FIREVIEWER_CDSE_OPENEO_ACCESS_TOKEN")
        return cls(
            token=SecretStr(token) if token else None,
            free_credits_verified=os.environ.get("FIREVIEWER_CDSE_OPENEO_FREE_CREDITS_VERIFIED")
            == "true",
            maximum_total_credits=float(
                os.environ.get("FIREVIEWER_CDSE_OPENEO_MAXIMUM_CREDITS", "0")
            ),
            maximum_request_credits=float(
                os.environ.get("FIREVIEWER_CDSE_OPENEO_REQUEST_CREDITS", "1")
            ),
        )


def process_graph(
    item: dict[str, Any], bbox: tuple[float, float, float, float], resolution: int
) -> dict[str, Any]:
    platform, orbit, direction = orbit_key(item)
    p = item["properties"]
    start = _timestamp(p["datetime"])
    end = _timestamp(p.get("end_datetime") or p["datetime"]) + timedelta(microseconds=1)
    if not 0 < (end - start).total_seconds() <= 120:
        raise ValueError("sentinel1_acquisition_window_invalid")
    properties = {
        "id": item["id"],
        "platform": platform,
        "sat:relative_orbit": orbit,
        "sat:orbit_state": direction,
        "sar:instrument_mode": "IW",
    }
    return {
        "load": {
            "process_id": "load_collection",
            "arguments": {
                "id": "SENTINEL1_GRD",
                "bands": ["VV", "VH"],
                "spatial_extent": dict(zip(("west", "south", "east", "north"), bbox, strict=True)),
                "temporal_extent": [
                    stamp.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
                    for stamp in (start, end)
                ],
                "properties": {
                    k: {
                        "process_graph": {
                            "eq": {
                                "process_id": "eq",
                                "arguments": {"x": {"from_parameter": "value"}, "y": v},
                                "result": True,
                            }
                        }
                    }
                    for k, v in properties.items()
                },
            },
        },
        "backscatter": {
            "process_id": "sar_backscatter",
            "arguments": {
                "data": {"from_node": "load"},
                "coefficient": "sigma0-ellipsoid",
                "elevation_model": "COPERNICUS_30",
            },
        },
        "grid": {
            "process_id": "resample_spatial",
            "arguments": {
                "data": {"from_node": "backscatter"},
                "resolution": resolution,
                "projection": 2154,
                "method": "bilinear",
            },
        },
        "time": {
            "process_id": "reduce_dimension",
            "arguments": {
                "data": {"from_node": "grid"},
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
            "arguments": {"data": {"from_node": "time"}, "format": "GTiff"},
            "result": True,
        },
    }


class PairReader(Protocol):
    def read_pair(
        self, pair: Sentinel1Pair, request: SatelliteCorpusRequest
    ) -> Sentinel1ChangeWindow: ...


class OpenEOCorpusReader:
    def __init__(self, client: httpx.Client, allowance: OpenEOAllowance) -> None:
        self.client, self.allowance = client, allowance
        self.reserved_credits = 0.0
        self.invocations = 0

    def _raster(
        self, item: dict[str, Any], request: SatelliteCorpusRequest, resolution: int
    ) -> tuple[Any, Any, str, SatelliteAssetReceipt]:
        allowance = self.allowance
        if allowance.token is None or not allowance.free_credits_verified:
            raise SatelliteCpuError("cdse_openeo_free_allowance_unavailable", retryable=False)
        if (
            self.reserved_credits + allowance.maximum_request_credits
            > allowance.maximum_total_credits
        ):
            raise SatelliteCpuError("cdse_openeo_corpus_budget_exhausted", retryable=False)
        graph = process_graph(item, request.bbox, resolution)
        token = allowance.token.get_secret_value()
        # General openEO uses its OIDC provider envelope, unlike Sentinel Hub.
        if not token.startswith("oidc/CDSE/"):
            token = "oidc/CDSE/" + token
        headers = {
            "Authorization": "Bearer " + token,
            "Accept-Encoding": "identity",
        }
        # Validation is not a processing invocation and cannot fall back to a wider graph.
        response = self.client.post(
            OPENEO_URL + "/validation",
            json={"process_graph": graph},
            headers=headers,
            timeout=30,
            follow_redirects=False,
        )
        if response.status_code != 200 or response.json().get("errors") != []:
            raise SatelliteCpuError("cdse_openeo_graph_rejected", retryable=False)
        self.reserved_credits += allowance.maximum_request_credits
        self.invocations += 1
        with self.client.stream(
            "POST",
            OPENEO_URL + "/result",
            json={
                "process": {"process_graph": graph},
                "budget": allowance.maximum_request_credits,
            },
            headers=headers,
            timeout=120,
            follow_redirects=False,
        ) as response:
            if response.status_code != 200:
                raise SatelliteCpuError("cdse_openeo_processing_failed", retryable=False)
            if response.headers.get("content-type", "").split(";")[0] not in {
                "image/tiff",
                "image/geotiff",
            }:
                raise SatelliteCpuError("cdse_openeo_content_type_invalid", retryable=False)
            content = bytearray()
            for chunk in response.iter_bytes(65536):
                content.extend(chunk)
                if len(content) > min(request.maximum_output_bytes, 64 * 1024**2):
                    raise SatelliteCpuError("cdse_openeo_response_size_invalid", retryable=False)
        with MemoryFile(bytes(content)) as memory, memory.open() as dataset:
            if (
                dataset.count != 2
                or dataset.crs is None
                or dataset.crs.to_epsg() != 2154
                or dataset.width * dataset.height > request.maximum_window_pixels
            ):
                raise SatelliteCpuError("cdse_openeo_raster_contract_invalid", retryable=False)
            if not math.isclose(abs(dataset.transform.a), resolution) or not math.isclose(
                abs(dataset.transform.e), resolution
            ):
                raise SatelliteCpuError("cdse_openeo_resolution_mismatch", retryable=False)
            arrays = {
                band: np.asarray(dataset.read(index, masked=True).astype("float64").filled(np.nan))
                for index, band in enumerate(("VV", "VH"), 1)
            }
            return (
                arrays,
                dataset.transform,
                str(dataset.crs),
                SatelliteAssetReceipt(
                    asset_name="openeo_vv_vh",
                    source_checksum=_digest(item),
                    derived_content_sha256=sha256(content).hexdigest(),
                    bytes_read=len(content),
                ),
            )

    def read_pair(
        self, pair: Sentinel1Pair, request: SatelliteCorpusRequest
    ) -> Sentinel1ChangeWindow:
        # Reserve enough room for BOTH calls before starting either one.
        if (
            self.reserved_credits + 2 * self.allowance.maximum_request_credits
            > self.allowance.maximum_total_credits
        ):
            raise SatelliteCpuError("cdse_openeo_corpus_budget_exhausted", retryable=False)
        west, south, east, north = transform_bounds("EPSG:4326", "EPSG:2154", *request.bbox)
        resolution = next(
            (
                r
                for r in (20, 50, 100, 250)
                if (math.ceil((east - west) / r) + 2) * (math.ceil((north - south) / r) + 2)
                <= request.maximum_window_pixels
            ),
            None,
        )
        if resolution is None:
            raise SatelliteCpuError("cdse_openeo_window_too_large", retryable=False)
        pre, transform, crs, pre_receipt = self._raster(pair.pre, request, resolution)
        post, post_transform, post_crs, post_receipt = self._raster(pair.post, request, resolution)
        if transform != post_transform or crs != post_crs or pre["VV"].shape != post["VV"].shape:
            raise SatelliteCpuError("cdse_openeo_grid_mismatch", retryable=False)
        return Sentinel1ChangeWindow(pre, post, transform, crs, (pre_receipt,), (post_receipt,))


def collect_sentinel1(
    request: SatelliteCorpusRequest,
    output_dir: Path,
    *,
    client: httpx.Client | None = None,
    reader: PairReader | None = None,
) -> dict[str, Any]:
    from fireviewer_evidence_ingestion.mvp.cdse_corpus import discover_items

    owned = client is None
    client = client or httpx.Client()
    result: dict[str, Any] = {
        "schema": "fireviewer.part4-cdse-corpus-result.v1",
        "products": [],
        "reason": "no_admissible_pair",
        "paid_service_used": False,
        "written_bytes": 0,
        "failures": [],
    }
    try:
        items, catalog = discover_items(client, request, "sentinel1")
        pairs, exclusions = select_pairs(items, request)
        result.update(
            catalog=catalog,
            pairing=exclusions,
            candidate_pairs=[
                {"pre": p.pre["id"], "post": p.post["id"], "orbit": orbit_key(p.post)}
                for p in pairs
            ],
        )
        if not pairs:
            return result
        if reader is None:
            allowance = OpenEOAllowance.from_environment()
            if not allowance.maximum_total_credits:
                result["reason"] = "cdse_openeo_free_allowance_unavailable"
                return result
            reader = OpenEOCorpusReader(client, allowance)
        for pair in pairs:
            try:
                window = reader.read_pair(pair, request)
            except (SatelliteCpuError, httpx.HTTPError) as exc:
                reason = (
                    str(exc) if isinstance(exc, SatelliteCpuError) else "cdse_openeo_network_error"
                )
                result["failures"].append({"pair_id": pair.identity, "reason": reason})
                continue
            resolution = max(40.0, abs(float(window.transform.a)))
            outcome = sentinel1_vvvh_from_window(
                incident_bbox=request.bbox,
                observed_at=_timestamp(pair.post["properties"]["datetime"]),
                source_revision_id=pair.identity,
                resolution_m=resolution,
                window=window,
            )
            arrays = [window.pre[b] for b in ("VV", "VH")] + [window.post[b] for b in ("VV", "VH")]
            reserve = sum(a.size * 4 for a in arrays) * 2 + 1024**2
            if result["written_bytes"] + reserve > request.maximum_output_bytes:
                raise ValueError("sentinel1_output_budget_exceeded")
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{pair.identity[:24]}-s1.tif"
            with rasterio.open(
                path,
                "w",
                driver="COG",
                width=arrays[0].shape[1],
                height=arrays[0].shape[0],
                count=4,
                dtype="float32",
                crs=window.crs,
                transform=window.transform,
                nodata=float("nan"),
                compress="DEFLATE",
            ) as raster:
                for index, array in enumerate(arrays, 1):
                    raster.write(array.astype("float32"), index)
                    raster.set_band_description(
                        index, ("pre_VV", "pre_VH", "post_VV", "post_VH")[index - 1]
                    )
                raster.update_tags(processor_revision=REVISION, source_revision=pair.identity)
            result["written_bytes"] += path.stat().st_size
            if result["written_bytes"] > request.maximum_output_bytes:
                raise ValueError("sentinel1_output_budget_exceeded")
            result["products"].append(
                {
                    "product_id": pair.post["id"],
                    "pre_product_id": pair.pre["id"],
                    "collection": "sentinel-1-grd",
                    "source_revision_sha256": pair.identity,
                    "processor_revision": REVISION,
                    "source_family_id": f"{orbit_key(pair.post)[0]}.c-sar",
                    "lineage_id": (
                        f"sentinel1-pair:{_digest([pair.pre['id'], pair.post['id']])[:32]}"
                    ),
                    "observation_kind": "modelled_perimeter",
                    "target_state": "affected",
                    "source_available_at": max(
                        _timestamp(i["properties"][k])
                        for i in (pair.pre, pair.post)
                        for k in ("created", "published", "processing:datetime")
                    ).isoformat(),
                    "resolution_m": resolution,
                    "observations": outcome.observations,
                    "coverage_geojson": outcome.valid_coverage_geojson,
                    "metrics": outcome.coverage_metrics,
                    "files": [{"path": path.name, "byte_count": path.stat().st_size}],
                    "source_items": [pair.pre, pair.post],
                    "calibration_state": "uncalibrated",
                    "independence_demonstrated": False,
                }
            )
        result["reason"] = "processed" if result["products"] else "processing_unavailable"
        if isinstance(reader, OpenEOCorpusReader):
            result["cost_control"] = {
                "backend": OPENEO_URL,
                "currency": "credits",
                "reserved_credits": reader.reserved_credits,
                "invocations": reader.invocations,
                "automatic_retry": False,
            }
        return result
    finally:
        if owned:
            client.close()
