"""Bounded, read-only CLMS and SLSTR acquisition for reference-isolated replays."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import rasterio
from pydantic import SecretStr
from rasterio.features import shapes
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from fireviewer_evidence_ingestion.mvp.satellite_corpus import SatelliteCorpusRequest, _timestamp
from fireviewer_evidence_ingestion.mvp.satellite_cpu import SatelliteCpuError
from fireviewer_evidence_ingestion.mvp.satellite_observations import (
    CdseObservationS3Config,
    CdseS3ObservationAssetReader,
    SatelliteObservationAsset,
    clms_observations_from_window,
    frp_observations_from_file,
)

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"
CLMS_COLLECTION = "clms_ba_global_300m_daily_v4_cog"
COLLECTIONS = {
    "sentinel1": ("sentinel-1-grd",),
    "clms": (CLMS_COLLECTION,),
    "sentinel3": ("sentinel-3-sl-2-frp-nrt", "sentinel-3-sl-2-frp-ntc"),
}
CLMS_ASSETS = ("ba300_dob_nrt", "ba300_cp_nrt", "ba300_bf_nrt")
REVISION = "fireviewer-cdse-corpus-cpu-1.0.0"


def discover_items(
    client: httpx.Client, request: SatelliteCorpusRequest, source: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Use publication time as well as acquisition time, without reading reference data."""
    if source not in COLLECTIONS:
        raise ValueError("unknown CDSE corpus source")
    start = request.event_started_at
    if source == "sentinel1":
        start -= timedelta(days=24)
    if source == "sentinel3":
        # Old thermal detections must not be reintroduced as today's active observations.
        start = max(
            start,
            request.evaluation_cutoff_at.astimezone(ZoneInfo("Europe/Paris"))
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .astimezone(UTC),
        )
    if request.evaluation_cutoff_at - start > timedelta(days=55 if source == "sentinel1" else 31):
        raise ValueError("CDSE acquisition window exceeds 31 days")
    url = STAC_URL
    params: dict[str, Any] | None = {
        "collections": ",".join(COLLECTIONS[source]),
        "bbox": ",".join(str(x) for x in request.bbox),
        "datetime": f"{start.isoformat()}/{request.evaluation_cutoff_at.isoformat()}",
        "limit": 100,
    }
    items: dict[str, dict[str, Any]] = {}
    counts = {"catalog_items": 0, "late_or_unknown_availability": 0, "reference_excluded": 0}
    seen_pages: set[str] = set()
    for _ in range(8):
        if url in seen_pages or urlsplit(url).hostname != "stac.dataspace.copernicus.eu":
            raise ValueError("CDSE pagination is cyclic or escaped its host")
        if not url.startswith(STAC_URL) or urlsplit(url).username or urlsplit(url).fragment:
            raise ValueError("CDSE pagination URL rejected")
        seen_pages.add(url)
        with client.stream(
            "GET", url, params=params, timeout=45, follow_redirects=False
        ) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_bytes(65536):
                data.extend(chunk)
                if len(data) > 8 * 1024**2:
                    raise ValueError("CDSE catalog response exceeds 8 MiB")
        payload = json.loads(data)
        for item in payload["features"]:
            counts["catalog_items"] += 1
            props = item["properties"]
            if item.get("collection") not in COLLECTIONS[source]:
                raise ValueError("CDSE collection mismatch")
            timestamps = [props.get("created"), props.get("published")]
            if source == "sentinel1":
                timestamps.append(props.get("processing:datetime"))
            acquired = _timestamp(props.get("end_datetime") or props["datetime"])
            if (
                not all(timestamps)
                or max(_timestamp(t) for t in timestamps) > request.evaluation_cutoff_at
                or acquired > request.evaluation_cutoff_at
                or _timestamp(props["datetime"]) < start
            ):
                counts["late_or_unknown_availability"] += 1
                continue
            item_url = f"https://stac.dataspace.copernicus.eu/v1/collections/{item['collection']}/items/{item['id']}"
            if {item["id"], item_url}.intersection(request.forbidden_input_refs):
                counts["reference_excluded"] += 1
                continue
            items[item["id"]] = item
        link = next((x for x in payload.get("links", []) if x.get("rel") == "next"), None)
        if link is None:
            break
        if link.get("method", "GET") != "GET":
            raise ValueError("CDSE pagination method rejected")
        url, params = link["href"], None
    else:
        raise ValueError("CDSE pagination limit exceeded")
    # NRT/NTC versions of the same granule are not independent observations.
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in sorted(items.values(), key=lambda x: (x["properties"]["published"], x["id"])):
        props = item["properties"]
        key = (props.get("platform", "sentinel-3"), props["datetime"], source)
        if source == "sentinel1":
            # Pairing below needs all intersecting granules, not one arbitrary tile.
            key = (item["id"], props["datetime"], source)
        selected.setdefault(key, item)
    if len(selected) > 64:
        raise ValueError("CDSE incident product limit exceeded")
    return list(selected.values()), counts


def processing_assets(item: dict[str, Any], source: str) -> tuple[SatelliteObservationAsset, ...]:
    names = (
        CLMS_ASSETS
        if source == "clms"
        else ("FRP_MWIR1km_STANDARD" if item["collection"].endswith("nrt") else "FRP_in",)
    )
    prefix = (
        "s3://eodata/CLMS/bio-geophysical/burnt_area/ba_global_300m_daily_v4/"
        if source == "clms"
        else "s3://eodata/Sentinel-3/SLSTR/SL_2_FRP___/"
    )
    assets = []
    for name in names:
        raw = item["assets"][name]
        url = urlsplit(raw["href"])
        if (
            not raw["href"].startswith(prefix)
            or url.query
            or url.fragment
            or ".." in url.path.split("/")
        ):
            raise ValueError("CDSE asset path rejected")
        # CLMS STAC does not publish per-band checksums. This identifies the metadata,
        # not a purported hash of remote raster bytes. The receipt states that basis.
        checksum = (
            raw.get("file:checksum") or sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
        )
        grid = (
            {}
            if source != "clms"
            else {
                "proj_code": raw["proj:code"],
                "proj_shape": raw["proj:shape"],
                "proj_transform": raw["proj:transform"],
                "nodata": raw["nodata"],
                "data_type": raw["data_type"],
                "raster_scale": raw.get("raster:scale", 1.0),
            }
        )
        assets.append(
            SatelliteObservationAsset(
                asset_name=name,
                object_uri=raw["href"],
                media_type=raw["type"],
                file_size_bytes=raw["file:size"],
                file_checksum=checksum,
                **grid,
            )
        )
    return tuple(assets)


def collect_cdse(
    request: SatelliteCorpusRequest,
    output_dir: Path,
    source: str,
    *,
    reader: CdseS3ObservationAssetReader | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    if source == "sentinel1":
        from fireviewer_evidence_ingestion.mvp.sentinel1_corpus import collect_sentinel1

        return collect_sentinel1(request, output_dir, client=client)
    output_dir.mkdir(parents=True, exist_ok=True)
    owned_client = client is None
    client = client or httpx.Client()
    owned_reader = reader is None
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    written = 0
    try:
        items, counts = discover_items(client, request, source)
        if items and reader is None:
            access = os.environ.get("FIREVIEWER_CDSE_S3_ACCESS_KEY", "")
            secret = os.environ.get("FIREVIEWER_CDSE_S3_SECRET_KEY", "")
            if not access or not secret:
                return {
                    "schema": "fireviewer.part4-cdse-corpus-result.v1",
                    "products": [],
                    "reason": "cdse_credentials_unavailable",
                    "catalog": counts,
                    "paid_service_used": False,
                    "written_bytes": 0,
                }
            reader = CdseS3ObservationAssetReader(
                CdseObservationS3Config(
                    access_key=SecretStr(access),
                    secret_key=SecretStr(secret),
                    maximum_window_pixels=request.maximum_window_pixels,
                )
            )
        for item in items:
            assert reader is not None
            props = item["properties"]
            identity = sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
            assets = processing_assets(item, source)
            files: list[dict[str, Any]] = []
            coverage = None
            acquisition = _timestamp(props["datetime"])
            try:
                if source == "clms":
                    window = reader.read_clms_window(assets=assets, bbox=request.bbox)
                    observations = clms_observations_from_window(
                        local_date=acquisition.date(),
                        incident_bbox=request.bbox,
                        observed_at=acquisition,
                        source_revision_id=identity,
                        resolution_m=300,
                        window=window,
                        probability_threshold=0.5,
                        fraction_threshold=0.1,
                    )
                    valid = np.logical_and.reduce(window.valid_masks) & (window.day_of_burn >= 0)
                    valid &= (window.burn_probability >= 0) & (window.burn_fraction >= 0)
                    polygons = [
                        shape(g)
                        for g, value in shapes(
                            valid.astype("uint8"), mask=valid, transform=window.transform
                        )
                        if value
                    ]
                    if polygons:
                        coverage = mapping(unary_union(polygons).intersection(box(*request.bbox)))
                    arrays = (window.day_of_burn, window.burn_probability, window.burn_fraction)
                    reserve = sum(a.size * 4 for a in arrays) * 2 + 1024**2
                    if written + reserve > request.maximum_output_bytes:
                        raise ValueError("CDSE output budget exceeded")
                    path = output_dir / f"{identity[:24]}-clms.tif"
                    with rasterio.open(
                        path,
                        "w",
                        driver="COG",
                        width=valid.shape[1],
                        height=valid.shape[0],
                        count=3,
                        dtype="float32",
                        crs="EPSG:4326",
                        transform=window.transform,
                        nodata=float("nan"),
                        compress="DEFLATE",
                    ) as raster:
                        for index, array in enumerate(arrays, 1):
                            raster.write(np.where(valid, array, np.nan).astype("float32"), index)
                            raster.set_band_description(index, CLMS_ASSETS[index - 1])
                        raster.update_tags(source_revision=identity, processor_revision=REVISION)
                    files.append({"path": path.name, "byte_count": path.stat().st_size})
                else:
                    if written + assets[0].file_size_bytes > request.maximum_output_bytes:
                        raise ValueError("CDSE output budget exceeded")
                    path = output_dir / f"{identity[:24]}-frp.nc"
                    reader.fetch_frp_file(asset=assets[0], output_path=path)
                    observations = frp_observations_from_file(
                        incident_bbox=request.bbox,
                        start=max(
                            request.event_started_at,
                            request.evaluation_cutoff_at.astimezone(
                                ZoneInfo("Europe/Paris")
                            ).replace(hour=0, minute=0, second=0, microsecond=0),
                        ),
                        end=request.evaluation_cutoff_at + timedelta(microseconds=1),
                        fallback_time=acquisition,
                        source_revision_id=identity,
                        resolution_m=1000,
                        path=path,
                        minimum_confidence=0.3,
                    )
                    files.append({"path": path.name, "byte_count": path.stat().st_size})
            except SatelliteCpuError as exc:
                failures.append({"product_id": item["id"], "reason": str(exc)})
                continue
            written += sum(f["byte_count"] for f in files)
            if written > request.maximum_output_bytes:
                raise ValueError("CDSE output budget exceeded")
            results.append(
                {
                    "product_id": item["id"],
                    "collection": item["collection"],
                    "source_revision_sha256": identity,
                    "processor_revision": REVISION,
                    "source_family_id": "clms.ba.sentinel3"
                    if source == "clms"
                    else f"{props['platform']}.slstr",
                    "lineage_id": (
                        f"{source}:{props.get('platform', 'sentinel3')}:"
                        f"{acquisition.astimezone(UTC).strftime('%Y%m%dT%H%M%S')}"
                    ),
                    "observation_kind": "burned_probability"
                    if source == "clms"
                    else "thermal_footprint",
                    "target_state": "affected" if source == "clms" else "active",
                    "source_available_at": max(
                        _timestamp(props[k]) for k in ("created", "published")
                    ).isoformat(),
                    "observed_at": acquisition.isoformat(),
                    "observed_end_at": props.get("end_datetime"),
                    "resolution_m": 300 if source == "clms" else 1000,
                    "observations": observations,
                    "coverage_geojson": coverage,
                    "files": files,
                    "source_item": item,
                    "asset_identity_basis": "provider_checksum_or_canonical_stac_asset_metadata",
                }
            )
        return {
            "schema": "fireviewer.part4-cdse-corpus-result.v1",
            "products": results,
            "reason": "processed" if results else "no_admissible_product",
            "catalog": counts,
            "failures": failures,
            "written_bytes": written,
            "paid_service_used": False,
        }
    finally:
        if owned_reader and reader is not None:
            reader.close()
        if owned_client:
            client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source", choices=tuple(COLLECTIONS), required=True)
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536:
        raise ValueError("CDSE corpus request too large")
    request = SatelliteCorpusRequest.model_validate_json(raw)
    print(
        json.dumps(
            collect_cdse(request, args.output_dir, args.source), sort_keys=True, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
