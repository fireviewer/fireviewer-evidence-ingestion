"""Bounded public Sentinel-2 COG acquisition for offline Part.4 corpus jobs.

No model, account, requester-pays bucket, or reference perimeter is used here.
The spectral processor is shared with the operational satellite CPU worker.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np
import rasterio
from pydantic import AwareDatetime, Field, model_validator
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from fireviewer_contracts.contracts import StrictModel
from fireviewer_evidence_ingestion.mvp.satellite_observations import (
    Sentinel2ChangeWindow,
    sentinel2_nbr_from_window,
)

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
ASSETS = {
    "red": "B04_20m",
    "nir08": "B8A_20m",
    "swir16": "B11_20m",
    "swir22": "B12_20m",
    "scl": "SCL_20m",
}
PROCESSOR_REVISION = "fireviewer-sentinel2-nbr-change-cpu-1.3.0"


class SatelliteCorpusRequest(StrictModel):
    incident_id: str = Field(min_length=3, max_length=128)
    bbox: tuple[float, float, float, float]
    event_started_at: AwareDatetime
    evaluation_cutoff_at: AwareDatetime
    forbidden_input_refs: tuple[str, ...] = ()
    maximum_pairs: int = Field(default=2, ge=1, le=4)
    maximum_window_pixels: int = Field(default=1_000_000, ge=256, le=4_000_000)
    maximum_output_bytes: int = Field(default=256 * 1024**2, ge=1024, le=1024**3)

    @model_validator(mode="after")
    def validate_extent(self) -> SatelliteCorpusRequest:
        west, south, east, north = self.bbox
        if not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
            raise ValueError("invalid satellite corpus extent")
        if east - west > 2 or north - south > 2:
            raise ValueError("satellite corpus AOI exceeds its bounded extent")
        if self.event_started_at >= self.evaluation_cutoff_at:
            raise ValueError("satellite corpus cutoff must follow the incident start")
        return self


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("satellite timestamp has no timezone")
    return parsed.astimezone(UTC)


def _public_cog(asset: dict[str, Any]) -> str:
    href = str(asset.get("href", ""))
    parsed = urlsplit(href)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "sentinel-cogs.s3.us-west-2.amazonaws.com"
        or not parsed.path.startswith("/sentinel-s2-l2a-cogs/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("satellite corpus only accepts allowlisted public COGs")
    return href


def _source_scene(value: str) -> tuple[str, str] | None:
    match = re.search(r"(S2[ABC]).*?(\d{8}T\d{6})", value.upper())
    return (match[1], match[2]) if match else None


def _availability_receipt(
    client: httpx.Client,
    item: dict[str, Any],
    cutoff: datetime,
) -> dict[str, Any] | None:
    """A migrated catalog is not a reprocessed image; require dated object evidence."""
    props = item["properties"]
    generated, created = props.get("s2:generation_time"), props.get("created")
    if not generated or not created or _timestamp(generated) > cutoff:
        return None
    if _timestamp(created) <= cutoff:
        return {
            "available_at": created,
            "basis": "source_generation_and_catalog_creation_before_cutoff",
        }
    receipts: list[dict[str, Any]] = []
    for band in ASSETS:
        href = _public_cog(item["assets"][band])
        response = client.head(href, timeout=20, follow_redirects=False)
        if response.status_code != 200:
            return None
        try:
            modified = parsedate_to_datetime(response.headers["last-modified"])
            byte_count = int(response.headers["content-length"])
            etag = response.headers["etag"]
        except (KeyError, ValueError, TypeError):
            return None
        if (
            modified.tzinfo is None
            or not _timestamp(generated) <= modified <= cutoff
            or byte_count <= 0
            or not etag
        ):
            return None
        receipts.append(
            {
                "asset": band,
                "href": href,
                "last_modified": modified.isoformat(),
                "byte_count": byte_count,
                "etag": etag,
            }
        )
    return {
        "available_at": max(r["last_modified"] for r in receipts),
        "basis": "original_public_cog_objects_and_source_generation_before_cutoff",
        "catalog_created_at": created,
        "object_headers": receipts,
    }


def discover_pairs(
    client: httpx.Client,
    request: SatelliteCorpusRequest,
) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    """Select pre/post pairs using metadata only, never the evaluation reference."""

    begin = request.event_started_at - timedelta(days=45)
    body: dict[str, Any] = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(request.bbox),
        "datetime": f"{begin.isoformat()}/{request.evaluation_cutoff_at.isoformat()}",
        "limit": 100,
    }
    url, method = STAC_URL, "POST"
    items: dict[str, dict[str, Any]] = {}
    forbidden_scenes = {
        scene
        for value in request.forbidden_input_refs
        if (scene := _source_scene(value)) is not None
    }
    # CEMS also names acquisitions without the spacecraft or tile. Exclude that
    # UTC day conservatively rather than treating a renamed derivative as input.
    forbidden_days = {
        match[1]
        for value in request.forbidden_input_refs
        if (match := re.search(r"SENTINEL[-_ ]?2[_ -]+(\d{8})", value.upper()))
    }
    for _ in range(4):
        with client.stream(method, url, json=body if method == "POST" else None) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_bytes():
                if len(content) + len(chunk) > 8 * 1024**2:
                    raise ValueError("satellite catalog response exceeds its byte limit")
                content.extend(chunk)
        payload = json.loads(content)
        for item in payload.get("features", []):
            props = item.get("properties", {})
            product = str(props.get("s2:product_uri", ""))
            exposed = {item.get("id", ""), product, props.get("s2:granule_id", "")}
            if exposed.intersection(request.forbidden_input_refs):
                continue
            if _source_scene(product) in forbidden_scenes:
                continue
            # Acquisition time alone cannot prove availability at the cutoff.
            generated = props.get("s2:generation_time")
            catalog_created = props.get("created")
            if not generated or not catalog_created:
                continue
            if _timestamp(generated) > request.evaluation_cutoff_at:
                continue
            acquired = _timestamp(props["datetime"])
            if not begin <= acquired <= request.evaluation_cutoff_at:
                continue
            if acquired.strftime("%Y%m%d") in forbidden_days:
                continue
            if float(props.get("eo:cloud_cover", 100)) > 80:
                continue
            if not props.get("grid:code") or not product:
                continue
            assets = item.get("assets", {})
            if not all(name in assets for name in ASSETS):
                continue
            for name in ASSETS:
                _public_cog(assets[name])
            items[str(item["id"])] = item
        next_link = next(
            (link for link in payload.get("links", []) if link.get("rel") == "next"), None
        )
        if next_link is None:
            break
        url = str(next_link["href"])
        if urlsplit(url).netloc != "earth-search.aws.element84.com" or not url.startswith(
            "https://"
        ):
            raise ValueError("satellite catalog pagination host differs")
        method = str(next_link.get("method", "GET")).upper()
        if method not in {"GET", "POST"}:
            raise ValueError("unsupported satellite pagination method")
        body = next_link.get("body", body)
    else:
        raise ValueError("satellite catalog pagination limit exceeded")
    ordered = sorted(
        items.values(),
        key=lambda item: (
            item["properties"]["datetime"],
            -float(item["properties"]["eo:cloud_cover"]),
            item["id"],
        ),
        reverse=True,
    )
    candidates: list[tuple[dict[str, Any], dict[str, Any], BaseGeometry]] = []
    extent = box(*request.bbox)
    used_grids: set[str] = set()
    checked: dict[str, bool] = {}

    def available(item: dict[str, Any]) -> bool:
        key = str(item["id"])
        if key not in checked:
            receipt = _availability_receipt(client, item, request.evaluation_cutoff_at)
            checked[key] = receipt is not None
            if receipt is not None:
                item["fireviewer:availability"] = receipt
        return checked[key]

    for post in ordered:
        props = post["properties"]
        grid = props["grid:code"]
        if grid in used_grids or _timestamp(props["datetime"]) < request.event_started_at:
            continue
        if not available(post):
            continue
        if not post.get("geometry"):
            continue
        post_footprint = shape(post["geometry"])
        if not post_footprint.is_valid or post_footprint.geom_type not in {
            "Polygon",
            "MultiPolygon",
        }:
            continue
        for pre in ordered:
            if (
                pre["properties"]["grid:code"] != grid
                or _timestamp(pre["properties"]["datetime"]) >= request.event_started_at
                or not pre.get("geometry")
            ):
                continue
            pre_footprint = shape(pre["geometry"])
            if not pre_footprint.is_valid or pre_footprint.geom_type not in {
                "Polygon",
                "MultiPolygon",
            }:
                continue
            footprint = extent.intersection(post_footprint).intersection(pre_footprint)
            if footprint.area <= 0 or not available(pre):
                continue
            candidates.append((pre, post, footprint))
            used_grids.add(grid)
            break
    # Choose complementary footprint coverage, not seconds between granules of
    # the same orbit. The only spatial target is the independent incident AOI.
    # Fractions are relative to this small WGS84 AOI, not physical area estimates.
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    uncovered = extent
    while candidates and len(pairs) < request.maximum_pairs:
        candidates.sort(
            key=lambda entry: (
                -uncovered.intersection(entry[2]).area,
                -_timestamp(entry[1]["properties"]["datetime"]).timestamp(),
                float(entry[1]["properties"]["eo:cloud_cover"]),
                entry[1]["id"],
            )
        )
        pre, post, footprint = candidates.pop(0)
        marginal = uncovered.intersection(footprint).area
        if marginal <= 0:
            break
        post["fireviewer:selection"] = {
            "method": "independent_aoi_complementary_pair_footprint_v1",
            "pair_aoi_coverage_fraction": footprint.area / extent.area,
            "additional_aoi_coverage_fraction": marginal / extent.area,
        }
        pairs.append((pre, post))
        uncovered = uncovered.difference(footprint)
    return tuple(pairs)


def read_pair(
    pre: dict[str, Any],
    post: dict[str, Any],
    request: SatelliteCorpusRequest,
) -> tuple[Sentinel2ChangeWindow, float]:
    """Read AOI ranges directly; never download a full scene to the local disk."""

    from fireviewer_evidence_ingestion.mvp.sentinel2_radiometry import verify_cog_radiometry

    with rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_TIMEOUT="30",
        GDAL_HTTP_MAX_RETRY="1",
        GDAL_CACHEMAX=32 * 1024**2,
        CPL_VSIL_CURL_CACHE_SIZE=16 * 1024**2,
    ):
        with rasterio.open(_public_cog(post["assets"]["nir08"])) as target:
            bounds = transform_bounds("EPSG:4326", target.crs, *request.bbox)
            window = (
                from_bounds(*bounds, transform=target.transform).round_offsets().round_lengths()
            )
            window = window.intersection(Window(0, 0, target.width, target.height))
            factor = max(
                1,
                math.ceil(math.sqrt(window.width * window.height / request.maximum_window_pixels)),
            )
            width, height = math.ceil(window.width / factor), math.ceil(window.height / factor)
            transform = target.window_transform(window) * rasterio.Affine.scale(
                window.width / width, window.height / height
            )
            crs = target.crs
        if width * height > request.maximum_window_pixels:
            raise ValueError("satellite window exceeds its pixel limit")
        arrays = []
        for item in (pre, post):
            bands = {}
            radiometry = {}
            for asset_name, band_name in ASSETS.items():
                asset = item["assets"][asset_name]
                raster_band = asset.get("raster:bands", [{}])[0]
                with (
                    rasterio.open(_public_cog(asset)) as dataset,
                    WarpedVRT(
                        dataset,
                        crs=crs,
                        transform=transform,
                        width=width,
                        height=height,
                        resampling=Resampling.nearest
                        if asset_name == "scl"
                        else Resampling.bilinear,
                    ) as vrt,
                ):
                    raw = vrt.read(1, masked=True, out_dtype="float32")
                if asset_name == "scl":
                    values = raw.filled(0)
                else:
                    if "scale" not in raster_band:
                        raise ValueError("satellite reflectance scale is missing")
                    verified = verify_cog_radiometry(item, asset_name)
                    radiometry[asset_name] = verified
                    values = (raw * verified["scale"] + verified["effective_offset"]).filled(np.nan)
                bands[band_name] = np.asarray(values, dtype=np.float32)
            item["fireviewer:radiometry"] = radiometry
            arrays.append(bands)
    return Sentinel2ChangeWindow(
        pre=arrays[0],
        post=arrays[1],
        transform=transform,
        crs=str(crs),
        receipts_pre=(),
        receipts_post=(),
    ), max(abs(transform.a), abs(transform.e))


def _within_directory(path: Path, directory: Path) -> bool:
    """Compare resolved paths, including HF's Windows extended-length spelling."""
    target, root = path.resolve(), directory.resolve()
    if sys.platform == "win32":
        # HF prefixes long drive paths with \\?\; it does not change their
        # location. Resolve junctions first and normalize only for comparison.
        target = Path(str(target).removeprefix("\\\\?\\"))
        root = Path(str(root).removeprefix("\\\\?\\"))
    return target.is_relative_to(root)


def _verify_archived_window(
    item: dict[str, Any], dataset: Any, client: httpx.Client, cutoff: datetime
) -> dict[str, Any]:
    """Recover missing object identity by reproducing every frozen AOI pixel.

    Catalog creation was sufficient for availability in the 1.2.0 writer, so
    recent items legitimately have no archived ETag. A current ETag alone is
    not historical identity: require dated objects AND exact old-writer output.
    No source search, new acquisition or full-scene file is involved.
    """
    availability = item.get("fireviewer:availability", {})
    props = item["properties"]
    available = _timestamp(availability["available_at"])
    if (
        availability.get("basis") != "source_generation_and_catalog_creation_before_cutoff"
        or not _timestamp(props["s2:generation_time"]) <= available <= cutoff
        or available != _timestamp(props["created"])
    ):
        raise ValueError("archived Sentinel-2 catalog availability evidence differs")
    bands = []
    with rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_TIMEOUT="30",
        GDAL_HTTP_MAX_RETRY="1",
        GDAL_CACHEMAX=32 * 1024**2,
        CPL_VSIL_CURL_CACHE_SIZE=16 * 1024**2,
    ):
        for index, (name, _) in enumerate(ASSETS.items(), 1):
            asset = item["assets"][name]
            url = _public_cog(asset)
            before = client.head(url)
            before.raise_for_status()
            modified = parsedate_to_datetime(before.headers["last-modified"])
            etag, size = before.headers["etag"], int(before.headers["content-length"])
            if (
                modified.tzinfo is None
                or not _timestamp(props["s2:generation_time"]) <= modified <= available
                or not etag
                or size <= 0
            ):
                raise ValueError("Sentinel-2 source object cannot prove archived availability")
            with (
                rasterio.open(url) as source,
                WarpedVRT(
                    source,
                    crs=dataset.crs,
                    transform=dataset.transform,
                    width=dataset.width,
                    height=dataset.height,
                    resampling=Resampling.nearest if name == "scl" else Resampling.bilinear,
                ) as vrt,
            ):
                raw = vrt.read(1, masked=True, out_dtype="float32")
            if name == "scl":
                reproduced = raw.filled(0)
            else:
                coefficients = asset["raster:bands"][0]
                reproduced = (raw * coefficients["scale"] + coefficients.get("offset", 0)).filled(
                    np.nan
                )
            archived = dataset.read(index).astype(np.float32)
            if not np.array_equal(
                archived, np.asarray(reproduced, dtype=np.float32), equal_nan=True
            ):
                raise ValueError("Sentinel-2 source pixels differ from the archived AOI")
            if name != "scl" and not np.any(np.isfinite(archived)):
                raise ValueError("Sentinel-2 archived AOI has no radiometric identity evidence")
            after = client.head(url)
            after.raise_for_status()
            if any(
                before.headers[key] != after.headers.get(key)
                for key in ("etag", "content-length", "last-modified")
            ):
                raise ValueError("Sentinel-2 source object changed during archive verification")
            bands.append(
                {
                    "asset": name,
                    "href": url,
                    "etag": etag,
                    "byte_count": size,
                    "last_modified": modified.isoformat(),
                    "compared_pixel_count": int(archived.size),
                    "archived_window_sha256": sha256(archived.tobytes()).hexdigest(),
                }
            )
    return {
        "method": "exact-archived-aoi-old-writer-reproduction-v1",
        "bands": bands,
        "evaluation_reference_accessed": False,
    }


def read_archived_pair(
    pair: dict[str, Any],
    request: SatelliteCorpusRequest,
    input_root: Path,
) -> tuple[Sentinel2ChangeWindow, float]:
    """Repair a frozen AOI only, retaining its grid, samples and acquisition dates."""
    from fireviewer_evidence_ingestion.mvp.sentinel2_radiometry import verify_cog_radiometry

    if pair["processor_revision"] != "fireviewer-sentinel2-nbr-change-cpu-1.2.0":
        raise ValueError("archived Sentinel-2 encoding requires the known 1.2.0 writer")
    pre, post = pair["source_items"]
    if (
        not _timestamp(pre["properties"]["datetime"])
        < request.event_started_at
        <= _timestamp(post["properties"]["datetime"])
        <= request.evaluation_cutoff_at
    ):
        raise ValueError("archived Sentinel-2 pair is outside the incident/cutoff window")
    if _timestamp(pair["source_available_at"]) > request.evaluation_cutoff_at:
        raise ValueError("archived Sentinel-2 pair was unavailable at cutoff")
    exposed = {str(item[key]) for item in (pre, post) for key in ("id",)}
    exposed.update(str(item["properties"]["s2:product_uri"]) for item in (pre, post))
    if exposed.intersection(request.forbidden_input_refs):
        raise ValueError("evaluation reference entered archived Sentinel-2 repair")
    arrays, grid = [], None
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        for item in (pre, post):
            headers = item.get("fireviewer:availability", {}).get("object_headers", [])
            if not headers and item.get("fireviewer:availability", {}).get("basis") == (
                "source_generation_and_catalog_creation_before_cutoff"
            ):
                # Validated below against every archived pixel before correction.
                continue
            if {h["asset"] for h in headers} != set(ASSETS):
                raise ValueError("archived Sentinel-2 COG identity evidence is incomplete")
            for header in headers:
                url = _public_cog(item["assets"][header["asset"]])
                if url != header["href"]:
                    raise ValueError("archived Sentinel-2 COG identity differs")
                response = client.head(url)
                response.raise_for_status()
                if (
                    response.headers.get("etag") != header["etag"]
                    or int(response.headers.get("content-length", -1)) != header["byte_count"]
                ):
                    raise ValueError("original Sentinel-2 COG changed after archive freeze")
    for role, item in zip(("pre", "post"), (pre, post), strict=True):
        path = Path(pair["archived_paths"][role]).resolve()
        if not _within_directory(path, input_root) or not path.is_file():
            raise ValueError("archived Sentinel-2 path escaped input root")
        file = next((f for f in pair["files"] if f["path"].endswith(f"-{role}.tif")), None)
        if file is None or path.stat().st_size != file["byte_count"]:
            raise ValueError("archived Sentinel-2 raster size differs")
        with rasterio.open(path) as dataset:
            if (
                dataset.descriptions != tuple(ASSETS.values())
                or dataset.width * dataset.height > request.maximum_window_pixels
            ):
                raise ValueError("archived Sentinel-2 bands or pixel budget differ")
            if (
                dataset.tags().get("source_revision") != pair["source_revision_sha256"]
                or dataset.tags().get("processor_revision") != pair["processor_revision"]
            ):
                raise ValueError("archived Sentinel-2 raster identity differs")
            current_grid = (str(dataset.crs), dataset.transform, dataset.shape)
            if grid is not None and current_grid != grid:
                raise ValueError("archived Sentinel-2 pre/post grids differ")
            grid = current_grid
            if not item.get("fireviewer:availability", {}).get("object_headers"):
                with httpx.Client(timeout=30, follow_redirects=False) as client:
                    item["fireviewer:archive_identity"] = _verify_archived_window(
                        item, dataset, client, request.evaluation_cutoff_at
                    )
            bands = {}
            radiometry = {}
            for index, (name, band) in enumerate(ASSETS.items(), 1):
                values = dataset.read(index).astype(np.float32)
                if name != "scl":
                    verified = verify_cog_radiometry(item, name)
                    # Invert the exact old writer's additive offset; verified
                    # scale equality is required. NaN/no-data remain unchanged.
                    old_offset = float(item["assets"][name]["raster:bands"][0].get("offset", 0))
                    values = (
                        values - np.float32(old_offset) + np.float32(verified["effective_offset"])
                    )
                    radiometry[name] = verified
                bands[band] = values
            item["fireviewer:radiometry"] = radiometry
            arrays.append(bands)
    if grid is None:
        raise ValueError("archived Sentinel-2 grid is absent")
    return Sentinel2ChangeWindow(
        pre=arrays[0],
        post=arrays[1],
        crs=grid[0],
        transform=grid[1],
        receipts_pre=(),
        receipts_post=(),
    ), max(abs(grid[1].a), abs(grid[1].e))


def collect(
    request: SatelliteCorpusRequest,
    output_dir: Path,
    *,
    archived_pairs: list[dict[str, Any]] | None = None,
    input_root: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if archived_pairs is None:
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            pairs = discover_pairs(client, request)
    else:
        if input_root is None or not 1 <= len(archived_pairs) <= request.maximum_pairs:
            raise ValueError("archived Sentinel-2 repair has no bounded input root/pairs")
        pairs = tuple(tuple(p["source_items"]) for p in archived_pairs)
    results = []
    written = 0
    for pair_index, (pre, post) in enumerate(pairs):
        if archived_pairs is None:
            window, resolution = read_pair(pre, post, request)
        else:
            assert input_root is not None
            window, resolution = read_archived_pair(archived_pairs[pair_index], request, input_root)
        identity = sha256(
            json.dumps([PROCESSOR_REVISION, pre, post], sort_keys=True).encode()
        ).hexdigest()
        outcome = sentinel2_nbr_from_window(
            incident_bbox=request.bbox,
            observed_at=_timestamp(post["properties"]["datetime"]),
            source_revision_id=identity,
            resolution_m=resolution,
            window=window,
            dnbr_threshold=0.1,
            minimum_probability=0.5,
        )
        files = []
        # Conservative reservation covers TIFF metadata and compression overhead.
        reserve = (
            sum(array.nbytes for bands in (window.pre, window.post) for array in bands.values()) * 2
            + 1024**2
        )
        if written + reserve > request.maximum_output_bytes:
            raise ValueError("satellite materialization exceeds its output byte budget")
        for name, bands in (("pre", window.pre), ("post", window.post)):
            path = output_dir / f"{identity[:24]}-{name}.tif"
            with rasterio.open(
                path,
                "w",
                driver="COG",
                width=next(iter(bands.values())).shape[1],
                height=next(iter(bands.values())).shape[0],
                count=len(ASSETS),
                dtype="float32",
                crs=window.crs,
                transform=window.transform,
                compress="DEFLATE",
                nodata=float("nan"),
            ) as dataset:
                for index, band_name in enumerate(ASSETS.values(), 1):
                    dataset.write(bands[band_name], index)
                    dataset.set_band_description(index, band_name)
                dataset.update_tags(
                    source_revision=identity,
                    provider="EarthSearch/Sentinel-2",
                    processor_revision=PROCESSOR_REVISION,
                )
            written += path.stat().st_size
            files.append({"path": path.name, "byte_count": path.stat().st_size})
        results.append(
            {
                "source_revision_sha256": identity,
                "processor_revision": PROCESSOR_REVISION,
                "pre_product_id": pre["properties"]["s2:product_uri"],
                "post_product_id": post["properties"]["s2:product_uri"],
                "pre_item_id": pre["id"],
                "post_item_id": post["id"],
                "source_family_id": f"{post['properties']['platform']}.msi",
                "source_generated_at": max(
                    _timestamp(item["properties"]["s2:generation_time"]) for item in (pre, post)
                ).isoformat(),
                "source_available_at": max(
                    _timestamp(
                        item.get("fireviewer:availability", {}).get(
                            "available_at", item["properties"]["created"]
                        )
                    )
                    for item in (pre, post)
                ).isoformat(),
                "availability_basis": [
                    item.get(
                        "fireviewer:availability",
                        {"basis": "source_generation_and_catalog_creation_before_cutoff"},
                    )
                    for item in (pre, post)
                ],
                "observed_at": post["properties"]["datetime"],
                "resolution_m": resolution,
                "observations": list(outcome.observations),
                "coverage_geojson": outcome.valid_coverage_geojson,
                "metrics": outcome.coverage_metrics,
                "files": files,
                "source_items": [pre, post],
                "archived_parent_source_revision": None
                if archived_pairs is None
                else archived_pairs[pair_index]["source_revision_sha256"],
            }
        )
    return {
        "schema": "fireviewer.part4-satellite-corpus-result.v1",
        "pairs": results,
        "reason": "processed" if pairs else "no_admissible_pre_post_pair",
        "written_bytes": written,
        "paid_service_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archived-input", type=Path)
    args = parser.parse_args()
    if args.archived_input is not None:
        if args.archived_input.stat().st_size > 4 * 1024**2:
            raise ValueError("archived Sentinel-2 repair request exceeds byte limit")
        payload = json.loads(args.archived_input.read_text(encoding="utf-8"))
        request = SatelliteCorpusRequest.model_validate(payload["request"])
        print(
            json.dumps(
                collect(
                    request,
                    args.output_dir,
                    archived_pairs=payload["pairs"],
                    input_root=args.archived_input.parent,
                ),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    raw = sys.stdin.buffer.read(64 * 1024 + 1)
    if len(raw) > 64 * 1024:
        raise ValueError("satellite corpus request exceeds its byte limit")
    request = SatelliteCorpusRequest.model_validate_json(raw)
    print(json.dumps(collect(request, args.output_dir), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
