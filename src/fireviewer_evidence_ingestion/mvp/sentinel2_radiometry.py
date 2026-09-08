"""Verify EarthSearch COG encoding against the same SAFE product, not fire labels.

Some migrated 2022 catalog records apply the SAFE offset to already harmonized
COGs. Neither the catalog flag nor a plausible-looking NBR is encoding evidence.
Small, native-grid DN samples determine whether that offset is already applied.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from typing import Any
from xml.etree import ElementTree

import httpx
import numpy as np
import rasterio
from defusedxml.ElementTree import fromstring as safe_fromstring
from rasterio.windows import Window

_ORIGINAL_ROOT = "https://sentinel-s2-l2a.s3.eu-central-1.amazonaws.com/"
_COG_ROOT = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/"
_BANDS = {
    "red": ("B04", 3, 10),
    "nir08": ("B8A", 8, 20),
    "swir16": ("B11", 11, 20),
    "swir22": ("B12", 12, 20),
}
RADIOMETRY_REVISION = "same-product-native-dn-verification-v1"


def _small_get(client: httpx.Client, url: str) -> bytes:
    with client.stream("GET", url) as response:
        response.raise_for_status()
        result = bytearray()
        for chunk in response.iter_bytes():
            if len(result) + len(chunk) > 2 * 1024**2:
                raise ValueError("Sentinel-2 encoding metadata exceeds byte limit")
            result.extend(chunk)
    return bytes(result)


def safe_coefficients(xml: bytes, product_uri: str, band_id: int) -> tuple[float, float]:
    """Read the original quantification and per-band DN offset, fail closed."""
    if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
        raise ValueError("unsafe Sentinel-2 encoding XML")
    root = safe_fromstring(xml, forbid_dtd=True)
    values: dict[str, list[ElementTree.Element]] = {}
    for element in root.iter():
        values.setdefault(element.tag.rsplit("}", 1)[-1], []).append(element)
    identities = values.get("PRODUCT_URI", [])
    if len(identities) != 1 or identities[0].text != product_uri:
        raise ValueError("Sentinel-2 original product identity differs")
    quantifications = values.get("BOA_QUANTIFICATION_VALUE", [])
    if len(quantifications) != 1:
        raise ValueError("Sentinel-2 original quantification is ambiguous")
    quantification = float(quantifications[0].text or "nan")
    if not math.isfinite(quantification) or quantification <= 0:
        raise ValueError("Sentinel-2 original quantification is invalid")
    offsets = [e for e in values.get("BOA_ADD_OFFSET", []) if e.get("band_id") == str(band_id)]
    if len(offsets) == 1:
        offset = float(offsets[0].text or "nan")
    elif not values.get("BOA_ADD_OFFSET"):
        baseline = values.get("PROCESSING_BASELINE", [])
        if len(baseline) != 1 or not 0 < float(baseline[0].text or "nan") < 4:
            raise ValueError("Sentinel-2 original offset is missing")
        offset = 0.0
    else:
        raise ValueError("Sentinel-2 original offset is ambiguous")
    if not math.isfinite(offset) or offset > 0:
        raise ValueError("Sentinel-2 original offset is invalid")
    return 1.0 / quantification, offset


def native_dn_delta(
    cog: np.ndarray, original: np.ndarray, offset_dn: float
) -> tuple[int | None, int]:
    """Accept an exact encoding transform only; never fit it to an outcome."""
    if cog.shape != original.shape:
        raise ValueError("Sentinel-2 radiometry sample shapes differ")
    valid = (cog > 0) & (original > 0) & (cog < 65535) & (original < 65535)
    count = int(np.count_nonzero(valid))
    if not count:
        return None, 0
    original_dn = original[valid].astype(np.float64)
    cog_dn = cog[valid].astype(np.float64)
    # The COG writer reserves zero for no-data and clamps harmonized positive
    # DN to one. Verify that exact transform, including the clamped samples;
    # those samples alone cannot identify which encoding was used.
    candidates = {
        int(delta)
        for delta in {0.0, -offset_dn}
        if np.all(cog_dn == np.maximum(original_dn - delta, 1))
    }
    if not candidates:
        raise ValueError("Sentinel-2 COG encoding is inconsistent with original DN")
    informative = int(np.count_nonzero(cog_dn > 1))
    if len(candidates) != 1 or informative == 0:
        return None, 0
    return next(iter(candidates)), informative


@lru_cache(maxsize=128)
def _verify(cog_url: str, product_uri: str, jp2_base: str, asset_name: str) -> dict[str, Any]:
    band, band_id, resolution = _BANDS[asset_name]
    jp2_url = _ORIGINAL_ROOT + jp2_base + f"/R{resolution}m/{band}.jp2"
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        tile = json.loads(_small_get(client, _ORIGINAL_ROOT + jp2_base + "/tileInfo.json"))
        product_name = product_uri.removesuffix(".SAFE")
        if tile.get("productName") != product_name or tile.get("path") != jp2_base:
            raise ValueError("Sentinel-2 original tile identity differs")
        product_path = str(tile.get("productPath", ""))
        if (
            re.fullmatch(r"products/\d{4}/\d{1,2}/\d{1,2}/" + re.escape(product_name), product_path)
            is None
        ):
            raise ValueError("Sentinel-2 original product metadata path differs")
        metadata_url = _ORIGINAL_ROOT + product_path + "/metadata.xml"
        scale, offset_dn = safe_coefficients(_small_get(client, metadata_url), product_uri, band_id)
    sample_receipts = []
    deltas: set[int] = set()
    count = 0
    with (
        rasterio.Env(
            AWS_NO_SIGN_REQUEST="YES",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_TIMEOUT="30",
            GDAL_HTTP_MAX_RETRY="1",
            GDAL_CACHEMAX=32 * 1024**2,
            CPL_VSIL_CURL_CACHE_SIZE=16 * 1024**2,
        ),
        rasterio.open(cog_url) as cog,
        rasterio.open(jp2_url) as original,
    ):
        if (
            cog.crs != original.crs
            or cog.transform != original.transform
            or cog.shape != original.shape
        ):
            raise ValueError("Sentinel-2 original and COG native grids differ")
        # Locate valid samples from a small source overview, not an incident or
        # reference contour. Partial granules need not contain the tile centre.
        overview = cog.read(1, out_shape=(16, 16), masked=True)
        positions = np.argwhere((overview.filled(0) > 0) & ~np.ma.getmaskarray(overview))
        for index in np.unique(
            np.linspace(0, len(positions) - 1, min(8, len(positions)), dtype=int)
        ):
            y, x = positions[index]
            col = min(cog.width - 32, max(0, int((x + 0.5) * cog.width / 16) - 16))
            row = min(cog.height - 32, max(0, int((y + 0.5) * cog.height / 16) - 16))
            window = Window(col, row, 32, 32)
            delta, valid = native_dn_delta(
                cog.read(1, window=window), original.read(1, window=window), offset_dn
            )
            if delta is not None:
                deltas.add(delta)
                count += valid
                sample_receipts.append(
                    {
                        "window": list(window.flatten()),
                        "valid_pixels": valid,
                        "original_minus_cog_dn": delta,
                    }
                )
            if len(deltas) > 1:
                raise ValueError("Sentinel-2 COG encoding changes across native-grid samples")
            if len(sample_receipts) >= 3 and count >= 512:
                break
    if len(sample_receipts) < 3 or count < 512 or len(deltas) != 1:
        raise ValueError("Sentinel-2 COG encoding has insufficient native-grid evidence")
    delta = next(iter(deltas))
    return {
        "method": RADIOMETRY_REVISION,
        "product_uri": product_uri,
        "asset_name": asset_name,
        "cog_url": cog_url,
        "original_url": jp2_url,
        "metadata_url": metadata_url,
        "scale": scale,
        "safe_offset_dn": offset_dn,
        "original_minus_cog_dn": delta,
        "effective_offset": (offset_dn + delta) * scale,
        "positive_dn_floor": 1,
        "valid_pixels": count,
        "samples": sample_receipts,
        "evaluation_reference_accessed": False,
    }


def verify_cog_radiometry(item: dict[str, Any], asset_name: str) -> dict[str, Any]:
    """Public, read-only verification for exactly the selected acquisition."""
    if asset_name not in _BANDS:
        raise ValueError("unsupported Sentinel-2 reflectance asset")
    asset = item["assets"][asset_name]
    cog_url = str(asset["href"])
    if not cog_url.startswith(_COG_ROOT) or re.search(r"[^A-Za-z0-9_./:-]", cog_url):
        raise ValueError("untrusted Sentinel-2 COG encoding URL")
    href = str(item["assets"].get(asset_name + "-jp2", {}).get("href", ""))
    band = _BANDS[asset_name][0]
    match = re.fullmatch(
        r"s3://sentinel-s2-l2a/(tiles/\d{1,2}/[A-Z]/[A-Z]{2}/\d{4}/\d{1,2}/\d{1,2}/\d+)/(?:R\d+m/)?"
        + band
        + r"\.jp2",
        href,
    )
    if match is None:
        raise ValueError("Sentinel-2 original encoding asset is missing or untrusted")
    receipt = dict(
        _verify(cog_url, str(item["properties"]["s2:product_uri"]), match[1], asset_name)
    )
    raster_band = asset.get("raster:bands", [{}])[0]
    if "scale" not in raster_band:
        raise ValueError("satellite reflectance scale is missing")
    if not math.isclose(float(raster_band["scale"]), receipt["scale"], rel_tol=1e-9):
        raise ValueError("Sentinel-2 catalog scale differs from original quantification")
    receipt["catalog_offset"] = float(raster_band.get("offset", 0))
    receipt["catalog_offset_applied"] = item["properties"].get("earthsearch:boa_offset_applied")
    return receipt
