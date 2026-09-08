from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fireviewer_evidence_ingestion.mvp.sentinel2_radiometry import (
    native_dn_delta,
    safe_coefficients,
    verify_cog_radiometry,
)


def _xml(*, product: str = "test.SAFE", offset: str = "-1000", baseline: str = "04.00") -> bytes:
    return (
        f"<root><PRODUCT_URI>{product}</PRODUCT_URI><PROCESSING_BASELINE>{baseline}</PROCESSING_BASELINE>"
        f"<BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>{offset}</root>"
    ).encode()


def test_safe_metadata_not_catalog_flag_defines_original_encoding() -> None:
    xml = _xml(offset='<BOA_ADD_OFFSET band_id="8">-1000</BOA_ADD_OFFSET>')
    assert safe_coefficients(xml, "test.SAFE", 8) == (0.0001, -1000)
    assert safe_coefficients(_xml(offset="", baseline="03.01"), "test.SAFE", 8) == (0.0001, 0)
    for invalid in (_xml(offset=""), xml.replace(b"10000", b"nan"), xml + b"<!DOCTYPE forbidden>"):
        with pytest.raises(ValueError):
            safe_coefficients(invalid, "test.SAFE", 8)
    with pytest.raises(ValueError, match="identity"):
        safe_coefficients(xml, "different.SAFE", 8)
    with pytest.raises(ValueError, match="ambiguous"):
        safe_coefficients(xml, "test.SAFE", 12)


@pytest.mark.parametrize("delta", [0, 1000])
def test_raw_dn_verification_accepts_only_exact_known_transform(delta: int) -> None:
    cog = np.arange(1, 1025, dtype=np.uint16).reshape(32, 32)
    original = cog + delta
    assert native_dn_delta(cog, original, -1000) == (delta, 1023)
    original[0, 0] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        native_dn_delta(cog, original, -1000)
    assert native_dn_delta(np.zeros((32, 32)), original, -1000) == (None, 0)


def test_harmonized_dn_floor_is_verified_but_cannot_supply_encoding_evidence() -> None:
    original = np.array([1, 2, 900, 1000, 1001, 1100, 1300], dtype=np.uint16)
    cog = np.maximum(original.astype(np.int32) - 1000, 1).astype(np.uint16)
    assert native_dn_delta(cog, original, -1000) == (1000, 2)
    assert native_dn_delta(cog[:5], original[:5], -1000) == (None, 0)
    cog[2] = 2
    with pytest.raises(ValueError, match="inconsistent"):
        native_dn_delta(cog, original, -1000)


def test_catalog_flag_cannot_override_verified_transform(monkeypatch: Any) -> None:
    item = {
        "properties": {"s2:product_uri": "test.SAFE", "earthsearch:boa_offset_applied": False},
        "assets": {
            "nir08": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/test/B8A.tif",
                "raster:bands": [{"scale": 0.0001, "offset": -0.1}],
            },
            "nir08-jp2": {"href": "s3://sentinel-s2-l2a/tiles/30/T/XQ/2022/7/17/0/B8A.jp2"},
        },
    }
    monkeypatch.setattr(
        "fireviewer_evidence_ingestion.mvp.sentinel2_radiometry._verify",
        lambda *_args: {"scale": 0.0001, "effective_offset": 0.0},
    )
    receipt = verify_cog_radiometry(item, "nir08")
    assert receipt["effective_offset"] == 0
    assert receipt["catalog_offset"] == -0.1
    assert receipt["catalog_offset_applied"] is False
    item["assets"]["nir08"]["raster:bands"][0]["scale"] = 0.001  # type: ignore[index]
    with pytest.raises(ValueError, match="scale differs"):
        verify_cog_radiometry(item, "nir08")


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.test/a.tif",
        "https://sentinel-cogs.s3.us-west-2.amazonaws.com.evil.test/a.tif",
        "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/test.tif?token=test",
    ],
)
def test_encoding_verifier_rejects_untrusted_sources(url: str) -> None:
    with pytest.raises(ValueError, match="untrusted"):
        verify_cog_radiometry({"assets": {"nir08": {"href": url}}}, "nir08")
