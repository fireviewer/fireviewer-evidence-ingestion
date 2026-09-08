from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fireviewer_evidence_ingestion.mvp.cdse_corpus import (
    CLMS_ASSETS,
    CLMS_COLLECTION,
    collect_cdse,
    discover_items,
    processing_assets,
)
from fireviewer_evidence_ingestion.mvp.satellite_corpus import SatelliteCorpusRequest
from fireviewer_evidence_ingestion.mvp.satellite_observations import ClmsRasterWindow


def request() -> SatelliteCorpusRequest:
    return SatelliteCorpusRequest(
        incident_id="INC-test",
        bbox=(5, 44, 5.1, 44.1),
        event_started_at=datetime(2026, 7, 1, tzinfo=UTC),
        evaluation_cutoff_at=datetime(2026, 7, 8, tzinfo=UTC),
    )


def item(day: int, published: int) -> dict[str, Any]:
    return {
        "id": f"clms-{day}",
        "collection": CLMS_COLLECTION,
        "properties": {
            "datetime": f"2026-07-{day:02d}T00:00:00Z",
            "end_datetime": f"2026-07-{day:02d}T23:59:59Z",
            "created": f"2026-07-{published:02d}T00:00:00Z",
            "published": f"2026-07-{published:02d}T01:00:00Z",
        },
        "assets": {
            name: {
                "href": f"s3://eodata/CLMS/bio-geophysical/burnt_area/ba_global_300m_daily_v4/{day}/{name}.tiff",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "file:size": 1000,
                "proj:code": "EPSG:4326",
                "proj:shape": [10, 10],
                "proj:transform": [0.01, 0, 5, 0, -0.01, 44.1],
                "nodata": -32768,
                "data_type": "int16",
                "raster:scale": 1,
            }
            for name in CLMS_ASSETS
        },
    }


def test_cdse_rejects_future_availability_and_reference() -> None:
    missing = item(4, 6)
    del missing["properties"]["published"]
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200, json={"features": [item(3, 5), item(5, 7), item(6, 9), missing], "links": []}
            )
        )
    ) as client:
        selected, counts = discover_items(
            client, request().model_copy(update={"forbidden_input_refs": ("clms-3",)}), "clms"
        )
    assert [x["id"] for x in selected] == ["clms-5"]
    assert counts["late_or_unknown_availability"] == 2
    assert counts["reference_excluded"] == 1


def test_cdse_asset_must_stay_in_official_product_prefix() -> None:
    raw = item(5, 7)
    assert len(processing_assets(raw, "clms")) == 3
    raw["assets"][CLMS_ASSETS[0]]["href"] = "s3://eodata/private/foo.tif"
    with pytest.raises(ValueError, match="path rejected"):
        processing_assets(raw, "clms")


def test_clms_materializes_window_without_faking_negatives(tmp_path: Path) -> None:
    class Reader:
        def read_clms_window(self, **_: Any) -> ClmsRasterWindow:
            # No detections is not valid_negative evidence.
            return ClmsRasterWindow(
                day_of_burn=np.zeros((10, 10)),
                burn_probability=np.zeros((10, 10)),
                burn_fraction=np.zeros((10, 10)),
                valid_masks=tuple(np.ones((10, 10), dtype=bool) for _ in range(3)),
                transform=from_origin(5, 44.1, 0.01, 0.01),
                receipts=(),
            )

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, json={"features": [item(5, 7)], "links": []})
        )
    ) as client:
        result = collect_cdse(request(), tmp_path, "clms", reader=Reader(), client=client)  # type: ignore[arg-type]
    product = result["products"][0]
    assert product["observations"] == []
    assert product["observed_at"] == "2026-07-05T00:00:00+00:00"
    assert product["coverage_geojson"] is not None
    assert result["paid_service_used"] is False
    with rasterio.open(tmp_path / product["files"][0]["path"]) as raster:
        assert raster.count == 3
        assert raster.width == 10


def test_cdse_missing_credentials_is_explicit(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv("FIREVIEWER_CDSE_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("FIREVIEWER_CDSE_S3_SECRET_KEY", raising=False)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, json={"features": [item(5, 7)], "links": []})
        )
    ) as client:
        result = collect_cdse(request(), tmp_path, "clms", client=client)
    assert result["reason"] == "cdse_credentials_unavailable"
    assert result["products"] == []
    assert list(tmp_path.iterdir()) == []
