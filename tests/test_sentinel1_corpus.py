from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import rasterio
from pydantic import SecretStr
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform
from shapely.geometry import box, mapping

from fireviewer_evidence_ingestion.mvp.cdse_corpus import discover_items
from fireviewer_evidence_ingestion.mvp.satellite_corpus import SatelliteCorpusRequest
from fireviewer_evidence_ingestion.mvp.satellite_cpu import SatelliteCpuError
from fireviewer_evidence_ingestion.mvp.satellite_observations import Sentinel1ChangeWindow
from fireviewer_evidence_ingestion.mvp.sentinel1_corpus import (
    OpenEOAllowance,
    OpenEOCorpusReader,
    Sentinel1Pair,
    collect_sentinel1,
    process_graph,
    select_pairs,
)


def request() -> SatelliteCorpusRequest:
    return SatelliteCorpusRequest(
        incident_id="INC-test",
        bbox=(5, 44, 5.1, 44.1),
        event_started_at=datetime(2026, 7, 10, tzinfo=UTC),
        evaluation_cutoff_at=datetime(2026, 7, 20, tzinfo=UTC),
    )


def item(day: int, orbit: int = 88, name: str | None = None) -> dict[str, Any]:
    return {
        "id": name or f"S1A-{day}",
        "collection": "sentinel-1-grd",
        "geometry": mapping(box(4, 43, 6, 45)),
        "properties": {
            "datetime": f"2026-07-{day:02d}T10:00:00Z",
            "end_datetime": f"2026-07-{day:02d}T10:00:25Z",
            "created": f"2026-07-{day:02d}T11:00:00Z",
            "published": f"2026-07-{day:02d}T12:00:00Z",
            "processing:datetime": f"2026-07-{day:02d}T10:30:00Z",
            "platform": "sentinel-1a",
            "sat:relative_orbit": orbit,
            "sat:orbit_state": "ascending",
            "sar:instrument_mode": "IW",
            "sar:polarizations": ["VV", "VH"],
        },
    }


def test_discovery_rejects_late_reprocessing_and_unknown_availability() -> None:
    late, missing = item(12), item(13)
    late["properties"]["processing:datetime"] = "2026-08-01T00:00:00Z"
    del missing["properties"]["processing:datetime"]
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"features": [item(2), item(14), late, missing], "links": []}
            )
        )
    ) as client:
        selected, counts = discover_items(client, request(), "sentinel1")
    assert len(selected) == 2
    assert counts["late_or_unknown_availability"] == 2


def test_pairs_never_mix_orbit_platform_polarization_or_reference() -> None:
    wrong_platform, wrong_bands = item(17), item(18)
    wrong_platform["properties"]["platform"] = "sentinel-1b"
    wrong_bands["properties"]["sar:polarizations"] = ["HH", "HV"]
    selected, counts = select_pairs(
        [item(2), item(14), item(16, 89), wrong_platform, wrong_bands], request()
    )
    assert [(p.pre["id"], p.post["id"]) for p in selected] == [("S1A-2", "S1A-14")]
    assert counts["unpaired_post"] == 2
    assert counts["incompatible_orbit"] == 1
    selected, _ = select_pairs(
        [item(2), item(14)],
        request().model_copy(update={"forbidden_input_refs": ("CEMS_SENTINEL1_20260714_1000",)}),
    )
    assert selected == []


def test_graph_pins_product_track_and_calibrates_backscatter() -> None:
    graph = process_graph(item(14), request().bbox, 50)
    load = graph["load"]["arguments"]
    assert load["id"] == "SENTINEL1_GRD"
    assert load["properties"]["id"]["process_graph"]["eq"]["arguments"]["y"] == "S1A-14"
    assert load["properties"]["sat:relative_orbit"]["process_graph"]["eq"]["arguments"]["y"] == 88
    assert graph["backscatter"]["arguments"]["coefficient"] == "sigma0-ellipsoid"
    assert graph["grid"]["arguments"]["projection"] == 2154
    assert load["bands"] == ["VV", "VH"]
    assert all(
        stamp.endswith(".000000Z") or stamp.endswith(".000001Z")
        for stamp in load["temporal_extent"]
    )


def test_unconfigured_allowance_discovers_without_any_processing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.delenv("FIREVIEWER_CDSE_OPENEO_MAXIMUM_CREDITS", raising=False)
    calls = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(r.method)
        return httpx.Response(200, json={"features": [item(2), item(14)], "links": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = collect_sentinel1(request(), tmp_path, client=client)
    assert calls == ["GET"]
    assert len(result["candidate_pairs"]) == 1
    assert result["reason"] == "cdse_openeo_free_allowance_unavailable"
    assert result["products"] == []
    assert list(tmp_path.iterdir()) == []


def window(changed: bool) -> Sentinel1ChangeWindow:
    x, y = transform("EPSG:4326", "EPSG:2154", [5.02], [44.08])
    pre = {b: np.ones((10, 10), dtype="float32") for b in ("VV", "VH")}
    post = {b: np.full((10, 10), 0.1 if changed else 1, dtype="float32") for b in ("VV", "VH")}
    return Sentinel1ChangeWindow(pre, post, from_origin(x[0], y[0], 20, 20), "EPSG:2154", (), ())


@pytest.mark.parametrize("changed", [True, False])
def test_shared_processor_persists_bounded_cog_and_never_makes_negatives(
    tmp_path: Path, changed: bool
) -> None:
    class Reader:
        def read_pair(
            self, pair: Sentinel1Pair, request: SatelliteCorpusRequest
        ) -> Sentinel1ChangeWindow:
            return window(changed)

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"features": [item(2), item(14)], "links": []})
        )
    ) as client:
        result = collect_sentinel1(request(), tmp_path, client=client, reader=Reader())
    product = result["products"][0]
    assert product["observation_kind"] == "modelled_perimeter"
    assert product["target_state"] == "affected"
    assert product["source_family_id"] == "sentinel-1a.c-sar"
    assert product["resolution_m"] == 40
    assert product["coverage_geojson"] is not None
    assert bool(product["observations"]) is changed
    assert all(o["confidence"] <= 0.45 for o in product["observations"])
    assert product["independence_demonstrated"] is False
    with rasterio.open(tmp_path / product["files"][0]["path"]) as raster:
        assert raster.count == 4
        assert raster.descriptions == ("pre_VV", "pre_VH", "post_VV", "post_VH")


def allowance(total: float = 2) -> OpenEOAllowance:
    return OpenEOAllowance(
        token=SecretStr("test-only-" * 5), free_credits_verified=True, maximum_total_credits=total
    )


def test_budget_reserves_both_requests_and_failed_requests_are_not_refunded() -> None:
    calls = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(r.url.path)
        assert r.headers["authorization"].startswith("Bearer oidc/CDSE/")
        if r.url.path.endswith("/validation"):
            return httpx.Response(200, json={"errors": []})
        assert json.loads(r.content)["budget"] == 1
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        reader = OpenEOCorpusReader(client, allowance())
        with pytest.raises(SatelliteCpuError, match="processing_failed"):
            reader.read_pair(Sentinel1Pair(item(2), item(14)), request())
        assert reader.invocations == 1 and reader.reserved_credits == 1
        with pytest.raises(SatelliteCpuError, match="budget_exhausted"):
            reader.read_pair(Sentinel1Pair(item(2), item(14)), request())
    assert len(calls) == 2
    with pytest.raises(ValueError, match="verified free credits"):
        OpenEOAllowance(maximum_total_credits=1)


def test_reader_preserves_nodata_and_shared_grid() -> None:
    original = window(True)
    buffers = []
    for arrays in (original.pre, original.post):
        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff",
                width=10,
                height=10,
                count=2,
                dtype="float32",
                crs=original.crs,
                transform=original.transform,
                nodata=-999,
            ) as ds:
                for index, band in enumerate(("VV", "VH"), 1):
                    a = arrays[band].copy()
                    a[0, 0] = -999
                    ds.write(a, index)
            buffers.append(memory.read())

    def handler(r: httpx.Request) -> httpx.Response:
        if r.url.path.endswith("/validation"):
            return httpx.Response(200, json={"errors": []})
        return httpx.Response(200, content=buffers.pop(0), headers={"content-type": "image/tiff"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        reader = OpenEOCorpusReader(client, allowance())
        output = reader.read_pair(Sentinel1Pair(item(2), item(14)), request())
    assert np.isnan(output.pre["VV"][0, 0])
    assert reader.invocations == 2 and reader.reserved_credits == 2
    assert len(output.receipts_pre) == len(output.receipts_post) == 1
