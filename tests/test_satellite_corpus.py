from __future__ import annotations

from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping, shape

from fireviewer_evidence_ingestion.mvp.satellite_corpus import (
    ASSETS,
    SatelliteCorpusRequest,
    _public_cog,
    collect,
    discover_pairs,
    read_pair,
)
from fireviewer_evidence_ingestion.mvp.satellite_observations import (
    Sentinel2ChangeWindow,
    sentinel2_nbr_from_window,
)


def _request(**updates: Any) -> SatelliteCorpusRequest:
    return SatelliteCorpusRequest.model_validate(
        {
            "incident_id": "INC-REAL-SOURCE-TEST",
            "bbox": [5.0, 44.0, 5.1, 44.1],
            "event_started_at": datetime(2022, 7, 10, tzinfo=UTC),
            "evaluation_cutoff_at": datetime(2022, 7, 15, tzinfo=UTC),
            **updates,
        }
    )


def _item(day: int, *, generated_day: int | None = None) -> dict[str, Any]:
    stamp = f"2022-07-{day:02d}T10:00:00Z"
    return {
        "id": f"scene-{day}",
        "geometry": mapping(box(5.0, 44.0, 5.1, 44.1)),
        "properties": {
            "datetime": stamp,
            "created": f"2022-07-{generated_day or day:02d}T12:00:00Z",
            "s2:generation_time": f"2022-07-{generated_day or day:02d}T11:00:00Z",
            "s2:product_uri": f"S2A_MSIL2A_202207{day:02d}T100000_N0400_TEST.SAFE",
            "grid:code": "MGRS-31T",
            "eo:cloud_cover": 0,
            "platform": "sentinel-2a",
        },
        "assets": {
            key: {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
                f"sentinel-s2-l2a-cogs/{day}/{key}.tif"
            }
            for key in ASSETS
        },
    }


def test_pairs_reject_future_reprocessing_and_reference_scene() -> None:
    items = [_item(9), _item(12), _item(14, generated_day=20), _item(13)]
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _req: httpx.Response(200, json={"features": items, "links": []})
        )
    )
    request = _request(forbidden_input_refs=["S2A_MSIL1C_20220713T100000_reference.zip"])
    pairs = discover_pairs(client, request)
    assert len(pairs) == 1
    assert (pairs[0][0]["id"], pairs[0][1]["id"]) == ("scene-9", "scene-12")


def test_pairs_reject_late_catalog_availability_and_renamed_reference() -> None:
    late = _item(14)
    late["properties"]["created"] = "2022-07-20T00:00:00Z"
    items = [_item(9), _item(12), _item(13), late]
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _req: httpx.Response(200, json={"features": items, "links": []})
        )
    ) as client:
        pairs = discover_pairs(
            client, _request(forbidden_input_refs=["EMSR900_SENTINEL2_20220713_0959_ORTHO.tif"])
        )
    assert [post["id"] for _, post in pairs] == ["scene-12"]


def test_public_reader_rejects_paid_or_untrusted_assets() -> None:
    for href in (
        "s3://requester-pays/image.tif",
        "http://127.0.0.1/x",
        "https://example.com/image.tif",
    ):
        with pytest.raises(ValueError, match="allowlisted"):
            _public_cog({"href": href})


def test_pairs_select_complementary_aoi_not_later_granule_seconds() -> None:
    items = []
    for grid, bounds, second in (
        ("coastal", (5.095, 44.0, 5.1, 44.02), 9),
        ("west", (5.0, 44.0, 5.07, 44.1), 1),
        ("east", (5.06, 44.0, 5.1, 44.1), 2),
        ("duplicate", (5.0, 44.0, 5.069, 44.1), 3),
    ):
        for day in (9, 12):
            item = _item(day)
            item["id"] += "-" + grid
            item["properties"]["grid:code"] = grid
            item["properties"]["datetime"] = f"2022-07-{day:02d}T10:00:{second:02d}Z"
            item["geometry"] = mapping(box(*bounds))
            items.append(item)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _req: httpx.Response(200, json={"features": items, "links": []})
        )
    ) as client:
        pairs = discover_pairs(client, _request())
    assert [post["properties"]["grid:code"] for _, post in pairs] == ["west", "east"]
    assert sum(
        post["fireviewer:selection"]["additional_aoi_coverage_fraction"] for _, post in pairs
    ) == pytest.approx(1.0)


def test_pairs_require_overlapping_pre_post_footprints() -> None:
    pre, post = _item(9), _item(12)
    pre["geometry"] = mapping(box(5.08, 44.0, 5.1, 44.1))
    post["geometry"] = mapping(box(5.0, 44.0, 5.07, 44.1))
    for missing in (False, True):
        if missing:
            del pre["geometry"]
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(200, json={"features": [pre, post], "links": []})
            )
        ) as client:
            assert discover_pairs(client, _request()) == ()


@pytest.mark.parametrize("water_phase", ["pre", "post", "both"])
def test_nbr_water_is_observable_but_never_positive_or_negative(water_phase: str) -> None:
    pre = {name: np.full((10, 10), 0.8, dtype=np.float32) for name in ASSETS.values()}
    post = {name: value.copy() for name, value in pre.items()}
    pre["B12_20m"][:] = 0.1
    post["B12_20m"][:] = 0.7
    post["B8A_20m"][:] = 0.2
    pre["SCL_20m"][:] = post["SCL_20m"][:] = 4
    for phase, bands in (("pre", pre), ("post", post)):
        if water_phase in {phase, "both"}:
            bands["SCL_20m"][:, :5] = 6
    window = Sentinel2ChangeWindow(
        pre=pre,
        post=post,
        transform=from_origin(5, 44.1, 0.01, 0.01),
        crs="EPSG:4326",
        receipts_pre=(),
        receipts_post=(),
    )
    outcome = sentinel2_nbr_from_window(
        incident_bbox=_request().bbox,
        observed_at=_request().evaluation_cutoff_at,
        source_revision_id="water-regression",
        resolution_m=20,
        window=window,
        dnbr_threshold=0.1,
        minimum_probability=0.5,
    )
    assert outcome.coverage_metrics["valid_pixel_count"] == 100
    assert outcome.coverage_metrics["water_positive_excluded_count"] == 50
    assert outcome.coverage_metrics["burned_pixel_count"] == 50
    assert shape(outcome.valid_coverage_geojson).area == pytest.approx(0.01)
    assert all(shape(row["geometry_geojson"]).bounds[0] >= 5.05 for row in outcome.observations)
    # No-signal water-only windows retain coverage without manufacturing negatives.
    pre["SCL_20m"][:] = 6
    empty = sentinel2_nbr_from_window(
        incident_bbox=_request().bbox,
        observed_at=_request().evaluation_cutoff_at,
        source_revision_id="water-only",
        resolution_m=20,
        window=window,
        dnbr_threshold=0.1,
        minimum_probability=0.5,
    )
    assert empty.observations == ()
    assert empty.valid_coverage_geojson is not None


@pytest.mark.parametrize("late_object", [False, True])
def test_migrated_catalog_requires_all_original_object_dates(late_object: bool) -> None:
    items = [_item(9), _item(12)]
    for entry in items:
        entry["properties"]["created"] = "2022-11-03T00:00:00Z"
    headers_seen = []

    def respond(req: httpx.Request) -> httpx.Response:
        if req.method != "HEAD":
            return httpx.Response(200, json={"features": items, "links": []})
        headers_seen.append(str(req.url))
        day = int(req.url.path.split("/")[-2])
        stamp = (
            "Wed, 20 Jul 2022 14:00:00 GMT"
            if late_object and req.url.path.endswith("scl.tif")
            else f"{('Sat' if day == 9 else 'Tue')}, {day:02d} Jul 2022 14:00:00 GMT"
        )
        return httpx.Response(
            200, headers={"last-modified": stamp, "etag": '"stable"', "content-length": "100000"}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        pairs = discover_pairs(client, _request())
    assert len(headers_seen) == (5 if late_object else 10)
    assert len(pairs) == (0 if late_object else 1)
    if pairs:
        proof = pairs[0][1]["fireviewer:availability"]
        assert len(proof["object_headers"]) == 5
        assert proof["available_at"].startswith("2022-07-12")


def test_window_reader_aligns_bands_and_applies_stac_scale_offset(monkeypatch: Any) -> None:
    pre, post = _item(9), _item(12)
    transform = from_origin(500_000, 4_900_000, 20, 20)
    paths = {}
    with ExitStack() as stack:
        for item in (pre, post):
            for asset_name in ASSETS:
                memory = stack.enter_context(MemoryFile())
                with memory.open(
                    driver="GTiff",
                    width=32,
                    height=32,
                    count=1,
                    dtype="uint16",
                    crs="EPSG:32631",
                    transform=transform,
                    nodata=0,
                ) as dataset:
                    values = np.full((32, 32), 4 if asset_name == "scl" else 5000, dtype="uint16")
                    values[0, 0] = 0
                    dataset.write(values, 1)
                asset = item["assets"][asset_name]
                asset["raster:bands"] = [{"scale": 0.0001, "offset": -0.1}]
                paths[asset["href"]] = memory.name
        monkeypatch.setattr(
            "fireviewer_evidence_ingestion.mvp.satellite_corpus._public_cog",
            lambda asset: paths[asset["href"]],
        )
        monkeypatch.setattr(
            "fireviewer_evidence_ingestion.mvp.sentinel2_radiometry.verify_cog_radiometry",
            lambda *_args: {"scale": 0.0001, "effective_offset": -0.1},
        )
        request = _request(
            bbox=transform_bounds(
                "EPSG:32631", "EPSG:4326", 500_000, 4_899_360, 500_640, 4_900_000
            ),
            maximum_window_pixels=256,
        )
        window, resolution = read_pair(pre, post, request)
        assert resolution >= 40
        assert window.pre["B8A_20m"].size <= 256
        assert window.pre["B8A_20m"].shape == window.post["SCL_20m"].shape
        assert np.nanmax(window.pre["B8A_20m"]) == pytest.approx(0.4)
        assert np.nanmax(window.post["SCL_20m"]) == 4
        del pre["assets"]["nir08"]["raster:bands"][0]["scale"]
        with pytest.raises(ValueError, match="scale is missing"):
            read_pair(pre, post, request)


def test_shared_nbr_materializes_only_bounded_aoi_cogs(tmp_path: Path, monkeypatch: Any) -> None:
    pre = {name: np.full((10, 10), 0.8, dtype=np.float32) for name in ASSETS.values()}
    post = {name: array.copy() for name, array in pre.items()}
    pre["B12_20m"][:] = 0.1
    post["B12_20m"][:] = 0.7
    post["B8A_20m"][:] = 0.2
    pre["SCL_20m"][:] = 4
    post["SCL_20m"][:] = 4
    post["SCL_20m"][0, 0] = 9
    window = Sentinel2ChangeWindow(
        pre=pre,
        post=post,
        transform=from_origin(5, 44.1, 0.01, 0.01),
        crs="EPSG:4326",
        receipts_pre=(),
        receipts_post=(),
    )
    monkeypatch.setattr(
        "fireviewer_evidence_ingestion.mvp.satellite_corpus.discover_pairs",
        lambda *_args: ((_item(9), _item(12)),),
    )
    monkeypatch.setattr(
        "fireviewer_evidence_ingestion.mvp.satellite_corpus.read_pair", lambda *_args: (window, 20.0)
    )
    result = collect(_request(), tmp_path)
    assert result["paid_service_used"] is False
    pair = result["pairs"][0]
    assert pair["metrics"]["valid_pixel_count"] == 99
    assert pair["observations"]
    assert len(pair["files"]) == 2
    for record in pair["files"]:
        with rasterio.open(tmp_path / record["path"]) as dataset:
            assert dataset.count == 5
            assert dataset.width == 10
            assert dataset.descriptions == tuple(ASSETS.values())
    with pytest.raises(ValueError, match="output byte budget"):
        collect(_request(maximum_output_bytes=1024), tmp_path / "too-small")


def test_nbr_rejects_negative_spectra_and_separates_clouds_from_nodata() -> None:
    pre = {name: np.full((10, 10), 0.4, dtype=np.float32) for name in ASSETS.values()}
    post = {name: array.copy() for name, array in pre.items()}
    pre["SCL_20m"][:] = 4
    post["SCL_20m"][:] = 4
    pre["B12_20m"][:] = -0.39  # A positive sum used to admit NBR=79.
    pre["SCL_20m"][0, :5] = 0
    post["SCL_20m"][0, 5:] = 9
    result = sentinel2_nbr_from_window(
        incident_bbox=(5, 44, 5.1, 44.1),
        observed_at=datetime(2022, 7, 12, tzinfo=UTC),
        source_revision_id="a" * 64,
        resolution_m=20,
        window=Sentinel2ChangeWindow(
            pre=pre,
            post=post,
            transform=from_origin(5, 44.1, 0.01, 0.01),
            crs="EPSG:4326",
            receipts_pre=(),
            receipts_post=(),
        ),
        dnbr_threshold=0.1,
        minimum_probability=0.5,
    )
    assert result.observations == ()
    assert result.valid_coverage_geojson is None
    assert result.coverage_metrics["spectral_invalid_pixel_count"] == 90
    assert result.coverage_metrics["cloud_fraction"] == 0.05
    assert result.coverage_metrics["no_data_fraction"] == 0.05


def test_archive_containment_accepts_windows_extended_paths_not_siblings(tmp_path: Path) -> None:
    import sys

    from fireviewer_evidence_ingestion.mvp.satellite_corpus import _within_directory

    path = tmp_path / "nested" / "scene.tif"
    path.parent.mkdir()
    path.touch()
    assert _within_directory(path, tmp_path)
    assert not _within_directory(tmp_path.parent / "elsewhere" / path.name, tmp_path)
    if sys.platform == "win32":
        assert _within_directory(Path("\\\\?\\" + str(path.resolve())), tmp_path)
        assert not _within_directory(Path("\\\\?\\" + str(tmp_path.parent / "elsewhere")), tmp_path)


@pytest.mark.parametrize("catalog_only", [False, True])
def test_archived_repair_preserves_grid_and_uses_no_scene_discovery(
    tmp_path: Path, monkeypatch: Any, catalog_only: bool
) -> None:
    from fireviewer_evidence_ingestion.mvp.satellite_corpus import read_archived_pair

    pre_item, post_item = _item(9), _item(12)
    pair: dict[str, Any] = {
        "source_items": [pre_item, post_item],
        "source_revision_sha256": "a" * 64,
        "processor_revision": "fireviewer-sentinel2-nbr-change-cpu-1.2.0",
        "source_available_at": "2022-07-12T12:00:00Z",
        "archived_paths": {},
        "files": [],
    }
    transform = from_origin(5, 44.1, 0.01, 0.01)
    for role, item in (("pre", pre_item), ("post", post_item)):
        path = tmp_path / f"original-{role}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=5,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=float("nan"),
        ) as dst:
            for index, (name, band) in enumerate(ASSETS.items(), 1):
                dst.write(np.full((10, 10), 4 if name == "scl" else 0.3, dtype=np.float32), index)
                dst.set_band_description(index, band)
                item["assets"][name]["raster:bands"] = [{"scale": 0.0001, "offset": -0.1}]
            dst.update_tags(source_revision="a" * 64, processor_revision=pair["processor_revision"])
        pair["archived_paths"][role] = str(path)
        pair["files"].append({"path": path.name, "byte_count": path.stat().st_size})
        item["fireviewer:availability"] = {
            "available_at": item["properties"]["created"],
            "object_headers": [
                {
                    "asset": name,
                    "href": item["assets"][name]["href"],
                    "etag": "test",
                    "byte_count": 100,
                }
                for name in ASSETS
            ],
        }
    client_type = httpx.Client
    monkeypatch.setattr(
        "fireviewer_evidence_ingestion.mvp.satellite_corpus.httpx.Client",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, headers={"etag": "test", "content-length": "100"}, request=req
                )
            )
        ),
    )
    if catalog_only:
        for item in (pre_item, post_item):
            item["fireviewer:availability"].pop("object_headers")
            item["fireviewer:availability"]["basis"] = (
                "source_generation_and_catalog_creation_before_cutoff"
            )
        monkeypatch.setattr(
            "fireviewer_evidence_ingestion.mvp.satellite_corpus._verify_archived_window",
            lambda *args: {"method": "verified-test-window"},
        )
    monkeypatch.setattr(
        "fireviewer_evidence_ingestion.mvp.sentinel2_radiometry.verify_cog_radiometry",
        lambda *_args: {"scale": 0.0001, "effective_offset": 0.0},
    )
    monkeypatch.setattr(
        "fireviewer_evidence_ingestion.mvp.satellite_corpus.discover_pairs",
        lambda *_args: pytest.fail("archive repair must not discover scenes"),
    )
    result = collect(_request(), tmp_path / "out", archived_pairs=[pair], input_root=tmp_path)
    assert result["pairs"][0]["processor_revision"].endswith("1.3.0")
    assert result["pairs"][0]["archived_parent_source_revision"] == "a" * 64
    with rasterio.open(tmp_path / "out" / result["pairs"][0]["files"][0]["path"]) as dataset:
        assert dataset.transform == transform
        assert np.nanmean(dataset.read(2)) == pytest.approx(0.4)
    pair["processor_revision"] = "fireviewer-sentinel2-nbr-change-cpu-1.3.0"
    with pytest.raises(ValueError, match=r"known 1\.2\.0"):
        read_archived_pair(pair, _request(), tmp_path)


@pytest.mark.parametrize("fault", [None, "pixel", "mask", "late", "changed", "empty", "basis"])
def test_catalog_archive_identity_requires_exact_pixels_and_dated_stable_objects(
    tmp_path: Path, monkeypatch: Any, fault: str | None
) -> None:
    from fireviewer_evidence_ingestion.mvp.satellite_corpus import _verify_archived_window

    item = _item(9)
    item["fireviewer:availability"] = {
        "available_at": item["properties"]["created"],
        "basis": "unknown"
        if fault == "basis"
        else ("source_generation_and_catalog_creation_before_cutoff"),
    }
    transform = from_origin(5, 44.1, 0.01, 0.01)
    archived_path = tmp_path / "archived.tif"
    paths = {}
    with rasterio.open(
        archived_path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=5,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=float("nan"),
    ) as archived:
        for index, name in enumerate(ASSETS, 1):
            raw = np.full((10, 10), 4 if name == "scl" else 4000, dtype=np.uint16)
            raw[0, 0] = 0
            if fault == "empty" and name != "scl":
                raw[:] = 0
            values = raw.astype(np.float32)
            if name != "scl":
                values = (np.ma.array(values, mask=raw == 0) * 0.0001 - 0.1).filled(np.nan)
            archived.write(values, index)
            if fault == "pixel":
                raw[1, 1] += 1
            if fault == "mask":
                raw[1, 1] = 0
            path = tmp_path / f"{name}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=10,
                height=10,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=transform,
                nodata=0,
            ) as source:
                source.write(raw, 1)
            asset = item["assets"][name]
            asset["raster:bands"] = [{"scale": 0.0001, "offset": -0.1}]
            paths[asset["href"]] = str(path)
    monkeypatch.setattr(
        "fireviewer_evidence_ingestion.mvp.satellite_corpus._public_cog", lambda asset: paths[asset["href"]]
    )
    # HEAD is mocked but GDAL reads and reprojects real raster files.
    calls = 0

    def head(url: str) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            request=httpx.Request("HEAD", "https://source.test"),
            headers={
                "last-modified": "Sat, 09 Jul 2022 13:00:00 GMT"
                if fault == "late"
                else ("Sat, 09 Jul 2022 11:30:00 GMT"),
                "etag": str(calls) if fault == "changed" else "fixed",
                "content-length": "100",
            },
        )

    from types import SimpleNamespace

    with rasterio.open(archived_path) as archived:
        if fault is not None:
            with pytest.raises(ValueError):
                _verify_archived_window(
                    item, archived, SimpleNamespace(head=head), _request().evaluation_cutoff_at
                )
        else:
            receipt = _verify_archived_window(
                item, archived, SimpleNamespace(head=head), _request().evaluation_cutoff_at
            )
            assert len(receipt["bands"]) == 5
            assert all(b["compared_pixel_count"] == 100 for b in receipt["bands"])
            assert all(len(b["archived_window_sha256"]) == 64 for b in receipt["bands"])
