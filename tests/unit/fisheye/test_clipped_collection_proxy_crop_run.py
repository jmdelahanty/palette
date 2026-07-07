from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.shared import flat_roi_cache as flat_cache_mod
from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.flat_roi_cache import open_flat_roi_cache
from fisheye.utils.create_clipped_collection_proxy_crop_run import (
    PROXY_CROP_RUN_SCHEMA,
    create_clipped_collection_proxy_crop_run,
)
from tests.unit.fisheye.test_clipped_collection_flat_roi_cache import _make_clipped_collection_archive
from tests.unit.fisheye.test_flat_roi_cache import _FakePynvvcReader


def _build_tiny_clipped_cache(monkeypatch, tmp_path: Path) -> tuple[Path, dict, np.ndarray]:
    from fisheye.shared.clipped_collection_flat_roi_cache import build_clipped_collection_flat_roi_cache

    zarr_path, frame_index_path, frames, expected = _make_clipped_collection_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root.attrs["height"] = 4
    root.attrs["width"] = 5
    monkeypatch.setattr(
        flat_cache_mod,
        "_open_pynvvc_luma_reader",
        lambda _video_path: _FakePynvvcReader(frames),
    )
    manifest = build_clipped_collection_flat_roi_cache(
        zarr_path=zarr_path,
        collection_id="workflow_001",
        recording_frame_index=frame_index_path,
        manifest_path=tmp_path / "cache" / "workflow_001.flat_roi_cache.json",
        roi_size=(2, 2),
    )
    return zarr_path, manifest, expected


def test_create_clipped_collection_proxy_crop_run_writes_lineage_and_alias(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, manifest, expected = _build_tiny_clipped_cache(monkeypatch, tmp_path)

    result = create_clipped_collection_proxy_crop_run(
        zarr_path=zarr_path,
        manifest_path=manifest["manifest_path"],
        proxy_run_name="crop_proxy_clip_000000",
    )

    assert result["ok"] is True
    assert result["proxy_crop_run"] == "crop_proxy_clip_000000"
    assert result["row_count"] == 2
    assert result["source_clip_id"] == "clip_000000"

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    crop_parent = root["crop_runs"]
    assert crop_parent.attrs.get("latest") is None
    assert crop_parent.attrs.get("latest_complete") is None
    crop = crop_parent["crop_proxy_clip_000000"]
    assert crop.attrs["schema"] == PROXY_CROP_RUN_SCHEMA
    assert crop.attrs["stage_selector_eligible"] is False
    assert crop.attrs["proxy_crop_complete"] is True
    assert crop.attrs["palette_run_completion_status"] == "auxiliary"
    assert crop.attrs["status"] == "completed"
    assert crop.attrs["crop_storage_mode"] == "geometry_only"
    assert crop.attrs["source_roi_cache_required"] is True
    assert crop.attrs["source_roi_cache_manifest"] == manifest["manifest_path"]
    assert crop.attrs["source_roi_cache_alias_manifest"] == result["alias_manifest_path"]

    np.testing.assert_array_equal(crop["frame_indices"][:], np.array([11, 9], dtype=np.int64))
    np.testing.assert_array_equal(crop["source_frame_indices"][:], np.array([11, 9], dtype=np.int64))
    np.testing.assert_array_equal(crop["source_clip_indices"][:], np.array([0, 0], dtype=np.int64))
    np.testing.assert_array_equal(crop["source_clip_local_frame_indices"][:], np.array([2, 0], dtype=np.int64))
    np.testing.assert_array_equal(crop["source_refined_row_ids"][:], np.array([20, 10], dtype=np.int64))
    np.testing.assert_array_equal(crop["source_detect_row_index"][:], np.array([7, 3], dtype=np.int64))
    np.testing.assert_array_equal(crop["detection_indices"][:], np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(crop["source_crop_row_ids"][:], np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(crop["roi_coordinates_full"][:], np.array([[3, 2], [0, 0]], dtype=np.int32))

    cache = open_flat_roi_cache(
        result["alias_manifest_path"],
        expected_archive_path=zarr_path,
        expected_crop_run="crop_proxy_clip_000000",
        expected_shape=expected.shape,
    )
    try:
        np.testing.assert_array_equal(cache[:], expected)
    finally:
        cache.close()

    source = CropImageSource.open(
        root,
        crop_run="crop_proxy_clip_000000",
        zarr_path=zarr_path,
        roi_cache_manifest=result["alias_manifest_path"],
    )
    try:
        assert source.roi_read_mode == "flat_bin_roi_cache"
        assert source.roi_cache_used is True
        np.testing.assert_array_equal(source.read_slice(0, 2), expected)
    finally:
        source.close()


def test_create_clipped_collection_proxy_crop_run_refuses_existing_without_overwrite(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, manifest, _expected = _build_tiny_clipped_cache(monkeypatch, tmp_path)
    create_clipped_collection_proxy_crop_run(
        zarr_path=zarr_path,
        manifest_path=manifest["manifest_path"],
        proxy_run_name="crop_proxy_clip_000000",
    )

    try:
        create_clipped_collection_proxy_crop_run(
            zarr_path=zarr_path,
            manifest_path=manifest["manifest_path"],
            proxy_run_name="crop_proxy_clip_000000",
        )
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected existing proxy crop run to be refused")
