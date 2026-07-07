from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.utils.merge_clipped_proxy_crop_runs import merge_clipped_proxy_crop_runs


def _write_source_crop(root: zarr.Group, name: str, *, frames: list[int], clip_index: int) -> None:
    crop = root.require_group("crop_runs").create_group(name)
    frames_np = np.asarray(frames, dtype=np.int64)
    n_rows = int(frames_np.shape[0])
    local = np.arange(n_rows, dtype=np.int64)
    crop.create_array("frame_indices", data=frames_np, chunks=(max(1, n_rows),))
    crop.create_array("source_frame_indices", data=frames_np, chunks=(max(1, n_rows),))
    crop.create_array("source_clip_indices", data=np.full(n_rows, clip_index, dtype=np.int64), chunks=(max(1, n_rows),))
    crop.create_array("source_clip_local_frame_indices", data=local, chunks=(max(1, n_rows),))
    crop.create_array("source_refined_row_ids", data=local + clip_index * 100, chunks=(max(1, n_rows),))
    crop.create_array("source_detect_row_index", data=local + clip_index * 1000, chunks=(max(1, n_rows),))
    crop.create_array("detection_indices", data=local, chunks=(max(1, n_rows),))
    crop.create_array("source_crop_row_ids", data=local, chunks=(max(1, n_rows),))
    crop.create_array(
        "roi_coordinates_full",
        data=np.stack((local + clip_index * 10, local + clip_index * 20), axis=1).astype(np.int32),
        chunks=(max(1, n_rows), 2),
    )
    crop.attrs.update(
        {
            "source_clip_id": f"clip_{clip_index:06d}",
            "source_clip_index": clip_index,
            "source_collection_id": "collection_test",
            "source_roi_cache_manifest": f"/nrs/cache/clip_{clip_index:06d}.json",
            "source_roi_cache_alias_manifest": f"/nrs/cache/clip_{clip_index:06d}.alias.json",
            "roi_shape": [512, 512],
            "roi_size": [512, 512],
        }
    )


def test_merge_clipped_proxy_crop_runs_writes_collection_proxy(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.create_group("crop_runs")
    _write_source_crop(root, "crop_proxy_b", frames=[20, 21], clip_index=1)
    _write_source_crop(root, "crop_proxy_a", frames=[10, 11], clip_index=0)

    result = merge_clipped_proxy_crop_runs(
        zarr_path=zarr_path,
        source_crop_runs=["crop_proxy_b", "crop_proxy_a"],
        output_run="crop_proxy_collection",
    )

    assert result["ok"] is True
    assert result["row_count"] == 4
    assert result["source_proxy_crop_runs"] == ["crop_proxy_a", "crop_proxy_b"]

    reopened = zarr.open_group(store=zarr_path, mode="r")
    merged = reopened["crop_runs/crop_proxy_collection"]
    assert merged.attrs["schema"] == "palette_clipped_collection_merged_proxy_crop_run_v1"
    assert merged.attrs["stage_selector_eligible"] is False
    assert merged.attrs["source_proxy_crop_runs"] == ["crop_proxy_a", "crop_proxy_b"]
    np.testing.assert_array_equal(merged["frame_indices"][:], np.array([10, 11, 20, 21], dtype=np.int64))
    np.testing.assert_array_equal(merged["source_crop_row_ids"][:], np.arange(4, dtype=np.int64))
    np.testing.assert_array_equal(merged["detection_indices"][:], np.arange(4, dtype=np.int64))
    np.testing.assert_array_equal(merged["source_proxy_crop_run_index"][:], np.array([0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(merged["source_proxy_crop_row_ids"][:], np.array([0, 1, 0, 1], dtype=np.int64))
