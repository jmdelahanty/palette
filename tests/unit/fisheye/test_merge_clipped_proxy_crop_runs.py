from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from fisheye.utils.create_clipped_collection_proxy_crop_run import CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE
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
    crop.create_array(
        "bbox_norm_coords",
        data=np.column_stack(
            (
                np.full(n_rows, 0.25 + clip_index * 0.25, dtype=np.float32),
                np.linspace(0.2, 0.4, n_rows, dtype=np.float32),
                np.full(n_rows, 0.1, dtype=np.float32),
                np.full(n_rows, 0.1, dtype=np.float32),
            )
        ),
        chunks=(max(1, n_rows), 4),
    )
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
            "detection_source_type": CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE,
            "source_detect_run": f"{CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE}:collection_test",
            "source_detect_run_semantics": "synthetic_collection_rowset_label_not_detect_runs_child",
            "source_refined_runs": [f"refined_clip_{clip_index:06d}"],
            "source_refined_run_paths": [
                f"clips/clip_{clip_index:06d}/refined_detect_runs/refined_clip_{clip_index:06d}"
            ],
            "source_roi_cache_manifest": f"/nrs/cache/clip_{clip_index:06d}.json",
            "source_roi_cache_alias_manifest": f"/nrs/cache/clip_{clip_index:06d}.alias.json",
            "bbox_norm_coords_semantics": "bbox_xywh_normalized_to_full_frame",
            "roi_shape": [512, 512],
            "roi_size": [512, 512],
            "source_video_width": 4512,
            "source_video_height": 4512,
            "source_video_dimensions_source": f"detect_clip_{clip_index:06d}:attrs",
            "width": 4512,
            "height": 4512,
        }
    )


def _write_legacy_bbox_row_index(path: Path, *, clip_index: int, n_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "bbox_norm_cx": np.full(n_rows, 0.25 + clip_index * 0.25, dtype=np.float32),
                "bbox_norm_cy": np.linspace(0.2, 0.4, n_rows, dtype=np.float32),
                "bbox_norm_w": np.full(n_rows, 0.1, dtype=np.float32),
                "bbox_norm_h": np.full(n_rows, 0.1, dtype=np.float32),
            }
        ),
        path,
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
    assert merged.attrs["detection_source_type"] == CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE
    assert merged.attrs["source_detect_run"] == f"{CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE}:collection_test"
    assert merged.attrs["source_refined_runs"] == ["refined_clip_000000", "refined_clip_000001"]
    assert merged.attrs["bbox_norm_coords_semantics"] == "bbox_xywh_normalized_to_full_frame"
    assert merged.attrs["legacy_bbox_norm_coords_repair_count"] == 0
    assert merged.attrs["source_video_width"] == 4512
    assert merged.attrs["source_video_height"] == 4512
    assert merged.attrs["width"] == 4512
    assert merged.attrs["height"] == 4512
    np.testing.assert_array_equal(merged["frame_indices"][:], np.array([10, 11, 20, 21], dtype=np.int64))
    np.testing.assert_allclose(
        merged["bbox_norm_coords"][:],
        np.array(
            [
                [0.25, 0.2, 0.1, 0.1],
                [0.25, 0.4, 0.1, 0.1],
                [0.50, 0.2, 0.1, 0.1],
                [0.50, 0.4, 0.1, 0.1],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(merged["source_crop_row_ids"][:], np.arange(4, dtype=np.int64))
    np.testing.assert_array_equal(merged["detection_indices"][:], np.arange(4, dtype=np.int64))
    np.testing.assert_array_equal(merged["source_proxy_crop_run_index"][:], np.array([0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(merged["source_proxy_crop_row_ids"][:], np.array([0, 1, 0, 1], dtype=np.int64))


def test_merge_clipped_proxy_crop_runs_repairs_legacy_bbox_from_row_index(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.create_group("crop_runs")
    _write_source_crop(root, "crop_proxy_b", frames=[20, 21], clip_index=1)
    _write_source_crop(root, "crop_proxy_a", frames=[10, 11], clip_index=0)

    for run_name, clip_index in (("crop_proxy_a", 0), ("crop_proxy_b", 1)):
        crop = root[f"crop_runs/{run_name}"]
        del crop["bbox_norm_coords"]
        crop.attrs.pop("source_detect_run")
        crop.attrs.pop("detection_source_type")
        row_index_path = tmp_path / "row_indices" / f"{run_name}.parquet"
        _write_legacy_bbox_row_index(row_index_path, clip_index=clip_index, n_rows=2)
        crop.attrs["source_roi_cache_row_index_path"] = str(row_index_path)

    result = merge_clipped_proxy_crop_runs(
        zarr_path=zarr_path,
        source_crop_runs=["crop_proxy_a", "crop_proxy_b"],
        output_run="crop_proxy_collection",
    )

    assert result["ok"] is True
    reopened = zarr.open_group(store=zarr_path, mode="r")
    merged = reopened["crop_runs/crop_proxy_collection"]
    assert merged.attrs["detection_source_type"] == CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE
    assert merged.attrs["source_detect_run"] == f"{CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE}:collection_test"
    assert (
        merged.attrs["bbox_norm_coords_source"]
        == "source_proxy_crop_runs.bbox_norm_coords_or_repaired_from_source_roi_cache_row_index_path"
    )
    assert merged.attrs["legacy_bbox_norm_coords_repair_count"] == 2
    np.testing.assert_allclose(
        merged["bbox_norm_coords"][:],
        np.array(
            [
                [0.25, 0.2, 0.1, 0.1],
                [0.25, 0.4, 0.1, 0.1],
                [0.50, 0.2, 0.1, 0.1],
                [0.50, 0.4, 0.1, 0.1],
            ],
            dtype=np.float32,
        ),
    )
