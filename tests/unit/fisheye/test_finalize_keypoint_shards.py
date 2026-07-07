from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR
from fisheye.utils.finalize_keypoint_shards import finalize_keypoint_shards


def _counts(frames: np.ndarray, *, frame_axis_len: int) -> np.ndarray:
    out = np.zeros(frame_axis_len, dtype=np.int32)
    if frames.size:
        values, counts = np.unique(frames, return_counts=True)
        out[values.astype(np.int64)] = counts.astype(np.int32)
    return out


def _make_archive(path: Path) -> zarr.Group:
    root = zarr.open_group(store=path, mode="w")
    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_proxy")
    crop.create_array("frame_indices", data=np.array([0, 1, 2, 3, 4], dtype=np.int64), chunks=(5,))
    crop.create_array("roi_coordinates_full", data=np.zeros((5, 2), dtype=np.int32), chunks=(5, 2))
    return root


def _write_shard(
    root: zarr.Group,
    name: str,
    *,
    crop_rows: list[int],
    source_crop_run: str = "crop_proxy",
    success: list[bool] | None = None,
) -> None:
    parent = root.require_group("keypoint_shard_runs")
    shard = parent.create_group(name)
    crop_rows_np = np.asarray(crop_rows, dtype=np.int64)
    frames = crop_rows_np.copy()
    n_rows = int(crop_rows_np.shape[0])
    n_kpts = 3
    success_np = np.ones(n_rows, dtype=bool) if success is None else np.asarray(success, dtype=bool)
    base = np.arange(n_rows * n_kpts * 2, dtype=np.float64).reshape(n_rows, n_kpts, 2)
    frame_counts = _counts(frames, frame_axis_len=5)
    n_keypoints = np.zeros(5, dtype=np.int32)
    if n_rows:
        n_keypoints[frames[success_np]] = n_kpts

    shard.create_array("frame_indices", data=frames, chunks=(max(1, n_rows),))
    shard.create_array("source_crop_row_ids", data=crop_rows_np, chunks=(max(1, n_rows),))
    shard.create_array("detection_indices", data=crop_rows_np, chunks=(max(1, n_rows),))
    shard.create_array("source_frame_indices", data=frames, chunks=(max(1, n_rows),))
    shard.create_array("source_clip_indices", data=np.zeros(n_rows, dtype=np.int64), chunks=(max(1, n_rows),))
    shard.create_array("source_clip_local_frame_indices", data=frames, chunks=(max(1, n_rows),))
    shard.create_array("source_refined_row_ids", data=crop_rows_np + 100, chunks=(max(1, n_rows),))
    shard.create_array("source_detect_row_index", data=crop_rows_np + 200, chunks=(max(1, n_rows),))
    shard.create_array("keypoints_roi", data=base + crop_rows_np[:, None, None], chunks=(max(1, n_rows), n_kpts, 2))
    shard.create_array("keypoints_img", data=base + 10 + crop_rows_np[:, None, None], chunks=(max(1, n_rows), n_kpts, 2))
    shard.create_array("keypoints_norm", data=(base + 10 + crop_rows_np[:, None, None]) / 100.0, chunks=(max(1, n_rows), n_kpts, 2))
    shard.create_array("keypoint_confidences", data=np.ones((n_rows, n_kpts), dtype=np.float64), chunks=(max(1, n_rows), n_kpts))
    shard.create_array("confidence", data=np.linspace(0.5, 0.9, n_rows, dtype=np.float64), chunks=(max(1, n_rows),))
    shard.create_array("detection_success", data=success_np, chunks=(max(1, n_rows),))
    shard.create_array("heading", data=np.arange(n_rows, dtype=np.float64), chunks=(max(1, n_rows),))
    shard.create_array("heading_finite", data=np.ones(n_rows, dtype=bool), chunks=(max(1, n_rows),))
    shard.create_array("heading_usable", data=success_np, chunks=(max(1, n_rows),))
    shard.create_array("pose_bbox_xyxy_roi", data=np.zeros((n_rows, 4), dtype=np.float32), chunks=(max(1, n_rows), 4))
    shard.create_array("effective_threshold", data=np.full(n_rows, np.nan), chunks=(max(1, n_rows),))
    shard.create_array("effective_se2_radius", data=np.full(n_rows, np.nan), chunks=(max(1, n_rows),))
    shard.create_array("detection_source", data=np.zeros(n_rows, dtype=np.int8), chunks=(max(1, n_rows),))
    shard.create_array("frame_counts", data=frame_counts, chunks=(5,))
    shard.create_array("n_rois", data=frame_counts, chunks=(5,))
    shard.create_array("n_keypoints", data=n_keypoints, chunks=(5,))
    shard.attrs.update(
        {
            RUN_COMPLETION_STATUS_ATTR: "complete",
            "stage_selector_eligible": False,
            "is_collection_shard": True,
            "source_crop_run": source_crop_run,
            "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
            "keypoint_confidence_labels": ["swim_bladder", "eye_left", "eye_right"],
            "skeleton_id": "traditional_v2",
            "kpt_shape": [3, 2],
            "pose_schema": {"name": "traditional_v2", "skeleton_id": "traditional_v2", "kpt_shape": [3, 2]},
            "model_kpt_shape": [3, 2],
            "method": "yolo_pose",
            "model_name": "pose.pt",
        }
    )


def test_finalize_keypoint_shards_writes_canonical_keypoint_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = _make_archive(zarr_path)
    _write_shard(root, "shard_b", crop_rows=[2])
    _write_shard(root, "shard_a", crop_rows=[0, 1])

    result = finalize_keypoint_shards(
        zarr_path=zarr_path,
        shard_runs=["shard_b", "shard_a"],
        output_run="keypoints_collection_test",
    )

    assert result["ok"] is True
    assert result["total_rois"] == 3
    assert result["sort_changed_order"] is True

    reopened = zarr.open_group(store=zarr_path, mode="r")
    parent = reopened["keypoints_runs"]
    assert parent.attrs["latest_complete"] == "keypoints_collection_test"
    assert parent.attrs["latest"] == "keypoints_collection_test"
    assert reopened.attrs["current_keypoint_group_path"] == "keypoints_runs/keypoints_collection_test"

    run = parent["keypoints_collection_test"]
    assert run.attrs["collection_finalizer_schema"] == "palette_keypoint_shard_collection_finalizer_v1"
    assert run.attrs["source_keypoint_shard_runs"] == ["shard_b", "shard_a"]
    assert run.attrs["source_crop_run"] == "crop_proxy"
    assert run.attrs["stage_selector_eligible"] is True
    np.testing.assert_array_equal(run["source_crop_row_ids"][:], np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(run["frame_indices"][:], np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(run["frame_counts"][:], np.array([1, 1, 1, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(run["n_rois"][:], np.array([1, 1, 1, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(run["n_keypoints"][:], np.array([3, 3, 3, 0, 0], dtype=np.int32))


def test_finalize_keypoint_shards_rejects_duplicate_source_crop_rows(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = _make_archive(zarr_path)
    _write_shard(root, "shard_a", crop_rows=[0, 1])
    _write_shard(root, "shard_b", crop_rows=[1, 2])

    with pytest.raises(ValueError, match="Duplicate source_crop_row_ids"):
        finalize_keypoint_shards(
            zarr_path=zarr_path,
            shard_runs=["shard_a", "shard_b"],
            output_run="keypoints_collection_test",
        )

    reopened = zarr.open_group(store=zarr_path, mode="r")
    assert "keypoints_runs" not in reopened


def test_finalize_keypoint_shards_rejects_mixed_source_crop_runs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = _make_archive(zarr_path)
    crop_other = root["crop_runs"].create_group("crop_other")
    crop_other.create_array("frame_indices", data=np.array([0, 1, 2, 3, 4], dtype=np.int64), chunks=(5,))
    _write_shard(root, "shard_a", crop_rows=[0], source_crop_run="crop_proxy")
    _write_shard(root, "shard_b", crop_rows=[1], source_crop_run="crop_other")

    with pytest.raises(ValueError, match="Mixed source_crop_run"):
        finalize_keypoint_shards(
            zarr_path=zarr_path,
            shard_runs=["shard_a", "shard_b"],
            output_run="keypoints_collection_test",
        )
