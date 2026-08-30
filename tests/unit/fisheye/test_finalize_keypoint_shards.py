from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.keypoint_publication_profile import (
    COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE,
    STRICT_V2_KEYPOINT_PUBLICATION_PROFILE,
)
from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR
from fisheye.utils.finalize_keypoint_shards import finalize_keypoint_shards
from fisheye.utils.merge_clipped_proxy_crop_runs import merge_clipped_proxy_crop_runs


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
    crop.create_array("instance_key", data=np.arange(1, 6, dtype=np.uint64), chunks=(5,))
    return root


def _write_proxy_crop(
    root: zarr.Group,
    name: str,
    *,
    frames: list[int],
    clip_index: int,
    refined_offset: int,
) -> None:
    crop = root.require_group("crop_runs").create_group(name)
    frames_np = np.asarray(frames, dtype=np.int64)
    n_rows = int(frames_np.shape[0])
    local = np.arange(n_rows, dtype=np.int64)
    crop.create_array("frame_indices", data=frames_np, chunks=(max(1, n_rows),))
    crop.create_array("source_frame_indices", data=frames_np, chunks=(max(1, n_rows),))
    crop.create_array("source_clip_indices", data=np.full(n_rows, clip_index, dtype=np.int64), chunks=(max(1, n_rows),))
    crop.create_array("source_clip_local_frame_indices", data=local, chunks=(max(1, n_rows),))
    crop.create_array("source_refined_row_ids", data=local + refined_offset, chunks=(max(1, n_rows),))
    crop.create_array("source_detect_row_index", data=local + refined_offset + 1000, chunks=(max(1, n_rows),))
    crop.create_array(
        "instance_key",
        data=(local + 1 + clip_index * 10_000).astype(np.uint64),
        chunks=(max(1, n_rows),),
    )
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
            "source_detect_run": "finalized_clipped_refined_detect_collection_proxy:collection_test",
            "detection_source_type": "finalized_clipped_refined_detect_collection_proxy",
            "crop_policy": "centered_refined_bbox",
            "bbox_norm_coords_semantics": "bbox_xywh_normalized_to_full_frame",
            "roi_shape": [512, 512],
            "roi_size": [512, 512],
            "source_video_width": 4512,
            "source_video_height": 4512,
            "width": 4512,
            "height": 4512,
        }
    )


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
    n_rows = int(crop_rows_np.shape[0])
    crop = root["crop_runs"][source_crop_run]
    frames = np.asarray(crop["frame_indices"][:], dtype=np.int64)[crop_rows_np]
    source_frame_indices = (
        np.asarray(crop["source_frame_indices"][:], dtype=np.int64)[crop_rows_np]
        if "source_frame_indices" in crop
        else frames
    )
    source_clip_indices = (
        np.asarray(crop["source_clip_indices"][:], dtype=np.int64)[crop_rows_np]
        if "source_clip_indices" in crop
        else np.zeros(n_rows, dtype=np.int64)
    )
    source_clip_local_frame_indices = (
        np.asarray(crop["source_clip_local_frame_indices"][:], dtype=np.int64)[crop_rows_np]
        if "source_clip_local_frame_indices" in crop
        else frames
    )
    source_refined_row_ids = (
        np.asarray(crop["source_refined_row_ids"][:], dtype=np.int64)[crop_rows_np]
        if "source_refined_row_ids" in crop
        else crop_rows_np + 100
    )
    source_detect_row_index = (
        np.asarray(crop["source_detect_row_index"][:], dtype=np.int64)[crop_rows_np]
        if "source_detect_row_index" in crop
        else crop_rows_np + 200
    )
    instance_key = np.asarray(crop["instance_key"][:], dtype=np.uint64)[crop_rows_np]
    n_kpts = 3
    success_np = np.ones(n_rows, dtype=bool) if success is None else np.asarray(success, dtype=bool)
    base = np.arange(n_rows * n_kpts * 2, dtype=np.float64).reshape(n_rows, n_kpts, 2)
    frame_axis_len = max(5, int(frames.max(initial=0)) + 1)
    frame_counts = _counts(frames, frame_axis_len=frame_axis_len)
    n_keypoints = np.zeros(frame_axis_len, dtype=np.int32)
    if n_rows:
        n_keypoints[frames[success_np]] = n_kpts

    shard.create_array("frame_indices", data=frames, chunks=(max(1, n_rows),))
    shard.create_array("source_crop_row_ids", data=crop_rows_np, chunks=(max(1, n_rows),))
    shard.create_array("detection_indices", data=crop_rows_np, chunks=(max(1, n_rows),))
    shard.create_array("source_frame_indices", data=source_frame_indices, chunks=(max(1, n_rows),))
    shard.create_array("source_clip_indices", data=source_clip_indices, chunks=(max(1, n_rows),))
    shard.create_array("source_clip_local_frame_indices", data=source_clip_local_frame_indices, chunks=(max(1, n_rows),))
    shard.create_array("source_refined_row_ids", data=source_refined_row_ids, chunks=(max(1, n_rows),))
    shard.create_array("source_detect_row_index", data=source_detect_row_index, chunks=(max(1, n_rows),))
    shard.create_array("instance_key", data=instance_key, chunks=(max(1, n_rows),))
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
        publication_profile=COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE,
        output_run="keypoints_collection_test",
    )

    assert result["ok"] is True
    assert result["total_rois"] == 3
    assert result["sort_changed_order"] is True
    assert result["identity_validation"] == {
        "mode": "instance_key",
        "status": "exact",
        "order_check": "source_crop_row_ids_indexed_exact",
        "source_crop_run": "crop_proxy",
        "row_count": 3,
        "unique_count": 3,
    }

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
    assert run.attrs["keypoint_storage_layout"] == "indexed_sharding_v1"
    assert run.attrs["row_identity_mode"] == "instance_key"
    assert run.attrs["instance_key_alignment_status"] == "exact"
    assert run.attrs["row_identity_validation"] == result["identity_validation"]
    assert run.attrs["keypoint_roi_shard_rows"] == 131_072
    assert run["keypoints_roi"].shards == (3, 3, 2)
    assert run["instance_key"].shards == (3,)
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
            publication_profile=COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE,
            output_run="keypoints_collection_test",
        )

    reopened = zarr.open_group(store=zarr_path, mode="r")
    assert "keypoints_runs" not in reopened


def test_finalize_keypoint_shards_rejects_instance_key_mismatch_before_publication(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = _make_archive(zarr_path)
    _write_shard(root, "shard_a", crop_rows=[0, 1])
    root["keypoint_shard_runs/shard_a/instance_key"][1] = np.uint64(999)

    with pytest.raises(ValueError, match="does not exactly match.*source_crop_row_ids"):
        finalize_keypoint_shards(
            zarr_path=zarr_path,
            shard_runs=["shard_a"],
            publication_profile=COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE,
            output_run="keypoints_collection_test",
        )

    reopened = zarr.open_group(store=zarr_path, mode="r")
    assert "keypoints_runs" not in reopened


def test_finalize_keypoint_shards_rejects_duplicate_instance_keys_before_publication(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = _make_archive(zarr_path)
    _write_shard(root, "shard_a", crop_rows=[0, 1])
    root["keypoint_shard_runs/shard_a/instance_key"][1] = np.uint64(1)

    with pytest.raises(ValueError, match="instance_key contains 1 duplicate"):
        finalize_keypoint_shards(
            zarr_path=zarr_path,
            shard_runs=["shard_a"],
            publication_profile=COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE,
            output_run="keypoints_collection_test",
        )

    reopened = zarr.open_group(store=zarr_path, mode="r")
    assert "keypoints_runs" not in reopened


def test_finalize_keypoint_shards_rejects_keyless_crop_as_legacy_positional(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = _make_archive(zarr_path)
    _write_shard(root, "shard_a", crop_rows=[0, 1])
    del root["crop_runs/crop_proxy/instance_key"]

    with pytest.raises(ValueError, match="legacy positional finalization is not permitted"):
        finalize_keypoint_shards(
            zarr_path=zarr_path,
            shard_runs=["shard_a"],
            publication_profile=COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE,
            output_run="keypoints_collection_test",
        )

    reopened = zarr.open_group(store=zarr_path, mode="r")
    assert "keypoints_runs" not in reopened


def test_finalize_keypoint_shards_rejects_keyless_shard_as_legacy_positional(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = _make_archive(zarr_path)
    _write_shard(root, "shard_a", crop_rows=[0, 1])
    del root["keypoint_shard_runs/shard_a/instance_key"]

    with pytest.raises(ValueError, match="missing required arrays.*instance_key"):
        finalize_keypoint_shards(
            zarr_path=zarr_path,
            shard_runs=["shard_a"],
            publication_profile=COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE,
            output_run="keypoints_collection_test",
        )

    reopened = zarr.open_group(store=zarr_path, mode="r")
    assert "keypoints_runs" not in reopened


def test_finalize_keypoint_shards_rejects_mixed_source_crop_runs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = _make_archive(zarr_path)
    crop_other = root["crop_runs"].create_group("crop_other")
    crop_other.create_array("frame_indices", data=np.array([0, 1, 2, 3, 4], dtype=np.int64), chunks=(5,))
    crop_other.create_array("instance_key", data=np.arange(101, 106, dtype=np.uint64), chunks=(5,))
    _write_shard(root, "shard_a", crop_rows=[0], source_crop_run="crop_proxy")
    _write_shard(root, "shard_b", crop_rows=[1], source_crop_run="crop_other")

    with pytest.raises(ValueError, match="Mixed source_crop_run"):
        finalize_keypoint_shards(
            zarr_path=zarr_path,
            shard_runs=["shard_a", "shard_b"],
            publication_profile=COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE,
            output_run="keypoints_collection_test",
        )


def test_finalize_keypoint_shards_rebases_mixed_proxy_crop_runs_to_target(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.create_group("crop_runs")
    _write_proxy_crop(root, "crop_proxy_a", frames=[10, 11], clip_index=0, refined_offset=100)
    _write_proxy_crop(root, "crop_proxy_b", frames=[20, 21], clip_index=1, refined_offset=200)

    merge_result = merge_clipped_proxy_crop_runs(
        zarr_path=zarr_path,
        source_crop_runs=["crop_proxy_b", "crop_proxy_a"],
        output_run="crop_proxy_collection",
    )
    assert merge_result["row_count"] == 4

    reopened = zarr.open_group(store=zarr_path, mode="a")
    _write_shard(reopened, "shard_a", crop_rows=[0, 1], source_crop_run="crop_proxy_a")
    _write_shard(reopened, "shard_b", crop_rows=[0, 1], source_crop_run="crop_proxy_b")

    result = finalize_keypoint_shards(
        zarr_path=zarr_path,
        shard_runs=["shard_b", "shard_a"],
        publication_profile=COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE,
        output_run="keypoints_collection_test",
        target_crop_run="crop_proxy_collection",
    )

    assert result["ok"] is True
    assert result["source_crop_run"] == "crop_proxy_collection"
    assert result["source_keypoint_shard_crop_runs"] == ["crop_proxy_a", "crop_proxy_b"]
    assert result["sort_changed_order"] is True
    assert result["identity_validation"]["status"] == "exact"
    assert result["identity_validation"]["source_crop_run"] == "crop_proxy_collection"

    root_after = zarr.open_group(store=zarr_path, mode="r")
    run = root_after["keypoints_runs/keypoints_collection_test"]
    assert run.attrs["source_crop_run"] == "crop_proxy_collection"
    assert run.attrs["source_crop_rebased_from_shards"] is True
    np.testing.assert_array_equal(run["source_crop_row_ids"][:], np.array([0, 1, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(run["frame_indices"][:], np.array([10, 11, 20, 21], dtype=np.int64))
    np.testing.assert_array_equal(run["detection_indices"][:], np.array([0, 1, 2, 3], dtype=np.int64))


def test_finalize_keypoint_shards_rebase_rejects_target_instance_key_mismatch(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.create_group("crop_runs")
    _write_proxy_crop(root, "crop_proxy_a", frames=[10, 11], clip_index=0, refined_offset=100)
    _write_proxy_crop(root, "crop_proxy_b", frames=[20, 21], clip_index=1, refined_offset=200)
    merge_clipped_proxy_crop_runs(
        zarr_path=zarr_path,
        source_crop_runs=["crop_proxy_b", "crop_proxy_a"],
        output_run="crop_proxy_collection",
    )

    reopened = zarr.open_group(store=zarr_path, mode="a")
    _write_shard(reopened, "shard_a", crop_rows=[0, 1], source_crop_run="crop_proxy_a")
    _write_shard(reopened, "shard_b", crop_rows=[0, 1], source_crop_run="crop_proxy_b")
    reopened["crop_runs/crop_proxy_collection/instance_key"][0] = np.uint64(999_999)

    with pytest.raises(ValueError, match="Could not map.*into crop_runs/crop_proxy_collection"):
        finalize_keypoint_shards(
            zarr_path=zarr_path,
            shard_runs=["shard_b", "shard_a"],
            publication_profile=COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE,
            output_run="keypoints_collection_test",
            target_crop_run="crop_proxy_collection",
        )

    root_after = zarr.open_group(store=zarr_path, mode="r")
    assert "keypoints_runs" not in root_after


def test_finalize_keypoint_shards_rejects_strict_v2_before_zarr_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="compatibility-only producer"):
        finalize_keypoint_shards(
            zarr_path=tmp_path / "does_not_exist.zarr",
            shard_runs=["shard_a"],
            publication_profile=STRICT_V2_KEYPOINT_PUBLICATION_PROFILE,
            output_run="must_not_publish",
        )
