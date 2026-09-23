from __future__ import annotations

from fisheye.shared.zarr.merged_keypoint_training_storage import (
    plan_merged_keypoint_training_arrays,
    storage_plan_manifest,
)


def test_pose_roi_storage_uses_one_sample_inner_chunks() -> None:
    plans = plan_merged_keypoint_training_arrays(
        run_name="merged_export_test",
        n_samples=10_717,
        roi_shape=(192, 192),
        keypoint_shape=(3, 2),
        n_sources=51,
        split_counts={"train": 8_538, "val": 2_179, "test": 0},
    )
    roi = plans["crop_runs/merged_export_test/roi_images"].plan

    assert roi.access_pattern == "per_row"
    assert roi.chunk_shape == (1, 192, 192)
    assert roi.shard_shape is not None
    assert roi.shard_shape[0] > 1
    assert roi.profile_id == "training_random_row_immutable_v1"
    assert storage_plan_manifest(plans)["profile_id"] == roi.profile_id

    small_plans = plan_merged_keypoint_training_arrays(
        run_name="merged_export_small",
        n_samples=1_000,
        roi_shape=(32, 32),
        keypoint_shape=(3, 2),
        n_sources=2,
        split_counts={"train": 800, "val": 200, "test": 0},
    )
    assert small_plans["crop_runs/merged_export_small/roi_images"].plan.chunk_shape == (
        1,
        32,
        32,
    )
