from __future__ import annotations

import json

from fisheye.shared.zarr.storage_intent import StoragePlan
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskDimensions,
    SubjectMaskProbabilityEncoding,
)
from fisheye.shared.zarr.subject_mask_storage import (
    SubjectMaskStoragePlanSet,
    plan_raw_subject_mask_storage,
    plan_refined_subject_mask_publication_storage,
)


def _sleepyfish_dimensions() -> SubjectMaskDimensions:
    return SubjectMaskDimensions(
        n_frames=1_188_000,
        n_rois=1_169_010,
        n_channels=4,
        roi_height=512,
        roi_width=512,
    )


def _plans_by_path(plan_set: SubjectMaskStoragePlanSet) -> dict[str, StoragePlan]:
    return {entry.rule.path: entry.plan for entry in plan_set.entries}


def test_raw_uint8_plan_derives_rows_from_bytes_and_access_shape() -> None:
    plan_set = plan_raw_subject_mask_storage(
        _sleepyfish_dimensions(),
        encoding=SubjectMaskProbabilityEncoding.LINEAR_UINT8_0_255,
    )
    plans = _plans_by_path(plan_set)

    assert plans["mask_probs_roi"].chunk_shape == (4, 1, 512, 512)
    assert plans["mask_probs_roi"].chunk_nbytes == 1024 * 1024
    assert plans["mask_probs_roi"].shard_shape == (1_144, 1, 512, 512)
    assert plans["mask_probs_roi"].estimated_payload_objects == 4_088
    assert plans["source_crop_xywh"].chunk_shape == (65_536, 4)
    assert plans["frame_row_offsets"].chunk_shape == (131_072,)
    assert plans["metrics/bbox_xyxy"].chunk_shape == (16_384, 4, 4)
    assert plan_set.arrays_over_object_budget == ()


def test_full_duration_float16_reports_current_object_budget_miss() -> None:
    plan_set = plan_raw_subject_mask_storage(
        _sleepyfish_dimensions(),
        encoding=SubjectMaskProbabilityEncoding.UNIT_FLOAT16,
    )
    probabilities = _plans_by_path(plan_set)["mask_probs_roi"]

    assert probabilities.chunk_shape == (2, 1, 512, 512)
    assert probabilities.chunk_nbytes == 1024 * 1024
    assert probabilities.shard_shape == (1_024, 1, 512, 512)
    assert probabilities.shard_nbytes == 512 * 1024 * 1024
    assert probabilities.estimated_payload_objects == 4_568
    assert probabilities.object_budget_satisfied is False
    assert plan_set.arrays_over_object_budget == ("mask_probs_roi",)


def test_optional_raw_threshold_cache_gets_its_own_per_component_plan() -> None:
    plan_set = plan_raw_subject_mask_storage(
        _sleepyfish_dimensions(),
        encoding=SubjectMaskProbabilityEncoding.LINEAR_UINT8_0_255,
        include_threshold_cache=True,
    )
    plans = _plans_by_path(plan_set)

    assert "masks_roi" in plans
    assert plans["masks_roi"].access_pattern == "per_row"
    assert plans["masks_roi"].access_unit_shape == (1, 1, 512, 512)
    assert plans["masks_roi"].chunk_shape == (4, 1, 512, 512)


def test_refined_publication_core_uses_dense_per_component_access() -> None:
    plan_set = plan_refined_subject_mask_publication_storage(_sleepyfish_dimensions())
    plans = _plans_by_path(plan_set)

    assert "mask_probs_roi" not in plans
    assert plans["masks_roi"].chunk_shape == (4, 1, 512, 512)
    assert plans["masks_roi"].write_mode == "immutable"
    assert plans["masks_roi"].write_ownership == "whole_shard_single_writer"
    assert plans["available_channels"].chunk_shape == (4,)
    assert plan_set.arrays_over_object_budget == ()


def test_storage_manifest_is_json_safe_and_reports_budget_scope() -> None:
    plan_set = plan_raw_subject_mask_storage(
        _sleepyfish_dimensions(),
        encoding=SubjectMaskProbabilityEncoding.UNIT_FLOAT16,
    )
    manifest = plan_set.as_manifest()

    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["stage_kind"] == "raw_probability_float16"
    assert manifest["object_estimate"]["arrays_over_object_budget"] == [
        "mask_probs_roi"
    ]
    assert manifest["object_estimate"]["all_array_object_budgets_satisfied"] is False
    assert manifest["metadata_open_contract"]["published"] == (
        "validated_consolidated_root"
    )


def test_shorter_recording_uses_same_byte_policy_without_row_constants() -> None:
    dimensions = SubjectMaskDimensions(
        n_frames=200_000,
        n_rois=198_000,
        n_channels=4,
        roi_height=512,
        roi_width=512,
    )
    plan_set = plan_raw_subject_mask_storage(
        dimensions,
        encoding=SubjectMaskProbabilityEncoding.LINEAR_UINT8_0_255,
    )
    plans = _plans_by_path(plan_set)

    assert plans["mask_probs_roi"].chunk_shape == (4, 1, 512, 512)
    assert plans["source_crop_xywh"].chunk_shape == (65_536, 4)
    assert plans["frame_row_offsets"].chunk_shape == (200_001,)
    assert plans["frame_row_offsets"].shard_shape is None
    assert plans["frame_row_offsets"].chunk_nbytes == 200_001 * 8
