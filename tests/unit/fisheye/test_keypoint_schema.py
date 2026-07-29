from __future__ import annotations

import copy

import numpy as np
import pytest

from fisheye.shared.zarr.keypoint_schema import (
    FORBIDDEN_KEYPOINT_V2_ARRAYS,
    KEYPOINT_SCHEMA_V2,
    REFINED_KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    KeypointSchemaError,
    derive_frame_row_offsets,
    derive_keypoint_row_signatures,
)

SKELETON_DIGEST = "42" * 32


def _fixture() -> tuple[
    KeypointDimensions,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    dimensions = KeypointDimensions(
        n_frames=4,
        n_instances=4,
        n_keypoints=3,
        source_width=640,
        source_height=480,
    )
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    keys = np.asarray([101, 102, 201, 301], dtype=np.uint64)
    origins = np.asarray([[10, 20], [100, 50], [200, 100], [300, 200]], dtype=np.int32)
    sizes = np.asarray([[64, 48], [80, 60], [96, 72], [112, 88]], dtype=np.int32)
    crop_signatures = np.arange(4 * 32, dtype=np.uint8).reshape(4, 32)
    source_crop = {
        "instance_key": keys.copy(),
        "frame_indices": frames.copy(),
        "source_acquisition_frame_index": frames.copy(),
        "source_row_signature": crop_signatures.copy(),
        "roi_coordinates_full": origins,
        "roi_sizes_full": sizes,
    }

    points_roi = np.asarray(
        [
            [[5, 6], [10, 11], [15, 16]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
            [[20, 21], [30, 31], [np.nan, np.nan]],
            [[40, 41], [50, 51], [60, 61]],
        ],
        dtype=np.float32,
    )
    valid = np.all(np.isfinite(points_roi), axis=2)
    points_img = points_roi + origins.astype(np.float32)[:, None, :]
    keypoint_confidences = np.asarray(
        [
            [0.95, 0.90, 0.85],
            [np.nan, np.nan, np.nan],
            [0.75, 0.70, np.nan],
            [0.99, 0.98, 0.97],
        ],
        dtype=np.float32,
    )
    bbox_roi = np.asarray(
        [
            [1, 2, 30, 35],
            [np.nan, np.nan, np.nan, np.nan],
            [4, 5, 60, 65],
            [6, 7, 100, 80],
        ],
        dtype=np.float32,
    )
    bbox_img = bbox_roi + np.column_stack((origins, origins)).astype(np.float32)
    pose_confidence = np.asarray([0.92, np.nan, 0.81, 0.97], dtype=np.float32)
    pose_success = np.asarray([True, False, True, True], dtype=bool)
    row_signatures = derive_keypoint_row_signatures(
        instance_key=keys,
        source_crop_row_signature=crop_signatures,
        keypoints_roi=points_roi,
        keypoint_valid=valid,
        skeleton_digest=SKELETON_DIGEST,
    )
    arrays = {
        "instance_key": keys,
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "source_acquisition_frame_index": frames.copy(),
        "frame_indices": frames,
        "frame_row_offsets": derive_frame_row_offsets(frames, n_frames=4),
        "source_crop_row_signature": crop_signatures,
        "keypoint_row_signature": row_signatures,
        "keypoints_roi": points_roi,
        "keypoints_img": points_img,
        "keypoint_confidences": keypoint_confidences,
        "keypoint_valid": valid,
        "pose_confidence": pose_confidence,
        "pose_bbox_xyxy_roi": bbox_roi,
        "pose_bbox_xyxy_img": bbox_img,
        "pose_success": pose_success,
    }
    return dimensions, arrays, source_crop


def _refined_fixture() -> tuple[
    KeypointDimensions,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    dimensions, raw, crop = _fixture()
    refined = {
        name: value.copy() for name, value in raw.items() if name != "pose_success"
    }
    refined.update(
        {
            "source_success": raw["pose_success"].copy(),
            "refined_success": raw["pose_success"].copy(),
            "keypoint_edit_flags": np.zeros((4, 3), dtype=bool),
            "flip_corrected": np.zeros(4, dtype=bool),
            "confidence_valid": np.asarray([True, False, True, True], dtype=bool),
            "geometry_valid": np.asarray([True, False, True, True], dtype=bool),
            "usable_keypoints": np.asarray([True, False, True, True], dtype=bool),
            "review_state_codes": np.asarray([1, 0, 1, 1], dtype=np.uint8),
            "reason_codes": np.asarray([0, 1, 0, 0], dtype=np.uint16),
        }
    )
    return dimensions, refined, crop


def _codes() -> tuple[dict[int, str], dict[int, str]]:
    return {0: "unreviewed", 1: "accepted"}, {0: "none", 1: "source_failed"}


def test_raw_keypoint_v2_accepts_multirow_and_empty_frames() -> None:
    dimensions, arrays, source_crop = _fixture()

    KEYPOINT_SCHEMA_V2.require(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop,
        skeleton_digest=SKELETON_DIGEST,
    )

    assert arrays["frame_row_offsets"].tolist() == [0, 2, 2, 3, 4]
    manifest = KEYPOINT_SCHEMA_V2.as_manifest(dimensions=dimensions)
    assert len(manifest["bindings"]) == 15
    assert manifest["invariants"]["instances_per_frame"] == "zero_one_or_many"
    assert "heading" in manifest["forbidden_v1_arrays"]


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    (
        (
            lambda arrays: arrays["frame_row_offsets"].__setitem__(2, 1),
            "frame_row_offsets_mismatch",
        ),
        (
            lambda arrays: arrays["instance_key"].__setitem__(
                1, arrays["instance_key"][0]
            ),
            "duplicate_instance_key",
        ),
        (
            lambda arrays: arrays["keypoints_img"].__setitem__((0, 0, 0), 999.0),
            "keypoints_img_projection_mismatch",
        ),
        (
            lambda arrays: arrays["keypoint_row_signature"].__setitem__((0, 0), 0),
            "keypoint_row_signature_mismatch",
        ),
    ),
)
def test_raw_keypoint_v2_rejects_cross_array_tampering(
    mutate: object, expected_code: str
) -> None:
    dimensions, arrays, source_crop = _fixture()
    mutate(arrays)  # type: ignore[operator]

    issues = KEYPOINT_SCHEMA_V2.validate(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop,
        skeleton_digest=SKELETON_DIGEST,
    )

    assert expected_code in {issue.code for issue in issues}


def test_raw_keypoint_v2_rejects_v1_heading_and_wrong_dtype() -> None:
    dimensions, arrays, source_crop = _fixture()
    arrays["heading"] = np.zeros(4, dtype=np.float64)
    arrays["keypoints_roi"] = arrays["keypoints_roi"].astype(np.float64)

    issues = KEYPOINT_SCHEMA_V2.validate(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop,
        skeleton_digest=SKELETON_DIGEST,
    )

    assert any(
        issue.code == "forbidden_v1_array" and issue.path == "heading"
        for issue in issues
    )
    assert any(
        issue.code == "array_contract_violation" and issue.path == "keypoints_roi"
        for issue in issues
    )


def test_raw_keypoint_v2_requires_bound_crop_evidence() -> None:
    dimensions, arrays, _ = _fixture()

    with pytest.raises(KeypointSchemaError, match="missing_source_crop_evidence"):
        KEYPOINT_SCHEMA_V2.require(
            arrays,
            dimensions=dimensions,
            source_crop_arrays=None,
            skeleton_digest=SKELETON_DIGEST,
        )


def test_refined_keypoint_v2_keeps_review_qc_in_snapshot() -> None:
    dimensions, arrays, source_crop = _refined_fixture()
    review_states, reasons = _codes()

    REFINED_KEYPOINT_SCHEMA_V2.require(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop,
        skeleton_digest=SKELETON_DIGEST,
        review_state_map=review_states,
        reason_code_map=reasons,
    )

    manifest = REFINED_KEYPOINT_SCHEMA_V2.as_manifest(dimensions=dimensions)
    assert len(manifest["bindings"]) == 23
    assert "refined_success" in REFINED_KEYPOINT_SCHEMA_V2.binding_paths
    assert "pose_success" not in REFINED_KEYPOINT_SCHEMA_V2.binding_paths


def test_refined_keypoint_v2_rejects_unknown_codes_and_invalid_usable_row() -> None:
    dimensions, arrays, source_crop = _refined_fixture()
    arrays = copy.deepcopy(arrays)
    arrays["review_state_codes"][2] = np.uint8(9)
    arrays["confidence_valid"][2] = False
    review_states, reasons = _codes()

    issues = REFINED_KEYPOINT_SCHEMA_V2.validate(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop,
        skeleton_digest=SKELETON_DIGEST,
        review_state_map=review_states,
        reason_code_map=reasons,
    )

    codes = {issue.code for issue in issues}
    assert "unknown_persisted_code" in codes
    assert "usable_keypoints_policy_mismatch" in codes


def test_refined_keypoint_v2_requires_explicit_code_registries() -> None:
    dimensions, arrays, source_crop = _refined_fixture()

    issues = REFINED_KEYPOINT_SCHEMA_V2.validate(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop,
        skeleton_digest=SKELETON_DIGEST,
    )

    assert [issue.code for issue in issues].count("missing_code_registry") == 2


def test_forbidden_v1_surface_is_a_stable_exact_set() -> None:
    assert "heading" in FORBIDDEN_KEYPOINT_V2_ARRAYS
    assert "keypoints_norm" in FORBIDDEN_KEYPOINT_V2_ARRAYS
    assert "heading_temporal_outlier" in FORBIDDEN_KEYPOINT_V2_ARRAYS
