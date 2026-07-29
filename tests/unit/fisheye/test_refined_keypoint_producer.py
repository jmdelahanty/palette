from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.keypoint_quality_producer import (
    ObservationLocalKeypointQualityPolicy,
    prepare_observation_local_keypoint_quality,
)
from fisheye.shared.zarr.keypoint_quality_schema import KeypointQualitySourceReference
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    REFINED_KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    derive_frame_row_offsets,
    derive_keypoint_row_signatures,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_keypoint_producer import (
    LandmarkCoordinateEdit,
    LandmarkValidityEdit,
    RefinedKeypointDecision,
    prepare_refined_keypoint_snapshot,
)


SKELETON_DIGEST = "42" * 32
REVIEW_STATE_MAP = {0: "unreviewed", 1: "accepted", 2: "rejected"}
REASON_CODE_MAP = {0: "none", 1: "manual_reject", 2: "manual_correction"}


def _fixture():  # type: ignore[no-untyped-def]
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
    sizes = np.full((4, 2), 100, dtype=np.int32)
    crop_signatures = np.arange(4 * 32, dtype=np.uint8).reshape(4, 32)
    crop = {
        "instance_key": keys.copy(),
        "frame_indices": frames.copy(),
        "source_acquisition_frame_index": frames.copy(),
        "source_row_signature": crop_signatures.copy(),
        "roi_coordinates_full": origins,
        "roi_sizes_full": sizes,
    }
    points_roi = np.asarray(
        [
            [[5, 15], [15, 10], [15, 20]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
            [[20, 20], [25, 15], [15, 25]],
            [[40, 50], [45, 35], [35, 45]],
        ],
        dtype=np.float32,
    )
    valid = np.all(np.isfinite(points_roi), axis=2)
    points_img = points_roi + origins.astype(np.float32)[:, None, :]
    bbox_roi = np.asarray(
        [[1, 2, 50, 60], [np.nan] * 4, [2, 3, 60, 70], [5, 6, 80, 90]],
        dtype=np.float32,
    )
    bbox_img = bbox_roi + np.column_stack((origins, origins)).astype(np.float32)
    row_signatures = derive_keypoint_row_signatures(
        instance_key=keys,
        source_crop_row_signature=crop_signatures,
        keypoints_roi=points_roi,
        keypoint_valid=valid,
        skeleton_digest=SKELETON_DIGEST,
    )
    raw = {
        "instance_key": keys,
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "source_acquisition_frame_index": frames.copy(),
        "frame_indices": frames,
        "frame_row_offsets": derive_frame_row_offsets(frames, n_frames=4),
        "source_crop_row_signature": crop_signatures,
        "keypoint_row_signature": row_signatures,
        "keypoints_roi": points_roi,
        "keypoints_img": points_img,
        "keypoint_confidences": np.asarray(
            [
                [0.9, 0.9, 0.7],
                [np.nan, np.nan, np.nan],
                [0.7, 0.9, 0.9],
                [0.95, 0.4, 0.97],
            ],
            dtype=np.float32,
        ),
        "keypoint_valid": valid,
        "pose_confidence": np.asarray([0.9, np.nan, 0.8, 0.95], dtype=np.float32),
        "pose_bbox_xyxy_roi": bbox_roi,
        "pose_bbox_xyxy_img": bbox_img,
        "pose_success": np.asarray([True, False, True, True], dtype=bool),
    }
    source_manifest: dict[str, object] = {
        "schema_id": "palette.keypoint.test_source_manifest",
        "schema_version": 1,
        "run_id": "raw_pose_v2_001",
        "logical_schema": KEYPOINT_SCHEMA_V2.as_manifest(dimensions=dimensions),
    }
    source = KeypointQualitySourceReference(
        run_name="raw_pose_v2_001",
        manifest_digest=canonical_json_sha256(source_manifest),
        skeleton_id="sleepyfish_three_point_test",
        skeleton_digest=SKELETON_DIGEST,
        keypoint_row_signatures_digest=sha256_array(row_signatures),
    )
    quality = prepare_observation_local_keypoint_quality(
        raw,
        source_dimensions=dimensions,
        source_crop_arrays=crop,
        source=source,
        skeleton_digest=SKELETON_DIGEST,
        policy=ObservationLocalKeypointQualityPolicy(
            confidence_threshold=0.8,
            minimum_valid_keypoints=2,
        ),
    )
    return dimensions, raw, crop, quality


def _prepare(decisions=()):  # type: ignore[no-untyped-def]
    dimensions, raw, crop, quality = _fixture()
    prepared = prepare_refined_keypoint_snapshot(
        raw,
        dimensions=dimensions,
        source_crop_arrays=crop,
        skeleton_digest=SKELETON_DIGEST,
        keypoint_quality_arrays=quality.arrays,
        quality_dimensions=quality.dimensions,
        quality_profile=quality.profile,
        decisions=tuple(decisions),
        review_state_map=REVIEW_STATE_MAP,
        reason_code_map=REASON_CODE_MAP,
    )
    return prepared, dimensions, raw, crop, quality


def test_noop_refinement_preserves_identity_order_offsets_and_coordinates() -> None:
    dimensions, raw, _, quality = _fixture()
    before = {path: value.copy() for path, value in raw.items()}

    prepared, _, _, crop, _ = _prepare()
    arrays = prepared.arrays

    assert set(arrays) == set(REFINED_KEYPOINT_SCHEMA_V2.binding_paths)
    assert arrays["frame_row_offsets"].tolist() == [0, 2, 2, 3, 4]
    assert arrays["instance_key"].tolist() == [101, 102, 201, 301]
    for path in (
        "instance_key",
        "source_crop_row_ids",
        "source_acquisition_frame_index",
        "frame_indices",
        "frame_row_offsets",
        "source_crop_row_signature",
        "keypoint_row_signature",
        "keypoints_roi",
        "keypoints_img",
    ):
        np.testing.assert_equal(arrays[path], raw[path])
    np.testing.assert_array_equal(arrays["source_success"], raw["pose_success"])
    np.testing.assert_array_equal(arrays["refined_success"], raw["pose_success"])
    np.testing.assert_array_equal(
        arrays["confidence_valid"], quality.arrays["proposed_pose_usable"]
    )
    assert not np.any(arrays["keypoint_edit_flags"])
    assert not np.any(arrays["flip_corrected"])
    assert "heading" not in " ".join(arrays)
    REFINED_KEYPOINT_SCHEMA_V2.require(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=crop,
        skeleton_digest=SKELETON_DIGEST,
        review_state_map=REVIEW_STATE_MAP,
        reason_code_map=REASON_CODE_MAP,
    )
    for path, expected in before.items():
        np.testing.assert_equal(raw[path], expected)


def test_coordinate_validity_and_flip_edits_reproject_and_resign_rows() -> None:
    decisions = (
        RefinedKeypointDecision(
            instance_key=201,
            accepted=True,
            review_state_code=1,
            reason_code=2,
            coordinate_edits=(LandmarkCoordinateEdit(0, (22.0, 23.0)),),
            validity_edits=(LandmarkValidityEdit(1, False),),
            confidence_valid=True,
            geometry_valid=True,
        ),
        RefinedKeypointDecision(
            instance_key=301,
            accepted=True,
            review_state_code=1,
            reason_code=2,
            flip_permutation=(0, 2, 1),
            confidence_valid=True,
            geometry_valid=True,
        ),
    )

    prepared, _, raw, _, _ = _prepare(decisions)
    arrays = prepared.arrays

    np.testing.assert_array_equal(arrays["keypoints_roi"][2, 0], [22.0, 23.0])
    assert np.all(np.isnan(arrays["keypoints_roi"][2, 1]))
    np.testing.assert_array_equal(arrays["keypoints_img"][2, 0], [222.0, 123.0])
    assert arrays["keypoint_edit_flags"][2].tolist() == [True, True, False]
    assert not np.array_equal(
        arrays["keypoint_row_signature"][2], raw["keypoint_row_signature"][2]
    )
    np.testing.assert_array_equal(
        arrays["keypoints_roi"][3, 1], raw["keypoints_roi"][3, 2]
    )
    np.testing.assert_array_equal(
        arrays["keypoints_roi"][3, 2], raw["keypoints_roi"][3, 1]
    )
    assert arrays["flip_corrected"][3]
    assert arrays["keypoint_edit_flags"][3].tolist() == [False, True, True]
    assert arrays["usable_keypoints"][2]
    assert arrays["usable_keypoints"][3]


def test_rejection_clears_row_and_manual_edits_can_recover_source_failure() -> None:
    decisions = (
        RefinedKeypointDecision(
            instance_key=101,
            accepted=False,
            review_state_code=2,
            reason_code=1,
        ),
        RefinedKeypointDecision(
            instance_key=102,
            accepted=True,
            review_state_code=1,
            reason_code=2,
            coordinate_edits=(
                LandmarkCoordinateEdit(0, (10.0, 11.0)),
                LandmarkCoordinateEdit(1, (20.0, 21.0)),
            ),
            confidence_valid=True,
            geometry_valid=True,
        ),
    )

    prepared, _, _, _, _ = _prepare(decisions)
    arrays = prepared.arrays

    assert not arrays["refined_success"][0]
    assert not arrays["usable_keypoints"][0]
    assert np.all(np.isnan(arrays["keypoints_roi"][0]))
    assert arrays["keypoint_edit_flags"][0].tolist() == [True, True, True]
    assert not arrays["source_success"][1]
    assert arrays["refined_success"][1]
    assert arrays["usable_keypoints"][1]
    assert arrays["keypoint_valid"][1].tolist() == [True, True, False]
    np.testing.assert_array_equal(arrays["keypoints_img"][1, 0], [110.0, 61.0])
    assert arrays["review_state_codes"].tolist() == [2, 1, 0, 0]
    assert arrays["reason_codes"].tolist() == [1, 2, 0, 0]


@pytest.mark.parametrize(
    ("decisions", "match"),
    [
        (
            (
                RefinedKeypointDecision(101, True, 1, 0),
                RefinedKeypointDecision(101, True, 1, 0),
            ),
            "At most one",
        ),
        ((RefinedKeypointDecision(999, True, 1, 0),), "unknown instance_key"),
        (
            (RefinedKeypointDecision(101, True, 1, 0, flip_permutation=(0, 1)),),
            "complete skeleton permutation",
        ),
        (
            (
                RefinedKeypointDecision(
                    101,
                    True,
                    1,
                    2,
                    coordinate_edits=(LandmarkCoordinateEdit(0, (500.0, 10.0)),),
                ),
            ),
            "outside crop bounds",
        ),
        (
            (
                RefinedKeypointDecision(
                    102,
                    True,
                    1,
                    2,
                    validity_edits=(LandmarkValidityEdit(0, True),),
                ),
            ),
            "cannot be made valid",
        ),
    ],
)
def test_invalid_decisions_fail_closed(decisions: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _prepare(decisions)


def test_decision_rejects_conflicting_or_ambiguous_edits() -> None:
    with pytest.raises(ValueError, match="nonzero reason_code"):
        RefinedKeypointDecision(101, False, 2, 0)
    with pytest.raises(ValueError, match="conflict"):
        RefinedKeypointDecision(
            101,
            True,
            1,
            2,
            coordinate_edits=(LandmarkCoordinateEdit(0, (1.0, 2.0)),),
            validity_edits=(LandmarkValidityEdit(0, False),),
        )
    with pytest.raises(ValueError, match="cannot be the identity"):
        _prepare(
            (RefinedKeypointDecision(101, True, 1, 2, flip_permutation=(0, 1, 2)),)
        )
