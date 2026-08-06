from __future__ import annotations

import copy

import numpy as np
import pytest

from fisheye.shared.keypoint_manual_review_qc import (
    ManualKeypointQcPolicy,
    build_default_manual_keypoint_qc_policy,
    build_manual_keypoint_review_derivation,
    evaluate_manual_keypoint_qc,
    manual_keypoint_qc_policy_from_manifest,
    validate_manual_keypoint_review_derivation,
)


LABELS = ("swim_bladder", "eye_left", "eye_right", "tail_tip")


def _policy() -> ManualKeypointQcPolicy:
    return build_default_manual_keypoint_qc_policy(
        skeleton_id="test_pose_v1",
        skeleton_digest="a" * 64,
        keypoint_labels=LABELS,
    )


def _valid_points() -> np.ndarray:
    return np.asarray(
        [[0.0, 0.0], [20.0, 0.0], [0.0, 20.0], [30.0, 30.0]],
        dtype=np.float64,
    )


def test_policy_round_trip_is_exact_and_digest_bound() -> None:
    policy = _policy()
    manifest = policy.as_manifest()

    assert manual_keypoint_qc_policy_from_manifest(manifest) == policy

    tampered = copy.deepcopy(manifest)
    tampered["thresholds"]["confidence_greater_equal"] = 0.9
    with pytest.raises(ValueError, match="canonical form"):
        manual_keypoint_qc_policy_from_manifest(tampered)


def test_policy_rejects_indices_that_disagree_with_skeleton_labels() -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="differ from skeleton labels"):
        ManualKeypointQcPolicy(
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            skeleton_id=policy.skeleton_id,
            skeleton_digest=policy.skeleton_digest,
            keypoint_labels=policy.keypoint_labels,
            head_triangle_indices=(1, 0, 2),
            confidence_threshold=policy.confidence_threshold,
            min_triangle_angle_deg=policy.min_triangle_angle_deg,
            min_triangle_area_px2=policy.min_triangle_area_px2,
            max_triangle_area_px2=policy.max_triangle_area_px2,
            replacement_confidence=policy.replacement_confidence,
        )


def test_shared_evaluator_uses_inclusive_thresholds() -> None:
    policy = _policy()
    result = evaluate_manual_keypoint_qc(
        _valid_points(),
        np.full(len(LABELS), policy.confidence_threshold, dtype=np.float64),
        policy=policy,
    )

    assert result.refined_success is True
    assert result.confidence_valid is True
    assert result.geometry_valid is True
    assert result.usable_keypoints is True
    assert result.triangle_area_px2 == 200.0
    assert result.minimum_triangle_angle_deg == pytest.approx(45.0)


def test_shared_evaluator_rejects_missing_and_low_confidence_landmarks() -> None:
    policy = _policy()
    missing = _valid_points()
    missing[-1] = np.nan
    missing_result = evaluate_manual_keypoint_qc(
        missing,
        np.full(len(LABELS), policy.replacement_confidence, dtype=np.float64),
        policy=policy,
    )
    low_confidence = np.full(
        len(LABELS), policy.replacement_confidence, dtype=np.float64
    )
    low_confidence[-1] = policy.confidence_threshold - 0.01
    low_result = evaluate_manual_keypoint_qc(
        _valid_points(), low_confidence, policy=policy
    )

    assert missing_result.refined_success is True
    assert missing_result.confidence_valid is False
    assert missing_result.geometry_valid is True
    assert missing_result.usable_keypoints is False
    assert low_result.confidence_valid is False
    assert low_result.usable_keypoints is False


def test_review_derivation_rejects_policy_or_lineage_tampering() -> None:
    policy = _policy()
    derivation = build_manual_keypoint_review_derivation(
        base_run_path="refined_keypoints_runs/base",
        delta_run="review",
        generation="generation_000001",
        generation_sha256="b" * 64,
        overlay_sha256="c" * 64,
        partition_count=2,
        event_count=11,
        policy=policy,
    )

    assert validate_manual_keypoint_review_derivation(derivation) == ()

    tampered = copy.deepcopy(derivation)
    tampered["review_qc_policy_digest"] = "d" * 64
    assert validate_manual_keypoint_review_derivation(tampered) == (
        "manual keypoint review derivation is not canonical",
    )
