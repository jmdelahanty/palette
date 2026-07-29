from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from fisheye.shared.zarr.keypoint_quality_schema import (
    KEYPOINT_QUALITY_SCHEMA_V1,
    KeypointQualityDimensions,
    KeypointQualityProfile,
    KeypointQualitySchemaError,
    KeypointQualitySourceReference,
    QualityMetricDefinition,
)
from fisheye.shared.zarr.keypoint_schema import derive_frame_row_offsets


def _profile() -> KeypointQualityProfile:
    return KeypointQualityProfile(
        profile_id="observation_local_pose_qc",
        profile_version=1,
        policy_digest="ab" * 32,
        keypoint_metrics=(
            QualityMetricDefinition(
                metric_id="confidence_margin",
                metric_version=1,
                units="probability",
                higher_is_worse=False,
                description="Confidence above the profile threshold.",
            ),
            QualityMetricDefinition(
                metric_id="pca_singleview_error",
                metric_version=1,
                units="pixels",
                higher_is_worse=True,
                description="Single-view pose plausibility reconstruction error.",
            ),
        ),
        pose_metrics=(
            QualityMetricDefinition(
                metric_id="valid_landmark_fraction",
                metric_version=1,
                units="fraction",
                higher_is_worse=False,
                description="Fraction of landmarks valid in the source pose.",
            ),
        ),
        keypoint_flag_map={1: "low_confidence", 2: "implausible_geometry"},
        pose_flag_map={1: "insufficient_landmarks"},
    )


def _source_reference() -> KeypointQualitySourceReference:
    return KeypointQualitySourceReference(
        run_name="raw_pose_v2_001",
        manifest_digest="11" * 32,
        skeleton_id="sleepyfish_five_point",
        skeleton_digest="22" * 32,
        keypoint_row_signatures_digest="33" * 32,
    )


def _fixture() -> tuple[
    KeypointQualityDimensions,
    KeypointQualityProfile,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    dimensions = KeypointQualityDimensions(
        n_frames=4,
        n_instances=4,
        n_keypoints=3,
        n_keypoint_metrics=2,
        n_pose_metrics=1,
    )
    profile = _profile()
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    keys = np.asarray([101, 102, 201, 301], dtype=np.uint64)
    signatures = np.arange(4 * 32, dtype=np.uint8).reshape(4, 32)
    source_valid = np.asarray(
        [
            [True, True, True],
            [False, False, False],
            [True, True, False],
            [True, True, True],
        ],
        dtype=bool,
    )
    source = {
        "instance_key": keys.copy(),
        "frame_indices": frames.copy(),
        "keypoint_row_signature": signatures.copy(),
        "keypoint_valid": source_valid.copy(),
        "pose_success": np.asarray([True, False, True, True], dtype=bool),
    }

    keypoint_metrics = np.full((4, 3, 2), np.nan, dtype=np.float32)
    keypoint_metrics[source_valid, 0] = np.float32(0.25)
    keypoint_metrics[source_valid, 1] = np.float32(1.5)
    pose_metrics = np.asarray([[1.0], [np.nan], [2.0 / 3.0], [1.0]], dtype=np.float32)
    arrays = {
        "instance_key": keys,
        "source_keypoint_row_ids": np.arange(4, dtype=np.int64),
        "source_keypoint_row_signature": signatures,
        "frame_indices": frames,
        "frame_row_offsets": derive_frame_row_offsets(frames, n_frames=4),
        "keypoint_metric_values": keypoint_metrics,
        "keypoint_metric_valid": np.isfinite(keypoint_metrics),
        "pose_metric_values": pose_metrics,
        "pose_metric_valid": np.isfinite(pose_metrics),
        "keypoint_quality_flags": np.asarray(
            [[0, 0, 1], [0, 0, 0], [0, 2, 0], [0, 0, 0]],
            dtype=np.uint16,
        ),
        "pose_quality_flags": np.asarray([0, 1, 0, 0], dtype=np.uint16),
        "proposed_keypoint_valid": source_valid.copy(),
        "proposed_pose_usable": np.asarray([True, False, True, True], dtype=bool),
    }
    return dimensions, profile, arrays, source


def test_quality_v1_accepts_multirow_and_empty_frames() -> None:
    dimensions, profile, arrays, source = _fixture()

    KEYPOINT_QUALITY_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        profile=profile,
        source_keypoint_arrays=source,
    )

    assert arrays["frame_row_offsets"].tolist() == [0, 2, 2, 3, 4]
    manifest = KEYPOINT_QUALITY_SCHEMA_V1.as_manifest(
        dimensions=dimensions,
        profile=profile,
        source=_source_reference(),
    )
    assert len(manifest["bindings"]) == 13
    assert manifest["source"]["stage"] == "keypoints"
    assert manifest["source"]["run_path"] == "keypoints_runs/raw_pose_v2_001"
    assert manifest["invariants"]["instances_per_frame"] == "zero_one_or_many"
    assert manifest["invariants"]["heading"].startswith("forbidden")
    assert json.loads(json.dumps(manifest)) == manifest
    assert len(manifest["profile"]["profile_digest"]) == 64


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        (
            lambda arrays, source: arrays["source_keypoint_row_ids"].__setitem__(
                2, 1
            ),
            "incomplete_source_row_coverage",
        ),
        (
            lambda arrays, source: source["instance_key"].__setitem__(0, 999),
            "source_keypoint_binding_mismatch",
        ),
        (
            lambda arrays, source: arrays["keypoint_metric_valid"].__setitem__(
                (0, 0, 0), False
            ),
            "metric_validity_mismatch",
        ),
        (
            lambda arrays, source: arrays["keypoint_quality_flags"].__setitem__(
                (0, 0), 8
            ),
            "undeclared_quality_flag",
        ),
        (
            lambda arrays, source: arrays["proposed_keypoint_valid"].__setitem__(
                (1, 0), True
            ),
            "proposed_keypoint_resurrects_invalid_source",
        ),
        (
            lambda arrays, source: arrays["proposed_pose_usable"].__setitem__(
                1, True
            ),
            "proposed_pose_resurrects_failed_source",
        ),
    ),
)
def test_quality_v1_rejects_cross_array_and_source_tampering(
    mutation: object,
    expected_code: str,
) -> None:
    dimensions, profile, arrays, source = _fixture()
    mutation(arrays, source)  # type: ignore[operator]

    issues = KEYPOINT_QUALITY_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        profile=profile,
        source_keypoint_arrays=source,
    )

    assert expected_code in {issue.code for issue in issues}


def test_quality_v1_forbids_heading_and_coordinate_payloads() -> None:
    dimensions, profile, arrays, source = _fixture()
    arrays["heading_deg"] = np.zeros(4, dtype=np.float32)
    arrays["keypoints_img"] = np.zeros((4, 3, 2), dtype=np.float32)

    issues = KEYPOINT_QUALITY_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        profile=profile,
        source_keypoint_arrays=source,
    )

    forbidden = {
        issue.path
        for issue in issues
        if issue.code == "heading_or_coordinate_payload_forbidden"
    }
    assert forbidden == {"heading_deg", "keypoints_img"}


def test_quality_v1_requires_raw_source_evidence() -> None:
    dimensions, profile, arrays, _ = _fixture()

    with pytest.raises(
        KeypointQualitySchemaError,
        match="missing_source_keypoint_evidence",
    ):
        KEYPOINT_QUALITY_SCHEMA_V1.require(
            arrays,
            dimensions=dimensions,
            profile=profile,
            source_keypoint_arrays=None,
        )


def test_quality_profile_rejects_longitudinal_or_heading_metrics() -> None:
    base = _profile()
    for metric_id in ("temporal_displacement", "heading_error", "track_jump"):
        with pytest.raises(ValueError, match="observation-local"):
            KeypointQualityProfile(
                profile_id=base.profile_id,
                profile_version=base.profile_version,
                policy_digest=base.policy_digest,
                keypoint_metrics=(
                    QualityMetricDefinition(
                        metric_id=metric_id,
                        metric_version=1,
                        units="pixels",
                        higher_is_worse=True,
                        description="Not valid without longitudinal lineage.",
                    ),
                ),
                pose_metrics=(),
                keypoint_flag_map={},
                pose_flag_map={},
            )


def test_profile_digest_changes_with_semantic_policy_declaration() -> None:
    profile = _profile()
    changed = KeypointQualityProfile(
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        policy_digest="cd" * 32,
        keypoint_metrics=profile.keypoint_metrics,
        pose_metrics=profile.pose_metrics,
        keypoint_flag_map=copy.deepcopy(dict(profile.keypoint_flag_map)),
        pose_flag_map=copy.deepcopy(dict(profile.pose_flag_map)),
    )

    assert changed.profile_digest != profile.profile_digest
