"""Deterministic observation-local producer for keypoint-quality v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.keypoint_quality_schema import (
    KEYPOINT_QUALITY_SCHEMA_V1,
    KeypointQualityDimensions,
    KeypointQualityProfile,
    KeypointQualitySourceReference,
    QualityMetricDefinition,
)
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    derive_frame_row_offsets,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


KEYPOINT_QUALITY_POLICY_SCHEMA_ID = "palette.keypoint_quality.policy"
KEYPOINT_QUALITY_POLICY_SCHEMA_VERSION = 1
OBSERVATION_LOCAL_QUALITY_PROFILE_ID = "observation_local_baseline"

KEYPOINT_FLAG_LOW_CONFIDENCE = 1
KEYPOINT_FLAG_SOURCE_INVALID = 2
POSE_FLAG_SOURCE_FAILED = 1
POSE_FLAG_INSUFFICIENT_VALID_LANDMARKS = 2


@dataclass(frozen=True)
class ObservationLocalKeypointQualityPolicy:
    """Initial non-temporal keypoint-quality decision policy."""

    confidence_threshold: float = 0.5
    minimum_valid_keypoints: int = 1
    policy_version: int = KEYPOINT_QUALITY_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        threshold = float(self.confidence_threshold)
        if not np.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
            raise ValueError("confidence_threshold must be finite in [0, 1].")
        if (
            type(self.minimum_valid_keypoints) is not int
            or self.minimum_valid_keypoints <= 0
        ):
            raise ValueError("minimum_valid_keypoints must be a positive integer.")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer.")
        object.__setattr__(self, "confidence_threshold", threshold)

    def payload(self) -> dict[str, object]:
        return {
            "schema_id": KEYPOINT_QUALITY_POLICY_SCHEMA_ID,
            "schema_version": self.policy_version,
            "policy_id": OBSERVATION_LOCAL_QUALITY_PROFILE_ID,
            "confidence_threshold": self.confidence_threshold,
            "minimum_valid_keypoints": self.minimum_valid_keypoints,
            "temporal_metrics": "forbidden",
            "heading_metrics": "forbidden",
        }

    @property
    def policy_digest(self) -> str:
        return canonical_json_sha256(self.payload())

    def as_manifest(self) -> dict[str, object]:
        return {**self.payload(), "policy_digest": self.policy_digest}


def quality_profile_for_policy(
    policy: ObservationLocalKeypointQualityPolicy,
) -> KeypointQualityProfile:
    """Return the exact metric axes and flag registries for the policy."""

    return KeypointQualityProfile(
        profile_id=OBSERVATION_LOCAL_QUALITY_PROFILE_ID,
        profile_version=1,
        policy_digest=policy.policy_digest,
        keypoint_metrics=(
            QualityMetricDefinition(
                metric_id="confidence_margin",
                metric_version=1,
                units="probability",
                higher_is_worse=False,
                description=(
                    "Source confidence minus the declared confidence threshold."
                ),
            ),
        ),
        pose_metrics=(
            QualityMetricDefinition(
                metric_id="valid_landmark_fraction",
                metric_version=1,
                units="fraction",
                higher_is_worse=False,
                description=(
                    "Fraction of source landmarks retained by the quality policy."
                ),
            ),
        ),
        keypoint_flag_map={
            KEYPOINT_FLAG_LOW_CONFIDENCE: "low_confidence",
            KEYPOINT_FLAG_SOURCE_INVALID: "source_invalid",
        },
        pose_flag_map={
            POSE_FLAG_SOURCE_FAILED: "source_pose_failed",
            POSE_FLAG_INSUFFICIENT_VALID_LANDMARKS: (
                "insufficient_valid_landmarks"
            ),
        },
    )


@dataclass(frozen=True)
class PreparedKeypointQualitySnapshot:
    dimensions: KeypointQualityDimensions
    profile: KeypointQualityProfile
    policy: ObservationLocalKeypointQualityPolicy
    source: KeypointQualitySourceReference
    arrays: Mapping[str, np.ndarray]
    source_arrays: Mapping[str, np.ndarray]


def _array_values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def prepare_observation_local_keypoint_quality(
    source_keypoint_arrays: Mapping[str, Any],
    *,
    source_dimensions: KeypointDimensions,
    source_crop_arrays: Mapping[str, Any],
    source: KeypointQualitySourceReference,
    skeleton_digest: str,
    policy: ObservationLocalKeypointQualityPolicy = (
        ObservationLocalKeypointQualityPolicy()
    ),
) -> PreparedKeypointQualitySnapshot:
    """Validate raw keypoints and derive one complete quality snapshot."""

    if source.skeleton_digest != skeleton_digest:
        raise ValueError("Source reference and validation skeleton digests differ.")
    if policy.minimum_valid_keypoints > source_dimensions.n_keypoints:
        raise ValueError("minimum_valid_keypoints exceeds the source skeleton size.")
    KEYPOINT_SCHEMA_V2.require(
        source_keypoint_arrays,
        dimensions=source_dimensions,
        source_crop_arrays=source_crop_arrays,
        skeleton_digest=skeleton_digest,
    )
    source_evidence_paths = (
        "instance_key",
        "frame_indices",
        "keypoint_row_signature",
        "keypoint_valid",
        "pose_success",
    )
    source_arrays = {
        path: _array_values(source_keypoint_arrays[path]).copy()
        for path in source_evidence_paths
    }
    observed_signature_digest = sha256_array(
        source_arrays["keypoint_row_signature"]
    )
    if observed_signature_digest != source.keypoint_row_signatures_digest:
        raise ValueError(
            "Source keypoint-row signature digest differs from the source reference."
        )

    profile = quality_profile_for_policy(policy)
    dimensions = KeypointQualityDimensions(
        n_frames=source_dimensions.n_frames,
        n_instances=source_dimensions.n_instances,
        n_keypoints=source_dimensions.n_keypoints,
        n_keypoint_metrics=len(profile.keypoint_metrics),
        n_pose_metrics=len(profile.pose_metrics),
    )
    valid = source_arrays["keypoint_valid"]
    confidence = _array_values(source_keypoint_arrays["keypoint_confidences"])
    pose_success = source_arrays["pose_success"]

    confidence_margin = np.full(valid.shape, np.nan, dtype=np.float32)
    confidence_margin[valid] = (
        confidence[valid] - np.float32(policy.confidence_threshold)
    )
    keypoint_metric_values = confidence_margin[..., None]
    keypoint_metric_valid = np.isfinite(keypoint_metric_values)

    proposed_keypoint_valid = valid & (
        confidence >= np.float32(policy.confidence_threshold)
    )
    proposed_count = np.count_nonzero(proposed_keypoint_valid, axis=1)
    valid_fraction = (
        proposed_count.astype(np.float32)
        / np.float32(source_dimensions.n_keypoints)
    )
    pose_metric_values = valid_fraction[:, None]
    pose_metric_valid = np.ones(pose_metric_values.shape, dtype=bool)
    proposed_pose_usable = pose_success & (
        proposed_count >= policy.minimum_valid_keypoints
    )

    keypoint_flags = np.zeros(valid.shape, dtype=np.uint16)
    keypoint_flags[~valid] |= np.uint16(KEYPOINT_FLAG_SOURCE_INVALID)
    keypoint_flags[valid & ~proposed_keypoint_valid] |= np.uint16(
        KEYPOINT_FLAG_LOW_CONFIDENCE
    )
    pose_flags = np.zeros(source_dimensions.n_instances, dtype=np.uint16)
    pose_flags[~pose_success] |= np.uint16(POSE_FLAG_SOURCE_FAILED)
    pose_flags[proposed_count < policy.minimum_valid_keypoints] |= np.uint16(
        POSE_FLAG_INSUFFICIENT_VALID_LANDMARKS
    )

    frames = source_arrays["frame_indices"]
    arrays = {
        "instance_key": source_arrays["instance_key"].copy(),
        "source_keypoint_row_ids": np.arange(
            source_dimensions.n_instances, dtype=np.int64
        ),
        "source_keypoint_row_signature": source_arrays[
            "keypoint_row_signature"
        ].copy(),
        "frame_indices": frames.copy(),
        "frame_row_offsets": derive_frame_row_offsets(
            frames, n_frames=source_dimensions.n_frames
        ),
        "keypoint_metric_values": keypoint_metric_values,
        "keypoint_metric_valid": keypoint_metric_valid,
        "pose_metric_values": pose_metric_values,
        "pose_metric_valid": pose_metric_valid,
        "keypoint_quality_flags": keypoint_flags,
        "pose_quality_flags": pose_flags,
        "proposed_keypoint_valid": proposed_keypoint_valid,
        "proposed_pose_usable": proposed_pose_usable,
    }
    KEYPOINT_QUALITY_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        profile=profile,
        source_keypoint_arrays=source_arrays,
    )
    return PreparedKeypointQualitySnapshot(
        dimensions=dimensions,
        profile=profile,
        policy=policy,
        source=source,
        arrays=arrays,
        source_arrays=source_arrays,
    )


__all__ = [
    "KEYPOINT_FLAG_LOW_CONFIDENCE",
    "KEYPOINT_FLAG_SOURCE_INVALID",
    "KEYPOINT_QUALITY_POLICY_SCHEMA_ID",
    "KEYPOINT_QUALITY_POLICY_SCHEMA_VERSION",
    "OBSERVATION_LOCAL_QUALITY_PROFILE_ID",
    "POSE_FLAG_INSUFFICIENT_VALID_LANDMARKS",
    "POSE_FLAG_SOURCE_FAILED",
    "ObservationLocalKeypointQualityPolicy",
    "PreparedKeypointQualitySnapshot",
    "prepare_observation_local_keypoint_quality",
    "quality_profile_for_policy",
]
