"""Deterministic observation-local producer for subject-mask quality v1.

The first profile is intentionally conservative.  It records stable local
geometry and topology, applies exact current-schema containment/exclusion
rules, and emits advisory findings.  It never changes source pixels or owns an
accepted review decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from fisheye.shared.mask_geometry import connected_component_labels, hole_stats
from fisheye.shared.refined_subject_component_contours import (
    extract_largest_external_contour,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SUBJECT_MASK_QUALITY_SCHEMA_V1,
    SubjectMaskQualityDimensions,
    SubjectMaskQualityMetricDefinition,
    SubjectMaskQualityProfile,
    SubjectMaskQualitySourceReference,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    derive_subject_mask_frame_row_offsets,
)

SUBJECT_MASK_QUALITY_POLICY_SCHEMA_ID = "palette.subject_mask_quality.policy"
SUBJECT_MASK_QUALITY_POLICY_SCHEMA_VERSION = 1
SUBJECT_V1_LR_QUALITY_PROFILE_ID = "subject_v1_lr_observation_local"
SUBJECT_V1_LR_COMPONENTS = (
    "subject_body",
    "eye_left",
    "eye_right",
    "swim_bladder",
)

COMPONENT_FLAG_SOURCE_UNAVAILABLE = 1
COMPONENT_FLAG_MISSING = 2
COMPONENT_FLAG_MULTIPLE_COMPONENTS = 4
COMPONENT_FLAG_HOLES_PRESENT = 8
COMPONENT_FLAG_TOUCHES_ROI_BORDER = 16

OBSERVATION_FLAG_MISSING_REQUIRED_COMPONENT = 1
OBSERVATION_FLAG_EYE_LEFT_OUTSIDE_BODY = 2
OBSERVATION_FLAG_EYE_RIGHT_OUTSIDE_BODY = 4
OBSERVATION_FLAG_SWIM_BLADDER_OUTSIDE_BODY = 8
OBSERVATION_FLAG_EYE_PAIR_OVERLAP = 16
OBSERVATION_FLAG_EYE_LEFT_SWIM_BLADDER_OVERLAP = 32
OBSERVATION_FLAG_EYE_RIGHT_SWIM_BLADDER_OVERLAP = 64


@dataclass(frozen=True)
class SubjectV1LrObservationQualityPolicy:
    """Exact current-schema relation policy for four-component subject masks."""

    maximum_outside_body_fraction: float = 0.0
    maximum_exclusive_pair_overlap_fraction: float = 0.0
    policy_version: int = SUBJECT_MASK_QUALITY_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "maximum_outside_body_fraction",
            "maximum_exclusive_pair_overlap_fraction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be finite in [0, 1].")
            object.__setattr__(self, name, value)
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer.")

    def payload(self) -> dict[str, object]:
        return {
            "schema_id": SUBJECT_MASK_QUALITY_POLICY_SCHEMA_ID,
            "schema_version": self.policy_version,
            "policy_id": SUBJECT_V1_LR_QUALITY_PROFILE_ID,
            "required_components": list(SUBJECT_V1_LR_COMPONENTS),
            "component_semantics": {
                "subject_body": "whole_subject_silhouette_including_organs",
                "eye_left": "contained_by_subject_body",
                "eye_right": "contained_by_subject_body",
                "swim_bladder": "contained_by_subject_body",
            },
            "allowed_overlap": [
                ["subject_body", "eye_left"],
                ["subject_body", "eye_right"],
                ["subject_body", "swim_bladder"],
            ],
            "exclusive_pairs": [
                ["eye_left", "eye_right"],
                ["eye_left", "swim_bladder"],
                ["eye_right", "swim_bladder"],
            ],
            "maximum_outside_body_fraction": self.maximum_outside_body_fraction,
            "maximum_exclusive_pair_overlap_fraction": (
                self.maximum_exclusive_pair_overlap_fraction
            ),
            "relation_fraction_denominators": {
                "outside_body": "organ_foreground_area",
                "exclusive_pair_overlap": "smaller_component_foreground_area",
            },
            "component_proposal_unusable_flags": [
                "source_component_unavailable",
                "missing_component",
            ],
            "observation_proposal_unusable_flags": [
                "missing_required_component",
                "eye_left_outside_subject_body",
                "eye_right_outside_subject_body",
                "swim_bladder_outside_subject_body",
                "eye_pair_overlap",
                "eye_left_swim_bladder_overlap",
                "eye_right_swim_bladder_overlap",
            ],
            "automatic_pixel_mutation": "forbidden",
            "accepted_review_state_ownership": "forbidden",
            "temporal_metrics": "forbidden",
        }

    @property
    def policy_digest(self) -> str:
        return canonical_json_sha256(self.payload())

    def as_manifest(self) -> dict[str, object]:
        return {**self.payload(), "policy_digest": self.policy_digest}


def quality_profile_for_policy(
    policy: SubjectV1LrObservationQualityPolicy,
) -> SubjectMaskQualityProfile:
    """Return the exact ordered metric axes and finding registries."""

    return SubjectMaskQualityProfile(
        profile_id=SUBJECT_V1_LR_QUALITY_PROFILE_ID,
        profile_version=1,
        policy_digest=policy.policy_digest,
        component_metrics=(
            SubjectMaskQualityMetricDefinition(
                metric_id="foreground_area_fraction_roi",
                metric_version=1,
                units="fraction",
                higher_is_worse=None,
                description="Foreground pixels divided by total ROI pixels.",
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="connected_component_count_8",
                metric_version=1,
                units="count",
                higher_is_worse=True,
                description="Count of 8-connected foreground components.",
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="largest_component_fraction_of_foreground",
                metric_version=1,
                units="fraction",
                higher_is_worse=False,
                description=(
                    "Largest 8-connected component area divided by all foreground area."
                ),
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="hole_count_8",
                metric_version=1,
                units="count",
                higher_is_worse=True,
                description="Count of enclosed 8-connected background regions.",
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="hole_area_fraction_of_filled_area",
                metric_version=1,
                units="fraction",
                higher_is_worse=True,
                description=(
                    "Enclosed hole pixels divided by foreground plus enclosed-hole pixels."
                ),
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="largest_external_contour_solidity",
                metric_version=1,
                units="fraction",
                higher_is_worse=False,
                description=(
                    "Largest canonical external-contour area divided by its convex-hull area."
                ),
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="foreground_roi_border_fraction",
                metric_version=1,
                units="fraction",
                higher_is_worse=True,
                description=(
                    "Unique foreground pixels on the one-pixel ROI perimeter divided by foreground area."
                ),
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="largest_external_contour_isoperimetric_ratio",
                metric_version=1,
                units="ratio",
                higher_is_worse=True,
                description=(
                    "Squared canonical contour perimeter divided by four pi times contour area."
                ),
            ),
        ),
        observation_metrics=(
            SubjectMaskQualityMetricDefinition(
                metric_id="required_component_present_fraction",
                metric_version=1,
                units="fraction",
                higher_is_worse=False,
                description=(
                    "Present required components divided by the four-component policy count."
                ),
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="eye_left_outside_subject_body_fraction",
                metric_version=1,
                units="fraction",
                higher_is_worse=True,
                description="Left-eye pixels outside subject_body divided by left-eye area.",
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="eye_right_outside_subject_body_fraction",
                metric_version=1,
                units="fraction",
                higher_is_worse=True,
                description="Right-eye pixels outside subject_body divided by right-eye area.",
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="swim_bladder_outside_subject_body_fraction",
                metric_version=1,
                units="fraction",
                higher_is_worse=True,
                description=(
                    "Swim-bladder pixels outside subject_body divided by swim-bladder area."
                ),
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="eye_pair_intersection_over_smaller_area",
                metric_version=1,
                units="fraction",
                higher_is_worse=True,
                description=(
                    "Left/right-eye intersection divided by the smaller eye area."
                ),
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="eye_left_swim_bladder_intersection_over_smaller_area",
                metric_version=1,
                units="fraction",
                higher_is_worse=True,
                description=(
                    "Left-eye/swim-bladder intersection divided by the smaller component area."
                ),
            ),
            SubjectMaskQualityMetricDefinition(
                metric_id="eye_right_swim_bladder_intersection_over_smaller_area",
                metric_version=1,
                units="fraction",
                higher_is_worse=True,
                description=(
                    "Right-eye/swim-bladder intersection divided by the smaller component area."
                ),
            ),
        ),
        component_flag_map={
            COMPONENT_FLAG_SOURCE_UNAVAILABLE: "source_component_unavailable",
            COMPONENT_FLAG_MISSING: "missing_component",
            COMPONENT_FLAG_MULTIPLE_COMPONENTS: "multiple_components",
            COMPONENT_FLAG_HOLES_PRESENT: "holes_present",
            COMPONENT_FLAG_TOUCHES_ROI_BORDER: "touches_roi_border",
        },
        observation_flag_map={
            OBSERVATION_FLAG_MISSING_REQUIRED_COMPONENT: (
                "missing_required_component"
            ),
            OBSERVATION_FLAG_EYE_LEFT_OUTSIDE_BODY: (
                "eye_left_outside_subject_body"
            ),
            OBSERVATION_FLAG_EYE_RIGHT_OUTSIDE_BODY: (
                "eye_right_outside_subject_body"
            ),
            OBSERVATION_FLAG_SWIM_BLADDER_OUTSIDE_BODY: (
                "swim_bladder_outside_subject_body"
            ),
            OBSERVATION_FLAG_EYE_PAIR_OVERLAP: "eye_pair_overlap",
            OBSERVATION_FLAG_EYE_LEFT_SWIM_BLADDER_OVERLAP: (
                "eye_left_swim_bladder_overlap"
            ),
            OBSERVATION_FLAG_EYE_RIGHT_SWIM_BLADDER_OVERLAP: (
                "eye_right_swim_bladder_overlap"
            ),
        },
    )


@dataclass(frozen=True)
class PreparedSubjectMaskQualitySnapshot:
    dimensions: SubjectMaskQualityDimensions
    profile: SubjectMaskQualityProfile
    policy: SubjectV1LrObservationQualityPolicy
    source: SubjectMaskQualitySourceReference
    components: SubjectMaskComponentRegistry
    arrays: Mapping[str, np.ndarray]
    source_arrays: Mapping[str, np.ndarray]


def _array_values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def _component_metric_row(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    height, width = (int(binary.shape[0]), int(binary.shape[1]))
    area = int(np.count_nonzero(binary))
    values = np.full((8,), np.nan, dtype=np.float32)
    valid = np.zeros((8,), dtype=bool)

    values[0] = np.float32(area / float(height * width))
    valid[0] = True
    labels, component_count = connected_component_labels(binary)
    values[1] = np.float32(component_count)
    valid[1] = True
    hole_count, hole_fraction, _hole_area = hole_stats(binary)
    values[3] = np.float32(hole_count)
    valid[3] = True
    if area <= 0:
        return values, valid

    sizes = np.bincount(labels[binary].reshape(-1))
    largest_area = int(np.max(sizes[1:])) if sizes.size > 1 else 0
    values[2] = np.float32(largest_area / float(area))
    valid[2] = True
    values[4] = np.float32(hole_fraction)
    valid[4] = True

    border = np.zeros(binary.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    values[6] = np.float32(np.count_nonzero(binary & border) / float(area))
    valid[6] = True

    contour = extract_largest_external_contour(binary, min_points=3)
    if contour is None:
        return values, valid
    contour_cv = contour.reshape(-1, 1, 2)
    contour_area = float(abs(cv2.contourArea(contour_cv)))
    hull = cv2.convexHull(contour_cv)
    hull_area = float(abs(cv2.contourArea(hull)))
    perimeter = float(cv2.arcLength(contour_cv, True))
    if contour_area > 0.0 and hull_area > 0.0:
        values[5] = np.float32(contour_area / hull_area)
        valid[5] = True
    if contour_area > 0.0 and perimeter > 0.0:
        values[7] = np.float32(
            (perimeter * perimeter) / (4.0 * float(np.pi) * contour_area)
        )
        valid[7] = True
    return values, valid


def _outside_fraction(component: np.ndarray, body: np.ndarray) -> float:
    area = int(np.count_nonzero(component))
    if area <= 0:
        return np.nan
    return float(np.count_nonzero(component & ~body) / float(area))


def _intersection_over_smaller_area(left: np.ndarray, right: np.ndarray) -> float:
    smaller = min(int(np.count_nonzero(left)), int(np.count_nonzero(right)))
    if smaller <= 0:
        return np.nan
    return float(np.count_nonzero(left & right) / float(smaller))


def prepare_in_memory_observation_local_subject_mask_quality(
    source_mask_arrays: Mapping[str, Any],
    *,
    n_frames: int,
    components: SubjectMaskComponentRegistry,
    source: SubjectMaskQualitySourceReference,
    policy: SubjectV1LrObservationQualityPolicy = (
        SubjectV1LrObservationQualityPolicy()
    ),
) -> PreparedSubjectMaskQualitySnapshot:
    """Derive one small complete QC snapshot from already materialized masks.

    This is the deterministic reference kernel and fixture path. It refuses a
    lazy/persisted dense array so callers cannot accidentally materialize a
    full-duration mask run. The future publication writer must invoke the same
    policy over bounded, whole-output-shard row blocks.
    """

    if tuple(components.labels) != SUBJECT_V1_LR_COMPONENTS:
        raise ValueError(
            "The first QC profile requires canonical subject_v1_lr component order."
        )
    required_paths = {
        "masks_roi",
        "instance_key",
        "source_acquisition_frame_index",
        "available_channels",
    }
    if not required_paths <= set(source_mask_arrays):
        missing = sorted(required_paths - set(source_mask_arrays))
        raise ValueError(f"Source mask evidence is missing arrays: {missing!r}.")

    if not isinstance(source_mask_arrays["masks_roi"], np.ndarray):
        raise TypeError(
            "The in-memory QC reference producer requires a NumPy masks_roi; "
            "the persisted writer must stream bounded row blocks."
        )
    masks = source_mask_arrays["masks_roi"]
    keys = _array_values(source_mask_arrays["instance_key"])
    frames = _array_values(source_mask_arrays["source_acquisition_frame_index"])
    available = _array_values(source_mask_arrays["available_channels"])
    if masks.ndim != 4 or int(masks.shape[1]) != len(components.labels):
        raise ValueError("masks_roi must have shape (N,4,H,W).")
    if masks.dtype != np.dtype(np.uint8) or np.any((masks != 0) & (masks != 1)):
        raise ValueError("masks_roi must contain exact binary uint8 values.")
    n_rois, n_channels, height, width = (int(value) for value in masks.shape)
    if keys.shape != (n_rois,) or keys.dtype != np.dtype(np.uint64):
        raise ValueError("instance_key must be uint64[N].")
    if frames.shape != (n_rois,) or frames.dtype != np.dtype(np.int64):
        raise ValueError("source_acquisition_frame_index must be int64[N].")
    if available.shape != (n_channels,) or available.dtype != np.dtype(bool):
        raise ValueError("available_channels must be bool[C].")
    if source.component_registry_digest != canonical_json_sha256(
        components.as_manifest()
    ):
        raise ValueError(
            "Source component-registry digest differs from the declared registry."
        )
    offsets = derive_subject_mask_frame_row_offsets(frames, n_frames=int(n_frames))

    profile = quality_profile_for_policy(policy)
    dimensions = SubjectMaskQualityDimensions(
        n_frames=int(n_frames),
        n_rois=n_rois,
        n_channels=n_channels,
        roi_height=height,
        roi_width=width,
        n_component_metrics=len(profile.component_metrics),
        n_observation_metrics=len(profile.observation_metrics),
    )
    component_values = np.full((n_rois, n_channels, 8), np.nan, dtype=np.float32)
    component_valid = np.zeros(component_values.shape, dtype=bool)
    component_flags = np.zeros((n_rois, n_channels), dtype=np.uint16)
    proposed_component_usable = np.zeros((n_rois, n_channels), dtype=bool)

    binary = masks.astype(bool, copy=False)
    present = np.any(binary, axis=(2, 3))
    for channel in range(n_channels):
        if not bool(available[channel]):
            component_flags[:, channel] |= np.uint16(
                COMPONENT_FLAG_SOURCE_UNAVAILABLE
            )
            continue
        for row in range(n_rois):
            metric_values, metric_valid = _component_metric_row(binary[row, channel])
            component_values[row, channel] = metric_values
            component_valid[row, channel] = metric_valid
            if not bool(present[row, channel]):
                component_flags[row, channel] |= np.uint16(COMPONENT_FLAG_MISSING)
                continue
            proposed_component_usable[row, channel] = True
            if metric_values[1] > 1.0:
                component_flags[row, channel] |= np.uint16(
                    COMPONENT_FLAG_MULTIPLE_COMPONENTS
                )
            if metric_values[3] > 0.0:
                component_flags[row, channel] |= np.uint16(
                    COMPONENT_FLAG_HOLES_PRESENT
                )
            if metric_values[6] > 0.0:
                component_flags[row, channel] |= np.uint16(
                    COMPONENT_FLAG_TOUCHES_ROI_BORDER
                )

    indexes = {label: index for index, label in enumerate(components.labels)}
    body_index = indexes["subject_body"]
    left_index = indexes["eye_left"]
    right_index = indexes["eye_right"]
    bladder_index = indexes["swim_bladder"]
    observation_values = np.full((n_rois, 7), np.nan, dtype=np.float32)
    observation_valid = np.zeros(observation_values.shape, dtype=bool)
    observation_flags = np.zeros((n_rois,), dtype=np.uint16)

    required_available = available[
        [body_index, left_index, right_index, bladder_index]
    ]
    for row in range(n_rois):
        required_present = present[
            row, [body_index, left_index, right_index, bladder_index]
        ] & required_available
        observation_values[row, 0] = np.float32(
            np.count_nonzero(required_present) / 4.0
        )
        observation_valid[row, 0] = True
        if not bool(np.all(required_present)):
            observation_flags[row] |= np.uint16(
                OBSERVATION_FLAG_MISSING_REQUIRED_COMPONENT
            )

        body = binary[row, body_index]
        left = binary[row, left_index]
        right = binary[row, right_index]
        bladder = binary[row, bladder_index]
        relation_values = (
            _outside_fraction(left, body),
            _outside_fraction(right, body),
            _outside_fraction(bladder, body),
            _intersection_over_smaller_area(left, right),
            _intersection_over_smaller_area(left, bladder),
            _intersection_over_smaller_area(right, bladder),
        )
        relation_availability = (
            available[body_index] and available[left_index],
            available[body_index] and available[right_index],
            available[body_index] and available[bladder_index],
            available[left_index] and available[right_index],
            available[left_index] and available[bladder_index],
            available[right_index] and available[bladder_index],
        )
        for metric_offset, (value, relation_available) in enumerate(
            zip(relation_values, relation_availability, strict=True), start=1
        ):
            if relation_available and np.isfinite(value):
                observation_values[row, metric_offset] = np.float32(value)
                observation_valid[row, metric_offset] = True

        outside_flags = (
            OBSERVATION_FLAG_EYE_LEFT_OUTSIDE_BODY,
            OBSERVATION_FLAG_EYE_RIGHT_OUTSIDE_BODY,
            OBSERVATION_FLAG_SWIM_BLADDER_OUTSIDE_BODY,
        )
        overlap_flags = (
            OBSERVATION_FLAG_EYE_PAIR_OVERLAP,
            OBSERVATION_FLAG_EYE_LEFT_SWIM_BLADDER_OVERLAP,
            OBSERVATION_FLAG_EYE_RIGHT_SWIM_BLADDER_OVERLAP,
        )
        for metric_offset, flag in enumerate(outside_flags, start=1):
            if (
                observation_valid[row, metric_offset]
                and observation_values[row, metric_offset]
                > policy.maximum_outside_body_fraction
            ):
                observation_flags[row] |= np.uint16(flag)
        for metric_offset, flag in enumerate(overlap_flags, start=4):
            if (
                observation_valid[row, metric_offset]
                and observation_values[row, metric_offset]
                > policy.maximum_exclusive_pair_overlap_fraction
            ):
                observation_flags[row] |= np.uint16(flag)

    arrays = {
        "instance_key": keys.copy(),
        "source_mask_row_ids": np.arange(n_rois, dtype=np.int64),
        "source_acquisition_frame_index": frames.copy(),
        "frame_row_offsets": offsets,
        "component_metric_values": component_values,
        "component_metric_valid": component_valid,
        "observation_metric_values": observation_values,
        "observation_metric_valid": observation_valid,
        "component_quality_flags": component_flags,
        "observation_quality_flags": observation_flags,
        "proposed_component_usable": proposed_component_usable,
        "proposed_observation_usable": observation_flags == 0,
    }
    source_evidence = {
        "instance_key": keys.copy(),
        "source_acquisition_frame_index": frames.copy(),
        "available_channels": available.copy(),
    }
    SUBJECT_MASK_QUALITY_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        components=components,
        profile=profile,
        source_mask_arrays=source_evidence,
    )
    return PreparedSubjectMaskQualitySnapshot(
        dimensions=dimensions,
        profile=profile,
        policy=policy,
        source=source,
        components=components,
        arrays=arrays,
        source_arrays=source_evidence,
    )


__all__ = [
    "COMPONENT_FLAG_HOLES_PRESENT",
    "COMPONENT_FLAG_MISSING",
    "COMPONENT_FLAG_MULTIPLE_COMPONENTS",
    "COMPONENT_FLAG_SOURCE_UNAVAILABLE",
    "COMPONENT_FLAG_TOUCHES_ROI_BORDER",
    "OBSERVATION_FLAG_EYE_LEFT_OUTSIDE_BODY",
    "OBSERVATION_FLAG_EYE_LEFT_SWIM_BLADDER_OVERLAP",
    "OBSERVATION_FLAG_EYE_PAIR_OVERLAP",
    "OBSERVATION_FLAG_EYE_RIGHT_OUTSIDE_BODY",
    "OBSERVATION_FLAG_EYE_RIGHT_SWIM_BLADDER_OVERLAP",
    "OBSERVATION_FLAG_MISSING_REQUIRED_COMPONENT",
    "OBSERVATION_FLAG_SWIM_BLADDER_OUTSIDE_BODY",
    "PreparedSubjectMaskQualitySnapshot",
    "SUBJECT_MASK_QUALITY_POLICY_SCHEMA_ID",
    "SUBJECT_MASK_QUALITY_POLICY_SCHEMA_VERSION",
    "SUBJECT_V1_LR_COMPONENTS",
    "SUBJECT_V1_LR_QUALITY_PROFILE_ID",
    "SubjectV1LrObservationQualityPolicy",
    "prepare_in_memory_observation_local_subject_mask_quality",
    "quality_profile_for_policy",
]
