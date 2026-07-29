"""Pure transformation from raw keypoint-v2 plus quality-v1 to refined-v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.keypoint_quality_schema import (
    KEYPOINT_QUALITY_SCHEMA_V1,
    KeypointQualityDimensions,
    KeypointQualityProfile,
)
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    REFINED_KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    derive_keypoint_row_signatures,
)


def _array_values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


@dataclass(frozen=True)
class LandmarkCoordinateEdit:
    """One accepted ROI-coordinate replacement in final skeleton order."""

    keypoint_index: int
    xy_roi: tuple[float, float]

    def __post_init__(self) -> None:
        if type(self.keypoint_index) is not int or self.keypoint_index < 0:
            raise ValueError("keypoint_index must be a nonnegative exact integer.")
        values = tuple(float(value) for value in self.xy_roi)
        if len(values) != 2 or not np.all(np.isfinite(values)):
            raise ValueError("xy_roi must contain exactly two finite coordinates.")
        object.__setattr__(self, "xy_roi", values)


@dataclass(frozen=True)
class LandmarkValidityEdit:
    """One explicit final landmark-validity decision."""

    keypoint_index: int
    valid: bool

    def __post_init__(self) -> None:
        if type(self.keypoint_index) is not int or self.keypoint_index < 0:
            raise ValueError("keypoint_index must be a nonnegative exact integer.")
        if type(self.valid) is not bool:
            raise TypeError("valid must be an exact bool.")


@dataclass(frozen=True)
class RefinedKeypointDecision:
    """Immutable accepted review/edit decision for one observation identity.

    A flip permutation is applied first. Coordinate and validity edits then
    address final anatomical landmark indices. Rejected rows are cleared and
    cannot also carry coordinate, validity, or flip edits.
    """

    instance_key: int
    accepted: bool
    review_state_code: int
    reason_code: int
    coordinate_edits: tuple[LandmarkCoordinateEdit, ...] = ()
    validity_edits: tuple[LandmarkValidityEdit, ...] = ()
    flip_permutation: tuple[int, ...] | None = None
    confidence_valid: bool | None = None
    geometry_valid: bool | None = None

    def __post_init__(self) -> None:
        if type(self.instance_key) is not int or not (
            0 <= self.instance_key <= 2**64 - 1
        ):
            raise ValueError("instance_key must be an exact uint64-compatible integer.")
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be an exact bool.")
        if type(self.review_state_code) is not int or not (
            0 <= self.review_state_code <= 255
        ):
            raise ValueError(
                "review_state_code must be an exact uint8-compatible integer."
            )
        if type(self.reason_code) is not int or not (0 <= self.reason_code <= 65535):
            raise ValueError("reason_code must be an exact uint16-compatible integer.")
        if not self.accepted and self.reason_code == 0:
            raise ValueError("A rejected decision must declare a nonzero reason_code.")
        coordinate_edits = tuple(self.coordinate_edits)
        validity_edits = tuple(self.validity_edits)
        if any(
            not isinstance(item, LandmarkCoordinateEdit) for item in coordinate_edits
        ):
            raise TypeError(
                "coordinate_edits must contain LandmarkCoordinateEdit values."
            )
        if any(not isinstance(item, LandmarkValidityEdit) for item in validity_edits):
            raise TypeError("validity_edits must contain LandmarkValidityEdit values.")
        coordinate_indices = [item.keypoint_index for item in coordinate_edits]
        validity_indices = [item.keypoint_index for item in validity_edits]
        if len(coordinate_indices) != len(set(coordinate_indices)):
            raise ValueError("A decision cannot edit one landmark coordinate twice.")
        if len(validity_indices) != len(set(validity_indices)):
            raise ValueError("A decision cannot edit one landmark validity twice.")
        validity_by_index = {item.keypoint_index: item.valid for item in validity_edits}
        conflicts = sorted(
            index
            for index in coordinate_indices
            if validity_by_index.get(index) is False
        )
        if conflicts:
            raise ValueError(
                f"Finite coordinate edits conflict with invalidity edits at {conflicts!r}."
            )
        if not self.accepted and (
            coordinate_edits or validity_edits or self.flip_permutation is not None
        ):
            raise ValueError("A rejected decision cannot also carry landmark edits.")
        for name in ("confidence_valid", "geometry_valid"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"{name} must be an exact bool or None.")
        permutation = None
        if self.flip_permutation is not None:
            permutation = tuple(self.flip_permutation)
            if any(type(value) is not int or value < 0 for value in permutation):
                raise ValueError(
                    "flip_permutation must contain nonnegative exact integers."
                )
        object.__setattr__(self, "coordinate_edits", coordinate_edits)
        object.__setattr__(self, "validity_edits", validity_edits)
        object.__setattr__(self, "flip_permutation", permutation)


@dataclass(frozen=True)
class PreparedRefinedKeypointSnapshot:
    dimensions: KeypointDimensions
    quality_dimensions: KeypointQualityDimensions
    quality_profile: KeypointQualityProfile
    decisions: tuple[RefinedKeypointDecision, ...]
    arrays: Mapping[str, np.ndarray]


def _require_decisions(
    decisions: tuple[RefinedKeypointDecision, ...],
    *,
    keys: np.ndarray,
    n_keypoints: int,
    review_state_map: Mapping[int, str],
    reason_code_map: Mapping[int, str],
) -> dict[int, RefinedKeypointDecision]:
    if any(not isinstance(decision, RefinedKeypointDecision) for decision in decisions):
        raise TypeError("decisions must contain RefinedKeypointDecision values.")
    decision_keys = [decision.instance_key for decision in decisions]
    if len(decision_keys) != len(set(decision_keys)):
        raise ValueError("At most one refined decision may target each instance_key.")
    row_by_key = {int(value): row for row, value in enumerate(keys)}
    unknown = sorted(set(decision_keys) - set(row_by_key))
    if unknown:
        raise ValueError(
            f"Refined decisions reference unknown instance_key values: {unknown!r}."
        )
    for decision in decisions:
        if decision.review_state_code not in review_state_map:
            raise ValueError(
                f"review_state_code {decision.review_state_code} is absent from its registry."
            )
        if decision.reason_code not in reason_code_map:
            raise ValueError(
                f"reason_code {decision.reason_code} is absent from its registry."
            )
        indices = [
            *(edit.keypoint_index for edit in decision.coordinate_edits),
            *(edit.keypoint_index for edit in decision.validity_edits),
        ]
        if any(index >= n_keypoints for index in indices):
            raise ValueError("A refined landmark edit index exceeds the skeleton size.")
        permutation = decision.flip_permutation
        if permutation is not None:
            if len(permutation) != n_keypoints or set(permutation) != set(
                range(n_keypoints)
            ):
                raise ValueError(
                    "flip_permutation must be one complete skeleton permutation."
                )
            if permutation == tuple(range(n_keypoints)):
                raise ValueError("flip_permutation cannot be the identity permutation.")
    return {
        row_by_key[key]: decision
        for key, decision in zip(decision_keys, decisions, strict=True)
    }


def prepare_refined_keypoint_snapshot(
    raw_keypoint_arrays: Mapping[str, Any],
    *,
    dimensions: KeypointDimensions,
    source_crop_arrays: Mapping[str, Any],
    skeleton_digest: str,
    keypoint_quality_arrays: Mapping[str, Any],
    quality_dimensions: KeypointQualityDimensions,
    quality_profile: KeypointQualityProfile,
    decisions: tuple[RefinedKeypointDecision, ...],
    review_state_map: Mapping[int, str],
    reason_code_map: Mapping[int, str],
    default_review_state_code: int = 0,
    default_reason_code: int = 0,
) -> PreparedRefinedKeypointSnapshot:
    """Return one exact refined-keypoint-v2 snapshot without mutating inputs."""

    KEYPOINT_SCHEMA_V2.require(
        raw_keypoint_arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop_arrays,
        skeleton_digest=skeleton_digest,
    )
    expected_quality_dimensions = (
        quality_dimensions.n_frames,
        quality_dimensions.n_instances,
        quality_dimensions.n_keypoints,
    )
    if expected_quality_dimensions != (
        dimensions.n_frames,
        dimensions.n_instances,
        dimensions.n_keypoints,
    ):
        raise ValueError("Keypoint-quality dimensions differ from the raw snapshot.")
    KEYPOINT_QUALITY_SCHEMA_V1.require(
        keypoint_quality_arrays,
        dimensions=quality_dimensions,
        profile=quality_profile,
        source_keypoint_arrays=raw_keypoint_arrays,
    )
    if (
        type(default_review_state_code) is not int
        or default_review_state_code not in review_state_map
    ):
        raise ValueError("default_review_state_code must exist in review_state_map.")
    if (
        type(default_reason_code) is not int
        or default_reason_code not in reason_code_map
    ):
        raise ValueError("default_reason_code must exist in reason_code_map.")

    raw = {
        path: _array_values(raw_keypoint_arrays[path])
        for path in KEYPOINT_SCHEMA_V2.binding_paths
    }
    quality = {
        path: _array_values(keypoint_quality_arrays[path])
        for path in KEYPOINT_QUALITY_SCHEMA_V1.binding_paths
    }
    resolved_decisions = tuple(decisions)
    by_row = _require_decisions(
        resolved_decisions,
        keys=raw["instance_key"],
        n_keypoints=dimensions.n_keypoints,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
    )

    # Every output owns its buffers. Source order, identity, frame CSR, crop
    # lineage, bbox, and source-model evidence remain byte-for-byte preserved.
    arrays = {
        path: raw[path].copy()
        for path in KEYPOINT_SCHEMA_V2.binding_paths
        if path not in {"keypoint_row_signature", "keypoints_img", "pose_success"}
    }
    points_roi = raw["keypoints_roi"].copy()
    confidence = raw["keypoint_confidences"].copy()
    source_success = raw["pose_success"].copy()
    refined_success = source_success.copy()
    confidence_valid = quality["proposed_pose_usable"].copy()
    geometry_valid = source_success & np.any(raw["keypoint_valid"], axis=1)
    review_codes = np.full(
        dimensions.n_instances, default_review_state_code, dtype=np.uint8
    )
    reason_codes = np.full(dimensions.n_instances, default_reason_code, dtype=np.uint16)
    flip_corrected = np.zeros(dimensions.n_instances, dtype=bool)

    for row, decision in by_row.items():
        review_codes[row] = np.uint8(decision.review_state_code)
        reason_codes[row] = np.uint16(decision.reason_code)
        if not decision.accepted:
            points_roi[row] = np.float32(np.nan)
            confidence[row] = np.float32(np.nan)
            refined_success[row] = False
            confidence_valid[row] = False
            geometry_valid[row] = False
            continue

        permutation = decision.flip_permutation
        if permutation is not None:
            order = np.asarray(permutation, dtype=np.int64)
            points_roi[row] = points_roi[row, order]
            confidence[row] = confidence[row, order]
            flip_corrected[row] = True
        for edit in decision.coordinate_edits:
            points_roi[row, edit.keypoint_index] = np.asarray(
                edit.xy_roi, dtype=np.float32
            )
        for edit in decision.validity_edits:
            if not edit.valid:
                points_roi[row, edit.keypoint_index] = np.float32(np.nan)
                confidence[row, edit.keypoint_index] = np.float32(np.nan)
            elif not np.all(np.isfinite(points_roi[row, edit.keypoint_index])):
                raise ValueError(
                    "A landmark cannot be made valid without finite final coordinates."
                )
        refined_success[row] = True
        if decision.confidence_valid is not None:
            confidence_valid[row] = decision.confidence_valid
        if decision.geometry_valid is not None:
            geometry_valid[row] = decision.geometry_valid

    keypoint_valid = np.all(np.isfinite(points_roi), axis=2)
    confidence[~keypoint_valid] = np.float32(np.nan)
    source_rows = raw["source_crop_row_ids"]
    origins = _array_values(source_crop_arrays["roi_coordinates_full"])[source_rows]
    sizes = _array_values(source_crop_arrays["roi_sizes_full"])[source_rows]
    finite = keypoint_valid
    outside = finite & (
        (points_roi[..., 0] < np.float32(0.0))
        | (points_roi[..., 1] < np.float32(0.0))
        | (points_roi[..., 0] >= sizes[:, None, 0])
        | (points_roi[..., 1] >= sizes[:, None, 1])
    )
    if np.any(outside):
        row, keypoint = np.argwhere(outside)[0]
        raise ValueError(
            f"Refined ROI coordinate is outside crop bounds at row {row}, keypoint {keypoint}."
        )
    points_img = points_roi + origins.astype(np.float32)[:, None, :]
    edit_flags = ~np.all(
        (points_roi == raw["keypoints_roi"])
        | (np.isnan(points_roi) & np.isnan(raw["keypoints_roi"])),
        axis=2,
    )
    if np.any(flip_corrected & ~np.any(edit_flags, axis=1)):
        raise ValueError(
            "A flip correction must change at least one landmark coordinate."
        )
    if np.any(refined_success & ~np.any(keypoint_valid, axis=1)):
        raise ValueError(
            "An accepted refined row must retain at least one valid landmark."
        )
    geometry_valid &= np.any(keypoint_valid, axis=1)
    usable = refined_success & confidence_valid & geometry_valid
    row_signatures = derive_keypoint_row_signatures(
        instance_key=raw["instance_key"],
        source_crop_row_signature=raw["source_crop_row_signature"],
        keypoints_roi=points_roi,
        keypoint_valid=keypoint_valid,
        skeleton_digest=skeleton_digest,
    )

    arrays.update(
        {
            "keypoint_row_signature": row_signatures,
            "keypoints_roi": points_roi,
            "keypoints_img": points_img,
            "keypoint_confidences": confidence,
            "keypoint_valid": keypoint_valid,
            "source_success": source_success,
            "refined_success": refined_success,
            "keypoint_edit_flags": edit_flags,
            "flip_corrected": flip_corrected,
            "confidence_valid": confidence_valid,
            "geometry_valid": geometry_valid,
            "usable_keypoints": usable,
            "review_state_codes": review_codes,
            "reason_codes": reason_codes,
        }
    )
    REFINED_KEYPOINT_SCHEMA_V2.require(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop_arrays,
        skeleton_digest=skeleton_digest,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
    )
    return PreparedRefinedKeypointSnapshot(
        dimensions=dimensions,
        quality_dimensions=quality_dimensions,
        quality_profile=quality_profile,
        decisions=tuple(sorted(resolved_decisions, key=lambda item: item.instance_key)),
        arrays=arrays,
    )


__all__ = [
    "LandmarkCoordinateEdit",
    "LandmarkValidityEdit",
    "PreparedRefinedKeypointSnapshot",
    "RefinedKeypointDecision",
    "prepare_refined_keypoint_snapshot",
]
