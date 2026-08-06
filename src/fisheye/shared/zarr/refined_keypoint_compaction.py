"""Pure compaction of a verified keypoint delta overlay into a successor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fisheye.shared.tabular_deltas import (
    KEYPOINT_OPERATION_CODE_MAP,
    ResolvedKeypointDeltaOverlay,
    apply_keypoint_delta_overlay,
)
from fisheye.shared.zarr.keypoint_quality_schema import (
    KeypointQualityDimensions,
    KeypointQualityProfile,
)
from fisheye.shared.zarr.keypoint_schema import (
    REFINED_KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    derive_keypoint_row_signatures,
)
from fisheye.shared.zarr.refined_keypoint_producer import (
    PreparedRefinedKeypointSnapshot,
)


@dataclass(frozen=True)
class PreparedRefinedKeypointCompaction:
    """Complete logical successor plus its expanded exact registries."""

    prepared: PreparedRefinedKeypointSnapshot
    review_state_map: Mapping[int, str]
    reason_code_map: Mapping[int, str]
    edited_instance_keys: tuple[int, ...]
    overlay_sha256: str


def _array_values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value[:])


def _extend_code_map(
    existing: Mapping[int, str],
    labels: tuple[str, ...],
    *,
    maximum: int,
) -> tuple[dict[int, str], dict[str, int]]:
    result = {int(code): str(label) for code, label in existing.items()}
    if 0 not in result:
        raise ValueError("A refined-keypoint code registry must define code zero.")
    code_by_label = {label: code for code, label in result.items()}
    if len(code_by_label) != len(result):
        raise ValueError("A refined-keypoint code registry reuses a label.")
    next_code = max(result) + 1
    for label in labels:
        if label in code_by_label:
            continue
        if next_code > maximum:
            raise ValueError("Refined-keypoint code registry is exhausted.")
        result[next_code] = label
        code_by_label[label] = next_code
        next_code += 1
    return dict(sorted(result.items())), code_by_label


def prepare_refined_keypoint_compaction(
    parent_arrays: Mapping[str, Any],
    *,
    raw_arrays: Mapping[str, Any],
    dimensions: KeypointDimensions,
    source_crop_arrays: Mapping[str, Any],
    skeleton_digest: str,
    quality_dimensions: KeypointQualityDimensions,
    quality_profile: KeypointQualityProfile,
    parent_review_state_map: Mapping[int, str],
    parent_reason_code_map: Mapping[int, str],
    overlay: ResolvedKeypointDeltaOverlay,
) -> PreparedRefinedKeypointCompaction:
    """Resolve one verified generation into a full immutable logical snapshot.

    The rowset and all source facts remain fixed. Coordinate validity, image
    projection, confidence/QC summaries, row signatures, and review/reason
    codes are regenerated from the resolved final landmark state.
    """

    REFINED_KEYPOINT_SCHEMA_V2.require(
        parent_arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop_arrays,
        skeleton_digest=skeleton_digest,
        review_state_map=parent_review_state_map,
        reason_code_map=parent_reason_code_map,
    )
    if (
        quality_dimensions.n_frames,
        quality_dimensions.n_instances,
        quality_dimensions.n_keypoints,
    ) != (
        dimensions.n_frames,
        dimensions.n_instances,
        dimensions.n_keypoints,
    ):
        raise ValueError("Keypoint-quality dimensions differ during compaction.")
    if overlay.generation_status not in {"frozen", "compacted"}:
        raise ValueError("Keypoint compaction requires a frozen delta generation.")

    parent_keys = _array_values(parent_arrays["instance_key"]).astype(
        np.uint64, copy=False
    )
    points_roi, keypoint_valid = apply_keypoint_delta_overlay(
        parent_arrays["keypoints_roi"],
        instance_keys=parent_arrays["instance_key"],
        overlay=overlay,
    )
    arrays = {
        path: _array_values(parent_arrays[path]).copy()
        for path in REFINED_KEYPOINT_SCHEMA_V2.binding_paths
    }
    confidences = arrays["keypoint_confidences"]
    replace_code = KEYPOINT_OPERATION_CODE_MAP["replace_xy"]
    clear_code = KEYPOINT_OPERATION_CODE_MAP["clear_keypoint"]
    edited_rows: set[int] = set()
    coordinate_decision_rows: set[int] = set()
    newest_reason_by_row: dict[int, tuple[tuple[int, int, str, int], int]] = {}
    label_by_delta_reason_code = {
        int(code): str(label) for label, code in overlay.reason_code_map.items()
    }
    for edit in overlay.edits:
        edited_rows.add(edit.row_index)
        if edit.operation_code in {replace_code, clear_code}:
            coordinate_decision_rows.add(edit.row_index)
        confidences[edit.row_index, edit.keypoint_index] = (
            np.float32(1.0) if edit.valid else np.float32(np.nan)
        )
        order = (
            edit.revision,
            edit.timestamp_ns,
            edit.partition,
            edit.partition_row_index,
        )
        previous = newest_reason_by_row.get(edit.row_index)
        if previous is None or order > previous[0]:
            newest_reason_by_row[edit.row_index] = (order, edit.reason_code)

    review_map, review_code_by_label = _extend_code_map(
        parent_review_state_map,
        ("manual_reviewed", "manual_rejected"),
        maximum=np.iinfo(np.uint8).max,
    )
    delta_reason_labels = tuple(sorted(overlay.reason_code_map))
    reason_map, reason_code_by_label = _extend_code_map(
        parent_reason_code_map,
        delta_reason_labels,
        maximum=np.iinfo(np.uint16).max,
    )

    source_rows = arrays["source_crop_row_ids"].astype(np.int64, copy=False)
    origins = _array_values(source_crop_arrays["roi_coordinates_full"])[source_rows]
    points_img = points_roi + origins.astype(np.float32)[:, None, :]
    refined_success = arrays["refined_success"]
    confidence_valid = arrays["confidence_valid"]
    geometry_valid = arrays["geometry_valid"]
    usable = arrays["usable_keypoints"]
    review_codes = arrays["review_state_codes"]
    reason_codes = arrays["reason_codes"]
    flip_corrected = arrays["flip_corrected"]
    for row in sorted(edited_rows):
        all_valid = bool(np.all(keypoint_valid[row]))
        any_valid = bool(np.any(keypoint_valid[row]))
        conf_ok = bool(
            all_valid
            and np.all(np.isfinite(confidences[row]))
            and np.all(confidences[row] >= np.float32(0.0))
        )
        geom_ok = all_valid
        refined_success[row] = any_valid
        confidence_valid[row] = conf_ok
        geometry_valid[row] = geom_ok
        usable[row] = any_valid and conf_ok and geom_ok
        review_codes[row] = np.uint8(
            review_code_by_label["manual_reviewed" if any_valid else "manual_rejected"]
        )
        _order, delta_reason_code = newest_reason_by_row[row]
        if delta_reason_code == 0:
            reason_codes[row] = np.uint16(0)
        else:
            label = label_by_delta_reason_code.get(delta_reason_code)
            if label is None:
                raise ValueError(
                    f"Delta reason code {delta_reason_code} is undeclared."
                )
            reason_codes[row] = np.uint16(reason_code_by_label[label])
    for row in coordinate_decision_rows:
        flip_corrected[row] = False

    raw_points = _array_values(raw_arrays["keypoints_roi"])
    if raw_points.shape != points_roi.shape:
        raise ValueError("Raw and compacted keypoint coordinate shapes differ.")
    edit_flags = ~np.all(
        (points_roi == raw_points) | (np.isnan(points_roi) & np.isnan(raw_points)),
        axis=2,
    )
    row_signatures = derive_keypoint_row_signatures(
        instance_key=parent_keys,
        source_crop_row_signature=arrays["source_crop_row_signature"],
        keypoints_roi=points_roi,
        keypoint_valid=keypoint_valid,
        skeleton_digest=skeleton_digest,
    )
    arrays.update(
        {
            "keypoint_row_signature": row_signatures,
            "keypoints_roi": points_roi,
            "keypoints_img": points_img,
            "keypoint_confidences": confidences,
            "keypoint_valid": keypoint_valid,
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
        review_state_map=review_map,
        reason_code_map=reason_map,
    )
    prepared = PreparedRefinedKeypointSnapshot(
        dimensions=dimensions,
        quality_dimensions=quality_dimensions,
        quality_profile=quality_profile,
        decisions=(),
        arrays=arrays,
    )
    return PreparedRefinedKeypointCompaction(
        prepared=prepared,
        review_state_map=review_map,
        reason_code_map=reason_map,
        edited_instance_keys=tuple(
            sorted(int(parent_keys[row]) for row in edited_rows)
        ),
        overlay_sha256=overlay.overlay_sha256,
    )


__all__ = [
    "PreparedRefinedKeypointCompaction",
    "prepare_refined_keypoint_compaction",
]
