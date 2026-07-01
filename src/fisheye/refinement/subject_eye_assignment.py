"""Assign raw union eye masks into canonical left/right subject-mask components."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time
from typing import Mapping

import numpy as np

from ..shared.mask_geometry import measure_mask_ellipse as _measure_mask
from ..shared.mask_geometry import connected_component_labels as _connected_component_labels
from ..shared.mask_geometry import select_component_near_point as _select_component_near_point

EYES_UNION_ASSIGNMENT_METHOD = "subject_eyes_union_keypoint_assignment_v1"
EYE_COMPONENTS = ("eye_left", "eye_right")
_HALFPLANE_EXACT_BOUNDARY_EPS = np.float32(0.25)


@dataclass(frozen=True)
class EyesUnionAssignmentResult:
    masks: Mapping[str, np.ndarray]
    reason_labels: Mapping[str, np.ndarray]
    assignment_status: np.ndarray
    summary: dict[str, object]
    phase_seconds: Mapping[str, float]
    eye_geometry: Mapping[str, object]


def _split_union_by_keypoints(
    union_mask: np.ndarray,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_mask = np.zeros_like(union_mask, dtype=bool)
    right_mask = np.zeros_like(union_mask, dtype=bool)
    ys, xs = np.nonzero(np.asarray(union_mask, dtype=bool))
    if ys.size <= 0:
        return left_mask, right_mask

    x_coords = xs.astype(np.float32)
    y_coords = ys.astype(np.float32)
    dist_left = (x_coords - float(eye_left[0])) ** 2 + (y_coords - float(eye_left[1])) ** 2
    dist_right = (x_coords - float(eye_right[0])) ** 2 + (y_coords - float(eye_right[1])) ** 2
    assign_left = dist_left <= dist_right
    left_mask[ys[assign_left], xs[assign_left]] = True
    right_mask[ys[~assign_left], xs[~assign_left]] = True
    return left_mask, right_mask


def _split_union_by_keypoints_distance_batch_into(
    union_masks: np.ndarray,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    *,
    row_indices: np.ndarray,
    left_out: np.ndarray,
    right_out: np.ndarray,
    batch_size: int = 32,
) -> None:
    """Reference split by explicit squared distances into caller-owned buffers."""

    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        return
    if batch_size <= 0:
        raise ValueError("split batch_size must be positive.")

    union_bool = np.asarray(union_masks, dtype=bool)
    if union_bool.ndim != 3:
        raise ValueError(f"union_masks must have shape (N,H,W), got {tuple(union_bool.shape)}.")
    height = int(union_bool.shape[1])
    width = int(union_bool.shape[2])
    x_grid = np.arange(width, dtype=np.float32).reshape(1, 1, width)
    y_grid = np.arange(height, dtype=np.float32).reshape(1, height, 1)

    left_points = np.asarray(eye_left, dtype=np.float32)
    right_points = np.asarray(eye_right, dtype=np.float32)
    for offset in range(0, int(rows.size), int(batch_size)):
        batch_rows = rows[offset : offset + int(batch_size)]
        left = left_points[batch_rows]
        right = right_points[batch_rows]
        union = union_bool[batch_rows]
        left_dx = x_grid - left[:, 0].reshape(-1, 1, 1)
        left_dy = y_grid - left[:, 1].reshape(-1, 1, 1)
        right_dx = x_grid - right[:, 0].reshape(-1, 1, 1)
        right_dy = y_grid - right[:, 1].reshape(-1, 1, 1)
        dist_left = left_dx * left_dx + left_dy * left_dy
        dist_right = right_dx * right_dx + right_dy * right_dy
        assign_left = dist_left <= dist_right
        left_out[batch_rows] = np.asarray(union & assign_left, dtype=np.uint8)
        right_out[batch_rows] = np.asarray(union & ~assign_left, dtype=np.uint8)


def _split_union_by_keypoints_halfplane_batch_into(
    union_masks: np.ndarray,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    *,
    row_indices: np.ndarray,
    left_out: np.ndarray,
    right_out: np.ndarray,
    batch_size: int = 32,
) -> None:
    """Split candidate rows by the keypoint bisector half-plane.

    This is algebraically equivalent to ``dist_left <= dist_right`` but avoids
    materializing two dense squared-distance fields per row. Ties still go left.
    """

    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        return
    if batch_size <= 0:
        raise ValueError("split batch_size must be positive.")

    union_bool = np.asarray(union_masks, dtype=bool)
    if union_bool.ndim != 3:
        raise ValueError(f"union_masks must have shape (N,H,W), got {tuple(union_bool.shape)}.")
    height = int(union_bool.shape[1])
    width = int(union_bool.shape[2])
    x_grid = np.arange(width, dtype=np.float32).reshape(1, 1, width)
    y_grid = np.arange(height, dtype=np.float32).reshape(1, height, 1)

    left_points = np.asarray(eye_left, dtype=np.float32)
    right_points = np.asarray(eye_right, dtype=np.float32)
    for offset in range(0, int(rows.size), int(batch_size)):
        batch_rows = rows[offset : offset + int(batch_size)]
        left = left_points[batch_rows]
        right = right_points[batch_rows]
        union = union_bool[batch_rows]
        signed_distance_delta = (
            (np.float32(2.0) * (right[:, 0] - left[:, 0]).reshape(-1, 1, 1) * x_grid)
            + (np.float32(2.0) * (right[:, 1] - left[:, 1]).reshape(-1, 1, 1) * y_grid)
            + (
                left[:, 0] * left[:, 0]
                + left[:, 1] * left[:, 1]
                - right[:, 0] * right[:, 0]
                - right[:, 1] * right[:, 1]
            ).reshape(-1, 1, 1)
        )
        assign_left = signed_distance_delta <= 0.0
        boundary = union & (np.abs(signed_distance_delta) <= _HALFPLANE_EXACT_BOUNDARY_EPS)
        if bool(np.any(boundary)):
            local_rows, ys, xs = np.nonzero(boundary)
            x_coords = xs.astype(np.float32, copy=False)
            y_coords = ys.astype(np.float32, copy=False)
            boundary_left = left[local_rows]
            boundary_right = right[local_rows]
            dist_left = (
                (x_coords - boundary_left[:, 0]) * (x_coords - boundary_left[:, 0])
                + (y_coords - boundary_left[:, 1]) * (y_coords - boundary_left[:, 1])
            )
            dist_right = (
                (x_coords - boundary_right[:, 0]) * (x_coords - boundary_right[:, 0])
                + (y_coords - boundary_right[:, 1]) * (y_coords - boundary_right[:, 1])
            )
            assign_left[local_rows, ys, xs] = dist_left <= dist_right
        left_out[batch_rows] = np.asarray(union & assign_left, dtype=np.uint8)
        right_out[batch_rows] = np.asarray(union & ~assign_left, dtype=np.uint8)


def _split_union_by_keypoints_sparse_batch_into(
    union_masks: np.ndarray,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    *,
    row_indices: np.ndarray,
    left_out: np.ndarray,
    right_out: np.ndarray,
    batch_size: int = 32,
) -> None:
    """Split foreground pixels only, preserving the explicit distance rule.

    This avoids dense 512x512 distance fields when ``eyes_union`` is sparse.
    Ties still go left via ``dist_left <= dist_right``.
    """

    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        return
    if batch_size <= 0:
        raise ValueError("split batch_size must be positive.")

    union_bool = np.asarray(union_masks, dtype=bool)
    if union_bool.ndim != 3:
        raise ValueError(f"union_masks must have shape (N,H,W), got {tuple(union_bool.shape)}.")

    left_points = np.asarray(eye_left, dtype=np.float32)
    right_points = np.asarray(eye_right, dtype=np.float32)
    for offset in range(0, int(rows.size), int(batch_size)):
        batch_rows = rows[offset : offset + int(batch_size)]
        left_out[batch_rows] = 0
        right_out[batch_rows] = 0

        local_rows, ys, xs = np.nonzero(union_bool[batch_rows])
        if local_rows.size <= 0:
            continue

        global_rows = batch_rows[local_rows]
        x_coords = xs.astype(np.float32, copy=False)
        y_coords = ys.astype(np.float32, copy=False)
        left = left_points[global_rows]
        right = right_points[global_rows]
        dist_left = (
            (x_coords - left[:, 0]) * (x_coords - left[:, 0])
            + (y_coords - left[:, 1]) * (y_coords - left[:, 1])
        )
        dist_right = (
            (x_coords - right[:, 0]) * (x_coords - right[:, 0])
            + (y_coords - right[:, 1]) * (y_coords - right[:, 1])
        )
        assign_left = dist_left <= dist_right
        left_out[global_rows[assign_left], ys[assign_left], xs[assign_left]] = 1
        right_out[global_rows[~assign_left], ys[~assign_left], xs[~assign_left]] = 1


def _select_label_near_point(
    *,
    labels: np.ndarray,
    label_values: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    candidate_labels: np.ndarray,
    point_xy: np.ndarray,
    label_count: int,
) -> np.ndarray:
    selected = np.zeros(labels.shape, dtype=bool)
    if int(candidate_labels.size) <= 0:
        return selected
    if int(candidate_labels.size) == 1:
        return labels == int(candidate_labels[0])

    areas = np.bincount(label_values, minlength=int(label_count) + 1).astype(np.float32)
    sum_x = np.bincount(
        label_values,
        weights=xs.astype(np.float32, copy=False),
        minlength=int(label_count) + 1,
    ).astype(np.float32)
    sum_y = np.bincount(
        label_values,
        weights=ys.astype(np.float32, copy=False),
        minlength=int(label_count) + 1,
    ).astype(np.float32)
    centroid_x = np.divide(sum_x, areas, out=np.zeros_like(sum_x), where=areas > 0)
    centroid_y = np.divide(sum_y, areas, out=np.zeros_like(sum_y), where=areas > 0)
    point = np.asarray(point_xy, dtype=np.float32).reshape(-1)
    distances = (
        (centroid_x[candidate_labels] - float(point[0])) ** 2
        + (centroid_y[candidate_labels] - float(point[1])) ** 2
    )
    best_label = int(candidate_labels[int(np.argmin(distances))])
    if best_label > 0:
        selected = labels == best_label
    return selected


def _assign_union_components_fast_batch_into(
    union_masks: np.ndarray,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    *,
    row_indices: np.ndarray,
    left_out: np.ndarray,
    right_out: np.ndarray,
) -> np.ndarray:
    """Assign whole connected components when no component crosses the split.

    Returns the subset of ``row_indices`` that were handled exactly. Any row
    containing a component with pixels on both sides is left untouched so the
    caller can fall back to the standard split/select path.
    """

    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        return np.zeros((0,), dtype=np.int64)

    union_bool = np.asarray(union_masks, dtype=bool)
    if union_bool.ndim != 3:
        raise ValueError(f"union_masks must have shape (N,H,W), got {tuple(union_bool.shape)}.")

    left_points = np.asarray(eye_left, dtype=np.float32)
    right_points = np.asarray(eye_right, dtype=np.float32)
    handled: list[int] = []
    for row in rows:
        row_idx = int(row)
        binary = np.asarray(union_bool[row_idx], dtype=bool)
        if not bool(np.any(binary)):
            continue

        labels, component_count = _connected_component_labels(binary)
        if int(component_count) <= 0:
            continue

        ys, xs = np.nonzero(binary)
        label_values = labels[ys, xs].astype(np.int32, copy=False)
        x_coords = xs.astype(np.float32, copy=False)
        y_coords = ys.astype(np.float32, copy=False)
        left = left_points[row_idx]
        right = right_points[row_idx]
        dist_left = (
            (x_coords - left[0]) * (x_coords - left[0])
            + (y_coords - left[1]) * (y_coords - left[1])
        )
        dist_right = (
            (x_coords - right[0]) * (x_coords - right[0])
            + (y_coords - right[1]) * (y_coords - right[1])
        )
        assign_left = dist_left <= dist_right

        counts = np.bincount(label_values, minlength=int(component_count) + 1)
        left_counts = np.bincount(label_values[assign_left], minlength=int(component_count) + 1)
        right_counts = counts - left_counts
        crosses_split = (left_counts[1:] > 0) & (right_counts[1:] > 0)
        if bool(np.any(crosses_split)):
            continue

        left_labels = np.flatnonzero(left_counts > 0).astype(np.int32, copy=False)
        right_labels = np.flatnonzero(right_counts > 0).astype(np.int32, copy=False)
        left_labels = left_labels[left_labels > 0]
        right_labels = right_labels[right_labels > 0]

        left_out[row_idx] = _select_label_near_point(
            labels=labels,
            label_values=label_values,
            xs=xs,
            ys=ys,
            candidate_labels=left_labels,
            point_xy=left,
            label_count=int(component_count),
        ).astype(np.uint8, copy=False)
        right_out[row_idx] = _select_label_near_point(
            labels=labels,
            label_values=label_values,
            xs=xs,
            ys=ys,
            candidate_labels=right_labels,
            point_xy=right,
            label_count=int(component_count),
        ).astype(np.uint8, copy=False)
        handled.append(row_idx)

    return np.asarray(handled, dtype=np.int64)


def _split_union_by_keypoints_batch_into(
    union_masks: np.ndarray,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    *,
    row_indices: np.ndarray,
    left_out: np.ndarray,
    right_out: np.ndarray,
    batch_size: int = 32,
) -> None:
    """Split candidate rows by keypoint distance into caller-owned buffers."""

    _split_union_by_keypoints_distance_batch_into(
        union_masks,
        eye_left,
        eye_right,
        row_indices=row_indices,
        left_out=left_out,
        right_out=right_out,
        batch_size=batch_size,
    )


def _join_reason_tags(tags: list[str]) -> str:
    return "|".join(str(tag) for tag in tags if str(tag).strip())


def _add_phase(phase_seconds: dict[str, float], name: str, started_at: float) -> None:
    phase_seconds[str(name)] = float(phase_seconds.get(str(name), 0.0)) + float(time.perf_counter() - started_at)


def assign_eyes_union_to_lr(
    union_masks: np.ndarray,
    *,
    keypoints_roi: np.ndarray,
    keypoint_success: np.ndarray,
    eye_keypoint_indices: tuple[int, int],
    split_batch_size: int = 32,
    use_component_fast_path: bool = False,
    measure_ellipses: bool = True,
) -> EyesUnionAssignmentResult:
    """Convert raw ``eyes_union`` masks into canonical LR eye component masks.

    The assignment is intentionally fail-closed: anatomical identity requires
    valid left/right eye keypoints. Failed rows emit empty LR masks plus reason
    labels so finalization can still create a reviewable refined run without
    pretending those rows are valid.

    ``measure_ellipses=False`` is a diagnostic/benchmark mode for evaluating a
    possible future contract where left/right assignment is separated from
    ellipse-shape QC. Production callers should keep the default.
    """

    union = np.asarray(union_masks, dtype=np.uint8)
    if union.ndim != 3:
        raise ValueError(f"eyes_union masks must have shape (N,H,W), got {tuple(union.shape)}.")
    total_rows = int(union.shape[0])
    height = int(union.shape[1])
    width = int(union.shape[2])

    keypoints = np.asarray(keypoints_roi, dtype=np.float32)
    if keypoints.ndim != 3 or int(keypoints.shape[0]) != total_rows or int(keypoints.shape[2]) < 2:
        raise ValueError(
            "keypoints_roi must have shape (N,K,>=2) and match eyes_union rows; "
            f"got {tuple(keypoints.shape)} for {total_rows} rows."
        )
    success = np.asarray(keypoint_success, dtype=bool).reshape(-1)
    if int(success.shape[0]) != total_rows:
        raise ValueError(
            f"keypoint_success row count {int(success.shape[0])} does not match eyes_union rows {total_rows}."
        )

    left_idx, right_idx = int(eye_keypoint_indices[0]), int(eye_keypoint_indices[1])
    if left_idx >= int(keypoints.shape[1]) or right_idx >= int(keypoints.shape[1]):
        raise ValueError(
            f"eye keypoint indices {eye_keypoint_indices!r} exceed keypoints_roi shape {tuple(keypoints.shape)}."
        )

    left_masks = np.zeros((total_rows, height, width), dtype=np.uint8)
    right_masks = np.zeros((total_rows, height, width), dtype=np.uint8)
    reason_labels = {
        "eye_left": np.empty((total_rows,), dtype=object),
        "eye_right": np.empty((total_rows,), dtype=object),
    }
    assignment_status = np.empty((total_rows,), dtype=object)
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    phase_seconds: dict[str, float] = {}
    ellipse_params = np.full((total_rows, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((total_rows, 2), dtype=bool)
    centroids = np.full((total_rows, 2, 2), np.nan, dtype=np.float32)
    contours: dict[str, list[np.ndarray | None]] = {
        "eye_left": [None] * total_rows,
        "eye_right": [None] * total_rows,
    }

    phase_start = time.perf_counter()
    union_bool = union > 0
    union_nonempty = np.any(union_bool, axis=(1, 2))
    left_points = np.asarray(keypoints[:, left_idx, :2], dtype=np.float32)
    right_points = np.asarray(keypoints[:, right_idx, :2], dtype=np.float32)
    left_valid = np.all(np.isfinite(left_points), axis=1)
    right_valid = np.all(np.isfinite(right_points), axis=1)
    coincident = np.all(np.isclose(left_points, right_points, atol=1e-3), axis=1)
    candidate_rows = np.flatnonzero(union_nonempty & success & left_valid & right_valid & ~coincident)
    fast_path_rows = np.zeros((0,), dtype=np.int64)
    fast_path_mask = np.zeros((total_rows,), dtype=bool)
    if bool(use_component_fast_path):
        fast_phase_start = time.perf_counter()
        fast_path_rows = _assign_union_components_fast_batch_into(
            union_bool,
            left_points,
            right_points,
            row_indices=candidate_rows,
            left_out=left_masks,
            right_out=right_masks,
        )
        fast_path_mask[fast_path_rows] = True
        _add_phase(phase_seconds, "component_fast_path", fast_phase_start)
    fallback_rows = candidate_rows[~fast_path_mask[candidate_rows]]
    _split_union_by_keypoints_batch_into(
        union_bool,
        left_points,
        right_points,
        row_indices=fallback_rows,
        left_out=left_masks,
        right_out=right_masks,
        batch_size=int(split_batch_size),
    )
    _add_phase(phase_seconds, "split_by_keypoint", phase_start)

    for row_idx in range(total_rows):
        tags = ["assigned_from_eyes_union"]
        status = "assigned"

        if not bool(union_nonempty[row_idx]):
            tags.append("eyes_union_empty")
            status = "failed_empty_union"
        elif not bool(success[row_idx]):
            tags.append("keypoint_fail")
            status = "failed_keypoint_status"
        else:
            eye_left = left_points[row_idx]
            eye_right = right_points[row_idx]
            if not bool(left_valid[row_idx]) or not bool(right_valid[row_idx]):
                tags.append("missing_eye_keypoints")
                status = "failed_missing_eye_keypoints"
            elif bool(coincident[row_idx]):
                tags.append("coincident_eye_keypoints")
                status = "failed_coincident_eye_keypoints"
            else:
                if bool(fast_path_mask[row_idx]):
                    selected_left = np.asarray(left_masks[row_idx], dtype=bool).copy()
                    selected_right = np.asarray(right_masks[row_idx], dtype=bool).copy()
                    left_masks[row_idx] = 0
                    right_masks[row_idx] = 0
                else:
                    split_left = np.asarray(left_masks[row_idx], dtype=bool).copy()
                    split_right = np.asarray(right_masks[row_idx], dtype=bool).copy()
                    left_masks[row_idx] = 0
                    right_masks[row_idx] = 0
                    phase_start = time.perf_counter()
                    selected_left = _select_component_near_point(split_left, eye_left)
                    selected_right = _select_component_near_point(split_right, eye_right)
                    _add_phase(phase_seconds, "select_components", phase_start)
                if int(np.count_nonzero(selected_left)) <= 0 or int(np.count_nonzero(selected_right)) <= 0:
                    tags.append("split_empty_component")
                    status = "failed_empty_split_component"
                elif bool(np.any(np.logical_and(selected_left, selected_right))):
                    tags.append("split_overlap")
                    status = "failed_split_overlap"
                else:
                    left_masks[row_idx] = selected_left.astype(np.uint8, copy=False)
                    right_masks[row_idx] = selected_right.astype(np.uint8, copy=False)
                    tags.append("split_by_keypoint")
                    if measure_ellipses:
                        phase_start = time.perf_counter()
                        left_success, _left_ellipse, _left_centroid, _left_contour, left_failure = _measure_mask(
                            left_masks[row_idx]
                        )
                        right_success, _right_ellipse, _right_centroid, _right_contour, right_failure = _measure_mask(
                            right_masks[row_idx]
                        )
                        ellipse_params[row_idx, 0] = np.asarray(_left_ellipse, dtype=np.float32)
                        ellipse_params[row_idx, 1] = np.asarray(_right_ellipse, dtype=np.float32)
                        ellipse_success[row_idx, 0] = bool(left_success)
                        ellipse_success[row_idx, 1] = bool(right_success)
                        centroids[row_idx, 0] = np.asarray(_left_centroid, dtype=np.float32)
                        centroids[row_idx, 1] = np.asarray(_right_centroid, dtype=np.float32)
                        contours["eye_left"][row_idx] = _left_contour
                        contours["eye_right"][row_idx] = _right_contour
                        _add_phase(phase_seconds, "measure_ellipse", phase_start)
                        if not bool(left_success):
                            tags.append("ellipse_fail_left")
                            if left_failure:
                                tags.append(f"{left_failure}_left")
                        if not bool(right_success):
                            tags.append("ellipse_fail_right")
                            if right_failure:
                                tags.append(f"{right_failure}_right")
                        if not bool(left_success and right_success):
                            status = "assigned_needs_review"

        phase_start = time.perf_counter()
        reason = _join_reason_tags(tags)
        assignment_status[row_idx] = status
        reason_labels["eye_left"][row_idx] = reason
        reason_labels["eye_right"][row_idx] = reason
        status_counts[str(status)] += 1
        reason_counts[reason] += 1
        _add_phase(phase_seconds, "reason_labels", phase_start)

    assigned_count = int(status_counts.get("assigned", 0))
    needs_review_count = int(status_counts.get("assigned_needs_review", 0))
    failed_count = int(total_rows - assigned_count - needs_review_count)
    summary = {
        "assignment_method": EYES_UNION_ASSIGNMENT_METHOD,
        "total_rows": total_rows,
        "assigned_rows": assigned_count,
        "assigned_needs_review_rows": needs_review_count,
        "failed_rows": failed_count,
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "component_fast_path_requested": bool(use_component_fast_path),
        "component_fast_path_rows": int(fast_path_rows.size),
        "distance_split_rows": int(fallback_rows.size),
        "ellipse_measurement_requested": bool(measure_ellipses),
    }
    return EyesUnionAssignmentResult(
        masks={"eye_left": left_masks, "eye_right": right_masks},
        reason_labels=reason_labels,
        assignment_status=assignment_status,
        summary=summary,
        phase_seconds=phase_seconds,
        eye_geometry={
            "ellipse_params": ellipse_params,
            "ellipse_success": ellipse_success,
            "centroids": centroids,
            "contours": contours,
        },
    )


__all__ = [
    "EYE_COMPONENTS",
    "EYES_UNION_ASSIGNMENT_METHOD",
    "EyesUnionAssignmentResult",
    "assign_eyes_union_to_lr",
]
