"""Assign raw union eye masks into canonical left/right subject-mask components."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from .refine_eye_masks import _measure_mask

EYES_UNION_ASSIGNMENT_METHOD = "subject_eyes_union_keypoint_assignment_v1"
EYE_COMPONENTS = ("eye_left", "eye_right")


@dataclass(frozen=True)
class EyesUnionAssignmentResult:
    masks: Mapping[str, np.ndarray]
    reason_labels: Mapping[str, np.ndarray]
    assignment_status: np.ndarray
    summary: dict[str, object]


def _valid_eye_point(point: np.ndarray) -> bool:
    arr = np.asarray(point, dtype=np.float32).reshape(-1)
    return arr.size >= 2 and bool(np.all(np.isfinite(arr[:2])))


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


def _select_component_near_point(mask: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    selected = np.zeros_like(binary, dtype=bool)
    if int(np.count_nonzero(binary)) <= 0:
        return selected

    label_count, labels, _stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if label_count <= 1:
        return selected

    point = np.asarray(point_xy, dtype=np.float32).reshape(-1)
    best_label = 0
    best_distance = float("inf")
    for label_idx in range(1, int(label_count)):
        centroid = np.asarray(centroids[label_idx], dtype=np.float32)
        distance = float(np.sum(np.square(centroid[:2] - point[:2], dtype=np.float32), dtype=np.float32))
        if distance < best_distance:
            best_distance = distance
            best_label = int(label_idx)

    if best_label > 0:
        selected = labels == best_label
    return selected


def _join_reason_tags(tags: list[str]) -> str:
    return "|".join(str(tag) for tag in tags if str(tag).strip())


def assign_eyes_union_to_lr(
    union_masks: np.ndarray,
    *,
    keypoints_roi: np.ndarray,
    keypoint_success: np.ndarray,
    eye_keypoint_indices: tuple[int, int],
) -> EyesUnionAssignmentResult:
    """Convert raw ``eyes_union`` masks into canonical LR eye component masks.

    The assignment is intentionally fail-closed: anatomical identity requires
    valid left/right eye keypoints. Failed rows emit empty LR masks plus reason
    labels so finalization can still create a reviewable refined run without
    pretending those rows are valid.
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

    for row_idx in range(total_rows):
        union_mask = np.asarray(union[row_idx], dtype=bool)
        tags = ["assigned_from_eyes_union"]
        status = "assigned"

        if int(np.count_nonzero(union_mask)) <= 0:
            tags.append("eyes_union_empty")
            status = "failed_empty_union"
        elif not bool(success[row_idx]):
            tags.append("keypoint_fail")
            status = "failed_keypoint_status"
        else:
            eye_left = np.asarray(keypoints[row_idx, left_idx, :2], dtype=np.float32)
            eye_right = np.asarray(keypoints[row_idx, right_idx, :2], dtype=np.float32)
            if not _valid_eye_point(eye_left) or not _valid_eye_point(eye_right):
                tags.append("missing_eye_keypoints")
                status = "failed_missing_eye_keypoints"
            elif bool(np.allclose(eye_left, eye_right, atol=1e-3)):
                tags.append("coincident_eye_keypoints")
                status = "failed_coincident_eye_keypoints"
            else:
                split_left, split_right = _split_union_by_keypoints(union_mask, eye_left, eye_right)
                selected_left = _select_component_near_point(split_left, eye_left)
                selected_right = _select_component_near_point(split_right, eye_right)
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
                    left_success, _left_ellipse, _left_centroid, _left_contour, left_failure = _measure_mask(
                        left_masks[row_idx]
                    )
                    right_success, _right_ellipse, _right_centroid, _right_contour, right_failure = _measure_mask(
                        right_masks[row_idx]
                    )
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

        reason = _join_reason_tags(tags)
        assignment_status[row_idx] = status
        reason_labels["eye_left"][row_idx] = reason
        reason_labels["eye_right"][row_idx] = reason
        status_counts[str(status)] += 1
        reason_counts[reason] += 1

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
    }
    return EyesUnionAssignmentResult(
        masks={"eye_left": left_masks, "eye_right": right_masks},
        reason_labels=reason_labels,
        assignment_status=assignment_status,
        summary=summary,
    )


__all__ = [
    "EYE_COMPONENTS",
    "EYES_UNION_ASSIGNMENT_METHOD",
    "EyesUnionAssignmentResult",
    "assign_eyes_union_to_lr",
]
