"""Deterministic selection for sampled-training detection instances.

This module contains no Zarr I/O.  It classifies every sampled crop proposal
against detections made on the corresponding full camera frame.  The strict
policy deliberately rejects an entire sampled frame when it has zero or more
than one detector candidate; it never silently chooses among multiple animals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SAMPLED_TRAINING_STRONG_SINGLE_POLICY_SCHEMA_ID = (
    "palette.sampled_training_detection.strong_single_policy"
)
SAMPLED_TRAINING_STRONG_SINGLE_POLICY_SCHEMA_VERSION = 1
SAMPLED_TRAINING_STRONG_SINGLE_POLICY_CURRENT_SCHEMA_VERSION = 2
SAMPLED_TRAINING_STRONG_SINGLE_POLICY_SUPPORTED_SCHEMA_VERSIONS = (1, 2)

SELECTION_REASON_ACCEPTED = np.int8(0)
SELECTION_REASON_MISSING = np.int8(1)
SELECTION_REASON_MULTIPLE = np.int8(2)
SELECTION_REASON_WEAK_OR_BAD_MATCH = np.int8(3)
SELECTION_REASON_OUTSIDE_TARGET_CROP = np.int8(4)

SELECTION_REASON_LABELS = {
    int(SELECTION_REASON_ACCEPTED): "accepted_strong_single",
    int(SELECTION_REASON_MISSING): "missing_detection",
    int(SELECTION_REASON_MULTIPLE): "multiple_candidates",
    int(SELECTION_REASON_WEAK_OR_BAD_MATCH): "weak_or_bad_match",
    int(SELECTION_REASON_OUTSIDE_TARGET_CROP): "outside_target_crop",
}


@dataclass(frozen=True)
class StrongSingleSelection:
    """Exact all-row receipt plus the accepted detector-row projection."""

    candidate_count: np.ndarray
    selected_detection_row_index: np.ndarray
    selected_score: np.ndarray
    proposal_iou: np.ndarray
    included: np.ndarray
    reason_code: np.ndarray
    accepted_crop_row_indices: np.ndarray
    accepted_detection_row_indices: np.ndarray

    @property
    def source_row_count(self) -> int:
        return int(self.included.shape[0])

    @property
    def accepted_row_count(self) -> int:
        return int(self.accepted_crop_row_indices.shape[0])

    def reason_counts(self) -> dict[str, int]:
        return {
            label: int(np.count_nonzero(self.reason_code == code))
            for code, label in SELECTION_REASON_LABELS.items()
        }


def strong_single_policy_record(
    *,
    minimum_score: float,
    minimum_proposal_iou: float,
    target_roi_size: tuple[int, int],
    policy_schema_version: int = (
        SAMPLED_TRAINING_STRONG_SINGLE_POLICY_CURRENT_SCHEMA_VERSION
    ),
) -> dict[str, Any]:
    """Return the canonical JSON-safe policy declaration."""

    score = float(minimum_score)
    iou = float(minimum_proposal_iou)
    height, width = (int(target_roi_size[0]), int(target_roi_size[1]))
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("minimum_score must be finite and in [0, 1].")
    if not np.isfinite(iou) or not 0.0 <= iou <= 1.0:
        raise ValueError("minimum_proposal_iou must be finite and in [0, 1].")
    if height <= 0 or width <= 0:
        raise ValueError("target_roi_size must contain positive height and width.")
    if (
        type(policy_schema_version) is not int
        or policy_schema_version
        not in SAMPLED_TRAINING_STRONG_SINGLE_POLICY_SUPPORTED_SCHEMA_VERSIONS
    ):
        raise ValueError("policy_schema_version is unsupported.")
    record = {
        "schema_id": SAMPLED_TRAINING_STRONG_SINGLE_POLICY_SCHEMA_ID,
        "schema_version": policy_schema_version,
        "candidate_cardinality": "exactly_one_before_quality_tests",
        "minimum_score_inclusive": score,
        "minimum_proposal_iou_inclusive": iou,
        "target_roi_size_height_width": [height, width],
        "target_crop_containment": "full_half_open_bbox_inclusive_edges",
        "multiple_candidate_policy": "exclude_entire_sampled_frame",
        "row_order": "source_crop_row_order",
        "reason_precedence": [
            "missing_detection",
            "multiple_candidates",
            "weak_or_bad_match",
            "outside_target_crop",
            "accepted_strong_single",
        ],
    }
    if policy_schema_version >= 2:
        record["target_crop_source_camera_containment"] = (
            "full_crop_window_required_no_padding"
        )
    return record


def normalized_cxcywh_to_image_xyxy(
    bbox_norm_coords: Any,
    *,
    source_width: int,
    source_height: int,
) -> np.ndarray:
    """Project normalized ``cx,cy,w,h`` while preserving floating dtype."""

    boxes = np.asarray(bbox_norm_coords)
    if boxes.dtype.kind != "f" or boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError("bbox_norm_coords must be a floating (N,4) array.")
    if type(source_width) is not int or type(source_height) is not int:
        raise TypeError("source dimensions must be exact integers.")
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive.")
    half = np.asarray(0.5, dtype=boxes.dtype)
    width_px = np.asarray(source_width, dtype=boxes.dtype)
    height_px = np.asarray(source_height, dtype=boxes.dtype)
    cx, cy, width, height = (boxes[:, index] for index in range(4))
    return np.ascontiguousarray(
        np.column_stack(
            (
                (cx - width * half) * width_px,
                (cy - height * half) * height_px,
                (cx + width * half) * width_px,
                (cy + height * half) * height_px,
            )
        ).astype(boxes.dtype, copy=False)
    )


def _bbox_iou(one: np.ndarray, two: np.ndarray) -> float:
    x1 = max(float(one[0]), float(two[0]))
    y1 = max(float(one[1]), float(two[1]))
    x2 = min(float(one[2]), float(two[2]))
    y2 = min(float(one[3]), float(two[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    one_area = max(0.0, float(one[2] - one[0])) * max(
        0.0, float(one[3] - one[1])
    )
    two_area = max(0.0, float(two[2] - two[0])) * max(
        0.0, float(two[3] - two[1])
    )
    union = one_area + two_area - intersection
    return intersection / union if union > 0.0 else 0.0


def select_strong_single_detections(
    *,
    crop_source_acquisition_frame_index: Any,
    proposal_bbox_img_xyxy: Any,
    target_roi_top_left_xy: Any,
    target_roi_size: tuple[int, int],
    detection_source_acquisition_frame_index: Any,
    detection_bbox_norm_coords: Any,
    detection_scores: Any,
    source_width: int,
    source_height: int,
    minimum_score: float = 0.40,
    minimum_proposal_iou: float = 0.50,
    policy_schema_version: int = (
        SAMPLED_TRAINING_STRONG_SINGLE_POLICY_CURRENT_SCHEMA_VERSION
    ),
) -> StrongSingleSelection:
    """Classify every crop row under the strict strong-single policy."""

    policy = strong_single_policy_record(
        minimum_score=minimum_score,
        minimum_proposal_iou=minimum_proposal_iou,
        target_roi_size=target_roi_size,
        policy_schema_version=policy_schema_version,
    )
    crop_frames = np.asarray(
        crop_source_acquisition_frame_index, dtype=np.int64
    ).reshape(-1)
    proposal = np.asarray(proposal_bbox_img_xyxy)
    top_left = np.asarray(target_roi_top_left_xy)
    detect_frames = np.asarray(
        detection_source_acquisition_frame_index, dtype=np.int64
    ).reshape(-1)
    detect_norm = np.asarray(detection_bbox_norm_coords)
    scores = np.asarray(detection_scores)
    crop_rows = int(crop_frames.shape[0])
    detection_rows = int(detect_frames.shape[0])
    if proposal.dtype.kind != "f" or proposal.shape != (crop_rows, 4):
        raise ValueError("proposal_bbox_img_xyxy must be floating and row-aligned.")
    if top_left.dtype.kind not in "iu" or top_left.shape != (crop_rows, 2):
        raise ValueError("target_roi_top_left_xy must be integral and row-aligned.")
    if detect_norm.dtype.kind != "f" or detect_norm.shape != (detection_rows, 4):
        raise ValueError("detection_bbox_norm_coords must be floating and row-aligned.")
    if scores.dtype.kind != "f" or scores.shape != (detection_rows,):
        raise ValueError("detection_scores must be floating and row-aligned.")
    if crop_rows != np.unique(crop_frames).shape[0]:
        raise ValueError("Sampled crop acquisition-frame indices must be unique.")
    if np.any(crop_frames < 0) or np.any(detect_frames < 0):
        raise ValueError("Acquisition-frame indices cannot be negative.")

    detect_img = normalized_cxcywh_to_image_xyxy(
        detect_norm,
        source_width=source_width,
        source_height=source_height,
    )
    candidates_by_frame: dict[int, list[int]] = {}
    for row_index, frame_index in enumerate(detect_frames):
        candidates_by_frame.setdefault(int(frame_index), []).append(int(row_index))

    counts = np.zeros(crop_rows, dtype=np.int32)
    selected = np.full(crop_rows, -1, dtype=np.int64)
    selected_score = np.full(crop_rows, np.nan, dtype=np.float32)
    proposal_iou = np.full(crop_rows, np.nan, dtype=np.float64)
    included = np.zeros(crop_rows, dtype=np.bool_)
    reasons = np.full(crop_rows, SELECTION_REASON_MISSING, dtype=np.int8)
    target_height, target_width = policy["target_roi_size_height_width"]

    for crop_row, frame_index in enumerate(crop_frames):
        candidates = candidates_by_frame.get(int(frame_index), [])
        counts[crop_row] = len(candidates)
        if not candidates:
            continue
        if len(candidates) != 1:
            reasons[crop_row] = SELECTION_REASON_MULTIPLE
            continue
        detection_row = int(candidates[0])
        selected[crop_row] = detection_row
        selected_score[crop_row] = np.float32(scores[detection_row])
        detection_box = detect_img[detection_row]
        iou = _bbox_iou(detection_box, proposal[crop_row])
        proposal_iou[crop_row] = iou
        norm_box = detect_norm[detection_row]
        valid_geometry = bool(
            np.isfinite(norm_box).all()
            and np.isfinite(detection_box).all()
            and float(norm_box[2]) > 0.0
            and float(norm_box[3]) > 0.0
            and float(detection_box[0]) >= 0.0
            and float(detection_box[1]) >= 0.0
            and float(detection_box[2]) <= float(source_width)
            and float(detection_box[3]) <= float(source_height)
            and float(detection_box[2]) > float(detection_box[0])
            and float(detection_box[3]) > float(detection_box[1])
        )
        if (
            not valid_geometry
            or not np.isfinite(scores[detection_row])
            or float(scores[detection_row]) < float(minimum_score)
            or iou < float(minimum_proposal_iou)
        ):
            reasons[crop_row] = SELECTION_REASON_WEAK_OR_BAD_MATCH
            continue
        x, y = (int(top_left[crop_row, 0]), int(top_left[crop_row, 1]))
        target_window_in_frame = (
            x >= 0
            and y >= 0
            and x + target_width <= source_width
            and y + target_height <= source_height
        )
        if not (
            (
                policy_schema_version < 2
                or target_window_in_frame
            )
            and float(detection_box[0]) >= float(x)
            and float(detection_box[1]) >= float(y)
            and float(detection_box[2]) <= float(x + target_width)
            and float(detection_box[3]) <= float(y + target_height)
        ):
            reasons[crop_row] = SELECTION_REASON_OUTSIDE_TARGET_CROP
            continue
        included[crop_row] = True
        reasons[crop_row] = SELECTION_REASON_ACCEPTED

    accepted_crop_rows = np.flatnonzero(included).astype(np.int64, copy=False)
    accepted_detection_rows = selected[accepted_crop_rows]
    if np.any(accepted_detection_rows < 0):  # defensive
        raise RuntimeError("Accepted rows lack an exact detector-row selection.")
    return StrongSingleSelection(
        candidate_count=np.ascontiguousarray(counts),
        selected_detection_row_index=np.ascontiguousarray(selected),
        selected_score=np.ascontiguousarray(selected_score),
        proposal_iou=np.ascontiguousarray(proposal_iou),
        included=np.ascontiguousarray(included),
        reason_code=np.ascontiguousarray(reasons),
        accepted_crop_row_indices=np.ascontiguousarray(accepted_crop_rows),
        accepted_detection_row_indices=np.ascontiguousarray(
            accepted_detection_rows
        ),
    )


__all__ = [
    "SAMPLED_TRAINING_STRONG_SINGLE_POLICY_CURRENT_SCHEMA_VERSION",
    "SAMPLED_TRAINING_STRONG_SINGLE_POLICY_SCHEMA_ID",
    "SAMPLED_TRAINING_STRONG_SINGLE_POLICY_SCHEMA_VERSION",
    "SAMPLED_TRAINING_STRONG_SINGLE_POLICY_SUPPORTED_SCHEMA_VERSIONS",
    "SELECTION_REASON_ACCEPTED",
    "SELECTION_REASON_LABELS",
    "SELECTION_REASON_MISSING",
    "SELECTION_REASON_MULTIPLE",
    "SELECTION_REASON_OUTSIDE_TARGET_CROP",
    "SELECTION_REASON_WEAK_OR_BAD_MATCH",
    "StrongSingleSelection",
    "normalized_cxcywh_to_image_xyxy",
    "select_strong_single_detections",
    "strong_single_policy_record",
]
