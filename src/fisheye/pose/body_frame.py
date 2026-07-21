"""Helpers for fish-relative body-frame geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from fisheye.shared.detect_reason_codec import encode_reason_bytes

BODY_FRAME_SCHEMA_ID = "fish_anatomical_body_frame"
BODY_FRAME_SCHEMA_VERSION = 1
BODY_FRAME_ESTIMATOR_KEYPOINT_HEAD_AXIS = "keypoint_head_axis"
BODY_FRAME_ANGLE_CONVENTION = "math_ccw_degrees_after_y_flip"
BODY_FRAME_COORDINATE_SPACE_ROI = "roi_pixels"

BODY_FRAME_REASON_OK = "ok"
BODY_FRAME_REASON_DETECTION_FAILURE = "detection_failure"
BODY_FRAME_REASON_MISSING_SOURCE_ANCHOR = "missing_source_anchor"
BODY_FRAME_REASON_DEGENERATE_FORWARD_AXIS = "degenerate_forward_axis"
BODY_FRAME_REASON_LEFT_RIGHT_UNRESOLVED = "left_right_unresolved"

_EPS = 1.0e-9


@dataclass(frozen=True)
class BodyFrameResult:
    """Row-aligned fish body-frame arrays."""

    origin_xy: np.ndarray
    forward_axis_xy: np.ndarray
    left_axis_xy: np.ndarray
    heading_deg: np.ndarray
    valid: np.ndarray
    failure_reason_bytes: np.ndarray


def _unit_or_nan(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(norms) & (norms > _EPS)
    out = np.full(vectors.shape, np.nan, dtype=np.float32)
    out[valid] = (vectors[valid] / norms[valid, None]).astype(np.float32, copy=False)
    return out, valid


def compute_keypoint_body_frame(
    keypoints_xy: np.ndarray,
    *,
    keypoint_indices: Mapping[str, int],
    detection_success: Optional[np.ndarray] = None,
) -> BodyFrameResult:
    """Compute a body frame from swim-bladder and eye keypoints.

    The estimator is the current keypoint-only fallback:
    ``swim_bladder -> midpoint(eye_left, eye_right)`` defines the forward axis,
    while the labeled eye pair resolves anatomical left.
    """

    points = np.asarray(keypoints_xy, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] < 2:
        raise ValueError(f"keypoints_xy must have shape (N, K, >=2); got {points.shape}.")

    row_count = int(points.shape[0])
    origin_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    forward_axis_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    left_axis_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    heading_deg = np.full(row_count, np.nan, dtype=np.float32)
    valid = np.zeros(row_count, dtype=bool)
    reasons = np.full(row_count, BODY_FRAME_REASON_OK, dtype=object)

    success = np.ones(row_count, dtype=bool)
    if detection_success is not None:
        success = np.asarray(detection_success, dtype=bool).reshape(row_count)

    try:
        bladder = points[:, int(keypoint_indices["swim_bladder"]), :2]
        eye_left = points[:, int(keypoint_indices["eye_left"]), :2]
        eye_right = points[:, int(keypoint_indices["eye_right"]), :2]
    except KeyError as exc:
        raise ValueError(f"Missing required body-frame keypoint index: {exc}") from exc

    anchor_finite = (
        np.all(np.isfinite(bladder), axis=1)
        & np.all(np.isfinite(eye_left), axis=1)
        & np.all(np.isfinite(eye_right), axis=1)
    )

    origin = 0.5 * (eye_left + eye_right)
    forward_raw = origin - bladder
    lateral_raw = eye_left - eye_right
    forward_unit, forward_ok = _unit_or_nan(forward_raw)
    lateral_unit, lateral_ok = _unit_or_nan(lateral_raw)

    candidate_left = np.empty_like(forward_unit)
    candidate_left[:, 0] = forward_unit[:, 1]
    candidate_left[:, 1] = -forward_unit[:, 0]

    orientable = anchor_finite & forward_ok & lateral_ok
    if np.any(orientable):
        dot_left = np.einsum("ij,ij->i", candidate_left[orientable], lateral_unit[orientable])
        signs = np.where(dot_left >= 0.0, 1.0, -1.0).astype(np.float32)
        left_axis_xy[orientable] = candidate_left[orientable] * signs[:, None]
        forward_axis_xy[orientable] = forward_unit[orientable]
        origin_xy[orientable] = origin[orientable].astype(np.float32, copy=False)
        heading_deg[orientable] = np.rad2deg(
            np.arctan2(-forward_unit[orientable, 1], forward_unit[orientable, 0])
        ).astype(np.float32, copy=False)

    valid = success & orientable
    reasons[~success] = BODY_FRAME_REASON_DETECTION_FAILURE
    reasons[success & ~anchor_finite] = BODY_FRAME_REASON_MISSING_SOURCE_ANCHOR
    reasons[success & anchor_finite & ~forward_ok] = BODY_FRAME_REASON_DEGENERATE_FORWARD_AXIS
    reasons[success & anchor_finite & forward_ok & ~lateral_ok] = BODY_FRAME_REASON_LEFT_RIGHT_UNRESOLVED

    origin_xy[~valid] = np.nan
    forward_axis_xy[~valid] = np.nan
    left_axis_xy[~valid] = np.nan
    heading_deg[~valid] = np.nan

    return BodyFrameResult(
        origin_xy=origin_xy,
        forward_axis_xy=forward_axis_xy,
        left_axis_xy=left_axis_xy,
        heading_deg=heading_deg,
        valid=valid,
        failure_reason_bytes=encode_reason_bytes(reasons),
    )


def signed_angle_in_body_frame_deg(vectors_xy: np.ndarray, body_frame: BodyFrameResult) -> np.ndarray:
    """Return signed fish-frame angles for row-aligned vectors.

    Positive values point toward anatomical left. Invalid body-frame rows or
    non-finite vectors produce NaN.
    """

    vectors = np.asarray(vectors_xy, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] < 2:
        raise ValueError(f"vectors_xy must have shape (N, >=2); got {vectors.shape}.")
    if vectors.shape[0] != body_frame.valid.shape[0]:
        raise ValueError("vectors_xy and body_frame row counts differ.")

    out = np.full(vectors.shape[0], np.nan, dtype=np.float32)
    finite = body_frame.valid & np.all(np.isfinite(vectors[:, :2]), axis=1)
    if not np.any(finite):
        return out

    forward = np.einsum("ij,ij->i", vectors[finite, :2], body_frame.forward_axis_xy[finite])
    left = np.einsum("ij,ij->i", vectors[finite, :2], body_frame.left_axis_xy[finite])
    out[finite] = np.rad2deg(np.arctan2(left, forward)).astype(np.float32, copy=False)
    return out


def build_keypoint_body_frame_contract_attrs(
    *,
    source_keypoints_run: Optional[str] = None,
    source_refined_keypoints_run: Optional[str] = None,
    coordinate_space: str = BODY_FRAME_COORDINATE_SPACE_ROI,
) -> dict[str, object]:
    """Return stable attrs for a keypoint-derived body-frame estimator."""

    if source_keypoints_run and source_refined_keypoints_run:
        raise ValueError(
            "A keypoint body frame cannot name both canonical base and refined sources."
        )
    source_refs: dict[str, object] = {}
    if source_keypoints_run:
        source_refs["keypoints_run"] = source_keypoints_run
        source_refs["keypoints_path"] = f"keypoints_runs/{source_keypoints_run}"
    if source_refined_keypoints_run:
        source_refs["refined_keypoints_run"] = source_refined_keypoints_run
        source_refs["refined_keypoints_path"] = f"refined_keypoints_runs/{source_refined_keypoints_run}"

    return {
        "body_frame_schema_id": BODY_FRAME_SCHEMA_ID,
        "body_frame_schema_version": BODY_FRAME_SCHEMA_VERSION,
        "body_frame_estimator": BODY_FRAME_ESTIMATOR_KEYPOINT_HEAD_AXIS,
        "body_frame_estimator_version": 1,
        "body_frame_coordinate_space": coordinate_space,
        "body_frame_angle_convention": BODY_FRAME_ANGLE_CONVENTION,
        "body_frame_source_refs": source_refs,
    }
