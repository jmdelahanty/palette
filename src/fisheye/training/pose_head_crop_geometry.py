"""Fixed sensor-pixel geometry for the versioned three-point pose-head crop.

The recipe intentionally matches the device crop kernel: truncate each box
centroid toward zero, subtract half the fixed window, and clamp to the frame.
It differs from Palette's ordinary rounded-center, padding-capable crops.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


POSE_HEAD_CROP_RECIPE_ID = "pose_head_fixed_192_truncated_centroid_v1"
POSE_HEAD_KEYPOINT_LABELS = ("swim_bladder", "eye_left", "eye_right")
POSE_HEAD_CROP_SIZE_PX = 192


def fixed_pose_head_origins(
    boxes_xyxy: np.ndarray,
    *,
    frame_shape_hw: tuple[int, int],
    crop_size_px: int = POSE_HEAD_CROP_SIZE_PX,
) -> np.ndarray:
    """Return integer `(x, y)` origins for full-frame `xyxy` boxes."""

    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape (rows, 4)")
    if not np.isfinite(boxes).all():
        raise ValueError("boxes_xyxy must be finite")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("boxes_xyxy must have positive width and height")
    height, width = frame_shape_hw
    if (
        type(height) is not int
        or type(width) is not int
        or type(crop_size_px) is not int
        or crop_size_px <= 0
        or height < crop_size_px
        or width < crop_size_px
        or crop_size_px % 2
    ):
        raise ValueError("frame must contain a positive, even-sized crop")
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    truncated_centers = np.trunc(centers).astype(np.int64)
    origins = truncated_centers - crop_size_px // 2
    return np.clip(origins, (0, 0), (width - crop_size_px, height - crop_size_px))


def project_pose_head_keypoints(
    points_img_xy: np.ndarray,
    *,
    source_labels: Sequence[str],
    origins_xy: np.ndarray,
    crop_size_px: int = POSE_HEAD_CROP_SIZE_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the canonical three labels and project to local crop pixels.

    Coordinates remain translated even for out-of-crop points; visibility 0
    tells consumers that those positions must not train a keypoint target.
    """

    points = np.asarray(points_img_xy)
    origins = np.asarray(origins_xy)
    if points.ndim != 3 or points.shape[2] != 2:
        raise ValueError("points_img_xy must have shape (rows, keypoints, 2)")
    if origins.shape != (len(points), 2) or not np.issubdtype(
        origins.dtype, np.integer
    ):
        raise ValueError("origins_xy must be integer (rows, 2)")
    labels = tuple(str(label) for label in source_labels)
    if len(labels) != points.shape[1]:
        raise ValueError("source_labels must match the keypoint axis")
    counts = Counter(labels)
    for label in POSE_HEAD_KEYPOINT_LABELS:
        if counts[label] == 0:
            raise ValueError(f"source skeleton is missing {label}")
        if counts[label] != 1:
            raise ValueError(f"source skeleton has duplicate {label}")
    if type(crop_size_px) is not int or crop_size_px <= 0:
        raise ValueError("crop_size_px must be a positive integer")
    indices = [labels.index(label) for label in POSE_HEAD_KEYPOINT_LABELS]
    selected = points[:, indices, :].astype(np.float64, copy=False)
    coordinates = selected - origins[:, None, :]
    inside = np.isfinite(coordinates).all(axis=2)
    inside &= np.logical_and(coordinates >= 0, coordinates < crop_size_px).all(axis=2)
    visibility = np.where(inside, 2, 0).astype(np.uint8)
    return coordinates, visibility
