from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr

from .type_conversions import as_int, normalize_attr
from .zarr_run_completion import resolve_latest_complete_run_name

DEFAULT_PREFERRED_CROP_POLICY_NAME = "centered_fixed_size_roi_v1"
DEFAULT_PREFERRED_ROI_SIZE = (512, 512)  # (height, width)


def bbox_norm_cxcywh_to_img_xyxy(
    bbox_norm_coords: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert normalized [cx, cy, w, h] boxes to full-image [x0, y0, x1, y1]."""
    bbox = np.asarray(bbox_norm_coords, dtype=np.float64).reshape(-1, 4)
    clipped = np.clip(bbox, 0.0, 1.0)
    cx = clipped[:, 0] * float(width)
    cy = clipped[:, 1] * float(height)
    bw = clipped[:, 2] * float(width)
    bh = clipped[:, 3] * float(height)
    return np.stack(
        (
            cx - bw * 0.5,
            cy - bh * 0.5,
            cx + bw * 0.5,
            cy + bh * 0.5,
        ),
        axis=1,
    ).astype(np.float64, copy=False)


def bbox_img_xyxy_to_norm_cxcywh(
    bbox_img_xyxy: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert full-image [x0, y0, x1, y1] boxes to normalized [cx, cy, w, h]."""
    bbox = np.asarray(bbox_img_xyxy, dtype=np.float64).reshape(-1, 4)
    x0 = bbox[:, 0]
    y0 = bbox[:, 1]
    x1 = bbox[:, 2]
    y1 = bbox[:, 3]
    cx = ((x0 + x1) * 0.5) / float(width)
    cy = ((y0 + y1) * 0.5) / float(height)
    bw = (x1 - x0) / float(width)
    bh = (y1 - y0) / float(height)
    return np.stack((cx, cy, bw, bh), axis=1).astype(np.float64, copy=False)


def compute_centered_roi_mapping(
    bbox_img_xyxy: np.ndarray,
    *,
    roi_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return translation-only ROI mapping compatible with current crop centering."""
    bbox = np.asarray(bbox_img_xyxy, dtype=np.float64).reshape(-1, 4)
    roi_h, roi_w = int(roi_size[0]), int(roi_size[1])
    centers = np.round(
        np.stack(
            (
                (bbox[:, 0] + bbox[:, 2]) * 0.5,
                (bbox[:, 1] + bbox[:, 3]) * 0.5,
            ),
            axis=1,
        )
    ).astype(np.int32, copy=False)
    offsets = np.empty((bbox.shape[0], 2), dtype=np.int32)
    offsets[:, 0] = centers[:, 0] - (roi_w // 2)
    offsets[:, 1] = centers[:, 1] - (roi_h // 2)
    sizes = np.tile(np.asarray([[roi_w, roi_h]], dtype=np.int32), (bbox.shape[0], 1))
    return offsets, sizes


def compute_roi_bbox_img_xyxy(
    roi_offset_xy_full: np.ndarray,
    roi_size_wh: np.ndarray,
) -> np.ndarray:
    offsets = np.asarray(roi_offset_xy_full, dtype=np.int32).reshape(-1, 2)
    sizes = np.asarray(roi_size_wh, dtype=np.int32).reshape(-1, 2)
    x0 = offsets[:, 0].astype(np.float64)
    y0 = offsets[:, 1].astype(np.float64)
    x1 = x0 + sizes[:, 0].astype(np.float64)
    y1 = y0 + sizes[:, 1].astype(np.float64)
    return np.stack((x0, y0, x1, y1), axis=1)


def normalize_roi_size(value: Optional[Sequence[int]]) -> Tuple[int, int]:
    if value is None:
        return DEFAULT_PREFERRED_ROI_SIZE
    if len(value) != 2:
        raise ValueError("roi_size must contain exactly 2 integers: [height, width].")
    roi_h = int(value[0])
    roi_w = int(value[1])
    if roi_h <= 0 or roi_w <= 0:
        raise ValueError("roi_size dimensions must be positive integers.")
    return roi_h, roi_w


def infer_preferred_roi_size(root: zarr.Group) -> Tuple[int, int]:
    crop_parent = root.get("crop_runs")
    if crop_parent is not None:
        latest = normalize_attr(resolve_latest_complete_run_name(crop_parent, legacy_default=True))
        if latest and latest in crop_parent:
            latest_group = crop_parent[latest]
            roi_size_attr = latest_group.attrs.get("roi_size")
            if isinstance(roi_size_attr, (list, tuple)) and len(roi_size_attr) == 2:
                return normalize_roi_size([int(roi_size_attr[0]), int(roi_size_attr[1])])

    pipeline_params = root.get("pipeline_params")
    if pipeline_params is not None:
        crop_params = pipeline_params.attrs.get("crop")
        if isinstance(crop_params, Mapping):
            roi_sz = crop_params.get("roi_sz")
            if isinstance(roi_sz, (list, tuple)) and len(roi_sz) == 2:
                return normalize_roi_size([int(roi_sz[0]), int(roi_sz[1])])

    width = as_int(root.attrs.get("width"))
    height = as_int(root.attrs.get("height"))
    if width is not None and height is not None and width > 0 and height > 0:
        return normalize_roi_size([min(DEFAULT_PREFERRED_ROI_SIZE[0], height), min(DEFAULT_PREFERRED_ROI_SIZE[1], width)])
    return DEFAULT_PREFERRED_ROI_SIZE


def build_crop_policy_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
