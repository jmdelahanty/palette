"""Shared ROI background and preview helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import zarr


@dataclass(frozen=True)
class KeypointSource:
    group_name: str
    run_name: str
    success_name: Optional[str]


@dataclass(frozen=True)
class BackgroundSource:
    run_name: str
    array_name: str
    full_shape: Optional[Tuple[int, int]]


def _group_keys(group: zarr.Group) -> List[str]:
    keys_fn = getattr(group, "group_keys", None)
    if callable(keys_fn):
        return sorted(str(name) for name in keys_fn())
    names: List[str] = []
    for key in group.keys():
        try:
            obj = group[key]
        except Exception:
            continue
        if isinstance(obj, zarr.Group):
            names.append(str(key))
    return sorted(names)


def _resolve_run_name(root: zarr.Group, parent_name: str, explicit: Optional[str]) -> str:
    parent = root.get(parent_name)
    if parent is None:
        raise ValueError(f"Missing '{parent_name}' group.")
    if explicit:
        if explicit not in parent:
            raise ValueError(f"Run '{explicit}' not found under {parent_name}.")
        return str(explicit)
    latest = parent.attrs.get("latest")
    if isinstance(latest, str) and latest in parent:
        return latest
    names = _group_keys(parent)
    if not names:
        raise ValueError(f"No runs found under {parent_name}.")
    return names[-1]


def _resolve_success_dataset_name(group: zarr.Group) -> Optional[str]:
    for candidate in ("detection_success", "refined_success", "source_success"):
        if candidate in group:
            return candidate
    return None


def _resolve_keypoint_source(root: zarr.Group, explicit: Optional[str]) -> Optional[KeypointSource]:
    refined = root.get("refined_keypoints_runs")
    raw = root.get("keypoints_runs")

    if explicit:
        if refined is not None and explicit in refined:
            group = refined[explicit]
            success_name = _resolve_success_dataset_name(group)
            return KeypointSource("refined_keypoints_runs", str(explicit), success_name)
        if raw is not None and explicit in raw:
            group = raw[explicit]
            success_name = _resolve_success_dataset_name(group)
            return KeypointSource("keypoints_runs", str(explicit), success_name)
        raise ValueError(
            f"Keypoint run '{explicit}' not found in refined_keypoints_runs or keypoints_runs."
        )

    refined_latest = refined.attrs.get("latest") if refined is not None else None
    if (
        refined is not None
        and isinstance(refined_latest, str)
        and refined_latest in refined
    ):
        group = refined[refined_latest]
        success_name = _resolve_success_dataset_name(group)
        return KeypointSource("refined_keypoints_runs", str(refined_latest), success_name)

    raw_latest = raw.attrs.get("latest") if raw is not None else None
    if raw is not None and isinstance(raw_latest, str) and raw_latest in raw:
        group = raw[raw_latest]
        success_name = _resolve_success_dataset_name(group)
        return KeypointSource("keypoints_runs", str(raw_latest), success_name)

    return None


def _resolve_background_source(root: zarr.Group, run_name: str) -> BackgroundSource:
    bg_parent = root.get("background_runs")
    if bg_parent is None:
        raise ValueError("Missing background_runs group.")
    if run_name not in bg_parent:
        raise ValueError(f"Background run '{run_name}' not found.")

    bg_group = bg_parent[run_name]
    if "background_full" in bg_group:
        return BackgroundSource(run_name=str(run_name), array_name="background_full", full_shape=None)
    if "background_ds" not in bg_group:
        raise ValueError(f"Background run '{run_name}' has neither background_full nor background_ds.")

    full_shape: Optional[Tuple[int, int]] = None
    raw_group = root.get("raw_video")
    images_full = raw_group.get("images_full") if raw_group is not None else None
    if images_full is not None and len(images_full.shape) >= 3:
        full_shape = (int(images_full.shape[1]), int(images_full.shape[2]))
    if full_shape is None:
        raise ValueError(
            "background_full is unavailable and full image shape could not be inferred "
            "(expected raw_video/images_full)."
        )

    return BackgroundSource(run_name=str(run_name), array_name="background_ds", full_shape=full_shape)


def _extract_background_roi_full(
    background_full: np.ndarray,
    top_left_xy: Sequence[float | int],
    roi_shape: Tuple[int, int],
) -> np.ndarray:
    roi_h, roi_w = int(roi_shape[0]), int(roi_shape[1])
    out = np.zeros((roi_h, roi_w), dtype=background_full.dtype)
    if roi_h <= 0 or roi_w <= 0:
        return out

    x = int(round(float(top_left_xy[0])))
    y = int(round(float(top_left_xy[1])))

    src_x0 = max(0, x)
    src_y0 = max(0, y)
    src_x1 = min(int(background_full.shape[1]), x + roi_w)
    src_y1 = min(int(background_full.shape[0]), y + roi_h)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = src_x0 - x
    dst_y0 = src_y0 - y
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = background_full[src_y0:src_y1, src_x0:src_x1]
    return out


def _extract_background_roi_ds(
    background_ds: np.ndarray,
    top_left_xy: Sequence[float | int],
    roi_shape: Tuple[int, int],
    full_shape: Tuple[int, int],
) -> np.ndarray:
    roi_h, roi_w = int(roi_shape[0]), int(roi_shape[1])
    out = np.zeros((roi_h, roi_w), dtype=background_ds.dtype)
    if roi_h <= 0 or roi_w <= 0:
        return out

    full_h, full_w = int(full_shape[0]), int(full_shape[1])
    if full_h <= 0 or full_w <= 0:
        return out

    ds_h, ds_w = int(background_ds.shape[0]), int(background_ds.shape[1])
    scale_x = float(ds_w) / float(full_w)
    scale_y = float(ds_h) / float(full_h)

    x = int(round(float(top_left_xy[0])))
    y = int(round(float(top_left_xy[1])))

    x0_ds = int(np.floor(x * scale_x))
    y0_ds = int(np.floor(y * scale_y))
    x1_ds = int(np.ceil((x + roi_w) * scale_x))
    y1_ds = int(np.ceil((y + roi_h) * scale_y))

    src_x0 = max(0, x0_ds)
    src_y0 = max(0, y0_ds)
    src_x1 = min(ds_w, x1_ds)
    src_y1 = min(ds_h, y1_ds)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    patch = np.asarray(background_ds[src_y0:src_y1, src_x0:src_x1])
    resized = cv2.resize(patch, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    if resized.dtype != out.dtype:
        resized = resized.astype(out.dtype, copy=False)
    return resized


def _extract_patch(
    image: np.ndarray,
    center_xy: Sequence[float | int],
    half_width: int,
) -> np.ndarray:
    half = max(0, int(half_width))
    size = half * 2 + 1
    patch = np.zeros((size, size), dtype=image.dtype)

    cx = int(round(float(center_xy[0])))
    cy = int(round(float(center_xy[1])))

    x0 = cx - half
    y0 = cy - half
    x1 = x0 + size
    y1 = y0 + size

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(int(image.shape[1]), x1)
    src_y1 = min(int(image.shape[0]), y1)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return patch

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    patch[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return patch


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _prepare_panel(image: np.ndarray, label: str, panel_size: int, *, nearest: bool = False) -> np.ndarray:
    panel = _to_bgr(image)
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    panel = cv2.resize(panel, (panel_size, panel_size), interpolation=interpolation)
    cv2.rectangle(panel, (0, 0), (panel_size - 1, 24), (0, 0, 0), -1)
    cv2.putText(panel, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
    return panel
