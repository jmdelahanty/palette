#!/usr/bin/env python3
"""Interactive swim-bladder patch viewer/editor for refined subject masks."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import cv2
import numpy as np
import zarr

from ..shared.crop_image_source import CropImageSource
from ..shared.mask_store import MaskStoreError, open_mask_store
from ..shared.provenance_attrs import resolve_source_keypoints_run
from ..tune.refined_subject_mask_review import (
    DEFAULT_REVIEW_INTENDED_USE,
    DEFAULT_REVIEW_METHOD,
    _load_refined_component_source_runs,
    apply_component_review_status,
    prepare_refined_subject_run,
    save_refined_subject_roi,
)
from ..shared.zarr_io import open_zarr_root

try:
    cv2_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "2")))
except (TypeError, ValueError):
    cv2_threads = 2
cv2.setNumThreads(cv2_threads)

WINDOW_NAME = "Swim Bladder Mask Patch Viewer"
CONTROL_WINDOW_NAME = "Swim Bladder Mask Patch Controls"
DEFAULT_PADDING = 18
MAX_PADDING = 128
DEFAULT_SCALE_PERCENT = 220
MAX_SCALE_PERCENT = 500
DEFAULT_EDIT_ZOOM = 8
MAX_EDIT_ZOOM = 24
DEFAULT_BRUSH = 4
BRUSH_MAX = 40
PANEL_LABEL_HEIGHT = 18
MASK_COLOR = (255, 220, 0)
KEYPOINT_COLOR = (255, 0, 255)
CENTER_COLOR = (0, 255, 0)


def _debug_json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_debug_json_value(item) for item in value]
    if isinstance(value, list):
        return [_debug_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _debug_json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class _UiDebugLog:
    def __init__(self, path: Path, *, zarr_path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)
        self._session_id = f"{os.getpid()}-{time.time_ns()}"
        self._seq = 0
        self._zarr_path = str(zarr_path)

    def emit(self, event: str, /, **fields: object) -> None:
        payload = {
            "session_id": self._session_id,
            "seq": self._seq,
            "event": str(event),
            "pid": int(os.getpid()),
            "t_monotonic_ns": int(time.monotonic_ns()),
            "zarr_path": self._zarr_path,
        }
        payload.update({str(key): _debug_json_value(value) for key, value in fields.items()})
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._seq += 1

    def close(self) -> None:
        self._fh.close()


def _open_ui_debug_log(path: Optional[Path], *, zarr_path: Path) -> Optional[_UiDebugLog]:
    if path is None:
        return None
    try:
        return _UiDebugLog(path, zarr_path=zarr_path)
    except Exception as exc:
        print(f"Warning: failed to open UI debug log {path}: {exc}")
        return None


def _require_gui_display() -> None:
    display = str(os.environ.get("DISPLAY") or "").strip()
    wayland_display = str(os.environ.get("WAYLAND_DISPLAY") or "").strip()
    if display or wayland_display:
        return
    raise RuntimeError(
        "No GUI display detected for OpenCV review window. DISPLAY and WAYLAND_DISPLAY are unset."
    )


def _mouse_modifier_state(flags: int) -> Tuple[bool, bool, bool]:
    ctrl_down = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
    shift_down = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
    left_down = bool(flags & cv2.EVENT_FLAG_LBUTTON)
    return ctrl_down, shift_down, left_down


def _resolve_erase_mode(base_erase_mode: bool, shift_down: bool) -> bool:
    # Shift acts as a temporary inverse while drawing.
    return (not base_erase_mode) if shift_down else base_erase_mode


def _extract_patch_bounds(
    roi_shape: Tuple[int, int],
    center_xy: Tuple[float, float],
    padding: int,
) -> Tuple[int, int, int, int]:
    roi_h, roi_w = int(roi_shape[0]), int(roi_shape[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    pad = max(1, int(padding))

    x0 = max(0, int(round(cx)) - pad)
    x1 = min(roi_w, int(round(cx)) + pad + 1)
    y0 = max(0, int(round(cy)) - pad)
    y1 = min(roi_h, int(round(cy)) + pad + 1)
    return x0, x1, y0, y1


def _mask_centroid_xy(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=np.uint8) > 0)
    if ys.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _normalize_label(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _resolve_swim_bladder_keypoint_index(keypoint_labels: Optional[Sequence[str]], keypoints_row: Optional[np.ndarray]) -> Optional[int]:
    if keypoint_labels:
        for idx, label in enumerate(keypoint_labels):
            if _normalize_label(label) == "swim_bladder":
                return int(idx)
    if keypoints_row is None:
        return None
    if keypoints_row.ndim != 2 or int(keypoints_row.shape[0]) <= 0:
        return None
    return 0


def _resolve_swim_bladder_center_with_source(
    keypoints_row: Optional[np.ndarray],
    keypoint_labels: Optional[Sequence[str]],
    mask_row: np.ndarray,
    roi_shape: Tuple[int, int],
) -> Tuple[Tuple[float, float], str]:
    kp_idx = _resolve_swim_bladder_keypoint_index(keypoint_labels, keypoints_row)
    if kp_idx is not None and keypoints_row is not None and keypoints_row.ndim == 2 and kp_idx < int(keypoints_row.shape[0]):
        kp = np.asarray(keypoints_row[kp_idx], dtype=np.float32)
        if kp.shape[0] >= 2 and np.all(np.isfinite(kp[:2])):
            return (float(kp[0]), float(kp[1])), "keypoint"

    centroid = _mask_centroid_xy(mask_row)
    if centroid is not None:
        return centroid, "mask_centroid"

    roi_h, roi_w = int(roi_shape[0]), int(roi_shape[1])
    return (float(roi_w) / 2.0, float(roi_h) / 2.0), "roi_center"


def _resolve_swim_bladder_keypoint_center(
    keypoints_row: Optional[np.ndarray],
    keypoint_labels: Optional[Sequence[str]],
    *,
    success_flag: Optional[bool],
) -> Tuple[Optional[Tuple[float, float]], str]:
    if success_flag is False:
        return None, "unsuccessful_keypoint"
    kp_idx = _resolve_swim_bladder_keypoint_index(keypoint_labels, keypoints_row)
    if kp_idx is not None and keypoints_row is not None and keypoints_row.ndim == 2 and kp_idx < int(keypoints_row.shape[0]):
        kp = np.asarray(keypoints_row[kp_idx], dtype=np.float32)
        if kp.shape[0] >= 2 and np.all(np.isfinite(kp[:2])):
            return (float(kp[0]), float(kp[1])), "keypoint"
    return None, "missing_keypoint"


def _source_component_mask_row(
    source: object,
    component_name: str,
    roi_idx: int,
    *,
    fallback_shape: tuple[int, int],
) -> np.ndarray:
    mask_labels = tuple(str(label) for label in getattr(source, "mask_labels", ()) or ())
    available_channels = np.asarray(getattr(source, "available_channels", ()), dtype=bool)
    fallback = np.zeros((int(fallback_shape[0]), int(fallback_shape[1])), dtype=np.uint8)
    if component_name not in mask_labels:
        return fallback
    component_idx = int(mask_labels.index(component_name))
    if component_idx >= int(available_channels.shape[0]) or not bool(available_channels[component_idx]):
        return fallback

    masks_roi = getattr(source, "masks_roi", None)
    if masks_roi is None:
        group = getattr(source, "group", None)
        if group is not None and "masks_roi" in group:
            masks_roi = group["masks_roi"]
        elif group is not None:
            try:
                return np.asarray(
                    open_mask_store(group, prefer="dense").read_dense(
                        rows=int(roi_idx),
                        channels=component_idx,
                    )[0, 0],
                    dtype=np.uint8,
                )
            except (MaskStoreError, ValueError, KeyError):
                return fallback
    if masks_roi is None:
        return fallback
    return np.asarray(masks_roi[int(roi_idx), component_idx], dtype=np.uint8)


def _parse_roi_indices_arg(raw: str) -> list[int]:
    text = str(raw).strip()
    if not text:
        raise argparse.ArgumentTypeError("ROI index list must not be empty.")
    values: list[int] = []
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid ROI index {item!r}.") from exc
    if not values:
        raise argparse.ArgumentTypeError("ROI index list must include at least one integer.")
    return values


def _normalize_visible_roi_indices(
    roi_indices: Optional[Sequence[int]],
    *,
    total_rois: int,
) -> list[int]:
    if roi_indices is None:
        return list(range(int(total_rois)))

    seen: set[int] = set()
    normalized: list[int] = []
    for value in roi_indices:
        roi_idx = int(value)
        if roi_idx < 0 or roi_idx >= int(total_rois):
            raise RuntimeError(
                f"ROI index {roi_idx} is out of range for crop ROI rows 0..{int(total_rois) - 1}."
            )
        if roi_idx in seen:
            continue
        seen.add(roi_idx)
        normalized.append(roi_idx)
    if not normalized:
        raise RuntimeError("No ROI indices remain after normalization.")
    return normalized


def _clamp_queue_pos(queue_pos: int, *, queue_size: int) -> int:
    if queue_size <= 0:
        raise RuntimeError("ROI queue is empty.")
    return max(0, min(int(queue_pos), int(queue_size) - 1))


def _labeled_panel(image_bgr: np.ndarray, label: str) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    canvas = np.zeros((h + PANEL_LABEL_HEIGHT, w, 3), dtype=np.uint8)
    canvas[PANEL_LABEL_HEIGHT:, :] = image_bgr
    cv2.putText(
        canvas,
        label,
        (4, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 255, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _stack_h(panels: Sequence[np.ndarray], gap: int = 6) -> np.ndarray:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h = max(panel.shape[0] for panel in panels)
    padded: List[np.ndarray] = []
    for panel in panels:
        pad_bottom = max(0, h - panel.shape[0])
        if pad_bottom:
            panel = np.pad(panel, ((0, pad_bottom), (0, 0), (0, 0)), mode="constant")
        padded.append(panel)
    if gap <= 0:
        return np.hstack(padded)
    spacer = np.zeros((h, gap, 3), dtype=np.uint8)
    out = padded[0]
    for panel in padded[1:]:
        out = np.hstack([out, spacer, panel])
    return out


def _stack_v(panels: Sequence[np.ndarray], gap: int = 8) -> np.ndarray:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    w = max(panel.shape[1] for panel in panels)
    padded: List[np.ndarray] = []
    for panel in panels:
        pad_right = max(0, w - panel.shape[1])
        if pad_right:
            panel = np.pad(panel, ((0, 0), (0, pad_right), (0, 0)), mode="constant")
        padded.append(panel)
    if gap <= 0:
        return np.vstack(padded)
    spacer = np.zeros((gap, w, 3), dtype=np.uint8)
    out = padded[0]
    for panel in padded[1:]:
        out = np.vstack([out, spacer, panel])
    return out


def _is_group_like(value: object) -> bool:
    return hasattr(value, "attrs") and hasattr(value, "get")


def _iter_group_names(group: Any) -> List[str]:
    if hasattr(group, "group_keys"):
        return [str(name) for name in group.group_keys()]
    return [str(name) for name in group.keys()]


def _candidate_run_names(parent: Any) -> List[str]:
    names: List[str] = []
    latest = parent.attrs.get("latest") if hasattr(parent, "attrs") else None
    if isinstance(latest, str) and latest in parent:
        names.append(latest)
    for name in sorted(_iter_group_names(parent), reverse=True):
        if name not in names:
            names.append(name)
    return names


def _required_array_equal(name: str, left: Any, right: Any) -> None:
    left_arr = np.asarray(left[:])
    right_arr = np.asarray(right[:])
    if left_arr.shape != right_arr.shape or not np.array_equal(left_arr, right_arr):
        raise RuntimeError(f"Alignment mismatch for {name}.")


def _optional_array_equal(name: str, left: Any | None, right: Any | None) -> bool:
    if left is None and right is None:
        return False
    if left is None or right is None:
        raise RuntimeError(f"Alignment mismatch for optional {name}: one source is missing it.")
    _required_array_equal(name, left, right)
    return True


def _validate_keypoint_group_alignment(
    crop_group: Any,
    crop_run: str,
    keypoint_group: Any,
    *,
    total_rois: int,
) -> None:
    if "keypoints_roi" not in keypoint_group:
        raise RuntimeError("Resolved keypoint run is missing keypoints_roi.")
    if int(keypoint_group["keypoints_roi"].shape[0]) != int(total_rois):
        raise RuntimeError(
            f"Keypoint rows {int(keypoint_group['keypoints_roi'].shape[0])} "
            f"do not match crop ROI rows {int(total_rois)}."
        )

    source_crop_run = keypoint_group.attrs.get("source_crop_run") if hasattr(keypoint_group, "attrs") else None
    if source_crop_run is not None and str(source_crop_run) != str(crop_run):
        raise RuntimeError(
            "Resolved keypoint run is not aligned to the selected crop run: "
            f"source_crop_run={source_crop_run!r}, crop_run={crop_run!r}."
        )

    validated = False
    validated |= _optional_array_equal(
        "frame_indices",
        crop_group.get("frame_indices"),
        keypoint_group.get("frame_indices"),
    )
    validated |= _optional_array_equal(
        "detection_indices",
        crop_group.get("detection_indices"),
        keypoint_group.get("detection_indices"),
    )
    if source_crop_run is None and not validated:
        raise RuntimeError(
            "Resolved keypoint run is missing source_crop_run and alignment arrays; "
            "cannot verify crop/keypoint alignment."
        )


def _resolve_keypoint_group(
    root: zarr.Group,
    *,
    subject_group: Optional[zarr.Group],
    refined_group: Optional[zarr.Group],
    explicit_run: Optional[str],
    explicit_group: Optional[str],
    expected_crop_run: Optional[str] = None,
) -> Tuple[Optional[zarr.Group], Optional[str], Optional[str]]:
    if explicit_group and explicit_run:
        parent = root.get(explicit_group)
        if _is_group_like(parent) and explicit_run in parent:
            return parent[explicit_run], explicit_group, explicit_run
        return None, None, None

    if explicit_run and not explicit_group:
        for group_name in ("refined_keypoints_runs", "keypoints_runs"):
            parent = root.get(group_name)
            if _is_group_like(parent) and explicit_run in parent:
                return parent[explicit_run], group_name, explicit_run
        return None, None, None

    for attrs in (
        dict(refined_group.attrs) if _is_group_like(refined_group) else {},
        dict(subject_group.attrs) if _is_group_like(subject_group) else {},
    ):
        source_group = attrs.get("source_keypoint_group")
        source_run = resolve_source_keypoints_run(attrs)
        if isinstance(source_group, str) and isinstance(source_run, str):
            parent = root.get(source_group)
            if _is_group_like(parent) and source_run in parent:
                return parent[source_run], source_group, source_run

    if expected_crop_run:
        for group_name in ("refined_keypoints_runs", "keypoints_runs"):
            parent = root.get(group_name)
            if not _is_group_like(parent):
                continue
            for run_name in _candidate_run_names(parent):
                candidate = parent[run_name]
                source_crop_run = candidate.attrs.get("source_crop_run")
                if source_crop_run is not None and str(source_crop_run) == str(expected_crop_run):
                    return candidate, group_name, run_name

    for group_name in ("refined_keypoints_runs", "keypoints_runs"):
        parent = root.get(group_name)
        if not _is_group_like(parent):
            continue
        for run_name in _candidate_run_names(parent):
            return parent[run_name], group_name, run_name
    return None, None, None


def _load_keypoint_success_flags(
    kp_group: zarr.Group,
    *,
    total_rois: int,
) -> np.ndarray:
    raw: Optional[np.ndarray] = None
    for name in ("refined_success", "detection_success", "source_success"):
        if name not in kp_group:
            continue
        raw = np.asarray(kp_group[name][:], dtype=bool)
        break
    if raw is None:
        raw = np.ones((total_rois,), dtype=bool)
    if int(raw.shape[0]) != int(total_rois):
        raise RuntimeError(
            f"Keypoint success rows {int(raw.shape[0])} do not match crop ROI rows {int(total_rois)}."
        )
    return raw


def _message_panel(
    shape: Tuple[int, int],
    *,
    title: str,
    message: str,
) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    panel = np.zeros((max(24, h), max(24, w), 3), dtype=np.uint8)
    cv2.putText(panel, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
    for idx, line in enumerate(str(message).splitlines()[:3]):
        cv2.putText(
            panel,
            line,
            (8, 42 + idx * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
    )
    return panel


def _target_canvas_shape(
    roi_shape: Tuple[int, int],
    *,
    padding: int,
    edit_zoom: int,
) -> Tuple[int, int]:
    roi_h, roi_w = int(roi_shape[0]), int(roi_shape[1])
    patch_dim = max(1, int(padding) * 2 + 1)
    zoom = max(1, int(edit_zoom))

    top_height = roi_h + PANEL_LABEL_HEIGHT
    top_width = (roi_w * 2) + 6

    bottom_height = max(patch_dim + PANEL_LABEL_HEIGHT, patch_dim * zoom + PANEL_LABEL_HEIGHT)
    bottom_width = (patch_dim * 3) + (patch_dim * zoom) + (6 * 3)

    return top_height + 8 + bottom_height, max(top_width, bottom_width)


def _pad_canvas_to_shape(canvas: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    canvas_h, canvas_w = int(canvas.shape[0]), int(canvas.shape[1])
    pad_bottom = max(0, target_h - canvas_h)
    pad_right = max(0, target_w - canvas_w)
    if pad_bottom == 0 and pad_right == 0:
        return canvas
    return np.pad(canvas, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="constant")


def _build_missing_keypoint_view(
    roi_img: np.ndarray,
    source_mask: np.ndarray,
    current_mask: np.ndarray,
    *,
    center_source: str,
) -> Tuple[np.ndarray, None]:
    roi_panel = cv2.cvtColor(np.asarray(roi_img, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    roi_overlay = roi_panel.copy()
    overlay_full = roi_overlay.copy()
    overlay_full[np.asarray(current_mask, dtype=np.uint8) > 0] = MASK_COLOR
    roi_overlay = cv2.addWeighted(overlay_full, 0.45, roi_overlay, 0.55, 0.0)

    message = (
        "Swim bladder keypoint row is marked unsuccessful."
        if center_source == "unsuccessful_keypoint"
        else "Swim bladder keypoint is missing or non-finite."
    )
    top = _stack_h(
        [
            _labeled_panel(roi_panel, f"Crop ROI ({center_source})"),
            _labeled_panel(roi_overlay, "ROI Overlay"),
        ],
        gap=6,
    )
    bottom = _stack_h(
        [
            _message_panel(roi_img.shape, title="Swim Bladder Patch", message=message),
            _labeled_panel(cv2.cvtColor((np.asarray(source_mask, dtype=np.uint8) * 255), cv2.COLOR_GRAY2BGR), "Source Mask"),
            _labeled_panel(cv2.cvtColor((np.asarray(current_mask, dtype=np.uint8) * 255), cv2.COLOR_GRAY2BGR), "Stored Mask"),
            _message_panel(roi_img.shape, title="Edit Patch", message="Editing is disabled until a valid swim-bladder keypoint exists."),
        ],
        gap=6,
    )
    return _stack_v([top, bottom], gap=8), None


def _build_view(
    roi_img: np.ndarray,
    source_mask: np.ndarray,
    current_mask: np.ndarray,
    *,
    center_xy: Tuple[float, float],
    center_source: str,
    padding: int,
    edit_zoom: int,
    brush_radius: int,
    cursor_patch_xy: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, Dict[str, int]]:
    x0, x1, y0, y1 = _extract_patch_bounds(roi_img.shape, center_xy, padding)
    crop_patch = np.asarray(roi_img[y0:y1, x0:x1], dtype=np.uint8)
    source_patch = (np.asarray(source_mask[y0:y1, x0:x1], dtype=np.uint8) > 0).astype(np.uint8)
    current_patch = (np.asarray(current_mask[y0:y1, x0:x1], dtype=np.uint8) > 0).astype(np.uint8)

    patch_crop_panel = cv2.cvtColor(crop_patch, cv2.COLOR_GRAY2BGR)
    source_mask_panel = cv2.cvtColor(source_patch * 255, cv2.COLOR_GRAY2BGR)
    current_mask_panel = cv2.cvtColor(current_patch * 255, cv2.COLOR_GRAY2BGR)

    patch_overlay = patch_crop_panel.copy()
    overlay = patch_overlay.copy()
    overlay[current_patch > 0] = MASK_COLOR
    patch_overlay = cv2.addWeighted(overlay, 0.45, patch_overlay, 0.55, 0.0)

    edit_panel = patch_overlay.copy()
    zoom = max(1, int(edit_zoom))
    if zoom > 1:
        edit_panel = cv2.resize(edit_panel, None, fx=float(zoom), fy=float(zoom), interpolation=cv2.INTER_NEAREST)

    cx_local = int(round(float(center_xy[0]) - x0))
    cy_local = int(round(float(center_xy[1]) - y0))
    for panel in (patch_crop_panel, patch_overlay):
        cv2.drawMarker(panel, (cx_local, cy_local), CENTER_COLOR, cv2.MARKER_CROSS, 8, 1)

    cv2.drawMarker(
        edit_panel,
        (int(round(cx_local * zoom)), int(round(cy_local * zoom))),
        CENTER_COLOR,
        cv2.MARKER_CROSS,
        max(8, int(round(8 * zoom))),
        max(1, int(round(zoom / 2))),
    )
    if cursor_patch_xy is not None:
        cv2.circle(
            edit_panel,
            (int(round(cursor_patch_xy[0] * zoom)), int(round(cursor_patch_xy[1] * zoom))),
            max(1, int(round(brush_radius * zoom))),
            MASK_COLOR,
            1,
            cv2.LINE_AA,
        )

    roi_panel = cv2.cvtColor(np.asarray(roi_img, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    roi_overlay = roi_panel.copy()
    overlay_full = roi_overlay.copy()
    overlay_full[np.asarray(current_mask, dtype=np.uint8) > 0] = MASK_COLOR
    roi_overlay = cv2.addWeighted(overlay_full, 0.45, roi_overlay, 0.55, 0.0)
    center_int = (int(round(float(center_xy[0]))), int(round(float(center_xy[1]))))
    cv2.drawMarker(roi_panel, center_int, CENTER_COLOR, cv2.MARKER_CROSS, 10, 1)
    cv2.drawMarker(roi_overlay, center_int, CENTER_COLOR, cv2.MARKER_CROSS, 10, 1)

    top = _stack_h(
        [
            _labeled_panel(roi_panel, f"Crop ROI ({center_source})"),
            _labeled_panel(roi_overlay, "ROI Overlay"),
        ],
        gap=6,
    )
    bottom_panels = [
        _labeled_panel(patch_crop_panel, "Swim Bladder Patch"),
        _labeled_panel(source_mask_panel, "Source Mask"),
        _labeled_panel(current_mask_panel, "Stored Mask"),
        _labeled_panel(edit_panel, "Edit Patch"),
    ]
    bottom = _stack_h(bottom_panels, gap=6)
    combined = _stack_v([top, bottom], gap=8)

    top_height = int(top.shape[0]) + 8
    edit_panel_widths = [panel.shape[1] for panel in bottom_panels]
    edit_index = len(bottom_panels) - 1
    edit_x = sum(edit_panel_widths[:edit_index]) + (6 * edit_index)
    edit_meta = {
        "x": int(edit_x),
        "y": int(top_height),
        "w": int(bottom_panels[edit_index].shape[1]),
        "h": int(bottom_panels[edit_index].shape[0]),
        "label_h": int(PANEL_LABEL_HEIGHT),
        "patch_x0": int(x0),
        "patch_y0": int(y0),
        "patch_w": int(crop_patch.shape[1]),
        "patch_h": int(crop_patch.shape[0]),
        "zoom": int(zoom),
    }
    return combined, edit_meta


def create_viewer(
    zarr_path: Path,
    *,
    subject_run: Optional[str],
    refined_run: Optional[str],
    crop_run: Optional[str],
    keypoint_run: Optional[str],
    keypoint_group: Optional[str],
    start_roi: int,
    roi_indices: Optional[Sequence[int]],
    padding: int,
    scale_percent: int,
    edit_zoom: int,
    review_state: str,
    review_method: str,
    review_intended_use: str,
    reviewer: Optional[str],
    review_notes: Optional[str],
    debug_ui_log: Optional[Path] = None,
) -> str:
    root = open_zarr_root(zarr_path, mode="a")
    source, refined = prepare_refined_subject_run(
        root,
        subject_run=subject_run,
        refined_run=refined_run,
        components=("swim_bladder",),
    )
    _primary_source, component_sources = _load_refined_component_source_runs(root, refined, default_source=source)
    swim_source = component_sources["swim_bladder"]
    if "swim_bladder" not in refined.component_to_index:
        raise RuntimeError(
            f"refined_subject_masks_runs/{refined.run_name} does not contain swim_bladder."
        )
    component_idx = int(refined.component_to_index["swim_bladder"])
    crop_run_name = str(crop_run or swim_source.crop_run)
    crop_source = CropImageSource.open(root, crop_run=crop_run_name, zarr_path=zarr_path)
    total_rois = int(crop_source.total_rois)
    if total_rois <= 0:
        raise RuntimeError("No ROIs found in crop run.")
    visible_roi_indices = _normalize_visible_roi_indices(roi_indices, total_rois=total_rois)
    queue_size = len(visible_roi_indices)
    subset_active = roi_indices is not None

    kp_group, _kp_group_name, _kp_run_name = _resolve_keypoint_group(
        root,
        subject_group=swim_source.group,
        refined_group=refined.group,
        explicit_run=keypoint_run,
        explicit_group=keypoint_group,
        expected_crop_run=crop_run_name,
    )
    if kp_group is None or "keypoints_roi" not in kp_group:
        raise RuntimeError("No keypoint run resolved for swim-bladder patch review.")
    _validate_keypoint_group_alignment(
        crop_source.crop_group,
        crop_run_name,
        kp_group,
        total_rois=total_rois,
    )
    keypoints_roi = np.asarray(kp_group["keypoints_roi"][:], dtype=np.float32)
    keypoint_success = _load_keypoint_success_flags(kp_group, total_rois=total_rois)
    keypoint_labels: Optional[Sequence[str]] = None
    labels_raw = kp_group.attrs.get("keypoint_labels")
    if isinstance(labels_raw, (list, tuple)):
        keypoint_labels = [str(item) for item in labels_raw]

    masks_arr = refined.group["masks_roi"]
    if int(masks_arr.shape[0]) != total_rois:
        raise RuntimeError("Refined masks rows do not match crop ROI rows.")

    _require_gui_display()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    ui_framework = str(cv2.currentUIFramework()).strip().upper() if hasattr(cv2, "currentUIFramework") else ""
    use_callback_roi_sync = not ui_framework.startswith("QT")
    debug_log = _open_ui_debug_log(debug_ui_log, zarr_path=zarr_path)

    def _debug_emit(event: str, /, **fields: object) -> None:
        if debug_log is not None:
            debug_log.emit(event, **fields)

    _debug_emit(
        "session_start",
        ui_framework=ui_framework or "unknown",
        use_callback_roi_sync=use_callback_roi_sync,
        control_window=CONTROL_WINDOW_NAME,
        total_rois=total_rois,
        queue_size=queue_size,
        subset_active=subset_active,
        start_roi=start_roi,
        initial_subject_run=subject_run,
        initial_refined_run=refined_run,
    )

    def _noop(_value: int) -> None:
        return

    initial_actual_roi = max(0, min(int(start_roi), total_rois - 1))
    if initial_actual_roi in visible_roi_indices:
        current_pos = visible_roi_indices.index(initial_actual_roi)
    else:
        current_pos = 0
    current_pos = _clamp_queue_pos(current_pos, queue_size=queue_size)
    last_loaded_pos = -1
    # GTK-backed HighGUI windows can return stale ROI positions via
    # getTrackbarPos() during active drags, so keep the callback path there.
    # Qt behaves better with direct polling and avoids visible slider wobble.
    _trackbar_drag_pos: List[int] = [current_pos]
    _displayed_trackbar_pos: List[int] = [current_pos]
    _pending_programmatic_roi_pos: List[Optional[int]] = [None]
    _last_polled_roi_slider: List[Optional[int]] = [None]
    _last_window_rect: List[Optional[Tuple[int, int, int, int]]] = [None]
    _setting_trackbar = [False]

    def _on_roi_trackbar(value: int) -> None:
        if not _setting_trackbar[0]:
            clamped = _clamp_queue_pos(value, queue_size=queue_size)
            _trackbar_drag_pos[0] = clamped
            _displayed_trackbar_pos[0] = clamped
            _debug_emit(
                "roi_trackbar_callback",
                value=value,
                clamped=clamped,
                current_pos=current_pos,
            )

    def _set_roi_trackbar(pos: int) -> None:
        clamped = _clamp_queue_pos(pos, queue_size=queue_size)
        if _displayed_trackbar_pos[0] == clamped:
            _debug_emit(
                "roi_trackbar_set_skipped",
                requested=pos,
                clamped=clamped,
                displayed=_displayed_trackbar_pos[0],
                pending_programmatic=_pending_programmatic_roi_pos[0],
            )
            return
        if not use_callback_roi_sync:
            _pending_programmatic_roi_pos[0] = clamped
        _debug_emit(
            "roi_trackbar_set",
            requested=pos,
            clamped=clamped,
            displayed_before=_displayed_trackbar_pos[0],
            pending_before=_pending_programmatic_roi_pos[0],
        )
        _setting_trackbar[0] = True
        cv2.setTrackbarPos("ROI", CONTROL_WINDOW_NAME, clamped)
        _setting_trackbar[0] = False
        _displayed_trackbar_pos[0] = clamped

    roi_trackbar_max = max(1, queue_size - 1)
    cv2.createTrackbar(
        "ROI",
        CONTROL_WINDOW_NAME,
        current_pos,
        roi_trackbar_max,
        _on_roi_trackbar if use_callback_roi_sync else _noop,
    )
    cv2.createTrackbar("Padding", CONTROL_WINDOW_NAME, max(1, min(int(padding), MAX_PADDING)), MAX_PADDING, _noop)
    cv2.createTrackbar("Brush", CONTROL_WINDOW_NAME, max(1, min(int(DEFAULT_BRUSH), BRUSH_MAX)), BRUSH_MAX, _noop)
    cv2.createTrackbar("Edit Zoom", CONTROL_WINDOW_NAME, max(1, min(int(edit_zoom), MAX_EDIT_ZOOM)), MAX_EDIT_ZOOM, _noop)
    cv2.createTrackbar(
        "Scale %",
        CONTROL_WINDOW_NAME,
        max(25, min(int(scale_percent), MAX_SCALE_PERCENT)),
        MAX_SCALE_PERCENT,
        _noop,
    )

    state: Dict[str, object] = {
        "loaded_roi_idx": -1,
        "loaded_queue_pos": -1,
        "roi_img": None,
        "stored_masks_row": None,
        "edit_masks_row": None,
        "edit_meta": None,
        "display_scale": 1.0,
        "cursor_patch_xy": None,
        "center_source": "unknown",
        "center_xy": None,
        "drawing": False,
        "erase_mode": False,
        "needs_redraw": False,
        "redraw_reason": None,
    }

    def _request_redraw(reason: str) -> None:
        previous_reason = state.get("redraw_reason")
        state["needs_redraw"] = True
        state["redraw_reason"] = str(reason)
        _debug_emit(
            "request_redraw",
            reason=reason,
            previous_reason=previous_reason,
            loaded_roi_idx=state.get("loaded_roi_idx"),
            current_pos=current_pos,
        )

    def _load_roi(queue_pos: int) -> None:
        roi_idx = int(visible_roi_indices[int(queue_pos)])
        roi_img = np.asarray(crop_source.read_slice(int(roi_idx), int(roi_idx) + 1)[0], dtype=np.uint8)
        masks_row = np.asarray(masks_arr[int(roi_idx)], dtype=np.uint8)
        keypoints_row = keypoints_roi[int(roi_idx)]
        center_xy, center_source = _resolve_swim_bladder_keypoint_center(
            keypoints_row,
            keypoint_labels,
            success_flag=bool(keypoint_success[int(roi_idx)]),
        )
        state["loaded_roi_idx"] = int(roi_idx)
        state["loaded_queue_pos"] = int(queue_pos)
        state["roi_img"] = roi_img
        state["stored_masks_row"] = masks_row.copy()
        state["edit_masks_row"] = masks_row.copy()
        state["cursor_patch_xy"] = None
        state["center_xy"] = center_xy
        state["center_source"] = center_source
        _debug_emit(
            "load_roi",
            queue_pos=queue_pos,
            roi_idx=roi_idx,
            center_source=center_source,
        )

    def _update_display() -> None:
        roi_img = state.get("roi_img")
        edit_masks_row = state.get("edit_masks_row")
        if roi_img is None or edit_masks_row is None:
            return
        roi_arr = np.asarray(roi_img, dtype=np.uint8)
        masks_row = np.asarray(edit_masks_row, dtype=np.uint8)
        source_mask = _source_component_mask_row(
            swim_source,
            "swim_bladder",
            int(state["loaded_roi_idx"]),
            fallback_shape=tuple(int(value) for value in masks_row[component_idx].shape),
        )
        padding_val = max(1, int(cv2.getTrackbarPos("Padding", CONTROL_WINDOW_NAME)))
        brush_val = max(1, int(cv2.getTrackbarPos("Brush", CONTROL_WINDOW_NAME)))
        edit_zoom_val = max(1, int(cv2.getTrackbarPos("Edit Zoom", CONTROL_WINDOW_NAME)))
        scale_percent_val = max(25, int(cv2.getTrackbarPos("Scale %", CONTROL_WINDOW_NAME)))
        state["display_scale"] = max(0.25, float(scale_percent_val) / 100.0)

        center_xy = state.get("center_xy")
        if center_xy is None:
            canvas, edit_meta = _build_missing_keypoint_view(
                roi_arr,
                source_mask,
                masks_row[component_idx],
                center_source=str(state["center_source"]),
            )
        else:
            canvas, edit_meta = _build_view(
                roi_arr,
                source_mask,
                masks_row[component_idx],
                center_xy=tuple(center_xy),
                center_source=str(state["center_source"]),
                padding=padding_val,
                edit_zoom=edit_zoom_val,
                brush_radius=brush_val,
                cursor_patch_xy=state.get("cursor_patch_xy"),
            )
        canvas = _pad_canvas_to_shape(
            canvas,
            _target_canvas_shape(
                tuple(roi_arr.shape),
                padding=padding_val,
                edit_zoom=edit_zoom_val,
            ),
        )
        review_payload = dict(refined.group.attrs.get("component_review_statuses") or {}).get("swim_bladder", {})
        review_state_text = str(review_payload.get("state") or "pending")
        brush_mode_text = "erase" if bool(state.get("erase_mode")) else "paint"
        queue_pos = int(state.get("loaded_queue_pos") or 0)
        roi_position_text = f"ROI {int(state['loaded_roi_idx']) + 1}/{total_rois}"
        if subset_active:
            roi_position_text = f"Queue {queue_pos + 1}/{queue_size}  {roi_position_text}"
        cv2.putText(
            canvas,
            (
                f"{roi_position_text}  "
                f"Center={state['center_source']}  "
                f"Brush={brush_val} ({brush_mode_text})  Review={review_state_text}"
            ),
            (12, max(16, canvas.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        state["edit_meta"] = edit_meta
        if float(state["display_scale"]) != 1.0:
            canvas = cv2.resize(
                canvas,
                None,
                fx=float(state["display_scale"]),
                fy=float(state["display_scale"]),
                interpolation=cv2.INTER_NEAREST,
            )
        cv2.imshow(WINDOW_NAME, canvas)
        window_rect: Optional[Tuple[int, int, int, int]] = None
        if hasattr(cv2, "getWindowImageRect"):
            try:
                rect_raw = cv2.getWindowImageRect(WINDOW_NAME)
                if rect_raw is not None:
                    window_rect = tuple(int(v) for v in rect_raw)
            except Exception:
                window_rect = None
        redraw_reason = state.get("redraw_reason")
        if redraw_reason is not None or window_rect != _last_window_rect[0]:
            _debug_emit(
                "update_display",
                redraw_reason=redraw_reason,
                canvas_shape=tuple(int(v) for v in canvas.shape),
                window_rect=window_rect,
                loaded_roi_idx=state.get("loaded_roi_idx"),
                current_pos=current_pos,
            )
        _last_window_rect[0] = window_rect
        state["needs_redraw"] = False
        state["redraw_reason"] = None

    def _save_current() -> None:
        if state.get("center_xy") is None:
            print(
                "Cannot save swim-bladder patch edits for this ROI: "
                f"{state.get('center_source')}."
            )
            return
        save_refined_subject_roi(
            source=source,
            refined=refined,
            roi_idx=int(state["loaded_roi_idx"]),
            edited_masks=np.asarray(state["edit_masks_row"], dtype=np.uint8),
            component_sources=component_sources,
        )
        state["stored_masks_row"] = np.asarray(state["edit_masks_row"], dtype=np.uint8).copy()
        print(f"Saved swim bladder patch edits for ROI {int(state['loaded_roi_idx'])}.")
        _debug_emit("save_current", loaded_roi_idx=state.get("loaded_roi_idx"))

    def _apply_patch_at(local_x: int, local_y: int, erase: bool) -> None:
        edit_meta = state.get("edit_meta")
        edit_masks_row = state.get("edit_masks_row")
        if not isinstance(edit_meta, dict) or edit_masks_row is None:
            return
        patch_x = int(local_x / max(1, int(edit_meta["zoom"])))
        patch_y = int((local_y - int(edit_meta["label_h"])) / max(1, int(edit_meta["zoom"])))
        patch_w = int(edit_meta["patch_w"])
        patch_h = int(edit_meta["patch_h"])
        if not (0 <= patch_x < patch_w and 0 <= patch_y < patch_h):
            return
        state["cursor_patch_xy"] = (patch_x, patch_y)
        mask = np.asarray(edit_masks_row, dtype=np.uint8)[component_idx]
        full_x = int(edit_meta["patch_x0"]) + patch_x
        full_y = int(edit_meta["patch_y0"]) + patch_y
        brush_val = max(1, int(cv2.getTrackbarPos("Brush", CONTROL_WINDOW_NAME)))
        cv2.circle(mask, (full_x, full_y), brush_val, 0 if erase else 1, -1)
        _debug_emit(
            "apply_patch",
            patch_xy=(patch_x, patch_y),
            full_xy=(full_x, full_y),
            brush=brush_val,
            erase=erase,
        )
        _request_redraw("mouse_apply_patch")

    def _on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        edit_meta = state.get("edit_meta")
        if not isinstance(edit_meta, dict):
            return
        ctrl_down, shift_down, left_down = _mouse_modifier_state(int(_flags))
        scale = float(state.get("display_scale") or 1.0)
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
        in_edit = (
            int(edit_meta["x"]) <= x < int(edit_meta["x"]) + int(edit_meta["w"])
            and int(edit_meta["y"]) <= y < int(edit_meta["y"]) + int(edit_meta["h"])
        )
        if not in_edit:
            if state.get("cursor_patch_xy") is not None:
                state["cursor_patch_xy"] = None
                _request_redraw("mouse_leave_edit")
            if event == cv2.EVENT_LBUTTONUP or not left_down:
                if bool(state.get("drawing")):
                    _debug_emit("drawing_state", active=False, reason="mouse_leave_or_button_up")
                state["drawing"] = False
            return

        local_x = x - int(edit_meta["x"])
        local_y = y - int(edit_meta["y"])
        if event == cv2.EVENT_LBUTTONDOWN:
            # Brush edits are gated behind Ctrl to avoid accidental draws while navigating.
            state["drawing"] = bool(ctrl_down)
            _debug_emit("drawing_state", active=bool(state["drawing"]), reason="lbutton_down", ctrl_down=ctrl_down)
        elif event == cv2.EVENT_MOUSEMOVE and bool(state.get("drawing")):
            if not (ctrl_down and left_down):
                state["drawing"] = False
                _debug_emit("drawing_state", active=False, reason="mouse_move_without_ctrl")
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            _debug_emit("drawing_state", active=False, reason="lbutton_up")

        patch_x = int(local_x / max(1, int(edit_meta["zoom"])))
        patch_y = int((local_y - int(edit_meta["label_h"])) / max(1, int(edit_meta["zoom"])))
        if bool(state.get("drawing")) and 0 <= patch_x < int(edit_meta["patch_w"]) and 0 <= patch_y < int(edit_meta["patch_h"]):
            erase_mode = _resolve_erase_mode(bool(state.get("erase_mode")), shift_down)
            _apply_patch_at(local_x, local_y, erase_mode)
        elif event == cv2.EVENT_MOUSEMOVE:
            if 0 <= patch_x < int(edit_meta["patch_w"]) and 0 <= patch_y < int(edit_meta["patch_h"]):
                next_cursor = (patch_x, patch_y)
                if state.get("cursor_patch_xy") != next_cursor:
                    state["cursor_patch_xy"] = next_cursor
                    _request_redraw("mouse_cursor_move")

    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)
    _load_roi(current_pos)
    last_loaded_pos = current_pos
    _update_display()

    print("\nSwim Bladder Patch Review")
    print(f"  Source run: subject_mask_runs/{source.run_name}")
    print(f"  Refined run: refined_subject_masks_runs/{refined.run_name}")
    print(f"  Crop run: {crop_run_name}")
    if subset_active:
        print(f"  ROI subset: {queue_size} rows")
    print("Controls:")
    print("  Mouse on Edit Patch: hold Ctrl+LMB to paint")
    print("  Mouse while drawing: hold Shift to temporarily invert brush mode")
    print("  [ / ]: brush size")
    print("  x: toggle brush mode (paint/erase)")
    print("  n/p: next/previous ROI")
    print("  s: save current ROI edits")
    print("  r: reset current ROI to stored refined mask")
    print("  a: approve swim bladder")
    print("  N/R/P: needs_review / rejected / pending")
    print("  q/ESC: quit")

    while True:
        if use_callback_roi_sync:
            # Accept mouse-driven trackbar drags via callback updates rather than
            # getTrackbarPos(), which can be stale on GTK backends.
            if _trackbar_drag_pos[0] != current_pos:
                current_pos = _trackbar_drag_pos[0]
                _debug_emit("current_pos_from_callback", current_pos=current_pos)
        else:
            roi_slider = _clamp_queue_pos(int(cv2.getTrackbarPos("ROI", CONTROL_WINDOW_NAME)), queue_size=queue_size)
            if _last_polled_roi_slider[0] != roi_slider or _pending_programmatic_roi_pos[0] is not None:
                _debug_emit(
                    "roi_trackbar_poll",
                    roi_slider=roi_slider,
                    current_pos=current_pos,
                    pending_programmatic=_pending_programmatic_roi_pos[0],
                )
                _last_polled_roi_slider[0] = roi_slider
            pending_programmatic = _pending_programmatic_roi_pos[0]
            if pending_programmatic is not None:
                if roi_slider == pending_programmatic:
                    _displayed_trackbar_pos[0] = roi_slider
                    _pending_programmatic_roi_pos[0] = None
                    _debug_emit("roi_trackbar_programmatic_ack", roi_slider=roi_slider)
            elif roi_slider != current_pos:
                current_pos = roi_slider
                _displayed_trackbar_pos[0] = roi_slider
                _debug_emit("current_pos_from_poll", current_pos=current_pos)

        if current_pos != last_loaded_pos:
            _set_roi_trackbar(current_pos)
            last_loaded_pos = current_pos
            _load_roi(current_pos)
        if not use_callback_roi_sync or bool(state.get("needs_redraw")):
            _update_display()

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            _debug_emit("key_event", key=key, key_chr=chr(key) if 32 <= key <= 126 else None)
            break

        if key == ord("n"):
            _debug_emit("key_event", key=key, key_chr="n", current_pos=current_pos)
            if current_pos < queue_size - 1:
                current_pos += 1
                if use_callback_roi_sync:
                    _trackbar_drag_pos[0] = current_pos
                else:
                    _set_roi_trackbar(current_pos)
        elif key == ord("p"):
            _debug_emit("key_event", key=key, key_chr="p", current_pos=current_pos)
            if current_pos > 0:
                current_pos -= 1
                if use_callback_roi_sync:
                    _trackbar_drag_pos[0] = current_pos
                else:
                    _set_roi_trackbar(current_pos)
        elif key == ord("["):
            current = max(1, int(cv2.getTrackbarPos("Brush", CONTROL_WINDOW_NAME)))
            cv2.setTrackbarPos("Brush", CONTROL_WINDOW_NAME, max(1, current - 1))
            _request_redraw("key_brush_dec")
        elif key == ord("]"):
            current = max(1, int(cv2.getTrackbarPos("Brush", CONTROL_WINDOW_NAME)))
            cv2.setTrackbarPos("Brush", CONTROL_WINDOW_NAME, min(BRUSH_MAX, current + 1))
            _request_redraw("key_brush_inc")
        elif key == ord("r"):
            state["edit_masks_row"] = np.asarray(state["stored_masks_row"], dtype=np.uint8).copy()
            _request_redraw("key_reset")
        elif key == ord("x"):
            new_mode = not bool(state.get("erase_mode"))
            state["erase_mode"] = new_mode
            print(f"Brush mode: {'erase' if new_mode else 'paint'}")
            _request_redraw("key_toggle_erase")
        elif key == ord("s"):
            _save_current()
        elif key in (ord("a"), ord("N"), ord("R"), ord("P")):
            _debug_emit("key_event", key=key, key_chr=chr(key), current_pos=current_pos)
            state_value = {
                ord("a"): "approved",
                ord("N"): "needs_review",
                ord("R"): "rejected",
                ord("P"): "pending",
            }[key]
            reviewer_name = reviewer or os.environ.get("USER") or os.environ.get("USERNAME")
            payload, run_payload = apply_component_review_status(
                refined.parent,
                refined.run_name,
                refined.group,
                component_name="swim_bladder",
                state=state_value,
                method=review_method,
                intended_use=review_intended_use,
                reviewer=reviewer_name,
                notes=review_notes,
                zarr_path=zarr_path,
            )
            print(
                f"Set swim_bladder review to {payload.get('state')} "
                f"(run={run_payload.get('state')})"
            )
            _request_redraw("key_review_state")

    cv2.destroyAllWindows()
    _debug_emit("session_end", loaded_roi_idx=state.get("loaded_roi_idx"))
    if debug_log is not None:
        debug_log.close()
    return refined.run_name


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual patch review/editor for swim-bladder masks.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette zarr archive.")
    parser.add_argument("--subject-run", help="Source subject_mask_runs/<run> to refine (default: latest).")
    parser.add_argument("--refined-run", help="Existing refined_subject_masks run to open, or target name to create.")
    parser.add_argument("--crop-run", help="Crop run to use for ROI images (default: source subject-mask crop run).")
    parser.add_argument("--keypoint-run", help="Optional keypoint run to anchor swim-bladder patch centers.")
    parser.add_argument("--keypoint-group", help="Optional keypoint parent group for --keypoint-run.")
    parser.add_argument("--roi-index", type=int, default=0, help="Initial ROI index.")
    parser.add_argument(
        "--roi-indices",
        type=_parse_roi_indices_arg,
        help="Optional comma-separated ROI indices to review as the active queue.",
    )
    parser.add_argument("--padding", type=int, default=DEFAULT_PADDING)
    parser.add_argument("--scale-percent", type=int, default=DEFAULT_SCALE_PERCENT)
    parser.add_argument("--edit-zoom", type=int, default=DEFAULT_EDIT_ZOOM)
    parser.add_argument("--review-state", default="approved", choices=["approved", "pending", "rejected", "needs_review"])
    parser.add_argument("--review-method", default=DEFAULT_REVIEW_METHOD)
    parser.add_argument("--review-intended-use", default=DEFAULT_REVIEW_INTENDED_USE)
    parser.add_argument("--reviewer", help="Reviewer name to record in review payloads.")
    parser.add_argument("--review-notes", help="Optional note attached to review payload updates.")
    parser.add_argument(
        "--debug-ui-log",
        type=Path,
        help="Optional JSONL log path for UI event instrumentation.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> str:
    args = parse_args(argv)
    return create_viewer(
        args.zarr_path,
        subject_run=args.subject_run,
        refined_run=args.refined_run,
        crop_run=args.crop_run,
        keypoint_run=args.keypoint_run,
        keypoint_group=args.keypoint_group,
        start_roi=args.roi_index,
        roi_indices=args.roi_indices,
        padding=args.padding,
        scale_percent=args.scale_percent,
        edit_zoom=args.edit_zoom,
        review_state=args.review_state,
        review_method=args.review_method,
        review_intended_use=args.review_intended_use,
        reviewer=args.reviewer,
        review_notes=args.review_notes,
        debug_ui_log=args.debug_ui_log,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
