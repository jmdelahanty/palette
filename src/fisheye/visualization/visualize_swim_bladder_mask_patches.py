#!/usr/bin/env python3
"""Interactive swim-bladder patch viewer/editor for refined subject masks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import cv2
import numpy as np
import zarr

from ..shared.crop_image_source import CropImageSource
from ..shared.provenance_attrs import resolve_source_keypoints_run
from ..tune.refined_subject_mask_review import (
    DEFAULT_REVIEW_INTENDED_USE,
    DEFAULT_REVIEW_METHOD,
    apply_component_review_status,
    prepare_refined_subject_run,
    save_refined_subject_roi,
)
from ..utils.zarr_io import open_zarr_root

try:
    cv2_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "2")))
except (TypeError, ValueError):
    cv2_threads = 2
cv2.setNumThreads(cv2_threads)

WINDOW_NAME = "Swim Bladder Mask Patch Viewer"
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


def _resolve_keypoint_group(
    root: zarr.Group,
    *,
    subject_group: Optional[zarr.Group],
    refined_group: Optional[zarr.Group],
    explicit_run: Optional[str],
    explicit_group: Optional[str],
) -> Tuple[Optional[zarr.Group], Optional[str], Optional[str]]:
    if explicit_group and explicit_run:
        parent = root.get(explicit_group)
        if isinstance(parent, zarr.Group) and explicit_run in parent:
            return parent[explicit_run], explicit_group, explicit_run
        return None, None, None

    if explicit_run and not explicit_group:
        for group_name in ("refined_keypoints_runs", "keypoints_runs"):
            parent = root.get(group_name)
            if isinstance(parent, zarr.Group) and explicit_run in parent:
                return parent[explicit_run], group_name, explicit_run
        return None, None, None

    for attrs in (
        dict(refined_group.attrs) if isinstance(refined_group, zarr.Group) else {},
        dict(subject_group.attrs) if isinstance(subject_group, zarr.Group) else {},
    ):
        source_group = attrs.get("source_keypoint_group")
        source_run = resolve_source_keypoints_run(attrs)
        if isinstance(source_group, str) and isinstance(source_run, str):
            parent = root.get(source_group)
            if isinstance(parent, zarr.Group) and source_run in parent:
                return parent[source_run], source_group, source_run

    for group_name in ("refined_keypoints_runs", "keypoints_runs"):
        parent = root.get(group_name)
        if not isinstance(parent, zarr.Group):
            continue
        latest = parent.attrs.get("latest")
        if isinstance(latest, str) and latest in parent:
            return parent[latest], group_name, latest
        keys = sorted(parent.group_keys()) if hasattr(parent, "group_keys") else sorted(parent.keys())
        if keys:
            run_name = str(keys[-1])
            return parent[run_name], group_name, run_name
    return None, None, None


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
    padding: int,
    scale_percent: int,
    edit_zoom: int,
    review_state: str,
    review_method: str,
    review_intended_use: str,
    reviewer: Optional[str],
    review_notes: Optional[str],
) -> str:
    root = open_zarr_root(zarr_path, mode="a")
    source, refined = prepare_refined_subject_run(
        root,
        subject_run=subject_run,
        refined_run=refined_run,
        components=("swim_bladder",),
    )
    if "swim_bladder" not in refined.component_to_index:
        raise RuntimeError(
            f"refined_subject_masks_runs/{refined.run_name} does not contain swim_bladder."
        )
    component_idx = int(refined.component_to_index["swim_bladder"])
    crop_run_name = str(crop_run or source.crop_run)
    crop_source = CropImageSource.open(root, crop_run=crop_run_name, zarr_path=zarr_path)
    total_rois = int(crop_source.total_rois)
    if total_rois <= 0:
        raise RuntimeError("No ROIs found in crop run.")

    kp_group, _kp_group_name, _kp_run_name = _resolve_keypoint_group(
        root,
        subject_group=source.group,
        refined_group=refined.group,
        explicit_run=keypoint_run,
        explicit_group=keypoint_group,
    )
    keypoints_roi = None
    keypoint_labels: Optional[Sequence[str]] = None
    if kp_group is not None and "keypoints_roi" in kp_group:
        candidate = np.asarray(kp_group["keypoints_roi"][:], dtype=np.float32)
        if int(candidate.shape[0]) == total_rois:
            keypoints_roi = candidate
            labels_raw = kp_group.attrs.get("keypoint_labels")
            if isinstance(labels_raw, (list, tuple)):
                keypoint_labels = [str(item) for item in labels_raw]

    masks_arr = refined.group["masks_roi"]
    if int(masks_arr.shape[0]) != total_rois:
        raise RuntimeError("Refined masks rows do not match crop ROI rows.")

    _require_gui_display()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1900, 1100)

    def _noop(_value: int) -> None:
        return

    cv2.createTrackbar("ROI", WINDOW_NAME, max(0, min(int(start_roi), total_rois - 1)), total_rois - 1, _noop)
    cv2.createTrackbar("Padding", WINDOW_NAME, max(1, min(int(padding), MAX_PADDING)), MAX_PADDING, _noop)
    cv2.createTrackbar("Brush", WINDOW_NAME, max(1, min(int(DEFAULT_BRUSH), BRUSH_MAX)), BRUSH_MAX, _noop)
    cv2.createTrackbar("Edit Zoom", WINDOW_NAME, max(1, min(int(edit_zoom), MAX_EDIT_ZOOM)), MAX_EDIT_ZOOM, _noop)
    cv2.createTrackbar(
        "Scale %",
        WINDOW_NAME,
        max(25, min(int(scale_percent), MAX_SCALE_PERCENT)),
        MAX_SCALE_PERCENT,
        _noop,
    )

    state: Dict[str, object] = {
        "loaded_roi_idx": -1,
        "roi_img": None,
        "stored_masks_row": None,
        "edit_masks_row": None,
        "edit_meta": None,
        "display_scale": 1.0,
        "cursor_patch_xy": None,
        "center_source": "unknown",
        "center_xy": (0.0, 0.0),
        "drawing": False,
        "erase_mode": False,
    }
    current_pos = max(0, min(int(start_roi), total_rois - 1))

    def _load_roi(roi_idx: int) -> None:
        roi_img = np.asarray(crop_source.read_slice(int(roi_idx), int(roi_idx) + 1)[0], dtype=np.uint8)
        masks_row = np.asarray(masks_arr[int(roi_idx)], dtype=np.uint8)
        keypoints_row = keypoints_roi[int(roi_idx)] if keypoints_roi is not None else None
        center_xy, center_source = _resolve_swim_bladder_center_with_source(
            keypoints_row,
            keypoint_labels,
            masks_row[component_idx],
            tuple(roi_img.shape),
        )
        state["loaded_roi_idx"] = int(roi_idx)
        state["roi_img"] = roi_img
        state["stored_masks_row"] = masks_row.copy()
        state["edit_masks_row"] = masks_row.copy()
        state["cursor_patch_xy"] = None
        state["center_xy"] = center_xy
        state["center_source"] = center_source

    def _update_display() -> None:
        roi_img = state.get("roi_img")
        edit_masks_row = state.get("edit_masks_row")
        if roi_img is None or edit_masks_row is None:
            return
        roi_arr = np.asarray(roi_img, dtype=np.uint8)
        masks_row = np.asarray(edit_masks_row, dtype=np.uint8)
        source_mask = np.asarray(
            source.group["masks_roi"][int(state["loaded_roi_idx"]), source.mask_labels.index("swim_bladder")]
            if "swim_bladder" in source.mask_labels and bool(source.available_channels[source.mask_labels.index("swim_bladder")])
            else np.zeros_like(masks_row[component_idx]),
            dtype=np.uint8,
        )
        padding_val = max(1, int(cv2.getTrackbarPos("Padding", WINDOW_NAME)))
        brush_val = max(1, int(cv2.getTrackbarPos("Brush", WINDOW_NAME)))
        edit_zoom_val = max(1, int(cv2.getTrackbarPos("Edit Zoom", WINDOW_NAME)))
        scale_percent_val = max(25, int(cv2.getTrackbarPos("Scale %", WINDOW_NAME)))
        state["display_scale"] = max(0.25, float(scale_percent_val) / 100.0)

        canvas, edit_meta = _build_view(
            roi_arr,
            source_mask,
            masks_row[component_idx],
            center_xy=tuple(state["center_xy"]),
            center_source=str(state["center_source"]),
            padding=padding_val,
            edit_zoom=edit_zoom_val,
            brush_radius=brush_val,
            cursor_patch_xy=state.get("cursor_patch_xy"),
        )
        review_payload = dict(refined.group.attrs.get("component_review_statuses") or {}).get("swim_bladder", {})
        review_state_text = str(review_payload.get("state") or "pending")
        brush_mode_text = "erase" if bool(state.get("erase_mode")) else "paint"
        cv2.putText(
            canvas,
            (
                f"ROI {int(state['loaded_roi_idx']) + 1}/{total_rois}  "
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

    def _save_current() -> None:
        save_refined_subject_roi(
            source=source,
            refined=refined,
            roi_idx=int(state["loaded_roi_idx"]),
            edited_masks=np.asarray(state["edit_masks_row"], dtype=np.uint8),
        )
        state["stored_masks_row"] = np.asarray(state["edit_masks_row"], dtype=np.uint8).copy()
        print(f"Saved swim bladder patch edits for ROI {int(state['loaded_roi_idx'])}.")

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
        brush_val = max(1, int(cv2.getTrackbarPos("Brush", WINDOW_NAME)))
        cv2.circle(mask, (full_x, full_y), brush_val, 0 if erase else 1, -1)
        _update_display()

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
                _update_display()
            if event == cv2.EVENT_LBUTTONUP or not left_down:
                state["drawing"] = False
            return

        local_x = x - int(edit_meta["x"])
        local_y = y - int(edit_meta["y"])
        if event == cv2.EVENT_LBUTTONDOWN:
            # Brush edits are gated behind Ctrl to avoid accidental draws while navigating.
            state["drawing"] = bool(ctrl_down)
        elif event == cv2.EVENT_MOUSEMOVE and bool(state.get("drawing")):
            if not (ctrl_down and left_down):
                state["drawing"] = False
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False

        patch_x = int(local_x / max(1, int(edit_meta["zoom"])))
        patch_y = int((local_y - int(edit_meta["label_h"])) / max(1, int(edit_meta["zoom"])))
        if bool(state.get("drawing")) and 0 <= patch_x < int(edit_meta["patch_w"]) and 0 <= patch_y < int(edit_meta["patch_h"]):
            erase_mode = _resolve_erase_mode(bool(state.get("erase_mode")), shift_down)
            _apply_patch_at(local_x, local_y, erase_mode)
        elif event == cv2.EVENT_MOUSEMOVE:
            if 0 <= patch_x < int(edit_meta["patch_w"]) and 0 <= patch_y < int(edit_meta["patch_h"]):
                state["cursor_patch_xy"] = (patch_x, patch_y)
                _update_display()

    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)
    _load_roi(current_pos)
    _update_display()

    print("\nSwim Bladder Patch Review")
    print(f"  Source run: subject_mask_runs/{source.run_name}")
    print(f"  Refined run: refined_subject_masks_runs/{refined.run_name}")
    print(f"  Crop run: {crop_run_name}")
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
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("["):
            current = max(1, int(cv2.getTrackbarPos("Brush", WINDOW_NAME)))
            cv2.setTrackbarPos("Brush", WINDOW_NAME, max(1, current - 1))
            _update_display()
        elif key == ord("]"):
            current = max(1, int(cv2.getTrackbarPos("Brush", WINDOW_NAME)))
            cv2.setTrackbarPos("Brush", WINDOW_NAME, min(BRUSH_MAX, current + 1))
            _update_display()
        elif key == ord("r"):
            state["edit_masks_row"] = np.asarray(state["stored_masks_row"], dtype=np.uint8).copy()
            _update_display()
        elif key == ord("x"):
            new_mode = not bool(state.get("erase_mode"))
            state["erase_mode"] = new_mode
            print(f"Brush mode: {'erase' if new_mode else 'paint'}")
            _update_display()
        elif key == ord("s"):
            _save_current()
        elif key == ord("n"):
            if current_pos < total_rois - 1:
                current_pos += 1
                cv2.setTrackbarPos("ROI", WINDOW_NAME, current_pos)
                _load_roi(current_pos)
                _update_display()
        elif key == ord("p"):
            if current_pos > 0:
                current_pos -= 1
                cv2.setTrackbarPos("ROI", WINDOW_NAME, current_pos)
                _load_roi(current_pos)
                _update_display()
        elif key in (ord("a"), ord("N"), ord("R"), ord("P")):
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
            )
            print(
                f"Set swim_bladder review to {payload.get('state')} "
                f"(run={run_payload.get('state')})"
            )
            _update_display()
        else:
            roi_slider = int(cv2.getTrackbarPos("ROI", WINDOW_NAME))
            if roi_slider != current_pos:
                current_pos = roi_slider
                _load_roi(current_pos)
                _update_display()

    cv2.destroyAllWindows()
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
    parser.add_argument("--padding", type=int, default=DEFAULT_PADDING)
    parser.add_argument("--scale-percent", type=int, default=DEFAULT_SCALE_PERCENT)
    parser.add_argument("--edit-zoom", type=int, default=DEFAULT_EDIT_ZOOM)
    parser.add_argument("--review-state", default="approved", choices=["approved", "pending", "rejected", "needs_review"])
    parser.add_argument("--review-method", default=DEFAULT_REVIEW_METHOD)
    parser.add_argument("--review-intended-use", default=DEFAULT_REVIEW_INTENDED_USE)
    parser.add_argument("--reviewer", help="Reviewer name to record in review payloads.")
    parser.add_argument("--review-notes", help="Optional note attached to review payload updates.")
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
        padding=args.padding,
        scale_percent=args.scale_percent,
        edit_zoom=args.edit_zoom,
        review_state=args.review_state,
        review_method=args.review_method,
        review_intended_use=args.review_intended_use,
        reviewer=args.reviewer,
        review_notes=args.review_notes,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
