"""
Manual review tool for detection runs.

Edits are written to refined_detect_runs/<run>/<output_group> (default: manual)
so raw detection outputs remain untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from rich.console import Console
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, erosion

from fisheye.detection.detect_traditional import create_dish_mask, get_detection_parameters
from fisheye.shared.refined_detect_review import (
    DEFAULT_DETECT_GROUP_PREFERENCE,
    resolve_refined_detect_group,
)


def _get_latest_refined_run(root: zarr.Group) -> str:
    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_detect_runs found in archive.")
    latest = refined_parent.attrs.get("latest")
    if not latest:
        raise RuntimeError("No refined detect runs recorded.")
    return latest


def _pick_variant(refined_run: zarr.Group, requested: Optional[str]) -> str:
    if requested:
        if requested in refined_run:
            return requested
        raise RuntimeError(f"Requested variant '{requested}' not found in refined run.")
    if "interpolated" in refined_run:
        return "interpolated"
    if "filtered" in refined_run:
        return "filtered"
    raise RuntimeError("Refined run has no interpolated/filtered variants to review.")


def _norm_to_rect(norm: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
    cx, cy, w, h = norm.tolist()
    x1 = (cx - w * 0.5) * width
    y1 = (cy - h * 0.5) * height
    x2 = (cx + w * 0.5) * width
    y2 = (cy + h * 0.5) * height
    return x1, y1, x2, y2


def _rect_to_norm(rect: Tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = rect
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    cx = (x_min + x_max) * 0.5 / width
    cy = (y_min + y_max) * 0.5 / height
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height
    return np.array([cx, cy, w, h], dtype=np.float64)


def _group_by_frame(frame_indices: np.ndarray) -> Dict[int, np.ndarray]:
    frame_map: Dict[int, List[int]] = {}
    for idx, frame in enumerate(frame_indices.astype(int)):
        frame_map.setdefault(int(frame), []).append(idx)
    return {frame: np.asarray(indices, dtype=np.int32) for frame, indices in frame_map.items()}


def _compute_frame_counts(frame_indices: np.ndarray, n_frames: int) -> np.ndarray:
    if frame_indices.size == 0:
        return np.zeros(n_frames, dtype=np.int32)
    return np.bincount(frame_indices.astype(np.int32, copy=False), minlength=n_frames).astype(np.int32, copy=False)


def _parse_frames_arg(value: Optional[str]) -> Optional[np.ndarray]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    items: List[object]
    path = Path(text)
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except Exception:
            data = None
        if isinstance(data, list):
            items = data
        else:
            items = re.split(r"[,\s]+", raw.strip())
    else:
        items = re.split(r"[,\s]+", text)
    frames: List[int] = []
    for item in items:
        if isinstance(item, (int, np.integer)):
            frames.append(int(item))
            continue
        token = str(item).strip()
        if not token:
            continue
        try:
            frames.append(int(token))
        except ValueError:
            continue
    if not frames:
        return None
    return np.asarray(sorted(set(frames)), dtype=np.int32)


def _get_or_create_retune_id(refined_run: zarr.Group, params: Dict[str, object]) -> int:
    existing = refined_run.attrs.get("retune_params")
    retune_params = existing if isinstance(existing, dict) else {}

    def signature(values: Dict[str, object]) -> tuple:
        return tuple(sorted(values.items()))

    target = signature(params)
    for key, value in retune_params.items():
        if isinstance(value, dict) and signature(value) == target:
            try:
                return int(key)
            except ValueError:
                continue

    existing_ids = [int(k) for k in retune_params.keys() if str(k).isdigit()]
    next_id = max(existing_ids, default=0) + 1
    retune_params[str(next_id)] = params
    refined_run.attrs["retune_params"] = retune_params
    return next_id


def _select_retune_base(
    refined_run: zarr.Group,
    variant: str,
    output_group: str,
) -> Tuple[zarr.Group, str, bool]:
    if output_group and output_group in refined_run:
        return refined_run[output_group], output_group, True
    if output_group == "manual":
        manual_latest = refined_run.attrs.get("manual_review_latest")
        if manual_latest and manual_latest in refined_run:
            return refined_run[manual_latest], str(manual_latest), True
    return refined_run[variant], variant, False


def _write_manual_group(
    refined_run: zarr.Group,
    output_group: str,
    frame_indices: np.ndarray,
    bbox_norm: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    retune_id: Optional[np.ndarray],
    frame_counts: np.ndarray,
    detection_source: Optional[np.ndarray],
    reason: Optional[np.ndarray],
    metadata: Dict[str, object],
    overwrite: bool,
) -> None:
    if output_group in refined_run:
        if not overwrite:
            raise RuntimeError(f"Output group '{output_group}' already exists. Use --overwrite to replace it.")
        del refined_run[output_group]

    group = refined_run.create_group(output_group)

    det_chunk = max(1, min(max(1, frame_indices.size), 4096))
    counts_chunk = max(1, min(frame_counts.size, 16384))

    group.create_array("frame_indices", data=frame_indices, chunks=(det_chunk,), overwrite=True)
    group.create_array("bbox_norm_coords", data=bbox_norm, chunks=(det_chunk, 4), overwrite=True)
    group.create_array("scores", data=scores, chunks=(det_chunk,), overwrite=True)
    group.create_array("class_ids", data=class_ids, chunks=(det_chunk,), overwrite=True)
    if retune_id is not None:
        group.create_array("retune_id", data=retune_id, chunks=(det_chunk,), overwrite=True)
    group.create_array("frame_counts", data=frame_counts, chunks=(counts_chunk,), overwrite=True)
    group.create_array("n_detections", data=frame_counts, chunks=(counts_chunk,), overwrite=True)
    group.create_array("frame_mapping", data=frame_indices, chunks=(det_chunk,), overwrite=True)

    column_fields = ["frame_indices", "bbox_norm_coords", "scores", "class_ids"]
    if retune_id is not None:
        column_fields.append("retune_id")
    if detection_source is not None:
        group.create_array("detection_source", data=detection_source, chunks=(det_chunk,), overwrite=True)
        column_fields.append("detection_source")

    if reason is not None:
        reason_arr = group.create_array(
            "reason",
            shape=(reason.shape[0],),
            chunks=(det_chunk,),
            dtype=VariableLengthUTF8(),
            fill_value="",
            overwrite=True,
        )
        reason_arr[:] = np.asarray(reason, dtype=str)
        column_fields.append("reason")

    group.attrs["storage_layout"] = "columnar"
    group.attrs["column_fields"] = column_fields
    group.attrs["field_names"] = column_fields
    group.attrs.update(metadata)


def _save_detection_tuning(
    root: zarr.Group,
    params: Dict[str, object],
    tuned_frame: Optional[int] = None,
) -> None:
    if "analysis_metadata" not in root:
        analysis_meta = root.create_group("analysis_metadata")
    else:
        analysis_meta = root["analysis_metadata"]

    metadata = dict(analysis_meta.attrs) if analysis_meta.attrs else {}
    metadata["detection_tuning"] = {
        "method": "blob_detection",
        "version": "1.0",
        "tuned_timestamp": datetime.now(timezone.utc).isoformat(),
        "tuned_parameters": {
            "ds_thresh": int(params["ds_thresh"]),
            "se1_radius": int(params["se1_radius"]),
            "se4_radius": int(params["se4_radius"]),
            "min_area": int(params["min_area"]),
            "max_area": int(params["max_area"]),
            "max_fish": int(params.get("max_fish", 20)),
        },
        "tuned_on_frame": int(tuned_frame) if tuned_frame is not None else None,
    }
    analysis_meta.attrs.update(metadata)


def run_manual_review(
    zarr_path: str,
    refined_run_name: Optional[str] = None,
    variant: Optional[str] = None,
    output_group: str = "manual",
    overwrite: bool = False,
    review_all: bool = False,
    max_frames: Optional[int] = None,
    target_frames: Optional[np.ndarray] = None,
    manual_score: float = 1.0,
    manual_class_id: int = 0,
    use_full_res: bool = False,
    review_state: str = "approved",
    review_method: str = "manual",
    review_intended_use: str = "training",
    reviewer: Optional[str] = None,
    review_notes: Optional[str] = None,
) -> None:
    root = zarr.open_group(zarr_path, mode="a")

    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_detect_runs found in archive.")

    refined_run_name = refined_run_name or refined_parent.attrs.get("latest")
    if not refined_run_name or refined_run_name not in refined_parent:
        raise RuntimeError("Refined detect run not found.")

    refined_run = refined_parent[refined_run_name]
    variant = _pick_variant(refined_run, variant)
    base_group, _, _ = _select_retune_base(refined_run, variant, output_group)

    if "raw_video" not in root:
        raise RuntimeError("Zarr archive is missing raw_video group.")

    if use_full_res:
        raise RuntimeError("Retune uses downsampled frames; omit --use-full-res.")

    images = root["raw_video"]["images_ds"]

    n_frames = int(images.shape[0])
    height = int(images.shape[1])
    width = int(images.shape[2])

    frame_indices = np.asarray(base_group["frame_indices"][:], dtype=np.int32)
    bbox_norm = np.asarray(base_group["bbox_norm_coords"][:], dtype=np.float64)
    scores = np.asarray(base_group["scores"][:], dtype=np.float32)
    class_ids = np.asarray(base_group["class_ids"][:], dtype=np.int32)
    detection_source = base_group.get("detection_source")
    detection_source_arr = np.asarray(detection_source[:], dtype=np.int8) if detection_source is not None else None

    frame_counts = base_group.get("frame_counts")
    if frame_counts is None:
        frame_counts_arr = _compute_frame_counts(frame_indices, n_frames)
    else:
        frame_counts_arr = np.asarray(frame_counts[:], dtype=np.int32)
        if frame_counts_arr.shape[0] != n_frames:
            n_frames = min(n_frames, frame_counts_arr.shape[0])
            frame_counts_arr = frame_counts_arr[:n_frames]

    if target_frames is not None:
        unique_frames = np.unique(target_frames.astype(np.int32, copy=False))
        review_frames = unique_frames[(unique_frames >= 0) & (unique_frames < n_frames)]
    elif review_all:
        review_frames = np.arange(n_frames, dtype=np.int32)
    else:
        review_frames = np.where(frame_counts_arr == 0)[0].astype(np.int32, copy=False)

    if max_frames is not None:
        review_frames = review_frames[:max_frames]

    if review_frames.size == 0:
        print("No frames to review.")
        return

    detections_by_frame = _group_by_frame(frame_indices)
    manual_changes: Dict[int, Optional[np.ndarray]] = {}

    def _apply_review_status() -> None:
        resolved = resolve_refined_detect_group(
            refined_run, preference=DEFAULT_DETECT_GROUP_PREFERENCE
        )
        reviewer_name = reviewer or os.environ.get("USER") or os.environ.get("USERNAME")
        payload: Dict[str, object] = {
            "state": review_state,
            "method": review_method,
            "intended_use": review_intended_use,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolved_group": resolved.label or resolved.group,
            "preference_chain": list(DEFAULT_DETECT_GROUP_PREFERENCE),
        }
        if reviewer_name:
            payload["reviewer"] = reviewer_name
        if review_notes:
            payload["notes"] = review_notes
        refined_run.attrs["detect_review_status"] = payload
        refined_parent.attrs["detect_review_status_latest"] = refined_run_name
        print(f"Set detect_review_status on refined_detect_runs/{refined_run_name}")

    idx_pos = 0
    current_rect: Optional[Tuple[float, float, float, float]] = None
    selector: Optional[RectangleSelector] = None
    approve_on_exit = False

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(bottom=0.15)
    base_patch = Rectangle((0, 0), 1, 1, fill=False, edgecolor="cyan", linewidth=1.5, linestyle="--")
    manual_patch = Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", linewidth=2)

    base_patch.set_visible(False)
    manual_patch.set_visible(False)

    def load_frame() -> None:
        nonlocal current_rect
        frame = int(review_frames[idx_pos])
        img = images[frame]
        ax.clear()
        ax.imshow(img, cmap="gray")
        ax.set_axis_off()
        ax.add_patch(base_patch)
        ax.add_patch(manual_patch)

        # Base detection (if any) - show highest score
        base_patch.set_visible(False)
        manual_patch.set_visible(False)
        current_rect = None

        base_indices = detections_by_frame.get(frame, np.array([], dtype=np.int32))
        if base_indices.size > 0:
            best_idx = base_indices[np.argmax(scores[base_indices])]
            rect = _norm_to_rect(bbox_norm[best_idx], width, height)
            base_patch.set_bounds(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
            base_patch.set_visible(True)

        if frame in manual_changes and manual_changes[frame] is not None:
            manual_rect = _norm_to_rect(manual_changes[frame], width, height)
            manual_patch.set_bounds(manual_rect[0], manual_rect[1], manual_rect[2] - manual_rect[0], manual_rect[3] - manual_rect[1])
            manual_patch.set_visible(True)
            current_rect = manual_rect

        status = "missing" if base_indices.size == 0 else "has detection"
        if frame in manual_changes:
            status = "manual" if manual_changes[frame] is not None else "cleared"

        if selector is not None:
            selector.clear()
            selector.set_visible(False)

        ax.set_title(
            f"Frame {frame} | {idx_pos + 1}/{len(review_frames)} | {status}",
            fontsize=10,
        )
        fig.canvas.draw_idle()

    def on_select(eclick, erelease):
        nonlocal current_rect
        if eclick.xdata is None or erelease.xdata is None:
            return
        current_rect = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
        rect_norm = _rect_to_norm(current_rect, width, height)
        if rect_norm[2] <= 0 or rect_norm[3] <= 0:
            return
        rect_norm[:2] = np.clip(rect_norm[:2], 0.0, 1.0)
        rect_norm[2:] = np.clip(rect_norm[2:], 0.0, 1.0)
        manual_patch.set_bounds(
            min(eclick.xdata, erelease.xdata),
            min(eclick.ydata, erelease.ydata),
            abs(erelease.xdata - eclick.xdata),
            abs(erelease.ydata - eclick.ydata),
        )
        manual_patch.set_visible(True)
        fig.canvas.draw_idle()
        frame = int(review_frames[idx_pos])
        manual_changes[frame] = rect_norm
        if selector is not None:
            selector.set_visible(False)

    def on_key(event):
        nonlocal idx_pos, approve_on_exit
        key = (event.key or "").lower()
        if key == "n":
            idx_pos = min(idx_pos + 1, len(review_frames) - 1)
            load_frame()
        elif key == "p":
            idx_pos = max(idx_pos - 1, 0)
            load_frame()
        elif key == "c":
            frame = int(review_frames[idx_pos])
            manual_changes[frame] = None
            load_frame()
        elif key == "r":
            frame = int(review_frames[idx_pos])
            if frame in manual_changes:
                del manual_changes[frame]
            load_frame()
        elif key == "a":
            approve_on_exit = True
            plt.close(fig)
        elif key == "q":
            plt.close(fig)
        elif key == "h":
            print("Keys: n=next, p=prev, c=clear detection, r=reset, a=approve, q=quit")

    # Keep a reference; otherwise the selector can be garbage-collected.
    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        interactive=True,
    )
    selector.set_visible(False)

    fig.canvas.mpl_connect("key_press_event", on_key)

    load_frame()
    print("\nManual review controls:")
    print("  - Drag to draw a bounding box on the current frame")
    print("  - n / p: next / previous frame")
    print("  - c: clear detection for this frame")
    print("  - r: reset changes for this frame")
    print("  - a: approve recording (sets detect_review_status)")
    print("  - q: save changes and quit")
    print("  - h: print this help again")
    plt.show()

    if not manual_changes:
        print("No manual changes recorded.")
        if approve_on_exit:
            _apply_review_status()
        return

    # Build updated detection arrays
    frames_to_replace = np.array(sorted(manual_changes.keys()), dtype=np.int32)
    keep_mask = ~np.isin(frame_indices, frames_to_replace)

    kept_frames = frame_indices[keep_mask]
    kept_bboxes = bbox_norm[keep_mask]
    kept_scores = scores[keep_mask]
    kept_class_ids = class_ids[keep_mask]
    kept_source = detection_source_arr[keep_mask] if detection_source_arr is not None else None

    manual_frames: List[int] = []
    manual_bboxes: List[np.ndarray] = []
    manual_scores: List[float] = []
    manual_class_ids: List[int] = []
    manual_source: List[int] = []

    removed_frames = 0
    for frame, rect_norm in manual_changes.items():
        if rect_norm is None:
            removed_frames += 1
            continue
        manual_frames.append(int(frame))
        manual_bboxes.append(rect_norm)
        manual_scores.append(float(manual_score))
        manual_class_ids.append(int(manual_class_id))
        manual_source.append(0)

    if manual_frames:
        manual_frames_arr = np.asarray(manual_frames, dtype=np.int32)
        manual_bboxes_arr = np.stack(manual_bboxes, axis=0).astype(np.float64, copy=False)
        manual_scores_arr = np.asarray(manual_scores, dtype=np.float32)
        manual_class_ids_arr = np.asarray(manual_class_ids, dtype=np.int32)
        manual_source_arr = np.asarray(manual_source, dtype=np.int8)
    else:
        manual_frames_arr = np.empty((0,), dtype=np.int32)
        manual_bboxes_arr = np.empty((0, 4), dtype=np.float64)
        manual_scores_arr = np.empty((0,), dtype=np.float32)
        manual_class_ids_arr = np.empty((0,), dtype=np.int32)
        manual_source_arr = np.empty((0,), dtype=np.int8)

    out_frames = np.concatenate([kept_frames, manual_frames_arr])
    out_bboxes = np.concatenate([kept_bboxes, manual_bboxes_arr])
    out_scores = np.concatenate([kept_scores, manual_scores_arr])
    out_class_ids = np.concatenate([kept_class_ids, manual_class_ids_arr])

    if detection_source_arr is not None:
        out_source = np.concatenate([kept_source, manual_source_arr])
    else:
        out_source = None

    reason = None
    if manual_frames_arr.size > 0:
        kept_reason = np.full(kept_frames.shape[0], "kept", dtype=object)
        manual_reason = np.full(manual_frames_arr.shape[0], "manual_correction", dtype=object)
        reason = np.concatenate([kept_reason, manual_reason])

    if out_frames.size > 0:
        order = np.argsort(out_frames)
        out_frames = out_frames[order]
        out_bboxes = out_bboxes[order]
        out_scores = out_scores[order]
        out_class_ids = out_class_ids[order]
        if out_source is not None:
            out_source = out_source[order]
        if reason is not None:
            reason = reason[order]

    out_counts = _compute_frame_counts(out_frames, n_frames)
    out_retune_id = np.full(out_frames.shape[0], -1, dtype=np.int32)

    metadata = {
        "manual_review_timestamp": datetime.now(timezone.utc).isoformat(),
        "manual_review_frames": int(len(manual_changes)),
        "manual_review_added": int(len(manual_frames)),
        "manual_review_removed": int(removed_frames),
        "source_variant": variant,
        "source_refined_run": refined_run_name,
        "source_detect_run": refined_run.attrs.get("source_detect_run"),
        "manual_score": float(manual_score),
        "manual_class_id": int(manual_class_id),
        "detection_source_type": "manual",
        "detection_source_path": f"{refined_run.path}/{output_group}",
    }

    _write_manual_group(
        refined_run,
        output_group,
        out_frames,
        out_bboxes,
        out_scores,
        out_class_ids,
        out_retune_id,
        out_counts,
        out_source,
        reason,
        metadata,
        overwrite,
    )

    refined_run.attrs["manual_review_latest"] = output_group
    print(f"Saved manual detections to refined_detect_runs/{refined_run_name}/{output_group}")
    if approve_on_exit:
        _apply_review_status()


def _detect_frames_with_params(
    images_ds: np.ndarray,
    background_ds: np.ndarray,
    frame_indices: np.ndarray,
    detect_params: Dict[str, object],
    dish_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    se1 = disk(int(detect_params["se1_radius"]))
    se4 = disk(int(detect_params["se4_radius"]))
    min_area = int(detect_params["min_area"])
    max_area = int(detect_params["max_area"])
    max_fish = int(detect_params.get("max_fish", 20))
    base_thresh = int(detect_params["ds_thresh"])

    all_frames: List[int] = []
    all_bboxes: List[List[float]] = []

    ds_height = int(images_ds.shape[1])
    ds_width = int(images_ds.shape[2])

    for frame_idx in frame_indices:
        img = images_ds[frame_idx]
        diff_ds = np.clip(
            background_ds.astype(np.int16) - img.astype(np.int16),
            0,
            255,
        ).astype(np.uint8)
        if dish_mask is not None:
            diff_ds = np.where(dish_mask > 0, diff_ds, 0)

        current_thresh = base_thresh
        valid_blobs: List[object] = []
        for _ in range(5):
            im_ds = erosion(dilation(erosion(diff_ds >= current_thresh, se1), se4), se1)
            all_blobs = regionprops(label(im_ds))
            valid_blobs = [r for r in all_blobs if min_area <= r.area <= max_area]
            if valid_blobs:
                break
            current_thresh -= 5

        if not valid_blobs:
            continue

        sorted_blobs = sorted(valid_blobs, key=lambda r: r.area, reverse=True)[:max_fish]
        for blob in sorted_blobs:
            min_r, min_c, max_r, max_c = blob.bbox
            center_y, center_x = (min_r + max_r) / 2.0, (min_c + max_c) / 2.0
            height, width = max_r - min_r, max_c - min_c
            center_norm = np.array([center_x / ds_width, center_y / ds_height])
            size_norm = np.array([width / ds_width, height / ds_height])
            all_frames.append(int(frame_idx))
            all_bboxes.append([*center_norm, *size_norm])

    if not all_frames:
        return np.empty((0,), dtype=np.int32), np.empty((0, 4), dtype=np.float64)

    return (
        np.asarray(all_frames, dtype=np.int32),
        np.asarray(all_bboxes, dtype=np.float64),
    )


def _build_retune_dashboard(
    image: np.ndarray,
    background: np.ndarray,
    dish_mask: Optional[np.ndarray],
    detect_params: Dict[str, object],
) -> Tuple[np.ndarray, int, int]:
    ds_thresh = int(detect_params["ds_thresh"])
    se1 = disk(int(detect_params["se1_radius"]))
    se4 = disk(int(detect_params["se4_radius"]))
    min_area = int(detect_params["min_area"])
    max_area = int(detect_params["max_area"])
    max_fish = int(detect_params.get("max_fish", 20))

    diff = np.clip(
        background.astype(np.int16) - image.astype(np.int16),
        0,
        255,
    ).astype(np.uint8)
    if dish_mask is not None:
        diff = np.where(dish_mask > 0, diff, 0)

    used_thresh = ds_thresh
    valid_blobs: List[object] = []
    processed_mask = None
    for _ in range(5):
        processed_mask = erosion(dilation(erosion(diff >= used_thresh, se1), se4), se1)
        all_blobs = regionprops(label(processed_mask))
        valid_blobs = [r for r in all_blobs if min_area <= r.area <= max_area]
        if valid_blobs:
            break
        used_thresh -= 5

    panel_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    panel_diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    panel_mask = cv2.cvtColor((processed_mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    panel_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if valid_blobs:
        sorted_blobs = sorted(valid_blobs, key=lambda r: r.area, reverse=True)[:max_fish]
        for blob in sorted_blobs:
            min_r, min_c, max_r, max_c = blob.bbox
            cv2.rectangle(panel_overlay, (min_c, min_r), (max_c, max_r), (0, 255, 0), 2)

    display_size = (480, 480)
    top_row = np.hstack((cv2.resize(panel_image, display_size), cv2.resize(panel_diff, display_size)))
    bottom_row = np.hstack((cv2.resize(panel_mask, display_size), cv2.resize(panel_overlay, display_size)))
    dashboard = np.vstack((top_row, bottom_row))

    return dashboard, used_thresh, len(valid_blobs)


def run_retune_interactive(
    zarr_path: str,
    refined_run_name: Optional[str] = None,
    variant: Optional[str] = None,
    output_group: str = "manual",
    overwrite: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = "pipeline_config.yaml",
    retune_score: float = 1.0,
    retune_class_id: int = 0,
    target_frames: Optional[np.ndarray] = None,
) -> None:
    console = Console()
    root = zarr.open_group(zarr_path, mode="a")

    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_detect_runs found in archive.")

    refined_run_name = refined_run_name or refined_parent.attrs.get("latest")
    if not refined_run_name or refined_run_name not in refined_parent:
        raise RuntimeError("Refined detect run not found.")

    refined_run = refined_parent[refined_run_name]
    variant = _pick_variant(refined_run, variant)
    base_group, base_name, base_is_manual = _select_retune_base(refined_run, variant, output_group)

    source_detect_run = refined_run.attrs.get("source_detect_run")
    if source_detect_run:
        detect_group = root.get(f"detect_runs/{source_detect_run}")
        if detect_group is not None:
            detection_method = str(detect_group.attrs.get("detection_method", "")).lower()
            pipeline_type = str(detect_group.attrs.get("pipeline_type", "")).lower()
            model_type = str(detect_group.attrs.get("model_type", "")).lower()
            if "yolo" in detection_method or "yolo" in pipeline_type or "yolo" in model_type:
                raise RuntimeError(
                    "Retune is not supported for YOLO detection runs. "
                    "Rerun YOLO detection with updated thresholds/model instead."
                )

    if "raw_video" not in root or "images_ds" not in root["raw_video"]:
        raise RuntimeError("Zarr archive is missing raw_video/images_ds.")

    images = root["raw_video"]["images_ds"]
    n_frames = int(images.shape[0])

    frame_counts = base_group.get("frame_counts")
    if frame_counts is None:
        frame_counts_arr = _compute_frame_counts(base_group["frame_indices"][:], n_frames)
    else:
        frame_counts_arr = np.asarray(frame_counts[:], dtype=np.int32)
        if frame_counts_arr.shape[0] != n_frames:
            n_frames = min(n_frames, frame_counts_arr.shape[0])
            frame_counts_arr = frame_counts_arr[:n_frames]

    if target_frames is not None:
        unique_frames = np.unique(target_frames.astype(np.int32, copy=False))
        retune_frames = unique_frames[(unique_frames >= 0) & (unique_frames < n_frames)]
    else:
        retune_frames = np.where(frame_counts_arr == 0)[0].astype(np.int32, copy=False)
    if max_frames is not None:
        retune_frames = retune_frames[:max_frames]

    if retune_frames.size == 0:
        print("No frames to retune.")
        return

    import yaml

    config: Dict[str, object] = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    detect_params, param_source = get_detection_parameters(root, config, console)

    if "background_runs" not in root:
        raise RuntimeError("Background stage not run. Run background computation first.")
    latest_bg_run = root["background_runs"].attrs.get("latest")
    if not latest_bg_run:
        raise RuntimeError("No background runs found (missing latest background run).")

    background_ds = root[f"background_runs/{latest_bg_run}/background_ds"][:]
    dish_mask = create_dish_mask(detect_params.get("dish_mask", {}), background_ds.shape, console)
    if dish_mask is None:
        console.print("[yellow]No dish mask loaded; using full frame.[/yellow]")
    else:
        console.print("[green]✓ Dish mask loaded for retune UI.[/green]")

    window_name = "Detection Retune (Interactive)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 900)

    current_idx = 0

    def update_idx(val):
        nonlocal current_idx
        current_idx = int(np.clip(val, 0, max(0, len(retune_frames) - 1)))

    def update_param(key: str):
        def _inner(val):
            detect_params[key] = val
        return _inner

    max_failure = len(retune_frames) - 1
    if max_failure >= 1:
        cv2.createTrackbar("Failure", window_name, current_idx, max_failure, update_idx)
    else:
        console.print("[yellow]Single missing frame; failure slider disabled.[/yellow]")
    cv2.createTrackbar("ds_thresh", window_name, int(detect_params["ds_thresh"]), 255, update_param("ds_thresh"))
    cv2.createTrackbar("se1_radius", window_name, int(detect_params["se1_radius"]), 10, update_param("se1_radius"))
    cv2.createTrackbar("se4_radius", window_name, int(detect_params["se4_radius"]), 10, update_param("se4_radius"))
    cv2.createTrackbar("min_area", window_name, int(detect_params["min_area"]), 5000, update_param("min_area"))
    cv2.createTrackbar("max_area", window_name, int(detect_params["max_area"]), 50000, update_param("max_area"))
    cv2.createTrackbar("max_fish", window_name, int(detect_params.get("max_fish", 20)), 50, update_param("max_fish"))

    print("\nDetection Retune (Interactive)")
    print(f"  Zarr: {zarr_path}")
    print(f"  Refined run: {refined_run_name}")
    if base_is_manual:
        print(f"  Base group: {base_name} (incremental)")
    else:
        print(f"  Base group: {base_name}")
    if target_frames is not None:
        print(f"  Targeted frames: {len(retune_frames)}")
    else:
        print(f"  Missing frames: {len(retune_frames)}")
    print("Controls:")
    frame_label = "targeted frames" if target_frames is not None else "missing frames"
    print(f"  - Adjust sliders to test parameters on {frame_label}")
    print("  - Left/Right arrows to step through failures")
    print("  - Press 's' to save tuning to zarr (analysis_metadata)")
    print(f"  - Press 'a' to APPLY retune to remaining {frame_label} (does not update analysis_metadata tuning)")
    print("  - Press 'q' or Esc to quit")

    while True:
        frame_idx = int(retune_frames[current_idx])
        dashboard, used_thresh, blob_count = _build_retune_dashboard(
            images[frame_idx], background_ds, dish_mask, detect_params
        )
        cv2.putText(
            dashboard,
            f"Frame {frame_idx} | Failure {current_idx+1}/{len(retune_frames)} | Used thresh {used_thresh} | blobs {blob_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.imshow(window_name, dashboard)
        cv2.setTrackbarPos("Failure", window_name, current_idx)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == 83:  # Right arrow
            current_idx = min(len(retune_frames) - 1, current_idx + 1)
        elif key == 81:  # Left arrow
            current_idx = max(0, current_idx - 1)
        elif key == ord("s"):
            _save_detection_tuning(root, detect_params, tuned_frame=frame_idx)
            print("✓ Saved detection tuning to analysis_metadata.")
        elif key == ord("a"):
            cv2.destroyAllWindows()
            run_retune_review(
                zarr_path,
                refined_run_name=refined_run_name,
                variant=variant,
                output_group=output_group,
                overwrite=overwrite,
                max_frames=max_frames,
                config_path=config_path,
                retune_score=retune_score,
                retune_class_id=retune_class_id,
                target_frames=retune_frames,
                use_full_res=False,
                detect_params_override=dict(detect_params),
                param_source_override="retune_ui",
            )
            return

    cv2.destroyAllWindows()

def run_retune_review(
    zarr_path: str,
    refined_run_name: Optional[str] = None,
    variant: Optional[str] = None,
    output_group: str = "manual",
    overwrite: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = "pipeline_config.yaml",
    retune_score: float = 1.0,
    retune_class_id: int = 0,
    target_frames: Optional[np.ndarray] = None,
    use_full_res: bool = False,
    detect_params_override: Optional[Dict[str, object]] = None,
    param_source_override: Optional[str] = None,
) -> None:
    console = Console()
    root = zarr.open_group(zarr_path, mode="a")

    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_detect_runs found in archive.")

    refined_run_name = refined_run_name or refined_parent.attrs.get("latest")
    if not refined_run_name or refined_run_name not in refined_parent:
        raise RuntimeError("Refined detect run not found.")

    refined_run = refined_parent[refined_run_name]
    variant = _pick_variant(refined_run, variant)
    base_group, _, _ = _select_retune_base(refined_run, variant, output_group)

    source_detect_run = refined_run.attrs.get("source_detect_run")
    if source_detect_run:
        detect_group = root.get(f"detect_runs/{source_detect_run}")
        if detect_group is not None:
            detection_method = str(detect_group.attrs.get("detection_method", "")).lower()
            pipeline_type = str(detect_group.attrs.get("pipeline_type", "")).lower()
            model_type = str(detect_group.attrs.get("model_type", "")).lower()
            if "yolo" in detection_method or "yolo" in pipeline_type or "yolo" in model_type:
                raise RuntimeError(
                    "Retune is not supported for YOLO detection runs. "
                    "Rerun YOLO detection with updated thresholds/model instead."
                )

    if "raw_video" not in root:
        raise RuntimeError("Zarr archive is missing raw_video group.")

    if use_full_res or "images_ds" not in root["raw_video"]:
        images = root["raw_video"]["images_full"]
    else:
        images = root["raw_video"]["images_ds"]

    n_frames = int(images.shape[0])

    frame_counts = base_group.get("frame_counts")
    if frame_counts is None:
        frame_counts_arr = _compute_frame_counts(base_group["frame_indices"][:], n_frames)
    else:
        frame_counts_arr = np.asarray(frame_counts[:], dtype=np.int32)
        if frame_counts_arr.shape[0] != n_frames:
            n_frames = min(n_frames, frame_counts_arr.shape[0])
            frame_counts_arr = frame_counts_arr[:n_frames]

    missing_before = int(np.sum(frame_counts_arr == 0))
    if target_frames is not None:
        unique_frames = np.unique(target_frames.astype(np.int32, copy=False))
        retune_frames = unique_frames[(unique_frames >= 0) & (unique_frames < n_frames)]
    else:
        retune_frames = np.where(frame_counts_arr == 0)[0].astype(np.int32, copy=False)
    if max_frames is not None:
        retune_frames = retune_frames[:max_frames]

    if retune_frames.size == 0:
        print("No frames to retune.")
        return

    # Resolve detection parameters (config defaults + zarr tuning), unless overridden.
    if detect_params_override is None:
        import yaml

        config: Dict[str, object] = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

        detect_params, param_source = get_detection_parameters(root, config, console)
    else:
        detect_params = detect_params_override
        param_source = param_source_override or "override"

    if "background_runs" not in root:
        raise RuntimeError("Background stage not run. Run background computation first.")
    latest_bg_run = root["background_runs"].attrs.get("latest")
    if not latest_bg_run:
        raise RuntimeError("No background runs found (missing latest background run).")

    background_ds = root[f"background_runs/{latest_bg_run}/background_ds"][:]
    dish_mask = create_dish_mask(detect_params.get("dish_mask", {}), background_ds.shape, console)

    retune_id = _get_or_create_retune_id(refined_run, detect_params)

    if target_frames is not None:
        console.print(f"[cyan]Retuning frames:[/cyan] {len(retune_frames)} targeted")
    else:
        console.print(f"[cyan]Retuning frames:[/cyan] {len(retune_frames)} missing detections")
    console.print(f"[cyan]Retune base:[/cyan] {base_group.path}")
    console.print(f"[cyan]Using parameters from:[/cyan] {param_source}")

    new_frames, new_bboxes = _detect_frames_with_params(
        images,
        background_ds,
        retune_frames,
        detect_params,
        dish_mask,
    )

    # Build updated detection arrays
    frame_indices = np.asarray(base_group["frame_indices"][:], dtype=np.int32)
    bbox_norm = np.asarray(base_group["bbox_norm_coords"][:], dtype=np.float64)
    scores = np.asarray(base_group["scores"][:], dtype=np.float32)
    class_ids = np.asarray(base_group["class_ids"][:], dtype=np.int32)
    detection_source = base_group.get("detection_source")
    detection_source_arr = np.asarray(detection_source[:], dtype=np.int8) if detection_source is not None else None
    base_retune_id = base_group.get("retune_id")
    base_retune_id_arr = np.asarray(base_retune_id[:], dtype=np.int32) if base_retune_id is not None else None
    base_reason = base_group.get("reason")
    base_reason_arr = np.asarray(base_reason[:], dtype=object) if base_reason is not None else None

    frames_to_replace = retune_frames
    keep_mask = ~np.isin(frame_indices, frames_to_replace)

    kept_frames = frame_indices[keep_mask]
    kept_bboxes = bbox_norm[keep_mask]
    kept_scores = scores[keep_mask]
    kept_class_ids = class_ids[keep_mask]
    kept_source = detection_source_arr[keep_mask] if detection_source_arr is not None else None

    if new_frames.size > 0:
        retune_scores = np.full((new_frames.shape[0],), float(retune_score), dtype=np.float32)
        retune_class_ids = np.full((new_frames.shape[0],), int(retune_class_id), dtype=np.int32)
        retune_source = np.zeros((new_frames.shape[0],), dtype=np.int8)
        retune_ids = np.full((new_frames.shape[0],), int(retune_id), dtype=np.int32)
    else:
        retune_scores = np.empty((0,), dtype=np.float32)
        retune_class_ids = np.empty((0,), dtype=np.int32)
        retune_source = np.empty((0,), dtype=np.int8)
        retune_ids = np.empty((0,), dtype=np.int32)

    out_frames = np.concatenate([kept_frames, new_frames])
    out_bboxes = np.concatenate([kept_bboxes, new_bboxes])
    out_scores = np.concatenate([kept_scores, retune_scores])
    out_class_ids = np.concatenate([kept_class_ids, retune_class_ids])
    kept_retune_id = (
        base_retune_id_arr[keep_mask]
        if base_retune_id_arr is not None
        else np.full(kept_frames.shape[0], -1, dtype=np.int32)
    )
    out_retune_id = np.concatenate([kept_retune_id, retune_ids])

    if detection_source_arr is not None:
        out_source = np.concatenate([kept_source, retune_source])
    else:
        out_source = None

    reason = None
    if out_frames.size > 0:
        if base_reason_arr is not None:
            kept_reason = base_reason_arr[keep_mask]
        else:
            kept_reason = np.full(kept_frames.shape[0], "kept", dtype=object)
        retune_reason = np.full(new_frames.shape[0], "retune", dtype=object)
        reason = np.concatenate([kept_reason, retune_reason])

    if out_frames.size > 0:
        order = np.argsort(out_frames)
        out_frames = out_frames[order]
        out_bboxes = out_bboxes[order]
        out_scores = out_scores[order]
        out_class_ids = out_class_ids[order]
        out_retune_id = out_retune_id[order]
        if out_source is not None:
            out_source = out_source[order]
        if reason is not None:
            reason = reason[order]

    out_counts = _compute_frame_counts(out_frames, n_frames)
    missing_after = int(np.sum(out_counts == 0))
    remaining_targeted = int(np.sum(out_counts[retune_frames] == 0)) if retune_frames.size else 0
    corrected = missing_before - missing_after

    metadata = {
        "retune_timestamp": datetime.now(timezone.utc).isoformat(),
        "retune_frames_requested": int(len(retune_frames)),
        "retune_detections_added": int(new_frames.shape[0]),
        "source_variant": variant,
        "source_refined_run": refined_run_name,
        "source_detect_run": refined_run.attrs.get("source_detect_run"),
        "retune_score": float(retune_score),
        "retune_class_id": int(retune_class_id),
        "retune_parameters": detect_params,
        "retune_parameter_source": param_source,
        "retune_base_group": base_group.path,
        "detection_source_type": "retune",
        "detection_source_path": f"{refined_run.path}/{output_group}",
    }

    _write_manual_group(
        refined_run,
        output_group,
        out_frames,
        out_bboxes,
        out_scores,
        out_class_ids,
        out_retune_id,
        out_counts,
        out_source,
        reason,
        metadata,
        overwrite,
    )

    refined_run.attrs["manual_review_latest"] = output_group
    print(f"Saved retuned detections to refined_detect_runs/{refined_run_name}/{output_group}")
    console.print("[green]Retune summary:[/green]")
    console.print(f"  Missing before: {missing_before}")
    console.print(f"  Retune detections added: {int(new_frames.shape[0])}")
    console.print(f"  Targeted this run: {len(retune_frames)}")
    console.print(f"  Corrected this run: {corrected}")
    console.print(f"  Still missing (targeted): {remaining_targeted}")
    console.print(f"  Missing after (overall): {missing_after}")

def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Manual review for detection runs (draw bounding boxes on missing frames)."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr directory.")
    parser.add_argument(
        "--refined-run",
        help="Refined detect run to review (defaults to latest).",
    )
    parser.add_argument(
        "--variant",
        choices=["filtered", "interpolated"],
        help="Refined variant to review (defaults to interpolated if available).",
    )
    parser.add_argument(
        "--output-group",
        default="manual",
        help="Group name to write manual detections under refined run (default: manual).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output group.")
    parser.add_argument("--all", action="store_true", help="Review all frames (not just missing detections).")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit number of frames to review (useful for quick passes).",
    )
    parser.add_argument(
        "--manual-score",
        type=float,
        default=1.0,
        help="Score value to assign to manual detections.",
    )
    parser.add_argument(
        "--manual-class-id",
        type=int,
        default=0,
        help="Class id to assign to manual detections.",
    )
    parser.add_argument(
        "--use-full-res",
        action="store_true",
        help="Use full-resolution frames instead of downsampled.",
    )
    parser.add_argument(
        "--retune",
        action="store_true",
        help="Retune missing detections using tuned parameters.",
    )
    parser.add_argument(
        "--retune-ui",
        action="store_true",
        help="Interactive retune (slider UI) for missing detections.",
    )
    parser.add_argument(
        "--config",
        default="pipeline_config.yaml",
        help="Config file to source defaults for retune (default: pipeline_config.yaml).",
    )
    parser.add_argument(
        "--retune-score",
        type=float,
        default=1.0,
        help="Score to assign to retuned detections (default: 1.0).",
    )
    parser.add_argument(
        "--retune-class-id",
        type=int,
        default=0,
        help="Class id to assign to retuned detections (default: 0).",
    )
    parser.add_argument(
        "--review-state",
        default="approved",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Review state to set when approving in manual review (default: approved).",
    )
    parser.add_argument(
        "--review-method",
        default="manual",
        choices=["manual", "algorithmic", "hybrid", "spotcheck"],
        help="Review method label (default: manual).",
    )
    parser.add_argument(
        "--review-intended-use",
        default="training",
        choices=["training", "full_recording"],
        help="Intended use label for review status (default: training).",
    )
    parser.add_argument("--reviewer", help="Reviewer name (defaults to $USER).")
    parser.add_argument("--review-notes", help="Optional review notes.")
    parser.add_argument(
        "--frames",
        type=str,
        help="Comma/space-separated frame indices (or path to a text/JSON list) to target.",
    )

    args = parser.parse_args(argv)
    target_frames = _parse_frames_arg(args.frames)
    if args.frames and args.all:
        print("Note: --frames provided; ignoring --all.")

    if args.retune_ui:
        run_retune_interactive(
            str(args.zarr_path),
            refined_run_name=args.refined_run,
            variant=args.variant,
            output_group=args.output_group,
            overwrite=args.overwrite,
            max_frames=args.max_frames,
            config_path=args.config,
            retune_score=args.retune_score,
            retune_class_id=args.retune_class_id,
            target_frames=target_frames,
        )
    elif args.retune:
        run_retune_review(
            str(args.zarr_path),
            refined_run_name=args.refined_run,
            variant=args.variant,
            output_group=args.output_group,
            overwrite=args.overwrite,
            max_frames=args.max_frames,
            config_path=args.config,
            retune_score=args.retune_score,
            retune_class_id=args.retune_class_id,
            target_frames=target_frames,
            use_full_res=args.use_full_res,
        )
    else:
        run_manual_review(
            str(args.zarr_path),
            refined_run_name=args.refined_run,
            variant=args.variant,
            output_group=args.output_group,
            overwrite=args.overwrite,
            review_all=args.all,
            max_frames=args.max_frames,
            target_frames=target_frames,
            manual_score=args.manual_score,
            manual_class_id=args.manual_class_id,
            use_full_res=args.use_full_res,
            review_state=args.review_state,
            review_method=args.review_method,
            review_intended_use=args.review_intended_use,
            reviewer=args.reviewer,
            review_notes=args.review_notes,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
