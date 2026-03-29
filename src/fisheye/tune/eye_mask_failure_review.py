"""Manual eye mask failure review and correction UI (OpenCV)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any

# Limit threading BEFORE importing numpy/cv2/skimage to avoid saturating all CPU cores.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import cv2

# Keep OpenCV from spinning up many worker threads in review UIs.
try:
    cv2_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "2")))
except (TypeError, ValueError):
    cv2_threads = 2
cv2.setNumThreads(cv2_threads)

import numpy as np
import zarr
from skimage import measure
from skimage.measure import EllipseModel


_DEFAULT_SUCCESS_MIN_EYE_AREA_PX = 50.0


def _get_latest_refined_run(root: zarr.Group) -> str:
    parent = root.get("refined_eye_masks_runs")
    if parent is None:
        raise RuntimeError("No refined_eye_masks_runs found in archive.")
    latest = parent.attrs.get("latest")
    if not latest:
        raise RuntimeError("No refined eye mask runs recorded.")
    return latest


def _get_sep_limits(root: zarr.Group, refined: zarr.Group) -> tuple[Optional[float], Optional[float]]:
    analysis = root.get("analysis_metadata")
    if analysis is not None:
        tuning = analysis.attrs.get("eye_mask_tuning")
        if isinstance(tuning, dict):
            params = tuning.get("tuned_parameters", {})
            min_sep = params.get("min_eye_separation")
            max_sep = params.get("max_eye_separation")
            return (
                float(min_sep) if min_sep is not None else None,
                float(max_sep) if max_sep is not None else None,
            )

    source_run = refined.attrs.get("source_eye_masks_run")
    if source_run and "eye_masks_runs" in root and source_run in root["eye_masks_runs"]:
        src = root["eye_masks_runs"][source_run]
        min_sep = src.attrs.get("min_eye_separation")
        max_sep = src.attrs.get("max_eye_separation")
        return (
            float(min_sep) if min_sep is not None else None,
            float(max_sep) if max_sep is not None else None,
        )
    return None, None


def _positive_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


def _resolve_success_min_eye_area_px(refined: zarr.Group) -> Optional[float]:
    raw = refined.attrs.get("success_min_eye_area_px")
    if raw is None:
        raw = refined.attrs.get("min_eye_area_success_px")
    threshold = _positive_float(raw)
    if threshold is not None:
        return threshold
    return float(_DEFAULT_SUCCESS_MIN_EYE_AREA_PX)


def _load_area_refined(refined: zarr.Group) -> Optional[np.ndarray]:
    masks_arr = refined.get("masks_roi")
    shape = getattr(masks_arr, "shape", None) if masks_arr is not None else None
    if isinstance(shape, tuple) and len(shape) == 4 and int(shape[1]) >= 2:
        total = int(shape[0])
        if total <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        chunks = getattr(masks_arr, "chunks", None)
        if isinstance(chunks, tuple) and chunks:
            step = max(1, int(chunks[0]))
        elif isinstance(chunks, int):
            step = max(1, int(chunks))
        else:
            step = 256
        out = np.zeros((total, 2), dtype=np.float32)
        for start in range(0, total, step):
            stop = min(total, start + step)
            block = np.asarray(masks_arr[start:stop], dtype=np.uint8)
            if block.ndim != 4 or int(block.shape[1]) < 2:
                return None
            area = np.sum((block[:, :2] > 0).reshape(block.shape[0], 2, -1), axis=2, dtype=np.int64)
            out[start:stop, :] = np.asarray(area, dtype=np.float32)
        return out

    metrics = refined.get("metrics")
    if not isinstance(metrics, zarr.Group):
        return None
    area_arr = metrics.get("area_refined")
    if area_arr is None:
        return None
    area_refined = np.asarray(area_arr[:], dtype=np.float32)
    if area_refined.ndim != 2 or area_refined.shape[1] < 2:
        return None
    return area_refined[:, :2]


def _compute_success_mask(
    ellipse_success: np.ndarray,
    eye_separation: np.ndarray,
    min_sep: Optional[float],
    max_sep: Optional[float],
    area_refined: Optional[np.ndarray],
    min_eye_area_px: Optional[float],
) -> np.ndarray:
    pair_success = np.all(ellipse_success, axis=1)
    sep_ok = np.ones_like(pair_success, dtype=bool)
    if eye_separation.size:
        sep_ok = np.isfinite(eye_separation)
        if min_sep is not None:
            sep_ok &= eye_separation >= float(min_sep)
        if max_sep is not None:
            sep_ok &= eye_separation <= float(max_sep)

    area_ok = np.ones_like(pair_success, dtype=bool)
    threshold = _positive_float(min_eye_area_px) if min_eye_area_px is not None else None
    if threshold is not None and area_refined is not None and area_refined.shape[0] == pair_success.shape[0]:
        finite = np.all(np.isfinite(area_refined[:, :2]), axis=1)
        area_ok = finite & np.all(area_refined[:, :2] >= float(threshold), axis=1)
    return pair_success & sep_ok & area_ok


def _load_failure_indices(
    refined: zarr.Group,
    min_sep: Optional[float],
    max_sep: Optional[float],
) -> np.ndarray:
    ellipse_success = np.asarray(refined["ellipse_success"][:], dtype=bool)
    eye_separation = np.asarray(refined["eye_separation"][:], dtype=np.float32)
    area_refined = _load_area_refined(refined)
    min_eye_area_px = _resolve_success_min_eye_area_px(refined)
    success_mask = _compute_success_mask(
        ellipse_success,
        eye_separation,
        min_sep,
        max_sep,
        area_refined,
        min_eye_area_px,
    )
    return np.where(~success_mask)[0].astype("i4", copy=False)


def _load_frame_flags(path: Path) -> Dict[str, list[Dict[str, Optional[int]]]]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to load frame flags from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Frame flag file must contain a JSON object: {path}")

    parsed: Dict[str, list[Dict[str, Optional[int]]]] = {}
    for key, value in data.items():
        entries: list[Dict[str, Optional[int]]] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    frame_val = item.get("frame_idx")
                    roi_val = item.get("roi_idx")
                    try:
                        frame_idx = int(frame_val) if frame_val is not None else None
                    except (TypeError, ValueError):
                        frame_idx = None
                    try:
                        roi_idx = int(roi_val) if roi_val is not None else None
                    except (TypeError, ValueError):
                        roi_idx = None
                    if frame_idx is not None or roi_idx is not None:
                        entries.append({"frame_idx": frame_idx, "roi_idx": roi_idx})
                else:
                    try:
                        frame_idx = int(item)
                    except (TypeError, ValueError):
                        continue
                    entries.append({"frame_idx": frame_idx, "roi_idx": None})
        parsed[str(key)] = entries
    return parsed


def _candidate_flag_keys(zarr_path: str) -> list[str]:
    raw = str(zarr_path)
    expanded = Path(raw).expanduser()
    candidates = {raw, str(expanded)}
    try:
        candidates.add(str(expanded.resolve(strict=False)))
    except Exception:
        pass
    return list(candidates)


def _collect_flagged_roi_indices(
    *,
    flag_path: Path,
    zarr_path: str,
    total_rois: int,
    frame_indices: Optional[np.ndarray],
) -> np.ndarray:
    payload = _load_frame_flags(flag_path)
    if not payload:
        return np.zeros((0,), dtype=np.int32)

    entries: list[Dict[str, Optional[int]]] = []
    for key in _candidate_flag_keys(zarr_path):
        value = payload.get(key)
        if isinstance(value, list):
            entries.extend(value)
    if not entries:
        return np.zeros((0,), dtype=np.int32)

    roi_set: set[int] = set()
    for entry in entries:
        roi_idx = entry.get("roi_idx")
        frame_idx = entry.get("frame_idx")

        if roi_idx is not None and 0 <= int(roi_idx) < int(total_rois):
            roi_set.add(int(roi_idx))
        if frame_idx is not None and frame_indices is not None:
            matches = np.where(frame_indices == int(frame_idx))[0]
            for idx in matches.tolist():
                if 0 <= int(idx) < int(total_rois):
                    roi_set.add(int(idx))

    if not roi_set:
        return np.zeros((0,), dtype=np.int32)
    return np.asarray(sorted(roi_set), dtype=np.int32)


def _combine_review_indices(
    failure_indices: np.ndarray,
    pending_flagged: set[int],
) -> np.ndarray:
    if failure_indices.size > 0 and pending_flagged:
        flagged_arr = np.asarray(sorted(pending_flagged), dtype=np.int32)
        return np.unique(np.concatenate([failure_indices, flagged_arr])).astype("i4", copy=False)
    if failure_indices.size > 0:
        return failure_indices.astype("i4", copy=False)
    if pending_flagged:
        return np.asarray(sorted(pending_flagged), dtype=np.int32).astype("i4", copy=False)
    return np.zeros((0,), dtype=np.int32)


def _next_review_position(
    failures: np.ndarray,
    previous_roi_idx: int,
    previous_pos: int,
) -> int:
    if failures.size <= 1:
        return 0
    matches = np.where(failures == int(previous_roi_idx))[0]
    if matches.size == 0:
        # Current ROI was cleared; continue forward in circular order.
        return int(previous_pos) % int(failures.size)
    current_idx = int(matches[0])
    return (current_idx + 1) % int(failures.size)


def _merge_reason(existing: str, tags: Sequence[str]) -> str:
    existing_tags = [tag for tag in existing.split("|") if tag]
    merged = sorted(set(existing_tags + list(tags)))
    return "|".join(merged) if merged else "clean"


def _apply_review_status(
    refined_parent: zarr.Group,
    refined_run: str,
    refined: zarr.Group,
    *,
    state: str,
    method: str,
    intended_use: str,
    reviewer: Optional[str],
    notes: Optional[str],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "state": state,
        "method": method,
        "intended_use": intended_use,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if reviewer:
        payload["reviewer"] = reviewer
    if notes:
        payload["notes"] = notes

    for key in ("source_eye_masks_run", "source_keypoints_run", "source_keypoint_group"):
        value = refined.attrs.get(key)
        if value is not None:
            payload[key] = value

    refined.attrs["eye_mask_review_status"] = payload
    refined_parent.attrs["eye_mask_review_status_latest"] = refined_run
    return payload


def _largest_region(mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
    labeled = measure.label(mask)
    if labeled.max() == 0:
        return None, None
    regions = measure.regionprops(labeled)
    if not regions:
        return None, None
    region = max(regions, key=lambda r: r.area)
    region_mask = labeled == region.label
    cy, cx = region.centroid
    return region_mask, (float(cx), float(cy))


def _fit_ellipse(mask: np.ndarray) -> Tuple[np.ndarray, bool, Optional[np.ndarray], Optional[Tuple[float, float]]]:
    if mask is None or mask.size == 0 or np.count_nonzero(mask) == 0:
        return np.full(5, np.nan, dtype=np.float32), False, None, None

    region_mask, centroid = _largest_region(mask.astype(bool))
    if region_mask is None:
        return np.full(5, np.nan, dtype=np.float32), False, None, None

    contour = measure.find_contours(region_mask.astype(float), 0.5)
    if not contour:
        return np.full(5, np.nan, dtype=np.float32), False, None, centroid
    best = max(contour, key=lambda c: c.shape[0])
    if best.shape[0] < 5:
        return np.full(5, np.nan, dtype=np.float32), False, None, centroid

    contour_xy = best[:, ::-1]
    ellipse_model = EllipseModel()
    if not ellipse_model.estimate(contour_xy) or ellipse_model.params is None:
        return np.full(5, np.nan, dtype=np.float32), False, contour_xy.astype(np.float32), centroid

    xc, yc, a, b, theta = ellipse_model.params
    if a < b:
        a, b = b, a
        theta = theta + np.pi / 2

    params = np.array(
        [float(xc), float(yc), float(a * 2), float(b * 2), float(np.degrees(theta))],
        dtype=np.float32,
    )
    return params, True, contour_xy.astype(np.float32), centroid


MASK_COLORS = [(0, 255, 255), (255, 0, 255)]  # left: yellow, right: pink
DISPLAY_SCALE = 3.0


def _overlay_masks(base: np.ndarray, masks: Sequence[np.ndarray]) -> np.ndarray:
    display = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = display.copy()
    for idx, mask in enumerate(masks):
        if mask is None:
            continue
        overlay[mask > 0] = MASK_COLORS[idx]
    return cv2.addWeighted(overlay, 0.4, display, 0.6, 0)


def _draw_ellipse(display: np.ndarray, params: np.ndarray, color: Tuple[int, int, int]) -> None:
    if params is None or not np.all(np.isfinite(params[:4])):
        return
    cx, cy, major, minor, angle = params
    if major <= 0 or minor <= 0:
        return
    center = (int(round(cx)), int(round(cy)))
    axes = (int(round(major / 2)), int(round(minor / 2)))
    cv2.ellipse(display, center, axes, float(angle), 0, 360, color, 1)


def _update_contour_arrays(
    refined: zarr.Group,
    roi_idx: int,
    contour: Optional[np.ndarray],
    *,
    side: str,
) -> None:
    ptr_name = f"contour_{side}_ptr"
    len_name = f"contour_{side}_len"
    cont_name = f"contours_{side}"

    if ptr_name not in refined or len_name not in refined or cont_name not in refined:
        return

    ptr_arr = refined[ptr_name]
    len_arr = refined[len_name]
    cont_arr = refined[cont_name]

    if contour is None or contour.size == 0:
        ptr_arr[roi_idx] = -1
        len_arr[roi_idx] = 0
        return

    contour = np.asarray(contour, dtype=np.float32)
    n_points = contour.shape[0]
    existing_ptr = int(ptr_arr[roi_idx])
    existing_len = int(len_arr[roi_idx])

    if existing_ptr >= 0 and existing_len >= n_points:
        cont_arr[existing_ptr:existing_ptr + n_points] = contour
        len_arr[roi_idx] = n_points
        return

    current_len = cont_arr.shape[0]
    new_len = current_len + n_points
    try:
        cont_arr.resize((new_len, 2))
    except Exception:
        old = cont_arr[:]
        del refined[cont_name]
        cont_arr = refined.create_array(
            cont_name,
            shape=(new_len, 2),
            chunks=(max(1, min(4096, new_len)), 2),
            dtype=old.dtype,
            overwrite=True,
        )
        cont_arr[:old.shape[0]] = old
    cont_arr[current_len:new_len] = contour
    ptr_arr[roi_idx] = current_len
    len_arr[roi_idx] = n_points


def launch_review(
    zarr_path: str,
    *,
    refined_run: Optional[str] = None,
    crop_run: Optional[str] = None,
    frame_flag_file: Optional[str] = None,
    review_state: str = "approved",
    review_method: str = "manual",
    review_intended_use: str = "training",
    reviewer: Optional[str] = None,
    review_notes: Optional[str] = None,
) -> None:
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    refined_parent = root.get("refined_eye_masks_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_eye_masks_runs found.")

    refined_run = refined_run or _get_latest_refined_run(root)
    if refined_run not in refined_parent:
        raise RuntimeError(f"Refined run '{refined_run}' not found.")
    refined = refined_parent[refined_run]

    crop_run = crop_run or refined.attrs.get("source_crop_run") or root["crop_runs"].attrs.get("latest")
    if not crop_run or "crop_runs" not in root or crop_run not in root["crop_runs"]:
        raise RuntimeError("Crop run not found for eye mask review.")
    crop_group = root["crop_runs"][crop_run]

    roi_images = crop_group["roi_images"]
    masks_arr = refined["masks_roi"]
    edit_applied_arr = refined.get("edit_applied")
    ellipse_params_arr = refined["ellipse_params"]
    ellipse_success_arr = refined["ellipse_success"]
    eye_separation_arr = refined["eye_separation"]

    metrics_group = refined.get("metrics")
    reason_arr = metrics_group.get("reason") if isinstance(metrics_group, zarr.Group) else None

    min_sep, max_sep = _get_sep_limits(root, refined)
    frame_indices = (
        np.asarray(crop_group["frame_indices"][:], dtype=np.int64)
        if "frame_indices" in crop_group
        else None
    )
    failure_indices = _load_failure_indices(refined, min_sep, max_sep)
    flagged_indices = np.zeros((0,), dtype=np.int32)
    flag_path: Optional[Path] = Path(frame_flag_file).expanduser() if frame_flag_file else None
    if flag_path is not None:
        try:
            flagged_indices = _collect_flagged_roi_indices(
                flag_path=flag_path,
                zarr_path=str(zarr_path),
                total_rois=int(roi_images.shape[0]),
                frame_indices=frame_indices,
            )
        except RuntimeError as exc:
            print(f"Warning: failed to load frame flags ({exc}).")
            flagged_indices = np.zeros((0,), dtype=np.int32)

    if failure_indices.size > 0 and flagged_indices.size > 0:
        failures = np.unique(np.concatenate([failure_indices, flagged_indices])).astype("i4", copy=False)
    elif failure_indices.size > 0:
        failures = failure_indices.astype("i4", copy=False)
    else:
        failures = flagged_indices.astype("i4", copy=False)
    pending_flagged: set[int] = {int(idx) for idx in flagged_indices.tolist()}

    if failures.size == 0:
        print("No failures or flagged ROIs to review.")
        return

    window_name = "Eye Mask Failure Review"
    window_flags = cv2.WINDOW_AUTOSIZE
    if hasattr(cv2, "WINDOW_GUI_NORMAL"):
        window_flags |= cv2.WINDOW_GUI_NORMAL
    elif hasattr(cv2, "WINDOW_NORMAL"):
        window_flags |= cv2.WINDOW_NORMAL
    cv2.namedWindow(window_name, window_flags)
    if hasattr(cv2, "WND_PROP_AUTOSIZE"):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 1)

    current_pos = 0
    active_eye = 0
    brush_radius = 6
    drawing = False
    erase_mode = False
    edit_masks = [None, None]
    original_masks = [None, None]
    roi_img = None
    roi_idx = int(failures[current_pos])
    display_layout = {"roi": (0, 0, 0, 0), "edit": (0, 0, 0, 0)}
    display_scale = max(1.0, float(DISPLAY_SCALE))
    cursor_pos: Optional[Tuple[str, int, int]] = None

    def load_current_roi() -> None:
        nonlocal roi_idx, roi_img, edit_masks, original_masks, active_eye
        roi_idx = int(failures[current_pos])
        roi_img = np.asarray(roi_images[roi_idx])
        left = np.asarray(masks_arr[roi_idx, 0]).astype(np.uint8)
        right = np.asarray(masks_arr[roi_idx, 1]).astype(np.uint8)
        original_masks = [left.copy(), right.copy()]
        edit_masks = [left.copy(), right.copy()]
        active_eye = 0

    def set_active_eye(idx: int) -> None:
        nonlocal active_eye
        active_eye = int(idx)

    def update_display() -> None:
        if roi_img is None:
            return

        left_params, left_success, left_contour, left_centroid = _fit_ellipse(edit_masks[0])
        right_params, right_success, right_contour, right_centroid = _fit_ellipse(edit_masks[1])

        overlay = _overlay_masks(roi_img, edit_masks)
        _draw_ellipse(overlay, left_params, (255, 255, 0))
        _draw_ellipse(overlay, right_params, (0, 255, 255))

        if left_centroid is not None and right_centroid is not None:
            sep = float(
                np.hypot(
                    left_centroid[0] - right_centroid[0],
                    left_centroid[1] - right_centroid[1],
                )
            )
        else:
            sep = float("nan")

        status = "PAIR OK" if (left_success and right_success) else "INCOMPLETE"
        if np.isfinite(sep):
            if min_sep is not None and sep < min_sep:
                status = "TOO CLOSE"
            elif max_sep is not None and sep > max_sep:
                status = "TOO FAR"

        header = overlay.copy()
        cv2.putText(
            header,
            f"Failure {current_pos + 1}/{len(failures)}  ROI {roi_idx}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            header,
            f"Active eye: {'Left' if active_eye == 0 else 'Right'}  Brush: {brush_radius}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            header,
            f"Status: {status}  Separation: {sep:.1f}px",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        edit_base = cv2.cvtColor(roi_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        edit_overlay = edit_base.copy()
        edit_overlay[edit_masks[active_eye] > 0] = MASK_COLORS[active_eye]
        edit_panel = cv2.addWeighted(edit_overlay, 0.4, edit_base, 0.6, 0)
        cv2.putText(
            edit_panel,
            "Mask Editor (draw here or on ROI)",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        spacer = np.full((roi_img.shape[0], 10, 3), 10, dtype=np.uint8)
        combined = np.hstack([header, spacer, edit_panel])

        display_layout["roi"] = (0, 0, header.shape[1], header.shape[0])
        display_layout["edit"] = (
            header.shape[1] + spacer.shape[1],
            0,
            edit_panel.shape[1],
            edit_panel.shape[0],
        )

        if display_scale != 1.0:
            scaled = cv2.resize(
                combined,
                None,
                fx=display_scale,
                fy=display_scale,
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow(window_name, scaled)
        else:
            if cursor_pos is not None:
                panel, cx, cy = cursor_pos
                if panel == "roi":
                    base_x, base_y, panel_w, panel_h = display_layout["roi"]
                else:
                    base_x, base_y, panel_w, panel_h = display_layout["edit"]
                x1 = max(base_x, base_x + cx - brush_radius)
                y1 = max(base_y, base_y + cy - brush_radius)
                x2 = min(base_x + panel_w - 1, base_x + cx + brush_radius)
                y2 = min(base_y + panel_h - 1, base_y + cy + brush_radius)
                cv2.rectangle(combined, (x1, y1), (x2, y2), MASK_COLORS[active_eye], 1)

        if display_scale != 1.0:
            scaled = cv2.resize(
                combined,
                None,
                fx=display_scale,
                fy=display_scale,
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow(window_name, scaled)
        else:
            cv2.imshow(window_name, combined)

    def save_current() -> None:
        existing_masks = np.asarray(masks_arr[roi_idx], dtype=np.uint8)
        left_params, left_success, left_contour, left_centroid = _fit_ellipse(edit_masks[0])
        right_params, right_success, right_contour, right_centroid = _fit_ellipse(edit_masks[1])

        if left_centroid is not None and right_centroid is not None:
            separation = float(
                np.hypot(
                    left_centroid[0] - right_centroid[0],
                    left_centroid[1] - right_centroid[1],
                )
            )
        else:
            separation = float("nan")

        reject_reason = None
        if not left_success and not right_success:
            reject_reason = "incomplete"
        elif not left_success:
            reject_reason = "left_empty"
        elif not right_success:
            reject_reason = "right_empty"
        else:
            overlap = np.logical_and(edit_masks[0] > 0, edit_masks[1] > 0).any()
            if overlap:
                reject_reason = "overlap"
            elif np.isfinite(separation):
                if min_sep is not None and separation < min_sep:
                    reject_reason = "too_close"
                elif max_sep is not None and separation > max_sep:
                    reject_reason = "too_far"

        next_masks = np.asarray(edit_masks, dtype=np.uint8)
        changed_channels = np.any(existing_masks != next_masks, axis=(1, 2))
        masks_arr[roi_idx] = next_masks
        if isinstance(edit_applied_arr, zarr.Array) and np.any(changed_channels):
            prior = np.asarray(edit_applied_arr[roi_idx], dtype=bool)
            edit_applied_arr[roi_idx] = np.logical_or(prior, changed_channels)
        ellipse_params_arr[roi_idx, 0] = left_params
        ellipse_params_arr[roi_idx, 1] = right_params
        ellipse_success_arr[roi_idx, 0] = left_success
        ellipse_success_arr[roi_idx, 1] = right_success
        eye_separation_arr[roi_idx] = separation

        _update_contour_arrays(refined, roi_idx, left_contour, side="left")
        _update_contour_arrays(refined, roi_idx, right_contour, side="right")

        if reason_arr is not None:
            existing = str(reason_arr[roi_idx]) if reason_arr[roi_idx] is not None else ""
            tags = ["manual_correction"]
            if reject_reason:
                tags.append(reject_reason)
            reason_value = _merge_reason(existing, tags)
            reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)

        print(f"Saved manual correction for ROI {roi_idx}.")

    def on_mouse(event: int, x: int, y: int, flags: int, _param: object) -> None:
        nonlocal drawing, erase_mode, cursor_pos
        if display_scale != 1.0:
            x = int(x / display_scale)
            y = int(y / display_scale)
        roi_x, roi_y, roi_w, roi_h = display_layout["roi"]
        edit_x, edit_y, edit_w, edit_h = display_layout["edit"]
        in_roi = roi_x <= x < roi_x + roi_w and roi_y <= y < roi_y + roi_h
        in_edit = edit_x <= x < edit_x + edit_w and edit_y <= y < edit_y + edit_h
        if not (in_roi or in_edit):
            if cursor_pos is not None:
                cursor_pos = None
                update_display()
            if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                drawing = False
            return
        if in_roi:
            local_x = x - roi_x
            local_y = y - roi_y
            panel = "roi"
        else:
            local_x = x - edit_x
            local_y = y - edit_y
            panel = "edit"

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            erase_mode = False
            cursor_pos = (panel, int(local_x), int(local_y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing = True
            erase_mode = True
            cursor_pos = (panel, int(local_x), int(local_y))
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            drawing = False
            cursor_pos = None
            update_display()

        if 0 <= local_x < roi_w and 0 <= local_y < roi_h:
            if event == cv2.EVENT_MOUSEMOVE and not drawing:
                cursor_pos = (panel, int(local_x), int(local_y))
                update_display()
            elif drawing:
                cursor_pos = (panel, int(local_x), int(local_y))
                color = 0 if erase_mode else 1
                cv2.circle(edit_masks[active_eye], (local_x, local_y), brush_radius, color, -1)
                update_display()

    cv2.setMouseCallback(window_name, on_mouse)
    load_current_roi()
    update_display()

    print("\nEye Mask Failure Review")
    print(f"  Refined run: {refined_run}")
    print(f"  Failure ROIs: {int(failure_indices.size)}")
    if flag_path is not None:
        print(f"  Flagged ROIs ({flag_path.expanduser().resolve(strict=False)}): {int(flagged_indices.size)}")
    print(f"  Total ROIs to review: {len(failures)}")
    print("Controls:")
    print("  1/2: select left/right eye")
    print("  Mouse: paint (LMB) / erase (RMB) on ROI or mask editor")
    print("  [ / ]: brush size")
    print("  s: save correction + advance")
    print("  r: reset masks")
    print("  n/p: next/previous failure")
    print("  a: approve eye masks")
    print("  N: mark needs_review")
    print("  R: mark rejected")
    print("  P: mark pending")
    print("  q/ESC: quit")

    def apply_state(state: str) -> None:
        payload = _apply_review_status(
            refined_parent,
            refined_run,
            refined,
            state=state,
            method=review_method,
            intended_use=review_intended_use,
            reviewer=reviewer or os.environ.get("USER") or os.environ.get("USERNAME"),
            notes=review_notes,
        )
        print(
            f"✓ Eye mask review set: {payload.get('state')} "
            f"({payload.get('method')}/{payload.get('intended_use')})"
        )

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("1"):
            set_active_eye(0)
            update_display()
        elif key == ord("2"):
            set_active_eye(1)
            update_display()
        elif key == ord("["):
            brush_radius = max(1, brush_radius - 1)
            update_display()
        elif key == ord("]"):
            brush_radius = min(40, brush_radius + 1)
            update_display()
        elif key == ord("r"):
            edit_masks = [original_masks[0].copy(), original_masks[1].copy()]
            update_display()
        elif key == ord("s"):
            previous_roi_idx = int(roi_idx)
            previous_pos = int(current_pos)
            save_current()
            pending_flagged.discard(previous_roi_idx)
            updated_failure_indices = _load_failure_indices(refined, min_sep, max_sep)
            updated_indices = _combine_review_indices(updated_failure_indices, pending_flagged)
            if updated_indices.size == 0:
                print("All failures cleared.")
                break
            failures = updated_indices
            current_pos = _next_review_position(
                failures,
                previous_roi_idx=previous_roi_idx,
                previous_pos=previous_pos,
            )
            load_current_roi()
            update_display()
        elif key == ord("n"):
            if current_pos < len(failures) - 1:
                current_pos += 1
                load_current_roi()
                update_display()
        elif key == ord("p"):
            if current_pos > 0:
                current_pos -= 1
                load_current_roi()
                update_display()
        elif key == ord("a"):
            apply_state(review_state)
        elif key == ord("R"):
            apply_state("rejected")
        elif key == ord("N"):
            apply_state("needs_review")
        elif key == ord("P"):
            apply_state("pending")

    cv2.destroyAllWindows()


def main(argv: Optional[Sequence[str]] = None) -> None:
    raise SystemExit(
        "The eye_mask_failure_review entrypoint has been removed. "
        "Use `python -m fisheye.tune.eye_mask_review --manual`."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
