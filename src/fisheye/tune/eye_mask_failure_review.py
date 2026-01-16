"""Manual eye mask failure review and correction UI (OpenCV)."""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence, Tuple, Dict, Any

import cv2
import numpy as np
import zarr
from skimage import measure
from skimage.measure import EllipseModel


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


def _compute_success_mask(
    ellipse_success: np.ndarray,
    eye_separation: np.ndarray,
    min_sep: Optional[float],
    max_sep: Optional[float],
) -> np.ndarray:
    pair_success = np.all(ellipse_success, axis=1)
    sep_ok = np.ones_like(pair_success, dtype=bool)
    if eye_separation.size:
        sep_ok = np.isfinite(eye_separation)
        if min_sep is not None:
            sep_ok &= eye_separation >= float(min_sep)
        if max_sep is not None:
            sep_ok &= eye_separation <= float(max_sep)
    return pair_success & sep_ok


def _load_failure_indices(
    refined: zarr.Group,
    min_sep: Optional[float],
    max_sep: Optional[float],
) -> np.ndarray:
    ellipse_success = np.asarray(refined["ellipse_success"][:], dtype=bool)
    eye_separation = np.asarray(refined["eye_separation"][:], dtype=np.float32)
    success_mask = _compute_success_mask(ellipse_success, eye_separation, min_sep, max_sep)
    return np.where(~success_mask)[0].astype("i4", copy=False)


def _merge_reason(existing: str, tags: Sequence[str]) -> str:
    existing_tags = [tag for tag in existing.split("|") if tag]
    merged = sorted(set(existing_tags + list(tags)))
    return "|".join(merged) if merged else "clean"


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
) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
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
    ellipse_params_arr = refined["ellipse_params"]
    ellipse_success_arr = refined["ellipse_success"]
    eye_separation_arr = refined["eye_separation"]

    metrics_group = refined.get("metrics")
    reason_arr = metrics_group.get("reason") if isinstance(metrics_group, zarr.Group) else None

    min_sep, max_sep = _get_sep_limits(root, refined)
    failures = _load_failure_indices(refined, min_sep, max_sep)
    if failures.size == 0:
        print("No failures to review.")
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

        masks_arr[roi_idx, 0] = edit_masks[0]
        masks_arr[roi_idx, 1] = edit_masks[1]
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
    print(f"  Failures to review: {len(failures)}")
    print("Controls:")
    print("  1/2: select left/right eye")
    print("  Mouse: paint (LMB) / erase (RMB) on ROI or mask editor")
    print("  [ / ]: brush size")
    print("  s: save correction")
    print("  r: reset masks")
    print("  n/p: next/previous failure")
    print("  q/ESC: quit")

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
            save_current()
            updated_failures = _load_failure_indices(refined, min_sep, max_sep)
            if updated_failures.size == 0:
                print("All failures cleared.")
                break
            failures = updated_failures
            current_pos = min(current_pos, len(failures) - 1)
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

    cv2.destroyAllWindows()


def main(argv: Optional[Sequence[str]] = None) -> None:
    raise SystemExit(
        "The eye_mask_failure_review entrypoint has been removed. "
        "Use `python -m fisheye.tune.eye_mask_review --manual`."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
