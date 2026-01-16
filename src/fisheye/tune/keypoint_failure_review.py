"""
Manual review tool for failed keypoints in a refined keypoint run.

Edits are applied to the refined run only (raw keypoints remain untouched).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import zarr

from ..refinement.keypoint_quality import compute_geometry_metrics
from ..refinement.refine_keypoints import _compute_heading_from_points


_DEFAULT_LABELS = ("bladder", "eye_left", "eye_right")
_DEFAULT_COLORS = ("#22c55e", "#1a66f3", "#f85151")


def _get_latest_run(root: zarr.Group, group_name: str) -> str:
    parent_name = f"{group_name}_runs"
    if parent_name not in root:
        raise RuntimeError(f"No '{parent_name}' group found in Zarr store.")
    latest = root[parent_name].attrs.get("latest")
    if not latest:
        raise RuntimeError(f"No runs recorded under '{parent_name}'.")
    return latest


def _load_failure_indices(refined: zarr.Group) -> np.ndarray:
    if "refined_success" in refined:
        success = np.asarray(refined["refined_success"][:], dtype=bool)
        return np.where(~success)[0].astype("i4", copy=False)
    if "source_success" in refined:
        success = np.asarray(refined["source_success"][:], dtype=bool)
        return np.where(~success)[0].astype("i4", copy=False)
    if "failure_indices" in refined:
        return np.asarray(refined["failure_indices"][:], dtype="i4")
    return np.zeros(0, dtype="i4")


def _clean_reason(existing: str, new_tags: Sequence[str]) -> str:
    tags = [tag for tag in existing.split("|") if tag and tag != "detection_failed"]
    tags.extend(new_tags)
    unique = sorted(set(tags))
    return "|".join(unique) if unique else "manual_correction"


def _sanitize_reason_array(reason_arr: zarr.Array) -> None:
    try:
        raw = reason_arr[:]
    except Exception:
        return
    if raw.size == 0:
        return

    def coerce(val: object) -> str:
        if val is None:
            return ""
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return ""
            if val.size == 1:
                return str(val.item())
            return "|".join(str(item) for item in val.tolist())
        return str(val)

    cleaned = np.array([coerce(v) for v in raw], dtype=object)
    reason_arr[:] = cleaned


def launch_review(
    zarr_path: str,
    refined_run: Optional[str] = None,
    crop_run: Optional[str] = None,
) -> None:
    root = zarr.open_group(zarr_path, mode="a")

    refined_parent = root.get("refined_keypoints_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_keypoints_runs found in archive.")

    using_latest = refined_run is None
    if refined_run is None:
        refined_run = refined_parent.attrs.get("latest")
    if not refined_run or refined_run not in refined_parent:
        raise RuntimeError("Refined keypoint run not found.")
    refined = refined_parent[refined_run]

    crop_run = crop_run or refined.attrs.get("source_crop_run") or _get_latest_run(root, "crop")
    crop_group = root[f"crop_runs/{crop_run}"]
    roi_images = crop_group["roi_images"]
    roi_coords = crop_group["roi_coordinates_full"]
    frame_indices = crop_group["frame_indices"][:]

    full_h, full_w = root["raw_video/images_full"].shape[1:]
    norm_factor = np.array([full_w, full_h], dtype=np.float64)

    failures = _load_failure_indices(refined)
    if failures.size == 0:
        print("No failed keypoints to review.")
        return

    summary_raw = refined.attrs.get("summary_statistics", {})
    summary = summary_raw.get("refine", summary_raw) if isinstance(summary_raw, dict) else {}
    confidence_threshold = float(summary.get("confidence_threshold", 0.3))
    min_triangle_angle = float(summary.get("min_triangle_angle", 10.0))
    min_triangle_area = float(summary.get("min_triangle_area", 100.0))

    if "keypoints_roi" not in refined:
        raise RuntimeError("Refined run is missing keypoints_roi.")
    kp_roi_arr = refined["keypoints_roi"]
    kp_img_arr = refined.get("keypoints_img")
    kp_norm_arr = refined.get("keypoints_norm")
    heading_arr = refined.get("heading")
    confidence_arr = refined.get("confidence")
    conf_arr = refined.get("keypoint_confidences")
    triangle_area_arr = refined.get("triangle_area")
    min_angle_arr = refined.get("min_angle")
    triangle_angles_arr = refined.get("triangle_angles")
    refined_success_arr = refined.get("refined_success")
    flip_corrected_arr = refined.get("flip_corrected")
    quality_labels_arr = refined.get("quality_labels")
    confidence_valid_arr = refined.get("confidence_valid")
    geometry_valid_arr = refined.get("geometry_valid")
    usable_arr = refined.get("usable_keypoints")
    reason_arr = refined.get("reason")
    heading_valid_arr = refined.get("heading_valid")
    detection_source_arr = refined.get("detection_source")
    if reason_arr is not None:
        _sanitize_reason_array(reason_arr)

    labels = list(refined.attrs.get("keypoint_labels", _DEFAULT_LABELS))
    colors = _DEFAULT_COLORS[: len(labels)]

    idx_pos = 0
    active_idx = 0
    points = np.full((len(labels), 2), np.nan, dtype=np.float64)

    mpl.rcParams["keymap.save"] = []
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)

    def load_current_points() -> None:
        nonlocal points
        roi_idx = int(failures[idx_pos])
        existing = np.asarray(kp_roi_arr[roi_idx], dtype=np.float64)
        if existing.shape == points.shape and np.isfinite(existing).any():
            points = existing.copy()
        else:
            points = np.full_like(points, np.nan)

    def update_display() -> None:
        roi_idx = int(failures[idx_pos])
        roi_img = roi_images[roi_idx]
        frame_idx = int(frame_indices[roi_idx])

        ax.clear()
        ax.imshow(roi_img, cmap="gray")

        for i, (label, color) in enumerate(zip(labels, colors)):
            if np.isfinite(points[i]).all():
                ax.scatter(points[i, 0], points[i, 1], s=60, c=color, edgecolors="black", linewidths=1.0)
                ax.text(points[i, 0] + 3, points[i, 1] - 3, label, color=color, fontsize=8, weight="bold")

        active_label = labels[active_idx] if active_idx < len(labels) else "unknown"
        ax.set_title(
            f"ROI {roi_idx} | Frame {frame_idx} | {idx_pos + 1}/{len(failures)} "
            f"| Active: {active_label}",
            fontsize=10,
        )
        ax.set_axis_off()
        fig.canvas.draw_idle()

    def save_current() -> None:
        nonlocal active_idx, idx_pos
        roi_idx = int(failures[idx_pos])
        if not np.isfinite(points).all():
            print("Set all three keypoints before saving.")
            return

        kp_roi_arr[roi_idx] = points
        full_points = points + roi_coords[roi_idx]
        if kp_img_arr is not None:
            kp_img_arr[roi_idx] = full_points
        if kp_norm_arr is not None:
            kp_norm_arr[roi_idx] = full_points / norm_factor

        heading_val = _compute_heading_from_points(points[0], points[1], points[2])
        if heading_arr is not None:
            heading_arr[roi_idx] = heading_val

        metrics = compute_geometry_metrics(points)
        geom_ok = bool(
            np.isfinite(metrics.min_angle)
            and np.isfinite(metrics.area)
            and metrics.min_angle >= min_triangle_angle
            and metrics.area >= min_triangle_area
        )

        if triangle_area_arr is not None:
            triangle_area_arr[roi_idx] = metrics.area
        if min_angle_arr is not None:
            min_angle_arr[roi_idx] = metrics.min_angle
        if triangle_angles_arr is not None:
            triangle_angles_arr[roi_idx] = metrics.angles

        conf_ok = True
        if conf_arr is not None:
            conf_vals = np.ones(len(labels), dtype=np.float64)
            conf_arr[roi_idx] = conf_vals
            conf_ok = bool(np.all(conf_vals >= confidence_threshold))

        if confidence_arr is not None:
            confidence_arr[roi_idx] = 1.0

        refined_success_val = True
        if refined_success_arr is not None:
            refined_success_arr[roi_idx] = refined_success_val
        if flip_corrected_arr is not None:
            flip_corrected_arr[roi_idx] = False
        if quality_labels_arr is not None:
            quality_labels_arr[roi_idx] = 0
        if confidence_valid_arr is not None:
            confidence_valid_arr[roi_idx] = conf_ok
        if geometry_valid_arr is not None:
            geometry_valid_arr[roi_idx] = geom_ok
        if usable_arr is not None:
            usable_arr[roi_idx] = conf_ok and geom_ok
        if heading_valid_arr is not None:
            det_src = int(detection_source_arr[roi_idx]) if detection_source_arr is not None else 0
            heading_valid_arr[roi_idx] = refined_success_val and det_src == 0
        if reason_arr is not None:
            existing = str(reason_arr[roi_idx]) if reason_arr[roi_idx] is not None else ""
            tags = ["manual_correction"]
            if not geom_ok:
                tags.append("geometry_issue")
            reason_value = str(_clean_reason(existing, tags))
            reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)

        print(f"Saved manual correction for ROI {roi_idx}.")

        active_idx = 0
        if idx_pos < len(failures) - 1:
            idx_pos += 1
            load_current_points()
            update_display()

    def next_failure() -> None:
        nonlocal idx_pos
        if idx_pos < len(failures) - 1:
            idx_pos += 1
            load_current_points()
            update_display()

    def prev_failure() -> None:
        nonlocal idx_pos
        if idx_pos > 0:
            idx_pos -= 1
            load_current_points()
            update_display()

    def reset_points() -> None:
        load_current_points()
        update_display()

    def on_click(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        points[active_idx] = [event.xdata, event.ydata]
        update_display()

    def on_key(event) -> None:
        nonlocal active_idx
        if event.key in {"1", "2", "3"}:
            active_idx = int(event.key) - 1
            update_display()
        elif event.key == "n":
            next_failure()
        elif event.key == "p":
            prev_failure()
        elif event.key == "r":
            reset_points()
        elif event.key == "s":
            save_current()
        elif event.key == "q":
            plt.close(fig)

    print("\nKeypoint Failure Review")
    print(f"  Zarr: {zarr_path}")
    refined_label = f"{refined_run} (latest)" if using_latest else refined_run
    print(f"  Refined run: {refined_label}")
    print(f"  Crop run: {crop_run}")
    print(f"  Failures to review: {len(failures)}")
    print("\nControls:")
    print("  Click: set active keypoint")
    print("  1/2/3: select bladder/left/right")
    print("  s: save correction")
    print("  r: reset points from current data")
    print("  n/p: next/previous failure")
    print("  q: quit")

    load_current_points()
    update_display()
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def main(argv: Optional[Sequence[str]] = None) -> None:
    raise SystemExit(
        "The keypoint_failure_review entrypoint has been removed. "
        "Use `python -m fisheye.tune.keypoint_review --manual`."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
