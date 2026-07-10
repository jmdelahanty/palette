from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from _common import (
    cfg_path,
    cfg_value,
    compute_body_transform,
    crop_row_frame_id,
    ensure_output_dir,
    get_video_info,
    keypoints_to_crop_pixels,
    load_config,
    load_keypoint_data,
    mask_mean_intensity,
    polygon_mask,
    project_subject_mask_to_crop_frame,
    read_crop_meta,
    resolve_roi_rect,
    roi_rect_corners,
    selected_crop_rows,
    transform_points,
)
from compare_roi_pixel_strategies import _estimate_trace, _estimate_columns, _primary_values
from map_pixel_band_contributions import _load_mask_component, _split_components
from measure_live_mask_stability import _dilate, _load_fixed_masks
from render_local_rostral_alignment_comparison import (
    _component_edge_midpoint,
    _reference_component_mask,
    _rigid_from_anchor_pair,
    _swim_anchor,
)
from render_roi_mask_overlay_video import _read_status_csv


def _invert_affine(matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.eye(3, dtype=np.float64)
    homogeneous[:2, :] = np.asarray(matrix, dtype=np.float64)
    inverse = np.linalg.inv(homogeneous)
    return inverse[:2, :]


def _warp_crop_mask_to_stable(
    mask: np.ndarray,
    *,
    crop_to_stable: np.ndarray,
    stable_width: int,
    stable_height: int,
) -> np.ndarray:
    import cv2

    return (
        cv2.warpAffine(
            np.asarray(mask, dtype=np.uint8) * 255,
            np.asarray(crop_to_stable, dtype=np.float32),
            (int(stable_width), int(stable_height)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )


def _project_component_crop(
    *,
    mask_data: Any,
    frame_id: int,
    crop_row: dict[str, str],
    video: Any,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    projected = project_subject_mask_to_crop_frame(
        mask_data,
        frame_id=int(frame_id),
        crop_row=crop_row,
        video_width=int(video.width),
        video_height=int(video.height),
    )
    details = {
        "valid": int(bool(projected.valid)),
        "reason": projected.reason,
        "mask_row": int(projected.mask_row),
        "source_crop_row_id": int(projected.source_crop_row_id),
        "pixel_count": int(projected.mask_pixel_count),
    }
    if not projected.valid or projected.mask is None:
        return None, details
    return np.asarray(projected.mask, dtype=bool), details


def _sample_mean(
    frame: np.ndarray,
    sample_mask: np.ndarray,
    *,
    min_pixels: int,
    min_mean_intensity: float | None,
) -> tuple[bool, str, float, int]:
    mean, count = mask_mean_intensity(frame, sample_mask)
    if int(count) < int(min_pixels):
        return False, "too_few_sample_pixels", math.nan, int(count)
    if not np.isfinite(mean):
        return False, "nonfinite_mean", math.nan, int(count)
    if min_mean_intensity is not None and float(mean) <= float(min_mean_intensity):
        return False, "low_mean_intensity", float(mean), int(count)
    return True, "ok", float(mean), int(count)


def _status_add(counter: dict[str, int], value: str) -> None:
    counter[str(value)] = counter.get(str(value), 0) + 1


def _interpolate_short_gaps_1d(values: np.ndarray, valid: np.ndarray, *, max_gap: int) -> tuple[np.ndarray, np.ndarray, int]:
    out = values.astype(np.float64, copy=True)
    finite = np.asarray(valid, dtype=bool) & np.isfinite(out)
    interpolated = 0
    idx = 0
    while idx < len(finite):
        if finite[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(finite) and not finite[idx]:
            idx += 1
        stop = idx
        if start == 0 or stop >= len(finite) or stop - start > int(max_gap):
            continue
        left = out[start - 1]
        right = out[stop]
        steps = float(stop - start + 1)
        for offset, row in enumerate(range(start, stop), start=1):
            frac = float(offset) / steps
            out[row] = (1.0 - frac) * left + frac * right
            finite[row] = True
            interpolated += 1
    return out, finite, int(interpolated)


def _longest_contiguous_true(mask: np.ndarray) -> tuple[int, int]:
    best_start = 0
    best_stop = 0
    run_start: int | None = None
    for idx, ok in enumerate(np.asarray(mask, dtype=bool)):
        if bool(ok) and run_start is None:
            run_start = idx
        if ((not bool(ok)) or idx == len(mask) - 1) and run_start is not None:
            run_stop = idx if not bool(ok) else idx + 1
            if run_stop - run_start > best_stop - best_start:
                best_start, best_stop = run_start, run_stop
            run_start = None
    return int(best_start), int(best_stop)


def _window_rows(
    *,
    frame_indices: np.ndarray,
    frame_start: int,
    frame_count: int,
    fps: float,
    window_seconds: float,
    step_seconds: float,
) -> list[tuple[int, float, float, np.ndarray]]:
    duration_s = float(frame_count) / float(fps)
    windows: list[tuple[int, float, float, np.ndarray]] = []
    index = 0
    start_s = 0.0
    while start_s + float(window_seconds) <= duration_s + 1e-9:
        stop_s = start_s + float(window_seconds)
        lo = int(frame_start) + int(round(start_s * float(fps)))
        hi = int(frame_start) + int(round(stop_s * float(fps)))
        rows = np.flatnonzero((frame_indices >= lo) & (frame_indices < hi))
        windows.append((index, float(start_s), float(stop_s), rows))
        index += 1
        start_s += float(step_seconds)
    return windows


def _estimate_windows(
    *,
    frame_indices: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    strategy: str,
    fps: float,
    frame_start: int,
    frame_count: int,
    window_seconds: float,
    step_seconds: float,
    max_gap: int,
    min_valid_fraction: float,
    band_min_hz: float,
    band_max_hz: float,
    primary_estimator: str,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    min_samples = max(16, int(math.ceil(float(window_seconds) * float(fps) * float(min_valid_fraction))))
    for window_index, start_s, stop_s, rows in _window_rows(
        frame_indices=frame_indices,
        frame_start=frame_start,
        frame_count=frame_count,
        fps=fps,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    ):
        base: dict[str, Any] = {
            "strategy": strategy,
            "window_index": int(window_index),
            "window_start_s": float(start_s),
            "window_stop_s": float(stop_s),
            "window_frame_start": int(frame_start + round(start_s * fps)),
            "window_frame_stop_inclusive": int(frame_start + round(stop_s * fps) - 1),
            "window_rows": int(rows.size),
        }
        if rows.size == 0:
            base.update({"status": "empty_window", "valid_rows": 0, "valid_fraction": 0.0, "sample_count": 0})
            base.update(_primary_values(_nan_estimates_for_import(), primary_estimator=primary_estimator))
            base.update(_nan_estimates_for_import())
            rows_out.append(base)
            continue
        interpolated, finite, interpolated_count = _interpolate_short_gaps_1d(values[rows], valid[rows], max_gap=max_gap)
        start, stop = _longest_contiguous_true(finite)
        selected_count = max(0, stop - start)
        valid_fraction = float(np.count_nonzero(valid[rows] & np.isfinite(values[rows])) / max(1, int(rows.size)))
        base.update(
            {
                "valid_rows": int(np.count_nonzero(valid[rows] & np.isfinite(values[rows]))),
                "valid_fraction": valid_fraction,
                "interpolated_rows": int(interpolated_count),
                "contiguous_rows_after_interpolation": int(selected_count),
            }
        )
        if selected_count < min_samples:
            base.update({"status": "too_few_contiguous_samples", "sample_count": int(selected_count)})
            base.update(_primary_values(_nan_estimates_for_import(), primary_estimator=primary_estimator))
            base.update(_nan_estimates_for_import())
            rows_out.append(base)
            continue
        segment_frames = frame_indices[rows][start:stop]
        segment_values = interpolated[start:stop]
        status, estimates, sample_count = _estimate_trace(
            frame_index=segment_frames,
            values=segment_values,
            fps=float(fps),
            band_min_hz=float(band_min_hz),
            band_max_hz=float(band_max_hz),
            primary_estimator=str(primary_estimator),
        )
        base.update({"status": status, "sample_count": int(sample_count)})
        base.update(_primary_values(estimates, primary_estimator=primary_estimator))
        base.update(estimates)
        rows_out.append(base)
    return rows_out


def _nan_estimates_for_import() -> dict[str, float]:
    return {key: math.nan for key in _estimate_columns()}


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _aggregate_strategy(rows: list[dict[str, Any]], *, strategy: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("strategy") == strategy]
    ok = [row for row in selected if row.get("status") == "ok" and np.isfinite(float(row.get("peak_bpm", math.nan)))]
    bpm = _finite([float(row.get("peak_bpm", math.nan)) for row in ok])
    score = _finite([float(row.get("peak_score", math.nan)) for row in ok])
    return {
        "windows": int(len(selected)),
        "ok_windows": int(len(ok)),
        "median_peak_bpm": float(np.median(bpm)) if bpm.size else None,
        "min_peak_bpm": float(np.min(bpm)) if bpm.size else None,
        "max_peak_bpm": float(np.max(bpm)) if bpm.size else None,
        "iqr_peak_bpm": float(np.quantile(bpm, 0.75) - np.quantile(bpm, 0.25)) if bpm.size >= 2 else None,
        "median_peak_score": float(np.median(score)) if score.size else None,
        "status_counts": {str(status): int(sum(1 for row in selected if row.get("status") == status)) for status in sorted({row.get("status") for row in selected})},
    }


def _write_plot(
    path: Path,
    *,
    sample_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal

    frame = np.asarray([int(row["crop_video_frame_index"]) for row in sample_rows], dtype=np.int64)
    t = (frame - frame[0]).astype(np.float64) / float(fps)

    def series(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray([float(row.get(f"{prefix}_mean_intensity", math.nan) or math.nan) for row in sample_rows])
        valid = np.asarray([str(row.get(f"{prefix}_valid", "0")) == "1" for row in sample_rows], dtype=bool)
        return values, valid & np.isfinite(values)

    current, current_valid = series("current")
    local, local_valid = series("local")

    def filtered(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, math.nan, dtype=np.float64)
        interpolated, finite, _count = _interpolate_short_gaps_1d(values, valid, max_gap=5)
        start, stop = _longest_contiguous_true(finite)
        if stop - start < 32:
            return out
        x = interpolated[start:stop]
        x = signal.detrend(x - np.nanmedian(x), type="linear")
        sos = signal.butter(3, [float(band_min_hz), float(band_max_hz)], btype="bandpass", fs=float(fps), output="sos")
        out[start:stop] = signal.sosfiltfilt(sos, x)
        return out

    current_f = filtered(current, current_valid)
    local_f = filtered(local, local_valid)
    local_rot = np.asarray([float(row.get("local_rotation_deg", math.nan) or math.nan) for row in sample_rows])
    local_trans = np.asarray([float(row.get("local_translation_px", math.nan) or math.nan) for row in sample_rows])

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=False, constrained_layout=True)
    axes[0].plot(t, current, lw=0.7, label="current fixed ROI")
    axes[0].plot(t, local, lw=0.7, label="gated local ROI")
    axes[0].set_ylabel("mean intensity")
    axes[0].set_title("Source-pixel ROI mean")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, current_f, lw=0.8, label="current")
    axes[1].plot(t, local_f, lw=0.8, label="local")
    axes[1].set_ylabel("bandpassed")
    axes[1].set_title(f"Bandpassed trace {band_min_hz:g}-{band_max_hz:g} Hz")
    axes[1].legend(loc="upper right")

    for strategy, color in (("current", "tab:blue"), ("local", "tab:orange")):
        rows = [row for row in summary_rows if row.get("strategy") == strategy and row.get("status") == "ok"]
        xs = np.asarray([0.5 * (float(row["window_start_s"]) + float(row["window_stop_s"])) for row in rows])
        ys = np.asarray([float(row["peak_bpm"]) for row in rows])
        axes[2].plot(xs, ys, marker="o", lw=1.0, color=color, label=strategy)
    axes[2].set_ylabel("bpm")
    axes[2].set_title("Windowed primary estimates")
    axes[2].legend(loc="upper right")

    axes[3].plot(t, local_rot, lw=0.7, label="local rotation deg")
    axes[3].plot(t, local_trans, lw=0.7, label="local translation px")
    axes[3].set_xlabel("time in selected clip (s)")
    axes[3].set_ylabel("correction")
    axes[3].set_title("Local correction diagnostics")
    axes[3].legend(loc="upper right")
    ensure_output_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Compare fixed and gated local-rostral ROI signals from source crop pixels.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--roi-json", type=Path, required=True)
    parser.add_argument("--mask-npz", type=Path, required=True)
    parser.add_argument("--status-csv", type=Path, default=None)
    parser.add_argument("--frame-start", type=int, default=30000)
    parser.add_argument("--frame-count", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--band-min-hz", type=float, default=1.5)
    parser.add_argument("--band-max-hz", type=float, default=3.5)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--window-step-seconds", type=float, default=2.5)
    parser.add_argument("--max-interpolated-gap-samples", type=int, default=2)
    parser.add_argument("--min-window-valid-fraction", type=float, default=0.80)
    parser.add_argument("--primary-estimator", choices=("autocorr", "welch", "periodogram"), default="autocorr")
    parser.add_argument("--min-sample-pixels", type=int, default=20)
    parser.add_argument("--min-roi-mean-intensity", type=float, default=1.0)
    parser.add_argument("--mask-parent", type=str, default=None, help="Mask parent group, or 'auto'.")
    parser.add_argument("--mask-run", type=str, default=None, help="Mask run name, or 'latest'.")
    parser.add_argument("--body-component", type=str, default="subject_body")
    parser.add_argument("--eye-components", type=str, default="eye_left,eye_right")
    parser.add_argument("--swim-component", type=str, default="swim_bladder")
    parser.add_argument("--eye-dilate-px", type=int, default=2)
    parser.add_argument("--anchor-band-px", type=int, default=2)
    parser.add_argument("--eye-occupancy-threshold", type=float, default=0.50)
    parser.add_argument("--swim-occupancy-threshold", type=float, default=0.25)
    parser.add_argument("--swim-anchor-mode", choices=("upper_edge", "centroid"), default="upper_edge")
    parser.add_argument("--reference-frame-start", type=int, default=30000)
    parser.add_argument("--reference-frame-count", type=int, default=3000)
    parser.add_argument("--reference-stride", type=int, default=10)
    parser.add_argument("--min-axis-length-px", type=float, default=8.0)
    parser.add_argument("--max-local-rotation-deg", type=float, default=50.0)
    parser.add_argument("--max-local-translation-px", type=float, default=150.0)
    parser.add_argument("--allow-local-scale", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/local_roi_signal_compare"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    crop_video = cfg_path(config, "inputs", "crop_video")
    crop_meta_csv = cfg_path(config, "inputs", "crop_meta_csv")
    zarr_path = cfg_path(config, "inputs", "zarr_path")
    keypoint_group = str(cfg_value(config, "inputs", "keypoint_group"))
    frame_id_column = str(cfg_value(config, "alignment", "frame_id_column", "camera_frame_id"))
    frame_array = str(cfg_value(config, "alignment", "keypoint_frame_array", "frame_indices"))
    keypoint_array = str(cfg_value(config, "alignment", "keypoint_coordinate_array", "keypoints_img"))
    valid_array = str(cfg_value(config, "alignment", "keypoint_valid_array", "usable_keypoints"))
    stable_width = int(cfg_value(config, "alignment", "stable_width", 256))
    stable_height = int(cfg_value(config, "alignment", "stable_height", 256))
    stable_center_x = float(cfg_value(config, "alignment", "stable_center_x", stable_width / 2.0))
    stable_center_y = float(cfg_value(config, "alignment", "stable_center_y", stable_height / 2.0))
    origin = str(cfg_value(config, "alignment", "origin", "eye_swim_midpoint"))
    target_forward = str(cfg_value(config, "alignment", "target_forward", "up"))
    scale = float(cfg_value(config, "alignment", "scale", 1.0))
    min_forward = float(cfg_value(config, "alignment", "min_forward_length_px", 8.0))
    min_eye_span = float(cfg_value(config, "alignment", "min_eye_span_px", 4.0))
    mask_parent = str(args.mask_parent or cfg_value(config, "mask", "parent", "auto"))
    mask_run = str(args.mask_run or cfg_value(config, "mask", "run", "latest"))

    output_prefix = Path(args.output_prefix)
    ensure_output_dir(output_prefix.parent)
    sample_csv = output_prefix.with_suffix(".samples.csv")
    summary_csv = output_prefix.with_suffix(".summary.csv")
    summary_json = output_prefix.with_suffix(".summary.json")
    plot_png = output_prefix.with_suffix(".png")

    roi_rect = resolve_roi_rect(config, roi_json=args.roi_json)
    roi_corners_stable = roi_rect_corners(roi_rect)
    crop_rows = read_crop_meta(crop_meta_csv)
    selected = selected_crop_rows(
        crop_rows,
        frame_id_column=frame_id_column,
        frame_start=int(args.frame_start),
        frame_count=int(args.frame_count),
        stride=max(1, int(args.stride)),
    )
    if not selected:
        raise ValueError("No frames selected.")
    video = get_video_info(crop_video)
    keypoints = load_keypoint_data(
        zarr_path,
        keypoint_group,
        frame_array=frame_array,
        keypoint_array=keypoint_array,
        valid_array=valid_array,
    )
    fixed = _load_fixed_masks(args.mask_npz, shape_hw=(stable_height, stable_width))
    status = _read_status_csv(args.status_csv)

    body_data = _load_mask_component(
        zarr_path,
        parent=mask_parent,
        run_name=mask_run,
        component_name=str(args.body_component),
    )
    eye_data = {
        component: _load_mask_component(
            zarr_path,
            parent=mask_parent,
            run_name=mask_run,
            component_name=component,
        )
        for component in _split_components(args.eye_components)
    }
    if not eye_data:
        raise ValueError("--eye-components must name at least one component.")
    swim_data = _load_mask_component(
        zarr_path,
        parent=mask_parent,
        run_name=mask_run,
        component_name=str(args.swim_component),
    )

    fixed_swim, fixed_swim_summary = _reference_component_mask(
        mask_data=swim_data,
        crop_rows=crop_rows,
        keypoints=keypoints,
        video=video,
        frame_id_column=frame_id_column,
        keypoint_coordinate_array=keypoint_array,
        stable_width=stable_width,
        stable_height=stable_height,
        stable_center_x=stable_center_x,
        stable_center_y=stable_center_y,
        origin=origin,
        target_forward=target_forward,
        scale=scale,
        min_forward=min_forward,
        min_eye_span=min_eye_span,
        frame_start=int(args.reference_frame_start),
        frame_count=int(args.reference_frame_count),
        stride=int(args.reference_stride),
        occupancy_threshold=float(args.swim_occupancy_threshold),
    )
    fixed_eye_components: dict[str, np.ndarray] = {}
    fixed_eye_component_summaries: dict[str, Any] = {}
    for component, data in eye_data.items():
        component_mask, component_summary = _reference_component_mask(
            mask_data=data,
            crop_rows=crop_rows,
            keypoints=keypoints,
            video=video,
            frame_id_column=frame_id_column,
            keypoint_coordinate_array=keypoint_array,
            stable_width=stable_width,
            stable_height=stable_height,
            stable_center_x=stable_center_x,
            stable_center_y=stable_center_y,
            origin=origin,
            target_forward=target_forward,
            scale=scale,
            min_forward=min_forward,
            min_eye_span=min_eye_span,
            frame_start=int(args.reference_frame_start),
            frame_count=int(args.reference_frame_count),
            stride=int(args.reference_stride),
            occupancy_threshold=float(args.eye_occupancy_threshold),
        )
        fixed_eye_components[component] = component_mask
        fixed_eye_component_summaries[component] = component_summary
    fixed_anterior = _component_edge_midpoint(
        list(fixed_eye_components.values()),
        fallback_mask=fixed["eye_mask"],
        quantile=0.98,
        side="lower",
        band_px=int(args.anchor_band_px),
    )
    fixed_posterior = _swim_anchor(fixed_swim, mode=str(args.swim_anchor_mode), band_px=int(args.anchor_band_px))
    if fixed_anterior is None or fixed_posterior is None:
        raise ValueError("Could not compute fixed local anchors.")

    min_mean = float(args.min_roi_mean_intensity)
    min_mean_threshold = min_mean if np.isfinite(min_mean) else None
    capture = cv2.VideoCapture(str(crop_video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {crop_video}")

    fieldnames = [
        "crop_video_frame_index",
        "frame_id",
        "status_valid",
        "read_valid",
        "current_valid",
        "current_reason",
        "current_mean_intensity",
        "current_sample_pixels",
        "local_valid",
        "local_reason",
        "local_correction_status",
        "local_mean_intensity",
        "local_sample_pixels",
        "local_rotation_deg",
        "local_translation_px",
        "source_axis_length_px",
        "target_axis_length_px",
        "body_reason",
        "swim_reason",
        "eye_reason",
    ]
    sample_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    correction_counts: dict[str, int] = {}
    next_expected: int | None = None
    try:
        for crop_video_index, crop_row in selected:
            row: dict[str, Any] = {
                "crop_video_frame_index": int(crop_video_index),
                "frame_id": "",
                "status_valid": 0,
                "read_valid": 0,
                "current_valid": 0,
                "current_reason": "not_evaluated",
                "current_mean_intensity": math.nan,
                "current_sample_pixels": 0,
                "local_valid": 0,
                "local_reason": "not_evaluated",
                "local_correction_status": "",
                "local_mean_intensity": math.nan,
                "local_sample_pixels": 0,
                "local_rotation_deg": math.nan,
                "local_translation_px": math.nan,
                "source_axis_length_px": math.nan,
                "target_axis_length_px": math.nan,
                "body_reason": "",
                "swim_reason": "",
                "eye_reason": "",
            }
            frame_id = crop_row_frame_id(int(crop_video_index), crop_row, frame_id_column)
            row["frame_id"] = int(frame_id)
            status_valid = bool(status.get(int(crop_video_index), True))
            row["status_valid"] = int(status_valid)

            if next_expected is None or int(crop_video_index) != int(next_expected):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(crop_video_index))
            ok, frame = capture.read()
            next_expected = int(crop_video_index) + 1
            if not ok:
                row["current_reason"] = "video_read_failed"
                row["local_reason"] = "video_read_failed"
                sample_rows.append(row)
                _status_add(reason_counts, "video_read_failed")
                continue
            row["read_valid"] = 1
            if not status_valid:
                row["current_reason"] = "status_invalid"
                row["local_reason"] = "status_invalid"
                sample_rows.append(row)
                _status_add(reason_counts, "status_invalid")
                continue

            keypoint_row = keypoints.frame_to_row.get(frame_id)
            if keypoint_row is None:
                row["current_reason"] = "missing_keypoint_frame"
                row["local_reason"] = "missing_keypoint_frame"
                sample_rows.append(row)
                _status_add(reason_counts, "missing_keypoint_frame")
                continue
            if not bool(keypoints.valid[keypoint_row]):
                row["current_reason"] = "invalid_keypoints"
                row["local_reason"] = "invalid_keypoints"
                sample_rows.append(row)
                _status_add(reason_counts, "invalid_keypoints")
                continue

            kp_crop = keypoints_to_crop_pixels(
                keypoints.keypoints_img[keypoint_row],
                crop_row,
                video_width=video.width,
                video_height=video.height,
            )
            transform = compute_body_transform(
                kp_crop,
                stable_width=stable_width,
                stable_height=stable_height,
                stable_center_x=stable_center_x,
                stable_center_y=stable_center_y,
                origin=origin,
                target_forward=target_forward,
                scale=scale,
                min_forward_length_px=min_forward,
                min_eye_span_px=min_eye_span,
            )
            if not transform.valid:
                row["current_reason"] = transform.reason
                row["local_reason"] = transform.reason
                sample_rows.append(row)
                _status_add(reason_counts, transform.reason)
                continue

            body_crop, body_details = _project_component_crop(
                mask_data=body_data,
                frame_id=frame_id,
                crop_row=crop_row,
                video=video,
            )
            swim_crop, swim_details = _project_component_crop(
                mask_data=swim_data,
                frame_id=frame_id,
                crop_row=crop_row,
                video=video,
            )
            row["body_reason"] = body_details["reason"]
            row["swim_reason"] = swim_details["reason"]
            eye_crop = np.zeros(frame.shape[:2], dtype=bool)
            eye_component_crop: list[np.ndarray] = []
            eye_reasons: dict[str, str] = {}
            for component, data in eye_data.items():
                component_crop, component_details = _project_component_crop(
                    mask_data=data,
                    frame_id=frame_id,
                    crop_row=crop_row,
                    video=video,
                )
                eye_reasons[component] = str(component_details["reason"])
                if component_crop is not None:
                    eye_component_crop.append(component_crop)
                    eye_crop |= component_crop
            row["eye_reason"] = ",".join(f"{name}:{reason}" for name, reason in eye_reasons.items())
            if body_crop is None:
                row["current_reason"] = f"body:{body_details['reason']}"
                row["local_reason"] = row["current_reason"]
                sample_rows.append(row)
                _status_add(reason_counts, row["current_reason"])
                continue
            if not np.any(eye_crop):
                row["current_reason"] = "empty_eye_union"
                row["local_reason"] = "empty_eye_union"
                sample_rows.append(row)
                _status_add(reason_counts, "empty_eye_union")
                continue

            eye_exclusion_crop = _dilate(eye_crop, int(args.eye_dilate_px))
            current_polygon_crop = transform_points(transform.stable_to_crop, roi_corners_stable)
            current_roi_mask = polygon_mask(frame.shape[:2], current_polygon_crop)
            current_sample = current_roi_mask & body_crop & ~eye_exclusion_crop
            ok_current, current_reason, current_mean, current_pixels = _sample_mean(
                frame,
                current_sample,
                min_pixels=int(args.min_sample_pixels),
                min_mean_intensity=min_mean_threshold,
            )
            row["current_valid"] = int(ok_current)
            row["current_reason"] = current_reason
            row["current_mean_intensity"] = current_mean
            row["current_sample_pixels"] = int(current_pixels)

            if swim_crop is None:
                local_reason = f"swim:{swim_details['reason']}"
                row["local_reason"] = local_reason
                row["local_correction_status"] = local_reason
                _status_add(correction_counts, local_reason)
                sample_rows.append(row)
                continue

            eye_component_stable = [
                _warp_crop_mask_to_stable(
                    component,
                    crop_to_stable=transform.crop_to_stable,
                    stable_width=stable_width,
                    stable_height=stable_height,
                )
                for component in eye_component_crop
            ]
            eye_stable = np.zeros((stable_height, stable_width), dtype=bool)
            for component in eye_component_stable:
                eye_stable |= component
            swim_stable = _warp_crop_mask_to_stable(
                swim_crop,
                crop_to_stable=transform.crop_to_stable,
                stable_width=stable_width,
                stable_height=stable_height,
            )
            live_anterior = _component_edge_midpoint(
                eye_component_stable,
                fallback_mask=eye_stable,
                quantile=0.98,
                side="lower",
                band_px=int(args.anchor_band_px),
            )
            live_posterior = _swim_anchor(swim_stable, mode=str(args.swim_anchor_mode), band_px=int(args.anchor_band_px))
            if live_anterior is None or live_posterior is None:
                row["local_reason"] = "missing_live_anchor"
                row["local_correction_status"] = "missing_live_anchor"
                _status_add(correction_counts, "missing_live_anchor")
                sample_rows.append(row)
                continue

            local_matrix, local_details = _rigid_from_anchor_pair(
                source_posterior=live_posterior,
                source_anterior=live_anterior,
                target_posterior=fixed_posterior,
                target_anterior=fixed_anterior,
                allow_scale=bool(args.allow_local_scale),
                min_axis_length_px=float(args.min_axis_length_px),
            )
            for key in ("local_rotation_deg", "local_translation_px", "source_axis_length_px", "target_axis_length_px"):
                if key in local_details:
                    row[key] = local_details[key]
            correction_status = str(local_details.get("reason", "ok"))
            if local_matrix is not None:
                rotation = abs(float(local_details.get("local_rotation_deg", math.nan)))
                translation = float(local_details.get("local_translation_px", math.nan))
                if np.isfinite(rotation) and rotation > float(args.max_local_rotation_deg):
                    local_matrix = None
                    correction_status = "rejected_rotation_limit"
                elif np.isfinite(translation) and translation > float(args.max_local_translation_px):
                    local_matrix = None
                    correction_status = "rejected_translation_limit"
            row["local_correction_status"] = correction_status
            _status_add(correction_counts, correction_status)
            if local_matrix is None:
                row["local_reason"] = correction_status
                sample_rows.append(row)
                continue

            current_stable_to_fixed_stable = np.asarray(local_matrix, dtype=np.float64)
            fixed_stable_to_current_stable = _invert_affine(current_stable_to_fixed_stable)
            local_roi_corners_current_stable = transform_points(fixed_stable_to_current_stable, roi_corners_stable)
            local_polygon_crop = transform_points(transform.stable_to_crop, local_roi_corners_current_stable)
            local_roi_mask = polygon_mask(frame.shape[:2], local_polygon_crop)
            local_sample = local_roi_mask & body_crop & ~eye_exclusion_crop
            ok_local, local_reason, local_mean, local_pixels = _sample_mean(
                frame,
                local_sample,
                min_pixels=int(args.min_sample_pixels),
                min_mean_intensity=min_mean_threshold,
            )
            row["local_valid"] = int(ok_local)
            row["local_reason"] = local_reason
            row["local_mean_intensity"] = local_mean
            row["local_sample_pixels"] = int(local_pixels)
            sample_rows.append(row)
    finally:
        capture.release()

    with sample_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    frame_indices = np.asarray([int(row["crop_video_frame_index"]) for row in sample_rows], dtype=np.int64)
    current_values = np.asarray([float(row["current_mean_intensity"]) for row in sample_rows], dtype=np.float64)
    local_values = np.asarray([float(row["local_mean_intensity"]) for row in sample_rows], dtype=np.float64)
    current_valid = np.asarray([int(row["current_valid"]) == 1 for row in sample_rows], dtype=bool)
    local_valid = np.asarray([int(row["local_valid"]) == 1 for row in sample_rows], dtype=bool)
    current_windows = _estimate_windows(
        frame_indices=frame_indices,
        values=current_values,
        valid=current_valid,
        strategy="current",
        fps=float(args.fps) / max(1, int(args.stride)),
        frame_start=int(args.frame_start),
        frame_count=int(args.frame_count),
        window_seconds=float(args.window_seconds),
        step_seconds=float(args.window_step_seconds),
        max_gap=int(args.max_interpolated_gap_samples),
        min_valid_fraction=float(args.min_window_valid_fraction),
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
        primary_estimator=str(args.primary_estimator),
    )
    local_windows = _estimate_windows(
        frame_indices=frame_indices,
        values=local_values,
        valid=local_valid,
        strategy="local",
        fps=float(args.fps) / max(1, int(args.stride)),
        frame_start=int(args.frame_start),
        frame_count=int(args.frame_count),
        window_seconds=float(args.window_seconds),
        step_seconds=float(args.window_step_seconds),
        max_gap=int(args.max_interpolated_gap_samples),
        min_valid_fraction=float(args.min_window_valid_fraction),
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
        primary_estimator=str(args.primary_estimator),
    )
    summary_rows = current_windows + local_windows
    _write_summary_csv(summary_csv, summary_rows)
    _write_plot(
        plot_png,
        sample_rows=sample_rows,
        summary_rows=summary_rows,
        fps=float(args.fps) / max(1, int(args.stride)),
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
    )

    aggregate = {
        "config": str(args.config),
        "crop_video": str(crop_video),
        "roi_json": str(args.roi_json),
        "mask_npz": str(args.mask_npz),
        "status_csv": str(args.status_csv) if args.status_csv is not None else None,
        "frame_start": int(args.frame_start),
        "frame_count": int(args.frame_count),
        "stride": int(args.stride),
        "fps_effective": float(args.fps) / max(1, int(args.stride)),
        "band_hz": [float(args.band_min_hz), float(args.band_max_hz)],
        "window_seconds": float(args.window_seconds),
        "window_step_seconds": float(args.window_step_seconds),
        "primary_estimator": str(args.primary_estimator),
        "sample_rows": int(len(sample_rows)),
        "current_valid_frames": int(np.count_nonzero(current_valid)),
        "local_valid_frames": int(np.count_nonzero(local_valid)),
        "current_valid_fraction": float(np.count_nonzero(current_valid) / max(1, len(sample_rows))),
        "local_valid_fraction": float(np.count_nonzero(local_valid) / max(1, len(sample_rows))),
        "reason_counts": reason_counts,
        "local_correction_counts": correction_counts,
        "local_correction_limits": {
            "max_rotation_deg": float(args.max_local_rotation_deg),
            "max_translation_px": float(args.max_local_translation_px),
        },
        "fixed_anchors": {
            "anterior_eye_bottom_xy": [float(value) for value in fixed_anterior.tolist()],
            "posterior_swim_rostral_xy": [float(value) for value in fixed_posterior.tolist()],
        },
        "fixed_swim_reference": fixed_swim_summary,
        "fixed_eye_component_references": fixed_eye_component_summaries,
        "strategies": {
            "current": _aggregate_strategy(summary_rows, strategy="current"),
            "local": _aggregate_strategy(summary_rows, strategy="local"),
        },
        "outputs": {
            "samples_csv": str(sample_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "plot_png": str(plot_png),
        },
    }
    with summary_json.open("w") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"samples_csv: {sample_csv}")
    print(f"summary_csv: {summary_csv}")
    print(f"summary_json: {summary_json}")
    print(f"plot_png: {plot_png}")
    print(f"current_valid_frames: {aggregate['current_valid_frames']}")
    print(f"local_valid_frames: {aggregate['local_valid_frames']}")
    print(f"current_median_bpm: {aggregate['strategies']['current']['median_peak_bpm']}")
    print(f"local_median_bpm: {aggregate['strategies']['local']['median_peak_bpm']}")


if __name__ == "__main__":
    main()
