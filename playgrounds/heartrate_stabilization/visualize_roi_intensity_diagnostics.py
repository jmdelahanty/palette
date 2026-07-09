from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from _common import ensure_output_dir, load_config, polygon_mask, resolve_roi_rect, roi_rect_corners
from map_pixel_band_contributions import (
    _bandpass_matrix,
    _draw_polygon,
    _load_roi_pixel_traces,
    _safe_corr_with_reference,
)


def _mask_from_npz(path: Path, *, shape_hw: tuple[int, int]) -> np.ndarray:
    with np.load(path) as data:
        if "roi_mask" in data:
            mask = np.asarray(data["roi_mask"], dtype=bool)
            if mask.shape != shape_hw:
                raise ValueError(f"{path} roi_mask shape {mask.shape} does not match video shape {shape_hw}")
            return mask
        if "roi_x" not in data or "roi_y" not in data:
            raise ValueError(f"{path} lacks roi_mask or roi_x/roi_y arrays")
        x = np.asarray(data["roi_x"], dtype=np.int64)
        y = np.asarray(data["roi_y"], dtype=np.int64)
    mask = np.zeros(shape_hw, dtype=bool)
    inside = (x >= 0) & (x < shape_hw[1]) & (y >= 0) & (y < shape_hw[0])
    mask[y[inside], x[inside]] = True
    return mask


def _interpolate_short_gaps_all_segments(
    *,
    traces: np.ndarray,
    valid: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, int, list[tuple[int, int]]]:
    out = traces.astype(np.float64, copy=True)
    finite = valid & np.isfinite(out).all(axis=1)
    if int(np.count_nonzero(finite)) < 16:
        raise ValueError("Not enough valid trace rows for intensity diagnostics.")

    interpolated_rows = 0
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
            interpolated_rows += 1

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, ok in enumerate(finite.tolist()):
        if ok and start is None:
            start = idx
        if (not ok or idx == len(finite) - 1) and start is not None:
            stop = idx if not ok else idx + 1
            if stop - start >= 32:
                segments.append((start, stop))
            start = None
    if not segments:
        raise ValueError("No contiguous valid segment with at least 32 samples after interpolation.")
    return out, finite, interpolated_rows, segments


def _zscore(values: np.ndarray, *, axis: int | None = None) -> np.ndarray:
    arr = values.astype(np.float64, copy=False)
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    out = np.zeros_like(arr, dtype=np.float64)
    ok = std > np.finfo(float).eps
    np.divide(arr - mean, std, out=out, where=ok)
    return out


def _scatter_to_image(
    values: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    shape_hw: tuple[int, int],
    fill: float = math.nan,
) -> np.ndarray:
    image = np.full(shape_hw, float(fill), dtype=np.float32)
    image[y, x] = np.asarray(values, dtype=np.float32)
    return image


def _downsample_indices(length: int, max_count: int) -> np.ndarray:
    if length <= max_count:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, length - 1, int(max_count))).astype(np.int64))


def _window_summaries(
    *,
    frame_index: np.ndarray,
    bandpassed: np.ndarray,
    roi_signal: np.ndarray,
    fps: float,
    window_seconds: float,
    step_seconds: float,
    min_samples: int,
) -> list[dict[str, Any]]:
    time_s = (frame_index - frame_index[0]).astype(np.float64) / float(fps)
    duration_s = float(time_s[-1] - time_s[0]) if len(time_s) else 0.0
    row_valid = np.isfinite(roi_signal) & np.isfinite(bandpassed).all(axis=1)
    rows: list[dict[str, Any]] = []
    start_s = 0.0
    window_index = 0
    while start_s + window_seconds <= duration_s + 1e-9:
        stop_s = start_s + window_seconds
        selection = np.flatnonzero((time_s >= start_s) & (time_s < stop_s) & row_valid)
        if selection.size >= min_samples:
            traces = bandpassed[selection]
            reference = roi_signal[selection]
            corr = _safe_corr_with_reference(traces, reference)
            finite = corr[np.isfinite(corr)]
            if finite.size:
                rows.append(
                    {
                        "window_index": int(window_index),
                        "window_start_s": float(start_s),
                        "window_stop_s": float(stop_s),
                        "time_s": float(0.5 * (start_s + stop_s)),
                        "sample_count": int(selection.size),
                        "correlation_min": float(np.min(finite)),
                        "correlation_p10": float(np.percentile(finite, 10)),
                        "correlation_median": float(np.median(finite)),
                        "correlation_p90": float(np.percentile(finite, 90)),
                        "correlation_max": float(np.max(finite)),
                        "mean_abs_correlation": float(np.mean(np.abs(finite))),
                        "fraction_positive": float(np.mean(finite > 0)),
                        "fraction_gt_0p3": float(np.mean(finite > 0.3)),
                        "roi_bandpassed_std": float(np.std(reference)),
                    }
                )
        start_s += step_seconds
        window_index += 1
    return rows


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_output_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_frame_signal_csv(
    path: Path,
    *,
    frame_index: np.ndarray,
    fps: float,
    raw_mean: np.ndarray,
    roi_signal: np.ndarray,
    roi_signal_z: np.ndarray,
    top_mean: np.ndarray,
    agreement_fraction: np.ndarray,
    covariance_support: np.ndarray,
) -> None:
    ensure_output_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "time_s",
                "time_min",
                "raw_mask_mean_intensity",
                "roi_bandpassed_mean",
                "roi_bandpassed_z",
                "top_pixel_bandpassed_mean",
                "top_pixel_agreement_fraction",
                "top_pixel_covariance_support",
            ],
        )
        writer.writeheader()
        t = (frame_index - frame_index[0]).astype(np.float64) / float(fps)
        for idx in range(len(frame_index)):
            writer.writerow(
                {
                    "frame_index": int(frame_index[idx]),
                    "time_s": float(t[idx]),
                    "time_min": float(t[idx] / 60.0),
                    "raw_mask_mean_intensity": float(raw_mean[idx]),
                    "roi_bandpassed_mean": float(roi_signal[idx]),
                    "roi_bandpassed_z": float(roi_signal_z[idx]),
                    "top_pixel_bandpassed_mean": float(top_mean[idx]),
                    "top_pixel_agreement_fraction": float(agreement_fraction[idx]),
                    "top_pixel_covariance_support": float(covariance_support[idx]),
                }
            )


def _write_pixel_summary_csv(
    path: Path,
    *,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    band_power: np.ndarray,
    correlation: np.ndarray,
    signed_covariance: np.ndarray,
    darkening_support: np.ndarray,
    darkening_contrast: np.ndarray,
    order: np.ndarray,
    top_indices: np.ndarray,
) -> None:
    selected = np.zeros(pixel_x.shape[0], dtype=bool)
    selected[top_indices] = True
    rank = np.empty(pixel_x.shape[0], dtype=np.int64)
    rank[order] = np.arange(1, pixel_x.shape[0] + 1)
    ensure_output_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "stable_x",
                "stable_y",
                "selected_for_raster",
                "band_power",
                "correlation_with_roi_mean",
                "signed_covariance_with_roi_mean",
                "darkening_support",
                "darkening_contrast",
            ],
        )
        writer.writeheader()
        for idx in order.tolist():
            writer.writerow(
                {
                    "rank": int(rank[idx]),
                    "stable_x": int(pixel_x[idx]),
                    "stable_y": int(pixel_y[idx]),
                    "selected_for_raster": int(selected[idx]),
                    "band_power": float(band_power[idx]),
                    "correlation_with_roi_mean": float(correlation[idx]),
                    "signed_covariance_with_roi_mean": float(signed_covariance[idx]),
                    "darkening_support": float(darkening_support[idx]),
                    "darkening_contrast": float(darkening_contrast[idx]),
                }
            )


def _load_hr_csv(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"time_s", "hr_bpm"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
        for row in reader:
            try:
                time_s = float(row["time_s"])
                bpm = float(row["hr_bpm"])
            except ValueError:
                continue
            if not (np.isfinite(time_s) and np.isfinite(bpm)):
                continue
            rolling = row.get("hr_bpm_rolling_median", "")
            try:
                rolling_bpm = float(rolling) if rolling not in {"", None} else math.nan
            except ValueError:
                rolling_bpm = math.nan
            rows.append({"time_s": time_s, "hr_bpm": bpm, "hr_bpm_rolling_median": rolling_bpm})
    if not rows:
        return None
    return {
        "time_s": np.asarray([row["time_s"] for row in rows], dtype=np.float64),
        "hr_bpm": np.asarray([row["hr_bpm"] for row in rows], dtype=np.float64),
        "hr_bpm_rolling_median": np.asarray([row["hr_bpm_rolling_median"] for row in rows], dtype=np.float64),
    }


def _time_axis(frame_index: np.ndarray, fps: float) -> tuple[np.ndarray, float, str]:
    time_s = (frame_index - frame_index[0]).astype(np.float64) / float(fps)
    if time_s.size and float(time_s[-1]) >= 180.0:
        return time_s / 60.0, 60.0, "time (min)"
    return time_s, 1.0, "time (s)"


def _write_plot(
    path: Path,
    *,
    mean_frame: np.ndarray,
    roi_polygon: np.ndarray,
    roi_mask: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    correlation: np.ndarray,
    frame_index: np.ndarray,
    fps: float,
    raw_mean: np.ndarray,
    roi_signal: np.ndarray,
    agreement_fraction: np.ndarray,
    covariance_support: np.ndarray,
    top_z: np.ndarray,
    window_rows: list[dict[str, Any]],
    hr: dict[str, np.ndarray] | None,
    band_min_hz: float,
    band_max_hz: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_output_dir(path.parent)
    shape = roi_mask.shape
    corr_image = _scatter_to_image(correlation, x=pixel_x, y=pixel_y, shape_hw=shape)
    pad = 8
    y0 = max(0, int(np.min(pixel_y)) - pad)
    y1 = min(shape[0], int(np.max(pixel_y)) + pad + 1)
    x0 = max(0, int(np.min(pixel_x)) - pad)
    x1 = min(shape[1], int(np.max(pixel_x)) + pad + 1)
    extent = [x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5]

    t, divisor, xlabel = _time_axis(frame_index, fps)
    time_indices = _downsample_indices(len(frame_index), 4500)
    raster_indices = _downsample_indices(len(frame_index), 2500)

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.2])

    ax = fig.add_subplot(grid[0, 0])
    mean_with_roi = _draw_polygon(mean_frame, roi_polygon, color=(0, 255, 255))
    crop = cv2.cvtColor(mean_with_roi[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    ax.imshow(crop, extent=extent)
    ax.scatter(pixel_x, pixel_y, s=6, c="#00ffff", alpha=0.75, linewidths=0)
    ax.set_title("Sampled pixels inside masked ROI")
    ax.set_xlim(x0 - 0.5, x1 - 0.5)
    ax.set_ylim(y1 - 0.5, y0 - 0.5)

    ax = fig.add_subplot(grid[0, 1])
    base = cv2.cvtColor(mean_frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    ax.imshow(base, extent=extent, alpha=0.65)
    masked = np.ma.masked_invalid(corr_image[y0:y1, x0:x1])
    im = ax.imshow(masked, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest", extent=extent, alpha=0.85)
    ax.contour(roi_mask[y0:y1, x0:x1].astype(np.uint8), levels=[0.5], colors="white", linewidths=0.6, extent=extent)
    ax.set_title("Full-recording pixel correlation")
    ax.set_xlim(x0 - 0.5, x1 - 0.5)
    ax.set_ylim(y1 - 0.5, y0 - 0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(t[time_indices], raw_mean[time_indices], lw=0.7, color="#2f5d7c")
    ax.set_title("Raw masked-ROI mean intensity")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("uint8 mean")
    ax.grid(True, alpha=0.2)
    if hr is not None:
        ax_hr = ax.twinx()
        hr_time = hr["time_s"] / divisor
        rolling = hr["hr_bpm_rolling_median"]
        if np.isfinite(rolling).any():
            ax_hr.plot(hr_time, rolling, lw=1.0, color="#a23b72", alpha=0.9, label="HR rolling median")
        else:
            ax_hr.plot(hr_time, hr["hr_bpm"], lw=0.8, color="#a23b72", alpha=0.8, label="HR")
        ax_hr.set_ylabel("HR bpm")
        ax_hr.tick_params(axis="y", labelcolor="#a23b72")

    ax = fig.add_subplot(grid[1, 1])
    ax.plot(t[time_indices], roi_signal[time_indices], lw=0.7, color="#6042a6", label="band-passed ROI mean")
    ax.set_title(f"Band-passed ROI signal and per-frame support, {band_min_hz:g}-{band_max_hz:g} Hz")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("intensity delta")
    ax.grid(True, alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(t[time_indices], agreement_fraction[time_indices], lw=0.7, color="#2d7f5e", alpha=0.75)
    ax2.set_ylabel("top-pixel sign agreement")
    ax2.set_ylim(-0.03, 1.03)
    ax2.tick_params(axis="y", labelcolor="#2d7f5e")

    ax = fig.add_subplot(grid[2, 0])
    raster = top_z[raster_indices].T
    raster_extent = [float(t[raster_indices[0]]), float(t[raster_indices[-1]]), raster.shape[0] + 0.5, 0.5]
    im = ax.imshow(raster, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3, interpolation="nearest", extent=raster_extent)
    ax.set_title("Top pixels: band-passed intensity z-score")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("pixel rank")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(grid[2, 1])
    if window_rows:
        win_t = np.asarray([float(row["time_s"]) / divisor for row in window_rows], dtype=np.float64)
        median_corr = np.asarray([float(row["correlation_median"]) for row in window_rows], dtype=np.float64)
        p90_corr = np.asarray([float(row["correlation_p90"]) for row in window_rows], dtype=np.float64)
        frac_gt = np.asarray([float(row["fraction_gt_0p3"]) for row in window_rows], dtype=np.float64)
        ax.plot(win_t, median_corr, lw=1.0, color="#1f77b4", label="median corr")
        ax.plot(win_t, p90_corr, lw=1.0, color="#d62728", label="p90 corr")
        ax.plot(win_t, frac_gt, lw=1.0, color="#2ca02c", label="fraction corr > 0.3")
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Sliding-window pixel agreement")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("correlation / fraction")
    ax.set_ylim(-1.02, 1.02)
    ax.grid(True, alpha=0.2)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_darkening_map(
    path: Path,
    *,
    mean_frame: np.ndarray,
    roi_polygon: np.ndarray,
    roi_mask: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    darkening_support: np.ndarray,
    darkening_contrast: np.ndarray,
    darkening_threshold_z: float,
    dark_frame_count: int,
    bright_frame_count: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_output_dir(path.parent)
    shape = roi_mask.shape
    support_image = _scatter_to_image(darkening_support, x=pixel_x, y=pixel_y, shape_hw=shape)
    contrast_image = _scatter_to_image(darkening_contrast, x=pixel_x, y=pixel_y, shape_hw=shape)
    pad = 8
    y0 = max(0, int(np.min(pixel_y)) - pad)
    y1 = min(shape[0], int(np.max(pixel_y)) + pad + 1)
    x0 = max(0, int(np.min(pixel_x)) - pad)
    x1 = min(shape[1], int(np.max(pixel_x)) + pad + 1)
    extent = [x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5]

    finite = darkening_support[np.isfinite(darkening_support)]
    vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    mean_with_roi = _draw_polygon(mean_frame, roi_polygon, color=(0, 255, 255))
    axes[0].imshow(cv2.cvtColor(mean_with_roi[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), extent=extent)
    axes[0].scatter(pixel_x, pixel_y, s=6, c="#00ffff", alpha=0.75, linewidths=0)
    axes[0].set_title("Sampled masked ROI pixels")

    masked_support = np.ma.masked_invalid(support_image[y0:y1, x0:x1])
    im = axes[1].imshow(
        masked_support,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        extent=extent,
    )
    axes[1].contour(roi_mask[y0:y1, x0:x1].astype(np.uint8), levels=[0.5], colors="white", linewidths=0.6, extent=extent)
    axes[1].set_title("Darkening support\npositive = darker during ROI dark phase")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    masked_contrast = np.ma.masked_invalid(contrast_image[y0:y1, x0:x1])
    im = axes[2].imshow(
        masked_contrast,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        extent=extent,
    )
    axes[2].contour(roi_mask[y0:y1, x0:x1].astype(np.uint8), levels=[0.5], colors="white", linewidths=0.6, extent=extent)
    axes[2].set_title("Raw contrast\nmean dark phase - mean bright phase")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for axis in axes:
        axis.set_xlim(x0 - 0.5, x1 - 0.5)
        axis.set_ylim(y1 - 0.5, y0 - 0.5)
    fig.suptitle(
        f"Directional darkening QC, ROI z <= -{darkening_threshold_z:g} "
        f"n={dark_frame_count}, bright n={bright_frame_count}"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _sort_order(
    *,
    mode: str,
    band_power: np.ndarray,
    correlation: np.ndarray,
    signed_covariance: np.ndarray,
) -> np.ndarray:
    if mode == "correlation":
        score = correlation
    elif mode == "abs_correlation":
        score = np.abs(correlation)
    elif mode == "band_power":
        score = band_power
    elif mode == "abs_covariance":
        score = np.abs(signed_covariance)
    elif mode == "covariance":
        score = signed_covariance
    else:
        raise ValueError(f"Unsupported sort mode: {mode}")
    return np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize frame-wise and window-wise intensity diagnostics inside a stabilized ROI mask."
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--video", type=Path, required=True, help="Stabilized video to sample.")
    parser.add_argument("--roi-json", type=Path, default=None, help="ROI JSON written by draw_roi.py.")
    parser.add_argument("--roi", type=str, default=None, help="Stabilized ROI rectangle x,y,width,height.")
    parser.add_argument("--status-csv", type=Path, default=None, help="Optional stabilized-video status CSV.")
    parser.add_argument("--mask-npz", type=Path, default=None, help="Optional pixel-band NPZ whose roi_mask defines pixels to sample.")
    parser.add_argument("--hr-csv", type=Path, default=None, help="Optional HR time-series CSV to overlay.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=0, help="0 means through the end of the video.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--band-min-hz", type=float, default=1.5)
    parser.add_argument("--band-max-hz", type=float, default=3.0)
    parser.add_argument(
        "--min-roi-mean-intensity",
        type=float,
        default=None,
        help=(
            "Mark frames invalid when the sampled ROI mean intensity is at or below this threshold. "
            "Use 0 or 1 to reject all-black acquisition-dropout crop frames."
        ),
    )
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--window-step-seconds", type=float, default=2.5)
    parser.add_argument("--min-window-samples", type=int, default=64)
    parser.add_argument("--max-interpolated-gap-samples", type=int, default=5)
    parser.add_argument(
        "--darkening-phase-z-threshold",
        type=float,
        default=0.75,
        help=(
            "Use ROI band-passed z <= -threshold as dark phase and z >= threshold as bright phase "
            "when computing directional darkening support."
        ),
    )
    parser.add_argument("--top-pixels", type=int, default=120)
    parser.add_argument(
        "--sort-by",
        choices=("covariance", "abs_covariance", "correlation", "abs_correlation", "band_power"),
        default="covariance",
        help="Pixel ordering for the raster and per-frame support traces.",
    )
    parser.add_argument(
        "--save-top-matrix",
        action="store_true",
        help="Store the top-pixel bandpassed z-score matrix in the NPZ output.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/roi_intensity_diagnostics"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    roi_rect = resolve_roi_rect(config, roi=args.roi, roi_json=args.roi_json)
    roi_polygon = roi_rect_corners(roi_rect)
    sample_mask = None
    if args.mask_npz is not None:
        from map_pixel_band_contributions import _video_shape

        sample_mask = _mask_from_npz(args.mask_npz, shape_hw=_video_shape(args.video))

    loaded = _load_roi_pixel_traces(
        video_path=args.video,
        roi_polygon=roi_polygon,
        status_csv=args.status_csv,
        frame_start=max(0, int(args.frame_start)),
        frame_count=max(0, int(args.frame_count)),
        stride=max(1, int(args.stride)),
        sample_mask=sample_mask,
        min_roi_mean_intensity=args.min_roi_mean_intensity,
    )
    frame_index = loaded["frame_indices"]
    traces, finite_rows, interpolated_rows, segments = _interpolate_short_gaps_all_segments(
        traces=loaded["traces"],
        valid=loaded["valid"],
        max_gap=max(0, int(args.max_interpolated_gap_samples)),
    )
    effective_fps = float(args.fps) / max(1, int(args.stride))
    raw_mean = np.full(traces.shape[0], math.nan, dtype=np.float64)
    raw_mean[finite_rows] = np.mean(traces[finite_rows], axis=1)
    bandpassed = np.full(traces.shape, math.nan, dtype=np.float64)
    skipped_segments: list[dict[str, Any]] = []
    for start, stop in segments:
        try:
            bandpassed[start:stop] = _bandpass_matrix(
                traces[start:stop],
                fps=effective_fps,
                band_min_hz=float(args.band_min_hz),
                band_max_hz=float(args.band_max_hz),
            )
        except Exception as exc:
            skipped_segments.append(
                {
                    "start_loaded_row": int(start),
                    "stop_loaded_row": int(stop),
                    "length": int(stop - start),
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )

    analysis_rows = np.isfinite(bandpassed).all(axis=1)
    if int(np.count_nonzero(analysis_rows)) < 16:
        raise ValueError("No band-passed segment produced at least 16 finite samples.")
    roi_signal = np.full(traces.shape[0], math.nan, dtype=np.float64)
    roi_signal[analysis_rows] = np.mean(bandpassed[analysis_rows], axis=1)
    roi_signal_z = np.full_like(roi_signal, math.nan)
    roi_signal_z[analysis_rows] = _zscore(roi_signal[analysis_rows])

    analysis_bandpassed = bandpassed[analysis_rows]
    analysis_roi_signal = roi_signal[analysis_rows]
    band_power = np.mean(analysis_bandpassed * analysis_bandpassed, axis=0)
    correlation = _safe_corr_with_reference(analysis_bandpassed, analysis_roi_signal)
    signed_covariance = np.mean(
        (analysis_bandpassed - np.mean(analysis_bandpassed, axis=0))
        * (analysis_roi_signal[:, None] - np.mean(analysis_roi_signal)),
        axis=0,
    )
    analysis_pixel_z = _zscore(analysis_bandpassed, axis=0)
    analysis_roi_z = _zscore(analysis_roi_signal)
    dark_threshold = abs(float(args.darkening_phase_z_threshold))
    dark_phase = analysis_roi_z <= -dark_threshold
    bright_phase = analysis_roi_z >= dark_threshold
    if int(np.count_nonzero(dark_phase)) and int(np.count_nonzero(bright_phase)):
        darkening_contrast = np.mean(analysis_pixel_z[dark_phase], axis=0) - np.mean(
            analysis_pixel_z[bright_phase],
            axis=0,
        )
        darkening_support = -darkening_contrast
    else:
        darkening_contrast = np.full(analysis_bandpassed.shape[1], math.nan, dtype=np.float64)
        darkening_support = np.full(analysis_bandpassed.shape[1], math.nan, dtype=np.float64)

    order = _sort_order(
        mode=str(args.sort_by),
        band_power=band_power,
        correlation=correlation,
        signed_covariance=signed_covariance,
    )
    top_count = min(max(1, int(args.top_pixels)), int(order.size))
    top_indices = order[:top_count]
    top_bandpassed = np.full((bandpassed.shape[0], top_count), math.nan, dtype=np.float64)
    top_bandpassed[analysis_rows] = analysis_bandpassed[:, top_indices]
    top_z = np.full_like(top_bandpassed, math.nan)
    top_z[analysis_rows] = _zscore(top_bandpassed[analysis_rows], axis=0)
    top_mean = np.full(bandpassed.shape[0], math.nan, dtype=np.float64)
    top_mean[analysis_rows] = np.mean(top_bandpassed[analysis_rows], axis=1)
    roi_sign = np.sign(roi_signal_z)
    pixel_sign = np.sign(top_z)
    nonzero = (
        np.isfinite(roi_signal_z[:, None])
        & np.isfinite(top_z)
        & (np.abs(roi_signal_z[:, None]) > 1e-12)
        & (np.abs(top_z) > 1e-12)
    )
    same_sign = pixel_sign == roi_sign[:, None]
    agreement = np.where(nonzero, same_sign.astype(np.float64), math.nan)
    agreement_fraction = np.full(top_z.shape[0], math.nan, dtype=np.float64)
    covariance_support = np.full(top_z.shape[0], math.nan, dtype=np.float64)
    rows_with_support = np.any(nonzero, axis=1)
    agreement_fraction[rows_with_support] = np.nanmean(agreement[rows_with_support], axis=1)
    covariance_support[rows_with_support] = np.nanmean(
        (top_z * roi_signal_z[:, None])[rows_with_support],
        axis=1,
    )

    window_rows = _window_summaries(
        frame_index=frame_index,
        bandpassed=bandpassed,
        roi_signal=roi_signal,
        fps=effective_fps,
        window_seconds=float(args.window_seconds),
        step_seconds=float(args.window_step_seconds),
        min_samples=max(16, int(args.min_window_samples)),
    )
    hr = _load_hr_csv(args.hr_csv)

    output_prefix = args.output_prefix
    ensure_output_dir(output_prefix.parent)
    plot_path = output_prefix.with_suffix(".png")
    darkening_map_path = output_prefix.with_suffix(".darkening_support.png")
    frame_csv = output_prefix.with_suffix(".frame_signal.csv")
    window_csv = output_prefix.with_suffix(".window_correlation.csv")
    pixel_csv = output_prefix.with_suffix(".pixel_summary.csv")
    npz_path = output_prefix.with_suffix(".npz")
    summary_path = output_prefix.with_suffix(".summary.json")

    _write_plot(
        plot_path,
        mean_frame=loaded["mean_frame"],
        roi_polygon=roi_polygon,
        roi_mask=loaded["roi_mask"],
        pixel_x=loaded["roi_x"],
        pixel_y=loaded["roi_y"],
        correlation=correlation,
        frame_index=frame_index,
        fps=effective_fps,
        raw_mean=raw_mean,
        roi_signal=roi_signal,
        agreement_fraction=agreement_fraction,
        covariance_support=covariance_support,
        top_z=top_z,
        window_rows=window_rows,
        hr=hr,
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
    )
    _write_darkening_map(
        darkening_map_path,
        mean_frame=loaded["mean_frame"],
        roi_polygon=roi_polygon,
        roi_mask=loaded["roi_mask"],
        pixel_x=loaded["roi_x"],
        pixel_y=loaded["roi_y"],
        darkening_support=darkening_support,
        darkening_contrast=darkening_contrast,
        darkening_threshold_z=dark_threshold,
        dark_frame_count=int(np.count_nonzero(dark_phase)),
        bright_frame_count=int(np.count_nonzero(bright_phase)),
    )
    _write_frame_signal_csv(
        frame_csv,
        frame_index=frame_index,
        fps=effective_fps,
        raw_mean=raw_mean,
        roi_signal=roi_signal,
        roi_signal_z=roi_signal_z,
        top_mean=top_mean,
        agreement_fraction=agreement_fraction,
        covariance_support=covariance_support,
    )
    _write_dict_rows(window_csv, window_rows)
    _write_pixel_summary_csv(
        pixel_csv,
        pixel_x=loaded["roi_x"],
        pixel_y=loaded["roi_y"],
        band_power=band_power,
        correlation=correlation,
        signed_covariance=signed_covariance,
        darkening_support=darkening_support,
        darkening_contrast=darkening_contrast,
        order=order,
        top_indices=top_indices,
    )

    npz_payload: dict[str, Any] = {
        "frame_index": frame_index.astype(np.int64),
        "raw_mean": raw_mean.astype(np.float32),
        "roi_bandpassed_mean": roi_signal.astype(np.float32),
        "roi_bandpassed_z": roi_signal_z.astype(np.float32),
        "top_pixel_agreement_fraction": agreement_fraction.astype(np.float32),
        "top_pixel_covariance_support": covariance_support.astype(np.float32),
        "pixel_x": loaded["roi_x"].astype(np.int32),
        "pixel_y": loaded["roi_y"].astype(np.int32),
        "band_power": band_power.astype(np.float32),
        "correlation": correlation.astype(np.float32),
        "signed_covariance": signed_covariance.astype(np.float32),
        "darkening_support": darkening_support.astype(np.float32),
        "darkening_contrast": darkening_contrast.astype(np.float32),
        "top_pixel_indices": top_indices.astype(np.int32),
    }
    if args.save_top_matrix:
        npz_payload["top_pixel_bandpassed_z"] = top_z.astype(np.float32)
    np.savez_compressed(npz_path, **npz_payload)

    finite_corr = correlation[np.isfinite(correlation)]
    finite_darkening_support = darkening_support[np.isfinite(darkening_support)]
    finite_raw_mean = raw_mean[np.isfinite(raw_mean)]
    finite_roi_signal = roi_signal[np.isfinite(roi_signal)]
    summary = {
        "source_video": str(args.video),
        "status_csv": str(args.status_csv) if args.status_csv is not None else None,
        "roi_json": str(args.roi_json) if args.roi_json is not None else None,
        "mask_npz": str(args.mask_npz) if args.mask_npz is not None else None,
        "hr_csv": str(args.hr_csv) if args.hr_csv is not None else None,
        "roi_rect_stable_xywh": [float(value) for value in roi_rect],
        "frame_start": int(args.frame_start),
        "frame_count_requested": int(args.frame_count),
        "stride": int(args.stride),
        "fps": effective_fps,
        "band_hz": [float(args.band_min_hz), float(args.band_max_hz)],
        "window_seconds": float(args.window_seconds),
        "window_step_seconds": float(args.window_step_seconds),
        "loaded_frames": int(loaded["traces"].shape[0]),
        "valid_loaded_frames": int(np.count_nonzero(loaded["valid"])),
        "min_roi_mean_intensity": loaded["min_roi_mean_intensity"],
        "low_intensity_frame_count": int(loaded["low_intensity_frame_count"]),
        "analysis_frames": int(np.count_nonzero(analysis_rows)),
        "valid_segments": [
            {
                "start_loaded_row": int(start),
                "stop_loaded_row": int(stop),
                "start_frame_index": int(frame_index[start]),
                "stop_frame_index_inclusive": int(frame_index[stop - 1]),
                "length": int(stop - start),
            }
            for start, stop in segments
        ],
        "skipped_segments": skipped_segments,
        "interpolated_rows": int(interpolated_rows),
        "roi_pixel_count": int(traces.shape[1]),
        "top_pixels": int(top_count),
        "sort_by": str(args.sort_by),
        "darkening_phase_z_threshold": float(dark_threshold),
        "darkening_phase_frame_count": int(np.count_nonzero(dark_phase)),
        "bright_phase_frame_count": int(np.count_nonzero(bright_phase)),
        "darkening_support_median": float(np.median(finite_darkening_support))
        if finite_darkening_support.size
        else math.nan,
        "darkening_support_p90": float(np.percentile(finite_darkening_support, 90))
        if finite_darkening_support.size
        else math.nan,
        "fraction_pixels_positive_darkening_support": float(np.mean(finite_darkening_support > 0))
        if finite_darkening_support.size
        else math.nan,
        "raw_mean_intensity_median": float(np.median(finite_raw_mean)) if finite_raw_mean.size else math.nan,
        "raw_mean_intensity_std": float(np.std(finite_raw_mean)) if finite_raw_mean.size else math.nan,
        "roi_bandpassed_std": float(np.std(finite_roi_signal)) if finite_roi_signal.size else math.nan,
        "correlation_min": float(np.min(finite_corr)) if finite_corr.size else math.nan,
        "correlation_median": float(np.median(finite_corr)) if finite_corr.size else math.nan,
        "correlation_p90": float(np.percentile(finite_corr, 90)) if finite_corr.size else math.nan,
        "correlation_max": float(np.max(finite_corr)) if finite_corr.size else math.nan,
        "fraction_pixels_positive_correlation": float(np.mean(finite_corr > 0)) if finite_corr.size else math.nan,
        "fraction_pixels_correlation_gt_0p3": float(np.mean(finite_corr > 0.3)) if finite_corr.size else math.nan,
        "agreement_fraction_median": float(np.nanmedian(agreement_fraction)),
        "agreement_fraction_p10": float(np.nanpercentile(agreement_fraction, 10)),
        "agreement_fraction_p90": float(np.nanpercentile(agreement_fraction, 90)),
        "outputs": {
            "plot_png": str(plot_path),
            "darkening_support_png": str(darkening_map_path),
            "frame_signal_csv": str(frame_csv),
            "window_correlation_csv": str(window_csv),
            "pixel_summary_csv": str(pixel_csv),
            "npz": str(npz_path),
        },
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"summary_json: {summary_path}")
    print(f"plot_png: {plot_path}")
    print(f"darkening_support_png: {darkening_map_path}")
    print(f"frame_signal_csv: {frame_csv}")
    print(f"window_correlation_csv: {window_csv}")
    print(f"pixel_summary_csv: {pixel_csv}")
    print(f"npz: {npz_path}")
    print(f"analysis_frames: {summary['analysis_frames']}")
    print(f"roi_pixel_count: {summary['roi_pixel_count']}")
    print(f"correlation_median: {summary['correlation_median']:.6g}")
    print(f"correlation_p90: {summary['correlation_p90']:.6g}")
    print(f"agreement_fraction_median: {summary['agreement_fraction_median']:.6g}")


if __name__ == "__main__":
    main()
