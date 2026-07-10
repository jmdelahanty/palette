from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np

from fisheye.analysis.dynamic_heart_support import (
    CrossfitHeartPhaseSeries,
    DynamicHeartSupportResult,
)
from fisheye.analysis.local_rostral_heartrate import LocalCoordinateDataset


def _robust_scale(values: np.ndarray, *, axis: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(arr, axis=axis)
        if axis is None:
            deviations = np.abs(arr - median)
        else:
            deviations = np.abs(arr - np.expand_dims(median, axis=axis))
        scale = 1.4826 * np.nanmedian(deviations, axis=axis)
        fallback = np.nanstd(arr, axis=axis)
    return np.where(
        np.isfinite(scale) & (scale > np.finfo(float).eps),
        scale,
        np.where(np.isfinite(fallback) & (fallback > np.finfo(float).eps), fallback, 1.0),
    )


def _pixel_geometry(dataset: LocalCoordinateDataset, margin_px: int = 1) -> dict[str, Any]:
    xy = np.rint(np.asarray(dataset.pixel_xy, dtype=np.float64)).astype(np.int64)
    height, width = dataset.image_shape_hw
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    if not np.any(inside):
        raise ValueError("dataset has no pixels inside the canonical image")
    x0 = max(0, int(np.min(xy[inside, 0])) - int(margin_px))
    y0 = max(0, int(np.min(xy[inside, 1])) - int(margin_px))
    x1 = min(width, int(np.max(xy[inside, 0])) + int(margin_px) + 1)
    y1 = min(height, int(np.max(xy[inside, 1])) + int(margin_px) + 1)
    return {
        "xy": xy,
        "inside": inside,
        "bbox_xyxy": (x0, y0, x1, y1),
        "local_x": xy[:, 0] - x0,
        "local_y": xy[:, 1] - y0,
        "shape_hw": (y1 - y0, x1 - x0),
    }


def _scatter(values: np.ndarray, geometry: Mapping[str, Any], *, fill: float = np.nan) -> np.ndarray:
    image = np.full(tuple(geometry["shape_hw"]), fill, dtype=np.float64)
    inside = np.asarray(geometry["inside"], dtype=bool)
    image[
        np.asarray(geometry["local_y"])[inside],
        np.asarray(geometry["local_x"])[inside],
    ] = np.asarray(values, dtype=np.float64)[inside]
    return image


def _support_raster(
    selected: np.ndarray,
    geometry: Mapping[str, Any],
) -> np.ndarray:
    return np.asarray(_scatter(np.asarray(selected, dtype=np.float64), geometry, fill=0.0) > 0.5)


def _blend_values(
    base_bgr: np.ndarray,
    values: np.ndarray,
    alpha: np.ndarray,
    *,
    cmap_name: str,
    value_min: float,
    value_max: float,
) -> np.ndarray:
    from matplotlib import colormaps

    output = np.asarray(base_bgr, dtype=np.uint8).copy()
    finite = np.isfinite(values) & np.isfinite(alpha) & (alpha > 0.0)
    if not np.any(finite):
        return output
    normalized = np.clip(
        (np.asarray(values, dtype=np.float64) - float(value_min))
        / max(float(value_max) - float(value_min), np.finfo(float).eps),
        0.0,
        1.0,
    )
    rgb = np.asarray(colormaps[cmap_name](normalized)[..., :3] * 255.0, dtype=np.uint8)
    bgr = rgb[..., ::-1]
    local_alpha = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 1.0)[..., None]
    blended = output.astype(np.float64) * (1.0 - local_alpha) + bgr.astype(np.float64) * local_alpha
    output[finite] = np.clip(blended[finite], 0, 255).astype(np.uint8)
    return output


def _draw_contour(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    import cv2

    contours, _hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(image, contours, -1, color, int(thickness), cv2.LINE_AA)


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.45,
    color: tuple[int, int, int] = (25, 25, 25),
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _resize_nearest(image: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(
        np.asarray(image, dtype=np.uint8),
        (int(size), int(size)),
        interpolation=cv2.INTER_NEAREST,
    )


def _trace_panel(
    phase: CrossfitHeartPhaseSeries,
    timestamps_s: np.ndarray,
    row: int,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    import cv2

    panel = np.full((int(height), int(width), 3), 250, dtype=np.uint8)
    left, right, top, bottom = 56, 14, 24, 22
    plot_width = max(1, width - left - right)
    plot_height = max(1, height - top - bottom)
    times = np.asarray(timestamps_s, dtype=np.float64)
    relative = times - float(times[0])
    t_stop = max(float(relative[-1]), np.finfo(float).eps)
    x = left + np.rint(relative / t_stop * (plot_width - 1)).astype(np.int64)
    latent = np.asarray(phase.latent_analytic.real, dtype=np.float64)
    center = float(np.nanmedian(latent))
    scale = float(_robust_scale(latent))
    normalized = np.clip((latent - center) / (3.0 * scale), -1.0, 1.0)
    drawable = np.nan_to_num(normalized, nan=0.0)
    y = top + np.rint((1.0 - (drawable + 1.0) * 0.5) * (plot_height - 1)).astype(np.int64)
    valid = np.asarray(phase.frame_valid, dtype=bool) & np.isfinite(latent)
    for start in range(1, len(times)):
        if valid[start - 1] and valid[start]:
            cv2.line(
                panel,
                (int(x[start - 1]), int(y[start - 1])),
                (int(x[start]), int(y[start])),
                (30, 95, 185),
                1,
                cv2.LINE_AA,
            )
    zero_y = top + plot_height // 2
    cv2.line(panel, (left, zero_y), (width - right, zero_y), (205, 205, 205), 1)
    cursor_x = int(x[row])
    cv2.line(panel, (cursor_x, top), (cursor_x, top + plot_height), (25, 25, 25), 1)
    _draw_text(panel, "phase-aligned latent trace", (6, 16), scale=0.42)
    _draw_text(panel, "cross-fit diagnostic; not event validation", (width - 300, 16), scale=0.38)
    _draw_text(panel, "0 s", (left, height - 5), scale=0.34, color=(80, 80, 80))
    _draw_text(
        panel,
        f"{relative[-1]:.1f} s",
        (width - right - 46, height - 5),
        scale=0.34,
        color=(80, 80, 80),
    )
    return panel


def _write_phase_strip(
    path: Path,
    dataset: LocalCoordinateDataset,
    dynamic: DynamicHeartSupportResult,
    phase: CrossfitHeartPhaseSeries,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    support_pixels = np.flatnonzero(phase.heart_support)
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    relative = timestamps - float(timestamps[0])
    actual = np.asarray(phase.analytic_residual[:, support_pixels], dtype=np.complex128)
    aligned = np.full_like(actual, np.nan + 0j)
    loading_phase_by_row = np.full_like(actual, np.nan + 0j)
    for row, fold_index in enumerate(np.asarray(phase.model_fold_indices, dtype=np.int64)):
        if fold_index < 0:
            continue
        loading = phase.fold_loadings[fold_index, support_pixels]
        amplitude = np.abs(loading)
        loading_phase_by_row[row] = np.divide(
            loading,
            amplitude,
            out=np.full(loading.shape, np.nan + 0j),
            where=amplitude > np.finfo(float).eps,
        )
        aligned[row] = np.conjugate(loading_phase_by_row[row]) * actual[row]
    mean_loading = np.nanmean(phase.fold_loadings[:, support_pixels], axis=0)
    order = np.argsort(np.angle(mean_loading))
    actual_phase = np.angle(actual[:, order]).T
    aligned_phase = np.angle(aligned[:, order]).T
    maximum_columns = 1600
    stride = max(1, int(math.ceil(dataset.frame_count / maximum_columns)))
    actual_phase = actual_phase[:, ::stride]
    aligned_phase = aligned_phase[:, ::stride]
    strip_times = relative[::stride]

    fig = plt.figure(figsize=(14, 8), constrained_layout=True, facecolor="white")
    grid = fig.add_gridspec(3, 1, height_ratios=(1.0, 1.0, 0.75))
    axes = [fig.add_subplot(grid[index, 0]) for index in range(3)]
    cmap = matplotlib.colormaps["twilight_shifted"].copy()
    cmap.set_bad("0.82")
    extent = [float(strip_times[0]), float(strip_times[-1]), 0, len(order)]
    image = axes[0].imshow(
        np.ma.masked_invalid(actual_phase),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-np.pi,
        vmax=np.pi,
        extent=extent,
        origin="lower",
    )
    axes[0].set_title("Observed per-pixel phase, ordered by frozen spatial phase")
    axes[0].set_ylabel("heart-support pixel")
    axes[1].imshow(
        np.ma.masked_invalid(aligned_phase),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-np.pi,
        vmax=np.pi,
        extent=extent,
        origin="lower",
    )
    axes[1].set_title("Phase after subtracting the opposite-fold pixel loading")
    axes[1].set_ylabel("same pixel order")
    latent = np.asarray(phase.latent_analytic, dtype=np.complex128)
    latent_scale = float(_robust_scale(latent.real))
    axes[2].plot(relative, latent.real / latent_scale, color="black", lw=0.8, label="latent real / robust scale")
    axes[2].plot(relative, phase.spatial_alignment, color="#c23b33", lw=0.8, label="spatial alignment")
    axes[2].axhline(0.0, color="0.7", lw=0.8)
    axes[2].set_xlabel("time from interval start (s)")
    axes[2].set_ylabel("diagnostic value")
    axes[2].legend(loc="upper right", fontsize=8)
    fig.colorbar(image, ax=axes[:2], label="instantaneous phase (rad)", fraction=0.025)
    fig.suptitle(
        f"Frozen support cross-fit phase diagnostic | {phase.frequency_hz:.3f} Hz | "
        f"band {phase.band_min_hz:.2f}-{phase.band_max_hz:.2f} Hz\n"
        "Frequency was selected by the dynamic-support search; this is not cardiac-event validation.",
        fontsize=12,
    )
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def _write_phase_video(
    path: Path,
    dataset: LocalCoordinateDataset,
    dynamic: DynamicHeartSupportResult,
    phase: CrossfitHeartPhaseSeries,
    *,
    frame_stride: int,
    playback_fps: float,
    panel_size: int,
) -> int:
    import cv2

    geometry = _pixel_geometry(dataset, margin_px=1)
    heart = _support_raster(phase.heart_support, geometry)
    fold0 = _support_raster(dynamic.pixel_groups["fold0_only"], geometry)
    fold1 = _support_raster(dynamic.pixel_groups["fold1_only"], geometry)
    anatomical = _support_raster(dynamic.pixel_groups["anatomical_only"], geometry)
    support_pixels = np.flatnonzero(phase.heart_support)
    traces = np.asarray(dataset.traces, dtype=np.float64)
    finite_traces = traces[np.isfinite(traces)]
    low, high = np.quantile(finite_traces, [0.02, 0.98])
    if not high > low:
        high = low + 1.0
    median_values = np.nanmedian(traces, axis=0)
    filtered_scale = np.asarray(_robust_scale(phase.bandpassed_residual, axis=0), dtype=np.float64)
    analytic_amplitude = np.abs(phase.analytic_residual)
    median_amplitude = np.ones(dataset.pixel_count, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_amplitude[support_pixels] = np.nanmedian(
            analytic_amplitude[:, support_pixels],
            axis=0,
        )
    median_amplitude = np.where(
        np.isfinite(median_amplitude) & (median_amplitude > np.finfo(float).eps),
        median_amplitude,
        1.0,
    )

    gap = 8
    title_height = 50
    trace_height = 124
    width = 3 * int(panel_size) + 4 * gap
    height = title_height + int(panel_size) + gap + trace_height
    width += width % 2
    height += height % 2
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(playback_fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")

    rendered = 0
    for row in range(0, dataset.frame_count, int(frame_stride)):
        raw_values = traces[row].copy()
        missing = ~np.isfinite(raw_values)
        raw_values[missing] = median_values[missing]
        raw = _scatter(raw_values, geometry)
        gray = np.clip((raw - low) / (high - low), 0.0, 1.0)
        base = np.asarray(np.nan_to_num(gray, nan=0.5) * 255.0, dtype=np.uint8)
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        first = base_bgr.copy()
        _draw_contour(first, heart, color=(15, 15, 15), thickness=1)
        _draw_contour(first, fold0, color=(20, 40, 230), thickness=1)
        _draw_contour(first, fold1, color=(230, 80, 30), thickness=1)
        _draw_contour(first, anatomical, color=(30, 180, 220), thickness=1)

        residual_values = np.divide(
            phase.bandpassed_residual[row],
            filtered_scale,
            out=np.full(dataset.pixel_count, np.nan),
            where=filtered_scale > np.finfo(float).eps,
        )
        residual_image = _scatter(residual_values, geometry)
        residual_alpha = np.zeros(dataset.pixel_count, dtype=np.float64)
        fold_index = int(phase.model_fold_indices[row])
        if fold_index >= 0:
            residual_alpha = 0.15 + 0.75 * phase.fold_loading_weights[fold_index]
        residual_alpha[~phase.heart_support] = 0.0
        residual_alpha[~np.isfinite(phase.bandpassed_residual[row])] = 0.0
        second = _blend_values(
            base_bgr,
            residual_image,
            _scatter(residual_alpha, geometry, fill=0.0),
            cmap_name="coolwarm",
            value_min=-3.0,
            value_max=3.0,
        )
        _draw_contour(second, heart, color=(20, 20, 20), thickness=1)

        phase_values = np.angle(phase.analytic_residual[row])
        phase_image = _scatter(phase_values, geometry)
        phase_alpha = np.zeros(dataset.pixel_count, dtype=np.float64)
        if fold_index >= 0:
            relative_amplitude = np.clip(
                analytic_amplitude[row] / (1.5 * median_amplitude),
                0.0,
                1.0,
            )
            phase_alpha = phase.fold_loading_weights[fold_index] * relative_amplitude
        phase_alpha[~phase.heart_support] = 0.0
        phase_alpha[~np.isfinite(phase.analytic_residual[row])] = 0.0
        third = base_bgr.copy()
        neutral = np.zeros(dataset.pixel_count, dtype=np.float64)
        neutral[phase.heart_support] = 0.30
        neutral_image = _scatter(neutral, geometry, fill=0.0)
        gray_target = np.full_like(third, 150)
        gray_alpha = neutral_image[..., None]
        third = np.asarray(third * (1.0 - gray_alpha) + gray_target * gray_alpha, dtype=np.uint8)
        visible_phase_alpha = np.zeros(dataset.pixel_count, dtype=np.float64)
        finite_phase = phase.heart_support & np.isfinite(phase.analytic_residual[row])
        visible_phase_alpha[finite_phase] = 0.35 + 0.65 * phase_alpha[finite_phase]
        third = _blend_values(
            third,
            phase_image,
            _scatter(visible_phase_alpha, geometry, fill=0.0),
            cmap_name="twilight_shifted",
            value_min=-np.pi,
            value_max=np.pi,
        )
        _draw_contour(third, heart, color=(20, 20, 20), thickness=1)

        panels = [_resize_nearest(item, int(panel_size)) for item in (first, second, third)]
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        titles = (
            "Local stabilized ROI samples + frozen contours",
            "Held-out band-limited activation",
            "Observed pixel phase (hue; confidence=opacity)",
        )
        for index, (panel, title) in enumerate(zip(panels, titles)):
            x0 = gap + index * (int(panel_size) + gap)
            canvas[title_height : title_height + int(panel_size), x0 : x0 + int(panel_size)] = panel
            _draw_text(canvas, title, (x0 + 3, 18), scale=0.40)
        _draw_text(
            canvas,
            "red=fold 0 only | blue=fold 1 only | gold=anatomical only",
            (gap + 3, 38),
            scale=0.32,
            color=(70, 70, 70),
        )
        phase_x0 = gap + 2 * (int(panel_size) + gap) + 26
        phase_width = int(panel_size) - 52
        phase_gradient = np.linspace(-np.pi, np.pi, phase_width, dtype=np.float64)[None, :]
        phase_bar = _blend_values(
            np.full((8, phase_width, 3), 255, dtype=np.uint8),
            np.repeat(phase_gradient, 8, axis=0),
            np.ones((8, phase_width), dtype=np.float64),
            cmap_name="twilight_shifted",
            value_min=-np.pi,
            value_max=np.pi,
        )
        canvas[29:37, phase_x0 : phase_x0 + phase_width] = phase_bar
        _draw_text(canvas, "-pi", (phase_x0 - 24, 38), scale=0.28, color=(70, 70, 70))
        _draw_text(
            canvas,
            "+pi",
            (phase_x0 + phase_width + 2, 38),
            scale=0.28,
            color=(70, 70, 70),
        )
        relative_time = float(dataset.timestamps_s[row] - dataset.timestamps_s[0])
        status = "valid" if bool(phase.frame_valid[row]) else "no phase estimate"
        alignment = float(phase.spatial_alignment[row])
        alignment_text = f"{alignment:.2f}" if np.isfinite(alignment) else "n/a"
        _draw_text(
            canvas,
            f"frame {int(dataset.frame_indices[row])} | t={relative_time:.3f}s | fold={fold_index} | "
            f"alignment={alignment_text} | {status}",
            (gap + 4, title_height + int(panel_size) - 10),
            scale=0.40,
            color=(245, 245, 245),
        )
        trace = _trace_panel(
            phase,
            dataset.timestamps_s,
            row,
            width=width - 2 * gap,
            height=trace_height,
        )
        trace_y = title_height + int(panel_size) + gap
        canvas[trace_y : trace_y + trace_height, gap : width - gap] = trace
        writer.write(canvas)
        rendered += 1
    writer.release()
    return rendered


def write_dynamic_phase_outputs(
    output_prefix: Path,
    dataset: LocalCoordinateDataset,
    dynamic: DynamicHeartSupportResult,
    phase: CrossfitHeartPhaseSeries,
    *,
    frame_stride: int = 3,
    playback_fps: float = 30.0,
    panel_size: int = 360,
) -> dict[str, Path]:
    if int(frame_stride) < 1:
        raise ValueError("frame_stride must be positive")
    if float(playback_fps) <= 0.0:
        raise ValueError("playback_fps must be positive")
    if int(panel_size) < 128:
        raise ValueError("panel_size must be at least 128")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = output_prefix.with_suffix(".dynamic_phase.arrays.npz")
    summary_path = output_prefix.with_suffix(".dynamic_phase.summary.json")
    strip_path = output_prefix.with_suffix(".dynamic_phase.strip.png")
    video_path = output_prefix.with_suffix(".dynamic_phase.mp4")
    np.savez_compressed(
        arrays_path,
        frame_indices=np.asarray(dataset.frame_indices, dtype=np.int64),
        timestamps_s=np.asarray(dataset.timestamps_s, dtype=np.float64),
        pixel_xy=np.asarray(dataset.pixel_xy, dtype=np.float32),
        heart_support=np.asarray(phase.heart_support, dtype=np.uint8),
        model_fold_indices=np.asarray(phase.model_fold_indices, dtype=np.int16),
        fold_loadings_real=np.asarray(phase.fold_loadings.real, dtype=np.float32),
        fold_loadings_imag=np.asarray(phase.fold_loadings.imag, dtype=np.float32),
        fold_loading_weights=np.asarray(phase.fold_loading_weights, dtype=np.float32),
        bandpassed_residual=np.asarray(phase.bandpassed_residual, dtype=np.float32),
        analytic_residual_real=np.asarray(phase.analytic_residual.real, dtype=np.float32),
        analytic_residual_imag=np.asarray(phase.analytic_residual.imag, dtype=np.float32),
        latent_analytic_real=np.asarray(phase.latent_analytic.real, dtype=np.float32),
        latent_analytic_imag=np.asarray(phase.latent_analytic.imag, dtype=np.float32),
        spatial_alignment=np.asarray(phase.spatial_alignment, dtype=np.float32),
        frame_valid=np.asarray(phase.frame_valid, dtype=np.uint8),
    )
    _write_phase_strip(strip_path, dataset, dynamic, phase)
    rendered = _write_phase_video(
        video_path,
        dataset,
        dynamic,
        phase,
        frame_stride=int(frame_stride),
        playback_fps=float(playback_fps),
        panel_size=int(panel_size),
    )
    timestamp_diffs = np.diff(np.asarray(dataset.timestamps_s, dtype=np.float64))
    effective_fps = 1.0 / float(np.median(timestamp_diffs))
    summary = {
        "diagnostic_only": True,
        "event_validation": False,
        "frequency_hz": float(phase.frequency_hz),
        "band_min_hz": float(phase.band_min_hz),
        "band_max_hz": float(phase.band_max_hz),
        "frequency_selection": "selected_by_dynamic_support_search",
        "support_source": str(dynamic.support_source),
        "confirmatory_eligible_support": bool(dynamic.confirmatory_eligible),
        "frame_count": int(dataset.frame_count),
        "phase_valid_frame_count": int(np.count_nonzero(phase.frame_valid)),
        "phase_valid_fraction": float(np.mean(phase.frame_valid)),
        "median_spatial_alignment": float(np.nanmedian(phase.spatial_alignment)),
        "rendered_frame_count": int(rendered),
        "frame_stride": int(frame_stride),
        "playback_fps": float(playback_fps),
        "source_effective_fps": float(effective_fps),
        "playback_speed_ratio": float(frame_stride) * float(playback_fps) / effective_fps,
        "pixel_phase_definition": "angle of held-out band-limited analytic residual before spatial alignment",
        "latent_trace_definition": "opposite-fold loading-phase-aligned weighted analytic residual",
        "invalid_policy": "long gaps, partition guards, and filter edges are not colored",
        "caveat": "The visualization diagnoses the selected periodic pattern; it does not identify cardiac contraction phase or validate beat events.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {
        "dynamic_phase_arrays_npz": arrays_path,
        "dynamic_phase_summary_json": summary_path,
        "dynamic_phase_strip_png": strip_path,
        "dynamic_phase_video_mp4": video_path,
    }
