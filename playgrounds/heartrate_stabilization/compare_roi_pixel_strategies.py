from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from _common import ensure_output_dir, load_config, polygon_mask, resolve_roi_rect, roi_rect_corners
from analyze_roi_rhythm import _analyze_trace
from map_pixel_band_contributions import (
    _bandpass_matrix,
    _load_roi_pixel_traces,
    _safe_corr_with_reference,
)


@dataclass(frozen=True)
class TraceSet:
    name: str
    traces: np.ndarray
    pixel_x: np.ndarray
    pixel_y: np.ndarray
    source: str


@dataclass(frozen=True)
class WindowData:
    index: int
    start_s: float
    stop_s: float
    rows: np.ndarray
    frame_index: np.ndarray
    traces: np.ndarray
    ok: bool
    reason: str


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"Expected positive top-k value, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one top-k value is required.")
    return tuple(dict.fromkeys(values))


def _parse_score_modes(raw: str) -> tuple[str, ...]:
    modes: list[str] = []
    for part in raw.split(","):
        mode = part.strip().lower()
        if not mode:
            continue
        if mode not in {"covariance", "correlation"}:
            raise ValueError(f"Unsupported score mode {mode!r}; expected covariance or correlation")
        modes.append(mode)
    if not modes:
        raise ValueError("At least one score mode is required.")
    return tuple(dict.fromkeys(modes))


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


def _window_ranges(
    *,
    frame_indices: np.ndarray,
    fps: float,
    frame_start: int,
    window_seconds: float,
    step_seconds: float,
    min_samples: int,
) -> list[tuple[int, float, float, np.ndarray]]:
    if window_seconds <= 0:
        raise ValueError("--window-seconds must be positive")
    if step_seconds <= 0:
        raise ValueError("--window-step-seconds must be positive")
    if frame_indices.size == 0:
        raise ValueError("No frame indices loaded")
    selected_start = int(frame_start)
    selected_stop = int(frame_indices[-1]) + 1
    duration_s = max(0.0, (selected_stop - selected_start) / float(fps))
    out: list[tuple[int, float, float, np.ndarray]] = []
    start_s = 0.0
    index = 0
    eps = 1e-9
    while start_s + window_seconds <= duration_s + eps:
        stop_s = start_s + window_seconds
        lo = selected_start + int(round(start_s * fps))
        hi = selected_start + int(round(stop_s * fps))
        rows = np.flatnonzero((frame_indices >= lo) & (frame_indices < hi))
        if rows.size >= min_samples:
            out.append((index, start_s, stop_s, rows))
        start_s += step_seconds
        index += 1
    if not out:
        raise ValueError(
            f"No windows with at least {min_samples} samples; "
            f"duration={duration_s:.3f}s window={window_seconds:.3f}s"
        )
    return out


def _interpolate_window(
    *,
    frame_index: np.ndarray,
    traces: np.ndarray,
    valid: np.ndarray,
    rows: np.ndarray,
    max_gap: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    selected_frames = frame_index[rows]
    selected_traces = traces[rows].astype(np.float64, copy=True)
    finite = valid[rows] & np.isfinite(selected_traces).all(axis=1)
    if int(np.count_nonzero(finite)) < min_samples:
        return selected_frames, selected_traces, False, "too_few_valid_rows"

    idx = 0
    interpolated = 0
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
        left = selected_traces[start - 1]
        right = selected_traces[stop]
        steps = float(stop - start + 1)
        for offset, row in enumerate(range(start, stop), start=1):
            frac = float(offset) / steps
            selected_traces[row] = (1.0 - frac) * left + frac * right
            finite[row] = True
            interpolated += 1

    best_start = 0
    best_stop = 0
    run_start: int | None = None
    for row, ok in enumerate(finite):
        if ok and run_start is None:
            run_start = row
        if (not ok or row == len(finite) - 1) and run_start is not None:
            run_stop = row if not ok else row + 1
            if run_stop - run_start > best_stop - best_start:
                best_start, best_stop = run_start, run_stop
            run_start = None

    if best_stop - best_start < min_samples:
        return selected_frames, selected_traces, False, "no_long_enough_contiguous_segment"
    reason = "ok" if interpolated == 0 else f"ok_interpolated_{interpolated}_rows"
    return selected_frames[best_start:best_stop], selected_traces[best_start:best_stop], True, reason


def _make_windows(
    *,
    trace_set: TraceSet,
    frame_index: np.ndarray,
    valid: np.ndarray,
    fps: float,
    frame_start: int,
    window_seconds: float,
    step_seconds: float,
    max_gap: int,
    min_samples: int,
) -> list[WindowData]:
    windows: list[WindowData] = []
    for index, start_s, stop_s, rows in _window_ranges(
        frame_indices=frame_index,
        fps=fps,
        frame_start=frame_start,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        min_samples=min_samples,
    ):
        frames, traces, ok, reason = _interpolate_window(
            frame_index=frame_index,
            traces=trace_set.traces,
            valid=valid,
            rows=rows,
            max_gap=max_gap,
            min_samples=min_samples,
        )
        windows.append(
            WindowData(
                index=index,
                start_s=float(start_s),
                stop_s=float(stop_s),
                rows=rows,
                frame_index=frames,
                traces=traces,
                ok=bool(ok),
                reason=reason,
            )
        )
    return windows


def _window_scores(
    *,
    windows: list[WindowData],
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    covariances: list[np.ndarray | None] = []
    correlations: list[np.ndarray | None] = []
    for window in windows:
        if not window.ok:
            covariances.append(None)
            correlations.append(None)
            continue
        bandpassed = _bandpass_matrix(
            window.traces,
            fps=fps,
            band_min_hz=band_min_hz,
            band_max_hz=band_max_hz,
        )
        roi_signal = np.mean(bandpassed, axis=1)
        covariance = np.mean(
            (bandpassed - np.mean(bandpassed, axis=0))
            * (roi_signal[:, None] - np.mean(roi_signal)),
            axis=0,
        )
        covariances.append(covariance.astype(np.float64))
        correlations.append(_safe_corr_with_reference(bandpassed, roi_signal).astype(np.float64))
    return covariances, correlations


def _selected_top_pixels(
    *,
    score_by_window: list[np.ndarray | None],
    eval_window_index: int,
    top_k: int,
    positive_only: bool,
) -> tuple[np.ndarray, int, bool]:
    training_scores = [
        score for index, score in enumerate(score_by_window) if index != eval_window_index and score is not None
    ]
    selection_overlap = False
    if not training_scores:
        current = score_by_window[eval_window_index]
        if current is None:
            return np.zeros(0, dtype=np.int64), 0, True
        training_scores = [current]
        selection_overlap = True
    score = np.nanmean(np.stack(training_scores, axis=0), axis=0)
    finite = np.isfinite(score)
    positive = finite & (score > 0)
    positive_count = int(np.count_nonzero(positive))
    eligible = positive if positive_only else finite
    if int(np.count_nonzero(eligible)) == 0:
        return np.zeros(0, dtype=np.int64), positive_count, selection_overlap
    order = np.argsort(score)[::-1]
    selected = [int(idx) for idx in order if bool(eligible[idx])][: int(top_k)]
    return np.asarray(selected, dtype=np.int64), positive_count, selection_overlap


def _nan_estimates() -> dict[str, float]:
    return {
        "welch_peak_frequency_hz": math.nan,
        "welch_peak_bpm": math.nan,
        "welch_peak_to_median_band_power": math.nan,
        "welch_peak_power": math.nan,
        "welch_median_band_power": math.nan,
        "periodogram_peak_frequency_hz": math.nan,
        "periodogram_peak_bpm": math.nan,
        "periodogram_peak_to_median_band_power": math.nan,
        "periodogram_peak_power": math.nan,
        "periodogram_median_band_power": math.nan,
        "autocorr_peak_frequency_hz": math.nan,
        "autocorr_peak_bpm": math.nan,
        "autocorr_peak_strength": math.nan,
        "autocorr_peak_lag_samples": math.nan,
        "autocorr_median_band_strength": math.nan,
    }


def _estimate_columns() -> list[str]:
    return list(_nan_estimates().keys())


def _parabolic_peak_position(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    y0 = float(values[index - 1])
    y1 = float(values[index])
    y2 = float(values[index + 1])
    denom = y0 - 2.0 * y1 + y2
    if not np.isfinite(denom) or abs(denom) < np.finfo(float).eps:
        return float(index)
    offset = 0.5 * (y0 - y2) / denom
    if not np.isfinite(offset) or abs(offset) > 1.0:
        return float(index)
    return float(index) + float(offset)


def _periodogram_estimate(
    values: np.ndarray,
    *,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
) -> dict[str, float]:
    from scipy import signal

    sample_rate = float(fps)
    nyquist = sample_rate / 2.0
    low = max(0.001, float(band_min_hz))
    high = min(float(band_max_hz), nyquist * 0.98)
    if len(values) < 16 or low >= high:
        return {}

    raw = values.astype(np.float64)
    centered = raw - np.nanmedian(raw)
    detrended = signal.detrend(centered, type="linear")
    window = np.hanning(len(detrended))
    window_power = float(np.sum(window * window))
    if window_power <= 0:
        return {}
    spectrum = np.fft.rfft(detrended * window)
    power = (np.abs(spectrum) ** 2) / window_power
    frequencies = np.fft.rfftfreq(len(detrended), d=1.0 / sample_rate)
    band = (frequencies >= low) & (frequencies <= high)
    if int(np.count_nonzero(band)) == 0:
        return {}
    band_indices = np.flatnonzero(band)
    band_power = power[band_indices]
    peak_global = int(band_indices[int(np.argmax(band_power))])
    refined = _parabolic_peak_position(power, peak_global)
    refined = max(float(band_indices[0]), min(float(band_indices[-1]), refined))
    peak_hz = float(refined * sample_rate / float(len(detrended)))
    peak_power = float(power[peak_global])
    median_power = float(np.median(band_power))
    return {
        "periodogram_peak_frequency_hz": peak_hz,
        "periodogram_peak_bpm": peak_hz * 60.0,
        "periodogram_peak_to_median_band_power": float(peak_power / median_power)
        if median_power > 0
        else math.inf,
        "periodogram_peak_power": peak_power,
        "periodogram_median_band_power": median_power,
    }


def _autocorr_estimate(
    values: np.ndarray,
    *,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
) -> dict[str, float]:
    from scipy import signal

    sample_rate = float(fps)
    nyquist = sample_rate / 2.0
    low = max(0.001, float(band_min_hz))
    high = min(float(band_max_hz), nyquist * 0.98)
    if len(values) < 16 or low >= high:
        return {}

    raw = values.astype(np.float64)
    centered = raw - np.nanmedian(raw)
    detrended = signal.detrend(centered, type="linear")
    filtered = detrended
    padlen = 3 * 2 * 3
    if len(detrended) > padlen + 4:
        sos = signal.butter(3, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        filtered = signal.sosfiltfilt(sos, detrended)
    filtered = filtered - float(np.mean(filtered))
    zero_lag = float(np.sum(filtered * filtered))
    if zero_lag <= np.finfo(float).eps:
        return {}

    autocorr = signal.correlate(filtered, filtered, mode="full", method="fft")[len(filtered) - 1 :]
    autocorr = autocorr.astype(np.float64) / zero_lag
    lag_min = max(1, int(math.floor(sample_rate / high)))
    lag_max = min(len(autocorr) - 2, int(math.ceil(sample_rate / low)))
    if lag_min > lag_max:
        return {}
    lag_segment = autocorr[lag_min : lag_max + 1]
    peaks, _ = signal.find_peaks(lag_segment)
    if peaks.size:
        local_peak = int(peaks[int(np.argmax(lag_segment[peaks]))])
    else:
        local_peak = int(np.argmax(lag_segment))
    peak_lag_index = lag_min + local_peak
    peak_lag = _parabolic_peak_position(autocorr, peak_lag_index)
    peak_lag = max(float(lag_min), min(float(lag_max), peak_lag))
    peak_hz = sample_rate / peak_lag
    return {
        "autocorr_peak_frequency_hz": float(peak_hz),
        "autocorr_peak_bpm": float(peak_hz * 60.0),
        "autocorr_peak_strength": float(autocorr[peak_lag_index]),
        "autocorr_peak_lag_samples": float(peak_lag),
        "autocorr_median_band_strength": float(np.median(lag_segment)),
    }


def _estimate_trace(
    *,
    frame_index: np.ndarray,
    values: np.ndarray,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
    primary_estimator: str,
) -> tuple[str, dict[str, float], int]:
    estimates = _nan_estimates()
    sample_count = int(len(values))
    status = "ok"
    try:
        result = _analyze_trace(
            frame_index=frame_index,
            values=values,
            fps=fps,
            band_min_hz=band_min_hz,
            band_max_hz=band_max_hz,
        )
        summary = result["summary"]
        sample_count = int(summary.get("sample_count", sample_count))
        estimates.update(
            {
                "welch_peak_frequency_hz": float(summary.get("peak_frequency_hz", math.nan)),
                "welch_peak_bpm": float(summary.get("peak_bpm", math.nan)),
                "welch_peak_to_median_band_power": float(
                    summary.get("peak_to_median_band_power", math.nan)
                ),
                "welch_peak_power": float(summary.get("peak_power", math.nan)),
                "welch_median_band_power": float(summary.get("median_band_power", math.nan)),
            }
        )
    except Exception as exc:
        status = f"welch_failed:{type(exc).__name__}:{exc}"

    try:
        estimates.update(
            _periodogram_estimate(
                values,
                fps=fps,
                band_min_hz=band_min_hz,
                band_max_hz=band_max_hz,
            )
        )
    except Exception as exc:
        if primary_estimator == "periodogram":
            status = f"periodogram_failed:{type(exc).__name__}:{exc}"

    try:
        estimates.update(
            _autocorr_estimate(
                values,
                fps=fps,
                band_min_hz=band_min_hz,
                band_max_hz=band_max_hz,
            )
        )
    except Exception as exc:
        if primary_estimator == "autocorr":
            status = f"autocorr_failed:{type(exc).__name__}:{exc}"

    primary_bpm = estimates.get(f"{primary_estimator}_peak_bpm", math.nan)
    if not np.isfinite(primary_bpm):
        status = f"{primary_estimator}_failed:no_finite_peak"
    return status, estimates, sample_count


def _primary_values(estimates: dict[str, float], *, primary_estimator: str) -> dict[str, float]:
    if primary_estimator == "autocorr":
        score = estimates["autocorr_peak_strength"]
    else:
        score = estimates[f"{primary_estimator}_peak_to_median_band_power"]
    return {
        "peak_frequency_hz": estimates[f"{primary_estimator}_peak_frequency_hz"],
        "peak_bpm": estimates[f"{primary_estimator}_peak_bpm"],
        "peak_score": score,
        "peak_to_median_band_power": score,
        "peak_power": estimates.get(f"{primary_estimator}_peak_power", math.nan),
        "median_band_power": estimates.get(f"{primary_estimator}_median_band_power", math.nan),
    }


def _summary_row(
    *,
    strategy: str,
    strategy_kind: str,
    source_trace_set: str,
    window: WindowData,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
    selected_pixels: np.ndarray,
    selection_positive_pixels: int | None = None,
    selection_overlap: bool = False,
    top_k: int | None = None,
    score_mode: str | None = None,
    primary_estimator: str = "welch",
) -> dict[str, Any]:
    empty_estimates = _nan_estimates()
    if not window.ok:
        return {
            "strategy": strategy,
            "strategy_kind": strategy_kind,
            "source_trace_set": source_trace_set,
            "primary_estimator": primary_estimator,
            "window_index": int(window.index),
            "window_start_s": float(window.start_s),
            "window_stop_s": float(window.stop_s),
            "window_frame_start": "",
            "window_frame_stop_inclusive": "",
            "status": window.reason,
            "analysis_samples": 0,
            "pixel_count": int(selected_pixels.size),
            "selection_positive_pixels": selection_positive_pixels,
            "selection_overlap": int(selection_overlap),
            "top_k": top_k,
            "score_mode": score_mode,
            "peak_frequency_hz": math.nan,
            "peak_bpm": math.nan,
            "peak_score": math.nan,
            "peak_to_median_band_power": math.nan,
            "peak_power": math.nan,
            "median_band_power": math.nan,
            **empty_estimates,
        }
    if selected_pixels.size == 0:
        status = "no_selected_pixels"
        return {
            "strategy": strategy,
            "strategy_kind": strategy_kind,
            "source_trace_set": source_trace_set,
            "primary_estimator": primary_estimator,
            "window_index": int(window.index),
            "window_start_s": float(window.start_s),
            "window_stop_s": float(window.stop_s),
            "window_frame_start": int(window.frame_index[0]),
            "window_frame_stop_inclusive": int(window.frame_index[-1]),
            "status": status,
            "analysis_samples": int(window.frame_index.size),
            "pixel_count": 0,
            "selection_positive_pixels": selection_positive_pixels,
            "selection_overlap": int(selection_overlap),
            "top_k": top_k,
            "score_mode": score_mode,
            "peak_frequency_hz": math.nan,
            "peak_bpm": math.nan,
            "peak_score": math.nan,
            "peak_to_median_band_power": math.nan,
            "peak_power": math.nan,
            "median_band_power": math.nan,
            **empty_estimates,
        }

    values = np.mean(window.traces[:, selected_pixels], axis=1)
    status, estimates, sample_count = _estimate_trace(
        frame_index=window.frame_index,
        values=values,
        fps=fps,
        band_min_hz=band_min_hz,
        band_max_hz=band_max_hz,
        primary_estimator=primary_estimator,
    )
    primary = _primary_values(estimates, primary_estimator=primary_estimator)

    return {
        "strategy": strategy,
        "strategy_kind": strategy_kind,
        "source_trace_set": source_trace_set,
        "primary_estimator": primary_estimator,
        "window_index": int(window.index),
        "window_start_s": float(window.start_s),
        "window_stop_s": float(window.stop_s),
        "window_frame_start": int(window.frame_index[0]) if window.frame_index.size else "",
        "window_frame_stop_inclusive": int(window.frame_index[-1]) if window.frame_index.size else "",
        "status": status,
        "analysis_samples": int(sample_count),
        "pixel_count": int(selected_pixels.size),
        "selection_positive_pixels": selection_positive_pixels,
        "selection_overlap": int(selection_overlap),
        "top_k": top_k,
        "score_mode": score_mode,
        **primary,
        **estimates,
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_output_dir(path.parent)
    fieldnames = [
        "strategy",
        "strategy_kind",
        "source_trace_set",
        "primary_estimator",
        "window_index",
        "window_start_s",
        "window_stop_s",
        "window_frame_start",
        "window_frame_stop_inclusive",
        "status",
        "analysis_samples",
        "pixel_count",
        "selection_positive_pixels",
        "selection_overlap",
        "top_k",
        "score_mode",
        "peak_frequency_hz",
        "peak_bpm",
        "peak_score",
        "peak_to_median_band_power",
        "peak_power",
        "median_band_power",
    ]
    fieldnames.extend(_estimate_columns())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    strategies = sorted({str(row["strategy"]) for row in rows})
    for strategy in strategies:
        selected = [
            row
            for row in rows
            if row["strategy"] == strategy and row["status"] == "ok" and int(row["window_index"]) >= 0
        ]
        bpm = np.asarray([float(row["peak_bpm"]) for row in selected], dtype=np.float64)
        score = np.asarray([float(row.get("peak_score", row["peak_to_median_band_power"])) for row in selected], dtype=np.float64)
        bpm = bpm[np.isfinite(bpm)]
        score = score[np.isfinite(score)]
        out[strategy] = {
            "ok_windows": int(len(selected)),
            "peak_bpm_median": float(np.median(bpm)) if bpm.size else math.nan,
            "peak_bpm_min": float(np.min(bpm)) if bpm.size else math.nan,
            "peak_bpm_max": float(np.max(bpm)) if bpm.size else math.nan,
            "peak_bpm_range": float(np.max(bpm) - np.min(bpm)) if bpm.size else math.nan,
            "peak_score_median": float(np.median(score)) if score.size else math.nan,
            "pixel_count_median": float(np.median([float(row["pixel_count"]) for row in selected]))
            if selected
            else math.nan,
        }
    return out


def _write_qc_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok_rows = [row for row in rows if row["status"] == "ok" and int(row["window_index"]) >= 0]
    strategies = sorted({str(row["strategy"]) for row in ok_rows})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    for strategy in strategies:
        selected = [row for row in ok_rows if row["strategy"] == strategy]
        selected.sort(key=lambda row: float(row["window_start_s"]))
        x = [0.5 * (float(row["window_start_s"]) + float(row["window_stop_s"])) for row in selected]
        bpm = [float(row["peak_bpm"]) for row in selected]
        score = [float(row.get("peak_score", row["peak_to_median_band_power"])) for row in selected]
        axes[0].plot(x, bpm, marker="o", lw=1.0, ms=3, label=strategy)
        axes[1].plot(x, score, marker="o", lw=1.0, ms=3, label=strategy)
    axes[0].set_title("Windowed peak heart rate by pixel strategy")
    axes[0].set_ylabel("peak bpm")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Windowed primary peak score")
    axes[1].set_xlabel("window midpoint (s)")
    axes[1].set_ylabel("score")
    axes[1].grid(True, alpha=0.25)
    if strategies:
        axes[0].legend(loc="best", fontsize=8, ncols=2)
    ensure_output_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_selection_map(
    path: Path,
    *,
    mean_frame: np.ndarray,
    roi_mask: np.ndarray,
    trace_sets: dict[str, TraceSet],
    selection_counts: dict[str, np.ndarray],
) -> None:
    if not selection_counts:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(selection_counts.keys())[:8]
    cols = min(4, len(names))
    rows = int(math.ceil(len(names) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    for axis in axes_arr.ravel():
        axis.axis("off")
    for axis, name in zip(axes_arr.ravel(), names):
        trace_set = trace_sets[name.split(":", 1)[0]]
        counts = selection_counts[name]
        image = np.full(roi_mask.shape, np.nan, dtype=np.float32)
        image[trace_set.pixel_y, trace_set.pixel_x] = counts.astype(np.float32)
        base = cv2.cvtColor(mean_frame, cv2.COLOR_BGR2RGB)
        axis.imshow(base)
        masked = np.ma.masked_invalid(image)
        vmax = float(np.nanmax(image)) if np.isfinite(image).any() else 1.0
        if vmax <= 0:
            vmax = 1.0
        axis.imshow(masked, cmap="magma", vmin=0, vmax=vmax, alpha=np.where(np.isfinite(image), 0.9, 0.0))
        axis.contour(roi_mask.astype(np.uint8), levels=[0.5], colors="white", linewidths=0.7)
        axis.set_title(name, fontsize=9)
        axis.axis("off")
    ensure_output_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare windowed heart-rate estimates across ROI pixel-selection strategies."
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--video", type=Path, required=True, help="Stabilized video to sample.")
    parser.add_argument("--roi-json", type=Path, default=None, help="ROI JSON written by draw_roi.py.")
    parser.add_argument("--roi", type=str, default=None, help="Stabilized ROI rectangle x,y,width,height.")
    parser.add_argument("--status-csv", type=Path, default=None, help="Optional stabilized-video status CSV.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=6000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--band-min-hz", type=float, default=1.0)
    parser.add_argument("--band-max-hz", type=float, default=4.0)
    parser.add_argument(
        "--min-roi-mean-intensity",
        type=float,
        default=None,
        help=(
            "Mark frames invalid when the sampled ROI mean intensity is at or below this threshold. "
            "Use 0 or 1 to reject all-black acquisition-dropout crop frames."
        ),
    )
    parser.add_argument(
        "--primary-estimator",
        choices=("welch", "periodogram", "autocorr"),
        default="welch",
        help="Estimator used for primary peak_bpm/peak_score columns. All estimators are still written.",
    )
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--window-step-seconds", type=float, default=None)
    parser.add_argument("--max-interpolated-gap-samples", type=int, default=5)
    parser.add_argument("--min-window-samples", type=int, default=64)
    parser.add_argument(
        "--mask-npz",
        type=Path,
        default=None,
        help="Optional pixel-band .npz whose roi_mask/roi_x/roi_y define an eye-excluded or otherwise filtered pixel set.",
    )
    parser.add_argument("--mask-label", type=str, default="mask_filtered")
    parser.add_argument("--top-k-values", type=str, default="25,50,100")
    parser.add_argument("--top-score-modes", type=str, default="covariance,correlation")
    parser.add_argument(
        "--top-source",
        choices=("auto", "all_roi", "mask"),
        default="auto",
        help="Pixel set used for top-K selection. auto uses --mask-npz when present, otherwise all_roi.",
    )
    parser.add_argument(
        "--allow-nonpositive-top-pixels",
        action="store_true",
        help="Allow top-K selection from nonpositive scores if there are fewer than K positive pixels.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/roi_pixel_strategy_compare"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    roi_rect = resolve_roi_rect(config, roi=args.roi, roi_json=args.roi_json)
    roi_polygon = roi_rect_corners(roi_rect)
    effective_fps = float(args.fps) / max(1, int(args.stride))
    window_step = float(args.window_step_seconds) if args.window_step_seconds is not None else float(args.window_seconds)

    loaded = _load_roi_pixel_traces(
        video_path=args.video,
        roi_polygon=roi_polygon,
        status_csv=args.status_csv,
        frame_start=max(0, int(args.frame_start)),
        frame_count=max(0, int(args.frame_count)),
        stride=max(1, int(args.stride)),
        min_roi_mean_intensity=args.min_roi_mean_intensity,
    )
    frame_index = loaded["frame_indices"]
    valid = loaded["valid"]
    base = TraceSet(
        name="all_roi",
        traces=loaded["traces"].astype(np.float64),
        pixel_x=loaded["roi_x"],
        pixel_y=loaded["roi_y"],
        source="roi_polygon",
    )
    trace_sets: dict[str, TraceSet] = {"all_roi": base}

    if args.mask_npz is not None:
        mask = _mask_from_npz(args.mask_npz, shape_hw=loaded["roi_mask"].shape)
        keep = mask[base.pixel_y, base.pixel_x]
        if int(np.count_nonzero(keep)) == 0:
            raise ValueError(f"{args.mask_npz} removed every loaded ROI pixel")
        trace_sets["mask"] = TraceSet(
            name=str(args.mask_label),
            traces=base.traces[:, keep],
            pixel_x=base.pixel_x[keep],
            pixel_y=base.pixel_y[keep],
            source=str(args.mask_npz),
        )

    if args.top_source == "mask" and "mask" not in trace_sets:
        raise ValueError("--top-source mask requires --mask-npz")
    top_source_key = "mask" if args.top_source == "auto" and "mask" in trace_sets else args.top_source
    if top_source_key == "auto":
        top_source_key = "all_roi"
    top_source = trace_sets[top_source_key]

    top_k_values = _parse_int_csv(args.top_k_values)
    score_modes = _parse_score_modes(args.top_score_modes)
    positive_only = not bool(args.allow_nonpositive_top_pixels)

    rows: list[dict[str, Any]] = []
    selection_counts: dict[str, np.ndarray] = {}
    windows_by_trace_set: dict[str, list[WindowData]] = {}
    for key, trace_set in trace_sets.items():
        windows_by_trace_set[key] = _make_windows(
            trace_set=trace_set,
            frame_index=frame_index,
            valid=valid,
            fps=effective_fps,
            frame_start=max(0, int(args.frame_start)),
            window_seconds=float(args.window_seconds),
            step_seconds=window_step,
            max_gap=max(0, int(args.max_interpolated_gap_samples)),
            min_samples=max(16, int(args.min_window_samples)),
        )
        all_pixels = np.arange(trace_set.traces.shape[1], dtype=np.int64)
        for window in windows_by_trace_set[key]:
            rows.append(
                _summary_row(
                    strategy=trace_set.name,
                    strategy_kind="all_pixels_in_trace_set",
                    source_trace_set=trace_set.name,
                    window=window,
                    fps=effective_fps,
                    band_min_hz=float(args.band_min_hz),
                    band_max_hz=float(args.band_max_hz),
                    selected_pixels=all_pixels,
                    primary_estimator=str(args.primary_estimator),
                )
            )

    top_windows = windows_by_trace_set[top_source_key]
    covariances, correlations = _window_scores(
        windows=top_windows,
        fps=effective_fps,
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
    )
    score_by_mode = {"covariance": covariances, "correlation": correlations}
    for mode in score_modes:
        for top_k in top_k_values:
            strategy = f"{top_source.name}_top_{mode}_k{top_k}"
            count_key = f"{top_source_key}:{strategy}"
            selection_counts[count_key] = np.zeros(top_source.traces.shape[1], dtype=np.uint16)
            for window in top_windows:
                selected, positive_count, selection_overlap = _selected_top_pixels(
                    score_by_window=score_by_mode[mode],
                    eval_window_index=window.index,
                    top_k=int(top_k),
                    positive_only=positive_only,
                )
                if selected.size:
                    selection_counts[count_key][selected] += 1
                rows.append(
                    _summary_row(
                        strategy=strategy,
                        strategy_kind="leave_one_window_out_top_pixels",
                        source_trace_set=top_source.name,
                        window=window,
                        fps=effective_fps,
                        band_min_hz=float(args.band_min_hz),
                        band_max_hz=float(args.band_max_hz),
                        selected_pixels=selected,
                        selection_positive_pixels=positive_count,
                        selection_overlap=selection_overlap,
                        top_k=int(top_k),
                        score_mode=mode,
                        primary_estimator=str(args.primary_estimator),
                    )
                )

    output_prefix = args.output_prefix
    ensure_output_dir(output_prefix.parent)
    csv_path = output_prefix.with_suffix(".summary.csv")
    json_path = output_prefix.with_suffix(".summary.json")
    plot_path = output_prefix.with_suffix(".windowed_qc.png")
    selection_map_path = output_prefix.with_suffix(".selected_pixels.png")

    _write_rows_csv(csv_path, rows)
    _write_qc_plot(plot_path, rows)
    selection_trace_sets = {key: value for key, value in trace_sets.items()}
    _write_selection_map(
        selection_map_path,
        mean_frame=loaded["mean_frame"],
        roi_mask=polygon_mask(loaded["roi_mask"].shape, roi_polygon),
        trace_sets=selection_trace_sets,
        selection_counts=selection_counts,
    )

    summary = {
        "source_video": str(args.video),
        "status_csv": str(args.status_csv) if args.status_csv is not None else None,
        "roi_json": str(args.roi_json) if args.roi_json is not None else None,
        "roi_rect_stable_xywh": [float(value) for value in roi_rect],
        "frame_start": int(args.frame_start),
        "frame_count": int(args.frame_count),
        "stride": int(args.stride),
        "fps": effective_fps,
        "min_roi_mean_intensity": loaded["min_roi_mean_intensity"],
        "low_intensity_frame_count": int(loaded["low_intensity_frame_count"]),
        "valid_loaded_frames": int(np.count_nonzero(loaded["valid"])),
        "band_hz": [float(args.band_min_hz), float(args.band_max_hz)],
        "primary_estimator": str(args.primary_estimator),
        "window_seconds": float(args.window_seconds),
        "window_step_seconds": window_step,
        "trace_sets": {
            key: {
                "name": trace_set.name,
                "pixel_count": int(trace_set.traces.shape[1]),
                "source": trace_set.source,
            }
            for key, trace_set in trace_sets.items()
        },
        "top_source": top_source.name,
        "top_k_values": list(top_k_values),
        "top_score_modes": list(score_modes),
        "top_selection": "positive_scores_only" if positive_only else "finite_scores",
        "aggregate": _aggregate(rows),
        "outputs": {
            "summary_csv": str(csv_path),
            "windowed_qc_png": str(plot_path),
            "selected_pixels_png": str(selection_map_path),
        },
    }
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"summary_json: {json_path}")
    print(f"summary_csv: {csv_path}")
    print(f"windowed_qc_png: {plot_path}")
    if selection_counts:
        print(f"selected_pixels_png: {selection_map_path}")
    for strategy, stats in summary["aggregate"].items():
        print(
            f"{strategy}: ok_windows={stats['ok_windows']} "
            f"median_bpm={stats['peak_bpm_median']:.3f} "
            f"range_bpm={stats['peak_bpm_range']:.3f} "
            f"median_peak_score={stats['peak_score_median']:.3f}"
        )


if __name__ == "__main__":
    main()
