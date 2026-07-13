from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.local_rostral_heartrate import (
    HeartrateConfig,
    bridge_short_gaps,
    build_risk_surfaces,
    contiguous_segments,
)


def _sample_rate(timestamps_s: np.ndarray) -> float:
    differences = np.diff(np.asarray(timestamps_s, dtype=np.float64))
    positive = differences[differences > 0.0]
    if positive.size == 0:
        raise ValueError("timestamps do not contain a positive sampling interval")
    return 1.0 / float(np.median(positive))


def _common_valid_segments(
    timestamps_s: np.ndarray,
    frame_valid: np.ndarray,
    pixel_valid: np.ndarray,
    eligible: np.ndarray,
    *,
    min_seconds: float,
    max_interpolated_gap_seconds: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    selected = np.asarray(eligible, dtype=bool)
    if not np.any(selected):
        raise ValueError("no eligible pixels")
    common = (
        np.asarray(frame_valid, dtype=bool)
        & np.all(np.asarray(pixel_valid, dtype=bool)[:, selected], axis=1)
    )
    bridged, interpolated = bridge_short_gaps(
        timestamps_s,
        common,
        max_gap_seconds=float(max_interpolated_gap_seconds),
    )
    return (
        contiguous_segments(
            timestamps_s,
            bridged,
            max_gap_factor=1.75,
            min_seconds=float(min_seconds),
        ),
        interpolated,
    )


def _interpolate_segment(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    output = source.copy()
    sample_index = np.arange(source.shape[0], dtype=np.float64)
    for column in range(source.shape[1]):
        finite = np.isfinite(source[:, column])
        if np.count_nonzero(finite) < 2:
            raise ValueError("segment pixel has fewer than two finite samples")
        if not np.all(finite):
            output[:, column] = np.interp(
                sample_index,
                sample_index[finite],
                source[finite, column],
            )
    return output


def _segmented_pca(
    values: np.ndarray,
    segments: list[np.ndarray],
    *,
    fps: float,
    band_hz: tuple[float, float],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float]:
    source = np.asarray(values, dtype=np.float64)
    residuals = [
        detrend(_interpolate_segment(source[rows]), axis=0, type="linear")
        for rows in segments
    ]
    joined = np.concatenate(residuals, axis=0)
    center = np.median(joined, axis=0)
    scale = 1.4826 * np.median(np.abs(joined - center), axis=0)
    usable = np.isfinite(scale) & (scale > 1e-6)
    if np.count_nonzero(usable) < 3:
        raise ValueError("fewer than three varying pixels support segmented PCA")
    sos = butter(3, band_hz, btype="bandpass", fs=fps, output="sos")
    filtered = [
        sosfiltfilt(sos, (residual[:, usable] - center[usable]) / scale[usable], axis=0)
        for residual in residuals
    ]
    pca = PCA(n_components=1, svd_solver="randomized", random_state=0)
    with threadpool_limits(limits=1):
        pca.fit(np.concatenate(filtered, axis=0))
    scores = [pca.transform(item).reshape(-1) for item in filtered]
    loading = np.zeros(source.shape[1], dtype=np.float64)
    loading[usable] = pca.components_[0]
    return scores, loading, usable, float(pca.explained_variance_ratio_[0])


def _frequency_power(
    scores: list[np.ndarray],
    segments: list[np.ndarray],
    timestamps_s: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    edge_seconds: float,
) -> tuple[np.ndarray, int]:
    power = np.zeros(frequencies_hz.size, dtype=np.float64)
    sample_count = 0
    for score, rows in zip(scores, segments):
        local_time = np.asarray(timestamps_s, dtype=np.float64)[rows]
        keep = (
            (local_time >= local_time[0] + float(edge_seconds))
            & (local_time <= local_time[-1] - float(edge_seconds))
        )
        if np.count_nonzero(keep) < 8:
            continue
        values = score[keep] - float(np.mean(score[keep]))
        time = local_time[keep]
        coefficients = np.exp(-2j * np.pi * frequencies_hz[:, None] * time[None, :]) @ values
        power += np.abs(coefficients) ** 2
        sample_count += int(values.size)
    if sample_count == 0:
        return np.full(frequencies_hz.size, np.nan), 0
    return power / float(sample_count), sample_count


def _window_rows(
    scores: list[np.ndarray],
    segments: list[np.ndarray],
    timestamps_s: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    window_seconds: float,
    edge_seconds: float,
    minimum_samples: int,
) -> list[dict[str, Any]]:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    recording_start = float(timestamps[0])
    recording_stop = float(timestamps[-1])
    window_start = recording_start
    while window_start <= recording_stop:
        window_stop = min(recording_stop + 1e-9, window_start + float(window_seconds))
        pieces: list[np.ndarray] = []
        piece_rows: list[np.ndarray] = []
        for score, segment in zip(scores, segments):
            local_time = timestamps[segment]
            keep = (
                (local_time >= local_time[0] + float(edge_seconds))
                & (local_time <= local_time[-1] - float(edge_seconds))
                & (local_time >= window_start)
                & (local_time < window_stop)
            )
            if np.count_nonzero(keep) >= 8:
                pieces.append(score[keep])
                piece_rows.append(segment[keep])
        power, count = _frequency_power(
            pieces,
            piece_rows,
            timestamps,
            frequencies_hz,
            edge_seconds=0.0,
        )
        if count >= int(minimum_samples):
            peak_index = int(np.nanargmax(power))
            peak = float(frequencies_hz[peak_index])
            ratio = float(power[peak_index] / max(np.nanmedian(power), np.finfo(float).tiny))
            status = "ok"
        else:
            peak = float("nan")
            ratio = float("nan")
            status = "insufficient_common_valid_samples"
        rows.append(
            {
                "window_start_s": window_start - recording_start,
                "window_stop_s": window_stop - recording_start,
                "window_mid_s": 0.5 * (window_start + window_stop) - recording_start,
                "status": status,
                "common_valid_samples": count,
                "candidate_frequency_hz": peak,
                "candidate_cycles_per_min": 60.0 * peak,
                "peak_to_band_median": ratio,
            }
        )
        window_start += float(window_seconds)
    return rows


def _segmented_events(
    scores: list[np.ndarray],
    segments: list[np.ndarray],
    timestamps_s: np.ndarray,
    *,
    fps: float,
    band_hz: tuple[float, float],
    edge_seconds: float,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    alternatives: list[tuple[float, int, list[float], list[float]]] = []
    distance = max(1, int(np.floor(fps / band_hz[1])))
    all_values = np.concatenate(scores)
    scale = max(1.4826 * np.median(np.abs(all_values - np.median(all_values))), 1e-9)
    for polarity in (1, -1):
        event_times: list[float] = []
        intervals: list[float] = []
        prominences: list[float] = []
        for score, rows in zip(scores, segments):
            local_time = timestamps[rows]
            keep = (
                (local_time >= local_time[0] + float(edge_seconds))
                & (local_time <= local_time[-1] - float(edge_seconds))
            )
            local_indices = np.flatnonzero(keep)
            if local_indices.size < 8:
                continue
            peaks, properties = find_peaks(
                polarity * score[keep] / scale,
                distance=distance,
                prominence=0.5,
            )
            selected_times = local_time[local_indices[peaks]]
            event_times.extend(selected_times.tolist())
            local_intervals = np.diff(selected_times)
            plausible = local_intervals[
                (local_intervals >= 1.0 / band_hz[1])
                & (local_intervals <= 1.0 / band_hz[0])
            ]
            intervals.extend(plausible.tolist())
            prominences.extend(properties["prominences"].tolist())
        interval_array = np.asarray(intervals, dtype=np.float64)
        cv = (
            float(np.std(interval_array) / np.mean(interval_array))
            if interval_array.size >= 2
            else float("inf")
        )
        prominence = float(np.median(prominences)) if prominences else 0.0
        alternatives.append((cv - 0.01 * prominence, polarity, event_times, intervals))
    _, polarity, event_times, intervals = min(alternatives, key=lambda item: item[0])
    interval_array = np.asarray(intervals, dtype=np.float64)
    cv = (
        float(np.std(interval_array) / np.mean(interval_array))
        if interval_array.size >= 2
        else float("nan")
    )
    return polarity, np.asarray(event_times), interval_array, cv


def _scalar_control(
    values: np.ndarray,
    segments: list[np.ndarray],
    timestamps_s: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    fps: float,
    band_hz: tuple[float, float],
    edge_seconds: float,
) -> tuple[np.ndarray, float, float]:
    sos = butter(3, band_hz, btype="bandpass", fs=fps, output="sos")
    scores: list[np.ndarray] = []
    kept_segments: list[np.ndarray] = []
    for rows in segments:
        source = np.asarray(values, dtype=np.float64)[rows]
        if source.size < 8 or np.count_nonzero(np.isfinite(source)) < 2:
            continue
        filled = _interpolate_segment(source[:, None]).reshape(-1)
        scores.append(sosfiltfilt(sos, detrend(filled, type="linear")))
        kept_segments.append(rows)
    power, _ = _frequency_power(
        scores,
        kept_segments,
        timestamps_s,
        frequencies_hz,
        edge_seconds=edge_seconds,
    )
    if not np.isfinite(power).any():
        return power, float("nan"), float("nan")
    peak_index = int(np.nanargmax(power))
    ratio = float(power[peak_index] / max(np.nanmedian(power), np.finfo(float).tiny))
    return power, float(frequencies_hz[peak_index]), ratio


def _read_mask_membership(path: Path, key: str, pixel_xy: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        mask = np.asarray(data[key], dtype=bool)
    xy = np.asarray(pixel_xy, dtype=np.float64).round().astype(np.int64)
    return mask[xy[:, 1], xy[:, 0]]


def _write_plot(
    path: Path,
    *,
    frequencies_hz: np.ndarray,
    signal_power: np.ndarray,
    motion_power: np.ndarray,
    windows: list[dict[str, Any]],
    pixel_xy: np.ndarray,
    eligible: np.ndarray,
    loading: np.ndarray,
    segment_rows: list[np.ndarray],
    scores: list[np.ndarray],
    timestamps_s: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(frequencies_hz * 60.0, signal_power / np.nanmax(signal_power), label="observed PCA")
    if np.isfinite(motion_power).any():
        axes[0, 0].plot(
            frequencies_hz * 60.0,
            motion_power / np.nanmax(motion_power),
            label="motion-prediction PCA",
        )
    axes[0, 0].set(title="Segment-preserving spectrum", xlabel="Cycles/min", ylabel="Relative power")
    axes[0, 0].legend()

    ok = [row for row in windows if row["status"] == "ok"]
    axes[0, 1].plot(
        [row["window_mid_s"] / 60.0 for row in ok],
        [row["candidate_cycles_per_min"] for row in ok],
        marker="o",
        label="observed PCA",
    )
    motion_ok = [
        row
        for row in ok
        if np.isfinite(float(row.get("motion_candidate_cycles_per_min", np.nan)))
    ]
    if motion_ok:
        axes[0, 1].plot(
            [row["window_mid_s"] / 60.0 for row in motion_ok],
            [row["motion_candidate_cycles_per_min"] for row in motion_ok],
            marker="o",
            label="motion-prediction PCA",
        )
    window_duration = (
        float(windows[0]["window_stop_s"]) - float(windows[0]["window_start_s"])
        if windows
        else float("nan")
    )
    axes[0, 1].set(
        title=f"{window_duration:g} s exploratory trajectory",
        xlabel="Recording time (min)",
        ylabel="Cycles/min",
    )
    axes[0, 1].set_ylim(90, 245)
    axes[0, 1].legend()

    xy = np.asarray(pixel_xy)[eligible]
    load = np.asarray(loading)[eligible]
    limit = max(float(np.max(np.abs(load))), 1e-9)
    scatter = axes[1, 0].scatter(xy[:, 0], xy[:, 1], c=load, cmap="coolwarm", vmin=-limit, vmax=limit, s=45)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set(title="Whole-ROI PCA loading", xlabel="Canonical x", ylabel="Canonical y")
    figure.colorbar(scatter, ax=axes[1, 0], label="Signed loading")

    longest = int(np.argmax([rows.size for rows in segment_rows]))
    rows = segment_rows[longest]
    time = np.asarray(timestamps_s)[rows] - float(np.asarray(timestamps_s)[rows[0]])
    axes[1, 1].plot(time, scores[longest], linewidth=1.2)
    axes[1, 1].set(title="Longest valid segment PCA trace", xlabel="Segment time (s)", ylabel="PCA score")
    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.2)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the simple embedded-fish band-PCA idea to gap-segmented moving-fish photometry.")
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--band-hz", type=float, nargs=2, default=(1.5, 4.0))
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--minimum-segment-seconds", type=float, default=2.0)
    parser.add_argument("--max-interpolated-gap-seconds", type=float, default=0.02)
    parser.add_argument("--filter-edge-seconds", type=float, default=0.75)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--analysis-mask-npz", type=Path)
    parser.add_argument("--analysis-mask-key", default="heart_support_mask")
    parser.add_argument("--frozen-mask-npz", type=Path)
    parser.add_argument("--frozen-mask-key", default="heart_support_mask")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_npz)
    band_hz = (float(args.band_hz[0]), float(args.band_hz[1]))
    config = HeartrateConfig(band_min_hz=band_hz[0], band_max_hz=band_hz[1]).validated()
    risks = build_risk_surfaces(dataset, config)
    eligible = np.asarray(risks.eligible, dtype=bool)
    base_eligible_count = int(np.count_nonzero(eligible))
    analysis_mask_source: dict[str, str] | None = None
    if args.analysis_mask_npz is not None:
        analysis_membership = _read_mask_membership(
            args.analysis_mask_npz,
            args.analysis_mask_key,
            dataset.pixel_xy,
        )
        eligible &= analysis_membership
        analysis_mask_source = {
            "path": str(args.analysis_mask_npz.resolve()),
            "key": str(args.analysis_mask_key),
        }
    fps = _sample_rate(dataset.timestamps_s)
    segments, interpolated_rows = _common_valid_segments(
        dataset.timestamps_s,
        dataset.frame_valid,
        dataset.pixel_valid,
        eligible,
        min_seconds=float(args.minimum_segment_seconds),
        max_interpolated_gap_seconds=float(args.max_interpolated_gap_seconds),
    )
    if not segments:
        raise ValueError("no common-valid segments meet the duration requirement")
    observed_scores, observed_loading, _, explained = _segmented_pca(
        np.asarray(dataset.traces)[:, eligible],
        segments,
        fps=fps,
        band_hz=band_hz,
    )
    motion_values = np.asarray(dataset.motion_prediction, dtype=np.float64)[:, eligible].copy()
    motion_values[interpolated_rows] = np.nan
    motion_scores, _, _, motion_explained = _segmented_pca(
        motion_values,
        segments,
        fps=fps,
        band_hz=band_hz,
    )
    frequencies = np.arange(
        band_hz[0], band_hz[1] + float(args.frequency_step_hz) * 0.5, float(args.frequency_step_hz)
    )
    observed_power, analyzed_samples = _frequency_power(
        observed_scores,
        segments,
        dataset.timestamps_s,
        frequencies,
        edge_seconds=float(args.filter_edge_seconds),
    )
    motion_power, _ = _frequency_power(
        motion_scores,
        segments,
        dataset.timestamps_s,
        frequencies,
        edge_seconds=float(args.filter_edge_seconds),
    )
    peak_index = int(np.nanargmax(observed_power))
    peak_hz = float(frequencies[peak_index])
    peak_ratio = float(observed_power[peak_index] / max(np.nanmedian(observed_power), np.finfo(float).tiny))
    motion_peak_index = int(np.nanargmax(motion_power))
    motion_peak_hz = float(frequencies[motion_peak_index])
    motion_ratio_at_observed = float(
        motion_power[peak_index] / max(np.nanmedian(motion_power), np.finfo(float).tiny)
    )
    score_joined = np.concatenate(observed_scores)
    motion_joined = np.concatenate(motion_scores)
    motion_correlation = float(abs(np.corrcoef(score_joined, motion_joined)[0, 1]))
    windows = _window_rows(
        observed_scores,
        segments,
        dataset.timestamps_s,
        frequencies,
        window_seconds=float(args.window_seconds),
        edge_seconds=float(args.filter_edge_seconds),
        minimum_samples=max(8, int(round(2.0 * fps))),
    )
    motion_windows = _window_rows(
        motion_scores,
        segments,
        dataset.timestamps_s,
        frequencies,
        window_seconds=float(args.window_seconds),
        edge_seconds=float(args.filter_edge_seconds),
        minimum_samples=max(8, int(round(2.0 * fps))),
    )
    if len(motion_windows) != len(windows):
        raise RuntimeError("observed and motion window counts differ")
    for observed_window, motion_window in zip(windows, motion_windows):
        if (
            float(observed_window["window_start_s"])
            != float(motion_window["window_start_s"])
        ):
            raise RuntimeError("observed and motion window bounds differ")
        motion_frequency = float(motion_window["candidate_frequency_hz"])
        observed_frequency = float(observed_window["candidate_frequency_hz"])
        observed_window["motion_status"] = motion_window["status"]
        observed_window["motion_candidate_frequency_hz"] = motion_frequency
        observed_window["motion_candidate_cycles_per_min"] = 60.0 * motion_frequency
        observed_window["motion_peak_to_band_median"] = float(
            motion_window["peak_to_band_median"]
        )
        observed_window["observed_motion_frequency_difference_hz"] = (
            abs(observed_frequency - motion_frequency)
            if np.isfinite(observed_frequency) and np.isfinite(motion_frequency)
            else float("nan")
        )
    polarity, event_times, intervals, interval_cv = _segmented_events(
        observed_scores,
        segments,
        dataset.timestamps_s,
        fps=fps,
        band_hz=band_hz,
        edge_seconds=float(args.filter_edge_seconds),
    )

    nuisance_controls: dict[str, dict[str, float]] = {}
    for column, name in enumerate(dataset.nuisance_names):
        control_values = np.asarray(dataset.nuisance_values, dtype=np.float64)[:, column].copy()
        control_values[interpolated_rows] = np.nan
        control_power, control_peak, control_ratio = _scalar_control(
            control_values,
            segments,
            dataset.timestamps_s,
            frequencies,
            fps=fps,
            band_hz=band_hz,
            edge_seconds=float(args.filter_edge_seconds),
        )
        nuisance_controls[str(name)] = {
            "peak_hz": control_peak,
            "peak_to_band_median": control_ratio,
            "ratio_at_candidate_frequency": (
                float(
                    control_power[peak_index]
                    / max(np.nanmedian(control_power), np.finfo(float).tiny)
                )
                if np.isfinite(control_power).any()
                else float("nan")
            ),
        }

    loading_full = np.zeros(dataset.pixel_count, dtype=np.float64)
    loading_full[eligible] = observed_loading
    frozen_overlap: dict[str, float | int] | None = None
    if args.frozen_mask_npz is not None:
        membership = _read_mask_membership(
            args.frozen_mask_npz,
            args.frozen_mask_key,
            dataset.pixel_xy,
        )
        energy = loading_full**2
        frozen_overlap = {
            "mask_pixel_count": int(np.count_nonzero(membership & eligible)),
            "loading_energy_fraction": float(
                np.sum(energy[membership & eligible]) / max(np.sum(energy[eligible]), np.finfo(float).tiny)
            ),
        }

    durations = [
        float(dataset.timestamps_s[rows[-1]] - dataset.timestamps_s[rows[0]] + 1.0 / fps)
        for rows in segments
    ]
    valid_windows = [row for row in windows if row["status"] == "ok"]
    paired_frequency_differences = np.asarray(
        [
            row["observed_motion_frequency_difference_hz"]
            for row in valid_windows
            if np.isfinite(row["observed_motion_frequency_difference_hz"])
        ],
        dtype=np.float64,
    )
    summary = {
        "analysis_status": "exploratory_same_cache_fit_and_description",
        "method": "robust bandpass PCA fitted across every qualifying common-valid segment",
        "folds_used": False,
        "discovery_confirmation_partition_used": False,
        "dataset_npz": str(args.dataset_npz.resolve()),
        "band_hz": list(band_hz),
        "frequency_step_hz": float(args.frequency_step_hz),
        "effective_fps": fps,
        "frame_count": dataset.frame_count,
        "base_anatomically_eligible_pixels": base_eligible_count,
        "eligible_pixels": int(np.count_nonzero(eligible)),
        "analysis_mask": analysis_mask_source,
        "common_valid_segment_count": len(segments),
        "common_valid_duration_s": float(np.sum(durations)),
        "longest_segment_s": float(np.max(durations)),
        "bounded_interpolated_gap_frames": int(np.count_nonzero(interpolated_rows)),
        "max_interpolated_gap_seconds": float(args.max_interpolated_gap_seconds),
        "filter_edge_seconds_per_segment": float(args.filter_edge_seconds),
        "analyzed_post_edge_samples": analyzed_samples,
        "pca_explained_variance_fraction": explained,
        "candidate_peak_hz": peak_hz,
        "candidate_cycles_per_min": 60.0 * peak_hz,
        "candidate_peak_to_band_median": peak_ratio,
        "motion_pca_explained_variance_fraction": motion_explained,
        "motion_pca_peak_hz": motion_peak_hz,
        "motion_pca_ratio_at_candidate_frequency": motion_ratio_at_observed,
        "absolute_observed_motion_pca_correlation": motion_correlation,
        "event_polarity": polarity,
        "event_count": int(event_times.size),
        "within_segment_interval_count": int(intervals.size),
        "median_interval_cycles_per_min": (
            float(60.0 / np.median(intervals)) if intervals.size else None
        ),
        "interval_cv": interval_cv,
        "window_seconds": float(args.window_seconds),
        "scorable_windows": len(valid_windows),
        "window_candidate_cycles_per_min": [
            float(row["candidate_cycles_per_min"]) for row in valid_windows
        ],
        "motion_window_candidate_cycles_per_min": [
            float(row["motion_candidate_cycles_per_min"])
            for row in valid_windows
            if np.isfinite(row["motion_candidate_cycles_per_min"])
        ],
        "observed_motion_window_frequency_difference_hz": {
            "paired_windows": int(paired_frequency_differences.size),
            "median": (
                float(np.median(paired_frequency_differences))
                if paired_frequency_differences.size
                else None
            ),
            "within_0p10_hz": int(
                np.count_nonzero(paired_frequency_differences <= 0.10 + 1e-12)
            ),
            "within_0p20_hz": int(
                np.count_nonzero(paired_frequency_differences <= 0.20 + 1e-12)
            ),
        },
        "nuisance_controls": nuisance_controls,
        "frozen_mask_loading_overlap": frozen_overlap,
        "limitations": [
            "PCA loadings and event polarity are fit on the same cache being described.",
            "There is no cardiac-rate reference for this freely moving recording.",
            "The moving-cache sampling geometry differs from the embedded 22x28 source-pixel box.",
            "This is not a surrogate-calibrated detection and emits no validated heart rate.",
        ],
    }

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_suffix(".summary.json")
    window_path = prefix.with_suffix(".windows.csv")
    arrays_path = prefix.with_suffix(".arrays.npz")
    plot_path = prefix.with_suffix(".diagnostic.png")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with window_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0].keys()))
        writer.writeheader()
        writer.writerows(windows)
    np.savez_compressed(
        arrays_path,
        frequencies_hz=frequencies,
        observed_power=observed_power,
        motion_power=motion_power,
        eligible=eligible,
        pca_loading=loading_full,
        event_times_s=event_times,
        intervals_s=intervals,
    )
    _write_plot(
        plot_path,
        frequencies_hz=frequencies,
        signal_power=observed_power,
        motion_power=motion_power,
        windows=windows,
        pixel_xy=dataset.pixel_xy,
        eligible=eligible,
        loading=loading_full,
        segment_rows=segments,
        scores=observed_scores,
        timestamps_s=dataset.timestamps_s,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")
    print(f"wrote {window_path}")
    print(f"wrote {arrays_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
