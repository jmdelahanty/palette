from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, hilbert, sosfiltfilt
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from analyze_segmented_cache_pca import (
    _common_valid_segments,
    _frequency_power,
    _interpolate_segment,
    _read_mask_membership,
    _sample_rate,
    _segmented_events,
    _window_rows,
)
from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.local_rostral_heartrate import HeartrateConfig, build_risk_surfaces


_METHODS = (
    "masked_pca",
    "full_equal_mean",
    "upper_equal_mean",
    "lower_equal_mean",
    "lower_raw_mean",
)


def segmented_filtered_pixels(
    values: np.ndarray,
    segments: list[np.ndarray],
    *,
    fps: float,
    band_hz: tuple[float, float],
) -> tuple[list[np.ndarray], np.ndarray]:
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
        raise ValueError("fewer than three varying pixels support the comparison")
    sos = butter(3, band_hz, btype="bandpass", fs=fps, output="sos")
    filtered = [
        sosfiltfilt(
            sos,
            (residual[:, usable] - center[usable]) / scale[usable],
            axis=0,
        )
        for residual in residuals
    ]
    return filtered, usable


def segmented_raw_region_mean(
    values: np.ndarray,
    segments: list[np.ndarray],
    *,
    fps: float,
    band_hz: tuple[float, float],
) -> list[np.ndarray]:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] < 1:
        raise ValueError("raw region mean requires at least one pixel trace")
    sos = butter(3, band_hz, btype="bandpass", fs=fps, output="sos")
    output: list[np.ndarray] = []
    for rows in segments:
        filled = _interpolate_segment(source[rows])
        raw_mean = np.mean(filled, axis=1)
        output.append(sosfiltfilt(sos, detrend(raw_mean, type="linear")))
    return output


def projection_scores(
    filtered: list[np.ndarray],
    *,
    upper_usable: np.ndarray,
    lower_usable: np.ndarray,
) -> tuple[dict[str, list[np.ndarray]], np.ndarray, float]:
    joined = np.concatenate(filtered, axis=0)
    pca = PCA(n_components=1, svd_solver="randomized", random_state=0)
    with threadpool_limits(limits=1):
        pca.fit(joined)
    scores = {
        "masked_pca": [pca.transform(values).reshape(-1) for values in filtered],
        "full_equal_mean": [np.mean(values, axis=1) for values in filtered],
        "upper_equal_mean": [np.mean(values[:, upper_usable], axis=1) for values in filtered],
        "lower_equal_mean": [np.mean(values[:, lower_usable], axis=1) for values in filtered],
    }
    return scores, pca.components_[0], float(pca.explained_variance_ratio_[0])


def _post_edge_joined(
    scores: list[np.ndarray],
    segments: list[np.ndarray],
    timestamps_s: np.ndarray,
    *,
    edge_seconds: float,
) -> np.ndarray:
    kept: list[np.ndarray] = []
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    for score, rows in zip(scores, segments):
        local_time = timestamps[rows]
        keep = (
            (local_time >= local_time[0] + float(edge_seconds))
            & (local_time <= local_time[-1] - float(edge_seconds))
        )
        if np.any(keep):
            kept.append(np.asarray(score)[keep])
    return np.concatenate(kept) if kept else np.empty(0, dtype=np.float64)


def regional_phase_summary(
    upper_scores: list[np.ndarray],
    lower_scores: list[np.ndarray],
    segments: list[np.ndarray],
    timestamps_s: np.ndarray,
    *,
    edge_seconds: float,
) -> dict[str, float | int]:
    vectors: list[np.ndarray] = []
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    for upper, lower, rows in zip(upper_scores, lower_scores, segments):
        local_time = timestamps[rows]
        keep = (
            (local_time >= local_time[0] + float(edge_seconds))
            & (local_time <= local_time[-1] - float(edge_seconds))
        )
        if np.count_nonzero(keep) < 8:
            continue
        upper_phase = np.angle(hilbert(np.asarray(upper)[keep]))
        lower_phase = np.angle(hilbert(np.asarray(lower)[keep]))
        vectors.append(np.exp(1j * (lower_phase - upper_phase)))
    if not vectors:
        return {"sample_count": 0, "mean_phase_deg_lower_minus_upper": float("nan"), "phase_locking_value": float("nan")}
    joined = np.concatenate(vectors)
    vector = np.mean(joined)
    return {
        "sample_count": int(joined.size),
        "mean_phase_deg_lower_minus_upper": float(np.degrees(np.angle(vector))),
        "phase_locking_value": float(np.abs(vector)),
    }


def _method_summary(
    scores: list[np.ndarray],
    segments: list[np.ndarray],
    timestamps_s: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    fps: float,
    band_hz: tuple[float, float],
    edge_seconds: float,
    window_seconds: tuple[float, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    power, analyzed_samples = _frequency_power(
        scores,
        segments,
        timestamps_s,
        frequencies_hz,
        edge_seconds=edge_seconds,
    )
    peak_index = int(np.nanargmax(power))
    polarity, event_times, intervals, interval_cv = _segmented_events(
        scores,
        segments,
        timestamps_s,
        fps=fps,
        band_hz=band_hz,
        edge_seconds=edge_seconds,
    )
    window_output: list[dict[str, Any]] = []
    window_summaries: dict[str, Any] = {}
    for duration in window_seconds:
        rows = _window_rows(
            scores,
            segments,
            timestamps_s,
            frequencies_hz,
            window_seconds=duration,
            edge_seconds=edge_seconds,
            minimum_samples=max(8, int(round(2.0 * fps))),
        )
        valid = [row for row in rows if row["status"] == "ok"]
        candidates = np.asarray(
            [float(row["candidate_cycles_per_min"]) for row in valid], dtype=np.float64
        )
        window_summaries[f"{duration:g}s"] = {
            "total_windows": len(rows),
            "scorable_windows": len(valid),
            "median_cycles_per_min": float(np.median(candidates)) if candidates.size else None,
            "iqr_cycles_per_min": (
                [float(np.quantile(candidates, 0.25)), float(np.quantile(candidates, 0.75))]
                if candidates.size
                else None
            ),
            "fraction_180_to_216_per_min": (
                float(np.mean((candidates >= 180.0) & (candidates <= 216.0)))
                if candidates.size
                else None
            ),
        }
        for row in rows:
            window_output.append({"window_seconds": duration, **row})
    summary = {
        "analyzed_post_edge_samples": int(analyzed_samples),
        "aggregate_peak_hz": float(frequencies_hz[peak_index]),
        "aggregate_cycles_per_min": float(60.0 * frequencies_hz[peak_index]),
        "peak_to_band_median": float(
            power[peak_index] / max(np.nanmedian(power), np.finfo(float).tiny)
        ),
        "event_polarity": int(polarity),
        "event_count": int(event_times.size),
        "within_segment_interval_count": int(intervals.size),
        "median_interval_cycles_per_min": (
            float(60.0 / np.median(intervals)) if intervals.size else None
        ),
        "interval_cv": float(interval_cv),
        "windows": window_summaries,
    }
    return summary, window_output, power, event_times, intervals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PCA and frozen full/upper/lower equal means in moving-fish photometry."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--mask-npz", type=Path, required=True)
    parser.add_argument("--mask-key", default="original_38_mask")
    parser.add_argument("--regions-npz", type=Path, required=True)
    parser.add_argument("--upper-key", default="upper_mask")
    parser.add_argument("--lower-key", default="lower_mask")
    parser.add_argument("--band-hz", type=float, nargs=2, default=(2.0, 4.0))
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--window-seconds", type=float, nargs="+", default=(4.0, 8.0))
    parser.add_argument("--minimum-segment-seconds", type=float, default=2.0)
    parser.add_argument("--max-interpolated-gap-seconds", type=float, default=0.02)
    parser.add_argument("--filter-edge-seconds", type=float, default=0.75)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_npz)
    band_hz = (float(args.band_hz[0]), float(args.band_hz[1]))
    config = HeartrateConfig(band_min_hz=band_hz[0], band_max_hz=band_hz[1]).validated()
    eligible = np.asarray(build_risk_surfaces(dataset, config).eligible, dtype=bool)
    mask = _read_mask_membership(args.mask_npz, args.mask_key, dataset.pixel_xy)
    upper = _read_mask_membership(args.regions_npz, args.upper_key, dataset.pixel_xy) & mask
    lower = _read_mask_membership(args.regions_npz, args.lower_key, dataset.pixel_xy) & mask
    eligible &= mask
    if np.any(upper & lower) or not np.array_equal(upper | lower, mask):
        raise ValueError("upper/lower regions must partition the frozen mask")
    if not np.array_equal(mask & eligible, mask):
        raise ValueError("some frozen mask pixels fail the anatomical eligibility contract")

    fps = _sample_rate(dataset.timestamps_s)
    segments, interpolated_rows = _common_valid_segments(
        dataset.timestamps_s,
        dataset.frame_valid,
        dataset.pixel_valid,
        eligible,
        min_seconds=float(args.minimum_segment_seconds),
        max_interpolated_gap_seconds=float(args.max_interpolated_gap_seconds),
    )
    filtered, usable = segmented_filtered_pixels(
        np.asarray(dataset.traces)[:, eligible], segments, fps=fps, band_hz=band_hz
    )
    upper_local = upper[eligible][usable]
    lower_local = lower[eligible][usable]
    if not np.any(upper_local) or not np.any(lower_local):
        raise ValueError("upper or lower region has no usable pixels")
    scores, pca_loading, pca_variance = projection_scores(
        filtered, upper_usable=upper_local, lower_usable=lower_local
    )
    scores["lower_raw_mean"] = segmented_raw_region_mean(
        np.asarray(dataset.traces)[:, lower],
        segments,
        fps=fps,
        band_hz=band_hz,
    )

    motion_values = np.asarray(dataset.motion_prediction, dtype=np.float64)[:, eligible].copy()
    motion_values[interpolated_rows] = np.nan
    motion_filtered, motion_usable = segmented_filtered_pixels(
        motion_values, segments, fps=fps, band_hz=band_hz
    )
    motion_upper = upper[eligible][motion_usable]
    motion_lower = lower[eligible][motion_usable]
    motion_scores, _motion_loading, motion_pca_variance = projection_scores(
        motion_filtered, upper_usable=motion_upper, lower_usable=motion_lower
    )
    motion_scores["lower_raw_mean"] = segmented_raw_region_mean(
        motion_values[:, lower[eligible]],
        segments,
        fps=fps,
        band_hz=band_hz,
    )

    frequencies = np.arange(
        band_hz[0],
        band_hz[1] + float(args.frequency_step_hz) * 0.5,
        float(args.frequency_step_hz),
    )
    summaries: dict[str, Any] = {}
    powers: dict[str, np.ndarray] = {}
    events: dict[str, np.ndarray] = {}
    intervals: dict[str, np.ndarray] = {}
    window_rows: list[dict[str, Any]] = []
    for method in _METHODS:
        summary, rows, power, event_times, method_intervals = _method_summary(
            scores[method],
            segments,
            dataset.timestamps_s,
            frequencies,
            fps=fps,
            band_hz=band_hz,
            edge_seconds=float(args.filter_edge_seconds),
            window_seconds=tuple(float(value) for value in args.window_seconds),
        )
        motion_power, _ = _frequency_power(
            motion_scores[method],
            segments,
            dataset.timestamps_s,
            frequencies,
            edge_seconds=float(args.filter_edge_seconds),
        )
        motion_peak = int(np.nanargmax(motion_power))
        summary["motion_aggregate_peak_hz"] = float(frequencies[motion_peak])
        summary["motion_peak_to_band_median"] = float(
            motion_power[motion_peak]
            / max(np.nanmedian(motion_power), np.finfo(float).tiny)
        )
        observed_joined = _post_edge_joined(
            scores[method], segments, dataset.timestamps_s, edge_seconds=float(args.filter_edge_seconds)
        )
        motion_joined = _post_edge_joined(
            motion_scores[method], segments, dataset.timestamps_s, edge_seconds=float(args.filter_edge_seconds)
        )
        summary["absolute_motion_waveform_correlation"] = float(
            abs(np.corrcoef(observed_joined, motion_joined)[0, 1])
        )
        summaries[method] = summary
        powers[method] = power
        events[method] = event_times
        intervals[method] = method_intervals
        window_rows.extend({"method": method, **row} for row in rows)

    joined = {
        method: _post_edge_joined(
            scores[method], segments, dataset.timestamps_s, edge_seconds=float(args.filter_edge_seconds)
        )
        for method in _METHODS
    }
    correlations: dict[str, float] = {}
    for left_index, left in enumerate(_METHODS):
        for right in _METHODS[left_index + 1 :]:
            correlations[f"{left}__{right}"] = float(
                np.corrcoef(joined[left], joined[right])[0, 1]
            )
    phase = regional_phase_summary(
        scores["upper_equal_mean"],
        scores["lower_equal_mean"],
        segments,
        dataset.timestamps_s,
        edge_seconds=float(args.filter_edge_seconds),
    )
    frequency_agreement: dict[str, Any] = {}
    for duration in (float(value) for value in args.window_seconds):
        by_method: dict[str, dict[float, float]] = {}
        for method in _METHODS:
            by_method[method] = {
                float(row["window_start_s"]): float(row["candidate_frequency_hz"])
                for row in window_rows
                if row["method"] == method
                and float(row["window_seconds"]) == duration
                and row["status"] == "ok"
            }
        for left, right in (
            ("masked_pca", "full_equal_mean"),
            ("masked_pca", "upper_equal_mean"),
            ("masked_pca", "lower_equal_mean"),
            ("masked_pca", "lower_raw_mean"),
            ("lower_equal_mean", "lower_raw_mean"),
            ("upper_equal_mean", "lower_equal_mean"),
        ):
            common = sorted(set(by_method[left]) & set(by_method[right]))
            differences = np.asarray(
                [abs(by_method[left][start] - by_method[right][start]) for start in common],
                dtype=np.float64,
            )
            frequency_agreement[f"{duration:g}s:{left}__{right}"] = {
                "paired_windows": int(differences.size),
                "median_absolute_difference_hz": (
                    float(np.median(differences)) if differences.size else None
                ),
                "fraction_within_0p10_hz": (
                    float(np.mean(differences <= 0.10 + 1e-12))
                    if differences.size
                    else None
                ),
                "fraction_within_0p20_hz": (
                    float(np.mean(differences <= 0.20 + 1e-12))
                    if differences.size
                    else None
                ),
            }

    full_traces = {method: np.full(dataset.frame_count, np.nan, dtype=np.float32) for method in _METHODS}
    for segment_index, rows in enumerate(segments):
        for method in _METHODS:
            full_traces[method][rows] = np.asarray(scores[method][segment_index], dtype=np.float32)

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_suffix(".summary.json")
    windows_path = prefix.with_suffix(".windows.csv")
    arrays_path = prefix.with_suffix(".arrays.npz")
    plot_path = prefix.with_suffix(".png")
    payload = {
        "analysis_status": "exploratory_same_cache_frozen_mask_projection_comparison",
        "dataset_npz": str(args.dataset_npz.resolve()),
        "mask_npz": str(args.mask_npz.resolve()),
        "mask_key": str(args.mask_key),
        "regions_npz": str(args.regions_npz.resolve()),
        "upper_key": str(args.upper_key),
        "lower_key": str(args.lower_key),
        "band_hz": list(band_hz),
        "window_seconds": [float(value) for value in args.window_seconds],
        "pixel_counts": {
            "full": int(np.count_nonzero(mask)),
            "upper": int(np.count_nonzero(upper)),
            "lower": int(np.count_nonzero(lower)),
            "usable_full": int(np.count_nonzero(usable)),
            "usable_upper": int(np.count_nonzero(upper_local)),
            "usable_lower": int(np.count_nonzero(lower_local)),
        },
        "common_valid_segment_count": len(segments),
        "pca_explained_variance_fraction": pca_variance,
        "motion_pca_explained_variance_fraction": motion_pca_variance,
        "summaries": summaries,
        "pairwise_signed_waveform_correlations": correlations,
        "paired_window_frequency_agreement": frequency_agreement,
        "upper_lower_phase": phase,
        "limitations": [
            "All projections and event polarities are fit or selected on the same moving-fish cache.",
            "The frozen spatial mask and upper/lower split are reused, not newly discovered here.",
            "There is no cardiac reference for the moving fish.",
            "This comparison describes candidate oscillators and does not validate heartbeats.",
        ],
        "projection_definitions": {
            "lower_equal_mean": "Equal mean after per-pixel robust scaling and bandpass.",
            "lower_raw_mean": "Literal Mono8 lower-mask mean formed before segment detrending and bandpass.",
        },
        "windows_csv": str(windows_path),
        "arrays_npz": str(arrays_path),
        "plot": str(plot_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with windows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)
    np.savez_compressed(
        arrays_path,
        frequencies_hz=frequencies,
        upper_membership=upper,
        lower_membership=lower,
        pca_loading_usable=pca_loading,
        **{f"power_{method}": powers[method] for method in _METHODS},
        **{f"trace_{method}": full_traces[method] for method in _METHODS},
        **{f"event_times_s_{method}": events[method] for method in _METHODS},
        **{f"intervals_s_{method}": intervals[method] for method in _METHODS},
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = {
        "masked_pca": "black",
        "full_equal_mean": "tab:purple",
        "upper_equal_mean": "tab:red",
        "lower_equal_mean": "tab:blue",
        "lower_raw_mean": "tab:green",
    }
    for method in _METHODS:
        axes[0, 0].plot(
            frequencies * 60.0,
            powers[method] / max(np.nanmax(powers[method]), np.finfo(float).tiny),
            color=colors[method],
            label=method,
        )
    axes[0, 0].set(title="Segment-preserving spectra", xlabel="Cycles/min", ylabel="Relative power")
    axes[0, 0].legend(fontsize=8)
    for panel, duration in enumerate((4.0, 8.0)):
        axis = axes.reshape(-1)[panel + 1]
        for method in _METHODS:
            valid = [
                row
                for row in window_rows
                if row["method"] == method
                and float(row["window_seconds"]) == duration
                and row["status"] == "ok"
            ]
            axis.plot(
                [float(row["window_mid_s"]) / 60.0 for row in valid],
                [float(row["candidate_cycles_per_min"]) for row in valid],
                color=colors[method],
                marker=".",
                linewidth=0.8,
                label=method,
            )
        axis.set(title=f"{duration:g} s candidates", xlabel="Recording time (min)", ylabel="Cycles/min", ylim=(115, 240))
    axes[0, 1].legend(fontsize=7, ncol=2)
    longest = int(np.argmax([rows.size for rows in segments]))
    local_time = dataset.timestamps_s[segments[longest]] - dataset.timestamps_s[segments[longest][0]]
    for method in _METHODS:
        values = scores[method][longest]
        scale = max(1.4826 * np.median(np.abs(values - np.median(values))), 1e-9)
        axes[1, 1].plot(local_time, values / scale, color=colors[method], linewidth=1.0, label=method)
    axes[1, 1].set(title="Longest common-valid segment", xlabel="Segment time (s)", ylabel="Robust scale")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.2)
    figure.savefig(plot_path, dpi=170)
    plt.close(figure)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
