from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from analyze_segmented_cache_pca import (
    _common_valid_segments,
    _read_mask_membership,
    _sample_rate,
)
from compare_moving_frozen_mask_means import segmented_raw_region_mean
from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.heart_photometry_motion_controls import tracking_feature_traces
from fisheye.analysis.local_rostral_heartrate import HeartrateConfig, build_risk_surfaces
from render_segmented_cache_pca_overlay import _analysis_core_rows


def _finite_stat(values: np.ndarray, statistic: str) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    if statistic == "median":
        return float(np.median(finite))
    if statistic == "mean":
        return float(np.mean(finite))
    if statistic == "minimum":
        return float(np.min(finite))
    if statistic == "p95":
        return float(np.quantile(finite, 0.95))
    if statistic == "rms":
        return float(np.sqrt(np.mean(np.square(finite))))
    raise ValueError(f"unknown statistic: {statistic}")


def _absolute_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 8:
        return float("nan")
    left_finite = left[finite]
    right_finite = right[finite]
    if np.std(left_finite) <= 1e-12 or np.std(right_finite) <= 1e-12:
        return float("nan")
    return float(abs(np.corrcoef(left_finite, right_finite)[0, 1]))


def _cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if not left.size or not right.size:
        return float("nan")
    differences = left[:, None] - right[None, :]
    return float((np.count_nonzero(differences > 0) - np.count_nonzero(differences < 0)) / differences.size)


def _classification(
    deviation_bpm: float,
    *,
    stable_threshold_bpm: float,
    excursion_threshold_bpm: float,
) -> str:
    if deviation_bpm <= stable_threshold_bpm:
        return "stable"
    if deviation_bpm >= excursion_threshold_bpm:
        return "excursion"
    return "intermediate"


def _metric_summary(
    rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    deviation = np.asarray([float(row["absolute_deviation_bpm"]) for row in rows])
    values = np.asarray([float(row[metric]) for row in rows])
    stable = values[[row["classification"] == "stable" for row in rows]]
    excursion = values[[row["classification"] == "excursion" for row in rows]]
    finite = np.isfinite(deviation) & np.isfinite(values)
    supports_rank_correlation = (
        np.count_nonzero(finite) >= 3
        and np.unique(deviation[finite]).size >= 2
        and np.unique(values[finite]).size >= 2
    )
    rho = (
        float(spearmanr(deviation[finite], values[finite]).statistic)
        if supports_rank_correlation
        else float("nan")
    )
    return {
        "all_finite_windows": int(np.count_nonzero(np.isfinite(values))),
        "stable_median": _finite_stat(stable, "median"),
        "excursion_median": _finite_stat(excursion, "median"),
        "excursion_minus_stable_median": (
            _finite_stat(excursion, "median") - _finite_stat(stable, "median")
        ),
        "cliffs_delta_excursion_vs_stable": _cliffs_delta(excursion, stable),
        "spearman_vs_absolute_rate_deviation": rho,
    }


def _nuisance_column(dataset: Any, name: str) -> np.ndarray:
    try:
        index = tuple(dataset.nuisance_names).index(name)
    except ValueError as error:
        raise ValueError(f"dataset lacks required nuisance trace {name!r}") from error
    return np.asarray(dataset.nuisance_values, dtype=np.float64)[:, index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare stable and excursion 4-second lower-mask windows against tracking covariates."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--full-mask-npz", type=Path, required=True)
    parser.add_argument("--full-mask-key", default="original_38_mask")
    parser.add_argument("--lower-mask-npz", type=Path, required=True)
    parser.add_argument("--lower-mask-key", default="lower_mask")
    parser.add_argument("--comparison-arrays", type=Path, required=True)
    parser.add_argument("--window-csv", type=Path, required=True)
    parser.add_argument("--method", default="lower_raw_mean")
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--band-hz", type=float, nargs=2, default=(2.0, 4.0))
    parser.add_argument("--stable-threshold-bpm", type=float, default=6.0)
    parser.add_argument("--excursion-threshold-bpm", type=float, default=24.0)
    parser.add_argument("--minimum-segment-seconds", type=float, default=2.0)
    parser.add_argument("--max-interpolated-gap-seconds", type=float, default=0.02)
    parser.add_argument("--filter-edge-seconds", type=float, default=0.75)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if args.stable_threshold_bpm >= args.excursion_threshold_bpm:
        parser.error("stable threshold must be below excursion threshold")

    dataset = load_dataset(args.dataset_npz)
    fps = _sample_rate(dataset.timestamps_s)
    band_hz = (float(args.band_hz[0]), float(args.band_hz[1]))
    config = HeartrateConfig(band_min_hz=band_hz[0], band_max_hz=band_hz[1]).validated()
    eligible = np.asarray(build_risk_surfaces(dataset, config).eligible, dtype=bool)
    full = _read_mask_membership(args.full_mask_npz, args.full_mask_key, dataset.pixel_xy)
    lower = _read_mask_membership(args.lower_mask_npz, args.lower_mask_key, dataset.pixel_xy)
    full_selected = eligible & full
    lower_selected = eligible & lower
    if np.any(lower_selected & ~full_selected):
        raise ValueError("lower mask must be a subset of the frozen full mask")

    segments, interpolated = _common_valid_segments(
        dataset.timestamps_s,
        dataset.frame_valid,
        dataset.pixel_valid,
        full_selected,
        min_seconds=float(args.minimum_segment_seconds),
        max_interpolated_gap_seconds=float(args.max_interpolated_gap_seconds),
    )
    analysis_valid = _analysis_core_rows(
        segments,
        dataset.timestamps_s,
        edge_seconds=float(args.filter_edge_seconds),
        frame_count=dataset.frame_count,
    )

    with np.load(args.comparison_arrays) as arrays:
        observed = np.asarray(arrays[f"trace_{args.method}"], dtype=np.float64)
        saved_lower = np.asarray(arrays["lower_membership"], dtype=bool)
    if not np.array_equal(saved_lower, lower_selected):
        raise ValueError("comparison lower membership differs from the requested mask")

    motion_values = np.asarray(dataset.motion_prediction, dtype=np.float64)[:, lower_selected].copy()
    motion_values[interpolated] = np.nan
    motion_segments = segmented_raw_region_mean(
        motion_values,
        segments,
        fps=fps,
        band_hz=band_hz,
    )
    motion_trace = np.full(dataset.frame_count, np.nan, dtype=np.float64)
    for segment_rows, values in zip(segments, motion_segments):
        motion_trace[segment_rows] = values

    tracking = tracking_feature_traces(dataset, lower_selected)
    detection_confidence = _nuisance_column(dataset, "detection_confidence")
    local_rotation = np.abs(_nuisance_column(dataset, "local_rotation_deg"))
    local_translation = _nuisance_column(dataset, "local_translation_px")
    body_scale = _nuisance_column(dataset, "body_scale")

    with args.window_csv.open(newline="") as handle:
        source_rows = [
            row
            for row in csv.DictReader(handle)
            if row["method"] == args.method
            and np.isclose(float(row["window_seconds"]), float(args.window_seconds))
            and row["status"] == "ok"
        ]
    if not source_rows:
        raise ValueError("no matching scorable window rows")
    candidate_values = np.asarray(
        [float(row["candidate_cycles_per_min"]) for row in source_rows], dtype=np.float64
    )
    center_bpm = float(np.median(candidate_values))
    relative_time = np.asarray(dataset.timestamps_s, dtype=np.float64) - float(dataset.timestamps_s[0])

    output_rows: list[dict[str, Any]] = []
    for row in source_rows:
        start = float(row["window_start_s"])
        stop = float(row["window_stop_s"])
        candidate = float(row["candidate_cycles_per_min"])
        deviation = abs(candidate - center_bpm)
        window_rows = np.flatnonzero(
            (relative_time >= start)
            & (relative_time < stop)
            & analysis_valid
        )
        source_step = tracking.source_step_px[window_rows]
        gradient_displacement = tracking.abs_gradient_displacement[window_rows]
        gradient = tracking.gradient_magnitude[window_rows]
        uncertainty = tracking.transform_uncertainty[window_rows]
        valid_fraction = tracking.valid_pixel_fraction[window_rows]
        scale_values = body_scale[window_rows]
        scale_median = _finite_stat(scale_values, "median")
        scale_relative_sd = (
            float(np.nanstd(scale_values) / max(abs(scale_median), 1e-12))
            if np.any(np.isfinite(scale_values))
            else float("nan")
        )
        output_rows.append(
            {
                "window_start_s": start,
                "window_stop_s": stop,
                "window_mid_s": float(row["window_mid_s"]),
                "candidate_cycles_per_min": candidate,
                "absolute_deviation_bpm": deviation,
                "classification": _classification(
                    deviation,
                    stable_threshold_bpm=float(args.stable_threshold_bpm),
                    excursion_threshold_bpm=float(args.excursion_threshold_bpm),
                ),
                "peak_to_band_median": float(row["peak_to_band_median"]),
                "reported_common_valid_samples": int(row["common_valid_samples"]),
                "reconstructed_analysis_valid_samples": int(window_rows.size),
                "source_step_px_median": _finite_stat(source_step, "median"),
                "source_step_px_p95": _finite_stat(source_step, "p95"),
                "abs_gradient_displacement_median": _finite_stat(gradient_displacement, "median"),
                "abs_gradient_displacement_p95": _finite_stat(gradient_displacement, "p95"),
                "gradient_magnitude_median": _finite_stat(gradient, "median"),
                "gradient_magnitude_p95": _finite_stat(gradient, "p95"),
                "transform_uncertainty_median": _finite_stat(uncertainty, "median"),
                "transform_uncertainty_p95": _finite_stat(uncertainty, "p95"),
                "valid_pixel_fraction_mean": _finite_stat(valid_fraction, "mean"),
                "valid_pixel_fraction_minimum": _finite_stat(valid_fraction, "minimum"),
                "detection_confidence_median": _finite_stat(detection_confidence[window_rows], "median"),
                "detection_confidence_minimum": _finite_stat(detection_confidence[window_rows], "minimum"),
                "absolute_local_rotation_deg_p95": _finite_stat(local_rotation[window_rows], "p95"),
                "local_translation_px_p95": _finite_stat(local_translation[window_rows], "p95"),
                "body_scale_relative_sd": scale_relative_sd,
                "observed_trace_rms": _finite_stat(observed[window_rows], "rms"),
                "motion_trace_rms": _finite_stat(motion_trace[window_rows], "rms"),
                "absolute_observed_motion_correlation": _absolute_correlation(
                    observed[window_rows], motion_trace[window_rows]
                ),
            }
        )

    metric_names = [
        "peak_to_band_median",
        "reported_common_valid_samples",
        "source_step_px_p95",
        "abs_gradient_displacement_p95",
        "gradient_magnitude_p95",
        "transform_uncertainty_p95",
        "valid_pixel_fraction_mean",
        "detection_confidence_minimum",
        "absolute_local_rotation_deg_p95",
        "local_translation_px_p95",
        "body_scale_relative_sd",
        "motion_trace_rms",
        "absolute_observed_motion_correlation",
    ]
    metric_summaries = {
        metric: _metric_summary(output_rows, metric) for metric in metric_names
    }
    class_counts = {
        label: sum(row["classification"] == label for row in output_rows)
        for label in ("stable", "intermediate", "excursion")
    }
    ranked_excursions = sorted(
        (row for row in output_rows if row["classification"] == "excursion"),
        key=lambda row: (-float(row["absolute_deviation_bpm"]), float(row["window_start_s"])),
    )

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".windows.csv")
    summary_path = prefix.with_suffix(".summary.json")
    plot_path = prefix.with_suffix(".png")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    payload = {
        "analysis_status": "descriptive_same_cache_rate_excursion_tracking_diagnostic",
        "dataset_npz": str(args.dataset_npz.resolve()),
        "comparison_arrays": str(args.comparison_arrays.resolve()),
        "source_window_csv": str(args.window_csv.resolve()),
        "method": args.method,
        "window_seconds": float(args.window_seconds),
        "band_hz": list(band_hz),
        "center_cycles_per_min": center_bpm,
        "stable_threshold_bpm": float(args.stable_threshold_bpm),
        "excursion_threshold_bpm": float(args.excursion_threshold_bpm),
        "class_counts": class_counts,
        "metric_summaries": metric_summaries,
        "ranked_excursions": ranked_excursions,
        "limitations": [
            "Groups are defined from the same candidate-rate outcome being diagnosed.",
            "Window observations are temporally dependent; effect sizes are descriptive, not inferential p-values.",
            "Tracking covariates can miss anatomically incorrect placement that remains geometrically smooth.",
            "No synchronized cardiac reference exists for this moving recording.",
        ],
        "windows_csv": str(csv_path),
        "plot": str(plot_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    panels = [
        ("peak_to_band_median", "Peak / band median"),
        ("source_step_px_p95", "Source-coordinate step p95 (px)"),
        ("abs_gradient_displacement_p95", "|gradient x displacement| p95"),
        ("transform_uncertainty_p95", "Transform uncertainty p95"),
        ("detection_confidence_minimum", "Detection confidence minimum"),
        ("absolute_observed_motion_correlation", "|observed-motion correlation|"),
    ]
    colors = {"stable": "#2a9d55", "intermediate": "#777777", "excursion": "#d1493f"}
    for axis, (metric, label) in zip(axes.reshape(-1), panels):
        for classification in ("intermediate", "stable", "excursion"):
            selected_rows = [row for row in output_rows if row["classification"] == classification]
            axis.scatter(
                [float(row["absolute_deviation_bpm"]) for row in selected_rows],
                [float(row[metric]) for row in selected_rows],
                s=28,
                alpha=0.8,
                color=colors[classification],
                label=classification,
            )
        rho = metric_summaries[metric]["spearman_vs_absolute_rate_deviation"]
        axis.set(
            xlabel=f"Absolute deviation from {center_bpm:.0f}/min",
            ylabel=label,
            title=f"Spearman rho={rho:.2f}",
        )
        axis.grid(True, alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Moving lower-mask 4 s excursions versus tracking diagnostics")
    figure.savefig(plot_path, dpi=170)
    plt.close(figure)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
