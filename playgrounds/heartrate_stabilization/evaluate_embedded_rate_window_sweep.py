from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_embedded_positive_control import (
    _reference_at_times,
    _ridge,
    _safe_correlation,
    read_numeric_xlsx_row,
)


def finite_runs(values: np.ndarray) -> list[np.ndarray]:
    finite = np.flatnonzero(np.isfinite(np.asarray(values, dtype=np.float64)))
    if not finite.size:
        return []
    return [run for run in np.split(finite, np.flatnonzero(np.diff(finite) > 1) + 1) if run.size]


def evaluate_window(
    trace: np.ndarray,
    reference_hz: np.ndarray,
    *,
    fps: float,
    reference_fps: float,
    window_seconds: float,
    step_seconds: float,
    band_hz: tuple[float, float],
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
    ridge_times: list[np.ndarray] = []
    ridge_values: list[np.ndarray] = []
    for run in finite_runs(trace):
        if run.size < int(round(window_seconds * fps)):
            continue
        local_time, local_ridge = _ridge(
            np.asarray(trace, dtype=np.float64)[run],
            fps=fps,
            band_hz=band_hz,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )
        ridge_times.append(local_time + float(run[0]) / fps)
        ridge_values.append(local_ridge)
    if not ridge_times:
        raise ValueError(f"no finite run supports a {window_seconds:g} second window")
    times = np.concatenate(ridge_times)
    ridge = np.concatenate(ridge_values)
    reference = _reference_at_times(reference_hz, times, reference_fps)
    error_bpm = 60.0 * np.abs(ridge - reference)
    summary: dict[str, float | int] = {
        "window_seconds": float(window_seconds),
        "step_seconds": float(step_seconds),
        "ridge_samples": int(ridge.size),
        "ridge_correlation": _safe_correlation(ridge, reference),
        "ridge_mae_bpm": float(np.mean(error_bpm)),
        "ridge_median_absolute_error_bpm": float(np.median(error_bpm)),
        "ridge_p90_absolute_error_bpm": float(np.quantile(error_bpm, 0.9)),
    }
    return summary, times, ridge, reference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate shorter spectral-ridge windows on a frozen embedded top-view projection."
    )
    parser.add_argument("--analysis-arrays", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, default=1)
    parser.add_argument("--method", default="masked_equal_mean")
    parser.add_argument("--fps", type=float, default=200.0)
    parser.add_argument("--reference-fps", type=float, default=100.0)
    parser.add_argument("--band-hz", type=float, nargs=2, default=(1.5, 4.0))
    parser.add_argument("--window-seconds", type=float, nargs="+", default=(2, 3, 4, 6, 8, 10))
    parser.add_argument("--step-seconds", type=float, default=0.5)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    arrays = np.load(args.analysis_arrays)
    trace_key = f"trace_{args.method}"
    if trace_key not in arrays:
        raise KeyError(f"analysis arrays do not contain {trace_key}")
    trace = np.asarray(arrays[trace_key], dtype=np.float64)
    reference_hz = read_numeric_xlsx_row(
        args.workbook,
        sheet_name="heart_rate_trace",
        row_number=int(args.trial_number) + 1,
    )
    band_hz = (float(args.band_hz[0]), float(args.band_hz[1]))
    results: list[dict[str, float | int]] = []
    trajectories: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for window_seconds in args.window_seconds:
        summary, times, ridge, reference = evaluate_window(
            trace,
            reference_hz,
            fps=float(args.fps),
            reference_fps=float(args.reference_fps),
            window_seconds=float(window_seconds),
            step_seconds=float(args.step_seconds),
            band_hz=band_hz,
        )
        results.append(summary)
        trajectories[float(window_seconds)] = (times, ridge, reference)

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".summary.json")
    plot_path = prefix.with_suffix(".png")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    windows = np.asarray([float(row["window_seconds"]) for row in results])
    correlations = np.asarray([float(row["ridge_correlation"]) for row in results])
    errors = np.asarray([float(row["ridge_mae_bpm"]) for row in results])
    axes[0].plot(windows, correlations, marker="o", label="correlation")
    error_axis = axes[0].twinx()
    error_axis.plot(windows, errors, marker="s", color="tab:red", label="MAE")
    axes[0].set(xlabel="Window (s)", ylabel="Reference correlation", title="Frozen masked-mean window sweep")
    error_axis.set_ylabel("MAE (bpm)", color="tab:red")
    for window_seconds in windows:
        times, ridge, _reference = trajectories[float(window_seconds)]
        axes[1].plot(times, ridge * 60.0, linewidth=1.0, label=f"{window_seconds:g} s")
    longest = float(np.max(windows))
    times, _ridge_values, reference = trajectories[longest]
    axes[1].plot(times, reference * 60.0, color="black", linewidth=2.0, label="side reference")
    axes[1].set(xlabel="Source time (s)", ylabel="Rate (bpm)", title="Held-out ridge trajectories")
    axes[1].legend(ncol=4, fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.2)
    figure.savefig(plot_path, dpi=170)
    plt.close(figure)

    payload = {
        "analysis_status": "descriptive_frozen_projection_window_sweep",
        "analysis_arrays": str(args.analysis_arrays.resolve()),
        "workbook": str(args.workbook.resolve()),
        "trial_number": int(args.trial_number),
        "method": str(args.method),
        "fps": float(args.fps),
        "reference_fps": float(args.reference_fps),
        "band_hz": list(band_hz),
        "step_seconds": float(args.step_seconds),
        "gap_policy": "ridge windows are computed separately within each finite held-out block",
        "results": results,
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
