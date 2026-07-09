from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _common import ensure_output_dir
from align_hr_to_chase_trials import _load_chase_trials, _unwrap_monotonic_counter


def _read_hr_timeseries(path: Path, *, strategy: str, smooth_windows: int) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "strategy",
        "status",
        "primary_estimator",
        "window_index",
        "window_start_s",
        "window_stop_s",
        "window_frame_start",
        "window_frame_stop_inclusive",
        "peak_frequency_hz",
        "peak_bpm",
        "peak_score",
        "pixel_count",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")

    selected = table[(table["strategy"] == strategy) & (table["status"] == "ok")].copy()
    if selected.empty:
        raise ValueError(f"No ok rows found for strategy={strategy!r} in {path}")
    selected = selected.sort_values(["window_start_s", "window_stop_s"]).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "window_index": selected["window_index"].astype(int),
            "window_start_s": selected["window_start_s"].astype(float),
            "window_stop_s": selected["window_stop_s"].astype(float),
            "time_s": (selected["window_start_s"].astype(float) + selected["window_stop_s"].astype(float)) / 2.0,
            "window_frame_start": selected["window_frame_start"].astype(int),
            "window_frame_stop_inclusive": selected["window_frame_stop_inclusive"].astype(int),
            "strategy": selected["strategy"].astype(str),
            "primary_estimator": selected["primary_estimator"].astype(str),
            "hr_frequency_hz": selected["peak_frequency_hz"].astype(float),
            "hr_bpm": selected["peak_bpm"].astype(float),
            "peak_score": selected["peak_score"].astype(float),
            "pixel_count": selected["pixel_count"].astype(int),
        }
    )
    out["time_min"] = out["time_s"] / 60.0
    smooth_windows = max(1, int(smooth_windows))
    out["hr_bpm_rolling_median"] = (
        out["hr_bpm"].rolling(window=smooth_windows, center=True, min_periods=1).median().astype(float)
    )
    out["peak_score_rolling_median"] = (
        out["peak_score"].rolling(window=smooth_windows, center=True, min_periods=1).median().astype(float)
    )

    optional_columns = [
        "welch_peak_bpm",
        "periodogram_peak_bpm",
        "autocorr_peak_bpm",
        "autocorr_peak_lag_samples",
        "autocorr_peak_strength",
    ]
    for column in optional_columns:
        if column in selected.columns:
            out[column] = selected[column].astype(float)
    return out


def _load_chase_trial_times(
    *,
    stimulus_h5: Path,
    crop_meta_csv: Path,
    fps: float,
) -> pd.DataFrame:
    trials = _load_chase_trials(stimulus_h5)
    crop = pd.read_csv(crop_meta_csv, usecols=["camera_frame_id"])
    camera_frames = _unwrap_monotonic_counter(crop["camera_frame_id"].to_numpy(dtype=np.int64))
    first_camera_frame = float(camera_frames[0])
    out = trials.copy()
    out["start_time_s"] = (out["start_camera_frame_id"].astype(float) - first_camera_frame) / float(fps)
    out["end_time_s"] = (out["end_camera_frame_id"].astype(float) - first_camera_frame) / float(fps)
    out["start_time_min"] = out["start_time_s"] / 60.0
    out["end_time_min"] = out["end_time_s"] / 60.0
    return out


def _write_plot(
    path: Path,
    *,
    series: pd.DataFrame,
    trials: pd.DataFrame | None,
    title: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    x = series["time_min"].to_numpy(dtype=float)
    hr = series["hr_bpm"].to_numpy(dtype=float)
    smooth = series["hr_bpm_rolling_median"].to_numpy(dtype=float)
    score = series["peak_score"].to_numpy(dtype=float)
    score_smooth = series["peak_score_rolling_median"].to_numpy(dtype=float)

    axes[0].plot(x, hr, color="#8db7dd", lw=0.7, alpha=0.8, label="window HR")
    axes[0].plot(x, smooth, color="#1f4e79", lw=1.8, label="rolling median")
    axes[0].set_ylabel("HR (bpm)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, score, color="#c7a76c", lw=0.7, alpha=0.75, label="window score")
    axes[1].plot(x, score_smooth, color="#8a5a00", lw=1.5, label="rolling median")
    axes[1].set_ylabel("peak score")
    axes[1].set_xlabel("recording time (min)")
    axes[1].grid(True, alpha=0.25)

    if trials is not None and not trials.empty:
        for index, (_, trial) in enumerate(trials.iterrows()):
            start = float(trial["start_time_min"])
            stop = float(trial["end_time_min"])
            for axis in axes:
                axis.axvspan(
                    start,
                    stop,
                    color="#d95f02",
                    alpha=0.20,
                    label="chase trial" if index == 0 and axis is axes[0] else None,
                )

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    ensure_output_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _summary(series: pd.DataFrame, *, trials: pd.DataFrame | None, smooth_windows: int) -> dict[str, Any]:
    hr = series["hr_bpm"].to_numpy(dtype=float)
    score = series["peak_score"].to_numpy(dtype=float)
    payload: dict[str, Any] = {
        "window_count": int(len(series)),
        "strategy": str(series["strategy"].iloc[0]),
        "primary_estimator": str(series["primary_estimator"].iloc[0]),
        "smooth_windows": int(smooth_windows),
        "duration_s": float(series["window_stop_s"].max() - series["window_start_s"].min()),
        "hr_bpm_min": float(np.nanmin(hr)),
        "hr_bpm_median": float(np.nanmedian(hr)),
        "hr_bpm_mean": float(np.nanmean(hr)),
        "hr_bpm_max": float(np.nanmax(hr)),
        "peak_score_median": float(np.nanmedian(score)),
    }
    if trials is not None:
        payload["chase_trial_count"] = int(len(trials))
        payload["first_chase_start_s"] = float(trials["start_time_s"].min()) if len(trials) else math.nan
        payload["last_chase_end_s"] = float(trials["end_time_s"].max()) if len(trials) else math.nan
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and plot a whole-recording HR time series.")
    parser.add_argument("--hr-csv", type=Path, required=True)
    parser.add_argument("--strategy", default="eye_excluded")
    parser.add_argument("--smooth-windows", type=int, default=5)
    parser.add_argument("--stimulus-h5", type=Path, default=None, help="Optional GoodCopBadCop H5 for chase shading.")
    parser.add_argument("--crop-meta-csv", type=Path, default=None, help="Required with --stimulus-h5.")
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/hr_timeseries"),
    )
    args = parser.parse_args()

    series = _read_hr_timeseries(args.hr_csv, strategy=str(args.strategy), smooth_windows=int(args.smooth_windows))
    trials = None
    if args.stimulus_h5 is not None:
        if args.crop_meta_csv is None:
            raise ValueError("--crop-meta-csv is required when --stimulus-h5 is provided")
        trials = _load_chase_trial_times(
            stimulus_h5=args.stimulus_h5,
            crop_meta_csv=args.crop_meta_csv,
            fps=float(args.fps),
        )

    output_prefix = args.output_prefix
    ensure_output_dir(output_prefix.parent)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".summary.json")
    png_path = output_prefix.with_suffix(".png")
    chase_csv = output_prefix.with_suffix(".chase_trials.csv")

    series.to_csv(csv_path, index=False)
    if trials is not None:
        trials.to_csv(chase_csv, index=False)
    title = (
        f"HR time series: {series['strategy'].iloc[0]}, "
        f"{series['primary_estimator'].iloc[0]}, n={len(series)} windows"
    )
    _write_plot(png_path, series=series, trials=trials, title=title)
    summary = _summary(series, trials=trials, smooth_windows=int(args.smooth_windows))
    summary["hr_csv"] = str(args.hr_csv)
    summary["outputs"] = {
        "timeseries_csv": str(csv_path),
        "plot_png": str(png_path),
        "summary_json": str(json_path),
    }
    if trials is not None:
        summary["outputs"]["chase_trials_csv"] = str(chase_csv)
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"summary_json: {json_path}")
    print(f"timeseries_csv: {csv_path}")
    print(f"plot_png: {png_path}")
    if trials is not None:
        print(f"chase_trials_csv: {chase_csv}")
    print(
        f"windows={summary['window_count']} median_bpm={summary['hr_bpm_median']:.3f} "
        f"mean_bpm={summary['hr_bpm_mean']:.3f} range={summary['hr_bpm_min']:.3f}..{summary['hr_bpm_max']:.3f}"
    )


if __name__ == "__main__":
    main()
