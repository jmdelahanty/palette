from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _common import ensure_output_dir


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").rstrip("\x00")
    return str(value)


def _load_chase_trials(path: Path) -> pd.DataFrame:
    import h5py

    with h5py.File(path, "r") as handle:
        trials = handle["trials/trial_index"][:]
        kinds = {
            int(row["id"]): _decode(row["name"]).lower()
            for row in handle["enums/runtime_trial_kinds"][:]
        }
        end_reasons = {
            int(row["id"]): _decode(row["name"]).lower()
            for row in handle["enums/runtime_trial_end_reasons"][:]
        }
    rows: list[dict[str, Any]] = []
    for row in trials:
        kind = kinds.get(int(row["trial_kind_id"]), str(int(row["trial_kind_id"])))
        if kind != "chaser_chase":
            continue
        details_raw = _decode(row["details_json"])
        try:
            details = json.loads(details_raw)
        except json.JSONDecodeError:
            details = {}
        rows.append(
            {
                "trial_id": int(row["trial_id"]),
                "trial_kind": kind,
                "start_camera_frame_id": int(row["start_camera_frame_id"]),
                "end_camera_frame_id": int(row["end_camera_frame_id"]),
                "start_stimulus_frame_num": int(row["start_stimulus_frame_num"]),
                "end_stimulus_frame_num": int(row["end_stimulus_frame_num"]),
                "end_reason": end_reasons.get(int(row["end_reason_id"]), str(int(row["end_reason_id"]))),
                "duration_camera_frames": int(row["end_camera_frame_id"]) - int(row["start_camera_frame_id"]),
                "pre_chase_dist_px": _nested_float(details, "start", "pre_chase_dist_px"),
                "in_danger_zone": _nested_bool(details, "start", "in_danger_zone"),
            }
        )
    return pd.DataFrame(rows)


def _nested_float(payload: dict[str, Any], *keys: str) -> float:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return math.nan
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return math.nan


def _nested_bool(payload: dict[str, Any], *keys: str) -> int | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, bool):
        return int(current)
    return None


def _read_hr_windows(path: Path, *, strategy: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"strategy", "status", "window_frame_start", "window_frame_stop_inclusive", "peak_bpm", "peak_score"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    selected = table[(table["strategy"] == strategy) & (table["status"] == "ok")].copy()
    if selected.empty:
        raise ValueError(f"No ok rows found for strategy={strategy!r} in {path}")
    selected["window_frame_start"] = selected["window_frame_start"].astype(int)
    selected["window_frame_stop_inclusive"] = selected["window_frame_stop_inclusive"].astype(int)
    selected["peak_bpm"] = selected["peak_bpm"].astype(float)
    selected["peak_score"] = selected["peak_score"].astype(float)
    return selected


def _add_camera_frame_mapping(hr: pd.DataFrame, *, crop_meta_csv: Path, fps: float) -> pd.DataFrame:
    crop = pd.read_csv(crop_meta_csv, usecols=["camera_frame_id"])
    camera_frames = _unwrap_monotonic_counter(crop["camera_frame_id"].to_numpy(dtype=np.int64))
    max_index = len(camera_frames) - 1
    starts = hr["window_frame_start"].to_numpy(dtype=np.int64)
    stops = hr["window_frame_stop_inclusive"].to_numpy(dtype=np.int64)
    if np.any(starts < 0) or np.any(stops < 0) or np.any(starts > max_index) or np.any(stops > max_index):
        raise ValueError(f"HR window frame range exceeds crop metadata rows 0..{max_index}")
    out = hr.copy()
    out["window_start_camera_frame_id"] = camera_frames[starts]
    out["window_stop_camera_frame_id"] = camera_frames[stops]
    out["window_mid_camera_frame_id"] = (
        out["window_start_camera_frame_id"].astype(float) + out["window_stop_camera_frame_id"].astype(float)
    ) / 2.0
    first_camera_frame = int(camera_frames[0])
    out["window_mid_recording_s"] = (out["window_mid_camera_frame_id"] - first_camera_frame) / float(fps)
    return out


def _unwrap_monotonic_counter(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.int64)
    out = raw.copy()
    offset = 0
    previous = int(raw[0]) if raw.size else 0
    for index, value in enumerate(raw):
        current = int(value)
        if index > 0 and current < previous:
            # Orange camera_frame_id wraps from 65535 to 1 in this recording.
            offset += previous
        out[index] = current + offset
        previous = current
    return out


def _overlap_frames(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _classify_windows(
    hr: pd.DataFrame,
    trials: pd.DataFrame,
    *,
    fps: float,
    pre_seconds: float,
    post_seconds: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pre_frames = float(pre_seconds * fps)
    post_frames = float(post_seconds * fps)
    for _, window in hr.iterrows():
        w0 = float(window["window_start_camera_frame_id"])
        w1 = float(window["window_stop_camera_frame_id"] + 1)
        midpoint = float(window["window_mid_camera_frame_id"])
        overlap_ids: list[int] = []
        overlap_frames = 0.0
        nearest_trial_id: int | None = None
        nearest_trial_delta_s = math.inf
        phase = "outside"
        for _, trial in trials.iterrows():
            t0 = float(trial["start_camera_frame_id"])
            t1 = float(trial["end_camera_frame_id"])
            overlap = _overlap_frames(w0, w1, t0, t1)
            if overlap > 0:
                overlap_frames += overlap
                overlap_ids.append(int(trial["trial_id"]))
            if midpoint < t0:
                delta_frames = midpoint - t0
            elif midpoint > t1:
                delta_frames = midpoint - t1
            else:
                delta_frames = 0.0
            if abs(delta_frames) < abs(nearest_trial_delta_s * fps):
                nearest_trial_delta_s = delta_frames / float(fps)
                nearest_trial_id = int(trial["trial_id"])
        if overlap_frames > 0:
            phase = "during_chase"
        else:
            for _, trial in trials.iterrows():
                t0 = float(trial["start_camera_frame_id"])
                t1 = float(trial["end_camera_frame_id"])
                if t0 - pre_frames <= midpoint < t0:
                    phase = "pre_chase"
                    break
                if t1 <= midpoint < t1 + post_frames:
                    phase = "post_chase"
                    break
        row = dict(window)
        row.update(
            {
                "chase_phase": phase,
                "chase_overlap_s": overlap_frames / float(fps),
                "chase_overlap_fraction": overlap_frames / max(1.0, w1 - w0),
                "overlap_trial_ids": ";".join(str(value) for value in overlap_ids),
                "nearest_trial_id": nearest_trial_id,
                "nearest_trial_delta_s": nearest_trial_delta_s,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_by_phase(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phase, group in windows.groupby("chase_phase", sort=True):
        bpm = group["peak_bpm"].to_numpy(dtype=float)
        score = group["peak_score"].to_numpy(dtype=float)
        rows.append(
            {
                "chase_phase": phase,
                "window_count": int(len(group)),
                "peak_bpm_median": float(np.nanmedian(bpm)),
                "peak_bpm_mean": float(np.nanmean(bpm)),
                "peak_bpm_min": float(np.nanmin(bpm)),
                "peak_bpm_max": float(np.nanmax(bpm)),
                "peak_score_median": float(np.nanmedian(score)),
            }
        )
    return pd.DataFrame(rows)


def _summarize_by_trial(
    windows: pd.DataFrame,
    trials: pd.DataFrame,
    *,
    fps: float,
    pre_seconds: float,
    post_seconds: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, trial in trials.iterrows():
        t0 = float(trial["start_camera_frame_id"])
        t1 = float(trial["end_camera_frame_id"])
        midpoint = windows["window_mid_camera_frame_id"].astype(float)
        pre = windows[(midpoint >= t0 - pre_seconds * fps) & (midpoint < t0)]
        during = windows[windows["overlap_trial_ids"].astype(str).str.split(";").apply(lambda ids: str(int(trial["trial_id"])) in ids)]
        post = windows[(midpoint >= t1) & (midpoint < t1 + post_seconds * fps)]
        row = dict(trial)
        for label, group in (("pre", pre), ("during", during), ("post", post)):
            row[f"{label}_window_count"] = int(len(group))
            row[f"{label}_peak_bpm_median"] = float(np.nanmedian(group["peak_bpm"])) if len(group) else math.nan
            row[f"{label}_peak_score_median"] = float(np.nanmedian(group["peak_score"])) if len(group) else math.nan
        row["during_minus_pre_bpm"] = row["during_peak_bpm_median"] - row["pre_peak_bpm_median"]
        row["post_minus_pre_bpm"] = row["post_peak_bpm_median"] - row["pre_peak_bpm_median"]
        rows.append(row)
    return pd.DataFrame(rows)


def _write_plot(path: Path, *, windows: pd.DataFrame, trials: pd.DataFrame, fps: float) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    ax.plot(
        windows["window_mid_recording_s"].astype(float),
        windows["peak_bpm"].astype(float),
        marker="o",
        ms=2.5,
        lw=0.9,
        color="#245c8a",
        label="HR estimate",
    )
    first_camera_frame = float(windows["window_start_camera_frame_id"].min())
    for idx, (_, trial) in enumerate(trials.iterrows()):
        x0 = (float(trial["start_camera_frame_id"]) - first_camera_frame) / float(fps)
        x1 = (float(trial["end_camera_frame_id"]) - first_camera_frame) / float(fps)
        ax.axvspan(x0, x1, color="#d95f02", alpha=0.22, label="chase trial" if idx == 0 else None)
    ax.set_title("Autocorrelation HR estimate aligned to chase trials")
    ax.set_xlabel("recording time (s)")
    ax.set_ylabel("peak bpm")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ensure_output_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align windowed HR estimates to GoodCopBadCop chase trials.")
    parser.add_argument("--hr-csv", type=Path, required=True)
    parser.add_argument("--strategy", default="eye_excluded")
    parser.add_argument("--crop-meta-csv", type=Path, required=True)
    parser.add_argument("--stimulus-h5", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--pre-seconds", type=float, default=20.0)
    parser.add_argument("--post-seconds", type=float, default=30.0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/hr_chase_alignment"),
    )
    args = parser.parse_args()

    trials = _load_chase_trials(args.stimulus_h5)
    hr = _read_hr_windows(args.hr_csv, strategy=str(args.strategy))
    mapped = _add_camera_frame_mapping(hr, crop_meta_csv=args.crop_meta_csv, fps=float(args.fps))
    windows = _classify_windows(
        mapped,
        trials,
        fps=float(args.fps),
        pre_seconds=float(args.pre_seconds),
        post_seconds=float(args.post_seconds),
    )
    phase_summary = _summarize_by_phase(windows)
    trial_summary = _summarize_by_trial(
        windows,
        trials,
        fps=float(args.fps),
        pre_seconds=float(args.pre_seconds),
        post_seconds=float(args.post_seconds),
    )

    output_prefix = args.output_prefix
    ensure_output_dir(output_prefix.parent)
    windows_csv = output_prefix.with_suffix(".windows.csv")
    phase_csv = output_prefix.with_suffix(".phase_summary.csv")
    trials_csv = output_prefix.with_suffix(".trial_summary.csv")
    plot_png = output_prefix.with_suffix(".png")
    summary_json = output_prefix.with_suffix(".summary.json")
    windows.to_csv(windows_csv, index=False)
    phase_summary.to_csv(phase_csv, index=False)
    trial_summary.to_csv(trials_csv, index=False)
    _write_plot(plot_png, windows=windows, trials=trials, fps=float(args.fps))
    summary = {
        "hr_csv": str(args.hr_csv),
        "strategy": str(args.strategy),
        "crop_meta_csv": str(args.crop_meta_csv),
        "stimulus_h5": str(args.stimulus_h5),
        "fps": float(args.fps),
        "pre_seconds": float(args.pre_seconds),
        "post_seconds": float(args.post_seconds),
        "trial_count": int(len(trials)),
        "window_count": int(len(windows)),
        "outputs": {
            "windows_csv": str(windows_csv),
            "phase_summary_csv": str(phase_csv),
            "trial_summary_csv": str(trials_csv),
            "plot_png": str(plot_png),
        },
    }
    with summary_json.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"summary_json: {summary_json}")
    print(f"windows_csv: {windows_csv}")
    print(f"phase_summary_csv: {phase_csv}")
    print(f"trial_summary_csv: {trials_csv}")
    print(f"plot_png: {plot_png}")
    print(phase_summary.to_string(index=False))


if __name__ == "__main__":
    main()
