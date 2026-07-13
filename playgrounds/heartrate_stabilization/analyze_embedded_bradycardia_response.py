from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

from analyze_embedded_positive_control import (
    _SHEET_NS,
    _column_index,
    _reference_at_times,
    _worksheet_path,
    read_numeric_xlsx_row,
)
from evaluate_embedded_rate_window_sweep import evaluate_window


def read_xlsx_row_values(
    workbook_path: Path,
    *,
    sheet_name: str,
    row_number: int,
) -> dict[int, str | float]:
    with zipfile.ZipFile(workbook_path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.iter(f"{{{_SHEET_NS}}}t"))
                for item in root.findall(f"{{{_SHEET_NS}}}si")
            ]
        worksheet = ET.fromstring(archive.read(_worksheet_path(archive, sheet_name)))
    row = next(
        (
            item
            for item in worksheet.findall(f".//{{{_SHEET_NS}}}row")
            if int(item.get("r", "-1")) == int(row_number)
        ),
        None,
    )
    if row is None:
        raise ValueError(f"sheet {sheet_name!r} has no row {row_number}")
    values: dict[int, str | float] = {}
    for cell in row.findall(f"{{{_SHEET_NS}}}c"):
        column = _column_index(str(cell.get("r")))
        value = cell.find(f"{{{_SHEET_NS}}}v")
        if cell.get("t") == "s" and value is not None and value.text is not None:
            values[column] = shared[int(value.text)]
        elif cell.get("t") == "inlineStr":
            values[column] = "".join(
                node.text or "" for node in cell.iter(f"{{{_SHEET_NS}}}t")
            )
        elif value is not None and value.text is not None:
            values[column] = float(value.text)
    return values


def threshold_spans(
    times_s: np.ndarray,
    values_bpm: np.ndarray,
    *,
    start_s: float,
    stop_s: float,
    threshold_bpm: float,
) -> list[tuple[float, float, float]]:
    times = np.asarray(times_s, dtype=np.float64)
    values = np.asarray(values_bpm, dtype=np.float64)
    selected = (
        (times >= float(start_s))
        & (times < float(stop_s))
        & np.isfinite(values)
        & (values <= float(threshold_bpm))
    )
    indices = np.flatnonzero(selected)
    if not indices.size:
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, boundaries)
    sample_step = float(np.median(np.diff(times))) if times.size >= 2 else 0.0
    return [
        (
            float(times[run[0]]),
            min(float(stop_s), float(times[run[-1]] + sample_step)),
            float(np.nanmin(values[run])),
        )
        for run in runs
    ]


def response_metrics(
    times_s: np.ndarray,
    values_hz: np.ndarray,
    *,
    baseline_s: tuple[float, float],
    stimulus_s: tuple[float, float],
) -> dict[str, float]:
    times = np.asarray(times_s, dtype=np.float64)
    values = np.asarray(values_hz, dtype=np.float64) * 60.0
    baseline = (
        (times >= baseline_s[0])
        & (times < baseline_s[1])
        & np.isfinite(values)
    )
    stimulus = (
        (times >= stimulus_s[0])
        & (times < stimulus_s[1])
        & np.isfinite(values)
    )
    if not np.any(baseline) or not np.any(stimulus):
        raise ValueError("baseline or stimulus response interval has no finite samples")
    baseline_bpm = float(np.median(values[baseline]))
    stimulus_indices = np.flatnonzero(stimulus)
    nadir_index = int(stimulus_indices[np.argmin(values[stimulus_indices])])
    nadir_bpm = float(values[nadir_index])
    return {
        "baseline_bpm": baseline_bpm,
        "nadir_bpm": nadir_bpm,
        "drop_bpm": baseline_bpm - nadir_bpm,
        "nadir_time_s": float(times[nadir_index]),
        "nadir_latency_s": float(times[nadir_index] - stimulus_s[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen top-camera projection for a stimulus-evoked bradycardia."
    )
    parser.add_argument("--projection-arrays", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, default=1)
    parser.add_argument("--method", default="masked_equal_mean")
    parser.add_argument("--fps", type=float, default=200.0)
    parser.add_argument("--reference-fps", type=float, default=100.0)
    parser.add_argument("--band-hz", type=float, nargs=2, default=(1.5, 4.0))
    parser.add_argument("--ridge-window-seconds", type=float, default=2.0)
    parser.add_argument("--ridge-step-seconds", type=float, default=1.0)
    parser.add_argument("--baseline-s", type=float, nargs=2, default=(28.0, 32.0))
    parser.add_argument("--stimulus-s", type=float, nargs=2, default=(32.0, 36.0))
    parser.add_argument("--drop-threshold-bpm", type=float, default=30.0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    arrays = np.load(args.projection_arrays)
    trace = np.asarray(arrays[f"trace_{args.method}"], dtype=np.float64)
    event_rate_hz = np.asarray(arrays[f"event_rate_hz_{args.method}"], dtype=np.float64)
    frame_time_s = np.arange(trace.size, dtype=np.float64) / float(args.fps)
    reference_hz = read_numeric_xlsx_row(
        args.workbook,
        sheet_name="heart_rate_trace",
        row_number=int(args.trial_number) + 1,
    )
    reference_frame_hz = _reference_at_times(
        reference_hz, frame_time_s, float(args.reference_fps)
    )
    _ridge_summary, ridge_time_s, ridge_hz, _ridge_reference = evaluate_window(
        trace,
        reference_hz,
        fps=float(args.fps),
        reference_fps=float(args.reference_fps),
        window_seconds=float(args.ridge_window_seconds),
        step_seconds=float(args.ridge_step_seconds),
        band_hz=(float(args.band_hz[0]), float(args.band_hz[1])),
    )
    baseline_s = (float(args.baseline_s[0]), float(args.baseline_s[1]))
    stimulus_s = (float(args.stimulus_s[0]), float(args.stimulus_s[1]))
    metrics = {
        "side_reference": response_metrics(
            frame_time_s, reference_frame_hz, baseline_s=baseline_s, stimulus_s=stimulus_s
        ),
        "top_masked_mean_event_interval": response_metrics(
            frame_time_s, event_rate_hz, baseline_s=baseline_s, stimulus_s=stimulus_s
        ),
        "top_masked_mean_spectral_ridge": response_metrics(
            ridge_time_s, ridge_hz, baseline_s=baseline_s, stimulus_s=stimulus_s
        ),
    }
    reference_drop = metrics["side_reference"]["drop_bpm"]
    for label in ("top_masked_mean_event_interval", "top_masked_mean_spectral_ridge"):
        metrics[label]["reference_depth_fraction"] = (
            metrics[label]["drop_bpm"] / reference_drop
        )

    brady_row = read_xlsx_row_values(
        args.workbook,
        sheet_name="Bradyinfo",
        row_number=int(args.trial_number) + 1,
    )
    brady_bouts_frames = ast.literal_eval(str(brady_row[4]))
    brady_bouts_s = [
        (float(start) / float(args.reference_fps), float(stop) / float(args.reference_fps))
        for start, stop in brady_bouts_frames
    ]
    stimulus_brady_bouts_s = [
        [start, stop]
        for start, stop in brady_bouts_s
        if stop > stimulus_s[0] and start < stimulus_s[1]
    ]

    event_metrics = metrics["top_masked_mean_event_interval"]
    threshold_bpm = event_metrics["baseline_bpm"] - float(args.drop_threshold_bpm)
    detected_spans = threshold_spans(
        frame_time_s,
        event_rate_hz * 60.0,
        start_s=stimulus_s[0],
        stop_s=stimulus_s[1],
        threshold_bpm=threshold_bpm,
    )
    primary_span = min(detected_spans, key=lambda item: item[2]) if detected_spans else None
    timing_evaluation: dict[str, float] | None = None
    if primary_span is not None and stimulus_brady_bouts_s:
        reference_span = stimulus_brady_bouts_s[0]
        intersection = max(
            0.0,
            min(primary_span[1], reference_span[1])
            - max(primary_span[0], reference_span[0]),
        )
        union = max(primary_span[1], reference_span[1]) - min(primary_span[0], reference_span[0])
        timing_evaluation = {
            "onset_error_s": float(primary_span[0] - reference_span[0]),
            "stop_error_s": float(primary_span[1] - reference_span[1]),
            "interval_iou": float(intersection / union) if union > 0 else float("nan"),
        }

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_path = prefix.with_suffix(".png")
    summary_path = prefix.with_suffix(".summary.json")
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(frame_time_s, reference_frame_hz * 60.0, color="black", linewidth=2, label="side reference")
    axes[0].plot(ridge_time_s, ridge_hz * 60.0, color="tab:orange", marker="o", label="top 2 s ridge")
    axes[0].plot(frame_time_s, event_rate_hz * 60.0, color="tab:blue", linewidth=1.2, label="top peak interval")
    axes[0].axvspan(*stimulus_s, color="orchid", alpha=0.18, label="stimulus epoch")
    for index, (start, stop) in enumerate(stimulus_brady_bouts_s):
        axes[0].axvspan(start, stop, color="red", alpha=0.20, label="supplied brady bout" if index == 0 else None)
    if primary_span is not None:
        axes[0].axvspan(primary_span[0], primary_span[1], color="tab:blue", alpha=0.18, label="top detected response")
    axes[0].set(xlim=(baseline_s[0], 40.0), ylabel="Rate (bpm)", title="Stimulus-locked bradycardia recovery")
    axes[0].legend(ncol=3, fontsize=8)
    finite_trace = trace[np.isfinite(trace)]
    scale = max(1.4826 * float(np.median(np.abs(finite_trace - np.median(finite_trace)))), 1e-9)
    axes[1].plot(frame_time_s, trace / scale, color="tab:orange", linewidth=1)
    axes[1].axvspan(*stimulus_s, color="orchid", alpha=0.18)
    for start, stop in stimulus_brady_bouts_s:
        axes[1].axvspan(start, stop, color="red", alpha=0.20)
    axes[1].set(xlim=(baseline_s[0], 40.0), xlabel="Source time (s)", ylabel="Robust scale", title="Held-out top equal-mask mean")
    for axis in axes:
        axis.grid(True, alpha=0.2)
    figure.savefig(plot_path, dpi=170)
    plt.close(figure)

    payload = {
        "analysis_status": "descriptive_stimulus_locked_bradycardia_evaluation",
        "projection_arrays": str(args.projection_arrays.resolve()),
        "workbook": str(args.workbook.resolve()),
        "trial_number": int(args.trial_number),
        "method": str(args.method),
        "baseline_s": list(baseline_s),
        "stimulus_s": list(stimulus_s),
        "stimulus_timing_source": "declared CLI interval matching the supplied Trial 1 highlighted stimulus epoch",
        "brady_bouts_frames_reference_timebase": brady_bouts_frames,
        "stimulus_brady_bouts_s": stimulus_brady_bouts_s,
        "drop_threshold_bpm": float(args.drop_threshold_bpm),
        "top_event_threshold_bpm": threshold_bpm,
        "top_detected_response_spans": [list(item) for item in detected_spans],
        "top_primary_response_span": list(primary_span) if primary_span is not None else None,
        "timing_evaluation": timing_evaluation,
        "metrics": metrics,
        "interpretation": "stimulus-locked rate response; top peak intervals remain candidate events",
        "plot": str(plot_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
