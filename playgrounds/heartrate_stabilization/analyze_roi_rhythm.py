from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from _common import ensure_output_dir


def _load_roi_rect(path: Path) -> tuple[float, float, float, float]:
    with path.open() as handle:
        payload = json.load(handle)
    values = payload.get("roi_rect_stable_xywh")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"ROI JSON does not contain roi_rect_stable_xywh[4]: {path}")
    x, y, w, h = (float(value) for value in values)
    if not all(np.isfinite([x, y, w, h])) or w <= 0 or h <= 0:
        raise ValueError(f"Invalid ROI rectangle in {path}: {values}")
    return x, y, w, h


def _read_status_csv(path: Path) -> dict[int, bool]:
    status: dict[int, bool] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "output_frame_index" not in (reader.fieldnames or []):
            raise ValueError(f"status CSV lacks output_frame_index: {path}")
        for row in reader:
            idx = int(row["output_frame_index"])
            status[idx] = str(row.get("valid", "1")).strip() not in {"", "0", "false", "False"}
    return status


def _sample_video_roi(
    *,
    video_path: Path,
    roi_json: Path,
    status_csv: Path | None,
    output_csv: Path,
    frame_start: int,
    frame_count: int | None,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    x, y, w, h = _load_roi_rect(roi_json)
    x0 = max(0, int(math.floor(x)))
    y0 = max(0, int(math.floor(y)))
    x1 = int(math.ceil(x + w))
    y1 = int(math.ceil(y + h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI is empty after integer clipping: {(x, y, w, h)}")

    status = _read_status_csv(status_csv) if status_csv is not None else {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count is None:
        frame_stop = total_frames if total_frames > 0 else None
    else:
        frame_stop = frame_start + max(0, int(frame_count))

    ensure_output_dir(output_csv.parent)
    frame_indices: list[int] = []
    intensities: list[float] = []
    valid_values: list[bool] = []
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "output_frame_index",
                "valid",
                "mean_intensity",
                "roi_x0",
                "roi_y0",
                "roi_x1",
                "roi_y1",
            ],
        )
        writer.writeheader()
        next_expected: int | None = None
        frame_index = int(frame_start)
        while frame_stop is None or frame_index < frame_stop:
            if next_expected is None or frame_index != next_expected:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            next_expected = frame_index + 1
            if not ok:
                break

            height, width = frame.shape[:2]
            roi = frame[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1), :3]
            status_valid = status.get(frame_index, True)
            valid = bool(status_valid and roi.size)
            mean = float(np.mean(roi)) if valid else math.nan
            writer.writerow(
                {
                    "output_frame_index": frame_index,
                    "valid": int(valid),
                    "mean_intensity": mean,
                    "roi_x0": max(0, x0),
                    "roi_y0": max(0, y0),
                    "roi_x1": min(width, x1),
                    "roi_y1": min(height, y1),
                }
            )
            frame_indices.append(frame_index)
            intensities.append(mean)
            valid_values.append(valid)

            frame_index += max(1, int(stride))
    capture.release()

    return (
        np.asarray(frame_indices, dtype=np.int64),
        np.asarray(intensities, dtype=np.float64),
        np.asarray(valid_values, dtype=bool),
    )


def _load_signal_csv(path: Path, *, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    table = pd.read_csv(path)
    if column not in table.columns:
        raise ValueError(f"{path} lacks signal column {column!r}; columns={list(table.columns)}")
    if "output_frame_index" in table.columns:
        frame_index = table["output_frame_index"].to_numpy(dtype=np.int64)
    elif "crop_video_frame_index" in table.columns:
        frame_index = table["crop_video_frame_index"].to_numpy(dtype=np.int64)
    else:
        frame_index = np.arange(len(table), dtype=np.int64)
    signal = table[column].to_numpy(dtype=np.float64)
    if "valid" in table.columns:
        valid = table["valid"].astype(bool).to_numpy()
    else:
        valid = np.ones(len(table), dtype=bool)
    valid &= np.isfinite(signal)
    return frame_index, signal, valid


def _continuous_segment_with_interpolated_gaps(
    frame_index: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    max_interpolated_gap_samples: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(frame_index) == 0:
        raise ValueError("No samples were loaded.")
    finite = valid & np.isfinite(values)
    if int(np.count_nonzero(finite)) < 8:
        raise ValueError("Fewer than 8 finite valid samples are available.")

    first_valid = int(np.flatnonzero(finite)[0])
    last_valid = int(np.flatnonzero(finite)[-1])
    max_gap = max(0, int(max_interpolated_gap_samples))

    candidate_start = first_valid
    best_start = first_valid
    best_stop = first_valid + 1
    idx = first_valid
    while idx <= last_valid:
        if finite[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx <= last_valid and not finite[idx]:
            idx += 1
        gap_stop = idx
        if gap_stop - gap_start > max_gap:
            if gap_start - candidate_start > best_stop - best_start:
                best_start, best_stop = candidate_start, gap_start
            candidate_start = gap_stop
    if last_valid + 1 - candidate_start > best_stop - best_start:
        best_start, best_stop = candidate_start, last_valid + 1

    segment_frames = frame_index[best_start:best_stop]
    segment_values = values[best_start:best_stop].astype(np.float64, copy=True)
    segment_valid = finite[best_start:best_stop]
    interpolated_count = int(np.count_nonzero(~segment_valid))
    if interpolated_count:
        x = np.arange(len(segment_values), dtype=np.float64)
        valid_x = x[segment_valid]
        segment_values[~segment_valid] = np.interp(x[~segment_valid], valid_x, segment_values[segment_valid])
    return segment_frames, segment_values, interpolated_count


def _analyze_trace(
    *,
    frame_index: np.ndarray,
    values: np.ndarray,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
) -> dict[str, Any]:
    from scipy import signal

    if len(values) < 16:
        raise ValueError("Need at least 16 contiguous valid samples for rhythm analysis.")
    sample_rate = float(fps)
    if sample_rate <= 0 or not np.isfinite(sample_rate):
        raise ValueError(f"Invalid fps: {fps}")
    nyquist = sample_rate / 2.0
    band_min_hz = max(0.001, float(band_min_hz))
    band_max_hz = min(float(band_max_hz), nyquist * 0.98)
    if band_min_hz >= band_max_hz:
        raise ValueError(f"Invalid search band after Nyquist clamp: {band_min_hz}..{band_max_hz} Hz")

    raw = values.astype(np.float64)
    raw_centered = raw - np.nanmedian(raw)
    detrended = signal.detrend(raw_centered, type="linear")

    filtered = detrended
    filter_applied = False
    padlen = 3 * 2 * 3
    if len(detrended) > padlen + 4:
        sos = signal.butter(3, [band_min_hz, band_max_hz], btype="bandpass", fs=sample_rate, output="sos")
        filtered = signal.sosfiltfilt(sos, detrended)
        filter_applied = True

    nperseg = min(len(detrended), max(64, int(round(sample_rate * 8))))
    if nperseg > len(detrended):
        nperseg = len(detrended)
    frequencies, psd = signal.welch(
        detrended,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2 if nperseg >= 4 else 0,
        detrend="constant",
        scaling="density",
    )
    band = (frequencies >= band_min_hz) & (frequencies <= band_max_hz)
    if int(np.count_nonzero(band)) == 0:
        raise ValueError("No PSD bins fell inside the requested search band.")
    band_freq = frequencies[band]
    band_psd = psd[band]
    peak_idx = int(np.argmax(band_psd))
    peak_hz = float(band_freq[peak_idx])
    peak_power = float(band_psd[peak_idx])
    median_power = float(np.median(band_psd)) if len(band_psd) else math.nan
    snr = float(peak_power / median_power) if median_power > 0 else math.inf

    peaks, _ = signal.find_peaks(band_psd)
    if len(peaks) == 0:
        ranked = [peak_idx]
    else:
        ranked = sorted((int(idx) for idx in peaks), key=lambda idx: band_psd[idx], reverse=True)
        if peak_idx not in ranked:
            ranked.insert(0, peak_idx)
    top_peaks = [
        {
            "frequency_hz": float(band_freq[idx]),
            "bpm": float(band_freq[idx] * 60.0),
            "power": float(band_psd[idx]),
        }
        for idx in ranked[:8]
    ]

    return {
        "frame_index": frame_index,
        "raw_values": raw,
        "detrended": detrended,
        "filtered": filtered,
        "frequencies": frequencies,
        "psd": psd,
        "summary": {
            "sample_count": int(len(raw)),
            "frame_start": int(frame_index[0]),
            "frame_stop_inclusive": int(frame_index[-1]),
            "duration_s": float((len(raw) - 1) / sample_rate),
            "fps": sample_rate,
            "search_band_hz": [float(band_min_hz), float(band_max_hz)],
            "peak_frequency_hz": peak_hz,
            "peak_bpm": float(peak_hz * 60.0),
            "peak_power": peak_power,
            "median_band_power": median_power,
            "peak_to_median_band_power": snr,
            "filter_applied": bool(filter_applied),
            "top_peaks": top_peaks,
            "raw_mean": float(np.mean(raw)),
            "raw_std": float(np.std(raw)),
            "filtered_std": float(np.std(filtered)),
        },
    }


def _write_plot(result: dict[str, Any], *, output_png: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = result["summary"]
    fps = float(summary["fps"])
    frame_index = result["frame_index"]
    t = (frame_index - frame_index[0]).astype(np.float64) / fps
    frequencies = result["frequencies"]
    psd = result["psd"]
    band_min, band_max = summary["search_band_hz"]
    peak_hz = summary["peak_frequency_hz"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].plot(t, result["raw_values"], lw=0.8, color="#2f5d7c")
    axes[0].set_title("ROI mean intensity")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("uint8 mean")

    axes[1].plot(t, result["filtered"], lw=0.8, color="#8a3ffc")
    axes[1].set_title("Band-passed / detrended ROI signal")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("intensity delta")

    axes[2].semilogy(frequencies, np.maximum(psd, np.finfo(float).tiny), lw=1.0, color="#2d7f5e")
    axes[2].axvspan(band_min, band_max, color="#d8d8d8", alpha=0.3)
    axes[2].axvline(peak_hz, color="#d62728", lw=1.2)
    axes[2].set_title(f"Welch PSD, peak {peak_hz:.3f} Hz / {peak_hz * 60.0:.1f} bpm")
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("power")
    axes[2].set_xlim(0, min(max(12.0, band_max * 1.25), fps / 2.0))

    ensure_output_dir(output_png.parent)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze rhythmic intensity changes in a stabilized ROI.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Existing ROI signal CSV.")
    source.add_argument("--video", type=Path, help="Stabilized video to sample directly.")
    parser.add_argument("--roi-json", type=Path, help="ROI JSON; required with --video.")
    parser.add_argument("--status-csv", type=Path, help="Optional stabilized-video status CSV for valid-frame filtering.")
    parser.add_argument("--sample-output", type=Path, help="CSV to write when sampling --video.")
    parser.add_argument("--column", type=str, default="mean_intensity", help="Signal column for --csv.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--band-min-hz", type=float, default=0.5)
    parser.add_argument("--band-max-hz", type=float, default=10.0)
    parser.add_argument(
        "--max-interpolated-gap-samples",
        type=int,
        default=5,
        help="Interpolate across invalid gaps up to this many samples before PSD analysis.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/roi_rhythm_analysis"),
    )
    args = parser.parse_args()

    output_prefix = args.output_prefix
    ensure_output_dir(output_prefix.parent)

    if args.video is not None:
        if args.roi_json is None:
            raise ValueError("--roi-json is required with --video")
        sample_output = args.sample_output or output_prefix.with_suffix(".samples.csv")
        frame_index, values, valid = _sample_video_roi(
            video_path=args.video,
            roi_json=args.roi_json,
            status_csv=args.status_csv,
            output_csv=sample_output,
            frame_start=max(0, int(args.frame_start)),
            frame_count=args.frame_count,
            stride=max(1, int(args.stride)),
        )
        print(f"sample_csv: {sample_output}")
    else:
        frame_index, values, valid = _load_signal_csv(args.csv, column=args.column)

    segment_frames, segment_values, interpolated_count = _continuous_segment_with_interpolated_gaps(
        frame_index,
        values,
        valid,
        max_interpolated_gap_samples=int(args.max_interpolated_gap_samples),
    )
    result = _analyze_trace(
        frame_index=segment_frames,
        values=segment_values,
        fps=float(args.fps) / max(1, int(args.stride)),
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
    )

    summary = dict(result["summary"])
    summary["source_csv"] = str(args.csv) if args.csv is not None else None
    summary["source_video"] = str(args.video) if args.video is not None else None
    summary["roi_json"] = str(args.roi_json) if args.roi_json is not None else None
    summary["status_csv"] = str(args.status_csv) if args.status_csv is not None else None
    summary["total_loaded_samples"] = int(len(frame_index))
    summary["total_valid_samples"] = int(np.count_nonzero(valid & np.isfinite(values)))
    summary["analysis_samples_after_short_gap_interpolation"] = int(len(segment_values))
    summary["interpolated_gap_samples"] = int(interpolated_count)
    summary["max_interpolated_gap_samples"] = int(args.max_interpolated_gap_samples)

    output_json = output_prefix.with_suffix(".summary.json")
    output_png = output_prefix.with_suffix(".png")
    with output_json.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_plot(result, output_png=output_png)

    print(f"summary_json: {output_json}")
    print(f"plot_png: {output_png}")
    print(f"loaded_samples: {summary['total_loaded_samples']}")
    print(f"valid_samples: {summary['total_valid_samples']}")
    print(f"analysis_samples: {summary['sample_count']}")
    print(f"duration_s: {summary['duration_s']:.3f}")
    print(f"peak_hz: {summary['peak_frequency_hz']:.6g}")
    print(f"peak_bpm: {summary['peak_bpm']:.3f}")
    print(f"peak_to_median_band_power: {summary['peak_to_median_band_power']:.3f}")


if __name__ == "__main__":
    main()
