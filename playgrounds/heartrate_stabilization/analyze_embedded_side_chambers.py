from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.signal import detrend

from analyze_embedded_positive_control import (
    _bandpass,
    _reference_at_times,
    analyze_candidate,
    read_numeric_xlsx_row,
)


_LABELS = ("chamber_a", "chamber_b")


def chamber_masks_from_json(path: Path) -> tuple[Path, dict[str, np.ndarray], int]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_id") != "palette.playground.fixed_video_chambers.v1":
        raise ValueError("unsupported chamber annotation schema")
    height, width = (int(value) for value in payload["frame_shape_hw"])
    masks: dict[str, np.ndarray] = {}
    for label in _LABELS:
        polygon = np.asarray(payload["chambers"][label]["polygon_xy"], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.reshape(-1, 1, 2)], 1)
        masks[label] = mask.astype(bool)
    return Path(payload["source_video"]), masks, int(payload["frame_index"])


def mask_variants(mask: np.ndarray, *, radius_px: int) -> dict[str, np.ndarray]:
    source = np.asarray(mask, dtype=bool)
    if radius_px < 1:
        raise ValueError("sensitivity radius must be positive")
    structure = ndimage.generate_binary_structure(2, 2)
    eroded = ndimage.binary_erosion(source, structure=structure, iterations=int(radius_px))
    dilated = ndimage.binary_dilation(source, structure=structure, iterations=int(radius_px))
    if int(np.count_nonzero(eroded)) < 3:
        raise ValueError("erosion removed nearly all chamber pixels")
    return {"base": source, f"erode_{radius_px}px": eroded, f"dilate_{radius_px}px": dilated}


def extract_mask_means(video: Path, masks: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    traces = {name: [] for name in masks}
    preview: np.ndarray | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if preview is None:
            preview = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for name, mask in masks.items():
            traces[name].append(float(np.mean(gray[mask])))
    capture.release()
    if preview is None or len(next(iter(traces.values()))) < 3:
        raise ValueError("video yielded too few frames")
    return {name: np.asarray(values) for name, values in traces.items()}, fps, preview


def window_lag_summary(
    first: np.ndarray,
    second: np.ndarray,
    *,
    fps: float,
    window_seconds: float = 8.0,
    step_seconds: float = 2.0,
    max_lag_seconds: float = 0.25,
) -> dict[str, object]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    window = int(round(window_seconds * fps))
    step = int(round(step_seconds * fps))
    max_lag = int(round(max_lag_seconds * fps))
    times: list[float] = []
    lags: list[float] = []
    correlations: list[float] = []
    for start in range(0, left.size - window + 1, step):
        a = left[start : start + window]
        b = right[start : start + window]
        options: list[tuple[float, float, int]] = []
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                x, y = a[:-lag], b[lag:]
            elif lag < 0:
                x, y = a[-lag:], b[:lag]
            else:
                x, y = a, b
            correlation = float(np.corrcoef(x, y)[0, 1])
            options.append((abs(correlation), correlation, lag))
        _absolute, correlation, lag = max(options, key=lambda item: item[0])
        times.append((start + (window - 1) / 2.0) / fps)
        lags.append(lag / fps)
        correlations.append(correlation)
    lag_array = np.asarray(lags)
    correlation_array = np.asarray(correlations)
    return {
        "window_center_s": np.asarray(times),
        "lag_s": lag_array,
        "correlation": correlation_array,
        "lag_convention": "positive means chamber_b follows chamber_a",
        "median_lag_ms": float(1000.0 * np.median(lag_array)),
        "lag_iqr_ms": [
            float(1000.0 * np.quantile(lag_array, 0.25)),
            float(1000.0 * np.quantile(lag_array, 0.75)),
        ],
        "median_absolute_correlation": float(np.median(np.abs(correlation_array))),
        "median_signed_correlation": float(np.median(correlation_array)),
        "negative_correlation_fraction": float(np.mean(correlation_array < 0.0)),
        "positive_lag_fraction": float(np.mean(lag_array > 0.0)),
        "negative_lag_fraction": float(np.mean(lag_array < 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two fixed side-camera chamber polygons.")
    parser.add_argument("--chambers-json", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, default=1)
    parser.add_argument("--reference-fps", type=float, default=100.0)
    parser.add_argument("--band-hz", type=float, nargs=2, default=(1.5, 4.0))
    parser.add_argument("--sensitivity-radius-px", type=int, default=2)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    video, base_masks, annotation_frame = chamber_masks_from_json(args.chambers_json)
    variants: dict[str, np.ndarray] = {}
    for label in _LABELS:
        for variant, mask in mask_variants(
            base_masks[label], radius_px=int(args.sensitivity_radius_px)
        ).items():
            variants[f"{label}_{variant}"] = mask
    variants["chamber_union_base"] = base_masks["chamber_a"] | base_masks["chamber_b"]
    raw_traces, fps, first_frame = extract_mask_means(video, variants)
    band_hz = (float(args.band_hz[0]), float(args.band_hz[1]))
    filtered = {
        name: _bandpass(detrend(trace), fps=fps, band_hz=band_hz)
        for name, trace in raw_traces.items()
    }
    reference_hz = read_numeric_xlsx_row(
        args.workbook,
        sheet_name="heart_rate_trace",
        row_number=int(args.trial_number) + 1,
    )
    summaries = {}
    details = {}
    for name, trace in filtered.items():
        summary, detail = analyze_candidate(
            name,
            trace,
            fps=fps,
            band_hz=band_hz,
            reference_hz=reference_hz,
            reference_fps=float(args.reference_fps),
        )
        summaries[name] = asdict(summary)
        details[name] = detail
    lag = window_lag_summary(
        filtered["chamber_a_base"], filtered["chamber_b_base"], fps=fps
    )

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = prefix.with_suffix(".arrays.npz")
    plot_path = prefix.with_suffix(".diagnostic.png")
    summary_path = prefix.with_suffix(".summary.json")
    np.savez_compressed(
        arrays_path,
        **{f"mask_{name}": mask for name, mask in variants.items()},
        **{f"raw_{name}": trace for name, trace in raw_traces.items()},
        **{f"filtered_{name}": trace for name, trace in filtered.items()},
        lag_window_center_s=lag["window_center_s"],
        lag_s=lag["lag_s"],
        lag_correlation=lag["correlation"],
    )

    colors = {"chamber_a_base": "tab:red", "chamber_b_base": "tab:blue", "chamber_union_base": "tab:purple"}
    time = np.arange(len(filtered["chamber_a_base"])) / fps
    reference_time = np.arange(reference_hz.size) / float(args.reference_fps)
    preview = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    overlay = preview.copy()
    for label, color in zip(_LABELS, ((255, 40, 60), (40, 120, 255))):
        overlay[base_masks[label]] = color
    preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0.0)
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    axes[0, 0].imshow(preview); axes[0, 0].set(title="Fixed chamber polygons"); axes[0, 0].axis("off")
    for name in colors:
        scale = max(1.4826 * np.median(np.abs(filtered[name] - np.median(filtered[name]))), 1e-9)
        axes[0, 1].plot(time, filtered[name] / scale, color=colors[name], label=name)
    axes[0, 1].set(xlim=(0, 10), title="Bandpassed chamber means, first 10 s", xlabel="Time (s)"); axes[0, 1].legend(fontsize=8)
    axes[0, 2].plot(lag["window_center_s"], 1000.0 * lag["lag_s"], marker="o")
    axes[0, 2].axhline(0, color="black", linewidth=1); axes[0, 2].set(title="B relative to A lag", xlabel="Time (s)", ylabel="Lag (ms)")
    axes[1, 0].plot(reference_time, reference_hz * 60.0, color="black", label="reference")
    for name in colors:
        axes[1, 0].plot(details[name]["ridge_time_s"], details[name]["ridge_hz"] * 60.0, color=colors[name], label=name)
    axes[1, 0].set(title="8 s rate ridge", xlabel="Time (s)", ylabel="Rate (bpm)"); axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(reference_time, reference_hz * 60.0, color="black", label="reference")
    for name in colors:
        axes[1, 1].scatter(details[name]["event_time_s"], details[name]["event_hz"] * 60.0, s=8, color=colors[name], label=name)
    axes[1, 1].set(title="Peak-interval rates", xlabel="Time (s)", ylabel="Rate (bpm)"); axes[1, 1].legend(fontsize=8)
    sensitivity_names = [name for name in summaries if name != "chamber_union_base"]
    axes[1, 2].barh(sensitivity_names, [summaries[name]["ridge_mae_bpm"] for name in sensitivity_names])
    axes[1, 2].set(title="Boundary sensitivity", xlabel="Ridge MAE (bpm)")
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    lag_json = {key: value for key, value in lag.items() if not isinstance(value, np.ndarray)}
    payload = {
        "analysis_status": "exploratory_fixed_side_chamber_photometry",
        "source_video": str(video.resolve()),
        "chambers_json": str(args.chambers_json.resolve()),
        "workbook": str(args.workbook.resolve()),
        "trial_number": int(args.trial_number),
        "annotation_frame_index": annotation_frame,
        "anatomical_identity_status": "unassigned_chamber_a_chamber_b",
        "source_fps": fps,
        "source_frames": int(len(time)),
        "band_hz": list(band_hz),
        "sensitivity_radius_px": int(args.sensitivity_radius_px),
        "mask_pixel_counts": {name: int(np.count_nonzero(mask)) for name, mask in variants.items()},
        "base_mask_overlap_pixels": int(np.count_nonzero(base_masks["chamber_a"] & base_masks["chamber_b"])),
        "summaries": summaries,
        "chamber_lag": lag_json,
        "arrays_npz": str(arrays_path),
        "diagnostic_png": str(plot_path),
        "interpretation_guard": (
            "fixed polygons and same-clip event fitting; not anatomical segmentation or beat ground truth"
        ),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
