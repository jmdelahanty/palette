from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from _common import ensure_output_dir, load_config, resolve_roi_rect, roi_rect_corners
from map_pixel_band_contributions import (
    _bandpass_matrix,
    _draw_polygon,
    _load_roi_pixel_traces,
)
from visualize_roi_intensity_diagnostics import (
    _interpolate_short_gaps_all_segments,
    _mask_from_npz,
    _scatter_to_image,
    _zscore,
)


def _welch_pixel_scores(
    traces: np.ndarray,
    *,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
) -> dict[str, np.ndarray]:
    from scipy import signal

    if traces.shape[0] < 32:
        raise ValueError("Need at least 32 samples for per-pixel Welch scores.")
    centered = traces.astype(np.float64) - np.mean(traces, axis=0, keepdims=True)
    detrended = signal.detrend(centered, axis=0, type="linear")
    nperseg = min(detrended.shape[0], max(64, int(round(float(fps) * 8.0))))
    if nperseg < 16:
        raise ValueError(f"nperseg too small: {nperseg}")
    frequencies, psd = signal.welch(
        detrended,
        fs=float(fps),
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2 if nperseg >= 4 else 0,
        detrend="constant",
        axis=0,
        scaling="density",
    )
    band = (frequencies >= float(band_min_hz)) & (frequencies <= float(band_max_hz))
    if int(np.count_nonzero(band)) == 0:
        raise ValueError("No Welch bins fall inside the requested band.")
    broad = (frequencies >= max(0.05, float(band_min_hz) * 0.25)) & (
        frequencies <= min(float(fps) * 0.49, float(band_max_hz) * 2.5)
    )
    band_psd = psd[band, :]
    band_power = np.trapezoid(band_psd, frequencies[band], axis=0)
    broad_power = np.trapezoid(psd[broad, :], frequencies[broad], axis=0) if np.any(broad) else np.sum(psd, axis=0)
    band_fraction = np.divide(
        band_power,
        broad_power,
        out=np.zeros_like(band_power, dtype=np.float64),
        where=broad_power > np.finfo(float).eps,
    )
    band_indices = np.flatnonzero(band)
    peak_local = np.argmax(band_psd, axis=0)
    peak_indices = band_indices[peak_local]
    peak_frequency = frequencies[peak_indices]
    peak_power = psd[peak_indices, np.arange(psd.shape[1])]
    median_band_power = np.median(band_psd, axis=0)
    peak_to_median = np.divide(
        peak_power,
        median_band_power,
        out=np.zeros_like(peak_power, dtype=np.float64),
        where=median_band_power > np.finfo(float).eps,
    )
    return {
        "band_power": band_power.astype(np.float64),
        "band_fraction": band_fraction.astype(np.float64),
        "peak_frequency_hz": peak_frequency.astype(np.float64),
        "peak_power": peak_power.astype(np.float64),
        "peak_to_median": peak_to_median.astype(np.float64),
    }


def _iter_chunks(
    *,
    segments: list[tuple[int, int]],
    fps: float,
    chunk_seconds: float,
    step_seconds: float,
    min_samples: int,
) -> list[tuple[int, int, int]]:
    chunk_len = max(1, int(round(float(chunk_seconds) * float(fps))))
    step_len = max(1, int(round(float(step_seconds) * float(fps))))
    chunks: list[tuple[int, int, int]] = []
    chunk_index = 0
    for segment_start, segment_stop in segments:
        start = int(segment_start)
        while start + chunk_len <= int(segment_stop):
            stop = start + chunk_len
            if stop - start >= int(min_samples):
                chunks.append((chunk_index, start, stop))
                chunk_index += 1
            start += step_len
    return chunks


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float64)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return np.zeros_like(values, dtype=np.float64)
    out = (values.astype(np.float64) - median) / scale
    out[~np.isfinite(out)] = 0.0
    return out


def _aggregate_pixel_scores(
    chunk_scores: list[dict[str, np.ndarray]],
    *,
    top_fraction: float,
) -> dict[str, np.ndarray]:
    if not chunk_scores:
        raise ValueError("No chunk scores were computed.")
    band_power = np.stack([score["band_power"] for score in chunk_scores], axis=0)
    band_fraction = np.stack([score["band_fraction"] for score in chunk_scores], axis=0)
    peak_to_median = np.stack([score["peak_to_median"] for score in chunk_scores], axis=0)
    peak_frequency = np.stack([score["peak_frequency_hz"] for score in chunk_scores], axis=0)
    power_z = np.stack([_robust_zscore(row) for row in band_power], axis=0)
    fraction_z = np.stack([_robust_zscore(row) for row in band_fraction], axis=0)
    peak_z = np.stack([_robust_zscore(row) for row in peak_to_median], axis=0)
    score = power_z + 0.5 * fraction_z + 0.25 * peak_z
    top_k = max(1, int(math.ceil(score.shape[1] * float(top_fraction))))
    top_counts = np.zeros(score.shape[1], dtype=np.int64)
    for row in score:
        order = np.argsort(row)[::-1]
        top_counts[order[:top_k]] += 1
    return {
        "selection_score": np.median(score, axis=0),
        "mean_selection_score": np.mean(score, axis=0),
        "median_band_power": np.median(band_power, axis=0),
        "median_band_fraction": np.median(band_fraction, axis=0),
        "median_peak_to_median": np.median(peak_to_median, axis=0),
        "median_peak_frequency_hz": np.median(peak_frequency, axis=0),
        "top_chunk_count": top_counts.astype(np.int64),
        "chunk_count": np.full(score.shape[1], score.shape[0], dtype=np.int64),
    }


def _select_pixels(
    aggregate: dict[str, np.ndarray],
    *,
    top_k: int,
    min_top_chunk_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    score = aggregate["selection_score"]
    chunk_count = np.maximum(1, aggregate["chunk_count"])
    top_fraction = aggregate["top_chunk_count"] / chunk_count
    hard_eligible = np.asarray(aggregate.get("hard_eroded_eligible", np.ones(score.shape[0], dtype=bool)), dtype=bool)
    if int(np.count_nonzero(hard_eligible)) == 0:
        raise ValueError("Hard erosion removed every candidate pixel.")
    eligible = hard_eligible & (top_fraction >= float(min_top_chunk_fraction))
    if int(np.count_nonzero(eligible)) == 0:
        eligible = hard_eligible
    eligible_idx = np.flatnonzero(eligible)
    eligible_order = eligible_idx[np.argsort(score[eligible_idx])[::-1]]
    remaining_idx = np.flatnonzero(~eligible)
    remaining_order = remaining_idx[np.argsort(score[remaining_idx])[::-1]]
    selected = eligible_order[: min(max(1, int(top_k)), int(eligible_order.size))]
    order = np.concatenate([eligible_order, remaining_order])
    return selected.astype(np.int64), order.astype(np.int64)


def _apply_boundary_penalty(
    aggregate: dict[str, np.ndarray],
    *,
    usable_mask: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    penalty_width_px: float,
    penalty_weight: float,
    hard_erode_px: float,
) -> dict[str, np.ndarray]:
    import cv2

    out = dict(aggregate)
    raw_score = np.asarray(aggregate["selection_score"], dtype=np.float64)
    usable = np.asarray(usable_mask, dtype=bool)
    if usable.shape[0] <= 0 or usable.shape[1] <= 0:
        raise ValueError("usable mask is empty")
    distance_image = cv2.distanceTransform(usable.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float64)
    boundary_distance = distance_image[pixel_y, pixel_x]
    width = float(penalty_width_px)
    weight = float(penalty_weight)
    if width > 0.0 and weight != 0.0:
        boundary_penalty = np.clip((width - boundary_distance) / width, 0.0, 1.0)
    else:
        boundary_penalty = np.zeros_like(raw_score, dtype=np.float64)
    out["unpenalized_selection_score"] = raw_score
    out["boundary_distance_px"] = boundary_distance.astype(np.float64)
    out["boundary_penalty"] = boundary_penalty.astype(np.float64)
    out["hard_eroded_eligible"] = (boundary_distance > float(hard_erode_px)).astype(bool)
    out["selection_score"] = (raw_score - weight * boundary_penalty).astype(np.float64)
    return out


def _write_pixel_scores_csv(
    path: Path,
    *,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    aggregate: dict[str, np.ndarray],
    selected: np.ndarray,
    order: np.ndarray,
) -> None:
    selected_mask = np.zeros(pixel_x.shape[0], dtype=bool)
    selected_mask[selected] = True
    rank = np.empty(pixel_x.shape[0], dtype=np.int64)
    rank[order] = np.arange(1, pixel_x.shape[0] + 1)
    ensure_output_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "stable_x",
                "stable_y",
                "selected",
                "selection_score",
                "unpenalized_selection_score",
                "boundary_distance_px",
                "boundary_penalty",
                "hard_eroded_eligible",
                "mean_selection_score",
                "median_band_power",
                "median_band_fraction",
                "median_peak_to_median",
                "median_peak_frequency_hz",
                "top_chunk_count",
                "chunk_count",
            ],
        )
        writer.writeheader()
        for idx in order.tolist():
            writer.writerow(
                {
                    "rank": int(rank[idx]),
                    "stable_x": int(pixel_x[idx]),
                    "stable_y": int(pixel_y[idx]),
                    "selected": int(selected_mask[idx]),
                    "selection_score": float(aggregate["selection_score"][idx]),
                    "unpenalized_selection_score": float(
                        aggregate.get("unpenalized_selection_score", aggregate["selection_score"])[idx]
                    ),
                    "boundary_distance_px": float(aggregate.get("boundary_distance_px", np.zeros_like(pixel_x))[idx]),
                    "boundary_penalty": float(aggregate.get("boundary_penalty", np.zeros_like(pixel_x))[idx]),
                    "hard_eroded_eligible": int(
                        np.asarray(aggregate.get("hard_eroded_eligible", np.ones_like(pixel_x, dtype=bool)))[idx]
                    ),
                    "mean_selection_score": float(aggregate["mean_selection_score"][idx]),
                    "median_band_power": float(aggregate["median_band_power"][idx]),
                    "median_band_fraction": float(aggregate["median_band_fraction"][idx]),
                    "median_peak_to_median": float(aggregate["median_peak_to_median"][idx]),
                    "median_peak_frequency_hz": float(aggregate["median_peak_frequency_hz"][idx]),
                    "top_chunk_count": int(aggregate["top_chunk_count"][idx]),
                    "chunk_count": int(aggregate["chunk_count"][idx]),
                }
            )


def _detect_beats(
    darkening_z: np.ndarray,
    *,
    frame_index: np.ndarray,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
    prominence_z: float,
) -> dict[str, np.ndarray]:
    from scipy import signal

    valid = np.isfinite(darkening_z)
    peaks_all: list[int] = []
    prominences_all: list[float] = []
    min_distance = max(1, int(math.ceil(float(fps) / float(band_max_hz))))
    start: int | None = None
    for idx, ok in enumerate(valid.tolist()):
        if ok and start is None:
            start = idx
        if (not ok or idx == len(valid) - 1) and start is not None:
            stop = idx if not ok else idx + 1
            if stop - start >= max(16, min_distance * 3):
                segment = darkening_z[start:stop]
                peaks, props = signal.find_peaks(
                    segment,
                    distance=min_distance,
                    prominence=float(prominence_z),
                )
                peaks_all.extend((peaks + start).astype(int).tolist())
                prominences_all.extend(np.asarray(props.get("prominences", []), dtype=np.float64).tolist())
            start = None
    peaks_arr = np.asarray(peaks_all, dtype=np.int64)
    prominences = np.asarray(prominences_all, dtype=np.float64)
    beat_frames = frame_index[peaks_arr] if peaks_arr.size else np.zeros(0, dtype=np.int64)
    beat_times = (beat_frames - frame_index[0]).astype(np.float64) / float(fps) if peaks_arr.size else np.zeros(0)
    ibi_s = np.diff(beat_times)
    bpm = np.full(ibi_s.shape, math.nan, dtype=np.float64)
    if ibi_s.size:
        min_ibi = 1.0 / float(band_max_hz)
        max_ibi = 1.0 / float(band_min_hz)
        plausible = (ibi_s >= min_ibi) & (ibi_s <= max_ibi)
        bpm[plausible] = 60.0 / ibi_s[plausible]
    return {
        "peak_rows": peaks_arr,
        "frame_index": beat_frames.astype(np.int64),
        "time_s": beat_times.astype(np.float64),
        "darkening_z": darkening_z[peaks_arr].astype(np.float64) if peaks_arr.size else np.zeros(0),
        "prominence_z": prominences.astype(np.float64),
        "ibi_s": ibi_s.astype(np.float64),
        "instant_bpm": bpm.astype(np.float64),
    }


def _write_trace_csv(
    path: Path,
    *,
    frame_index: np.ndarray,
    fps: float,
    raw_mean: np.ndarray,
    bandpassed_mean: np.ndarray,
    darkening_z: np.ndarray,
) -> None:
    ensure_output_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "time_s",
                "time_min",
                "selected_raw_mean",
                "selected_bandpassed_mean",
                "selected_darkening_z",
            ],
        )
        writer.writeheader()
        t = (frame_index - frame_index[0]).astype(np.float64) / float(fps)
        for idx in range(len(frame_index)):
            writer.writerow(
                {
                    "frame_index": int(frame_index[idx]),
                    "time_s": float(t[idx]),
                    "time_min": float(t[idx] / 60.0),
                    "selected_raw_mean": float(raw_mean[idx]) if np.isfinite(raw_mean[idx]) else math.nan,
                    "selected_bandpassed_mean": float(bandpassed_mean[idx])
                    if np.isfinite(bandpassed_mean[idx])
                    else math.nan,
                    "selected_darkening_z": float(darkening_z[idx]) if np.isfinite(darkening_z[idx]) else math.nan,
                }
            )


def _write_beats_csv(path: Path, beats: dict[str, np.ndarray]) -> None:
    ensure_output_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_index",
                "frame_index",
                "time_s",
                "darkening_z",
                "prominence_z",
                "inter_event_interval_s",
                "event_rate_per_min",
            ],
        )
        writer.writeheader()
        for idx in range(len(beats["frame_index"])):
            writer.writerow(
                {
                    "event_index": idx,
                    "frame_index": int(beats["frame_index"][idx]),
                    "time_s": float(beats["time_s"][idx]),
                    "darkening_z": float(beats["darkening_z"][idx]),
                    "prominence_z": float(beats["prominence_z"][idx]) if idx < len(beats["prominence_z"]) else math.nan,
                    "inter_event_interval_s": float(beats["ibi_s"][idx - 1])
                    if idx > 0 and idx - 1 < len(beats["ibi_s"])
                    else math.nan,
                    "event_rate_per_min": float(beats["instant_bpm"][idx - 1])
                    if idx > 0 and idx - 1 < len(beats["instant_bpm"])
                    else math.nan,
                }
            )


def _write_outputs(
    *,
    output_prefix: Path,
    mean_frame: np.ndarray,
    roi_polygon: np.ndarray,
    roi_mask: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    aggregate: dict[str, np.ndarray],
    selected: np.ndarray,
    trace_frame_index: np.ndarray,
    fps: float,
    darkening_z: np.ndarray,
    beats: dict[str, np.ndarray],
) -> dict[str, str]:
    import cv2
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_output_dir(output_prefix.parent)
    selected_mask = np.zeros(roi_mask.shape, dtype=bool)
    selected_mask[pixel_y[selected], pixel_x[selected]] = True
    score_image = _scatter_to_image(aggregate["selection_score"], x=pixel_x, y=pixel_y, shape_hw=roi_mask.shape)
    selected_image = mean_frame.copy()
    selected_image[selected_mask] = (0, 0, 255)
    selected_image = _draw_polygon(selected_image, roi_polygon, color=(0, 255, 255))

    pad = 8
    y0 = max(0, int(np.min(pixel_y)) - pad)
    y1 = min(roi_mask.shape[0], int(np.max(pixel_y)) + pad + 1)
    x0 = max(0, int(np.min(pixel_x)) - pad)
    x1 = min(roi_mask.shape[1], int(np.max(pixel_x)) + pad + 1)
    extent = [x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5]

    mask_png = output_prefix.with_suffix(".candidate_mask.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(cv2.cvtColor(_draw_polygon(mean_frame, roi_polygon, color=(0, 255, 255))[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), extent=extent)
    axes[0].set_title("ROI")
    score_crop = np.ma.masked_invalid(score_image[y0:y1, x0:x1])
    im = axes[1].imshow(score_crop, cmap="magma", interpolation="nearest", extent=extent)
    axes[1].set_title("Per-pixel spectral selection score")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(cv2.cvtColor(selected_image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), extent=extent)
    axes[2].set_title("Selected candidate pixels")
    for axis in axes:
        axis.set_xlim(x0 - 0.5, x1 - 0.5)
        axis.set_ylim(y1 - 0.5, y0 - 0.5)
    fig.savefig(mask_png, dpi=160)
    plt.close(fig)

    trace_png = output_prefix.with_suffix(".trace_events.png")
    t = (trace_frame_index - trace_frame_index[0]).astype(np.float64) / float(fps)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True, sharex=True)
    axes[0].plot(t / 60.0, darkening_z, lw=0.6, color="#9c1b30")
    if beats["time_s"].size:
        axes[0].plot(beats["time_s"] / 60.0, beats["darkening_z"], "o", ms=2.5, color="#1f1f1f")
    axes[0].set_title("Selected-pixel darkening trace and detected events")
    axes[0].set_ylabel("darkening z")
    axes[0].grid(True, alpha=0.25)
    if beats["instant_bpm"].size:
        axes[1].plot(beats["time_s"][1:] / 60.0, beats["instant_bpm"], lw=0.8, color="#2f5d7c")
    axes[1].set_title("Event rate from time-domain peak intervals")
    axes[1].set_xlabel("time (min)")
    axes[1].set_ylabel("events/min")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(trace_png, dpi=160)
    plt.close(fig)

    mask_npz = output_prefix.with_suffix(".candidate_mask.npz")
    np.savez_compressed(
        mask_npz,
        roi_mask=roi_mask.astype(np.uint8),
        candidate_mask=selected_mask.astype(np.uint8),
        pixel_x=pixel_x.astype(np.int32),
        pixel_y=pixel_y.astype(np.int32),
        selected_pixel_indices=selected.astype(np.int32),
        selection_score=aggregate["selection_score"].astype(np.float32),
        unpenalized_selection_score=aggregate.get(
            "unpenalized_selection_score",
            aggregate["selection_score"],
        ).astype(np.float32),
        boundary_distance_px=aggregate.get("boundary_distance_px", np.zeros_like(aggregate["selection_score"])).astype(
            np.float32
        ),
        boundary_penalty=aggregate.get("boundary_penalty", np.zeros_like(aggregate["selection_score"])).astype(
            np.float32
        ),
        hard_eroded_eligible=aggregate.get(
            "hard_eroded_eligible",
            np.ones_like(aggregate["selection_score"], dtype=bool),
        ).astype(np.uint8),
        median_band_power=aggregate["median_band_power"].astype(np.float32),
        median_band_fraction=aggregate["median_band_fraction"].astype(np.float32),
        median_peak_frequency_hz=aggregate["median_peak_frequency_hz"].astype(np.float32),
    )
    return {
        "candidate_mask_png": str(mask_png),
        "trace_events_png": str(trace_png),
        "candidate_mask_npz": str(mask_npz),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a candidate periodic-signal mask from per-pixel spectral power, then extract event timings."
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--video", type=Path, required=True, help="Stabilized video to sample.")
    parser.add_argument("--roi-json", type=Path, default=None, help="ROI JSON written by draw_roi.py.")
    parser.add_argument("--roi", type=str, default=None, help="Stabilized ROI rectangle x,y,width,height.")
    parser.add_argument("--status-csv", type=Path, default=None, help="Optional stabilized-video status CSV.")
    parser.add_argument("--mask-npz", type=Path, default=None, help="Optional NPZ whose roi_mask defines candidate pixels.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=0, help="0 means through the end of the video.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--band-min-hz", type=float, default=1.5)
    parser.add_argument("--band-max-hz", type=float, default=3.5)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--chunk-step-seconds", type=float, default=None)
    parser.add_argument("--min-chunk-samples", type=int, default=512)
    parser.add_argument("--top-pixels", type=int, default=50)
    parser.add_argument("--chunk-top-fraction", type=float, default=0.20)
    parser.add_argument("--min-top-chunk-fraction", type=float, default=0.10)
    parser.add_argument(
        "--boundary-penalty-width-px",
        type=float,
        default=0.0,
        help="Softly penalize candidate pixels within this distance of the usable-mask boundary. 0 disables it.",
    )
    parser.add_argument(
        "--boundary-penalty-weight",
        type=float,
        default=0.0,
        help="Selection-score penalty applied to pixels on the usable-mask boundary.",
    )
    parser.add_argument(
        "--hard-erode-usable-mask-px",
        type=float,
        default=0.0,
        help="Exclude candidate pixels whose distance to unusable pixels is at or below this value.",
    )
    parser.add_argument("--min-roi-mean-intensity", type=float, default=1.0)
    parser.add_argument("--max-interpolated-gap-samples", type=int, default=5)
    parser.add_argument("--event-prominence-z", "--beat-prominence-z", dest="event_prominence_z", type=float, default=0.75)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/candidate_periodic_pixel_mask"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    roi_rect = resolve_roi_rect(config, roi=args.roi, roi_json=args.roi_json)
    roi_polygon = roi_rect_corners(roi_rect)
    sample_mask = None
    if args.mask_npz is not None:
        from map_pixel_band_contributions import _video_shape

        sample_mask = _mask_from_npz(args.mask_npz, shape_hw=_video_shape(args.video))

    loaded = _load_roi_pixel_traces(
        video_path=args.video,
        roi_polygon=roi_polygon,
        status_csv=args.status_csv,
        frame_start=max(0, int(args.frame_start)),
        frame_count=max(0, int(args.frame_count)),
        stride=max(1, int(args.stride)),
        sample_mask=sample_mask,
        min_roi_mean_intensity=args.min_roi_mean_intensity,
    )
    effective_fps = float(args.fps) / max(1, int(args.stride))
    traces, finite_rows, interpolated_rows, segments = _interpolate_short_gaps_all_segments(
        traces=loaded["traces"],
        valid=loaded["valid"],
        max_gap=max(0, int(args.max_interpolated_gap_samples)),
    )
    chunk_step = float(args.chunk_step_seconds) if args.chunk_step_seconds is not None else float(args.chunk_seconds)
    chunks = _iter_chunks(
        segments=segments,
        fps=effective_fps,
        chunk_seconds=float(args.chunk_seconds),
        step_seconds=chunk_step,
        min_samples=max(32, int(args.min_chunk_samples)),
    )
    if not chunks:
        raise ValueError("No chunks were long enough for per-pixel spectral scoring.")

    chunk_scores: list[dict[str, np.ndarray]] = []
    chunk_rows: list[dict[str, Any]] = []
    for chunk_index, start, stop in chunks:
        scores = _welch_pixel_scores(
            traces[start:stop],
            fps=effective_fps,
            band_min_hz=float(args.band_min_hz),
            band_max_hz=float(args.band_max_hz),
        )
        chunk_scores.append(scores)
        best = int(np.argmax(scores["band_power"]))
        chunk_rows.append(
            {
                "chunk_index": int(chunk_index),
                "start_frame_index": int(loaded["frame_indices"][start]),
                "stop_frame_index_inclusive": int(loaded["frame_indices"][stop - 1]),
                "start_s": float((loaded["frame_indices"][start] - loaded["frame_indices"][0]) / effective_fps),
                "stop_s": float((loaded["frame_indices"][stop - 1] - loaded["frame_indices"][0]) / effective_fps),
                "sample_count": int(stop - start),
                "best_pixel_x": int(loaded["roi_x"][best]),
                "best_pixel_y": int(loaded["roi_y"][best]),
                "best_pixel_band_power": float(scores["band_power"][best]),
                "best_pixel_peak_hz": float(scores["peak_frequency_hz"][best]),
            }
        )

    aggregate = _aggregate_pixel_scores(chunk_scores, top_fraction=float(args.chunk_top_fraction))
    aggregate = _apply_boundary_penalty(
        aggregate,
        usable_mask=loaded["roi_mask"],
        pixel_x=loaded["roi_x"],
        pixel_y=loaded["roi_y"],
        penalty_width_px=float(args.boundary_penalty_width_px),
        penalty_weight=float(args.boundary_penalty_weight),
        hard_erode_px=float(args.hard_erode_usable_mask_px),
    )
    selected, order = _select_pixels(
        aggregate,
        top_k=int(args.top_pixels),
        min_top_chunk_fraction=float(args.min_top_chunk_fraction),
    )

    bandpassed = np.full(traces.shape, math.nan, dtype=np.float64)
    for start, stop in segments:
        bandpassed[start:stop] = _bandpass_matrix(
            traces[start:stop],
            fps=effective_fps,
            band_min_hz=float(args.band_min_hz),
            band_max_hz=float(args.band_max_hz),
        )
    analysis_rows = np.isfinite(bandpassed[:, selected]).all(axis=1)
    raw_mean = np.full(traces.shape[0], math.nan, dtype=np.float64)
    bandpassed_mean = np.full(traces.shape[0], math.nan, dtype=np.float64)
    darkening_z = np.full(traces.shape[0], math.nan, dtype=np.float64)
    raw_rows = finite_rows & np.isfinite(traces[:, selected]).all(axis=1)
    raw_mean[raw_rows] = np.mean(traces[raw_rows][:, selected], axis=1)
    bandpassed_mean[analysis_rows] = np.mean(bandpassed[analysis_rows][:, selected], axis=1)
    darkening_z[analysis_rows] = -_zscore(bandpassed_mean[analysis_rows])

    beats = _detect_beats(
        darkening_z,
        frame_index=loaded["frame_indices"],
        fps=effective_fps,
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
        prominence_z=float(args.event_prominence_z),
    )

    output_prefix = args.output_prefix
    ensure_output_dir(output_prefix.parent)
    pixel_csv = output_prefix.with_suffix(".pixel_scores.csv")
    chunk_csv = output_prefix.with_suffix(".chunk_scores.csv")
    trace_csv = output_prefix.with_suffix(".trace.csv")
    beats_csv = output_prefix.with_suffix(".events.csv")
    summary_json = output_prefix.with_suffix(".summary.json")
    outputs = _write_outputs(
        output_prefix=output_prefix,
        mean_frame=loaded["mean_frame"],
        roi_polygon=roi_polygon,
        roi_mask=loaded["roi_mask"],
        pixel_x=loaded["roi_x"],
        pixel_y=loaded["roi_y"],
        aggregate=aggregate,
        selected=selected,
        trace_frame_index=loaded["frame_indices"],
        fps=effective_fps,
        darkening_z=darkening_z,
        beats=beats,
    )
    _write_pixel_scores_csv(
        pixel_csv,
        pixel_x=loaded["roi_x"],
        pixel_y=loaded["roi_y"],
        aggregate=aggregate,
        selected=selected,
        order=order,
    )
    ensure_output_dir(chunk_csv.parent)
    with chunk_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chunk_rows[0].keys()))
        writer.writeheader()
        writer.writerows(chunk_rows)
    _write_trace_csv(
        trace_csv,
        frame_index=loaded["frame_indices"],
        fps=effective_fps,
        raw_mean=raw_mean,
        bandpassed_mean=bandpassed_mean,
        darkening_z=darkening_z,
    )
    _write_beats_csv(beats_csv, beats)
    outputs.update(
        {
            "pixel_scores_csv": str(pixel_csv),
            "chunk_scores_csv": str(chunk_csv),
            "trace_csv": str(trace_csv),
            "events_csv": str(beats_csv),
            "summary_json": str(summary_json),
        }
    )

    finite_bpm = beats["instant_bpm"][np.isfinite(beats["instant_bpm"])]
    summary = {
        "source_video": str(args.video),
        "status_csv": str(args.status_csv) if args.status_csv is not None else None,
        "roi_json": str(args.roi_json) if args.roi_json is not None else None,
        "mask_npz": str(args.mask_npz) if args.mask_npz is not None else None,
        "frame_start": int(args.frame_start),
        "frame_count_requested": int(args.frame_count),
        "stride": int(args.stride),
        "fps": effective_fps,
        "band_hz": [float(args.band_min_hz), float(args.band_max_hz)],
        "chunk_seconds": float(args.chunk_seconds),
        "chunk_step_seconds": float(chunk_step),
        "chunk_count": int(len(chunks)),
        "boundary_penalty_width_px": float(args.boundary_penalty_width_px),
        "boundary_penalty_weight": float(args.boundary_penalty_weight),
        "hard_erode_usable_mask_px": float(args.hard_erode_usable_mask_px),
        "hard_eroded_candidate_pixel_count": int(np.count_nonzero(aggregate["hard_eroded_eligible"])),
        "selection_method": "per-pixel Welch band power/fraction/peak score for mask selection only",
        "trace_method": "selected-pixel luminance mean, band-passed in time, darkening peaks detected in time domain",
        "loaded_frames": int(loaded["traces"].shape[0]),
        "valid_loaded_frames": int(np.count_nonzero(loaded["valid"])),
        "low_intensity_frame_count": int(loaded["low_intensity_frame_count"]),
        "interpolated_rows": int(interpolated_rows),
        "roi_pixel_count": int(loaded["roi_x"].size),
        "selected_pixel_count": int(selected.size),
        "selected_boundary_distance_px_median": float(np.median(aggregate["boundary_distance_px"][selected]))
        if selected.size
        else math.nan,
        "selected_boundary_distance_px_min": float(np.min(aggregate["boundary_distance_px"][selected]))
        if selected.size
        else math.nan,
        "event_count": int(beats["frame_index"].size),
        "event_prominence_z": float(args.event_prominence_z),
        "event_rate_per_min_median": float(np.median(finite_bpm)) if finite_bpm.size else math.nan,
        "event_rate_per_min_mean": float(np.mean(finite_bpm)) if finite_bpm.size else math.nan,
        "event_rate_per_min_min": float(np.min(finite_bpm)) if finite_bpm.size else math.nan,
        "event_rate_per_min_max": float(np.max(finite_bpm)) if finite_bpm.size else math.nan,
        "outputs": outputs,
    }
    with summary_json.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"summary_json: {summary_json}")
    print(f"candidate_mask_png: {outputs['candidate_mask_png']}")
    print(f"trace_events_png: {outputs['trace_events_png']}")
    print(f"pixel_scores_csv: {pixel_csv}")
    print(f"trace_csv: {trace_csv}")
    print(f"events_csv: {beats_csv}")
    print(f"chunks: {len(chunks)}")
    print(f"selected_pixels: {selected.size}")
    print(f"events: {beats['frame_index'].size}")
    if finite_bpm.size:
        print(f"event_rate_per_min_median: {np.median(finite_bpm):.3f}")


if __name__ == "__main__":
    main()
