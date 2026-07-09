from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from _common import ensure_output_dir, load_config, polygon_mask, resolve_roi_rect, roi_rect_corners
from map_pixel_band_contributions import _draw_polygon, _gray, _read_status_csv, _video_shape
from visualize_roi_intensity_diagnostics import _mask_from_npz


def _clip_bounds(mask: np.ndarray, *, pad: int) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        raise ValueError("Mask contains no pixels.")
    y0 = max(0, int(np.min(yy)) - int(pad))
    y1 = min(mask.shape[0], int(np.max(yy)) + int(pad) + 1)
    x0 = max(0, int(np.min(xx)) - int(pad))
    x1 = min(mask.shape[1], int(np.max(xx)) + int(pad) + 1)
    return x0, y0, x1, y1


def _valid_segments(valid: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, ok in enumerate(valid.tolist()):
        if ok and start is None:
            start = idx
        if (not ok or idx == len(valid) - 1) and start is not None:
            stop = idx if not ok else idx + 1
            segments.append((start, stop))
            start = None
    return segments


def _interpolate_short_invalid_stack(
    stack: np.ndarray,
    valid: np.ndarray,
    *,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    out = stack.astype(np.float32, copy=True)
    finite = np.asarray(valid, dtype=bool).copy()
    interpolated = 0
    idx = 0
    while idx < finite.size:
        if finite[idx]:
            idx += 1
            continue
        start = idx
        while idx < finite.size and not finite[idx]:
            idx += 1
        stop = idx
        if start == 0 or stop >= finite.size or stop - start > int(max_gap):
            continue
        left = out[start - 1].astype(np.float32)
        right = out[stop].astype(np.float32)
        steps = float(stop - start + 1)
        for offset, row in enumerate(range(start, stop), start=1):
            frac = float(offset) / steps
            out[row] = (1.0 - frac) * left + frac * right
            finite[row] = True
            interpolated += 1
    return out, finite, interpolated


def _load_roi_stack(
    *,
    video_path: Path,
    roi_polygon: np.ndarray,
    status_csv: Path | None,
    frame_start: int,
    frame_count: int,
    stride: int,
    sample_mask: np.ndarray | None,
    min_roi_mean_intensity: float | None,
    pad_px: int,
    duplicate_mean_tol: float,
) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if int(frame_count) <= 0:
        frame_count = max(0, total_frames - int(frame_start))
    frame_indices = np.arange(
        int(frame_start),
        int(frame_start) + int(frame_count),
        max(1, int(stride)),
        dtype=np.int64,
    )
    if frame_indices.size == 0:
        raise ValueError("No frames selected.")

    base_roi_mask = polygon_mask((height, width), roi_polygon)
    candidate_mask = base_roi_mask.copy()
    if sample_mask is not None:
        provided = np.asarray(sample_mask, dtype=bool)
        if provided.shape != candidate_mask.shape:
            raise ValueError(f"sample mask shape {provided.shape} does not match video shape {candidate_mask.shape}")
        candidate_mask &= provided
    if int(np.count_nonzero(candidate_mask)) == 0:
        raise ValueError("Candidate mask contains no pixels.")

    x0, y0, x1, y1 = _clip_bounds(base_roi_mask, pad=pad_px)
    base_crop_mask = base_roi_mask[y0:y1, x0:x1]
    candidate_crop_mask = candidate_mask[y0:y1, x0:x1]
    if int(np.count_nonzero(candidate_crop_mask)) == 0:
        raise ValueError("Candidate crop mask contains no pixels.")

    stack = np.full((int(frame_indices.size), y1 - y0, x1 - x0), np.nan, dtype=np.float32)
    valid = np.zeros(int(frame_indices.size), dtype=bool)
    low_intensity_count = 0
    read_failure_count = 0
    status = _read_status_csv(status_csv)
    intensity_threshold = (
        float(min_roi_mean_intensity)
        if min_roi_mean_intensity is not None and np.isfinite(float(min_roi_mean_intensity))
        else None
    )

    next_expected: int | None = None
    try:
        for out_row, frame_index in enumerate(frame_indices.tolist()):
            if next_expected is None or int(frame_index) != int(next_expected):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            next_expected = int(frame_index) + 1
            if not ok:
                read_failure_count += 1
                continue
            if not status.get(int(frame_index), True):
                continue
            crop = _gray(frame)[y0:y1, x0:x1].astype(np.float32)
            mean_value = float(np.mean(crop[candidate_crop_mask]))
            if intensity_threshold is not None and mean_value <= intensity_threshold:
                low_intensity_count += 1
                continue
            stack[out_row] = crop
            valid[out_row] = True
    finally:
        capture.release()

    if int(np.count_nonzero(valid)) >= 2 and float(duplicate_mean_tol) >= 0.0:
        means = np.full(valid.shape, np.nan, dtype=np.float64)
        means[valid] = np.mean(stack[valid][:, candidate_crop_mask], axis=1)
        valid_positions = np.flatnonzero(valid)
        duplicate = np.zeros(valid.shape, dtype=bool)
        for prev, cur in zip(valid_positions[:-1], valid_positions[1:]):
            if abs(float(means[cur] - means[prev])) <= float(duplicate_mean_tol):
                duplicate[cur] = True
        valid &= ~duplicate
        stack[duplicate] = np.nan
    else:
        duplicate = np.zeros(valid.shape, dtype=bool)

    if int(np.count_nonzero(valid)) < 32:
        raise ValueError("Fewer than 32 valid frames were loaded.")

    return {
        "stack": stack,
        "valid": valid,
        "frame_indices": frame_indices,
        "base_crop_mask": base_crop_mask,
        "candidate_crop_mask": candidate_crop_mask,
        "bbox_xyxy": (x0, y0, x1, y1),
        "video_shape_hw": (height, width),
        "low_intensity_frame_count": int(low_intensity_count),
        "duplicate_frame_count": int(np.count_nonzero(duplicate)),
        "read_failure_count": int(read_failure_count),
    }


def _prepare_stack(
    stack: np.ndarray,
    valid: np.ndarray,
    *,
    max_interpolated_gap: int,
    min_segment_samples: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], int]:
    interpolated_stack, finite, interpolated = _interpolate_short_invalid_stack(
        stack,
        valid,
        max_gap=max_interpolated_gap,
    )
    segments = [(start, stop) for start, stop in _valid_segments(finite) if stop - start >= int(min_segment_samples)]
    if not segments:
        raise ValueError("No contiguous valid segment remains after short-gap interpolation.")
    start, stop = max(segments, key=lambda item: item[1] - item[0])
    return interpolated_stack[start:stop], finite[start:stop], (start, stop), int(interpolated)


def _nuisance_regressors(stack: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = stack.reshape(stack.shape[0], -1)
    mask_flat = mask.reshape(-1)
    traces = flat[:, mask_flat]
    mean_image = np.mean(stack, axis=0)
    grad_y, grad_x = np.gradient(mean_image)
    gx = grad_x.reshape(-1)[mask_flat]
    gy = grad_y.reshape(-1)[mask_flat]
    gradient_design = np.stack([gx, gy], axis=1)
    if gradient_design.shape[0] < 2:
        raise ValueError("Too few candidate pixels for nuisance regression.")
    pinv = np.linalg.pinv(gradient_design)
    residual = traces - mean_image.reshape(-1)[mask_flat][None, :]
    displacement = (pinv @ residual.T).T
    dx = displacement[:, 0]
    dy = displacement[:, 1]
    global_mean = np.mean(traces, axis=1)
    motion_pred = (gx[None, :] * dx[:, None] + gy[None, :] * dy[:, None]).T
    scalar = np.stack([np.ones(stack.shape[0]), global_mean, dx, dy], axis=1)
    return motion_pred, scalar, mean_image


def _residualize(traces: np.ndarray, motion_pred: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    adjusted = traces - motion_pred
    beta, *_ = np.linalg.lstsq(scalar, adjusted.T, rcond=None)
    return (adjusted.T - scalar @ beta).T


def _inband_snr(
    traces: np.ndarray,
    *,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
    nperseg: int | None = None,
) -> np.ndarray:
    from scipy import signal

    if traces.shape[-1] < 32:
        raise ValueError("Need at least 32 samples for spectral scoring.")
    effective_nperseg = nperseg or min(traces.shape[-1], max(64, int(round(float(fps) * 4.0))))
    frequencies, psd = signal.welch(
        traces,
        fs=float(fps),
        nperseg=effective_nperseg,
        axis=-1,
        detrend="linear",
    )
    band = (frequencies >= float(band_min_hz)) & (frequencies <= float(band_max_hz))
    if int(np.count_nonzero(band)) == 0:
        raise ValueError("No spectral bins fall inside the requested band.")
    nyquist = float(fps) / 2.0
    noise = (
        ((frequencies > 0.2) & (frequencies < float(band_min_hz) - 0.2))
        | ((frequencies > float(band_max_hz) + 0.2) & (frequencies < min(nyquist * 0.9, float(band_max_hz) + 3.0)))
    )
    if int(np.count_nonzero(noise)) < 3:
        noise = (~band) & (frequencies > 0.1)
    peak = np.max(psd[..., band], axis=-1)
    floor = np.median(psd[..., noise], axis=-1) + 1e-12
    score = np.log10(np.divide(peak, floor, out=np.zeros_like(peak), where=floor > 0.0))
    score[~np.isfinite(score)] = 0.0
    return score.astype(np.float64)


def _chunk_stability(
    traces: np.ndarray,
    *,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
    chunk_seconds: float,
    min_chunk_samples: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    chunk_len = max(int(min_chunk_samples), int(round(float(chunk_seconds) * float(fps))))
    if traces.shape[1] < chunk_len:
        return np.zeros(traces.shape[0], dtype=np.float64), np.full(traces.shape[0], np.nan), 0
    scores: list[np.ndarray] = []
    for start in range(0, traces.shape[1] - chunk_len + 1, chunk_len):
        stop = start + chunk_len
        scores.append(
            _inband_snr(
                traces[:, start:stop],
                fps=fps,
                band_min_hz=band_min_hz,
                band_max_hz=band_max_hz,
            )
        )
    if not scores:
        return np.zeros(traces.shape[0], dtype=np.float64), np.full(traces.shape[0], np.nan), 0
    stacked = np.stack(scores, axis=1)
    per_chunk_median = np.median(stacked, axis=0, keepdims=True)
    frac_above = np.mean(stacked > per_chunk_median, axis=1)
    cv = np.std(stacked, axis=1) / (np.abs(np.mean(stacked, axis=1)) + 1e-9)
    return frac_above.astype(np.float64), cv.astype(np.float64), int(stacked.shape[1])


def _scatter(values: np.ndarray, *, mask: np.ndarray) -> np.ndarray:
    image = np.full(mask.shape, np.nan, dtype=np.float64)
    image[mask] = values
    return image


def _edge_penalty_map(mean_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(mean_image)
    grad = np.hypot(grad_x, grad_y)
    grad[~mask] = np.nan
    return grad


def _select_and_validate(
    score_map: np.ndarray,
    *,
    mask: np.ndarray,
    mean_image: np.ndarray,
    hard_erode_px: int,
    gradient_percentile: float,
    score_percentile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import ndimage

    usable = ndimage.binary_erosion(mask, iterations=max(0, int(hard_erode_px)))
    if int(np.count_nonzero(usable)) == 0:
        return np.zeros(mask.shape, dtype=bool), {
            "candidate_pixel_count": 0,
            "selected_pixel_count": 0,
            "n_components": 0,
            "largest_size": 0,
            "largest_fraction": 0.0,
            "compactness": 0.0,
            "dist_to_boundary_px": 0.0,
            "participation_ratio": 0.0,
        }
    grad = _edge_penalty_map(mean_image, mask)
    grad_threshold = float(np.nanpercentile(grad[usable], float(gradient_percentile)))
    interior = usable & (grad <= grad_threshold) & np.isfinite(score_map)
    if int(np.count_nonzero(interior)) == 0:
        return np.zeros(mask.shape, dtype=bool), {
            "candidate_pixel_count": int(np.count_nonzero(usable)),
            "selected_pixel_count": 0,
            "n_components": 0,
            "largest_size": 0,
            "largest_fraction": 0.0,
            "compactness": 0.0,
            "dist_to_boundary_px": 0.0,
            "participation_ratio": 0.0,
        }
    threshold = float(np.nanpercentile(score_map[interior], float(score_percentile)))
    selected = interior & (score_map >= threshold)
    labels, n_components = ndimage.label(selected)
    if int(n_components) == 0:
        return selected, {
            "candidate_pixel_count": int(np.count_nonzero(interior)),
            "selected_pixel_count": 0,
            "n_components": 0,
            "largest_size": 0,
            "largest_fraction": 0.0,
            "compactness": 0.0,
            "dist_to_boundary_px": 0.0,
            "participation_ratio": 0.0,
        }
    sizes = ndimage.sum(np.ones_like(labels), labels, range(1, int(n_components) + 1))
    largest_label = 1 + int(np.argmax(sizes))
    component = labels == largest_label
    yy, xx = np.nonzero(component)
    area = int(np.count_nonzero(component))
    total = int(np.count_nonzero(selected))
    bbox_area = int((yy.max() - yy.min() + 1) * (xx.max() - xx.min() + 1)) if area else 1
    compactness = float(area / max(1, bbox_area)) if area >= 4 else 0.0
    distance = ndimage.distance_transform_edt(mask)
    score_values = score_map[component]
    score_values = score_values - np.nanmin(score_values) + 1e-9 if area else np.asarray([1.0])
    participation_ratio = float((np.nansum(score_values) ** 2) / (np.nansum(score_values**2) + 1e-9))
    centroid = [float(np.mean(yy)), float(np.mean(xx))] if area else [math.nan, math.nan]
    return selected, {
        "candidate_pixel_count": int(np.count_nonzero(interior)),
        "selected_pixel_count": total,
        "n_components": int(n_components),
        "largest_size": area,
        "largest_fraction": float(area / max(1, total)),
        "compactness": compactness,
        "dist_to_boundary_px": float(np.mean(distance[component])) if area else 0.0,
        "participation_ratio": participation_ratio,
        "centroid_yx": centroid,
    }


def _phase_randomize(traces: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(traces, axis=-1)
    amplitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    randomized = rng.uniform(-np.pi, np.pi, phase.shape)
    randomized[..., 0] = 0.0
    if traces.shape[-1] % 2 == 0:
        randomized[..., -1] = 0.0
    return np.fft.irfft(amplitude * np.exp(1j * randomized), n=traces.shape[-1], axis=-1)


def _null_scores(
    traces: np.ndarray,
    *,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
    mask: np.ndarray,
    random_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    from scipy import ndimage

    distance = ndimage.distance_transform_edt(mask)
    distance_in_mask = distance[mask]
    interior_idx = np.flatnonzero(distance_in_mask >= 4.0)
    boundary_idx = np.flatnonzero((distance_in_mask > 0.0) & (distance_in_mask <= 2.0))
    output: dict[str, np.ndarray] = {}
    if interior_idx.size:
        chosen = rng.choice(interior_idx, size=min(int(random_samples), int(interior_idx.size)), replace=False)
        output["random_interior"] = _inband_snr(
            traces[chosen],
            fps=fps,
            band_min_hz=band_min_hz,
            band_max_hz=band_max_hz,
        )
        shuffled = traces[chosen][:, rng.permutation(traces.shape[1])]
        output["shuffled_time"] = _inband_snr(
            shuffled,
            fps=fps,
            band_min_hz=band_min_hz,
            band_max_hz=band_max_hz,
        )
        output["phase_random"] = _inband_snr(
            _phase_randomize(traces[chosen], rng),
            fps=fps,
            band_min_hz=band_min_hz,
            band_max_hz=band_max_hz,
        )
    else:
        empty = np.zeros(0, dtype=np.float64)
        output["random_interior"] = empty
        output["shuffled_time"] = empty
        output["phase_random"] = empty
    if boundary_idx.size:
        chosen_boundary = rng.choice(boundary_idx, size=min(int(random_samples), int(boundary_idx.size)), replace=False)
        output["boundary_only"] = _inband_snr(
            traces[chosen_boundary],
            fps=fps,
            band_min_hz=band_min_hz,
            band_max_hz=band_max_hz,
        )
    else:
        output["boundary_only"] = np.zeros(0, dtype=np.float64)
    return output


def _bandpass_trace(trace: np.ndarray, *, fps: float, band_min_hz: float, band_max_hz: float) -> np.ndarray:
    from scipy import signal

    nyquist = float(fps) / 2.0
    low = float(band_min_hz) / nyquist
    high = min(float(band_max_hz) / nyquist, 0.99)
    if not (0.0 < low < high < 1.0):
        raise ValueError(f"Invalid band for fps={fps}: {band_min_hz}..{band_max_hz}")
    sos = signal.butter(3, [low, high], btype="band", output="sos")
    return signal.sosfiltfilt(sos, trace)


def _extract_events(
    trace: np.ndarray,
    *,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
    prominence_scale: float,
) -> dict[str, np.ndarray | float | int]:
    from scipy import signal

    filtered = _bandpass_trace(trace, fps=fps, band_min_hz=band_min_hz, band_max_hz=band_max_hz)
    refractory = max(1, int(round(float(fps) / float(band_max_hz) * 0.8)))
    prominence = float(prominence_scale) * float(np.nanstd(filtered))
    peaks, properties = signal.find_peaks(filtered, distance=refractory, prominence=prominence)
    times_s = peaks.astype(np.float64) / float(fps)
    intervals_s = np.diff(times_s)
    low_interval = 1.0 / float(band_max_hz)
    high_interval = 1.0 / float(band_min_hz)
    in_band = (intervals_s >= low_interval * 0.7) & (intervals_s <= high_interval * 1.3) if intervals_s.size else np.zeros(0, dtype=bool)
    rates = np.divide(60.0, intervals_s, out=np.full(intervals_s.shape, math.nan), where=intervals_s > 0)
    return {
        "filtered": filtered.astype(np.float64),
        "peaks": peaks.astype(np.int64),
        "times_s": times_s.astype(np.float64),
        "intervals_s": intervals_s.astype(np.float64),
        "event_rate_per_min": rates.astype(np.float64),
        "prominence": np.asarray(properties.get("prominences", []), dtype=np.float64),
        "fraction_intervals_in_band": float(np.mean(in_band)) if in_band.size else 0.0,
        "rejected_interval_count": int(np.count_nonzero(~in_band)) if in_band.size else 0,
    }


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.percentile(finite, percentile))


def _trace_spike_metrics(trace: np.ndarray, *, threshold_z: float = 12.0) -> dict[str, float]:
    finite = np.asarray(trace, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "selected_trace_abs_robust_z_p99": math.nan,
            "selected_trace_spike_fraction": math.nan,
            "selected_trace_spike_threshold_z": float(threshold_z),
        }
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return {
            "selected_trace_abs_robust_z_p99": 0.0,
            "selected_trace_spike_fraction": 0.0,
            "selected_trace_spike_threshold_z": float(threshold_z),
        }
    abs_z = np.abs((finite - median) / scale)
    return {
        "selected_trace_abs_robust_z_p99": float(np.percentile(abs_z, 99)),
        "selected_trace_spike_fraction": float(np.mean(abs_z > float(threshold_z))),
        "selected_trace_spike_threshold_z": float(threshold_z),
    }


def _build_verdict(
    *,
    metrics: dict[str, Any],
    nulls: dict[str, np.ndarray],
    selected_scores: np.ndarray,
    events: dict[str, np.ndarray | float | int] | None,
    min_component_pixels: int,
    min_participation_ratio: float,
    min_largest_fraction: float,
    min_boundary_distance_px: float,
    min_fraction_intervals_in_band: float,
    max_spike_fraction: float,
) -> dict[str, bool]:
    selected_p90 = _safe_percentile(selected_scores, 90)
    boundary_p95 = _safe_percentile(nulls.get("boundary_only", np.zeros(0)), 95)
    phase_p95 = _safe_percentile(nulls.get("phase_random", np.zeros(0)), 95)
    verdict = {
        "survives_hard_erosion": int(metrics.get("largest_size", 0)) >= int(min_component_pixels),
        "concentrated": float(metrics.get("participation_ratio", 0.0)) >= float(min_participation_ratio),
        "single_dominant_cluster": float(metrics.get("largest_fraction", 0.0)) >= float(min_largest_fraction),
        "interior": float(metrics.get("dist_to_boundary_px", 0.0)) >= float(min_boundary_distance_px),
        "exceeds_boundary_null": bool(
            np.isfinite(selected_p90) and np.isfinite(boundary_p95) and selected_p90 > boundary_p95
        ),
        "exceeds_phase_null": bool(np.isfinite(selected_p90) and np.isfinite(phase_p95) and selected_p90 > phase_p95),
        "limited_transient_spikes": bool(
            np.isfinite(float(metrics.get("selected_trace_spike_fraction", math.nan)))
            and float(metrics.get("selected_trace_spike_fraction", math.inf)) <= float(max_spike_fraction)
        ),
    }
    if events is None:
        verdict["intervals_plausible"] = False
    else:
        verdict["intervals_plausible"] = float(events["fraction_intervals_in_band"]) >= float(min_fraction_intervals_in_band)
    verdict["trust_event_series"] = bool(all(verdict.values()))
    return verdict


def _write_events_csv(
    path: Path,
    *,
    events: dict[str, np.ndarray | float | int],
    frame_indices: np.ndarray,
    segment_start_row: int,
    stride: int,
) -> None:
    ensure_output_dir(path.parent)
    peaks = np.asarray(events["peaks"], dtype=np.int64)
    times = np.asarray(events["times_s"], dtype=np.float64)
    prominences = np.asarray(events["prominence"], dtype=np.float64)
    intervals = np.asarray(events["intervals_s"], dtype=np.float64)
    rates = np.asarray(events["event_rate_per_min"], dtype=np.float64)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_index",
                "frame_index",
                "time_s",
                "prominence",
                "inter_event_interval_s",
                "event_rate_per_min",
            ],
        )
        writer.writeheader()
        for idx, peak in enumerate(peaks.tolist()):
            original_row = int(segment_start_row) + int(peak)
            if int(stride) != 1:
                original_row = int(segment_start_row) + int(peak)
            writer.writerow(
                {
                    "event_index": idx,
                    "frame_index": int(frame_indices[original_row]),
                    "time_s": float(times[idx]),
                    "prominence": float(prominences[idx]) if idx < prominences.size else math.nan,
                    "inter_event_interval_s": float(intervals[idx - 1]) if idx > 0 and idx - 1 < intervals.size else math.nan,
                    "event_rate_per_min": float(rates[idx - 1]) if idx > 0 and idx - 1 < rates.size else math.nan,
                }
            )


def _write_figure(
    path: Path,
    *,
    mean_image: np.ndarray,
    score_raw: np.ndarray,
    score_residual: np.ndarray,
    score_stable: np.ndarray,
    selected: np.ndarray,
    mask: np.ndarray,
    nulls: dict[str, np.ndarray],
    selected_trace: np.ndarray,
    events: dict[str, np.ndarray | float | int] | None,
    fps: float,
    title: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_output_dir(path.parent)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, image, label in (
        (axes[0, 0], score_raw, "In-band SNR, raw"),
        (axes[0, 1], score_residual, "In-band SNR, nuisance-removed"),
        (axes[0, 2], score_stable, "Chunk-stable score"),
    ):
        masked = np.ma.masked_invalid(image)
        handle = axis.imshow(masked, cmap="magma", interpolation="nearest")
        axis.contour(mask.astype(np.uint8), levels=[0.5], colors="white", linewidths=0.5)
        axis.set_title(label)
        axis.axis("off")
        fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)

    axes[1, 0].imshow(mean_image, cmap="gray", interpolation="nearest")
    axes[1, 0].contour(mask.astype(np.uint8), levels=[0.5], colors="yellow", linewidths=0.7)
    axes[1, 0].contour(selected.astype(np.uint8), levels=[0.5], colors="red", linewidths=0.9)
    axes[1, 0].set_title("Candidate mask and selected cluster")
    axes[1, 0].axis("off")

    axes[1, 1].set_title("Null score distributions")
    colors = {
        "random_interior": "#666666",
        "boundary_only": "#c23b22",
        "shuffled_time": "#345995",
        "phase_random": "#2e7d32",
    }
    for key, color in colors.items():
        values = nulls.get(key, np.zeros(0, dtype=np.float64))
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        axes[1, 1].hist(finite, bins=30, histtype="step", color=color, label=key, density=True)
        axes[1, 1].axvline(np.percentile(finite, 95), color=color, linestyle="--", linewidth=0.8)
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].set_xlabel("log in-band SNR")

    time_s = np.arange(selected_trace.size, dtype=np.float64) / float(fps)
    axes[1, 2].plot(time_s, selected_trace, linewidth=0.6, color="#666666", label="selected trace")
    if events is not None:
        filtered = np.asarray(events["filtered"], dtype=np.float64)
        peaks = np.asarray(events["peaks"], dtype=np.int64)
        axes[1, 2].plot(time_s, filtered, linewidth=0.8, color="#111111", label="band-passed")
        if peaks.size:
            axes[1, 2].plot(peaks / float(fps), filtered[peaks], "v", ms=4, color="#c23b22", label="events")
        rates = np.asarray(events["event_rate_per_min"], dtype=np.float64)
        finite_rates = rates[np.isfinite(rates)]
        median_rate = float(np.median(finite_rates)) if finite_rates.size else math.nan
        axes[1, 2].set_title(
            f"Event extraction, median={median_rate:.1f}/min, "
            f"in-band={float(events['fraction_intervals_in_band']) * 100:.0f}%"
        )
    else:
        axes[1, 2].set_title("Event extraction skipped")
    axes[1, 2].set_xlabel("time (s)")
    axes[1, 2].legend(fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_probe(
    *,
    stack: np.ndarray,
    mask: np.ndarray,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
    hard_erode_px: int,
    gradient_percentile: float,
    score_percentile: float,
    chunk_seconds: float,
    min_chunk_samples: int,
    random_samples: int,
    event_prominence_scale: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    stack = np.asarray(stack, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    flat = stack.reshape(stack.shape[0], -1)
    mask_flat = mask.reshape(-1)
    traces = flat[:, mask_flat].T.astype(np.float64)

    motion_pred, scalar, mean_image = _nuisance_regressors(stack, mask)
    residual = _residualize(traces, motion_pred, scalar)

    score_raw = _inband_snr(
        traces,
        fps=fps,
        band_min_hz=band_min_hz,
        band_max_hz=band_max_hz,
    )
    score_residual = _inband_snr(
        residual,
        fps=fps,
        band_min_hz=band_min_hz,
        band_max_hz=band_max_hz,
    )
    frac_above, cv, chunk_count = _chunk_stability(
        residual,
        fps=fps,
        band_min_hz=band_min_hz,
        band_max_hz=band_max_hz,
        chunk_seconds=chunk_seconds,
        min_chunk_samples=min_chunk_samples,
    )
    stable = score_residual + 0.5 * (frac_above - 0.5) - 0.3 * np.clip(np.nan_to_num(cv, nan=0.0), 0.0, 3.0)

    score_raw_map = _scatter(score_raw, mask=mask)
    score_residual_map = _scatter(score_residual, mask=mask)
    stable_map = _scatter(stable, mask=mask)
    selected, metrics = _select_and_validate(
        stable_map,
        mask=mask,
        mean_image=mean_image,
        hard_erode_px=hard_erode_px,
        gradient_percentile=gradient_percentile,
        score_percentile=score_percentile,
    )
    selected_idx = np.flatnonzero(selected.reshape(-1)[mask_flat])
    nulls = _null_scores(
        residual,
        fps=fps,
        band_min_hz=band_min_hz,
        band_max_hz=band_max_hz,
        mask=mask,
        random_samples=random_samples,
        rng=rng,
    )

    events: dict[str, np.ndarray | float | int] | None = None
    selected_trace = np.zeros(stack.shape[0], dtype=np.float64)
    selected_scores = np.zeros(0, dtype=np.float64)
    if selected_idx.size:
        selected_trace = np.median(residual[selected_idx], axis=0)
        selected_scores = stable[selected_idx]
        events = _extract_events(
            selected_trace,
            fps=fps,
            band_min_hz=band_min_hz,
            band_max_hz=band_max_hz,
            prominence_scale=event_prominence_scale,
        )
    metrics.update(_trace_spike_metrics(selected_trace))

    verdict = _build_verdict(
        metrics=metrics,
        nulls=nulls,
        selected_scores=selected_scores,
        events=events,
        min_component_pixels=8,
        min_participation_ratio=5.0,
        min_largest_fraction=0.30,
        min_boundary_distance_px=float(hard_erode_px),
        min_fraction_intervals_in_band=0.70,
        max_spike_fraction=0.01,
    )
    metrics["chunk_count"] = int(chunk_count)
    metrics["selected_score_p90"] = _safe_percentile(selected_scores, 90)
    return {
        "mean_image": mean_image,
        "score_raw_map": score_raw_map,
        "score_residual_map": score_residual_map,
        "stable_map": stable_map,
        "selected_mask": selected,
        "selected_trace": selected_trace,
        "metrics": metrics,
        "nulls": nulls,
        "events": events,
        "verdict": verdict,
    }


def _make_synthetic(kind: str, *, frames: int = 1200, height: int = 64, width: int = 64, fps: float = 60.0) -> tuple[np.ndarray, np.ndarray, float]:
    from scipy import signal

    rng = np.random.default_rng(1)
    time = np.arange(frames, dtype=np.float64) / float(fps)
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.hypot(yy - height / 2.0, xx - width / 2.0)
    mask = radius < 26
    base = 0.4 + 0.2 * np.exp(-((radius - 15.0) ** 2) / 40.0)
    base += 0.15 * (radius > 24)
    stack = np.repeat(base[None], frames, axis=0).astype(np.float32)
    stack += rng.normal(0.0, 0.01, stack.shape).astype(np.float32)
    stack += (0.02 * np.sin(2 * np.pi * 0.15 * time))[:, None, None].astype(np.float32)
    frequency = 2.4
    if kind == "interior":
        blob = np.exp(-((yy - 24) ** 2 + (xx - 30) ** 2) / 12.0)
        pulse = signal.square(2 * np.pi * frequency * time, duty=0.3) * 0.5 + 0.5
        stack += (0.06 * blob[None] * pulse[:, None, None]).astype(np.float32)
    elif kind == "boundary_motion":
        dx = 0.4 * np.sin(2 * np.pi * frequency * time)
        dy = 0.4 * np.cos(2 * np.pi * frequency * time)
        grad_y, grad_x = np.gradient(base)
        stack += (grad_x[None] * dx[:, None, None] + grad_y[None] * dy[:, None, None]).astype(np.float32)
    elif kind == "global_flicker":
        stack += (0.03 * np.sin(2 * np.pi * frequency * time))[:, None, None].astype(np.float32)
    else:
        raise ValueError(f"Unknown synthetic kind: {kind}")
    return stack, mask, float(fps)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (bool, str, int, float)) or value is None:
        return value
    return str(value)


def _run_selftest(args: argparse.Namespace) -> None:
    output_dir = args.output_prefix.parent
    ensure_output_dir(output_dir)
    results: dict[str, Any] = {}
    for kind in ("interior", "boundary_motion", "global_flicker"):
        stack, mask, fps = _make_synthetic(kind)
        out = run_probe(
            stack=stack,
            mask=mask,
            fps=fps,
            band_min_hz=float(args.band_min_hz),
            band_max_hz=float(args.band_max_hz),
            hard_erode_px=int(args.hard_erode_px),
            gradient_percentile=float(args.gradient_percentile),
            score_percentile=float(args.score_percentile),
            chunk_seconds=float(args.chunk_seconds),
            min_chunk_samples=int(args.min_chunk_samples),
            random_samples=int(args.random_samples),
            event_prominence_scale=float(args.event_prominence_scale),
            seed=int(args.seed),
        )
        figure_path = output_dir / f"localized_periodic_signal_selftest_{kind}.png"
        _write_figure(
            figure_path,
            mean_image=out["mean_image"],
            score_raw=out["score_raw_map"],
            score_residual=out["score_residual_map"],
            score_stable=out["stable_map"],
            selected=out["selected_mask"],
            mask=mask,
            nulls=out["nulls"],
            selected_trace=out["selected_trace"],
            events=out["events"],
            fps=fps,
            title=f"self-test: {kind}",
        )
        results[kind] = {"verdict": out["verdict"], "metrics": out["metrics"], "figure": str(figure_path)}
        print(f"{kind}: trust_event_series={out['verdict']['trust_event_series']} figure={figure_path}")
    summary_path = args.output_prefix.with_suffix(".selftest_summary.json")
    with summary_path.open("w") as handle:
        json.dump(_json_ready(results), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"selftest_summary_json: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a localized periodic luminance signal with nuisance controls.")
    parser.add_argument("--selftest", action="store_true", help="Run synthetic controls instead of loading a video.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--video", type=Path, default=None, help="Stabilized video to sample.")
    parser.add_argument("--roi-json", type=Path, default=None, help="ROI JSON written by draw_roi.py.")
    parser.add_argument("--roi", type=str, default=None, help="Stabilized ROI rectangle x,y,width,height.")
    parser.add_argument("--status-csv", type=Path, default=None)
    parser.add_argument("--mask-npz", type=Path, default=None, help="Optional NPZ whose roi_mask defines usable pixels.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--band-min-hz", type=float, default=1.5)
    parser.add_argument("--band-max-hz", type=float, default=3.5)
    parser.add_argument("--pad-px", type=int, default=4)
    parser.add_argument("--min-roi-mean-intensity", type=float, default=1.0)
    parser.add_argument("--duplicate-mean-tol", type=float, default=1e-6)
    parser.add_argument("--max-interpolated-gap-samples", type=int, default=5)
    parser.add_argument("--min-segment-samples", type=int, default=512)
    parser.add_argument("--hard-erode-px", type=int, default=5)
    parser.add_argument("--gradient-percentile", type=float, default=85.0)
    parser.add_argument("--score-percentile", type=float, default=90.0)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--min-chunk-samples", type=int, default=512)
    parser.add_argument("--random-samples", type=int, default=200)
    parser.add_argument("--event-prominence-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).with_name("outputs") / "localized_periodic_signal_probe",
    )
    args = parser.parse_args()

    if args.selftest:
        _run_selftest(args)
        return

    if args.video is None:
        raise ValueError("--video is required unless --selftest is used.")
    config = load_config(args.config)
    roi_rect = resolve_roi_rect(config, roi=args.roi, roi_json=args.roi_json)
    roi_polygon = roi_rect_corners(roi_rect)
    sample_mask = None
    if args.mask_npz is not None:
        sample_mask = _mask_from_npz(args.mask_npz, shape_hw=_video_shape(args.video))

    loaded = _load_roi_stack(
        video_path=args.video,
        roi_polygon=roi_polygon,
        status_csv=args.status_csv,
        frame_start=max(0, int(args.frame_start)),
        frame_count=max(0, int(args.frame_count)),
        stride=max(1, int(args.stride)),
        sample_mask=sample_mask,
        min_roi_mean_intensity=float(args.min_roi_mean_intensity),
        pad_px=max(0, int(args.pad_px)),
        duplicate_mean_tol=float(args.duplicate_mean_tol),
    )
    stack, finite_valid, segment, interpolated = _prepare_stack(
        loaded["stack"],
        loaded["valid"],
        max_interpolated_gap=max(0, int(args.max_interpolated_gap_samples)),
        min_segment_samples=max(32, int(args.min_segment_samples)),
    )
    segment_start, segment_stop = segment
    frame_indices = loaded["frame_indices"][segment_start:segment_stop]
    effective_fps = float(args.fps) / max(1, int(args.stride))
    mask = loaded["candidate_crop_mask"]
    out = run_probe(
        stack=stack,
        mask=mask,
        fps=effective_fps,
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
        hard_erode_px=int(args.hard_erode_px),
        gradient_percentile=float(args.gradient_percentile),
        score_percentile=float(args.score_percentile),
        chunk_seconds=float(args.chunk_seconds),
        min_chunk_samples=int(args.min_chunk_samples),
        random_samples=int(args.random_samples),
        event_prominence_scale=float(args.event_prominence_scale),
        seed=int(args.seed),
    )

    output_prefix = args.output_prefix
    ensure_output_dir(output_prefix.parent)
    figure_path = output_prefix.with_suffix(".diagnostic.png")
    summary_path = output_prefix.with_suffix(".summary.json")
    events_path = output_prefix.with_suffix(".events.csv")
    mask_path = output_prefix.with_suffix(".selected_mask.npz")

    _write_figure(
        figure_path,
        mean_image=out["mean_image"],
        score_raw=out["score_raw_map"],
        score_residual=out["score_residual_map"],
        score_stable=out["stable_map"],
        selected=out["selected_mask"],
        mask=mask,
        nulls=out["nulls"],
        selected_trace=out["selected_trace"],
        events=out["events"],
        fps=effective_fps,
        title="localized periodic signal probe",
    )
    if out["events"] is not None:
        _write_events_csv(
            events_path,
            events=out["events"],
            frame_indices=loaded["frame_indices"],
            segment_start_row=int(segment_start),
            stride=max(1, int(args.stride)),
        )
    np.savez_compressed(
        mask_path,
        candidate_mask=mask.astype(np.uint8),
        selected_mask=out["selected_mask"].astype(np.uint8),
        bbox_xyxy=np.asarray(loaded["bbox_xyxy"], dtype=np.int32),
        score_raw=out["score_raw_map"].astype(np.float32),
        score_residual=out["score_residual_map"].astype(np.float32),
        score_stable=out["stable_map"].astype(np.float32),
    )

    event_rates = (
        np.asarray(out["events"]["event_rate_per_min"], dtype=np.float64)
        if out["events"] is not None
        else np.zeros(0, dtype=np.float64)
    )
    finite_rates = event_rates[np.isfinite(event_rates)]
    summary = {
        "source_video": str(args.video),
        "roi_json": str(args.roi_json) if args.roi_json is not None else None,
        "mask_npz": str(args.mask_npz) if args.mask_npz is not None else None,
        "frame_start": int(args.frame_start),
        "frame_count_requested": int(args.frame_count),
        "loaded_frame_count": int(loaded["stack"].shape[0]),
        "valid_loaded_frame_count": int(np.count_nonzero(loaded["valid"])),
        "duplicate_frame_count": int(loaded["duplicate_frame_count"]),
        "low_intensity_frame_count": int(loaded["low_intensity_frame_count"]),
        "interpolated_short_gap_frame_count": int(interpolated),
        "analysis_segment_rows": [int(segment_start), int(segment_stop)],
        "analysis_frame_indices": [int(frame_indices[0]), int(frame_indices[-1])],
        "analysis_frame_count": int(stack.shape[0]),
        "bbox_xyxy": [int(value) for value in loaded["bbox_xyxy"]],
        "candidate_pixel_count": int(np.count_nonzero(mask)),
        "fps": float(effective_fps),
        "band_hz": [float(args.band_min_hz), float(args.band_max_hz)],
        "hard_erode_px": int(args.hard_erode_px),
        "gradient_percentile": float(args.gradient_percentile),
        "score_percentile": float(args.score_percentile),
        "metrics": out["metrics"],
        "verdict": out["verdict"],
        "event_count": int(np.asarray(out["events"]["peaks"]).size) if out["events"] is not None else 0,
        "event_rate_per_min_median": float(np.median(finite_rates)) if finite_rates.size else math.nan,
        "event_rate_per_min_mean": float(np.mean(finite_rates)) if finite_rates.size else math.nan,
        "outputs": {
            "diagnostic_png": str(figure_path),
            "summary_json": str(summary_path),
            "events_csv": str(events_path) if out["events"] is not None else None,
            "selected_mask_npz": str(mask_path),
        },
    }
    with summary_path.open("w") as handle:
        json.dump(_json_ready(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"summary_json: {summary_path}")
    print(f"diagnostic_png: {figure_path}")
    print(f"selected_mask_npz: {mask_path}")
    if out["events"] is not None:
        print(f"events_csv: {events_path}")
    print(f"trust_event_series: {out['verdict']['trust_event_series']}")
    print(f"largest_size: {out['metrics'].get('largest_size', 0)}")
    print(f"selected_pixels: {out['metrics'].get('selected_pixel_count', 0)}")
    print(f"event_count: {summary['event_count']}")
    if finite_rates.size:
        print(f"event_rate_per_min_median: {summary['event_rate_per_min_median']:.3f}")


if __name__ == "__main__":
    main()
