from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from analyze_embedded_positive_control import (
    _extract_video,
    _event_train,
    _load_roi,
    _reference_at_times,
    _ridge,
    _safe_correlation,
    read_numeric_xlsx_row,
)


@dataclass(frozen=True)
class MaskFold:
    fold_index: int
    discovery_interval_s: tuple[float, float]
    confirmation_interval_s: tuple[float, float]
    discovery_block_count: int
    selected_pixel_count: int
    cluster_score_mass: float
    confirmation_peak_hz: float
    confirmation_ridge_correlation: float
    confirmation_ridge_mae_bpm: float
    event_polarity: int
    event_count: int
    valid_event_interval_count: int
    event_interval_cv: float
    event_rate_mae_bpm: float


def temporal_half_partitions(
    frame_count: int,
    *,
    fps: float,
    guard_seconds: float,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if frame_count < 4 or fps <= 0.0 or guard_seconds < 0.0:
        raise ValueError("invalid temporal partition parameters")
    midpoint = frame_count // 2
    guard = int(np.ceil(float(guard_seconds) * float(fps)))
    left = np.zeros(frame_count, dtype=bool)
    right = np.zeros(frame_count, dtype=bool)
    left[: max(0, midpoint - guard)] = True
    right[min(frame_count, midpoint + guard) :] = True
    if not np.any(left) or not np.any(right):
        raise ValueError("partition guard removed an entire temporal half")
    return ((left, right), (right, left))


def _complete_blocks(rows: np.ndarray, *, fps: float, block_seconds: float) -> tuple[np.ndarray, ...]:
    selected = np.flatnonzero(np.asarray(rows, dtype=bool))
    block_size = int(round(float(block_seconds) * float(fps)))
    if block_size < 3 or selected.size < block_size:
        return ()
    splits = np.split(selected, np.flatnonzero(np.diff(selected) > 1) + 1)
    blocks: list[np.ndarray] = []
    for run in splits:
        for start in range(0, run.size - block_size + 1, block_size):
            blocks.append(run[start : start + block_size])
    return tuple(blocks)


def _fit_scale(pixels: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(pixels, dtype=np.float64)
    training = values[np.asarray(rows, dtype=bool)]
    center = np.median(training, axis=0)
    scale = 1.4826 * np.median(np.abs(training - center), axis=0)
    usable = np.isfinite(scale) & (scale > 1e-6)
    if int(np.count_nonzero(usable)) < 3:
        raise ValueError("fewer than three varying discovery pixels")
    standardized = np.zeros_like(values)
    standardized[:, usable] = (values[:, usable] - center[usable]) / scale[usable]
    return standardized, scale, usable


def _bandpass(values: np.ndarray, *, fps: float, band_hz: tuple[float, float]) -> np.ndarray:
    sos = butter(3, band_hz, btype="bandpass", fs=float(fps), output="sos")
    return sosfiltfilt(sos, np.asarray(values, dtype=np.float64), axis=0)


def _aligned_block_loadings(
    filtered: np.ndarray,
    blocks: tuple[np.ndarray, ...],
    usable: np.ndarray,
) -> np.ndarray:
    loadings: list[np.ndarray] = []
    anchor: np.ndarray | None = None
    for rows in blocks:
        pca = PCA(n_components=1, svd_solver="randomized", random_state=0)
        with threadpool_limits(limits=1):
            pca.fit(filtered[rows][:, usable])
        loading = np.zeros(filtered.shape[1], dtype=np.float64)
        loading[usable] = pca.components_[0]
        if anchor is not None and float(np.dot(anchor, loading)) < 0.0:
            loading *= -1.0
        if anchor is None:
            anchor = loading.copy()
        else:
            anchor += loading
        loadings.append(loading)
    if not loadings:
        raise ValueError("discovery partition has no complete blocks")
    return np.stack(loadings, axis=0)


def select_loading_cluster(
    block_loadings: np.ndarray,
    *,
    roi_shape_hw: tuple[int, int],
    score_threshold_z: float,
    min_cluster_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    loadings = np.asarray(block_loadings, dtype=np.float64)
    height, width = roi_shape_hw
    if loadings.ndim != 2 or loadings.shape[1] != height * width:
        raise ValueError("block loadings do not match ROI shape")
    magnitude = np.median(np.abs(loadings), axis=0)
    median = float(np.median(magnitude))
    scale = max(1.4826 * float(np.median(np.abs(magnitude - median))), 1e-9)
    scores = (magnitude - median) / scale
    supra = (scores >= float(score_threshold_z)).reshape(height, width)
    labels, count = ndimage.label(supra, structure=np.ones((3, 3), dtype=np.uint8))
    best_mask = np.zeros((height, width), dtype=bool)
    best_mass = 0.0
    for label in range(1, count + 1):
        component = labels == label
        if int(np.count_nonzero(component)) < int(min_cluster_pixels):
            continue
        mass = float(np.sum(np.maximum(scores.reshape(height, width)[component], 0.0)))
        if mass > best_mass:
            best_mask = component
            best_mass = mass
    selected = best_mask.reshape(-1)
    signed_loading = np.median(loadings, axis=0)
    signed_loading[~selected] = 0.0
    norm = float(np.linalg.norm(signed_loading))
    if norm > np.finfo(float).eps:
        signed_loading /= norm
    return selected, scores, signed_loading, best_mass


def _dilated_overlap(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if not np.any(a) or not np.any(b):
        return 0.0
    a_image = ndimage.binary_dilation(a, iterations=1)
    b_image = ndimage.binary_dilation(b, iterations=1)
    a_supported = np.count_nonzero(a & b_image) / float(np.count_nonzero(a))
    b_supported = np.count_nonzero(b & a_image) / float(np.count_nonzero(b))
    return float(min(a_supported, b_supported))


def _peak_frequency(signal: np.ndarray, *, fps: float, band_hz: tuple[float, float]) -> float:
    frequencies, power = welch(
        np.asarray(signal, dtype=np.float64),
        fs=float(fps),
        nperseg=min(len(signal), int(round(8.0 * fps))),
    )
    inside = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    return float(frequencies[inside][np.argmax(power[inside])])


def segmented_welch(signal: np.ndarray, *, fps: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=np.float64)
    finite_rows = np.flatnonzero(np.isfinite(values))
    if finite_rows.size < 3:
        raise ValueError("segmented Welch needs at least three finite samples")
    runs = np.split(finite_rows, np.flatnonzero(np.diff(finite_rows) > 1) + 1)
    runs = [run for run in runs if run.size >= 3]
    if not runs:
        raise ValueError("segmented Welch has no usable contiguous run")
    nperseg = min(1600, min(run.size for run in runs))
    spectra: list[np.ndarray] = []
    frequencies: np.ndarray | None = None
    for run in runs:
        local_frequency, local_power = welch(values[run], fs=fps, nperseg=nperseg)
        if frequencies is None:
            frequencies = local_frequency
        elif not np.array_equal(frequencies, local_frequency):
            raise AssertionError("segmented Welch frequency grids differ")
        spectra.append(local_power)
    assert frequencies is not None
    return frequencies, np.mean(np.stack(spectra, axis=0), axis=0)


def heldout_event_intervals(
    trace: np.ndarray,
    discovery_rows: np.ndarray,
    confirmation_rows: np.ndarray,
    *,
    fps: float,
    band_hz: tuple[float, float],
    prominence_mad: float,
    edge_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    values = np.asarray(trace, dtype=np.float64)
    discovery = np.flatnonzero(np.asarray(discovery_rows, dtype=bool))
    confirmation = np.flatnonzero(np.asarray(confirmation_rows, dtype=bool))
    if discovery.size < 3 or confirmation.size < 3:
        raise ValueError("event detection requires non-empty discovery and confirmation rows")
    if np.any(np.diff(discovery) != 1) or np.any(np.diff(confirmation) != 1):
        raise ValueError("event detection expects contiguous temporal halves")
    training = values[discovery]
    center = float(np.median(training))
    scale = max(1.4826 * float(np.median(np.abs(training - center))), 1e-9)
    _training_peaks, polarity, _training_cv = _event_train(
        training,
        fps=float(fps),
        max_hz=float(band_hz[1]),
    )
    standardized = int(polarity) * (values[confirmation] - center) / scale
    local_peaks, _properties = find_peaks(
        standardized,
        distance=max(1, int(np.floor(float(fps) / float(band_hz[1])))),
        prominence=float(prominence_mad),
    )
    edge = int(np.ceil(float(edge_seconds) * float(fps)))
    local_peaks = local_peaks[
        (local_peaks >= edge) & (local_peaks < confirmation.size - edge)
    ]
    event_frames = confirmation[local_peaks]
    intervals = np.diff(event_frames).astype(np.float64) / float(fps)
    valid = (
        (intervals >= 1.0 / float(band_hz[1]))
        & (intervals <= 1.0 / float(band_hz[0]))
    )
    interval_end_frames = event_frames[1:][valid]
    interval_hz = 1.0 / intervals[valid]
    rate_trace = np.full(values.size, np.nan, dtype=np.float64)
    for start, stop, is_valid, interval in zip(
        event_frames[:-1], event_frames[1:], valid, intervals
    ):
        if is_valid:
            rate_trace[int(start) : int(stop)] = 1.0 / float(interval)
    return event_frames, interval_end_frames, interval_hz, rate_trace, int(polarity)


def _plot(
    output: Path,
    *,
    frame: np.ndarray,
    roi_xywh: tuple[int, int, int, int],
    masks: np.ndarray,
    scores: np.ndarray,
    crossfit_trace: np.ndarray,
    ridge_time_s: np.ndarray,
    ridge_bpm: np.ndarray,
    reference_hz: np.ndarray,
    reference_fps: float,
    fps: float,
) -> None:
    x, y, width, height = roi_xywh
    preview = frame.copy()
    cv2.rectangle(preview, (x, y), (x + width - 1, y + height - 1), (0, 0, 255), 2)
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    time = np.arange(crossfit_trace.size, dtype=np.float64) / fps
    reference_time = np.arange(reference_hz.size, dtype=np.float64) / reference_fps

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes[0, 0].imshow(preview)
    axes[0, 0].set(title="Embedded top camera and fixed ROI")
    axes[0, 0].axis("off")
    overlap = np.zeros((height, width, 3), dtype=np.float64)
    overlap[..., 0] = masks[0].reshape(height, width)
    overlap[..., 2] = masks[1].reshape(height, width)
    axes[0, 1].imshow(overlap)
    axes[0, 1].set(title="Cross-fit masks: fold 0 red, fold 1 blue")
    axes[0, 1].axis("off")
    score_limit = max(float(np.nanquantile(np.abs(scores), 0.99)), 1.0)
    axes[0, 2].imshow(
        np.mean(scores, axis=0).reshape(height, width),
        cmap="coolwarm",
        vmin=-score_limit,
        vmax=score_limit,
    )
    axes[0, 2].set(title="Mean fold loading score")
    axes[0, 2].axis("off")
    axes[1, 0].plot(time, crossfit_trace, linewidth=0.8)
    axes[1, 0].axvline(time.size / (2.0 * fps), color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(title="Cross-fitted held-out projection", xlabel="Time (s)")
    axes[1, 1].plot(reference_time, reference_hz * 60.0, color="black", label="side reference")
    axes[1, 1].plot(ridge_time_s, ridge_bpm, marker="o", label="top cross-fit ridge")
    axes[1, 1].set(title="Reference used only after mask discovery", xlabel="Time (s)", ylabel="Rate (bpm)")
    axes[1, 1].legend()
    frequencies, power = segmented_welch(crossfit_trace, fps=fps)
    axes[1, 2].plot(frequencies, power)
    axes[1, 2].set(xlim=(1.0, 4.5), title="Mean contiguous-fold spectrum", xlabel="Frequency (Hz)")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover a cross-fitted embedded top-camera heart support without using the rate workbook."
    )
    parser.add_argument("--roi-json", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, default=1)
    parser.add_argument("--reference-fps", type=float, default=100.0)
    parser.add_argument("--band-hz", type=float, nargs=2, default=(1.5, 4.0))
    parser.add_argument("--guard-seconds", type=float, default=1.0)
    parser.add_argument("--block-seconds", type=float, default=4.0)
    parser.add_argument("--score-threshold-z", type=float, default=1.5)
    parser.add_argument("--min-cluster-pixels", type=int, default=3)
    parser.add_argument("--event-prominence-mad", type=float, default=0.5)
    parser.add_argument("--event-filter-edge-seconds", type=float, default=0.75)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    video, roi_xywh = _load_roi(args.roi_json)
    pixels, _, duplicates, first_frame, fps, frame_shape = _extract_video(
        video, roi_xywh, ring_margin=8
    )
    band_hz = (float(args.band_hz[0]), float(args.band_hz[1]))
    partitions = temporal_half_partitions(
        pixels.shape[0], fps=fps, guard_seconds=float(args.guard_seconds)
    )
    roi_shape = (int(roi_xywh[3]), int(roi_xywh[2]))
    masks: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    loadings: list[np.ndarray] = []
    filtered_by_fold: list[np.ndarray] = []
    block_counts: list[int] = []
    masses: list[float] = []

    # No workbook values are loaded until both spatial masks and loadings are frozen.
    for discovery_rows, _confirmation_rows in partitions:
        standardized, _scale, usable = _fit_scale(pixels, discovery_rows)
        filtered = _bandpass(standardized, fps=fps, band_hz=band_hz)
        blocks = _complete_blocks(
            discovery_rows, fps=fps, block_seconds=float(args.block_seconds)
        )
        block_loading = _aligned_block_loadings(filtered, blocks, usable)
        mask, score, loading, mass = select_loading_cluster(
            block_loading,
            roi_shape_hw=roi_shape,
            score_threshold_z=float(args.score_threshold_z),
            min_cluster_pixels=int(args.min_cluster_pixels),
        )
        masks.append(mask)
        scores.append(score)
        loadings.append(loading)
        filtered_by_fold.append(filtered)
        block_counts.append(len(blocks))
        masses.append(mass)

    reference_hz = read_numeric_xlsx_row(
        args.workbook,
        sheet_name="heart_rate_trace",
        row_number=int(args.trial_number) + 1,
    )
    crossfit_trace = np.full(pixels.shape[0], np.nan, dtype=np.float64)
    ridge_times: list[np.ndarray] = []
    ridge_rates: list[np.ndarray] = []
    ridge_references: list[np.ndarray] = []
    event_frames_by_fold: list[np.ndarray] = []
    event_interval_end_frames: list[np.ndarray] = []
    event_interval_rates: list[np.ndarray] = []
    event_rate_trace = np.full(pixels.shape[0], np.nan, dtype=np.float64)
    fold_summaries: list[MaskFold] = []
    for fold_index, ((discovery_rows, confirmation_rows), mask, loading, filtered) in enumerate(
        zip(partitions, masks, loadings, filtered_by_fold)
    ):
        if np.any(mask):
            trace = filtered @ loading
            crossfit_trace[confirmation_rows] = trace[confirmation_rows]
            confirmation_indices = np.flatnonzero(confirmation_rows)
            local_trace = trace[confirmation_indices]
            local_time, local_ridge = _ridge(
                local_trace,
                fps=fps,
                band_hz=band_hz,
                window_seconds=8.0,
                step_seconds=1.0,
            )
            absolute_time = local_time + confirmation_indices[0] / fps
            local_reference = _reference_at_times(
                reference_hz, absolute_time, float(args.reference_fps)
            )
            peak_hz = _peak_frequency(local_trace, fps=fps, band_hz=band_hz)
            correlation = _safe_correlation(local_ridge, local_reference)
            mae = float(60.0 * np.mean(np.abs(local_ridge - local_reference)))
            ridge_times.append(absolute_time)
            ridge_rates.append(local_ridge)
            ridge_references.append(local_reference)
            (
                fold_event_frames,
                fold_interval_end_frames,
                fold_interval_hz,
                fold_event_rate_trace,
                event_polarity,
            ) = heldout_event_intervals(
                trace,
                discovery_rows,
                confirmation_rows,
                fps=fps,
                band_hz=band_hz,
                prominence_mad=float(args.event_prominence_mad),
                edge_seconds=float(args.event_filter_edge_seconds),
            )
            event_rate_trace[np.isfinite(fold_event_rate_trace)] = fold_event_rate_trace[
                np.isfinite(fold_event_rate_trace)
            ]
            event_frames_by_fold.append(fold_event_frames)
            event_interval_end_frames.append(fold_interval_end_frames)
            event_interval_rates.append(fold_interval_hz)
            fold_event_reference = _reference_at_times(
                reference_hz,
                fold_interval_end_frames.astype(np.float64) / fps,
                float(args.reference_fps),
            )
            fold_event_mae = (
                float(60.0 * np.mean(np.abs(fold_interval_hz - fold_event_reference)))
                if fold_interval_hz.size
                else float("nan")
            )
            fold_event_intervals = 1.0 / fold_interval_hz
            fold_event_cv = (
                float(np.std(fold_event_intervals) / np.mean(fold_event_intervals))
                if fold_event_intervals.size >= 2
                else float("nan")
            )
        else:
            peak_hz = float("nan")
            correlation = float("nan")
            mae = float("nan")
            event_polarity = 0
            fold_event_frames = np.empty(0, dtype=np.int64)
            fold_interval_hz = np.empty(0, dtype=np.float64)
            fold_event_mae = float("nan")
            fold_event_cv = float("nan")
        discovery_indices = np.flatnonzero(discovery_rows)
        confirmation_indices = np.flatnonzero(confirmation_rows)
        fold_summaries.append(
            MaskFold(
                fold_index=fold_index,
                discovery_interval_s=(discovery_indices[0] / fps, discovery_indices[-1] / fps),
                confirmation_interval_s=(confirmation_indices[0] / fps, confirmation_indices[-1] / fps),
                discovery_block_count=block_counts[fold_index],
                selected_pixel_count=int(np.count_nonzero(mask)),
                cluster_score_mass=float(masses[fold_index]),
                confirmation_peak_hz=peak_hz,
                confirmation_ridge_correlation=correlation,
                confirmation_ridge_mae_bpm=mae,
                event_polarity=event_polarity,
                event_count=int(fold_event_frames.size),
                valid_event_interval_count=int(fold_interval_hz.size),
                event_interval_cv=fold_event_cv,
                event_rate_mae_bpm=fold_event_mae,
            )
        )

    all_ridge_time = np.concatenate(ridge_times) if ridge_times else np.empty(0)
    all_ridge_hz = np.concatenate(ridge_rates) if ridge_rates else np.empty(0)
    all_reference = np.concatenate(ridge_references) if ridge_references else np.empty(0)
    order = np.argsort(all_ridge_time)
    all_ridge_time = all_ridge_time[order]
    all_ridge_hz = all_ridge_hz[order]
    all_reference = all_reference[order]
    aggregate_correlation = _safe_correlation(all_ridge_hz, all_reference)
    aggregate_mae = (
        float(60.0 * np.mean(np.abs(all_ridge_hz - all_reference)))
        if all_ridge_hz.size
        else float("nan")
    )
    all_event_frames = (
        np.sort(np.concatenate(event_frames_by_fold))
        if event_frames_by_fold
        else np.empty(0, dtype=np.int64)
    )
    all_interval_end_frames = (
        np.concatenate(event_interval_end_frames)
        if event_interval_end_frames
        else np.empty(0, dtype=np.int64)
    )
    all_interval_hz = (
        np.concatenate(event_interval_rates)
        if event_interval_rates
        else np.empty(0, dtype=np.float64)
    )
    event_reference = _reference_at_times(
        reference_hz,
        all_interval_end_frames.astype(np.float64) / fps,
        float(args.reference_fps),
    )
    event_rate_mae = (
        float(60.0 * np.mean(np.abs(all_interval_hz - event_reference)))
        if all_interval_hz.size
        else float("nan")
    )
    event_intervals = 1.0 / all_interval_hz
    event_interval_cv = (
        float(np.std(event_intervals) / np.mean(event_intervals))
        if event_intervals.size >= 2
        else float("nan")
    )
    mask_array = np.stack(masks, axis=0)
    raw_overlap = (
        float(np.count_nonzero(mask_array[0] & mask_array[1]))
        / float(np.count_nonzero(mask_array[0] | mask_array[1]))
        if np.any(mask_array[0] | mask_array[1])
        else 0.0
    )

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = prefix.with_suffix(".arrays.npz")
    summary_path = prefix.with_suffix(".summary.json")
    plot_path = prefix.with_suffix(".diagnostic.png")
    np.savez_compressed(
        arrays_path,
        fold_masks=mask_array.reshape(2, *roi_shape),
        fold_scores=np.stack(scores, axis=0).reshape(2, *roi_shape),
        fold_loadings=np.stack(loadings, axis=0).reshape(2, *roi_shape),
        crossfit_trace=crossfit_trace,
        ridge_time_s=all_ridge_time,
        ridge_hz=all_ridge_hz,
        reference_hz_at_ridge=all_reference,
        event_frame_indices=all_event_frames,
        event_interval_end_frames=all_interval_end_frames,
        event_interval_hz=all_interval_hz,
        event_rate_trace_hz=event_rate_trace,
    )
    _plot(
        plot_path,
        frame=first_frame,
        roi_xywh=roi_xywh,
        masks=mask_array,
        scores=np.stack(scores, axis=0),
        crossfit_trace=crossfit_trace,
        ridge_time_s=all_ridge_time,
        ridge_bpm=all_ridge_hz * 60.0,
        reference_hz=reference_hz,
        reference_fps=float(args.reference_fps),
        fps=fps,
    )
    payload = {
        "analysis_status": "exploratory_reference_blind_crossfit_mask_positive_control",
        "source_video": str(video.resolve()),
        "roi_json": str(args.roi_json.resolve()),
        "roi_xywh": list(roi_xywh),
        "frame_shape_hw": list(frame_shape),
        "source_frames": int(pixels.shape[0]),
        "source_fps": float(fps),
        "duplicate_roi_frame_fraction": float(np.mean(duplicates[1:])),
        "band_hz": list(band_hz),
        "guard_seconds": float(args.guard_seconds),
        "block_seconds": float(args.block_seconds),
        "score_threshold_z": float(args.score_threshold_z),
        "min_cluster_pixels": int(args.min_cluster_pixels),
        "event_prominence_mad": float(args.event_prominence_mad),
        "event_filter_edge_seconds": float(args.event_filter_edge_seconds),
        "selection_contract": (
            "median absolute PCA loading across discovery-only blocks; robust spatial z threshold; "
            "strongest 8-connected component; empty selection allowed"
        ),
        "reference_use": "loaded only after both fold masks and signed loadings were frozen",
        "folds": [asdict(item) for item in fold_summaries],
        "raw_mask_jaccard": raw_overlap,
        "dilated_mask_overlap": _dilated_overlap(mask_array[0], mask_array[1]),
        "crossfit_ridge_correlation": aggregate_correlation,
        "crossfit_ridge_mae_bpm": aggregate_mae,
        "crossfit_ridge_point_count": int(all_ridge_hz.size),
        "crossfit_event_count": int(all_event_frames.size),
        "crossfit_valid_event_interval_count": int(all_interval_hz.size),
        "crossfit_event_interval_cv": event_interval_cv,
        "crossfit_event_rate_mae_bpm": event_rate_mae,
        "event_contract": (
            "polarity and robust scale learned on discovery half; fixed prominence/spacing on "
            "opposite half; 0.75 s confirmation edges rejected; intervals never cross guard"
        ),
        "arrays_npz": str(arrays_path),
        "diagnostic_png": str(plot_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
