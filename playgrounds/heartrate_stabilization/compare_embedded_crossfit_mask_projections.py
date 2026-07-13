from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from analyze_embedded_crossfit_mask import (
    _bandpass,
    _fit_scale,
    _peak_frequency,
    heldout_event_intervals,
    temporal_half_partitions,
)
from analyze_embedded_positive_control import (
    _extract_video,
    _load_roi,
    _reference_at_times,
    _ridge,
    _safe_correlation,
    read_numeric_xlsx_row,
)


_METHODS = (
    "masked_discovery_loading",
    "masked_refit_pca",
    "masked_equal_mean",
)


@dataclass(frozen=True)
class ProjectionSummary:
    method: str
    crossfit_ridge_correlation: float
    crossfit_ridge_mae_bpm: float
    crossfit_event_count: int
    crossfit_valid_event_interval_count: int
    crossfit_event_interval_cv: float
    crossfit_event_rate_mae_bpm: float
    fold_confirmation_peak_hz: tuple[float, float]
    fold_event_polarities: tuple[int, int]


def projection_weights(
    filtered_pixels: np.ndarray,
    discovery_rows: np.ndarray,
    mask: np.ndarray,
    discovery_loading: np.ndarray,
) -> dict[str, np.ndarray]:
    values = np.asarray(filtered_pixels, dtype=np.float64)
    rows = np.asarray(discovery_rows, dtype=bool)
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    base = np.asarray(discovery_loading, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or selected.shape != (values.shape[1],) or base.shape != selected.shape:
        raise ValueError("projection inputs have inconsistent pixel dimensions")
    if int(np.count_nonzero(selected)) < 3 or int(np.count_nonzero(rows)) < 3:
        raise ValueError("projection comparison requires at least three mask pixels and rows")

    masked_loading = base.copy()
    masked_loading[~selected] = 0.0
    loading_norm = float(np.linalg.norm(masked_loading))
    if loading_norm <= np.finfo(float).eps:
        raise ValueError("masked discovery loading has zero norm")
    masked_loading /= loading_norm

    pca = PCA(n_components=1, svd_solver="randomized", random_state=0)
    with threadpool_limits(limits=1):
        pca.fit(values[rows][:, selected])
    refit = np.zeros(values.shape[1], dtype=np.float64)
    refit[selected] = pca.components_[0]
    refit /= max(float(np.linalg.norm(refit)), np.finfo(float).eps)

    equal = np.zeros(values.shape[1], dtype=np.float64)
    equal[selected] = 1.0 / float(np.count_nonzero(selected))
    return {
        "masked_discovery_loading": masked_loading,
        "masked_refit_pca": refit,
        "masked_equal_mean": equal,
    }


def _robust_trace_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    center = float(np.median(finite))
    return max(1.4826 * float(np.median(np.abs(finite - center))), 1e-9)


def _plot(
    output: Path,
    *,
    masks: np.ndarray,
    traces: dict[str, np.ndarray],
    ridge_times: dict[str, np.ndarray],
    ridge_hz: dict[str, np.ndarray],
    event_rate_hz: dict[str, np.ndarray],
    reference_hz: np.ndarray,
    reference_fps: float,
    fps: float,
) -> None:
    colors = {
        "masked_discovery_loading": "tab:blue",
        "masked_refit_pca": "tab:green",
        "masked_equal_mean": "tab:orange",
    }
    labels = {
        "masked_discovery_loading": "discovery loading",
        "masked_refit_pca": "within-mask PCA refit",
        "masked_equal_mean": "within-mask equal mean",
    }
    time = np.arange(next(iter(traces.values())).size, dtype=np.float64) / fps
    reference_time = np.arange(reference_hz.size, dtype=np.float64) / reference_fps
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    overlap = np.zeros((*masks.shape[1:], 3), dtype=np.float64)
    overlap[..., 0] = masks[0]
    overlap[..., 2] = masks[1]
    axes[0, 0].imshow(overlap)
    axes[0, 0].set(title="Frozen fold masks: fold 0 red, fold 1 blue")
    axes[0, 0].axis("off")

    for offset, method in enumerate(_METHODS):
        trace = traces[method] / _robust_trace_scale(traces[method]) + 4.0 * offset
        axes[0, 1].plot(time, trace, color=colors[method], linewidth=0.7, label=labels[method])
    axes[0, 1].axvspan(time[len(time) // 2] - 1.0, time[len(time) // 2] + 1.0, color="gray", alpha=0.2)
    axes[0, 1].set(title="Held-out traces, robust-scaled and vertically offset", xlabel="Time (s)")
    axes[0, 1].legend()

    axes[1, 0].plot(reference_time, reference_hz * 60.0, color="black", linewidth=1.5, label="side reference")
    for method in _METHODS:
        axes[1, 0].plot(
            ridge_times[method],
            ridge_hz[method] * 60.0,
            marker="o",
            color=colors[method],
            label=labels[method],
        )
    axes[1, 0].set(title="Held-out 8 s spectral ridge", xlabel="Time (s)", ylabel="Rate (bpm)")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(reference_time, reference_hz * 60.0, color="black", linewidth=1.5, label="side reference")
    for method in _METHODS:
        axes[1, 1].plot(
            time,
            event_rate_hz[method] * 60.0,
            color=colors[method],
            linewidth=0.9,
            label=labels[method],
        )
    axes[1, 1].set(title="Held-out event-interval rates", xlabel="Time (s)", ylabel="Rate (bpm)")
    axes[1, 1].legend(fontsize=8)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare three opposite-half projections over frozen embedded masks."
    )
    parser.add_argument("--roi-json", type=Path, required=True)
    parser.add_argument("--mask-summary", type=Path, required=True)
    parser.add_argument("--mask-arrays", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, default=1)
    parser.add_argument("--reference-fps", type=float, default=100.0)
    parser.add_argument("--event-prominence-mad", type=float, default=0.5)
    parser.add_argument("--event-filter-edge-seconds", type=float, default=0.75)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    mask_summary = json.loads(args.mask_summary.read_text())
    mask_arrays = np.load(args.mask_arrays)
    video, roi_xywh = _load_roi(args.roi_json)
    pixels, _, _duplicates, _frame, fps, _frame_shape = _extract_video(
        video, roi_xywh, ring_margin=8
    )
    if Path(mask_summary["source_video"]).resolve() != video.resolve():
        raise ValueError("mask summary belongs to a different source video")
    masks = np.asarray(mask_arrays["fold_masks"], dtype=bool).reshape(2, -1)
    discovery_loadings = np.asarray(mask_arrays["fold_loadings"], dtype=np.float64).reshape(2, -1)
    if masks.shape != (2, pixels.shape[1]) or discovery_loadings.shape != masks.shape:
        raise ValueError("frozen masks/loadings do not match decoded ROI pixels")
    band_hz = tuple(float(value) for value in mask_summary["band_hz"])
    partitions = temporal_half_partitions(
        pixels.shape[0],
        fps=fps,
        guard_seconds=float(mask_summary["guard_seconds"]),
    )

    filtered_by_fold: list[np.ndarray] = []
    weights_by_method = {method: [] for method in _METHODS}
    # All three weighting rules are frozen before the supplied reference is loaded.
    for fold_index, (discovery_rows, _confirmation_rows) in enumerate(partitions):
        standardized, _scale, _usable = _fit_scale(pixels, discovery_rows)
        filtered = _bandpass(standardized, fps=fps, band_hz=band_hz)
        weights = projection_weights(
            filtered,
            discovery_rows,
            masks[fold_index],
            discovery_loadings[fold_index],
        )
        filtered_by_fold.append(filtered)
        for method in _METHODS:
            weights_by_method[method].append(weights[method])

    reference_hz = read_numeric_xlsx_row(
        args.workbook,
        sheet_name="heart_rate_trace",
        row_number=int(args.trial_number) + 1,
    )
    traces = {method: np.full(pixels.shape[0], np.nan, dtype=np.float64) for method in _METHODS}
    event_rates = {method: np.full(pixels.shape[0], np.nan, dtype=np.float64) for method in _METHODS}
    ridge_times_by_method: dict[str, np.ndarray] = {}
    ridge_hz_by_method: dict[str, np.ndarray] = {}
    summaries: list[ProjectionSummary] = []
    weight_array = np.zeros((len(_METHODS), 2, pixels.shape[1]), dtype=np.float64)

    for method_index, method in enumerate(_METHODS):
        ridge_times: list[np.ndarray] = []
        ridge_values: list[np.ndarray] = []
        ridge_references: list[np.ndarray] = []
        event_frames: list[np.ndarray] = []
        interval_end_frames: list[np.ndarray] = []
        interval_values: list[np.ndarray] = []
        fold_peaks: list[float] = []
        fold_polarities: list[int] = []
        for fold_index, (discovery_rows, confirmation_rows) in enumerate(partitions):
            weights = weights_by_method[method][fold_index]
            weight_array[method_index, fold_index] = weights
            trace = filtered_by_fold[fold_index] @ weights
            confirmation_indices = np.flatnonzero(confirmation_rows)
            traces[method][confirmation_rows] = trace[confirmation_rows]
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
            ridge_times.append(absolute_time)
            ridge_values.append(local_ridge)
            ridge_references.append(local_reference)
            fold_peaks.append(_peak_frequency(local_trace, fps=fps, band_hz=band_hz))
            events, ends, interval_hz, rate_trace, polarity = heldout_event_intervals(
                trace,
                discovery_rows,
                confirmation_rows,
                fps=fps,
                band_hz=band_hz,
                prominence_mad=float(args.event_prominence_mad),
                edge_seconds=float(args.event_filter_edge_seconds),
            )
            finite_rate = np.isfinite(rate_trace)
            event_rates[method][finite_rate] = rate_trace[finite_rate]
            event_frames.append(events)
            interval_end_frames.append(ends)
            interval_values.append(interval_hz)
            fold_polarities.append(polarity)

        all_ridge_time = np.concatenate(ridge_times)
        all_ridge_hz = np.concatenate(ridge_values)
        all_ridge_reference = np.concatenate(ridge_references)
        order = np.argsort(all_ridge_time)
        all_ridge_time = all_ridge_time[order]
        all_ridge_hz = all_ridge_hz[order]
        all_ridge_reference = all_ridge_reference[order]
        ridge_times_by_method[method] = all_ridge_time
        ridge_hz_by_method[method] = all_ridge_hz
        all_events = np.concatenate(event_frames)
        all_ends = np.concatenate(interval_end_frames)
        all_intervals_hz = np.concatenate(interval_values)
        event_reference = _reference_at_times(
            reference_hz,
            all_ends.astype(np.float64) / fps,
            float(args.reference_fps),
        )
        interval_seconds = 1.0 / all_intervals_hz
        summaries.append(
            ProjectionSummary(
                method=method,
                crossfit_ridge_correlation=_safe_correlation(all_ridge_hz, all_ridge_reference),
                crossfit_ridge_mae_bpm=float(
                    60.0 * np.mean(np.abs(all_ridge_hz - all_ridge_reference))
                ),
                crossfit_event_count=int(all_events.size),
                crossfit_valid_event_interval_count=int(all_intervals_hz.size),
                crossfit_event_interval_cv=float(
                    np.std(interval_seconds) / np.mean(interval_seconds)
                ),
                crossfit_event_rate_mae_bpm=float(
                    60.0 * np.mean(np.abs(all_intervals_hz - event_reference))
                ),
                fold_confirmation_peak_hz=(float(fold_peaks[0]), float(fold_peaks[1])),
                fold_event_polarities=(int(fold_polarities[0]), int(fold_polarities[1])),
            )
        )

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = prefix.with_suffix(".arrays.npz")
    plot_path = prefix.with_suffix(".diagnostic.png")
    summary_path = prefix.with_suffix(".summary.json")
    np.savez_compressed(
        arrays_path,
        method_names=np.asarray(_METHODS),
        fold_masks=masks.reshape(2, roi_xywh[3], roi_xywh[2]),
        fold_projection_weights=weight_array.reshape(len(_METHODS), 2, roi_xywh[3], roi_xywh[2]),
        **{f"trace_{method}": traces[method] for method in _METHODS},
        **{f"ridge_time_s_{method}": ridge_times_by_method[method] for method in _METHODS},
        **{f"ridge_hz_{method}": ridge_hz_by_method[method] for method in _METHODS},
        **{f"event_rate_hz_{method}": event_rates[method] for method in _METHODS},
    )
    _plot(
        plot_path,
        masks=masks.reshape(2, roi_xywh[3], roi_xywh[2]),
        traces=traces,
        ridge_times=ridge_times_by_method,
        ridge_hz=ridge_hz_by_method,
        event_rate_hz=event_rates,
        reference_hz=reference_hz,
        reference_fps=float(args.reference_fps),
        fps=fps,
    )
    pairwise_correlations: dict[str, float] = {}
    for left_index, left in enumerate(_METHODS):
        for right in _METHODS[left_index + 1 :]:
            common = np.isfinite(traces[left]) & np.isfinite(traces[right])
            pairwise_correlations[f"{left}__{right}"] = _safe_correlation(
                traces[left][common], traces[right][common]
            )
    payload = {
        "analysis_status": "exploratory_frozen_mask_crossfit_projection_comparison",
        "source_video": str(video.resolve()),
        "roi_json": str(args.roi_json.resolve()),
        "mask_summary": str(args.mask_summary.resolve()),
        "mask_arrays": str(args.mask_arrays.resolve()),
        "workbook": str(args.workbook.resolve()),
        "reference_use": "loaded only after all fold-specific projection weights were frozen",
        "methods": {
            "masked_discovery_loading": "existing block-median PCA loading, zero outside frozen mask",
            "masked_refit_pca": "PCA refit on frozen mask pixels in discovery half",
            "masked_equal_mean": "equal-weight mean of frozen mask pixels using discovery scaling",
        },
        "event_prominence_mad": float(args.event_prominence_mad),
        "event_filter_edge_seconds": float(args.event_filter_edge_seconds),
        "summaries": [asdict(summary) for summary in summaries],
        "pairwise_trace_correlations": pairwise_correlations,
        "arrays_npz": str(arrays_path),
        "diagnostic_png": str(plot_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
