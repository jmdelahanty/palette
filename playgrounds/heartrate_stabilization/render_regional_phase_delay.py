from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.local_rostral_heartrate import LocalCoordinateDataset
from fisheye.analysis.regional_phase_delay import RegionalPhaseDelayResult


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _scatter_pixels(
    dataset: LocalCoordinateDataset,
    values: np.ndarray,
    *,
    fill: float = np.nan,
) -> np.ndarray:
    image = np.full(dataset.image_shape_hw, fill, dtype=np.float64)
    xy = np.rint(np.asarray(dataset.pixel_xy, dtype=np.float64)).astype(np.int64)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < dataset.image_shape_hw[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < dataset.image_shape_hw[0])
    )
    image[xy[inside, 1], xy[inside, 0]] = np.asarray(values, dtype=np.float64)[inside]
    return image


def _robust_scale(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    median = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - median)))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(finite))
    return scale if np.isfinite(scale) and scale > np.finfo(float).eps else 1.0


def _json_summary(result: RegionalPhaseDelayResult) -> dict[str, Any]:
    cycle_lags = np.asarray(
        [float(row["lower_minus_upper_ms"]) for row in result.cycle_rows],
        dtype=np.float64,
    )
    return {
        "diagnostic_only": True,
        "event_validation": False,
        "region_source": str(result.region_source),
        "regions_independent": bool(result.regions_independent),
        "interpretation": str(result.interpretation),
        "frequency_hz": float(result.frequency_hz),
        "period_ms": float(result.period_ms),
        "split_y": float(result.split_y),
        "split_gap_px": float(result.split_gap_px),
        "upper_pixel_count": int(np.count_nonzero(result.upper_pixels)),
        "lower_pixel_count": int(np.count_nonzero(result.lower_pixels)),
        "phase_valid_frame_count": int(np.count_nonzero(result.frame_valid)),
        "phase_valid_fraction": float(np.mean(result.frame_valid)),
        "block_count": int(len(result.block_summary)),
        "paired_cycle_count": int(len(result.cycle_rows)),
        "median_cycle_lower_lag_ms": float(np.median(cycle_lags))
        if cycle_lags.size
        else math.nan,
        "cycle_lower_lag_mad_ms": float(
            np.median(np.abs(cycle_lags - np.median(cycle_lags)))
        )
        if cycle_lags.size
        else math.nan,
        "across_block_mean_phase_deg_lower_minus_upper": float(
            result.across_block_mean_phase_deg
        ),
        "across_block_lower_lag_ms": float(result.across_block_lower_lag_ms),
        "across_block_phase_locking_value": float(
            result.across_block_phase_locking_value
        ),
        "median_within_block_phase_locking_value": float(
            result.median_within_block_phase_locking_value
        ),
        "stable_delay_score": float(result.stable_delay_score),
        "stable_delay_p_value": float(result.stable_delay_p_value),
        "stable_delay_exceeds_null": bool(result.stable_delay_exceeds_null),
        "surrogate_count": int(result.null_stable_delay_scores.size),
        "lag_sign_contract": "positive lower_lag_ms means the lower region reaches the same phase after the upper region",
        "null_contract": "independent random phase rotation of each held-out lower-region block; conditional on the frozen support, frequency, filtering, and regional split",
        "caveat": "Narrow-band filtering creates phase cycles. This analysis tests repeatability of the regional delay, not whether the source is cardiac.",
    }


def _write_figure(
    path: Path,
    dataset: LocalCoordinateDataset,
    result: RegionalPhaseDelayResult,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    relative = timestamps - float(timestamps[0])
    mean_image = _scatter_pixels(dataset, np.nanmedian(dataset.traces, axis=0))
    upper_image = _scatter_pixels(dataset, result.upper_pixels.astype(np.float64), fill=0.0) > 0.5
    lower_image = _scatter_pixels(dataset, result.lower_pixels.astype(np.float64), fill=0.0) > 0.5
    support = upper_image | lower_image
    yy, xx = np.nonzero(support)
    if xx.size:
        margin = 2
        x0, x1 = max(0, int(xx.min()) - margin), min(mean_image.shape[1], int(xx.max()) + margin + 1)
        y0, y1 = max(0, int(yy.min()) - margin), min(mean_image.shape[0], int(yy.max()) + margin + 1)
    else:
        x0, x1, y0, y1 = 0, mean_image.shape[1], 0, mean_image.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True, facecolor="white")
    axis = axes[0, 0]
    axis.imshow(mean_image, cmap="gray", interpolation="nearest")
    overlay = np.zeros((*mean_image.shape, 4), dtype=np.float64)
    overlay[upper_image] = (0.85, 0.18, 0.16, 0.58)
    overlay[lower_image] = (0.12, 0.35, 0.82, 0.58)
    axis.imshow(overlay, interpolation="nearest")
    if np.isfinite(result.split_y):
        axis.axhline(result.split_y, color="white", lw=1.2, ls="--")
    axis.set_xlim(x0 - 0.5, x1 - 0.5)
    axis.set_ylim(y1 - 0.5, y0 - 0.5)
    axis.set_title(
        f"Frozen regions: upper red ({np.count_nonzero(result.upper_pixels)} px), "
        f"lower blue ({np.count_nonzero(result.lower_pixels)} px)"
    )
    axis.set_axis_off()

    upper_real = np.asarray(result.upper_analytic.real, dtype=np.float64)
    lower_real = np.asarray(result.lower_analytic.real, dtype=np.float64)
    upper_plot = upper_real / _robust_scale(upper_real)
    lower_plot = lower_real / _robust_scale(lower_real)
    axes[0, 1].plot(relative, upper_plot, color="#c9362e", lw=0.8, label="upper")
    axes[0, 1].plot(relative, lower_plot, color="#2458b8", lw=0.8, label="lower")
    axes[0, 1].axhline(0.0, color="0.75", lw=0.8)
    axes[0, 1].set_title("Cross-fitted regional analytic traces")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("real component / robust scale")
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot(relative, result.lower_lag_ms, color="black", lw=0.8)
    axes[0, 2].axhline(0.0, color="0.6", lw=0.8, ls="--")
    axes[0, 2].set_ylim(-0.5 * result.period_ms, 0.5 * result.period_ms)
    axes[0, 2].set_title("Instantaneous lower-region lag")
    axes[0, 2].set_xlabel("time (s)")
    axes[0, 2].set_ylabel("lower minus upper (ms)")

    block_indices = np.arange(len(result.block_summary), dtype=np.int64)
    block_lags = np.asarray(
        [float(row["mean_lower_lag_ms"]) for row in result.block_summary],
        dtype=np.float64,
    )
    block_sd = np.asarray(
        [float(row["circular_sd_deg"]) for row in result.block_summary],
        dtype=np.float64,
    ) / 360.0 * result.period_ms
    axes[1, 0].errorbar(
        block_indices,
        block_lags,
        yerr=block_sd,
        fmt="o",
        color="black",
        ecolor="0.55",
        capsize=3,
        label="block circular mean +/- circular SD",
    )
    for row in result.cycle_rows:
        axes[1, 0].scatter(
            int(row["block_index"]),
            float(row["lower_minus_upper_ms"]),
            color="#3576c4",
            s=12,
            alpha=0.55,
        )
    axes[1, 0].axhline(0.0, color="0.65", lw=0.8, ls="--")
    axes[1, 0].set_title("Block and paired-cycle delays")
    axes[1, 0].set_xlabel("held-out block")
    axes[1, 0].set_ylabel("lower minus upper (ms)")
    axes[1, 0].legend(fontsize=7)

    within = [float(row["phase_locking_value"]) for row in result.block_summary]
    upper_coherence = [
        float(row["median_upper_spatial_coherence"]) for row in result.block_summary
    ]
    lower_coherence = [
        float(row["median_lower_spatial_coherence"]) for row in result.block_summary
    ]
    axes[1, 1].plot(block_indices, within, marker="o", color="black", label="regional phase locking")
    axes[1, 1].plot(block_indices, upper_coherence, marker="o", color="#c9362e", label="upper spatial coherence")
    axes[1, 1].plot(block_indices, lower_coherence, marker="o", color="#2458b8", label="lower spatial coherence")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_title("Within-block repeatability")
    axes[1, 1].set_xlabel("held-out block")
    axes[1, 1].legend(fontsize=7)

    axes[1, 2].hist(
        result.null_stable_delay_scores,
        bins=20,
        color="0.75",
        edgecolor="0.4",
    )
    axes[1, 2].axvline(result.stable_delay_score, color="black", lw=1.5)
    lines = [
        f"mean lower lag: {result.across_block_lower_lag_ms:.1f} ms",
        f"across-block PLV: {result.across_block_phase_locking_value:.3f}",
        f"median within-block PLV: {result.median_within_block_phase_locking_value:.3f}",
        f"stable-delay p: {result.stable_delay_p_value:.3g}",
        f"independent regions: {result.regions_independent}",
        result.interpretation,
    ]
    axes[1, 2].text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=axes[1, 2].transAxes,
        va="top",
        fontsize=8,
        family="monospace",
    )
    axes[1, 2].set_title("Block-phase randomization null")
    axes[1, 2].set_xlabel("stable-delay score")
    fig.suptitle(
        f"Upper-to-lower phase delay at {result.frequency_hz:.3f} Hz | "
        "positive lag means lower follows upper\n"
        "Conditional narrow-band diagnostic; not cardiac-event validation",
        fontsize=12,
    )
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def write_regional_phase_delay_outputs(
    output_prefix: Path,
    dataset: LocalCoordinateDataset,
    result: RegionalPhaseDelayResult,
) -> dict[str, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_prefix.with_suffix(".regional_phase_delay.summary.json")
    blocks_path = output_prefix.with_suffix(".regional_phase_delay.blocks.csv")
    cycles_path = output_prefix.with_suffix(".regional_phase_delay.cycles.csv")
    arrays_path = output_prefix.with_suffix(".regional_phase_delay.arrays.npz")
    figure_path = output_prefix.with_suffix(".regional_phase_delay.diagnostic.png")
    summary_path.write_text(json.dumps(_json_summary(result), indent=2, sort_keys=True) + "\n")
    _write_rows(blocks_path, result.block_summary)
    _write_rows(cycles_path, result.cycle_rows)
    upper_image = _scatter_pixels(dataset, result.upper_pixels.astype(np.float64), fill=0.0) > 0.5
    lower_image = _scatter_pixels(dataset, result.lower_pixels.astype(np.float64), fill=0.0) > 0.5
    np.savez_compressed(
        arrays_path,
        frame_indices=np.asarray(dataset.frame_indices, dtype=np.int64),
        timestamps_s=np.asarray(dataset.timestamps_s, dtype=np.float64),
        pixel_xy=np.asarray(dataset.pixel_xy, dtype=np.float32),
        upper_pixels=np.asarray(result.upper_pixels, dtype=np.uint8),
        lower_pixels=np.asarray(result.lower_pixels, dtype=np.uint8),
        upper_mask=np.asarray(upper_image, dtype=np.uint8),
        lower_mask=np.asarray(lower_image, dtype=np.uint8),
        upper_analytic_real=np.asarray(result.upper_analytic.real, dtype=np.float32),
        upper_analytic_imag=np.asarray(result.upper_analytic.imag, dtype=np.float32),
        lower_analytic_real=np.asarray(result.lower_analytic.real, dtype=np.float32),
        lower_analytic_imag=np.asarray(result.lower_analytic.imag, dtype=np.float32),
        upper_spatial_coherence=np.asarray(result.upper_spatial_coherence, dtype=np.float32),
        lower_spatial_coherence=np.asarray(result.lower_spatial_coherence, dtype=np.float32),
        phase_offset_rad=np.asarray(result.phase_offset_rad, dtype=np.float32),
        lower_lag_ms=np.asarray(result.lower_lag_ms, dtype=np.float32),
        frame_valid=np.asarray(result.frame_valid, dtype=np.uint8),
        block_indices=np.asarray(result.block_indices, dtype=np.int16),
        null_stable_delay_scores=np.asarray(result.null_stable_delay_scores, dtype=np.float32),
        split_y=np.asarray(result.split_y, dtype=np.float32),
        split_gap_px=np.asarray(result.split_gap_px, dtype=np.float32),
    )
    _write_figure(figure_path, dataset, result)
    return {
        "regional_phase_delay_summary_json": summary_path,
        "regional_phase_delay_blocks_csv": blocks_path,
        "regional_phase_delay_cycles_csv": cycles_path,
        "regional_phase_delay_arrays_npz": arrays_path,
        "regional_phase_delay_diagnostic_png": figure_path,
    }
