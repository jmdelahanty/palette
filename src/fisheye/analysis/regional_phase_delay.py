from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.dynamic_heart_support import CrossfitHeartPhaseSeries
from fisheye.analysis.local_rostral_heartrate import (
    LocalCoordinateDataset,
    contiguous_segments,
)


@dataclass(frozen=True)
class RegionalPhaseDelayResult:
    """Conditional top-to-bottom phase-delay diagnostics for a frozen support."""

    region_source: str
    regions_independent: bool
    interpretation: str
    frequency_hz: float
    period_ms: float
    split_y: float
    split_gap_px: float
    upper_pixels: np.ndarray
    lower_pixels: np.ndarray
    upper_analytic: np.ndarray
    lower_analytic: np.ndarray
    upper_spatial_coherence: np.ndarray
    lower_spatial_coherence: np.ndarray
    phase_offset_rad: np.ndarray
    lower_lag_ms: np.ndarray
    frame_valid: np.ndarray
    block_indices: np.ndarray
    block_summary: tuple[Mapping[str, Any], ...]
    cycle_rows: tuple[Mapping[str, Any], ...]
    across_block_mean_phase_deg: float
    across_block_lower_lag_ms: float
    across_block_phase_locking_value: float
    median_within_block_phase_locking_value: float
    stable_delay_score: float
    stable_delay_p_value: float
    stable_delay_exceeds_null: bool
    null_stable_delay_scores: np.ndarray


def _balanced_horizontal_regions(
    dataset: LocalCoordinateDataset,
    support: np.ndarray,
    *,
    split_y: float | None,
    split_gap_px: float,
    min_region_pixels: int,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    selected = np.asarray(support, dtype=bool)
    if selected.shape != (dataset.pixel_count,):
        raise ValueError("support shape does not match dataset pixels")
    y = np.asarray(dataset.pixel_xy, dtype=np.float64)[:, 1]
    if split_y is None:
        unique_y = np.unique(y[selected & np.isfinite(y)])
        candidates = 0.5 * (unique_y[:-1] + unique_y[1:])
        viable: list[tuple[int, float, int, int]] = []
        for candidate in candidates:
            upper_count = int(np.count_nonzero(selected & (y < candidate)))
            lower_count = int(np.count_nonzero(selected & (y > candidate)))
            if upper_count >= int(min_region_pixels) and lower_count >= int(min_region_pixels):
                viable.append(
                    (
                        abs(upper_count - lower_count),
                        float(candidate),
                        upper_count,
                        lower_count,
                    )
                )
        if not viable:
            raise ValueError("could not find a balanced horizontal support split")
        _imbalance, resolved_split, _upper_count, _lower_count = min(
            viable,
            key=lambda item: (item[0], abs(item[1] - float(np.median(y[selected])))),
        )
        source = "mask_geometry_balanced_horizontal_split"
    else:
        resolved_split = float(split_y)
        source = "explicit_horizontal_split"
    gap = float(split_gap_px)
    if gap < 0.0:
        raise ValueError("split_gap_px cannot be negative")
    upper = selected & (y < resolved_split - 0.5 * gap)
    lower = selected & (y > resolved_split + 0.5 * gap)
    if int(np.count_nonzero(upper)) < int(min_region_pixels):
        raise ValueError("upper region has too few pixels")
    if int(np.count_nonzero(lower)) < int(min_region_pixels):
        raise ValueError("lower region has too few pixels")
    return upper, lower, resolved_split, source


def _validate_explicit_regions(
    support: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    *,
    min_region_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(support, dtype=bool)
    upper_pixels = np.asarray(upper, dtype=bool).copy()
    lower_pixels = np.asarray(lower, dtype=bool).copy()
    if upper_pixels.shape != selected.shape or lower_pixels.shape != selected.shape:
        raise ValueError("explicit regional masks must match the dataset pixel axis")
    upper_pixels &= selected
    lower_pixels &= selected
    if np.any(upper_pixels & lower_pixels):
        raise ValueError("upper and lower masks must be disjoint")
    if int(np.count_nonzero(upper_pixels)) < int(min_region_pixels):
        raise ValueError("explicit upper region has too few support pixels")
    if int(np.count_nonzero(lower_pixels)) < int(min_region_pixels):
        raise ValueError("explicit lower region has too few support pixels")
    return upper_pixels, lower_pixels


def _regional_analytic(
    phase: CrossfitHeartPhaseSeries,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    region = np.asarray(selected, dtype=bool)
    output = np.full(phase.frame_valid.shape, np.nan + 0j, dtype=np.complex128)
    coherence = np.full(phase.frame_valid.shape, np.nan, dtype=np.float64)
    for row, fold_index in enumerate(np.asarray(phase.model_fold_indices, dtype=np.int64)):
        if fold_index < 0:
            continue
        values = np.asarray(phase.analytic_residual[row], dtype=np.complex128)
        weights = np.asarray(phase.fold_loading_weights[fold_index], dtype=np.float64)
        valid = region & np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if int(np.count_nonzero(valid)) < 3:
            continue
        denominator = float(np.sum(weights[valid]))
        if denominator <= np.finfo(float).eps:
            continue
        weighted = weights[valid] * values[valid]
        output[row] = np.sum(weighted) / denominator
        absolute_denominator = float(np.sum(weights[valid] * np.abs(values[valid])))
        if absolute_denominator > np.finfo(float).eps:
            coherence[row] = float(np.abs(np.sum(weighted)) / absolute_denominator)
    return output, coherence


def _circular_summary(angles_rad: np.ndarray) -> tuple[float, float, float]:
    angles = np.asarray(angles_rad, dtype=np.float64)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return math.nan, math.nan, math.nan
    vector = np.mean(np.exp(1j * angles))
    locking = float(np.abs(vector))
    circular_sd = float(math.sqrt(max(0.0, -2.0 * math.log(max(locking, 1e-12)))))
    return float(np.angle(vector)), locking, circular_sd


def _phase_crossings(
    timestamps_s: np.ndarray,
    analytic: np.ndarray,
) -> np.ndarray:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    values = np.asarray(analytic, dtype=np.complex128)
    finite = np.isfinite(values) & np.isfinite(timestamps)
    if int(np.count_nonzero(finite)) < 3:
        return np.zeros(0, dtype=np.float64)
    timestamps = timestamps[finite]
    unwrapped = np.unwrap(np.angle(values[finite]))
    start_level = int(math.ceil(float(np.min(unwrapped)) / (2.0 * np.pi)))
    stop_level = int(math.floor(float(np.max(unwrapped)) / (2.0 * np.pi)))
    crossings: list[float] = []
    for level_index in range(start_level, stop_level + 1):
        level = float(level_index) * 2.0 * np.pi
        indices = np.flatnonzero((unwrapped[:-1] < level) & (unwrapped[1:] >= level))
        if indices.size == 0:
            continue
        index = int(indices[0])
        phase_step = float(unwrapped[index + 1] - unwrapped[index])
        if phase_step <= np.finfo(float).eps:
            continue
        fraction = (level - float(unwrapped[index])) / phase_step
        crossings.append(
            float(timestamps[index] + fraction * (timestamps[index + 1] - timestamps[index]))
        )
    return np.asarray(crossings, dtype=np.float64)


def _pair_cycle_crossings(
    upper_times: np.ndarray,
    lower_times: np.ndarray,
    *,
    maximum_lag_s: float,
) -> list[tuple[float, float, float]]:
    upper = np.asarray(upper_times, dtype=np.float64)
    lower = np.asarray(lower_times, dtype=np.float64)
    used = np.zeros(lower.size, dtype=bool)
    output: list[tuple[float, float, float]] = []
    for upper_time in upper:
        available = np.flatnonzero(~used)
        if available.size == 0:
            break
        local = int(np.argmin(np.abs(lower[available] - upper_time)))
        lower_index = int(available[local])
        delay = float(lower[lower_index] - upper_time)
        if abs(delay) > float(maximum_lag_s):
            continue
        used[lower_index] = True
        output.append((float(upper_time), float(lower[lower_index]), delay))
    return output


def analyze_regional_phase_delay(
    dataset: LocalCoordinateDataset,
    phase: CrossfitHeartPhaseSeries,
    *,
    upper_pixels: np.ndarray | None = None,
    lower_pixels: np.ndarray | None = None,
    split_y: float | None = None,
    split_gap_px: float = 0.0,
    regions_independent: bool = False,
    min_region_pixels: int = 3,
    surrogate_count: int = 199,
    alpha: float = 0.05,
    max_gap_factor: float = 1.75,
    seed: int = 0,
) -> RegionalPhaseDelayResult:
    """Measure held-out upper-to-lower phase lag conditional on a detected band."""

    dataset.validated()
    support = np.asarray(phase.heart_support, dtype=bool)
    if (upper_pixels is None) != (lower_pixels is None):
        raise ValueError("upper_pixels and lower_pixels must be supplied together")
    if upper_pixels is None:
        upper, lower, resolved_split, region_source = _balanced_horizontal_regions(
            dataset,
            support,
            split_y=split_y,
            split_gap_px=float(split_gap_px),
            min_region_pixels=int(min_region_pixels),
        )
    else:
        upper, lower = _validate_explicit_regions(
            support,
            upper_pixels,
            lower_pixels,
            min_region_pixels=int(min_region_pixels),
        )
        resolved_split = math.nan
        region_source = "external_explicit_region_masks"
    frequency = float(phase.frequency_hz)
    if not frequency > 0.0:
        raise ValueError("phase frequency must be positive")
    period_s = 1.0 / frequency
    upper_analytic, upper_coherence = _regional_analytic(phase, upper)
    lower_analytic, lower_coherence = _regional_analytic(phase, lower)
    phase_offset = np.angle(lower_analytic * np.conjugate(upper_analytic))
    lower_lag_ms = -phase_offset / (2.0 * np.pi * frequency) * 1000.0
    frame_valid = (
        np.asarray(phase.frame_valid, dtype=bool)
        & np.isfinite(upper_analytic)
        & np.isfinite(lower_analytic)
        & np.isfinite(upper_coherence)
        & np.isfinite(lower_coherence)
    )
    phase_offset[~frame_valid] = np.nan
    lower_lag_ms[~frame_valid] = np.nan
    minimum_block_seconds = max(2.0 * period_s, 0.5)
    blocks = contiguous_segments(
        dataset.timestamps_s,
        frame_valid,
        max_gap_factor=float(max_gap_factor),
        min_seconds=minimum_block_seconds,
    )
    block_indices = np.full(dataset.frame_count, -1, dtype=np.int16)
    block_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    block_mean_angles: list[float] = []
    block_locking: list[float] = []
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    for block_index, rows in enumerate(blocks):
        block_indices[rows] = int(block_index)
        mean_angle, locking, circular_sd = _circular_summary(phase_offset[rows])
        block_mean_angles.append(mean_angle)
        block_locking.append(locking)
        block_lags = lower_lag_ms[rows]
        mean_lag_ms = -mean_angle / (2.0 * np.pi * frequency) * 1000.0
        sign = 1.0 if mean_lag_ms >= 0.0 else -1.0
        direction_fraction = float(np.mean(sign * block_lags >= 0.0))
        upper_crossings = _phase_crossings(timestamps[rows], upper_analytic[rows])
        lower_crossings = _phase_crossings(timestamps[rows], lower_analytic[rows])
        pairs = _pair_cycle_crossings(
            upper_crossings,
            lower_crossings,
            maximum_lag_s=0.5 * period_s,
        )
        paired_lags_ms = np.asarray([pair[2] * 1000.0 for pair in pairs], dtype=np.float64)
        for cycle_index, (upper_time, lower_time, delay_s) in enumerate(pairs):
            cycle_rows.append(
                {
                    "block_index": int(block_index),
                    "cycle_index": int(cycle_index),
                    "upper_crossing_s": float(upper_time),
                    "lower_crossing_s": float(lower_time),
                    "lower_minus_upper_ms": float(delay_s * 1000.0),
                }
            )
        block_rows.append(
            {
                "block_index": int(block_index),
                "start_s": float(timestamps[rows[0]]),
                "stop_s": float(timestamps[rows[-1]]),
                "frame_count": int(rows.size),
                "mean_phase_offset_deg_lower_minus_upper": float(np.degrees(mean_angle)),
                "mean_lower_lag_ms": float(mean_lag_ms),
                "phase_locking_value": float(locking),
                "circular_sd_deg": float(np.degrees(circular_sd)),
                "same_direction_frame_fraction": direction_fraction,
                "median_upper_spatial_coherence": float(np.nanmedian(upper_coherence[rows])),
                "median_lower_spatial_coherence": float(np.nanmedian(lower_coherence[rows])),
                "median_upper_amplitude": float(np.nanmedian(np.abs(upper_analytic[rows]))),
                "median_lower_amplitude": float(np.nanmedian(np.abs(lower_analytic[rows]))),
                "paired_cycle_count": int(len(pairs)),
                "median_cycle_lower_lag_ms": float(np.median(paired_lags_ms))
                if paired_lags_ms.size
                else math.nan,
                "cycle_lower_lag_mad_ms": float(
                    np.median(np.abs(paired_lags_ms - np.median(paired_lags_ms)))
                )
                if paired_lags_ms.size
                else math.nan,
            }
        )
    mean_phase, across_locking, _across_sd = _circular_summary(
        np.asarray(block_mean_angles, dtype=np.float64)
    )
    median_within = (
        float(np.median(np.asarray(block_locking, dtype=np.float64)))
        if block_locking
        else 0.0
    )
    observed_score = float(across_locking * median_within) if np.isfinite(across_locking) else 0.0
    count = int(surrogate_count)
    if count < 0:
        raise ValueError("surrogate_count cannot be negative")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be between zero and one")
    rng = np.random.default_rng(int(seed))
    null_scores = np.zeros(count, dtype=np.float64)
    block_angles = np.asarray(block_mean_angles, dtype=np.float64)
    if block_angles.size >= 3:
        for surrogate_index in range(count):
            rotations = rng.uniform(-np.pi, np.pi, block_angles.size)
            _mean, randomized_locking, _sd = _circular_summary(block_angles + rotations)
            null_scores[surrogate_index] = float(randomized_locking * median_within)
        p_value = (
            float(1 + np.count_nonzero(null_scores >= observed_score)) / float(count + 1)
            if count
            else 1.0
        )
        threshold = (
            float(np.quantile(null_scores, 1.0 - float(alpha), method="higher"))
            if count
            else math.inf
        )
        exceeds = bool(count and p_value <= float(alpha) and observed_score > threshold)
    else:
        p_value = 1.0
        threshold = math.inf
        exceeds = False
    if not regions_independent:
        interpretation = "exploratory_regions_were_not_independently_prespecified"
    elif len(block_rows) < 3:
        interpretation = "insufficient_heldout_blocks_for_regional_delay"
    elif not exceeds:
        interpretation = "regional_delay_not_stable_above_block_phase_null"
    else:
        interpretation = "stable_regional_delay_above_block_phase_null"
    return RegionalPhaseDelayResult(
        region_source=region_source,
        regions_independent=bool(regions_independent),
        interpretation=interpretation,
        frequency_hz=frequency,
        period_ms=period_s * 1000.0,
        split_y=float(resolved_split),
        split_gap_px=float(split_gap_px),
        upper_pixels=upper,
        lower_pixels=lower,
        upper_analytic=upper_analytic,
        lower_analytic=lower_analytic,
        upper_spatial_coherence=upper_coherence,
        lower_spatial_coherence=lower_coherence,
        phase_offset_rad=phase_offset,
        lower_lag_ms=lower_lag_ms,
        frame_valid=frame_valid,
        block_indices=block_indices,
        block_summary=tuple(block_rows),
        cycle_rows=tuple(cycle_rows),
        across_block_mean_phase_deg=float(np.degrees(mean_phase)),
        across_block_lower_lag_ms=float(
            -mean_phase / (2.0 * np.pi * frequency) * 1000.0
        ),
        across_block_phase_locking_value=float(across_locking),
        median_within_block_phase_locking_value=median_within,
        stable_delay_score=observed_score,
        stable_delay_p_value=p_value,
        stable_delay_exceeds_null=exceeds,
        null_stable_delay_scores=null_scores,
    )
