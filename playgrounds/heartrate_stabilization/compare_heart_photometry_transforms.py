from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import spearmanr

from analyze_frozen_heart_masks_longitudinal import _read_mask, _window_dataset
from diagnose_frozen_mask_longitudinal_tracking import _mask_at_pixels
from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.heart_photometry_transforms import (
    masked_gaussian_smooth,
    normalized_signed_lag_difference,
    reference_normalize,
    regional_pool,
    regional_spatial_std,
    segmented_savgol_derivative,
)
from fisheye.analysis.local_rostral_heartrate import alternating_block_partitions


_INTERPRETATION = (
    "conditional_descriptive_transform_comparison_without_transform_family_null"
)


@dataclass(frozen=True)
class TraceSet:
    target: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    control: np.ndarray


@dataclass(frozen=True)
class WindowMetrics:
    spectral_ratio: float
    control_ratio: float
    phase_offset_deg: float
    phase_locking_value: float
    lag_cycle_fraction: float
    lag_ms: float
    tracking_spearman_r: float
    block_count: int
    control_block_count: int
    median_transform_uncertainty: float
    target_coefficients: np.ndarray
    control_coefficients: np.ndarray
    block_phase_offsets_rad: np.ndarray
    block_spectral_ratios: np.ndarray
    block_tracking_risk: np.ndarray


def _read_source_windows(path: Path, mask_name: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["mask"] == mask_name]
    if not rows:
        raise ValueError(f"no longitudinal rows found for {mask_name!r}")
    return sorted(rows, key=lambda row: int(row["window_index"]))


def _parse_int_list(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _finite_median(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else math.nan


def _finite_iqr(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return math.nan
    return float(np.quantile(array, 0.75) - np.quantile(array, 0.25))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if int(np.count_nonzero(valid)) < 4:
        return math.nan
    if np.ptp(first[valid]) <= np.finfo(float).eps:
        return math.nan
    if np.ptp(second[valid]) <= np.finfo(float).eps:
        return math.nan
    result = spearmanr(first[valid], second[valid])
    return float(result.statistic)


def _auto_reference_and_control_masks(
    original: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_outer = binary_dilation(np.asarray(original, dtype=bool), iterations=2)
    control_outer = binary_dilation(np.asarray(original, dtype=bool), iterations=5)
    reference = reference_outer & ~np.asarray(original, dtype=bool)
    control = control_outer & ~reference_outer
    return reference, control


def _validate_pixel_mask(name: str, mask: np.ndarray, *, minimum: int = 3) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(selected))
    if count < int(minimum):
        raise ValueError(f"{name} has only {count} cached pixels; need at least {minimum}")
    return selected


def _stored_mask_at_pixels(
    stored: np.ndarray,
    dataset: Any,
    *,
    name: str,
) -> np.ndarray:
    mask = np.asarray(stored, dtype=bool)
    if mask.shape == tuple(dataset.image_shape_hw):
        return _mask_at_pixels(mask, dataset.pixel_xy)
    if mask.shape == (int(dataset.pixel_count),):
        return mask.copy()
    raise ValueError(
        f"{name} shape {mask.shape} is neither the cached image shape "
        f"{dataset.image_shape_hw} nor pixel axis ({dataset.pixel_count},)"
    )


def _pool_trace_set(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    target: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    control: np.ndarray,
    method: str,
    min_valid_pixels: int,
) -> TraceSet:
    def pool(region: np.ndarray) -> np.ndarray:
        return regional_pool(
            values,
            region,
            valid=valid,
            method=method,
            trim_fraction=0.1,
            min_valid_pixels=min(min_valid_pixels, int(np.count_nonzero(region))),
        )

    return TraceSet(
        target=pool(target),
        upper=pool(upper),
        lower=pool(lower),
        control=pool(control),
    )


def _smooth_pool(
    values: np.ndarray,
    valid: np.ndarray,
    pixel_xy: np.ndarray,
    region: np.ndarray,
    *,
    sigma_px: float,
) -> np.ndarray:
    smoothed = masked_gaussian_smooth(
        values,
        pixel_xy,
        region,
        sigma_px=float(sigma_px),
        valid=valid,
    )
    return regional_pool(
        smoothed,
        region,
        valid=np.isfinite(smoothed),
        method="huber",
        min_valid_pixels=min(3, int(np.count_nonzero(region))),
    )


def _unary_trace_set(
    traces: TraceSet,
    operation: Callable[[np.ndarray], np.ndarray],
) -> TraceSet:
    return TraceSet(
        target=operation(traces.target),
        upper=operation(traces.upper),
        lower=operation(traces.lower),
        control=operation(traces.control),
    )


def _reference_trace_set(
    robust: TraceSet,
    reference: np.ndarray,
    *,
    mode: str,
    control_is_reference: bool,
) -> TraceSet:
    def normalize(values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values) & np.isfinite(reference)
        return reference_normalize(
            values,
            reference,
            mode=mode,
            valid=valid,
            epsilon=1e-6,
        )

    return TraceSet(
        target=normalize(robust.target),
        upper=normalize(robust.upper),
        lower=normalize(robust.lower),
        control=robust.control if control_is_reference else normalize(robust.control),
    )


def _candidate_traces(
    dataset: Any,
    *,
    target: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    reference: np.ndarray,
    control: np.ndarray,
    sg_windows: Sequence[int],
    lag_frames: Sequence[int],
    gaussian_sigma_px: float,
) -> dict[str, TraceSet]:
    values = np.asarray(dataset.traces, dtype=np.float64)
    valid = (
        np.asarray(dataset.pixel_valid, dtype=bool)
        & np.asarray(dataset.frame_valid, dtype=bool)[:, None]
        & np.isfinite(values)
    )
    mean = _pool_trace_set(
        values,
        valid,
        target=target,
        upper=upper,
        lower=lower,
        control=control,
        method="mean",
        min_valid_pixels=3,
    )
    robust = _pool_trace_set(
        values,
        valid,
        target=target,
        upper=upper,
        lower=lower,
        control=control,
        method="huber",
        min_valid_pixels=3,
    )
    reference_trace = regional_pool(
        values,
        reference,
        valid=valid,
        method="huber",
        min_valid_pixels=min(3, int(np.count_nonzero(reference))),
    )
    control_is_reference = np.array_equal(control, reference)
    gaussian = TraceSet(
        target=_smooth_pool(
            values, valid, dataset.pixel_xy, target, sigma_px=gaussian_sigma_px
        ),
        upper=_smooth_pool(
            values, valid, dataset.pixel_xy, upper, sigma_px=gaussian_sigma_px
        ),
        lower=_smooth_pool(
            values, valid, dataset.pixel_xy, lower, sigma_px=gaussian_sigma_px
        ),
        control=_smooth_pool(
            values, valid, dataset.pixel_xy, control, sigma_px=gaussian_sigma_px
        ),
    )
    spatial_std = TraceSet(
        target=regional_spatial_std(values, target, valid=valid, min_valid_pixels=3),
        upper=regional_spatial_std(values, upper, valid=valid, min_valid_pixels=3),
        lower=regional_spatial_std(values, lower, valid=valid, min_valid_pixels=3),
        control=regional_spatial_std(values, control, valid=valid, min_valid_pixels=3),
    )
    candidates = {
        "baseline_mean_intensity": mean,
        "robust_huber_intensity": robust,
        "reference_log_ratio": _reference_trace_set(
            robust,
            reference_trace,
            mode="log_ratio",
            control_is_reference=control_is_reference,
        ),
        "reference_fractional_difference": _reference_trace_set(
            robust,
            reference_trace,
            mode="fractional_difference",
            control_is_reference=control_is_reference,
        ),
        f"masked_gaussian_huber_sigma{gaussian_sigma_px:g}": gaussian,
        "regional_spatial_std": spatial_std,
    }
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    for window_length in sg_windows:
        candidates[f"huber_savgol_derivative_w{window_length}"] = _unary_trace_set(
            robust,
            lambda trace, length=window_length: segmented_savgol_derivative(
                trace,
                timestamps,
                valid=np.isfinite(trace),
                window_length=int(length),
                polyorder=min(2, int(length) - 1),
                max_gap_factor=1.75,
            ),
        )
    candidates[f"gaussian_savgol_derivative_w7_sigma{gaussian_sigma_px:g}"] = (
        _unary_trace_set(
            gaussian,
            lambda trace: segmented_savgol_derivative(
                trace,
                timestamps,
                valid=np.isfinite(trace),
                window_length=7,
                polyorder=2,
                max_gap_factor=1.75,
            ),
        )
    )
    for lag in lag_frames:
        candidates[f"huber_normalized_signed_lag{lag}"] = _unary_trace_set(
            robust,
            lambda trace, resolved_lag=lag: normalized_signed_lag_difference(
                trace,
                timestamps,
                lag_frames=int(resolved_lag),
                valid=np.isfinite(trace),
                max_gap_factor=1.75,
                alignment="center",
            ),
        )
    return candidates


def _nuisance_design(dataset: Any) -> np.ndarray:
    base = np.asarray(dataset.nuisance_values, dtype=np.float64)
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    derivatives = np.full_like(base, np.nan)
    if base.shape[1]:
        dt = np.diff(timestamps)
        valid_dt = np.isfinite(dt) & (dt > 0.0)
        derivatives[1:] = np.divide(
            np.diff(base, axis=0),
            dt[:, None],
            out=np.full_like(base[1:], np.nan),
            where=valid_dt[:, None],
        )
        derivatives[0] = derivatives[1] if base.shape[0] > 1 else 0.0
    return np.column_stack(
        [base, derivatives, np.asarray(dataset.transform_uncertainty, dtype=np.float64)]
    )


def _crossfit_nuisance_residual(
    trace: np.ndarray,
    dataset: Any,
    partitions: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    ridge: float,
) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64)
    design = _nuisance_design(dataset)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    frame_valid = np.asarray(dataset.frame_valid, dtype=bool)
    for discovery_rows, confirmation_rows in partitions:
        discovery = (
            np.asarray(discovery_rows, dtype=bool)
            & frame_valid
            & np.isfinite(values)
        )
        confirmation = (
            np.asarray(confirmation_rows, dtype=bool)
            & frame_valid
            & np.isfinite(values)
        )
        if int(np.count_nonzero(discovery)) < max(16, design.shape[1] + 3):
            continue
        center = np.nanmedian(design[discovery], axis=0)
        mad = np.nanmedian(np.abs(design[discovery] - center[None, :]), axis=0)
        scale = 1.4826 * mad
        fallback = np.nanstd(design[discovery], axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, fallback)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        standardized = (design - center[None, :]) / scale[None, :]
        standardized[~np.isfinite(standardized)] = 0.0
        matrix = np.column_stack([np.ones(values.size), standardized])
        fit_matrix = matrix[discovery]
        fit_values = values[discovery]
        penalty = np.eye(fit_matrix.shape[1], dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(
            fit_matrix.T @ fit_matrix + penalty,
            fit_matrix.T @ fit_values,
        )
        predicted = matrix @ beta
        output[confirmation] = values[confirmation] - predicted[confirmation]
    return output


def _crossfit_trace_set(
    traces: TraceSet,
    dataset: Any,
    partitions: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    ridge: float,
) -> TraceSet:
    return TraceSet(
        target=_crossfit_nuisance_residual(
            traces.target, dataset, partitions, ridge=ridge
        ),
        upper=_crossfit_nuisance_residual(
            traces.upper, dataset, partitions, ridge=ridge
        ),
        lower=_crossfit_nuisance_residual(
            traces.lower, dataset, partitions, ridge=ridge
        ),
        control=_crossfit_nuisance_residual(
            traces.control, dataset, partitions, ridge=ridge
        ),
    )


def _matched_projection_trace_set(
    dataset: Any,
    *,
    target: np.ndarray,
    control: np.ndarray,
    partitions: Sequence[tuple[np.ndarray, np.ndarray]],
    frequency_hz: float,
) -> TraceSet:
    from fisheye.analysis.heart_photometry_projection import (
        MatchedProjectionConfig,
        crossfit_matched_spatial_projection,
    )

    config = MatchedProjectionConfig().validated()
    target_result = crossfit_matched_spatial_projection(
        dataset,
        target,
        partitions,
        frequency_hz=float(frequency_hz),
        config=config,
    )
    control_result = crossfit_matched_spatial_projection(
        dataset,
        control,
        partitions,
        frequency_hz=float(frequency_hz),
        config=config,
    )
    target_trace = np.asarray(target_result.projected_trace, dtype=np.float64).copy()
    control_trace = np.asarray(control_result.projected_trace, dtype=np.float64).copy()
    target_trace[~np.asarray(target_result.frame_valid, dtype=bool)] = np.nan
    control_trace[~np.asarray(control_result.frame_valid, dtype=bool)] = np.nan
    unavailable = np.full(dataset.frame_count, np.nan, dtype=np.float64)
    return TraceSet(
        target=target_trace,
        upper=unavailable.copy(),
        lower=unavailable.copy(),
        control=control_trace,
    )


def _complex_coefficients(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    trace = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(timestamps) & np.isfinite(trace)
    if int(np.count_nonzero(valid)) < 8:
        return np.full(frequencies_hz.shape, np.nan + 0j, dtype=np.complex128)
    timestamps = timestamps[valid]
    trace = trace[valid]
    relative = timestamps - timestamps[0]
    trend = np.column_stack([np.ones(relative.size), relative])
    trace = trace - trend @ np.linalg.lstsq(trend, trace, rcond=None)[0]
    window = np.hanning(trace.size)
    denominator = float(np.sum(window))
    if denominator <= np.finfo(float).eps:
        return np.full(frequencies_hz.shape, np.nan + 0j, dtype=np.complex128)
    phase = np.exp(
        -2j * np.pi * np.asarray(frequencies_hz, dtype=np.float64)[:, None] * relative
    )
    return 2.0 * np.sum(phase * (window * trace)[None, :], axis=1) / denominator


def _logical_blocks(
    dataset: Any,
    traces: TraceSet,
    *,
    block_seconds: float,
    min_block_seconds: float,
    min_valid_fraction: float,
    max_interpolated_gap_seconds: float,
) -> list[np.ndarray]:
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    relative = timestamps - timestamps[0]
    block_ids = np.floor(relative / float(block_seconds)).astype(np.int64)
    common = np.isfinite(traces.target)
    if np.any(np.isfinite(traces.upper)) and np.any(np.isfinite(traces.lower)):
        common &= np.isfinite(traces.upper) & np.isfinite(traces.lower)
    output: list[np.ndarray] = []
    frame_valid = np.asarray(dataset.frame_valid, dtype=bool)
    for block_id in np.unique(block_ids):
        possible = block_ids == block_id
        base_valid_rows = np.flatnonzero(possible & frame_valid)
        if base_valid_rows.size < 8:
            continue
        # Derivative kernels deliberately widen NaN margins around an invalid raw
        # frame. Test the underlying frame stream for long gaps so those margins
        # do not disqualify every derivative candidate.
        gaps = np.diff(timestamps[base_valid_rows])
        nominal = float(np.median(np.diff(timestamps)))
        # Fourier coefficients use the real timestamps, so accepted missing rows do
        # not compress time. This bound mirrors a short-gap interpolation policy:
        # a missing duration of N*dt has a bounding timestamp gap of (N+1)*dt.
        maximum_bounding_gap = float(max_interpolated_gap_seconds) + 1.05 * nominal
        cuts = np.flatnonzero(gaps > maximum_bounding_gap) + 1
        for base_segment in np.split(base_valid_rows, cuts):
            if base_segment.size < 8:
                continue
            valid_rows = base_segment[common[base_segment]]
            if valid_rows.size < 8:
                continue
            if float(valid_rows.size / base_segment.size) < float(min_valid_fraction):
                continue
            duration = float(timestamps[valid_rows[-1]] - timestamps[valid_rows[0]])
            if duration < float(min_block_seconds):
                continue
            output.append(valid_rows)
    return output


def _measure_window(
    dataset: Any,
    traces: TraceSet,
    *,
    frequency_hz: float,
    frequency_min_hz: float,
    frequency_max_hz: float,
    frequency_step_hz: float,
    block_seconds: float,
    min_block_seconds: float,
    min_valid_fraction: float,
    max_interpolated_gap_seconds: float,
) -> WindowMetrics:
    grid = float(frequency_min_hz) + np.arange(
        int(round((float(frequency_max_hz) - float(frequency_min_hz)) / frequency_step_hz))
        + 1,
        dtype=np.float64,
    ) * float(frequency_step_hz)
    target_index = int(np.argmin(np.abs(grid - float(frequency_hz))))
    if abs(float(grid[target_index] - frequency_hz)) > 0.51 * float(frequency_step_hz):
        raise ValueError("frozen frequency is outside the configured spectral grid")
    sideband = np.abs(grid - float(grid[target_index])) >= 0.2 - 1e-12
    blocks = _logical_blocks(
        dataset,
        traces,
        block_seconds=float(block_seconds),
        min_block_seconds=float(min_block_seconds),
        min_valid_fraction=float(min_valid_fraction),
        max_interpolated_gap_seconds=float(max_interpolated_gap_seconds),
    )
    target_coefficients: list[complex] = []
    control_coefficients: list[complex] = []
    phase_offsets: list[float] = []
    spectral_ratios: list[float] = []
    control_spectral_ratios: list[float] = []
    tracking_risk: list[float] = []
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    uncertainty = np.asarray(dataset.transform_uncertainty, dtype=np.float64)
    for rows in blocks:
        target_curve = _complex_coefficients(timestamps[rows], traces.target[rows], grid)
        target_value = target_curve[target_index]
        target_noise = _finite_median(np.abs(target_curve[sideband]))
        target_ratio = (
            float(abs(target_value) / target_noise)
            if np.isfinite(target_value) and np.isfinite(target_noise) and target_noise > 0.0
            else math.nan
        )
        target_coefficients.append(target_value)
        spectral_ratios.append(target_ratio)
        tracking_risk.append(_finite_median(uncertainty[rows]))
        if np.all(np.isfinite(traces.upper[rows])) and np.all(
            np.isfinite(traces.lower[rows])
        ):
            upper_value = _complex_coefficients(
                timestamps[rows], traces.upper[rows], grid[target_index : target_index + 1]
            )[0]
            lower_value = _complex_coefficients(
                timestamps[rows], traces.lower[rows], grid[target_index : target_index + 1]
            )[0]
            if np.isfinite(upper_value) and np.isfinite(lower_value):
                phase_offsets.append(float(np.angle(lower_value * np.conjugate(upper_value))))
        control_valid = np.isfinite(traces.control[rows])
        if float(np.mean(control_valid)) >= float(min_valid_fraction):
            control_curve = _complex_coefficients(
                timestamps[rows][control_valid], traces.control[rows][control_valid], grid
            )
            control_value = control_curve[target_index]
            control_noise = _finite_median(np.abs(control_curve[sideband]))
            control_ratio = (
                float(abs(control_value) / control_noise)
                if np.isfinite(control_value)
                and np.isfinite(control_noise)
                and control_noise > 0.0
                else math.nan
            )
        else:
            control_value = np.nan + 0j
            control_ratio = math.nan
        control_coefficients.append(control_value)
        control_spectral_ratios.append(control_ratio)
    spectral = _finite_median(spectral_ratios)
    control_spectral = _finite_median(control_spectral_ratios)
    target_to_control = (
        float(spectral / control_spectral)
        if np.isfinite(spectral)
        and np.isfinite(control_spectral)
        and control_spectral > 0.0
        else math.nan
    )
    phase_array = np.asarray(phase_offsets, dtype=np.float64)
    if phase_array.size:
        vector = np.mean(np.exp(1j * phase_array))
        mean_phase = float(np.angle(vector))
        locking = float(np.abs(vector))
    else:
        mean_phase = math.nan
        locking = math.nan
    lag_fraction = -mean_phase / (2.0 * np.pi) if np.isfinite(mean_phase) else math.nan
    lag_ms = lag_fraction / float(frequency_hz) * 1000.0 if np.isfinite(lag_fraction) else math.nan
    spectral_array = np.asarray(spectral_ratios, dtype=np.float64)
    tracking_array = np.asarray(tracking_risk, dtype=np.float64)
    return WindowMetrics(
        spectral_ratio=spectral,
        control_ratio=target_to_control,
        phase_offset_deg=float(np.degrees(mean_phase)) if np.isfinite(mean_phase) else math.nan,
        phase_locking_value=locking,
        lag_cycle_fraction=lag_fraction,
        lag_ms=lag_ms,
        tracking_spearman_r=_safe_spearman(spectral_array, tracking_array),
        block_count=len(blocks),
        control_block_count=int(np.count_nonzero(np.isfinite(control_spectral_ratios))),
        median_transform_uncertainty=_finite_median(uncertainty),
        target_coefficients=np.asarray(target_coefficients, dtype=np.complex64),
        control_coefficients=np.asarray(control_coefficients, dtype=np.complex64),
        block_phase_offsets_rad=phase_array.astype(np.float32),
        block_spectral_ratios=spectral_array.astype(np.float32),
        block_tracking_risk=tracking_array.astype(np.float32),
    )


def _candidate_summary(
    rows: Sequence[Mapping[str, Any]],
    candidate_names: Sequence[str],
    *,
    min_discovery_windows: int,
    min_discovery_spectral_ratio: float,
    min_discovery_control_ratio: float,
) -> tuple[dict[str, Any], str | None]:
    summaries: dict[str, Any] = {}
    eligible: list[tuple[float, str]] = []
    for name in candidate_names:
        selected = [row for row in rows if row["candidate"] == name and row["status"] == "ok"]
        discovery = [row for row in selected if row["outer_role"] == "discovery"]
        confirmation = [row for row in selected if row["outer_role"] == "confirmation"]

        def values(group: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in group], dtype=np.float64)

        discovery_spectral = _finite_median(values(discovery, "spectral_ratio"))
        discovery_control = _finite_median(values(discovery, "control_ratio"))
        discovery_plv = _finite_median(values(discovery, "phase_locking_value"))
        selection_score = (
            float(math.log2(discovery_spectral) + math.log2(discovery_control))
            if np.isfinite(discovery_spectral)
            and discovery_spectral > 0.0
            and np.isfinite(discovery_control)
            and discovery_control > 0.0
            else math.nan
        )
        passes = bool(
            len(discovery) >= int(min_discovery_windows)
            and np.isfinite(discovery_spectral)
            and discovery_spectral >= float(min_discovery_spectral_ratio)
            and np.isfinite(discovery_control)
            and discovery_control >= float(min_discovery_control_ratio)
            and np.isfinite(selection_score)
        )
        if passes:
            eligible.append((selection_score, name))
        summaries[name] = {
            "discovery_window_count": len(discovery),
            "confirmation_window_count": len(confirmation),
            "passes_descriptive_discovery_gate": passes,
            "discovery_selection_score": selection_score,
            "discovery": {
                "spectral_ratio_median": discovery_spectral,
                "control_ratio_median": discovery_control,
                "phase_locking_value_median": discovery_plv,
                "tracking_spearman_r_median": _finite_median(
                    values(discovery, "tracking_spearman_r")
                ),
            },
            "confirmation_display_only": {
                "spectral_ratio_median": _finite_median(
                    values(confirmation, "spectral_ratio")
                ),
                "spectral_ratio_iqr": _finite_iqr(values(confirmation, "spectral_ratio")),
                "control_ratio_median": _finite_median(values(confirmation, "control_ratio")),
                "phase_locking_value_median": _finite_median(
                    values(confirmation, "phase_locking_value")
                ),
                "tracking_spearman_r_median": _finite_median(
                    values(confirmation, "tracking_spearman_r")
                ),
            },
        }
    winner = max(eligible, key=lambda item: (item[0], item[1]))[1] if eligible else None
    return summaries, winner


def _write_plot(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    candidate_names: Sequence[str],
    *,
    winner: str | None,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [name.replace("_", "\n") for name in candidate_names]
    positions = np.arange(len(candidate_names))
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    metrics = (
        ("spectral_ratio", "held-out target / sideband", 1.0),
        ("control_ratio", "target spectral ratio / control", 1.0),
        ("phase_locking_value", "upper/lower phase-locking value", None),
        ("tracking_spearman_r", "block response vs tracking-risk Spearman r", 0.0),
    )
    for axis, (key, ylabel, baseline) in zip(axes.flat, metrics):
        confirmation_values: list[np.ndarray] = []
        discovery_medians: list[float] = []
        for name in candidate_names:
            confirmation_values.append(
                np.asarray(
                    [
                        float(row[key])
                        for row in rows
                        if row["candidate"] == name
                        and row["outer_role"] == "confirmation"
                        and row["status"] == "ok"
                        and np.isfinite(float(row[key]))
                    ],
                    dtype=np.float64,
                )
            )
            discovery_medians.append(
                _finite_median(
                    [
                        float(row[key])
                        for row in rows
                        if row["candidate"] == name
                        and row["outer_role"] == "discovery"
                        and row["status"] == "ok"
                    ]
                )
            )
        nonempty_positions = [index for index, values in enumerate(confirmation_values) if values.size]
        if nonempty_positions:
            axis.boxplot(
                [confirmation_values[index] for index in nonempty_positions],
                positions=np.asarray(nonempty_positions),
                widths=0.55,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "#d9e5ef", "edgecolor": "#42677f"},
                medianprops={"color": "#152b36", "linewidth": 1.5},
            )
        axis.scatter(
            positions,
            discovery_medians,
            marker="D",
            s=28,
            color="#c03d2b",
            label="discovery median",
            zorder=3,
        )
        if winner in candidate_names:
            winner_index = candidate_names.index(str(winner))
            axis.axvspan(winner_index - 0.45, winner_index + 0.45, color="#e6b44c", alpha=0.16)
        if baseline is not None:
            axis.axhline(float(baseline), color="0.5", ls="--", lw=0.8)
        axis.set_xticks(positions, labels, fontsize=7)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "Photometry transform comparison: boxplots are outer confirmation display only; "
        "diamonds are discovery medians\n"
        "Every trace is internally cross-fit at the frozen compact-core frequency; no "
        "transform-family null was run"
    )
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cross-fit photometry transforms at frozen per-window frequencies. "
            "This is a descriptive challenger comparison, not an inferential test."
        )
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--longitudinal-csv", type=Path, required=True)
    parser.add_argument("--original-mask-npz", type=Path, required=True)
    parser.add_argument("--original-mask-key", default="heart_support_mask")
    parser.add_argument("--consensus-mask-npz", type=Path, required=True)
    parser.add_argument("--consensus-mask-key", default="consensus_mask")
    parser.add_argument("--regions-npz", type=Path, required=True)
    parser.add_argument("--upper-key", default="upper_mask")
    parser.add_argument("--lower-key", default="lower_mask")
    parser.add_argument("--reference-mask-npz", type=Path)
    parser.add_argument("--reference-mask-key", default="reference_mask")
    parser.add_argument("--control-mask-npz", type=Path)
    parser.add_argument("--control-mask-key", default="control_mask")
    parser.add_argument("--frequency-source-mask", default="intersection_8")
    parser.add_argument("--frequency-min-hz", type=float, default=2.0)
    parser.add_argument("--frequency-max-hz", type=float, default=4.0)
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--block-seconds", type=float, default=4.0)
    parser.add_argument("--guard-seconds", type=float, default=0.25)
    parser.add_argument("--min-block-seconds", type=float, default=2.0)
    parser.add_argument("--min-block-valid-fraction", type=float, default=0.7)
    parser.add_argument("--max-interpolated-gap-seconds", type=float, default=0.02)
    parser.add_argument("--nuisance-ridge", type=float, default=1e-6)
    parser.add_argument("--sg-windows", default="5,7,11")
    parser.add_argument("--lag-frames", default="8,12,16")
    parser.add_argument("--gaussian-sigma-px", type=float, default=0.8)
    parser.add_argument("--outer-discovery-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--min-discovery-windows", type=int, default=3)
    parser.add_argument("--min-discovery-spectral-ratio", type=float, default=1.5)
    parser.add_argument("--min-discovery-control-ratio", type=float, default=1.1)
    parser.add_argument("--window-indices")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument(
        "--frame-count",
        type=int,
        help="Smoke-only cap applied independently to the start of each selected window.",
    )
    parser.add_argument("--skip-matched-projection", action="store_true")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    sg_windows = sorted(_parse_int_list(args.sg_windows) or ())
    lag_frames = sorted(_parse_int_list(args.lag_frames) or ())
    if any(value < 3 or value % 2 == 0 for value in sg_windows):
        raise ValueError("Savitzky-Golay windows must be odd integers of at least 3")
    if any(value < 2 or value % 2 != 0 for value in lag_frames):
        raise ValueError("center-aligned lag differences require positive even lags")
    if args.max_windows is not None and int(args.max_windows) < 1:
        raise ValueError("max-windows must be positive")
    if args.frame_count is not None and int(args.frame_count) < 16:
        raise ValueError("frame-count must be at least 16")

    dataset = load_dataset(args.dataset_npz)
    original_image = _read_mask(args.original_mask_npz, args.original_mask_key)
    consensus_image = _read_mask(args.consensus_mask_npz, args.consensus_mask_key)
    upper_stored = _read_mask(args.regions_npz, args.upper_key)
    lower_stored = _read_mask(args.regions_npz, args.lower_key)
    for name, image in (
        ("original", original_image),
        ("consensus", consensus_image),
    ):
        if image.shape != dataset.image_shape_hw:
            raise ValueError(f"{name} mask shape does not match the cached image shape")
    target_pixels = _validate_pixel_mask(
        "original target", _mask_at_pixels(original_image, dataset.pixel_xy)
    )
    consensus_pixels = _validate_pixel_mask(
        "consensus", _mask_at_pixels(consensus_image, dataset.pixel_xy)
    )
    intersection_pixels = _validate_pixel_mask(
        "original/consensus intersection", target_pixels & consensus_pixels
    )
    upper_pixels = _validate_pixel_mask(
        "upper region",
        _stored_mask_at_pixels(
            upper_stored, dataset, name="upper frozen region"
        )
        & target_pixels,
    )
    lower_pixels = _validate_pixel_mask(
        "lower region",
        _stored_mask_at_pixels(
            lower_stored, dataset, name="lower frozen region"
        )
        & target_pixels,
    )
    if np.any(upper_pixels & lower_pixels):
        raise ValueError("upper and lower frozen regions overlap")
    if not np.array_equal(upper_pixels | lower_pixels, target_pixels):
        raise ValueError("upper and lower frozen regions do not partition the target mask")

    auto_reference_image, auto_control_image = _auto_reference_and_control_masks(original_image)
    reference_source = "automatic_two_pixel_geometric_annulus"
    control_source = "automatic_two_to_five_pixel_geometric_annulus"
    if args.reference_mask_npz is not None:
        reference_image = _read_mask(args.reference_mask_npz, args.reference_mask_key)
        reference_source = str(args.reference_mask_npz)
    else:
        reference_image = auto_reference_image
    if args.control_mask_npz is not None:
        control_image = _read_mask(args.control_mask_npz, args.control_mask_key)
        control_source = str(args.control_mask_npz)
    else:
        control_image = auto_control_image
    if reference_image.shape != dataset.image_shape_hw or control_image.shape != dataset.image_shape_hw:
        raise ValueError("reference and control masks must match the cached image shape")
    reference_pixels = _validate_pixel_mask(
        "reference", _mask_at_pixels(reference_image, dataset.pixel_xy)
    )
    control_pixels = _validate_pixel_mask(
        "control", _mask_at_pixels(control_image, dataset.pixel_xy)
    )
    if np.any(reference_pixels & target_pixels) or np.any(control_pixels & target_pixels):
        raise ValueError("reference/control pixels must not overlap the frozen target")

    selected_window_indices = _parse_int_list(args.window_indices)
    windows = _read_source_windows(args.longitudinal_csv, str(args.frequency_source_mask))
    if selected_window_indices is not None:
        windows = [row for row in windows if int(row["window_index"]) in selected_window_indices]
    if args.max_windows is not None:
        windows = windows[: int(args.max_windows)]
    if not windows:
        raise ValueError("no longitudinal windows remain after filtering")

    expected_names = [
        "baseline_mean_intensity",
        "robust_huber_intensity",
        "reference_log_ratio",
        "reference_fractional_difference",
        f"masked_gaussian_huber_sigma{float(args.gaussian_sigma_px):g}",
        "regional_spatial_std",
        *[f"huber_savgol_derivative_w{value}" for value in sg_windows],
        f"gaussian_savgol_derivative_w7_sigma{float(args.gaussian_sigma_px):g}",
        *[f"huber_normalized_signed_lag{value}" for value in lag_frames],
    ]
    if not args.skip_matched_projection:
        expected_names.append("crossfit_matched_spatial_projection")

    frame_indices = np.asarray(dataset.frame_indices, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    coefficient_records: list[dict[str, Any]] = []
    for position, source in enumerate(windows):
        window_index = int(source["window_index"])
        outer_role = (
            "discovery"
            if window_index % 2 == int(args.outer_discovery_parity)
            else "confirmation"
        )
        common: dict[str, Any] = {
            "window_index": window_index,
            "outer_role": outer_role,
            "window_start_s": float(source["window_start_s"]),
            "window_stop_s": float(source["window_stop_s"]),
            "window_mid_s": float(source["window_mid_s"]),
            "source_status": str(source["status"]),
            "frozen_frequency_hz": float(source["candidate_frequency_hz"] or "nan"),
            "frozen_cycles_per_min": float(source["candidate_cycles_per_min"] or "nan"),
        }
        if source["status"] != "ok":
            for name in expected_names:
                rows.append({**common, "candidate": name, "status": "source_window_unscorable"})
            continue
        start = int(
            np.searchsorted(frame_indices, int(source["window_frame_start"]), side="left")
        )
        stop = int(
            np.searchsorted(
                frame_indices,
                int(source["window_frame_stop_inclusive"]),
                side="right",
            )
        )
        if args.frame_count is not None:
            stop = min(stop, start + int(args.frame_count))
        if stop - start < 16:
            for name in expected_names:
                rows.append({**common, "candidate": name, "status": "too_few_rows"})
            continue
        local = _window_dataset(dataset, start, stop)
        partitions = alternating_block_partitions(
            local.timestamps_s,
            block_seconds=float(args.block_seconds),
            guard_seconds=float(args.guard_seconds),
        )
        candidates = _candidate_traces(
            local,
            target=target_pixels,
            upper=upper_pixels,
            lower=lower_pixels,
            reference=reference_pixels,
            control=control_pixels,
            sg_windows=sg_windows,
            lag_frames=lag_frames,
            gaussian_sigma_px=float(args.gaussian_sigma_px),
        )
        if not args.skip_matched_projection:
            candidates["crossfit_matched_spatial_projection"] = _matched_projection_trace_set(
                local,
                target=target_pixels,
                control=control_pixels,
                partitions=partitions,
                frequency_hz=float(common["frozen_frequency_hz"]),
            )
        if list(candidates) != expected_names:
            raise RuntimeError("candidate construction order does not match declared family")
        for name, raw_traces in candidates.items():
            try:
                traces = (
                    raw_traces
                    if name == "crossfit_matched_spatial_projection"
                    else _crossfit_trace_set(
                        raw_traces,
                        local,
                        partitions,
                        ridge=float(args.nuisance_ridge),
                    )
                )
                metrics = _measure_window(
                    local,
                    traces,
                    frequency_hz=float(common["frozen_frequency_hz"]),
                    frequency_min_hz=float(args.frequency_min_hz),
                    frequency_max_hz=float(args.frequency_max_hz),
                    frequency_step_hz=float(args.frequency_step_hz),
                    block_seconds=float(args.block_seconds),
                    min_block_seconds=float(args.min_block_seconds),
                    min_valid_fraction=float(args.min_block_valid_fraction),
                    max_interpolated_gap_seconds=float(
                        args.max_interpolated_gap_seconds
                    ),
                )
                status = "ok" if metrics.block_count >= 2 else "too_few_heldout_blocks"
                rows.append(
                    {
                        **common,
                        "candidate": name,
                        "status": status,
                        "frame_count": local.frame_count,
                        "heldout_block_count": metrics.block_count,
                        "control_block_count": metrics.control_block_count,
                        "spectral_ratio": metrics.spectral_ratio,
                        "control_ratio": metrics.control_ratio,
                        "phase_offset_deg_lower_minus_upper": metrics.phase_offset_deg,
                        "phase_locking_value": metrics.phase_locking_value,
                        "lag_cycle_fraction": metrics.lag_cycle_fraction,
                        "lag_ms": metrics.lag_ms,
                        "tracking_spearman_r": metrics.tracking_spearman_r,
                        "median_transform_uncertainty": metrics.median_transform_uncertainty,
                    }
                )
                coefficient_records.append(
                    {
                        "window_index": window_index,
                        "candidate": name,
                        "target": metrics.target_coefficients,
                        "control": metrics.control_coefficients,
                        "phase": metrics.block_phase_offsets_rad,
                        "spectral": metrics.block_spectral_ratios,
                        "tracking": metrics.block_tracking_risk,
                    }
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                rows.append(
                    {
                        **common,
                        "candidate": name,
                        "status": f"failed:{type(exc).__name__}",
                    }
                )
        print(
            f"window {position + 1}/{len(windows)} index={window_index} role={outer_role} "
            f"frequency={common['frozen_frequency_hz']:.2f}Hz",
            flush=True,
        )

    summaries, winner = _candidate_summary(
        rows,
        expected_names,
        min_discovery_windows=int(args.min_discovery_windows),
        min_discovery_spectral_ratio=float(args.min_discovery_spectral_ratio),
        min_discovery_control_ratio=float(args.min_discovery_control_ratio),
    )
    winner_confirmation = [
        row
        for row in rows
        if winner is not None
        and row["candidate"] == winner
        and row["outer_role"] == "confirmation"
        and row["status"] == "ok"
    ]
    winner_status = (
        "descriptive_discovery_challenger_selected_confirmation_display_only"
        if winner is not None and winner_confirmation
        else "descriptive_discovery_challenger_selected_no_confirmation_rows"
        if winner is not None
        else "no_candidate_passed_descriptive_discovery_gate"
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".photometry_transforms.windows.csv")
    arrays_path = output_prefix.with_suffix(".photometry_transforms.arrays.npz")
    summary_path = output_prefix.with_suffix(".photometry_transforms.summary.json")
    figure_path = output_prefix.with_suffix(".photometry_transforms.png")
    fieldnames = list(rows[0]) if rows else []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    window_indices = sorted({int(row["window_index"]) for row in rows})
    window_position = {value: index for index, value in enumerate(window_indices)}
    candidate_position = {value: index for index, value in enumerate(expected_names)}
    shape = (len(window_indices), len(expected_names))
    arrays: dict[str, np.ndarray] = {
        "spectral_ratio": np.full(shape, np.nan, dtype=np.float32),
        "control_ratio": np.full(shape, np.nan, dtype=np.float32),
        "phase_offset_deg": np.full(shape, np.nan, dtype=np.float32),
        "phase_locking_value": np.full(shape, np.nan, dtype=np.float32),
        "lag_cycle_fraction": np.full(shape, np.nan, dtype=np.float32),
        "tracking_spearman_r": np.full(shape, np.nan, dtype=np.float32),
        "heldout_block_count": np.zeros(shape, dtype=np.int16),
    }
    for row in rows:
        wi = window_position[int(row["window_index"])]
        ci = candidate_position[str(row["candidate"])]
        if row["status"] != "ok":
            continue
        for key in (
            "spectral_ratio",
            "control_ratio",
            "phase_locking_value",
            "lag_cycle_fraction",
            "tracking_spearman_r",
        ):
            arrays[key][wi, ci] = float(row[key])
        arrays["phase_offset_deg"][wi, ci] = float(
            row["phase_offset_deg_lower_minus_upper"]
        )
        arrays["heldout_block_count"][wi, ci] = int(row["heldout_block_count"])
    maximum_blocks = max(
        (int(np.asarray(record["target"]).size) for record in coefficient_records),
        default=0,
    )
    coefficient_shape = (len(coefficient_records), maximum_blocks)
    target_block_coefficients = np.full(
        coefficient_shape, np.nan + 0j, dtype=np.complex64
    )
    control_block_coefficients = np.full(
        coefficient_shape, np.nan + 0j, dtype=np.complex64
    )
    block_phase_offsets = np.full(coefficient_shape, np.nan, dtype=np.float32)
    block_spectral_ratios = np.full(coefficient_shape, np.nan, dtype=np.float32)
    block_tracking_risk = np.full(coefficient_shape, np.nan, dtype=np.float32)
    coefficient_block_counts = np.zeros(len(coefficient_records), dtype=np.int16)
    for record_index, record in enumerate(coefficient_records):
        count = int(np.asarray(record["target"]).size)
        coefficient_block_counts[record_index] = count
        target_block_coefficients[record_index, :count] = record["target"]
        control_block_coefficients[record_index, :count] = record["control"]
        block_spectral_ratios[record_index, :count] = record["spectral"]
        block_tracking_risk[record_index, :count] = record["tracking"]
        phase_count = int(np.asarray(record["phase"]).size)
        block_phase_offsets[record_index, :phase_count] = record["phase"]
    np.savez_compressed(
        arrays_path,
        interpretation=np.asarray(_INTERPRETATION),
        window_indices=np.asarray(window_indices, dtype=np.int32),
        outer_roles=np.asarray(
            [
                "discovery"
                if value % 2 == int(args.outer_discovery_parity)
                else "confirmation"
                for value in window_indices
            ]
        ),
        candidate_names=np.asarray(expected_names),
        selected_candidate=np.asarray(winner or ""),
        **arrays,
        coefficient_window_indices=np.asarray(
            [record["window_index"] for record in coefficient_records], dtype=np.int32
        ),
        coefficient_candidate_names=np.asarray(
            [record["candidate"] for record in coefficient_records]
        ),
        coefficient_block_counts=coefficient_block_counts,
        target_block_coefficients=target_block_coefficients,
        control_block_coefficients=control_block_coefficients,
        block_phase_offsets_rad=block_phase_offsets,
        block_spectral_ratios=block_spectral_ratios,
        block_tracking_risk=block_tracking_risk,
    )
    summary = {
        "interpretation": _INTERPRETATION,
        "inference": {
            "transform_family_surrogates_rerun": False,
            "p_values_computed": False,
            "warning": (
                "All candidates and kernel choices form one adaptive family. The optional "
                "winner is chosen only from outer discovery windows using uncalibrated "
                "descriptive gates. Confirmation values are display-only and cannot support "
                "a detection, cardiac identity, or validated event claim."
            ),
        },
        "sources": {
            "dataset_npz": str(args.dataset_npz),
            "longitudinal_csv": str(args.longitudinal_csv),
            "original_mask_npz": str(args.original_mask_npz),
            "consensus_mask_npz": str(args.consensus_mask_npz),
            "regions_npz": str(args.regions_npz),
            "reference": reference_source,
            "control": control_source,
        },
        "pixel_counts": {
            "target_original": int(np.count_nonzero(target_pixels)),
            "consensus": int(np.count_nonzero(consensus_pixels)),
            "intersection_frequency_source": int(np.count_nonzero(intersection_pixels)),
            "upper": int(np.count_nonzero(upper_pixels)),
            "lower": int(np.count_nonzero(lower_pixels)),
            "reference": int(np.count_nonzero(reference_pixels)),
            "control": int(np.count_nonzero(control_pixels)),
        },
        "crossfit": {
            "within_window": "alternating temporal blocks; fit on one parity, apply to the other, then reverse",
            "block_seconds": float(args.block_seconds),
            "guard_seconds": float(args.guard_seconds),
            "maximum_short_gap_seconds": float(args.max_interpolated_gap_seconds),
            "outer_discovery_window_parity": int(args.outer_discovery_parity),
            "outer_selection_uses_confirmation": False,
            "frequency_policy": (
                "per-window frequency is frozen from the supplied compact-mask longitudinal row"
            ),
            "matched_projection_regional_phase": (
                "not_computed_because_independently_fitted_regional_templates_have_arbitrary_phase_references"
            ),
        },
        "smoke_limits": {
            "window_indices": sorted(selected_window_indices)
            if selected_window_indices is not None
            else None,
            "max_windows": args.max_windows,
            "frame_count_per_window": args.frame_count,
        },
        "family": {
            "candidate_count": len(expected_names),
            "candidate_names": expected_names,
            "matched_projection_included": not bool(args.skip_matched_projection),
            "descriptive_discovery_gate": {
                "minimum_windows": int(args.min_discovery_windows),
                "minimum_median_spectral_ratio": float(args.min_discovery_spectral_ratio),
                "minimum_median_control_ratio": float(args.min_discovery_control_ratio),
                "selection_score": "log2(median spectral ratio) + log2(median control ratio)",
                "calibrated": False,
            },
            "selection_status": winner_status,
            "selected_candidate": winner,
            "candidate_summaries": summaries,
        },
        "outputs": {
            "window_csv": str(csv_path),
            "arrays_npz": str(arrays_path),
            "summary_json": str(summary_path),
            "diagnostic_png": str(figure_path),
        },
    }
    summary_path.write_text(json.dumps(_json_value(summary), indent=2, sort_keys=True) + "\n")
    _write_plot(figure_path, rows, expected_names, winner=winner)
    print(json.dumps(_json_value(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
