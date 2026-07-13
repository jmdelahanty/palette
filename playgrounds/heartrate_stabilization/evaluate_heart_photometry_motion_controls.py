from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_erosion

from analyze_frozen_heart_masks_longitudinal import _read_mask, _window_dataset
import compare_heart_photometry_transforms as photometry
from diagnose_frozen_mask_longitudinal_tracking import _mask_at_pixels
from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.heart_photometry_motion_controls import (
    MotionControlResult,
    integrate_gradient_displacement_control,
    resample_static_reference_control,
    tracking_feature_traces,
)
from fisheye.analysis.heart_photometry_transforms import (
    normalized_signed_lag_difference,
    regional_pool,
    regional_spatial_std,
    segmented_savgol_derivative,
)
from fisheye.analysis.local_rostral_heartrate import (
    LocalCoordinateDataset,
    alternating_block_partitions,
)


_INTERPRETATION = "descriptive_motion_only_controls_without_calibrated_null"
_SOURCE_NAMES = (
    "observed",
    "gradient_displacement_control",
    "static_reference_control",
)
_REGION_NAMES = ("full_mask", "boundary", "eroded_interior")
_TRANSFORM_NAMES = (
    "regional_spatial_std",
    "crossfit_matched_spatial_projection",
    "huber_savgol_derivative_w11",
    "huber_normalized_signed_lag12",
    "huber_normalized_signed_lag16",
)


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


def _validated_frequency_grid(
    timestamps_s: np.ndarray,
    *,
    frequency_min_hz: float,
    frequency_max_hz: float,
    frequency_step_hz: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("timestamps_s must be a one-dimensional array with two samples")
    differences = np.diff(timestamps)
    if not np.isfinite(timestamps).all() or np.any(differences <= 0.0):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    minimum = float(frequency_min_hz)
    maximum = float(frequency_max_hz)
    step = float(frequency_step_hz)
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("frequency_min_hz must be finite and positive")
    if not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError("frequency_max_hz must exceed frequency_min_hz")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("frequency_step_hz must be finite and positive")
    interval_count = int(round((maximum - minimum) / step))
    grid = minimum + np.arange(interval_count + 1, dtype=np.float64) * step
    tolerance = max(1e-12, 1e-9 * maximum)
    if grid.size < 3 or abs(float(grid[-1]) - maximum) > tolerance:
        raise ValueError("frequency range must contain an integral number of steps")
    nominal_dt = float(np.median(differences))
    nyquist_hz = 0.5 / nominal_dt
    if maximum >= nyquist_hz * (1.0 - 1e-12):
        raise ValueError(
            f"frequency_max_hz={maximum:g} must be below timestamp Nyquist "
            f"{nyquist_hz:g}"
        )
    if any(np.count_nonzero(np.abs(grid - frequency) >= 0.2 - 1e-12) < 2 for frequency in grid):
        raise ValueError("frequency grid is too narrow for the 0.2 Hz sideband exclusion")
    return grid, {
        "timestamp_count": int(timestamps.size),
        "nominal_interval_s": nominal_dt,
        "nominal_sampling_hz": float(1.0 / nominal_dt),
        "nyquist_hz": nyquist_hz,
        "minimum_interval_s": float(np.min(differences)),
        "maximum_interval_s": float(np.max(differences)),
        "maximum_absolute_interval_jitter_s": float(
            np.max(np.abs(differences - nominal_dt))
        ),
        "searched_frequency_count": int(grid.size),
        "searched_frequency_min_hz": float(grid[0]),
        "searched_frequency_max_hz": float(grid[-1]),
        "searched_frequency_step_hz": step,
    }


def _coefficient_ratio_curve(
    coefficients: np.ndarray,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    values = np.abs(np.asarray(coefficients, dtype=np.complex128))
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    output = np.full(frequencies.shape, np.nan, dtype=np.float64)
    for index, frequency in enumerate(frequencies):
        sideband = np.abs(frequencies - frequency) >= 0.2 - 1e-12
        noise = _finite_median(values[sideband])
        if np.isfinite(values[index]) and np.isfinite(noise) and noise > 0.0:
            output[index] = float(values[index] / noise)
    return output


def _searched_frequency_metrics(
    dataset: LocalCoordinateDataset,
    traces: photometry.TraceSet,
    *,
    frequency_min_hz: float,
    frequency_max_hz: float,
    frequency_step_hz: float,
    block_seconds: float,
    min_block_seconds: float,
    min_valid_fraction: float,
    max_interpolated_gap_seconds: float,
) -> dict[str, Any]:
    grid, _timebase = _validated_frequency_grid(
        dataset.timestamps_s,
        frequency_min_hz=float(frequency_min_hz),
        frequency_max_hz=float(frequency_max_hz),
        frequency_step_hz=float(frequency_step_hz),
    )
    blocks = photometry._logical_blocks(
        dataset,
        traces,
        block_seconds=float(block_seconds),
        min_block_seconds=float(min_block_seconds),
        min_valid_fraction=float(min_valid_fraction),
        max_interpolated_gap_seconds=float(max_interpolated_gap_seconds),
    )
    target_ratios: list[np.ndarray] = []
    control_ratios: list[np.ndarray] = []
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    for rows in blocks:
        target_coefficients = photometry._complex_coefficients(
            timestamps[rows],
            traces.target[rows],
            grid,
        )
        target_ratios.append(_coefficient_ratio_curve(target_coefficients, grid))
        control_valid = np.isfinite(traces.control[rows])
        if float(np.mean(control_valid)) >= float(min_valid_fraction):
            control_coefficients = photometry._complex_coefficients(
                timestamps[rows][control_valid],
                traces.control[rows][control_valid],
                grid,
            )
            control_ratios.append(_coefficient_ratio_curve(control_coefficients, grid))
        else:
            control_ratios.append(np.full(grid.shape, np.nan, dtype=np.float64))
    if not target_ratios:
        return {
            "searched_best_frequency_hz": math.nan,
            "searched_best_cycles_per_min": math.nan,
            "searched_best_spectral_ratio": math.nan,
            "searched_external_control_ratio_at_best": math.nan,
            "searched_best_frequency_at_boundary": False,
            "searched_block_count": 0,
            "searched_external_control_block_count": 0,
        }
    target_matrix = np.asarray(target_ratios, dtype=np.float64)
    control_matrix = np.asarray(control_ratios, dtype=np.float64)
    target_curve = np.asarray(
        [_finite_median(target_matrix[:, index]) for index in range(grid.size)],
        dtype=np.float64,
    )
    control_curve = np.asarray(
        [_finite_median(control_matrix[:, index]) for index in range(grid.size)],
        dtype=np.float64,
    )
    finite = np.isfinite(target_curve)
    if not np.any(finite):
        return {
            "searched_best_frequency_hz": math.nan,
            "searched_best_cycles_per_min": math.nan,
            "searched_best_spectral_ratio": math.nan,
            "searched_external_control_ratio_at_best": math.nan,
            "searched_best_frequency_at_boundary": False,
            "searched_block_count": len(blocks),
            "searched_external_control_block_count": int(
                np.count_nonzero(np.any(np.isfinite(control_matrix), axis=1))
            ),
        }
    finite_indices = np.flatnonzero(finite)
    # np.argmax deterministically chooses the lowest frequency on an exact tie.
    best_index = int(finite_indices[np.argmax(target_curve[finite_indices])])
    best_ratio = float(target_curve[best_index])
    control_ratio = float(control_curve[best_index])
    return {
        "searched_best_frequency_hz": float(grid[best_index]),
        "searched_best_cycles_per_min": float(60.0 * grid[best_index]),
        "searched_best_spectral_ratio": best_ratio,
        "searched_external_control_ratio_at_best": _safe_ratio(
            best_ratio,
            control_ratio,
        ),
        "searched_best_frequency_at_boundary": bool(
            best_index in {0, int(grid.size) - 1}
        ),
        "searched_block_count": len(blocks),
        "searched_external_control_block_count": int(
            np.count_nonzero(np.any(np.isfinite(control_matrix), axis=1))
        ),
    }


def _median_ratio_curve_on_blocks(
    timestamps_s: np.ndarray,
    trace: np.ndarray,
    blocks: Sequence[np.ndarray],
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    values = np.asarray(trace, dtype=np.float64)
    curves = [
        _coefficient_ratio_curve(
            photometry._complex_coefficients(timestamps[rows], values[rows], frequencies_hz),
            frequencies_hz,
        )
        for rows in blocks
    ]
    if not curves:
        return np.full(np.asarray(frequencies_hz).shape, np.nan, dtype=np.float64)
    matrix = np.asarray(curves, dtype=np.float64)
    return np.asarray(
        [_finite_median(matrix[:, index]) for index in range(matrix.shape[1])],
        dtype=np.float64,
    )


def _paired_support_metrics(
    observed_dataset: LocalCoordinateDataset | Any,
    observed_traces: photometry.TraceSet,
    control_dataset: LocalCoordinateDataset | Any,
    control_traces: photometry.TraceSet,
    *,
    frozen_frequency_hz: float,
    frequency_min_hz: float,
    frequency_max_hz: float,
    frequency_step_hz: float,
    block_seconds: float,
    min_block_seconds: float,
    min_valid_fraction: float,
    max_interpolated_gap_seconds: float,
    minimum_paired_block_count: int,
    minimum_paired_block_fraction: float,
) -> dict[str, Any]:
    observed_timestamps = np.asarray(observed_dataset.timestamps_s, dtype=np.float64)
    control_timestamps = np.asarray(control_dataset.timestamps_s, dtype=np.float64)
    if observed_timestamps.shape != control_timestamps.shape or not np.array_equal(
        observed_timestamps,
        control_timestamps,
    ):
        raise ValueError("observed and control timestamps must match exactly for pairing")
    minimum_count = int(minimum_paired_block_count)
    minimum_fraction = float(minimum_paired_block_fraction)
    if minimum_count < 1:
        raise ValueError("minimum_paired_block_count must be positive")
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_paired_block_fraction must be in (0, 1]")
    grid, _timebase = _validated_frequency_grid(
        observed_timestamps,
        frequency_min_hz=float(frequency_min_hz),
        frequency_max_hz=float(frequency_max_hz),
        frequency_step_hz=float(frequency_step_hz),
    )
    observed_blocks = photometry._logical_blocks(
        observed_dataset,
        observed_traces,
        block_seconds=float(block_seconds),
        min_block_seconds=float(min_block_seconds),
        min_valid_fraction=float(min_valid_fraction),
        max_interpolated_gap_seconds=float(max_interpolated_gap_seconds),
    )
    shared_rows = (
        np.asarray(observed_dataset.frame_valid, dtype=bool)
        & np.asarray(control_dataset.frame_valid, dtype=bool)
        & np.isfinite(observed_traces.target)
        & np.isfinite(control_traces.target)
    )
    paired_dataset = SimpleNamespace(
        timestamps_s=observed_timestamps,
        frame_valid=(
            np.asarray(observed_dataset.frame_valid, dtype=bool)
            & np.asarray(control_dataset.frame_valid, dtype=bool)
        ),
    )
    shared_trace = np.where(shared_rows, 0.0, np.nan)
    paired_trace_set = photometry.TraceSet(
        target=shared_trace,
        upper=np.full(shared_trace.shape, np.nan, dtype=np.float64),
        lower=np.full(shared_trace.shape, np.nan, dtype=np.float64),
        control=shared_trace.copy(),
    )
    paired_blocks = photometry._logical_blocks(
        paired_dataset,
        paired_trace_set,
        block_seconds=float(block_seconds),
        min_block_seconds=float(min_block_seconds),
        min_valid_fraction=float(min_valid_fraction),
        max_interpolated_gap_seconds=float(max_interpolated_gap_seconds),
    )

    relative = observed_timestamps - observed_timestamps[0]
    block_ids = np.floor(relative / float(block_seconds)).astype(np.int64)
    observed_ids = {int(block_ids[rows[0]]) for rows in observed_blocks if rows.size}
    paired_ids = {int(block_ids[rows[0]]) for rows in paired_blocks if rows.size}
    paired_block_fraction = (
        float(len(observed_ids & paired_ids) / len(observed_ids))
        if observed_ids
        else 0.0
    )
    observed_rows = (
        np.unique(np.concatenate(observed_blocks))
        if observed_blocks
        else np.empty(0, dtype=np.int64)
    )
    paired_rows = (
        np.unique(np.concatenate(paired_blocks))
        if paired_blocks
        else np.empty(0, dtype=np.int64)
    )
    paired_row_fraction = (
        float(paired_rows.size / observed_rows.size) if observed_rows.size else 0.0
    )
    support_gate = bool(
        len(paired_blocks) >= minimum_count
        and paired_block_fraction >= minimum_fraction
    )

    observed_curve = _median_ratio_curve_on_blocks(
        observed_timestamps,
        observed_traces.target,
        paired_blocks,
        grid,
    )
    control_curve = _median_ratio_curve_on_blocks(
        control_timestamps,
        control_traces.target,
        paired_blocks,
        grid,
    )

    def maximum(curve: np.ndarray) -> tuple[float, float, bool]:
        finite = np.flatnonzero(np.isfinite(curve))
        if not finite.size:
            return math.nan, math.nan, False
        index = int(finite[np.argmax(curve[finite])])
        return (
            float(grid[index]),
            float(curve[index]),
            bool(index in {0, int(grid.size) - 1}),
        )

    observed_best_frequency, observed_best_ratio, observed_boundary = maximum(
        observed_curve
    )
    control_best_frequency, control_best_ratio, control_boundary = maximum(control_curve)
    frozen_index = int(np.argmin(np.abs(grid - float(frozen_frequency_hz))))
    if abs(float(grid[frozen_index] - frozen_frequency_hz)) > 0.51 * float(
        frequency_step_hz
    ):
        raise ValueError("frozen frequency is outside the paired frequency grid")
    interior_gate = bool(
        np.isfinite(observed_best_frequency)
        and np.isfinite(control_best_frequency)
        and not observed_boundary
        and not control_boundary
    )
    return {
        "paired_observed_logical_block_count": len(observed_blocks),
        "paired_block_count": len(paired_blocks),
        "paired_block_fraction": paired_block_fraction,
        "paired_row_count": int(paired_rows.size),
        "paired_observed_row_count": int(observed_rows.size),
        "paired_row_fraction": paired_row_fraction,
        "minimum_paired_block_count": minimum_count,
        "minimum_paired_block_fraction": minimum_fraction,
        "paired_support_gate_passed": support_gate,
        "paired_observed_frozen_spectral_ratio": float(observed_curve[frozen_index]),
        "paired_control_frozen_spectral_ratio": float(control_curve[frozen_index]),
        "paired_observed_to_control_frozen_ratio": _safe_ratio(
            float(observed_curve[frozen_index]),
            float(control_curve[frozen_index]),
        ),
        "paired_observed_searched_best_frequency_hz": observed_best_frequency,
        "paired_control_searched_best_frequency_hz": control_best_frequency,
        "paired_observed_searched_best_spectral_ratio": observed_best_ratio,
        "paired_control_searched_best_spectral_ratio": control_best_ratio,
        "paired_observed_to_control_searched_max_ratio": _safe_ratio(
            observed_best_ratio,
            control_best_ratio,
        ),
        "paired_observed_best_frequency_at_boundary": observed_boundary,
        "paired_control_best_frequency_at_boundary": control_boundary,
        "paired_frequency_interior_gate_passed": interior_gate,
    }


def _paired_claim_decision(
    paired: Mapping[str, Any],
    *,
    transform_name: str,
) -> tuple[bool, str]:
    if not bool(paired.get("paired_support_gate_passed", False)):
        return False, "paired_support_gate_failed"
    if transform_name == "crossfit_matched_spatial_projection":
        return False, "matched_projection_requires_frequency_grid_refit"
    observed_boundary = bool(
        paired.get("paired_observed_best_frequency_at_boundary", False)
    )
    control_boundary = bool(
        paired.get("paired_control_best_frequency_at_boundary", False)
    )
    if observed_boundary and control_boundary:
        return False, "observed_and_control_maxima_at_search_boundaries"
    if observed_boundary:
        return False, "observed_maximum_at_search_boundary"
    if control_boundary:
        return False, "control_maximum_at_search_boundary"
    if not bool(paired.get("paired_frequency_interior_gate_passed", False)):
        return False, "paired_maximum_frequency_nonfinite_or_not_strictly_interior"
    if not np.isfinite(
        float(paired.get("paired_observed_to_control_searched_max_ratio", math.nan))
    ):
        return False, "paired_searched_max_ratio_nonfinite"
    return True, "eligible"


def _control_dataset(
    dataset: LocalCoordinateDataset,
    result: MotionControlResult,
    *,
    source_name: str,
) -> LocalCoordinateDataset:
    return replace(
        dataset,
        traces=np.asarray(result.values, dtype=np.float64),
        pixel_valid=np.asarray(result.pixel_valid, dtype=bool),
        frame_valid=np.asarray(result.frame_valid, dtype=bool),
        metadata={
            **dict(dataset.metadata),
            "photometry_motion_control": str(source_name),
            "photometry_motion_control_interpretation": str(
                result.diagnostics["interpretation"]
            ),
        },
    ).validated()


def _region_masks(
    original_image: np.ndarray,
    dataset: LocalCoordinateDataset,
    *,
    erosion_iterations: int,
) -> dict[str, np.ndarray]:
    original = np.asarray(original_image, dtype=bool)
    iterations = int(erosion_iterations)
    if iterations < 1:
        raise ValueError("erosion_iterations must be positive")
    structure = np.asarray(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=bool,
    )
    interior_image = binary_erosion(
        original,
        structure=structure,
        iterations=iterations,
        border_value=0,
    )
    images = {
        "full_mask": original,
        "boundary": original & ~interior_image,
        "eroded_interior": interior_image,
    }
    output: dict[str, np.ndarray] = {}
    for name, image in images.items():
        pixels = np.asarray(_mask_at_pixels(image, dataset.pixel_xy), dtype=bool)
        if int(np.count_nonzero(pixels)) < 3:
            raise ValueError(f"{name} has fewer than three cached pixels after erosion")
        output[name] = pixels
    return output


def _pooled_trace(
    dataset: LocalCoordinateDataset,
    region: np.ndarray,
) -> np.ndarray:
    values = np.asarray(dataset.traces, dtype=np.float64)
    valid = (
        np.asarray(dataset.pixel_valid, dtype=bool)
        & np.asarray(dataset.frame_valid, dtype=bool)[:, None]
        & np.isfinite(values)
    )
    return regional_pool(
        values,
        region,
        valid=valid,
        method="huber",
        min_valid_pixels=min(3, int(np.count_nonzero(region))),
    )


def _spatial_std_trace(
    dataset: LocalCoordinateDataset,
    region: np.ndarray,
) -> np.ndarray:
    values = np.asarray(dataset.traces, dtype=np.float64)
    valid = (
        np.asarray(dataset.pixel_valid, dtype=bool)
        & np.asarray(dataset.frame_valid, dtype=bool)[:, None]
        & np.isfinite(values)
    )
    return regional_spatial_std(
        values,
        region,
        valid=valid,
        min_valid_pixels=min(3, int(np.count_nonzero(region))),
    )


def _unavailable(frame_count: int) -> np.ndarray:
    return np.full(int(frame_count), np.nan, dtype=np.float64)


def _challenger_traces(
    dataset: LocalCoordinateDataset,
    *,
    target: np.ndarray,
    control: np.ndarray,
    transform_name: str,
    partitions: Sequence[tuple[np.ndarray, np.ndarray]],
    frequency_hz: float,
    nuisance_ridge: float,
) -> photometry.TraceSet:
    if transform_name == "crossfit_matched_spatial_projection":
        return photometry._matched_projection_trace_set(
            dataset,
            target=target,
            control=control,
            partitions=partitions,
            frequency_hz=float(frequency_hz),
        )

    if transform_name == "regional_spatial_std":
        target_trace = _spatial_std_trace(dataset, target)
        control_trace = _spatial_std_trace(dataset, control)
    else:
        target_trace = _pooled_trace(dataset, target)
        control_trace = _pooled_trace(dataset, control)
        timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
        if transform_name == "huber_savgol_derivative_w11":
            operation = lambda trace: segmented_savgol_derivative(
                trace,
                timestamps,
                valid=np.isfinite(trace),
                window_length=11,
                polyorder=2,
                max_gap_factor=1.75,
            )
        elif transform_name == "huber_normalized_signed_lag12":
            operation = lambda trace: normalized_signed_lag_difference(
                trace,
                timestamps,
                lag_frames=12,
                valid=np.isfinite(trace),
                max_gap_factor=1.75,
                alignment="center",
            )
        elif transform_name == "huber_normalized_signed_lag16":
            operation = lambda trace: normalized_signed_lag_difference(
                trace,
                timestamps,
                lag_frames=16,
                valid=np.isfinite(trace),
                max_gap_factor=1.75,
                alignment="center",
            )
        else:
            raise ValueError(f"unsupported challenger transform {transform_name!r}")
        target_trace = operation(target_trace)
        control_trace = operation(control_trace)

    raw = photometry.TraceSet(
        target=target_trace,
        upper=_unavailable(dataset.frame_count),
        lower=_unavailable(dataset.frame_count),
        control=control_trace,
    )
    return photometry._crossfit_trace_set(
        raw,
        dataset,
        partitions,
        ridge=float(nuisance_ridge),
    )


def _tracking_correlations(
    dataset: LocalCoordinateDataset,
    traces: photometry.TraceSet,
    spectral_ratios: np.ndarray,
    region: np.ndarray,
    *,
    block_seconds: float,
    min_block_seconds: float,
    min_valid_fraction: float,
    max_interpolated_gap_seconds: float,
) -> dict[str, float]:
    blocks = photometry._logical_blocks(
        dataset,
        traces,
        block_seconds=float(block_seconds),
        min_block_seconds=float(min_block_seconds),
        min_valid_fraction=float(min_valid_fraction),
        max_interpolated_gap_seconds=float(max_interpolated_gap_seconds),
    )
    ratios = np.asarray(spectral_ratios, dtype=np.float64)
    count = min(len(blocks), int(ratios.size))
    if count == 0:
        return {
            "source_step_px": math.nan,
            "abs_gradient_displacement": math.nan,
            "gradient_magnitude": math.nan,
            "transform_uncertainty": math.nan,
            "valid_pixel_fraction": math.nan,
        }
    features = tracking_feature_traces(dataset, region)
    output: dict[str, float] = {}
    for name in (
        "source_step_px",
        "abs_gradient_displacement",
        "gradient_magnitude",
        "transform_uncertainty",
        "valid_pixel_fraction",
    ):
        values = np.asarray(getattr(features, name), dtype=np.float64)
        block_values = np.asarray(
            [_finite_median(values[rows]) for rows in blocks[:count]],
            dtype=np.float64,
        )
        output[name] = photometry._safe_spearman(ratios[:count], block_values)
    return output


def _safe_ratio(numerator: float, denominator: float) -> float:
    return (
        float(numerator / denominator)
        if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0.0
        else math.nan
    )


def _comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    paired_records: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    accepted = {
        (
            int(row["window_index"]),
            str(row["source"]),
            str(row["region"]),
            str(row["transform"]),
        ): row
        for row in rows
        if row["status"] == "ok"
    }
    paired = {
        (
            int(row["window_index"]),
            str(row["control_source"]),
            str(row["region"]),
            str(row["transform"]),
        ): row
        for row in paired_records
        if row["status"] == "ok"
    }
    window_indices = sorted({int(row["window_index"]) for row in rows})
    output: list[dict[str, Any]] = []
    for window_index in window_indices:
        for region in _REGION_NAMES:
            for transform in _TRANSFORM_NAMES:
                observed = accepted.get((window_index, "observed", region, transform))
                gradient = accepted.get(
                    (window_index, "gradient_displacement_control", region, transform)
                )
                static = accepted.get(
                    (window_index, "static_reference_control", region, transform)
                )
                gradient_pair = paired.get(
                    (
                        window_index,
                        "gradient_displacement_control",
                        region,
                        transform,
                    )
                )
                static_pair = paired.get(
                    (window_index, "static_reference_control", region, transform)
                )
                gradient_searched_eligible = bool(
                    gradient_pair
                    and gradient_pair.get("paired_support_gate_passed", False)
                    and gradient_pair.get(
                        "paired_frequency_interior_gate_passed", False
                    )
                    and gradient_pair.get(
                        "paired_adaptive_peak_claim_eligible", False
                    )
                )
                static_searched_eligible = bool(
                    static_pair
                    and static_pair.get("paired_support_gate_passed", False)
                    and static_pair.get(
                        "paired_frequency_interior_gate_passed", False
                    )
                    and static_pair.get("paired_adaptive_peak_claim_eligible", False)
                )
                output.append(
                    {
                        "comparison": "observed_vs_motion_controls",
                        "window_index": window_index,
                        "source": "observed",
                        "region": region,
                        "transform": transform,
                        "observed_spectral_ratio": (
                            float(observed["spectral_ratio"]) if observed else math.nan
                        ),
                        "gradient_control_spectral_ratio": (
                            float(gradient["spectral_ratio"]) if gradient else math.nan
                        ),
                        "static_control_spectral_ratio": (
                            float(static["spectral_ratio"]) if static else math.nan
                        ),
                        "observed_to_gradient_ratio": (
                            _safe_ratio(
                                float(observed["spectral_ratio"]),
                                float(gradient["spectral_ratio"]),
                            )
                            if observed and gradient
                            else math.nan
                        ),
                        "observed_to_static_ratio": (
                            _safe_ratio(
                                float(observed["spectral_ratio"]),
                                float(static["spectral_ratio"]),
                            )
                            if observed and static
                            else math.nan
                        ),
                        "frozen_frequency_peak_claim_eligible": False,
                        "observed_searched_best_frequency_hz": (
                            float(observed.get("searched_best_frequency_hz", math.nan))
                            if observed
                            else math.nan
                        ),
                        "gradient_searched_best_frequency_hz": (
                            float(gradient.get("searched_best_frequency_hz", math.nan))
                            if gradient
                            else math.nan
                        ),
                        "static_searched_best_frequency_hz": (
                            float(static.get("searched_best_frequency_hz", math.nan))
                            if static
                            else math.nan
                        ),
                        "observed_searched_best_spectral_ratio": (
                            float(observed.get("searched_best_spectral_ratio", math.nan))
                            if observed
                            else math.nan
                        ),
                        "gradient_searched_best_spectral_ratio": (
                            float(gradient.get("searched_best_spectral_ratio", math.nan))
                            if gradient
                            else math.nan
                        ),
                        "static_searched_best_spectral_ratio": (
                            float(static.get("searched_best_spectral_ratio", math.nan))
                            if static
                            else math.nan
                        ),
                        "observed_to_gradient_searched_max_ratio": (
                            float(
                                gradient_pair[
                                    "paired_observed_to_control_searched_max_ratio"
                                ]
                            )
                            if gradient_searched_eligible
                            else math.nan
                        ),
                        "observed_to_static_searched_max_ratio": (
                            float(
                                static_pair[
                                    "paired_observed_to_control_searched_max_ratio"
                                ]
                            )
                            if static_searched_eligible
                            else math.nan
                        ),
                        "unpaired_observed_to_gradient_searched_max_ratio": (
                            _safe_ratio(
                                float(observed["searched_best_spectral_ratio"]),
                                float(gradient["searched_best_spectral_ratio"]),
                            )
                            if observed and gradient
                            else math.nan
                        ),
                        "unpaired_observed_to_static_searched_max_ratio": (
                            _safe_ratio(
                                float(observed["searched_best_spectral_ratio"]),
                                float(static["searched_best_spectral_ratio"]),
                            )
                            if observed and static
                            else math.nan
                        ),
                        "observed_to_gradient_searched_claim_eligible": (
                            gradient_searched_eligible
                        ),
                        "observed_to_static_searched_claim_eligible": (
                            static_searched_eligible
                        ),
                        "gradient_paired_support_gate_passed": bool(
                            gradient_pair
                            and gradient_pair["paired_support_gate_passed"]
                        ),
                        "gradient_paired_frequency_interior_gate_passed": bool(
                            gradient_pair
                            and gradient_pair[
                                "paired_frequency_interior_gate_passed"
                            ]
                        ),
                        "gradient_paired_adaptive_peak_eligibility_reason": (
                            str(
                                gradient_pair.get(
                                    "paired_adaptive_peak_eligibility_reason",
                                    "paired_record_unavailable",
                                )
                            )
                            if gradient_pair
                            else "paired_record_unavailable"
                        ),
                        "gradient_paired_block_count": (
                            int(gradient_pair["paired_block_count"])
                            if gradient_pair
                            else 0
                        ),
                        "gradient_paired_block_fraction": (
                            float(gradient_pair["paired_block_fraction"])
                            if gradient_pair
                            else 0.0
                        ),
                        "gradient_paired_row_fraction": (
                            float(gradient_pair["paired_row_fraction"])
                            if gradient_pair
                            else 0.0
                        ),
                        "gradient_paired_observed_searched_best_spectral_ratio": (
                            float(
                                gradient_pair[
                                    "paired_observed_searched_best_spectral_ratio"
                                ]
                            )
                            if gradient_pair
                            else math.nan
                        ),
                        "gradient_paired_observed_searched_best_frequency_hz": (
                            float(
                                gradient_pair[
                                    "paired_observed_searched_best_frequency_hz"
                                ]
                            )
                            if gradient_pair
                            else math.nan
                        ),
                        "gradient_paired_control_searched_best_spectral_ratio": (
                            float(
                                gradient_pair[
                                    "paired_control_searched_best_spectral_ratio"
                                ]
                            )
                            if gradient_pair
                            else math.nan
                        ),
                        "gradient_paired_control_searched_best_frequency_hz": (
                            float(
                                gradient_pair[
                                    "paired_control_searched_best_frequency_hz"
                                ]
                            )
                            if gradient_pair
                            else math.nan
                        ),
                        "static_paired_support_gate_passed": bool(
                            static_pair and static_pair["paired_support_gate_passed"]
                        ),
                        "static_paired_frequency_interior_gate_passed": bool(
                            static_pair
                            and static_pair["paired_frequency_interior_gate_passed"]
                        ),
                        "static_paired_adaptive_peak_eligibility_reason": (
                            str(
                                static_pair.get(
                                    "paired_adaptive_peak_eligibility_reason",
                                    "paired_record_unavailable",
                                )
                            )
                            if static_pair
                            else "paired_record_unavailable"
                        ),
                        "static_paired_block_count": (
                            int(static_pair["paired_block_count"])
                            if static_pair
                            else 0
                        ),
                        "static_paired_block_fraction": (
                            float(static_pair["paired_block_fraction"])
                            if static_pair
                            else 0.0
                        ),
                        "static_paired_row_fraction": (
                            float(static_pair["paired_row_fraction"])
                            if static_pair
                            else 0.0
                        ),
                        "static_paired_observed_searched_best_spectral_ratio": (
                            float(
                                static_pair[
                                    "paired_observed_searched_best_spectral_ratio"
                                ]
                            )
                            if static_pair
                            else math.nan
                        ),
                        "static_paired_observed_searched_best_frequency_hz": (
                            float(
                                static_pair[
                                    "paired_observed_searched_best_frequency_hz"
                                ]
                            )
                            if static_pair
                            else math.nan
                        ),
                        "static_paired_control_searched_best_spectral_ratio": (
                            float(
                                static_pair[
                                    "paired_control_searched_best_spectral_ratio"
                                ]
                            )
                            if static_pair
                            else math.nan
                        ),
                        "static_paired_control_searched_best_frequency_hz": (
                            float(
                                static_pair[
                                    "paired_control_searched_best_frequency_hz"
                                ]
                            )
                            if static_pair
                            else math.nan
                        ),
                    }
                )
        for source in _SOURCE_NAMES:
            for transform in _TRANSFORM_NAMES:
                boundary = accepted.get((window_index, source, "boundary", transform))
                interior = accepted.get(
                    (window_index, source, "eroded_interior", transform)
                )
                output.append(
                    {
                        "comparison": "boundary_vs_eroded_interior",
                        "window_index": window_index,
                        "source": source,
                        "region": "boundary_over_eroded_interior",
                        "transform": transform,
                        "boundary_spectral_ratio": (
                            float(boundary["spectral_ratio"]) if boundary else math.nan
                        ),
                        "interior_spectral_ratio": (
                            float(interior["spectral_ratio"]) if interior else math.nan
                        ),
                        "boundary_to_interior_ratio": (
                            _safe_ratio(
                                float(boundary["spectral_ratio"]),
                                float(interior["spectral_ratio"]),
                            )
                            if boundary and interior
                            else math.nan
                        ),
                        "boundary_searched_best_spectral_ratio": (
                            float(boundary.get("searched_best_spectral_ratio", math.nan))
                            if boundary
                            else math.nan
                        ),
                        "interior_searched_best_spectral_ratio": (
                            float(interior.get("searched_best_spectral_ratio", math.nan))
                            if interior
                            else math.nan
                        ),
                        "boundary_to_interior_searched_max_ratio": (
                            _safe_ratio(
                                float(boundary["searched_best_spectral_ratio"]),
                                float(interior["searched_best_spectral_ratio"]),
                            )
                            if boundary
                            and interior
                            and boundary.get("searched_frequency_search_complete", False)
                            and interior.get("searched_frequency_search_complete", False)
                            else math.nan
                        ),
                    }
                )
    return output


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for source in _SOURCE_NAMES:
        summary[source] = {}
        for region in _REGION_NAMES:
            summary[source][region] = {}
            for transform in _TRANSFORM_NAMES:
                selected = [
                    row
                    for row in rows
                    if row["status"] == "ok"
                    and row["source"] == source
                    and row["region"] == region
                    and row["transform"] == transform
                ]
                summary[source][region][transform] = {
                    "scorable_window_count": len(selected),
                    "spectral_ratio_median": _finite_median(
                        [float(row["spectral_ratio"]) for row in selected]
                    ),
                    "spectral_ratio_iqr": _finite_iqr(
                        [float(row["spectral_ratio"]) for row in selected]
                    ),
                    "external_control_ratio_median": _finite_median(
                        [float(row["external_control_ratio"]) for row in selected]
                    ),
                    "searched_best_frequency_hz_median": _finite_median(
                        [float(row["searched_best_frequency_hz"]) for row in selected]
                    ),
                    "searched_best_spectral_ratio_median": _finite_median(
                        [float(row["searched_best_spectral_ratio"]) for row in selected]
                    ),
                    "searched_external_control_ratio_at_best_median": _finite_median(
                        [
                            float(row["searched_external_control_ratio_at_best"])
                            for row in selected
                        ]
                    ),
                    "searched_adaptive_peak_claim_eligible": bool(
                        selected
                        and all(
                            bool(row["searched_adaptive_peak_claim_eligible"])
                            for row in selected
                        )
                    ),
                    "source_step_spearman_median": _finite_median(
                        [float(row["source_step_spearman_r"]) for row in selected]
                    ),
                    "gradient_displacement_spearman_median": _finite_median(
                        [
                            float(row["gradient_displacement_spearman_r"])
                            for row in selected
                        ]
                    ),
                    "transform_uncertainty_spearman_median": _finite_median(
                        [
                            float(row["transform_uncertainty_spearman_r"])
                            for row in selected
                        ]
                    ),
                }
    return summary


def _paired_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for control_source in (
        "gradient_displacement_control",
        "static_reference_control",
    ):
        summary[control_source] = {}
        for region in _REGION_NAMES:
            summary[control_source][region] = {}
            for transform in _TRANSFORM_NAMES:
                selected = [
                    row
                    for row in rows
                    if row["status"] == "ok"
                    and row["control_source"] == control_source
                    and row["region"] == region
                    and row["transform"] == transform
                ]
                eligible = [
                    row
                    for row in selected
                    if row["paired_adaptive_peak_claim_eligible"]
                ]
                reason_counts: dict[str, int] = {}
                for row in selected:
                    reason = str(row["paired_adaptive_peak_eligibility_reason"])
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                summary[control_source][region][transform] = {
                    "computed_window_count": len(selected),
                    "support_gate_pass_count": int(
                        sum(bool(row["paired_support_gate_passed"]) for row in selected)
                    ),
                    "frequency_interior_gate_pass_count": int(
                        sum(
                            bool(row["paired_frequency_interior_gate_passed"])
                            for row in selected
                        )
                    ),
                    "claim_eligible_window_count": len(eligible),
                    "adaptive_peak_eligibility_reason_counts": reason_counts,
                    "paired_block_count_median": _finite_median(
                        [float(row["paired_block_count"]) for row in selected]
                    ),
                    "paired_block_fraction_median": _finite_median(
                        [float(row["paired_block_fraction"]) for row in selected]
                    ),
                    "paired_row_fraction_median": _finite_median(
                        [float(row["paired_row_fraction"]) for row in selected]
                    ),
                    "eligible_observed_to_control_searched_max_ratio_median": (
                        _finite_median(
                            [
                                float(
                                    row[
                                        "paired_observed_to_control_searched_max_ratio"
                                    ]
                                )
                                for row in eligible
                            ]
                        )
                    ),
                }
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_arrays(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> None:
    windows = sorted({int(row["window_index"]) for row in rows})
    positions = {
        "window": {value: index for index, value in enumerate(windows)},
        "source": {value: index for index, value in enumerate(_SOURCE_NAMES)},
        "region": {value: index for index, value in enumerate(_REGION_NAMES)},
        "transform": {value: index for index, value in enumerate(_TRANSFORM_NAMES)},
    }
    shape = (
        len(windows),
        len(_SOURCE_NAMES),
        len(_REGION_NAMES),
        len(_TRANSFORM_NAMES),
    )
    arrays = {
        "spectral_ratio": np.full(shape, np.nan, dtype=np.float32),
        "external_control_ratio": np.full(shape, np.nan, dtype=np.float32),
        "source_step_spearman_r": np.full(shape, np.nan, dtype=np.float32),
        "gradient_displacement_spearman_r": np.full(
            shape, np.nan, dtype=np.float32
        ),
        "transform_uncertainty_spearman_r": np.full(
            shape, np.nan, dtype=np.float32
        ),
        "searched_best_frequency_hz": np.full(shape, np.nan, dtype=np.float32),
        "searched_best_spectral_ratio": np.full(shape, np.nan, dtype=np.float32),
        "searched_external_control_ratio_at_best": np.full(
            shape, np.nan, dtype=np.float32
        ),
        "searched_best_frequency_at_boundary": np.zeros(shape, dtype=np.uint8),
        "searched_adaptive_peak_claim_eligible": np.zeros(shape, dtype=np.uint8),
        "heldout_block_count": np.zeros(shape, dtype=np.int16),
    }
    for row in rows:
        if row["status"] != "ok":
            continue
        index = (
            positions["window"][int(row["window_index"])],
            positions["source"][str(row["source"])],
            positions["region"][str(row["region"])],
            positions["transform"][str(row["transform"])],
        )
        for key in (
            "spectral_ratio",
            "external_control_ratio",
            "source_step_spearman_r",
            "gradient_displacement_spearman_r",
            "transform_uncertainty_spearman_r",
            "searched_best_frequency_hz",
            "searched_best_spectral_ratio",
            "searched_external_control_ratio_at_best",
        ):
            arrays[key][index] = float(row[key])
        arrays["searched_best_frequency_at_boundary"][index] = int(
            bool(row["searched_best_frequency_at_boundary"])
        )
        arrays["searched_adaptive_peak_claim_eligible"][index] = int(
            bool(row["searched_adaptive_peak_claim_eligible"])
        )
        arrays["heldout_block_count"][index] = int(row["heldout_block_count"])

    ratio_shape = (len(windows), len(_REGION_NAMES), len(_TRANSFORM_NAMES))
    observed_to_gradient = np.full(ratio_shape, np.nan, dtype=np.float32)
    observed_to_static = np.full(ratio_shape, np.nan, dtype=np.float32)
    observed_to_gradient_searched = np.full(ratio_shape, np.nan, dtype=np.float32)
    observed_to_static_searched = np.full(ratio_shape, np.nan, dtype=np.float32)
    searched_gradient_eligible = np.zeros(ratio_shape, dtype=np.uint8)
    searched_static_eligible = np.zeros(ratio_shape, dtype=np.uint8)
    gradient_paired_block_count = np.zeros(ratio_shape, dtype=np.int16)
    static_paired_block_count = np.zeros(ratio_shape, dtype=np.int16)
    gradient_paired_block_fraction = np.zeros(ratio_shape, dtype=np.float32)
    static_paired_block_fraction = np.zeros(ratio_shape, dtype=np.float32)
    gradient_paired_row_fraction = np.zeros(ratio_shape, dtype=np.float32)
    static_paired_row_fraction = np.zeros(ratio_shape, dtype=np.float32)
    gradient_paired_support_gate = np.zeros(ratio_shape, dtype=np.uint8)
    static_paired_support_gate = np.zeros(ratio_shape, dtype=np.uint8)
    gradient_paired_frequency_interior_gate = np.zeros(
        ratio_shape, dtype=np.uint8
    )
    static_paired_frequency_interior_gate = np.zeros(ratio_shape, dtype=np.uint8)
    gradient_paired_eligibility_reason = np.full(
        ratio_shape, "paired_record_unavailable", dtype="<U64"
    )
    static_paired_eligibility_reason = np.full(
        ratio_shape, "paired_record_unavailable", dtype="<U64"
    )
    gradient_paired_observed_best_score = np.full(
        ratio_shape, np.nan, dtype=np.float32
    )
    gradient_paired_control_best_score = np.full(
        ratio_shape, np.nan, dtype=np.float32
    )
    static_paired_observed_best_score = np.full(
        ratio_shape, np.nan, dtype=np.float32
    )
    static_paired_control_best_score = np.full(
        ratio_shape, np.nan, dtype=np.float32
    )
    gradient_paired_observed_best_frequency = np.full(
        ratio_shape, np.nan, dtype=np.float32
    )
    gradient_paired_control_best_frequency = np.full(
        ratio_shape, np.nan, dtype=np.float32
    )
    static_paired_observed_best_frequency = np.full(
        ratio_shape, np.nan, dtype=np.float32
    )
    static_paired_control_best_frequency = np.full(
        ratio_shape, np.nan, dtype=np.float32
    )
    boundary_shape = (len(windows), len(_SOURCE_NAMES), len(_TRANSFORM_NAMES))
    boundary_to_interior = np.full(boundary_shape, np.nan, dtype=np.float32)
    for row in comparisons:
        wi = positions["window"][int(row["window_index"])]
        ti = positions["transform"][str(row["transform"])]
        if row["comparison"] == "observed_vs_motion_controls":
            ri = positions["region"][str(row["region"])]
            observed_to_gradient[wi, ri, ti] = float(row["observed_to_gradient_ratio"])
            observed_to_static[wi, ri, ti] = float(row["observed_to_static_ratio"])
            observed_to_gradient_searched[wi, ri, ti] = float(
                row["observed_to_gradient_searched_max_ratio"]
            )
            observed_to_static_searched[wi, ri, ti] = float(
                row["observed_to_static_searched_max_ratio"]
            )
            searched_gradient_eligible[wi, ri, ti] = int(
                bool(row["observed_to_gradient_searched_claim_eligible"])
            )
            searched_static_eligible[wi, ri, ti] = int(
                bool(row["observed_to_static_searched_claim_eligible"])
            )
            gradient_paired_block_count[wi, ri, ti] = int(
                row["gradient_paired_block_count"]
            )
            static_paired_block_count[wi, ri, ti] = int(
                row["static_paired_block_count"]
            )
            gradient_paired_block_fraction[wi, ri, ti] = float(
                row["gradient_paired_block_fraction"]
            )
            static_paired_block_fraction[wi, ri, ti] = float(
                row["static_paired_block_fraction"]
            )
            gradient_paired_row_fraction[wi, ri, ti] = float(
                row["gradient_paired_row_fraction"]
            )
            static_paired_row_fraction[wi, ri, ti] = float(
                row["static_paired_row_fraction"]
            )
            gradient_paired_support_gate[wi, ri, ti] = int(
                bool(row["gradient_paired_support_gate_passed"])
            )
            static_paired_support_gate[wi, ri, ti] = int(
                bool(row["static_paired_support_gate_passed"])
            )
            gradient_paired_frequency_interior_gate[wi, ri, ti] = int(
                bool(row["gradient_paired_frequency_interior_gate_passed"])
            )
            static_paired_frequency_interior_gate[wi, ri, ti] = int(
                bool(row["static_paired_frequency_interior_gate_passed"])
            )
            gradient_paired_eligibility_reason[wi, ri, ti] = str(
                row["gradient_paired_adaptive_peak_eligibility_reason"]
            )
            static_paired_eligibility_reason[wi, ri, ti] = str(
                row["static_paired_adaptive_peak_eligibility_reason"]
            )
            gradient_paired_observed_best_score[wi, ri, ti] = float(
                row["gradient_paired_observed_searched_best_spectral_ratio"]
            )
            gradient_paired_control_best_score[wi, ri, ti] = float(
                row["gradient_paired_control_searched_best_spectral_ratio"]
            )
            static_paired_observed_best_score[wi, ri, ti] = float(
                row["static_paired_observed_searched_best_spectral_ratio"]
            )
            static_paired_control_best_score[wi, ri, ti] = float(
                row["static_paired_control_searched_best_spectral_ratio"]
            )
            gradient_paired_observed_best_frequency[wi, ri, ti] = float(
                row["gradient_paired_observed_searched_best_frequency_hz"]
            )
            gradient_paired_control_best_frequency[wi, ri, ti] = float(
                row["gradient_paired_control_searched_best_frequency_hz"]
            )
            static_paired_observed_best_frequency[wi, ri, ti] = float(
                row["static_paired_observed_searched_best_frequency_hz"]
            )
            static_paired_control_best_frequency[wi, ri, ti] = float(
                row["static_paired_control_searched_best_frequency_hz"]
            )
        else:
            si = positions["source"][str(row["source"])]
            boundary_to_interior[wi, si, ti] = float(
                row["boundary_to_interior_ratio"]
            )
    np.savez_compressed(
        path,
        interpretation=np.asarray(_INTERPRETATION),
        window_indices=np.asarray(windows, dtype=np.int32),
        source_names=np.asarray(_SOURCE_NAMES),
        region_names=np.asarray(_REGION_NAMES),
        transform_names=np.asarray(_TRANSFORM_NAMES),
        observed_to_gradient_ratio=observed_to_gradient,
        observed_to_static_ratio=observed_to_static,
        observed_to_gradient_searched_max_ratio=observed_to_gradient_searched,
        observed_to_static_searched_max_ratio=observed_to_static_searched,
        observed_to_gradient_searched_claim_eligible=searched_gradient_eligible,
        observed_to_static_searched_claim_eligible=searched_static_eligible,
        gradient_paired_block_count=gradient_paired_block_count,
        static_paired_block_count=static_paired_block_count,
        gradient_paired_block_fraction=gradient_paired_block_fraction,
        static_paired_block_fraction=static_paired_block_fraction,
        gradient_paired_row_fraction=gradient_paired_row_fraction,
        static_paired_row_fraction=static_paired_row_fraction,
        gradient_paired_support_gate_passed=gradient_paired_support_gate,
        static_paired_support_gate_passed=static_paired_support_gate,
        gradient_paired_frequency_interior_gate_passed=(
            gradient_paired_frequency_interior_gate
        ),
        static_paired_frequency_interior_gate_passed=(
            static_paired_frequency_interior_gate
        ),
        gradient_paired_adaptive_peak_eligibility_reason=(
            gradient_paired_eligibility_reason
        ),
        static_paired_adaptive_peak_eligibility_reason=(
            static_paired_eligibility_reason
        ),
        gradient_paired_observed_searched_best_spectral_ratio=(
            gradient_paired_observed_best_score
        ),
        gradient_paired_control_searched_best_spectral_ratio=(
            gradient_paired_control_best_score
        ),
        static_paired_observed_searched_best_spectral_ratio=(
            static_paired_observed_best_score
        ),
        static_paired_control_searched_best_spectral_ratio=(
            static_paired_control_best_score
        ),
        gradient_paired_observed_searched_best_frequency_hz=(
            gradient_paired_observed_best_frequency
        ),
        gradient_paired_control_searched_best_frequency_hz=(
            gradient_paired_control_best_frequency
        ),
        static_paired_observed_searched_best_frequency_hz=(
            static_paired_observed_best_frequency
        ),
        static_paired_control_searched_best_frequency_hz=(
            static_paired_control_best_frequency
        ),
        boundary_to_interior_ratio=boundary_to_interior,
        **arrays,
    )


def _write_plot(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "regional_spatial_std": "spatial SD",
        "crossfit_matched_spatial_projection": "matched projection",
        "huber_savgol_derivative_w11": "SG derivative",
        "huber_normalized_signed_lag12": "lag 12",
        "huber_normalized_signed_lag16": "lag 16",
    }
    source_colors = {
        "observed": "#0072B2",
        "gradient_displacement_control": "#D55E00",
        "static_reference_control": "#009E73",
    }
    region_colors = {
        "full_mask": "#4D4D4D",
        "boundary": "#CC79A7",
        "eroded_interior": "#56B4E9",
    }
    x = np.arange(len(_TRANSFORM_NAMES), dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for source_index, source in enumerate(_SOURCE_NAMES):
        medians = []
        for transform in _TRANSFORM_NAMES:
            values = [
                float(row["spectral_ratio"])
                for row in rows
                if row["status"] == "ok"
                and row["source"] == source
                and row["region"] == "full_mask"
                and row["transform"] == transform
            ]
            medians.append(_finite_median(values))
        axes[0, 0].plot(
            x,
            medians,
            marker="o",
            lw=1.5,
            color=source_colors[source],
            label=source.replace("_", " "),
        )
    axes[0, 0].set_ylabel("median frozen-frequency target / sideband")
    axes[0, 0].set_title("Observed-selected frozen frequency: conditioned diagnostic")
    axes[0, 0].legend(fontsize=8)

    width = 0.36
    for control_index, (key, color, label) in enumerate(
        (
            (
                "observed_to_gradient_searched_max_ratio",
                "#E69F00",
                "observed max / gradient-control max",
            ),
            (
                "observed_to_static_searched_max_ratio",
                "#009E73",
                "observed max / static-control max",
            ),
        )
    ):
        medians = []
        for transform in _TRANSFORM_NAMES:
            values = [
                float(row[key])
                for row in comparisons
                if row["comparison"] == "observed_vs_motion_controls"
                and row["region"] == "full_mask"
                and row["transform"] == transform
            ]
            medians.append(_finite_median(values))
        axes[0, 1].bar(
            x + (control_index - 0.5) * width,
            medians,
            width=width,
            color=color,
            label=label,
        )
    axes[0, 1].axhline(1.0, color="#333333", lw=1.0, ls="--")
    axes[0, 1].set_ylabel("median paired-support searched-max quotient")
    axes[0, 1].set_title(
        "Exact shared support; failed coverage, boundary maxima, and matched projection omitted"
    )
    axes[0, 1].legend(fontsize=8)

    for region in _REGION_NAMES:
        medians = []
        for transform in _TRANSFORM_NAMES:
            values = [
                float(row["spectral_ratio"])
                for row in rows
                if row["status"] == "ok"
                and row["source"] == "observed"
                and row["region"] == region
                and row["transform"] == transform
            ]
            medians.append(_finite_median(values))
        axes[1, 0].plot(
            x,
            medians,
            marker="o",
            lw=1.5,
            color=region_colors[region],
            label=region.replace("_", " "),
        )
    axes[1, 0].set_ylabel("median target / sideband")
    axes[1, 0].set_title("Observed boundary versus eroded interior")
    axes[1, 0].legend(fontsize=8)

    correlation_fields = (
        ("source_step_spearman_r", "source step"),
        ("gradient_displacement_spearman_r", "gradient x displacement"),
        ("transform_uncertainty_spearman_r", "transform uncertainty"),
    )
    correlation_colors = ("#0072B2", "#D55E00", "#CC79A7")
    for field_index, ((field, label), color) in enumerate(
        zip(correlation_fields, correlation_colors, strict=True)
    ):
        medians = []
        for transform in _TRANSFORM_NAMES:
            values = [
                float(row[field])
                for row in rows
                if row["status"] == "ok"
                and row["source"] == "observed"
                and row["region"] == "full_mask"
                and row["transform"] == transform
            ]
            medians.append(_finite_median(values))
        axes[1, 1].plot(x, medians, marker="o", lw=1.5, color=color, label=label)
    axes[1, 1].axhline(0.0, color="#333333", lw=1.0)
    axes[1, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].set_ylabel("median blockwise Spearman r")
    axes[1, 1].set_title("Oscillation score association with tracking exposure")
    axes[1, 1].legend(fontsize=8)

    tick_labels = [labels[name] for name in _TRANSFORM_NAMES]
    for axis in axes.flat:
        axis.set_xticks(x, tick_labels, rotation=22, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle(
        "Photometry motion controls at frozen per-window frequencies\n"
        "Adaptive-peak claims require paired-support grid maxima and coverage gates; descriptive only"
    )
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected photometry challengers with measured-coordinate "
            "motion-only controls at frozen longitudinal frequencies."
        )
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--longitudinal-csv", type=Path, required=True)
    parser.add_argument("--original-mask-npz", type=Path, required=True)
    parser.add_argument("--original-mask-key", default="heart_support_mask")
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
    parser.add_argument("--erosion-iterations", type=int, default=1)
    parser.add_argument("--static-epoch-seconds", type=float, default=4.0)
    parser.add_argument("--static-guard-seconds", type=float, default=0.1)
    parser.add_argument("--static-min-template-pixels", type=int, default=8)
    parser.add_argument("--minimum-paired-block-count", type=int, default=4)
    parser.add_argument("--minimum-paired-block-fraction", type=float, default=0.5)
    parser.add_argument("--window-indices")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument(
        "--frame-count",
        type=int,
        help="Smoke-only cap applied independently to the start of each window.",
    )
    parser.add_argument("--skip-static-control", action="store_true")
    parser.add_argument("--skip-matched-projection", action="store_true")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    selected_windows = photometry._parse_int_list(args.window_indices)
    transforms = tuple(
        name
        for name in _TRANSFORM_NAMES
        if not (
            args.skip_matched_projection
            and name == "crossfit_matched_spatial_projection"
        )
    )
    sources = tuple(
        name
        for name in _SOURCE_NAMES
        if not (args.skip_static_control and name == "static_reference_control")
    )
    if args.max_windows is not None and int(args.max_windows) < 1:
        raise ValueError("max_windows must be positive")
    if args.frame_count is not None and int(args.frame_count) < 800:
        raise ValueError("frame_count must be at least 800 for 4-second cross-fitting")
    if int(args.minimum_paired_block_count) < 1:
        raise ValueError("minimum_paired_block_count must be positive")
    if not 0.0 < float(args.minimum_paired_block_fraction) <= 1.0:
        raise ValueError("minimum_paired_block_fraction must be in (0, 1]")

    dataset = load_dataset(args.dataset_npz)
    _frequency_grid, timebase_summary = _validated_frequency_grid(
        dataset.timestamps_s,
        frequency_min_hz=float(args.frequency_min_hz),
        frequency_max_hz=float(args.frequency_max_hz),
        frequency_step_hz=float(args.frequency_step_hz),
    )
    original_image = _read_mask(args.original_mask_npz, args.original_mask_key)
    if original_image.shape != tuple(dataset.image_shape_hw):
        raise ValueError("original mask shape does not match cached image shape")
    regions = _region_masks(
        original_image,
        dataset,
        erosion_iterations=int(args.erosion_iterations),
    )
    _reference_image, automatic_control_image = photometry._auto_reference_and_control_masks(
        original_image
    )
    control_image = (
        _read_mask(args.control_mask_npz, args.control_mask_key)
        if args.control_mask_npz is not None
        else automatic_control_image
    )
    if control_image.shape != tuple(dataset.image_shape_hw):
        raise ValueError("control mask shape does not match cached image shape")
    external_control = photometry._validate_pixel_mask(
        "external control",
        _mask_at_pixels(control_image, dataset.pixel_xy),
    )
    if np.any(external_control & regions["full_mask"]):
        raise ValueError("external control overlaps the original target mask")

    windows = photometry._read_source_windows(
        args.longitudinal_csv,
        str(args.frequency_source_mask),
    )
    if selected_windows is not None:
        windows = [row for row in windows if int(row["window_index"]) in selected_windows]
    if args.max_windows is not None:
        windows = windows[: int(args.max_windows)]
    if not windows:
        raise ValueError("no longitudinal windows remain after filtering")

    frame_indices = np.asarray(dataset.frame_indices, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    paired_records: list[dict[str, Any]] = []
    generation: list[dict[str, Any]] = []
    for position, source_window in enumerate(windows):
        window_index = int(source_window["window_index"])
        common = {
            "window_index": window_index,
            "window_start_s": float(source_window["window_start_s"]),
            "window_stop_s": float(source_window["window_stop_s"]),
            "window_mid_s": float(source_window["window_mid_s"]),
            "frozen_frequency_hz": float(
                source_window["candidate_frequency_hz"] or "nan"
            ),
            "frozen_cycles_per_min": float(
                source_window["candidate_cycles_per_min"] or "nan"
            ),
        }
        if source_window["status"] != "ok":
            for source_name in sources:
                for region_name in _REGION_NAMES:
                    for transform_name in transforms:
                        rows.append(
                            {
                                **common,
                                "source": source_name,
                                "region": region_name,
                                "transform": transform_name,
                                "status": "source_window_unscorable",
                            }
                        )
            continue
        start = int(
            np.searchsorted(
                frame_indices,
                int(source_window["window_frame_start"]),
                side="left",
            )
        )
        stop = int(
            np.searchsorted(
                frame_indices,
                int(source_window["window_frame_stop_inclusive"]),
                side="right",
            )
        )
        if args.frame_count is not None:
            stop = min(stop, start + int(args.frame_count))
        if stop - start < 800:
            for source_name in sources:
                for region_name in _REGION_NAMES:
                    for transform_name in transforms:
                        rows.append(
                            {
                                **common,
                                "source": source_name,
                                "region": region_name,
                                "transform": transform_name,
                                "status": "too_few_rows",
                            }
                        )
            continue
        local = _window_dataset(dataset, start, stop)
        source_datasets: dict[str, LocalCoordinateDataset] = {"observed": local}
        gradient = integrate_gradient_displacement_control(local)
        source_datasets["gradient_displacement_control"] = _control_dataset(
            local,
            gradient,
            source_name="gradient_displacement_control",
        )
        generation.append(
            {
                "window_index": window_index,
                "source": "gradient_displacement_control",
                **dict(gradient.diagnostics),
            }
        )
        if not args.skip_static_control:
            static = resample_static_reference_control(
                local,
                epoch_seconds=float(args.static_epoch_seconds),
                guard_seconds=float(args.static_guard_seconds),
                minimum_template_pixels=int(args.static_min_template_pixels),
                maximum_interpolated_gap_seconds=float(
                    args.max_interpolated_gap_seconds
                ),
            )
            source_datasets["static_reference_control"] = _control_dataset(
                local,
                static,
                source_name="static_reference_control",
            )
            generation.append(
                {
                    "window_index": window_index,
                    "source": "static_reference_control",
                    **dict(static.diagnostics),
                }
            )

        trace_cache: dict[
            tuple[str, str, str],
            tuple[LocalCoordinateDataset, photometry.TraceSet],
        ] = {}
        for source_name in sources:
            local_source = source_datasets[source_name]
            partitions = alternating_block_partitions(
                local_source.timestamps_s,
                block_seconds=float(args.block_seconds),
                guard_seconds=float(args.guard_seconds),
            )
            for region_name in _REGION_NAMES:
                region = regions[region_name]
                region_valid_fraction = float(
                    np.mean(
                        np.asarray(local_source.pixel_valid, dtype=bool)[:, region]
                    )
                )
                for transform_name in transforms:
                    try:
                        challenger = _challenger_traces(
                            local_source,
                            target=region,
                            control=external_control,
                            transform_name=transform_name,
                            partitions=partitions,
                            frequency_hz=float(common["frozen_frequency_hz"]),
                            nuisance_ridge=float(args.nuisance_ridge),
                        )
                        trace_cache[(source_name, region_name, transform_name)] = (
                            local_source,
                            challenger,
                        )
                        metrics = photometry._measure_window(
                            local_source,
                            challenger,
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
                        searched = _searched_frequency_metrics(
                            local_source,
                            challenger,
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
                        correlations = _tracking_correlations(
                            local_source,
                            challenger,
                            metrics.block_spectral_ratios,
                            region,
                            block_seconds=float(args.block_seconds),
                            min_block_seconds=float(args.min_block_seconds),
                            min_valid_fraction=float(args.min_block_valid_fraction),
                            max_interpolated_gap_seconds=float(
                                args.max_interpolated_gap_seconds
                            ),
                        )
                        status = (
                            "ok" if metrics.block_count >= 2 else "too_few_heldout_blocks"
                        )
                        rows.append(
                            {
                                **common,
                                "source": source_name,
                                "region": region_name,
                                "transform": transform_name,
                                "status": status,
                                "frame_count": local_source.frame_count,
                                "region_pixel_count": int(np.count_nonzero(region)),
                                "region_valid_sample_fraction": region_valid_fraction,
                                "heldout_block_count": metrics.block_count,
                                "external_control_block_count": metrics.control_block_count,
                                "spectral_ratio": metrics.spectral_ratio,
                                "frozen_frequency_spectral_ratio": metrics.spectral_ratio,
                                "external_control_ratio": metrics.control_ratio,
                                **searched,
                                "searched_frequency_scope": (
                                    "fixed_frozen_frequency_projection_output_only"
                                    if transform_name
                                    == "crossfit_matched_spatial_projection"
                                    else "complete_grid_on_fixed_transform_trace"
                                ),
                                "searched_frequency_search_complete": bool(
                                    transform_name
                                    != "crossfit_matched_spatial_projection"
                                ),
                                "searched_adaptive_peak_claim_eligible": False,
                                "searched_claim_requires_paired_support": True,
                                "frozen_frequency_peak_claim_eligible": False,
                                "source_step_spearman_r": correlations["source_step_px"],
                                "gradient_displacement_spearman_r": correlations[
                                    "abs_gradient_displacement"
                                ],
                                "gradient_magnitude_spearman_r": correlations[
                                    "gradient_magnitude"
                                ],
                                "transform_uncertainty_spearman_r": correlations[
                                    "transform_uncertainty"
                                ],
                                "valid_fraction_spearman_r": correlations[
                                    "valid_pixel_fraction"
                                ],
                            }
                        )
                    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                        rows.append(
                            {
                                **common,
                                "source": source_name,
                                "region": region_name,
                                "transform": transform_name,
                                "status": f"failed:{type(exc).__name__}",
                                "error": str(exc),
                                "region_pixel_count": int(np.count_nonzero(region)),
                                "region_valid_sample_fraction": region_valid_fraction,
                            }
                        )
        for control_source in (
            source for source in sources if source != "observed"
        ):
            for region_name in _REGION_NAMES:
                for transform_name in transforms:
                    observed_cached = trace_cache.get(
                        ("observed", region_name, transform_name)
                    )
                    control_cached = trace_cache.get(
                        (control_source, region_name, transform_name)
                    )
                    if observed_cached is None or control_cached is None:
                        paired_records.append(
                            {
                                **common,
                                "control_source": control_source,
                                "region": region_name,
                                "transform": transform_name,
                                "status": "trace_unavailable",
                                "paired_support_gate_passed": False,
                                "paired_frequency_interior_gate_passed": False,
                                "paired_adaptive_peak_claim_eligible": False,
                                "paired_adaptive_peak_eligibility_reason": (
                                    "paired_trace_unavailable"
                                ),
                            }
                        )
                        continue
                    observed_dataset, observed_challenger = observed_cached
                    control_dataset, control_challenger = control_cached
                    try:
                        paired = _paired_support_metrics(
                            observed_dataset,
                            observed_challenger,
                            control_dataset,
                            control_challenger,
                            frozen_frequency_hz=float(common["frozen_frequency_hz"]),
                            frequency_min_hz=float(args.frequency_min_hz),
                            frequency_max_hz=float(args.frequency_max_hz),
                            frequency_step_hz=float(args.frequency_step_hz),
                            block_seconds=float(args.block_seconds),
                            min_block_seconds=float(args.min_block_seconds),
                            min_valid_fraction=float(args.min_block_valid_fraction),
                            max_interpolated_gap_seconds=float(
                                args.max_interpolated_gap_seconds
                            ),
                            minimum_paired_block_count=int(
                                args.minimum_paired_block_count
                            ),
                            minimum_paired_block_fraction=float(
                                args.minimum_paired_block_fraction
                            ),
                        )
                        claim_eligible, ineligible_reason = _paired_claim_decision(
                            paired,
                            transform_name=transform_name,
                        )
                        paired_records.append(
                            {
                                **common,
                                "control_source": control_source,
                                "region": region_name,
                                "transform": transform_name,
                                "status": "ok",
                                **paired,
                                "paired_adaptive_peak_claim_eligible": claim_eligible,
                                "paired_adaptive_peak_eligibility_reason": (
                                    ineligible_reason
                                ),
                                "paired_frozen_frequency_peak_claim_eligible": False,
                            }
                        )
                    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                        paired_records.append(
                            {
                                **common,
                                "control_source": control_source,
                                "region": region_name,
                                "transform": transform_name,
                                "status": f"failed:{type(exc).__name__}",
                                "error": str(exc),
                                "paired_support_gate_passed": False,
                                "paired_frequency_interior_gate_passed": False,
                                "paired_adaptive_peak_claim_eligible": False,
                                "paired_adaptive_peak_eligibility_reason": (
                                    "paired_analysis_failed"
                                ),
                                "paired_frozen_frequency_peak_claim_eligible": False,
                            }
                        )
        print(
            f"window {position + 1}/{len(windows)} index={window_index} "
            f"frequency={common['frozen_frequency_hz']:.2f}Hz",
            flush=True,
        )

    comparisons = _comparison_rows(rows, paired_records)
    aggregates = _aggregate(rows)
    paired_aggregates = _paired_aggregate(paired_records)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    windows_csv = output_prefix.with_suffix(".motion_controls.windows.csv")
    comparisons_csv = output_prefix.with_suffix(".motion_controls.comparisons.csv")
    paired_csv = output_prefix.with_suffix(".motion_controls.paired_support.csv")
    arrays_path = output_prefix.with_suffix(".motion_controls.arrays.npz")
    summary_path = output_prefix.with_suffix(".motion_controls.summary.json")
    figure_path = output_prefix.with_suffix(".motion_controls.png")
    _write_csv(windows_csv, rows)
    _write_csv(comparisons_csv, comparisons)
    _write_csv(paired_csv, paired_records)
    _write_arrays(arrays_path, rows, comparisons)
    _write_plot(figure_path, rows, comparisons)

    summary = {
        "interpretation": _INTERPRETATION,
        "inference": {
            "status": "descriptive_control_comparison_only",
            "frequency_policy": {
                "frozen_frequency": (
                    "retained as a conditioned diagnostic selected from the observed "
                    "compact-mask longitudinal analysis; never eligible by itself for "
                    "a claim that motion failed to reproduce an adaptive peak"
                ),
                "searched_frequency": (
                    "each source independently maximizes target/sideband over the same "
                    "validated configured frequency grid"
                ),
                "adaptive_peak_claim_rule": (
                    "only searched-max comparisons marked claim-eligible may address "
                    "whether a motion control reproduced an adaptive peak"
                ),
                "matched_projection_exception": (
                    "the reported search scans the output of weights learned at the "
                    "frozen frequency, so matched-projection searched comparisons are "
                    "marked ineligible until the spatial projection is refit at every "
                    "searched frequency"
                ),
                "paired_support_rule": (
                    "observed and each motion control are rescored on identical usable "
                    "rows and identical accepted logical blocks; claim eligibility also "
                    "requires the predeclared paired block count and coverage gates"
                ),
                "frequency_interior_rule": (
                    "both paired observed and paired control searched maxima must lie "
                    "strictly inside the configured grid; endpoint maxima remain "
                    "descriptive and are explicitly claim-ineligible"
                ),
            },
            "motion_controls": {
                "gradient_displacement_control": (
                    "integrated cached image-gradient dot measured source-coordinate step"
                ),
                "static_reference_control": (
                    "low-uncertainty cached-frame surface linearly resampled inside its "
                    "source-coordinate convex hull"
                ),
            },
            "not_optical_flow": True,
            "limitations": [
                "cached gradients come from observed dynamic frames",
                "the static control reconstructs only the cached local grid, not a full source frame",
                "neither motion control is a calibrated null distribution",
                "a favorable observed-to-control ratio does not establish cardiac identity",
                "frozen-frequency observed/control differences are selection-conditioned",
                "unpaired source-specific searched maxima are display-only",
                "matched-projection adaptive-frequency comparison requires a future grid-wide refit",
            ],
        },
        "sources": {
            "dataset_npz": str(args.dataset_npz),
            "longitudinal_csv": str(args.longitudinal_csv),
            "original_mask_npz": str(args.original_mask_npz),
            "external_control": (
                str(args.control_mask_npz)
                if args.control_mask_npz is not None
                else "automatic_two_to_five_pixel_geometric_annulus"
            ),
        },
        "pixel_counts": {
            **{name: int(np.count_nonzero(mask)) for name, mask in regions.items()},
            "external_control": int(np.count_nonzero(external_control)),
        },
        "configuration": {
            "frequency_source_mask": str(args.frequency_source_mask),
            "frequency_range_hz": [
                float(args.frequency_min_hz),
                float(args.frequency_max_hz),
            ],
            "frequency_step_hz": float(args.frequency_step_hz),
            "timebase_and_nyquist_validation": timebase_summary,
            "block_seconds": float(args.block_seconds),
            "guard_seconds": float(args.guard_seconds),
            "erosion": (
                f"{int(args.erosion_iterations)} iteration(s), 4-connected 3x3 structure"
            ),
            "static_epoch_seconds": float(args.static_epoch_seconds),
            "static_guard_seconds": float(args.static_guard_seconds),
            "static_min_template_pixels": int(args.static_min_template_pixels),
            "minimum_paired_block_count": int(args.minimum_paired_block_count),
            "minimum_paired_block_fraction": float(
                args.minimum_paired_block_fraction
            ),
            "transforms": list(transforms),
            "sources": list(sources),
            "window_indices": sorted(
                {int(row["window_index"]) for row in rows}
            ),
            "frame_count_per_window": args.frame_count,
        },
        "control_generation": generation,
        "aggregates": aggregates,
        "paired_support_aggregates": paired_aggregates,
        "outputs": {
            "windows_csv": str(windows_csv),
            "comparisons_csv": str(comparisons_csv),
            "paired_support_csv": str(paired_csv),
            "arrays_npz": str(arrays_path),
            "summary_json": str(summary_path),
            "figure_png": str(figure_path),
        },
    }
    summary_path.write_text(json.dumps(_json_value(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_value(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
