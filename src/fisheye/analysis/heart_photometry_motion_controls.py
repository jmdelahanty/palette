from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError

from fisheye.analysis.local_rostral_heartrate import LocalCoordinateDataset


@dataclass(frozen=True)
class MotionControlResult:
    """Synthetic cached-pixel intensities generated from measured coordinate motion."""

    values: np.ndarray
    pixel_valid: np.ndarray
    frame_valid: np.ndarray
    segment_index: np.ndarray
    reference_row: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class MotionTrackingFeatures:
    """Per-frame tracking exposures aligned to a local-coordinate dataset."""

    source_step_px: np.ndarray
    abs_gradient_displacement: np.ndarray
    gradient_magnitude: np.ndarray
    transform_uncertainty: np.ndarray
    valid_pixel_fraction: np.ndarray


def _nominal_dt(timestamps_s: np.ndarray) -> float:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    differences = np.diff(timestamps)
    finite = differences[np.isfinite(differences) & (differences > 0.0)]
    if not finite.size:
        raise ValueError("timestamps_s must contain a positive interval")
    return float(np.median(finite))


def _base_valid(dataset: LocalCoordinateDataset) -> np.ndarray:
    traces = np.asarray(dataset.traces, dtype=np.float64)
    source_xy = np.asarray(dataset.source_xy, dtype=np.float64)
    return (
        np.asarray(dataset.frame_valid, dtype=bool)[:, None]
        & np.asarray(dataset.pixel_valid, dtype=bool)
        & np.isfinite(traces)
        & np.all(np.isfinite(source_xy), axis=2)
    )


def _finite_median_axis0(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    output = np.full(array.shape[1], np.nan, dtype=np.float64)
    for column in range(array.shape[1]):
        sample = array[finite[:, column], column]
        if sample.size:
            output[column] = float(np.median(sample))
    return output


def integrate_gradient_displacement_control(
    dataset: LocalCoordinateDataset,
    *,
    maximum_timestamp_gap_factor: float = 1.75,
) -> MotionControlResult:
    """Integrate the cached ``gradient dot coordinate-step`` motion prediction.

    The extraction cache computes ``motion_prediction[t, p]`` as the current
    image gradient at local pixel ``p`` dotted with the measured source-coordinate
    displacement from ``t - 1`` to ``t``. This function integrates those signed
    increments into a pseudo-intensity trace. Every invalid or non-adjacent row
    starts a new segment at a fixed per-pixel median baseline; gaps are never
    bridged.

    This is a first-order coordinate-motion control. It is not optical flow and
    the image gradients still come from the observed frames.
    """

    dataset.validated()
    gap_factor = float(maximum_timestamp_gap_factor)
    if not np.isfinite(gap_factor) or gap_factor <= 1.0:
        raise ValueError("maximum_timestamp_gap_factor must be greater than one")
    traces = np.asarray(dataset.traces, dtype=np.float64)
    motion = np.asarray(dataset.motion_prediction, dtype=np.float64)
    source_xy = np.asarray(dataset.source_xy, dtype=np.float64)
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    base_valid = _base_valid(dataset)
    baseline = _finite_median_axis0(np.where(base_valid, traces, np.nan))
    nominal_dt = _nominal_dt(timestamps)
    maximum_gap = nominal_dt * gap_factor

    values = np.full(traces.shape, np.nan, dtype=np.float64)
    valid = np.zeros(traces.shape, dtype=bool)
    segment_index = np.full(traces.shape, -1, dtype=np.int32)
    reference_row = np.full(traces.shape, -1, dtype=np.int64)
    next_epoch = 0
    for pixel in range(dataset.pixel_count):
        previous = -2
        current_epoch = -1
        segment_reference = -1
        for row in np.flatnonzero(base_valid[:, pixel]):
            adjacent = (
                row == previous + 1
                and timestamps[row] - timestamps[previous] <= maximum_gap
                and np.isfinite(motion[row, pixel])
                and np.all(np.isfinite(source_xy[[previous, row], pixel]))
            )
            if adjacent and np.isfinite(values[previous, pixel]):
                values[row, pixel] = values[previous, pixel] + motion[row, pixel]
            elif np.isfinite(baseline[pixel]):
                values[row, pixel] = baseline[pixel]
                current_epoch = next_epoch
                next_epoch += 1
                segment_reference = int(row)
            if np.isfinite(values[row, pixel]):
                valid[row, pixel] = True
                segment_index[row, pixel] = int(current_epoch)
                reference_row[row, pixel] = int(segment_reference)
            previous = int(row)

    frame_valid = np.asarray(dataset.frame_valid, dtype=bool) & np.any(valid, axis=1)
    values[~valid] = np.nan
    return MotionControlResult(
        values=values,
        pixel_valid=valid,
        frame_valid=frame_valid,
        segment_index=segment_index,
        reference_row=reference_row,
        diagnostics={
            "control": "integrated_cached_gradient_dot_source_coordinate_step",
            "interpretation": "first_order_coordinate_motion_control_not_optical_flow",
            "maximum_timestamp_gap_factor": gap_factor,
            "valid_sample_fraction": float(np.mean(valid)),
            "valid_frame_fraction": float(np.mean(frame_valid)),
            "segment_start_count_across_pixels": int(next_epoch),
            "provenance_axis": "time_by_pixel",
        },
    )


def _contiguous_frame_segments(
    dataset: LocalCoordinateDataset,
    *,
    maximum_timestamp_gap_factor: float,
    maximum_interpolated_gap_seconds: float,
) -> tuple[np.ndarray, ...]:
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    frame_valid = np.asarray(dataset.frame_valid, dtype=bool)
    rows = np.flatnonzero(frame_valid)
    if not rows.size:
        return ()
    nominal_dt = _nominal_dt(timestamps)
    maximum_gap = max(
        nominal_dt * float(maximum_timestamp_gap_factor),
        float(maximum_interpolated_gap_seconds) + 1.05 * nominal_dt,
    )
    breaks = np.flatnonzero(
        np.diff(timestamps[rows]) > maximum_gap
    ) + 1
    return tuple(segment for segment in np.split(rows, breaks) if segment.size)


def _epoch_ranges(
    timestamps_s: np.ndarray,
    segment: np.ndarray,
    *,
    epoch_seconds: float,
) -> tuple[np.ndarray, ...]:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    epochs: list[np.ndarray] = []
    start = 0
    while start < segment.size:
        stop_time = float(timestamps[segment[start]]) + float(epoch_seconds)
        stop = int(np.searchsorted(timestamps[segment], stop_time, side="left"))
        stop = max(start + 1, min(stop, int(segment.size)))
        epochs.append(segment[start:stop])
        start = stop
    return tuple(epochs)


def _reference_row_for_epoch(
    dataset: LocalCoordinateDataset,
    rows: np.ndarray,
    *,
    minimum_template_pixels: int,
) -> int | None:
    base_valid = _base_valid(dataset)
    counts = np.sum(base_valid[rows], axis=1)
    eligible = counts >= int(minimum_template_pixels)
    if not np.any(eligible):
        return None
    candidates = rows[eligible]
    candidate_counts = counts[eligible]
    uncertainty = np.asarray(dataset.transform_uncertainty, dtype=np.float64)[candidates]
    finite_uncertainty = uncertainty[np.isfinite(uncertainty)]
    fallback = float(np.max(finite_uncertainty) + 1.0) if finite_uncertainty.size else 1.0
    uncertainty = np.where(np.isfinite(uncertainty), uncertainty, fallback)
    center_time = 0.5 * (
        float(dataset.timestamps_s[rows[0]]) + float(dataset.timestamps_s[rows[-1]])
    )
    distance = np.abs(np.asarray(dataset.timestamps_s)[candidates] - center_time)
    # Primary preference is a low-uncertainty frame, followed by broad cached
    # coverage and proximity to the epoch center. Row index breaks exact ties.
    order = np.lexsort((candidates, distance, -candidate_counts, uncertainty))
    return int(candidates[int(order[0])])


def _linear_static_surface(
    points_xy: np.ndarray,
    values: np.ndarray,
) -> LinearNDInterpolator | None:
    points = np.asarray(points_xy, dtype=np.float64)
    intensities = np.asarray(values, dtype=np.float64)
    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(intensities)
    points = points[finite]
    intensities = intensities[finite]
    if points.shape[0] < 3:
        return None
    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    if unique_points.shape[0] < 3:
        return None
    if unique_points.shape[0] != points.shape[0]:
        sums = np.bincount(inverse, weights=intensities)
        counts = np.bincount(inverse)
        intensities = sums / np.maximum(counts, 1)
        points = unique_points
    centered = points - np.mean(points, axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered) < 2:
        return None
    try:
        return LinearNDInterpolator(points, intensities, fill_value=np.nan)
    except (QhullError, ValueError):
        return None


def resample_static_reference_control(
    dataset: LocalCoordinateDataset,
    *,
    epoch_seconds: float = 4.0,
    guard_seconds: float = 0.1,
    minimum_template_pixels: int = 8,
    maximum_timestamp_gap_factor: float = 1.75,
    maximum_interpolated_gap_seconds: float = 0.02,
) -> MotionControlResult:
    """Resample frozen cached-frame surfaces at measured source coordinates.

    Within each contiguous epoch, one low-transform-uncertainty frame supplies a
    static intensity surface over its cached source-coordinate samples. Every
    other row is evaluated at its measured source coordinates using linear
    interpolation inside that sampled convex hull. No extrapolation is used.

    Because the cache contains only local-grid samples rather than complete
    source frames, this is a local static-template reconstruction, not a replay
    of a full static video and not optical flow.
    """

    dataset.validated()
    duration = float(epoch_seconds)
    guard = float(guard_seconds)
    minimum = int(minimum_template_pixels)
    gap_factor = float(maximum_timestamp_gap_factor)
    short_gap = float(maximum_interpolated_gap_seconds)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("epoch_seconds must be finite and positive")
    if not np.isfinite(guard) or guard < 0.0 or 2.0 * guard >= duration:
        raise ValueError("guard_seconds must be nonnegative and less than half an epoch")
    if minimum < 3:
        raise ValueError("minimum_template_pixels must be at least three")
    if not np.isfinite(gap_factor) or gap_factor <= 1.0:
        raise ValueError("maximum_timestamp_gap_factor must be greater than one")
    if not np.isfinite(short_gap) or short_gap < 0.0:
        raise ValueError("maximum_interpolated_gap_seconds must be nonnegative")

    traces = np.asarray(dataset.traces, dtype=np.float64)
    source_xy = np.asarray(dataset.source_xy, dtype=np.float64)
    base_valid = _base_valid(dataset)
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    values = np.full(traces.shape, np.nan, dtype=np.float64)
    valid = np.zeros(traces.shape, dtype=bool)
    segment_index = np.full(traces.shape, -1, dtype=np.int32)
    reference_row = np.full(traces.shape, -1, dtype=np.int64)
    attempted_epochs = 0
    successful_epochs = 0
    failed_surface_epochs = 0
    next_epoch = 0

    segments = _contiguous_frame_segments(
        dataset,
        maximum_timestamp_gap_factor=gap_factor,
        maximum_interpolated_gap_seconds=short_gap,
    )
    for segment in segments:
        for rows in _epoch_ranges(timestamps, segment, epoch_seconds=duration):
            attempted_epochs += 1
            reference = _reference_row_for_epoch(
                dataset,
                rows,
                minimum_template_pixels=minimum,
            )
            if reference is None:
                next_epoch += 1
                continue
            template_valid = base_valid[reference]
            surface = _linear_static_surface(
                source_xy[reference, template_valid],
                traces[reference, template_valid],
            )
            if surface is None:
                failed_surface_epochs += 1
                next_epoch += 1
                continue
            epoch_start = float(timestamps[rows[0]])
            epoch_stop = float(timestamps[rows[-1]])
            usable_rows = rows[
                (timestamps[rows] >= epoch_start + guard)
                & (timestamps[rows] <= epoch_stop - guard)
            ]
            for row in usable_rows:
                row_valid = base_valid[row]
                if not np.any(row_valid):
                    continue
                predicted = np.asarray(surface(source_xy[row, row_valid]), dtype=np.float64)
                finite = np.isfinite(predicted)
                pixels = np.flatnonzero(row_valid)[finite]
                values[row, pixels] = predicted[finite]
                valid[row, pixels] = True
                segment_index[row, pixels] = int(next_epoch)
                reference_row[row, pixels] = int(reference)
            if np.any(segment_index[rows] == next_epoch):
                successful_epochs += 1
            next_epoch += 1

    frame_valid = np.asarray(dataset.frame_valid, dtype=bool) & np.any(valid, axis=1)
    values[~valid] = np.nan
    return MotionControlResult(
        values=values,
        pixel_valid=valid,
        frame_valid=frame_valid,
        segment_index=segment_index,
        reference_row=reference_row,
        diagnostics={
            "control": "cached_static_reference_linear_resampling",
            "interpretation": "local_convex_hull_static_template_control_not_optical_flow",
            "epoch_seconds": duration,
            "guard_seconds": guard,
            "minimum_template_pixels": minimum,
            "maximum_timestamp_gap_factor": gap_factor,
            "maximum_interpolated_gap_seconds": short_gap,
            "attempted_epoch_count": int(attempted_epochs),
            "successful_epoch_count": int(successful_epochs),
            "failed_surface_epoch_count": int(failed_surface_epochs),
            "valid_sample_fraction": float(np.mean(valid)),
            "valid_frame_fraction": float(np.mean(frame_valid)),
            "provenance_axis": "time_by_pixel",
        },
    )


def tracking_feature_traces(
    dataset: LocalCoordinateDataset,
    region_mask: np.ndarray,
) -> MotionTrackingFeatures:
    """Summarize measured tracking exposure per frame for a frozen pixel region."""

    dataset.validated()
    selected = np.asarray(region_mask, dtype=bool)
    if selected.shape != (dataset.pixel_count,) or not np.any(selected):
        raise ValueError(f"region_mask must select pixels on axis ({dataset.pixel_count},)")
    base_valid = _base_valid(dataset)
    source_xy = np.asarray(dataset.source_xy, dtype=np.float64)
    motion = np.asarray(dataset.motion_prediction, dtype=np.float64)
    gradient = np.asarray(dataset.gradient_magnitude, dtype=np.float64)
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    maximum_gap = _nominal_dt(timestamps) * 1.75
    step = np.full((dataset.frame_count, dataset.pixel_count), np.nan, dtype=np.float64)
    adjacent = (
        np.asarray(dataset.frame_valid[1:], dtype=bool)
        & np.asarray(dataset.frame_valid[:-1], dtype=bool)
        & (np.diff(timestamps) <= maximum_gap)
    )
    pair_valid = base_valid[1:] & base_valid[:-1] & adjacent[:, None]
    differences = np.linalg.norm(source_xy[1:] - source_xy[:-1], axis=2)
    step[1:][pair_valid] = differences[pair_valid]

    def regional_median(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        output = np.full(dataset.frame_count, np.nan, dtype=np.float64)
        local_values = np.asarray(values, dtype=np.float64)[:, selected]
        local_valid = np.asarray(valid, dtype=bool)[:, selected] & np.isfinite(local_values)
        for row in range(dataset.frame_count):
            sample = local_values[row, local_valid[row]]
            if sample.size:
                output[row] = float(np.median(sample))
        return output

    region_valid = base_valid & np.isfinite(motion)
    selected_count = int(np.count_nonzero(selected))
    valid_fraction = np.sum(base_valid[:, selected], axis=1) / float(selected_count)
    return MotionTrackingFeatures(
        source_step_px=regional_median(step, np.isfinite(step)),
        abs_gradient_displacement=regional_median(np.abs(motion), region_valid),
        gradient_magnitude=regional_median(gradient, base_valid & np.isfinite(gradient)),
        transform_uncertainty=np.asarray(dataset.transform_uncertainty, dtype=np.float64).copy(),
        valid_pixel_fraction=np.asarray(valid_fraction, dtype=np.float64),
    )


__all__ = [
    "MotionControlResult",
    "MotionTrackingFeatures",
    "integrate_gradient_displacement_control",
    "resample_static_reference_control",
    "tracking_feature_traces",
]
