from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal


PoolingMethod = Literal["mean", "median", "trimmed_mean", "huber"]
ReferenceMode = Literal["log_ratio", "fractional_difference"]
LagAlignment = Literal["center", "trailing"]


def _time_channel_view(values: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if array.size < 2:
            raise ValueError("values must contain at least two time samples")
        return array[:, None], True
    if array.ndim == 2:
        if array.shape[0] < 2 or array.shape[1] < 1:
            raise ValueError("values must be nonempty time x channel data")
        return array, False
    raise ValueError("values must be one- or two-dimensional")


def _valid_for_values(values: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    if valid is None:
        return np.isfinite(values)
    supplied = np.asarray(valid, dtype=bool)
    if supplied.shape == (values.shape[0],):
        supplied = np.broadcast_to(supplied[:, None], values.shape)
    elif supplied.shape != values.shape:
        raise ValueError(
            f"valid shape {supplied.shape} must be ({values.shape[0]},) or {values.shape}"
        )
    return supplied & np.isfinite(values)


def _region_inputs(
    values: np.ndarray,
    region_mask: np.ndarray,
    valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("values must be nonempty time x pixel data")
    selected = np.asarray(region_mask, dtype=bool)
    if selected.shape != (array.shape[1],):
        raise ValueError(f"region_mask shape {selected.shape} must be ({array.shape[1]},)")
    if not np.any(selected):
        raise ValueError("region_mask must select at least one pixel")
    usable = _valid_for_values(array, valid)
    return array, selected, usable


def _huber_locations(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    c: float,
    min_count: int,
) -> np.ndarray:
    counts = np.sum(valid, axis=1)
    eligible = counts >= int(min_count)
    masked = np.where(valid, values, np.nan)
    locations = np.full(values.shape[0], np.nan, dtype=np.float64)
    scales = np.full(values.shape[0], np.nan, dtype=np.float64)
    if np.any(eligible):
        eligible_values = masked[eligible]
        eligible_locations = np.nanmedian(eligible_values, axis=1)
        locations[eligible] = eligible_locations
        scales[eligible] = 1.4826 * np.nanmedian(
            np.abs(eligible_values - eligible_locations[:, None]), axis=1
        )
    output = np.full(values.shape[0], np.nan, dtype=np.float64)
    scale_floor = np.finfo(np.float64).eps * np.maximum(1.0, np.abs(locations)) * 16.0
    fixed = eligible & np.isfinite(locations) & (
        ~np.isfinite(scales) | (scales <= scale_floor)
    )
    output[fixed] = locations[fixed]
    active = eligible & np.isfinite(locations) & np.isfinite(scales) & ~fixed
    if not np.any(active):
        return output

    cutoffs = float(c) * scales
    for _iteration in range(30):
        residual = np.abs(values - locations[:, None])
        weights = np.where(valid, 1.0, 0.0)
        outside = valid & (residual > cutoffs[:, None])
        weights[outside] = np.broadcast_to(cutoffs[:, None], values.shape)[outside] / residual[
            outside
        ]
        denominators = np.sum(weights, axis=1)
        updated = np.divide(
            np.sum(weights * np.where(valid, values, 0.0), axis=1),
            denominators,
            out=locations.copy(),
            where=denominators > 0.0,
        )
        tolerances = 1e-10 * np.maximum.reduce(
            [np.ones(locations.shape), np.abs(locations), scales]
        )
        converged = np.abs(updated - locations) <= tolerances
        locations[active] = updated[active]
        active &= ~converged
        if not np.any(active):
            break
    output[eligible & np.isfinite(locations)] = locations[eligible & np.isfinite(locations)]
    return output


def regional_pool(
    values: np.ndarray,
    region_mask: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    method: PoolingMethod = "trimmed_mean",
    trim_fraction: float = 0.1,
    huber_c: float = 1.345,
    min_valid_pixels: int = 1,
) -> np.ndarray:
    """Pool each frame over a fixed region while rejecting invalid pixels.

    ``trimmed_mean`` removes the same fraction from both tails independently
    in each frame. ``huber`` uses a median/MAD initialization and fixed Huber
    tuning, so a few spatial outliers cannot dominate the regional trace.
    """

    array, selected, usable = _region_inputs(values, region_mask, valid)
    if method not in {"mean", "median", "trimmed_mean", "huber"}:
        raise ValueError(f"unsupported pooling method {method!r}")
    trim = float(trim_fraction)
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    if not float(huber_c) > 0.0:
        raise ValueError("huber_c must be positive")
    minimum = int(min_valid_pixels)
    if minimum < 1:
        raise ValueError("min_valid_pixels must be positive")

    region_values = array[:, selected]
    region_valid = usable[:, selected]
    if method == "huber":
        return _huber_locations(
            region_values,
            region_valid,
            c=float(huber_c),
            min_count=minimum,
        )
    output = np.full(array.shape[0], np.nan, dtype=np.float64)
    for row in range(array.shape[0]):
        sample = region_values[row, region_valid[row]]
        if sample.size < minimum:
            continue
        if method == "mean":
            output[row] = float(np.mean(sample))
        elif method == "median":
            output[row] = float(np.median(sample))
        elif method == "trimmed_mean":
            tail = int(np.floor(trim * sample.size))
            ordered = np.sort(sample)
            retained = ordered[tail : sample.size - tail] if tail else ordered
            if retained.size >= minimum:
                output[row] = float(np.mean(retained))
    return output


def masked_gaussian_smooth(
    values: np.ndarray,
    pixel_xy: np.ndarray,
    region_mask: np.ndarray,
    *,
    sigma_px: float = 0.8,
    valid: np.ndarray | None = None,
    truncate: float = 3.0,
    min_weight: float = 1e-12,
) -> np.ndarray:
    """Apply fixed normalized Gaussian smoothing using only same-mask pixels.

    The fixed pairwise kernel is evaluated on the local pixel grid. Each frame
    is normalized by its available spatial weights, equivalent to smoothing
    ``image * valid_mask`` and dividing by the smoothed valid mask. Values from
    outside ``region_mask`` can never contribute, and no temporal filling is
    performed.
    """

    array, selected, usable = _region_inputs(values, region_mask, valid)
    coordinates = np.asarray(pixel_xy, dtype=np.float64)
    if coordinates.shape != (array.shape[1], 2):
        raise ValueError(f"pixel_xy shape {coordinates.shape} must be ({array.shape[1]}, 2)")
    if not np.isfinite(coordinates).all():
        raise ValueError("pixel_xy must be finite")
    sigma = float(sigma_px)
    if sigma <= 0.0:
        raise ValueError("sigma_px must be positive")
    cutoff = float(truncate)
    if cutoff <= 0.0:
        raise ValueError("truncate must be positive")
    threshold = float(min_weight)
    if threshold <= 0.0:
        raise ValueError("min_weight must be positive")

    selected_indices = np.flatnonzero(selected)
    local_xy = coordinates[selected_indices]
    offsets = local_xy[:, None, :] - local_xy[None, :, :]
    support_radius = cutoff * sigma
    within_support = np.all(np.abs(offsets) <= support_radius, axis=2)
    squared_distance = np.sum(np.square(offsets), axis=2)
    weights = np.exp(-0.5 * squared_distance / (sigma * sigma))
    weights[~within_support] = 0.0

    local_values = array[:, selected_indices]
    local_valid = usable[:, selected_indices]
    numerator = np.where(local_valid, local_values, 0.0) @ weights.T
    denominator = local_valid.astype(np.float64) @ weights.T
    smoothed = np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator >= threshold,
    )
    output = np.full(array.shape, np.nan, dtype=np.float64)
    output[:, selected_indices] = smoothed
    return output


def reference_normalize(
    signal_values: np.ndarray,
    reference_values: np.ndarray,
    *,
    mode: ReferenceMode = "log_ratio",
    valid: np.ndarray | None = None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize signal and reference intensities with a fixed transformation."""

    signal_array = np.asarray(signal_values, dtype=np.float64)
    reference_array = np.asarray(reference_values, dtype=np.float64)
    if signal_array.shape != reference_array.shape or signal_array.ndim not in {1, 2}:
        raise ValueError("signal_values and reference_values must have the same 1D or 2D shape")
    if signal_array.size == 0:
        raise ValueError("signal_values cannot be empty")
    if mode not in {"log_ratio", "fractional_difference"}:
        raise ValueError(f"unsupported reference normalization mode {mode!r}")
    offset = float(epsilon)
    if offset <= 0.0:
        raise ValueError("epsilon must be positive")

    signal_matrix, squeezed = _time_channel_view(signal_array)
    reference_matrix, _ = _time_channel_view(reference_array)
    usable = _valid_for_values(signal_matrix, valid) & np.isfinite(reference_matrix)
    output = np.full(signal_matrix.shape, np.nan, dtype=np.float64)
    if mode == "log_ratio":
        positive = (signal_matrix + offset > 0.0) & (reference_matrix + offset > 0.0)
        use = usable & positive
        output[use] = np.log(signal_matrix[use] + offset) - np.log(
            reference_matrix[use] + offset
        )
    else:
        denominator = signal_matrix + reference_matrix + offset
        use = usable & np.isfinite(denominator) & (
            np.abs(denominator) > np.finfo(np.float64).eps
        )
        output[use] = 2.0 * (signal_matrix[use] - reference_matrix[use]) / denominator[use]
    return output[:, 0] if squeezed else output


def _validated_timestamps(timestamps_s: np.ndarray, expected_count: int) -> tuple[np.ndarray, float]:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.shape != (expected_count,):
        raise ValueError(f"timestamps_s shape {timestamps.shape} must be ({expected_count},)")
    differences = np.diff(timestamps)
    if not np.isfinite(timestamps).all() or np.any(differences <= 0.0):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    return timestamps, float(np.median(differences))


def _valid_segments(
    timestamps_s: np.ndarray,
    valid: np.ndarray,
    *,
    nominal_dt: float,
    max_gap_factor: float,
) -> tuple[np.ndarray, ...]:
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return ()
    adjacent_index = np.diff(indices) == 1
    adjacent_time = np.diff(timestamps_s[indices]) <= float(max_gap_factor) * nominal_dt
    breaks = np.flatnonzero(~(adjacent_index & adjacent_time)) + 1
    return tuple(part for part in np.split(indices, breaks) if part.size)


def _irregular_polynomial_derivative(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    *,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    half_window = window_length // 2
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for center in range(half_window, values.size - half_window):
        window = slice(center - half_window, center + half_window + 1)
        offsets = timestamps_s[window] - timestamps_s[center]
        scale = float(np.max(np.abs(offsets)))
        if not scale > 0.0:
            continue
        design = np.vander(offsets / scale, N=polyorder + 1, increasing=True)
        coefficients, _residuals, rank, _singular = np.linalg.lstsq(
            design,
            values[window],
            rcond=None,
        )
        if int(rank) == polyorder + 1:
            output[center] = float(coefficients[1] / scale)
    return output


def segmented_savgol_derivative(
    values: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    window_length: int = 7,
    polyorder: int = 3,
    max_gap_factor: float = 1.75,
) -> np.ndarray:
    """Compute a centered signed derivative within contiguous valid segments.

    Uniform segments use SciPy's Savitzky-Golay implementation with their
    measured median timestep. Irregular segments use the equivalent centered
    local polynomial fit on the actual timestamps. Half a window at every
    segment boundary remains ``NaN``; invalid gaps are never interpolated.
    """

    matrix, squeezed = _time_channel_view(values)
    timestamps, nominal_dt = _validated_timestamps(timestamps_s, matrix.shape[0])
    usable = _valid_for_values(matrix, valid)
    window = int(window_length)
    order = int(polyorder)
    if window < 3 or window % 2 != 1:
        raise ValueError("window_length must be an odd integer of at least three")
    if order < 1 or order >= window:
        raise ValueError("polyorder must be positive and less than window_length")
    gap_factor = float(max_gap_factor)
    if gap_factor <= 1.0:
        raise ValueError("max_gap_factor must be greater than one")

    output = np.full(matrix.shape, np.nan, dtype=np.float64)
    half_window = window // 2
    for channel in range(matrix.shape[1]):
        segments = _valid_segments(
            timestamps,
            usable[:, channel],
            nominal_dt=nominal_dt,
            max_gap_factor=gap_factor,
        )
        for rows in segments:
            if rows.size < window:
                continue
            segment_timestamps = timestamps[rows]
            segment_values = matrix[rows, channel]
            differences = np.diff(segment_timestamps)
            segment_dt = float(np.median(differences))
            regular_tolerance = max(1e-12, 1e-4 * segment_dt)
            if np.max(np.abs(differences - segment_dt)) <= regular_tolerance:
                derivative = signal.savgol_filter(
                    segment_values,
                    window_length=window,
                    polyorder=order,
                    deriv=1,
                    delta=segment_dt,
                    mode="interp",
                )
            else:
                derivative = _irregular_polynomial_derivative(
                    segment_timestamps,
                    segment_values,
                    window_length=window,
                    polyorder=order,
                )
            retained = rows[half_window : rows.size - half_window]
            output[retained, channel] = derivative[half_window : rows.size - half_window]
    return output[:, 0] if squeezed else output


def normalized_signed_lag_difference(
    values: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    lag_frames: int,
    valid: np.ndarray | None = None,
    max_gap_factor: float = 1.75,
    alignment: LagAlignment = "center",
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute ``2 * (later - earlier) / (later + earlier + epsilon)``.

    Pairs are formed only within contiguous valid segments. Center alignment
    associates each difference with the midpoint sample and therefore requires
    an even lag. Trailing alignment associates it with the later sample.
    """

    matrix, squeezed = _time_channel_view(values)
    timestamps, nominal_dt = _validated_timestamps(timestamps_s, matrix.shape[0])
    usable = _valid_for_values(matrix, valid)
    lag = int(lag_frames)
    if lag < 1 or lag >= matrix.shape[0]:
        raise ValueError("lag_frames must be positive and smaller than the time axis")
    if alignment not in {"center", "trailing"}:
        raise ValueError(f"unsupported lag alignment {alignment!r}")
    if alignment == "center" and lag % 2:
        raise ValueError("center alignment requires an even lag_frames value")
    gap_factor = float(max_gap_factor)
    if gap_factor <= 1.0:
        raise ValueError("max_gap_factor must be greater than one")
    offset = float(epsilon)
    if offset <= 0.0:
        raise ValueError("epsilon must be positive")

    output = np.full(matrix.shape, np.nan, dtype=np.float64)
    for channel in range(matrix.shape[1]):
        segments = _valid_segments(
            timestamps,
            usable[:, channel],
            nominal_dt=nominal_dt,
            max_gap_factor=gap_factor,
        )
        for rows in segments:
            if rows.size <= lag:
                continue
            earlier = matrix[rows[:-lag], channel]
            later = matrix[rows[lag:], channel]
            denominator = later + earlier + offset
            differences = np.divide(
                2.0 * (later - earlier),
                denominator,
                out=np.full(denominator.shape, np.nan, dtype=np.float64),
                where=np.isfinite(denominator)
                & (np.abs(denominator) > np.finfo(np.float64).eps),
            )
            if alignment == "center":
                targets = rows[lag // 2 : rows.size - lag // 2]
            else:
                targets = rows[lag:]
            output[targets, channel] = differences
    return output[:, 0] if squeezed else output


def regional_spatial_std(
    values: np.ndarray,
    region_mask: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    ddof: int = 1,
    min_valid_pixels: int = 2,
) -> np.ndarray:
    """Return the per-frame spatial standard deviation inside a fixed region."""

    array, selected, usable = _region_inputs(values, region_mask, valid)
    correction = int(ddof)
    minimum = int(min_valid_pixels)
    if correction < 0:
        raise ValueError("ddof cannot be negative")
    if minimum < 1:
        raise ValueError("min_valid_pixels must be positive")

    region_values = array[:, selected]
    region_valid = usable[:, selected]
    counts = np.sum(region_valid, axis=1)
    sums = np.sum(np.where(region_valid, region_values, 0.0), axis=1)
    means = np.divide(
        sums,
        counts,
        out=np.zeros(array.shape[0], dtype=np.float64),
        where=counts > 0,
    )
    squared_deviation = np.where(
        region_valid,
        np.square(region_values - means[:, None]),
        0.0,
    )
    denominator = counts - correction
    variance = np.divide(
        np.sum(squared_deviation, axis=1),
        denominator,
        out=np.full(array.shape[0], np.nan, dtype=np.float64),
        where=(counts >= minimum) & (denominator > 0),
    )
    return np.sqrt(np.maximum(variance, 0.0))


__all__ = [
    "masked_gaussian_smooth",
    "normalized_signed_lag_difference",
    "reference_normalize",
    "regional_pool",
    "regional_spatial_std",
    "segmented_savgol_derivative",
]
