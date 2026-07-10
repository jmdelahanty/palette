from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np


def _quiet_nanmedian(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.asarray(np.nanmedian(values, axis=axis))


def _finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(values)
    finite = np.isfinite(arr)
    numerator = np.sum(np.where(finite, arr, 0), axis=axis)
    denominator = np.sum(finite, axis=axis)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=arr.dtype),
        where=denominator > 0,
    )


@dataclass(frozen=True)
class LocalCoordinateDataset:
    """Source-pixel samples expressed on a fixed local anatomical grid."""

    frame_indices: np.ndarray
    timestamps_s: np.ndarray
    traces: np.ndarray
    pixel_xy: np.ndarray
    pixel_valid: np.ndarray
    frame_valid: np.ndarray
    source_xy: np.ndarray
    bilinear_weights: np.ndarray
    body_occupancy: np.ndarray
    eye_occupancy: np.ndarray
    gradient_magnitude: np.ndarray
    motion_prediction: np.ndarray
    nuisance_values: np.ndarray
    nuisance_names: tuple[str, ...]
    image_shape_hw: tuple[int, int]
    administrative_boundary_distance_px: np.ndarray
    physical_boundary_distance_px: np.ndarray
    transform_uncertainty: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validated(self) -> LocalCoordinateDataset:
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        timestamps_s = np.asarray(self.timestamps_s, dtype=np.float64)
        traces = np.asarray(self.traces, dtype=np.float64)
        pixel_xy = np.asarray(self.pixel_xy, dtype=np.float64)
        pixel_valid = np.asarray(self.pixel_valid, dtype=bool)
        frame_valid = np.asarray(self.frame_valid, dtype=bool)
        source_xy = np.asarray(self.source_xy, dtype=np.float64)
        bilinear_weights = np.asarray(self.bilinear_weights, dtype=np.float64)
        t_count, pixel_count = traces.shape
        expected = {
            "frame_indices": (t_count,),
            "timestamps_s": (t_count,),
            "pixel_xy": (pixel_count, 2),
            "pixel_valid": (t_count, pixel_count),
            "frame_valid": (t_count,),
            "source_xy": (t_count, pixel_count, 2),
            "bilinear_weights": (t_count, pixel_count, 4),
            "body_occupancy": (t_count, pixel_count),
            "eye_occupancy": (t_count, pixel_count),
            "gradient_magnitude": (t_count, pixel_count),
            "motion_prediction": (t_count, pixel_count),
            "transform_uncertainty": (t_count,),
            "administrative_boundary_distance_px": (pixel_count,),
            "physical_boundary_distance_px": (pixel_count,),
        }
        actual = {
            "frame_indices": frame_indices.shape,
            "timestamps_s": timestamps_s.shape,
            "pixel_xy": pixel_xy.shape,
            "pixel_valid": pixel_valid.shape,
            "frame_valid": frame_valid.shape,
            "source_xy": source_xy.shape,
            "bilinear_weights": bilinear_weights.shape,
            "body_occupancy": np.asarray(self.body_occupancy).shape,
            "eye_occupancy": np.asarray(self.eye_occupancy).shape,
            "gradient_magnitude": np.asarray(self.gradient_magnitude).shape,
            "motion_prediction": np.asarray(self.motion_prediction).shape,
            "transform_uncertainty": np.asarray(self.transform_uncertainty).shape,
            "administrative_boundary_distance_px": np.asarray(
                self.administrative_boundary_distance_px
            ).shape,
            "physical_boundary_distance_px": np.asarray(self.physical_boundary_distance_px).shape,
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} shape {actual[name]} does not match {shape}")
        nuisance = np.asarray(self.nuisance_values, dtype=np.float64)
        if nuisance.ndim != 2 or nuisance.shape[0] != t_count:
            raise ValueError(f"nuisance_values shape {nuisance.shape} must be ({t_count}, K)")
        if nuisance.shape[1] != len(self.nuisance_names):
            raise ValueError("nuisance_names length does not match nuisance_values columns")
        if t_count < 2 or pixel_count < 1:
            raise ValueError("dataset needs at least two frames and one pixel")
        if not np.isfinite(timestamps_s).all() or np.any(np.diff(timestamps_s) <= 0.0):
            raise ValueError("timestamps_s must be finite and strictly increasing")
        if np.any(np.diff(frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        valid_values = pixel_valid
        if not np.isfinite(traces[valid_values]).all():
            raise ValueError("valid pixel samples must have finite trace values")
        if not np.isfinite(source_xy[valid_values]).all():
            raise ValueError("valid pixel samples must have finite source coordinates")
        if not np.isfinite(bilinear_weights[valid_values]).all():
            raise ValueError("valid pixel samples must have finite bilinear weights")
        if not np.allclose(np.nansum(bilinear_weights, axis=2)[pixel_valid], 1.0, atol=1e-5):
            raise ValueError("valid bilinear weights must sum to one")
        return self

    @property
    def frame_count(self) -> int:
        return int(np.asarray(self.traces).shape[0])

    @property
    def pixel_count(self) -> int:
        return int(np.asarray(self.traces).shape[1])


@dataclass(frozen=True)
class HeartrateConfig:
    band_min_hz: float = 1.5
    band_max_hz: float = 3.5
    frequency_step_hz: float = 0.05
    partition_block_seconds: float = 4.0
    partition_guard_seconds: float = 0.25
    min_partition_blocks_per_fold: int = 2
    discovery_chunk_seconds: float = 4.0
    min_chunk_seconds: float = 2.0
    min_chunk_valid_fraction: float = 0.8
    max_timestamp_gap_factor: float = 1.75
    max_interpolated_gap_seconds: float = 0.02
    min_pixel_valid_fraction: float = 0.8
    min_body_occupancy: float = 0.8
    max_eye_occupancy: float = 0.05
    min_physical_boundary_distance_px: float = 1.0
    max_warp_invalid_fraction: float = 0.1
    gradient_risk_weight: float = 0.25
    boundary_risk_weight: float = 0.5
    warp_risk_weight: float = 1.0
    transform_risk_weight: float = 0.25
    pixel_score_threshold_z: float = 1.5
    min_cluster_pixels: int = 3
    nuisance_ridge: float = 1e-6
    surrogate_count: int = 199
    surrogate_spatial_block_px: int = 2
    surrogate_min_shift_seconds: float = 1.0
    alpha: float = 0.05
    min_control_ratio: float = 1.1
    min_crossfit_dilated_overlap: float = 0.5
    max_crossfit_frequency_difference_hz: float = 0.1
    event_polarity: str = "darkening"
    event_prominence_mad: float = 1.0
    event_filter_edge_seconds: float = 0.75
    random_seed: int = 0

    def validated(self) -> HeartrateConfig:
        if not (0.0 < float(self.band_min_hz) < float(self.band_max_hz)):
            raise ValueError("band_min_hz must be positive and below band_max_hz")
        if float(self.frequency_step_hz) <= 0.0:
            raise ValueError("frequency_step_hz must be positive")
        if float(self.partition_block_seconds) <= 0.0:
            raise ValueError("partition_block_seconds must be positive")
        if int(self.min_partition_blocks_per_fold) < 1:
            raise ValueError("min_partition_blocks_per_fold must be positive")
        if float(self.discovery_chunk_seconds) <= 0.0 or float(self.min_chunk_seconds) <= 0.0:
            raise ValueError("chunk durations must be positive")
        if float(self.max_interpolated_gap_seconds) < 0.0:
            raise ValueError("max_interpolated_gap_seconds cannot be negative")
        if int(self.surrogate_count) < 0:
            raise ValueError("surrogate_count cannot be negative")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be between zero and one")
        if float(self.max_crossfit_frequency_difference_hz) < 0.0:
            raise ValueError("max_crossfit_frequency_difference_hz cannot be negative")
        if str(self.event_polarity) not in {"darkening", "brightening", "auto"}:
            raise ValueError("event_polarity must be darkening, brightening, or auto")
        return self


@dataclass(frozen=True)
class RiskSurfaces:
    eligible: np.ndarray
    combined_penalty: np.ndarray
    body_occupancy: np.ndarray
    eye_occupancy: np.ndarray
    warp_invalid_fraction: np.ndarray
    gradient_risk: np.ndarray
    transform_risk: np.ndarray
    physical_boundary_risk: np.ndarray
    physical_boundary_distance_px: np.ndarray
    administrative_boundary_distance_px: np.ndarray


@dataclass(frozen=True)
class NuisanceModel:
    center: np.ndarray
    scale: np.ndarray
    beta: np.ndarray
    motion_beta: np.ndarray
    fitted_pixels: np.ndarray
    nuisance_names: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryCandidate:
    frequency_hz: float
    cluster_mass: float
    pixel_indices: np.ndarray
    pixel_weights: np.ndarray
    pixel_scores: np.ndarray
    cluster_mask: np.ndarray
    spatial_phase_coherence: float
    chunk_phase_coherence: float
    chunk_count: int


@dataclass(frozen=True)
class CalibratedDiscovery:
    candidate: DiscoveryCandidate
    null_max_cluster_mass: np.ndarray
    p_value: float
    threshold: float
    detected: bool
    nuisance_model: NuisanceModel
    residual: np.ndarray


@dataclass(frozen=True)
class EventSeries:
    frame_indices: np.ndarray
    timestamps_s: np.ndarray
    filtered_values: np.ndarray
    prominences: np.ndarray
    intervals_s: np.ndarray
    instantaneous_bpm: np.ndarray
    rejected_edge_events: int
    analyzed_intervals_s: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class FoldResult:
    fold_index: int
    discovery: CalibratedDiscovery
    confirmation_p_value: float
    confirmation_score: float
    confirmation_null_scores: np.ndarray
    confirmation_chunk_scores: np.ndarray
    confirmation_chunk_p_values: np.ndarray
    confirmed_chunk_count: int
    confirmation_chunk_count: int
    control_scores: Mapping[str, float]
    control_ratio: float
    confirmed: bool
    polarity: str
    confirmation_trace: np.ndarray
    confirmation_valid: np.ndarray
    event_valid: np.ndarray
    events: EventSeries | None


@dataclass(frozen=True)
class HeartrateResult:
    detected: bool
    reason: str
    folds: tuple[FoldResult, ...]
    crossfit_dilated_overlap: float
    crossfit_frequency_difference_hz: float
    event_frame_indices: np.ndarray
    event_timestamps_s: np.ndarray
    event_intervals_s: np.ndarray
    instantaneous_bpm: np.ndarray
    coverage_fraction: float
    no_estimate_intervals_s: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class InjectionSpec:
    amplitude_sigma: float
    frequency_hz: float
    center_xy: tuple[float, float]
    radius_px: float
    phase_rad: float = 0.0
    phase_drift_hz_per_s: float = 0.0
    active_fraction: float = 1.0


def bilinear_sample(
    image: np.ndarray,
    points_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a 2-D image and return values, validity, and four source weights."""

    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"bilinear_sample expects a 2-D image, got {arr.shape}")
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (
        np.isfinite(points).all(axis=1)
        & (x0 >= 0)
        & (y0 >= 0)
        & (x1 < arr.shape[1])
        & (y1 < arr.shape[0])
    )
    dx = x - x0
    dy = y - y0
    weights = np.column_stack(
        [
            (1.0 - dx) * (1.0 - dy),
            dx * (1.0 - dy),
            (1.0 - dx) * dy,
            dx * dy,
        ]
    )
    weights[~valid] = np.nan
    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    rows = np.flatnonzero(valid)
    if rows.size:
        values[rows] = (
            weights[rows, 0] * arr[y0[rows], x0[rows]]
            + weights[rows, 1] * arr[y0[rows], x1[rows]]
            + weights[rows, 2] * arr[y1[rows], x0[rows]]
            + weights[rows, 3] * arr[y1[rows], x1[rows]]
        )
    return values, valid, weights


def _robust_center_scale(values: np.ndarray, axis: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=axis)
    if axis is None:
        deviations = np.abs(values - median)
    else:
        deviations = np.abs(values - np.expand_dims(median, axis=axis))
    scale = 1.4826 * np.nanmedian(deviations, axis=axis)
    fallback = np.nanstd(values, axis=axis)
    scale = np.where(np.isfinite(scale) & (scale > np.finfo(float).eps), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > np.finfo(float).eps), scale, 1.0)
    return np.asarray(median), np.asarray(scale)


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    center, scale = _robust_center_scale(np.asarray(values, dtype=np.float64))
    out = (np.asarray(values, dtype=np.float64) - float(center)) / float(scale)
    out[~np.isfinite(out)] = 0.0
    return out


def build_risk_surfaces(dataset: LocalCoordinateDataset, config: HeartrateConfig) -> RiskSurfaces:
    dataset.validated()
    config.validated()
    frame_ok = np.asarray(dataset.frame_valid, dtype=bool)
    pixel_valid = np.asarray(dataset.pixel_valid, dtype=bool) & frame_ok[:, None]
    denominator = max(1, int(np.count_nonzero(frame_ok)))
    valid_fraction = np.count_nonzero(pixel_valid, axis=0) / float(denominator)
    warp_invalid = 1.0 - valid_fraction
    body = np.nanmedian(np.where(pixel_valid, dataset.body_occupancy, np.nan), axis=0)
    eye = np.nanquantile(np.where(pixel_valid, dataset.eye_occupancy, np.nan), 0.95, axis=0)
    gradient = np.nanmedian(np.where(pixel_valid, dataset.gradient_magnitude, np.nan), axis=0)
    motion = np.nanmedian(np.abs(np.where(pixel_valid, dataset.motion_prediction, np.nan)), axis=0)
    uncertainty_values = np.asarray(dataset.transform_uncertainty, dtype=np.float64)
    uncertainty_scale = float(np.nanmedian(uncertainty_values[frame_ok])) if np.any(frame_ok) else 0.0
    boundary_distance = np.asarray(dataset.physical_boundary_distance_px, dtype=np.float64)
    boundary_risk = np.divide(
        1.0,
        np.maximum(boundary_distance, 0.25),
        out=np.full(boundary_distance.shape, 4.0),
        where=np.isfinite(boundary_distance),
    )
    eligible = (
        (valid_fraction >= float(config.min_pixel_valid_fraction))
        & (body >= float(config.min_body_occupancy))
        & (eye <= float(config.max_eye_occupancy))
        & (boundary_distance >= float(config.min_physical_boundary_distance_px))
        & (warp_invalid <= float(config.max_warp_invalid_fraction))
        & np.isfinite(gradient)
    )
    # Standardize first, then apply weights. This keeps configured weights effective.
    transform_risk = motion + np.nan_to_num(gradient, nan=0.0) * max(0.0, uncertainty_scale)
    penalty = (
        float(config.gradient_risk_weight) * np.maximum(0.0, _robust_zscore(gradient))
        + float(config.boundary_risk_weight) * np.maximum(0.0, _robust_zscore(boundary_risk))
        + float(config.warp_risk_weight) * np.maximum(0.0, _robust_zscore(warp_invalid))
        + float(config.transform_risk_weight) * np.maximum(0.0, _robust_zscore(transform_risk))
    )
    penalty[~np.isfinite(penalty)] = 0.0
    return RiskSurfaces(
        eligible=eligible,
        combined_penalty=penalty,
        body_occupancy=body,
        eye_occupancy=eye,
        warp_invalid_fraction=warp_invalid,
        gradient_risk=gradient,
        transform_risk=transform_risk,
        physical_boundary_risk=boundary_risk,
        physical_boundary_distance_px=boundary_distance,
        administrative_boundary_distance_px=np.asarray(
            dataset.administrative_boundary_distance_px, dtype=np.float64
        ),
    )


def alternating_block_partitions(
    timestamps_s: np.ndarray,
    *,
    block_seconds: float,
    guard_seconds: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.size < 2:
        raise ValueError("at least two timestamps are required")
    relative = timestamps - timestamps[0]
    block = np.floor(relative / float(block_seconds)).astype(np.int64)
    offset = np.mod(relative, float(block_seconds))
    half_guard = max(0.0, float(guard_seconds)) / 2.0
    kept = (offset >= half_guard) & (offset <= float(block_seconds) - half_guard)
    even = kept & ((block % 2) == 0)
    odd = kept & ((block % 2) == 1)
    if not np.any(even) or not np.any(odd):
        raise ValueError("recording is too short for two disjoint block partitions")
    return ((even, odd), (odd, even))


def _nuisance_design(dataset: LocalCoordinateDataset) -> tuple[np.ndarray, tuple[str, ...]]:
    base = np.asarray(dataset.nuisance_values, dtype=np.float64)
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    dt = np.diff(timestamps, prepend=timestamps[0])
    median_dt = float(np.nanmedian(np.diff(timestamps)))
    dt[0] = median_dt
    derivatives = np.full_like(base, np.nan, dtype=np.float64)
    if base.shape[1]:
        derivatives[1:] = np.diff(base, axis=0) / np.maximum(dt[1:, None], np.finfo(float).eps)
        derivatives[0] = derivatives[1] if base.shape[0] > 1 else 0.0
    names = tuple(dataset.nuisance_names) + tuple(f"d_{name}_dt" for name in dataset.nuisance_names)
    return np.column_stack([base, derivatives]), names


def fit_nuisance_model(
    dataset: LocalCoordinateDataset,
    fit_rows: np.ndarray,
    *,
    ridge: float,
) -> NuisanceModel:
    design, names = _nuisance_design(dataset)
    rows = np.asarray(fit_rows, dtype=bool) & np.asarray(dataset.frame_valid, dtype=bool)
    if int(np.count_nonzero(rows)) < max(8, design.shape[1] + 2):
        raise ValueError("too few discovery rows to fit nuisance model")
    center, scale = _robust_center_scale(design[rows], axis=0)
    standardized = (design - center[None, :]) / scale[None, :]
    standardized[~np.isfinite(standardized)] = 0.0
    scalar = np.column_stack([np.ones(dataset.frame_count), standardized])
    traces = np.asarray(dataset.traces, dtype=np.float64)
    motion = np.asarray(dataset.motion_prediction, dtype=np.float64)
    pixel_valid = np.asarray(dataset.pixel_valid, dtype=bool)
    beta = np.full((scalar.shape[1], dataset.pixel_count), np.nan, dtype=np.float64)
    motion_beta = np.full(dataset.pixel_count, np.nan, dtype=np.float64)
    fitted = np.zeros(dataset.pixel_count, dtype=bool)
    penalty = float(ridge)
    for pixel in range(dataset.pixel_count):
        ok = rows & pixel_valid[:, pixel] & np.isfinite(traces[:, pixel]) & np.isfinite(motion[:, pixel])
        if int(np.count_nonzero(ok)) < scalar.shape[1] + 2:
            continue
        x = np.column_stack([scalar[ok], motion[ok, pixel]])
        y = traces[ok, pixel]
        gram = x.T @ x
        regularizer = np.eye(gram.shape[0], dtype=np.float64) * penalty
        regularizer[0, 0] = 0.0
        coefficients = np.linalg.solve(gram + regularizer, x.T @ y)
        beta[:, pixel] = coefficients[:-1]
        motion_beta[pixel] = coefficients[-1]
        fitted[pixel] = True
    return NuisanceModel(
        center=np.asarray(center, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        beta=beta,
        motion_beta=motion_beta,
        fitted_pixels=fitted,
        nuisance_names=names,
    )


def apply_nuisance_model(dataset: LocalCoordinateDataset, model: NuisanceModel) -> np.ndarray:
    design, names = _nuisance_design(dataset)
    if names != model.nuisance_names:
        raise ValueError("nuisance model columns do not match dataset")
    standardized = (design - model.center[None, :]) / model.scale[None, :]
    standardized[~np.isfinite(standardized)] = 0.0
    scalar = np.column_stack([np.ones(dataset.frame_count), standardized])
    predicted = scalar @ model.beta
    predicted += np.asarray(dataset.motion_prediction, dtype=np.float64) * model.motion_beta[None, :]
    residual = np.asarray(dataset.traces, dtype=np.float64) - predicted
    valid = np.asarray(dataset.pixel_valid, dtype=bool) & np.asarray(dataset.frame_valid, dtype=bool)[:, None]
    residual[~valid] = np.nan
    residual[:, ~model.fitted_pixels] = np.nan
    return residual


def contiguous_segments(
    timestamps_s: np.ndarray,
    valid: np.ndarray,
    *,
    max_gap_factor: float,
    min_seconds: float = 0.0,
) -> list[np.ndarray]:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    accepted = np.asarray(valid, dtype=bool) & np.isfinite(timestamps)
    if timestamps.size != accepted.size:
        raise ValueError("timestamps and valid must have the same length")
    finite_diffs = np.diff(timestamps[np.isfinite(timestamps)])
    positive = finite_diffs[finite_diffs > 0.0]
    if positive.size == 0:
        return []
    nominal_dt = float(np.median(positive))
    max_gap = nominal_dt * float(max_gap_factor)
    segments: list[np.ndarray] = []
    current: list[int] = []
    previous = -1
    for index in np.flatnonzero(accepted).tolist():
        separated = previous >= 0 and (
            index != previous + 1 or float(timestamps[index] - timestamps[previous]) > max_gap
        )
        if separated and current:
            rows = np.asarray(current, dtype=np.int64)
            duration = float(timestamps[rows[-1]] - timestamps[rows[0]] + nominal_dt)
            if duration >= float(min_seconds):
                segments.append(rows)
            current = []
        current.append(int(index))
        previous = int(index)
    if current:
        rows = np.asarray(current, dtype=np.int64)
        duration = float(timestamps[rows[-1]] - timestamps[rows[0]] + nominal_dt)
        if duration >= float(min_seconds):
            segments.append(rows)
    return segments


def bridge_short_gaps(
    timestamps_s: np.ndarray,
    valid: np.ndarray,
    *,
    max_gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark bounded short gaps as interpolable without removing their rows."""

    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    output = np.asarray(valid, dtype=bool).copy()
    interpolated = np.zeros(output.shape, dtype=bool)
    if float(max_gap_seconds) <= 0.0 or output.size < 3:
        return output, interpolated
    positive = np.diff(timestamps)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return output, interpolated
    nominal_dt = float(np.median(positive))
    index = 0
    while index < output.size:
        if output[index]:
            index += 1
            continue
        start = index
        while index < output.size and not output[index]:
            index += 1
        stop = index
        if start == 0 or stop >= output.size:
            continue
        missing_duration = float(stop - start) * nominal_dt
        bounding_gap = float(timestamps[stop] - timestamps[start - 1])
        if (
            missing_duration <= float(max_gap_seconds) + nominal_dt * 1e-3
            and bounding_gap <= float(max_gap_seconds) + 2.0 * nominal_dt
        ):
            output[start:stop] = True
            interpolated[start:stop] = True
    return output, interpolated


def _analysis_chunks(
    dataset: LocalCoordinateDataset,
    rows_mask: np.ndarray,
    eligible_pixels: np.ndarray,
    config: HeartrateConfig,
) -> list[np.ndarray]:
    eligible = np.asarray(eligible_pixels, dtype=bool)
    if not np.any(eligible):
        return []
    per_row_fraction = np.mean(np.asarray(dataset.pixel_valid, dtype=bool)[:, eligible], axis=1)
    valid = (
        np.asarray(rows_mask, dtype=bool)
        & np.asarray(dataset.frame_valid, dtype=bool)
        & (per_row_fraction >= float(config.min_chunk_valid_fraction))
    )
    within_partition = np.asarray(rows_mask, dtype=bool)
    bridged, _interpolated = bridge_short_gaps(
        dataset.timestamps_s,
        valid,
        max_gap_seconds=float(config.max_interpolated_gap_seconds),
    )
    valid = bridged & within_partition
    segments = contiguous_segments(
        dataset.timestamps_s,
        valid,
        max_gap_factor=float(config.max_timestamp_gap_factor),
        min_seconds=float(config.min_chunk_seconds),
    )
    chunks: list[np.ndarray] = []
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    target = float(config.discovery_chunk_seconds)
    minimum = float(config.min_chunk_seconds)
    for segment in segments:
        start = 0
        while start < segment.size:
            start_time = float(timestamps[segment[start]])
            stop = int(
                np.searchsorted(
                    timestamps[segment],
                    start_time + target,
                    side="left",
                )
            )
            stop = max(start + 1, min(stop, int(segment.size)))
            candidate = segment[start:stop]
            if candidate.size:
                duration = float(timestamps[candidate[-1]] - timestamps[candidate[0]])
                if duration + np.median(np.diff(timestamps)) >= minimum:
                    chunks.append(candidate)
            start = stop
    return chunks


def balanced_valid_partitions(
    dataset: LocalCoordinateDataset,
    risks: RiskSurfaces,
    config: HeartrateConfig,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Alternate usable contiguous blocks so both folds receive real coverage."""

    block_config = replace(
        config,
        discovery_chunk_seconds=float(config.partition_block_seconds),
    )
    blocks = _analysis_chunks(
        dataset,
        np.ones(dataset.frame_count, dtype=bool),
        risks.eligible,
        block_config,
    )
    minimum = int(config.min_partition_blocks_per_fold)
    if len(blocks) < 2 * minimum:
        raise ValueError(
            f"need at least {2 * minimum} usable blocks for two-fold confirmation; found {len(blocks)}"
        )
    masks = [np.zeros(dataset.frame_count, dtype=bool), np.zeros(dataset.frame_count, dtype=bool)]
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    half_guard = max(0.0, float(config.partition_guard_seconds)) / 2.0
    for block_index, rows in enumerate(blocks):
        kept = rows[
            (timestamps[rows] >= float(timestamps[rows[0]]) + half_guard)
            & (timestamps[rows] <= float(timestamps[rows[-1]]) - half_guard)
        ]
        if kept.size:
            masks[block_index % 2][kept] = True
    if any(int(np.count_nonzero(mask)) == 0 for mask in masks):
        raise ValueError("partition guard removed every row from one fold")
    return ((masks[0], masks[1]), (masks[1], masks[0]))


def _frequency_grid(config: HeartrateConfig) -> np.ndarray:
    count = int(
        math.floor(
            (float(config.band_max_hz) - float(config.band_min_hz))
            / float(config.frequency_step_hz)
        )
    )
    return float(config.band_min_hz) + np.arange(count + 1, dtype=np.float64) * float(
        config.frequency_step_hz
    )


def _chunk_frequency_coefficients(
    residual: np.ndarray,
    timestamps_s: np.ndarray,
    chunks: Sequence[np.ndarray],
    frequencies_hz: np.ndarray,
    *,
    min_valid_fraction: float,
    max_interpolated_gap_seconds: float,
) -> np.ndarray:
    traces = np.asarray(residual, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    coefficients = np.full(
        (len(frequencies_hz), len(chunks), traces.shape[1]),
        np.nan + 0j,
        dtype=np.complex128,
    )
    for chunk_index, rows in enumerate(chunks):
        y = traces[rows].astype(np.float64, copy=True)
        finite = np.isfinite(y)
        good = np.mean(finite, axis=0) >= float(min_valid_fraction)
        if not np.any(good):
            continue
        local_t = timestamps[rows] - float(timestamps[rows[0]])
        phase_t = timestamps[rows] - float(timestamps[0])
        selected_pixels = np.flatnonzero(good)
        selected = y[:, selected_pixels]
        usable = np.ones(selected_pixels.size, dtype=bool)
        for column in range(selected.shape[1]):
            finite_column = np.isfinite(selected[:, column])
            bridged, _interpolated = bridge_short_gaps(
                timestamps[rows],
                finite_column,
                max_gap_seconds=float(max_interpolated_gap_seconds),
            )
            if not np.all(bridged):
                usable[column] = False
                continue
            if not np.all(finite_column):
                selected[~finite_column, column] = np.interp(
                    local_t[~finite_column],
                    local_t[finite_column],
                    selected[finite_column, column],
                )
        if not np.any(usable):
            continue
        selected = selected[:, usable]
        selected_pixels = selected_pixels[usable]
        trend_design = np.column_stack([np.ones(local_t.size), local_t - np.mean(local_t)])
        trend_beta = np.linalg.pinv(trend_design) @ selected
        detrended = selected - trend_design @ trend_beta
        window = np.hanning(local_t.size)
        if float(np.sum(window)) <= np.finfo(float).eps:
            continue
        exponent = np.exp(-2j * np.pi * frequencies_hz[:, None] * phase_t[None, :])
        raw = (2.0 / float(np.sum(window))) * (exponent @ (detrended * window[:, None]))
        coefficients[:, chunk_index, selected_pixels] = raw
    return coefficients


def _empty_candidate(dataset: LocalCoordinateDataset, chunk_count: int = 0) -> DiscoveryCandidate:
    return DiscoveryCandidate(
        frequency_hz=math.nan,
        cluster_mass=0.0,
        pixel_indices=np.zeros(0, dtype=np.int64),
        pixel_weights=np.zeros(0, dtype=np.float64),
        pixel_scores=np.full(dataset.pixel_count, np.nan, dtype=np.float64),
        cluster_mask=np.zeros(dataset.image_shape_hw, dtype=bool),
        spatial_phase_coherence=0.0,
        chunk_phase_coherence=0.0,
        chunk_count=int(chunk_count),
    )


def _candidate_for_frequency(
    dataset: LocalCoordinateDataset,
    coefficients: np.ndarray,
    noise_floor: np.ndarray,
    frequency_hz: float,
    risks: RiskSurfaces,
    config: HeartrateConfig,
) -> DiscoveryCandidate:
    from scipy import ndimage

    eligible = np.asarray(risks.eligible, dtype=bool)
    if coefficients.ndim != 2 or coefficients.shape[1] != dataset.pixel_count:
        raise ValueError("coefficients must be chunk x pixel")
    chunk_count = int(coefficients.shape[0])
    floor = np.asarray(noise_floor, dtype=np.float64)
    scaled_coefficients = np.divide(
        coefficients,
        floor[None, :] + np.finfo(float).eps,
        out=np.full_like(coefficients, np.nan + 0j),
        where=np.isfinite(floor[None, :]),
    )
    absolute_amplitude = _quiet_nanmedian(np.abs(coefficients), axis=0)
    spectral_contrast = _quiet_nanmedian(np.abs(scaled_coefficients), axis=0)
    unit = np.divide(
        scaled_coefficients,
        np.abs(scaled_coefficients),
        out=np.full_like(scaled_coefficients, np.nan + 0j),
        where=np.abs(scaled_coefficients) > np.finfo(float).eps,
    )
    phase_stability = np.abs(_finite_mean(unit, axis=0))
    raw_score = (
        absolute_amplitude
        * np.sqrt(np.maximum(spectral_contrast, 0.0))
        * (0.25 + 0.75 * np.nan_to_num(phase_stability, nan=0.0))
    )
    score = np.full(dataset.pixel_count, np.nan, dtype=np.float64)
    if not np.any(eligible & np.isfinite(raw_score)):
        return _empty_candidate(dataset, chunk_count=chunk_count)
    score[eligible] = _robust_zscore(raw_score[eligible]) - risks.combined_penalty[eligible]
    selected = eligible & np.isfinite(score) & (score >= float(config.pixel_score_threshold_z))
    image = np.zeros(dataset.image_shape_hw, dtype=bool)
    xy = np.rint(np.asarray(dataset.pixel_xy, dtype=np.float64)).astype(np.int64)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < dataset.image_shape_hw[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < dataset.image_shape_hw[0])
    )
    selected_rows = np.flatnonzero(selected & inside)
    image[xy[selected_rows, 1], xy[selected_rows, 0]] = True
    labels, count = ndimage.label(image, structure=np.ones((3, 3), dtype=np.uint8))
    best = _empty_candidate(dataset, chunk_count=chunk_count)
    mean_coeff = _finite_mean(scaled_coefficients, axis=0)
    for label in range(1, int(count) + 1):
        component_image = labels == label
        component_at_pixel = component_image[
            xy[:, 1].clip(0, dataset.image_shape_hw[0] - 1),
            xy[:, 0].clip(0, dataset.image_shape_hw[1] - 1),
        ]
        component = np.flatnonzero(inside & component_at_pixel)
        if component.size < int(config.min_cluster_pixels):
            continue
        excess = np.maximum(score[component] - float(config.pixel_score_threshold_z), 0.0) + 0.05
        pixel_phase = mean_coeff[component]
        spatial_coherence = float(
            np.abs(np.nansum(excess * pixel_phase))
            / (np.nansum(excess * np.abs(pixel_phase)) + np.finfo(float).eps)
        )
        chunk_coeff = np.nansum(scaled_coefficients[:, component] * excess[None, :], axis=1)
        chunk_unit = np.divide(
            chunk_coeff,
            np.abs(chunk_coeff),
            out=np.zeros_like(chunk_coeff),
            where=np.abs(chunk_coeff) > np.finfo(float).eps,
        )
        chunk_coherence = float(np.abs(np.mean(chunk_unit))) if chunk_coeff.size else 0.0
        contrast_gain = float(
            np.nanmedian(np.log1p(np.maximum(spectral_contrast[component] - 1.0, 0.0)))
        )
        mass = float(
            np.sum(excess)
            * spatial_coherence
            * chunk_coherence
            * max(contrast_gain, np.finfo(float).eps)
        )
        if mass > best.cluster_mass:
            weights = excess / float(np.sum(excess))
            best = DiscoveryCandidate(
                frequency_hz=float(frequency_hz),
                cluster_mass=mass,
                pixel_indices=component.astype(np.int64),
                pixel_weights=weights.astype(np.float64),
                pixel_scores=score.copy(),
                cluster_mask=component_image,
                spatial_phase_coherence=spatial_coherence,
                chunk_phase_coherence=chunk_coherence,
                chunk_count=chunk_count,
            )
    return best


def discover_candidate(
    dataset: LocalCoordinateDataset,
    residual: np.ndarray,
    discovery_rows: np.ndarray,
    risks: RiskSurfaces,
    config: HeartrateConfig,
) -> DiscoveryCandidate:
    chunks = _analysis_chunks(dataset, discovery_rows, risks.eligible, config)
    if len(chunks) < int(config.min_partition_blocks_per_fold):
        return _empty_candidate(dataset, chunk_count=len(chunks))
    frequencies = _frequency_grid(config)
    coefficients = _chunk_frequency_coefficients(
        residual,
        dataset.timestamps_s,
        chunks,
        frequencies,
        min_valid_fraction=float(config.min_chunk_valid_fraction),
        max_interpolated_gap_seconds=float(config.max_interpolated_gap_seconds),
    )
    best = _empty_candidate(dataset, chunk_count=len(chunks))
    for frequency_index, frequency in enumerate(frequencies.tolist()):
        sideband = np.abs(frequencies - float(frequency)) >= max(
            2.0 * float(config.frequency_step_hz),
            0.2,
        )
        if not np.any(sideband):
            sideband = np.arange(frequencies.size) != int(frequency_index)
        noise_floor = _quiet_nanmedian(np.abs(coefficients[sideband]), axis=(0, 1))
        candidate = _candidate_for_frequency(
            dataset,
            coefficients[frequency_index],
            noise_floor,
            frequency,
            risks,
            config,
        )
        if candidate.cluster_mass > best.cluster_mass:
            best = candidate
    return best


def _spatial_block_ids(pixel_xy: np.ndarray, block_px: int) -> np.ndarray:
    xy = np.floor(np.asarray(pixel_xy, dtype=np.float64) / max(1, int(block_px))).astype(np.int64)
    unique: dict[tuple[int, int], int] = {}
    output = np.zeros(xy.shape[0], dtype=np.int64)
    for index, pair in enumerate(xy.tolist()):
        key = (int(pair[0]), int(pair[1]))
        if key not in unique:
            unique[key] = len(unique)
        output[index] = unique[key]
    return output


def autocorrelation_preserving_surrogate(
    dataset: LocalCoordinateDataset,
    active_rows: np.ndarray,
    *,
    rng: np.random.Generator,
    spatial_block_px: int,
    min_shift_seconds: float,
    max_gap_factor: float,
) -> LocalCoordinateDataset:
    traces = np.asarray(dataset.traces, dtype=np.float64).copy()
    segments = contiguous_segments(
        dataset.timestamps_s,
        np.asarray(active_rows, dtype=bool),
        max_gap_factor=float(max_gap_factor),
    )
    block_ids = _spatial_block_ids(dataset.pixel_xy, int(spatial_block_px))
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    for segment in segments:
        if segment.size < 4:
            continue
        dt = float(np.median(np.diff(timestamps[segment])))
        requested_minimum = max(1, int(math.ceil(float(min_shift_seconds) / dt)))
        # Short valid segments still need independent random surrogates. Restrict
        # shifts to the middle half when the requested temporal separation does
        # not fit, instead of deterministically rolling every segment by half.
        minimum = min(requested_minimum, max(1, int(segment.size // 4)))
        for block in np.unique(block_ids).tolist():
            pixels = np.flatnonzero(block_ids == int(block))
            shift = int(rng.integers(minimum, int(segment.size) - minimum + 1))
            traces[np.ix_(segment, pixels)] = np.roll(
                traces[np.ix_(segment, pixels)],
                shift=shift,
                axis=0,
            )
    surrogate_valid = np.asarray(dataset.pixel_valid, dtype=bool) & np.isfinite(traces)
    return replace(dataset, traces=traces, pixel_valid=surrogate_valid)


def calibrate_discovery(
    dataset: LocalCoordinateDataset,
    discovery_rows: np.ndarray,
    risks: RiskSurfaces,
    config: HeartrateConfig,
    *,
    seed: int,
) -> CalibratedDiscovery:
    model = fit_nuisance_model(dataset, discovery_rows, ridge=float(config.nuisance_ridge))
    residual = apply_nuisance_model(dataset, model)
    observed = discover_candidate(dataset, residual, discovery_rows, risks, config)
    rng = np.random.default_rng(int(seed))
    null = np.zeros(int(config.surrogate_count), dtype=np.float64)
    for index in range(int(config.surrogate_count)):
        surrogate = autocorrelation_preserving_surrogate(
            dataset,
            discovery_rows,
            rng=rng,
            spatial_block_px=int(config.surrogate_spatial_block_px),
            min_shift_seconds=float(config.surrogate_min_shift_seconds),
            max_gap_factor=float(config.max_timestamp_gap_factor),
        )
        surrogate_model = fit_nuisance_model(
            surrogate,
            discovery_rows,
            ridge=float(config.nuisance_ridge),
        )
        surrogate_residual = apply_nuisance_model(surrogate, surrogate_model)
        null[index] = discover_candidate(
            surrogate,
            surrogate_residual,
            discovery_rows,
            risks,
            config,
        ).cluster_mass
    p_value = (
        float(1 + np.count_nonzero(null >= observed.cluster_mass)) / float(null.size + 1)
        if null.size
        else 1.0
    )
    threshold = (
        float(np.quantile(null, 1.0 - float(config.alpha), method="higher"))
        if null.size
        else math.inf
    )
    detected = bool(
        observed.pixel_indices.size
        and null.size
        and p_value <= float(config.alpha)
        and observed.cluster_mass > threshold
    )
    return CalibratedDiscovery(
        candidate=observed,
        null_max_cluster_mass=null,
        p_value=p_value,
        threshold=threshold,
        detected=detected,
        nuisance_model=model,
        residual=residual,
    )


def _weighted_pixel_trace(
    residual: np.ndarray,
    pixel_indices: np.ndarray,
    pixel_weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float64)[:, np.asarray(pixel_indices, dtype=np.int64)]
    weights = np.asarray(pixel_weights, dtype=np.float64)
    if values.shape[1] != weights.size:
        raise ValueError("pixel weights do not match selected pixel count")
    finite = np.isfinite(values)
    effective = finite * weights[None, :]
    denominator = np.sum(effective, axis=1)
    output = np.full(values.shape[0], np.nan, dtype=np.float64)
    ok = denominator > np.finfo(float).eps
    output[ok] = np.nansum(values[ok] * weights[None, :], axis=1) / denominator[ok]
    return output


def _trace_chunks(
    dataset: LocalCoordinateDataset,
    trace: np.ndarray,
    rows_mask: np.ndarray,
    config: HeartrateConfig,
) -> list[np.ndarray]:
    valid = (
        np.asarray(rows_mask, dtype=bool)
        & np.asarray(dataset.frame_valid, dtype=bool)
        & np.isfinite(np.asarray(trace, dtype=np.float64))
    )
    valid, _interpolated = bridge_short_gaps(
        dataset.timestamps_s,
        valid,
        max_gap_seconds=float(config.max_interpolated_gap_seconds),
    )
    valid &= np.asarray(rows_mask, dtype=bool)
    segments = contiguous_segments(
        dataset.timestamps_s,
        valid,
        max_gap_factor=float(config.max_timestamp_gap_factor),
        min_seconds=float(config.min_chunk_seconds),
    )
    chunks: list[np.ndarray] = []
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    for segment in segments:
        start = 0
        while start < segment.size:
            t0 = float(timestamps[segment[start]])
            stop = int(np.searchsorted(timestamps[segment], t0 + float(config.discovery_chunk_seconds)))
            stop = max(start + 1, min(stop, int(segment.size)))
            candidate = segment[start:stop]
            dt = float(np.median(np.diff(timestamps)))
            duration = float(timestamps[candidate[-1]] - timestamps[candidate[0]] + dt)
            if duration >= float(config.min_chunk_seconds):
                chunks.append(candidate)
            start = stop
    return chunks


def trace_frequency_score(
    dataset: LocalCoordinateDataset,
    trace: np.ndarray,
    rows_mask: np.ndarray,
    frequency_hz: float,
    config: HeartrateConfig,
) -> float:
    chunks = _trace_chunks(dataset, trace, rows_mask, config)
    if not chunks:
        return 0.0
    coefficient = _chunk_frequency_coefficients(
        np.asarray(trace, dtype=np.float64)[:, None],
        dataset.timestamps_s,
        chunks,
        np.asarray([float(frequency_hz)], dtype=np.float64),
        min_valid_fraction=float(config.min_chunk_valid_fraction),
        max_interpolated_gap_seconds=float(config.max_interpolated_gap_seconds),
    )[0, :, 0]
    finite = coefficient[np.isfinite(coefficient)]
    if finite.size == 0:
        return 0.0
    unit = finite / np.maximum(np.abs(finite), np.finfo(float).eps)
    phase_coherence = float(np.abs(np.mean(unit)))
    return float(np.median(np.abs(finite)) * (0.25 + 0.75 * phase_coherence))


def _select_control_pixels(
    dataset: LocalCoordinateDataset,
    risks: RiskSurfaces,
    candidate: DiscoveryCandidate,
) -> dict[str, np.ndarray]:
    from scipy import ndimage

    count = int(candidate.pixel_indices.size)
    if count == 0:
        return {"interior": np.zeros(0, dtype=np.int64), "boundary": np.zeros(0, dtype=np.int64)}
    xy = np.rint(dataset.pixel_xy).astype(np.int64)
    dilated = ndimage.binary_dilation(candidate.cluster_mask, iterations=2)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < dataset.image_shape_hw[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < dataset.image_shape_hw[0])
    )
    near_cluster = np.zeros(dataset.pixel_count, dtype=bool)
    near_cluster[inside] = dilated[xy[inside, 1], xy[inside, 0]]
    target_penalty = float(np.median(risks.combined_penalty[candidate.pixel_indices]))
    interior_pool = np.flatnonzero(risks.eligible & ~near_cluster)
    interior_order = interior_pool[
        np.argsort(np.abs(risks.combined_penalty[interior_pool] - target_penalty))
    ]
    boundary_pool = np.flatnonzero(
        (risks.body_occupancy >= 0.5)
        & (risks.warp_invalid_fraction <= 0.25)
        & (
            risks.physical_boundary_distance_px
            < max(2.0, float(np.nanmedian(risks.physical_boundary_distance_px[candidate.pixel_indices])))
        )
        & ~near_cluster
    )
    centroid = np.mean(np.asarray(dataset.pixel_xy)[candidate.pixel_indices], axis=0)
    boundary_order = boundary_pool[
        np.argsort(np.linalg.norm(np.asarray(dataset.pixel_xy)[boundary_pool] - centroid[None, :], axis=1))
    ]
    return {
        "interior": interior_order[:count].astype(np.int64),
        "boundary": boundary_order[:count].astype(np.int64),
    }


def _uniform_weights(pixel_indices: np.ndarray) -> np.ndarray:
    count = int(np.asarray(pixel_indices).size)
    return np.full(count, 1.0 / count, dtype=np.float64) if count else np.zeros(0, dtype=np.float64)


def _named_control_scores(
    dataset: LocalCoordinateDataset,
    residual: np.ndarray,
    confirmation_rows: np.ndarray,
    candidate: DiscoveryCandidate,
    risks: RiskSurfaces,
    config: HeartrateConfig,
) -> dict[str, float]:
    controls = _select_control_pixels(dataset, risks, candidate)
    output: dict[str, float] = {}
    for label, pixels in controls.items():
        if pixels.size:
            trace = _weighted_pixel_trace(residual, pixels, _uniform_weights(pixels))
            output[label] = trace_frequency_score(
                dataset,
                trace,
                confirmation_rows,
                candidate.frequency_hz,
                config,
            )
        else:
            output[label] = math.nan
    for label in ("global_mean", "body_control_mean", "external_control_mean"):
        if label not in dataset.nuisance_names:
            output[label] = math.nan
            continue
        column = dataset.nuisance_names.index(label)
        output[label] = trace_frequency_score(
            dataset,
            np.asarray(dataset.nuisance_values)[:, column],
            confirmation_rows,
            candidate.frequency_hz,
            config,
        )
    return output


def _choose_polarity(
    dataset: LocalCoordinateDataset,
    trace: np.ndarray,
    rows_mask: np.ndarray,
    config: HeartrateConfig,
) -> str:
    configured = str(config.event_polarity)
    if configured != "auto":
        return configured
    values = np.asarray(trace, dtype=np.float64)[np.asarray(rows_mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size < 16:
        return "darkening"
    center = float(np.median(values))
    upper = float(np.quantile(values - center, 0.95))
    lower = abs(float(np.quantile(values - center, 0.05)))
    return "brightening" if upper > lower else "darkening"


def extract_segmented_events(
    dataset: LocalCoordinateDataset,
    trace: np.ndarray,
    rows_mask: np.ndarray,
    *,
    polarity: str,
    config: HeartrateConfig,
) -> EventSeries:
    from scipy import signal

    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    values = np.asarray(trace, dtype=np.float64)
    valid = (
        np.asarray(rows_mask, dtype=bool)
        & np.asarray(dataset.frame_valid, dtype=bool)
        & np.isfinite(values)
    )
    valid, _interpolated = bridge_short_gaps(
        timestamps,
        valid,
        max_gap_seconds=float(config.max_interpolated_gap_seconds),
    )
    valid &= np.asarray(rows_mask, dtype=bool)
    minimum_duration = max(
        float(config.min_chunk_seconds),
        2.0 * float(config.event_filter_edge_seconds) + 3.0 / float(config.band_max_hz),
    )
    segments = contiguous_segments(
        timestamps,
        valid,
        max_gap_factor=float(config.max_timestamp_gap_factor),
        min_seconds=minimum_duration,
    )
    event_frames: list[int] = []
    event_times: list[float] = []
    event_values: list[float] = []
    event_prominences: list[float] = []
    intervals: list[float] = []
    bpm: list[float] = []
    analyzed: list[tuple[float, float]] = []
    rejected_edge = 0
    sign = -1.0 if str(polarity) == "darkening" else 1.0
    for rows in segments:
        local_t = timestamps[rows]
        dt = float(np.median(np.diff(local_t)))
        grid = np.arange(local_t[0], local_t[-1] + 0.5 * dt, dt, dtype=np.float64)
        local_values = values[rows]
        finite_local = np.isfinite(local_values)
        if int(np.count_nonzero(finite_local)) < 3:
            continue
        interpolated_local = np.interp(
            local_t,
            local_t[finite_local],
            local_values[finite_local],
        )
        sampled = np.interp(grid, local_t, interpolated_local)
        sampled = signal.detrend(sampled, type="linear")
        nyquist = 0.5 / dt
        high = min(float(config.band_max_hz), nyquist * 0.98)
        if not (0.0 < float(config.band_min_hz) < high):
            continue
        sos = signal.butter(
            3,
            [float(config.band_min_hz), high],
            btype="bandpass",
            fs=1.0 / dt,
            output="sos",
        )
        filtered = sign * signal.sosfiltfilt(sos, sampled)
        _center, robust_scale = _robust_center_scale(filtered)
        prominence = float(config.event_prominence_mad) * float(robust_scale)
        minimum_distance = max(1, int(math.floor((1.0 / float(config.band_max_hz)) / dt * 0.8)))
        peaks, properties = signal.find_peaks(
            filtered,
            distance=minimum_distance,
            prominence=prominence,
        )
        edge = float(config.event_filter_edge_seconds)
        keep = (grid[peaks] >= grid[0] + edge) & (grid[peaks] <= grid[-1] - edge)
        rejected_edge += int(np.count_nonzero(~keep))
        peaks = peaks[keep]
        prominences = np.asarray(properties.get("prominences", []), dtype=np.float64)[keep]
        analyzed_start = float(grid[0] + edge)
        analyzed_stop = float(grid[-1] - edge)
        if analyzed_stop > analyzed_start:
            analyzed.append((analyzed_start, analyzed_stop))
        segment_times = grid[peaks]
        segment_frames = np.asarray(
            [
                int(dataset.frame_indices[int(np.argmin(np.abs(timestamps - event_time)))])
                for event_time in segment_times.tolist()
            ],
            dtype=np.int64,
        )
        event_frames.extend(segment_frames.tolist())
        event_times.extend(segment_times.tolist())
        event_values.extend(filtered[peaks].tolist())
        event_prominences.extend(prominences.tolist())
        segment_intervals = np.diff(segment_times)
        intervals.extend(segment_intervals.tolist())
        bpm.extend((60.0 / segment_intervals).tolist())
    order = (
        np.argsort(np.asarray(event_times, dtype=np.float64))
        if event_times
        else np.zeros(0, dtype=np.int64)
    )
    return EventSeries(
        frame_indices=np.asarray(event_frames, dtype=np.int64)[order],
        timestamps_s=np.asarray(event_times, dtype=np.float64)[order],
        filtered_values=np.asarray(event_values, dtype=np.float64)[order],
        prominences=np.asarray(event_prominences, dtype=np.float64)[order],
        intervals_s=np.asarray(intervals, dtype=np.float64),
        instantaneous_bpm=np.asarray(bpm, dtype=np.float64),
        rejected_edge_events=int(rejected_edge),
        analyzed_intervals_s=tuple(analyzed),
    )


def _confirm_candidate(
    dataset: LocalCoordinateDataset,
    confirmation_rows: np.ndarray,
    discovery_rows: np.ndarray,
    discovery: CalibratedDiscovery,
    risks: RiskSurfaces,
    config: HeartrateConfig,
    *,
    fold_index: int,
    seed: int,
) -> FoldResult:
    candidate = discovery.candidate
    if not discovery.detected:
        return FoldResult(
            fold_index=int(fold_index),
            discovery=discovery,
            confirmation_p_value=1.0,
            confirmation_score=0.0,
            confirmation_null_scores=np.zeros(0, dtype=np.float64),
            confirmation_chunk_scores=np.zeros(0, dtype=np.float64),
            confirmation_chunk_p_values=np.zeros(0, dtype=np.float64),
            confirmed_chunk_count=0,
            confirmation_chunk_count=0,
            control_scores={},
            control_ratio=0.0,
            confirmed=False,
            polarity=str(config.event_polarity if config.event_polarity != "auto" else "darkening"),
            confirmation_trace=np.full(dataset.frame_count, np.nan, dtype=np.float64),
            confirmation_valid=np.zeros(dataset.frame_count, dtype=bool),
            event_valid=np.zeros(dataset.frame_count, dtype=bool),
            events=None,
        )
    residual = apply_nuisance_model(dataset, discovery.nuisance_model)
    trace = _weighted_pixel_trace(residual, candidate.pixel_indices, candidate.pixel_weights)
    confirmation_valid = np.asarray(confirmation_rows, dtype=bool) & np.isfinite(trace)
    observed = trace_frequency_score(
        dataset,
        trace,
        confirmation_rows,
        candidate.frequency_hz,
        config,
    )
    confirmation_chunks = _trace_chunks(dataset, trace, confirmation_rows, config)
    chunk_scores = np.asarray(
        [
            trace_frequency_score(
                dataset,
                trace,
                np.isin(np.arange(dataset.frame_count), chunk),
                candidate.frequency_hz,
                config,
            )
            for chunk in confirmation_chunks
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    null = np.zeros(int(config.surrogate_count), dtype=np.float64)
    chunk_null = np.zeros((int(config.surrogate_count), len(confirmation_chunks)), dtype=np.float64)
    for index in range(int(config.surrogate_count)):
        surrogate = autocorrelation_preserving_surrogate(
            dataset,
            confirmation_rows,
            rng=rng,
            spatial_block_px=int(config.surrogate_spatial_block_px),
            min_shift_seconds=float(config.surrogate_min_shift_seconds),
            max_gap_factor=float(config.max_timestamp_gap_factor),
        )
        surrogate_residual = apply_nuisance_model(surrogate, discovery.nuisance_model)
        surrogate_trace = _weighted_pixel_trace(
            surrogate_residual,
            candidate.pixel_indices,
            candidate.pixel_weights,
        )
        null[index] = trace_frequency_score(
            surrogate,
            surrogate_trace,
            confirmation_rows,
            candidate.frequency_hz,
            config,
        )
        for chunk_index, chunk in enumerate(confirmation_chunks):
            chunk_mask = np.zeros(dataset.frame_count, dtype=bool)
            chunk_mask[chunk] = True
            chunk_null[index, chunk_index] = trace_frequency_score(
                surrogate,
                surrogate_trace,
                chunk_mask,
                candidate.frequency_hz,
                config,
            )
    p_value = (
        float(1 + np.count_nonzero(null >= observed)) / float(null.size + 1)
        if null.size
        else 1.0
    )
    controls = _named_control_scores(
        dataset,
        residual,
        confirmation_rows,
        candidate,
        risks,
        config,
    )
    finite_controls = np.asarray(
        [value for value in controls.values() if np.isfinite(value)],
        dtype=np.float64,
    )
    strongest_control = float(np.max(finite_controls)) if finite_controls.size else 0.0
    control_ratio = float(observed / max(strongest_control, np.finfo(float).eps))
    confirmed = bool(
        null.size
        and p_value <= float(config.alpha)
        and observed > float(np.quantile(null, 1.0 - float(config.alpha), method="higher"))
        and control_ratio >= float(config.min_control_ratio)
    )
    chunk_null_max = (
        np.max(chunk_null, axis=1)
        if chunk_null.shape[0] and chunk_null.shape[1]
        else np.zeros(0, dtype=np.float64)
    )
    chunk_p_values = np.asarray(
        [
            float(1 + np.count_nonzero(chunk_null_max >= chunk_scores[index]))
            / float(chunk_null_max.size + 1)
            if chunk_null_max.size
            else 1.0
            for index in range(len(confirmation_chunks))
        ],
        dtype=np.float64,
    )
    chunk_threshold = (
        float(
            np.quantile(
                chunk_null_max,
                1.0 - float(config.alpha),
                method="higher",
            )
        )
        if chunk_null_max.size
        else math.inf
    )
    confirmed_chunks = (
        (chunk_p_values <= float(config.alpha))
        & (chunk_scores > chunk_threshold)
        if len(confirmation_chunks)
        else np.zeros(0, dtype=bool)
    )
    event_rows = np.zeros(dataset.frame_count, dtype=bool)
    if confirmed:
        for accepted, chunk in zip(confirmed_chunks.tolist(), confirmation_chunks):
            if accepted:
                event_rows[chunk] = True
    discovery_trace = _weighted_pixel_trace(
        discovery.residual,
        candidate.pixel_indices,
        candidate.pixel_weights,
    )
    polarity = _choose_polarity(dataset, discovery_trace, discovery_rows, config)
    events = (
        extract_segmented_events(
            dataset,
            trace,
            event_rows,
            polarity=polarity,
            config=config,
        )
        if confirmed and np.any(event_rows)
        else None
    )
    return FoldResult(
        fold_index=int(fold_index),
        discovery=discovery,
        confirmation_p_value=p_value,
        confirmation_score=float(observed),
        confirmation_null_scores=null,
        confirmation_chunk_scores=chunk_scores,
        confirmation_chunk_p_values=chunk_p_values,
        confirmed_chunk_count=int(np.count_nonzero(confirmed_chunks)),
        confirmation_chunk_count=int(len(confirmation_chunks)),
        control_scores=controls,
        control_ratio=control_ratio,
        confirmed=confirmed,
        polarity=polarity,
        confirmation_trace=trace,
        confirmation_valid=confirmation_valid,
        event_valid=event_rows,
        events=events,
    )


def _dilated_overlap(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import ndimage

    first = np.asarray(a, dtype=bool)
    second = np.asarray(b, dtype=bool)
    if not np.any(first) or not np.any(second):
        return 0.0
    first_dilated = ndimage.binary_dilation(first, iterations=1)
    second_dilated = ndimage.binary_dilation(second, iterations=1)
    first_supported = np.count_nonzero(first & second_dilated) / float(np.count_nonzero(first))
    second_supported = np.count_nonzero(second & first_dilated) / float(np.count_nonzero(second))
    return float(min(first_supported, second_supported))


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(stop)) for start, stop in intervals if stop > start)
    merged: list[tuple[float, float]] = []
    for start, stop in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, stop))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
    return merged


def _interval_complement(
    start: float,
    stop: float,
    accepted: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    output: list[tuple[float, float]] = []
    cursor = float(start)
    for left, right in _merge_intervals(accepted):
        left = max(float(start), left)
        right = min(float(stop), right)
        if left > cursor:
            output.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < float(stop):
        output.append((cursor, float(stop)))
    return tuple(output)


def analyze_heartrate(
    dataset: LocalCoordinateDataset,
    config: HeartrateConfig,
) -> HeartrateResult:
    dataset.validated()
    config.validated()
    nominal_dt = float(np.median(np.diff(np.asarray(dataset.timestamps_s, dtype=np.float64))))
    if float(config.band_max_hz) >= 0.5 / nominal_dt:
        full_interval = ((float(dataset.timestamps_s[0]), float(dataset.timestamps_s[-1])),)
        return HeartrateResult(
            detected=False,
            reason="target_band_exceeds_timestamp_nyquist",
            folds=(),
            crossfit_dilated_overlap=0.0,
            crossfit_frequency_difference_hz=math.nan,
            event_frame_indices=np.zeros(0, dtype=np.int64),
            event_timestamps_s=np.zeros(0, dtype=np.float64),
            event_intervals_s=np.zeros(0, dtype=np.float64),
            instantaneous_bpm=np.zeros(0, dtype=np.float64),
            coverage_fraction=0.0,
            no_estimate_intervals_s=full_interval,
        )
    risks = build_risk_surfaces(dataset, config)
    if int(np.count_nonzero(risks.eligible)) < int(config.min_cluster_pixels):
        full_interval = ((float(dataset.timestamps_s[0]), float(dataset.timestamps_s[-1])),)
        return HeartrateResult(
            detected=False,
            reason="too_few_physically_eligible_pixels",
            folds=(),
            crossfit_dilated_overlap=0.0,
            crossfit_frequency_difference_hz=math.nan,
            event_frame_indices=np.zeros(0, dtype=np.int64),
            event_timestamps_s=np.zeros(0, dtype=np.float64),
            event_intervals_s=np.zeros(0, dtype=np.float64),
            instantaneous_bpm=np.zeros(0, dtype=np.float64),
            coverage_fraction=0.0,
            no_estimate_intervals_s=full_interval,
        )
    try:
        partitions = balanced_valid_partitions(dataset, risks, config)
    except ValueError as exc:
        full_interval = ((float(dataset.timestamps_s[0]), float(dataset.timestamps_s[-1])),)
        return HeartrateResult(
            detected=False,
            reason=f"insufficient_disjoint_valid_blocks:{exc}",
            folds=(),
            crossfit_dilated_overlap=0.0,
            crossfit_frequency_difference_hz=math.nan,
            event_frame_indices=np.zeros(0, dtype=np.int64),
            event_timestamps_s=np.zeros(0, dtype=np.float64),
            event_intervals_s=np.zeros(0, dtype=np.float64),
            instantaneous_bpm=np.zeros(0, dtype=np.float64),
            coverage_fraction=0.0,
            no_estimate_intervals_s=full_interval,
        )
    folds: list[FoldResult] = []
    for fold_index, (discovery_rows, confirmation_rows) in enumerate(partitions):
        discovery = calibrate_discovery(
            dataset,
            discovery_rows,
            risks,
            config,
            seed=int(config.random_seed) + 1009 * fold_index,
        )
        folds.append(
            _confirm_candidate(
                dataset,
                confirmation_rows,
                discovery_rows,
                discovery,
                risks,
                config,
                fold_index=fold_index,
                seed=int(config.random_seed) + 2003 * fold_index + 17,
            )
        )
    overlap = _dilated_overlap(
        folds[0].discovery.candidate.cluster_mask,
        folds[1].discovery.candidate.cluster_mask,
    )
    fold_frequencies = np.asarray(
        [fold.discovery.candidate.frequency_hz for fold in folds],
        dtype=np.float64,
    )
    frequency_difference = (
        float(abs(fold_frequencies[0] - fold_frequencies[1]))
        if np.isfinite(fold_frequencies).all()
        else math.inf
    )
    detected = bool(
        all(fold.confirmed for fold in folds)
        and overlap >= float(config.min_crossfit_dilated_overlap)
        and frequency_difference <= float(config.max_crossfit_frequency_difference_hz)
    )
    if not all(fold.discovery.detected for fold in folds):
        reason = "discovery_not_significant_in_both_folds"
    elif not all(fold.confirmed for fold in folds):
        reason = "held_out_confirmation_failed"
    elif (
        overlap < float(config.min_crossfit_dilated_overlap)
        and frequency_difference > float(config.max_crossfit_frequency_difference_hz)
    ):
        reason = "crossfit_frequency_and_cluster_not_reproducible"
    elif frequency_difference > float(config.max_crossfit_frequency_difference_hz):
        reason = "crossfit_frequency_not_reproducible"
    elif overlap < float(config.min_crossfit_dilated_overlap):
        reason = "crossfit_cluster_not_reproducible"
    else:
        reason = "confirmed"
    event_frames: list[int] = []
    event_times: list[float] = []
    event_intervals: list[float] = []
    event_bpm: list[float] = []
    analyzed_intervals: list[tuple[float, float]] = []
    if detected:
        for fold in folds:
            if fold.events is None:
                continue
            event_frames.extend(fold.events.frame_indices.tolist())
            event_times.extend(fold.events.timestamps_s.tolist())
            event_intervals.extend(fold.events.intervals_s.tolist())
            event_bpm.extend(fold.events.instantaneous_bpm.tolist())
            analyzed_intervals.extend(fold.events.analyzed_intervals_s)
    order = (
        np.argsort(np.asarray(event_times, dtype=np.float64))
        if event_times
        else np.zeros(0, dtype=np.int64)
    )
    merged = _merge_intervals(analyzed_intervals)
    recording_start = float(dataset.timestamps_s[0])
    recording_stop = float(dataset.timestamps_s[-1])
    covered = float(sum(stop - start for start, stop in merged))
    duration = max(recording_stop - recording_start, np.finfo(float).eps)
    no_estimate = _interval_complement(recording_start, recording_stop, merged if detected else ())
    return HeartrateResult(
        detected=detected,
        reason=reason,
        folds=tuple(folds),
        crossfit_dilated_overlap=overlap,
        crossfit_frequency_difference_hz=frequency_difference,
        event_frame_indices=np.asarray(event_frames, dtype=np.int64)[order],
        event_timestamps_s=np.asarray(event_times, dtype=np.float64)[order],
        event_intervals_s=np.asarray(event_intervals, dtype=np.float64),
        instantaneous_bpm=np.asarray(event_bpm, dtype=np.float64),
        coverage_fraction=float(covered / duration) if detected else 0.0,
        no_estimate_intervals_s=no_estimate,
    )


def make_real_noise_null(
    dataset: LocalCoordinateDataset,
    config: HeartrateConfig,
    *,
    seed: int,
) -> LocalCoordinateDataset:
    """Destroy cross-pixel periodic coherence while retaining real trace autocorrelation."""

    return autocorrelation_preserving_surrogate(
        dataset,
        np.asarray(dataset.frame_valid, dtype=bool),
        rng=np.random.default_rng(int(seed)),
        spatial_block_px=int(config.surrogate_spatial_block_px),
        min_shift_seconds=float(config.surrogate_min_shift_seconds),
        max_gap_factor=float(config.max_timestamp_gap_factor),
    )


def inject_local_signal(
    dataset: LocalCoordinateDataset,
    spec: InjectionSpec,
) -> tuple[LocalCoordinateDataset, np.ndarray]:
    """Inject on the source raster, then re-sample through the saved bilinear map."""

    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    relative = timestamps - timestamps[0]
    duration = max(float(relative[-1]), np.finfo(float).eps)
    phase = 2.0 * np.pi * (
        float(spec.frequency_hz) * relative
        + 0.5 * float(spec.phase_drift_hz_per_s) * relative**2
    ) + float(spec.phase_rad)
    waveform = np.sin(phase)
    active = relative <= duration * float(np.clip(spec.active_fraction, 0.0, 1.0))
    waveform = waveform * active.astype(np.float64)
    xy = np.asarray(dataset.pixel_xy, dtype=np.float64)
    center = np.asarray(spec.center_xy, dtype=np.float64)
    radius = max(float(spec.radius_px), 0.25)
    spatial = np.exp(-0.5 * np.sum((xy - center[None, :]) ** 2, axis=1) / (radius**2))
    source_shape = tuple(
        int(value)
        for value in dataset.metadata.get("source_image_shape_hw", dataset.image_shape_hw)
    )
    source_height, source_width = source_shape
    source_effect = np.full((dataset.frame_count, dataset.pixel_count), np.nan, dtype=np.float64)
    coordinates = np.asarray(dataset.source_xy, dtype=np.float64)
    interpolation_weights = np.asarray(dataset.bilinear_weights, dtype=np.float64)
    pixel_valid = np.asarray(dataset.pixel_valid, dtype=bool)
    for frame in range(dataset.frame_count):
        valid_pixels = np.flatnonzero(
            pixel_valid[frame]
            & np.isfinite(coordinates[frame]).all(axis=1)
            & np.isfinite(interpolation_weights[frame]).all(axis=1)
        )
        if valid_pixels.size == 0:
            continue
        points = coordinates[frame, valid_pixels]
        x0 = np.floor(points[:, 0]).astype(np.int64)
        y0 = np.floor(points[:, 1]).astype(np.int64)
        neighbor_indices = np.column_stack(
            [
                y0 * source_width + x0,
                y0 * source_width + (x0 + 1),
                (y0 + 1) * source_width + x0,
                (y0 + 1) * source_width + (x0 + 1),
            ]
        )
        inside = (
            (x0 >= 0)
            & (y0 >= 0)
            & (x0 + 1 < source_width)
            & (y0 + 1 < source_height)
        )
        if not np.any(inside):
            continue
        valid_pixels = valid_pixels[inside]
        neighbor_indices = neighbor_indices[inside]
        local_weights = interpolation_weights[frame, valid_pixels]
        flat_indices = neighbor_indices.reshape(-1)
        flat_weights = local_weights.reshape(-1)
        numerator = np.bincount(
            flat_indices,
            weights=(local_weights * spatial[valid_pixels, None]).reshape(-1),
            minlength=source_height * source_width,
        )
        denominator = np.bincount(
            flat_indices,
            weights=flat_weights,
            minlength=source_height * source_width,
        )
        source_raster = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > np.finfo(float).eps,
        )
        source_effect[frame, valid_pixels] = np.sum(
            local_weights * source_raster[neighbor_indices],
            axis=1,
        )
    traces = np.asarray(dataset.traces, dtype=np.float64)
    if traces.shape[0] >= 3:
        _center, per_pixel_scale = _robust_center_scale(np.diff(traces, axis=0), axis=0)
        per_pixel_scale = per_pixel_scale / math.sqrt(2.0)
    else:
        _center, per_pixel_scale = _robust_center_scale(traces, axis=0)
    noise_scale = float(np.nanmedian(per_pixel_scale[np.isfinite(per_pixel_scale)]))
    injected = traces + (
        float(spec.amplitude_sigma) * noise_scale * waveform[:, None] * source_effect
    )
    injected[~pixel_valid] = np.nan
    expected_peaks, _ = __import__("scipy.signal", fromlist=["find_peaks"]).find_peaks(
        -waveform,
        distance=max(1, int(0.8 / float(spec.frequency_hz) / np.median(np.diff(timestamps)))),
    )
    expected_peaks = expected_peaks[active[expected_peaks]]
    metadata = dict(dataset.metadata)
    metadata["injection"] = {
        "surface": "source_crop_raster_resampled_through_saved_bilinear_weights",
        "noise_scale": "median per-pixel robust first-difference scale divided by sqrt(2)",
        "noise_scale_intensity": noise_scale,
        "amplitude_intensity": float(spec.amplitude_sigma) * noise_scale,
    }
    return replace(dataset, traces=injected, metadata=metadata), timestamps[expected_peaks]


def _match_event_times(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    tolerance_s: float,
) -> tuple[np.ndarray, float, float]:
    observed_values = np.asarray(observed, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    candidates: list[tuple[float, int, int]] = []
    for expected_index, expected_time in enumerate(expected_values.tolist()):
        distances = np.abs(observed_values - float(expected_time))
        for observed_index in np.flatnonzero(distances <= float(tolerance_s)).tolist():
            candidates.append((float(distances[observed_index]), expected_index, int(observed_index)))
    matched_expected: set[int] = set()
    matched_observed: set[int] = set()
    errors: list[float] = []
    for distance, expected_index, observed_index in sorted(candidates):
        if expected_index in matched_expected or observed_index in matched_observed:
            continue
        matched_expected.add(expected_index)
        matched_observed.add(observed_index)
        errors.append(float(distance))
    precision = float(len(errors) / observed_values.size) if observed_values.size else 0.0
    recall = float(len(errors) / expected_values.size) if expected_values.size else 0.0
    return np.asarray(errors, dtype=np.float64), precision, recall


def _times_outside_intervals(
    timestamps_s: np.ndarray,
    excluded: Sequence[tuple[float, float]],
) -> np.ndarray:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    keep = np.ones(timestamps.shape, dtype=bool)
    for start, stop in excluded:
        keep &= ~((timestamps >= float(start)) & (timestamps <= float(stop)))
    return timestamps[keep]


def run_injection_recovery(
    dataset: LocalCoordinateDataset,
    config: HeartrateConfig,
    specs: Sequence[InjectionSpec],
    *,
    replicates: int = 1,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run the complete adaptive pipeline on injected real-noise null backgrounds."""

    rows: list[dict[str, Any]] = []
    for replicate in range(max(1, int(replicates))):
        background = make_real_noise_null(dataset, config, seed=int(seed) + 7919 * replicate)
        for case_index, spec in enumerate(specs):
            injected, expected_events = inject_local_signal(background, spec)
            case_config = replace(config, random_seed=int(seed) + replicate * 10007 + case_index * 101)
            result = analyze_heartrate(injected, case_config)
            frequencies = np.asarray(
                [fold.discovery.candidate.frequency_hz for fold in result.folds],
                dtype=np.float64,
            )
            finite_frequency = frequencies[np.isfinite(frequencies)]
            estimated_frequency = float(np.median(finite_frequency)) if finite_frequency.size else math.nan
            centroids: list[np.ndarray] = []
            for fold in result.folds:
                pixels = fold.discovery.candidate.pixel_indices
                if pixels.size:
                    centroids.append(np.mean(np.asarray(dataset.pixel_xy)[pixels], axis=0))
            centroid = np.mean(np.asarray(centroids), axis=0) if centroids else np.full(2, np.nan)
            localization_error = float(
                np.linalg.norm(centroid - np.asarray(spec.center_xy, dtype=np.float64))
            ) if np.isfinite(centroid).all() else math.nan
            evaluated_expected = _times_outside_intervals(
                expected_events,
                result.no_estimate_intervals_s,
            ) if result.detected else np.zeros(0, dtype=np.float64)
            match_errors, event_precision, event_recall = _match_event_times(
                result.event_timestamps_s,
                evaluated_expected,
                tolerance_s=0.35 / float(spec.frequency_hz),
            )
            rows.append(
                {
                    "replicate": int(replicate),
                    "case_index": int(case_index),
                    "amplitude_sigma": float(spec.amplitude_sigma),
                    "frequency_hz": float(spec.frequency_hz),
                    "phase_drift_hz_per_s": float(spec.phase_drift_hz_per_s),
                    "radius_px": float(spec.radius_px),
                    "center_x": float(spec.center_xy[0]),
                    "center_y": float(spec.center_xy[1]),
                    "active_fraction": float(spec.active_fraction),
                    "noise_scale_intensity": float(
                        injected.metadata["injection"]["noise_scale_intensity"]
                    ),
                    "amplitude_intensity": float(
                        injected.metadata["injection"]["amplitude_intensity"]
                    ),
                    "detected": bool(result.detected),
                    "reason": result.reason,
                    "crossfit_frequency_difference_hz": float(
                        result.crossfit_frequency_difference_hz
                    ),
                    "estimated_frequency_hz": estimated_frequency,
                    "frequency_bias_hz": float(estimated_frequency - spec.frequency_hz)
                    if np.isfinite(estimated_frequency)
                    else math.nan,
                    "localization_error_px": localization_error,
                    "expected_event_count_total": int(expected_events.size),
                    "expected_event_count_evaluated": int(evaluated_expected.size),
                    "recovered_event_count": int(result.event_timestamps_s.size),
                    "matched_event_count": int(match_errors.size),
                    "event_precision": event_precision,
                    "event_recall": event_recall,
                    "event_timing_rmse_s": float(np.sqrt(np.mean(match_errors**2)))
                    if match_errors.size
                    else math.nan,
                    "coverage_fraction": float(result.coverage_fraction),
                }
            )
    return rows


def injection_operating_characteristics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"case_count": 0, "false_positive_rate": math.nan, "sensitivity_by_amplitude": {}}
    null = [row for row in rows if float(row["amplitude_sigma"]) == 0.0]
    positive = [row for row in rows if float(row["amplitude_sigma"]) > 0.0]
    detected_positive = [row for row in positive if bool(row["detected"])]
    amplitudes = sorted({float(row["amplitude_sigma"]) for row in positive})
    sensitivity = {
        str(amplitude): float(
            np.mean([bool(row["detected"]) for row in positive if float(row["amplitude_sigma"]) == amplitude])
        )
        for amplitude in amplitudes
    }
    detectable = [amplitude for amplitude in amplitudes if sensitivity[str(amplitude)] >= 0.8]
    amplitude_case_counts = {
        str(amplitude): int(
            sum(float(row["amplitude_sigma"]) == amplitude for row in positive)
        )
        for amplitude in amplitudes
    }

    def finite_median(field: str, selected: Sequence[Mapping[str, Any]]) -> float:
        values = np.asarray([float(row[field]) for row in selected], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else math.nan

    return {
        "case_count": int(len(rows)),
        "null_case_count": int(len(null)),
        "positive_case_count": int(len(positive)),
        "positive_detected_count": int(sum(bool(row["detected"]) for row in positive)),
        "positive_detection_rate": float(np.mean([bool(row["detected"]) for row in positive]))
        if positive
        else math.nan,
        "false_positive_rate": float(np.mean([bool(row["detected"]) for row in null])) if null else math.nan,
        "sensitivity_by_amplitude": sensitivity,
        "case_count_by_amplitude": amplitude_case_counts,
        "minimum_tested_amplitude_sigma_at_80pct_detection": min(detectable) if detectable else None,
        "confirmed_frequency_bias_hz_median": finite_median(
            "frequency_bias_hz", detected_positive
        ),
        "confirmed_localization_error_px_median": finite_median(
            "localization_error_px", detected_positive
        ),
        "confirmed_event_timing_rmse_s_median": finite_median(
            "event_timing_rmse_s", detected_positive
        ),
        "confirmed_event_precision_median": finite_median(
            "event_precision", detected_positive
        ),
        "confirmed_event_recall_median": finite_median(
            "event_recall", detected_positive
        ),
        "confirmed_coverage_fraction_median": finite_median(
            "coverage_fraction", detected_positive
        ),
    }
