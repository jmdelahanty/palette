from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import scipy

from analyze_frozen_heart_masks_longitudinal import _read_mask, _window_dataset
import compare_heart_photometry_transforms as comparison
from diagnose_frozen_mask_longitudinal_tracking import _mask_at_pixels
from extract_reliable_local_rostral_heartrate import load_dataset
import fisheye.analysis.heart_photometry_injection as injection_analysis
import fisheye.analysis.heart_photometry_projection as projection_analysis
import fisheye.analysis.heart_photometry_transforms as transform_analysis
import fisheye.analysis.local_rostral_heartrate as local_analysis
from fisheye.analysis.heart_photometry_injection import (
    PhotometryInjectionSpec,
    circular_difference_rad,
    estimate_monotonic_phase_slope,
    inject_mono8_photometry,
    phase_equivalent_timing_error_ms,
    select_discovery_family,
    stable_spec_id,
)
from fisheye.analysis.heart_photometry_transforms import (
    normalized_signed_lag_difference,
    regional_pool,
    regional_spatial_std,
    segmented_savgol_derivative,
)
from fisheye.analysis.local_rostral_heartrate import alternating_block_partitions


_CANDIDATE_NAMES = (
    "regional_spatial_std",
    "crossfit_matched_spatial_projection",
    "huber_savgol_derivative_w11",
    "huber_normalized_signed_lag12",
    "huber_normalized_signed_lag16",
)
_PHASE_TIMING_CANDIDATES = {
    "huber_savgol_derivative_w11",
    "huber_normalized_signed_lag12",
    "huber_normalized_signed_lag16",
}
_TRANSFORM_BUILD_LOCK = threading.Lock()
_EXPERIMENT_SCHEMA_VERSION = 4
_JOB_NPZ_SCHEMA = "cached_photometry_recoverability_job"
_JOB_NPZ_SCHEMA_VERSION = 2
_CONSOLIDATED_NPZ_SCHEMA = "cached_photometry_recoverability_consolidated"
_CONSOLIDATED_NPZ_SCHEMA_VERSION = 2


def _runtime_version_identity() -> dict[str, str]:
    return {
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
    }


def _implementation_code_identity(
    paths: Mapping[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    """Hash every local implementation file that can change reported metrics."""

    playground = Path(__file__).resolve().parent
    resolved_paths = dict(paths) if paths is not None else {
        "injection_runner": Path(__file__).resolve(),
        "injection_analysis": Path(injection_analysis.__file__).resolve(),
        "photometry_transforms": Path(transform_analysis.__file__).resolve(),
        "matched_projection": Path(projection_analysis.__file__).resolve(),
        "local_rostral_analysis": Path(local_analysis.__file__).resolve(),
        "transform_comparison_helpers": Path(comparison.__file__).resolve(),
        "longitudinal_window_helpers": playground
        / "analyze_frozen_heart_masks_longitudinal.py",
        "tracking_mask_helpers": playground
        / "diagnose_frozen_mask_longitudinal_tracking.py",
        "dataset_loader": playground / "extract_reliable_local_rostral_heartrate.py",
    }
    identity: dict[str, dict[str, Any]] = {}
    for name, path in sorted(resolved_paths.items()):
        resolved = Path(path).resolve()
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        identity[str(name)] = {
            "path": str(resolved),
            "size_bytes": int(resolved.stat().st_size),
            "sha256": digest.hexdigest(),
        }
    return identity


@dataclass(frozen=True)
class InjectionJob:
    pattern: str
    frequency_hz: float
    amplitude_dn: float
    activity_mode: str
    replicate: int
    initial_phase_rad: float
    traveling_band_count: int
    traveling_direction: int

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def job_id(self) -> str:
        return stable_spec_id(self.payload())


@dataclass(frozen=True)
class GridMeasurement:
    spectral_ratio: np.ndarray
    control_ratio: np.ndarray
    phase_offset_rad: np.ndarray
    phase_locking_value: np.ndarray
    phase_valid_block_count: np.ndarray
    phase_total_block_count: np.ndarray
    phase_valid_block_fraction: np.ndarray
    phase_support_qualified_block_count: np.ndarray
    phase_support_qualified_block_fraction: np.ndarray
    upper_phase_support_ratio_median: np.ndarray
    lower_phase_support_ratio_median: np.ndarray
    upper_phase_coefficient_amplitude_median: np.ndarray
    lower_phase_coefficient_amplitude_median: np.ndarray
    target_truth_residual_rad: np.ndarray
    target_truth_phase_coherence: np.ndarray
    target_truth_valid_block_count: np.ndarray
    target_truth_total_block_count: np.ndarray
    target_truth_valid_block_fraction: np.ndarray
    target_truth_support_qualified_block_count: np.ndarray
    target_truth_support_qualified_block_fraction: np.ndarray
    target_phase_support_ratio_median: np.ndarray
    target_phase_coefficient_amplitude_median: np.ndarray
    block_count: int


def _parse_csv_values(value: str, cast: Any) -> list[Any]:
    parsed = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed:
        raise ValueError("comma-separated option cannot be empty")
    return parsed


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
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _finite_median(values: np.ndarray | Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else math.nan


def _validate_timebase(
    timestamps_s: np.ndarray,
    frame_indices: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    maximum_relative_jitter: float,
) -> dict[str, Any]:
    """Validate the actual cached timestamps and tested-frequency Nyquist support."""

    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 3:
        raise ValueError("timebase needs at least three timestamps")
    if frames.shape != timestamps.shape:
        raise ValueError("frame_indices must match timestamps_s")
    if frequencies.ndim != 1 or not frequencies.size:
        raise ValueError("frequencies_hz must be a nonempty vector")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    if np.any(np.diff(frames) <= 0):
        raise ValueError("frame_indices must be strictly increasing")
    if not np.isfinite(frequencies).all() or np.any(frequencies <= 0.0):
        raise ValueError("tested frequencies must be finite and positive")
    allowed_jitter = float(maximum_relative_jitter)
    if not 0.0 <= allowed_jitter < 1.0:
        raise ValueError("maximum_relative_jitter must be in [0, 1)")

    spacing = np.diff(timestamps)
    frame_steps = np.diff(frames)
    median_spacing = float(np.median(spacing))
    spacing_mad = float(np.median(np.abs(spacing - median_spacing)))
    relative_mad = float(1.4826 * spacing_mad / median_spacing)
    spacing_p01 = float(np.quantile(spacing, 0.01))
    spacing_p99 = float(np.quantile(spacing, 0.99))
    relative_quantile_deviation = float(
        max(abs(spacing_p01 - median_spacing), abs(spacing_p99 - median_spacing))
        / median_spacing
    )
    relative_jitter = max(relative_mad, relative_quantile_deviation)
    if relative_jitter > allowed_jitter:
        raise ValueError(
            "timestamp spacing relative jitter "
            f"{relative_jitter:.6g} exceeds predeclared limit {allowed_jitter:.6g}"
        )
    sample_rate_hz = float(1.0 / median_spacing)
    nyquist_hz = float(0.5 * sample_rate_hz)
    maximum_tested = float(np.max(frequencies))
    if maximum_tested >= nyquist_hz * (1.0 - 1e-12):
        raise ValueError(
            f"maximum tested frequency {maximum_tested:g} Hz is not below "
            f"timestamp-derived Nyquist {nyquist_hz:g} Hz"
        )
    return {
        "timestamp_count": int(timestamps.size),
        "timestamp_start_s": float(timestamps[0]),
        "timestamp_stop_s": float(timestamps[-1]),
        "spacing_median_s": median_spacing,
        "spacing_min_s": float(np.min(spacing)),
        "spacing_max_s": float(np.max(spacing)),
        "spacing_p01_s": spacing_p01,
        "spacing_p99_s": spacing_p99,
        "spacing_relative_mad": relative_mad,
        "spacing_relative_quantile_deviation": relative_quantile_deviation,
        "spacing_relative_jitter_statistic": relative_jitter,
        "maximum_allowed_spacing_relative_jitter": allowed_jitter,
        "timestamp_derived_sample_rate_hz": sample_rate_hz,
        "timestamp_derived_nyquist_hz": nyquist_hz,
        "nyquist_basis": "one_half_over_median_actual_timestamp_spacing",
        "tested_frequency_min_hz": float(np.min(frequencies)),
        "tested_frequency_max_hz": maximum_tested,
        "tested_frequency_count": int(frequencies.size),
        "tested_grid_below_nyquist": True,
        "frame_step_median": float(np.median(frame_steps)),
        "frame_step_max": int(np.max(frame_steps)),
        "frame_gap_count": int(np.count_nonzero(frame_steps > 1)),
    }


def _circular_mean(values: np.ndarray | Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return math.nan, math.nan
    vector = np.mean(np.exp(1j * array))
    return float(np.angle(vector)), float(abs(vector))


def _frequency_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    if not 0.0 < float(minimum) < float(maximum) or float(step) <= 0.0:
        raise ValueError("invalid frequency grid")
    count = int(round((float(maximum) - float(minimum)) / float(step))) + 1
    grid = float(minimum) + np.arange(count, dtype=np.float64) * float(step)
    if abs(float(grid[-1] - maximum)) > 1e-8:
        raise ValueError("frequency range must be an integer number of steps")
    sideband_counts = np.asarray(
        [np.count_nonzero(np.abs(grid - frequency) >= 0.2 - 1e-12) for frequency in grid]
    )
    if np.any(sideband_counts < 1):
        raise ValueError("frequency grid is too narrow to define 0.2 Hz sidebands")
    return grid


def _jobs(args: argparse.Namespace) -> list[InjectionJob]:
    patterns = _parse_csv_values(args.patterns, str)
    unknown = set(patterns) - {
        "synchronous",
        "opposite_upper_lower",
        "fixed_upper_lower_delay",
        "traveling_wave",
    }
    if unknown:
        raise ValueError(f"unsupported patterns: {sorted(unknown)}")
    frequencies = _parse_csv_values(args.injection_frequencies_hz, float)
    amplitudes = _parse_csv_values(args.amplitudes_dn, float)
    activities = _parse_csv_values(args.activity_modes, str)
    band_counts = _parse_csv_values(args.traveling_band_counts, int)
    directions = _parse_csv_values(args.traveling_directions, int)
    jobs: list[InjectionJob] = []
    for pattern in patterns:
        pattern_band_counts = band_counts if pattern == "traveling_wave" else [5]
        pattern_directions = directions if pattern == "traveling_wave" else [1]
        for frequency in frequencies:
            for amplitude in amplitudes:
                for activity in activities:
                    for replicate in range(int(args.replicates)):
                        for band_count in pattern_band_counts:
                            for direction in pattern_directions:
                                seed_payload = {
                                    "seed": int(args.seed),
                                    "pattern": pattern,
                                    "frequency_hz": frequency,
                                    "amplitude_dn": amplitude,
                                    "activity_mode": activity,
                                    "replicate": replicate,
                                    "band_count": band_count,
                                    "direction": direction,
                                }
                                seed = int(stable_spec_id(seed_payload, length=16), 16)
                                phase = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))
                                jobs.append(
                                    InjectionJob(
                                        pattern=pattern,
                                        frequency_hz=float(frequency),
                                        amplitude_dn=float(amplitude),
                                        activity_mode=activity,
                                        replicate=replicate,
                                        initial_phase_rad=phase,
                                        traveling_band_count=int(band_count),
                                        traveling_direction=int(direction),
                                    )
                                )
    # Amplitude-zero activity, phase, pattern, direction, and nominal truth
    # frequency and replicate are all observationally identical. Keep one
    # canonical sanity control instead of reporting duplicated zero-DN trials.
    canonical: dict[str, InjectionJob] = {}
    for job in jobs:
        if float(job.amplitude_dn) == 0.0:
            job = InjectionJob(
                pattern="synchronous",
                frequency_hz=float(frequencies[0]),
                amplitude_dn=0.0,
                activity_mode="continuous",
                replicate=0,
                initial_phase_rad=0.0,
                traveling_band_count=5,
                traveling_direction=1,
            )
        canonical[job.job_id] = job
    jobs = sorted(canonical.values(), key=lambda job: job.job_id)
    batch_count = int(args.batch_count)
    batch_index = int(args.batch_index)
    if batch_count < 1 or not 0 <= batch_index < batch_count:
        raise ValueError("batch-index must be in [0, batch-count)")
    return [job for index, job in enumerate(jobs) if index % batch_count == batch_index]


def _read_windows(
    path: Path,
    mask_name: str,
    selected: set[int] | None,
    maximum: int | None,
) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        windows = [
            row
            for row in csv.DictReader(handle)
            if row["mask"] == str(mask_name) and row["status"] == "ok"
        ]
    windows.sort(key=lambda row: int(row["window_index"]))
    if selected is not None:
        windows = [row for row in windows if int(row["window_index"]) in selected]
    if maximum is not None:
        windows = windows[: int(maximum)]
    if not windows:
        raise ValueError("no scorable windows remain")
    return windows


def _source_window_dataset(
    dataset: Any,
    source: Mapping[str, str],
    *,
    frame_count: int | None,
) -> Any:
    frame_indices = np.asarray(dataset.frame_indices, dtype=np.int64)
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
    if frame_count is not None:
        stop = min(stop, start + int(frame_count))
    if stop - start < 32:
        raise ValueError("window contains too few requested frames")
    return _window_dataset(dataset, start, stop)


def _valid_matrix(dataset: Any) -> np.ndarray:
    return (
        np.asarray(dataset.pixel_valid, dtype=bool)
        & np.asarray(dataset.frame_valid, dtype=bool)[:, None]
        & np.isfinite(np.asarray(dataset.traces, dtype=np.float64))
    )


def _base_candidates(
    dataset: Any,
    *,
    target: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    reference: np.ndarray,
    control: np.ndarray,
    partitions: Sequence[tuple[np.ndarray, np.ndarray]],
    nuisance_ridge: float,
) -> dict[str, comparison.TraceSet]:
    family = comparison._candidate_traces(
        dataset,
        target=target,
        upper=upper,
        lower=lower,
        reference=reference,
        control=control,
        sg_windows=(11,),
        lag_frames=(12, 16),
        gaussian_sigma_px=0.8,
    )
    selected: dict[str, comparison.TraceSet] = {}
    for name in _CANDIDATE_NAMES:
        if name == "crossfit_matched_spatial_projection":
            continue
        selected[name] = comparison._crossfit_trace_set(
            family[name],
            dataset,
            partitions,
            ridge=float(nuisance_ridge),
        )
    return selected


def _ratio_grid(curves: np.ndarray, frequencies_hz: np.ndarray) -> np.ndarray:
    amplitudes = np.abs(np.asarray(curves, dtype=np.complex128))
    output = np.full(amplitudes.shape, np.nan, dtype=np.float64)
    for frequency_index, frequency in enumerate(frequencies_hz):
        sideband = np.abs(frequencies_hz - frequency) >= 0.2 - 1e-12
        sideband_amplitudes = amplitudes[:, sideband]
        noise = np.full(amplitudes.shape[0], np.nan, dtype=np.float64)
        for row, values in enumerate(sideband_amplitudes):
            finite = values[np.isfinite(values)]
            if finite.size:
                noise[row] = float(np.median(finite))
        output[:, frequency_index] = np.divide(
            amplitudes[:, frequency_index],
            noise,
            out=np.full(amplitudes.shape[0], np.nan, dtype=np.float64),
            where=np.isfinite(noise) & (noise > np.finfo(float).eps),
        )
    return output


def _measure_grid(
    dataset: Any,
    traces: comparison.TraceSet,
    frequencies_hz: np.ndarray,
    *,
    block_seconds: float,
    min_block_seconds: float,
    min_valid_fraction: float,
    max_interpolated_gap_seconds: float,
    truth_frequency_hz: float,
    truth_target_phase_rad: float,
    truth_time_origin_s: float,
    activity_envelope: np.ndarray,
    truth_amplitude_dn: float,
    minimum_phase_support_ratio: float,
    minimum_phase_coefficient_amplitude: float,
) -> GridMeasurement:
    if float(minimum_phase_support_ratio) < 1.0:
        raise ValueError("minimum_phase_support_ratio must be at least one")
    if float(minimum_phase_coefficient_amplitude) <= 0.0:
        raise ValueError("minimum_phase_coefficient_amplitude must be positive")
    amplitude_floor = float(minimum_phase_coefficient_amplitude)
    support_floor = float(minimum_phase_support_ratio)
    blocks = comparison._logical_blocks(
        dataset,
        traces,
        block_seconds=float(block_seconds),
        min_block_seconds=float(min_block_seconds),
        min_valid_fraction=float(min_valid_fraction),
        max_interpolated_gap_seconds=float(max_interpolated_gap_seconds),
    )
    frequency_count = frequencies_hz.size
    if not blocks:
        empty = np.full(frequency_count, np.nan, dtype=np.float64)
        zero = np.zeros(frequency_count, dtype=np.int32)
        return GridMeasurement(
            spectral_ratio=empty,
            control_ratio=empty.copy(),
            phase_offset_rad=empty.copy(),
            phase_locking_value=empty.copy(),
            phase_valid_block_count=zero,
            phase_total_block_count=zero.copy(),
            phase_valid_block_fraction=empty.copy(),
            phase_support_qualified_block_count=zero.copy(),
            phase_support_qualified_block_fraction=empty.copy(),
            upper_phase_support_ratio_median=empty.copy(),
            lower_phase_support_ratio_median=empty.copy(),
            upper_phase_coefficient_amplitude_median=empty.copy(),
            lower_phase_coefficient_amplitude_median=empty.copy(),
            target_truth_residual_rad=empty.copy(),
            target_truth_phase_coherence=empty.copy(),
            target_truth_valid_block_count=zero.copy(),
            target_truth_total_block_count=zero.copy(),
            target_truth_valid_block_fraction=empty.copy(),
            target_truth_support_qualified_block_count=zero.copy(),
            target_truth_support_qualified_block_fraction=empty.copy(),
            target_phase_support_ratio_median=empty.copy(),
            target_phase_coefficient_amplitude_median=empty.copy(),
            block_count=0,
        )
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    target_curves: list[np.ndarray] = []
    control_curves: list[np.ndarray] = []
    upper_curves: list[np.ndarray] = []
    lower_curves: list[np.ndarray] = []
    truth_residuals: list[np.ndarray] = []
    truth_active: list[bool] = []
    for rows in blocks:
        target_curve = comparison._complex_coefficients(
            timestamps[rows], traces.target[rows], frequencies_hz
        )
        target_curves.append(target_curve)
        control_curves.append(
            comparison._complex_coefficients(
                timestamps[rows], traces.control[rows], frequencies_hz
            )
        )
        if np.all(np.isfinite(traces.upper[rows])) and np.all(
            np.isfinite(traces.lower[rows])
        ):
            upper_curve = comparison._complex_coefficients(
                timestamps[rows], traces.upper[rows], frequencies_hz
            )
            lower_curve = comparison._complex_coefficients(
                timestamps[rows], traces.lower[rows], frequencies_hz
            )
            upper_curves.append(upper_curve)
            lower_curves.append(lower_curve)
        else:
            upper_curves.append(
                np.full(frequency_count, np.nan + 0j, dtype=np.complex128)
            )
            lower_curves.append(
                np.full(frequency_count, np.nan + 0j, dtype=np.complex128)
            )
        residual = np.full(frequency_count, np.nan, dtype=np.float64)
        truth_index = int(np.argmin(np.abs(frequencies_hz - float(truth_frequency_hz))))
        target_block_support = _ratio_grid(
            target_curve[None, :], frequencies_hz
        )[0, truth_index]
        if (
            float(truth_amplitude_dn) > 0.0
            and np.isfinite(truth_target_phase_rad)
            and abs(float(frequencies_hz[truth_index] - truth_frequency_hz))
            <= 0.51 * float(np.min(np.diff(frequencies_hz)))
            and np.isfinite(target_curve[truth_index])
            and abs(target_curve[truth_index]) >= amplitude_floor
            and np.isfinite(target_block_support)
            and target_block_support >= support_floor
        ):
            expected_phase = (
                2.0
                * np.pi
                * float(truth_frequency_hz)
                * (float(timestamps[rows[0]]) - float(truth_time_origin_s))
                + float(truth_target_phase_rad)
                - np.pi / 2.0
            )
            residual[truth_index] = float(
                circular_difference_rad(np.angle(target_curve[truth_index]), expected_phase)
            )
        truth_residuals.append(residual)
        truth_active.append(float(np.mean(activity_envelope[rows])) >= 0.5)

    target_array = np.stack(target_curves)
    control_array = np.stack(control_curves)
    upper_array = np.stack(upper_curves)
    lower_array = np.stack(lower_curves)
    target_ratios = _ratio_grid(target_array, frequencies_hz)
    control_ratios = _ratio_grid(control_array, frequencies_hz)
    upper_support = _ratio_grid(upper_array, frequencies_hz)
    lower_support = _ratio_grid(lower_array, frequencies_hz)
    spectral = np.asarray(
        [_finite_median(target_ratios[:, index]) for index in range(frequency_count)]
    )
    control_spectral = np.asarray(
        [_finite_median(control_ratios[:, index]) for index in range(frequency_count)]
    )
    target_control = np.divide(
        spectral,
        control_spectral,
        out=np.full(spectral.shape, np.nan, dtype=np.float64),
        where=np.isfinite(control_spectral) & (control_spectral > 0.0),
    )
    upper_amplitude = np.abs(upper_array)
    lower_amplitude = np.abs(lower_array)
    phase_support_qualified = (
        np.isfinite(upper_support)
        & np.isfinite(lower_support)
        & (upper_support >= support_floor)
        & (lower_support >= support_floor)
        & np.isfinite(upper_amplitude)
        & np.isfinite(lower_amplitude)
        & (upper_amplitude >= amplitude_floor)
        & (lower_amplitude >= amplitude_floor)
    )
    phase_array = np.full(phase_support_qualified.shape, np.nan, dtype=np.float64)
    phase_products = lower_array * np.conjugate(upper_array)
    phase_array[phase_support_qualified] = np.angle(
        phase_products[phase_support_qualified]
    )
    phase_mean = np.full(frequency_count, np.nan, dtype=np.float64)
    phase_locking = np.full(frequency_count, np.nan, dtype=np.float64)
    phase_valid_count = np.sum(np.isfinite(phase_array), axis=0).astype(np.int32)
    phase_support_count = np.sum(phase_support_qualified, axis=0).astype(np.int32)
    phase_total_count = np.full(frequency_count, len(blocks), dtype=np.int32)
    phase_valid_fraction = phase_valid_count.astype(np.float64) / float(len(blocks))
    phase_support_fraction = phase_support_count.astype(np.float64) / float(len(blocks))
    for frequency_index in range(frequency_count):
        phase_mean[frequency_index], phase_locking[frequency_index] = _circular_mean(
            phase_array[:, frequency_index]
        )
    residual_array = np.stack(truth_residuals)
    active = np.asarray(truth_active, dtype=bool)
    target_amplitude = np.abs(target_array)
    target_support_qualified = (
        active[:, None]
        & np.isfinite(target_ratios)
        & (target_ratios >= support_floor)
        & np.isfinite(target_amplitude)
        & (target_amplitude >= amplitude_floor)
    )
    residual_array[~target_support_qualified] = np.nan
    residual_mean = np.full(frequency_count, np.nan, dtype=np.float64)
    residual_coherence = np.full(frequency_count, np.nan, dtype=np.float64)
    residual_valid_count = np.sum(
        np.isfinite(residual_array) & active[:, None], axis=0
    ).astype(np.int32)
    active_count = int(np.count_nonzero(active))
    residual_support_count = np.sum(target_support_qualified, axis=0).astype(np.int32)
    residual_total_count = np.full(frequency_count, active_count, dtype=np.int32)
    residual_valid_fraction = np.divide(
        residual_valid_count,
        residual_total_count,
        out=np.full(frequency_count, np.nan, dtype=np.float64),
        where=residual_total_count > 0,
    )
    residual_support_fraction = np.divide(
        residual_support_count,
        residual_total_count,
        out=np.full(frequency_count, np.nan, dtype=np.float64),
        where=residual_total_count > 0,
    )
    for frequency_index in range(frequency_count):
        residual_mean[frequency_index], residual_coherence[frequency_index] = _circular_mean(
            residual_array[active, frequency_index]
        )
    upper_support_median = np.asarray(
        [_finite_median(upper_support[:, index]) for index in range(frequency_count)]
    )
    lower_support_median = np.asarray(
        [_finite_median(lower_support[:, index]) for index in range(frequency_count)]
    )
    upper_amplitude_median = np.asarray(
        [_finite_median(upper_amplitude[:, index]) for index in range(frequency_count)]
    )
    lower_amplitude_median = np.asarray(
        [_finite_median(lower_amplitude[:, index]) for index in range(frequency_count)]
    )
    target_support_median = np.asarray(
        [
            _finite_median(target_ratios[active, index])
            for index in range(frequency_count)
        ]
    )
    target_amplitude_median = np.asarray(
        [
            _finite_median(target_amplitude[active, index])
            for index in range(frequency_count)
        ]
    )
    return GridMeasurement(
        spectral_ratio=spectral,
        control_ratio=target_control,
        phase_offset_rad=phase_mean,
        phase_locking_value=phase_locking,
        phase_valid_block_count=phase_valid_count,
        phase_total_block_count=phase_total_count,
        phase_valid_block_fraction=phase_valid_fraction,
        phase_support_qualified_block_count=phase_support_count,
        phase_support_qualified_block_fraction=phase_support_fraction,
        upper_phase_support_ratio_median=upper_support_median,
        lower_phase_support_ratio_median=lower_support_median,
        upper_phase_coefficient_amplitude_median=upper_amplitude_median,
        lower_phase_coefficient_amplitude_median=lower_amplitude_median,
        target_truth_residual_rad=residual_mean,
        target_truth_phase_coherence=residual_coherence,
        target_truth_valid_block_count=residual_valid_count,
        target_truth_total_block_count=residual_total_count,
        target_truth_valid_block_fraction=residual_valid_fraction,
        target_truth_support_qualified_block_count=residual_support_count,
        target_truth_support_qualified_block_fraction=residual_support_fraction,
        target_phase_support_ratio_median=target_support_median,
        target_phase_coefficient_amplitude_median=target_amplitude_median,
        block_count=len(blocks),
    )


def _band_candidates(
    dataset: Any,
    band_masks: Sequence[np.ndarray],
    partitions: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    nuisance_ridge: float,
) -> dict[str, list[np.ndarray]]:
    values = np.asarray(dataset.traces, dtype=np.float64)
    valid = _valid_matrix(dataset)
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    output = {
        name: []
        for name in _CANDIDATE_NAMES
        if name != "crossfit_matched_spatial_projection"
    }
    for mask in band_masks:
        huber = regional_pool(
            values,
            mask,
            valid=valid,
            method="huber",
            min_valid_pixels=min(2, int(np.count_nonzero(mask))),
        )
        raw = {
            "regional_spatial_std": regional_spatial_std(
                values,
                mask,
                valid=valid,
                min_valid_pixels=min(2, int(np.count_nonzero(mask))),
            ),
            "huber_savgol_derivative_w11": segmented_savgol_derivative(
                huber,
                timestamps,
                valid=np.isfinite(huber),
                window_length=11,
                polyorder=2,
                max_gap_factor=1.75,
            ),
            "huber_normalized_signed_lag12": normalized_signed_lag_difference(
                huber,
                timestamps,
                lag_frames=12,
                valid=np.isfinite(huber),
                max_gap_factor=1.75,
                alignment="center",
            ),
            "huber_normalized_signed_lag16": normalized_signed_lag_difference(
                huber,
                timestamps,
                lag_frames=16,
                valid=np.isfinite(huber),
                max_gap_factor=1.75,
                alignment="center",
            ),
        }
        for name, trace in raw.items():
            output[name].append(
                comparison._crossfit_nuisance_residual(
                    trace,
                    dataset,
                    partitions,
                    ridge=float(nuisance_ridge),
                )
            )
    return output


def _band_slope_grid(
    dataset: Any,
    traces_by_band: Sequence[np.ndarray],
    frequencies_hz: np.ndarray,
    *,
    block_seconds: float,
    min_block_seconds: float,
    min_valid_fraction: float,
    max_interpolated_gap_seconds: float,
    minimum_phase_support_ratio: float,
    minimum_phase_coefficient_amplitude: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if float(minimum_phase_support_ratio) < 1.0:
        raise ValueError("minimum_phase_support_ratio must be at least one")
    if float(minimum_phase_coefficient_amplitude) <= 0.0:
        raise ValueError("minimum_phase_coefficient_amplitude must be positive")
    unavailable = np.full(dataset.frame_count, np.nan, dtype=np.float64)
    common = np.ones(dataset.frame_count, dtype=bool)
    for trace in traces_by_band:
        common &= np.isfinite(trace)
    dummy = comparison.TraceSet(
        target=np.where(common, traces_by_band[0], np.nan),
        upper=unavailable,
        lower=unavailable.copy(),
        control=np.where(common, traces_by_band[-1], np.nan),
    )
    blocks = comparison._logical_blocks(
        dataset,
        dummy,
        block_seconds=float(block_seconds),
        min_block_seconds=float(min_block_seconds),
        min_valid_fraction=float(min_valid_fraction),
        max_interpolated_gap_seconds=float(max_interpolated_gap_seconds),
    )
    slopes = np.full((len(blocks), frequencies_hz.size), np.nan, dtype=np.float64)
    band_count = len(traces_by_band)
    support = np.full(
        (len(blocks), band_count, frequencies_hz.size), np.nan, dtype=np.float64
    )
    amplitude = np.full_like(support, np.nan)
    support_qualified = np.zeros(
        (len(blocks), frequencies_hz.size), dtype=bool
    )
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    for block_index, rows in enumerate(blocks):
        coefficients = np.stack(
            [
                comparison._complex_coefficients(
                    timestamps[rows], trace[rows], frequencies_hz
                )
                for trace in traces_by_band
            ]
        )
        block_support = _ratio_grid(coefficients, frequencies_hz)
        block_amplitude = np.abs(coefficients)
        support[block_index] = block_support
        amplitude[block_index] = block_amplitude
        support_qualified[block_index] = np.all(
            np.isfinite(block_support)
            & (block_support >= float(minimum_phase_support_ratio))
            & np.isfinite(block_amplitude)
            & (
                block_amplitude
                >= float(minimum_phase_coefficient_amplitude)
            ),
            axis=0,
        )
        for frequency_index in range(frequencies_hz.size):
            if not support_qualified[block_index, frequency_index]:
                continue
            slopes[block_index, frequency_index] = estimate_monotonic_phase_slope(
                np.angle(coefficients[:, frequency_index]),
                weights=np.abs(coefficients[:, frequency_index]),
            )
    mean_slope = np.full(frequencies_hz.size, np.nan, dtype=np.float64)
    slope_coherence = np.full(frequencies_hz.size, np.nan, dtype=np.float64)
    valid_count = np.sum(np.isfinite(slopes), axis=0).astype(np.int32)
    support_qualified_count = np.sum(support_qualified, axis=0).astype(np.int32)
    total_count = np.full(frequencies_hz.size, len(blocks), dtype=np.int32)
    valid_fraction = np.divide(
        valid_count,
        total_count,
        out=np.full(frequencies_hz.size, np.nan, dtype=np.float64),
        where=total_count > 0,
    )
    support_qualified_fraction = np.divide(
        support_qualified_count,
        total_count,
        out=np.full(frequencies_hz.size, np.nan, dtype=np.float64),
        where=total_count > 0,
    )
    band_support_median = np.full(
        (frequencies_hz.size, band_count), np.nan, dtype=np.float64
    )
    band_amplitude_median = np.full_like(band_support_median, np.nan)
    for frequency_index in range(frequencies_hz.size):
        for band_index in range(band_count):
            band_support_median[frequency_index, band_index] = _finite_median(
                support[:, band_index, frequency_index]
            )
            band_amplitude_median[frequency_index, band_index] = _finite_median(
                amplitude[:, band_index, frequency_index]
            )
    direction_consistency = np.full(frequencies_hz.size, np.nan, dtype=np.float64)
    for index in range(frequencies_hz.size):
        finite = slopes[:, index][np.isfinite(slopes[:, index])]
        mean_slope[index], slope_coherence[index] = _circular_mean(finite)
        if finite.size and np.isfinite(mean_slope[index]) and mean_slope[index] != 0.0:
            direction_consistency[index] = float(
                np.mean(np.sign(finite) == np.sign(mean_slope[index]))
            )
    return (
        mean_slope,
        direction_consistency,
        slope_coherence,
        valid_count,
        total_count,
        valid_fraction,
        support_qualified_count,
        support_qualified_fraction,
        band_support_median,
        band_amplitude_median,
    )


def _window_metrics(
    dataset: Any,
    source: Mapping[str, str],
    job: InjectionJob,
    frequencies_hz: np.ndarray,
    masks: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    *,
    time_origin_s: float,
) -> dict[str, np.ndarray]:
    local = _source_window_dataset(dataset, source, frame_count=args.frame_count)
    spec = PhotometryInjectionSpec(
        pattern=job.pattern,
        frequency_hz=float(job.frequency_hz),
        amplitude_dn=float(job.amplitude_dn),
        activity_mode=job.activity_mode,
        initial_phase_rad=float(job.initial_phase_rad),
        upper_lower_delay_rad=float(args.upper_lower_delay_deg) * np.pi / 180.0,
        traveling_band_count=int(job.traveling_band_count),
        traveling_phase_span_rad=float(args.traveling_phase_span_deg) * np.pi / 180.0,
        traveling_direction=int(job.traveling_direction),
        intermittent_period_seconds=float(args.intermittent_period_seconds),
        intermittent_active_fraction=float(args.intermittent_active_fraction),
        intermittent_ramp_seconds=float(args.intermittent_ramp_seconds),
    ).validated()
    injected, truth = inject_mono8_photometry(
        local,
        masks["target"],
        spec,
        upper_mask=masks["upper"],
        lower_mask=masks["lower"],
        band_axis="y",
        time_origin_s=float(time_origin_s),
    )
    partitions = alternating_block_partitions(
        injected.timestamps_s,
        block_seconds=float(args.block_seconds),
        guard_seconds=float(args.guard_seconds),
    )
    # The transform helpers use warnings.catch_warnings, whose filter state is
    # process-global. Serialize only this inexpensive construction step so
    # optional worker threads cannot race while changing warning filters.
    with _TRANSFORM_BUILD_LOCK:
        candidates = _base_candidates(
            injected,
            target=masks["target"],
            upper=masks["upper"],
            lower=masks["lower"],
            reference=masks["reference"],
            control=masks["control"],
            partitions=partitions,
            nuisance_ridge=float(args.nuisance_ridge),
        )
    candidate_count = len(_CANDIDATE_NAMES)
    frequency_count = frequencies_hz.size
    shape = (candidate_count, frequency_count)
    result = {
        "spectral_ratio": np.full(shape, np.nan, dtype=np.float64),
        "control_ratio": np.full(shape, np.nan, dtype=np.float64),
        "phase_offset_rad": np.full(shape, np.nan, dtype=np.float64),
        "phase_locking_value": np.full(shape, np.nan, dtype=np.float64),
        "phase_valid_block_count": np.zeros(shape, dtype=np.int32),
        "phase_total_block_count": np.zeros(shape, dtype=np.int32),
        "phase_valid_block_fraction": np.full(shape, np.nan, dtype=np.float64),
        "phase_support_qualified_block_count": np.zeros(shape, dtype=np.int32),
        "phase_support_qualified_block_fraction": np.full(
            shape, np.nan, dtype=np.float64
        ),
        "upper_phase_support_ratio_median": np.full(
            shape, np.nan, dtype=np.float64
        ),
        "lower_phase_support_ratio_median": np.full(
            shape, np.nan, dtype=np.float64
        ),
        "upper_phase_coefficient_amplitude_median": np.full(
            shape, np.nan, dtype=np.float64
        ),
        "lower_phase_coefficient_amplitude_median": np.full(
            shape, np.nan, dtype=np.float64
        ),
        "target_truth_residual_rad": np.full(shape, np.nan, dtype=np.float64),
        "target_truth_phase_coherence": np.full(shape, np.nan, dtype=np.float64),
        "target_truth_valid_block_count": np.zeros(shape, dtype=np.int32),
        "target_truth_total_block_count": np.zeros(shape, dtype=np.int32),
        "target_truth_valid_block_fraction": np.full(
            shape, np.nan, dtype=np.float64
        ),
        "target_truth_support_qualified_block_count": np.zeros(
            shape, dtype=np.int32
        ),
        "target_truth_support_qualified_block_fraction": np.full(
            shape, np.nan, dtype=np.float64
        ),
        "target_phase_support_ratio_median": np.full(
            shape, np.nan, dtype=np.float64
        ),
        "target_phase_coefficient_amplitude_median": np.full(
            shape, np.nan, dtype=np.float64
        ),
        "band_phase_slope_rad": np.full(shape, np.nan, dtype=np.float64),
        "band_direction_consistency": np.full(shape, np.nan, dtype=np.float64),
        "band_slope_coherence": np.full(shape, np.nan, dtype=np.float64),
        "band_slope_valid_block_count": np.zeros(shape, dtype=np.int32),
        "band_slope_total_block_count": np.zeros(shape, dtype=np.int32),
        "band_slope_valid_block_fraction": np.full(
            shape, np.nan, dtype=np.float64
        ),
        "band_slope_support_qualified_block_count": np.zeros(
            shape, dtype=np.int32
        ),
        "band_slope_support_qualified_block_fraction": np.full(
            shape, np.nan, dtype=np.float64
        ),
        "band_phase_support_ratio_median": np.full(
            (candidate_count, frequency_count, 5), np.nan, dtype=np.float64
        ),
        "band_phase_coefficient_amplitude_median": np.full(
            (candidate_count, frequency_count, 5), np.nan, dtype=np.float64
        ),
        "block_count": np.zeros(candidate_count, dtype=np.int32),
        "clipped_sample_count": np.asarray(truth.clipped_sample_count, dtype=np.int64),
        "target_phase_resultant": np.asarray(
            truth.target_phase_resultant, dtype=np.float64
        ),
        "expected_upper_lower_phase_rad": np.asarray(
            truth.expected_upper_lower_phase_rad, dtype=np.float64
        ),
        "expected_band_phase_slope_rad": np.asarray(
            truth.expected_band_phase_slope_rad, dtype=np.float64
        ),
    }
    target_phase = (
        float(truth.expected_target_phase_rad)
        if truth.target_phase_resultant
        >= float(args.min_pooled_target_phase_resultant)
        else math.nan
    )
    for candidate_index, name in enumerate(_CANDIDATE_NAMES):
        if name == "crossfit_matched_spatial_projection":
            if args.skip_matched_projection:
                continue
            for frequency_index, frequency in enumerate(frequencies_hz):
                try:
                    projection = comparison._matched_projection_trace_set(
                        injected,
                        target=masks["target"],
                        control=masks["control"],
                        partitions=partitions,
                        frequency_hz=float(frequency),
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    continue
                # The one-frequency projection trace is scored against the common
                # predeclared grid so its local noise comes from other frequencies.
                measurement = _measure_grid(
                    injected,
                    projection,
                    frequencies_hz,
                    block_seconds=float(args.block_seconds),
                    min_block_seconds=float(args.min_block_seconds),
                    min_valid_fraction=float(args.min_block_valid_fraction),
                    max_interpolated_gap_seconds=float(
                        args.max_interpolated_gap_seconds
                    ),
                    truth_frequency_hz=float(job.frequency_hz),
                    truth_target_phase_rad=float(target_phase),
                    truth_time_origin_s=float(time_origin_s),
                    activity_envelope=truth.activity_envelope,
                    truth_amplitude_dn=float(job.amplitude_dn),
                    minimum_phase_support_ratio=float(
                        args.min_phase_region_support_ratio
                    ),
                    minimum_phase_coefficient_amplitude=float(
                        args.min_phase_coefficient_amplitude
                    ),
                )
                for key in (
                    "spectral_ratio",
                    "control_ratio",
                    "phase_offset_rad",
                    "phase_locking_value",
                    "phase_valid_block_count",
                    "phase_total_block_count",
                    "phase_valid_block_fraction",
                    "phase_support_qualified_block_count",
                    "phase_support_qualified_block_fraction",
                    "upper_phase_support_ratio_median",
                    "lower_phase_support_ratio_median",
                    "upper_phase_coefficient_amplitude_median",
                    "lower_phase_coefficient_amplitude_median",
                    "target_truth_residual_rad",
                    "target_truth_phase_coherence",
                    "target_truth_valid_block_count",
                    "target_truth_total_block_count",
                    "target_truth_valid_block_fraction",
                    "target_truth_support_qualified_block_count",
                    "target_truth_support_qualified_block_fraction",
                    "target_phase_support_ratio_median",
                    "target_phase_coefficient_amplitude_median",
                ):
                    result[key][candidate_index, frequency_index] = getattr(
                        measurement, key
                    )[frequency_index]
                result["block_count"][candidate_index] = max(
                    result["block_count"][candidate_index], measurement.block_count
                )
            continue
        measurement = _measure_grid(
            injected,
            candidates[name],
            frequencies_hz,
            block_seconds=float(args.block_seconds),
            min_block_seconds=float(args.min_block_seconds),
            min_valid_fraction=float(args.min_block_valid_fraction),
            max_interpolated_gap_seconds=float(args.max_interpolated_gap_seconds),
            truth_frequency_hz=float(job.frequency_hz),
            truth_target_phase_rad=float(target_phase),
            truth_time_origin_s=float(time_origin_s),
            activity_envelope=truth.activity_envelope,
            truth_amplitude_dn=float(job.amplitude_dn),
            minimum_phase_support_ratio=float(args.min_phase_region_support_ratio),
            minimum_phase_coefficient_amplitude=float(
                args.min_phase_coefficient_amplitude
            ),
        )
        for key in (
            "spectral_ratio",
            "control_ratio",
            "phase_offset_rad",
            "phase_locking_value",
            "phase_valid_block_count",
            "phase_total_block_count",
            "phase_valid_block_fraction",
            "phase_support_qualified_block_count",
            "phase_support_qualified_block_fraction",
            "upper_phase_support_ratio_median",
            "lower_phase_support_ratio_median",
            "upper_phase_coefficient_amplitude_median",
            "lower_phase_coefficient_amplitude_median",
            "target_truth_residual_rad",
            "target_truth_phase_coherence",
            "target_truth_valid_block_count",
            "target_truth_total_block_count",
            "target_truth_valid_block_fraction",
            "target_truth_support_qualified_block_count",
            "target_truth_support_qualified_block_fraction",
            "target_phase_support_ratio_median",
            "target_phase_coefficient_amplitude_median",
        ):
            result[key][candidate_index] = getattr(measurement, key)
        result["block_count"][candidate_index] = measurement.block_count

    if job.pattern == "traveling_wave":
        band_masks = [
            truth.band_index_by_pixel == band_index
            for band_index in range(truth.band_count)
        ]
        with _TRANSFORM_BUILD_LOCK:
            band_candidates = _band_candidates(
                injected,
                band_masks,
                partitions,
                nuisance_ridge=float(args.nuisance_ridge),
            )
        for candidate_index, name in enumerate(_CANDIDATE_NAMES):
            if name == "crossfit_matched_spatial_projection":
                continue
            (
                slope,
                consistency,
                coherence,
                valid_count,
                total_count,
                valid_fraction,
                support_qualified_count,
                support_qualified_fraction,
                band_support_median,
                band_amplitude_median,
            ) = _band_slope_grid(
                injected,
                band_candidates[name],
                frequencies_hz,
                block_seconds=float(args.block_seconds),
                min_block_seconds=float(args.min_block_seconds),
                min_valid_fraction=float(args.min_block_valid_fraction),
                max_interpolated_gap_seconds=float(args.max_interpolated_gap_seconds),
                minimum_phase_support_ratio=float(
                    args.min_phase_region_support_ratio
                ),
                minimum_phase_coefficient_amplitude=float(
                    args.min_phase_coefficient_amplitude
                ),
            )
            result["band_phase_slope_rad"][candidate_index] = slope
            result["band_direction_consistency"][candidate_index] = consistency
            result["band_slope_coherence"][candidate_index] = coherence
            result["band_slope_valid_block_count"][candidate_index] = valid_count
            result["band_slope_total_block_count"][candidate_index] = total_count
            result["band_slope_valid_block_fraction"][candidate_index] = valid_fraction
            result["band_slope_support_qualified_block_count"][
                candidate_index
            ] = support_qualified_count
            result["band_slope_support_qualified_block_fraction"][
                candidate_index
            ] = support_qualified_fraction
            result["band_phase_support_ratio_median"][
                candidate_index, :, : truth.band_count
            ] = band_support_median
            result["band_phase_coefficient_amplitude_median"][
                candidate_index, :, : truth.band_count
            ] = band_amplitude_median
    return result


def _confirmation_summary(
    spectral_ratio: np.ndarray,
    control_ratio: np.ndarray,
    candidate_index: int,
    frequency_index: int,
    confirmation_windows: np.ndarray,
    *,
    minimum_windows: int,
    minimum_fraction: float,
    minimum_spectral_ratio: float,
    minimum_control_ratio: float,
) -> dict[str, Any]:
    """Apply the predeclared confirmation coverage and signal gates."""

    confirmation = np.asarray(confirmation_windows, dtype=bool)
    spectral_values = np.asarray(spectral_ratio, dtype=np.float64)[
        candidate_index, :, frequency_index
    ]
    control_values = np.asarray(control_ratio, dtype=np.float64)[
        candidate_index, :, frequency_index
    ]
    total = int(np.count_nonzero(confirmation))
    valid = (
        confirmation
        & np.isfinite(spectral_values)
        & np.isfinite(control_values)
        & (spectral_values > 0.0)
        & (control_values > 0.0)
    )
    valid_count = int(np.count_nonzero(valid))
    valid_fraction = float(valid_count / total) if total else 0.0
    spectral = _finite_median(spectral_values[valid])
    control = _finite_median(control_values[valid])
    coverage_passed = bool(
        valid_count >= int(minimum_windows)
        and valid_fraction >= float(minimum_fraction)
    )
    signal_passed = bool(
        np.isfinite(spectral)
        and spectral >= float(minimum_spectral_ratio)
        and np.isfinite(control)
        and control >= float(minimum_control_ratio)
    )
    return {
        "confirmation_window_count_total": total,
        "confirmation_window_count_valid": valid_count,
        "confirmation_window_fraction_valid": valid_fraction,
        "confirmation_coverage_gate_passed": coverage_passed,
        "confirmation_signal_gate_passed": signal_passed,
        "confirmation_spectral_ratio": spectral,
        "confirmation_control_ratio": control,
        "confirmation_gate_passed": bool(coverage_passed and signal_passed),
    }


def _phase_partition_summary(
    phase_rad: np.ndarray,
    measured_coherence: np.ndarray,
    valid_block_count: np.ndarray,
    total_block_count: np.ndarray,
    valid_block_fraction: np.ndarray,
    support_qualified_block_count: np.ndarray,
    support_qualified_block_fraction: np.ndarray,
    partition_windows: np.ndarray,
    *,
    minimum_resultant: float,
    minimum_valid_blocks: int,
    minimum_valid_block_fraction: float,
    minimum_valid_windows: int,
    minimum_valid_window_fraction: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Gate a partition's phase angle on block and window-level support."""

    phase = np.asarray(phase_rad, dtype=np.float64)
    coherence = np.asarray(measured_coherence, dtype=np.float64)
    valid_blocks = np.asarray(valid_block_count, dtype=np.int64)
    total_blocks = np.asarray(total_block_count, dtype=np.int64)
    block_fraction = np.asarray(valid_block_fraction, dtype=np.float64)
    support_blocks = np.asarray(support_qualified_block_count, dtype=np.int64)
    support_fraction = np.asarray(
        support_qualified_block_fraction, dtype=np.float64
    )
    partition = np.asarray(partition_windows, dtype=bool)
    expected_shape = partition.shape
    for name, values in {
        "phase_rad": phase,
        "measured_coherence": coherence,
        "valid_block_count": valid_blocks,
        "total_block_count": total_blocks,
        "valid_block_fraction": block_fraction,
        "support_qualified_block_count": support_blocks,
        "support_qualified_block_fraction": support_fraction,
    }.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} shape must match partition_windows")
    support_window = (
        partition
        & (support_blocks >= int(minimum_valid_blocks))
        & np.isfinite(support_fraction)
        & (support_fraction >= float(minimum_valid_block_fraction))
    )
    phase_valid = (
        support_window
        & np.isfinite(phase)
        & np.isfinite(coherence)
        & (coherence >= float(minimum_resultant))
        & (valid_blocks >= int(minimum_valid_blocks))
        & np.isfinite(block_fraction)
        & (block_fraction >= float(minimum_valid_block_fraction))
    )
    window_total = int(np.count_nonzero(partition))
    window_valid = int(np.count_nonzero(phase_valid))
    window_fraction = float(window_valid / window_total) if window_total else 0.0
    block_total = int(np.sum(total_blocks[partition], dtype=np.int64))
    block_valid = int(np.sum(valid_blocks[partition], dtype=np.int64))
    support_block_valid = int(np.sum(support_blocks[partition], dtype=np.int64))
    support_window_count = int(np.count_nonzero(support_window))
    aggregate_block_fraction = float(block_valid / block_total) if block_total else 0.0
    within_window_coherence_median = _finite_median(coherence[partition])
    mean_phase, resultant = _circular_mean(phase[phase_valid])
    window_coverage_passed = bool(
        window_valid >= int(minimum_valid_windows)
        and window_fraction >= float(minimum_valid_window_fraction)
    )
    resultant_passed = bool(
        np.isfinite(resultant) and resultant >= float(minimum_resultant)
    )
    available = bool(window_coverage_passed and resultant_passed)
    unavailable_reason = (
        None
        if available
        else "no_windows_passed_phase_signal_support_gates"
        if support_window_count == 0
        else "no_windows_passed_phase_coherence_gate"
        if window_valid == 0
        else "insufficient_phase_valid_window_count_or_fraction"
        if not window_coverage_passed
        else "partition_phase_resultant_below_predeclared_threshold"
    )
    return (
        {
            "phase_mean_rad": mean_phase if available else math.nan,
            "phase_measured_resultant": resultant,
            "phase_within_window_coherence_median": within_window_coherence_median,
            "phase_resultant_gate_passed": resultant_passed,
            "phase_window_count_total": window_total,
            "phase_window_count_valid": window_valid,
            "phase_window_fraction_valid": window_fraction,
            "phase_window_coverage_gate_passed": window_coverage_passed,
            "phase_block_count_total": block_total,
            "phase_block_count_valid": block_valid,
            "phase_block_fraction_valid": aggregate_block_fraction,
            "phase_support_qualified_block_count": support_block_valid,
            "phase_support_qualified_block_fraction": (
                float(support_block_valid / block_total) if block_total else 0.0
            ),
            "phase_support_qualified_window_count": support_window_count,
            "phase_available": available,
            "phase_unavailable_reason": unavailable_reason,
        },
        phase_valid,
    )


def _summarize_job(
    job: InjectionJob,
    arrays: Mapping[str, np.ndarray],
    frequencies_hz: np.ndarray,
    window_roles: np.ndarray,
    window_indices: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    discovery = window_roles == "discovery"
    confirmation = window_roles == "confirmation"
    target_phase_resultant = _finite_median(arrays["target_phase_resultant"])
    pooled_target_phase_interpretable = bool(
        float(job.amplitude_dn) > 0.0
        and np.isfinite(target_phase_resultant)
        and target_phase_resultant
        >= float(args.min_pooled_target_phase_resultant)
    )
    pooled_phase_unavailable_reason = (
        "zero_dn_sanity_control_has_no_injected_phase_timing"
        if float(job.amplitude_dn) == 0.0
        else "injected_target_phase_resultant_below_predeclared_threshold"
        if not pooled_target_phase_interpretable
        else None
    )
    available_names = [
        name
        for name in _CANDIDATE_NAMES
        if not (args.skip_matched_projection and name == "crossfit_matched_spatial_projection")
    ]
    available_indices = [_CANDIDATE_NAMES.index(name) for name in available_names]
    selection = select_discovery_family(
        arrays["spectral_ratio"][available_indices],
        arrays["control_ratio"][available_indices],
        available_names,
        frequencies_hz,
        discovery,
        minimum_windows=int(args.min_discovery_windows),
        minimum_spectral_ratio=float(args.min_spectral_ratio),
        minimum_control_ratio=float(args.min_control_ratio),
    )
    candidate_summaries: list[dict[str, Any]] = []
    for candidate_index, name in enumerate(_CANDIDATE_NAMES):
        if args.skip_matched_projection and name == "crossfit_matched_spatial_projection":
            continue
        own = select_discovery_family(
            arrays["spectral_ratio"][candidate_index : candidate_index + 1],
            arrays["control_ratio"][candidate_index : candidate_index + 1],
            (name,),
            frequencies_hz,
            discovery,
            minimum_windows=int(args.min_discovery_windows),
            minimum_spectral_ratio=float(args.min_spectral_ratio),
            minimum_control_ratio=float(args.min_control_ratio),
        )
        summary: dict[str, Any] = {
            "candidate": name,
            "selected_frequency_hz": own.selected_frequency_hz,
            "discovery_gate_passed": own.selected_candidate is not None,
            "discovery_selection_score": own.selection_score,
            "discovery_spectral_ratio": own.spectral_ratio,
            "discovery_control_ratio": own.control_ratio,
            "confirmation_spectral_ratio": math.nan,
            "confirmation_control_ratio": math.nan,
            "confirmation_window_count_total": int(np.count_nonzero(confirmation)),
            "confirmation_window_count_valid": 0,
            "confirmation_window_fraction_valid": 0.0,
            "confirmation_coverage_gate_passed": False,
            "confirmation_signal_gate_passed": False,
            "confirmation_gate_passed": False,
            "frequency_error_hz": math.nan,
            "upper_lower_phase_offset_deg": math.nan,
            "upper_lower_phase_error_deg": math.nan,
            "upper_lower_phase_locking_value": math.nan,
            "upper_lower_within_window_coherence_median": math.nan,
            "upper_lower_phase_available": False,
            "upper_lower_phase_unavailable_reason": "candidate_frequency_not_selected",
            "upper_lower_phase_window_count_total": int(
                np.count_nonzero(confirmation)
            ),
            "upper_lower_phase_window_count_valid": 0,
            "upper_lower_phase_window_fraction_valid": 0.0,
            "upper_lower_phase_block_count_total": 0,
            "upper_lower_phase_block_count_valid": 0,
            "upper_lower_phase_block_fraction_valid": 0.0,
            "upper_lower_phase_support_qualified_block_count": 0,
            "upper_lower_phase_support_qualified_block_fraction": 0.0,
            "upper_lower_phase_support_qualified_window_count": 0,
            "upper_phase_support_ratio_median": math.nan,
            "lower_phase_support_ratio_median": math.nan,
            "upper_phase_coefficient_amplitude_median": math.nan,
            "lower_phase_coefficient_amplitude_median": math.nan,
            "phase_equivalent_timing_rmse_ms": math.nan,
            "target_phase_resultant": target_phase_resultant,
            "pooled_target_phase_interpretable": pooled_target_phase_interpretable,
            "phase_timing_interpretable": False,
            "phase_timing_unavailable_reason": (
                "candidate_has_no_pooled_phase_timing_contract"
                if name not in _PHASE_TIMING_CANDIDATES
                else pooled_phase_unavailable_reason
                if pooled_phase_unavailable_reason is not None
                else "candidate_frequency_not_selected"
            ),
            "target_phase_calibration_measured_resultant": math.nan,
            "target_phase_calibration_within_window_coherence_median": math.nan,
            "target_phase_calibration_window_count_valid": 0,
            "target_phase_calibration_window_fraction_valid": 0.0,
            "target_phase_confirmation_measured_resultant": math.nan,
            "target_phase_confirmation_within_window_coherence_median": math.nan,
            "target_phase_confirmation_window_count_valid": 0,
            "target_phase_confirmation_window_fraction_valid": 0.0,
            "target_phase_support_ratio_median": math.nan,
            "target_phase_coefficient_amplitude_median": math.nan,
            "target_phase_confirmation_support_qualified_block_count": 0,
            "target_phase_confirmation_support_qualified_block_fraction": 0.0,
            "target_phase_confirmation_support_qualified_window_count": 0,
            "band_phase_slope_rad_per_band": math.nan,
            "band_phase_slope_error_rad_per_band": math.nan,
            "travel_direction_correct": None,
            "band_direction_consistency": math.nan,
            "travel_phase_measured_resultant": math.nan,
            "travel_within_window_coherence_median": math.nan,
            "travel_phase_available": False,
            "travel_phase_unavailable_reason": "candidate_frequency_not_selected",
            "travel_phase_window_count_total": int(np.count_nonzero(confirmation)),
            "travel_phase_window_count_valid": 0,
            "travel_phase_window_fraction_valid": 0.0,
            "travel_phase_block_count_total": 0,
            "travel_phase_block_count_valid": 0,
            "travel_phase_block_fraction_valid": 0.0,
            "travel_phase_support_qualified_block_count": 0,
            "travel_phase_support_qualified_block_fraction": 0.0,
            "travel_phase_support_qualified_window_count": 0,
            "travel_band_support_ratio_median": [],
            "travel_band_coefficient_amplitude_median": [],
        }
        if own.selected_frequency_index is not None:
            frequency_index = int(own.selected_frequency_index)
            confirmation_summary = _confirmation_summary(
                arrays["spectral_ratio"],
                arrays["control_ratio"],
                candidate_index,
                frequency_index,
                confirmation,
                minimum_windows=int(args.min_confirmation_windows),
                minimum_fraction=float(args.min_confirmation_window_fraction),
                minimum_spectral_ratio=float(args.min_spectral_ratio),
                minimum_control_ratio=float(args.min_control_ratio),
            )
            summary.update(confirmation_summary)
            if float(job.amplitude_dn) > 0.0:
                summary["frequency_error_hz"] = abs(
                    float(own.selected_frequency_hz) - float(job.frequency_hz)
                )
            upper_phase_gate, _upper_phase_windows = _phase_partition_summary(
                arrays["phase_offset_rad"][candidate_index, :, frequency_index],
                arrays["phase_locking_value"][candidate_index, :, frequency_index],
                arrays["phase_valid_block_count"][candidate_index, :, frequency_index],
                arrays["phase_total_block_count"][candidate_index, :, frequency_index],
                arrays["phase_valid_block_fraction"][candidate_index, :, frequency_index],
                arrays["phase_support_qualified_block_count"][
                    candidate_index, :, frequency_index
                ],
                arrays["phase_support_qualified_block_fraction"][
                    candidate_index, :, frequency_index
                ],
                confirmation,
                minimum_resultant=float(args.min_phase_resultant),
                minimum_valid_blocks=int(args.min_phase_valid_blocks),
                minimum_valid_block_fraction=float(
                    args.min_phase_valid_block_fraction
                ),
                minimum_valid_windows=int(args.min_phase_confirmation_windows),
                minimum_valid_window_fraction=float(
                    args.min_phase_confirmation_window_fraction
                ),
            )
            summary.update(
                {
                    "upper_lower_phase_locking_value": upper_phase_gate[
                        "phase_measured_resultant"
                    ],
                    "upper_lower_within_window_coherence_median": upper_phase_gate[
                        "phase_within_window_coherence_median"
                    ],
                    "upper_lower_phase_window_count_total": upper_phase_gate[
                        "phase_window_count_total"
                    ],
                    "upper_lower_phase_window_count_valid": upper_phase_gate[
                        "phase_window_count_valid"
                    ],
                    "upper_lower_phase_window_fraction_valid": upper_phase_gate[
                        "phase_window_fraction_valid"
                    ],
                    "upper_lower_phase_block_count_total": upper_phase_gate[
                        "phase_block_count_total"
                    ],
                    "upper_lower_phase_block_count_valid": upper_phase_gate[
                        "phase_block_count_valid"
                    ],
                    "upper_lower_phase_block_fraction_valid": upper_phase_gate[
                        "phase_block_fraction_valid"
                    ],
                    "upper_lower_phase_support_qualified_block_count": upper_phase_gate[
                        "phase_support_qualified_block_count"
                    ],
                    "upper_lower_phase_support_qualified_block_fraction": upper_phase_gate[
                        "phase_support_qualified_block_fraction"
                    ],
                    "upper_lower_phase_support_qualified_window_count": upper_phase_gate[
                        "phase_support_qualified_window_count"
                    ],
                    "upper_phase_support_ratio_median": _finite_median(
                        arrays["upper_phase_support_ratio_median"][
                            candidate_index, confirmation, frequency_index
                        ]
                    ),
                    "lower_phase_support_ratio_median": _finite_median(
                        arrays["lower_phase_support_ratio_median"][
                            candidate_index, confirmation, frequency_index
                        ]
                    ),
                    "upper_phase_coefficient_amplitude_median": _finite_median(
                        arrays["upper_phase_coefficient_amplitude_median"][
                            candidate_index, confirmation, frequency_index
                        ]
                    ),
                    "lower_phase_coefficient_amplitude_median": _finite_median(
                        arrays["lower_phase_coefficient_amplitude_median"][
                            candidate_index, confirmation, frequency_index
                        ]
                    ),
                }
            )
            upper_phase_available = bool(
                upper_phase_gate["phase_available"]
                and summary["confirmation_gate_passed"]
            )
            summary["upper_lower_phase_available"] = upper_phase_available
            summary["upper_lower_phase_unavailable_reason"] = (
                None
                if upper_phase_available
                else "transform_frequency_confirmation_gate_failed"
                if not summary["confirmation_gate_passed"]
                else upper_phase_gate["phase_unavailable_reason"]
            )
            phase = float(upper_phase_gate["phase_mean_rad"])
            if upper_phase_available:
                summary["upper_lower_phase_offset_deg"] = float(np.degrees(phase))
            expected_phase = _finite_median(arrays["expected_upper_lower_phase_rad"])
            if (
                upper_phase_available
                and float(job.amplitude_dn) > 0.0
                and np.isfinite(expected_phase)
            ):
                summary["upper_lower_phase_error_deg"] = float(
                    np.degrees(circular_difference_rad(phase, expected_phase))
                )
            residual = arrays["target_truth_residual_rad"][
                candidate_index, :, frequency_index
            ]
            timing_frequency_matches = bool(
                abs(float(own.selected_frequency_hz - job.frequency_hz))
                <= 0.51 * float(args.frequency_step_hz)
            )
            target_calibration, _target_discovery_windows = _phase_partition_summary(
                residual,
                arrays["target_truth_phase_coherence"][
                    candidate_index, :, frequency_index
                ],
                arrays["target_truth_valid_block_count"][
                    candidate_index, :, frequency_index
                ],
                arrays["target_truth_total_block_count"][
                    candidate_index, :, frequency_index
                ],
                arrays["target_truth_valid_block_fraction"][
                    candidate_index, :, frequency_index
                ],
                arrays["target_truth_support_qualified_block_count"][
                    candidate_index, :, frequency_index
                ],
                arrays["target_truth_support_qualified_block_fraction"][
                    candidate_index, :, frequency_index
                ],
                discovery,
                minimum_resultant=float(args.min_phase_resultant),
                minimum_valid_blocks=int(args.min_phase_valid_blocks),
                minimum_valid_block_fraction=float(
                    args.min_phase_valid_block_fraction
                ),
                minimum_valid_windows=int(args.min_phase_confirmation_windows),
                minimum_valid_window_fraction=float(
                    args.min_phase_confirmation_window_fraction
                ),
            )
            target_confirmation, target_confirmation_windows = _phase_partition_summary(
                residual,
                arrays["target_truth_phase_coherence"][
                    candidate_index, :, frequency_index
                ],
                arrays["target_truth_valid_block_count"][
                    candidate_index, :, frequency_index
                ],
                arrays["target_truth_total_block_count"][
                    candidate_index, :, frequency_index
                ],
                arrays["target_truth_valid_block_fraction"][
                    candidate_index, :, frequency_index
                ],
                arrays["target_truth_support_qualified_block_count"][
                    candidate_index, :, frequency_index
                ],
                arrays["target_truth_support_qualified_block_fraction"][
                    candidate_index, :, frequency_index
                ],
                confirmation,
                minimum_resultant=float(args.min_phase_resultant),
                minimum_valid_blocks=int(args.min_phase_valid_blocks),
                minimum_valid_block_fraction=float(
                    args.min_phase_valid_block_fraction
                ),
                minimum_valid_windows=int(args.min_phase_confirmation_windows),
                minimum_valid_window_fraction=float(
                    args.min_phase_confirmation_window_fraction
                ),
            )
            summary.update(
                {
                    "target_phase_calibration_measured_resultant": target_calibration[
                        "phase_measured_resultant"
                    ],
                    "target_phase_calibration_within_window_coherence_median": target_calibration[
                        "phase_within_window_coherence_median"
                    ],
                    "target_phase_calibration_window_count_valid": target_calibration[
                        "phase_window_count_valid"
                    ],
                    "target_phase_calibration_window_fraction_valid": target_calibration[
                        "phase_window_fraction_valid"
                    ],
                    "target_phase_confirmation_measured_resultant": target_confirmation[
                        "phase_measured_resultant"
                    ],
                    "target_phase_confirmation_within_window_coherence_median": target_confirmation[
                        "phase_within_window_coherence_median"
                    ],
                    "target_phase_confirmation_window_count_valid": target_confirmation[
                        "phase_window_count_valid"
                    ],
                    "target_phase_confirmation_window_fraction_valid": target_confirmation[
                        "phase_window_fraction_valid"
                    ],
                    "target_phase_support_ratio_median": _finite_median(
                        arrays["target_phase_support_ratio_median"][
                            candidate_index, confirmation, frequency_index
                        ]
                    ),
                    "target_phase_coefficient_amplitude_median": _finite_median(
                        arrays["target_phase_coefficient_amplitude_median"][
                            candidate_index, confirmation, frequency_index
                        ]
                    ),
                    "target_phase_confirmation_support_qualified_block_count": target_confirmation[
                        "phase_support_qualified_block_count"
                    ],
                    "target_phase_confirmation_support_qualified_block_fraction": target_confirmation[
                        "phase_support_qualified_block_fraction"
                    ],
                    "target_phase_confirmation_support_qualified_window_count": target_confirmation[
                        "phase_support_qualified_window_count"
                    ],
                }
            )
            timing_available = bool(
                name in _PHASE_TIMING_CANDIDATES
                and pooled_target_phase_interpretable
                and summary["confirmation_gate_passed"]
                and timing_frequency_matches
                and target_calibration["phase_available"]
                and target_confirmation["phase_available"]
            )
            summary["phase_timing_interpretable"] = timing_available
            summary["phase_timing_unavailable_reason"] = (
                None
                if timing_available
                else "candidate_has_no_pooled_phase_timing_contract"
                if name not in _PHASE_TIMING_CANDIDATES
                else pooled_phase_unavailable_reason
                if pooled_phase_unavailable_reason is not None
                else "transform_frequency_confirmation_gate_failed"
                if not summary["confirmation_gate_passed"]
                else "selected_frequency_does_not_match_injected_frequency"
                if not timing_frequency_matches
                else target_calibration["phase_unavailable_reason"]
                if not target_calibration["phase_available"]
                else target_confirmation["phase_unavailable_reason"]
            )
            if timing_available:
                calibration = float(target_calibration["phase_mean_rad"])
                timing = phase_equivalent_timing_error_ms(
                    residual[target_confirmation_windows],
                    calibration,
                    frequency_hz=float(job.frequency_hz),
                )
                timing = timing[np.isfinite(timing)]
                if timing.size:
                    summary["phase_equivalent_timing_rmse_ms"] = float(
                        np.sqrt(np.mean(np.square(timing)))
                    )
            travel_gate, travel_windows = _phase_partition_summary(
                arrays["band_phase_slope_rad"][candidate_index, :, frequency_index],
                arrays["band_slope_coherence"][candidate_index, :, frequency_index],
                arrays["band_slope_valid_block_count"][
                    candidate_index, :, frequency_index
                ],
                arrays["band_slope_total_block_count"][
                    candidate_index, :, frequency_index
                ],
                arrays["band_slope_valid_block_fraction"][
                    candidate_index, :, frequency_index
                ],
                arrays["band_slope_support_qualified_block_count"][
                    candidate_index, :, frequency_index
                ],
                arrays["band_slope_support_qualified_block_fraction"][
                    candidate_index, :, frequency_index
                ],
                confirmation,
                minimum_resultant=float(args.min_phase_resultant),
                minimum_valid_blocks=int(args.min_phase_valid_blocks),
                minimum_valid_block_fraction=float(
                    args.min_phase_valid_block_fraction
                ),
                minimum_valid_windows=int(args.min_phase_confirmation_windows),
                minimum_valid_window_fraction=float(
                    args.min_phase_confirmation_window_fraction
                ),
            )
            expected_slope = _finite_median(arrays["expected_band_phase_slope_rad"])
            reported_band_count = (
                int(job.traveling_band_count) if job.pattern == "traveling_wave" else 0
            )
            travel_band_support = [
                _finite_median(
                    arrays["band_phase_support_ratio_median"][
                        candidate_index, confirmation, frequency_index, band_index
                    ]
                )
                for band_index in range(reported_band_count)
            ]
            travel_band_amplitude = [
                _finite_median(
                    arrays["band_phase_coefficient_amplitude_median"][
                        candidate_index, confirmation, frequency_index, band_index
                    ]
                )
                for band_index in range(reported_band_count)
            ]
            travel_available = bool(
                travel_gate["phase_available"]
                and summary["confirmation_gate_passed"]
                and np.isfinite(expected_slope)
                and float(job.amplitude_dn) > 0.0
            )
            summary.update(
                {
                    "travel_phase_measured_resultant": travel_gate[
                        "phase_measured_resultant"
                    ],
                    "travel_within_window_coherence_median": travel_gate[
                        "phase_within_window_coherence_median"
                    ],
                    "travel_phase_available": travel_available,
                    "travel_phase_unavailable_reason": (
                        None
                        if travel_available
                        else "injection_pattern_has_no_traveling_slope_truth"
                        if not np.isfinite(expected_slope)
                        else "transform_frequency_confirmation_gate_failed"
                        if not summary["confirmation_gate_passed"]
                        else travel_gate["phase_unavailable_reason"]
                    ),
                    "travel_phase_window_count_total": travel_gate[
                        "phase_window_count_total"
                    ],
                    "travel_phase_window_count_valid": travel_gate[
                        "phase_window_count_valid"
                    ],
                    "travel_phase_window_fraction_valid": travel_gate[
                        "phase_window_fraction_valid"
                    ],
                    "travel_phase_block_count_total": travel_gate[
                        "phase_block_count_total"
                    ],
                    "travel_phase_block_count_valid": travel_gate[
                        "phase_block_count_valid"
                    ],
                    "travel_phase_block_fraction_valid": travel_gate[
                        "phase_block_fraction_valid"
                    ],
                    "travel_phase_support_qualified_block_count": travel_gate[
                        "phase_support_qualified_block_count"
                    ],
                    "travel_phase_support_qualified_block_fraction": travel_gate[
                        "phase_support_qualified_block_fraction"
                    ],
                    "travel_phase_support_qualified_window_count": travel_gate[
                        "phase_support_qualified_window_count"
                    ],
                    "travel_band_support_ratio_median": travel_band_support,
                    "travel_band_coefficient_amplitude_median": travel_band_amplitude,
                }
            )
            if travel_available:
                slope = float(travel_gate["phase_mean_rad"])
                consistency = _finite_median(
                    arrays["band_direction_consistency"][
                        candidate_index, travel_windows, frequency_index
                    ]
                )
                summary["band_phase_slope_rad_per_band"] = slope
                summary["band_direction_consistency"] = consistency
                summary["band_phase_slope_error_rad_per_band"] = float(
                    circular_difference_rad(slope, expected_slope)
                )
                summary["travel_direction_correct"] = bool(
                    np.isfinite(slope) and np.sign(slope) == np.sign(expected_slope)
                )
        candidate_summaries.append(summary)

    family_summary: dict[str, Any] = {
        "selected_candidate": selection.selected_candidate,
        "selected_frequency_hz": selection.selected_frequency_hz,
        "discovery_gate_passed": selection.selected_candidate is not None,
        "discovery_selection_score": selection.selection_score,
        "discovery_spectral_ratio": selection.spectral_ratio,
        "discovery_control_ratio": selection.control_ratio,
        "confirmation_spectral_ratio": math.nan,
        "confirmation_control_ratio": math.nan,
        "confirmation_window_count_total": int(np.count_nonzero(confirmation)),
        "confirmation_window_count_valid": 0,
        "confirmation_window_fraction_valid": 0.0,
        "confirmation_coverage_gate_passed": False,
        "confirmation_signal_gate_passed": False,
        "confirmation_gate_passed": False,
        "frequency_error_hz": math.nan,
        "frequency_recovered_within_half_step": False,
        "zero_dn_sanity_gate_passed": False,
    }
    if selection.selected_candidate_index is not None:
        candidate_index = available_indices[int(selection.selected_candidate_index)]
        frequency_index = int(selection.selected_frequency_index)
        confirmation_summary = _confirmation_summary(
            arrays["spectral_ratio"],
            arrays["control_ratio"],
            candidate_index,
            frequency_index,
            confirmation,
            minimum_windows=int(args.min_confirmation_windows),
            minimum_fraction=float(args.min_confirmation_window_fraction),
            minimum_spectral_ratio=float(args.min_spectral_ratio),
            minimum_control_ratio=float(args.min_control_ratio),
        )
        confirmed = bool(confirmation_summary["confirmation_gate_passed"])
        error = (
            abs(float(selection.selected_frequency_hz) - float(job.frequency_hz))
            if float(job.amplitude_dn) > 0.0
            else math.nan
        )
        family_summary.update(
            {
                **confirmation_summary,
                "frequency_error_hz": error,
                "frequency_recovered_within_half_step": bool(
                    confirmed
                    and np.isfinite(error)
                    and error <= 0.51 * float(args.frequency_step_hz)
                ),
                "zero_dn_sanity_gate_passed": bool(
                    float(job.amplitude_dn) == 0.0 and confirmed
                ),
            }
        )
    return {
        "job_id": job.job_id,
        "job": job.payload(),
        "window_indices": window_indices.tolist(),
        "window_roles": window_roles.tolist(),
        "pooled_target_phase_contract": {
            "injected_spatial_phase_resultant": target_phase_resultant,
            "minimum_resultant": float(args.min_pooled_target_phase_resultant),
            "interpretable": pooled_target_phase_interpretable,
            "unavailable_reason": pooled_phase_unavailable_reason,
        },
        "candidate_summaries": candidate_summaries,
        "family": family_summary,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _save_job(
    directory: Path,
    job: InjectionJob,
    arrays: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"{job.job_id}.npz"
    temporary_npz = directory / f".{job.job_id}.tmp.npz"
    experiment_id = summary.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("job summary must contain a nonempty experiment_id")
    reserved = {
        "schema_name",
        "schema_version",
        "experiment_schema_version",
        "experiment_id",
    }
    if reserved.intersection(arrays):
        raise ValueError("job arrays use a reserved identity field")
    np.savez_compressed(
        temporary_npz,
        schema_name=np.asarray(_JOB_NPZ_SCHEMA),
        schema_version=np.asarray(_JOB_NPZ_SCHEMA_VERSION, dtype=np.int16),
        experiment_schema_version=np.asarray(
            _EXPERIMENT_SCHEMA_VERSION, dtype=np.int16
        ),
        experiment_id=np.asarray(experiment_id),
        **arrays,
    )
    os.replace(temporary_npz, npz_path)
    _atomic_json(directory / f"{job.job_id}.json", summary)


def _load_job(
    directory: Path,
    job: InjectionJob,
    experiment_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    npz_path = directory / f"{job.job_id}.npz"
    json_path = directory / f"{job.job_id}.json"
    if not npz_path.exists() or not json_path.exists():
        return None
    summary = json.loads(json_path.read_text())
    if summary.get("job") != _json_value(job.payload()):
        raise ValueError(f"resume identity mismatch for {job.job_id}")
    if summary.get("experiment_id") != str(experiment_id):
        raise ValueError(
            f"resume experiment mismatch for {job.job_id}; use a new resume directory"
        )
    with np.load(npz_path, allow_pickle=False) as data:
        required = {
            "schema_name",
            "schema_version",
            "experiment_schema_version",
            "experiment_id",
        }
        if not required.issubset(data.files):
            raise ValueError(f"resume NPZ identity metadata missing for {job.job_id}")
        if str(data["schema_name"].item()) != _JOB_NPZ_SCHEMA:
            raise ValueError(f"resume NPZ schema mismatch for {job.job_id}")
        if int(data["schema_version"].item()) != _JOB_NPZ_SCHEMA_VERSION:
            raise ValueError(f"resume NPZ schema version mismatch for {job.job_id}")
        if (
            int(data["experiment_schema_version"].item())
            != _EXPERIMENT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"resume NPZ experiment schema version mismatch for {job.job_id}"
            )
        if str(data["experiment_id"].item()) != str(experiment_id):
            raise ValueError(f"resume NPZ experiment mismatch for {job.job_id}")
        arrays = {
            key: np.asarray(data[key]) for key in data.files if key not in required
        }
    return arrays, summary


def _write_consolidated_npz(
    path: Path,
    *,
    experiment_id: str,
    job_ids: np.ndarray,
    candidate_names: np.ndarray,
    frequencies_hz: np.ndarray,
    window_indices: np.ndarray,
    window_roles: np.ndarray,
    arrays: Mapping[str, np.ndarray],
) -> None:
    reserved = {
        "schema_name",
        "schema_version",
        "experiment_schema_version",
        "experiment_id",
        "numpy_version",
        "scipy_version",
        "job_ids",
        "candidate_names",
        "frequencies_hz",
        "window_indices",
        "window_roles",
    }
    if reserved.intersection(arrays):
        raise ValueError("consolidated arrays use a reserved identity field")
    np.savez_compressed(
        path,
        schema_name=np.asarray(_CONSOLIDATED_NPZ_SCHEMA),
        schema_version=np.asarray(_CONSOLIDATED_NPZ_SCHEMA_VERSION, dtype=np.int16),
        experiment_schema_version=np.asarray(
            _EXPERIMENT_SCHEMA_VERSION, dtype=np.int16
        ),
        experiment_id=np.asarray(str(experiment_id)),
        numpy_version=np.asarray(_runtime_version_identity()["numpy"]),
        scipy_version=np.asarray(_runtime_version_identity()["scipy"]),
        job_ids=np.asarray(job_ids),
        candidate_names=np.asarray(candidate_names),
        frequencies_hz=np.asarray(frequencies_hz),
        window_indices=np.asarray(window_indices),
        window_roles=np.asarray(window_roles),
        **arrays,
    )


def _run_job(
    dataset: Any,
    windows: Sequence[Mapping[str, str]],
    window_roles: np.ndarray,
    window_indices: np.ndarray,
    job: InjectionJob,
    frequencies_hz: np.ndarray,
    masks: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    resume_dir: Path,
    experiment_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    resumed = _load_job(resume_dir, job, experiment_id)
    if resumed is not None:
        return resumed[0], resumed[1], 0.0
    started = time.perf_counter()
    window_results = [
        _window_metrics(
            dataset,
            source,
            job,
            frequencies_hz,
            masks,
            args,
            time_origin_s=float(np.asarray(dataset.timestamps_s)[0]),
        )
        for source in windows
    ]
    arrays: dict[str, np.ndarray] = {}
    for key in window_results[0]:
        value = np.asarray(window_results[0][key])
        arrays[key] = np.stack([np.asarray(result[key]) for result in window_results])
        if value.ndim >= 1 and value.shape[0] == len(_CANDIDATE_NAMES):
            arrays[key] = np.moveaxis(arrays[key], 1, 0)
    summary = _summarize_job(
        job,
        arrays,
        frequencies_hz,
        window_roles,
        window_indices,
        args,
    )
    summary["experiment_id"] = str(experiment_id)
    elapsed = float(time.perf_counter() - started)
    summary["elapsed_seconds"] = elapsed
    _save_job(resume_dir, job, arrays, summary)
    return arrays, summary, elapsed


def _write_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        job = dict(summary["job"])
        family = dict(summary["family"])
        for candidate in summary["candidate_summaries"]:
            rows.append(
                {
                    "job_id": summary["job_id"],
                    **job,
                    "family_selected_candidate": family["selected_candidate"],
                    "family_selected_frequency_hz": family["selected_frequency_hz"],
                    "family_confirmation_gate_passed": family[
                        "confirmation_gate_passed"
                    ],
                    "family_frequency_error_hz": family["frequency_error_hz"],
                    "family_zero_dn_sanity_gate_passed": family[
                        "zero_dn_sanity_gate_passed"
                    ],
                    **candidate,
                }
            )
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_value(rows))


def _write_plot(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = {
        "synchronous": "#246a73",
        "opposite_upper_lower": "#b85c38",
        "fixed_upper_lower_delay": "#6b5b95",
        "traveling_wave": "#3c7d44",
    }
    timing_labels = {
        "huber_savgol_derivative_w11": "SG derivative (11)",
        "huber_normalized_signed_lag12": "signed lag (12)",
        "huber_normalized_signed_lag16": "signed lag (16)",
    }

    def plot_number(value: Any) -> float:
        return float(value) if value is not None else math.nan

    for summary in summaries:
        job = summary["job"]
        family = summary["family"]
        color = colors[str(job["pattern"])]
        marker = "o" if str(job["activity_mode"]) == "continuous" else "s"
        axes[0, 0].scatter(
            float(job["amplitude_dn"]),
            plot_number(family["confirmation_spectral_ratio"]),
            color=color,
            marker=marker,
            alpha=0.8,
        )
        axes[0, 1].scatter(
            float(job["amplitude_dn"]),
            plot_number(family["frequency_error_hz"]),
            color=color,
            marker=marker,
            alpha=0.8,
        )
        for candidate in summary["candidate_summaries"]:
            if candidate["candidate"] not in _PHASE_TIMING_CANDIDATES:
                continue
            timing = candidate["phase_equivalent_timing_rmse_ms"]
            if timing is not None and np.isfinite(float(timing)):
                axes[1, 0].scatter(
                    timing_labels[candidate["candidate"]],
                    float(timing),
                    color=color,
                    marker=marker,
                    alpha=0.7,
                )
            if str(job["pattern"]) == "traveling_wave":
                slope_error = candidate["band_phase_slope_error_rad_per_band"]
                if slope_error is not None and np.isfinite(float(slope_error)):
                    axes[1, 1].scatter(
                        timing_labels[candidate["candidate"]],
                        float(slope_error),
                        color=color,
                        marker=marker,
                        alpha=0.7,
                    )
    axes[0, 0].axhline(1.5, color="#555555", linestyle="--", linewidth=1)
    axes[0, 0].set(xlabel="injected amplitude (Mono8 DN)", ylabel="held-out spectral ratio")
    axes[0, 1].axhline(0.0, color="#555555", linewidth=1)
    axes[0, 1].set(xlabel="injected amplitude (Mono8 DN)", ylabel="family frequency error (Hz)")
    axes[1, 0].set(ylabel="phase-equivalent timing RMSE (ms)")
    axes[1, 1].axhline(0.0, color="#555555", linewidth=1)
    axes[1, 1].set(ylabel="traveling slope error (rad / band)")
    for axis in axes[1]:
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle(
        "Conditional cached-photometry transform/frequency recoverability"
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inject known Mono8 oscillators into frozen cached-mask pixels and "
            "measure conditional cross-fitted transform/frequency recoverability."
        )
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--longitudinal-csv", type=Path, required=True)
    parser.add_argument("--original-mask-npz", type=Path, required=True)
    parser.add_argument("--original-mask-key", default="heart_support_mask")
    parser.add_argument("--regions-npz", type=Path, required=True)
    parser.add_argument("--upper-key", default="upper_mask")
    parser.add_argument("--lower-key", default="lower_mask")
    parser.add_argument("--frequency-source-mask", default="intersection_8")
    parser.add_argument("--patterns", default="synchronous,opposite_upper_lower,fixed_upper_lower_delay,traveling_wave")
    parser.add_argument("--injection-frequencies-hz", default="2.6,3.25,3.8")
    parser.add_argument("--amplitudes-dn", default="0,0.5,1.5")
    parser.add_argument("--activity-modes", default="continuous,intermittent")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--upper-lower-delay-deg", type=float, default=90.0)
    parser.add_argument("--traveling-band-counts", default="3,5")
    parser.add_argument("--traveling-directions", default="-1,1")
    parser.add_argument("--traveling-phase-span-deg", type=float, default=120.0)
    parser.add_argument("--intermittent-period-seconds", type=float, default=12.0)
    parser.add_argument("--intermittent-active-fraction", type=float, default=0.5)
    parser.add_argument("--intermittent-ramp-seconds", type=float, default=0.25)
    parser.add_argument("--frequency-min-hz", type=float, default=2.0)
    parser.add_argument("--frequency-max-hz", type=float, default=4.0)
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--block-seconds", type=float, default=4.0)
    parser.add_argument("--guard-seconds", type=float, default=0.25)
    parser.add_argument("--min-block-seconds", type=float, default=2.0)
    parser.add_argument("--min-block-valid-fraction", type=float, default=0.7)
    parser.add_argument("--max-interpolated-gap-seconds", type=float, default=0.02)
    parser.add_argument("--nuisance-ridge", type=float, default=1e-6)
    parser.add_argument("--outer-discovery-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--min-discovery-windows", type=int, default=3)
    parser.add_argument("--min-confirmation-windows", type=int, default=3)
    parser.add_argument("--min-confirmation-window-fraction", type=float, default=0.5)
    parser.add_argument("--min-spectral-ratio", type=float, default=1.5)
    parser.add_argument("--min-control-ratio", type=float, default=1.1)
    parser.add_argument("--min-pooled-target-phase-resultant", type=float, default=0.1)
    parser.add_argument("--min-phase-resultant", type=float, default=0.5)
    parser.add_argument("--min-phase-region-support-ratio", type=float, default=1.5)
    parser.add_argument(
        "--min-phase-coefficient-amplitude", type=float, default=1e-8
    )
    parser.add_argument("--min-phase-valid-blocks", type=int, default=2)
    parser.add_argument("--min-phase-valid-block-fraction", type=float, default=0.5)
    parser.add_argument("--min-phase-confirmation-windows", type=int, default=3)
    parser.add_argument(
        "--min-phase-confirmation-window-fraction", type=float, default=0.5
    )
    parser.add_argument("--max-timebase-relative-jitter", type=float, default=0.05)
    parser.add_argument("--window-indices")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--skip-matched-projection", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-count", type=int, default=1)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if int(args.workers) < 1 or int(args.replicates) < 1:
        raise ValueError("workers and replicates must be positive")
    if int(args.min_discovery_windows) < 1 or int(args.min_confirmation_windows) < 1:
        raise ValueError("minimum discovery/confirmation window counts must be positive")
    if not 0.0 < float(args.min_confirmation_window_fraction) <= 1.0:
        raise ValueError("min-confirmation-window-fraction must be in (0, 1]")
    if not 0.0 < float(args.min_pooled_target_phase_resultant) <= 1.0:
        raise ValueError("min-pooled-target-phase-resultant must be in (0, 1]")
    if not 0.0 < float(args.min_phase_resultant) <= 1.0:
        raise ValueError("min-phase-resultant must be in (0, 1]")
    if float(args.min_phase_region_support_ratio) < 1.0:
        raise ValueError("min-phase-region-support-ratio must be at least 1")
    if float(args.min_phase_coefficient_amplitude) <= 0.0:
        raise ValueError("min-phase-coefficient-amplitude must be positive")
    if int(args.min_phase_valid_blocks) < 1 or int(
        args.min_phase_confirmation_windows
    ) < 1:
        raise ValueError("minimum phase block/window counts must be positive")
    if not 0.0 < float(args.min_phase_valid_block_fraction) <= 1.0:
        raise ValueError("min-phase-valid-block-fraction must be in (0, 1]")
    if not 0.0 < float(args.min_phase_confirmation_window_fraction) <= 1.0:
        raise ValueError("min-phase-confirmation-window-fraction must be in (0, 1]")
    if not 0.0 <= float(args.max_timebase_relative_jitter) < 1.0:
        raise ValueError("max-timebase-relative-jitter must be in [0, 1)")
    frequencies_hz = _frequency_grid(
        args.frequency_min_hz, args.frequency_max_hz, args.frequency_step_hz
    )
    injection_frequencies = _parse_csv_values(args.injection_frequencies_hz, float)
    if any(
        np.min(np.abs(frequencies_hz - frequency)) > 1e-8
        for frequency in injection_frequencies
    ):
        raise ValueError("all injection frequencies must lie exactly on the search grid")
    dataset = load_dataset(args.dataset_npz)
    timebase_diagnostics = _validate_timebase(
        dataset.timestamps_s,
        dataset.frame_indices,
        frequencies_hz,
        maximum_relative_jitter=float(args.max_timebase_relative_jitter),
    )
    original_image = _read_mask(args.original_mask_npz, args.original_mask_key)
    with np.load(args.regions_npz, allow_pickle=False) as data:
        upper_stored = np.asarray(data[args.upper_key], dtype=bool)
        lower_stored = np.asarray(data[args.lower_key], dtype=bool)
    target = comparison._stored_mask_at_pixels(
        original_image, dataset, name="original frozen target"
    )
    upper = comparison._stored_mask_at_pixels(
        upper_stored, dataset, name="upper frozen target"
    ) & target
    lower = comparison._stored_mask_at_pixels(
        lower_stored, dataset, name="lower frozen target"
    ) & target
    if np.any(upper & lower) or not np.array_equal(upper | lower, target):
        raise ValueError("upper/lower masks must partition original target")
    reference_image, control_image = comparison._auto_reference_and_control_masks(
        original_image
    )
    reference = _mask_at_pixels(reference_image, dataset.pixel_xy)
    control = _mask_at_pixels(control_image, dataset.pixel_xy)
    masks = {
        "target": comparison._validate_pixel_mask("target", target),
        "upper": comparison._validate_pixel_mask("upper", upper),
        "lower": comparison._validate_pixel_mask("lower", lower),
        "reference": comparison._validate_pixel_mask("reference", reference),
        "control": comparison._validate_pixel_mask("control", control),
    }
    selected_windows = (
        set(_parse_csv_values(args.window_indices, int))
        if args.window_indices
        else None
    )
    windows = _read_windows(
        args.longitudinal_csv,
        args.frequency_source_mask,
        selected_windows,
        args.max_windows,
    )
    window_indices = np.asarray(
        [int(source["window_index"]) for source in windows], dtype=np.int32
    )
    window_roles = np.asarray(
        [
            "discovery"
            if index % 2 == int(args.outer_discovery_parity)
            else "confirmation"
            for index in window_indices
        ]
    )
    if np.count_nonzero(window_roles == "discovery") < int(args.min_discovery_windows):
        raise ValueError("selected windows do not provide min-discovery-windows")
    if np.count_nonzero(window_roles == "confirmation") < int(
        args.min_confirmation_windows
    ):
        raise ValueError("selected windows do not provide min-confirmation-windows")

    def source_identity(path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    experiment_payload = {
        "schema_version": _EXPERIMENT_SCHEMA_VERSION,
        "implementation_code": _implementation_code_identity(),
        "runtime_versions": _runtime_version_identity(),
        "timebase": timebase_diagnostics,
        "sources": {
            "dataset_npz": source_identity(args.dataset_npz),
            "longitudinal_csv": source_identity(args.longitudinal_csv),
            "original_mask_npz": source_identity(args.original_mask_npz),
            "regions_npz": source_identity(args.regions_npz),
        },
        "candidate_names": list(_CANDIDATE_NAMES),
        "frequencies_hz": frequencies_hz.tolist(),
        "window_indices": window_indices.tolist(),
        "window_bounds": [
            [source["window_frame_start"], source["window_frame_stop_inclusive"]]
            for source in windows
        ],
        "frame_count": args.frame_count,
        "outer_discovery_parity": int(args.outer_discovery_parity),
        "block_seconds": float(args.block_seconds),
        "guard_seconds": float(args.guard_seconds),
        "min_block_seconds": float(args.min_block_seconds),
        "min_block_valid_fraction": float(args.min_block_valid_fraction),
        "max_interpolated_gap_seconds": float(args.max_interpolated_gap_seconds),
        "nuisance_ridge": float(args.nuisance_ridge),
        "skip_matched_projection": bool(args.skip_matched_projection),
        "traveling_phase_span_deg": float(args.traveling_phase_span_deg),
        "upper_lower_delay_deg": float(args.upper_lower_delay_deg),
        "intermittent_period_seconds": float(args.intermittent_period_seconds),
        "intermittent_active_fraction": float(args.intermittent_active_fraction),
        "intermittent_ramp_seconds": float(args.intermittent_ramp_seconds),
        "selection_gate": {
            "minimum_discovery_windows": int(args.min_discovery_windows),
            "minimum_confirmation_windows": int(args.min_confirmation_windows),
            "minimum_confirmation_window_fraction": float(
                args.min_confirmation_window_fraction
            ),
            "minimum_spectral_ratio": float(args.min_spectral_ratio),
            "minimum_control_ratio": float(args.min_control_ratio),
        },
        "minimum_pooled_target_phase_resultant": float(
            args.min_pooled_target_phase_resultant
        ),
        "phase_metric_gate": {
            "support_ratio_definition": (
                "selected-frequency coefficient amplitude divided by median "
                "coefficient amplitude at grid frequencies at least 0.2 Hz away"
            ),
            "all_contributing_regions_or_bands_must_pass_support": True,
            "support_ratio_default_rationale": (
                "the 1.5 default matches the predeclared descriptive spectral-ratio "
                "gate while remaining independently configurable"
            ),
            "minimum_resultant": float(args.min_phase_resultant),
            "minimum_region_target_to_sideband_support_ratio": float(
                args.min_phase_region_support_ratio
            ),
            "minimum_region_coefficient_amplitude": float(
                args.min_phase_coefficient_amplitude
            ),
            "minimum_valid_blocks": int(args.min_phase_valid_blocks),
            "minimum_valid_block_fraction": float(
                args.min_phase_valid_block_fraction
            ),
            "minimum_valid_partition_windows": int(
                args.min_phase_confirmation_windows
            ),
            "minimum_valid_partition_window_fraction": float(
                args.min_phase_confirmation_window_fraction
            ),
        },
        "mask_pixel_indices": {
            name: np.flatnonzero(mask).tolist() for name, mask in masks.items()
        },
    }
    experiment_id = stable_spec_id(experiment_payload, length=24)

    jobs = _jobs(args)
    if not jobs:
        raise ValueError("no jobs belong to this deterministic batch")
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    resume_dir = args.resume_dir or output_prefix.with_suffix(".injection_recovery.jobs")
    resume_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[str, tuple[dict[str, np.ndarray], dict[str, Any], float]] = {}
    if int(args.workers) == 1:
        for position, job in enumerate(jobs, start=1):
            completed[job.job_id] = _run_job(
                dataset,
                windows,
                window_roles,
                window_indices,
                job,
                frequencies_hz,
                masks,
                args,
                resume_dir,
                experiment_id,
            )
            print(f"job {position}/{len(jobs)} {job.job_id}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(
                    _run_job,
                    dataset,
                    windows,
                    window_roles,
                    window_indices,
                    job,
                    frequencies_hz,
                    masks,
                    args,
                    resume_dir,
                    experiment_id,
                ): job
                for job in jobs
            }
            for position, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                completed[job.job_id] = future.result()
                print(f"job {position}/{len(jobs)} {job.job_id}", flush=True)

    ordered = [completed[job.job_id] for job in jobs]
    summaries = [item[1] for item in ordered]
    csv_path = output_prefix.with_suffix(".injection_recovery.csv")
    arrays_path = output_prefix.with_suffix(".injection_recovery.arrays.npz")
    plot_path = output_prefix.with_suffix(".injection_recovery.png")
    summary_path = output_prefix.with_suffix(".injection_recovery.summary.json")
    _write_csv(csv_path, summaries)
    stacked: dict[str, np.ndarray] = {}
    for key in ordered[0][0]:
        stacked[key] = np.stack([item[0][key] for item in ordered])
    _write_consolidated_npz(
        arrays_path,
        experiment_id=experiment_id,
        job_ids=np.asarray([job.job_id for job in jobs]),
        candidate_names=np.asarray(_CANDIDATE_NAMES),
        frequencies_hz=frequencies_hz,
        window_indices=window_indices,
        window_roles=window_roles,
        arrays=stacked,
    )
    _write_plot(plot_path, summaries)
    invocation_elapsed = np.asarray([item[2] for item in ordered], dtype=np.float64)
    all_elapsed = np.asarray(
        [float(item[1].get("elapsed_seconds", math.nan)) for item in ordered],
        dtype=np.float64,
    )
    all_elapsed = all_elapsed[np.isfinite(all_elapsed) & (all_elapsed > 0.0)]
    per_job = float(np.median(all_elapsed)) if all_elapsed.size else math.nan
    full_frequency_count = 41
    current_projection_fits = frequencies_hz.size if not args.skip_matched_projection else 0
    cost_scale = (
        full_frequency_count / max(current_projection_fits, 1)
        if not args.skip_matched_projection
        else math.nan
    )
    zero_job_count = int(
        sum(float(item["job"]["amplitude_dn"]) == 0.0 for item in summaries)
    )
    zero_gate_pass_count = int(
        sum(
            bool(item["family"]["zero_dn_sanity_gate_passed"])
            for item in summaries
        )
    )
    summary = {
        "interpretation": (
            "conditional_cached_photometry_transform_frequency_recoverability_"
            "with_frozen_masks"
        ),
        "scope": {
            "conditions_on_existing_cached_traces": True,
            "conditions_on_frozen_target_reference_control_and_regional_masks": True,
            "tests_transform_family_recoverability": True,
            "tests_frequency_search_recoverability": True,
            "reruns_spatial_localization": False,
            "estimates_false_positive_rate": False,
            "validates_individual_events": False,
            "establishes_cardiac_identity": False,
        },
        "experiment_id": experiment_id,
        "experiment_payload": experiment_payload,
        "runtime_versions": _runtime_version_identity(),
        "timebase": timebase_diagnostics,
        "sources": {
            "dataset_npz": str(args.dataset_npz),
            "longitudinal_csv": str(args.longitudinal_csv),
            "original_mask_npz": str(args.original_mask_npz),
            "regions_npz": str(args.regions_npz),
        },
        "pixel_format": dataset.metadata.get("pixel_format", "mono8_reported_by_user"),
        "injection_contract": {
            "domain": "additive Mono8 digital numbers after cached bilinear sampling",
            "validity_nuisance_tracking_preserved": True,
            "clip_range_dn": [0.0, 255.0],
            "adaptive_transform_and_frequency_selection_rerun_per_job": True,
            "outer_selection_uses_confirmation_windows": False,
            "matched_projection_fit_per_tested_frequency": not args.skip_matched_projection,
            "phase_timing_metric": (
                "conditional nearest-cycle phase-equivalent error after discovery-only "
                "transfer-phase calibration; unavailable for low-resultant pooled "
                "spatial injections and not individual-event validation"
            ),
            "minimum_pooled_target_phase_resultant": float(
                args.min_pooled_target_phase_resultant
            ),
            "phase_metric_gate": experiment_payload["phase_metric_gate"],
        },
        "search": {
            "candidate_names": list(_CANDIDATE_NAMES),
            "frequencies_hz": frequencies_hz,
            "discovery_window_parity": int(args.outer_discovery_parity),
            "window_indices": window_indices,
            "minimum_discovery_windows": int(args.min_discovery_windows),
            "minimum_confirmation_windows": int(args.min_confirmation_windows),
            "minimum_confirmation_window_fraction": float(
                args.min_confirmation_window_fraction
            ),
            "descriptive_gate": {
                "spectral_ratio": float(args.min_spectral_ratio),
                "control_ratio": float(args.min_control_ratio),
                "calibrated": False,
            },
        },
        "batch": {
            "batch_index": int(args.batch_index),
            "batch_count": int(args.batch_count),
            "workers": int(args.workers),
            "job_count": len(jobs),
            "resume_dir": str(resume_dir),
        },
        "smoke_limits": {
            "frame_count_per_window": args.frame_count,
            "maximum_windows": args.max_windows,
            "skip_matched_projection": bool(args.skip_matched_projection),
        },
        "cost": {
            "median_observed_job_seconds": per_job,
            "new_job_count_this_invocation": int(
                np.count_nonzero(invocation_elapsed > 0.0)
            ),
            "resumed_job_count_this_invocation": int(
                np.count_nonzero(invocation_elapsed == 0.0)
            ),
            "observed_timed_job_count": int(all_elapsed.size),
            "rough_41_frequency_seconds_per_job": (
                per_job * cost_scale
                if np.isfinite(cost_scale) and np.isfinite(per_job)
                else None
            ),
            "rough_current_job_matrix_at_41_frequencies_seconds": (
                per_job * cost_scale * len(jobs)
                if np.isfinite(cost_scale) and np.isfinite(per_job)
                else None
            ),
            "note": "linear extrapolation; memory contention can make parallel runs slower",
        },
        "zero_amplitude_sanity_control": {
            "job_count": zero_job_count,
            "confirmation_gate_pass_count": zero_gate_pass_count,
            "purpose": "single fixed-background pipeline sanity control",
            "independent_cached_backgrounds": zero_job_count,
            "estimates_false_positive_rate": False,
            "reason": (
                "one canonical zero-DN job on one fixed cached background cannot "
                "estimate a false-positive rate"
            ),
        },
        "jobs": summaries,
        "outputs": {
            "csv": str(csv_path),
            "arrays_npz": str(arrays_path),
            "plot": str(plot_path),
            "summary_json": str(summary_path),
        },
    }
    _atomic_json(summary_path, summary)
    print(json.dumps(_json_value(summary["cost"]), indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
