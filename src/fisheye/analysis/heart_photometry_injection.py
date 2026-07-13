from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from fisheye.analysis.local_rostral_heartrate import LocalCoordinateDataset


InjectionPattern = Literal[
    "synchronous",
    "opposite_upper_lower",
    "fixed_upper_lower_delay",
    "traveling_wave",
]
ActivityMode = Literal["continuous", "intermittent"]
BandAxis = Literal["x", "y"]


@dataclass(frozen=True)
class PhotometryInjectionSpec:
    """A deterministic additive signal expressed in Mono8 digital numbers."""

    pattern: InjectionPattern
    frequency_hz: float
    amplitude_dn: float
    activity_mode: ActivityMode = "continuous"
    initial_phase_rad: float = 0.0
    upper_lower_delay_rad: float = math.pi / 2.0
    traveling_band_count: int = 5
    traveling_phase_span_rad: float = math.pi
    traveling_direction: int = 1
    intermittent_period_seconds: float = 12.0
    intermittent_active_fraction: float = 0.5
    intermittent_ramp_seconds: float = 0.25
    clip_min_dn: float = 0.0
    clip_max_dn: float = 255.0

    def validated(self) -> PhotometryInjectionSpec:
        if self.pattern not in {
            "synchronous",
            "opposite_upper_lower",
            "fixed_upper_lower_delay",
            "traveling_wave",
        }:
            raise ValueError(f"unsupported injection pattern {self.pattern!r}")
        if self.activity_mode not in {"continuous", "intermittent"}:
            raise ValueError(f"unsupported activity mode {self.activity_mode!r}")
        if not np.isfinite(self.frequency_hz) or float(self.frequency_hz) <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        if not np.isfinite(self.amplitude_dn) or float(self.amplitude_dn) < 0.0:
            raise ValueError("amplitude_dn must be finite and nonnegative")
        if not np.isfinite(self.initial_phase_rad):
            raise ValueError("initial_phase_rad must be finite")
        if not np.isfinite(self.upper_lower_delay_rad):
            raise ValueError("upper_lower_delay_rad must be finite")
        if int(self.traveling_band_count) < 3 or int(self.traveling_band_count) > 5:
            raise ValueError("traveling_band_count must be between three and five")
        if not np.isfinite(self.traveling_phase_span_rad):
            raise ValueError("traveling_phase_span_rad must be finite")
        if abs(float(self.traveling_phase_span_rad)) >= 2.0 * np.pi:
            raise ValueError("traveling_phase_span_rad magnitude must be below one cycle")
        if int(self.traveling_direction) not in {-1, 1}:
            raise ValueError("traveling_direction must be -1 or 1")
        if float(self.intermittent_period_seconds) <= 0.0:
            raise ValueError("intermittent_period_seconds must be positive")
        if not 0.0 < float(self.intermittent_active_fraction) < 1.0:
            raise ValueError("intermittent_active_fraction must be in (0, 1)")
        if float(self.intermittent_ramp_seconds) < 0.0:
            raise ValueError("intermittent_ramp_seconds cannot be negative")
        active_seconds = (
            float(self.intermittent_period_seconds)
            * float(self.intermittent_active_fraction)
        )
        if 2.0 * float(self.intermittent_ramp_seconds) > active_seconds:
            raise ValueError("intermittent ramps cannot exceed the active interval")
        if not float(self.clip_min_dn) < float(self.clip_max_dn):
            raise ValueError("clip_min_dn must be below clip_max_dn")
        return self


@dataclass(frozen=True)
class PhotometryInjectionTruth:
    spec: PhotometryInjectionSpec
    target_mask: np.ndarray
    phase_by_pixel_rad: np.ndarray
    band_index_by_pixel: np.ndarray
    band_count: int
    activity_envelope: np.ndarray
    injected_delta_dn: np.ndarray
    expected_target_phase_rad: float
    target_phase_resultant: float
    expected_upper_lower_phase_rad: float
    expected_band_phase_slope_rad: float
    clipped_sample_count: int
    time_origin_s: float


@dataclass(frozen=True)
class FamilySelection:
    selected_candidate_index: int | None
    selected_frequency_index: int | None
    selected_candidate: str | None
    selected_frequency_hz: float
    selection_score: float
    spectral_ratio: float
    control_ratio: float
    passing_pair_count: int
    evaluated_pair_count: int


def circular_difference_rad(observed: float | np.ndarray, expected: float | np.ndarray) -> np.ndarray:
    """Return observed-minus-expected wrapped to ``[-pi, pi)``."""

    difference = np.asarray(observed, dtype=np.float64) - np.asarray(
        expected, dtype=np.float64
    )
    return (difference + np.pi) % (2.0 * np.pi) - np.pi


def phase_equivalent_timing_error_ms(
    observed_phase_rad: float | np.ndarray,
    expected_phase_rad: float | np.ndarray,
    *,
    frequency_hz: float,
) -> np.ndarray:
    """Convert circular phase error to the nearest-cycle timing error.

    This is not an independently detected event error. It is the time shift that
    would produce the measured phase error at one declared frequency.
    """

    frequency = float(frequency_hz)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    return circular_difference_rad(observed_phase_rad, expected_phase_rad) / (
        2.0 * np.pi * frequency
    ) * 1000.0


def activity_envelope(
    timestamps_s: np.ndarray,
    spec: PhotometryInjectionSpec,
    *,
    time_origin_s: float,
) -> np.ndarray:
    """Build the fixed continuous or cosine-ramped intermittent envelope."""

    resolved = spec.validated()
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or not timestamps.size or not np.isfinite(timestamps).all():
        raise ValueError("timestamps_s must be a nonempty finite vector")
    if resolved.activity_mode == "continuous":
        return np.ones(timestamps.shape, dtype=np.float64)

    period = float(resolved.intermittent_period_seconds)
    active_seconds = period * float(resolved.intermittent_active_fraction)
    ramp = float(resolved.intermittent_ramp_seconds)
    position = np.mod(timestamps - float(time_origin_s), period)
    active = position < active_seconds
    envelope = active.astype(np.float64)
    if ramp <= 0.0:
        return envelope
    rising = active & (position < ramp)
    falling = active & (position > active_seconds - ramp)
    envelope[rising] = 0.5 - 0.5 * np.cos(np.pi * position[rising] / ramp)
    remaining = active_seconds - position[falling]
    envelope[falling] = 0.5 - 0.5 * np.cos(np.pi * remaining / ramp)
    return envelope


def ordered_spatial_bands(
    pixel_xy: np.ndarray,
    target_mask: np.ndarray,
    *,
    band_count: int,
    axis: BandAxis = "y",
) -> np.ndarray:
    """Assign target pixels to balanced deterministic spatial bands.

    Sorting uses the requested anatomical coordinate and the orthogonal
    coordinate as a deterministic tie break. Every nonempty band contains a
    contiguous section of that ordering.
    """

    coordinates = np.asarray(pixel_xy, dtype=np.float64)
    selected = np.asarray(target_mask, dtype=bool)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("pixel_xy must have shape (pixel_count, 2)")
    if selected.shape != (coordinates.shape[0],):
        raise ValueError("target_mask does not match pixel_xy")
    if not np.isfinite(coordinates).all():
        raise ValueError("pixel_xy must be finite")
    count = int(band_count)
    if count < 2 or count > int(np.count_nonzero(selected)):
        raise ValueError("band_count must be at least two and no larger than target size")
    if axis not in {"x", "y"}:
        raise ValueError("axis must be x or y")
    primary = 0 if axis == "x" else 1
    secondary = 1 - primary
    indices = np.flatnonzero(selected)
    order = np.lexsort((coordinates[indices, secondary], coordinates[indices, primary]))
    ordered = indices[order]
    groups = np.array_split(ordered, count)
    labels = np.full(coordinates.shape[0], -1, dtype=np.int16)
    for band_index, group in enumerate(groups):
        labels[group] = int(band_index)
    return labels


def injection_phase_map(
    pixel_xy: np.ndarray,
    target_mask: np.ndarray,
    spec: PhotometryInjectionSpec,
    *,
    upper_mask: np.ndarray | None = None,
    lower_mask: np.ndarray | None = None,
    band_axis: BandAxis = "y",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return per-pixel signal phase and optional ordered-band labels."""

    resolved = spec.validated()
    coordinates = np.asarray(pixel_xy, dtype=np.float64)
    target = np.asarray(target_mask, dtype=bool)
    if coordinates.shape != (target.size, 2):
        raise ValueError("pixel_xy and target_mask shapes are inconsistent")
    if not np.any(target):
        raise ValueError("target_mask cannot be empty")
    phases = np.full(target.shape, np.nan, dtype=np.float64)
    labels = np.full(target.shape, -1, dtype=np.int16)
    initial = float(resolved.initial_phase_rad)

    if resolved.pattern == "synchronous":
        phases[target] = initial
        return phases, labels, 0

    if resolved.pattern in {"opposite_upper_lower", "fixed_upper_lower_delay"}:
        if upper_mask is None or lower_mask is None:
            raise ValueError(f"{resolved.pattern} requires upper_mask and lower_mask")
        upper = np.asarray(upper_mask, dtype=bool)
        lower = np.asarray(lower_mask, dtype=bool)
        if upper.shape != target.shape or lower.shape != target.shape:
            raise ValueError("upper/lower mask shape does not match target")
        upper &= target
        lower &= target
        if np.any(upper & lower) or not np.array_equal(upper | lower, target):
            raise ValueError("upper and lower masks must disjointly partition target_mask")
        phases[upper] = initial
        delay = (
            np.pi
            if resolved.pattern == "opposite_upper_lower"
            else float(resolved.upper_lower_delay_rad)
        )
        phases[lower] = initial + delay
        labels[upper] = 0
        labels[lower] = 1
        return phases, labels, 2

    count = int(resolved.traveling_band_count)
    labels = ordered_spatial_bands(
        coordinates,
        target,
        band_count=count,
        axis=band_axis,
    )
    step = (
        int(resolved.traveling_direction)
        * float(resolved.traveling_phase_span_rad)
        / float(count - 1)
    )
    phases[target] = initial + step * labels[target]
    return phases, labels, count


def _regional_phase_summary(
    phases: np.ndarray,
    region: np.ndarray,
) -> tuple[float, float]:
    selected = np.asarray(region, dtype=bool) & np.isfinite(phases)
    if not np.any(selected):
        return math.nan, math.nan
    vector = np.mean(np.exp(1j * np.asarray(phases)[selected]))
    resultant = float(abs(vector))
    phase = float(np.angle(vector)) if resultant > np.finfo(float).eps else math.nan
    return phase, resultant


def inject_mono8_photometry(
    dataset: LocalCoordinateDataset,
    target_mask: np.ndarray,
    spec: PhotometryInjectionSpec,
    *,
    upper_mask: np.ndarray | None = None,
    lower_mask: np.ndarray | None = None,
    band_axis: BandAxis = "y",
    time_origin_s: float | None = None,
) -> tuple[LocalCoordinateDataset, PhotometryInjectionTruth]:
    """Add a known cache-domain oscillator to frozen target pixels.

    Only ``traces`` and injection provenance in ``metadata`` are changed. This
    deliberately leaves nuisance measurements, source coordinates, motion
    predictions, and transform uncertainty exactly as observed in the cache.
    The result tests conditional transform/frequency recoverability; it does not
    rerun spatial localization or create an event-level validation dataset.
    """

    dataset.validated()
    resolved = spec.validated()
    target = np.asarray(target_mask, dtype=bool)
    if target.shape != (dataset.pixel_count,) or not np.any(target):
        raise ValueError("target_mask must select cached pixels")
    origin = (
        float(np.asarray(dataset.timestamps_s)[0])
        if time_origin_s is None
        else float(time_origin_s)
    )
    phases, labels, band_count = injection_phase_map(
        dataset.pixel_xy,
        target,
        resolved,
        upper_mask=upper_mask,
        lower_mask=lower_mask,
        band_axis=band_axis,
    )
    envelope = activity_envelope(dataset.timestamps_s, resolved, time_origin_s=origin)
    relative_time = np.asarray(dataset.timestamps_s, dtype=np.float64) - origin
    phase_matrix = (
        2.0 * np.pi * float(resolved.frequency_hz) * relative_time[:, None]
        + phases[target][None, :]
    )
    delta = (
        float(resolved.amplitude_dn)
        * envelope[:, None]
        * np.sin(phase_matrix)
    )
    original = np.asarray(dataset.traces, dtype=np.float64)
    injected = original.copy()
    selected_values = injected[:, target]
    usable = (
        np.asarray(dataset.pixel_valid, dtype=bool)[:, target]
        & np.asarray(dataset.frame_valid, dtype=bool)[:, None]
        & np.isfinite(selected_values)
    )
    proposed = selected_values + delta
    clipped = usable & (
        (proposed < float(resolved.clip_min_dn))
        | (proposed > float(resolved.clip_max_dn))
    )
    selected_values[usable] = np.clip(
        proposed[usable],
        float(resolved.clip_min_dn),
        float(resolved.clip_max_dn),
    )
    injected[:, target] = selected_values
    injected[~np.asarray(dataset.pixel_valid, dtype=bool)] = np.nan

    target_phase, target_resultant = _regional_phase_summary(phases, target)
    upper_phase = (
        _regional_phase_summary(phases, np.asarray(upper_mask, dtype=bool))[0]
        if upper_mask is not None
        else math.nan
    )
    lower_phase = (
        _regional_phase_summary(phases, np.asarray(lower_mask, dtype=bool))[0]
        if lower_mask is not None
        else math.nan
    )
    upper_lower = (
        float(circular_difference_rad(lower_phase, upper_phase))
        if np.isfinite(upper_phase) and np.isfinite(lower_phase)
        else math.nan
    )
    slope = (
        int(resolved.traveling_direction)
        * float(resolved.traveling_phase_span_rad)
        / float(band_count - 1)
        if resolved.pattern == "traveling_wave"
        else math.nan
    )
    provenance = {
        "pattern": resolved.pattern,
        "frequency_hz": float(resolved.frequency_hz),
        "amplitude_dn": float(resolved.amplitude_dn),
        "activity_mode": resolved.activity_mode,
        "time_origin_s": origin,
        "band_axis": band_axis,
        "target_pixel_count": int(np.count_nonzero(target)),
        "target_phase_resultant": target_resultant,
        "clipped_sample_count": int(np.count_nonzero(clipped)),
        "nuisance_and_tracking_arrays_preserved": True,
    }
    output = replace(
        dataset,
        traces=injected,
        metadata={
            **dict(dataset.metadata),
            "synthetic_photometry_injection": provenance,
        },
    ).validated()
    truth = PhotometryInjectionTruth(
        spec=resolved,
        target_mask=target.copy(),
        phase_by_pixel_rad=phases,
        band_index_by_pixel=labels,
        band_count=band_count,
        activity_envelope=envelope,
        injected_delta_dn=delta,
        expected_target_phase_rad=target_phase,
        target_phase_resultant=target_resultant,
        expected_upper_lower_phase_rad=upper_lower,
        expected_band_phase_slope_rad=slope,
        clipped_sample_count=int(np.count_nonzero(clipped)),
        time_origin_s=origin,
    )
    return output, truth


def estimate_monotonic_phase_slope(
    phase_by_band_rad: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """Estimate an unwrapped phase slope across three to five ordered bands."""

    phases = np.asarray(phase_by_band_rad, dtype=np.float64)
    if phases.ndim != 1 or phases.size < 3 or phases.size > 5:
        raise ValueError("phase_by_band_rad must contain three to five bands")
    finite = np.isfinite(phases)
    if weights is None:
        resolved_weights = np.ones(phases.shape, dtype=np.float64)
    else:
        resolved_weights = np.asarray(weights, dtype=np.float64)
        if resolved_weights.shape != phases.shape:
            raise ValueError("weights shape must match phase_by_band_rad")
        finite &= np.isfinite(resolved_weights) & (resolved_weights > 0.0)
    if int(np.count_nonzero(finite)) < 3:
        return math.nan
    x = np.arange(phases.size, dtype=np.float64)[finite]
    y = np.unwrap(phases[finite])
    w = resolved_weights[finite]
    x_center = float(np.average(x, weights=w))
    y_center = float(np.average(y, weights=w))
    denominator = float(np.sum(w * np.square(x - x_center)))
    if denominator <= np.finfo(float).eps:
        return math.nan
    return float(np.sum(w * (x - x_center) * (y - y_center)) / denominator)


def select_discovery_family(
    spectral_ratio: np.ndarray,
    control_ratio: np.ndarray,
    candidate_names: Sequence[str],
    frequencies_hz: np.ndarray,
    discovery_windows: np.ndarray,
    *,
    minimum_windows: int = 3,
    minimum_spectral_ratio: float = 1.5,
    minimum_control_ratio: float = 1.1,
) -> FamilySelection:
    """Select a transform/frequency pair using discovery windows only."""

    spectral = np.asarray(spectral_ratio, dtype=np.float64)
    control = np.asarray(control_ratio, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    discovery = np.asarray(discovery_windows, dtype=bool)
    if spectral.ndim != 3 or control.shape != spectral.shape:
        raise ValueError("spectral_ratio and control_ratio must be candidate x window x frequency")
    if len(candidate_names) != spectral.shape[0]:
        raise ValueError("candidate_names does not match metric candidate axis")
    if frequencies.shape != (spectral.shape[2],):
        raise ValueError("frequencies_hz does not match metric frequency axis")
    if discovery.shape != (spectral.shape[1],):
        raise ValueError("discovery_windows does not match metric window axis")
    if int(minimum_windows) < 1:
        raise ValueError("minimum_windows must be positive")

    scores = np.full((spectral.shape[0], spectral.shape[2]), np.nan, dtype=np.float64)
    median_spectral = np.full(scores.shape, np.nan, dtype=np.float64)
    median_control = np.full(scores.shape, np.nan, dtype=np.float64)
    counts = np.zeros(scores.shape, dtype=np.int64)
    for candidate_index in range(spectral.shape[0]):
        for frequency_index in range(spectral.shape[2]):
            valid = (
                discovery
                & np.isfinite(spectral[candidate_index, :, frequency_index])
                & np.isfinite(control[candidate_index, :, frequency_index])
                & (spectral[candidate_index, :, frequency_index] > 0.0)
                & (control[candidate_index, :, frequency_index] > 0.0)
            )
            counts[candidate_index, frequency_index] = int(np.count_nonzero(valid))
            if not np.any(valid):
                continue
            median_spectral[candidate_index, frequency_index] = float(
                np.median(spectral[candidate_index, valid, frequency_index])
            )
            median_control[candidate_index, frequency_index] = float(
                np.median(control[candidate_index, valid, frequency_index])
            )
            scores[candidate_index, frequency_index] = float(
                np.log2(median_spectral[candidate_index, frequency_index])
                + np.log2(median_control[candidate_index, frequency_index])
            )
    passing = (
        (counts >= int(minimum_windows))
        & (median_spectral >= float(minimum_spectral_ratio))
        & (median_control >= float(minimum_control_ratio))
        & np.isfinite(scores)
    )
    if not np.any(passing):
        return FamilySelection(
            selected_candidate_index=None,
            selected_frequency_index=None,
            selected_candidate=None,
            selected_frequency_hz=math.nan,
            selection_score=math.nan,
            spectral_ratio=math.nan,
            control_ratio=math.nan,
            passing_pair_count=0,
            evaluated_pair_count=int(np.count_nonzero(np.isfinite(scores))),
        )
    ranked = np.where(passing, scores, -np.inf)
    flat = int(np.argmax(ranked))
    candidate_index, frequency_index = np.unravel_index(flat, ranked.shape)
    return FamilySelection(
        selected_candidate_index=int(candidate_index),
        selected_frequency_index=int(frequency_index),
        selected_candidate=str(candidate_names[candidate_index]),
        selected_frequency_hz=float(frequencies[frequency_index]),
        selection_score=float(scores[candidate_index, frequency_index]),
        spectral_ratio=float(median_spectral[candidate_index, frequency_index]),
        control_ratio=float(median_control[candidate_index, frequency_index]),
        passing_pair_count=int(np.count_nonzero(passing)),
        evaluated_pair_count=int(np.count_nonzero(np.isfinite(scores))),
    )


def stable_spec_id(payload: Mapping[str, Any], *, length: int = 16) -> str:
    """Return a stable, ordering-independent identifier for a job payload."""

    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[: int(length)]
