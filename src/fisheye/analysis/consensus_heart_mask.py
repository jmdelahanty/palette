from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from fisheye.analysis.local_rostral_heartrate import (
    DiscoveryCandidate,
    HeartrateConfig,
    LocalCoordinateDataset,
    RiskSurfaces,
    _analysis_chunks,
    _chunk_frequency_coefficients,
    _select_control_pixels,
    apply_nuisance_model,
    autocorrelation_preserving_surrogate,
    build_risk_surfaces,
    discover_candidate,
    fit_nuisance_model,
    trace_frequency_score,
)


@dataclass(frozen=True)
class ConsensusMaskConfig:
    outer_fold_count: int = 5
    outer_guard_seconds: float = 1.0
    min_selection_folds: int = 3
    min_confirmed_outer_folds: int = 3
    consensus_surrogate_count: int = 199
    heldout_surrogate_count: int = 39
    alpha: float = 0.05
    random_seed: int = 0

    def validated(self) -> ConsensusMaskConfig:
        if int(self.outer_fold_count) < 3:
            raise ValueError("outer_fold_count must be at least three")
        if float(self.outer_guard_seconds) < 0.0:
            raise ValueError("outer_guard_seconds cannot be negative")
        if not (1 <= int(self.min_selection_folds) <= int(self.outer_fold_count)):
            raise ValueError("min_selection_folds must be within the outer fold count")
        if not (1 <= int(self.min_confirmed_outer_folds) <= int(self.outer_fold_count)):
            raise ValueError("min_confirmed_outer_folds must be within the outer fold count")
        if int(self.consensus_surrogate_count) < 0 or int(self.heldout_surrogate_count) < 0:
            raise ValueError("surrogate counts cannot be negative")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be between zero and one")
        return self


@dataclass(frozen=True)
class OuterMaskFoldResult:
    fold_index: int
    discovery_interval_count: int
    confirmation_interval_s: tuple[float, float]
    discovery_frame_count: int
    confirmation_frame_count: int
    candidate: DiscoveryCandidate
    confirmation_score: float
    confirmation_p_value: float
    confirmation_null_scores: np.ndarray
    confirmation_chunk_count: int
    control_scores: Mapping[str, float]
    control_ratio: float
    confirmed: bool


@dataclass(frozen=True)
class ConsensusHeartMaskResult:
    detected: bool
    reason: str
    outer_folds: tuple[OuterMaskFoldResult, ...]
    selection_counts: np.ndarray
    selection_fractions: np.ndarray
    selection_p_values: np.ndarray
    consensus_pixels: np.ndarray
    consensus_mask: np.ndarray
    null_max_selection_counts: np.ndarray
    null_selection_count_threshold: float
    confirmed_outer_fold_count: int
    median_candidate_frequency_hz: float


def familywise_max_p_values(
    observed_by_hypothesis: Mapping[str, float],
    null_by_hypothesis: Mapping[str, np.ndarray],
    *,
    alpha: float,
) -> tuple[dict[str, float], float, dict[str, bool], np.ndarray]:
    """Correct prespecified hypotheses against their shared maximum null."""

    if not observed_by_hypothesis or set(observed_by_hypothesis) != set(null_by_hypothesis):
        raise ValueError("observed and null hypotheses must be non-empty and identical")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be between zero and one")
    arrays = [np.asarray(null_by_hypothesis[name], dtype=np.float64) for name in observed_by_hypothesis]
    sizes = {array.size for array in arrays}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise ValueError("all hypotheses must have the same non-zero null count")
    maximum_null = np.max(np.stack(arrays, axis=0), axis=0)
    threshold = float(np.quantile(maximum_null, 1.0 - float(alpha), method="higher"))
    p_values: dict[str, float] = {}
    exceeds: dict[str, bool] = {}
    for name, observed in observed_by_hypothesis.items():
        value = float(observed)
        p_values[name] = float(1 + np.count_nonzero(maximum_null >= value)) / float(
            maximum_null.size + 1
        )
        exceeds[name] = bool(p_values[name] <= float(alpha) and value > threshold)
    return p_values, threshold, exceeds, maximum_null


def contiguous_outer_partitions(
    dataset: LocalCoordinateDataset,
    config: ConsensusMaskConfig,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    dataset.validated()
    config.validated()
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    if timestamps.size < 2:
        raise ValueError("dataset must contain at least two timestamps")
    dt = float(np.median(np.diff(timestamps)))
    start = float(timestamps[0])
    stop = float(timestamps[-1] + dt)
    edges = np.linspace(start, stop, int(config.outer_fold_count) + 1)
    guard = float(config.outer_guard_seconds)
    partitions: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(int(config.outer_fold_count)):
        left = float(edges[fold_index])
        right = float(edges[fold_index + 1])
        confirmation_left = left + (guard if fold_index > 0 else 0.0)
        confirmation_right = right - (
            guard if fold_index + 1 < int(config.outer_fold_count) else 0.0
        )
        if confirmation_right <= confirmation_left:
            raise ValueError("outer guard removed an entire confirmation fold")
        confirmation = (timestamps >= confirmation_left) & (timestamps < confirmation_right)
        discovery = (timestamps < left - guard) | (timestamps >= right + guard)
        if fold_index == 0:
            discovery = timestamps >= right + guard
        elif fold_index + 1 == int(config.outer_fold_count):
            discovery = timestamps < left - guard
        if not np.any(discovery) or not np.any(confirmation):
            raise ValueError(f"outer fold {fold_index} has no discovery or confirmation rows")
        if np.any(discovery & confirmation):
            raise AssertionError("outer discovery and confirmation rows overlap")
        partitions.append((discovery, confirmation))
    return tuple(partitions)


def _candidate_pixel_mask(
    candidate: DiscoveryCandidate,
    pixel_count: int,
) -> np.ndarray:
    selected = np.zeros(int(pixel_count), dtype=bool)
    selected[np.asarray(candidate.pixel_indices, dtype=np.int64)] = True
    return selected


def _confirm_outer_candidate(
    dataset: LocalCoordinateDataset,
    risks: RiskSurfaces,
    analysis_config: HeartrateConfig,
    consensus_config: ConsensusMaskConfig,
    candidate: DiscoveryCandidate,
    nuisance_model,
    discovery_rows: np.ndarray,
    confirmation_rows: np.ndarray,
    *,
    fold_index: int,
) -> tuple[float, float, np.ndarray, int, Mapping[str, float], float, bool]:
    if candidate.pixel_indices.size == 0 or not np.isfinite(candidate.frequency_hz):
        return 0.0, 1.0, np.zeros(0), 0, {}, 0.0, False
    support = _candidate_pixel_mask(candidate, dataset.pixel_count)
    loading = _fit_phase_loading(
        dataset,
        nuisance_model,
        discovery_rows,
        support,
        candidate.frequency_hz,
        analysis_config,
    )
    observed, chunk_count = _score_frozen_phase_loading(
        dataset,
        nuisance_model,
        confirmation_rows,
        support,
        candidate.frequency_hz,
        loading,
        analysis_config,
    )
    rng = np.random.default_rng(
        int(consensus_config.random_seed) + 7919 * int(fold_index) + 101
    )
    null = np.zeros(int(consensus_config.heldout_surrogate_count), dtype=np.float64)
    for index in range(null.size):
        surrogate = autocorrelation_preserving_surrogate(
            dataset,
            confirmation_rows,
            rng=rng,
            spatial_block_px=int(analysis_config.surrogate_spatial_block_px),
            min_shift_seconds=float(analysis_config.surrogate_min_shift_seconds),
            max_gap_factor=float(analysis_config.max_timestamp_gap_factor),
        )
        null[index], _surrogate_chunk_count = _score_frozen_phase_loading(
            surrogate,
            nuisance_model,
            confirmation_rows,
            support,
            candidate.frequency_hz,
            loading,
            analysis_config,
        )
    p_value = (
        float(1 + np.count_nonzero(null >= observed)) / float(null.size + 1)
        if null.size
        else 1.0
    )
    controls = _outer_control_scores(
        dataset,
        nuisance_model,
        discovery_rows,
        confirmation_rows,
        candidate,
        risks,
        analysis_config,
    )
    finite_controls = np.asarray(
        [value for value in controls.values() if np.isfinite(value)],
        dtype=np.float64,
    )
    strongest_control = float(np.max(finite_controls)) if finite_controls.size else 0.0
    control_ratio = float(observed / max(strongest_control, np.finfo(float).eps))
    threshold = (
        float(np.quantile(null, 1.0 - float(consensus_config.alpha), method="higher"))
        if null.size
        else math.inf
    )
    confirmed = bool(
        null.size
        and p_value <= float(consensus_config.alpha)
        and observed > threshold
        and control_ratio >= float(analysis_config.min_control_ratio)
    )
    return observed, p_value, null, chunk_count, controls, control_ratio, confirmed


def _fit_phase_loading(
    dataset: LocalCoordinateDataset,
    nuisance_model,
    discovery_rows: np.ndarray,
    support: np.ndarray,
    frequency_hz: float,
    analysis_config: HeartrateConfig,
) -> np.ndarray:
    residual = apply_nuisance_model(dataset, nuisance_model)
    chunks = _analysis_chunks(dataset, discovery_rows, support, analysis_config)
    if not chunks:
        return np.full(dataset.pixel_count, np.nan + 0j, dtype=np.complex128)
    coefficients = _chunk_frequency_coefficients(
        residual,
        dataset.timestamps_s,
        chunks,
        np.asarray([frequency_hz], dtype=np.float64),
        min_valid_fraction=float(analysis_config.min_chunk_valid_fraction),
        max_interpolated_gap_seconds=float(analysis_config.max_interpolated_gap_seconds),
    )[0]
    finite = np.isfinite(coefficients)
    numerator = np.sum(np.where(finite, coefficients, 0.0), axis=0)
    denominator = np.sum(finite, axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.full(dataset.pixel_count, np.nan + 0j, dtype=np.complex128),
        where=denominator > 0,
    )


def _score_frozen_phase_loading(
    dataset: LocalCoordinateDataset,
    nuisance_model,
    confirmation_rows: np.ndarray,
    support: np.ndarray,
    frequency_hz: float,
    loading: np.ndarray,
    analysis_config: HeartrateConfig,
) -> tuple[float, int]:
    residual = apply_nuisance_model(dataset, nuisance_model)
    chunks = _analysis_chunks(dataset, confirmation_rows, support, analysis_config)
    if not chunks:
        return 0.0, 0
    coefficients = _chunk_frequency_coefficients(
        residual,
        dataset.timestamps_s,
        chunks,
        np.asarray([frequency_hz], dtype=np.float64),
        min_valid_fraction=float(analysis_config.min_chunk_valid_fraction),
        max_interpolated_gap_seconds=float(analysis_config.max_interpolated_gap_seconds),
    )[0]
    amplitude = np.abs(np.asarray(loading, dtype=np.complex128))
    usable = np.asarray(support, dtype=bool) & np.isfinite(loading) & (
        amplitude > np.finfo(float).eps
    )
    if int(np.count_nonzero(usable)) < 3:
        return 0.0, len(chunks)
    cap = float(np.quantile(amplitude[usable], 0.9))
    weights = np.minimum(amplitude, cap)
    phases = np.divide(
        loading,
        amplitude,
        out=np.full(dataset.pixel_count, np.nan + 0j),
        where=amplitude > np.finfo(float).eps,
    )
    matched = np.full(len(chunks), np.nan + 0j, dtype=np.complex128)
    alignment = np.full(len(chunks), np.nan, dtype=np.float64)
    for chunk_index, values in enumerate(coefficients):
        valid = usable & np.isfinite(values)
        denominator = float(np.sum(weights[valid]))
        if denominator <= np.finfo(float).eps:
            continue
        aligned = np.conjugate(phases[valid]) * values[valid]
        matched[chunk_index] = np.sum(weights[valid] * aligned) / denominator
        absolute_denominator = float(np.sum(weights[valid] * np.abs(values[valid])))
        if absolute_denominator > np.finfo(float).eps:
            alignment[chunk_index] = float(
                np.abs(np.sum(weights[valid] * aligned)) / absolute_denominator
            )
    finite = np.isfinite(matched) & np.isfinite(alignment)
    if not np.any(finite):
        return 0.0, len(chunks)
    score = (
        float(np.nanmedian(np.abs(matched)))
        * float(np.nanmedian(alignment))
        * float(np.mean(finite))
    )
    return score, len(chunks)


def _outer_control_scores(
    dataset: LocalCoordinateDataset,
    nuisance_model,
    discovery_rows: np.ndarray,
    confirmation_rows: np.ndarray,
    candidate: DiscoveryCandidate,
    risks: RiskSurfaces,
    analysis_config: HeartrateConfig,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for label, pixels in _select_control_pixels(dataset, risks, candidate).items():
        support = np.zeros(dataset.pixel_count, dtype=bool)
        support[np.asarray(pixels, dtype=np.int64)] = True
        if int(np.count_nonzero(support)) < 3:
            output[label] = math.nan
            continue
        loading = _fit_phase_loading(
            dataset,
            nuisance_model,
            discovery_rows,
            support,
            candidate.frequency_hz,
            analysis_config,
        )
        output[label], _chunk_count = _score_frozen_phase_loading(
            dataset,
            nuisance_model,
            confirmation_rows,
            support,
            candidate.frequency_hz,
            loading,
            analysis_config,
        )
    for label in ("global_mean", "body_control_mean", "external_control_mean"):
        if label not in dataset.nuisance_names:
            output[label] = math.nan
            continue
        column = dataset.nuisance_names.index(label)
        output[label] = trace_frequency_score(
            dataset,
            np.asarray(dataset.nuisance_values, dtype=np.float64)[:, column],
            confirmation_rows,
            candidate.frequency_hz,
            analysis_config,
        )
    return output


def _discover_outer_candidates(
    dataset: LocalCoordinateDataset,
    risks: RiskSurfaces,
    analysis_config: HeartrateConfig,
    partitions: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[tuple[DiscoveryCandidate, object], ...]:
    discovered: list[tuple[DiscoveryCandidate, object]] = []
    for discovery_rows, _confirmation_rows in partitions:
        model = fit_nuisance_model(
            dataset,
            discovery_rows,
            ridge=float(analysis_config.nuisance_ridge),
        )
        residual = apply_nuisance_model(dataset, model)
        candidate = discover_candidate(
            dataset,
            residual,
            discovery_rows,
            risks,
            analysis_config,
        )
        discovered.append((candidate, model))
    return tuple(discovered)


def _selection_counts(
    candidates: tuple[tuple[DiscoveryCandidate, object], ...],
    pixel_count: int,
) -> np.ndarray:
    counts = np.zeros(int(pixel_count), dtype=np.int16)
    for candidate, _model in candidates:
        counts += _candidate_pixel_mask(candidate, pixel_count).astype(np.int16)
    return counts


def _consensus_null(
    dataset: LocalCoordinateDataset,
    risks: RiskSurfaces,
    consensus_config: ConsensusMaskConfig,
    discovered: tuple[tuple[DiscoveryCandidate, object], ...],
) -> np.ndarray:
    """Calibrate repeated spatial selection conditional on discovered cluster shapes.

    Temporal circular shifts preserve narrow-band power at each pixel and are
    therefore unsuitable for a null whose target is recurrence at the same
    location. This null independently translates each observed cluster within
    the physically eligible pixel grid, preserving its shape and size.
    """

    count = int(consensus_config.consensus_surrogate_count)
    maxima = np.zeros(count, dtype=np.int16)
    if count == 0:
        return maxima
    rng = np.random.default_rng(int(consensus_config.random_seed) + 4241)
    for index in range(count):
        selections = np.zeros(dataset.pixel_count, dtype=np.int16)
        for candidate, _model in discovered:
            translated = _random_eligible_translation(
                dataset,
                risks.eligible,
                _candidate_pixel_mask(candidate, dataset.pixel_count),
                rng,
            )
            selections += translated.astype(np.int16)
        maxima[index] = int(np.max(selections)) if selections.size else 0
    return maxima


def _random_eligible_translation(
    dataset: LocalCoordinateDataset,
    eligible: np.ndarray,
    selected: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    chosen = np.flatnonzero(np.asarray(selected, dtype=bool))
    output = np.zeros(dataset.pixel_count, dtype=bool)
    if chosen.size == 0:
        return output
    xy = np.rint(np.asarray(dataset.pixel_xy, dtype=np.float64)).astype(np.int64)
    eligible_indices = np.flatnonzero(np.asarray(eligible, dtype=bool))
    coordinate_to_pixel = {
        (int(xy[pixel, 0]), int(xy[pixel, 1])): int(pixel)
        for pixel in eligible_indices.tolist()
    }
    source = xy[chosen]
    eligible_xy = xy[eligible_indices]
    dx_min = int(np.min(eligible_xy[:, 0]) - np.min(source[:, 0]))
    dx_max = int(np.max(eligible_xy[:, 0]) - np.max(source[:, 0]))
    dy_min = int(np.min(eligible_xy[:, 1]) - np.min(source[:, 1]))
    dy_max = int(np.max(eligible_xy[:, 1]) - np.max(source[:, 1]))
    placements: list[np.ndarray] = []
    for dy in range(dy_min, dy_max + 1):
        for dx in range(dx_min, dx_max + 1):
            translated_pixels: list[int] = []
            for x, y in source.tolist():
                pixel = coordinate_to_pixel.get((int(x + dx), int(y + dy)))
                if pixel is None:
                    translated_pixels = []
                    break
                translated_pixels.append(pixel)
            if len(translated_pixels) == chosen.size:
                placements.append(np.asarray(translated_pixels, dtype=np.int64))
    if placements:
        placement = placements[int(rng.integers(0, len(placements)))]
        output[placement] = True
        return output
    if eligible_indices.size >= chosen.size:
        output[rng.choice(eligible_indices, size=chosen.size, replace=False)] = True
    return output


def _scatter_consensus_mask(
    dataset: LocalCoordinateDataset,
    selected: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(dataset.image_shape_hw, dtype=bool)
    xy = np.rint(np.asarray(dataset.pixel_xy, dtype=np.float64)).astype(np.int64)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < dataset.image_shape_hw[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < dataset.image_shape_hw[0])
    )
    rows = np.flatnonzero(np.asarray(selected, dtype=bool) & inside)
    mask[xy[rows, 1], xy[rows, 0]] = True
    return mask


def learn_consensus_heart_mask(
    dataset: LocalCoordinateDataset,
    analysis_config: HeartrateConfig,
    consensus_config: ConsensusMaskConfig,
) -> ConsensusHeartMaskResult:
    dataset.validated()
    analysis_config.validated()
    consensus_config.validated()
    risks = build_risk_surfaces(dataset, analysis_config)
    partitions = contiguous_outer_partitions(dataset, consensus_config)
    discovered = _discover_outer_candidates(
        dataset,
        risks,
        analysis_config,
        partitions,
    )
    folds: list[OuterMaskFoldResult] = []
    for fold_index, ((discovery_rows, confirmation_rows), (candidate, model)) in enumerate(
        zip(partitions, discovered)
    ):
        observed, p_value, null, chunk_count, controls, ratio, confirmed = (
            _confirm_outer_candidate(
                dataset,
                risks,
                analysis_config,
                consensus_config,
                candidate,
                model,
                discovery_rows,
                confirmation_rows,
                fold_index=fold_index,
            )
        )
        times = np.asarray(dataset.timestamps_s, dtype=np.float64)[confirmation_rows]
        folds.append(
            OuterMaskFoldResult(
                fold_index=int(fold_index),
                discovery_interval_count=1 if fold_index in {0, len(partitions) - 1} else 2,
                confirmation_interval_s=(float(times[0]), float(times[-1])),
                discovery_frame_count=int(np.count_nonzero(discovery_rows)),
                confirmation_frame_count=int(np.count_nonzero(confirmation_rows)),
                candidate=candidate,
                confirmation_score=float(observed),
                confirmation_p_value=float(p_value),
                confirmation_null_scores=null,
                confirmation_chunk_count=int(chunk_count),
                control_scores=controls,
                control_ratio=float(ratio),
                confirmed=bool(confirmed),
            )
        )
    selection_counts = _selection_counts(discovered, dataset.pixel_count)
    null_max = _consensus_null(
        dataset,
        risks,
        consensus_config,
        discovered,
    )
    p_values = np.ones(dataset.pixel_count, dtype=np.float64)
    if null_max.size:
        for pixel in range(dataset.pixel_count):
            p_values[pixel] = float(1 + np.count_nonzero(null_max >= selection_counts[pixel])) / float(
                null_max.size + 1
            )
    selected = (
        risks.eligible
        & (selection_counts >= int(consensus_config.min_selection_folds))
        & (p_values <= float(consensus_config.alpha))
    )
    confirmed_count = int(sum(fold.confirmed for fold in folds))
    enough_pixels = int(np.count_nonzero(selected)) >= int(analysis_config.min_cluster_pixels)
    enough_confirmed = confirmed_count >= int(consensus_config.min_confirmed_outer_folds)
    detected = bool(null_max.size and enough_pixels and enough_confirmed)
    if not null_max.size:
        reason = "consensus_null_not_run"
    elif not enough_pixels:
        reason = "no_pixels_exceed_max_stability_null"
    elif not enough_confirmed:
        reason = "too_few_outer_folds_confirmed"
    else:
        reason = "consensus_mask_detected"
    frequencies = np.asarray(
        [fold.candidate.frequency_hz for fold in folds if np.isfinite(fold.candidate.frequency_hz)],
        dtype=np.float64,
    )
    threshold = (
        float(np.quantile(null_max, 1.0 - float(consensus_config.alpha), method="higher"))
        if null_max.size
        else math.inf
    )
    return ConsensusHeartMaskResult(
        detected=detected,
        reason=reason,
        outer_folds=tuple(folds),
        selection_counts=selection_counts,
        selection_fractions=selection_counts.astype(np.float64) / float(len(folds)),
        selection_p_values=p_values,
        consensus_pixels=selected,
        consensus_mask=_scatter_consensus_mask(dataset, selected),
        null_max_selection_counts=null_max,
        null_selection_count_threshold=threshold,
        confirmed_outer_fold_count=confirmed_count,
        median_candidate_frequency_hz=(float(np.median(frequencies)) if frequencies.size else math.nan),
    )
