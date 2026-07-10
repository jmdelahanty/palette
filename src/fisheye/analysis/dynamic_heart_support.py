from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.local_rostral_heartrate import (
    HeartrateConfig,
    HeartrateResult,
    LocalCoordinateDataset,
    NuisanceModel,
    _analysis_chunks,
    _chunk_frequency_coefficients,
    _frequency_grid,
    apply_nuisance_model,
    autocorrelation_preserving_surrogate,
    balanced_valid_partitions,
    bridge_short_gaps,
    build_risk_surfaces,
    fit_nuisance_model,
)


@dataclass(frozen=True)
class DynamicHeartSupportResult:
    """Cross-fitted evidence for a spatially varying anatomical heart support."""

    support_source: str
    frequency_search_source: str
    confirmatory_eligible: bool
    interpretation: str
    phase_pattern: str
    frequency_hz: float
    support_score: float
    support_p_value: float
    support_exceeds_null: bool
    shared_phase_score: float
    shared_phase_p_value: float
    shared_phase_exceeds_null: bool
    joint_p_value: float
    joint_exceeds_null: bool
    latent_score: float
    latent_p_value: float
    latent_exceeds_null: bool
    union_to_core_score_ratio: float
    strongest_control: str | None
    control_ratio: float
    control_scores: Mapping[str, float]
    pixel_groups: Mapping[str, np.ndarray]
    group_summary: Mapping[str, Mapping[str, Any]]
    block_summary: tuple[Mapping[str, Any], ...]
    block_rows: tuple[np.ndarray, ...]
    block_model_fold_indices: np.ndarray
    block_coefficients: np.ndarray
    frequency_grid_hz: np.ndarray
    frequency_support_scores: np.ndarray
    frequency_shared_phase_scores: np.ndarray
    frequency_latent_scores: np.ndarray
    null_max_support_scores: np.ndarray
    null_max_shared_phase_scores: np.ndarray
    null_max_latent_scores: np.ndarray
    latent_block_coefficients: np.ndarray
    latent_block_alignment_coherence: np.ndarray


@dataclass(frozen=True)
class CrossfitHeartPhaseSeries:
    """Frame-resolved held-out phase for a frozen anatomical support."""

    frequency_hz: float
    band_min_hz: float
    band_max_hz: float
    heart_support: np.ndarray
    model_fold_indices: np.ndarray
    fold_loadings: np.ndarray
    fold_loading_weights: np.ndarray
    crossfit_residual: np.ndarray
    bandpassed_residual: np.ndarray
    analytic_residual: np.ndarray
    latent_analytic: np.ndarray
    spatial_alignment: np.ndarray
    frame_valid: np.ndarray


def _bandpass_analytic_chunks(
    dataset: LocalCoordinateDataset,
    residual: np.ndarray,
    chunks: Sequence[np.ndarray],
    selected: np.ndarray,
    *,
    band_min_hz: float,
    band_max_hz: float,
    edge_seconds: float,
    min_valid_fraction: float,
    max_interpolated_gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy import signal

    filtered = np.full_like(np.asarray(residual, dtype=np.float64), np.nan)
    analytic = np.full(filtered.shape, np.nan + 0j, dtype=np.complex128)
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    pixels = np.flatnonzero(np.asarray(selected, dtype=bool))
    for rows in chunks:
        if rows.size < 3:
            continue
        local_t = timestamps[rows]
        dt = float(np.median(np.diff(local_t)))
        nyquist = 0.5 / dt
        high = min(float(band_max_hz), nyquist * 0.98)
        if not (0.0 < float(band_min_hz) < high):
            continue
        grid = np.arange(local_t[0], local_t[-1] + 0.5 * dt, dt, dtype=np.float64)
        sos = signal.butter(
            3,
            [float(band_min_hz), high],
            btype="bandpass",
            fs=1.0 / dt,
            output="sos",
        )
        keep = (local_t >= local_t[0] + float(edge_seconds)) & (
            local_t <= local_t[-1] - float(edge_seconds)
        )
        if not np.any(keep):
            continue
        for pixel in pixels:
            values = np.asarray(residual, dtype=np.float64)[rows, pixel]
            finite = np.isfinite(values)
            if float(np.mean(finite)) < float(min_valid_fraction):
                continue
            bridged, _interpolated = bridge_short_gaps(
                local_t,
                finite,
                max_gap_seconds=float(max_interpolated_gap_seconds),
            )
            if not np.all(bridged) or int(np.count_nonzero(finite)) < 3:
                continue
            filled = values.copy()
            if not np.all(finite):
                filled[~finite] = np.interp(
                    local_t[~finite],
                    local_t[finite],
                    filled[finite],
                )
            sampled = np.interp(grid, local_t, filled)
            sampled = signal.detrend(sampled, type="linear")
            try:
                filtered_grid = signal.sosfiltfilt(sos, sampled)
            except ValueError:
                continue
            analytic_grid = signal.hilbert(filtered_grid)
            filtered_local = np.interp(local_t, grid, filtered_grid)
            analytic_local = np.interp(local_t, grid, analytic_grid.real) + 1j * np.interp(
                local_t,
                grid,
                analytic_grid.imag,
            )
            output_rows = rows[keep]
            filtered[output_rows, pixel] = filtered_local[keep]
            analytic[output_rows, pixel] = analytic_local[keep]
    return filtered, analytic


def reconstruct_crossfit_heart_phase(
    dataset: LocalCoordinateDataset,
    config: HeartrateConfig,
    result: HeartrateResult,
    dynamic_result: DynamicHeartSupportResult,
    *,
    edge_seconds: float | None = None,
) -> CrossfitHeartPhaseSeries:
    """Reconstruct held-out pixel phase without training on displayed rows.

    The selected frequency is descriptive: it was chosen by the dynamic-support
    search. Within that frequency, nuisance models and pixel loadings are always
    learned on the opposite cross-fit partition from the rows they visualize.
    """

    dataset.validated()
    config.validated()
    risks = build_risk_surfaces(dataset, config)
    partitions = balanced_valid_partitions(dataset, risks, config)
    support = np.asarray(dynamic_result.pixel_groups["heart_support"], dtype=bool)
    if support.shape != (dataset.pixel_count,) or int(np.count_nonzero(support)) < 3:
        raise ValueError("heart support must contain at least three dataset pixels")
    frequency = float(dynamic_result.frequency_hz)
    frequency_grid = np.asarray(dynamic_result.frequency_grid_hz, dtype=np.float64)
    band_min = float(np.min(frequency_grid))
    band_max = float(np.max(frequency_grid))
    if not (band_min < frequency < band_max):
        half_width = max(float(config.frequency_step_hz), 0.25)
        band_min = max(float(config.band_min_hz), frequency - half_width)
        band_max = min(float(config.band_max_hz), frequency + half_width)
    edge = float(config.event_filter_edge_seconds if edge_seconds is None else edge_seconds)
    if edge < 0.0:
        raise ValueError("edge_seconds cannot be negative")

    fold_count = len(partitions)
    fold_loadings = np.full(
        (fold_count, dataset.pixel_count),
        np.nan + 0j,
        dtype=np.complex128,
    )
    fold_weights = np.zeros((fold_count, dataset.pixel_count), dtype=np.float64)
    crossfit_residual = np.full_like(np.asarray(dataset.traces, dtype=np.float64), np.nan)
    filtered = np.full_like(crossfit_residual, np.nan)
    analytic = np.full(crossfit_residual.shape, np.nan + 0j, dtype=np.complex128)
    model_fold = np.full(dataset.frame_count, -1, dtype=np.int16)

    for fold, (discovery_rows, confirmation_rows) in zip(result.folds, partitions):
        fold_index = int(fold.fold_index)
        if fold_index < 0 or fold_index >= fold_count:
            raise ValueError(f"unexpected fold index {fold_index}")
        model = fold.discovery.nuisance_model
        residual = apply_nuisance_model(dataset, model)
        discovery_chunks = _analysis_chunks(dataset, discovery_rows, support, config)
        confirmation_chunks = _analysis_chunks(dataset, confirmation_rows, support, config)
        if not discovery_chunks or not confirmation_chunks:
            continue
        coefficients = _chunk_frequency_coefficients(
            residual,
            dataset.timestamps_s,
            discovery_chunks,
            np.asarray([frequency], dtype=np.float64),
            min_valid_fraction=float(config.min_chunk_valid_fraction),
            max_interpolated_gap_seconds=float(config.max_interpolated_gap_seconds),
        )
        loading = _finite_complex_mean(coefficients, axis=1)[0]
        amplitude = np.abs(loading)
        usable = support & np.isfinite(loading) & (amplitude > np.finfo(float).eps)
        if int(np.count_nonzero(usable)) < 3:
            continue
        cap = float(np.quantile(amplitude[usable], 0.9))
        weights = np.minimum(amplitude, cap)
        weights[~usable] = 0.0
        if cap > np.finfo(float).eps:
            weights /= cap
        fold_loadings[fold_index] = loading
        fold_weights[fold_index] = weights
        rows = np.asarray(confirmation_rows, dtype=bool)
        crossfit_residual[rows] = residual[rows]
        model_fold[rows] = fold_index
        fold_filtered, fold_analytic = _bandpass_analytic_chunks(
            dataset,
            residual,
            confirmation_chunks,
            support,
            band_min_hz=band_min,
            band_max_hz=band_max,
            edge_seconds=edge,
            min_valid_fraction=float(config.min_chunk_valid_fraction),
            max_interpolated_gap_seconds=float(config.max_interpolated_gap_seconds),
        )
        finite = np.isfinite(fold_analytic)
        filtered[finite] = fold_filtered[finite]
        analytic[finite] = fold_analytic[finite]

    latent = np.full(dataset.frame_count, np.nan + 0j, dtype=np.complex128)
    alignment = np.full(dataset.frame_count, np.nan, dtype=np.float64)
    for row in range(dataset.frame_count):
        fold_index = int(model_fold[row])
        if fold_index < 0:
            continue
        loading = fold_loadings[fold_index]
        amplitude = np.abs(loading)
        phase = np.divide(
            loading,
            amplitude,
            out=np.full(dataset.pixel_count, np.nan + 0j),
            where=amplitude > np.finfo(float).eps,
        )
        values = analytic[row]
        weights = fold_weights[fold_index]
        valid = support & np.isfinite(values) & np.isfinite(phase) & (weights > 0.0)
        denominator = float(np.sum(weights[valid]))
        if int(np.count_nonzero(valid)) < 3 or denominator <= np.finfo(float).eps:
            continue
        aligned = np.conjugate(phase[valid]) * values[valid]
        latent[row] = np.sum(weights[valid] * aligned) / denominator
        absolute_denominator = float(np.sum(weights[valid] * np.abs(values[valid])))
        if absolute_denominator > np.finfo(float).eps:
            alignment[row] = float(
                np.abs(np.sum(weights[valid] * aligned)) / absolute_denominator
            )
    frame_valid = np.isfinite(latent) & np.isfinite(alignment)
    return CrossfitHeartPhaseSeries(
        frequency_hz=frequency,
        band_min_hz=band_min,
        band_max_hz=band_max,
        heart_support=support,
        model_fold_indices=model_fold,
        fold_loadings=fold_loadings,
        fold_loading_weights=fold_weights,
        crossfit_residual=crossfit_residual,
        bandpassed_residual=filtered,
        analytic_residual=analytic,
        latent_analytic=latent,
        spatial_alignment=alignment,
        frame_valid=frame_valid,
    )


def _finite_complex_mean(values: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(values)
    finite = np.isfinite(arr)
    numerator = np.sum(np.where(finite, arr, 0), axis=axis)
    denominator = np.sum(finite, axis=axis)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan + 0j, dtype=np.complex128),
        where=denominator > 0,
    )


def _quiet_nanmedian(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.asarray(np.nanmedian(values, axis=axis))


def _mask_at_pixels(mask: np.ndarray, dataset: LocalCoordinateDataset) -> np.ndarray:
    image = np.asarray(mask, dtype=bool)
    if image.shape != tuple(dataset.image_shape_hw):
        raise ValueError(
            f"anatomical mask shape {image.shape} does not match {dataset.image_shape_hw}"
        )
    xy = np.rint(np.asarray(dataset.pixel_xy, dtype=np.float64)).astype(np.int64)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < image.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < image.shape[0])
    )
    selected = np.zeros(dataset.pixel_count, dtype=bool)
    selected[inside] = image[xy[inside, 1], xy[inside, 0]]
    return selected


def _candidate_pixels(candidate_mask: np.ndarray, dataset: LocalCoordinateDataset) -> np.ndarray:
    return _mask_at_pixels(np.asarray(candidate_mask, dtype=bool), dataset)


def _build_pixel_groups(
    dataset: LocalCoordinateDataset,
    result: HeartrateResult,
    eligible: np.ndarray,
    *,
    heart_mask: np.ndarray | None,
    esophagus_mask: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], str]:
    if len(result.folds) != 2:
        raise ValueError("dynamic support analysis requires two completed cross-fit folds")
    fold0 = _candidate_pixels(result.folds[0].discovery.candidate.cluster_mask, dataset)
    fold1 = _candidate_pixels(result.folds[1].discovery.candidate.cluster_mask, dataset)
    cluster_union = fold0 | fold1
    if heart_mask is None:
        if not np.any(cluster_union):
            raise ValueError("dynamic support analysis requires non-empty fold clusters")
        base_support = cluster_union.copy()
        support_source = "posthoc_crossfit_cluster_union"
    else:
        base_support = _mask_at_pixels(heart_mask, dataset)
        support_source = "external_anatomical_mask"
    esophagus = (
        _mask_at_pixels(esophagus_mask, dataset)
        if esophagus_mask is not None
        else np.zeros(dataset.pixel_count, dtype=bool)
    )
    support = base_support & np.asarray(eligible, dtype=bool) & ~esophagus
    if int(np.count_nonzero(support)) < 3:
        raise ValueError("anatomical heart support has fewer than three eligible pixels")
    groups = {
        "heart_support": support,
        "core": support & fold0 & fold1,
        "fold0_only": support & fold0 & ~fold1,
        "fold1_only": support & fold1 & ~fold0,
        "anatomical_only": support & ~cluster_union,
        "esophagus_control": esophagus,
    }
    return groups, support_source


def _crossfit_residual(
    dataset: LocalCoordinateDataset,
    result: HeartrateResult,
    partitions: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    refit_nuisance: bool = False,
    nuisance_ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[NuisanceModel, ...]]:
    residual = np.full_like(np.asarray(dataset.traces, dtype=np.float64), np.nan)
    model_fold = np.full(dataset.frame_count, -1, dtype=np.int16)
    active = np.zeros(dataset.frame_count, dtype=bool)
    models: list[NuisanceModel] = []
    for fold, (discovery_rows, confirmation_rows) in zip(result.folds, partitions):
        model = (
            fit_nuisance_model(
                dataset,
                discovery_rows,
                ridge=float(nuisance_ridge),
            )
            if refit_nuisance
            else fold.discovery.nuisance_model
        )
        fold_residual = apply_nuisance_model(dataset, model)
        models.append(model)
        rows = np.asarray(confirmation_rows, dtype=bool)
        residual[rows] = fold_residual[rows]
        model_fold[rows] = int(fold.fold_index)
        active |= rows
    return residual, model_fold, active, tuple(models)


def _crossfit_latent_scores(
    dataset: LocalCoordinateDataset,
    partitions: Sequence[tuple[np.ndarray, np.ndarray]],
    models: Sequence[NuisanceModel],
    support: np.ndarray,
    frequencies_hz: np.ndarray,
    config: HeartrateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Learn pixel phase/polarity on discovery and match it on held-out blocks."""

    matched_parts: list[np.ndarray] = []
    coherence_parts: list[np.ndarray] = []
    selected = np.asarray(support, dtype=bool)
    for model, (discovery_rows, confirmation_rows) in zip(models, partitions):
        residual = apply_nuisance_model(dataset, model)
        discovery_chunks = _analysis_chunks(
            dataset,
            discovery_rows,
            selected,
            config,
        )
        confirmation_chunks = _analysis_chunks(
            dataset,
            confirmation_rows,
            selected,
            config,
        )
        if not discovery_chunks or not confirmation_chunks:
            continue
        discovery_coefficients = _chunk_frequency_coefficients(
            residual,
            dataset.timestamps_s,
            discovery_chunks,
            frequencies_hz,
            min_valid_fraction=float(config.min_chunk_valid_fraction),
            max_interpolated_gap_seconds=float(config.max_interpolated_gap_seconds),
        )
        confirmation_coefficients = _chunk_frequency_coefficients(
            residual,
            dataset.timestamps_s,
            confirmation_chunks,
            frequencies_hz,
            min_valid_fraction=float(config.min_chunk_valid_fraction),
            max_interpolated_gap_seconds=float(config.max_interpolated_gap_seconds),
        )
        loading = _finite_complex_mean(discovery_coefficients, axis=1)
        matched = np.full(
            (len(frequencies_hz), len(confirmation_chunks)),
            np.nan + 0j,
            dtype=np.complex128,
        )
        alignment = np.full(matched.shape, np.nan, dtype=np.float64)
        for frequency_index in range(len(frequencies_hz)):
            amplitude = np.abs(loading[frequency_index])
            usable = selected & np.isfinite(loading[frequency_index]) & (
                amplitude > np.finfo(float).eps
            )
            if int(np.count_nonzero(usable)) < 3:
                continue
            cap = float(np.quantile(amplitude[usable], 0.9))
            weights = np.minimum(amplitude, cap)
            phase = np.divide(
                loading[frequency_index],
                amplitude,
                out=np.full(dataset.pixel_count, np.nan + 0j),
                where=amplitude > np.finfo(float).eps,
            )
            for block_index in range(len(confirmation_chunks)):
                values = confirmation_coefficients[
                    frequency_index,
                    block_index,
                ]
                valid = usable & np.isfinite(values)
                denominator = float(np.sum(weights[valid]))
                if denominator <= np.finfo(float).eps:
                    continue
                aligned = np.conjugate(phase[valid]) * values[valid]
                matched[frequency_index, block_index] = np.sum(
                    weights[valid] * aligned
                ) / denominator
                absolute_denominator = float(
                    np.sum(weights[valid] * np.abs(values[valid]))
                )
                if absolute_denominator > np.finfo(float).eps:
                    alignment[frequency_index, block_index] = float(
                        np.abs(np.sum(weights[valid] * aligned))
                        / absolute_denominator
                    )
        matched_parts.append(matched)
        coherence_parts.append(alignment)
    if not matched_parts:
        empty = np.full((len(frequencies_hz), 0), np.nan + 0j)
        return np.zeros(len(frequencies_hz)), empty, empty.real
    matched_all = np.concatenate(matched_parts, axis=1)
    coherence_all = np.concatenate(coherence_parts, axis=1)
    median_amplitude = _quiet_nanmedian(np.abs(matched_all), axis=1)
    median_alignment = _quiet_nanmedian(coherence_all, axis=1)
    valid_fraction = np.mean(
        np.isfinite(matched_all) & np.isfinite(coherence_all),
        axis=1,
    )
    scores = (
        median_amplitude
        * np.nan_to_num(median_alignment, nan=0.0)
        * valid_fraction
    )
    return scores, matched_all, coherence_all


def _aggregate_coefficients(
    coefficients: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    pixels = np.asarray(selected, dtype=bool)
    if not np.any(pixels):
        return np.full(coefficients.shape[:2], np.nan + 0j, dtype=np.complex128)
    return _finite_complex_mean(np.asarray(coefficients)[:, :, pixels], axis=2)


def _spatial_phase_coherence(
    coefficients: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    pixels = np.asarray(selected, dtype=bool)
    if not np.any(pixels):
        return np.full(coefficients.shape[:2], np.nan, dtype=np.float64)
    values = np.asarray(coefficients)[:, :, pixels]
    numerator = np.abs(np.nansum(values, axis=2))
    denominator = np.nansum(np.abs(values), axis=2)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > np.finfo(float).eps,
    )


def _temporal_coherence_score(
    aggregate: np.ndarray,
    spatial_coherence: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(aggregate, dtype=np.complex128)
    amplitude = np.abs(values)
    unit = np.divide(
        values,
        amplitude,
        out=np.full(values.shape, np.nan + 0j, dtype=np.complex128),
        where=amplitude > np.finfo(float).eps,
    )
    temporal = np.abs(_finite_complex_mean(unit, axis=1))
    median_amplitude = _quiet_nanmedian(amplitude, axis=1)
    if spatial_coherence is None:
        spatial = np.ones(median_amplitude.shape, dtype=np.float64)
    else:
        spatial = _quiet_nanmedian(np.asarray(spatial_coherence), axis=1)
    return (
        median_amplitude
        * (0.25 + 0.75 * np.nan_to_num(temporal, nan=0.0))
        * (0.25 + 0.75 * np.nan_to_num(spatial, nan=0.0))
    )


def _sideband_noise(
    aggregate: np.ndarray,
    frequencies_hz: np.ndarray,
    frequency_index: int,
) -> np.ndarray:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    sideband = np.abs(frequencies - float(frequencies[frequency_index])) >= 0.2
    if not np.any(sideband):
        sideband = np.arange(frequencies.size) != int(frequency_index)
    if not np.any(sideband):
        return np.full(aggregate.shape[1], np.nan, dtype=np.float64)
    return _quiet_nanmedian(np.abs(np.asarray(aggregate)[sideband]), axis=0)


def _phase_relation(
    aggregate: np.ndarray,
    reference: np.ndarray,
    noise: np.ndarray,
    reference_noise: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(aggregate, dtype=np.complex128)
    reference_values = np.asarray(reference, dtype=np.complex128)
    amplitude_ratio = np.divide(
        np.abs(values),
        noise,
        out=np.full(values.shape, np.nan, dtype=np.float64),
        where=np.asarray(noise) > np.finfo(float).eps,
    )
    reference_ratio = np.divide(
        np.abs(reference_values),
        reference_noise,
        out=np.full(reference_values.shape, np.nan, dtype=np.float64),
        where=np.asarray(reference_noise) > np.finfo(float).eps,
    )
    offsets = np.angle(values * np.conjugate(reference_values))
    strength = np.minimum(amplitude_ratio, reference_ratio)
    active = (
        np.isfinite(offsets)
        & np.isfinite(strength)
        & (strength >= 0.75)
    )
    if int(np.count_nonzero(active)) >= 2:
        weights = np.minimum(strength[active], 3.0)
        mean_vector = np.sum(weights * np.exp(1j * offsets[active])) / np.sum(weights)
        concentration = float(np.abs(mean_vector))
        mean_offset = float(np.degrees(np.angle(mean_vector)))
        median_abs_offset = float(np.degrees(np.median(np.abs(offsets[active]))))
    else:
        concentration = math.nan
        mean_offset = math.nan
        median_abs_offset = math.nan
    return {
        "offsets_rad": offsets,
        "target_to_sideband_ratio": amplitude_ratio,
        "active": active,
        "active_block_count": int(np.count_nonzero(active)),
        "phase_offset_concentration": concentration,
        "mean_phase_offset_deg": mean_offset,
        "median_abs_phase_offset_deg": median_abs_offset,
    }


def _shared_phase_frequency_scores(
    coefficients: np.ndarray,
    frequencies_hz: np.ndarray,
    groups: Mapping[str, np.ndarray],
    support_scores: np.ndarray,
) -> np.ndarray:
    reference_name = "core" if np.count_nonzero(groups["core"]) >= 3 else "heart_support"
    reference = _aggregate_coefficients(coefficients, groups[reference_name])
    comparison_names = [
        name
        for name in ("fold0_only", "fold1_only", "anatomical_only")
        if np.any(groups[name])
    ]
    if not comparison_names:
        return np.asarray(support_scores, dtype=np.float64).copy()
    output = np.zeros(len(frequencies_hz), dtype=np.float64)
    group_aggregates = {
        name: _aggregate_coefficients(coefficients, groups[name])
        for name in comparison_names
    }
    for frequency_index in range(len(frequencies_hz)):
        reference_noise = _sideband_noise(reference, frequencies_hz, frequency_index)
        concentrations: list[float] = []
        for name in comparison_names:
            aggregate = group_aggregates[name]
            relation = _phase_relation(
                aggregate[frequency_index],
                reference[frequency_index],
                _sideband_noise(aggregate, frequencies_hz, frequency_index),
                reference_noise,
            )
            concentration = float(relation["phase_offset_concentration"])
            concentrations.append(concentration if np.isfinite(concentration) else 0.0)
        output[frequency_index] = float(support_scores[frequency_index]) * min(concentrations)
    return output


def _group_and_block_summary(
    coefficients: np.ndarray,
    frequencies_hz: np.ndarray,
    frequency_index: int,
    groups: Mapping[str, np.ndarray],
    chunks: Sequence[np.ndarray],
    model_fold: np.ndarray,
    timestamps_s: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
    reference_name = "core" if np.count_nonzero(groups["core"]) >= 3 else "heart_support"
    aggregates = {
        name: _aggregate_coefficients(coefficients, selected)
        for name, selected in groups.items()
    }
    spatial = {
        name: _spatial_phase_coherence(coefficients, selected)
        for name, selected in groups.items()
    }
    reference = aggregates[reference_name]
    reference_noise = _sideband_noise(reference, frequencies_hz, frequency_index)
    relations: dict[str, dict[str, Any]] = {}
    group_summary: dict[str, dict[str, Any]] = {}
    for name, selected in groups.items():
        aggregate = aggregates[name]
        noise = _sideband_noise(aggregate, frequencies_hz, frequency_index)
        relation = _phase_relation(
            aggregate[frequency_index],
            reference[frequency_index],
            noise,
            reference_noise,
        )
        relations[name] = relation
        ratios = np.asarray(relation["target_to_sideband_ratio"], dtype=np.float64)
        group_summary[name] = {
            "pixel_count": int(np.count_nonzero(selected)),
            "median_amplitude": float(
                _quiet_nanmedian(np.abs(aggregate[frequency_index]))
            ),
            "median_target_to_sideband_ratio": float(_quiet_nanmedian(ratios)),
            "median_spatial_phase_coherence": float(
                _quiet_nanmedian(spatial[name][frequency_index])
            ),
            "active_block_count": int(relation["active_block_count"]),
            "phase_offset_concentration": float(
                relation["phase_offset_concentration"]
            ),
            "mean_phase_offset_deg": float(relation["mean_phase_offset_deg"]),
            "median_abs_phase_offset_deg": float(
                relation["median_abs_phase_offset_deg"]
            ),
        }
    block_summary: list[dict[str, Any]] = []
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    for block_index, rows in enumerate(chunks):
        model_indices = np.unique(np.asarray(model_fold)[rows])
        model_indices = model_indices[model_indices >= 0]
        row: dict[str, Any] = {
            "block_index": int(block_index),
            "start_s": float(timestamps[rows[0]]),
            "stop_s": float(timestamps[rows[-1]]),
            "crossfit_model_fold_index": int(model_indices[0])
            if model_indices.size == 1
            else -1,
        }
        for name, selected in groups.items():
            value = aggregates[name][frequency_index, block_index]
            relation = relations[name]
            row[f"{name}_pixel_count"] = int(np.count_nonzero(selected))
            row[f"{name}_amplitude"] = float(np.abs(value))
            row[f"{name}_phase_deg"] = float(np.degrees(np.angle(value)))
            row[f"{name}_spatial_phase_coherence"] = float(
                spatial[name][frequency_index, block_index]
            )
            row[f"{name}_target_to_sideband_ratio"] = float(
                relation["target_to_sideband_ratio"][block_index]
            )
            row[f"{name}_phase_offset_to_reference_deg"] = float(
                np.degrees(relation["offsets_rad"][block_index])
            )
            row[f"{name}_phase_active"] = bool(relation["active"][block_index])
        block_summary.append(row)
    return group_summary, tuple(block_summary)


def _control_scores(
    dataset: LocalCoordinateDataset,
    coefficients: np.ndarray,
    chunks: Sequence[np.ndarray],
    frequencies_hz: np.ndarray,
    frequency_index: int,
    groups: Mapping[str, np.ndarray],
    config: HeartrateConfig,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in ("global_mean", "body_control_mean", "external_control_mean"):
        if name not in dataset.nuisance_names:
            continue
        column = dataset.nuisance_names.index(name)
        trace = np.asarray(dataset.nuisance_values, dtype=np.float64)[:, column, None]
        control_coefficients = _chunk_frequency_coefficients(
            trace,
            dataset.timestamps_s,
            chunks,
            np.asarray([frequencies_hz[frequency_index]], dtype=np.float64),
            min_valid_fraction=float(config.min_chunk_valid_fraction),
            max_interpolated_gap_seconds=float(config.max_interpolated_gap_seconds),
        )[0, :, 0]
        output[name] = float(_temporal_coherence_score(control_coefficients[None, :])[0])
    support = groups["heart_support"]
    motion_trace = _finite_complex_mean(
        np.asarray(dataset.motion_prediction, dtype=np.float64)[:, support].astype(
            np.complex128
        ),
        axis=1,
    ).real[:, None]
    motion_coefficients = _chunk_frequency_coefficients(
        motion_trace,
        dataset.timestamps_s,
        chunks,
        np.asarray([frequencies_hz[frequency_index]], dtype=np.float64),
        min_valid_fraction=float(config.min_chunk_valid_fraction),
        max_interpolated_gap_seconds=float(config.max_interpolated_gap_seconds),
    )[0, :, 0]
    output["heart_motion_prediction"] = float(
        _temporal_coherence_score(motion_coefficients[None, :])[0]
    )
    if np.any(groups["esophagus_control"]):
        esophagus_aggregate = _aggregate_coefficients(
            coefficients,
            groups["esophagus_control"],
        )
        esophagus_spatial = _spatial_phase_coherence(
            coefficients,
            groups["esophagus_control"],
        )
        output["esophagus_control"] = float(
            _temporal_coherence_score(
                esophagus_aggregate,
                esophagus_spatial,
            )[frequency_index]
        )
    return output


def analyze_dynamic_heart_support(
    dataset: LocalCoordinateDataset,
    config: HeartrateConfig,
    result: HeartrateResult,
    *,
    heart_mask: np.ndarray | None = None,
    esophagus_mask: np.ndarray | None = None,
    mask_is_independent: bool = False,
    frequency_margin_hz: float | None = None,
    frequency_min_hz: float | None = None,
    frequency_max_hz: float | None = None,
    surrogate_count: int | None = None,
    seed: int | None = None,
) -> DynamicHeartSupportResult:
    """Test a fixed anatomical support while allowing block-varying visibility.

    A support derived from the two observed clusters is always labeled post hoc.
    It can generate diagnostics and a fixed-support surrogate p-value, but it
    cannot reverse the original confirmatory decision on the same interval.
    """

    dataset.validated()
    config.validated()
    risks = build_risk_surfaces(dataset, config)
    partitions = balanced_valid_partitions(dataset, risks, config)
    groups, support_source = _build_pixel_groups(
        dataset,
        result,
        risks.eligible,
        heart_mask=heart_mask,
        esophagus_mask=esophagus_mask,
    )
    crossfit_residual, model_fold, active_rows, crossfit_models = _crossfit_residual(
        dataset,
        result,
        partitions,
    )
    chunks = _analysis_chunks(
        dataset,
        active_rows,
        groups["heart_support"],
        config,
    )
    if len(chunks) < 2:
        raise ValueError("dynamic support analysis requires at least two held-out blocks")
    full_frequency_grid = _frequency_grid(config)
    explicit_bounds = frequency_min_hz is not None or frequency_max_hz is not None
    if explicit_bounds and (frequency_min_hz is None or frequency_max_hz is None):
        raise ValueError("frequency_min_hz and frequency_max_hz must be supplied together")
    if explicit_bounds:
        frequency_min = float(frequency_min_hz)
        frequency_max = float(frequency_max_hz)
        frequency_search_source = "explicit_prespecified_bounds"
    elif heart_mask is not None and mask_is_independent:
        frequency_min = float(config.band_min_hz)
        frequency_max = float(config.band_max_hz)
        frequency_search_source = "preconfigured_full_band"
    else:
        fold_frequencies = np.asarray(
            [fold.discovery.candidate.frequency_hz for fold in result.folds],
            dtype=np.float64,
        )
        if not np.isfinite(fold_frequencies).all():
            raise ValueError("dynamic support analysis requires finite fold frequencies")
        margin = (
            max(
                1.0 / float(config.partition_block_seconds),
                2.0 * float(config.frequency_step_hz),
            )
            if frequency_margin_hz is None
            else float(frequency_margin_hz)
        )
        if margin < 0.0:
            raise ValueError("frequency_margin_hz cannot be negative")
        frequency_min = float(np.min(fold_frequencies) - margin)
        frequency_max = float(np.max(fold_frequencies) + margin)
        frequency_search_source = "posthoc_fold_centered_bounds"
    if not (frequency_min < frequency_max):
        raise ValueError("dynamic frequency bounds must be increasing")
    frequencies = full_frequency_grid[
        (full_frequency_grid >= frequency_min)
        & (full_frequency_grid <= frequency_max)
    ]
    if frequencies.size == 0:
        raise ValueError("candidate-centered dynamic frequency grid is empty")
    coefficients = _chunk_frequency_coefficients(
        crossfit_residual,
        dataset.timestamps_s,
        chunks,
        frequencies,
        min_valid_fraction=float(config.min_chunk_valid_fraction),
        max_interpolated_gap_seconds=float(config.max_interpolated_gap_seconds),
    )
    support_aggregate = _aggregate_coefficients(coefficients, groups["heart_support"])
    support_spatial = _spatial_phase_coherence(coefficients, groups["heart_support"])
    support_scores = _temporal_coherence_score(support_aggregate, support_spatial)
    shared_scores = _shared_phase_frequency_scores(
        coefficients,
        frequencies,
        groups,
        support_scores,
    )
    latent_scores, latent_matched, latent_alignment = _crossfit_latent_scores(
        dataset,
        partitions,
        crossfit_models,
        groups["heart_support"],
        frequencies,
        config,
    )
    selected_index = int(np.nanargmax(latent_scores))
    group_summary, block_summary = _group_and_block_summary(
        coefficients,
        frequencies,
        selected_index,
        groups,
        chunks,
        model_fold,
        dataset.timestamps_s,
    )
    core_aggregate = _aggregate_coefficients(coefficients, groups["core"])
    core_spatial = _spatial_phase_coherence(coefficients, groups["core"])
    core_scores = _temporal_coherence_score(core_aggregate, core_spatial)
    selected_support_score = float(support_scores[selected_index])
    selected_core_score = float(core_scores[selected_index])
    union_to_core = float(
        selected_support_score / max(selected_core_score, np.finfo(float).eps)
    )

    count = int(config.surrogate_count if surrogate_count is None else surrogate_count)
    rng = np.random.default_rng(int(config.random_seed if seed is None else seed))
    null_support = np.zeros(count, dtype=np.float64)
    null_shared = np.zeros(count, dtype=np.float64)
    null_latent = np.zeros(count, dtype=np.float64)
    for surrogate_index in range(count):
        surrogate = autocorrelation_preserving_surrogate(
            dataset,
            active_rows,
            rng=rng,
            spatial_block_px=int(config.surrogate_spatial_block_px),
            min_shift_seconds=float(config.surrogate_min_shift_seconds),
            max_gap_factor=float(config.max_timestamp_gap_factor),
        )
        (
            surrogate_residual,
            _surrogate_model_fold,
            _surrogate_active,
            surrogate_models,
        ) = _crossfit_residual(
            surrogate,
            result,
            partitions,
            refit_nuisance=True,
            nuisance_ridge=float(config.nuisance_ridge),
        )
        surrogate_coefficients = _chunk_frequency_coefficients(
            surrogate_residual,
            surrogate.timestamps_s,
            chunks,
            frequencies,
            min_valid_fraction=float(config.min_chunk_valid_fraction),
            max_interpolated_gap_seconds=float(config.max_interpolated_gap_seconds),
        )
        surrogate_support_aggregate = _aggregate_coefficients(
            surrogate_coefficients,
            groups["heart_support"],
        )
        surrogate_support_spatial = _spatial_phase_coherence(
            surrogate_coefficients,
            groups["heart_support"],
        )
        surrogate_support_scores = _temporal_coherence_score(
            surrogate_support_aggregate,
            surrogate_support_spatial,
        )
        surrogate_shared_scores = _shared_phase_frequency_scores(
            surrogate_coefficients,
            frequencies,
            groups,
            surrogate_support_scores,
        )
        surrogate_latent_scores, _matched, _alignment = _crossfit_latent_scores(
            surrogate,
            partitions,
            surrogate_models,
            groups["heart_support"],
            frequencies,
            config,
        )
        null_support[surrogate_index] = float(np.nanmax(surrogate_support_scores))
        null_shared[surrogate_index] = float(np.nanmax(surrogate_shared_scores))
        null_latent[surrogate_index] = float(np.nanmax(surrogate_latent_scores))
    observed_support = selected_support_score
    observed_shared = float(shared_scores[selected_index])
    observed_latent = float(latent_scores[selected_index])
    support_p = (
        float(1 + np.count_nonzero(null_support >= observed_support))
        / float(count + 1)
        if count
        else 1.0
    )
    shared_p = (
        float(1 + np.count_nonzero(null_shared >= observed_shared))
        / float(count + 1)
        if count
        else 1.0
    )
    latent_p = (
        float(1 + np.count_nonzero(null_latent >= observed_latent))
        / float(count + 1)
        if count
        else 1.0
    )
    support_threshold = (
        float(np.quantile(null_support, 1.0 - float(config.alpha), method="higher"))
        if count
        else math.inf
    )
    shared_threshold = (
        float(np.quantile(null_shared, 1.0 - float(config.alpha), method="higher"))
        if count
        else math.inf
    )
    latent_threshold = (
        float(np.quantile(null_latent, 1.0 - float(config.alpha), method="higher"))
        if count
        else math.inf
    )
    support_exceeds = bool(
        count
        and support_p <= float(config.alpha)
        and observed_support > support_threshold
    )
    shared_exceeds = bool(
        count
        and shared_p <= float(config.alpha)
        and observed_shared > shared_threshold
    )
    latent_exceeds = bool(
        count
        and latent_p <= float(config.alpha)
        and observed_latent > latent_threshold
    )
    comparison_names = [
        name
        for name in ("fold0_only", "fold1_only", "anatomical_only")
        if int(group_summary[name]["pixel_count"]) > 0
    ]
    insufficient = any(
        int(group_summary[name]["active_block_count"]) < 2
        for name in comparison_names
    )
    stable = bool(
        comparison_names
        and not insufficient
        and all(
            float(group_summary[name]["phase_offset_concentration"]) >= 0.8
            for name in comparison_names
        )
    )
    if insufficient:
        phase_pattern = "insufficient_exclusive_region_amplitude"
    elif stable and support_exceeds and shared_exceeds:
        phase_pattern = "stable_exclusive_region_phase_relationship_above_joint_null"
    elif stable:
        phase_pattern = "stable_offsets_without_joint_support_significance"
    else:
        phase_pattern = "exclusive_region_phase_relationship_not_stable"
    confirmatory_eligible = bool(
        heart_mask is not None and mask_is_independent and support_source == "external_anatomical_mask"
    )
    if not confirmatory_eligible:
        interpretation = "exploratory_only_support_was_not_independently_prespecified"
    elif not latent_exceeds:
        interpretation = "crossfit_latent_anatomical_pattern_not_significant"
    else:
        interpretation = "fixed_anatomical_support_has_confirmatory_latent_pattern_evidence"
    controls = _control_scores(
        dataset,
        coefficients,
        chunks,
        frequencies,
        selected_index,
        groups,
        config,
    )
    finite_controls = {
        name: float(value)
        for name, value in controls.items()
        if np.isfinite(value)
    }
    strongest_control = (
        max(finite_controls, key=finite_controls.get) if finite_controls else None
    )
    strongest_score = (
        finite_controls[strongest_control] if strongest_control is not None else 0.0
    )
    control_ratio = float(observed_latent / max(strongest_score, np.finfo(float).eps))
    block_models = np.asarray(
        [int(row["crossfit_model_fold_index"]) for row in block_summary],
        dtype=np.int16,
    )
    return DynamicHeartSupportResult(
        support_source=support_source,
        frequency_search_source=frequency_search_source,
        confirmatory_eligible=confirmatory_eligible,
        interpretation=interpretation,
        phase_pattern=phase_pattern,
        frequency_hz=float(frequencies[selected_index]),
        support_score=selected_support_score,
        support_p_value=support_p,
        support_exceeds_null=support_exceeds,
        shared_phase_score=float(shared_scores[selected_index]),
        shared_phase_p_value=shared_p,
        shared_phase_exceeds_null=shared_exceeds,
        joint_p_value=max(support_p, shared_p),
        joint_exceeds_null=bool(support_exceeds and shared_exceeds),
        latent_score=observed_latent,
        latent_p_value=latent_p,
        latent_exceeds_null=latent_exceeds,
        union_to_core_score_ratio=union_to_core,
        strongest_control=strongest_control,
        control_ratio=control_ratio,
        control_scores=controls,
        pixel_groups=groups,
        group_summary=group_summary,
        block_summary=block_summary,
        block_rows=tuple(np.asarray(rows, dtype=np.int64) for rows in chunks),
        block_model_fold_indices=block_models,
        block_coefficients=np.asarray(coefficients[selected_index], dtype=np.complex128),
        frequency_grid_hz=frequencies,
        frequency_support_scores=np.asarray(support_scores, dtype=np.float64),
        frequency_shared_phase_scores=np.asarray(shared_scores, dtype=np.float64),
        frequency_latent_scores=np.asarray(latent_scores, dtype=np.float64),
        null_max_support_scores=null_support,
        null_max_shared_phase_scores=null_shared,
        null_max_latent_scores=null_latent,
        latent_block_coefficients=np.asarray(
            latent_matched[selected_index], dtype=np.complex128
        ),
        latent_block_alignment_coherence=np.asarray(
            latent_alignment[selected_index], dtype=np.float64
        ),
    )
