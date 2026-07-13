from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.local_rostral_heartrate import (
    LocalCoordinateDataset,
    NuisanceModel,
    _chunk_frequency_coefficients,
    apply_nuisance_model,
    contiguous_segments,
    fit_nuisance_model,
)


@dataclass(frozen=True)
class MatchedProjectionConfig:
    """Training and missing-data policy for a frozen-frequency spatial filter."""

    covariance_mode: str = "diagonal"
    covariance_shrinkage: float = 0.25
    covariance_ridge_fraction: float = 1e-6
    nuisance_ridge: float = 1e-6
    discovery_chunk_seconds: float = 4.0
    minimum_chunk_cycles: float = 2.0
    minimum_pixel_valid_fraction: float = 0.8
    minimum_loading_coherence: float = 0.0
    minimum_effective_pixels: int = 2
    minimum_frame_weight_fraction: float = 0.75
    maximum_timestamp_gap_factor: float = 1.75
    maximum_interpolated_gap_seconds: float = 0.02

    def validated(self) -> MatchedProjectionConfig:
        if str(self.covariance_mode) not in {"diagonal", "shrinkage"}:
            raise ValueError("covariance_mode must be diagonal or shrinkage")
        if not 0.0 <= float(self.covariance_shrinkage) <= 1.0:
            raise ValueError("covariance_shrinkage must be between zero and one")
        if float(self.covariance_ridge_fraction) < 0.0:
            raise ValueError("covariance_ridge_fraction cannot be negative")
        if float(self.nuisance_ridge) < 0.0:
            raise ValueError("nuisance_ridge cannot be negative")
        if float(self.discovery_chunk_seconds) <= 0.0:
            raise ValueError("discovery_chunk_seconds must be positive")
        if float(self.minimum_chunk_cycles) <= 0.0:
            raise ValueError("minimum_chunk_cycles must be positive")
        if not 0.0 < float(self.minimum_pixel_valid_fraction) <= 1.0:
            raise ValueError("minimum_pixel_valid_fraction must be in (0, 1]")
        if not 0.0 <= float(self.minimum_loading_coherence) <= 1.0:
            raise ValueError("minimum_loading_coherence must be between zero and one")
        if int(self.minimum_effective_pixels) < 1:
            raise ValueError("minimum_effective_pixels must be positive")
        if not 0.0 < float(self.minimum_frame_weight_fraction) <= 1.0:
            raise ValueError("minimum_frame_weight_fraction must be in (0, 1]")
        if float(self.maximum_timestamp_gap_factor) <= 1.0:
            raise ValueError("maximum_timestamp_gap_factor must exceed one")
        if float(self.maximum_interpolated_gap_seconds) < 0.0:
            raise ValueError("maximum_interpolated_gap_seconds cannot be negative")
        return self


@dataclass(frozen=True)
class MatchedSpatialProjectionModel:
    """A spatial projection learned exclusively from one discovery partition."""

    frequency_hz: float
    config: MatchedProjectionConfig
    frozen_mask: np.ndarray
    discovery_rows: np.ndarray
    nuisance_model: NuisanceModel
    loadings: np.ndarray
    loading_coherence: np.ndarray
    noise_variance: np.ndarray
    effective_pixels: np.ndarray
    complex_weights: np.ndarray
    signed_weights: np.ndarray
    signed_template: np.ndarray
    reference_phase_rad: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FoldMatchedProjection:
    """One discovery model applied only to its held-out confirmation rows."""

    fold_index: int
    model: MatchedSpatialProjectionModel
    confirmation_rows: np.ndarray
    projected_trace: np.ndarray
    complex_projected_trace: np.ndarray
    frame_valid: np.ndarray
    weight_fraction: np.ndarray
    effective_pixel_count: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossfitMatchedSpatialProjection:
    """Gap-preserving projection in which every finite output row is held out."""

    frequency_hz: float
    frozen_mask: np.ndarray
    projected_trace: np.ndarray
    complex_projected_trace: np.ndarray
    frame_valid: np.ndarray
    fold_labels: np.ndarray
    weight_fraction: np.ndarray
    effective_pixel_count: np.ndarray
    folds: tuple[FoldMatchedProjection, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def weights(self) -> np.ndarray:
        return np.stack([fold.model.complex_weights for fold in self.folds], axis=0)

    @property
    def signed_weights(self) -> np.ndarray:
        return np.stack([fold.model.signed_weights for fold in self.folds], axis=0)

    @property
    def loadings(self) -> np.ndarray:
        return np.stack([fold.model.loadings for fold in self.folds], axis=0)

    @property
    def effective_pixels(self) -> np.ndarray:
        return np.stack([fold.model.effective_pixels for fold in self.folds], axis=0)


def _row_mask(rows: np.ndarray, frame_count: int, *, name: str) -> np.ndarray:
    values = np.asarray(rows)
    if values.dtype == bool:
        if values.shape != (int(frame_count),):
            raise ValueError(f"{name} boolean mask must have shape ({frame_count},)")
        return values.copy()
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional mask or index array")
    indices = values.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= int(frame_count)):
        raise ValueError(f"{name} contains an out-of-range row index")
    output = np.zeros(int(frame_count), dtype=bool)
    output[indices] = True
    return output


def _pixel_mask(mask: np.ndarray, pixel_count: int) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (int(pixel_count),):
        raise ValueError(f"frozen_mask must have shape ({pixel_count},)")
    if not np.any(selected):
        raise ValueError("frozen_mask cannot be empty")
    return selected.copy()


def _frequency_supported(dataset: LocalCoordinateDataset, frequency_hz: float) -> float:
    frequency = float(frequency_hz)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    spacing = np.diff(timestamps)
    spacing = spacing[np.isfinite(spacing) & (spacing > 0.0)]
    if not spacing.size:
        raise ValueError("dataset has no positive timestamp spacing")
    nyquist = 0.5 / float(np.median(spacing))
    if frequency >= nyquist:
        raise ValueError(
            f"frequency_hz={frequency:g} is not below timestamp Nyquist {nyquist:g}"
        )
    return frequency


def _split_segments_into_chunks(
    dataset: LocalCoordinateDataset,
    discovery_rows: np.ndarray,
    *,
    frequency_hz: float,
    config: MatchedProjectionConfig,
) -> list[np.ndarray]:
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    accepted = np.asarray(discovery_rows, dtype=bool) & np.asarray(dataset.frame_valid, dtype=bool)
    minimum_seconds = float(config.minimum_chunk_cycles) / float(frequency_hz)
    segments = contiguous_segments(
        timestamps,
        accepted,
        max_gap_factor=float(config.maximum_timestamp_gap_factor),
        min_seconds=minimum_seconds,
    )
    chunks: list[np.ndarray] = []
    for segment in segments:
        start = 0
        while start < segment.size:
            stop_time = float(timestamps[segment[start]]) + float(config.discovery_chunk_seconds)
            stop = int(np.searchsorted(timestamps[segment], stop_time, side="left"))
            stop = max(start + 1, min(stop, int(segment.size)))
            candidate = segment[start:stop]
            duration = float(timestamps[candidate[-1]] - timestamps[candidate[0]])
            if candidate.size > 1:
                duration += float(np.median(np.diff(timestamps[candidate])))
            if duration >= minimum_seconds:
                chunks.append(candidate)
            start = stop
    return chunks


def _complex_loading_summary(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(coefficients, dtype=np.complex128)
    finite = np.isfinite(values)
    count = np.sum(finite, axis=0)
    total = np.sum(np.where(finite, values, 0.0 + 0.0j), axis=0)
    loading = np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan + 0.0j, dtype=np.complex128),
        where=count > 0,
    )
    magnitude_sum = np.sum(np.where(finite, np.abs(values), 0.0), axis=0)
    coherence = np.divide(
        np.abs(total),
        magnitude_sum,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=magnitude_sum > np.finfo(float).eps,
    )
    return loading, coherence


def _robust_noise_variance(noise: np.ndarray) -> np.ndarray:
    values = np.asarray(noise, dtype=np.float64)
    variance = np.full(values.shape[1], np.nan, dtype=np.float64)
    for column in range(values.shape[1]):
        finite_values = values[np.isfinite(values[:, column]), column]
        if finite_values.size == 0:
            continue
        center = float(np.median(finite_values))
        mad = float(np.median(np.abs(finite_values - center)))
        estimate = float(np.square(1.4826 * mad))
        if (not np.isfinite(estimate) or estimate <= np.finfo(float).eps) and (
            finite_values.size > 1
        ):
            estimate = float(np.var(finite_values, ddof=1))
        variance[column] = estimate
    finite_positive = variance[np.isfinite(variance) & (variance > np.finfo(float).eps)]
    floor = (
        max(float(np.median(finite_positive)) * 1e-6, np.finfo(float).eps)
        if finite_positive.size
        else 1.0
    )
    variance[~np.isfinite(variance) | (variance < floor)] = floor
    return variance


def _noise_covariance(
    noise: np.ndarray,
    effective: np.ndarray,
    variance: np.ndarray,
    config: MatchedProjectionConfig,
) -> tuple[np.ndarray, float]:
    pixels = np.flatnonzero(effective)
    diagonal = np.asarray(variance[pixels], dtype=np.float64)
    if str(config.covariance_mode) == "diagonal":
        covariance = np.diag(diagonal)
    else:
        values = np.asarray(noise[:, pixels], dtype=np.float64)
        center = np.nanmedian(values, axis=0)
        centered = values - center[None, :]
        finite = np.isfinite(centered)
        filled = np.where(finite, centered, 0.0)
        pair_count = finite.T.astype(np.float64) @ finite.astype(np.float64)
        empirical = np.divide(
            filled.T @ filled,
            np.maximum(pair_count - 1.0, 1.0),
            out=np.zeros((pixels.size, pixels.size), dtype=np.float64),
            where=pair_count > 1.0,
        )
        empirical = 0.5 * (empirical + empirical.T)
        np.fill_diagonal(empirical, diagonal)
        shrinkage = float(config.covariance_shrinkage)
        covariance = (1.0 - shrinkage) * empirical + shrinkage * np.diag(diagonal)
    scale = float(np.median(diagonal)) if diagonal.size else 1.0
    ridge = max(scale * float(config.covariance_ridge_fraction), np.finfo(float).eps)
    covariance = covariance + np.eye(covariance.shape[0], dtype=np.float64) * ridge
    condition = float(np.linalg.cond(covariance))
    return covariance, condition


def fit_matched_spatial_projection(
    dataset: LocalCoordinateDataset,
    frozen_mask: np.ndarray,
    discovery_rows: np.ndarray,
    *,
    frequency_hz: float,
    config: MatchedProjectionConfig | None = None,
) -> MatchedSpatialProjectionModel:
    """Fit a signed and complex matched spatial filter on discovery rows only."""

    dataset.validated()
    resolved = (config or MatchedProjectionConfig()).validated()
    frequency = _frequency_supported(dataset, frequency_hz)
    selected = _pixel_mask(frozen_mask, dataset.pixel_count)
    discovery = _row_mask(discovery_rows, dataset.frame_count, name="discovery_rows")
    discovery &= np.asarray(dataset.frame_valid, dtype=bool)
    if int(np.count_nonzero(discovery)) < 8:
        raise ValueError("too few valid discovery rows")
    nuisance_model = fit_nuisance_model(
        dataset,
        discovery,
        ridge=float(resolved.nuisance_ridge),
    )
    residual = apply_nuisance_model(dataset, nuisance_model)
    chunks = _split_segments_into_chunks(
        dataset,
        discovery,
        frequency_hz=frequency,
        config=resolved,
    )
    if not chunks:
        raise ValueError("discovery rows contain no sufficiently long contiguous chunks")
    coefficients = _chunk_frequency_coefficients(
        residual,
        dataset.timestamps_s,
        chunks,
        np.asarray([frequency], dtype=np.float64),
        min_valid_fraction=float(resolved.minimum_pixel_valid_fraction),
        max_interpolated_gap_seconds=float(resolved.maximum_interpolated_gap_seconds),
    )[0]
    loadings, loading_coherence = _complex_loading_summary(coefficients)
    amplitude = np.abs(loadings)
    effective = (
        selected
        & np.asarray(nuisance_model.fitted_pixels, dtype=bool)
        & np.isfinite(loadings)
        & np.isfinite(loading_coherence)
        & (amplitude > np.finfo(float).eps)
        & (loading_coherence >= float(resolved.minimum_loading_coherence))
    )
    if int(np.count_nonzero(effective)) < int(resolved.minimum_effective_pixels):
        raise ValueError("too few frozen-mask pixels have stable discovery loadings")

    phase = np.exp(2j * np.pi * frequency * np.asarray(dataset.timestamps_s, dtype=np.float64))
    modeled = np.real(phase[:, None] * loadings[None, :])
    noise = np.asarray(residual, dtype=np.float64) - modeled
    noise[~discovery] = np.nan
    variance = _robust_noise_variance(noise)
    effective &= np.isfinite(variance) & (variance > 0.0)
    if int(np.count_nonzero(effective)) < int(resolved.minimum_effective_pixels):
        raise ValueError("too few frozen-mask pixels have estimable discovery noise")

    covariance, condition = _noise_covariance(noise, effective, variance, resolved)
    precision = np.linalg.pinv(covariance, hermitian=True)
    pixels = np.flatnonzero(effective)
    local_loading = loadings[pixels]
    # Tie polarity to a stable anatomical pixel rather than whichever polarity
    # happens to have the largest noisy amplitude in this fold.
    anchor_local = 0
    reference_phase = float(np.angle(local_loading[anchor_local]))
    signed_template_local = np.real(local_loading * np.exp(-1j * reference_phase))

    complex_raw = precision @ local_loading
    complex_response = np.vdot(complex_raw, local_loading)
    if not np.isfinite(complex_response) or abs(complex_response) <= np.finfo(float).eps:
        raise ValueError("complex matched filter has zero response")
    complex_local = complex_raw / np.conjugate(complex_response)
    signed_raw = precision @ signed_template_local
    signed_response = float(signed_raw @ signed_template_local)
    if not np.isfinite(signed_response) or abs(signed_response) <= np.finfo(float).eps:
        raise ValueError("signed matched filter has zero response")
    signed_local = signed_raw / signed_response

    complex_weights = np.zeros(dataset.pixel_count, dtype=np.complex128)
    signed_weights = np.zeros(dataset.pixel_count, dtype=np.float64)
    signed_template = np.full(dataset.pixel_count, np.nan, dtype=np.float64)
    complex_weights[pixels] = complex_local
    signed_weights[pixels] = signed_local
    signed_template[pixels] = signed_template_local
    bipolar_phase_coherence = float(
        np.abs(np.mean(np.exp(2j * np.angle(local_loading))))
    )
    diagnostics: dict[str, Any] = {
        "covariance_mode": str(resolved.covariance_mode),
        "covariance_shrinkage": float(resolved.covariance_shrinkage),
        "covariance_condition_number": condition,
        "discovery_row_count": int(np.count_nonzero(discovery)),
        "discovery_chunk_count": int(len(chunks)),
        "frozen_pixel_count": int(np.count_nonzero(selected)),
        "effective_pixel_count": int(pixels.size),
        "median_loading_amplitude": float(np.median(amplitude[pixels])),
        "median_chunk_loading_coherence": float(np.median(loading_coherence[pixels])),
        "bipolar_phase_coherence": bipolar_phase_coherence,
        "reference_pixel_index": int(pixels[anchor_local]),
        "reference_phase_rad": reference_phase,
        "weight_normalization": "unit_response_to_discovery_loading",
        "control_projection_metadata": {
            "pixel_indices": pixels.copy(),
            "complex_weight_l1_norm": float(np.sum(np.abs(complex_local))),
            "signed_weight_l1_norm": float(np.sum(np.abs(signed_local))),
        },
    }
    return MatchedSpatialProjectionModel(
        frequency_hz=frequency,
        config=resolved,
        frozen_mask=selected,
        discovery_rows=discovery,
        nuisance_model=nuisance_model,
        loadings=loadings,
        loading_coherence=loading_coherence,
        noise_variance=variance,
        effective_pixels=effective,
        complex_weights=complex_weights,
        signed_weights=signed_weights,
        signed_template=signed_template,
        reference_phase_rad=reference_phase,
        diagnostics=diagnostics,
    )


def apply_matched_spatial_projection(
    dataset: LocalCoordinateDataset,
    model: MatchedSpatialProjectionModel,
    confirmation_rows: np.ndarray,
    *,
    fold_index: int = 0,
    config: MatchedProjectionConfig | None = None,
) -> FoldMatchedProjection:
    """Apply a frozen discovery model without filling invalid confirmation samples."""

    dataset.validated()
    resolved = (config or model.config).validated()
    confirmation = _row_mask(
        confirmation_rows,
        dataset.frame_count,
        name="confirmation_rows",
    )
    if np.any(confirmation & np.asarray(model.discovery_rows, dtype=bool)):
        raise ValueError("discovery and confirmation rows must be disjoint")
    if model.frozen_mask.shape != (dataset.pixel_count,):
        raise ValueError("model pixel axis does not match dataset")
    residual = apply_nuisance_model(dataset, model.nuisance_model)
    pixels = np.flatnonzero(np.asarray(model.effective_pixels, dtype=bool))
    if pixels.size < int(resolved.minimum_effective_pixels):
        raise ValueError("model has too few effective pixels")
    signed_weights = np.asarray(model.signed_weights, dtype=np.float64)[pixels]
    complex_weights = np.asarray(model.complex_weights, dtype=np.complex128)[pixels]
    signed_template = np.asarray(model.signed_template, dtype=np.float64)[pixels]
    loading = np.asarray(model.loadings, dtype=np.complex128)[pixels]
    values = residual[:, pixels]
    finite = np.isfinite(values)
    signed_l1 = float(np.sum(np.abs(signed_weights)))
    per_frame_l1 = finite @ np.abs(signed_weights)
    weight_fraction = np.divide(
        per_frame_l1,
        signed_l1,
        out=np.zeros(dataset.frame_count, dtype=np.float64),
        where=signed_l1 > np.finfo(float).eps,
    )
    effective_count = np.sum(finite, axis=1).astype(np.int16)
    frame_valid = (
        confirmation
        & np.asarray(dataset.frame_valid, dtype=bool)
        & (effective_count >= int(resolved.minimum_effective_pixels))
        & (weight_fraction >= float(resolved.minimum_frame_weight_fraction))
    )
    projected = np.full(dataset.frame_count, np.nan, dtype=np.float64)
    complex_projected = np.full(
        dataset.frame_count,
        np.nan + 0.0j,
        dtype=np.complex128,
    )
    for row in np.flatnonzero(frame_valid):
        valid = finite[row]
        signed_response = float(signed_weights[valid] @ signed_template[valid])
        complex_response = np.vdot(complex_weights[valid], loading[valid])
        if (
            abs(signed_response) < float(resolved.minimum_frame_weight_fraction)
            or abs(complex_response) < float(resolved.minimum_frame_weight_fraction)
        ):
            frame_valid[row] = False
            continue
        projected[row] = float(signed_weights[valid] @ values[row, valid] / signed_response)
        complex_projected[row] = np.vdot(
            complex_weights[valid],
            values[row, valid],
        ) / complex_response
    valid_values = projected[frame_valid]
    diagnostics: dict[str, Any] = {
        "fold_index": int(fold_index),
        "confirmation_row_count": int(np.count_nonzero(confirmation)),
        "valid_projection_row_count": int(np.count_nonzero(frame_valid)),
        "valid_projection_fraction": float(
            np.count_nonzero(frame_valid) / max(1, np.count_nonzero(confirmation))
        ),
        "projected_trace_rms": (
            float(np.sqrt(np.mean(np.square(valid_values)))) if valid_values.size else math.nan
        ),
        "median_effective_pixel_count": (
            float(np.median(effective_count[frame_valid])) if np.any(frame_valid) else math.nan
        ),
        "median_weight_fraction": (
            float(np.median(weight_fraction[frame_valid])) if np.any(frame_valid) else math.nan
        ),
    }
    return FoldMatchedProjection(
        fold_index=int(fold_index),
        model=model,
        confirmation_rows=confirmation,
        projected_trace=projected,
        complex_projected_trace=complex_projected,
        frame_valid=frame_valid,
        weight_fraction=weight_fraction,
        effective_pixel_count=effective_count,
        diagnostics=diagnostics,
    )


def crossfit_matched_spatial_projection(
    dataset: LocalCoordinateDataset,
    frozen_mask: np.ndarray,
    partitions: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    frequency_hz: float,
    config: MatchedProjectionConfig | None = None,
) -> CrossfitMatchedSpatialProjection:
    """Learn on each discovery partition and concatenate held-out projections."""

    dataset.validated()
    resolved = (config or MatchedProjectionConfig()).validated()
    selected = _pixel_mask(frozen_mask, dataset.pixel_count)
    if not partitions:
        raise ValueError("partitions cannot be empty")
    projected = np.full(dataset.frame_count, np.nan, dtype=np.float64)
    complex_projected = np.full(
        dataset.frame_count,
        np.nan + 0.0j,
        dtype=np.complex128,
    )
    frame_valid = np.zeros(dataset.frame_count, dtype=bool)
    fold_labels = np.full(dataset.frame_count, -1, dtype=np.int16)
    weight_fraction = np.full(dataset.frame_count, np.nan, dtype=np.float64)
    effective_count = np.zeros(dataset.frame_count, dtype=np.int16)
    folds: list[FoldMatchedProjection] = []
    assigned = np.zeros(dataset.frame_count, dtype=bool)
    for fold_index, (raw_discovery, raw_confirmation) in enumerate(partitions):
        discovery = _row_mask(
            raw_discovery,
            dataset.frame_count,
            name=f"partitions[{fold_index}].discovery_rows",
        )
        confirmation = _row_mask(
            raw_confirmation,
            dataset.frame_count,
            name=f"partitions[{fold_index}].confirmation_rows",
        )
        if np.any(discovery & confirmation):
            raise ValueError(f"partition {fold_index} discovery and confirmation rows overlap")
        if np.any(assigned & confirmation):
            raise ValueError("confirmation rows cannot be assigned to more than one fold")
        model = fit_matched_spatial_projection(
            dataset,
            selected,
            discovery,
            frequency_hz=frequency_hz,
            config=resolved,
        )
        fold = apply_matched_spatial_projection(
            dataset,
            model,
            confirmation,
            fold_index=fold_index,
            config=resolved,
        )
        projected[confirmation] = fold.projected_trace[confirmation]
        complex_projected[confirmation] = fold.complex_projected_trace[confirmation]
        frame_valid[confirmation] = fold.frame_valid[confirmation]
        fold_labels[confirmation] = int(fold_index)
        weight_fraction[confirmation] = fold.weight_fraction[confirmation]
        effective_count[confirmation] = fold.effective_pixel_count[confirmation]
        assigned |= confirmation
        folds.append(fold)
    diagnostics: dict[str, Any] = {
        "fold_count": int(len(folds)),
        "assigned_confirmation_row_count": int(np.count_nonzero(assigned)),
        "valid_projection_row_count": int(np.count_nonzero(frame_valid)),
        "frozen_pixel_count": int(np.count_nonzero(selected)),
        "effective_pixels_per_fold": np.asarray(
            [np.count_nonzero(fold.model.effective_pixels) for fold in folds],
            dtype=np.int16,
        ),
        "held_out_only": True,
    }
    return CrossfitMatchedSpatialProjection(
        frequency_hz=float(frequency_hz),
        frozen_mask=selected,
        projected_trace=projected,
        complex_projected_trace=complex_projected,
        frame_valid=frame_valid,
        fold_labels=fold_labels,
        weight_fraction=weight_fraction,
        effective_pixel_count=effective_count,
        folds=tuple(folds),
        diagnostics=diagnostics,
    )
