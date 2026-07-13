from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
from numpy.testing import assert_allclose

from fisheye.analysis.heart_photometry_projection import (
    MatchedProjectionConfig,
    crossfit_matched_spatial_projection,
    fit_matched_spatial_projection,
)
from fisheye.analysis.local_rostral_heartrate import LocalCoordinateDataset


def _opposite_polarity_dataset(*, seed: int = 7) -> tuple[LocalCoordinateDataset, np.ndarray]:
    rng = np.random.default_rng(seed)
    fps = 60.0
    frame_count = 1200
    timestamps = np.arange(frame_count, dtype=np.float64) / fps
    pixel_xy = np.column_stack(
        [np.arange(8, dtype=np.float64), np.zeros(8, dtype=np.float64)]
    )
    frequency_hz = 3.0
    truth = np.sin(2.0 * np.pi * frequency_hz * timestamps)
    nuisance = 0.9 * np.sin(2.0 * np.pi * 0.41 * timestamps)
    signs = np.asarray([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0])
    traces = 120.0 + nuisance[:, None] + 2.0 * truth[:, None] * signs[None, :]
    common_noise = rng.normal(0.0, 0.22, (frame_count, 1))
    traces += 0.4 * common_noise + rng.normal(0.0, 0.32, traces.shape)
    pixel_valid = np.ones(traces.shape, dtype=bool)
    frame_valid = np.ones(frame_count, dtype=bool)
    frame_valid[480:510] = False
    pixel_valid[480:510] = False
    traces[~pixel_valid] = np.nan
    source_xy = np.broadcast_to(pixel_xy[None], (frame_count, 8, 2)).copy()
    bilinear_weights = np.full((frame_count, 8, 4), 0.25, dtype=np.float64)
    dataset = LocalCoordinateDataset(
        frame_indices=np.arange(frame_count, dtype=np.int64),
        timestamps_s=timestamps,
        traces=traces,
        pixel_xy=pixel_xy,
        pixel_valid=pixel_valid,
        frame_valid=frame_valid,
        source_xy=source_xy,
        bilinear_weights=bilinear_weights,
        body_occupancy=np.ones(traces.shape, dtype=np.float64),
        eye_occupancy=np.zeros(traces.shape, dtype=np.float64),
        gradient_magnitude=np.ones(traces.shape, dtype=np.float64),
        motion_prediction=np.zeros(traces.shape, dtype=np.float64),
        nuisance_values=nuisance[:, None],
        nuisance_names=("global_reference",),
        image_shape_hw=(2, 9),
        administrative_boundary_distance_px=np.ones(8, dtype=np.float64),
        physical_boundary_distance_px=np.ones(8, dtype=np.float64),
        transform_uncertainty=np.zeros(frame_count, dtype=np.float64),
        metadata={"pixel_format": "mono8", "fixture": "opposite_polarity"},
    ).validated()
    return dataset, truth


def _partitions(dataset: LocalCoordinateDataset) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    block = np.floor(np.asarray(dataset.timestamps_s) / 2.0).astype(np.int64)
    even = (block % 2) == 0
    odd = ~even
    return ((even, odd), (odd, even))


def _config(*, covariance_mode: str = "diagonal") -> MatchedProjectionConfig:
    return MatchedProjectionConfig(
        covariance_mode=covariance_mode,
        covariance_shrinkage=0.35,
        discovery_chunk_seconds=2.0,
        minimum_chunk_cycles=3.0,
        minimum_pixel_valid_fraction=0.75,
        minimum_effective_pixels=4,
        minimum_frame_weight_fraction=0.75,
    ).validated()


def test_crossfit_projection_recovers_signal_lost_to_polarity_cancellation() -> None:
    dataset, truth = _opposite_polarity_dataset()
    frozen_mask = np.asarray([True, True, True, True, True, True, False, False])

    result = crossfit_matched_spatial_projection(
        dataset,
        frozen_mask,
        _partitions(dataset),
        frequency_hz=3.0,
        config=_config(),
    )

    valid = result.frame_valid
    ordinary_mean = np.full(dataset.frame_count, np.nan, dtype=np.float64)
    ordinary_mean[valid] = np.mean(np.asarray(dataset.traces)[valid][:, frozen_mask], axis=1)
    matched_correlation = float(np.corrcoef(result.projected_trace[valid], truth[valid])[0, 1])
    ordinary_correlation = float(np.corrcoef(ordinary_mean[valid], truth[valid])[0, 1])
    assert matched_correlation > 0.95
    assert abs(ordinary_correlation) < 0.2
    assert np.all(np.nanmedian(result.signed_weights[:, :3], axis=1) > 0.0)
    assert np.all(np.nanmedian(result.signed_weights[:, 3:6], axis=1) < 0.0)
    assert result.diagnostics["held_out_only"] is True


def test_crossfit_projection_preserves_invalid_gaps_and_fold_assignment() -> None:
    dataset, _truth = _opposite_polarity_dataset()
    frozen_mask = np.arange(dataset.pixel_count) < 6
    partitions = _partitions(dataset)

    result = crossfit_matched_spatial_projection(
        dataset,
        frozen_mask,
        partitions,
        frequency_hz=3.0,
        config=_config(),
    )

    assert not np.any(result.frame_valid[480:510])
    assert np.isnan(result.projected_trace[480:510]).all()
    assert np.isnan(result.complex_projected_trace[480:510]).all()
    assert np.all(result.fold_labels[partitions[0][1]] == 0)
    assert np.all(result.fold_labels[partitions[1][1]] == 1)
    assert np.all(np.isfinite(result.projected_trace[result.frame_valid]))


def test_confirmation_values_cannot_change_discovery_weights() -> None:
    dataset, _truth = _opposite_polarity_dataset()
    discovery, confirmation = _partitions(dataset)[0]
    frozen_mask = np.arange(dataset.pixel_count) < 6
    config = _config(covariance_mode="shrinkage")
    baseline = fit_matched_spatial_projection(
        dataset,
        frozen_mask,
        discovery,
        frequency_hz=3.0,
        config=config,
    )
    changed_traces = np.asarray(dataset.traces).copy()
    changed_traces[confirmation] += 25.0 * np.sin(
        2.0 * np.pi * 7.0 * np.asarray(dataset.timestamps_s)[confirmation]
    )[:, None]
    changed = replace(dataset, traces=changed_traces).validated()
    repeated = fit_matched_spatial_projection(
        changed,
        frozen_mask,
        discovery,
        frequency_hz=3.0,
        config=config,
    )

    assert_allclose(repeated.loadings, baseline.loadings, rtol=0.0, atol=0.0, equal_nan=True)
    assert_allclose(repeated.complex_weights, baseline.complex_weights, rtol=0.0, atol=0.0)
    assert_allclose(repeated.signed_weights, baseline.signed_weights, rtol=0.0, atol=0.0)


def test_shrinkage_projection_is_deterministic_and_reports_diagnostics() -> None:
    dataset, _truth = _opposite_polarity_dataset(seed=19)
    frozen_mask = np.arange(dataset.pixel_count) < 6
    kwargs = dict(
        dataset=dataset,
        frozen_mask=frozen_mask,
        partitions=_partitions(dataset),
        frequency_hz=3.0,
        config=_config(covariance_mode="shrinkage"),
    )

    first = crossfit_matched_spatial_projection(**kwargs)
    second = crossfit_matched_spatial_projection(**kwargs)

    assert_allclose(first.projected_trace, second.projected_trace, equal_nan=True)
    assert_allclose(first.weights, second.weights, equal_nan=True)
    assert_allclose(first.loadings, second.loadings, equal_nan=True)
    for fold in first.folds:
        assert fold.model.diagnostics["covariance_mode"] == "shrinkage"
        assert fold.model.diagnostics["effective_pixel_count"] == 6
        assert 0.0 <= fold.model.diagnostics["bipolar_phase_coherence"] <= 1.0
        assert np.isfinite(fold.model.diagnostics["covariance_condition_number"])


def test_nonselected_all_nan_pixel_does_not_emit_runtime_warning() -> None:
    dataset, _truth = _opposite_polarity_dataset()
    traces = np.asarray(dataset.traces).copy()
    pixel_valid = np.asarray(dataset.pixel_valid).copy()
    traces[:, -1] = np.nan
    pixel_valid[:, -1] = False
    with_missing_pixel = replace(
        dataset,
        traces=traces,
        pixel_valid=pixel_valid,
    ).validated()
    discovery, _confirmation = _partitions(with_missing_pixel)[0]
    frozen_mask = np.arange(with_missing_pixel.pixel_count) < 6

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = fit_matched_spatial_projection(
            with_missing_pixel,
            frozen_mask,
            discovery,
            frequency_hz=3.0,
            config=_config(),
        )

    assert not model.effective_pixels[-1]
    assert np.isfinite(model.noise_variance[-1])
