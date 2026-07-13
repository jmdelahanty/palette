from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from fisheye.analysis.heart_photometry_nulls import (
    compute_photometry_family_null_batch,
    evaluate_photometry_family,
    familywise_p_values,
    load_photometry_null_batch,
    write_photometry_null_batch,
)
from fisheye.analysis.heart_photometry_transforms import regional_pool
from fisheye.analysis.local_rostral_heartrate import (
    LocalCoordinateDataset,
    autocorrelation_preserving_surrogate,
)


def _dataset() -> LocalCoordinateDataset:
    rng = np.random.default_rng(91)
    frame_count = 320
    pixel_count = 8
    timestamps = np.arange(frame_count, dtype=np.float64) / 40.0
    yy, xx = np.divmod(np.arange(pixel_count), 4)
    pixel_xy = np.column_stack([xx, yy]).astype(np.float64)
    traces = rng.normal(100.0, 0.8, (frame_count, pixel_count))
    traces += np.sin(2.0 * np.pi * 3.0 * timestamps)[:, None] * np.linspace(
        -0.5, 0.5, pixel_count
    )[None, :]
    valid = np.ones(traces.shape, dtype=bool)
    source_xy = np.broadcast_to(
        pixel_xy[None], (frame_count, pixel_count, 2)
    ).copy()
    return LocalCoordinateDataset(
        frame_indices=np.arange(frame_count, dtype=np.int64),
        timestamps_s=timestamps,
        traces=traces,
        pixel_xy=pixel_xy,
        pixel_valid=valid,
        frame_valid=np.ones(frame_count, dtype=bool),
        source_xy=source_xy,
        bilinear_weights=np.full((frame_count, pixel_count, 4), 0.25),
        body_occupancy=np.ones(traces.shape),
        eye_occupancy=np.zeros(traces.shape),
        gradient_magnitude=np.ones(traces.shape),
        motion_prediction=np.zeros(traces.shape),
        nuisance_values=np.zeros((frame_count, 1)),
        nuisance_names=("global",),
        image_shape_hw=(2, 4),
        administrative_boundary_distance_px=np.ones(pixel_count),
        physical_boundary_distance_px=np.ones(pixel_count),
        transform_uncertainty=np.zeros(frame_count),
        metadata={"fixture": "photometry_null"},
    ).validated()


def _lightweight_scorer(dataset: LocalCoordinateDataset):
    traces = np.asarray(dataset.traces, dtype=np.float64)
    spectral = np.zeros((4, 2), dtype=np.float64)
    control = np.zeros((4, 2), dtype=np.float64)
    for window_index, rows in enumerate(np.array_split(np.arange(dataset.frame_count), 4)):
        first = traces[rows, :4]
        second = traces[rows, 4:]
        spectral[window_index, 0] = 1.0 + abs(float(np.mean(np.diff(first, axis=0))))
        spectral[window_index, 1] = 1.0 + float(np.std(np.mean(first, axis=1)))
        control[window_index, 0] = 1.0 + float(np.std(np.mean(second, axis=1)))
        control[window_index, 1] = 1.0 + abs(float(np.mean(np.diff(second, axis=0))))
    return evaluate_photometry_family(
        candidate_names=("mean", "spatial"),
        window_indices=np.arange(4),
        discovery_windows=np.asarray([True, False, True, False]),
        spectral_ratios=spectral,
        control_ratios=control,
        min_discovery_windows=1,
        min_discovery_spectral_ratio=1.0,
        min_discovery_control_ratio=1.0,
        min_confirmation_windows=1,
        min_confirmation_scorable_fraction=0.5,
    )


def test_family_evaluation_allows_no_candidate() -> None:
    evaluation = evaluate_photometry_family(
        candidate_names=("mean", "derivative"),
        window_indices=np.arange(4),
        discovery_windows=np.asarray([True, False, True, False]),
        spectral_ratios=np.asarray(
            [[1.2, 2.0], [50.0, 50.0], [1.3, 2.1], [50.0, 50.0]]
        ),
        control_ratios=np.asarray(
            [[1.2, 0.8], [50.0, 50.0], [1.3, 0.9], [50.0, 50.0]]
        ),
        min_discovery_windows=2,
        min_discovery_spectral_ratio=1.5,
        min_discovery_control_ratio=1.1,
    )

    assert evaluation.selected_candidate_index == -1
    assert evaluation.selected_confirmation_statistic == -np.inf
    assert evaluation.maximum_window_index in (1, 3)


def test_adaptive_selection_never_uses_confirmation_values() -> None:
    common = dict(
        candidate_names=("baseline", "projection"),
        window_indices=np.arange(4),
        discovery_windows=np.asarray([True, False, True, False]),
        min_discovery_windows=2,
        min_discovery_spectral_ratio=1.0,
        min_discovery_control_ratio=1.0,
        min_confirmation_windows=2,
        min_confirmation_scorable_fraction=1.0,
    )
    first = evaluate_photometry_family(
        **common,
        spectral_ratios=np.asarray(
            [[4.0, 2.0], [0.1, 1e6], [4.0, 2.0], [0.1, 1e6]]
        ),
        control_ratios=np.asarray(
            [[2.0, 1.5], [0.1, 1e6], [2.0, 1.5], [0.1, 1e6]]
        ),
    )
    second = evaluate_photometry_family(
        **common,
        spectral_ratios=np.asarray(
            [[4.0, 2.0], [1e6, 0.1], [4.0, 2.0], [1e6, 0.1]]
        ),
        control_ratios=np.asarray(
            [[2.0, 1.5], [1e6, 0.1], [2.0, 1.5], [1e6, 0.1]]
        ),
    )

    assert first.selected_candidate_index == second.selected_candidate_index == 0
    assert_array_equal(
        first.discovery_selection_scores, second.discovery_selection_scores
    )
    assert first.selected_confirmation_statistic != second.selected_confirmation_statistic


def test_adaptive_confirmation_requires_predeclared_count_and_fraction() -> None:
    spectral = np.full((6, 1), 2.0)
    control = np.full((6, 1), 1.5)
    scorable = np.ones((6, 1), dtype=bool)
    scorable[[1, 5], 0] = False
    kwargs = {
        "candidate_names": ("spatial_std",),
        "window_indices": np.arange(6),
        "discovery_windows": np.asarray([True, False, True, False, True, False]),
        "spectral_ratios": spectral,
        "control_ratios": control,
        "min_discovery_windows": 3,
        "min_discovery_spectral_ratio": 1.0,
        "min_discovery_control_ratio": 1.0,
        "min_confirmation_windows": 2,
        "min_confirmation_scorable_fraction": 0.5,
    }

    rejected = evaluate_photometry_family(**kwargs, scorable=scorable)
    accepted = evaluate_photometry_family(
        **kwargs, scorable=np.ones((6, 1), dtype=bool)
    )

    assert rejected.selected_candidate_index == 0
    assert rejected.selected_confirmation_window_count == 1
    assert rejected.total_confirmation_window_count == 3
    assert rejected.selected_confirmation_scorable_fraction == 1.0 / 3.0
    assert not rejected.selected_confirmation_gate_passed
    assert rejected.selected_confirmation_statistic == -np.inf
    assert accepted.selected_confirmation_window_count == 3
    assert accepted.selected_confirmation_scorable_fraction == 1.0
    assert accepted.selected_confirmation_gate_passed
    assert np.isfinite(accepted.selected_confirmation_statistic)


def test_photometry_surrogate_shifts_missingness_with_trace_exactly() -> None:
    base = _dataset()
    traces = np.asarray(base.traces, dtype=np.float64).copy()
    valid = np.ones(traces.shape, dtype=bool)
    valid[15:29, 0] = False
    valid[80:87, 0] = False
    valid[42:58, 3] = False
    valid[120:133, 6] = False
    traces[~valid] = np.nan
    dataset = replace(base, traces=traces, pixel_valid=valid).validated()

    surrogate = autocorrelation_preserving_surrogate(
        dataset,
        np.ones(dataset.frame_count, dtype=bool),
        rng=np.random.default_rng(509),
        spatial_block_px=2,
        min_shift_seconds=0.5,
        max_gap_factor=1.75,
    )

    assert_array_equal(
        np.sum(surrogate.pixel_valid, axis=0), np.sum(dataset.pixel_valid, axis=0)
    )
    for pixel in range(dataset.pixel_count):
        matches = [
            shift
            for shift in range(dataset.frame_count)
            if np.array_equal(
                np.roll(dataset.pixel_valid[:, pixel], shift),
                surrogate.pixel_valid[:, pixel],
            )
            and np.allclose(
                np.roll(dataset.traces[:, pixel], shift),
                surrogate.traces[:, pixel],
                equal_nan=True,
            )
        ]
        assert matches, f"pixel {pixel} validity and trace were not shifted together"


def test_all_nan_huber_rows_are_warning_free_under_threads() -> None:
    values = np.arange(96, dtype=np.float64).reshape(12, 8)
    valid = np.ones(values.shape, dtype=bool)
    valid[[1, 4, 7, 10]] = False
    values[~valid] = np.nan
    region = np.ones(values.shape[1], dtype=bool)

    def pool() -> np.ndarray:
        return regional_pool(
            values, region, valid=valid, method="huber", min_valid_pixels=3
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _index: pool(), range(16)))

    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert all(np.allclose(result, results[0], equal_nan=True) for result in results)
    assert np.isnan(results[0][[1, 4, 7, 10]]).all()


def test_global_indices_are_batch_and_worker_invariant() -> None:
    dataset = _dataset()
    active = np.ones(dataset.frame_count, dtype=bool)
    whole = compute_photometry_family_null_batch(
        dataset,
        active,
        surrogate_indices=np.arange(6),
        seed=117,
        scorer=_lightweight_scorer,
        spatial_block_px=2,
        min_shift_seconds=0.5,
        workers=1,
    )
    first = compute_photometry_family_null_batch(
        dataset,
        active,
        surrogate_indices=np.arange(3),
        seed=117,
        scorer=_lightweight_scorer,
        spatial_block_px=2,
        min_shift_seconds=0.5,
        workers=2,
    )
    second = compute_photometry_family_null_batch(
        dataset,
        active,
        surrogate_indices=np.arange(3, 6),
        seed=117,
        scorer=_lightweight_scorer,
        spatial_block_px=2,
        min_shift_seconds=0.5,
        workers=2,
    )

    for field in (
        "maximum_cell_statistics",
        "selected_confirmation_statistics",
        "selected_candidate_indices",
        "selected_confirmation_window_counts",
        "selected_confirmation_scorable_fractions",
        "selected_confirmation_gate_passed",
        "maximum_window_indices",
        "maximum_candidate_indices",
    ):
        expected = getattr(whole, field)
        combined = np.concatenate([getattr(first, field), getattr(second, field)])
        assert_allclose(combined, expected, rtol=0.0, atol=0.0)


def test_batch_round_trip_rejects_stale_identity(tmp_path) -> None:
    dataset = _dataset()
    batch = compute_photometry_family_null_batch(
        dataset,
        np.ones(dataset.frame_count, dtype=bool),
        surrogate_indices=(4, 5),
        seed=3,
        scorer=_lightweight_scorer,
    )
    path = tmp_path / "batch.npz"
    write_photometry_null_batch(path, identity="current", batch=batch)

    loaded = load_photometry_null_batch(
        path, identity="current", expected_indices=np.asarray([4, 5])
    )

    assert loaded is not None
    assert_allclose(loaded.maximum_cell_statistics, batch.maximum_cell_statistics)
    assert (
        load_photometry_null_batch(
            path, identity="stale", expected_indices=np.asarray([4, 5])
        )
        is None
    )
    assert (
        load_photometry_null_batch(
            path, identity="current", expected_indices=np.asarray([5, 6])
        )
        is None
    )


def test_familywise_p_values_use_global_maximum_distribution() -> None:
    observed = np.asarray([[1.0, 2.0], [3.0, np.nan]])
    null_maximum = np.asarray([0.5, 1.5, 2.5, 3.5])

    result = familywise_p_values(observed, null_maximum)

    assert_allclose(result[:2, :1], np.asarray([[0.8], [0.4]]))
    assert result[0, 1] == 0.6
    assert np.isnan(result[1, 1])
