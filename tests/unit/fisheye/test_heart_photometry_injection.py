from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from fisheye.analysis.heart_photometry_injection import (
    PhotometryInjectionSpec,
    activity_envelope,
    circular_difference_rad,
    estimate_monotonic_phase_slope,
    inject_mono8_photometry,
    ordered_spatial_bands,
    phase_equivalent_timing_error_ms,
    select_discovery_family,
    stable_spec_id,
)
from fisheye.analysis.local_rostral_heartrate import LocalCoordinateDataset


_PLAYGROUND = (
    Path(__file__).resolve().parents[3] / "playgrounds" / "heartrate_stabilization"
)
sys.path.insert(0, str(_PLAYGROUND))
import inject_recover_heart_photometry as runner  # noqa: E402


def _dataset() -> LocalCoordinateDataset:
    frame_count = 600
    pixel_xy = np.asarray(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 2],
            [1, 2],
            [3, 0],
            [4, 0],
        ],
        dtype=np.float64,
    )
    timestamps = np.arange(frame_count, dtype=np.float64) / 60.0
    traces = np.full((frame_count, pixel_xy.shape[0]), 100.0, dtype=np.float64)
    traces += 0.1 * np.sin(2.0 * np.pi * 0.4 * timestamps)[:, None]
    pixel_valid = np.ones(traces.shape, dtype=bool)
    frame_valid = np.ones(frame_count, dtype=bool)
    frame_valid[200:210] = False
    pixel_valid[200:210] = False
    traces[~pixel_valid] = np.nan
    source_xy = np.broadcast_to(
        pixel_xy[None], (frame_count, pixel_xy.shape[0], 2)
    ).copy()
    weights = np.full((frame_count, pixel_xy.shape[0], 4), 0.25, dtype=np.float64)
    return LocalCoordinateDataset(
        frame_indices=np.arange(frame_count, dtype=np.int64),
        timestamps_s=timestamps,
        traces=traces,
        pixel_xy=pixel_xy,
        pixel_valid=pixel_valid,
        frame_valid=frame_valid,
        source_xy=source_xy,
        bilinear_weights=weights,
        body_occupancy=np.ones(traces.shape, dtype=np.float64),
        eye_occupancy=np.zeros(traces.shape, dtype=np.float64),
        gradient_magnitude=np.ones(traces.shape, dtype=np.float64),
        motion_prediction=np.zeros(traces.shape, dtype=np.float64),
        nuisance_values=np.column_stack(
            [np.sin(2.0 * np.pi * 0.4 * timestamps), timestamps]
        ),
        nuisance_names=("global", "drift"),
        image_shape_hw=(3, 5),
        administrative_boundary_distance_px=np.ones(pixel_xy.shape[0]),
        physical_boundary_distance_px=np.ones(pixel_xy.shape[0]),
        transform_uncertainty=np.linspace(0.0, 1.0, frame_count),
        metadata={"pixel_format": "mono8"},
    ).validated()


def _masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.arange(8) < 6
    upper = np.arange(8) < 3
    lower = (np.arange(8) >= 3) & (np.arange(8) < 6)
    return target, upper, lower


def test_zero_amplitude_is_exact_and_preserves_observed_contract() -> None:
    dataset = _dataset()
    target, upper, lower = _masks()
    injected, truth = inject_mono8_photometry(
        dataset,
        target,
        PhotometryInjectionSpec(
            pattern="fixed_upper_lower_delay",
            frequency_hz=3.0,
            amplitude_dn=0.0,
        ),
        upper_mask=upper,
        lower_mask=lower,
    )

    assert_allclose(injected.traces, dataset.traces, rtol=0.0, atol=0.0, equal_nan=True)
    assert_array_equal(injected.pixel_valid, dataset.pixel_valid)
    assert_array_equal(injected.frame_valid, dataset.frame_valid)
    assert_allclose(injected.nuisance_values, dataset.nuisance_values, rtol=0.0, atol=0.0)
    assert_allclose(injected.source_xy, dataset.source_xy, rtol=0.0, atol=0.0)
    assert_allclose(
        injected.transform_uncertainty,
        dataset.transform_uncertainty,
        rtol=0.0,
        atol=0.0,
    )
    assert_allclose(truth.injected_delta_dn, 0.0, rtol=0.0, atol=0.0)


def test_upper_lower_patterns_have_declared_circular_offset() -> None:
    dataset = _dataset()
    target, upper, lower = _masks()
    cases = (
        ("synchronous", 0.0),
        ("opposite_upper_lower", np.pi),
        ("fixed_upper_lower_delay", np.pi / 3.0),
    )
    for pattern, expected in cases:
        _injected, truth = inject_mono8_photometry(
            dataset,
            target,
            PhotometryInjectionSpec(
                pattern=pattern,
                frequency_hz=3.0,
                amplitude_dn=1.0,
                upper_lower_delay_rad=np.pi / 3.0,
            ),
            upper_mask=upper,
            lower_mask=lower,
        )
        assert abs(float(circular_difference_rad(truth.expected_upper_lower_phase_rad, expected))) < 1e-12


def test_balanced_opposite_pattern_has_no_pooled_target_phase_reference() -> None:
    dataset = _dataset()
    target, upper, lower = _masks()
    _injected, truth = inject_mono8_photometry(
        dataset,
        target,
        PhotometryInjectionSpec(
            pattern="opposite_upper_lower",
            frequency_hz=3.0,
            amplitude_dn=1.0,
        ),
        upper_mask=upper,
        lower_mask=lower,
    )

    assert truth.target_phase_resultant < 1e-12
    assert math.isnan(truth.expected_target_phase_rad)
    assert abs(
        float(circular_difference_rad(truth.expected_upper_lower_phase_rad, np.pi))
    ) < 1e-12


def test_traveling_wave_uses_balanced_monotonic_bands() -> None:
    dataset = _dataset()
    target, upper, lower = _masks()
    injected, truth = inject_mono8_photometry(
        dataset,
        target,
        PhotometryInjectionSpec(
            pattern="traveling_wave",
            frequency_hz=3.0,
            amplitude_dn=2.0,
            traveling_band_count=3,
            traveling_phase_span_rad=2.0 * np.pi / 3.0,
            traveling_direction=-1,
        ),
        upper_mask=upper,
        lower_mask=lower,
    )

    assert truth.band_count == 3
    assert_array_equal(np.bincount(truth.band_index_by_pixel[target]), [2, 2, 2])
    phases = np.asarray(
        [
            np.mean(truth.phase_by_pixel_rad[truth.band_index_by_pixel == index])
            for index in range(3)
        ]
    )
    assert_allclose(np.diff(phases), -np.pi / 3.0, rtol=0.0, atol=1e-12)
    assert_allclose(
        truth.expected_band_phase_slope_rad,
        -np.pi / 3.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isnan(injected.traces[~injected.pixel_valid]).all()


def test_ordered_bands_sort_top_to_bottom_with_deterministic_ties() -> None:
    dataset = _dataset()
    target, _upper, _lower = _masks()
    labels = ordered_spatial_bands(dataset.pixel_xy, target, band_count=3, axis="y")

    assert_array_equal(labels[target], [0, 0, 1, 1, 2, 2])
    assert np.all(labels[~target] == -1)


def test_intermittent_envelope_has_cosine_ramps_and_fixed_off_interval() -> None:
    timestamps = np.arange(0.0, 12.0, 0.25)
    spec = PhotometryInjectionSpec(
        pattern="synchronous",
        frequency_hz=3.0,
        amplitude_dn=1.0,
        activity_mode="intermittent",
        intermittent_period_seconds=8.0,
        intermittent_active_fraction=0.5,
        intermittent_ramp_seconds=0.5,
    )
    envelope = activity_envelope(timestamps, spec, time_origin_s=0.0)

    assert envelope[0] == 0.0
    assert 0.0 < envelope[1] < 1.0
    assert envelope[2] == 1.0
    assert np.all(envelope[(timestamps >= 4.0) & (timestamps < 8.0)] == 0.0)
    assert envelope[np.flatnonzero(timestamps == 8.0)[0]] == 0.0


def test_discovery_family_selection_cannot_see_confirmation_values() -> None:
    candidates = ("spatial_std", "derivative")
    frequencies = np.asarray([2.8, 3.0, 3.2])
    spectral = np.ones((2, 4, 3), dtype=np.float64)
    control = np.ones_like(spectral)
    discovery = np.asarray([True, False, True, False])
    spectral[0, discovery, 1] = 2.0
    control[0, discovery, 1] = 1.5
    spectral[1, discovery, 2] = 1.7
    control[1, discovery, 2] = 1.2
    first = select_discovery_family(
        spectral,
        control,
        candidates,
        frequencies,
        discovery,
        minimum_windows=2,
    )
    changed_spectral = spectral.copy()
    changed_control = control.copy()
    changed_spectral[1, ~discovery, 2] = 1e9
    changed_control[1, ~discovery, 2] = 1e9
    changed_spectral[0, ~discovery, 1] = 1e-9
    changed_control[0, ~discovery, 1] = 1e-9
    second = select_discovery_family(
        changed_spectral,
        changed_control,
        candidates,
        frequencies,
        discovery,
        minimum_windows=2,
    )

    assert first == second
    assert first.selected_candidate == "spatial_std"
    assert first.selected_frequency_hz == 3.0


def test_family_gate_permits_no_detection() -> None:
    metrics = np.full((2, 4, 3), 1.2, dtype=np.float64)
    result = select_discovery_family(
        metrics,
        metrics,
        ("one", "two"),
        np.asarray([2.8, 3.0, 3.2]),
        np.asarray([True, False, True, False]),
        minimum_windows=2,
        minimum_spectral_ratio=1.5,
        minimum_control_ratio=1.1,
    )

    assert result.selected_candidate is None
    assert result.passing_pair_count == 0


def test_confirmation_gate_requires_predeclared_count_and_fraction() -> None:
    spectral = np.full((1, 4, 1), 2.0, dtype=np.float64)
    control = np.full_like(spectral, 2.0)
    confirmation = np.asarray([False, True, False, True])
    spectral[0, 3, 0] = np.nan

    insufficient = runner._confirmation_summary(
        spectral,
        control,
        0,
        0,
        confirmation,
        minimum_windows=2,
        minimum_fraction=0.75,
        minimum_spectral_ratio=1.5,
        minimum_control_ratio=1.1,
    )
    spectral[0, 3, 0] = 2.0
    sufficient = runner._confirmation_summary(
        spectral,
        control,
        0,
        0,
        confirmation,
        minimum_windows=2,
        minimum_fraction=0.75,
        minimum_spectral_ratio=1.5,
        minimum_control_ratio=1.1,
    )

    assert insufficient["confirmation_window_count_total"] == 2
    assert insufficient["confirmation_window_count_valid"] == 1
    assert insufficient["confirmation_window_fraction_valid"] == 0.5
    assert insufficient["confirmation_signal_gate_passed"]
    assert not insufficient["confirmation_coverage_gate_passed"]
    assert not insufficient["confirmation_gate_passed"]
    assert sufficient["confirmation_window_count_valid"] == 2
    assert sufficient["confirmation_window_fraction_valid"] == 1.0
    assert sufficient["confirmation_gate_passed"]


def test_phase_gate_rejects_opposing_windows_and_low_within_window_coherence() -> None:
    partition = np.ones(4, dtype=bool)
    blocks = np.full(4, 4, dtype=np.int32)
    fractions = np.ones(4, dtype=np.float64)
    opposing, _ = runner._phase_partition_summary(
        np.asarray([0.0, np.pi, 0.0, np.pi]),
        np.ones(4, dtype=np.float64),
        blocks,
        blocks,
        fractions,
        blocks,
        fractions,
        partition,
        minimum_resultant=0.5,
        minimum_valid_blocks=2,
        minimum_valid_block_fraction=0.5,
        minimum_valid_windows=3,
        minimum_valid_window_fraction=0.75,
    )
    incoherent, _ = runner._phase_partition_summary(
        np.zeros(4, dtype=np.float64),
        np.full(4, 0.05, dtype=np.float64),
        blocks,
        blocks,
        fractions,
        blocks,
        fractions,
        partition,
        minimum_resultant=0.5,
        minimum_valid_blocks=2,
        minimum_valid_block_fraction=0.5,
        minimum_valid_windows=3,
        minimum_valid_window_fraction=0.75,
    )
    insufficient_blocks, _ = runner._phase_partition_summary(
        np.zeros(4, dtype=np.float64),
        np.ones(4, dtype=np.float64),
        np.ones(4, dtype=np.int32),
        blocks,
        np.full(4, 0.25, dtype=np.float64),
        np.ones(4, dtype=np.int32),
        np.full(4, 0.25, dtype=np.float64),
        partition,
        minimum_resultant=0.5,
        minimum_valid_blocks=2,
        minimum_valid_block_fraction=0.5,
        minimum_valid_windows=3,
        minimum_valid_window_fraction=0.75,
    )

    assert opposing["phase_window_count_valid"] == 4
    assert opposing["phase_measured_resultant"] < 1e-12
    assert not opposing["phase_available"]
    assert math.isnan(opposing["phase_mean_rad"])
    assert opposing["phase_unavailable_reason"] == (
        "partition_phase_resultant_below_predeclared_threshold"
    )
    assert incoherent["phase_window_count_valid"] == 0
    assert_allclose(incoherent["phase_within_window_coherence_median"], 0.05)
    assert not incoherent["phase_available"]
    assert math.isnan(incoherent["phase_mean_rad"])
    assert incoherent["phase_unavailable_reason"] == (
        "no_windows_passed_phase_coherence_gate"
    )
    assert insufficient_blocks["phase_window_count_valid"] == 0
    assert not insufficient_blocks["phase_available"]
    assert insufficient_blocks["phase_unavailable_reason"] == (
        "no_windows_passed_phase_signal_support_gates"
    )


def test_timebase_validation_reports_actual_nyquist_and_rejects_invalid_support() -> None:
    timestamps = np.arange(1000, dtype=np.float64) / 100.0
    frames = np.arange(1000, dtype=np.int64)
    diagnostics = runner._validate_timebase(
        timestamps,
        frames,
        np.asarray([2.0, 3.0, 4.0]),
        maximum_relative_jitter=0.05,
    )

    assert_allclose(diagnostics["timestamp_derived_sample_rate_hz"], 100.0)
    assert_allclose(diagnostics["timestamp_derived_nyquist_hz"], 50.0)
    assert diagnostics["tested_grid_below_nyquist"] is True
    with pytest.raises(ValueError, match="Nyquist"):
        runner._validate_timebase(
            timestamps,
            frames,
            np.asarray([2.0, 50.0]),
            maximum_relative_jitter=0.05,
        )
    duplicated = timestamps.copy()
    duplicated[10] = duplicated[9]
    with pytest.raises(ValueError, match="strictly increasing"):
        runner._validate_timebase(
            duplicated,
            frames,
            np.asarray([2.0, 3.0]),
            maximum_relative_jitter=0.05,
        )
    irregular_spacing = np.tile([0.005, 0.015], 500)
    irregular = np.cumsum(irregular_spacing)
    with pytest.raises(ValueError, match="relative jitter"):
        runner._validate_timebase(
            irregular,
            frames,
            np.asarray([2.0, 3.0]),
            maximum_relative_jitter=0.05,
        )


def test_grid_measurement_records_phase_valid_block_support() -> None:
    timestamps = np.arange(1200, dtype=np.float64) / 100.0
    target = np.sin(2.0 * np.pi * 3.0 * timestamps)
    traces = runner.comparison.TraceSet(
        target=target,
        upper=target,
        lower=np.sin(2.0 * np.pi * 3.0 * timestamps + np.pi / 3.0),
        control=np.sin(2.0 * np.pi * 1.0 * timestamps),
    )
    dataset = SimpleNamespace(
        timestamps_s=timestamps,
        frame_valid=np.ones(timestamps.size, dtype=bool),
    )
    frequencies = np.asarray([2.5, 3.0, 3.5])

    measured = runner._measure_grid(
        dataset,
        traces,
        frequencies,
        block_seconds=4.0,
        min_block_seconds=2.0,
        min_valid_fraction=0.7,
        max_interpolated_gap_seconds=0.02,
        truth_frequency_hz=3.0,
        truth_target_phase_rad=0.0,
        truth_time_origin_s=0.0,
        activity_envelope=np.ones(timestamps.size, dtype=np.float64),
        truth_amplitude_dn=1.0,
        minimum_phase_support_ratio=1.5,
        minimum_phase_coefficient_amplitude=1e-8,
    )

    truth_index = 1
    assert measured.block_count == 3
    assert measured.phase_valid_block_count[truth_index] == 3
    assert measured.phase_total_block_count[truth_index] == 3
    assert measured.phase_valid_block_fraction[truth_index] == 1.0
    assert measured.phase_support_qualified_block_count[truth_index] == 3
    assert measured.upper_phase_support_ratio_median[truth_index] > 1.5
    assert measured.lower_phase_support_ratio_median[truth_index] > 1.5
    assert measured.phase_locking_value[truth_index] > 0.99
    assert measured.target_truth_valid_block_count[truth_index] == 3
    assert measured.target_truth_total_block_count[truth_index] == 3
    assert measured.target_truth_valid_block_fraction[truth_index] == 1.0
    assert measured.target_truth_support_qualified_block_count[truth_index] == 3
    assert measured.target_truth_phase_coherence[truth_index] > 0.99


@pytest.mark.parametrize("weak_scale", [0.0, 1e-12])
def test_zero_or_near_zero_region_cannot_produce_phase_or_timing(
    weak_scale: float,
) -> None:
    timestamps = np.arange(1200, dtype=np.float64) / 100.0
    signal = np.sin(2.0 * np.pi * 3.0 * timestamps)
    weak = weak_scale * signal
    dataset = SimpleNamespace(
        timestamps_s=timestamps,
        frame_valid=np.ones(timestamps.size, dtype=bool),
    )
    frequencies = np.asarray([2.5, 3.0, 3.5])
    regional = runner._measure_grid(
        dataset,
        runner.comparison.TraceSet(
            target=signal,
            upper=weak,
            lower=signal,
            control=np.sin(2.0 * np.pi * timestamps),
        ),
        frequencies,
        block_seconds=4.0,
        min_block_seconds=2.0,
        min_valid_fraction=0.7,
        max_interpolated_gap_seconds=0.02,
        truth_frequency_hz=3.0,
        truth_target_phase_rad=0.0,
        truth_time_origin_s=0.0,
        activity_envelope=np.ones(timestamps.size),
        truth_amplitude_dn=1.0,
        minimum_phase_support_ratio=1.5,
        minimum_phase_coefficient_amplitude=1e-8,
    )
    pooled = runner._measure_grid(
        dataset,
        runner.comparison.TraceSet(
            target=weak,
            upper=signal,
            lower=signal,
            control=np.sin(2.0 * np.pi * timestamps),
        ),
        frequencies,
        block_seconds=4.0,
        min_block_seconds=2.0,
        min_valid_fraction=0.7,
        max_interpolated_gap_seconds=0.02,
        truth_frequency_hz=3.0,
        truth_target_phase_rad=0.0,
        truth_time_origin_s=0.0,
        activity_envelope=np.ones(timestamps.size),
        truth_amplitude_dn=1.0,
        minimum_phase_support_ratio=1.5,
        minimum_phase_coefficient_amplitude=1e-8,
    )

    assert regional.phase_support_qualified_block_count[1] == 0
    assert regional.phase_valid_block_count[1] == 0
    assert math.isnan(regional.phase_offset_rad[1])
    assert pooled.target_truth_support_qualified_block_count[1] == 0
    assert pooled.target_truth_valid_block_count[1] == 0
    assert math.isnan(pooled.target_truth_residual_rad[1])


@pytest.mark.parametrize("weak_scale", [0.0, 1e-12])
def test_zero_or_near_zero_travel_band_cannot_produce_slope_or_direction(
    weak_scale: float,
) -> None:
    timestamps = np.arange(1200, dtype=np.float64) / 100.0
    signal = np.sin(2.0 * np.pi * 3.0 * timestamps)
    dataset = SimpleNamespace(
        timestamps_s=timestamps,
        frame_valid=np.ones(timestamps.size, dtype=bool),
        frame_count=timestamps.size,
    )
    result = runner._band_slope_grid(
        dataset,
        (
            signal,
            weak_scale * signal,
            np.sin(2.0 * np.pi * 3.0 * timestamps + np.pi / 3.0),
        ),
        np.asarray([2.5, 3.0, 3.5]),
        block_seconds=4.0,
        min_block_seconds=2.0,
        min_valid_fraction=0.7,
        max_interpolated_gap_seconds=0.02,
        minimum_phase_support_ratio=1.5,
        minimum_phase_coefficient_amplitude=1e-8,
    )
    (
        slope,
        direction_consistency,
        _coherence,
        valid_count,
        _total_count,
        _valid_fraction,
        support_count,
        _support_fraction,
        _band_support,
        band_amplitude,
    ) = result

    assert support_count[1] == 0
    assert valid_count[1] == 0
    assert math.isnan(slope[1])
    assert math.isnan(direction_consistency[1])
    assert band_amplitude[1, 1] < 1e-8


def test_phase_slope_and_nearest_cycle_timing_are_circular() -> None:
    phases = np.asarray([2.8, -3.0, -2.5, -2.0])
    slope = estimate_monotonic_phase_slope(phases)
    timing = phase_equivalent_timing_error_ms(
        np.deg2rad(-179.0),
        np.deg2rad(179.0),
        frequency_hz=2.0,
    )

    assert_allclose(slope, 0.5, atol=0.03)
    assert_allclose(timing, 1000.0 * (2.0 / 360.0) / 2.0, atol=1e-12)


def test_stable_spec_id_ignores_mapping_order() -> None:
    assert stable_spec_id({"a": 1, "b": 2}) == stable_spec_id({"b": 2, "a": 1})


def test_clipping_is_limited_to_valid_selected_samples() -> None:
    dataset = _dataset()
    target, upper, lower = _masks()
    traces = np.asarray(dataset.traces).copy()
    traces[:, target] = 254.9
    traces[~np.asarray(dataset.pixel_valid)] = np.nan
    near_ceiling = replace(dataset, traces=traces).validated()
    injected, truth = inject_mono8_photometry(
        near_ceiling,
        target,
        PhotometryInjectionSpec(
            pattern="synchronous",
            frequency_hz=3.0,
            amplitude_dn=5.0,
        ),
        upper_mask=upper,
        lower_mask=lower,
    )

    assert np.nanmax(injected.traces[:, target]) <= 255.0
    assert truth.clipped_sample_count > 0
    assert_allclose(
        injected.traces[:, ~target],
        near_ceiling.traces[:, ~target],
        equal_nan=True,
    )


def test_job_matrix_canonicalizes_duplicate_zero_amplitude_trials() -> None:
    args = SimpleNamespace(
        patterns="synchronous,traveling_wave",
        injection_frequencies_hz="2.5,3.0",
        amplitudes_dn="0",
        activity_modes="continuous,intermittent",
        traveling_band_counts="3,5",
        traveling_directions="-1,1",
        replicates=2,
        seed=11,
        batch_count=1,
        batch_index=0,
    )

    jobs = runner._jobs(args)

    assert len(jobs) == 1
    assert jobs[0].replicate == 0
    assert all(job.pattern == "synchronous" for job in jobs)
    assert all(job.activity_mode == "continuous" for job in jobs)
    assert all(job.amplitude_dn == 0.0 for job in jobs)


def test_resume_identity_fails_closed_on_experiment_change(tmp_path: Path) -> None:
    job = runner.InjectionJob(
        pattern="synchronous",
        frequency_hz=3.0,
        amplitude_dn=1.0,
        activity_mode="continuous",
        replicate=0,
        initial_phase_rad=0.0,
        traveling_band_count=5,
        traveling_direction=1,
    )
    runner._save_job(
        tmp_path,
        job,
        {"metric": np.asarray([1.0])},
        {"job": job.payload(), "experiment_id": "experiment-a"},
    )

    loaded = runner._load_job(tmp_path, job, "experiment-a")
    assert loaded is not None
    assert_allclose(loaded[0]["metric"], [1.0])
    with np.load(tmp_path / f"{job.job_id}.npz", allow_pickle=False) as data:
        assert str(data["schema_name"].item()) == runner._JOB_NPZ_SCHEMA
        assert int(data["schema_version"].item()) == runner._JOB_NPZ_SCHEMA_VERSION
        assert int(data["experiment_schema_version"].item()) == (
            runner._EXPERIMENT_SCHEMA_VERSION
        )
        assert str(data["experiment_id"].item()) == "experiment-a"
    with pytest.raises(ValueError, match="experiment mismatch"):
        runner._load_job(tmp_path, job, "experiment-b")

    np.savez_compressed(
        tmp_path / f"{job.job_id}.npz",
        schema_name=np.asarray(runner._JOB_NPZ_SCHEMA),
        schema_version=np.asarray(runner._JOB_NPZ_SCHEMA_VERSION),
        experiment_schema_version=np.asarray(runner._EXPERIMENT_SCHEMA_VERSION),
        experiment_id=np.asarray("experiment-b"),
        metric=np.asarray([1.0]),
    )
    with pytest.raises(ValueError, match="NPZ experiment mismatch"):
        runner._load_job(tmp_path, job, "experiment-a")


def test_consolidated_npz_embeds_schema_experiment_and_runtime_versions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "consolidated.npz"
    runner._write_consolidated_npz(
        path,
        experiment_id="experiment-123",
        job_ids=np.asarray(["job-a"]),
        candidate_names=np.asarray(["candidate-a"]),
        frequencies_hz=np.asarray([3.0]),
        window_indices=np.asarray([0], dtype=np.int32),
        window_roles=np.asarray(["confirmation"]),
        arrays={"spectral_ratio": np.asarray([[[[2.0]]]])},
    )

    with np.load(path, allow_pickle=False) as data:
        assert str(data["schema_name"].item()) == runner._CONSOLIDATED_NPZ_SCHEMA
        assert int(data["schema_version"].item()) == (
            runner._CONSOLIDATED_NPZ_SCHEMA_VERSION
        )
        assert int(data["experiment_schema_version"].item()) == (
            runner._EXPERIMENT_SCHEMA_VERSION
        )
        assert str(data["experiment_id"].item()) == "experiment-123"
        assert str(data["numpy_version"].item()) == np.__version__
        assert str(data["scipy_version"].item()) == runner.scipy.__version__
        assert_allclose(data["spectral_ratio"], [[[[2.0]]]])


def test_implementation_hash_changes_resume_experiment_identity(tmp_path: Path) -> None:
    implementation = tmp_path / "implementation.py"
    implementation.write_text("VALUE = 1\n")
    first = runner._implementation_code_identity({"fixture": implementation})
    first_experiment = stable_spec_id({"implementation_code": first})

    implementation.write_text("VALUE = 2\n")
    second = runner._implementation_code_identity({"fixture": implementation})
    second_experiment = stable_spec_id({"implementation_code": second})

    assert len(first["fixture"]["sha256"]) == 64
    assert first["fixture"]["sha256"] != second["fixture"]["sha256"]
    assert first_experiment != second_experiment

    job = runner.InjectionJob(
        pattern="synchronous",
        frequency_hz=3.0,
        amplitude_dn=1.0,
        activity_mode="continuous",
        replicate=0,
        initial_phase_rad=0.0,
        traveling_band_count=5,
        traveling_direction=1,
    )
    runner._save_job(
        tmp_path / "jobs",
        job,
        {"metric": np.asarray([1.0])},
        {"job": job.payload(), "experiment_id": first_experiment},
    )
    with pytest.raises(ValueError, match="experiment mismatch"):
        runner._load_job(tmp_path / "jobs", job, second_experiment)


def test_runtime_versions_participate_in_experiment_identity() -> None:
    runtime = runner._runtime_version_identity()
    baseline = stable_spec_id({"runtime_versions": runtime})
    changed = stable_spec_id(
        {"runtime_versions": {**runtime, "numpy": runtime["numpy"] + "+changed"}}
    )

    assert runtime == {
        "numpy": np.__version__,
        "scipy": runner.scipy.__version__,
    }
    assert baseline != changed


def test_opposite_phase_disables_pooled_timing_but_keeps_regional_phase() -> None:
    candidate_count = len(runner._CANDIDATE_NAMES)
    window_count = 4
    frequency_count = 1
    shape = (candidate_count, window_count, frequency_count)
    arrays = {
        "spectral_ratio": np.full(shape, 2.0, dtype=np.float64),
        "control_ratio": np.full(shape, 2.0, dtype=np.float64),
        "phase_offset_rad": np.full(shape, np.pi, dtype=np.float64),
        "phase_locking_value": np.ones(shape, dtype=np.float64),
        "phase_valid_block_count": np.full(shape, 4, dtype=np.int32),
        "phase_total_block_count": np.full(shape, 4, dtype=np.int32),
        "phase_valid_block_fraction": np.ones(shape, dtype=np.float64),
        "phase_support_qualified_block_count": np.full(shape, 4, dtype=np.int32),
        "phase_support_qualified_block_fraction": np.ones(shape, dtype=np.float64),
        "upper_phase_support_ratio_median": np.full(shape, 2.0),
        "lower_phase_support_ratio_median": np.full(shape, 2.0),
        "upper_phase_coefficient_amplitude_median": np.ones(shape),
        "lower_phase_coefficient_amplitude_median": np.ones(shape),
        "target_truth_residual_rad": np.full(shape, 0.2, dtype=np.float64),
        "target_truth_phase_coherence": np.ones(shape, dtype=np.float64),
        "target_truth_valid_block_count": np.full(shape, 4, dtype=np.int32),
        "target_truth_total_block_count": np.full(shape, 4, dtype=np.int32),
        "target_truth_valid_block_fraction": np.ones(shape, dtype=np.float64),
        "target_truth_support_qualified_block_count": np.full(
            shape, 4, dtype=np.int32
        ),
        "target_truth_support_qualified_block_fraction": np.ones(
            shape, dtype=np.float64
        ),
        "target_phase_support_ratio_median": np.full(shape, 2.0),
        "target_phase_coefficient_amplitude_median": np.ones(shape),
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
            (candidate_count, window_count, frequency_count, 5), np.nan
        ),
        "band_phase_coefficient_amplitude_median": np.full(
            (candidate_count, window_count, frequency_count, 5), np.nan
        ),
        "target_phase_resultant": np.zeros(window_count, dtype=np.float64),
        "expected_upper_lower_phase_rad": np.full(
            window_count, np.pi, dtype=np.float64
        ),
        "expected_band_phase_slope_rad": np.full(
            window_count, np.nan, dtype=np.float64
        ),
    }
    args = SimpleNamespace(
        skip_matched_projection=False,
        min_discovery_windows=2,
        min_confirmation_windows=2,
        min_confirmation_window_fraction=1.0,
        min_spectral_ratio=1.5,
        min_control_ratio=1.1,
        min_pooled_target_phase_resultant=0.1,
        min_phase_resultant=0.5,
        min_phase_region_support_ratio=1.5,
        min_phase_coefficient_amplitude=1e-8,
        min_phase_valid_blocks=2,
        min_phase_valid_block_fraction=0.5,
        min_phase_confirmation_windows=2,
        min_phase_confirmation_window_fraction=1.0,
        frequency_step_hz=0.1,
    )
    job = runner.InjectionJob(
        pattern="opposite_upper_lower",
        frequency_hz=3.0,
        amplitude_dn=1.0,
        activity_mode="continuous",
        replicate=0,
        initial_phase_rad=0.0,
        traveling_band_count=5,
        traveling_direction=1,
    )

    summary = runner._summarize_job(
        job,
        arrays,
        np.asarray([3.0]),
        np.asarray(["discovery", "confirmation", "discovery", "confirmation"]),
        np.arange(window_count),
        args,
    )
    derivative = next(
        item
        for item in summary["candidate_summaries"]
        if item["candidate"] == "huber_savgol_derivative_w11"
    )

    assert derivative["confirmation_gate_passed"]
    assert derivative["upper_phase_support_ratio_median"] == 2.0
    assert derivative["lower_phase_support_ratio_median"] == 2.0
    assert derivative["upper_lower_phase_support_qualified_block_count"] == 8
    assert not derivative["pooled_target_phase_interpretable"]
    assert not derivative["phase_timing_interpretable"]
    assert math.isnan(derivative["phase_equivalent_timing_rmse_ms"])
    assert derivative["phase_timing_unavailable_reason"] == (
        "injected_target_phase_resultant_below_predeclared_threshold"
    )
    assert_allclose(derivative["upper_lower_phase_offset_deg"], 180.0)
    assert_allclose(derivative["upper_lower_phase_error_deg"], 0.0, atol=1e-12)


def test_summary_withholds_all_phase_angles_when_measured_coherence_is_low() -> None:
    candidate_count = len(runner._CANDIDATE_NAMES)
    shape = (candidate_count, 4, 1)
    arrays = {
        "spectral_ratio": np.full(shape, 2.0, dtype=np.float64),
        "control_ratio": np.full(shape, 2.0, dtype=np.float64),
        "phase_offset_rad": np.zeros(shape, dtype=np.float64),
        "phase_locking_value": np.full(shape, 0.05, dtype=np.float64),
        "phase_valid_block_count": np.full(shape, 4, dtype=np.int32),
        "phase_total_block_count": np.full(shape, 4, dtype=np.int32),
        "phase_valid_block_fraction": np.ones(shape, dtype=np.float64),
        "phase_support_qualified_block_count": np.full(shape, 4, dtype=np.int32),
        "phase_support_qualified_block_fraction": np.ones(shape, dtype=np.float64),
        "upper_phase_support_ratio_median": np.full(shape, 2.0),
        "lower_phase_support_ratio_median": np.full(shape, 2.0),
        "upper_phase_coefficient_amplitude_median": np.ones(shape),
        "lower_phase_coefficient_amplitude_median": np.ones(shape),
        "target_truth_residual_rad": np.zeros(shape, dtype=np.float64),
        "target_truth_phase_coherence": np.full(shape, 0.05, dtype=np.float64),
        "target_truth_valid_block_count": np.full(shape, 4, dtype=np.int32),
        "target_truth_total_block_count": np.full(shape, 4, dtype=np.int32),
        "target_truth_valid_block_fraction": np.ones(shape, dtype=np.float64),
        "target_truth_support_qualified_block_count": np.full(
            shape, 4, dtype=np.int32
        ),
        "target_truth_support_qualified_block_fraction": np.ones(shape),
        "target_phase_support_ratio_median": np.full(shape, 2.0),
        "target_phase_coefficient_amplitude_median": np.ones(shape),
        "band_phase_slope_rad": np.full(shape, 0.5, dtype=np.float64),
        "band_direction_consistency": np.ones(shape, dtype=np.float64),
        "band_slope_coherence": np.full(shape, 0.05, dtype=np.float64),
        "band_slope_valid_block_count": np.full(shape, 4, dtype=np.int32),
        "band_slope_total_block_count": np.full(shape, 4, dtype=np.int32),
        "band_slope_valid_block_fraction": np.ones(shape, dtype=np.float64),
        "band_slope_support_qualified_block_count": np.full(
            shape, 4, dtype=np.int32
        ),
        "band_slope_support_qualified_block_fraction": np.ones(shape),
        "band_phase_support_ratio_median": np.full(
            (candidate_count, 4, 1, 5), 2.0
        ),
        "band_phase_coefficient_amplitude_median": np.ones(
            (candidate_count, 4, 1, 5)
        ),
        "target_phase_resultant": np.ones(4, dtype=np.float64),
        "expected_upper_lower_phase_rad": np.zeros(4, dtype=np.float64),
        "expected_band_phase_slope_rad": np.full(4, 0.5, dtype=np.float64),
    }
    args = SimpleNamespace(
        skip_matched_projection=False,
        min_discovery_windows=2,
        min_confirmation_windows=2,
        min_confirmation_window_fraction=1.0,
        min_spectral_ratio=1.5,
        min_control_ratio=1.1,
        min_pooled_target_phase_resultant=0.1,
        min_phase_resultant=0.5,
        min_phase_region_support_ratio=1.5,
        min_phase_coefficient_amplitude=1e-8,
        min_phase_valid_blocks=2,
        min_phase_valid_block_fraction=0.5,
        min_phase_confirmation_windows=2,
        min_phase_confirmation_window_fraction=1.0,
        frequency_step_hz=0.1,
    )
    job = runner.InjectionJob(
        pattern="traveling_wave",
        frequency_hz=3.0,
        amplitude_dn=1.0,
        activity_mode="continuous",
        replicate=0,
        initial_phase_rad=0.0,
        traveling_band_count=3,
        traveling_direction=1,
    )

    result = runner._summarize_job(
        job,
        arrays,
        np.asarray([3.0]),
        np.asarray(["discovery", "confirmation", "discovery", "confirmation"]),
        np.arange(4),
        args,
    )
    derivative = next(
        item
        for item in result["candidate_summaries"]
        if item["candidate"] == "huber_savgol_derivative_w11"
    )

    assert derivative["confirmation_gate_passed"]
    assert_allclose(derivative["upper_lower_within_window_coherence_median"], 0.05)
    assert not derivative["upper_lower_phase_available"]
    assert math.isnan(derivative["upper_lower_phase_offset_deg"])
    assert math.isnan(derivative["upper_lower_phase_error_deg"])
    assert not derivative["phase_timing_interpretable"]
    assert math.isnan(derivative["phase_equivalent_timing_rmse_ms"])
    assert not derivative["travel_phase_available"]
    assert derivative["travel_phase_support_qualified_block_count"] == 8
    assert derivative["travel_band_support_ratio_median"] == [2.0, 2.0, 2.0]
    assert math.isnan(derivative["band_phase_slope_rad_per_band"])
    assert math.isnan(derivative["band_phase_slope_error_rad_per_band"])
    assert derivative["travel_direction_correct"] is None
    assert derivative["upper_lower_phase_unavailable_reason"] == (
        "no_windows_passed_phase_coherence_gate"
    )
    assert derivative["travel_phase_unavailable_reason"] == (
        "no_windows_passed_phase_coherence_gate"
    )
