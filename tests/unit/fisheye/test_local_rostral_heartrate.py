from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
from scipy import ndimage, signal

from fisheye.analysis.consensus_heart_mask import (
    ConsensusMaskConfig,
    contiguous_outer_partitions,
    familywise_max_p_values,
    learn_consensus_heart_mask,
)
from fisheye.analysis.dynamic_heart_support import (
    CrossfitHeartPhaseSeries,
    analyze_dynamic_heart_support,
    compute_dynamic_support_null_batch,
    prepare_dynamic_support_null_context,
    reconstruct_crossfit_heart_phase,
)
from fisheye.analysis.local_rostral_heartrate import (
    HeartrateConfig,
    InjectionSpec,
    LocalCoordinateDataset,
    _chunk_frequency_coefficients,
    alternating_block_partitions,
    analyze_heartrate,
    autocorrelation_preserving_surrogate,
    autocorrelation_surrogate_shift_diagnostics,
    bilinear_sample,
    build_risk_surfaces,
    bridge_short_gaps,
    contiguous_segments,
    inject_local_signal,
    injection_operating_characteristics,
    run_injection_recovery,
)
from fisheye.analysis.regional_phase_delay import analyze_regional_phase_delay


def _synthetic_dataset(*, amplitude: float = 0.0, seed: int = 1) -> LocalCoordinateDataset:
    rng = np.random.default_rng(seed)
    fps = 50.0
    frame_count = 1600
    timestamps = np.arange(frame_count, dtype=np.float64) / fps
    yy, xx = np.mgrid[1:9, 1:9]
    pixel_xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float64)
    pixel_count = int(pixel_xy.shape[0])
    global_mean = np.sin(2.0 * np.pi * 0.35 * timestamps)
    body_control = signal.lfilter([1.0], [1.0, -0.75], rng.normal(0.0, 0.15, frame_count))
    external_control = rng.normal(0.0, 0.1, frame_count)
    nuisance = np.column_stack([global_mean, body_control, external_control])
    traces = 100.0 + 1.5 * global_mean[:, None] + 0.5 * body_control[:, None]
    noise = signal.lfilter(
        [1.0],
        [1.0, -0.55],
        rng.normal(0.0, 0.22, (frame_count, pixel_count)),
        axis=0,
    )
    traces = traces + noise
    cluster = (
        (pixel_xy[:, 0] >= 3)
        & (pixel_xy[:, 0] <= 5)
        & (pixel_xy[:, 1] >= 3)
        & (pixel_xy[:, 1] <= 5)
    )
    traces[:, cluster] += float(amplitude) * np.sin(2.0 * np.pi * 2.2 * timestamps)[:, None]
    image_mask = np.ones((10, 10), dtype=bool)
    distance = ndimage.distance_transform_edt(image_mask)
    boundary_distance = distance[pixel_xy[:, 1].astype(int), pixel_xy[:, 0].astype(int)]
    source_xy = np.broadcast_to(pixel_xy[None], (frame_count, pixel_count, 2)).copy()
    weights = np.full((frame_count, pixel_count, 4), 0.25, dtype=np.float32)
    return LocalCoordinateDataset(
        frame_indices=np.arange(frame_count, dtype=np.int64),
        timestamps_s=timestamps,
        traces=traces.astype(np.float32),
        pixel_xy=pixel_xy,
        pixel_valid=np.ones((frame_count, pixel_count), dtype=bool),
        frame_valid=np.ones(frame_count, dtype=bool),
        source_xy=source_xy.astype(np.float32),
        bilinear_weights=weights,
        body_occupancy=np.ones((frame_count, pixel_count), dtype=np.float32),
        eye_occupancy=np.zeros((frame_count, pixel_count), dtype=np.float32),
        gradient_magnitude=np.ones((frame_count, pixel_count), dtype=np.float32),
        motion_prediction=np.zeros((frame_count, pixel_count), dtype=np.float32),
        nuisance_values=nuisance,
        nuisance_names=("global_mean", "body_control_mean", "external_control_mean"),
        image_shape_hw=(10, 10),
        administrative_boundary_distance_px=np.zeros(pixel_count, dtype=np.float64),
        physical_boundary_distance_px=boundary_distance,
        transform_uncertainty=np.zeros(frame_count, dtype=np.float64),
        metadata={"fixture": "synthetic"},
    ).validated()


def _test_config(**overrides) -> HeartrateConfig:
    base = HeartrateConfig(
        band_min_hz=1.8,
        band_max_hz=2.6,
        frequency_step_hz=0.1,
        partition_block_seconds=4.0,
        partition_guard_seconds=0.1,
        discovery_chunk_seconds=3.5,
        min_chunk_seconds=2.5,
        min_pixel_valid_fraction=0.9,
        min_body_occupancy=0.9,
        max_eye_occupancy=0.05,
        min_physical_boundary_distance_px=1.0,
        max_warp_invalid_fraction=0.05,
        gradient_risk_weight=0.0,
        boundary_risk_weight=0.0,
        warp_risk_weight=0.0,
        transform_risk_weight=0.0,
        pixel_score_threshold_z=1.0,
        min_cluster_pixels=4,
        surrogate_count=19,
        surrogate_spatial_block_px=1,
        surrogate_min_shift_seconds=0.5,
        alpha=0.1,
        min_control_ratio=1.25,
        min_crossfit_dilated_overlap=0.5,
        event_polarity="darkening",
        event_prominence_mad=0.75,
        event_filter_edge_seconds=0.5,
        random_seed=3,
    )
    return replace(base, **overrides).validated()


def test_bilinear_sample_preserves_four_weights() -> None:
    values, valid, weights = bilinear_sample(
        np.asarray([[0.0, 2.0], [4.0, 6.0]]),
        np.asarray([[0.25, 0.75], [-1.0, 0.0]]),
    )

    assert valid.tolist() == [True, False]
    assert np.isclose(values[0], 3.5)
    assert np.isclose(np.sum(weights[0]), 1.0)
    assert np.isnan(values[1])


def test_partitions_are_disjoint_and_gaps_are_not_compressed() -> None:
    timestamps = np.arange(120, dtype=np.float64) / 10.0
    partitions = alternating_block_partitions(timestamps, block_seconds=3.0, guard_seconds=0.2)
    for discovery, confirmation in partitions:
        assert not np.any(discovery & confirmation)
    valid = np.ones(120, dtype=bool)
    valid[40:60] = False
    segments = contiguous_segments(timestamps, valid, max_gap_factor=1.5)

    assert len(segments) == 2
    assert segments[0][-1] == 39
    assert segments[1][0] == 60

    short = np.ones(120, dtype=bool)
    short[40:42] = False
    bridged, interpolated = bridge_short_gaps(timestamps, short, max_gap_seconds=0.2)
    assert np.all(bridged)
    assert np.flatnonzero(interpolated).tolist() == [40, 41]
    not_bridged, _ = bridge_short_gaps(timestamps, valid, max_gap_seconds=0.2)
    assert not np.any(not_bridged[40:60])

    fine_timestamps = np.arange(120, dtype=np.float64) / 100.0
    values = np.sin(2.0 * np.pi * 2.0 * fine_timestamps)[:, None] * np.ones((1, 2))
    values[40:42, 0] = np.nan
    values[40:50, 1] = np.nan
    coefficients = _chunk_frequency_coefficients(
        values,
        fine_timestamps,
        [np.arange(120, dtype=np.int64)],
        np.asarray([2.0]),
        min_valid_fraction=0.5,
        max_interpolated_gap_seconds=0.02,
    )
    assert np.isfinite(coefficients[0, 0, 0])
    assert np.isnan(coefficients[0, 0, 1])


def test_cached_dataset_rejects_unsupported_timestamp_nyquist() -> None:
    dataset = _synthetic_dataset()
    slow_timestamps = np.arange(dataset.frame_count, dtype=np.float64) / 4.0
    dataset = replace(dataset, timestamps_s=slow_timestamps).validated()

    result = analyze_heartrate(dataset, _test_config())

    assert result.detected is False
    assert result.reason == "target_band_exceeds_timestamp_nyquist"
    assert result.no_estimate_intervals_s


def test_short_segment_surrogates_are_randomized_not_deterministic() -> None:
    dataset = _synthetic_dataset()
    active = np.zeros(dataset.frame_count, dtype=bool)
    active[20:70] = True
    active[120:170] = True

    first = autocorrelation_preserving_surrogate(
        dataset,
        active,
        rng=np.random.default_rng(10),
        spatial_block_px=1,
        min_shift_seconds=1.0,
        max_gap_factor=1.5,
    )
    second = autocorrelation_preserving_surrogate(
        dataset,
        active,
        rng=np.random.default_rng(11),
        spatial_block_px=1,
        min_shift_seconds=1.0,
        max_gap_factor=1.5,
    )

    assert np.array_equal(first.timestamps_s, dataset.timestamps_s)
    assert np.array_equal(second.timestamps_s, dataset.timestamps_s)
    assert not np.allclose(first.traces[active], second.traces[active], equal_nan=True)


def test_surrogate_shift_bounds_use_requested_distance_when_feasible() -> None:
    base = _synthetic_dataset()
    traces = np.arange(
        base.frame_count * base.pixel_count, dtype=np.float64
    ).reshape(base.frame_count, base.pixel_count)
    dataset = replace(
        base,
        traces=traces,
        pixel_valid=np.ones(traces.shape, dtype=bool),
    ).validated()
    dt = float(np.median(np.diff(dataset.timestamps_s)))
    cases = (
        (20, 8, 8, False),
        (20, 10, 10, False),
        (20, 11, 5, True),
        (21, 10, 10, False),
        (21, 11, 5, True),
    )
    for segment_size, requested_frames, effective_frames, fallback in cases:
        active = np.zeros(dataset.frame_count, dtype=bool)
        active[:segment_size] = True
        requested_seconds = (requested_frames - 0.25) * dt
        surrogate = autocorrelation_preserving_surrogate(
            dataset,
            active,
            rng=np.random.default_rng(301 + requested_frames + segment_size),
            spatial_block_px=2,
            min_shift_seconds=requested_seconds,
            max_gap_factor=1.75,
        )
        diagnostics = autocorrelation_surrogate_shift_diagnostics(
            dataset,
            active,
            min_shift_seconds=requested_seconds,
            max_gap_factor=1.75,
        )
        assert surrogate.metadata["autocorrelation_surrogate_shift"] == diagnostics
        assert diagnostics["segment_count"] == 1
        details = diagnostics["segments"][0]
        assert details["requested_minimum_shift_frames"] == requested_frames
        assert details["effective_minimum_circular_shift_frames"] == effective_frames
        assert details["requested_minimum_feasible"] is (not fallback)
        assert details["fallback_used"] is fallback

        anchor_rows = np.flatnonzero(
            surrogate.traces[:segment_size, 0] == dataset.traces[0, 0]
        )
        assert anchor_rows.size == 1
        shift = int(anchor_rows[0])
        circular_distance = min(shift, segment_size - shift)
        assert circular_distance >= effective_frames
        if not fallback:
            assert circular_distance >= requested_frames

    active = np.zeros(dataset.frame_count, dtype=bool)
    active[:3] = True
    too_short = autocorrelation_preserving_surrogate(
        dataset,
        active,
        rng=np.random.default_rng(411),
        spatial_block_px=2,
        min_shift_seconds=dt,
        max_gap_factor=1.75,
    )
    details = too_short.metadata["autocorrelation_surrogate_shift"]["segments"][0]
    assert details["policy"] == "unchanged_too_short"
    assert details["effective_minimum_circular_shift_frames"] == 0
    assert details["requested_minimum_feasible"] is False
    assert details["fallback_used"] is False
    np.testing.assert_array_equal(too_short.traces[:3], dataset.traces[:3])


def test_surrogate_shifts_pixel_validity_with_its_trace_exactly() -> None:
    base = _synthetic_dataset()
    traces = np.asarray(base.traces, dtype=np.float64).copy()
    pixel_valid = np.ones(traces.shape, dtype=bool)
    pixel_valid[15:29, 0] = False
    pixel_valid[80:87, 0] = False
    pixel_valid[42:58, 3] = False
    pixel_valid[120:133, 6] = False
    traces[~pixel_valid] = np.nan
    dataset = replace(
        base, traces=traces, pixel_valid=pixel_valid
    ).validated()

    surrogate = autocorrelation_preserving_surrogate(
        dataset,
        np.ones(dataset.frame_count, dtype=bool),
        rng=np.random.default_rng(509),
        spatial_block_px=2,
        min_shift_seconds=0.5,
        max_gap_factor=1.75,
    )

    np.testing.assert_array_equal(
        np.sum(surrogate.pixel_valid, axis=0),
        np.sum(dataset.pixel_valid, axis=0),
    )
    for pixel in range(dataset.pixel_count):
        original_anchor = float(dataset.traces[0, pixel])
        anchor_rows = np.flatnonzero(surrogate.traces[:, pixel] == original_anchor)
        assert anchor_rows.size == 1
        shift = int(anchor_rows[0])
        np.testing.assert_array_equal(
            surrogate.pixel_valid[:, pixel],
            np.roll(dataset.pixel_valid[:, pixel], shift),
        )
        np.testing.assert_allclose(
            surrogate.traces[:, pixel],
            np.roll(dataset.traces[:, pixel], shift),
            equal_nan=True,
        )


def test_all_valid_surrogate_trace_stream_is_backward_deterministic() -> None:
    base = _synthetic_dataset()
    traces = np.arange(
        base.frame_count * base.pixel_count, dtype=np.float64
    ).reshape(base.frame_count, base.pixel_count)
    dataset = replace(
        base,
        traces=traces,
        pixel_valid=np.ones(traces.shape, dtype=bool),
    ).validated()

    surrogate = autocorrelation_preserving_surrogate(
        dataset,
        np.ones(dataset.frame_count, dtype=bool),
        rng=np.random.default_rng(902),
        spatial_block_px=2,
        min_shift_seconds=1.0,
        max_gap_factor=1.75,
    )

    digest = hashlib.sha256(np.asarray(surrogate.traces).tobytes()).hexdigest()
    assert digest == "34360945cc27b0c7e96c84aea952ffd652ecc57215c09b35c056824d7251788c"
    np.testing.assert_array_equal(surrogate.pixel_valid, dataset.pixel_valid)
    details = surrogate.metadata["autocorrelation_surrogate_shift"]["segments"][0]
    assert details["requested_minimum_shift_frames"] == 51
    assert details["effective_minimum_circular_shift_frames"] == 51
    assert details["fallback_used"] is False


def test_administrative_boundary_is_report_only() -> None:
    dataset = _synthetic_dataset()
    risks = build_risk_surfaces(dataset, _test_config(surrogate_count=0))

    assert np.all(risks.administrative_boundary_distance_px == 0.0)
    assert np.all(risks.eligible)


def test_consensus_outer_partitions_are_contiguous_and_guarded() -> None:
    dataset = _synthetic_dataset()
    config = ConsensusMaskConfig(
        outer_fold_count=5,
        outer_guard_seconds=0.2,
        min_selection_folds=3,
        min_confirmed_outer_folds=3,
        consensus_surrogate_count=0,
        heldout_surrogate_count=0,
        alpha=0.1,
    )

    partitions = contiguous_outer_partitions(dataset, config)

    assert len(partitions) == 5
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    for discovery, confirmation in partitions:
        assert not np.any(discovery & confirmation)
        confirmation_times = timestamps[confirmation]
        assert confirmation_times.size > 0
        assert np.all(np.diff(np.flatnonzero(confirmation)) == 1)
        discovery_times = timestamps[discovery]
        distance = np.min(
            np.abs(discovery_times[:, None] - confirmation_times[[0, -1]][None, :])
        )
        assert distance >= 0.39


def test_consensus_mask_recovers_stable_source_across_outer_folds() -> None:
    dataset = _synthetic_dataset(amplitude=3.0, seed=41)
    analysis = _test_config(surrogate_count=0, min_control_ratio=1.0)
    consensus = ConsensusMaskConfig(
        outer_fold_count=5,
        outer_guard_seconds=0.2,
        min_selection_folds=3,
        min_confirmed_outer_folds=3,
        consensus_surrogate_count=9,
        heldout_surrogate_count=9,
        alpha=0.2,
        random_seed=43,
    )

    result = learn_consensus_heart_mask(dataset, analysis, consensus)

    assert result.detected is True
    assert result.reason == "consensus_mask_detected"
    assert result.confirmed_outer_fold_count >= 3
    assert np.count_nonzero(result.consensus_pixels) >= 4
    assert abs(result.median_candidate_frequency_hz - 2.2) <= 0.1
    assert np.all(result.selection_counts[result.consensus_pixels] >= 3)


def test_consensus_mask_has_explicit_empty_result_without_null_calibration() -> None:
    dataset = _synthetic_dataset(amplitude=3.0, seed=47)
    result = learn_consensus_heart_mask(
        dataset,
        _test_config(surrogate_count=0),
        ConsensusMaskConfig(
            outer_fold_count=5,
            outer_guard_seconds=0.2,
            min_selection_folds=3,
            min_confirmed_outer_folds=3,
            consensus_surrogate_count=0,
            heldout_surrogate_count=0,
            alpha=0.1,
            random_seed=49,
        ),
    )

    assert result.detected is False
    assert result.reason == "consensus_null_not_run"
    assert np.count_nonzero(result.consensus_pixels) == 0


def test_familywise_mask_comparison_uses_shared_maximum_null() -> None:
    p_values, threshold, exceeds, maximum = familywise_max_p_values(
        {"first": 6.0, "second": 4.0},
        {
            "first": np.asarray([1.0, 2.0, 5.0, 3.0]),
            "second": np.asarray([2.0, 4.0, 3.0, 1.0]),
        },
        alpha=0.25,
    )

    assert maximum.tolist() == [2.0, 4.0, 5.0, 3.0]
    assert threshold == 5.0
    assert p_values == {"first": 0.2, "second": 0.6}
    assert exceeds == {"first": True, "second": False}


def test_null_data_produces_explicit_no_estimate() -> None:
    result = analyze_heartrate(_synthetic_dataset(amplitude=0.0), _test_config())

    assert result.detected is False
    assert result.event_timestamps_s.size == 0
    assert result.coverage_fraction == 0.0
    assert result.no_estimate_intervals_s


def test_compact_oscillator_is_discovered_and_confirmed_crossfit() -> None:
    dataset = _synthetic_dataset(amplitude=2.5)
    traces = np.asarray(dataset.traces).copy()
    pixel_valid = np.asarray(dataset.pixel_valid).copy()
    frame_valid = np.asarray(dataset.frame_valid).copy()
    traces[300:302] = np.nan
    pixel_valid[300:302] = False
    frame_valid[300:302] = False
    dataset = replace(
        dataset,
        traces=traces,
        pixel_valid=pixel_valid,
        frame_valid=frame_valid,
    ).validated()
    result = analyze_heartrate(dataset, _test_config())

    assert result.detected is True
    assert result.reason == "confirmed"
    assert len(result.folds) == 2
    assert all(fold.discovery.detected for fold in result.folds)
    assert all(fold.confirmed for fold in result.folds)
    assert all(abs(fold.discovery.candidate.frequency_hz - 2.2) <= 0.1 for fold in result.folds)
    assert result.crossfit_dilated_overlap >= 0.5
    assert result.crossfit_frequency_difference_hz <= 0.1
    assert result.event_timestamps_s.size >= 20
    assert result.coverage_fraction > 0.4


def test_injection_recovery_runs_complete_pipeline() -> None:
    dataset = _synthetic_dataset(amplitude=0.0)
    injected, expected = inject_local_signal(
        dataset,
        InjectionSpec(
            amplitude_sigma=4.0,
            frequency_hz=2.2,
            center_xy=(4.0, 4.0),
            radius_px=1.5,
        ),
    )
    assert expected.size > 40
    assert not np.allclose(injected.traces, dataset.traces)

    rows = run_injection_recovery(
        dataset,
        _test_config(surrogate_count=9, alpha=0.2),
        [
            InjectionSpec(0.0, 2.2, (4.0, 4.0), 1.5),
            InjectionSpec(4.0, 2.2, (4.0, 4.0), 1.5),
        ],
        seed=11,
    )
    operating = injection_operating_characteristics(rows)

    assert len(rows) == 2
    assert operating["null_case_count"] == 1
    assert operating["positive_case_count"] == 1
    assert operating["positive_detected_count"] == 1
    assert operating["positive_detection_rate"] == 1.0
    assert operating["false_positive_rate"] == 0.0
    assert rows[0]["detected"] is False
    assert rows[1]["detected"] is True
    assert rows[1]["estimated_frequency_hz"] == 2.2
    assert rows[1]["expected_event_count_evaluated"] <= rows[1]["expected_event_count_total"]
    assert np.isfinite(operating["confirmed_event_precision_median"])
    assert np.isfinite(operating["confirmed_event_recall_median"])
    assert np.isfinite(operating["confirmed_event_timing_rmse_s_median"])


def _dynamic_support_fixture():
    dataset = _synthetic_dataset(amplitude=2.5)
    config = _test_config()
    result = analyze_heartrate(dataset, config)
    xy = np.asarray(dataset.pixel_xy, dtype=np.float64)
    fold0_selected = (
        (xy[:, 0] >= 3)
        & (xy[:, 0] <= 5)
        & (xy[:, 1] >= 3)
        & (xy[:, 1] <= 4)
    )
    fold1_selected = (
        (xy[:, 0] >= 3)
        & (xy[:, 0] <= 5)
        & (xy[:, 1] >= 4)
        & (xy[:, 1] <= 5)
    )

    def with_support(fold, selected):
        indices = np.flatnonzero(selected)
        mask = np.zeros(dataset.image_shape_hw, dtype=bool)
        integer_xy = np.rint(xy[indices]).astype(np.int64)
        mask[integer_xy[:, 1], integer_xy[:, 0]] = True
        candidate = replace(
            fold.discovery.candidate,
            pixel_indices=indices,
            pixel_weights=np.full(indices.size, 1.0 / indices.size),
            cluster_mask=mask,
        )
        return replace(
            fold,
            discovery=replace(fold.discovery, candidate=candidate),
        )

    result = replace(
        result,
        folds=(
            with_support(result.folds[0], fold0_selected),
            with_support(result.folds[1], fold1_selected),
        ),
    )
    return dataset, config, result, fold1_selected & ~fold0_selected


def test_dynamic_support_recognizes_stable_phase_across_exclusive_regions() -> None:
    dataset, config, result, _fold1_only = _dynamic_support_fixture()

    dynamic = analyze_dynamic_heart_support(
        dataset,
        config,
        result,
        surrogate_count=9,
        seed=29,
    )

    assert dynamic.support_source == "posthoc_crossfit_cluster_union"
    assert dynamic.confirmatory_eligible is False
    assert dynamic.interpretation == (
        "exploratory_only_support_was_not_independently_prespecified"
    )
    assert dynamic.phase_pattern == (
        "stable_exclusive_region_phase_relationship_above_joint_null"
    )
    assert abs(dynamic.frequency_hz - 2.2) <= 0.1
    assert np.count_nonzero(dynamic.pixel_groups["heart_support"]) == 9
    assert np.count_nonzero(dynamic.pixel_groups["core"]) == 3
    assert np.count_nonzero(dynamic.pixel_groups["fold0_only"]) == 3
    assert np.count_nonzero(dynamic.pixel_groups["fold1_only"]) == 3
    assert dynamic.group_summary["fold0_only"]["phase_offset_concentration"] > 0.9
    assert dynamic.group_summary["fold1_only"]["phase_offset_concentration"] > 0.9
    assert dynamic.shared_phase_exceeds_null is True


def test_dynamic_support_distinguishes_phase_incoherence_from_visibility() -> None:
    dataset, config, result, fold1_only = _dynamic_support_fixture()
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    traces = np.asarray(dataset.traces, dtype=np.float64).copy()
    base = 2.5 * np.sin(2.0 * np.pi * 2.2 * timestamps)
    block = np.floor((timestamps - timestamps[0]) / 4.0).astype(np.int64)
    phase_by_block = np.asarray([0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi])
    varying = 2.5 * np.sin(
        2.0 * np.pi * 2.2 * timestamps + phase_by_block[block % 4]
    )
    traces[:, fold1_only] -= base[:, None]
    traces[:, fold1_only] += varying[:, None]
    incoherent_dataset = replace(dataset, traces=traces).validated()

    dynamic = analyze_dynamic_heart_support(
        incoherent_dataset,
        config,
        result,
        surrogate_count=0,
    )

    assert dynamic.phase_pattern != (
        "stable_exclusive_region_phase_relationship_above_joint_null"
    )
    concentration = dynamic.group_summary["fold1_only"][
        "phase_offset_concentration"
    ]
    assert not np.isfinite(concentration) or concentration < 0.7


def test_dynamic_support_latent_pattern_aligns_opposite_contrast_polarity() -> None:
    dataset, config, result, fold1_only = _dynamic_support_fixture()
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    traces = np.asarray(dataset.traces, dtype=np.float64).copy()
    injected = 2.5 * np.sin(2.0 * np.pi * 2.2 * timestamps)
    traces[:, fold1_only] -= 2.0 * injected[:, None]
    opposite_polarity = replace(dataset, traces=traces).validated()

    dynamic = analyze_dynamic_heart_support(
        opposite_polarity,
        config,
        result,
        surrogate_count=9,
        seed=33,
    )

    assert abs(dynamic.frequency_hz - 2.2) <= 0.1
    assert dynamic.latent_exceeds_null is True
    assert dynamic.latent_score > 4.0 * dynamic.support_score
    offset = abs(dynamic.group_summary["fold1_only"]["mean_phase_offset_deg"])
    assert offset > 170.0

    phase = reconstruct_crossfit_heart_phase(
        opposite_polarity,
        config,
        result,
        dynamic,
    )
    assert np.count_nonzero(phase.frame_valid) > 0.3 * dataset.frame_count
    assert np.nanmedian(phase.spatial_alignment[phase.frame_valid]) > 0.8
    assert np.isfinite(phase.bandpassed_residual[:, dynamic.pixel_groups["heart_support"]]).any()
    assert np.isfinite(phase.analytic_residual[:, dynamic.pixel_groups["heart_support"]]).any()
    for fold_index in range(phase.fold_loadings.shape[0]):
        loading = phase.fold_loadings[fold_index]
        first = np.angle(np.nanmean(loading[dynamic.pixel_groups["fold0_only"]]))
        second = np.angle(np.nanmean(loading[dynamic.pixel_groups["fold1_only"]]))
        separation = abs(np.angle(np.exp(1j * (second - first))))
        assert separation > 2.5


def test_dynamic_support_accepts_independently_frozen_anatomical_mask() -> None:
    dataset, config, result, _fold1_only = _dynamic_support_fixture()
    heart_mask = np.zeros(dataset.image_shape_hw, dtype=bool)
    xy = np.rint(dataset.pixel_xy).astype(np.int64)
    selected = (
        result.folds[0].discovery.candidate.cluster_mask
        | result.folds[1].discovery.candidate.cluster_mask
    )
    at_pixel = selected[xy[:, 1], xy[:, 0]]
    heart_mask[xy[at_pixel, 1], xy[at_pixel, 0]] = True

    dynamic = analyze_dynamic_heart_support(
        dataset,
        config,
        result,
        heart_mask=heart_mask,
        mask_is_independent=True,
        frequency_min_hz=1.9,
        frequency_max_hz=2.5,
        surrogate_count=0,
    )

    assert dynamic.support_source == "external_anatomical_mask"
    assert dynamic.frequency_search_source == "explicit_prespecified_bounds"
    assert dynamic.confirmatory_eligible is True
    assert np.count_nonzero(dynamic.pixel_groups["heart_support"]) == 9


def test_shared_dynamic_nulls_are_invariant_to_batching_and_workers() -> None:
    dataset, config, result, _fold1_only = _dynamic_support_fixture()
    union = (
        result.folds[0].discovery.candidate.cluster_mask
        | result.folds[1].discovery.candidate.cluster_mask
    )
    masks = {"first": union, "second": union.copy()}
    context = prepare_dynamic_support_null_context(
        dataset,
        config,
        result,
        heart_masks=masks,
        frequency_min_hz=1.9,
        frequency_max_hz=2.5,
    )
    kwargs = {
        "heart_masks": masks,
        "seed": 71,
        "frequency_min_hz": 1.9,
        "frequency_max_hz": 2.5,
        "context": context,
    }

    complete = compute_dynamic_support_null_batch(
        dataset,
        config,
        result,
        surrogate_indices=range(4),
        workers=1,
        **kwargs,
    )
    first = compute_dynamic_support_null_batch(
        dataset,
        config,
        result,
        surrogate_indices=range(2),
        workers=2,
        **kwargs,
    )
    second = compute_dynamic_support_null_batch(
        dataset,
        config,
        result,
        surrogate_indices=range(2, 4),
        workers=2,
        **kwargs,
    )

    assert np.array_equal(complete.surrogate_indices, np.arange(4))
    for name in masks:
        for attribute in ("support_scores", "shared_phase_scores", "latent_scores"):
            expected = getattr(complete, attribute)[name]
            merged = np.concatenate(
                [getattr(first, attribute)[name], getattr(second, attribute)[name]]
            )
            np.testing.assert_array_equal(merged, expected)
        np.testing.assert_array_equal(
            complete.latent_scores[name],
            complete.latent_scores["first"],
        )


def _regional_phase_fixture(block_lags_s: list[float]):
    dataset = _synthetic_dataset(amplitude=0.0)
    timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
    xy = np.asarray(dataset.pixel_xy, dtype=np.float64)
    upper = (
        (xy[:, 0] >= 3)
        & (xy[:, 0] <= 5)
        & (xy[:, 1] >= 3)
        & (xy[:, 1] <= 4)
    )
    lower = (
        (xy[:, 0] >= 3)
        & (xy[:, 0] <= 5)
        & (xy[:, 1] >= 5)
        & (xy[:, 1] <= 6)
    )
    support = upper | lower
    analytic = np.full(dataset.traces.shape, np.nan + 0j, dtype=np.complex128)
    frame_valid = np.zeros(dataset.frame_count, dtype=bool)
    model_fold = np.full(dataset.frame_count, -1, dtype=np.int16)
    block_ranges = ((100, 300), (400, 600), (800, 1000), (1200, 1400))
    frequency = 2.2
    for block_index, ((start, stop), lag_s) in enumerate(zip(block_ranges, block_lags_s)):
        rows = np.arange(start, stop, dtype=np.int64)
        upper_values = np.exp(2j * np.pi * frequency * timestamps[rows])
        lower_values = np.exp(2j * np.pi * frequency * (timestamps[rows] - lag_s))
        analytic[np.ix_(rows, np.flatnonzero(upper))] = upper_values[:, None]
        analytic[np.ix_(rows, np.flatnonzero(lower))] = lower_values[:, None]
        frame_valid[rows] = True
        model_fold[rows] = block_index % 2
    weights = np.zeros((2, dataset.pixel_count), dtype=np.float64)
    weights[:, support] = 1.0
    loadings = weights.astype(np.complex128)
    latent = np.full(dataset.frame_count, np.nan + 0j, dtype=np.complex128)
    latent[frame_valid] = np.exp(2j * np.pi * frequency * timestamps[frame_valid])
    phase = CrossfitHeartPhaseSeries(
        frequency_hz=frequency,
        band_min_hz=2.0,
        band_max_hz=2.4,
        heart_support=support,
        model_fold_indices=model_fold,
        fold_loadings=loadings,
        fold_loading_weights=weights,
        crossfit_residual=np.full(dataset.traces.shape, np.nan),
        bandpassed_residual=analytic.real,
        analytic_residual=analytic,
        latent_analytic=latent,
        spatial_alignment=np.where(frame_valid, 1.0, np.nan),
        frame_valid=frame_valid,
    )
    return dataset, phase


def test_regional_phase_delay_recovers_stable_lower_lag() -> None:
    dataset, phase = _regional_phase_fixture([0.06, 0.06, 0.06, 0.06])

    result = analyze_regional_phase_delay(
        dataset,
        phase,
        surrogate_count=199,
        alpha=0.05,
        seed=71,
    )

    assert result.region_source == "mask_geometry_balanced_horizontal_split"
    assert result.split_y == 4.5
    assert np.count_nonzero(result.upper_pixels) == 6
    assert np.count_nonzero(result.lower_pixels) == 6
    assert len(result.block_summary) == 4
    assert abs(result.across_block_lower_lag_ms - 60.0) < 1.0
    assert result.across_block_phase_locking_value > 0.99
    assert result.stable_delay_exceeds_null is True
    cycle_lags = np.asarray(
        [row["lower_minus_upper_ms"] for row in result.cycle_rows],
        dtype=np.float64,
    )
    assert cycle_lags.size >= 20
    assert abs(float(np.median(cycle_lags)) - 60.0) < 1.0

    frozen = analyze_regional_phase_delay(
        dataset,
        phase,
        upper_pixels=result.upper_pixels,
        lower_pixels=result.lower_pixels,
        regions_independent=True,
        surrogate_count=199,
        alpha=0.05,
        seed=71,
    )
    assert frozen.region_source == "external_explicit_region_masks"
    assert frozen.regions_independent is True
    assert frozen.interpretation == "stable_regional_delay_above_block_phase_null"


def test_regional_phase_delay_rejects_block_varying_direction() -> None:
    period = 1.0 / 2.2
    dataset, phase = _regional_phase_fixture(
        [0.0, 0.25 * period, 0.5 * period, -0.25 * period]
    )

    result = analyze_regional_phase_delay(
        dataset,
        phase,
        regions_independent=True,
        surrogate_count=199,
        alpha=0.05,
        seed=73,
    )

    assert result.across_block_phase_locking_value < 0.1
    assert result.stable_delay_exceeds_null is False
    assert result.interpretation == "regional_delay_not_stable_above_block_phase_null"
