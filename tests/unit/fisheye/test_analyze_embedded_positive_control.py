from __future__ import annotations

from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest


_PLAYGROUND = (
    Path(__file__).resolve().parents[3] / "playgrounds" / "heartrate_stabilization"
)
sys.path.insert(0, str(_PLAYGROUND))

import analyze_embedded_positive_control as analysis  # noqa: E402
import analyze_embedded_crossfit_mask as mask_analysis  # noqa: E402
import analyze_embedded_side_chambers as chamber_analysis  # noqa: E402
import analyze_embedded_bradycardia_response as brady_analysis  # noqa: E402
import analyze_moving_lower_window_excursions as excursion_analysis  # noqa: E402
import analyze_segmented_cache_pca as segmented  # noqa: E402
import compare_embedded_crossfit_mask_projections as mask_projection  # noqa: E402
import compare_moving_frozen_mask_means as moving_means  # noqa: E402
import evaluate_embedded_rate_window_sweep as window_sweep  # noqa: E402
import render_embedded_positive_control_overlay as overlay  # noqa: E402
import render_embedded_crossfit_mask_overlay as mask_overlay  # noqa: E402
import render_embedded_masked_mean_window_overlay as mean_window_overlay  # noqa: E402
import render_embedded_side_chambers_overlay as chamber_overlay  # noqa: E402
import render_moving_lower_mean_overlay as lower_mean_overlay  # noqa: E402
import render_segmented_cache_pca_overlay as moving_overlay  # noqa: E402
import plot_moving_lower_oscillator_frequency as oscillator_plot  # noqa: E402


def test_read_numeric_xlsx_row_resolves_named_worksheet(tmp_path: Path) -> None:
    workbook = tmp_path / "reference.xlsx"
    with zipfile.ZipFile(workbook, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <sheets><sheet name="heart_rate_trace" sheetId="2" r:id="rId2"/></sheets>
</workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId2" Target="worksheets/sheet2.xml"/>
</Relationships>""",
        )
        archive.writestr(
            "xl/worksheets/sheet2.xml",
            """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <sheetData><row r="2"><c r="A2"><v>1.5</v></c><c r="B2"><v>2.25</v></c></row></sheetData>
</worksheet>""",
        )

    values = analysis.read_numeric_xlsx_row(
        workbook,
        sheet_name="heart_rate_trace",
        row_number=2,
    )

    np.testing.assert_allclose(values, [1.5, 2.25])


def test_band_pca_recovers_shared_pixel_oscillation() -> None:
    fps = 100.0
    time = np.arange(2000) / fps
    oscillator = np.sin(2.0 * np.pi * 2.25 * time)
    rng = np.random.default_rng(7)
    loading = np.linspace(-1.0, 1.0, 20)
    pixels = 100.0 + 2.0 * oscillator[:, None] * loading[None, :]
    pixels += rng.normal(scale=0.25, size=pixels.shape)
    ring = 100.0 + rng.normal(scale=0.05, size=time.size)

    candidates, variance = analysis.build_candidate_signals(
        pixels,
        ring,
        fps=fps,
        band_hz=(1.5, 4.0),
    )
    _, _, peak_hz, spectral_ratio = analysis._spectrum(
        candidates["band_pca1"],
        fps=fps,
        band_hz=(1.5, 4.0),
    )

    assert abs(peak_hz - 2.25) <= 0.125
    assert spectral_ratio > 10.0
    assert variance > 0.8


def test_offset_controls_preserve_box_size_and_drop_out_of_bounds() -> None:
    controls = analysis._offset_controls(
        (10, 10, 5, 6),
        frame_shape=(30, 40),
        offset_pixels=12,
    )

    assert controls == {
        "offset_right": (22, 10, 5, 6),
        "offset_posterior": (10, 22, 5, 6),
    }


def test_overlay_event_rate_is_defined_between_successive_peaks() -> None:
    rate = overlay._event_rate_trace(
        np.asarray([10, 60, 100]),
        frame_count=120,
        fps=100.0,
    )

    assert np.isnan(rate[9])
    np.testing.assert_allclose(rate[10:60], 2.0)
    np.testing.assert_allclose(rate[60:100], 2.5)
    assert np.isnan(rate[100])


def test_overlay_pixel_change_uses_blue_negative_and_red_positive() -> None:
    color = overlay._color_pixel_change(
        np.asarray([[-2.0, 0.0, 2.0]]),
        scale=2.0,
    )

    assert int(color[0, 0, 0]) > int(color[0, 0, 2])
    assert color[0, 1].tolist() == [42, 42, 42]
    assert int(color[0, 2, 2]) > int(color[0, 2, 0])


def test_overlay_roi_display_scale_fits_small_and_large_boxes() -> None:
    assert overlay._roi_display_scale(22, 28) == 10
    assert overlay._roi_display_scale(127, 68) == 2


def test_embedded_mask_partitions_are_guarded_and_crossfit() -> None:
    partitions = mask_analysis.temporal_half_partitions(
        100,
        fps=10.0,
        guard_seconds=0.5,
    )

    first_discovery, first_confirmation = partitions[0]
    second_discovery, second_confirmation = partitions[1]
    assert np.flatnonzero(first_discovery)[-1] == 44
    assert np.flatnonzero(first_confirmation)[0] == 55
    np.testing.assert_array_equal(first_discovery, second_confirmation)
    np.testing.assert_array_equal(first_confirmation, second_discovery)
    assert not np.any(first_discovery & first_confirmation)


def test_embedded_mask_cluster_selection_finds_compact_loading_support() -> None:
    rng = np.random.default_rng(17)
    loadings = rng.normal(scale=0.005, size=(5, 8, 8))
    loadings[:, 2:4, 4:7] += 0.25

    selected, scores, signed, mass = mask_analysis.select_loading_cluster(
        loadings.reshape(5, -1),
        roi_shape_hw=(8, 8),
        score_threshold_z=1.5,
        min_cluster_pixels=3,
    )

    selected_image = selected.reshape(8, 8)
    assert np.count_nonzero(selected_image[2:4, 4:7]) >= 5
    assert np.count_nonzero(selected_image) >= 5
    assert scores.shape == (64,)
    assert np.isclose(np.linalg.norm(signed), 1.0)
    assert mass > 0.0


def test_embedded_mask_cluster_selection_can_return_empty() -> None:
    selected, _scores, signed, mass = mask_analysis.select_loading_cluster(
        np.ones((4, 25), dtype=np.float64),
        roi_shape_hw=(5, 5),
        score_threshold_z=1.5,
        min_cluster_pixels=3,
    )

    assert not np.any(selected)
    assert not np.any(signed)
    assert mass == 0.0


def test_embedded_mask_segmented_welch_does_not_bridge_gap() -> None:
    fps = 20.0
    time = np.arange(200, dtype=np.float64) / fps
    signal = np.sin(2.0 * np.pi * 2.0 * time)
    signal[80:120] = np.nan

    frequencies, power = mask_analysis.segmented_welch(signal, fps=fps)

    assert abs(float(frequencies[np.argmax(power)]) - 2.0) <= 0.25


def test_embedded_mask_overlay_uses_opposite_half_and_blanks_guard() -> None:
    partitions = mask_analysis.temporal_half_partitions(
        100,
        fps=10.0,
        guard_seconds=0.5,
    )

    model = mask_overlay.crossfit_model_indices(partitions, frame_count=100)

    assert np.all(model[:45] == 1)
    assert np.all(model[45:55] == -1)
    assert np.all(model[55:] == 0)


def test_embedded_mask_event_intervals_remain_within_confirmation_halves() -> None:
    fps = 20.0
    frame_count = 400
    time = np.arange(frame_count, dtype=np.float64) / fps
    trace = np.sin(2.0 * np.pi * 2.0 * time)
    partitions = mask_analysis.temporal_half_partitions(
        frame_count,
        fps=fps,
        guard_seconds=1.0,
    )
    combined_rate = np.full(frame_count, np.nan, dtype=np.float64)

    for discovery, confirmation in partitions:
        events, _ends, interval_hz, rate, polarity = (
            mask_analysis.heldout_event_intervals(
                trace,
                discovery,
                confirmation,
                fps=fps,
                band_hz=(1.5, 4.0),
                prominence_mad=0.5,
                edge_seconds=0.75,
            )
        )
        assert np.all(confirmation[events])
        assert polarity in {-1, 1}
        assert abs(float(np.median(interval_hz)) - 2.0) <= 0.05
        assert float(np.max(np.abs(interval_hz - 2.0))) <= 0.25
        combined_rate[np.isfinite(rate)] = rate[np.isfinite(rate)]

    assert np.isnan(combined_rate[180:220]).all()


def test_embedded_mask_projection_variants_remain_inside_frozen_mask() -> None:
    rng = np.random.default_rng(31)
    filtered = rng.normal(size=(120, 9))
    filtered[:, 3:6] += np.sin(np.arange(120) / 4.0)[:, None]
    discovery = np.zeros(120, dtype=bool)
    discovery[:60] = True
    mask = np.zeros(9, dtype=bool)
    mask[3:6] = True
    loading = np.arange(1.0, 10.0)

    weights = mask_projection.projection_weights(
        filtered,
        discovery,
        mask,
        loading,
    )

    assert set(weights) == set(mask_projection._METHODS)
    for values in weights.values():
        assert not np.any(values[~mask])
    assert np.isclose(np.linalg.norm(weights["masked_discovery_loading"]), 1.0)
    assert np.isclose(np.linalg.norm(weights["masked_refit_pca"]), 1.0)
    assert np.isclose(np.sum(weights["masked_equal_mean"]), 1.0)


def test_side_chamber_mask_variants_preserve_base_and_expand_contract() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 6:14] = True

    variants = chamber_analysis.mask_variants(mask, radius_px=2)

    np.testing.assert_array_equal(variants["base"], mask)
    assert np.count_nonzero(variants["erode_2px"]) < np.count_nonzero(mask)
    assert np.count_nonzero(variants["dilate_2px"]) > np.count_nonzero(mask)


def test_side_chamber_lag_reports_second_trace_delay() -> None:
    rng = np.random.default_rng(41)
    first = rng.normal(size=2000)
    second = np.zeros_like(first)
    second[5:] = first[:-5]

    summary = chamber_analysis.window_lag_summary(
        first,
        second,
        fps=100.0,
        window_seconds=4.0,
        step_seconds=2.0,
        max_lag_seconds=0.1,
    )

    assert abs(float(summary["median_lag_ms"]) - 50.0) <= 1e-6
    assert float(summary["median_absolute_correlation"]) > 0.95


def test_side_chamber_overlay_crop_contains_both_masks_with_margin() -> None:
    chamber_a = np.zeros((20, 30), dtype=bool)
    chamber_b = np.zeros((20, 30), dtype=bool)
    chamber_a[5:10, 7:12] = True
    chamber_b[12:18, 20:27] = True

    bounds = chamber_overlay.chamber_crop_bounds(
        {"chamber_a": chamber_a, "chamber_b": chamber_b},
        margin_px=2,
    )

    assert bounds == (5, 3, 29, 20)


def test_embedded_window_sweep_never_joins_finite_blocks() -> None:
    trace = np.asarray([1.0, 2.0, np.nan, np.nan, 3.0, 4.0, 5.0, np.nan])

    runs = window_sweep.finite_runs(trace)

    assert [run.tolist() for run in runs] == [[0, 1], [4, 5, 6]]


def test_embedded_mean_window_overlay_limits_nearest_ridge_support() -> None:
    times = np.asarray([1.0, 2.0, 5.0])
    values = np.asarray([2.0, 2.5, 3.0])

    assert mean_window_overlay.nearest_ridge_value(
        times, values, time_s=2.4, maximum_distance_s=0.5
    ) == pytest.approx(2.5)
    assert np.isnan(
        mean_window_overlay.nearest_ridge_value(
            times, values, time_s=3.0, maximum_distance_s=0.5
        )
    )


def test_embedded_mean_window_overlay_smoothing_does_not_bridge_gap() -> None:
    values = np.asarray([1.0, 3.0, 1.0, np.nan, 9.0, 7.0, 9.0])

    smoothed = mean_window_overlay.smooth_finite_runs(values, sigma_samples=1.0)

    assert np.isnan(smoothed[3])
    assert float(np.max(smoothed[:3])) < 3.0
    assert float(np.min(smoothed[4:])) > 7.0


def test_embedded_bradycardia_threshold_spans_remain_separate() -> None:
    times = np.arange(8, dtype=np.float64) * 0.5
    values = np.asarray([150, 110, 105, 140, 115, 110, 150, 150], dtype=np.float64)

    spans = brady_analysis.threshold_spans(
        times,
        values,
        start_s=0.0,
        stop_s=4.0,
        threshold_bpm=120.0,
    )

    assert spans == [(0.5, 1.5, 105.0), (2.0, 3.0, 110.0)]


def test_segmented_cache_common_validity_preserves_real_gap() -> None:
    timestamps = np.arange(100, dtype=np.float64) / 10.0
    frame_valid = np.ones(100, dtype=bool)
    frame_valid[40:55] = False
    pixel_valid = np.ones((100, 3), dtype=bool)

    segments, interpolated = segmented._common_valid_segments(
        timestamps,
        frame_valid,
        pixel_valid,
        np.ones(3, dtype=bool),
        min_seconds=2.0,
        max_interpolated_gap_seconds=0.02,
    )

    assert not np.any(interpolated)
    assert [rows.tolist() for rows in segments] == [
        list(range(40)),
        list(range(55, 100)),
    ]


def test_segmented_cache_bridges_only_bounded_short_gap() -> None:
    timestamps = np.arange(100, dtype=np.float64) / 100.0
    frame_valid = np.ones(100, dtype=bool)
    frame_valid[20:22] = False
    frame_valid[60:64] = False
    pixel_valid = np.ones((100, 3), dtype=bool)

    segments, interpolated = segmented._common_valid_segments(
        timestamps,
        frame_valid,
        pixel_valid,
        np.ones(3, dtype=bool),
        min_seconds=0.1,
        max_interpolated_gap_seconds=0.02,
    )

    assert np.flatnonzero(interpolated).tolist() == [20, 21]
    assert [rows.tolist() for rows in segments] == [
        list(range(60)),
        list(range(64, 100)),
    ]


def test_segmented_cache_pca_recovers_frequency_without_joining_segments() -> None:
    fps = 100.0
    timestamps = np.arange(1200, dtype=np.float64) / fps
    segments = [np.arange(0, 500), np.arange(700, 1200)]
    oscillator = np.sin(2.0 * np.pi * 2.5 * timestamps)
    loading = np.linspace(-1.0, 1.0, 12)
    rng = np.random.default_rng(11)
    values = 100.0 + 3.0 * oscillator[:, None] * loading[None, :]
    values += rng.normal(scale=0.2, size=values.shape)

    scores, _, _, explained = segmented._segmented_pca(
        values,
        segments,
        fps=fps,
        band_hz=(1.5, 4.0),
    )
    frequencies = np.arange(1.5, 4.01, 0.05)
    power, count = segmented._frequency_power(
        scores,
        segments,
        timestamps,
        frequencies,
        edge_seconds=0.5,
    )

    assert count == 800
    assert frequencies[int(np.argmax(power))] == pytest.approx(2.5)
    assert explained > 0.9


def test_moving_overlay_event_rate_does_not_cross_long_gap() -> None:
    timestamps = np.arange(0.0, 6.0, 0.1)
    event_times = np.asarray([0.0, 0.5, 5.0, 5.3])

    rate = moving_overlay._event_rate_trace(
        event_times,
        timestamps,
        band_hz=(1.5, 4.0),
    )

    np.testing.assert_allclose(rate[:5], 120.0)
    assert np.isnan(rate[5:50]).all()
    np.testing.assert_allclose(rate[50:53], 200.0)


def test_moving_overlay_analysis_core_removes_each_segment_edge() -> None:
    timestamps = np.arange(20, dtype=np.float64) / 10.0
    segments = [np.arange(0, 10), np.arange(12, 20)]

    valid = moving_overlay._analysis_core_rows(
        segments,
        timestamps,
        edge_seconds=0.2,
        frame_count=20,
    )

    assert np.flatnonzero(valid).tolist() == [2, 3, 4, 5, 6, 7, 14, 15, 16, 17]


def test_moving_overlay_maps_only_scorable_window_candidates(tmp_path: Path) -> None:
    window_csv = tmp_path / "windows.csv"
    window_csv.write_text(
        "window_start_s,window_stop_s,status,candidate_cycles_per_min\n"
        "0,2,ok,120\n"
        "2,4,insufficient_common_valid_samples,nan\n"
        "4,6,ok,180\n"
    )

    candidate = moving_overlay._window_candidate_trace(
        window_csv,
        np.arange(6, dtype=np.float64),
    )

    np.testing.assert_allclose(candidate[[0, 1, 4, 5]], [120, 120, 180, 180])
    assert np.isnan(candidate[2:4]).all()


def test_moving_overlay_timeline_is_static_canvas_without_cursor() -> None:
    panel = moving_overlay._timeline_panel(
        event_rate_bpm=np.asarray([120.0, 120.0, np.nan, 180.0]),
        analysis_valid=np.asarray([True, True, False, True]),
        candidate_bpm=np.asarray([120.0, 120.0, np.nan, 180.0]),
    )

    assert panel.shape == (141, 541, 3)
    assert panel.dtype == np.uint8
    assert np.any(panel != 25)


def test_lower_mean_overlay_filters_method_and_window_duration(tmp_path: Path) -> None:
    window_csv = tmp_path / "windows.csv"
    window_csv.write_text(
        "method,window_seconds,window_start_s,window_stop_s,status,candidate_cycles_per_min\n"
        "masked_pca,4,0,2,ok,120\n"
        "lower_equal_mean,8,0,2,ok,150\n"
        "lower_equal_mean,4,0,2,ok,180\n"
        "lower_equal_mean,4,2,4,insufficient_common_valid_samples,nan\n"
        "lower_equal_mean,4,4,6,ok,210\n"
    )

    candidate = lower_mean_overlay._window_candidate_trace(
        window_csv,
        np.arange(6, dtype=np.float64),
        method="lower_equal_mean",
        window_seconds=4.0,
    )

    np.testing.assert_allclose(candidate[[0, 1, 4, 5]], [180, 180, 210, 210])
    assert np.isnan(candidate[2:4]).all()


def test_moving_frozen_mask_projection_scores_preserve_equal_means() -> None:
    filtered = [
        np.asarray(
            [
                [1.0, 3.0, 10.0, 14.0],
                [2.0, 4.0, 12.0, 16.0],
                [3.0, 5.0, 14.0, 18.0],
            ]
        )
    ]

    scores, _loading, _variance = moving_means.projection_scores(
        filtered,
        upper_usable=np.asarray([True, True, False, False]),
        lower_usable=np.asarray([False, False, True, True]),
    )

    np.testing.assert_allclose(scores["full_equal_mean"][0], [7.0, 8.5, 10.0])
    np.testing.assert_allclose(scores["upper_equal_mean"][0], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(scores["lower_equal_mean"][0], [12.0, 14.0, 16.0])


def test_moving_raw_region_mean_averages_mono8_before_filtering() -> None:
    fps = 100.0
    time = np.arange(1000, dtype=np.float64) / fps
    oscillator = np.sin(2.0 * np.pi * 3.0 * time)
    values = np.column_stack(
        [100.0 + oscillator, 200.0 + 3.0 * oscillator]
    )

    result = moving_means.segmented_raw_region_mean(
        values,
        [np.arange(time.size)],
        fps=fps,
        band_hz=(2.0, 4.0),
    )[0]

    assert result.shape == (time.size,)
    assert np.isfinite(result).all()
    assert np.corrcoef(result[100:-100], oscillator[100:-100])[0, 1] > 0.999
    assert np.std(result[100:-100]) == pytest.approx(
        np.std(2.0 * oscillator[100:-100]), rel=0.02
    )


def test_moving_window_excursion_classification_has_declared_middle() -> None:
    assert excursion_analysis._classification(
        6.0, stable_threshold_bpm=6.0, excursion_threshold_bpm=24.0
    ) == "stable"
    assert excursion_analysis._classification(
        12.0, stable_threshold_bpm=6.0, excursion_threshold_bpm=24.0
    ) == "intermediate"
    assert excursion_analysis._classification(
        24.0, stable_threshold_bpm=6.0, excursion_threshold_bpm=24.0
    ) == "excursion"


def test_moving_window_cliffs_delta_ignores_missing_values() -> None:
    assert excursion_analysis._cliffs_delta(
        np.asarray([3.0, 4.0, np.nan]),
        np.asarray([1.0, 2.0, np.nan]),
    ) == pytest.approx(1.0)
    assert np.isnan(
        excursion_analysis._cliffs_delta(
            np.asarray([np.nan]),
            np.asarray([1.0]),
        )
    )


def test_oscillator_plot_does_not_connect_across_window_gaps() -> None:
    rows = [
        {"window_start_s": "0", "window_stop_s": "8"},
        {"window_start_s": "8", "window_stop_s": "16"},
        {"window_start_s": "32", "window_stop_s": "40"},
    ]

    runs = oscillator_plot._contiguous_runs(rows)

    assert [[row["window_start_s"] for row in run] for run in runs] == [
        ["0", "8"],
        ["32"],
    ]
