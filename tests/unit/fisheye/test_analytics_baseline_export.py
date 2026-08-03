from __future__ import annotations

from dataclasses import replace

import numpy as np

from fisheye.analytics_exports.baseline import (
    BaselineArrays,
    BaselineWindow,
    FULL_SAMPLE_POLICY,
    SAMPLE_POLICY,
    build_sample_metrics,
    build_summary_metrics,
    build_time_bin_metrics,
    is_baseline_label,
)
from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.contracts import (
    BASELINE_BEHAVIOR_SUMMARY_TABLE,
    BASELINE_BEHAVIOR_TIME_BINS_TABLE,
    BASELINE_KINEMATIC_SAMPLES_TABLE,
    TABLE_CONTRACTS,
)


def _inputs() -> BaselineArrays:
    frames = np.arange(20, dtype=np.int64)
    xy = np.column_stack(
        [
            np.linspace(10.0, 19.0, 20),
            np.full(20, 10.0),
        ]
    )
    return BaselineArrays(
        fps=10.0,
        arena_xy_px=xy,
        position_valid=np.ones(20, dtype=bool),
        arena_center_x_px=10.0,
        arena_center_y_px=10.0,
        arena_radius_px=10.0,
        pixels_per_mm=1.0,
        wall_band_mm=2.0,
        track_frames=frames,
        track_time_s=frames / 10.0,
        speed_mm_s=np.arange(20, dtype=np.float64),
        frame_path_distance_mm=np.ones(20, dtype=np.float64),
        heading_deg=np.linspace(-45.0, 45.0, 20),
        sample_valid=np.ones(20, dtype=bool),
        bout_event_frames=np.asarray([1, 7, 12], dtype=np.int64),
    )


def _window() -> BaselineWindow:
    return BaselineWindow(
        window_id=0,
        label="pre_event",
        start_frame=0,
        end_frame=9,
        start_time_s=0.0,
        end_time_s=1.0,
        duration_s=1.0,
    )


def test_baseline_label_vocabulary_is_explicit() -> None:
    assert is_baseline_label("pre_event")
    assert is_baseline_label("Pre stimulus")
    assert is_baseline_label("baseline")
    assert not is_baseline_label("training_event")
    assert not is_baseline_label("post_event")


def test_baseline_summary_has_activity_spatial_and_quality_metrics() -> None:
    row = build_summary_metrics(_inputs(), _window(), spatial_grid_size=4)

    assert row["baseline_window_label"] == "pre_event"
    assert row["total_frame_count"] == 10
    assert row["valid_frame_count"] == 10
    assert row["tracking_dropout_fraction"] == 0.0
    assert row["mean_speed_mm_s"] == 4.5
    assert row["total_path_mm"] == 10.0
    assert row["bout_count"] == 2
    assert row["bout_rate_per_min"] == 120.0
    assert row["spatial_grid_size"] == 4
    assert 0.0 <= row["spatial_entropy_normalized"] <= 1.0
    assert 0.0 <= row["quadrant_entropy_normalized"] <= 1.0
    assert row["coordinate_frame"] == "arena_centered_mm"
    assert row["experimental_area_geometry_type"] == "circle"
    assert row["boundary_distance_method"] == "circle_radius_minus_center_distance_v1"
    assert row["wall_fraction_denominator"] == "valid_position_frames"
    expected_boundary = 10.0 - float(np.mean(np.linspace(0.0, 9.0, 20)[:10]))
    assert np.isclose(row["mean_distance_to_arena_boundary_mm"], expected_boundary)
    assert np.isclose(row["expected_uniform_wall_fraction"], 0.36)


def test_baseline_summary_projects_only_the_closed_source_quality_vocabulary() -> None:
    source_fields = {
        "median_bout_duration_s": 1.0,
        "mean_bout_duration_s": 2.0,
        "median_bout_path_length_mm": 3.0,
        "mean_bout_path_length_mm": 4.0,
        "median_abs_bout_net_heading_change_deg": 5.0,
        "mean_abs_bout_net_heading_change_deg": 6.0,
        "median_inter_bout_interval_s": 7.0,
        "mean_inter_bout_interval_s": 8.0,
        "future_source_metric": 9.0,
        "window_label": "must_not_replace_the_baseline_window",
    }

    row = build_summary_metrics(
        _inputs(),
        _window(),
        spatial_grid_size=4,
        source_summary=source_fields,
    )

    projected = tuple(name for name in source_fields if name in row)
    assert projected == (
        "median_bout_duration_s",
        "mean_bout_duration_s",
        "median_bout_path_length_mm",
        "mean_bout_path_length_mm",
        "median_abs_bout_net_heading_change_deg",
        "mean_abs_bout_net_heading_change_deg",
        "median_inter_bout_interval_s",
        "mean_inter_bout_interval_s",
    )
    assert row["baseline_window_label"] == "pre_event"
    assert "future_source_metric" not in row


def test_baseline_time_bins_preserve_temporal_change() -> None:
    rows = build_time_bin_metrics(_inputs(), _window(), time_bin_s=0.5)

    assert len(rows) == 2
    assert tuple(rows[0]) == (
        "baseline_method",
        "baseline_method_version",
        "baseline_window_id",
        "baseline_window_label",
        "time_bin_index",
        "relative_start_s",
        "relative_end_s",
        "time_bin_duration_s",
        "source_start_frame",
        "source_end_frame",
        "expected_frame_count",
        "valid_position_count",
        "valid_position_fraction",
        "speed_sample_count",
        "mean_speed_mm_s",
        "median_speed_mm_s",
        "p95_speed_mm_s",
        "distance_travelled_mm",
        "mean_center_distance_mm",
        "median_center_distance_mm",
        "mean_distance_to_arena_boundary_mm",
        "median_distance_to_arena_boundary_mm",
        "experimental_area_geometry_type",
        "boundary_distance_method",
        "wall_fraction_denominator",
        "wall_frame_count",
        "wall_fraction",
        "representative_position_method",
        "representative_x_mm",
        "representative_y_mm",
        "mean_heading_deg",
        "heading_resultant",
        "bout_count",
        "coordinate_frame",
        "coordinate_origin",
        "x_axis_direction",
        "y_axis_direction",
        "time_bin_policy",
    )
    assert [row["source_start_frame"] for row in rows] == [0, 5]
    assert [row["source_end_frame"] for row in rows] == [4, 9]
    assert [row["mean_speed_mm_s"] for row in rows] == [2.0, 7.0]
    assert [row["distance_travelled_mm"] for row in rows] == [5.0, 5.0]
    assert [row["bout_count"] for row in rows] == [1, 1]
    assert rows[0]["representative_x_mm"] < rows[1]["representative_x_mm"]
    assert rows[0]["mean_distance_to_arena_boundary_mm"] > rows[1][
        "mean_distance_to_arena_boundary_mm"
    ]
    assert all(row["wall_fraction_denominator"] == "valid_position_frames" for row in rows)


def test_baseline_samples_are_deterministic_and_support_full_resolution() -> None:
    sampled = build_sample_metrics(
        _inputs(),
        _window(),
        target_sample_rate_hz=2.0,
    )
    full = build_sample_metrics(
        _inputs(),
        _window(),
        full_resolution=True,
    )

    assert tuple(sampled[0]) == (
        "baseline_method",
        "baseline_method_version",
        "baseline_window_id",
        "baseline_window_label",
        "source_sample_index",
        "source_frame",
        "source_time_s",
        "relative_time_s",
        "x_arena_mm",
        "y_arena_mm",
        "x_arena_fraction",
        "y_arena_fraction",
        "speed_mm_s",
        "heading_deg",
        "frame_path_distance_mm",
        "center_distance_mm",
        "distance_to_arena_boundary_mm",
        "wall",
        "experimental_area_geometry_type",
        "boundary_distance_method",
        "position_valid",
        "sample_valid",
        "sampling_policy",
        "sampling_stride_frames",
        "requested_sample_rate_hz",
        "source_sample_rate_hz",
        "nominal_sample_rate_hz",
        "effective_sample_rate_hz",
        "coordinate_frame",
        "coordinate_origin",
        "x_axis_direction",
        "y_axis_direction",
    )
    assert [row["source_frame"] for row in sampled] == [0, 5]
    assert all(row["sampling_policy"] == SAMPLE_POLICY for row in sampled)
    assert all(row["sampling_stride_frames"] == 5 for row in sampled)
    assert all(row["nominal_sample_rate_hz"] == 2.0 for row in sampled)
    assert all(row["effective_sample_rate_hz"] == 2.0 for row in sampled)
    assert np.allclose(
        [row["distance_to_arena_boundary_mm"] for row in sampled],
        [10.0, 10.0 - np.linspace(0.0, 9.0, 20)[5]],
    )
    assert all(
        row["boundary_distance_method"] == "circle_radius_minus_center_distance_v1"
        for row in sampled
    )
    assert len(full) == 10
    assert all(row["sampling_policy"] == FULL_SAMPLE_POLICY for row in full)
    assert all(row["sampling_stride_frames"] == 1 for row in full)
    assert all(row["requested_sample_rate_hz"] is None for row in full)


def test_baseline_samples_use_nulls_not_sentinels_for_invalid_measurements() -> None:
    inputs = _inputs()
    invalid = replace(
        inputs,
        position_valid=np.zeros(20, dtype=bool),
        speed_mm_s=np.full(20, np.nan, dtype=np.float64),
        heading_deg=np.full(20, np.nan, dtype=np.float64),
        frame_path_distance_mm=np.full(20, np.nan, dtype=np.float64),
    )

    rows = build_sample_metrics(invalid, _window(), target_sample_rate_hz=2.0)

    assert rows
    for row in rows:
        assert row["position_valid"] is False
        assert row["sample_valid"] is True
        assert row["wall"] is None
        assert row["x_arena_mm"] is None
        assert row["y_arena_mm"] is None
        assert row["speed_mm_s"] is None
        assert row["heading_deg"] is None
        assert row["frame_path_distance_mm"] is None


def test_baseline_capabilities_require_their_strict_table_contracts() -> None:
    columns = {
        table: contract.required_columns
        for table, contract in TABLE_CONTRACTS.items()
        if table
        in {
            BASELINE_BEHAVIOR_SUMMARY_TABLE,
            BASELINE_BEHAVIOR_TIME_BINS_TABLE,
            BASELINE_KINEMATIC_SAMPLES_TABLE,
        }
    }
    statuses = {status.capability_id: status for status in resolve_capabilities(columns)}

    assert statuses["core.baseline.behavior_summary"].available
    assert statuses["core.baseline.behavior_time_bins"].available
    assert statuses["core.baseline.kinematic_samples"].available
