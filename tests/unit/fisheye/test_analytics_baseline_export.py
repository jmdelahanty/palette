from __future__ import annotations

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


def test_baseline_time_bins_preserve_temporal_change() -> None:
    rows = build_time_bin_metrics(_inputs(), _window(), time_bin_s=0.5)

    assert len(rows) == 2
    assert [row["source_start_frame"] for row in rows] == [0, 5]
    assert [row["source_end_frame"] for row in rows] == [4, 9]
    assert [row["mean_speed_mm_s"] for row in rows] == [2.0, 7.0]
    assert [row["distance_travelled_mm"] for row in rows] == [5.0, 5.0]
    assert [row["bout_count"] for row in rows] == [1, 1]
    assert rows[0]["representative_x_mm"] < rows[1]["representative_x_mm"]


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

    assert [row["source_frame"] for row in sampled] == [0, 5]
    assert all(row["sampling_policy"] == SAMPLE_POLICY for row in sampled)
    assert all(row["sampling_stride_frames"] == 5 for row in sampled)
    assert all(row["nominal_sample_rate_hz"] == 2.0 for row in sampled)
    assert all(row["effective_sample_rate_hz"] == 2.0 for row in sampled)
    assert len(full) == 10
    assert all(row["sampling_policy"] == FULL_SAMPLE_POLICY for row in full)
    assert all(row["sampling_stride_frames"] == 1 for row in full)


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
