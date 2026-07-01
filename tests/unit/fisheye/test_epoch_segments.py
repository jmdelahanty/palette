from __future__ import annotations

import numpy as np

from fisheye.analysis.epoch_segments import (
    EpochSegment,
    HistogramMetricSpec,
    assign_frames_to_segments,
    assign_intervals_to_segments,
    assign_point_events_to_segments,
    histogram_table,
    rate_per_minute,
)


def _segments() -> tuple[EpochSegment, ...]:
    return (
        EpochSegment(10, "pre_event", 0, 4, 0.0, 0.5, 0.5),
        EpochSegment(20, "post_event", 5, 9, 0.5, 1.0, 0.5),
    )


def test_assigns_frames_points_and_intervals_to_segments() -> None:
    segments = _segments()

    np.testing.assert_array_equal(
        assign_frames_to_segments(10, segments),
        np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        assign_point_events_to_segments(np.asarray([0, 4, 5, 9, 10]), segments),
        np.asarray([0, 0, 1, 1, -1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        assign_intervals_to_segments(
            np.asarray([1, 3, 5, 4]),
            np.asarray([2, 6, 8, 5]),
            segments,
            rule="contained",
        ),
        np.asarray([0, -1, 1, -1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        assign_intervals_to_segments(
            np.asarray([1, 3, 5, 4]),
            np.asarray([2, 6, 8, 5]),
            segments,
            rule="overlap",
        ),
        np.asarray([0, 1, 1, 1], dtype=np.int32),
    )


def test_histogram_table_uses_shared_bins_and_persisted_metadata() -> None:
    segments = _segments()
    table = histogram_table(
        segments=segments,
        values_by_segment=[
            np.asarray([0.01, 0.03, 0.04]),
            np.asarray([0.11, np.nan]),
        ],
        metric_spec=HistogramMetricSpec(
            metric_name="bout_duration_s",
            units="s",
            bin_policy="fixed_width_from_zero_to_component_max",
            bin_width=0.05,
            range_min=0.0,
        ),
    )

    assert table.dtype.names is not None
    assert table.shape == (6,)
    assert table["metric_name"][0] == b"bout_duration_s"
    np.testing.assert_allclose(table["bin_left"][:3], [0.0, 0.05, 0.10])
    np.testing.assert_array_equal(table["hist_count"][:3], [3, 0, 0])
    np.testing.assert_array_equal(table["hist_count"][3:], [0, 0, 1])
    assert int(table["source_sample_count"][3]) == 2
    assert int(table["finite_sample_count"][3]) == 1
    assert rate_per_minute(3, 30.0) == 6.0
