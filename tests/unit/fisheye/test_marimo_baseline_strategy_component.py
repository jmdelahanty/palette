from __future__ import annotations

from apps.marimo.components.baseline_strategy import (
    category_count_figure,
    feature_distribution_figure,
    feature_scatter_figure,
    filter_qc_rows,
    speed_trace_figure,
    trajectory_figure,
)


def _rows():
    return [
        {
            "recording_id": "a",
            "protocol_name": "RedScare",
            "classification_status": "complete",
            "primary_strategy": "broad_even_explorer",
            "wall_fraction": 0.1,
            "active_wall_fraction": 0.1,
            "occupancy_coverage_fraction": 0.9,
        },
        {
            "recording_id": "b",
            "protocol_name": "GoodCopBadCop",
            "classification_status": "invalid",
            "primary_strategy": "unavailable",
            "wall_fraction": 0.8,
            "active_wall_fraction": 0.8,
            "occupancy_coverage_fraction": 0.2,
        },
    ]


def test_filters_require_selected_protocol_and_status() -> None:
    filtered = filter_qc_rows(
        _rows(), protocols=("RedScare",), statuses=("complete",)
    )
    assert [row["recording_id"] for row in filtered] == ["a"]


def test_cohort_figures_preserve_protocol_and_strategy_groups() -> None:
    rows = _rows()
    counts = category_count_figure(
        rows, category_key="primary_strategy", title="Strategies"
    )
    distribution = feature_distribution_figure(
        rows, metric="wall_fraction", label="Wall fraction"
    )
    scatter = feature_scatter_figure(rows)

    assert counts is not None
    assert counts.layout.barmode == "group"
    assert {trace.name for trace in counts.data} == {"RedScare", "GoodCopBadCop"}
    assert distribution is not None
    assert {trace.name for trace in distribution.data} == {
        "RedScare",
        "GoodCopBadCop",
    }
    assert scatter is not None
    assert {trace.name for trace in scatter.data} == {
        "broad_even_explorer",
        "unavailable",
    }


def test_recording_figures_use_lines_and_valid_samples() -> None:
    samples = [
        {
            "relative_time_s": 0.0,
            "x_arena_mm": 1.0,
            "y_arena_mm": 2.0,
            "speed_mm_s": 3.0,
            "position_valid": True,
            "sample_valid": True,
        },
        {
            "relative_time_s": 0.1,
            "x_arena_mm": 2.0,
            "y_arena_mm": 3.0,
            "speed_mm_s": 4.0,
            "position_valid": True,
            "sample_valid": True,
        },
    ]
    trajectory = trajectory_figure(samples, recording_id="recording")
    speed = speed_trace_figure(samples, recording_id="recording")

    assert trajectory is not None
    assert trajectory.data[0].mode == "lines"
    assert speed is not None
    assert speed.data[0].mode == "lines"


def test_figures_return_none_without_usable_rows() -> None:
    assert category_count_figure([], category_key="x", title="empty") is None
    assert feature_distribution_figure([], metric="x", label="x") is None
    assert feature_scatter_figure([]) is None
    assert trajectory_figure([], recording_id="none") is None
    assert speed_trace_figure([], recording_id="none") is None
