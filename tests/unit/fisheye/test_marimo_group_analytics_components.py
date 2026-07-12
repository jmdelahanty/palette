from __future__ import annotations

from apps.marimo.components.group_analytics import (
    available_group_panels,
    egocentric_heatmap_figure,
    grouped_bar_figure,
    line_figure,
    sample_grain_status_rows,
)


def test_available_group_panels_follow_export_capabilities() -> None:
    panels = available_group_panels(
        {
            "chaser.epoch.behavior_summary",
            "chaser.egocentric",
            "group.statistics",
        },
        statistics_available=True,
    )

    assert [panel.panel_id for panel in panels] == [
        "behavior",
        "egocentric",
        "statistics",
        "inventory",
    ]


def test_sample_grain_status_is_scoped_to_selected_export() -> None:
    rows = sample_grain_status_rows({"kinematics.frame_trace"})
    rows_by_surface = {row["surface"]: row for row in rows}

    assert rows_by_surface["Frame kinematic traces"]["included_in_export"] is True
    assert rows_by_surface["Trajectory reconstruction"]["included_in_export"] is True
    assert rows_by_surface["Eye angles and convergence"]["status"] == (
        "not included in this exported dataset"
    )
    assert rows_by_surface["Tail motion"]["included_in_export"] is False


def test_plot_helpers_return_none_for_missing_required_columns() -> None:
    assert grouped_bar_figure(
        [],
        title="empty",
        x_key="condition",
        y_key="value",
        series_key="series",
        yaxis_title="Value",
    ) is None
    assert line_figure(
        [{"x": 1.0}],
        title="missing y",
        x_key="x",
        y_key="y",
        series_keys=(),
        xaxis_title="X",
        yaxis_title="Y",
    ) is None


def test_egocentric_heatmap_uses_persisted_probability_bins() -> None:
    figure = egocentric_heatmap_figure(
        [
            {
                "distance_bin_center_mm": 1.0,
                "bearing_bin_center_deg": -45.0,
                "pooled_probability": 0.25,
            },
            {
                "distance_bin_center_mm": 2.0,
                "bearing_bin_center_deg": -45.0,
                "pooled_probability": 0.75,
            },
        ],
        title="bearing",
    )

    assert figure is not None
    assert list(figure.data[0].x) == [1.0, 2.0]
    assert list(figure.data[0].y) == [-45.0]
    assert list(figure.data[0].z[0]) == [0.25, 0.75]
