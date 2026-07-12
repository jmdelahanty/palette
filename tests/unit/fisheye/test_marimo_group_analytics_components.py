from __future__ import annotations

from apps.marimo.components.group_analytics import (
    available_group_panels,
    chaser_selection_options,
    egocentric_heatmap_figure,
    filter_rows_by_chasers,
    grouped_bar_figure,
    line_figure,
    panel_control_spec,
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


def test_panel_controls_only_expose_relevant_filters() -> None:
    behavior = panel_control_spec("behavior")
    assert behavior.analysis_options_key == "epoch_speed_metrics"
    assert behavior.show_statistic is True
    assert behavior.show_window is False
    assert behavior.show_chaser is False

    bouts = panel_control_spec("bout_distributions")
    assert bouts.analysis_options_key == "epoch_bout_histogram_metrics"
    assert bouts.show_window is True
    assert bouts.show_chaser is False
    assert bouts.show_statistic is False

    egocentric = panel_control_spec("egocentric")
    assert egocentric.analysis_options_key == "egocentric_metrics"
    assert egocentric.show_window is True
    assert egocentric.show_chaser is True
    assert egocentric.show_statistic is True

    inventory = panel_control_spec("inventory")
    assert inventory.analysis_options_key is None
    assert inventory.show_window is False
    assert inventory.show_chaser is False
    assert inventory.show_statistic is False


def test_chaser_row_filter_supports_multiple_selected_chasers() -> None:
    rows = [
        {"chaser_index": 0, "value": 1.0},
        {"chaser_index": 1, "value": 2.0},
        {"chaser_index": 2, "value": 3.0},
    ]

    assert filter_rows_by_chasers(rows, (0, 1)) == rows[:2]
    assert filter_rows_by_chasers(rows, ()) == []


def test_every_available_chaser_is_selected_by_default() -> None:
    options, defaults = chaser_selection_options((0, 1))

    assert options == {"Chaser 0": 0, "Chaser 1": 1}
    assert defaults == ["Chaser 0", "Chaser 1"]


def test_grouped_bar_places_selected_chasers_side_by_side() -> None:
    figure = grouped_bar_figure(
        [
            {"window_label": "pre", "value": 1.0, "series": "Chaser 0"},
            {"window_label": "pre", "value": 2.0, "series": "Chaser 1"},
            {"window_label": "post", "value": 3.0, "series": "Chaser 0"},
            {"window_label": "post", "value": 4.0, "series": "Chaser 1"},
        ],
        title="Chaser comparison",
        x_key="window_label",
        y_key="value",
        series_key="series",
        yaxis_title="Value",
    )

    assert figure is not None
    assert figure.layout.barmode == "group"
    assert [trace.name for trace in figure.data] == ["Chaser 0", "Chaser 1"]
    assert [list(trace.x) for trace in figure.data] == [
        ["pre", "post"],
        ["pre", "post"],
    ]


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
