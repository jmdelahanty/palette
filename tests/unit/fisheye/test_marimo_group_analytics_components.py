from __future__ import annotations

from apps.marimo.components.group_analytics import (
    available_group_panels,
    chaser_selection_options,
    cra_quadrant_occupancy_figure,
    epoch_selection_options,
    egocentric_heatmap_figure,
    egocentric_polar_figure,
    egocentric_probability_color_max,
    filter_rows_by_chasers,
    filter_rows_by_windows,
    group_egocentric_histogram_rows,
    grouped_bar_figure,
    line_figure,
    panel_control_spec,
    position_occupancy_heatmap_figure,
    sample_grain_status_rows,
)
from fisheye.group_analytics_viewer.query import (
    egocentric_rebin_options,
    rebin_egocentric_histogram_rows,
)


def test_available_group_panels_follow_export_capabilities() -> None:
    panels = available_group_panels(
        {
            "chaser.epoch.behavior_summary",
            "chaser.quadrant_occupancy",
            "chaser.near_field_occupancy",
            "chaser.egocentric",
            "group.statistics",
            "report.semantic_montage",
        },
        statistics_available=True,
    )

    assert [panel.panel_id for panel in panels] == [
        "behavior",
        "quadrant_occupancy",
        "near_field",
        "egocentric",
        "statistics",
        "published_montages",
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
    assert behavior.show_window is True
    assert behavior.show_chaser is False

    bouts = panel_control_spec("bout_distributions")
    assert bouts.analysis_options_key == "epoch_bout_histogram_metrics"
    assert bouts.show_window is True
    assert bouts.show_chaser is False
    assert bouts.show_statistic is False

    spatial = panel_control_spec("spatial")
    assert spatial.show_window is True
    assert spatial.show_position_bins is True

    egocentric = panel_control_spec("egocentric")
    assert egocentric.analysis_options_key == "egocentric_metrics"
    assert egocentric.show_window is True
    assert egocentric.show_chaser is True
    assert egocentric.show_statistic is True
    assert egocentric.show_egocentric_bins is True

    quadrant_occupancy = panel_control_spec("quadrant_occupancy")
    assert quadrant_occupancy.analysis_label == "Quadrant-occupancy analysis"

    inventory = panel_control_spec("inventory")
    assert inventory.analysis_options_key is None
    assert inventory.show_window is False
    assert inventory.show_chaser is False
    assert inventory.show_statistic is False
    assert inventory.show_egocentric_bins is False
    assert inventory.show_position_bins is False


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


def test_every_available_epoch_is_selected_by_default() -> None:
    options, defaults = epoch_selection_options(
        ("pre_event", "training_event", "post_event", "pre_event")
    )

    assert options == ["pre_event", "training_event", "post_event"]
    assert defaults == options


def test_epoch_row_filter_supports_one_two_or_all_epochs() -> None:
    rows = [
        {"window_label": "pre_event", "value": 1.0},
        {"window_label": "training_event", "value": 2.0},
        {"window_label": "post_event", "value": 3.0},
        {"value": 4.0},
    ]

    assert filter_rows_by_windows(rows, ("pre_event",)) == rows[:1]
    assert filter_rows_by_windows(rows, ("pre_event", "post_event")) == [
        rows[0],
        rows[2],
    ]
    assert filter_rows_by_windows(
        rows,
        ("pre_event", "training_event", "post_event"),
    ) == rows[:3]


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


def test_grouped_bar_uses_stimulus_colors_and_patterns_duplicate_colors() -> None:
    figure = grouped_bar_figure(
        [
            {
                "window_label": "pre",
                "value": 1.0,
                "series": "aggressive · chaser 0",
                "raw_color_hex": "#ff0000",
            },
            {
                "window_label": "pre",
                "value": 2.0,
                "series": "inert · chaser 1",
                "raw_color_hex": "#ff0000",
            },
        ],
        title="RedScare",
        x_key="window_label",
        y_key="value",
        series_key="series",
        yaxis_title="Value",
        color_key="raw_color_hex",
    )

    assert figure is not None
    assert [trace.marker.color for trace in figure.data] == ["#ff0000", "#ff0000"]
    assert figure.data[0].marker.pattern.shape != figure.data[1].marker.pattern.shape


def test_line_figure_uses_distinct_stimulus_colors() -> None:
    figure = line_figure(
        [
            {"x": 1.0, "y": 2.0, "chaser": 0, "raw_color_hex": "#ff0000"},
            {"x": 1.0, "y": 3.0, "chaser": 1, "raw_color_hex": "#1600ff"},
        ],
        title="GoodCopBadCop",
        x_key="x",
        y_key="y",
        series_keys=("chaser",),
        xaxis_title="X",
        yaxis_title="Y",
        color_key="raw_color_hex",
    )

    assert figure is not None
    assert [trace.line.color for trace in figure.data] == ["#ff0000", "#1600ff"]


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


def test_egocentric_histogram_groups_every_selected_epoch_chaser_panel() -> None:
    rows = [
        {"window_label": "pre", "chaser_index": 0, "pooled_probability": 0.1},
        {"window_label": "post", "chaser_index": 1, "pooled_probability": 0.2},
    ]

    grouped = group_egocentric_histogram_rows(
        rows,
        window_labels=("pre", "post"),
        chaser_indices=(0, 1),
    )

    assert list(grouped) == [
        ("pre", 0),
        ("pre", 1),
        ("post", 0),
        ("post", 1),
    ]
    assert grouped[("pre", 0)] == rows[:1]
    assert grouped[("pre", 1)] == []
    assert grouped[("post", 1)] == rows[1:]


def test_egocentric_polar_uses_exported_bin_geometry_and_shared_scale() -> None:
    rows = [
        {
            "distance_bin_left_mm": 10.0,
            "distance_bin_width_mm": 5.0,
            "bearing_bin_center_deg": 15.0,
            "bearing_bin_width_deg": 30.0,
            "pooled_count": 25,
            "pooled_probability": 0.25,
        }
    ]

    figure = egocentric_polar_figure(
        rows,
        title="pre · aggressive · chaser 0",
        color_max=0.5,
        show_colorbar=False,
    )

    assert figure is not None
    trace = figure.data[0]
    assert trace.type == "barpolar"
    assert list(trace.theta) == [15.0]
    assert list(trace.r) == [5.0]
    assert list(trace.base) == [10.0]
    assert list(trace.width) == [30.0]
    assert trace.marker.cmin == 0.0
    assert trace.marker.cmax == 0.5
    assert trace.marker.showscale is False
    assert figure.layout.polar.angularaxis.rotation == 90
    assert figure.layout.polar.angularaxis.direction == "counterclockwise"


def test_egocentric_probability_scale_is_robust_across_panels() -> None:
    rows = [
        {"pooled_probability": 0.1},
        {"pooled_probability": 0.2},
        {"pooled_probability": 0.9},
    ]

    assert egocentric_probability_color_max(rows, quantile=0.5) == 0.2


def test_position_occupancy_heatmap_preserves_source_image_orientation() -> None:
    figure = position_occupancy_heatmap_figure(
        [
            {
                "x_bin_center_fraction": 0.25,
                "y_bin_center_fraction": 0.25,
                "pooled_probability": 0.4,
            },
            {
                "x_bin_center_fraction": 0.75,
                "y_bin_center_fraction": 0.25,
                "pooled_probability": 0.6,
            },
        ],
        title="pre position occupancy",
        color_max=0.7,
    )

    assert figure is not None
    assert list(figure.data[0].x) == [0.25, 0.75]
    assert list(figure.data[0].y) == [0.25]
    assert list(figure.data[0].z[0]) == [0.4, 0.6]
    assert figure.data[0].zmax == 0.7
    assert figure.layout.yaxis.autorange == "reversed"
    assert figure.layout.yaxis.scaleanchor == "x"


def test_cra_quadrant_figure_restores_recording_strips_means_and_densities() -> None:
    phases = [
        {"phase_axis_index": 0, "phase_label": "pre_static"},
        {"phase_axis_index": 1, "phase_label": "post_static"},
    ]
    rows = []
    quadrant_rows = []
    density_rows = []
    for phase in phases:
        phase_label = phase["phase_label"]
        for order, quadrant in enumerate(
            ("top_left", "top_right", "bottom_left", "bottom_right")
        ):
            rows.append(
                {
                    "recording_id": "recording_1",
                    "phase_label": phase_label,
                    "quadrant_id": quadrant,
                    "display_order": order,
                    "occupancy_fraction": 0.4 if order == 0 else 0.2,
                    "is_chaser_quadrant": order == 0,
                }
            )
            quadrant_rows.append(
                {
                    "phase_label": phase_label,
                    "quadrant_id": quadrant,
                    "quadrant_label": quadrant.replace("_", " ").title(),
                    "display_order": order,
                    "mean": 0.4 if order == 0 else 0.2,
                    "sem": 0.02,
                    "recording_count": 1,
                    "chaser_recording_count": 1 if order == 0 else 0,
                }
            )
        for series_role in ("chaser", "non_chaser"):
            density_rows.extend(
                {
                    "phase_label": phase_label,
                    "series_role": series_role,
                    "x": x,
                    "density": 1.0 - abs(x - 0.5),
                }
                for x in (0.0, 0.5, 1.0)
            )

    figure = cra_quadrant_occupancy_figure(
        {
            "available": True,
            "chance": 0.25,
            "phases": phases,
            "rows": rows,
            "quadrant_rows": quadrant_rows,
            "density_rows": density_rows,
            "statistics": {"n": 1, "p_value": 1.0, "median_difference": 0.0},
        }
    )

    assert figure is not None
    assert figure.layout.title.text.startswith("Descriptive gross quadrant relocation")
    assert [trace.name for trace in figure.data[:5]] == [
        "non-chaser quadrants",
        "chaser quadrant",
        "mean ± SEM",
        "non-chaser density",
        "chaser density",
    ]
    assert figure.data[0].marker.color == "#334155"
    assert figure.data[1].marker.color == "#b54848"
    assert figure.data[2].error_x.visible is True
    assert figure.data[3].fill == "tozeroy"
    assert len(figure.layout.shapes) == 4
    assert all(shape.x0 == 0.25 and shape.x1 == 0.25 for shape in figure.layout.shapes)


def _native_egocentric_histogram_rows() -> list[dict[str, object]]:
    counts = [1, 2, 3, 4, 5, 6, 7, 8]
    total = sum(counts)
    rows = []
    for distance_index in range(2):
        for bearing_index in range(4):
            count = counts[distance_index * 4 + bearing_index]
            rows.append(
                {
                    "window_index": 0,
                    "window_label": "pre",
                    "chaser_index": 0,
                    "distance_bin_index": distance_index,
                    "distance_bin_left_mm": float(distance_index * 2),
                    "distance_bin_right_mm": float((distance_index + 1) * 2),
                    "distance_bin_center_mm": float(distance_index * 2 + 1),
                    "distance_bin_width_mm": 2.0,
                    "bearing_bin_index": bearing_index,
                    "bearing_bin_left_deg": float(-180 + bearing_index * 90),
                    "bearing_bin_right_deg": float(-90 + bearing_index * 90),
                    "bearing_bin_center_deg": float(-135 + bearing_index * 90),
                    "bearing_bin_width_deg": 90.0,
                    "pooled_count": count,
                    "pooled_total_count": total,
                    "pooled_probability": count / total,
                }
            )
    return rows


def test_egocentric_rebin_options_only_offer_exact_native_multiples() -> None:
    options = egocentric_rebin_options(_native_egocentric_histogram_rows())

    assert [item["factor"] for item in options["distance"]] == [1, 2]
    assert [item["label"] for item in options["distance"]] == [
        "2 mm (native)",
        "4 mm (2× native)",
    ]
    assert [item["factor"] for item in options["bearing"]] == [1, 2, 4]
    assert [item["width"] for item in options["bearing"]] == [90.0, 180.0, 360.0]


def test_egocentric_rebin_sums_counts_and_recomputes_probabilities() -> None:
    rebinned = rebin_egocentric_histogram_rows(
        _native_egocentric_histogram_rows(),
        distance_bin_factor=2,
        bearing_bin_factor=2,
    )

    assert len(rebinned) == 2
    assert [row["pooled_count"] for row in rebinned] == [14, 22]
    assert all(row["pooled_total_count"] == 36 for row in rebinned)
    assert sum(float(row["pooled_probability"]) for row in rebinned) == 1.0
    assert all(row["distance_bin_width_mm"] == 4.0 for row in rebinned)
    assert all(row["bearing_bin_width_deg"] == 180.0 for row in rebinned)
    assert all(row["distance_bin_factor"] == 2 for row in rebinned)
    assert all(row["bearing_bin_factor"] == 2 for row in rebinned)


def test_egocentric_rebin_rejects_non_divisible_bearing_factor() -> None:
    import pytest

    with pytest.raises(ValueError, match="complete circular grid"):
        rebin_egocentric_histogram_rows(
            _native_egocentric_histogram_rows(),
            bearing_bin_factor=3,
        )
