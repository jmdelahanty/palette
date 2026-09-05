from __future__ import annotations

import numpy as np

from apps.marimo.components.validated_behavior_group_statistics import (
    distance_curve_metric_figure,
    grouped_epoch_metric_figure,
    spatial_occupancy_metric_figure,
)
from fisheye.visualization.validated_behavior_group_statistics import (
    distance_curve_figure,
    grouped_epoch_figure,
    spatial_occupancy_figure,
)


def _display_provenance() -> dict[str, object]:
    return {
        "condition_order": ["chaser_pre", "chaser_training", "chaser_post"],
        "condition_labels": {
            "chaser_pre": "Pre",
            "chaser_training": "Training",
            "chaser_post": "Post",
        },
        "condition_colors": {
            "chaser_pre": "#4C78A8",
            "chaser_training": "#E45756",
            "chaser_post": "#54A24B",
        },
        "behavior_role_colors": {"aggressive": "#D1495B"},
        "provider_line_styles": {"detection": "dashed"},
        "source_statistics": {"statistics_manifest_sha256": "a" * 64},
    }


def test_distance_renderers_preserve_open_ended_terminal_band():
    metric_id = "bout_response_by_distance.bout_rate_per_min"
    rows = []
    for condition_index, condition in enumerate(
        ("chaser_pre", "chaser_training", "chaser_post")
    ):
        for bin_index, (start, end) in enumerate(((0.0, 8.0), (50.0, None))):
            median = 10.0 + condition_index + bin_index
            rows.append(
                {
                    "metric_id": metric_id,
                    "condition": condition,
                    "behavior_role": "aggressive",
                    "distance_bin_index": bin_index,
                    "distance_bin_start_mm": start,
                    "distance_bin_end_mm": end,
                    "median": median,
                    "p25": median - 1.0,
                    "p75": median + 1.0,
                }
            )
    payload = {
        **_display_provenance(),
        "label": "Bout response by distance",
        "metric_catalog": [
            {
                "metric_id": metric_id,
                "value_column": "bout_rate_per_min",
                "unit": "1/min",
                "interpretation": "Bout rate within a persisted distance bin",
            }
        ],
        "descriptive_rows": rows,
        "recording_rows": [],
        "contrast_rows": [],
    }

    static = distance_curve_figure(payload)
    tick_labels = [item.get_text() for item in static.axes[0].get_xticklabels()]
    assert tick_labels == ["0–8", "50–∞"]

    interactive = distance_curve_metric_figure(payload, metric_id)
    median_traces = [
        trace
        for trace in interactive.data
        if trace.mode == "lines" and trace.customdata is not None
    ]
    assert median_traces
    assert any(
        "50–∞" in str(value)
        for trace in median_traces
        for row in trace.customdata
        for value in row
    )


def test_spatial_renderers_select_arena_member_stratum_at_mixed_boundary():
    metric_id = "spatial_occupancy.occupancy_density_valid_in_arena"
    common = {
        "metric_id": metric_id,
        "condition": "chaser_pre",
        "provider_role": "detection",
        "y_bin_index": 0,
        "y_bin_start_mm": 0.0,
        "y_bin_end_mm": 2.0,
        "finite_recording_count": 3,
    }
    rows = [
        {
            **common,
            "x_bin_index": 0,
            "x_bin_start_mm": 0.0,
            "x_bin_end_mm": 2.0,
            "arena_bin_center_member": False,
            "mean": 0.90,
            "median": 0.90,
        },
        {
            **common,
            "x_bin_index": 0,
            "x_bin_start_mm": 0.0,
            "x_bin_end_mm": 2.0,
            "arena_bin_center_member": True,
            "mean": 0.10,
            "median": 0.08,
        },
        {
            **common,
            "x_bin_index": 1,
            "x_bin_start_mm": 2.0,
            "x_bin_end_mm": 4.0,
            "arena_bin_center_member": True,
            "mean": 0.20,
            "median": 0.18,
        },
    ]
    payload = {
        **_display_provenance(),
        "condition_order": ["chaser_pre"],
        "label": "Spatial occupancy",
        "default_metric_id": metric_id,
        "metric_catalog": [
            {
                "metric_id": metric_id,
                "value_column": "occupancy_density_valid_in_arena",
                "unit": "fraction",
                "interpretation": "Occupancy density",
            }
        ],
        "descriptive_rows": rows,
        "recording_rows": [],
        "contrast_rows": [],
    }

    static = spatial_occupancy_figure(payload)
    static_grid = np.asarray(static.axes[0].collections[0].get_array()).reshape(1, 2)
    np.testing.assert_allclose(static_grid, [[10.0, 20.0]])

    interactive = spatial_occupancy_metric_figure(
        payload,
        metric_id,
        provider_role="detection",
        condition="chaser_pre",
    )
    np.testing.assert_allclose(np.asarray(interactive.data[0].z), [[10.0, 20.0]])


def test_grouped_renderers_use_protocol_color_and_independent_role_glyph():
    metric_id = "near_field.distance_mean_mm"
    rows = [
        {
            "metric_id": metric_id,
            "condition": condition,
            "provider_role": "keypoint",
            "behavior_role": "aggressive",
            "median": 10.0 + index,
            "p25": 9.0 + index,
            "p75": 11.0 + index,
            "finite_recording_count": 3,
        }
        for index, condition in enumerate(
            ("chaser_pre", "chaser_training", "chaser_post")
        )
    ]
    payload = {
        **_display_provenance(),
        "label": "Near field",
        "metric_catalog": [
            {
                "metric_id": metric_id,
                "value_column": "distance_mean_mm",
                "unit": "mm",
                "interpretation": "Mean fish-to-chaser distance",
            }
        ],
        "descriptive_rows": rows,
        "recording_rows": [],
        "contrast_rows": [],
        "behavior_role_styles": {
            "aggressive": {
                "aggregate_color_hex": "#0000ff",
                "aggregate_color_css": "rgba(0, 0, 255, 1)",
                "aggregate_color_policy": (
                    "unique_protocol_rgba_across_occurrences"
                ),
                "experimental_color_hex_values": ["#0000ff"],
                "experimental_color_css_values": ["rgba(0, 0, 255, 1)"],
                "plotly_role_symbol": "star",
                "matplotlib_role_marker": "*",
                "color_role_independence": True,
            }
        },
    }

    static = grouped_epoch_figure(payload)
    assert static.axes[0].lines[0].get_color() == "#0000ff"
    assert static.axes[0].lines[0].get_marker() == "*"

    interactive = grouped_epoch_metric_figure(payload, metric_id)
    assert interactive.data[0].line.color == "rgba(0, 0, 255, 1)"
    assert interactive.data[0].marker.symbol == "star"
