from __future__ import annotations

import time

import plotly.express as px
import plotly.graph_objects as go

from apps.marimo.components.goodcopbadcop_chaser import (
    GoodCopBadCopTimeWindow,
    build_arena_heatmap,
    build_controls,
    build_controls_panel_from_widgets,
    build_detection_occupancy_output,
    build_spatial_occupancy_output,
    is_goodcopbadcop_option,
    load_goodcopbadcop_view,
)
from apps.marimo.components.registry import (
    artifact_path_for,
    discover_interactive_spec_options,
    supported_renderer_ids,
)
from fisheye.analysis.chaser_distance_runs import write_chaser_distance_run
from fisheye.visualization.goodcopbadcop_interactive import (
    DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT,
    GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
)
from tests.unit.fisheye.test_goodcopbadcop_interactive import (
    _make_archive_with_detection_occupancy,
    _make_chaser_result,
)


def _make_archive_with_goodcopbadcop_spec(tmp_path):
    zarr_path = _make_archive_with_detection_occupancy(tmp_path)
    write_chaser_distance_run(zarr_path, _make_chaser_result(zarr_path), overwrite=True)
    return zarr_path


def test_palette_explorer_registry_discovers_goodcopbadcop_interactive_spec(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_spec(tmp_path)

    options = discover_interactive_spec_options(zarr_path)

    assert len(options) == 1
    option = options[0]
    assert option.renderer == GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER
    assert option.renderer in supported_renderer_ids()
    assert option.is_supported is True
    assert is_goodcopbadcop_option(option) is True
    assert option.run_path == "analysis/chaser_distance_runs/chaser_distance_1"
    assert option.artifact_name == DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT
    assert option.artifact_path == artifact_path_for(option.run_path, option.artifact_name)
    assert option.spec["source_runs"]["detection_occupancy"] == "occupancy_1"

    filtered = discover_interactive_spec_options(
        zarr_path,
        renderer_filter=GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
        run_path_filter=option.run_path,
        artifact_filter=option.artifact_name,
    )
    assert filtered == options
    assert discover_interactive_spec_options(zarr_path, renderer_filter="missing-renderer") == []


def test_goodcopbadcop_component_loads_selected_registry_option(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]

    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)

    assert loaded.data.run_name == "chaser_distance_1"
    assert loaded.data.distance_mm.shape == (9, 2)
    assert loaded.windows_df["label"].tolist() == ["pre_event", "training_event", "post_event"]
    assert "distance_mm_chaser_0" in loaded.distance_df.columns
    assert loaded.position_df["fish_valid"].tolist() == [True, True, True, True, False, True, True, True, True]
    assert loaded.spatial_occupancy_df["zone_set_id"].unique().tolist() == ["image_quadrants_v1"]
    assert loaded.spatial_occupancy_df["frame_count"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert loaded.load_duration_ms >= 0.0


def test_goodcopbadcop_controls_do_not_read_widget_values_during_creation(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]
    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)

    class _StrictWidget:
        def __init__(self, value):
            self._initial_value = value

        @property
        def value(self):
            raise RuntimeError("same-cell widget value access")

    class _Ui:
        @staticmethod
        def multiselect(*, options, value, label):
            return _StrictWidget(value)

        @staticmethod
        def range_slider(*, start, stop, value, step, label):
            return _StrictWidget(value)

        @staticmethod
        def dropdown(*, options, value, label):
            return _StrictWidget(value)

        @staticmethod
        def slider(*, start, stop, value, step, label):
            return _StrictWidget(value)

        @staticmethod
        def checkbox(*, value, label):
            return _StrictWidget(value)

    class _Mo:
        ui = _Ui()

        @staticmethod
        def vstack(items):
            return list(items)

    controls = build_controls(_Mo, loaded=loaded)

    assert len(controls.view) == 6


def test_goodcopbadcop_spatial_figures_use_image_y_axis(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]
    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)
    window = GoodCopBadCopTimeWindow(
        selected_epoch_id=1,
        selected_epoch_label="training_event",
        start_s=0.3,
        stop_s=0.6,
    )

    class _Value:
        def __init__(self, value):
            self.value = value

    arena_fig, _visible_positions = build_arena_heatmap(
        px,
        loaded=loaded,
        heatmap_bins=_Value(20),
        chaser_overlay=_Value(True),
        window=window,
    )
    occupancy_fig = build_detection_occupancy_output(
        object(),
        go,
        loaded=loaded,
        window=window,
    )

    assert arena_fig.layout.yaxis.autorange == "reversed"
    assert arena_fig.layout.yaxis.title.text == "Arena Y (px, down)"
    chaser_position_traces = [
        trace for trace in arena_fig.data if str(getattr(trace, "name", "")).endswith(" position")
    ]
    assert [trace.name for trace in chaser_position_traces] == ["chaser 0 position", "chaser 1 position"]
    assert [list(trace.text) for trace in chaser_position_traces] == [["C0"], ["C1"]]
    assert all(trace.marker.symbol == "diamond" for trace in chaser_position_traces)
    assert [trace.marker.color for trace in chaser_position_traces] == ["#ff0000", "#0000ff"]
    assert occupancy_fig.layout.yaxis.autorange == "reversed"
    assert occupancy_fig.layout.yaxis.title.text == "Source image Y (px, down)"


def test_goodcopbadcop_spatial_occupancy_panel_renders_zone_summary(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]
    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)
    window = GoodCopBadCopTimeWindow(
        selected_epoch_id=1,
        selected_epoch_label="training_event",
        start_s=0.3,
        stop_s=0.6,
    )

    class _Value:
        value = "image_quadrants_v1"

    class _Mo:
        @staticmethod
        def vstack(items):
            return list(items)

    figures = build_spatial_occupancy_output(
        _Mo,
        go,
        loaded=loaded,
        window=window,
        spatial_zone_set_picker=_Value(),
    )

    assert len(figures) == 2
    assert [figure.layout.title.text for figure in figures] == [
        "Spatial Occupancy Zones (image_quadrants_v1, pre_event)",
        "Spatial Occupancy Zones (image_quadrants_v1, post_event)",
    ]
    assert all(len(figure.data) == 4 for figure in figures)
    assert [figure.data[0].name for figure in figures] == ["pre_event", "post_event"]
    assert [trace.name for trace in figures[0].data[1:]] == [
        "chaser 0 zone",
        "chaser 1 zone",
        "multiple chasers",
    ]
    assert list(figures[0].data[1].marker.pattern.shape) == ["/"]
    assert list(figures[0].data[2].marker.pattern.shape) == ["|"]
    assert list(figures[0].data[3].marker.pattern.shape) == ["x"]
    assert figures[0].layout.legend.title.text == "Pattern key"
    assert list(figures[0].data[0].x) == ["Top left", "Top right", "Bottom left", "Bottom right"]
    assert list(figures[0].data[0].y) == [0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645]
    assert list(figures[1].data[0].y) == [0.8999999761581421, 1.0, 1.100000023841858, 1.2000000476837158]


def test_goodcopbadcop_spatial_occupancy_panel_marks_prepost_chaser_zones(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]
    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)
    window = GoodCopBadCopTimeWindow(
        selected_epoch_id=0,
        selected_epoch_label="pre_event",
        start_s=0.0,
        stop_s=0.3,
    )

    class _Value:
        value = "image_quadrants_v1"

    class _Mo:
        @staticmethod
        def vstack(items):
            return list(items)

    figures = build_spatial_occupancy_output(
        _Mo,
        go,
        loaded=loaded,
        window=window,
        spatial_zone_set_picker=_Value(),
    )

    pre_fig, post_fig = figures
    assert list(pre_fig.data[0].marker.pattern.shape) == ["/", "|", "", ""]
    assert [row[4] for row in pre_fig.data[0].customdata] == ["chaser 0", "chaser 1", "none", "none"]
    assert list(post_fig.data[0].marker.pattern.shape) == ["", "", "|", "/"]
    assert [row[4] for row in post_fig.data[0].customdata] == ["none", "none", "chaser 1", "chaser 0"]


def test_goodcopbadcop_controls_show_time_slider_only_for_custom_window() -> None:
    class _Widget:
        def __init__(self, value):
            self.value = value

    class _Mo:
        @staticmethod
        def vstack(items):
            return list(items)

    distance_picker = _Widget(["distance_mm_chaser_0"])
    time_slider = _Widget([0.0, 10.0])
    epoch_picker = _Widget("Custom time window")
    bins = _Widget(80)
    overlay = _Widget(True)
    epoch_options = {"Custom time window": None, "post_event": 2}

    custom_items = build_controls_panel_from_widgets(
        _Mo,
        distance_series_picker=distance_picker,
        time_window=time_slider,
        epoch_picker=epoch_picker,
        epoch_options=epoch_options,
        heatmap_bins=bins,
        chaser_overlay=overlay,
    )
    assert custom_items == [distance_picker, epoch_picker, time_slider, bins, overlay]

    epoch_picker.value = "post_event"
    epoch_items = build_controls_panel_from_widgets(
        _Mo,
        distance_series_picker=distance_picker,
        time_window=time_slider,
        epoch_picker=epoch_picker,
        epoch_options=epoch_options,
        heatmap_bins=bins,
        chaser_overlay=overlay,
    )
    assert epoch_items == [distance_picker, epoch_picker, bins, overlay]
