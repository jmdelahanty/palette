from __future__ import annotations

import sqlite3
import time

import apps.marimo.components.registry as registry_component
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import zarr

from apps.marimo.components.goodcopbadcop_chaser import (
    GoodCopBadCopTimeWindow,
    _minmax_line_display_frame,
    available_chaser_analysis_ids,
    build_arena_heatmap,
    build_controls,
    build_controls_panel_from_widgets,
    build_cra_near_field_output,
    build_cra_primary_endpoint_output,
    build_detection_occupancy_output,
    build_egocentric_alignment_output,
    build_egocentric_bearing_output,
    build_egocentric_polar_heatmap_output,
    build_egocentric_static_polar_output,
    build_epoch_summary_output,
    build_fish_heading_output,
    build_spatial_occupancy_output,
    is_goodcopbadcop_option,
    load_goodcopbadcop_view,
    resolve_time_windows_from_multiselect,
)
from apps.marimo.components.registry import (
    artifact_path_for,
    discover_interactive_spec_options,
    discover_recording_explorer_spec_options,
    discover_protocol_recording_options,
    infer_recordings_root_from_zarr_path,
    recording_id_from_analysis_zarr,
    supported_renderer_ids,
)
from fisheye.analysis.cra_primary_endpoint import (
    build_cra_primary_endpoint_result,
    write_cra_primary_endpoint_component,
)
from fisheye.analysis.cra_near_field import (
    build_cra_near_field_result,
    write_cra_near_field_component,
)
from fisheye.analysis.chaser_distance_runs import write_chaser_distance_run
from fisheye.analysis.chaser_state_interpolator import write_columnar_dataset
from fisheye.visualization.goodcopbadcop_interactive import (
    CHASER_DASHBOARD_RENDERER,
    DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT,
    GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
)
from fisheye.analysis.chaser_egocentric_bearing import (
    PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME,
    PRE_POST_POLAR_PNG_ARTIFACT_NAME,
    build_chaser_egocentric_bearing_result,
    write_chaser_egocentric_bearing_component,
)
from fisheye.analysis.goodcopbadcop_epoch_behavior_summary import (
    build_goodcopbadcop_epoch_behavior_summary_result,
    write_goodcopbadcop_epoch_behavior_summary_component,
)
from tests.unit.fisheye.test_chaser_egocentric_bearing import _add_track_kinematics_run
from tests.unit.fisheye.test_cra_near_field import _add_circle_geometry
from tests.unit.fisheye.test_goodcopbadcop_interactive import (
    _make_archive_with_detection_occupancy,
    _make_chaser_result,
)
from tests.unit.fisheye.test_export_cross_recording_analytics import _add_goodcopbadcop_cra_protocol_metadata


def _make_archive_with_goodcopbadcop_spec(tmp_path):
    zarr_path = _make_archive_with_detection_occupancy(tmp_path)
    write_chaser_distance_run(zarr_path, _make_chaser_result(zarr_path), overwrite=True)
    return zarr_path


def _make_archive_with_goodcopbadcop_egocentric_spec(tmp_path):
    zarr_path = _make_archive_with_goodcopbadcop_spec(tmp_path)
    _add_track_kinematics_run(zarr_path)
    result = build_chaser_egocentric_bearing_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
        track_kinematics_run="tk_1",
    )
    write_chaser_egocentric_bearing_component(zarr_path, result, overwrite=True)
    return zarr_path


def _add_swim_bout_run(zarr_path):
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    track = root["analysis/track_kinematics_runs/offline/tk_1/tracks/id_0"]
    track.create_array(
        "speed_filtered_mm",
        data=np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=np.float32),
        chunks=(8,),
        overwrite=True,
    )
    tk_run = root["analysis/track_kinematics_runs/offline/tk_1"]
    tk_run.attrs["fps"] = 10.0
    tk_run.attrs["pixel_to_mm"] = 0.02

    parent = root["analysis"].require_group("swim_bout_runs")
    parent.attrs["latest"] = "bouts_1"
    run = parent.create_group("bouts_1")
    run.attrs.update(
        {
            "default_level": "filtered",
            "source_track_kinematics_run": "tk_1",
            "track_id": 0,
            "detection_method": "threshold",
        }
    )
    level = run.create_group("speed_filtered")
    level.attrs["n_bouts"] = 4

    bout_dtype = np.dtype(
        [
            ("bout_id", np.int32),
            ("peak_time_s", np.float64),
            ("start_time_s", np.float64),
            ("end_time_s", np.float64),
            ("start_frame", np.int64),
            ("end_frame", np.int64),
            ("duration_s", np.float64),
            ("path_length_mm", np.float64),
        ]
    )
    bouts = np.zeros(4, dtype=bout_dtype)
    bouts["bout_id"] = [0, 1, 2, 3]
    bouts["peak_time_s"] = [0.10, 0.20, 0.45, 0.70]
    bouts["start_time_s"] = [0.08, 0.18, 0.42, 0.68]
    bouts["end_time_s"] = [0.12, 0.24, 0.50, 0.76]
    bouts["start_frame"] = [0, 1, 4, 7]
    bouts["end_frame"] = [1, 2, 5, 8]
    bouts["duration_s"] = [0.04, 0.06, 0.08, 0.08]
    bouts["path_length_mm"] = [0.2, 0.3, 0.4, 0.5]
    write_columnar_dataset(level, "bouts", bouts, {"n_bouts": 4})

    interval_dtype = np.dtype(
        [
            ("interval_id", np.int32),
            ("valid", bool),
            ("prev_end_time_s", np.float64),
            ("next_start_time_s", np.float64),
            ("interval_s", np.float64),
        ]
    )
    intervals = np.zeros(2, dtype=interval_dtype)
    intervals["interval_id"] = [0, 1]
    intervals["valid"] = [True, True]
    intervals["prev_end_time_s"] = [0.12, 0.50]
    intervals["next_start_time_s"] = [0.18, 0.68]
    intervals["interval_s"] = [0.06, 0.18]
    write_columnar_dataset(level, "inter_bout_intervals", intervals, {"n_intervals": 2})


def _make_archive_with_goodcopbadcop_cra_spec(tmp_path):
    zarr_path = _make_archive_with_goodcopbadcop_spec(tmp_path)
    _add_goodcopbadcop_cra_protocol_metadata(zarr_path)
    result = build_cra_primary_endpoint_result(zarr_path, chaser_distance_run="chaser_distance_1")
    write_cra_primary_endpoint_component(zarr_path, result, overwrite=True)
    return zarr_path


def _make_archive_with_goodcopbadcop_cra_near_field_spec(tmp_path):
    zarr_path = _make_archive_with_goodcopbadcop_cra_spec(tmp_path)
    _add_circle_geometry(zarr_path)
    result = build_cra_near_field_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
        cra_primary_endpoint_component="object_relative_pre_post_v1",
        r_zone_mm=2.0,
        r_in_mm=2.0,
        r_out_mm=3.0,
        percentile_values=(5.0, 10.0),
        radial_bin_edges_mm=(0.0, 2.0, 4.0, 8.0),
        cdf_thresholds_mm=(2.0, 4.0),
        perimeter_band_mm=2.0,
    )
    write_cra_near_field_component(zarr_path, result, overwrite=True)
    return zarr_path


def _make_recording_archive_with_goodcopbadcop_spec(recordings_root, recording_id):
    zarr_dir = recordings_root / recording_id / "zarr"
    zarr_dir.mkdir(parents=True)
    return _make_archive_with_goodcopbadcop_spec(zarr_dir)


def test_palette_explorer_registry_discovers_goodcopbadcop_interactive_spec(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_spec(tmp_path)

    options = discover_interactive_spec_options(zarr_path)

    assert len(options) == 1
    option = options[0]
    assert option.renderer == CHASER_DASHBOARD_RENDERER
    assert option.renderer in supported_renderer_ids()
    assert option.is_supported is True
    assert is_goodcopbadcop_option(option) is True
    assert option.run_path == "analysis/chaser_distance_runs/chaser_distance_1"
    assert option.artifact_name == DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT
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
    assert discover_recording_explorer_spec_options(zarr_path) == options


def test_palette_explorer_discovers_interactive_specs_from_manifest_without_recursive_walk(
    tmp_path,
    monkeypatch,
) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_spec(tmp_path)

    def _fail_recursive_walk(_root):
        raise AssertionError("recursive visualization walk should not be used when manifest entries exist")

    monkeypatch.setattr(registry_component, "iter_visualization_artifacts", _fail_recursive_walk)

    options = discover_interactive_spec_options(zarr_path)

    assert len(options) == 1
    assert options[0].renderer == CHASER_DASHBOARD_RENDERER


def test_palette_explorer_registry_backed_recording_list_is_lazy(
    tmp_path,
    monkeypatch,
) -> None:
    first = _make_archive_with_goodcopbadcop_spec(
        tmp_path / "2026-06-23T16-01-09Z_arena_1_RedScare" / "zarr"
    )
    second = _make_archive_with_goodcopbadcop_spec(
        tmp_path / "2026-06-23T17-16-51Z_arena_3_RedScare" / "zarr"
    )
    other = _make_archive_with_goodcopbadcop_spec(
        tmp_path / "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop" / "zarr"
    )
    registry = tmp_path / "palette_registry.sqlite"
    with sqlite3.connect(registry) as conn:
        conn.execute(
            """
            CREATE TABLE datasets (
                dataset_id TEXT,
                recording_id TEXT,
                zarr_path TEXT,
                zarr_use TEXT,
                status TEXT
            );
            """
        )
        conn.executemany(
            "INSERT INTO datasets(dataset_id, recording_id, zarr_path, zarr_use, status) VALUES (?, ?, ?, ?, ?);",
            [
                ("red-1", "2026-06-23T16-01-09Z_arena_1_RedScare", str(first), "analysis", "active"),
                ("red-2", "2026-06-23T17-16-51Z_arena_3_RedScare", str(second), "analysis", "active"),
                ("other-1", "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop", str(other), "analysis", "active"),
                ("missing-1", "2026-06-23T18-00-00Z_arena_4_RedScare", str(tmp_path / "missing.zarr"), "analysis", "missing"),
            ],
        )

    def _fail_directory_discovery(*_args, **_kwargs):
        raise AssertionError("directory discovery should not be used when registry_path is provided")

    def _fail_spec_discovery(*_args, **_kwargs):
        raise AssertionError("registry-backed recording list should not open each zarr eagerly")

    monkeypatch.setattr(registry_component, "_candidate_analysis_zarrs", _fail_directory_discovery)
    monkeypatch.setattr(registry_component, "discover_interactive_spec_options", _fail_spec_discovery)

    options = discover_protocol_recording_options(
        first,
        registry_path=registry,
        name_contains="RedScare",
    )

    assert [option.recording_id for option in options] == [
        "2026-06-23T16-01-09Z_arena_1_RedScare",
        "2026-06-23T17-16-51Z_arena_3_RedScare",
    ]
    assert {option.zarr_path for option in options} == {first, second}
    assert all(option.spec_counts_loaded is False for option in options)
    assert all(option.interactive_spec_count == 0 for option in options)
    assert all(option.supported_spec_count == 0 for option in options)


def test_palette_explorer_discovers_sibling_goodcopbadcop_recordings(tmp_path) -> None:
    recordings_root = tmp_path / "recordings"
    first = _make_recording_archive_with_goodcopbadcop_spec(
        recordings_root,
        "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop",
    )
    second = _make_recording_archive_with_goodcopbadcop_spec(
        recordings_root,
        "2026-06-14T21-12-08Z_arena_2_GoodCopBadCop",
    )
    other_dir = recordings_root / "2026-06-14T21-12-08Z_arena_3_OtherProtocol" / "zarr"
    other_dir.mkdir(parents=True)
    _make_archive_with_goodcopbadcop_spec(other_dir)

    assert infer_recordings_root_from_zarr_path(first) == recordings_root
    assert recording_id_from_analysis_zarr(first) == "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop"

    options = discover_protocol_recording_options(
        first,
        recordings_root=recordings_root,
        renderer_filter=GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
    )

    assert [option.recording_id for option in options] == [
        "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop",
        "2026-06-14T21-12-08Z_arena_2_GoodCopBadCop",
    ]
    assert {option.zarr_path for option in options} == {first, second}
    assert all(option.supported_spec_count == 1 for option in options)
    assert all(
        option.renderer_counts == {CHASER_DASHBOARD_RENDERER: 1}
        for option in options
    )


def test_palette_explorer_direct_launch_does_not_discover_siblings(tmp_path) -> None:
    recordings_root = tmp_path / "recordings"
    selected = _make_recording_archive_with_goodcopbadcop_spec(
        recordings_root,
        "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop",
    )
    _make_recording_archive_with_goodcopbadcop_spec(
        recordings_root,
        "2026-06-14T21-12-08Z_arena_2_GoodCopBadCop",
    )

    options = discover_protocol_recording_options(
        selected,
        name_contains=None,
        recording_explorer_only=True,
        include_collection=False,
    )

    assert [option.zarr_path for option in options] == [selected]


def test_palette_explorer_direct_launch_reports_selected_zarr_without_specs(tmp_path) -> None:
    selected = tmp_path / "recording_analysis.zarr"
    zarr.open_group(str(selected), mode="w", use_consolidated=False)

    options = discover_protocol_recording_options(
        selected,
        name_contains=None,
        recording_explorer_only=True,
        include_collection=False,
        include_seed_without_specs=True,
    )

    assert len(options) == 1
    assert options[0].zarr_path == selected
    assert options[0].interactive_spec_count == 0
    assert options[0].supported_spec_count == 0


def test_goodcopbadcop_component_loads_selected_registry_option(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]

    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)

    assert loaded.data.run_name == "chaser_distance_1"
    assert loaded.data.distance_mm.shape == (9, 2)
    assert loaded.windows_df["label"].to_list() == ["pre_event", "training_event", "post_event"]
    assert "distance_mm_chaser_0" in loaded.distance_df.columns
    assert loaded.position_df["fish_valid"].to_list() == [True, True, True, True, False, True, True, True, True]
    assert loaded.spatial_occupancy_df["zone_set_id"].unique().to_list() == ["image_quadrants_v1"]
    assert loaded.spatial_occupancy_df["frame_count"].to_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert loaded.epoch_summary_df.height == 6
    assert loaded.load_duration_ms >= 0.0


def test_goodcopbadcop_component_projects_only_selected_analysis_family(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_egocentric_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]

    loaded = load_goodcopbadcop_view(
        zarr_path,
        option,
        timer=time,
        include_companion_analyses=False,
        analysis_id="distance",
    )

    assert not loaded.distance_df.is_empty()
    assert loaded.position_df.is_empty()
    assert loaded.egocentric_bearing_df.is_empty()
    assert loaded.epoch_summary_df.is_empty()
    assert loaded.data.egocentric_bearing_deg is None
    assert loaded.data.occupancy_normalized is None


def test_chaser_analysis_choices_follow_persisted_components(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_cra_near_field_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]

    analysis_ids = available_chaser_analysis_ids(zarr_path, option)

    assert "distance" in analysis_ids
    assert "position_heatmap" in analysis_ids
    assert "cra_quadrant" in analysis_ids
    assert "cra_near_field" in analysis_ids
    assert "escape_freeze" not in analysis_ids


def test_goodcopbadcop_component_summarizes_swim_bouts_by_epoch(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_egocentric_spec(tmp_path)
    _add_swim_bout_run(zarr_path)
    option = discover_interactive_spec_options(zarr_path)[0]

    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)
    pre_chaser_0 = loaded.epoch_summary_df.filter(
        (pl.col("window_label") == "pre_event") & (pl.col("chaser_index") == 0)
    ).row(0, named=True)

    assert loaded.epoch_summary_source == "analysis/swim_bout_runs/bouts_1"
    assert pre_chaser_0["bout_count"] == 2
    assert pre_chaser_0["inter_bout_interval_count"] == 1
    assert pre_chaser_0["median_inter_bout_interval_s"] == 0.06
    assert pre_chaser_0["speed_sample_count"] == 3
    assert pre_chaser_0["mean_speed_mm_s"] == 20.0
    assert np.isfinite(pre_chaser_0["median_distance_mm"])

    class _Ui:
        @staticmethod
        def table(frame, selection=None, page_size=10):
            return frame

    class _Mo:
        ui = _Ui()

        @staticmethod
        def md(text):
            return text

        @staticmethod
        def stat(*, label, value):
            return {"label": label, "value": value}

        @staticmethod
        def hstack(items, **_kwargs):
            return list(items)

        @staticmethod
        def vstack(items):
            return list(items)

    class _ChaserPicker:
        value = "chaser 1"

    output = build_epoch_summary_output(_Mo, loaded=loaded, chaser_picker=_ChaserPicker())
    table = output[-1]
    assert table["chaser_index"].to_list() == [1, 1, 1]
    assert table["bout_count"].to_list() == [2, 1, 1]


def test_goodcopbadcop_component_prefers_persisted_epoch_behavior_summary(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_egocentric_spec(tmp_path)
    _add_swim_bout_run(zarr_path)
    result = build_goodcopbadcop_epoch_behavior_summary_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
    )
    component_path = write_goodcopbadcop_epoch_behavior_summary_component(
        zarr_path,
        result,
        overwrite=True,
    )
    option = discover_interactive_spec_options(zarr_path)[0]

    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)
    pre_chaser_0 = loaded.epoch_summary_df.filter(
        (pl.col("window_label") == "pre_event") & (pl.col("chaser_index") == 0)
    ).row(0, named=True)

    assert loaded.epoch_behavior is not None
    assert loaded.epoch_summary_source == component_path
    assert loaded.epoch_summary_error is None
    assert loaded.epoch_summary_computed_in_viewer is False
    assert loaded.epoch_behavior.per_epoch_fish_df.height == 3
    assert pre_chaser_0["bout_count"] == 2
    assert pre_chaser_0["mean_inter_bout_interval_s"] == 0.06

    class _Ui:
        @staticmethod
        def table(frame, selection=None, page_size=10):
            return frame

    class _Mo:
        ui = _Ui()

        @staticmethod
        def md(text):
            return text

        @staticmethod
        def stat(*, label, value):
            return {"label": label, "value": value}

        @staticmethod
        def hstack(items, **_kwargs):
            return list(items)

        @staticmethod
        def vstack(items):
            return list(items)

    output = build_epoch_summary_output(_Mo, go, loaded=loaded)

    assert "persisted zarr component" in output[0]
    summary_plot_rows = output[2]
    distribution_plot_rows = output[4]
    ibi_distribution_plot_rows = output[6]
    summary_plots = [figure for row in summary_plot_rows for figure in row if hasattr(figure, "layout")]
    distribution_plots = [
        figure for row in distribution_plot_rows for figure in row if hasattr(figure, "layout")
    ]
    ibi_distribution_plots = [
        figure for row in ibi_distribution_plot_rows for figure in row if hasattr(figure, "layout")
    ]
    assert [len(row) for row in summary_plot_rows] == [2, 2, 2, 2]
    assert [len(row) for row in distribution_plot_rows] == [2, 2, 2]
    assert [len(row) for row in ibi_distribution_plot_rows] == [2]
    assert [figure.layout.title.text for figure in summary_plots] == [
        "Bout Rate by Epoch",
        "Bout Count by Epoch",
        "Inter-Bout Interval Count by Epoch",
        "Mean Inter-Bout Interval by Epoch",
        "Mean Bout Duration by Epoch",
        "Mean Bout Distance by Epoch",
        "Mean Net Bout Heading Change by Epoch",
        "Mean Absolute Net Bout Heading Change by Epoch",
    ]
    assert output[3] == "## Swim Bout Distributions"
    assert [figure.layout.title.text for figure in distribution_plots] == [
        "Swim Bout Duration Distribution",
        "Swim Bout Distance Distribution",
        "Swim Bout Net Heading Change Distribution",
        "Swim Bout Absolute Net Heading Change Distribution",
        "Swim Bout Heading Path Distribution",
    ]
    assert output[5] == "## Inter-Bout Interval Distributions"
    assert [figure.layout.title.text for figure in ibi_distribution_plots] == [
        "Inter-Bout Interval Distribution",
    ]
    assert distribution_plots[0].data[0].type == "bar"
    assert ibi_distribution_plots[0].data[0].type == "bar"
    assert summary_plots[0].layout.yaxis.title.text == "Bouts / min"
    assert summary_plots[1].layout.yaxis.title.text == "Bouts"
    assert all(figure.layout.height == 360 for figure in summary_plots)
    assert all(figure.layout.height == 360 for figure in distribution_plots)
    assert "bout_rate_per_min" in output[-1].columns
    assert "tracking_dropout_fraction" in output[-1].columns
    assert "mean_bout_duration_s" in output[-1].columns
    assert "mean_bout_path_length_mm" in output[-1].columns
    assert "mean_bout_net_heading_change_deg" in output[-1].columns
    assert "mean_abs_bout_net_heading_change_deg" in output[-1].columns
    assert "mean_inter_bout_interval_s" in output[-1].columns
    assert output[-1]["mean_inter_bout_interval_s"].to_list()[0] == 0.06


def test_goodcopbadcop_component_loads_and_renders_egocentric_static_polar_png(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_egocentric_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]

    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)

    assert loaded.data.egocentric_component_path is not None
    assert loaded.egocentric_pre_post_polar_png_path is not None
    assert loaded.egocentric_pre_post_polar_png_path.endswith(
        f"/visualizations/{PRE_POST_POLAR_PNG_ARTIFACT_NAME}"
    )
    assert loaded.egocentric_pre_post_polar_png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert loaded.egocentric_pre_post_polar_png_error is None
    assert loaded.egocentric_pre_post_polar_point_cloud_png_path is not None
    assert loaded.egocentric_pre_post_polar_point_cloud_png_path.endswith(
        f"/visualizations/{PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME}"
    )
    assert loaded.egocentric_pre_post_polar_point_cloud_png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert loaded.egocentric_pre_post_polar_point_cloud_png_error is None

    class _Mo:
        @staticmethod
        def md(text):
            return text

        @staticmethod
        def vstack(items):
            return list(items)

    output = build_egocentric_static_polar_output(_Mo, loaded=loaded)

    assert "Persisted Egocentric Bearing Point Clouds" in output[0]
    assert PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME in output[0]
    assert "data:image/png;base64," in output[1]
    assert "Persisted Egocentric Bearing Circular Histograms" in output[2]
    assert PRE_POST_POLAR_PNG_ARTIFACT_NAME in output[2]
    assert "data:image/png;base64," in output[3]


def test_goodcopbadcop_component_loads_and_renders_cra_endpoint(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_cra_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]

    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)

    endpoint = loaded.cra_endpoint
    assert endpoint is not None
    assert endpoint.component_path.endswith("/cra_primary_endpoint/object_relative_pre_post_v1")
    assert endpoint.summary["delta_occ_agg"] == -1.0
    assert endpoint.summary["pre_aggressive_quadrant"] == "top_left"
    assert endpoint.summary["post_aggressive_quadrant"] == "bottom_right"

    root = zarr.open_group(str(zarr_path), mode="r")
    component = root[endpoint.component_path]
    stored_object_x = np.asarray(component["object_phase/object_x_px"][:], dtype=float).reshape(-1)
    stored_occupancy = np.asarray(component["per_object_phase/occupancy_fraction"][:], dtype=float).reshape(-1)
    loaded_object_rows = endpoint.object_phase_df.sort(["phase_index", "object_index"])
    loaded_metric_rows = endpoint.per_object_phase_df.sort(["phase_index", "object_index"])
    np.testing.assert_allclose(loaded_object_rows["object_x_px"].to_numpy(), stored_object_x)
    np.testing.assert_allclose(loaded_metric_rows["occupancy_fraction"].to_numpy(), stored_occupancy)
    assert loaded_object_rows["object_quadrant"].to_list() == [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    ]

    class _Ui:
        @staticmethod
        def table(data, *, selection=None, page_size=10):
            return data

    class _Mo:
        ui = _Ui()

        @staticmethod
        def md(text):
            return text

        @staticmethod
        def hstack(items):
            return list(items)

        @staticmethod
        def vstack(items):
            return list(items)

        @staticmethod
        def stat(*, label, value):
            return {"label": label, "value": value}

        @staticmethod
        def accordion(items):
            return dict(items)

    output = build_cra_primary_endpoint_output(_Mo, go, loaded=loaded)
    figures = [item for item in output if hasattr(item, "data")]

    assert "CRA Primary Endpoint" in output[0]
    assert [figure.layout.title.text for figure in figures[:2]] == [
        "CRA Primary Endpoint: Median Distance",
        "CRA Primary Endpoint: Object-Quadrant Occupancy",
    ]
    assert [figure.layout.title.text for figure in figures[2:]] == [
        "CRA Object-Relative Quadrants (pre_static)",
        "CRA Object-Relative Quadrants (post_static)",
    ]
    assert figures[2].layout.yaxis.autorange == "reversed"


def test_goodcopbadcop_component_loads_and_renders_cra_near_field(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_cra_near_field_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]

    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)

    near_field = loaded.cra_near_field
    assert near_field is not None
    assert near_field.component_path.endswith("/cra_near_field/object_relative_near_field_v1")
    assert near_field.geometry_status == "circle"
    assert near_field.arena_shape == "circle"
    assert near_field.summary["nearzone_occ_delta_agg"] == -1.0
    assert "approach_p05_mm" in near_field.per_object_phase_df.columns
    assert "approach_p05_cdf_fraction" in near_field.per_object_phase_df.columns
    assert "object_distance_to_wall_mm" in near_field.per_object_phase_df.columns
    assert near_field.radial_density_df.height == 12
    assert "radial_density_wall_excluded_per_mm2" in near_field.radial_density_df.columns
    assert near_field.cdf_df.height == 8
    assert near_field.control_reference_radial_density_df.height == 6
    assert near_field.control_reference_cdf_df.height == 4
    assert near_field.control_reference_phase_df.height == 2
    assert near_field.thigmotaxis_df.height == 2
    assert "mean_speed_mm_s" in near_field.thigmotaxis_df.columns

    class _Ui:
        @staticmethod
        def table(data, *, selection=None, page_size=10):
            return data

    class _Mo:
        ui = _Ui()

        @staticmethod
        def md(text):
            return text

        @staticmethod
        def hstack(items):
            return list(items)

        @staticmethod
        def vstack(items):
            return list(items)

        @staticmethod
        def stat(*, label, value):
            return {"label": label, "value": value}

        @staticmethod
        def accordion(items):
            return dict(items)

    output = build_cra_near_field_output(_Mo, go, loaded=loaded)
    figures = [item for item in output if hasattr(item, "data")]

    assert "CRA Near-Field Avoidance" in output[0]
    assert [figure.layout.title.text for figure in figures] == [
        "CRA Near-Field: Close-Approach Distance",
        "CRA Near-Field: Near-Zone Occupancy",
        "CRA Near-Field: Near-Zone Entry Rate",
        "CRA Near-Field: Radial Occupancy Density",
        "CRA Near-Field: Wall-Band-Excluded Radial Density",
        "CRA Near-Field: Distance CDF",
        "CRA Near-Field: Dish-Center Control Radial Density",
        "CRA Near-Field: Dish-Center Control CDF",
        "CRA Near-Field: Global-State QC",
    ]
    assert figures[5].layout.yaxis.range == (0, 1)


def test_goodcopbadcop_component_can_skip_companion_analysis_loads(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_cra_near_field_spec(tmp_path)
    _add_track_kinematics_run(zarr_path)
    _add_swim_bout_run(zarr_path)
    result = build_goodcopbadcop_epoch_behavior_summary_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
    )
    write_goodcopbadcop_epoch_behavior_summary_component(
        zarr_path,
        result,
        overwrite=True,
    )
    option = discover_interactive_spec_options(zarr_path)[0]

    loaded = load_goodcopbadcop_view(
        zarr_path,
        option,
        timer=time,
        include_companion_analyses=False,
    )

    assert loaded.cra_endpoint is None
    assert loaded.cra_near_field is None
    assert loaded.escape_freeze is None
    assert loaded.epoch_behavior is not None
    assert loaded.epoch_summary_computed_in_viewer is False


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

    assert len(controls.view) == 7
    assert len(controls.egocentric_epoch_picker._initial_value) == 3


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


def test_distance_line_display_budget_preserves_real_extrema() -> None:
    row_count = 10000
    distances = np.sin(np.arange(row_count, dtype=np.float64) / 100.0)
    distances[4321] = 50.0
    distances[6789] = -25.0
    frame = pl.DataFrame(
        {
            "time_s": np.arange(row_count, dtype=np.float64) / 100.0,
            "distance_mm": distances,
        }
    )

    display = _minmax_line_display_frame(
        frame,
        value_column="distance_mm",
        max_points=1000,
    )

    assert len(display) <= 1000
    assert display["time_s"].is_sorted()
    assert display["distance_mm"].max() == 50.0
    assert display["distance_mm"].min() == -25.0
    assert set(display["time_s"].to_list()).issubset(set(frame["time_s"].to_list()))


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

        @staticmethod
        def hstack(items, **_kwargs):
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


def test_goodcopbadcop_egocentric_panels_render_from_linked_component(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_egocentric_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]
    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)
    window = GoodCopBadCopTimeWindow(
        selected_epoch_id=2,
        selected_epoch_label="post_event",
        start_s=0.6,
        stop_s=0.9,
    )

    class _Mo:
        @staticmethod
        def md(text):
            return text

        @staticmethod
        def vstack(items):
            return list(items)

        @staticmethod
        def hstack(items, **_kwargs):
            return list(items)

    class _ChaserPicker:
        value = "chaser 1"

    bearing_fig = build_egocentric_bearing_output(
        _Mo,
        go,
        loaded=loaded,
        window=window,
    )
    density_figures = build_egocentric_polar_heatmap_output(
        _Mo,
        go,
        loaded=loaded,
        window=window,
    )
    heading_fig = build_fish_heading_output(
        _Mo,
        go,
        loaded=loaded,
        window=window,
    )
    alignment_fig = build_egocentric_alignment_output(
        _Mo,
        go,
        loaded=loaded,
        window=window,
    )
    single_bearing_fig = build_egocentric_bearing_output(
        _Mo,
        go,
        loaded=loaded,
        window=window,
        chaser_picker=_ChaserPicker(),
    )
    single_density_fig = build_egocentric_polar_heatmap_output(
        _Mo,
        go,
        loaded=loaded,
        window=window,
        chaser_picker=_ChaserPicker(),
    )
    single_alignment_fig = build_egocentric_alignment_output(
        _Mo,
        go,
        loaded=loaded,
        window=window,
        chaser_picker=_ChaserPicker(),
    )
    epoch_labels = loaded.windows_df["label"].to_list()
    epoch_windows = tuple(
        GoodCopBadCopTimeWindow(
            selected_epoch_id=int(row["window_id"]),
            selected_epoch_label=str(row["label"]),
            start_s=float(row["start_time_s"]),
            stop_s=float(row["end_time_s"]),
        )
        for row in loaded.windows_df.iter_rows(named=True)
    )
    multi_epoch_output = build_egocentric_bearing_output(
        _Mo,
        go,
        loaded=loaded,
        windows=epoch_windows,
    )

    assert loaded.data.egocentric_component_name == "track_offline_tk_1_id_0_smoothed"
    assert loaded.egocentric_bearing_df.height > 0
    assert loaded.egocentric_heading_df.height > 0
    assert [trace.name for trace in bearing_fig.data] == ["chaser 0", "chaser 1"]
    assert bearing_fig.layout.polar.angularaxis.rotation == 90
    assert list(bearing_fig.layout.polar.angularaxis.ticktext) == [
        "behind",
        "right",
        "front",
        "left",
        "behind",
    ]
    assert len(density_figures) == 2
    assert [figure.data[0].type for figure in density_figures] == ["barpolar", "barpolar"]
    assert density_figures[0].layout.polar.angularaxis.rotation == 90
    assert float(density_figures[0].data[0].r[0]) == 5.0
    assert float(density_figures[0].data[0].width[0]) == 30.0
    assert "5 mm x 30 deg display bins" in str(density_figures[0].layout.title.text)
    assert heading_fig.data[0].type == "scattergl"
    assert heading_fig.data[0].mode == "markers"
    assert heading_fig.layout.yaxis.title.text == "Fish heading (deg)"
    assert [trace.mode for trace in alignment_fig.data] == ["lines+markers", "lines+markers"]
    assert alignment_fig.layout.yaxis.title.text == "Mean cos(bearing)"
    assert [trace.name for trace in single_bearing_fig.data] == ["chaser 1"]
    assert single_density_fig.data[0].name == "chaser 1"
    assert [trace.name for trace in single_alignment_fig.data] == ["chaser 1"]
    assert multi_epoch_output[0] == "## Egocentric Chaser Bearing"
    multi_epoch_figures = multi_epoch_output[1]
    assert [figure.layout.title.text for figure in multi_epoch_figures] == epoch_labels
    assert len(multi_epoch_figures) == 3


def test_goodcopbadcop_multi_epoch_picker_resolves_selected_windows_in_time_order(tmp_path) -> None:
    zarr_path = _make_archive_with_goodcopbadcop_egocentric_spec(tmp_path)
    option = discover_interactive_spec_options(zarr_path)[0]
    loaded = load_goodcopbadcop_view(zarr_path, option, timer=time)

    class _Picker:
        value = ["post (0.6-0.9s)", "pre (0.0-0.3s)"]

    resolved = resolve_time_windows_from_multiselect(
        epoch_options={
            "Custom time window": None,
            "pre (0.0-0.3s)": 0,
            "training (0.3-0.6s)": 1,
            "post (0.6-0.9s)": 2,
        },
        epoch_picker=_Picker(),
        windows_df=loaded.windows_df,
    )

    assert [window.selected_epoch_id for window in resolved] == [0, 2]
    assert [window.selected_epoch_label for window in resolved] == ["pre_event", "post_event"]


def test_goodcopbadcop_controls_show_time_slider_only_for_custom_window() -> None:
    class _Widget:
        def __init__(self, value):
            self.value = value

    class _Mo:
        @staticmethod
        def vstack(items):
            return list(items)

    distance_picker = _Widget(["distance_mm_chaser_0"])
    chaser_picker = _Widget("All chasers")
    time_slider = _Widget([0.0, 10.0])
    epoch_picker = _Widget("Custom time window")
    bins = _Widget(80)
    overlay = _Widget(True)
    epoch_options = {"Custom time window": None, "post_event": 2}

    custom_items = build_controls_panel_from_widgets(
        _Mo,
        distance_series_picker=distance_picker,
        chaser_picker=chaser_picker,
        time_window=time_slider,
        epoch_picker=epoch_picker,
        epoch_options=epoch_options,
        heatmap_bins=bins,
        chaser_overlay=overlay,
    )
    assert custom_items == [distance_picker, chaser_picker, epoch_picker, time_slider, bins, overlay]

    epoch_picker.value = "post_event"
    epoch_items = build_controls_panel_from_widgets(
        _Mo,
        distance_series_picker=distance_picker,
        chaser_picker=chaser_picker,
        time_window=time_slider,
        epoch_picker=epoch_picker,
        epoch_options=epoch_options,
        heatmap_bins=bins,
        chaser_overlay=overlay,
    )
    assert epoch_items == [distance_picker, chaser_picker, epoch_picker, bins, overlay]
