from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import zarr

from apps.marimo.components.analysis_catalog import group_specs_by_provider
from apps.marimo.components.bout_kinematics import (
    available_bout_analysis_ids,
    build_bout_metric_figure,
    build_bout_kinematics_output,
    load_bout_metric_projection,
    load_bout_snapshot,
)
from apps.marimo.components.registry import (
    discover_interactive_spec_options,
    discover_recording_explorer_spec_options,
    supported_renderer_ids,
)
from fisheye.shared.plot_artifacts import (
    write_interactive_plot_spec_artifact,
    write_png_visualization_artifact,
)
from fisheye.shared.zarr.columnar import write_columnar_dataset
from fisheye.visualization.bout_kinematics_interactive import (
    BOUT_EYE_GAZE_PLOT_RENDERER,
    BOUT_EYE_GAZE_PLOT_SPEC_SCHEMA_ID,
    BOUT_HEADING_PLOT_RENDERER,
    BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
    BOUT_MOVEMENT_PLOT_RENDERER,
    BOUT_MOVEMENT_PLOT_SPEC_SCHEMA_ID,
    LEGACY_BOUT_PLOT_RENDERER,
)


class _Mo:
    @staticmethod
    def md(value):
        return ("md", value)

    @staticmethod
    def vstack(value):
        return list(value)

    @staticmethod
    def hstack(value, **_kwargs):
        return list(value)

    @staticmethod
    def accordion(value):
        return dict(value)

    @staticmethod
    def tree(value):
        return dict(value)

    @staticmethod
    def callout(value, *, kind):
        return (kind, value)

    @staticmethod
    def stat(*, label, value):
        return (label, value)


def _make_bout_archive(tmp_path, *, legacy: bool = True):
    zarr_path = tmp_path / "bout.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    run = root.create_group("analysis/bout_kinematics_runs/bout_1")
    run.attrs["layout"] = "compact_tabular_v2"
    heading = np.zeros(
        6,
        dtype=[
            ("heading_level_bytes", "S32"),
            ("net_delta_heading_deg", "f8"),
            ("abs_net_delta_heading_deg", "f8"),
            ("pre_window_valid", "?"),
            ("post_window_valid", "?"),
        ],
    )
    heading["heading_level_bytes"] = ["heading_smoothed"] * 3 + ["heading_raw"] * 3
    heading["net_delta_heading_deg"] = [-20.0, 0.0, 20.0, -40.0, 0.0, 40.0]
    heading["abs_net_delta_heading_deg"] = np.abs(heading["net_delta_heading_deg"])
    heading["pre_window_valid"] = [True, False, True, True, True, True]
    heading["post_window_valid"] = True
    write_columnar_dataset(run, "heading_metrics", heading, shard_rows=None)

    movement = np.zeros(
        3,
        dtype=[
            ("detector_duration_s", "f8"),
            ("physical_active_path_length_mm", "f8"),
            ("physical_active_valid", "?"),
        ],
    )
    movement["detector_duration_s"] = [0.1, 0.2, 0.3]
    movement["physical_active_path_length_mm"] = [1.0, 2.0, 3.0]
    movement["physical_active_valid"] = [True, False, True]
    write_columnar_dataset(run, "movement_metrics", movement, shard_rows=None)

    eye = np.zeros(
        3,
        dtype=[
            ("within_bout_vergence_gaze_mean_deg", "f8"),
            ("within_eye_window_valid", "?"),
        ],
    )
    eye["within_bout_vergence_gaze_mean_deg"] = [5.0, 10.0, 15.0]
    eye["within_eye_window_valid"] = [True, False, True]
    write_columnar_dataset(run, "eye_gaze_metrics", eye, shard_rows=None)
    renderers = {
        "heading": BOUT_HEADING_PLOT_RENDERER,
        "movement": BOUT_MOVEMENT_PLOT_RENDERER,
        "eye_gaze": BOUT_EYE_GAZE_PLOT_RENDERER,
    }
    schemas = {
        "heading": BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
        "movement": BOUT_MOVEMENT_PLOT_SPEC_SCHEMA_ID,
        "eye_gaze": BOUT_EYE_GAZE_PLOT_SPEC_SCHEMA_ID,
    }
    prefixes = {
        "heading": "bout_kinematics_summary",
        "movement": "bout_movement_summary",
        "eye_gaze": "bout_eye_gaze_summary",
    }
    source_run = "analysis/bout_kinematics_runs/copied_source_run"
    source_paths = {
        "heading": {
            "heading_smoothed.net_delta_heading_deg": (
                f"{source_run}/heading_metrics/net_delta_heading_deg"
            ),
            "heading_smoothed.abs_net_delta_heading_deg": (
                f"{source_run}/heading_metrics/abs_net_delta_heading_deg"
            ),
            "heading_raw.net_delta_heading_deg": (
                f"{source_run}/heading_metrics/net_delta_heading_deg"
            ),
            "heading_raw.abs_net_delta_heading_deg": (
                f"{source_run}/heading_metrics/abs_net_delta_heading_deg"
            ),
        },
        "movement": {
            "movement.detector_duration_s": (
                f"{source_run}/movement_metrics/detector_duration_s"
            ),
            "movement.physical_active_path_length_mm": (
                f"{source_run}/movement_metrics/physical_active_path_length_mm"
            ),
        },
        "eye_gaze": {
            "eye_gaze.within_bout_vergence_gaze_mean_deg": (
                f"{source_run}/eye_gaze_metrics/within_bout_vergence_gaze_mean_deg"
            ),
        },
    }
    panels = {
        "heading": [
            {
                "kind": "facet_histogram",
                "metrics": ["net_delta_heading_deg", "abs_net_delta_heading_deg"],
                "bins": 10,
            }
        ],
        "movement": [
            {
                "kind": "facet_histogram",
                "metrics": ["detector_duration_s", "physical_active_path_length_mm"],
                "bins": 10,
            }
        ],
        "eye_gaze": [
            {
                "kind": "facet_histogram",
                "metrics": ["within_bout_vergence_gaze_mean_deg"],
                "bins": 10,
            }
        ],
    }
    for analysis_id in ("heading", "movement", "eye_gaze"):
        png_name = f"{prefixes[analysis_id]}_track_0_png"
        spec_name = f"{prefixes[analysis_id]}_track_0_interactive"
        renderer = LEGACY_BOUT_PLOT_RENDERER if legacy else renderers[analysis_id]
        write_png_visualization_artifact(
            run,
            png_name,
            b"\x89PNG\r\n\x1a\nFAKE",
            description=f"{analysis_id} snapshot",
            created_by="test",
        )
        write_interactive_plot_spec_artifact(
            run,
            spec_name,
            {
                "schema_id": schemas[analysis_id],
                "title": f"Bout {analysis_id}",
                "run_name": "copied_source_run",
                "renderer": renderer,
                "source_paths": source_paths[analysis_id],
                "parameters": {"bins": 12},
                "panels": panels[analysis_id],
                **(
                    {
                        "heading_levels": ["heading_smoothed", "heading_raw"],
                        "default_heading_level": "heading_smoothed",
                    }
                    if analysis_id == "heading"
                    else {}
                ),
            },
            description=f"{analysis_id} spec",
            created_by="test",
            renderer=renderer,
            snapshot_artifact=png_name,
        )
    return zarr_path


def test_legacy_bout_specs_are_exactly_normalized_and_fast_discovered(tmp_path) -> None:
    zarr_path = _make_bout_archive(tmp_path, legacy=True)

    options = discover_recording_explorer_spec_options(zarr_path)

    assert {option.renderer for option in options} == {
        BOUT_HEADING_PLOT_RENDERER,
        BOUT_MOVEMENT_PLOT_RENDERER,
        BOUT_EYE_GAZE_PLOT_RENDERER,
    }
    assert all(option.is_supported for option in options)
    assert all(option.renderer in supported_renderer_ids() for option in options)
    assert len(
        discover_recording_explorer_spec_options(
            zarr_path, renderer_filter=BOUT_MOVEMENT_PLOT_RENDERER
        )
    ) == 1
    assert len(
        discover_recording_explorer_spec_options(
            zarr_path, renderer_filter=LEGACY_BOUT_PLOT_RENDERER
        )
    ) == 3
    grouped = group_specs_by_provider(options)
    assert list(grouped) == ["bout_kinematics"]
    assert grouped["bout_kinematics"][0].renderer == BOUT_HEADING_PLOT_RENDERER


def test_unrelated_legacy_generic_spec_is_not_claimed_as_bout(tmp_path) -> None:
    zarr_path = tmp_path / "other.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    run = root.create_group("analysis/other_runs/other_1")
    write_interactive_plot_spec_artifact(
        run,
        "other_interactive",
        {
            "schema_id": "palette.plot_spec.other.v1",
            "renderer": LEGACY_BOUT_PLOT_RENDERER,
        },
        description="other spec",
        created_by="test",
        renderer=LEGACY_BOUT_PLOT_RENDERER,
    )

    assert discover_recording_explorer_spec_options(zarr_path) == []
    broad = discover_interactive_spec_options(zarr_path)
    assert len(broad) == 1
    assert broad[0].renderer == LEGACY_BOUT_PLOT_RENDERER
    assert broad[0].is_supported is False


def test_bout_component_exposes_companions_and_loads_only_selected_snapshot(tmp_path) -> None:
    zarr_path = _make_bout_archive(tmp_path, legacy=False)
    options = discover_recording_explorer_spec_options(zarr_path)
    selected = next(option for option in options if option.renderer == BOUT_HEADING_PLOT_RENDERER)

    assert available_bout_analysis_ids(zarr_path, selected) == (
        "heading",
        "movement",
        "eye_gaze",
        "provenance",
    )
    movement = next(option for option in options if option.renderer == BOUT_MOVEMENT_PLOT_RENDERER)
    snapshot = load_bout_snapshot(zarr_path, movement, analysis_id="movement")
    assert snapshot.artifact_path.endswith("bout_movement_summary_track_0_png")
    assert snapshot.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    output = build_bout_kinematics_output(
        _Mo,
        zarr_path=zarr_path,
        selected_option=selected,
        analysis_id="movement",
    )
    assert output[0] == ("md", "### Bout movement")
    assert output[1][0] == "md"
    assert "data:image/png;base64" in output[1][1]


def test_bout_metric_projection_reads_selected_columns_and_filters_heading_level(tmp_path) -> None:
    zarr_path = _make_bout_archive(tmp_path, legacy=True)
    options = discover_recording_explorer_spec_options(zarr_path)
    selected = next(option for option in options if option.renderer == BOUT_HEADING_PLOT_RENDERER)

    projection = load_bout_metric_projection(
        zarr_path,
        selected,
        analysis_id="heading",
        metric="net_delta_heading_deg",
        heading_level="heading_smoothed",
        bins=12,
        valid_only=True,
    )

    assert projection.source_row_count == 6
    assert projection.selected_level_row_count == 3
    assert projection.finite_row_count == 3
    assert projection.validity_excluded_count == 1
    assert projection.plotted_row_count == 2
    assert projection.counts.sum() == 2
    assert projection.median == 0.0
    assert len(projection.source_paths_read) == 4
    assert all("/bout_1/" in path for path in projection.source_paths_read)
    assert not hasattr(projection, "values")

    figure = build_bout_metric_figure(go, projection)
    assert len(figure.data) == 2
    assert len(figure.data[0].x) == 12
    assert max(figure.data[1].y) == 100.0


def test_bout_metric_projection_applies_metric_specific_movement_validity(tmp_path) -> None:
    zarr_path = _make_bout_archive(tmp_path, legacy=False)
    options = discover_recording_explorer_spec_options(zarr_path)
    selected = next(option for option in options if option.renderer == BOUT_HEADING_PLOT_RENDERER)

    projection = load_bout_metric_projection(
        zarr_path,
        selected,
        analysis_id="movement",
        metric="physical_active_path_length_mm",
        bins=5,
        valid_only=True,
    )

    assert projection.selected_level_row_count == 3
    assert projection.validity_excluded_count == 1
    assert projection.plotted_row_count == 2
    assert projection.median == 2.0
    assert projection.source_paths_read[-1].endswith("physical_active_valid")

    output = build_bout_kinematics_output(
        _Mo,
        zarr_path=zarr_path,
        selected_option=selected,
        analysis_id="movement",
        go=go,
        projection=projection,
        show_snapshot=False,
    )
    assert output[0] == ("md", "### Bout movement")
    assert output[1].data[0].type == "bar"


def test_bout_metric_projection_reads_hierarchical_legacy_table(tmp_path) -> None:
    zarr_path = tmp_path / "hierarchical.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    run = root.create_group("analysis/bout_kinematics_runs/hierarchical_1")
    level = run.create_group("heading_smoothed")
    rows = np.zeros(
        3,
        dtype=[
            ("net_delta_heading_deg", "f8"),
            ("pre_window_valid", "?"),
            ("post_window_valid", "?"),
        ],
    )
    rows["net_delta_heading_deg"] = [-15.0, 0.0, 15.0]
    rows["pre_window_valid"] = True
    rows["post_window_valid"] = [True, False, True]
    write_columnar_dataset(level, "per_bout_metrics", rows, shard_rows=None)
    write_interactive_plot_spec_artifact(
        run,
        "bout_kinematics_summary_track_0_interactive",
        {
            "schema_id": BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "title": "Hierarchical bout heading",
            "run_name": "hierarchical_1",
            "renderer": LEGACY_BOUT_PLOT_RENDERER,
            "layout": "hierarchical_v1",
            "heading_levels": ["heading_smoothed"],
            "default_heading_level": "heading_smoothed",
            "source_paths": {
                "heading_smoothed.net_delta_heading_deg": (
                    "analysis/bout_kinematics_runs/hierarchical_1/"
                    "heading_smoothed/per_bout_metrics/net_delta_heading_deg"
                )
            },
            "panels": [
                {
                    "kind": "facet_histogram",
                    "metrics": ["net_delta_heading_deg"],
                    "bins": 10,
                }
            ],
        },
        description="hierarchical heading spec",
        created_by="test",
        renderer=LEGACY_BOUT_PLOT_RENDERER,
    )
    selected = discover_recording_explorer_spec_options(zarr_path)[0]

    projection = load_bout_metric_projection(
        zarr_path,
        selected,
        analysis_id="heading",
        metric="net_delta_heading_deg",
        heading_level="heading_smoothed",
        bins=10,
        valid_only=True,
    )

    assert projection.source_row_count == 3
    assert projection.selected_level_row_count == 3
    assert projection.plotted_row_count == 2
    assert len(projection.source_paths_read) == 3
