from __future__ import annotations

import zarr

from apps.marimo.components.analysis_catalog import group_specs_by_provider
from apps.marimo.components.bout_kinematics import (
    available_bout_analysis_ids,
    build_bout_kinematics_output,
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
    def accordion(value):
        return dict(value)

    @staticmethod
    def tree(value):
        return dict(value)

    @staticmethod
    def callout(value, *, kind):
        return (kind, value)


def _make_bout_archive(tmp_path, *, legacy: bool = True):
    zarr_path = tmp_path / "bout.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    run = root.create_group("analysis/bout_kinematics_runs/bout_1")
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
                "run_name": "bout_1",
                "renderer": renderer,
                "source_paths": {"metrics": f"{analysis_id}/per_bout_metrics"},
                "parameters": {"bins": 12},
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
