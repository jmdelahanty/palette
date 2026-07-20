from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from rich.console import Console

from fisheye.analysis import plot_track_kinematics as mod
from fisheye.shared.zarr.columnar import write_columnar_dataset
from fisheye.shared.plot_artifacts import INTERACTIVE_SPEC_SCHEMA_ID, PNG_ARTIFACT_SCHEMA_ID
from tests.unit.fisheye.test_track_kinematics_io import _patch_bound_loader


@pytest.fixture(autouse=True)
def _verified_track_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_bound_loader(monkeypatch)


def _write_track_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    group.create_array(name, data=values, chunks=values.shape, overwrite=True)


def _make_track_kinematics_archive(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    analysis = root.create_group("analysis")
    parent = analysis.create_group("track_kinematics_runs")
    offline = parent.create_group("offline")
    offline.attrs["latest"] = "track_kinematics_1"

    run = offline.create_group("track_kinematics_1")
    run.attrs["pixel_to_mm"] = 0.1
    run.attrs["fps"] = 200.0
    run.attrs["coordinate_space"] = "images_ds"
    run.create_array("track_ids", data=np.asarray([0], dtype=np.int32), overwrite=True)

    tracks = run.create_group("tracks")
    track = tracks.create_group("id_0")
    n = 6
    time_seconds = np.arange(n, dtype=np.float32) / 200.0
    positions_px = np.column_stack(
        [
            np.linspace(10.0, 20.0, n, dtype=np.float32),
            np.linspace(30.0, 45.0, n, dtype=np.float32),
        ]
    )
    positions_mm = positions_px * 0.1
    speed_px = np.linspace(1.0, 3.0, n, dtype=np.float32)
    speed_mm = speed_px * 0.1

    _write_track_array(track, "time_seconds", time_seconds)
    _write_track_array(track, "frame_indices", np.arange(n, dtype=np.int32))
    _write_track_array(
        track,
        "source_acquisition_frame_index",
        np.arange(n, dtype=np.int64),
    )
    _write_track_array(
        track,
        "track_sample_key",
        np.column_stack(
            [np.zeros(n, dtype=np.int64), np.arange(n, dtype=np.int64)]
        ),
    )
    _write_track_array(
        track,
        "source_frame_interpolation",
        np.zeros(n, dtype=np.int8),
    )
    _write_track_array(track, "source_instance_key", np.arange(n, dtype=np.int64))
    _write_track_array(track, "source_row_index", np.arange(n, dtype=np.int64))
    _write_track_array(track, "positions_px", positions_px)
    _write_track_array(track, "positions_mm", positions_mm)
    _write_track_array(
        track,
        "delta_seconds",
        np.asarray([np.nan, *([1.0 / 200.0] * (n - 1))], dtype=np.float32),
    )
    _write_track_array(track, "speed_raw_px", speed_px)
    _write_track_array(track, "speed_raw_mm", speed_mm)
    _write_track_array(track, "speed_filtered_px", speed_px)
    _write_track_array(track, "speed_filtered_mm", speed_mm)
    _write_track_array(track, "speed_smoothed_px", speed_px)
    _write_track_array(track, "speed_smoothed_mm", speed_mm)
    _write_track_array(track, "speed_averaged_px", speed_px)
    _write_track_array(track, "speed_averaged_mm", speed_mm)
    _write_track_array(track, "smoothed_heading_degrees", np.linspace(0.0, 5.0, n, dtype=np.float32))
    _write_track_array(track, "heading_degrees", np.linspace(0.0, 5.0, n, dtype=np.float32))
    _write_track_array(track, "heading_radians", np.deg2rad(np.linspace(0.0, 5.0, n)).astype(np.float32))
    _write_track_array(track, "smoothed_heading_radians", np.deg2rad(np.linspace(0.0, 5.0, n)).astype(np.float32))
    _write_track_array(track, "delta_heading_degrees", np.linspace(0.0, 2.0, n, dtype=np.float32))
    _write_track_array(track, "angular_velocity_deg_s", np.linspace(0.0, 20.0, n, dtype=np.float32))
    _write_track_array(track, "angular_velocity_raw_deg_s", np.linspace(0.0, 20.0, n, dtype=np.float32))
    _write_track_array(track, "angular_speed_raw_deg_s", np.linspace(0.0, 20.0, n, dtype=np.float32))
    _write_track_array(track, "delta_heading_smoothed_degrees", np.linspace(0.0, 1.0, n, dtype=np.float32))
    _write_track_array(track, "angular_velocity_smoothed_deg_s", np.linspace(0.0, 10.0, n, dtype=np.float32))
    _write_track_array(track, "angular_speed_smoothed_deg_s", np.linspace(0.0, 10.0, n, dtype=np.float32))
    _write_track_array(track, "detection_source", np.ones(n, dtype=np.int16))
    _write_track_array(track, "acceleration_px", np.zeros(n, dtype=np.float32))
    _write_track_array(track, "acceleration_mm", np.zeros(n, dtype=np.float32))
    _write_track_array(track, "smoothed_acceleration_px", np.zeros(n, dtype=np.float32))
    _write_track_array(track, "smoothed_acceleration_mm", np.zeros(n, dtype=np.float32))
    _write_track_array(track, "cumulative_path_distance_px", np.cumsum(speed_px).astype(np.float32))
    _write_track_array(track, "cumulative_path_distance_mm", np.cumsum(speed_mm).astype(np.float32))
    movement_speed = track.create_group("movement").create_group("speed")
    for idx, level in enumerate(("raw", "filtered", "smoothed", "averaged"), start=1):
        level_group = movement_speed.create_group(level)
        acceleration_px = np.full(n, float(idx), dtype=np.float32)
        acceleration_mm = acceleration_px * 0.1
        smoothed_acceleration_px = acceleration_px + 0.5
        smoothed_acceleration_mm = smoothed_acceleration_px * 0.1
        _write_track_array(level_group, "px", speed_px)
        _write_track_array(level_group, "mm", speed_mm)
        _write_track_array(level_group, "acceleration_px", acceleration_px)
        _write_track_array(level_group, "acceleration_mm", acceleration_mm)
        _write_track_array(level_group, "smoothed_acceleration_px", smoothed_acceleration_px)
        _write_track_array(level_group, "smoothed_acceleration_mm", smoothed_acceleration_mm)
        if level != "averaged":
            _write_track_array(level_group, "frame_path_distance_px", speed_px)
            _write_track_array(level_group, "frame_path_distance_mm", speed_mm)
    _write_track_array(track, "transition_valid", np.asarray([False, True, False, True, True, True], dtype=bool))
    _write_track_array(track, "transition_reason_code", np.asarray([1, 0, 2, 0, 0, 0], dtype=np.int16))
    _write_track_array(track, "sample_valid", np.asarray([True, True, True, False, True, True], dtype=bool))
    _write_track_array(track, "sample_reason_code", np.asarray([0, 0, 0, 4, 0, 0], dtype=np.int16))
    track.attrs["transition_reason_codes"] = {"0": "ok", "1": "first_sample", "2": "frame_gap"}
    track.attrs["sample_reason_codes"] = {"0": "ok", "4": "keypoint_failed"}
    return zarr_path


def _add_swim_bout_run(
    zarr_path: Path,
    *,
    run_name: str = "swim_bout_1",
    default_level: str = "speed_filtered",
    levels: tuple[str, ...] = ("speed_filtered",),
    use_frame_fields: bool = False,
) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    parent = root["analysis"].create_group("swim_bout_runs")
    parent.attrs["latest"] = run_name
    run = parent.create_group(run_name)
    run.attrs.update(
        {
            "default_level": default_level,
            "detection_method": "threshold",
            "fps": 100.0,
            "source_track_kinematics_run": "track_kinematics_1",
            "track_id": 0,
        }
    )
    if use_frame_fields:
        bouts = np.asarray(
            [(1, 10, 20), (2, 30, 45)],
            dtype=[("bout_id", "i4"), ("start_frame", "i8"), ("end_frame", "i8")],
        )
    else:
        bouts = np.asarray(
            [(1, 0.10, 0.20), (2, 0.30, 0.45)],
            dtype=[("bout_id", "i4"), ("start_time_s", "f8"), ("end_time_s", "f8")],
        )
    for level in levels:
        level_group = run.create_group(level)
        write_columnar_dataset(level_group, "bouts", bouts)


def _add_flat_legacy_swim_bout_run(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    parent = root["analysis"].create_group("swim_bout_runs")
    parent.attrs["latest"] = "legacy_bouts"
    run = parent.create_group("legacy_bouts")
    run.attrs.update({"detection_method": "legacy", "fps": 100.0})
    bouts = np.asarray(
        [(1, 5, 15)],
        dtype=[("bout_id", "i4"), ("start_frame", "i8"), ("end_frame", "i8")],
    )
    write_columnar_dataset(run, "bouts", bouts)


def test_resolve_swim_bout_spans_uses_resolver_for_selected_level(tmp_path: Path) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)
    _add_swim_bout_run(
        zarr_path,
        levels=("speed_filtered", "speed_smoothed"),
    )
    root = zarr.open_group(str(zarr_path), mode="r")

    spans, label = mod.resolve_swim_bout_spans(
        root,
        "latest",
        Console(record=True),
        speed_level="smoothed",
    )

    assert spans == [(0.10, 0.20), (0.30, 0.45)]
    assert label == "swim_bout_1 (speed_smoothed) (threshold)"


def test_resolve_swim_bout_spans_falls_back_to_default_and_converts_frames(
    tmp_path: Path,
) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)
    _add_swim_bout_run(zarr_path, use_frame_fields=True)
    root = zarr.open_group(str(zarr_path), mode="r")
    console = Console(record=True)

    spans, label = mod.resolve_swim_bout_spans(
        root,
        "swim_bout_1",
        console,
        speed_level="smoothed",
    )

    assert spans == [(0.10, 0.20), (0.30, 0.45)]
    assert label == "swim_bout_1 (speed_filtered) (threshold)"
    assert "using default 'speed_filtered'" in console.export_text()


def test_resolve_swim_bout_spans_keeps_flat_legacy_support(tmp_path: Path) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)
    _add_flat_legacy_swim_bout_run(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="r")

    spans, label = mod.resolve_swim_bout_spans(
        root,
        "latest",
        Console(record=True),
        speed_level="smoothed",
    )

    assert spans == [(0.05, 0.15)]
    assert label == "legacy_bouts (legacy)"


def test_plot_track_kinematics_writes_png_and_interactive_spec_artifacts(tmp_path: Path) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)

    mod.main(
        [
            str(zarr_path),
            "--offline-only",
            "--track-id",
            "0",
            "--swim-bout-run",
            "none",
            "--bins",
            "8",
        ]
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["track_kinematics_runs"]["offline"]["track_kinematics_1"]
    assert "visualizations" not in run
    render_parent = root["analysis"]["track_kinematics_visualization_runs"][
        "offline"
    ]["track_kinematics_1"]["tracks"]["id_0"]
    render = render_parent[render_parent.attrs["latest"]]
    assert render.attrs["palette_run_completion_status"] == "complete"
    assert render.attrs["stage_selector_eligible"] is True
    visualizations = render["visualizations"]

    png = visualizations["track_kinematics_summary_track_0_png"]
    assert bytes(np.asarray(png[:], dtype=np.uint8)[:8]) == b"\x89PNG\r\n\x1a\n"
    assert png.attrs["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
    assert png.attrs["visualization_contract_id"] == (
        mod.TRACK_KINEMATICS_VISUALIZATION_CONTRACT_ID
    )
    assert png.attrs["renderer"] == mod.TRACK_KINEMATICS_PLOT_RENDERER
    assert png.attrs["renderer_version"] == mod.TRACK_KINEMATICS_RENDERER_VERSION
    assert png.attrs["track_id"] == 0
    assert png.attrs["source_paths"]["time_seconds"].endswith("/tracks/id_0/time_seconds")
    assert png.attrs["parameters"]["bins"] == 8
    assert png.attrs["parameters"]["speed_level"] == mod.DEFAULT_SWIM_BOUT_OVERLAY_SPEED_LEVEL
    png_provenance = png.attrs["provenance"]
    assert png_provenance["contract"]["name"] == "palette_stage_provenance"
    assert png_provenance["stage"] == "track_kinematics_visualization"
    assert png_provenance["parameters"]["bins"] == 8
    assert png_provenance["parameters"]["renderer"] == mod.TRACK_KINEMATICS_PLOT_RENDERER
    assert png_provenance["inputs"]["source_runs"]["track_kinematics"] == "offline/track_kinematics_1"
    assert png_provenance["artifacts"]["png_artifact"].endswith("track_kinematics_summary_track_0_png")

    spec_group = visualizations["track_kinematics_summary_track_0_interactive"]
    assert spec_group.attrs["artifact_schema_id"] == INTERACTIVE_SPEC_SCHEMA_ID
    assert spec_group.attrs["snapshot_artifact"] == "track_kinematics_summary_track_0_png"
    spec_bytes = np.asarray(spec_group["spec_json"][:], dtype=np.uint8).tobytes()
    spec = json.loads(spec_bytes.decode("utf-8"))
    assert spec["schema_id"] == mod.TRACK_KINEMATICS_PLOT_SPEC_SCHEMA_ID
    assert spec["track_id"] == 0
    assert spec["source_paths"]["positions_mm"].endswith("/tracks/id_0/positions_mm")
    assert spec["source_paths"]["speed_smoothed_mm"].endswith(
        "/tracks/id_0/movement/speed/smoothed/mm"
    )
    assert spec["source_paths"]["smoothed_acceleration_mm"].endswith(
        "/tracks/id_0/movement/speed/smoothed/smoothed_acceleration_mm"
    )
    assert spec["source_paths"]["speed_filtered_acceleration_mm"].endswith(
        "/tracks/id_0/movement/speed/filtered/acceleration_mm"
    )
    assert spec["source_paths"]["speed_filtered_frame_path_distance_mm"].endswith(
        "/tracks/id_0/movement/speed/filtered/frame_path_distance_mm"
    )
    assert spec["source_paths"]["angular_speed_smoothed_deg_s"].endswith(
        "/tracks/id_0/angular_speed_smoothed_deg_s"
    )
    assert spec["source_paths"]["transition_valid"].endswith("/tracks/id_0/transition_valid")
    assert spec["source_paths"]["sample_valid"].endswith("/tracks/id_0/sample_valid")
    assert any(panel["id"] == "position_density" for panel in spec["panels"])
    assert any(panel["id"] == "turning" for panel in spec["panels"])
    spec_provenance = spec_group.attrs["provenance"]
    assert spec_provenance["stage"] == "track_kinematics_visualization"
    assert spec_provenance["artifacts"]["interactive_artifact"].endswith(
        "track_kinematics_summary_track_0_interactive"
    )

    position_xy = visualizations["position_xy_trace_track_0_png"]
    assert bytes(np.asarray(position_xy[:], dtype=np.uint8)[:8]) == b"\x89PNG\r\n\x1a\n"
    assert position_xy.attrs["visualization_contract_id"] == (
        mod.POSITION_XY_TRACE_VISUALIZATION_CONTRACT_ID
    )
    assert position_xy.attrs["renderer"] == mod.POSITION_XY_TRACE_RENDERER
    assert position_xy.attrs["renderer_version"] == mod.POSITION_XY_TRACE_RENDERER_VERSION
    assert position_xy.attrs["source_paths"]["positions_mm"].endswith(
        "/tracks/id_0/positions_mm"
    )

    manifest = render.attrs["visualizations"]
    assert manifest["track_kinematics_summary_track_0_png"]["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
    assert manifest["track_kinematics_summary_track_0_png"]["visualization_contract_id"] == (
        mod.TRACK_KINEMATICS_VISUALIZATION_CONTRACT_ID
    )
    assert manifest["position_xy_trace_track_0_png"]["visualization_contract_id"] == (
        mod.POSITION_XY_TRACE_VISUALIZATION_CONTRACT_ID
    )
    assert (
        manifest["track_kinematics_summary_track_0_interactive"]["artifact_schema_id"]
        == INTERACTIVE_SPEC_SCHEMA_ID
    )


def test_plot_track_kinematics_can_skip_default_zarr_artifacts(tmp_path: Path) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)
    save_path = tmp_path / "track_plot.png"

    mod.main(
        [
            str(zarr_path),
            "--offline-only",
            "--track-id",
            "0",
            "--swim-bout-run",
            "none",
            "--save",
            str(save_path),
            "--no-write-zarr-artifacts",
        ]
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["track_kinematics_runs"]["offline"]["track_kinematics_1"]
    assert "visualizations" not in run
    assert "track_kinematics_visualization_runs" not in root["analysis"]
    assert save_path.with_name("track_plot_offline_track_kinematics_1.png").exists()


def test_plot_track_kinematics_accepts_exponential_swim_bout_overlay(tmp_path: Path) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)
    _add_swim_bout_run(
        zarr_path,
        run_name="peak_event_bouts",
        default_level="speed_exponential",
        levels=("speed_filtered", "speed_exponential"),
    )

    mod.main(
        [
            str(zarr_path),
            "--offline-only",
            "--track-id",
            "0",
            "--swim-bout-run",
            "peak_event_bouts",
            "--speed-level",
            "exponential",
            "--write-zarr-artifacts",
        ]
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    render_parent = root["analysis"]["track_kinematics_visualization_runs"][
        "offline"
    ]["track_kinematics_1"]["tracks"]["id_0"]
    render = render_parent[render_parent.attrs["latest"]]
    spec_group = render["visualizations"]["track_kinematics_summary_track_0_interactive"]
    spec_bytes = np.asarray(spec_group["spec_json"][:], dtype=np.uint8).tobytes()
    spec = json.loads(spec_bytes.decode("utf-8"))

    assert spec["overlays"]["swim_bouts"]["speed_level"] == "exponential"
    assert spec["overlays"]["swim_bouts"]["resolved_label"] == (
        "peak_event_bouts (speed_exponential) (threshold)"
    )
    assert spec_group.attrs["parameters"]["speed_level"] == "exponential"


def test_visualization_publication_restores_selectors_on_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    tables = mod.load_track_kinematics_track(
        root,
        run_name="track_kinematics_1",
        scope="offline",
        track_id=0,
        required_speed_levels=("raw", "smoothed"),
    )
    parent = root.require_group(
        "analysis/track_kinematics_visualization_runs/offline/"
        "track_kinematics_1/tracks/id_0"
    )
    parent.create_group("prior")
    parent.attrs["latest"] = "prior"
    parent.attrs["latest_complete"] = "prior"

    def interrupt(**_kwargs) -> None:
        raise KeyboardInterrupt("hostile interrupt")

    monkeypatch.setattr(mod, "write_track_kinematics_plot_artifacts", interrupt)

    with pytest.raises(KeyboardInterrupt, match="hostile interrupt"):
        mod.publish_track_kinematics_plot_artifacts(
            root=root,
            zarr_path=zarr_path,
            run_name="offline/track_kinematics_1",
            track_tables=tables,
            track_id=0,
            png_bytes=b"not-used",
            bins=8,
            artifact_dpi=72,
            swim_bout_label=None,
            swim_bout_requested=None,
            speed_level="smoothed",
            distance_series_present=False,
            stimulus_run=None,
            console=Console(record=True),
        )

    assert parent.attrs["latest"] == "prior"
    assert parent.attrs["latest_complete"] == "prior"
    assert list(parent.group_keys()) == ["prior"]
    source_run = root[
        "analysis/track_kinematics_runs/offline/track_kinematics_1"
    ]
    assert "visualizations" not in source_run


def test_visualization_selectors_are_independent_per_track(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    tables_0 = mod.load_track_kinematics_track(
        root,
        run_name="track_kinematics_1",
        scope="offline",
        track_id=0,
        required_speed_levels=("raw", "smoothed"),
    )
    tables_1 = replace(
        tables_0,
        track_id=1,
        track_path=(
            "analysis/track_kinematics_runs/offline/"
            "track_kinematics_1/tracks/id_1"
        ),
    )

    def reload_track(_root, *, track_id: int, **_kwargs):
        return tables_0 if track_id == 0 else tables_1

    monkeypatch.setattr(mod, "load_track_kinematics_track", reload_track)
    for track_id, tables in ((0, tables_0), (1, tables_1)):
        mod.publish_track_kinematics_plot_artifacts(
            root=root,
            zarr_path=zarr_path,
            run_name="offline/track_kinematics_1",
            track_tables=tables,
            track_id=track_id,
            png_bytes=f"png-{track_id}".encode(),
            bins=8,
            artifact_dpi=72,
            swim_bout_label=None,
            swim_bout_requested=None,
            speed_level="smoothed",
            distance_series_present=False,
            stimulus_run=None,
            console=Console(record=True),
        )

    source_parent = root[
        "analysis/track_kinematics_visualization_runs/offline/track_kinematics_1"
    ]
    for track_id in (0, 1):
        parent = source_parent[f"tracks/id_{track_id}"]
        render = parent[parent.attrs["latest_complete"]]
        assert render.attrs["track_id"] == track_id
        assert (
            f"track_kinematics_summary_track_{track_id}_png"
            in render["visualizations"]
        )


def test_visualization_seal_rejects_equal_shaped_source_path_substitution(
    tmp_path: Path,
) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)
    mod.main(
        [
            str(zarr_path),
            "--offline-only",
            "--track-id",
            "0",
            "--swim-bout-run",
            "none",
        ]
    )
    root = zarr.open_group(str(zarr_path), mode="a")
    parent = root[
        "analysis/track_kinematics_visualization_runs/offline/"
        "track_kinematics_1/tracks/id_0"
    ]
    render = parent[parent.attrs["latest_complete"]]
    tables = mod.load_track_kinematics_track(
        root,
        run_name="track_kinematics_1",
        scope="offline",
        track_id=0,
        required_speed_levels=("raw", "smoothed"),
    )
    expected_paths = mod._track_source_paths("offline/track_kinematics_1", tables)
    artifact = render["visualizations/track_kinematics_summary_track_0_png"]
    artifact.attrs["source_paths"] = {
        **expected_paths,
        "positions_px": "decoy_positions_px",
    }

    with pytest.raises(ValueError, match="failed integrity validation"):
        mod._validate_track_visualization_run(
            render,
            track_id=0,
            source_authority=tables.authority_record(),
            expected_source_paths=expected_paths,
        )
