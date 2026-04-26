from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis import plot_track_kinematics as mod
from fisheye.shared.plot_artifacts import INTERACTIVE_SPEC_SCHEMA_ID, PNG_ARTIFACT_SCHEMA_ID


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
    _write_track_array(track, "positions_px", positions_px)
    _write_track_array(track, "positions_mm", positions_mm)
    _write_track_array(track, "speed_raw_px", speed_px)
    _write_track_array(track, "speed_raw_mm", speed_mm)
    _write_track_array(track, "speed_filtered_px", speed_px)
    _write_track_array(track, "speed_filtered_mm", speed_mm)
    _write_track_array(track, "speed_smoothed_px", speed_px)
    _write_track_array(track, "speed_smoothed_mm", speed_mm)
    _write_track_array(track, "speed_averaged_px", speed_px)
    _write_track_array(track, "speed_averaged_mm", speed_mm)
    _write_track_array(track, "smoothed_heading_degrees", np.linspace(0.0, 5.0, n, dtype=np.float32))
    _write_track_array(track, "smoothed_acceleration_px", np.zeros(n, dtype=np.float32))
    _write_track_array(track, "smoothed_acceleration_mm", np.zeros(n, dtype=np.float32))
    _write_track_array(track, "cumulative_distance_px", np.cumsum(speed_px).astype(np.float32))
    _write_track_array(track, "cumulative_distance_mm", np.cumsum(speed_mm).astype(np.float32))
    return zarr_path


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
            "--write-zarr-artifacts",
        ]
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["track_kinematics_runs"]["offline"]["track_kinematics_1"]
    visualizations = run["visualizations"]

    png = visualizations["track_kinematics_summary_track_0_png"]
    assert bytes(np.asarray(png[:], dtype=np.uint8)[:8]) == b"\x89PNG\r\n\x1a\n"
    assert png.attrs["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
    assert png.attrs["track_id"] == 0
    assert png.attrs["source_paths"]["time_seconds"].endswith("/tracks/id_0/time_seconds")
    assert png.attrs["parameters"]["bins"] == 8

    spec_group = visualizations["track_kinematics_summary_track_0_interactive"]
    assert spec_group.attrs["artifact_schema_id"] == INTERACTIVE_SPEC_SCHEMA_ID
    assert spec_group.attrs["snapshot_artifact"] == "track_kinematics_summary_track_0_png"
    spec_bytes = np.asarray(spec_group["spec_json"][:], dtype=np.uint8).tobytes()
    spec = json.loads(spec_bytes.decode("utf-8"))
    assert spec["schema_id"] == mod.TRACK_KINEMATICS_PLOT_SPEC_SCHEMA_ID
    assert spec["track_id"] == 0
    assert spec["source_paths"]["positions_mm"].endswith("/tracks/id_0/positions_mm")
    assert any(panel["id"] == "position_density" for panel in spec["panels"])

    manifest = run.attrs["visualizations"]
    assert manifest["track_kinematics_summary_track_0_png"]["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
    assert (
        manifest["track_kinematics_summary_track_0_interactive"]["artifact_schema_id"]
        == INTERACTIVE_SPEC_SCHEMA_ID
    )
