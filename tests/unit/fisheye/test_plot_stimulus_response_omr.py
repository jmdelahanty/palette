from __future__ import annotations

import json

import numpy as np
import zarr

from fisheye.analysis.plot_stimulus_response_omr import (
    OMR_BOUT_TRAJECTORY_INTERACTIVE_ARTIFACT_NAME,
    OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME,
    OMR_SUMMARY_INTERACTIVE_ARTIFACT_NAME,
    OMR_SUMMARY_PNG_ARTIFACT_NAME,
    STIMULUS_RESPONSE_OMR_PLOT_RENDERER,
    write_omr_summary_visualization,
)


def _make_omr_run() -> zarr.Group:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("stimulus_response_runs")
    run = parent.create_group("omr_test")
    parent.attrs["latest"] = "omr_test"
    run.attrs.update(
        {
            "source_track_kinematics_run": "tk_test",
            "source_track_kinematics_type": "offline",
            "source_stimulus_run": "stim_test",
            "source_bout_run": "bouts_test",
        }
    )
    tk_parent = analysis.create_group("track_kinematics_runs").create_group("offline")
    tk_run = tk_parent.create_group("tk_test")
    tk_parent.attrs["latest"] = "tk_test"
    tk_run.attrs["fps"] = 10.0
    tracks = tk_run.create_group("tracks")
    track = tracks.create_group("id_0")
    frames = np.arange(500, dtype=np.int64)
    track.create_array("frame_indices", data=frames)
    track.create_array("time_seconds", data=(frames / 10.0).astype(np.float32))
    positions = np.stack(
        [
            10.0 + np.cos(frames / 24.0) * 3.0 + frames * 0.01,
            10.0 + np.sin(frames / 24.0) * 2.0,
        ],
        axis=1,
    ).astype(np.float32)
    track.create_array("positions_mm", data=positions)
    track.create_array("heading_degrees", data=np.full(frames.shape, 25.0, dtype=np.float32))

    steps = run.create_group("steps")
    for step_index, direction in [(0, 0.0), (4, 180.0)]:
        step = steps.create_group(f"step_{step_index}")
        step.attrs.update(
            {
                "step_index": step_index,
                "step_name": f"moving_{step_index}",
                "start_frame": step_index * 100,
                "end_frame": step_index * 100 + 100,
                "duration_s": 10.0,
            }
        )
        omr = step.create_group("grating").create_group("omr")
        omr.attrs.update(
            {
                "stimulus_direction_deg": direction,
                "arena_center_mm": [10.0, 10.0],
                "arena_axis_extent_mm": 20.0,
                "arena_geometry_source": "test_calibration",
            }
        )
        per_fish = omr.create_group("per_fish")
        per_fish.create_array("fish_id", data=np.array([0, 1], dtype=np.int32))
        per_fish.create_array("omr_path_index", data=np.array([0.7, 0.5], dtype=np.float32))
        per_fish.create_array("bout_choice_index", data=np.array([1.0, 0.0], dtype=np.float32))
        per_fish.create_array("time_choice_index", data=np.array([0.5, -0.1], dtype=np.float32))
        per_fish.create_array("start_position_axis_norm", data=np.array([-0.5, 0.1], dtype=np.float32))
        per_fish.create_array("mean_position_axis_norm", data=np.array([0.0, 0.2], dtype=np.float32))
        per_fish.create_array("end_position_axis_norm", data=np.array([0.6, 0.4], dtype=np.float32))
        per_fish.create_array("fraction_time_correct_side", data=np.array([0.8, 0.6], dtype=np.float32))
        per_fish.create_array("first_aligned_bout_latency_s", data=np.array([0.5, np.nan], dtype=np.float32))
        per_fish.create_array("first_classified_bout_latency_s", data=np.array([0.2, 0.3], dtype=np.float32))
        per_fish.create_array("first_opposing_bout_latency_s", data=np.array([np.nan, 1.2], dtype=np.float32))

        per_bout = omr.create_group("per_bout")
        base = step_index * 100
        per_bout.create_array("fish_id", data=np.array([0, 0, 0], dtype=np.int32))
        per_bout.create_array("bout_id", data=np.array([base + 1, base + 2, base + 3], dtype=np.int32))
        per_bout.create_array("start_frame", data=np.array([base + 5, base + 20, base + 45], dtype=np.int64))
        per_bout.create_array("end_frame", data=np.array([base + 12, base + 28, base + 54], dtype=np.int64))
        per_bout.create_array("per_bout_omr_score", data=np.array([0.7, -0.4, 0.02], dtype=np.float32))
        per_bout.create_array("correct_label", data=np.array([1, -1, 0], dtype=np.int8))

        windows = omr.create_group("windows")
        windows.create_array("fish_id", data=np.array([0, 1, 0, 1], dtype=np.int32))
        windows.create_array("start_time_s", data=np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32))
        windows.create_array("window_length_s", data=np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32))
        windows.create_array("omr_path_index", data=np.array([0.5, 0.4, -0.2, 0.1], dtype=np.float32))

    return root


def test_write_omr_summary_visualization_writes_png_and_spec() -> None:
    root = _make_omr_run()

    result = write_omr_summary_visualization(
        root,
        run_name="omr_test",
        artifact_dpi=80,
        command="test command",
    )

    run = root["analysis"]["stimulus_response_runs"]["omr_test"]
    visualizations = run["visualizations"]
    png = visualizations[OMR_SUMMARY_PNG_ARTIFACT_NAME]
    spec = visualizations[OMR_SUMMARY_INTERACTIVE_ARTIFACT_NAME]
    trajectory_png = visualizations[OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME]
    trajectory_spec = visualizations[OMR_BOUT_TRAJECTORY_INTERACTIVE_ARTIFACT_NAME]

    assert result["n_omr_steps"] == 2
    assert result["bout_trajectory_png_artifact"] == f"visualizations/{OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME}"
    assert np.asarray(png[:], dtype=np.uint8).tobytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert np.asarray(trajectory_png[:], dtype=np.uint8).tobytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert png.attrs["n_omr_steps"] == 2
    assert trajectory_png.attrs["track_id"] == 0
    assert spec.attrs["renderer"] == STIMULUS_RESPONSE_OMR_PLOT_RENDERER
    assert spec.attrs["snapshot_artifact"] == OMR_SUMMARY_PNG_ARTIFACT_NAME
    assert trajectory_spec.attrs["snapshot_artifact"] == OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME

    spec_bytes = np.asarray(spec["spec_json"][:], dtype=np.uint8).tobytes()
    spec_json = json.loads(spec_bytes.decode("utf-8"))
    assert spec_json["steps"][0]["omr_path"].endswith("steps/step_0/grating/omr")
    assert "first_aligned_bout_latency_s" in spec_json["steps"][0]["primary_fields"]
    trajectory_spec_bytes = np.asarray(trajectory_spec["spec_json"][:], dtype=np.uint8).tobytes()
    trajectory_spec_json = json.loads(trajectory_spec_bytes.decode("utf-8"))
    assert trajectory_spec_json["track_path"].endswith("tracks/id_0")
    assert trajectory_spec_json["artifact_family"] == "stimulus_response_omr_bout_trajectory"

    manifest = run.attrs["visualizations"]
    assert manifest[OMR_SUMMARY_PNG_ARTIFACT_NAME]["path"] == f"visualizations/{OMR_SUMMARY_PNG_ARTIFACT_NAME}"
    assert manifest[OMR_SUMMARY_INTERACTIVE_ARTIFACT_NAME]["snapshot_artifact"] == OMR_SUMMARY_PNG_ARTIFACT_NAME
    assert manifest[OMR_BOUT_TRAJECTORY_INTERACTIVE_ARTIFACT_NAME]["snapshot_artifact"] == OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME
    json.dumps(dict(png.attrs), allow_nan=False)
    json.dumps(dict(spec.attrs), allow_nan=False)
    json.dumps(dict(trajectory_png.attrs), allow_nan=False)
    json.dumps(dict(trajectory_spec.attrs), allow_nan=False)
    json.dumps(dict(run.attrs), allow_nan=False)
