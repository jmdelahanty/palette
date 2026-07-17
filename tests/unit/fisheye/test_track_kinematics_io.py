from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis.detect_bouts_multi_level import _load_track_kinematics_track_speeds
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track


def _array(group: zarr.Group, name: str, values) -> None:
    group.create_array(name, data=np.asarray(values), overwrite=True)


def _make_track_kinematics_archive(path: Path | None = None) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w") if path is not None else zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("track_kinematics_runs")
    offline = parent.create_group("offline")
    offline.attrs["latest"] = "tk_test"
    run = offline.create_group("tk_test")
    run.attrs.update(
        {
            "fps": 60.0,
            "pixel_to_mm": 0.1,
            "created_at_utc": "2026-05-10T00:00:00Z",
            "provenance": {"stage": "track_kinematics", "version": "test", "git": {"commit": "abc"}},
        }
    )
    _array(run, "track_ids", [0])
    tracks = run.create_group("tracks")
    track = tracks.create_group("id_0")
    track.attrs["track_id"] = 0

    _array(track, "frame_indices", [10, 11, 12])
    _array(track, "positions_mm", [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    _array(track, "positions_px", [[10.0, 20.0], [20.0, 20.0], [30.0, 20.0]])
    _array(track, "delta_seconds", [np.nan, 1.0 / 60.0, 1.0 / 60.0])
    _array(track, "transition_valid", [False, True, True])
    _array(track, "sample_valid", [True, True, True])

    for base, values in {
        "raw": [1.0, 2.0, 3.0],
        "filtered": [4.0, 5.0, 6.0],
        "smoothed": [7.0, 8.0, 9.0],
        "averaged": [10.0, 11.0, 12.0],
    }.items():
        source = f"speed_{base}"
        _array(track, f"{source}_mm", values)
        _array(track, f"{source}_px", np.asarray(values) * 10.0)
    for base, values in {
        "raw": [0.1, 0.2, 0.3],
        "filtered": [0.4, 0.5, 0.6],
        "smoothed": [0.7, 0.8, 0.9],
    }.items():
        _array(track, f"frame_path_distance_{base}_mm", values)
        _array(track, f"frame_path_distance_{base}_px", np.asarray(values) * 10.0)

    movement = track.create_group("movement")
    speed = movement.create_group("speed")
    filtered = speed.create_group("filtered")
    _array(filtered, "mm", [40.0, 50.0, 60.0])
    _array(filtered, "px", [400.0, 500.0, 600.0])
    _array(filtered, "frame_path_distance_mm", [4.0, 5.0, 6.0])
    _array(filtered, "frame_path_distance_px", [40.0, 50.0, 60.0])
    return root


def test_track_kinematics_logical_loader_prefers_grouped_speed_surface() -> None:
    root = _make_track_kinematics_archive()

    track = load_track_kinematics_track(root, run_name="latest", scope="offline", track_id=0)

    assert track.run_name == "tk_test"
    assert track.track_path == "analysis/track_kinematics_runs/offline/tk_test/tracks/id_0"
    np.testing.assert_array_equal(track.frame_indices, [10, 11, 12])
    np.testing.assert_allclose(track.speed_mm_by_level["raw"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(track.speed_mm_by_level["filtered"], [40.0, 50.0, 60.0])
    np.testing.assert_allclose(track.frame_path_distance_mm_by_level["filtered"], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(track.positions_mm, [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])


def test_detect_bouts_load_speed_levels_uses_track_kinematics_resolver(tmp_path: Path) -> None:
    zarr_path = tmp_path / "track_io.zarr"
    _make_track_kinematics_archive(zarr_path)

    speeds, metadata = _load_track_kinematics_track_speeds(zarr_path, track_kinematics_run="latest", track_id=0)

    assert metadata["track_kinematics_run"] == "tk_test"
    assert metadata["track_kinematics_scope"] == "offline"
    assert metadata["track_kinematics_git_commit"] == "abc"
    assert metadata["source_frame_indices_dtype"] == "int64"
    assert metadata["source_frame_indices_shape"] == [3]
    assert metadata["source_array_paths"]["frame_indices"] == (
        "analysis/track_kinematics_runs/offline/tk_test/tracks/id_0/frame_indices"
    )
    np.testing.assert_array_equal(speeds["frames"], [10, 11, 12])
    np.testing.assert_allclose(speeds["speed_filtered_mm"], [40.0, 50.0, 60.0])
    np.testing.assert_allclose(speeds["frame_path_distance_filtered_mm"], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(metadata["positions_px"], [[10.0, 20.0], [20.0, 20.0], [30.0, 20.0]])
