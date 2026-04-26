from __future__ import annotations

import io

import numpy as np
import zarr
from rich.console import Console

from fisheye.analysis import track_kinematics as mod


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)

    def __getitem__(self, key):
        return self._data[key]


class _FakeAttrs(dict):
    pass


class _FakeGroup:
    def __init__(self) -> None:
        self.attrs = _FakeAttrs()
        self._children: dict[str, _FakeGroup | _FakeArray] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        group = _FakeGroup()
        self._children[name] = group
        return group

    def create_array(self, name: str, data, chunks=None, overwrite: bool = False):
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        array = _FakeArray(np.asarray(data))
        self._children[name] = array
        return array

    def __getitem__(self, key: str):
        return self._children[key]


def test_build_track_datasets_computes_turning_series_for_all_tracks() -> None:
    track_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    frames = np.array([0, 1, 0, 2], dtype=np.int64)
    positions_px = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [10.0, 0.0],
            [12.0, 0.0],
        ],
        dtype=np.float32,
    )
    headings_deg = np.array([350.0, 10.0, 10.0, 350.0], dtype=np.float32)
    keypoint_success = np.array([True, True, True, True], dtype=bool)

    tracks, summaries = mod.build_track_datasets(
        track_ids=track_ids,
        frames=frames,
        positions_px=positions_px,
        headings_deg=headings_deg,
        keypoint_success=keypoint_success,
        detection_source=None,
        fps=1.0,
        smooth_seconds=1.0,
        pixel_to_mm=None,
    )

    assert set(tracks) == {0, 1}
    assert len(summaries) == 2

    np.testing.assert_allclose(
        tracks[0]["delta_heading_degrees"],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        tracks[0]["angular_velocity_deg_s"],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
    )

    np.testing.assert_allclose(
        tracks[1]["delta_heading_degrees"],
        np.array([np.nan, -20.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        tracks[1]["angular_velocity_deg_s"],
        np.array([np.nan, -10.0], dtype=np.float32),
        equal_nan=True,
    )


def test_save_track_kinematics_tracks_persists_turning_arrays() -> None:
    track_ids = np.array([0, 0], dtype=np.int64)
    frames = np.array([0, 1], dtype=np.int64)
    positions_px = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    headings_deg = np.array([350.0, 10.0], dtype=np.float32)
    keypoint_success = np.array([True, True], dtype=bool)

    tracks, summaries = mod.build_track_datasets(
        track_ids=track_ids,
        frames=frames,
        positions_px=positions_px,
        headings_deg=headings_deg,
        keypoint_success=keypoint_success,
        detection_source=None,
        fps=1.0,
        smooth_seconds=1.0,
        pixel_to_mm=None,
    )

    run_group = _FakeGroup()
    ordered_ids = mod.save_track_kinematics_tracks(run_group, tracks, summaries)

    assert ordered_ids == [0]
    subgroup = run_group["tracks"]["id_0"]
    np.testing.assert_allclose(
        subgroup["delta_heading_degrees"][:],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        subgroup["angular_velocity_deg_s"][:],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
    )


def _quiet_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _make_run_group_with_tracks(root: zarr.Group, track_ids: tuple[int, ...]) -> zarr.Group:
    run_group = root.create_group("track_kinematics_run")
    tracks = run_group.create_group("tracks")
    for track_id in track_ids:
        tracks.create_group(f"id_{track_id}")
    return run_group


def _add_hierarchical_swim_bout_run(
    root: zarr.Group,
    *,
    run_name: str = "bout_run",
    track_id: int | None = 1,
    source_track_kinematics_run: str = "kin_run",
) -> None:
    swim_parent = root.require_group("analysis").require_group("swim_bout_runs")
    bout_run = swim_parent.create_group(run_name)
    swim_parent.attrs["latest"] = run_name
    bout_run.attrs["source_track_kinematics_run"] = source_track_kinematics_run
    if track_id is not None:
        bout_run.attrs["track_id"] = track_id
    bout_run.attrs["default_level"] = "speed_smoothed"
    dtype = np.dtype([
        ("start_frame", "<i4"),
        ("end_frame", "<i4"),
        ("duration_s", "<f4"),
    ])
    bouts = np.array([(10, 20, 1.0)], dtype=dtype)
    for level in ("speed_raw", "speed_filtered", "speed_smoothed", "speed_averaged"):
        level_group = bout_run.create_group(level)
        level_group.create_array("bouts", data=bouts)


def _add_flat_swim_bout_run_without_track_id(root: zarr.Group, *, run_name: str = "legacy_bout") -> None:
    swim_parent = root.require_group("analysis").require_group("swim_bout_runs")
    bout_run = swim_parent.create_group(run_name)
    swim_parent.attrs["latest"] = run_name
    dtype = np.dtype([
        ("start_frame", "<i4"),
        ("end_frame", "<i4"),
        ("duration_s", "<f4"),
    ])
    bout_run.create_array("bouts", data=np.array([(10, 20, 1.0)], dtype=dtype))


def test_swim_bout_mirror_only_writes_matching_track_id() -> None:
    root = zarr.group()
    run_group = _make_run_group_with_tracks(root, (0, 1))
    _add_hierarchical_swim_bout_run(root, track_id=1, source_track_kinematics_run="kin_run")

    result = mod._mirror_swim_bouts_to_tracks(
        root,
        run_group,
        [0, 1],
        "bout_run",
        _quiet_console(),
        expected_track_kinematics_run="kin_run",
    )

    assert result == "bout_run"
    tracks = run_group["tracks"]
    assert "swim_bouts" not in tracks["id_0"]
    mirrored = tracks["id_1"]["swim_bouts"]
    assert mirrored.attrs["source_swim_bout_track_id"] == 1
    assert mirrored.attrs["mirror_scope"] == "matched_track_id"
    np.testing.assert_array_equal(
        mirrored["speed_smoothed"]["start_frame"][:],
        np.array([10], dtype=np.int32),
    )


def test_swim_bout_mirror_skips_unscoped_legacy_run_with_multiple_tracks() -> None:
    root = zarr.group()
    run_group = _make_run_group_with_tracks(root, (0, 1))
    _add_flat_swim_bout_run_without_track_id(root)

    result = mod._mirror_swim_bouts_to_tracks(
        root,
        run_group,
        [0, 1],
        "legacy_bout",
        _quiet_console(),
        expected_track_kinematics_run="kin_run",
    )

    assert result is None
    tracks = run_group["tracks"]
    assert "swim_bouts" not in tracks["id_0"]
    assert "swim_bouts" not in tracks["id_1"]
