from __future__ import annotations

import numpy as np

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
