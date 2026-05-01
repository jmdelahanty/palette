from __future__ import annotations

import io

import numpy as np
import pytest
import zarr
from rich.console import Console

from fisheye.analysis import track_kinematics as mod
from fisheye.analysis.compute_speed import (
    TRANSITION_REASON_FIRST_SAMPLE,
    TRANSITION_REASON_FRAME_GAP,
    TRANSITION_REASON_OK,
    TRANSITION_REASON_TELEPORT,
    compute_track_speed,
)


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

    def __contains__(self, key: str) -> bool:
        return key in self._children


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
        tracks[0]["angular_velocity_raw_deg_s"],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        tracks[0]["angular_speed_raw_deg_s"],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        tracks[0]["angular_velocity_smoothed_deg_s"],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        tracks[0]["angular_speed_smoothed_deg_s"],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        tracks[1]["delta_heading_degrees"],
        np.array([np.nan, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        tracks[1]["angular_velocity_deg_s"],
        np.array([np.nan, np.nan], dtype=np.float32),
        equal_nan=True,
    )


def test_load_keypoint_usability_array_prefers_heading_usable() -> None:
    group = _FakeGroup()
    group.create_array("refined_success", data=np.array([False, True], dtype=bool))
    group.create_array("heading_usable", data=np.array([True, False], dtype=bool))

    values, dataset_name = mod.load_keypoint_usability_array(group, expected_length=2)

    assert dataset_name == "heading_usable"
    assert values.tolist() == [True, False]


def test_build_track_datasets_materializes_sample_validity() -> None:
    tracks, _summaries = mod.build_track_datasets(
        track_ids=np.array([0, 0, 0, 0], dtype=np.int64),
        frames=np.array([0, 1, 2, 3], dtype=np.int64),
        positions_px=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float32,
        ),
        headings_deg=np.array([0.0, 0.0, 0.0, np.nan], dtype=np.float32),
        keypoint_success=np.array([True, True, False, True], dtype=bool),
        detection_source=np.array([0, 1, 0, 0], dtype=np.int8),
        fps=1.0,
        smooth_seconds=1.0,
        pixel_to_mm=None,
    )

    track = tracks[0]
    assert track["sample_observed"].tolist() == [True, True, True, True]
    assert track["source_observed"].tolist() == [True, False, True, True]
    assert track["keypoint_usable"].tolist() == [True, True, False, False]
    assert track["position_finite"].tolist() == [True, True, True, True]
    assert track["heading_usable"].tolist() == [True, True, False, False]
    assert track["sample_valid"].tolist() == [True, False, False, False]
    assert track["sample_reason_code"].tolist() == [
        mod.SAMPLE_REASON_OK,
        mod.SAMPLE_REASON_SOURCE_INTERPOLATED,
        mod.SAMPLE_REASON_KEYPOINT_FAILED,
        mod.SAMPLE_REASON_HEADING_UNUSABLE,
    ]
    np.testing.assert_allclose(
        track["angular_velocity_raw_deg_s"],
        np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32),
        equal_nan=True,
    )


def test_build_track_datasets_materializes_transition_validity() -> None:
    tracks, _summaries = mod.build_track_datasets(
        track_ids=np.array([0, 0, 0, 0], dtype=np.int64),
        frames=np.array([0, 1, 3, 4], dtype=np.int64),
        positions_px=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [3.0, 0.0],
                [1000.0, 0.0],
            ],
            dtype=np.float32,
        ),
        headings_deg=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        keypoint_success=np.array([True, True, True, True], dtype=bool),
        detection_source=None,
        fps=1.0,
        smooth_seconds=1.0,
        pixel_to_mm=None,
    )

    track = tracks[0]
    assert track["delta_frames"].tolist() == [0, 1, 2, 1]
    np.testing.assert_allclose(
        track["delta_seconds"],
        np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float32),
    )
    assert track["transition_valid"].tolist() == [False, True, False, False]
    assert track["transition_reason_code"].tolist() == [
        TRANSITION_REASON_FIRST_SAMPLE,
        TRANSITION_REASON_OK,
        TRANSITION_REASON_FRAME_GAP,
        TRANSITION_REASON_TELEPORT,
    ]
    np.testing.assert_allclose(
        track["frame_path_distance_raw_px"],
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        track["angular_velocity_raw_deg_s"],
        np.array([np.nan, 0.0, np.nan, np.nan], dtype=np.float32),
        equal_nan=True,
    )


def test_causal_smoothing_does_not_leak_future_motion_into_onset() -> None:
    frames = np.arange(7, dtype=np.int64)
    positions = np.column_stack(
        [
            np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0], dtype=np.float32),
            np.zeros(7, dtype=np.float32),
        ]
    )

    centered = compute_track_speed(
        frames,
        positions,
        fps=1.0,
        smooth_seconds=3.0,
        distance_smooth_seconds=3.0,
        smoothing_method="moving_average",
        smoothing_alignment="centered",
    )
    causal = compute_track_speed(
        frames,
        positions,
        fps=1.0,
        smooth_seconds=3.0,
        distance_smooth_seconds=3.0,
        smoothing_method="moving_average",
        smoothing_alignment="causal",
    )

    assert centered.speed_smoothed[3] > 0.0
    assert causal.speed_smoothed[3] == 0.0
    assert causal.speed_smoothed[4] > 0.0


def test_causal_smoothing_rejects_savitzky_golay_for_now() -> None:
    with pytest.raises(ValueError, match="causal smoothing"):
        compute_track_speed(
            np.arange(4, dtype=np.int64),
            np.column_stack([np.arange(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]),
            fps=1.0,
            smooth_seconds=3.0,
            smoothing_method="savitzky_golay",
            smoothing_alignment="causal",
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
    np.testing.assert_allclose(
        subgroup["angular_velocity_raw_deg_s"][:],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        subgroup["angular_speed_raw_deg_s"][:],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        subgroup["delta_heading_smoothed_degrees"][:],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        subgroup["angular_velocity_smoothed_deg_s"][:],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        subgroup["angular_speed_smoothed_deg_s"][:],
        np.array([np.nan, 20.0], dtype=np.float32),
        equal_nan=True,
        atol=1e-4,
    )
    assert subgroup["delta_frames"][:].tolist() == [0, 1]
    np.testing.assert_allclose(
        subgroup["delta_seconds"][:],
        np.array([0.0, 1.0], dtype=np.float32),
    )
    assert subgroup["transition_valid"][:].tolist() == [False, True]
    assert subgroup["transition_reason_code"][:].tolist() == [
        TRANSITION_REASON_FIRST_SAMPLE,
        TRANSITION_REASON_OK,
    ]
    assert (
        subgroup.attrs["transition_validity_schema_id"]
        == "palette.track_transition_validity.v1"
    )
    assert (
        subgroup.attrs["sample_validity_schema_id"]
        == "palette.track_sample_validity.v1"
    )
    assert subgroup["sample_observed"][:].tolist() == [True, True]
    assert subgroup["sample_valid"][:].tolist() == [True, True]
    assert subgroup["source_observed"][:].tolist() == [True, True]
    assert subgroup["keypoint_usable"][:].tolist() == [True, True]
    assert subgroup["position_finite"][:].tolist() == [True, True]
    assert subgroup["heading_usable"][:].tolist() == [True, True]
    assert subgroup["sample_reason_code"][:].tolist() == [
        mod.SAMPLE_REASON_OK,
        mod.SAMPLE_REASON_OK,
    ]
    assert (
        subgroup.attrs["sample_reason_codes"][str(mod.SAMPLE_REASON_SOURCE_INTERPOLATED)]
        == "source_interpolated"
    )
    assert (
        subgroup.attrs["transition_reason_codes"][str(TRANSITION_REASON_FRAME_GAP)]
        == "frame_gap"
    )


def test_save_track_kinematics_tracks_persists_speed_derivative_hierarchy() -> None:
    track_ids = np.array([0, 0, 0], dtype=np.int64)
    frames = np.array([0, 1, 2], dtype=np.int64)
    positions_px = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    headings_deg = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    keypoint_success = np.array([True, True, True], dtype=bool)

    tracks, summaries = mod.build_track_datasets(
        track_ids=track_ids,
        frames=frames,
        positions_px=positions_px,
        headings_deg=headings_deg,
        keypoint_success=keypoint_success,
        detection_source=None,
        fps=1.0,
        smooth_seconds=1.0,
        pixel_to_mm=2.0,
    )

    run_group = _FakeGroup()
    mod.save_track_kinematics_tracks(run_group, tracks, summaries)

    subgroup = run_group["tracks"]["id_0"]
    derivatives = subgroup["speed_derivatives"]
    movement_speed = subgroup["movement"]["speed"]
    assert derivatives.attrs["schema_id"] == "palette.track_speed_derivatives.v1"
    assert derivatives.attrs["default_source_speed_level"] == "speed_smoothed"
    assert subgroup["movement"].attrs["schema_id"] == "palette.track_movement.v2"
    assert movement_speed.attrs["schema_id"] == "palette.track_movement_speed.v2"

    expected_accel_px = np.array([np.nan, np.nan, 1.0], dtype=np.float32)
    expected_accel_mm = np.array([np.nan, np.nan, 2.0], dtype=np.float32)
    for level, movement_level in (
        ("speed_raw", "raw"),
        ("speed_filtered", "filtered"),
        ("speed_smoothed", "smoothed"),
        ("speed_averaged", "averaged"),
    ):
        level_group = derivatives[level]
        movement_group = movement_speed[movement_level]
        assert level_group.attrs["schema_id"] == "palette.track_speed_derivative.v1"
        assert level_group.attrs["source_speed_level"] == level
        assert level_group.attrs["source_speed_px_array"] == f"../../{level}_px"
        assert level_group.attrs["time_delta_array"] == "../../delta_seconds"
        assert movement_group.attrs["schema_id"] == "palette.track_movement_speed_level.v2"
        assert movement_group.attrs["source_speed_level"] == level
        np.testing.assert_allclose(
            level_group["acceleration_px"][:],
            expected_accel_px,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            level_group["acceleration_mm"][:],
            expected_accel_mm,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            movement_group["acceleration_px"][:],
            level_group["acceleration_px"][:],
            equal_nan=True,
        )

    np.testing.assert_allclose(
        subgroup["acceleration_px"][:],
        derivatives["speed_smoothed"]["acceleration_px"][:],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        subgroup["smoothed_acceleration_mm"][:],
        derivatives["speed_smoothed"]["smoothed_acceleration_mm"][:],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        movement_speed["smoothed"]["mm"][:],
        subgroup["speed_smoothed_mm"][:],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        movement_speed["filtered"]["frame_path_distance_mm"][:],
        subgroup["frame_path_distance_filtered_mm"][:],
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
