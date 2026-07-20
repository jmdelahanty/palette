from __future__ import annotations

import io

import numpy as np
import pytest
import zarr
from rich.console import Console

from fisheye.analysis import track_kinematics as mod
from fisheye.shared.zarr.columnar import write_columnar_dataset
from fisheye.analysis.compute_speed import (
    TRANSITION_REASON_FIRST_SAMPLE,
    TRANSITION_REASON_FRAME_GAP,
    TRANSITION_REASON_OK,
    TRANSITION_REASON_TELEPORT,
    compute_track_speed,
)
from tests.unit.fisheye.test_directed_transform_chain import _world
from tests.unit.fisheye.test_track_coordinate_publication import (
    _source,
)
from tests.unit.fisheye.test_track_kinematics_coordinate_contract import (
    _WritableGroup,
    _selected_stimulus_physical_authority,
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


def _make_memory_group() -> zarr.Group:
    return zarr.open_group("memory://", mode="w")


def test_track_kinematics_contract_attrs_are_machine_readable() -> None:
    attrs = mod._track_kinematics_contract_attrs(
        run_type="online",
        method="track_kinematics_online_refined",
        parameters={"fps": 30.0},
        inputs={
            "refined_online_run": "refined_a",
            "stimulus_run": "stim_a",
            "chaser_index": 1,
        },
    )

    assert attrs["schema_id"] == "analysis.track_kinematics_runs"
    assert attrs["schema_version"] == 1
    assert attrs["method_version"] == "track_kinematics.v1"
    assert attrs["row_axis"] == "track_samples"
    assert attrs["parameters"] == {"fps": 30.0}
    assert attrs["source_refs"] == {
        "source_refined_online_path": "refined_online_runs/refined_a",
        "source_stimulus_path": "analysis/stimulus_runs/stim_a",
        "source_chaser_index": 1,
    }


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


def test_hysteresis_band_policy_reset_preserves_legacy_debounce() -> None:
    # Displacements: high, low, in-band, low, low, low.
    # Legacy reset treats the in-band sample as enough motion evidence to
    # restart exit debounce, so the final low frame is the first zeroed sample.
    displacements = np.array([5.0, 1.0, 3.0, 1.0, 1.0, 1.0], dtype=np.float32)
    positions = np.column_stack(
        [np.concatenate([[0.0], np.cumsum(displacements)]), np.zeros(displacements.size + 1)]
    )

    speeds = compute_track_speed(
        np.arange(positions.shape[0], dtype=np.int64),
        positions,
        fps=1.0,
        smooth_seconds=1.0,
        hysteresis_high_px=4.0,
        hysteresis_low_px=2.0,
        hysteresis_min_frames=3,
        hysteresis_band_policy="reset",
    )

    np.testing.assert_allclose(
        speeds.frame_path_distance_filtered,
        np.array([0.0, 5.0, 1.0, 3.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def test_hysteresis_band_policy_latch_is_schmitt_style_dead_band() -> None:
    # Schmitt-style latch leaves low_count unchanged in the low/high dead band,
    # so the exit debounce fires earlier than legacy reset on the same signal.
    displacements = np.array([5.0, 1.0, 3.0, 1.0, 1.0, 1.0], dtype=np.float32)
    positions = np.column_stack(
        [np.concatenate([[0.0], np.cumsum(displacements)]), np.zeros(displacements.size + 1)]
    )

    speeds = compute_track_speed(
        np.arange(positions.shape[0], dtype=np.int64),
        positions,
        fps=1.0,
        smooth_seconds=1.0,
        hysteresis_high_px=4.0,
        hysteresis_low_px=2.0,
        hysteresis_min_frames=3,
        hysteresis_band_policy="latch",
    )

    np.testing.assert_allclose(
        speeds.frame_path_distance_filtered,
        np.array([0.0, 5.0, 1.0, 3.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )


def test_hysteresis_band_policy_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="hysteresis band policy"):
        compute_track_speed(
            np.arange(3, dtype=np.int64),
            np.column_stack([np.arange(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]),
            fps=1.0,
            smooth_seconds=1.0,
            hysteresis_high_px=4.0,
            hysteresis_low_px=2.0,
            hysteresis_band_policy="unknown",
        )


def test_save_track_kinematics_tracks_persists_turning_arrays() -> None:
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, temporal = _source(world)
    track_ids = np.array([7, 7], dtype=np.int64)
    frames = np.array([0, 1], dtype=np.int64)
    positions_px = np.asarray(source.coordinate_node[:])
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
        source_row_index=np.asarray([0, 1], dtype=np.int64),
        source_temporal_authority=temporal,
    )

    run_group = _WritableGroup(
        path="analysis/track_kinematics_runs/offline/turning",
        archive_token=world["archive_token"],
    )
    ordered_ids = mod.save_track_kinematics_tracks(
        run_group,
        tracks,
        summaries,
        source_temporal_authority=temporal,
        positions_px_source=source,
    )

    assert ordered_ids == [7]
    subgroup = run_group["tracks"]["id_7"]
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
    world = _world(convention="pixel_center", archive_token=object())
    _, _, source, temporal = _source(world)
    physical_authority = _selected_stimulus_physical_authority(world)
    track_ids = np.array([7, 7], dtype=np.int64)
    frames = np.array([0, 1], dtype=np.int64)
    positions_px = np.asarray(source.coordinate_node[:])
    headings_deg = np.array([0.0, 0.0], dtype=np.float32)
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
        pixel_to_mm=physical_authority.mm_per_pixel,
        source_row_index=np.asarray([0, 1], dtype=np.int64),
        source_temporal_authority=temporal,
    )

    run_group = _WritableGroup(
        path="analysis/track_kinematics_runs/offline/derivatives",
        archive_token=world["archive_token"],
    )
    mod.save_track_kinematics_tracks(
        run_group,
        tracks,
        summaries,
        source_temporal_authority=temporal,
        positions_px_source=source,
        physical_authority=physical_authority,
    )

    subgroup = run_group["tracks"]["id_7"]
    derivatives = subgroup["speed_derivatives"]
    movement_speed = subgroup["movement"]["speed"]
    assert derivatives.attrs["schema_id"] == "palette.track_speed_derivatives.v1"
    assert derivatives.attrs["default_source_speed_level"] == "speed_smoothed"
    assert subgroup["movement"].attrs["schema_id"] == "palette.track_movement.v2"
    assert movement_speed.attrs["schema_id"] == "palette.track_movement_speed.v2"

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
            tracks[7]["speed_derivatives"][level]["acceleration_px"],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            level_group["acceleration_mm"][:],
            tracks[7]["speed_derivatives"][level]["acceleration_mm"],
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


def test_speed_derivative_mm_arrays_are_exact_scaled_float32_pairs() -> None:
    scale = 0.018846914829982055
    derivative = mod._compute_speed_derivative(  # noqa: SLF001
        np.asarray([0.0, 1.234567, 5.678901, 2.345678], dtype=np.float64),
        np.asarray([0.0, 1.0 / 30.0, 1.0 / 30.0, 1.0 / 30.0]),
        pixel_to_mm=scale,
        smooth_seconds=0.05,
        fps=30.0,
    )

    for pixel_name, physical_name in (
        ("acceleration_px", "acceleration_mm"),
        ("smoothed_acceleration_px", "smoothed_acceleration_mm"),
    ):
        pixel = np.asarray(derivative[pixel_name])
        physical = np.asarray(derivative[physical_name])
        assert pixel.dtype == np.dtype("<f4")
        assert physical.dtype == pixel.dtype
        assert np.array_equal(
            physical,
            pixel * np.asarray(scale, dtype=pixel.dtype),
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


def _add_compact_swim_bout_run(root: zarr.Group, *, track_id: int = 1, run_name: str = "compact_bout") -> None:
    swim_parent = root.require_group("analysis").require_group("swim_bout_runs")
    bout_run = swim_parent.create_group(run_name)
    swim_parent.attrs["latest"] = run_name
    bout_run.attrs.update(
        {
            "schema_id": "palette.swim_bout_runs",
            "schema_version": 7,
            "layout": "compact_tabular_v2",
            "source_track_kinematics_run": "kin_run",
            "track_id": track_id,
            "default_candidate_id": 0,
            "default_signal_id": 1,
        }
    )
    indexes = bout_run.create_group("indexes")
    tables = bout_run.create_group("tables")
    candidates = np.zeros(
        1,
        dtype=[
            ("candidate_id", "i4"),
            ("candidate_name", "S32"),
            ("is_default", "?"),
            ("detection_method", "S32"),
            ("parameters_json", "S64"),
        ],
    )
    candidates[0] = (0, b"candidate", True, b"peak_event", b"{}")
    write_columnar_dataset(indexes, "candidates", candidates)
    signals = np.zeros(
        2,
        dtype=[
            ("signal_id", "i4"),
            ("speed_level", "S32"),
            ("signal_name", "S32"),
            ("role", "S32"),
            ("source_level", "S32"),
        ],
    )
    signals[0] = (0, b"speed_filtered", b"filtered", b"physical_estimator", b"speed_filtered")
    signals[1] = (1, b"speed_exponential", b"exponential", b"detector_response", b"speed_filtered")
    write_columnar_dataset(indexes, "signal_variants", signals)
    bouts = np.zeros(
        2,
        dtype=[
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("track_id", "i4"),
            ("bout_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("duration_s", "f8"),
        ],
    )
    bouts[0] = (0, 0, track_id, 10, 5, 9, 0.1)
    bouts[1] = (0, 1, track_id, 20, 10, 20, 0.2)
    write_columnar_dataset(tables, "bouts", bouts)


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


def test_swim_bout_mirror_reads_compact_v2_logical_signals() -> None:
    root = zarr.group()
    run_group = _make_run_group_with_tracks(root, (0, 1))
    _add_compact_swim_bout_run(root, track_id=1)

    result = mod._mirror_swim_bouts_to_tracks(
        root,
        run_group,
        [0, 1],
        "compact_bout",
        _quiet_console(),
        expected_track_kinematics_run="kin_run",
    )

    assert result == "compact_bout"
    tracks = run_group["tracks"]
    assert "swim_bouts" not in tracks["id_0"]
    mirrored = tracks["id_1"]["swim_bouts"]
    assert mirrored.attrs["layout"] == "compact_tabular_v2"
    assert mirrored.attrs["source_swim_bout_candidate_id"] == 0
    assert mirrored.attrs["source_swim_bout_default_signal_id"] == 1
    assert set(mirrored.group_keys()) == {"speed_filtered", "speed_exponential"}
    np.testing.assert_array_equal(
        mirrored["speed_exponential"]["start_frame"][:],
        np.array([10], dtype=np.int32),
    )
    assert mirrored["speed_exponential"].attrs["signal_id"] == 1
