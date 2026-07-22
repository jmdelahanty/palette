from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

import fisheye.analysis.chaser_escape_freeze_summary as escape_freeze_summary
from fisheye.analysis.chaser_escape_freeze import (
    DEFAULT_BASELINE_WINDOW_S,
    DEFAULT_COMPONENT_NAME,
    DEFAULT_ESCAPE_PATH_THRESHOLD_MM,
    DEFAULT_FREEZE_WINDOW_S,
    DEFAULT_HEADING_MIN_SPEED_MM_S,
    DEFAULT_LOW_SPEED_THRESHOLD_MM_S,
    DEFAULT_RESPONSE_WINDOW_S,
    EscapeFreezeCanaryResult,
    _assert_chaser_trace_moves,
    _classify_escape_attempt_by_path,
    _controller_trial_segments,
    _contiguous_true_segments,
    _metric_dtype,
    _select_trial_trigger,
    _trajectory_dtype,
    _trial_dtype,
    _trigger_radius_from_chaser_states,
    _trigger_radius_from_protocol_json,
    chaser_frame_transform,
    write_escape_freeze_canary_component,
)


def test_chaser_frame_transform_maps_chaser_heading_to_positive_y() -> None:
    fish_xy = np.asarray([[0.0, -10.0], [10.0, -10.0]], dtype=np.float32)
    chaser_xy = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    heading_right_rad = np.asarray([0.0, 0.0], dtype=np.float32)

    transformed, radius, bearing = chaser_frame_transform(
        fish_xy,
        chaser_xy,
        heading_right_rad,
        pixels_per_mm=1.0,
    )

    np.testing.assert_allclose(transformed[:, 0], [-10.0, -10.0], atol=1e-6)
    np.testing.assert_allclose(transformed[:, 1], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(radius, [10.0, 10.0], atol=1e-6)
    np.testing.assert_allclose(bearing, [-90.0, -90.0], atol=1e-6)

    fish_ahead = np.asarray([[10.0, 0.0]], dtype=np.float32)
    chaser = np.asarray([[0.0, 0.0]], dtype=np.float32)
    ahead, ahead_radius, ahead_bearing = chaser_frame_transform(
        fish_ahead,
        chaser,
        np.asarray([0.0], dtype=np.float32),
        pixels_per_mm=1.0,
    )
    np.testing.assert_allclose(ahead, [[0.0, 10.0]], atol=1e-6)
    np.testing.assert_allclose(ahead_radius, [10.0], atol=1e-6)
    np.testing.assert_allclose(ahead_bearing, [0.0], atol=1e-6)


def test_chaser_frame_transform_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="matching shape"):
        chaser_frame_transform(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((3, 2), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            pixels_per_mm=1.0,
        )

    with pytest.raises(ValueError, match="pixels_per_mm"):
        chaser_frame_transform(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            pixels_per_mm=0.0,
        )


def test_contiguous_true_segments() -> None:
    assert _contiguous_true_segments(np.asarray([False, True, True, False, True])) == [(1, 2), (4, 4)]
    assert _contiguous_true_segments(np.asarray([False, False])) == []
    assert _contiguous_true_segments(np.asarray([True, True])) == [(0, 1)]


def test_controller_trial_segments_bridge_alignment_gaps_with_same_logged_id() -> None:
    active = np.asarray([False, True, True, False, True, False, True, True, False])
    trial_id = np.asarray([0, 7, 7, 0, 7, 0, 8, 8, 0])

    segments, source = _controller_trial_segments(active, trial_id)

    assert segments == [(1, 4, 7), (6, 7, 8)]
    assert source == "chase_trial_id"


def test_controller_trial_segments_fall_back_when_logged_ids_are_unavailable() -> None:
    active = np.asarray([False, True, True, False, True])
    trial_id = np.zeros(active.shape, dtype=np.int64)

    segments, source = _controller_trial_segments(active, trial_id)

    assert segments == [(1, 2, 1), (4, 4, 2)]
    assert source == "contiguous_chase_sequence_active_fallback"


def test_select_trial_trigger_labels_already_inside_radius() -> None:
    distance = np.asarray([30.0, 18.0, 17.0, 16.0], dtype=np.float32)

    trigger, proximity, source = _select_trial_trigger(
        distance,
        np.asarray([1, 2, 3], dtype=np.int64),
        trigger_radius_mm=20.0,
    )

    assert trigger == 1
    assert proximity == 1
    assert source == "bout_onset_already_inside_radius"


def test_select_trial_trigger_labels_crossing_and_no_crossing() -> None:
    distance = np.asarray([30.0, 25.0, 19.0, 18.0, 40.0, 39.0], dtype=np.float32)

    trigger, proximity, source = _select_trial_trigger(
        distance,
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        trigger_radius_mm=20.0,
    )
    assert trigger == 2
    assert proximity == 2
    assert source == "proximity"

    trigger, proximity, source = _select_trial_trigger(
        distance,
        np.asarray([4, 5], dtype=np.int64),
        trigger_radius_mm=20.0,
    )
    assert trigger == 4
    assert proximity == -1
    assert source == "bout_onset_no_proximity"


def test_select_trial_trigger_labels_first_valid_inside_radius() -> None:
    distance = np.asarray([np.nan, np.nan, 9.0, 8.0], dtype=np.float32)

    trigger, proximity, source = _select_trial_trigger(
        distance,
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        trigger_radius_mm=20.0,
    )

    assert trigger == 2
    assert proximity == 2
    assert source == "first_valid_already_inside_radius"


def test_trigger_radius_from_chaser_states_uses_active_selected_chaser() -> None:
    dtype = np.dtype(
        [
            ("chaser_index", np.int16),
            ("chase_sequence_active", np.bool_),
            ("initial_distance_mm", np.float32),
        ]
    )
    states = np.asarray(
        [
            (0, False, 99.0),
            (0, True, 20.0),
            (0, True, 20.0),
            (1, True, 30.0),
        ],
        dtype=dtype,
    )

    assert _trigger_radius_from_chaser_states(states, chaser_index=0) == 20.0
    assert _trigger_radius_from_chaser_states(states, chaser_index=1) == 30.0


def test_trigger_radius_from_protocol_json_uses_matching_chaser() -> None:
    protocol = {
        "steps": [
            {
                "parameters": {
                    "chasers": [
                        {"initial_distance_mm": 20.0},
                        {"initial_distance_mm": 30.0},
                    ]
                }
            }
        ]
    }

    assert _trigger_radius_from_protocol_json(protocol, chaser_index=0) == 20.0
    assert _trigger_radius_from_protocol_json(protocol, chaser_index=1) == 30.0


def test_classify_escape_attempt_by_full_trial_path_threshold() -> None:
    assert _classify_escape_attempt_by_path(148.0, threshold_mm=40.0) == (True, "escape_attempt")
    assert _classify_escape_attempt_by_path(7.8, threshold_mm=40.0) == (False, "not_escape")
    assert _classify_escape_attempt_by_path(np.nan, threshold_mm=40.0) == (False, "unclassified")


def test_escape_freeze_writer_distinguishes_trajectory_coordinate_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "escape_freeze.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    run = root.create_group("analysis").create_group("chaser_distance_runs").create_group("chaser_distance_1")
    result = EscapeFreezeCanaryResult(
        zarr_path=str(zarr_path),
        recording_id="recording_1",
        component_name=DEFAULT_COMPONENT_NAME,
        chaser_distance_run_name="chaser_distance_1",
        chaser_distance_run_path="analysis/chaser_distance_runs/chaser_distance_1",
        source_stimulus_run="stimulus_1",
        source_stimulus_path="analysis/stimulus_runs/stimulus_1",
        fps=100.0,
        total_frames=0,
        pixels_per_mm_projector=2.0,
        chaser_index=0,
        response_window_s=DEFAULT_RESPONSE_WINDOW_S,
        freeze_window_s=DEFAULT_FREEZE_WINDOW_S,
        baseline_window_s=DEFAULT_BASELINE_WINDOW_S,
        trigger_radius_mm=20.0,
        trigger_radius_source="test",
        trigger_radius_override=False,
        escape_path_threshold_mm=DEFAULT_ESCAPE_PATH_THRESHOLD_MM,
        low_speed_threshold_mm_s=DEFAULT_LOW_SPEED_THRESHOLD_MM_S,
        heading_min_speed_mm_s=DEFAULT_HEADING_MIN_SPEED_MM_S,
        trial_table=np.zeros(0, dtype=_trial_dtype()),
        metric_table=np.zeros(0, dtype=_metric_dtype()),
        trajectory_table=np.zeros(0, dtype=_trajectory_dtype()),
        summary={},
        diagnostics={},
        warnings=(),
    )
    monkeypatch.setattr(
        escape_freeze_summary,
        "reject_unsealed_chaser_derived_publication",
        lambda *_args, **_kwargs: None,
    )

    component_path = write_escape_freeze_canary_component(zarr_path, result, overwrite=True, write_png=False)

    attrs = dict(root[component_path]["trial_trajectories"].attrs)
    assert run["chaser_escape_freeze"].attrs["latest_complete"] == DEFAULT_COMPONENT_NAME
    assert attrs["coordinate_frame"] == "chaser_centric_mm"
    assert attrs["coordinate_frame_scope"] == "default for fish_x_chaser_frame_mm, fish_y_chaser_frame_mm, and bearing_deg"
    assert attrs["chaser_centric_x_axis_direction"] == "right_relative_to_chaser_heading"
    assert attrs["chaser_centric_bearing_deg_convention"] == "0=chaser_heading_forward; positive=right_relative_to_chaser_heading"
    assert attrs["fish_centered_coordinate_frame"] == "fish_centered_world_mm"
    assert attrs["fish_centered_y_axis_direction"] == "up"
    assert attrs["column_coordinate_frames"]["chaser_x_fish_centered_mm"] == "fish_centered_world_mm"


def test_assert_chaser_trace_moves_passes_for_moving_trace() -> None:
    chaser_xy = np.column_stack([np.arange(10.0), np.arange(10.0) * 2.0])
    chaser_valid = np.ones(10, dtype=bool)
    _assert_chaser_trace_moves(chaser_xy, chaser_valid, "chaser_distance_1")


def test_assert_chaser_trace_moves_passes_when_only_one_axis_moves() -> None:
    chaser_xy = np.column_stack([np.arange(10.0), np.full(10, 5.0)])
    chaser_valid = np.ones(10, dtype=bool)
    _assert_chaser_trace_moves(chaser_xy, chaser_valid, "chaser_distance_1")


def test_assert_chaser_trace_moves_raises_for_constant_trace() -> None:
    chaser_xy = np.full((10, 2), 7.0)
    chaser_valid = np.ones(10, dtype=bool)
    with pytest.raises(ValueError, match="is constant"):
        _assert_chaser_trace_moves(chaser_xy, chaser_valid, "chaser_distance_1")


def test_assert_chaser_trace_moves_ignores_movement_in_invalid_samples() -> None:
    chaser_xy = np.column_stack([np.arange(10.0), np.arange(10.0)])
    chaser_valid = np.ones(10, dtype=bool)
    chaser_valid[1:] = False  # only one valid sample; the rest "move" but are invalid
    chaser_xy[0] = [3.0, 3.0]
    _assert_chaser_trace_moves(chaser_xy, chaser_valid, "chaser_distance_1")


def test_assert_chaser_trace_moves_raises_when_valid_samples_are_stuck() -> None:
    chaser_xy = np.arange(20.0).reshape(10, 2)  # every row distinct...
    chaser_valid = np.zeros(10, dtype=bool)
    chaser_valid[[2, 5, 8]] = True
    chaser_xy[[2, 5, 8]] = [4.0, 9.0]  # ...but the valid ones are identical
    with pytest.raises(ValueError, match="is constant"):
        _assert_chaser_trace_moves(chaser_xy, chaser_valid, "chaser_distance_1")
