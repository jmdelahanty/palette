from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.analysis.chaser_state_interpolator import write_columnar_dataset
from fisheye.analysis.megabouts_classifier_inputs import (
    build_megabouts_classifier_input_pack,
    summarize_input_pack,
)


def _build_root() -> zarr.Group:
    root = zarr.group()
    analysis = root.create_group("analysis")

    posture_parent = analysis.create_group("tail_posture_view_runs")
    posture_parent.attrs["latest"] = "posture_001"
    posture_parent.attrs["latest_megabouts_compatible"] = "posture_001"
    posture = posture_parent.create_group("posture_001")
    frames = np.arange(8, dtype=np.int64)
    posture.create_array("frame_index", data=frames, overwrite=True)
    posture.create_array(
        "valid",
        data=np.asarray([True, True, False, True, True, True, True, True], dtype=bool),
        overwrite=True,
    )
    tail_angle = np.arange(8 * 10, dtype=np.float32).reshape(8, 10) / 100.0
    posture.create_array("tail_angle_rad", data=tail_angle, overwrite=True)

    track_parent = analysis.create_group("track_kinematics_runs")
    offline = track_parent.create_group("offline")
    offline.attrs["latest"] = "tk_001"
    track_run = offline.create_group("tk_001")
    track_run.attrs["fps"] = 10.0
    track = track_run.create_group("tracks").create_group("id_0")
    track.create_array("frame_indices", data=frames, overwrite=True)
    track.create_array(
        "positions_mm",
        data=np.stack([frames.astype(np.float32), frames.astype(np.float32) + 10.0], axis=1),
        overwrite=True,
    )
    track.create_array("smoothed_heading_radians", data=np.linspace(0.0, 0.7, 8, dtype=np.float32), overwrite=True)
    track.create_array(
        "sample_valid",
        data=np.asarray([True, True, True, True, True, False, True, True], dtype=bool),
        overwrite=True,
    )

    swim_parent = analysis.create_group("swim_bout_runs")
    swim_parent.attrs["latest"] = "bouts_001"
    bout_run = swim_parent.create_group("bouts_001")
    bout_run.attrs["default_level"] = "speed_filtered"
    level = bout_run.create_group("speed_filtered")
    level.attrs["fps"] = 10.0
    bouts = np.asarray(
        [(11, 1, 3), (12, 4, 6)],
        dtype=[("bout_id", "i4"), ("start_frame", "i8"), ("end_frame", "i8")],
    )
    write_columnar_dataset(level, "bouts", bouts)
    return root


def test_build_megabouts_classifier_input_pack_resolves_sources_and_shapes() -> None:
    root = _build_root()

    pack = build_megabouts_classifier_input_pack(
        root,
        bout_duration_frames=4,
        min_tail_valid_fraction=0.75,
        min_traj_valid_fraction=0.75,
        max_consecutive_invalid_frames=1,
    )

    assert pack.tail_array.shape == (2, 10, 4)
    assert pack.traj_array.shape == (2, 3, 4)
    assert pack.source_bout_id.tolist() == [11, 12]
    assert pack.window_start_frame.tolist() == [1, 4]
    assert pack.window_end_frame.tolist() == [4, 7]
    np.testing.assert_allclose(pack.tail_array[0, :, 0], np.arange(10, 20, dtype=np.float32) / 100.0)
    assert pack.tail_valid.tolist() == [
        [True, False, True, True],
        [True, True, True, True],
    ]
    assert pack.traj_valid.tolist() == [
        [True, True, True, True],
        [True, False, True, True],
    ]
    np.testing.assert_allclose(pack.tail_valid_fraction, [0.75, 1.0])
    np.testing.assert_allclose(pack.traj_valid_fraction, [1.0, 0.75])
    assert pack.valid_bout.tolist() == [True, True]
    assert pack.failure_reason.tolist() == ["ok", "ok"]
    assert pack.parameters["window_policy"] == "start_frame_fixed_duration"
    assert pack.parameters["calls_megabouts"] is False
    assert pack.source_refs["tail_angle_rad"].endswith("/tail_angle_rad")


def test_build_megabouts_classifier_input_pack_reports_invalid_coverage() -> None:
    root = _build_root()

    pack = build_megabouts_classifier_input_pack(root, bout_duration_frames=4)
    summary = summarize_input_pack(pack)

    assert pack.valid_bout.tolist() == [False, False]
    assert summary["valid_bout_count"] == 0
    assert summary["invalid_bout_count"] == 2
    assert summary["failure_reason_counts"] == {
        "tail_valid_fraction_below_threshold": 1,
        "traj_valid_fraction_below_threshold": 1,
    }


def test_build_megabouts_classifier_input_pack_can_resolve_time_only_bouts() -> None:
    root = _build_root()
    level = root["analysis/swim_bout_runs/bouts_001/speed_filtered"]
    bouts = np.asarray(
        [(21, 0.1, 0.3)],
        dtype=[("bout_id", "i4"), ("start_time_s", "f8"), ("end_time_s", "f8")],
    )
    write_columnar_dataset(level, "bouts", bouts)

    pack = build_megabouts_classifier_input_pack(
        root,
        bout_duration_s=0.2,
        min_tail_valid_fraction=0.5,
        min_traj_valid_fraction=0.5,
    )

    assert pack.tail_array.shape == (1, 10, 2)
    assert pack.source_start_frame.tolist() == [1]
    assert pack.source_end_frame.tolist() == [3]
    assert pack.window_end_frame.tolist() == [2]


def test_build_megabouts_classifier_input_pack_rejects_wrong_tail_segment_count() -> None:
    root = _build_root()
    posture = root["analysis/tail_posture_view_runs/posture_001"]
    del posture["tail_angle_rad"]
    posture.create_array("tail_angle_rad", data=np.zeros((8, 9), dtype=np.float32), overwrite=True)

    with pytest.raises(ValueError, match="10 tail-angle channels"):
        build_megabouts_classifier_input_pack(root, bout_duration_frames=4)


def test_build_megabouts_classifier_input_pack_rejects_windows_too_long_for_megabouts() -> None:
    root = _build_root()

    with pytest.raises(ValueError, match="capped at 140 frames"):
        build_megabouts_classifier_input_pack(root, bout_duration_frames=141)
