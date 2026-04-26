from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.detect_bouts_multi_level import (
    _detect_bouts_from_speed,
    detect_and_save_bouts,
    normalize_speed_level,
)


def _write_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    group.create_array(name, data=data, chunks=data.shape, overwrite=True)


def _make_track_kinematics_archive(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    analysis = root.create_group("analysis")
    parent = analysis.create_group("track_kinematics_runs")
    offline = parent.create_group("offline")
    offline.attrs["latest"] = "tk_1"

    run = offline.create_group("tk_1")
    run.attrs["fps"] = 100.0
    run.attrs["pixel_to_mm"] = 0.1

    track = run.create_group("tracks").create_group("id_0")
    frames = np.arange(12, dtype=np.int64)
    speed = np.asarray([0, 0, 3, 4, 5, 5, 4, 3, 0, 0, 0, 0], dtype=np.float32)
    positions_px = np.column_stack(
        [
            np.linspace(10.0, 22.0, frames.size, dtype=np.float32),
            np.linspace(30.0, 42.0, frames.size, dtype=np.float32),
        ]
    )

    _write_array(track, "frame_indices", frames)
    _write_array(track, "speed_raw_mm", speed)
    _write_array(track, "speed_filtered_mm", speed)
    _write_array(track, "speed_smoothed_mm", speed)
    _write_array(track, "speed_averaged_mm", speed)
    _write_array(track, "positions_px", positions_px)
    _write_array(track, "positions_mm", positions_px * 0.1)
    return zarr_path


def test_normalize_speed_level_accepts_aliases() -> None:
    assert normalize_speed_level("filtered") == "speed_filtered"
    assert normalize_speed_level("speed_filtered") == "speed_filtered"


def test_normalize_speed_level_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported speed level"):
        normalize_speed_level("median")


def test_detect_and_save_bouts_records_filtered_default_level(tmp_path: Path) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)

    run_name = detect_and_save_bouts(
        zarr_path=zarr_path,
        run_name="bouts_filtered_default",
        track_kinematics_run="tk_1",
        track_id=0,
        threshold_mm=2.0,
        min_bout_duration_s=0.01,
        min_gap_duration_s=0.01,
        default_level="filtered",
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    bout_run = root["analysis"]["swim_bout_runs"][run_name]

    assert bout_run.attrs["default_level"] == "speed_filtered"
    assert bout_run.attrs["boundary_mode"] == "threshold"
    provenance = bout_run.attrs["provenance"]
    assert provenance["contract"]["name"] == "palette_stage_provenance"
    assert provenance["stage"] == "detect_bouts_multi_level"
    assert provenance["version"] == "detect_bouts_multi_level.v1"
    assert provenance["parameters"]["threshold_mm"] == 2.0
    assert provenance["parameters"]["default_level"] == "speed_filtered"
    assert provenance["parameters"]["boundary_mode"] == "threshold"
    assert provenance["inputs"]["source_track_kinematics_run"] == "tk_1"
    assert provenance["inputs"]["source_track_path"].endswith("/tk_1/tracks/id_0")
    assert provenance["artifacts"]["run_path"] == "analysis/swim_bout_runs/bouts_filtered_default"
    assert bout_run["speed_filtered"]["bouts"].attrs["is_default_level"] is True
    assert bout_run["speed_smoothed"]["bouts"].attrs["is_default_level"] is False
    assert "core_start_frame" in bout_run["speed_filtered"]["bouts"]
    intervals = bout_run["speed_filtered"]["inter_bout_intervals"]
    assert intervals.attrs["field_names"] == [
        "prev_bout_id",
        "next_bout_id",
        "prev_end_frame",
        "next_start_frame",
        "interval_frames",
        "prev_end_time_s",
        "next_start_time_s",
        "interval_s",
    ]
    assert intervals["interval_s"].shape == (0,)


def test_threshold_bouts_can_expand_to_local_minimum_boundaries() -> None:
    frames = np.arange(10, dtype=np.int64)
    speed = np.asarray([0, 0, 1, 3, 5, 3, 1, 0, 0, 0], dtype=np.float32)

    bouts = _detect_bouts_from_speed(
        speed,
        frames,
        fps=10.0,
        threshold=2.0,
        min_bout_duration_s=0.01,
        min_gap_duration_s=0.01,
        boundary_mode="local_minimum",
        boundary_window_s=0.3,
    )

    assert len(bouts) == 1
    assert bouts["core_start_frame"][0] == 3
    assert bouts["core_end_frame"][0] == 5
    assert bouts["start_frame"][0] == 1
    assert bouts["end_frame"][0] == 7
    assert bouts["duration_frames"][0] > bouts["core_duration_frames"][0]


def test_detect_and_save_bouts_requires_overwrite_for_existing_run(tmp_path: Path) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)

    detect_and_save_bouts(
        zarr_path=zarr_path,
        run_name="candidate",
        track_kinematics_run="tk_1",
        track_id=0,
        threshold_mm=2.0,
        min_bout_duration_s=0.01,
        min_gap_duration_s=0.01,
        default_level="filtered",
    )

    with pytest.raises(ValueError, match="Use --overwrite"):
        detect_and_save_bouts(
            zarr_path=zarr_path,
            run_name="candidate",
            track_kinematics_run="tk_1",
            track_id=0,
            threshold_mm=4.0,
            min_bout_duration_s=0.01,
            min_gap_duration_s=0.01,
            default_level="filtered",
        )

    detect_and_save_bouts(
        zarr_path=zarr_path,
        run_name="candidate",
        track_kinematics_run="tk_1",
        track_id=0,
        threshold_mm=4.0,
        min_bout_duration_s=0.01,
        min_gap_duration_s=0.01,
        default_level="smoothed",
        overwrite=True,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    parent = root["analysis"]["swim_bout_runs"]
    bout_run = parent["candidate"]
    assert parent.attrs["latest"] == "candidate"
    assert bout_run.attrs["threshold_mm"] == 4.0
    assert bout_run.attrs["default_level"] == "speed_smoothed"
