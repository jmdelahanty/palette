from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.detect_bouts_multi_level import (
    _causal_exponential_speed_response,
    _detect_bouts_from_peak_events,
    _detect_bouts_from_speed,
    _duration_seconds_to_frames,
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
    transition_valid = np.asarray(
        [False, True, True, True, True, False, True, True, True, True, True, True],
        dtype=bool,
    )
    delta_seconds = np.zeros(frames.size, dtype=np.float32)
    delta_seconds[1:] = 0.01
    path_mm = np.full(frames.size, 0.5, dtype=np.float32)
    path_mm[0] = 0.0
    path_px = path_mm / 0.1
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
    _write_array(track, "frame_path_distance_raw_mm", path_mm)
    _write_array(track, "frame_path_distance_raw_px", path_px)
    _write_array(track, "frame_path_distance_filtered_mm", path_mm)
    _write_array(track, "frame_path_distance_filtered_px", path_px)
    _write_array(track, "frame_path_distance_smoothed_mm", path_mm)
    _write_array(track, "frame_path_distance_smoothed_px", path_px)
    _write_array(track, "delta_seconds", delta_seconds)
    _write_array(track, "transition_valid", transition_valid)
    _write_array(track, "sample_valid", np.ones(frames.size, dtype=bool))
    _write_array(track, "positions_px", positions_px)
    _write_array(track, "positions_mm", positions_px * 0.1)
    return zarr_path


def test_normalize_speed_level_accepts_aliases() -> None:
    assert normalize_speed_level("filtered") == "speed_filtered"
    assert normalize_speed_level("speed_filtered") == "speed_filtered"
    assert normalize_speed_level("exponential") == "speed_exponential"


def test_normalize_speed_level_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported speed level"):
        normalize_speed_level("median")


def test_duration_seconds_to_frames_uses_ceiling() -> None:
    assert _duration_seconds_to_frames(0.03, 60.0) == 2
    assert _duration_seconds_to_frames(0.1, 60.0) == 6
    assert _duration_seconds_to_frames(0.0, 60.0) == 0


def test_min_gap_duration_resolves_to_ceiled_frames() -> None:
    frames = np.arange(8, dtype=np.int64)
    speed = np.asarray([0, 2, 2, 0, 2, 2, 0, 0], dtype=np.float32)

    merged = _detect_bouts_from_speed(
        speed,
        frames,
        fps=60.0,
        threshold=1.0,
        min_bout_duration_s=0.01,
        min_gap_duration_s=0.03,
    )
    assert merged["start_frame"].tolist() == [1]
    assert merged["end_frame"].tolist() == [5]

    split = _detect_bouts_from_speed(
        speed,
        frames,
        fps=60.0,
        threshold=1.0,
        min_bout_duration_s=0.01,
        min_gap_duration_s=0.03,
        min_gap_frames=1,
    )
    assert split["start_frame"].tolist() == [1, 4]
    assert split["end_frame"].tolist() == [2, 5]


def test_interpolated_core_gap_policy_can_split_subframe_threshold_gaps() -> None:
    frames = np.arange(7, dtype=np.int64)
    speed = np.asarray([0.0, 3.0, 3.0, 0.0, 3.0, 3.0, 0.0], dtype=np.float32)

    sampled = _detect_bouts_from_speed(
        speed,
        frames,
        fps=10.0,
        threshold=2.0,
        min_bout_duration_s=0.01,
        min_gap_duration_s=0.12,
        gap_merge_policy="sampled_frame_gap",
    )
    assert sampled["start_frame"].tolist() == [1]
    assert sampled["end_frame"].tolist() == [5]

    interpolated = _detect_bouts_from_speed(
        speed,
        frames,
        fps=10.0,
        threshold=2.0,
        min_bout_duration_s=0.01,
        min_gap_duration_s=0.12,
        gap_merge_policy="interpolated_core_gap",
    )
    assert interpolated["start_frame"].tolist() == [1, 4]
    assert interpolated["end_frame"].tolist() == [2, 5]
    np.testing.assert_allclose(
        interpolated["core_start_time_s_interpolated"],
        [1.0 / 15.0, 11.0 / 30.0],
    )
    np.testing.assert_allclose(
        interpolated["core_end_time_s_interpolated"],
        [7.0 / 30.0, 8.0 / 15.0],
    )


def test_threshold_bouts_record_interpolated_core_crossing_times() -> None:
    frames = np.arange(6, dtype=np.int64)
    speed = np.asarray([0.0, 1.0, 3.0, 5.0, 1.0, 0.0], dtype=np.float32)

    bouts = _detect_bouts_from_speed(
        speed,
        frames,
        fps=10.0,
        threshold=2.0,
        min_bout_duration_s=0.01,
        min_gap_duration_s=0.01,
    )

    assert bouts["start_frame"].tolist() == [2]
    assert bouts["end_frame"].tolist() == [3]
    assert bouts["core_start_time_interpolated_valid"].tolist() == [True]
    assert bouts["core_end_time_interpolated_valid"].tolist() == [True]
    np.testing.assert_allclose(bouts["core_start_time_s_interpolated"], [0.15])
    np.testing.assert_allclose(bouts["core_end_time_s_interpolated"], [0.375])
    np.testing.assert_allclose(bouts["core_duration_s_interpolated"], [0.225])


def test_peak_event_detector_splits_two_prominent_peaks_in_one_threshold_blob() -> None:
    frames = np.arange(7, dtype=np.int64)
    speed = np.asarray([0.0, 1.0, 10.0, 2.0, 9.0, 1.0, 0.0], dtype=np.float32)

    bouts, peak_events = _detect_bouts_from_peak_events(
        speed,
        frames,
        fps=10.0,
        min_peak_height_mm_s=5.0,
        min_peak_prominence_mm_s=4.0,
        min_peak_distance_s=0.1,
        peak_width_rel_height=0.9,
        min_bout_duration_s=0.01,
    )

    assert bouts["bout_id"].tolist() == [1, 2]
    assert peak_events["bout_id"].tolist() == [1, 2]
    assert peak_events["peak_frame"].tolist() == [2, 4]
    np.testing.assert_allclose(peak_events["peak_signal_value_mm_s"], [10.0, 9.0])
    assert (bouts["start_frame"] <= peak_events["peak_frame"]).all()
    assert (peak_events["peak_frame"] <= bouts["end_frame"]).all()
    assert bouts["end_frame"][0] < bouts["start_frame"][1]


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
    assert provenance["version"] == "detect_bouts_multi_level.v7"
    assert provenance["parameters"]["threshold_mm"] == 2.0
    assert provenance["parameters"]["default_level"] == "speed_filtered"
    assert provenance["parameters"]["boundary_mode"] == "threshold"
    assert provenance["parameters"]["duration_frame_rounding_policy"] == "ceil_seconds_times_fps"
    assert provenance["parameters"]["threshold_crossing_interpolation"] == "linear_between_samples"
    assert provenance["parameters"]["resolved_min_bout_frames"] == 1
    assert provenance["parameters"]["resolved_min_gap_frames"] == 1
    assert provenance["parameters"]["min_gap_frame_source"] == "ceil_seconds_times_fps"
    assert provenance["parameters"]["gap_merge_policy"] == "sampled_frame_gap"
    assert provenance["parameters"]["gap_merge_policy_active"] is True
    assert provenance["parameters"]["gap_merge_min_gap_duration_s"] == 0.01
    assert provenance["parameters"]["gap_merge_min_gap_source"] == "seconds"
    assert provenance["parameters"]["exponential_tau_s"] == 0.05
    assert provenance["parameters"]["exponential_source_level"] == "speed_filtered"
    assert provenance["inputs"]["source_track_kinematics_run"] == "tk_1"
    assert provenance["inputs"]["source_track_path"].endswith("/tk_1/tracks/id_0")
    assert provenance["artifacts"]["run_path"] == "analysis/swim_bout_runs/bouts_filtered_default"
    assert bout_run["speed_filtered"]["bouts"].attrs["is_default_level"] is True
    assert bout_run.attrs["schema_id"] == "palette.swim_bout_runs"
    assert bout_run.attrs["schema_version"] == 6
    assert bout_run.attrs["detection_signal_schema_id"] == "palette.swim_bout_detection_signal.v1"
    assert bout_run.attrs["resolved_min_gap_frames"] == 1
    assert bout_run.attrs["gap_merge_policy"] == "sampled_frame_gap"
    assert bout_run.attrs["gap_merge_policy_active"] is True
    assert bout_run.attrs["gap_merge_min_gap_duration_s"] == 0.01
    assert bout_run.attrs["gap_merge_min_gap_source"] == "seconds"
    assert bout_run.attrs["threshold_crossing_interpolation"] == "linear_between_samples"
    assert bout_run["speed_filtered"].attrs["resolved_min_gap_frames"] == 1
    assert bout_run["speed_filtered"].attrs["gap_merge_policy"] == "sampled_frame_gap"
    assert "peak_events" in bout_run["speed_filtered"]
    assert bout_run["speed_filtered"]["peak_events"].attrs["peak_event_schema_id"] == (
        "palette.swim_bout_peak_events.v1"
    )
    assert bout_run["speed_filtered"]["peak_events"]["bout_id"].shape == (0,)
    assert (
        bout_run["speed_filtered"]["bouts"].attrs["bout_metric_schema_id"]
        == "palette.swim_bout_metrics.v3"
    )
    assert bout_run["speed_smoothed"]["bouts"].attrs["is_default_level"] is False
    assert "speed_exponential" in bout_run
    assert bout_run["speed_filtered"].attrs["detection_signal_transform_type"] == "identity"
    assert bout_run["speed_filtered"].attrs["detection_signal_is_primary_physical_speed"] is True
    assert bout_run["speed_filtered"].attrs["movement_metric_source_level"] == "filtered"
    assert bout_run["speed_filtered"].attrs["peak_detection_signal_field"] == "peak_detection_signal_mm_s"
    assert bout_run["speed_exponential"].attrs["speed_transform"] == "causal_exponential_response"
    assert bout_run["speed_exponential"].attrs["exponential_source_level"] == "speed_filtered"
    assert bout_run["speed_exponential"].attrs["detection_signal_transform_type"] == "convolution"
    assert bout_run["speed_exponential"].attrs["detection_signal_kernel_family"] == "causal_exponential"
    assert bout_run["speed_exponential"].attrs["detection_signal_source_level"] == "speed_filtered"
    assert bout_run["speed_exponential"].attrs["detection_signal_is_primary_physical_speed"] is False
    assert bout_run["speed_exponential"]["bouts"].attrs["speed_transform"] == "causal_exponential_response"
    assert bout_run["speed_exponential"]["bouts"].attrs["exponential_source_level"] == "speed_filtered"
    assert "detection_signal_mm_s" in bout_run["speed_exponential"]
    assert bout_run["speed_exponential"]["detection_signal_mm_s"].attrs["detection_signal_transform_type"] == (
        "convolution"
    )
    assert bout_run["speed_exponential"]["detection_signal_mm_s"].shape == (12,)
    assert "core_start_frame" in bout_run["speed_filtered"]["bouts"]
    assert "distance" not in bout_run["speed_filtered"]["bouts"].attrs["field_names"]
    assert "path_length_mm" in bout_run["speed_filtered"]["bouts"].attrs["field_names"]
    assert "peak_detection_signal_mm_s" in bout_run["speed_filtered"]["bouts"].attrs["field_names"]
    assert "peak_physical_speed_mm_s" in bout_run["speed_filtered"]["bouts"].attrs["field_names"]
    assert "peak_speed_mm_s" not in bout_run["speed_filtered"]["bouts"].attrs["field_names"]
    assert "observed_duration_s" in bout_run["speed_filtered"]["bouts"].attrs["field_names"]
    assert "gap_censored" in bout_run["speed_filtered"]["bouts"].attrs["field_names"]
    assert "core_start_time_s_interpolated" in bout_run["speed_filtered"]["bouts"].attrs["field_names"]
    assert "core_end_time_s_interpolated" in bout_run["speed_filtered"]["bouts"].attrs["field_names"]
    bouts = bout_run["speed_filtered"]["bouts"]
    assert bouts["n_invalid_transitions"][:].tolist() == [1]
    assert bouts["gap_censored"][:].tolist() == [True]
    np.testing.assert_allclose(bouts["observed_duration_s"][:], [0.05])
    np.testing.assert_allclose(bouts["path_length_mm"][:], [2.5])
    np.testing.assert_allclose(bouts["path_length_px"][:], [25.0])
    np.testing.assert_allclose(bouts["mean_speed_mm_s"][:], [50.0])
    np.testing.assert_allclose(bouts["peak_detection_signal_mm_s"][:], [5.0])
    np.testing.assert_allclose(bouts["peak_physical_speed_mm_s"][:], [5.0])
    np.testing.assert_allclose(bouts["valid_transition_fraction"][:], [5 / 6])
    global_metrics = bout_run["speed_filtered"]["global_metrics"]
    assert "total_distance" not in global_metrics.attrs["field_names"]
    assert "total_path_length_mm" in global_metrics.attrs["field_names"]
    np.testing.assert_allclose(global_metrics["total_path_length_mm"][:], [2.5])
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


def test_detect_and_save_bouts_writes_peak_event_metadata(tmp_path: Path) -> None:
    zarr_path = _make_track_kinematics_archive(tmp_path)

    run_name = detect_and_save_bouts(
        zarr_path=zarr_path,
        run_name="bouts_peak_event",
        track_kinematics_run="tk_1",
        track_id=0,
        method="peak_event",
        min_peak_height_mm_s=3.0,
        min_peak_prominence_mm_s=2.0,
        min_peak_distance_s=0.05,
        peak_width_rel_height=0.9,
        min_bout_duration_s=0.01,
        default_level="filtered",
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    bout_run = root["analysis"]["swim_bout_runs"][run_name]
    assert bout_run.attrs["detection_method"] == "peak_event"
    assert bout_run.attrs["schema_version"] == 6
    assert bout_run.attrs["peak_event_schema_version"] == 1
    assert bout_run.attrs["min_peak_height_mm_s"] == 3.0
    assert bout_run.attrs["min_peak_prominence_mm_s"] == 2.0
    assert bout_run.attrs["min_peak_distance_s"] == 0.05
    assert bout_run.attrs["peak_event_boundary_mode"] == "relative_prominence_width"
    assert bout_run.attrs["shape_split_policy"] == "none"
    assert bout_run.attrs["gap_merge_policy_active"] is False
    provenance = bout_run.attrs["provenance"]
    assert provenance["version"] == "detect_bouts_multi_level.v7"
    assert provenance["parameters"]["method"] == "peak_event"
    assert provenance["parameters"]["min_peak_height_mm_s"] == 3.0
    assert provenance["parameters"]["min_peak_prominence_mm_s"] == 2.0
    assert provenance["parameters"]["peak_event_schema_version"] == 1
    assert provenance["parameters"]["gap_merge_policy_active"] is False

    level = bout_run["speed_filtered"]
    assert level.attrs["n_peak_events"] == level.attrs["n_bouts"]
    peak_events = level["peak_events"]
    bouts = level["bouts"]
    assert peak_events["bout_id"].shape == bouts["bout_id"].shape
    assert peak_events["peak_frame"][:].tolist() == [4]
    assert peak_events["peak_signal_value_mm_s"][:].tolist() == [5.0]
    assert peak_events["boundary_mode"][:].shape[0] == 1
    assert "peak_prominence_mm_s" in peak_events.attrs["field_names"]


def test_causal_exponential_speed_response_resets_on_gap() -> None:
    speed = np.asarray([0.0, 10.0, 0.0, 10.0], dtype=np.float32)
    frames = np.asarray([0, 1, 2, 10], dtype=np.int64)
    response = _causal_exponential_speed_response(
        speed,
        frames,
        fps=10.0,
        tau_s=0.1,
        transition_valid=np.asarray([False, True, True, False]),
    )

    expected_alpha = 1.0 - np.exp(-1.0)
    np.testing.assert_allclose(response[0], 0.0)
    np.testing.assert_allclose(response[1], 10.0 * expected_alpha)
    assert response[2] < response[1]
    np.testing.assert_allclose(response[3], 10.0)


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
