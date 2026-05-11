from __future__ import annotations

import numpy as np
import pytest
import zarr

import fisheye.analysis.detect_bouts_multi_level as detect_bouts_multi_level
from fisheye.analysis.detect_bouts_multi_level import (
    PATH_DISTANCE_LEVEL_SOURCE,
    SPEED_LEVELS,
    SWIM_BOUT_STORED_LAYOUT_COMPACT_V2,
    _compute_global_metrics,
    _compute_inter_bout_intervals,
    _create_bout_points,
    _empty_peak_events,
    _write_compact_v2_swim_bout_payloads,
    _bout_dtype,
)
from fisheye.analysis.swim_bout_io import load_default_swim_bout_tables


def _one_bout(bout_id: int, start_frame: int, end_frame: int) -> np.ndarray:
    records = np.zeros(1, dtype=_bout_dtype())
    records["bout_id"] = bout_id
    records["start_frame"] = start_frame
    records["end_frame"] = end_frame
    records["core_start_frame"] = start_frame
    records["core_end_frame"] = end_frame
    records["duration_frames"] = end_frame - start_frame + 1
    records["duration_s"] = 0.1
    records["observed_duration_s"] = 0.1
    records["core_duration_s"] = 0.1
    records["path_length_mm"] = 1.5
    records["path_length_px"] = 15.0
    records["mean_speed_mm_s"] = 15.0
    records["peak_detection_signal_mm_s"] = 20.0
    records["peak_physical_speed_mm_s"] = 18.0
    records["valid_transition_fraction"] = 1.0
    records["start_time_s"] = start_frame / 60.0
    records["end_time_s"] = end_frame / 60.0
    records["core_start_time_s"] = start_frame / 60.0
    records["core_end_time_s"] = end_frame / 60.0
    return records


def test_compact_v2_writer_helper_outputs_resolver_readable_tables() -> None:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("swim_bout_runs")
    parent.attrs["latest"] = "compact_run"
    run = parent.create_group("compact_run")
    run.attrs.update(
        {
            "schema_id": "palette.swim_bout_runs",
            "schema_version": 7,
            "source_track_kinematics_run": "tk_run",
            "track_id": 0,
            "default_level": "speed_exponential",
        }
    )

    speed_levels = list(SPEED_LEVELS)
    signal_id_by_level = {level: idx for idx, level in enumerate(speed_levels)}
    path_distance_source = {**PATH_DISTANCE_LEVEL_SOURCE, "speed_exponential": "filtered"}
    estimator_signal_id_by_level = {
        level: signal_id_by_level[f"speed_{path_distance_source[level]}"]
        for level in speed_levels
    }
    frames = np.arange(20, dtype=np.int64)
    level_payloads = {}
    for idx, level in enumerate(speed_levels):
        bouts = _one_bout(idx + 1, idx * 2, idx * 2 + 1)
        intervals, _interval_metrics, hist = _compute_inter_bout_intervals(bouts, fps=60.0)
        level_payloads[level] = {
            "bouts": bouts,
            "peak_events": _empty_peak_events(),
            "intervals": intervals,
            "interval_histogram": hist,
            "global_metrics": _compute_global_metrics(bouts, fps=60.0, total_frames=20),
            "bout_points": _create_bout_points(bouts, None, None, frames, fps=60.0),
            "attrs": {"n_bouts": int(bouts.size), "speed_level": level},
        }

    _write_compact_v2_swim_bout_payloads(
        run,
        run_name="compact_run",
        speed_levels=speed_levels,
        level_payloads=level_payloads,
        signal_id_by_level=signal_id_by_level,
        estimator_signal_id_by_level=estimator_signal_id_by_level,
        default_level_key="speed_exponential",
        method="peak_event",
        parameters={
            "method": "peak_event",
            "boundary_mode": "threshold",
            "boundary_window_s": 0.25,
            "gap_merge_policy": "sampled_frame_gap",
            "min_bout_duration_s": 0.05,
            "min_gap_duration_s": 0.1,
            "min_gap_frames": None,
        },
        provenance={},
        track_id=0,
        pixel_to_mm=0.1,
        path_distance_level_source=path_distance_source,
        source_track_path="analysis/track_kinematics_runs/offline/tk_run/tracks/id_0",
        exponential_source_key="speed_filtered",
        exponential_tau_s=0.025,
        frames=frames,
        speeds={
            "speed_exponential_mm": np.linspace(0.0, 1.0, frames.size, dtype=np.float64),
        },
    )

    payload = load_default_swim_bout_tables(root)

    assert run.attrs["layout"] == SWIM_BOUT_STORED_LAYOUT_COMPACT_V2
    assert "speed_exponential" not in run
    assert "indexes" in run
    assert "tables" in run
    assert payload.signal.speed_level == "speed_exponential"
    assert payload.signal.role == "detector_response"
    assert payload.bouts["signal_id"].tolist() == [signal_id_by_level["speed_exponential"]]
    assert payload.bouts["estimator_signal_id"].tolist() == [signal_id_by_level["speed_filtered"]]
    assert payload.global_metrics["n_bouts"][0] == 1.0
    assert payload.series["detection_signal_mm_s"].shape == (frames.size,)


def test_detect_and_save_bouts_defaults_to_compact_v2_layout(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")
    frames = np.arange(12, dtype=np.int64)
    speed = np.asarray([0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0], dtype=np.float64)
    transition_valid = np.ones(frames.size, dtype=bool)
    transition_valid[0] = False

    def fake_load_track(_zarr_path, _track_kinematics_run, track_id):
        speeds = {
            "speed_raw_mm": speed,
            "speed_filtered_mm": speed,
            "speed_smoothed_mm": speed,
            "speed_averaged_mm": speed,
            "frames": frames,
            "frame_path_distance_raw_mm": speed / 60.0,
            "frame_path_distance_raw_px": speed / 6.0,
            "frame_path_distance_filtered_mm": speed / 60.0,
            "frame_path_distance_filtered_px": speed / 6.0,
            "frame_path_distance_smoothed_mm": speed / 60.0,
            "frame_path_distance_smoothed_px": speed / 6.0,
            "delta_seconds": np.full(frames.size, 1.0 / 60.0),
            "transition_valid": transition_valid,
            "sample_valid": np.ones(frames.size, dtype=bool),
        }
        metadata = {
            "fps": 60.0,
            "pixel_to_mm": 0.1,
            "n_frames": frames.size,
            "track_kinematics_run": "tk_run",
            "track_id": track_id,
            "positions_mm": np.column_stack((frames, frames)).astype(np.float64),
            "positions_px": np.column_stack((frames * 10, frames * 10)).astype(np.float64),
        }
        return speeds, metadata

    monkeypatch.setattr(
        detect_bouts_multi_level,
        "_load_track_kinematics_track_speeds",
        fake_load_track,
    )

    detect_bouts_multi_level.detect_and_save_bouts(
        zarr_path=zarr_path,
        run_name="compact_run",
        track_kinematics_run="tk_run",
        method="threshold",
        threshold_mm=0.5,
        min_bout_duration_s=0.01,
        min_gap_duration_s=0.01,
        default_level="exponential",
        command="test",
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["swim_bout_runs"]["compact_run"]
    payload = load_default_swim_bout_tables(root)

    assert run.attrs["schema_version"] == 7
    assert run.attrs["layout"] == SWIM_BOUT_STORED_LAYOUT_COMPACT_V2
    assert "speed_exponential" not in run
    assert payload.signal.speed_level == "speed_exponential"
    assert payload.bouts.size > 0
