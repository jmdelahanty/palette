from __future__ import annotations

import numpy as np
import zarr

from fisheye.analysis.chaser_state_interpolator import store_array, write_columnar_dataset
from fisheye.analysis.swim_bout_io import (
    discover_swim_bout_candidates,
    load_default_swim_bout_tables,
    load_swim_bout_tables,
    structured_records_to_dicts,
)
from fisheye.utils.export_cross_recording_analytics import _load_swim_bout_metrics


def _bout_records(offset: int = 0) -> np.ndarray:
    records = np.zeros(
        2,
        dtype=[
            ("bout_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("duration_s", "f8"),
            ("path_length_mm", "f8"),
        ],
    )
    records["bout_id"] = [0 + offset, 1 + offset]
    records["start_frame"] = [10 + offset, 30 + offset]
    records["end_frame"] = [20 + offset, 42 + offset]
    records["duration_s"] = [0.16, 0.20]
    records["path_length_mm"] = [1.5, 2.25]
    return records


def _peak_records() -> np.ndarray:
    records = np.zeros(
        2,
        dtype=[
            ("bout_id", "i8"),
            ("peak_frame", "i8"),
            ("peak_time_s", "f8"),
            ("peak_signal_value_mm_s", "f8"),
        ],
    )
    records["bout_id"] = [0, 1]
    records["peak_frame"] = [15, 35]
    records["peak_time_s"] = [0.25, 0.58]
    records["peak_signal_value_mm_s"] = [42.0, 51.0]
    return records


def _interval_records() -> np.ndarray:
    records = np.zeros(
        1,
        dtype=[
            ("prev_bout_id", "i8"),
            ("next_bout_id", "i8"),
            ("interval_s", "f8"),
        ],
    )
    records["prev_bout_id"] = [0]
    records["next_bout_id"] = [1]
    records["interval_s"] = [0.18]
    return records


def _build_v1_swim_bout_root() -> zarr.Group:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("swim_bout_runs")
    parent.attrs["latest"] = "bouts_canary"

    run = parent.create_group("bouts_canary")
    run.attrs.update(
        {
            "schema_id": "palette.swim_bout_runs",
            "schema_version": 6,
            "source_track_kinematics_run": "tk_hyst4_low2_s005",
            "track_id": 0,
            "detection_method": "peak_event",
            "default_level": "speed_exponential",
            "exponential_tau_s": 0.025,
        }
    )

    filtered = run.create_group("speed_filtered")
    filtered.attrs.update(
        {
            "n_bouts": 2,
            "speed_level": "speed_filtered",
            "path_distance_source_level": "filtered",
        }
    )
    write_columnar_dataset(filtered, "bouts", _bout_records(offset=100))
    write_columnar_dataset(filtered, "peak_events", _peak_records())
    write_columnar_dataset(filtered, "inter_bout_intervals", _interval_records())

    exponential = run.create_group("speed_exponential")
    exponential.attrs.update(
        {
            "n_bouts": 2,
            "speed_level": "speed_exponential",
            "path_distance_source_level": "filtered",
            "detection_signal_transform_type": "exponential",
            "detection_signal_source_level": "filtered",
        }
    )
    write_columnar_dataset(exponential, "bouts", _bout_records())
    write_columnar_dataset(exponential, "peak_events", _peak_records())
    write_columnar_dataset(exponential, "inter_bout_intervals", _interval_records())
    store_array(exponential, "detection_signal_mm_s", np.asarray([0.0, 4.0, 8.0], dtype=np.float32))
    store_array(exponential, "frame_indices", np.asarray([10, 11, 12], dtype=np.int64))
    return root


def test_discover_swim_bout_candidates_maps_v1_levels_to_signals() -> None:
    root = _build_v1_swim_bout_root()

    candidates = discover_swim_bout_candidates(
        root,
        track_run_name="offline/tk_hyst4_low2_s005",
        track_id=0,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.run_name == "bouts_canary"
    assert candidate.is_latest is True
    assert candidate.detection_method == "peak_event"
    assert candidate.default_speed_level == "speed_exponential"
    assert [signal.speed_level for signal in candidate.signals] == [
        "speed_filtered",
        "speed_exponential",
    ]
    assert [signal.role for signal in candidate.signals] == [
        "physical_estimator",
        "detector_response",
    ]
    assert candidate.default_signal_id == 1


def test_load_default_swim_bout_tables_uses_default_level() -> None:
    root = _build_v1_swim_bout_root()

    payload = load_default_swim_bout_tables(root)

    assert payload.run_name == "bouts_canary"
    assert payload.signal.speed_level == "speed_exponential"
    assert payload.signal.role == "detector_response"
    assert payload.level_path == "analysis/swim_bout_runs/bouts_canary/speed_exponential"
    assert payload.bouts["bout_id"].tolist() == [0, 1]
    assert payload.inter_bout_intervals["interval_s"].tolist() == [0.18]
    assert payload.series["detection_signal_mm_s"].tolist() == [0.0, 4.0, 8.0]
    assert structured_records_to_dicts(payload.bouts)[0]["start_frame"] == 10


def test_load_swim_bout_tables_can_select_non_default_speed_level() -> None:
    root = _build_v1_swim_bout_root()

    payload = load_swim_bout_tables(root, speed_level="filtered")

    assert payload.signal.speed_level == "speed_filtered"
    assert payload.signal.role == "physical_estimator"
    assert payload.bouts["bout_id"].tolist() == [100, 101]


def test_cross_recording_export_uses_swim_bout_resolver() -> None:
    root = _build_v1_swim_bout_root()

    rows = _load_swim_bout_metrics(
        root,
        export_run_id="export_test",
        zarr_path="/tmp/example_analysis.zarr",
        recording_id="recording_1",
        stimulus_run=None,
        protocol_signature=None,
        steps=[],
        tables={"swim_bout_metrics"},
        diagnostics=[],
    )

    assert len(rows) == 2
    assert rows[0]["swim_bout_run"] == "bouts_canary"
    assert rows[0]["speed_level"] == "speed_exponential"
    assert rows[0]["candidate_id"] == 0
    assert rows[0]["signal_id"] == 1
    assert rows[0]["signal_role"] == "detector_response"
    assert rows[0]["signal_source_level"] == "filtered"
    assert rows[0]["bout_id"] == 0
