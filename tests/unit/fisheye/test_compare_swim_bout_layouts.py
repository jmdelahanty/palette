from __future__ import annotations

import numpy as np
import zarr

from fisheye.shared.zarr.columnar import store_array, write_columnar_dataset
from fisheye.utils.compare_swim_bout_layouts import (
    compare_swim_bout_layouts,
    comparison_to_dict,
)


def _bouts(*, compact: bool = False) -> np.ndarray:
    fields = []
    if compact:
        fields.extend(
            [
                ("candidate_id", "i4"),
                ("signal_id", "i4"),
                ("estimator_signal_id", "i4"),
                ("track_id", "i4"),
            ]
        )
    fields.extend(
        [
            ("bout_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("duration_s", "f8"),
            ("path_length_mm", "f8"),
            ("mean_speed_mm_s", "f8"),
            ("peak_detection_signal_mm_s", "f8"),
        ]
    )
    records = np.zeros(2, dtype=fields)
    if compact:
        records["candidate_id"] = 0
        records["signal_id"] = 1
        records["estimator_signal_id"] = 0
        records["track_id"] = 0
    records["bout_id"] = [0, 1]
    records["start_frame"] = [10, 30]
    records["end_frame"] = [20, 44]
    records["duration_s"] = [1.0 / 6.0, 14.0 / 60.0]
    records["path_length_mm"] = [1.5, 2.25]
    records["mean_speed_mm_s"] = [9.0, 9.642857142857142]
    records["peak_detection_signal_mm_s"] = [42.0, 51.0]
    return records


def _peak_events(*, compact: bool = False) -> np.ndarray:
    fields = []
    if compact:
        fields.extend(
            [
                ("peak_event_id", "i8"),
                ("candidate_id", "i4"),
                ("signal_id", "i4"),
            ]
        )
    fields.extend(
        [
            ("bout_id", "i8"),
            ("peak_frame", "i8"),
            ("peak_time_s", "f8"),
            ("peak_signal_value_mm_s", "f8"),
            ("peak_prominence_mm_s", "f8"),
        ]
    )
    records = np.zeros(2, dtype=fields)
    if compact:
        records["peak_event_id"] = [0, 1]
        records["candidate_id"] = 0
        records["signal_id"] = 1
    records["bout_id"] = [0, 1]
    records["peak_frame"] = [15, 35]
    records["peak_time_s"] = [0.25, 0.5833333333333334]
    records["peak_signal_value_mm_s"] = [42.0, 51.0]
    records["peak_prominence_mm_s"] = [4.0, 5.0]
    return records


def _intervals(*, compact: bool = False) -> np.ndarray:
    fields = []
    if compact:
        fields.extend(
            [
                ("interval_id", "i8"),
                ("candidate_id", "i4"),
                ("signal_id", "i4"),
            ]
        )
    fields.extend(
        [
            ("prev_bout_id", "i8"),
            ("next_bout_id", "i8"),
            ("prev_end_frame", "i8"),
            ("next_start_frame", "i8"),
            ("prev_end_time_s", "f8"),
            ("next_start_time_s", "f8"),
            ("interval_s", "f8"),
        ]
    )
    records = np.zeros(1, dtype=fields)
    if compact:
        records["interval_id"] = [0]
        records["candidate_id"] = 0
        records["signal_id"] = 1
    records["prev_bout_id"] = [0]
    records["next_bout_id"] = [1]
    records["prev_end_frame"] = [20]
    records["next_start_frame"] = [30]
    records["prev_end_time_s"] = [20.0 / 60.0]
    records["next_start_time_s"] = [0.5]
    records["interval_s"] = [10.0 / 60.0]
    return records


def _write_reference_run(parent: zarr.Group) -> None:
    run = parent.create_group("bouts_v1")
    run.attrs.update(
        {
            "schema_id": "palette.swim_bout_runs",
            "schema_version": 6,
            "source_track_kinematics_run": "tk_hyst4_low2_s005",
            "track_id": 0,
            "detection_method": "peak_event",
            "default_level": "speed_exponential",
        }
    )
    level = run.create_group("speed_exponential")
    level.attrs.update(
        {
            "speed_level": "speed_exponential",
            "path_distance_source_level": "filtered",
            "detection_signal_transform_type": "exponential",
            "detection_signal_source_level": "filtered",
        }
    )
    write_columnar_dataset(level, "bouts", _bouts())
    write_columnar_dataset(level, "peak_events", _peak_events())
    write_columnar_dataset(level, "inter_bout_intervals", _intervals())
    store_array(level, "detection_signal_mm_s", np.asarray([0.0, 4.0, 8.0], dtype=np.float32))
    store_array(level, "frame_indices", np.asarray([10, 11, 12], dtype=np.int64))


def _write_candidate_run(parent: zarr.Group) -> None:
    run = parent.create_group("bouts_compact")
    run.attrs.update(
        {
            "schema_id": "palette.swim_bout_runs",
            "schema_version": 7,
            "layout": "compact_tabular_v2",
            "source_track_kinematics_run": "tk_hyst4_low2_s005",
            "track_id": 0,
            "default_candidate_id": 0,
            "default_signal_id": 1,
        }
    )
    indexes = run.create_group("indexes")
    tables = run.create_group("tables")
    signals = run.create_group("signals")

    candidates = np.zeros(
        1,
        dtype=[
            ("candidate_id", "i4"),
            ("candidate_name", "S32"),
            ("is_default", "?"),
            ("detection_method", "S32"),
            ("parameters_json", "S128"),
        ],
    )
    candidates[0] = (0, b"compact_candidate", True, b"peak_event", b'{"method":"peak_event"}')
    write_columnar_dataset(indexes, "candidates", candidates)

    signal_variants = np.zeros(
        1,
        dtype=[
            ("signal_id", "i4"),
            ("speed_level", "S32"),
            ("signal_name", "S32"),
            ("role", "S32"),
            ("source_level", "S32"),
            ("transform_type", "S32"),
            ("transform_source_signal_id", "i4"),
            ("tau_s", "f8"),
            ("units", "S16"),
            ("path_distance_source_level", "S32"),
        ],
    )
    signal_variants[0] = (
        1,
        b"speed_exponential",
        b"exponential",
        b"detector_response",
        b"speed_filtered",
        b"exponential",
        0,
        0.025,
        b"mm/s",
        b"filtered",
    )
    write_columnar_dataset(indexes, "signal_variants", signal_variants)
    write_columnar_dataset(tables, "bouts", _bouts(compact=True))
    write_columnar_dataset(tables, "peak_events", _peak_events(compact=True))
    write_columnar_dataset(tables, "inter_bout_intervals", _intervals(compact=True))
    store_array(signals, "detector_signal_mm_s", np.asarray([[0.0, 4.0, 8.0]], dtype=np.float32))
    store_array(signals, "detector_signal_signal_ids", np.asarray([1], dtype=np.int32))
    store_array(signals, "frame_indices", np.asarray([10, 11, 12], dtype=np.int64))


def test_compare_swim_bout_layouts_passes_for_equivalent_v1_and_compact_v2(tmp_path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(zarr_path, mode="w")
    parent = root.create_group("analysis").create_group("swim_bout_runs")
    parent.attrs["latest"] = "bouts_v1"
    _write_reference_run(parent)
    _write_candidate_run(parent)

    comparison = compare_swim_bout_layouts(
        zarr_path,
        reference_run="bouts_v1",
        candidate_run="bouts_compact",
        legacy_compatibility=True,
    )
    payload = comparison_to_dict(comparison)

    assert comparison.passed
    assert comparison.checks_failed == 0
    assert comparison.reference_signal == "speed_exponential"
    assert comparison.candidate_signal == "speed_exponential"
    assert payload["checks_failed"] == 0
    assert comparison.reference_object_counts.zarr_json_count is not None
    assert comparison.candidate_object_counts.zarr_json_count is not None
