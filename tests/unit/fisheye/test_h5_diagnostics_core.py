from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from fisheye.diagnostics.h5 import build_h5_report


EVENTS_DTYPE = np.dtype([
    ("timestamp_ns_epoch", "<i8"),
    ("timestamp_ns_session", "<i8"),
    ("event_type_id", "<i4"),
    ("current_step_index", "<i4"),
    ("stimulus_frame_num", "<i8"),
    ("camera_frame_id", "<i8"),
    ("reserved", "<i4"),
    ("name_or_context", "S64"),
    ("stimulus_mode_id", "<i4"),
    ("details_json", "S256"),
])
FRAME_METADATA_DTYPE = np.dtype([
    ("stimulus_frame_num", "<i8"),
    ("triggering_camera_frame_id", "<i8"),
    ("timestamp_ns", "<i8"),
])
CHASER_DTYPE = np.dtype([
    ("stimulus_frame_num", "<i8"),
    ("chaser_pos_x", "<f4"),
    ("chaser_pos_y", "<f4"),
])
BBOX_DTYPE = np.dtype([
    ("payload_frame_id", "<i8"),
    ("x_min", "<f4"),
    ("y_min", "<f4"),
    ("width", "<f4"),
    ("height", "<f4"),
])
ENUM_DTYPE = np.dtype([("id", "<i4"), ("name", "S32")])


def _write_scalar_text(group: h5py.Group, name: str, payload: str) -> None:
    group.create_dataset(name, data=np.bytes_(payload))


def _build_frame_metadata(camera_counts: list[int]) -> np.ndarray:
    rows = sum(camera_counts)
    frame_metadata = np.zeros(rows, dtype=FRAME_METADATA_DTYPE)
    stimulus_frame = 0
    offset = 0
    for camera_frame_id, count in enumerate(camera_counts, start=1):
        end = offset + count
        frame_metadata["stimulus_frame_num"][offset:end] = np.arange(stimulus_frame, stimulus_frame + count, dtype=np.int64)
        frame_metadata["triggering_camera_frame_id"][offset:end] = camera_frame_id
        frame_metadata["timestamp_ns"][offset:end] = np.arange(offset, end, dtype=np.int64) * 10
        stimulus_frame += count
        offset = end
    return frame_metadata


def _create_recording_h5(
    base: Path,
    *,
    include_frame_metadata: bool = True,
    malformed_tracking: bool = False,
    camera_counts: list[int] | None = None,
) -> Path:
    recording_dir = base / "2026-01-01T00-00-00Z_example_DefaultScreen"
    raw_dir = recording_dir / "raw"
    raw_dir.mkdir(parents=True)
    h5_path = raw_dir / "example.h5"
    with h5py.File(h5_path, "w") as h5:
        events = np.zeros(4, dtype=EVENTS_DTYPE)
        events["timestamp_ns_epoch"] = [100, 200, 300, 400]
        events["timestamp_ns_session"] = [0, 100, 200, 300]
        events["event_type_id"] = [0, 11, 11, 1]
        events["current_step_index"] = [0, 0, 1, 1]
        events["stimulus_frame_num"] = [0, 2, 4, 6]
        events["camera_frame_id"] = [1, 1, 2, 3]
        events["name_or_context"] = [b"start", b"step_a", b"step_b", b"end"]
        events["stimulus_mode_id"] = [0, 12, 12, 0]
        events["details_json"] = [
            json.dumps({"protocol_name": "DefaultScreen"}).encode(),
            json.dumps({"step_name": "a"}).encode(),
            json.dumps({"step_name": "b"}).encode(),
            b"",
        ]
        h5.create_dataset("events", data=events)

        if include_frame_metadata:
            counts = camera_counts or [2, 2, 2]
            video_metadata = h5.require_group("video_metadata")
            video_metadata.create_dataset("frame_metadata", data=_build_frame_metadata(counts))

        tracking = h5.require_group("tracking_data")
        if malformed_tracking:
            tracking.create_dataset("chaser_states", data=np.zeros(2, dtype=np.dtype([("stimulus_frame_num", "<i8")])))
        else:
            tracking.create_dataset("chaser_states", data=np.zeros(0, dtype=CHASER_DTYPE))
        tracking.create_dataset("bounding_boxes", data=np.zeros(0, dtype=BBOX_DTYPE))

        protocol_snapshot = h5.require_group("protocol_snapshot")
        _write_scalar_text(protocol_snapshot, "protocol_definition_json", json.dumps({"protocol_name": "DefaultScreen"}))

        calibration_snapshot = h5.require_group("calibration_snapshot")
        _write_scalar_text(calibration_snapshot, "arena_config_json", json.dumps({"arena": "arena_1"}))

        recording_snapshot = h5.require_group("recording_snapshot")
        _write_scalar_text(recording_snapshot, "recording_snapshot_json", json.dumps({"recording_id": "abc"}))
        _write_scalar_text(recording_snapshot, "recording_pointer_json", json.dumps({"pointer": "xyz"}))

        subject_metadata = h5.require_group("subject_metadata")
        subject_metadata.attrs["fish_count"] = 10
        subject_metadata.attrs["dish_id"] = "dish-1"

        stimulus_coordinates = h5.require_group("stimulus_coordinates")
        stimulus_coordinates.require_group("arena_1")

        enums = h5.require_group("enums")
        enums.create_dataset(
            "events",
            data=np.array([(0, b"PROTOCOL_START"), (11, b"STEP_START")], dtype=ENUM_DTYPE),
        )
    return recording_dir


def test_h5_report_passes_on_minimal_palette_importable_recording(tmp_path: Path) -> None:
    recording_dir = _create_recording_h5(tmp_path)
    report = build_h5_report(recording_dir)
    assert report.overall_status == "pass"
    assert report.core_status == "pass"
    assert report.optional_status == "pass"
    assert report.tooling_status == "pass"
    assert report.file_info.source_kind == "raw"
    assert report.core.events_present is True
    assert report.core.frame_metadata_present is True
    assert report.events.rows == 4
    assert report.frame_metadata.rows == 6
    assert report.frame_metadata.max_abs_cumulative_drift == 0.0
    assert report.tracking.datasets["chaser_states"].status == "pass"
    assert report.snapshots.protocol_json_parseable is True
    assert report.enums.dataset_counts["events"] == 2


def test_h5_report_fails_when_frame_metadata_missing_for_palette_import(tmp_path: Path) -> None:
    recording_dir = _create_recording_h5(tmp_path, include_frame_metadata=False)
    report = build_h5_report(recording_dir, profile="palette-import")
    assert report.overall_status == "fail"
    assert report.core_status == "fail"
    assert any(f.code == "h5.frame_metadata_missing" for f in report.findings)


def test_h5_report_warns_when_frame_metadata_missing_for_citrus_contract(tmp_path: Path) -> None:
    recording_dir = _create_recording_h5(tmp_path, include_frame_metadata=False)
    report = build_h5_report(recording_dir, profile="citrus-contract")
    assert report.overall_status == "warn"
    assert report.core_status == "warn"
    assert report.frame_metadata.status == "skip"


def test_h5_report_separates_optional_tracking_failures(tmp_path: Path) -> None:
    recording_dir = _create_recording_h5(tmp_path, malformed_tracking=True)
    report = build_h5_report(recording_dir)
    assert report.overall_status == "pass"
    assert report.core_status == "pass"
    assert report.optional_status == "fail"
    assert report.tracking.status == "fail"
    assert any(f.code == "h5.tracking_chaser_states_fields_missing" for f in report.findings)


def test_h5_report_tolerates_sparse_compensated_frame_ratio_pairs(tmp_path: Path) -> None:
    recording_dir = _create_recording_h5(tmp_path, camera_counts=[2, 2, 3, 1, 2, 2])
    report = build_h5_report(recording_dir)
    assert report.overall_status == "pass"
    assert report.core_status == "pass"
    assert report.frame_metadata.status == "pass"
    assert report.frame_metadata.ratio_warn_count == 2
    assert report.frame_metadata.max_ratio_warn_run_length == 2
    assert report.frame_metadata.max_abs_cumulative_drift == 1.0
    assert not any(f.code == "h5.frame_metadata_alignment_irregular" for f in report.findings)


def test_h5_report_warns_on_large_cumulative_frame_ratio_drift(tmp_path: Path) -> None:
    recording_dir = _create_recording_h5(tmp_path, camera_counts=[2, 2, 3, 3, 3, 3, 3, 3])
    report = build_h5_report(recording_dir)
    assert report.overall_status == "warn"
    assert report.core_status == "warn"
    assert report.frame_metadata.status == "warn"
    assert report.frame_metadata.max_abs_cumulative_drift == 6.0
    assert any(f.code == "h5.frame_metadata_alignment_irregular" for f in report.findings)
