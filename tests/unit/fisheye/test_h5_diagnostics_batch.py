from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from fisheye.diagnostics.h5.batch import build_batch_report


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


def _create_h5(path: Path, *, include_frame_metadata: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        events = np.zeros(2, dtype=EVENTS_DTYPE)
        events["timestamp_ns_epoch"] = [100, 200]
        events["timestamp_ns_session"] = [0, 100]
        events["event_type_id"] = [0, 1]
        events["current_step_index"] = [0, 1]
        events["stimulus_frame_num"] = [0, 2]
        events["camera_frame_id"] = [1, 2]
        events["name_or_context"] = [b"start", b"end"]
        events["stimulus_mode_id"] = [0, 0]
        events["details_json"] = [b"{}", b"{}"]
        h5.create_dataset("events", data=events)
        if include_frame_metadata:
            frame_metadata = np.zeros(4, dtype=FRAME_METADATA_DTYPE)
            frame_metadata["stimulus_frame_num"] = [0, 1, 2, 3]
            frame_metadata["triggering_camera_frame_id"] = [1, 1, 2, 2]
            frame_metadata["timestamp_ns"] = [0, 10, 20, 30]
            video_metadata = h5.require_group("video_metadata")
            video_metadata.create_dataset("frame_metadata", data=frame_metadata)


def test_batch_report_groups_by_recording_root(tmp_path: Path) -> None:
    first = tmp_path / "recording_one" / "raw" / "one.h5"
    second = tmp_path / "recording_two" / "raw" / "two.h5"
    _create_h5(first)
    _create_h5(second, include_frame_metadata=False)
    report = build_batch_report([tmp_path], recursive=True)
    assert report.summary.scanned == 2
    assert len(report.recordings) == 2
    roots = {item.recording_root for item in report.recordings}
    assert str(tmp_path / "recording_one") in roots
    assert str(tmp_path / "recording_two") in roots
    assert report.summary.recording_counts["pass"] == 1
    assert report.summary.recording_counts["fail"] == 1
