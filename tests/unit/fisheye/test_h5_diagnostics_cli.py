from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from fisheye.diagnostics.h5.cli import main


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


def _create_h5(path: Path) -> None:
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
        frame_metadata = np.zeros(4, dtype=FRAME_METADATA_DTYPE)
        frame_metadata["stimulus_frame_num"] = [0, 1, 2, 3]
        frame_metadata["triggering_camera_frame_id"] = [1, 1, 2, 2]
        frame_metadata["timestamp_ns"] = [0, 10, 20, 30]
        video_metadata = h5.require_group("video_metadata")
        video_metadata.create_dataset("frame_metadata", data=frame_metadata)


def test_cli_report_json_resolves_recording_dir(tmp_path: Path, capsys) -> None:
    recording_dir = tmp_path / "recording_a"
    _create_h5(recording_dir / "raw" / "recording_a.h5")
    exit_code = main(["report", str(recording_dir), "--json"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["core_status"] == "pass"
    assert payload["file_info"]["source_kind"] == "raw"


def test_cli_batch_writes_jsonl(tmp_path: Path, capsys) -> None:
    recording_dir = tmp_path / "recording_b"
    _create_h5(recording_dir / "raw" / "recording_b.h5")
    jsonl_path = tmp_path / "report.jsonl"
    exit_code = main(["batch", str(tmp_path), "--jsonl", str(jsonl_path)])
    assert exit_code == 0
    lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["core_status"] == "pass"
    summary_text = capsys.readouterr().out
    assert "core_files" in summary_text
