from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Sequence

from fisheye.utils.verify_orange_style_clips import VerifyOptions, verify_recording_clips


def _write_clip_recording(tmp_path: Path, *, metadata_rows: int = 4) -> Path:
    root = tmp_path / "recording"
    clip_dir = root / "clips" / "clip_000000"
    clip_dir.mkdir(parents=True)
    (clip_dir / "Cam2010093_example.mp4").write_bytes(b"video")
    with (clip_dir / "Cam2010093_example_meta.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame_id", "timestamp", "timestamp_sys"])
        writer.writeheader()
        for idx in range(metadata_rows):
            writer.writerow(
                {
                    "frame_id": 101 + idx,
                    "timestamp": 1001 + idx,
                    "timestamp_sys": 2001 + idx,
                }
            )
    (clip_dir / "Cam2010093_example_keyframe.json").write_text(
        json.dumps({"total_frames": 4, "fps": 2, "keyframe_frames": [0, 2]}),
        encoding="utf-8",
    )
    (clip_dir / "clip_manifest.json").write_text(
        json.dumps({"clip_id": "clip_000000", "camera_artifacts": []}),
        encoding="utf-8",
    )
    (root / "recording_clip_index.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "mode": "materialized_stream_copy",
                "clip_count": 1,
                "clips": [
                    {
                        "clip_id": "clip_000000",
                        "clip_index": 0,
                        "video_path": "clips/clip_000000/Cam2010093_example.mp4",
                        "metadata_path": "clips/clip_000000/Cam2010093_example_meta.csv",
                        "keyframe_path": "clips/clip_000000/Cam2010093_example_keyframe.json",
                        "clip_manifest_path": "clips/clip_000000/clip_manifest.json",
                        "frame_count": 4,
                        "first_recording_frame_id": 101,
                        "last_recording_frame_id": 104,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_verify_recording_clips_structural_ok(tmp_path: Path) -> None:
    root = _write_clip_recording(tmp_path)

    result = verify_recording_clips(
        VerifyOptions(
            recording_dir=root,
            index_json=root / "recording_clip_index.json",
            probe_video=False,
        )
    )

    assert result["status"] == "ok"
    assert result["clip_count"] == 1
    assert result["clips"][0]["metadata_stats"]["row_count"] == 4


def test_verify_recording_clips_reports_metadata_mismatch(tmp_path: Path) -> None:
    root = _write_clip_recording(tmp_path, metadata_rows=3)

    result = verify_recording_clips(
        VerifyOptions(
            recording_dir=root,
            index_json=root / "recording_clip_index.json",
            probe_video=False,
        )
    )

    assert result["status"] == "fail"
    failures = [
        check for check in result["clips"][0]["checks"]
        if check["status"] != "ok"
    ]
    assert any(check["code"] == "metadata_row_count_matches_index" for check in failures)
    assert any(check["code"] == "metadata_last_frame_id_matches_index" for check in failures)


def test_verify_recording_clips_probe_video_ok(tmp_path: Path) -> None:
    root = _write_clip_recording(tmp_path)
    calls: list[list[str]] = []
    progress_events: list[dict[str, object]] = []

    def fake_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        return subprocess.CompletedProcess(
            list(command),
            0,
            stdout=json.dumps({"streams": [{"nb_read_packets": "4", "avg_frame_rate": "2/1"}]}),
            stderr="",
        )

    result = verify_recording_clips(
        VerifyOptions(
            recording_dir=root,
            index_json=root / "recording_clip_index.json",
            probe_video=True,
        ),
        runner=fake_runner,
        progress=lambda event: progress_events.append(dict(event)),
    )

    assert result["status"] == "ok"
    assert calls
    assert "-count_packets" in calls[0]
    assert [event["event"] for event in progress_events] == [
        "video_probe_start",
        "video_probe_done",
    ]
    assert progress_events[0]["ordinal"] == 1
    assert progress_events[0]["total"] == 1
    assert progress_events[1]["observed_frame_count"] == 4
    assert result["clips"][0]["video_probe"]["elapsed_s"] >= 0.0


def test_verify_recording_clips_probe_video_mismatch(tmp_path: Path) -> None:
    root = _write_clip_recording(tmp_path)

    def fake_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(command),
            0,
            stdout=json.dumps({"streams": [{"nb_read_packets": "3"}]}),
            stderr="",
        )

    result = verify_recording_clips(
        VerifyOptions(
            recording_dir=root,
            index_json=root / "recording_clip_index.json",
            probe_video=True,
        ),
        runner=fake_runner,
    )

    assert result["status"] == "fail"
    failures = [
        check for check in result["clips"][0]["checks"]
        if check["status"] != "ok"
    ]
    assert any(check["code"] == "video_packet_count_matches_index" for check in failures)
