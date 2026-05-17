from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from fisheye.utils.plan_orange_style_clips import (
    build_clip_plan,
    choose_clip_boundaries,
    write_plan_artifacts,
)


def test_choose_clip_boundaries_snaps_to_next_keyframe() -> None:
    boundaries = choose_clip_boundaries(
        total_frames=101,
        fps=10.0,
        keyframe_frames=np.array([0, 31, 61, 91], dtype="int64"),
        target_duration_minutes=0.05,  # 30 frames
        snap_direction="next",
    )

    assert [row["start_frame"] for row in boundaries] == [0, 31, 61, 91]
    assert [row["end_frame_exclusive"] for row in boundaries] == [31, 61, 91, 101]
    assert boundaries[1]["snap_delta_frames"] == 1
    assert boundaries[-1]["final_clip"] is True


def test_build_clip_plan_uses_metadata_rows_as_local_frame_map(tmp_path: Path) -> None:
    video = tmp_path / "Cam2010093_example.mp4"
    video.write_bytes(b"placeholder")
    metadata = tmp_path / "Cam2010093_example_meta.csv"
    keyframe = tmp_path / "Cam2010093_example_keyframe.json"

    with metadata.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame_id", "timestamp", "timestamp_sys"])
        writer.writeheader()
        for local_idx in range(100):
            writer.writerow(
                {
                    "frame_id": 1001 + local_idx,
                    "timestamp": 2001 + local_idx,
                    "timestamp_sys": 3001 + local_idx,
                }
            )
    keyframe.write_text(
        json.dumps(
            {
                "total_frames": 100,
                "fps": 10,
                "keyframe_frames": [0, 30, 60, 90],
            }
        ),
        encoding="utf-8",
    )

    plan = build_clip_plan(
        video_path=video,
        metadata_csv=metadata,
        keyframe_json=keyframe,
        target_duration_minutes=0.05,
        snap_direction="next",
        recording_id="rec_a",
    )

    assert plan["status"] == "ok"
    assert plan["camera_serial"] == "2010093"
    assert plan["clip_count"] == 4
    first = plan["clips"][0]
    assert first["clip_id"] == "clip_000000"
    assert first["actual_start_frame"] == 0
    assert first["end_frame_exclusive"] == 30
    assert first["first_recording_frame_id"] == 1001
    assert first["last_recording_frame_id"] == 1030
    assert first["first_clip_local_frame_index"] == 0
    assert first["last_clip_local_frame_index"] == 29
    second = plan["clips"][1]
    assert second["rollover_at_recording_frame_id"] == 1031
    assert second["first_timestamp"] == 2031


def test_build_clip_plan_reports_frame_id_gaps(tmp_path: Path) -> None:
    video = tmp_path / "Cam2010093_gap.mp4"
    video.write_bytes(b"placeholder")
    metadata = tmp_path / "Cam2010093_gap_meta.csv"
    keyframe = tmp_path / "Cam2010093_gap_keyframe.json"

    with metadata.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame_id", "timestamp", "timestamp_sys"])
        writer.writeheader()
        for frame_id in [1, 2, 4, 5]:
            writer.writerow({"frame_id": frame_id, "timestamp": frame_id, "timestamp_sys": frame_id})
    keyframe.write_text(
        json.dumps({"total_frames": 4, "fps": 2, "keyframe_frames": [0, 2]}),
        encoding="utf-8",
    )

    plan = build_clip_plan(
        video_path=video,
        metadata_csv=metadata,
        keyframe_json=keyframe,
        target_duration_minutes=1.0,
    )

    assert plan["status"] == "fail"
    gap_checks = [row for row in plan["checks"] if row["code"] == "recording_frame_id_continuity"]
    assert gap_checks[0]["recording_frame_id_gaps"] == 1


def test_write_plan_artifacts(tmp_path: Path) -> None:
    plan = {
        "status": "ok",
        "clips": [
            {
                "recording_id": "rec",
                "clip_id": "clip_000000",
                "clip_index": 0,
                "frame_count": 10,
            }
        ],
    }

    artifacts = write_plan_artifacts(plan, tmp_path / "out", prefix="recording_clip_index")

    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["csv"]).exists()
    rows = list(csv.DictReader(Path(artifacts["csv"]).open("r", encoding="utf-8")))
    assert rows[0]["clip_id"] == "clip_000000"
