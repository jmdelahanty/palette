from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Sequence

from fisheye.utils.materialize_orange_style_clips import (
    MaterializeOptions,
    build_ffmpeg_stream_copy_command,
    materialize_clip_plan,
)
from fisheye.utils.plan_orange_style_clips import build_clip_plan


def _write_source_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    video = tmp_path / "source" / "cams" / "Cam2010093_example.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"source-video")
    metadata = video.with_name(f"{video.stem}_meta.csv")
    keyframe = video.with_name(f"{video.stem}_keyframe.json")
    with metadata.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame_id", "timestamp", "timestamp_sys"])
        writer.writeheader()
        for idx in range(12):
            writer.writerow(
                {
                    "frame_id": 101 + idx,
                    "timestamp": 1001 + idx,
                    "timestamp_sys": 2001 + idx,
                }
            )
    keyframe.write_text(
        json.dumps(
            {
                "codec": "hevc",
                "fps": 2,
                "total_frames": 12,
                "keyframe_frames": [0, 4, 8],
            }
        ),
        encoding="utf-8",
    )
    return video, metadata, keyframe


def test_build_ffmpeg_stream_copy_command_uses_frame_count_limit(tmp_path: Path) -> None:
    command = build_ffmpeg_stream_copy_command(
        ffmpeg_bin="ffmpeg",
        source_video=tmp_path / "in.mp4",
        output_video=tmp_path / "out.mp4",
        start_time_s=2.0,
        frame_count=4,
        overwrite=False,
    )

    assert command[:6] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    assert command[command.index("-ss") + 1] == "2"
    assert command[command.index("-frames:v") + 1] == "4"
    assert command[-1].endswith("out.mp4")


def test_materialize_clip_plan_dry_run_writes_nothing(tmp_path: Path) -> None:
    video, metadata, keyframe = _write_source_bundle(tmp_path)
    plan = build_clip_plan(
        video_path=video,
        metadata_csv=metadata,
        keyframe_json=keyframe,
        target_duration_minutes=2 / 60,
        snap_direction="next",
        recording_id="rec",
    )

    result = materialize_clip_plan(
        plan,
        options=MaterializeOptions(output_recording_dir=tmp_path / "source", apply=False),
    )

    assert result["status"] == "ok"
    assert result["apply"] is False
    assert result["clip_count"] == 3
    assert not (tmp_path / "source" / "clips").exists()


def test_materialize_clip_plan_writes_orange_style_sidecars(tmp_path: Path) -> None:
    video, metadata, keyframe = _write_source_bundle(tmp_path)
    plan = build_clip_plan(
        video_path=video,
        metadata_csv=metadata,
        keyframe_json=keyframe,
        target_duration_minutes=2 / 60,
        snap_direction="next",
        recording_id="rec",
    )
    calls: list[list[str]] = []

    def fake_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        Path(command[-1]).write_bytes(b"clip-video")
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")

    result = materialize_clip_plan(
        plan,
        options=MaterializeOptions(
            output_recording_dir=tmp_path / "source",
            apply=True,
            overwrite=False,
        ),
        runner=fake_runner,
    )

    assert result["status"] == "ok"
    assert result["apply"] is True
    assert len(calls) == 3

    root = tmp_path / "source"
    first_clip = root / "clips" / "clip_000000"
    assert (first_clip / "Cam2010093_example.mp4").read_bytes() == b"clip-video"
    assert (first_clip / "clip_manifest.json").exists()

    metadata_rows = list(
        csv.DictReader((first_clip / "Cam2010093_example_meta.csv").open("r", encoding="utf-8"))
    )
    assert [row["frame_id"] for row in metadata_rows] == ["101", "102", "103", "104"]

    first_keyframe = json.loads((first_clip / "Cam2010093_example_keyframe.json").read_text())
    assert first_keyframe["total_frames"] == 4
    assert first_keyframe["keyframe_frames"] == [0]
    assert first_keyframe["palette_retro_clip"]["source_start_frame"] == 0

    second_keyframe = json.loads(
        (root / "clips" / "clip_000001" / "Cam2010093_example_keyframe.json").read_text()
    )
    assert second_keyframe["total_frames"] == 4
    assert second_keyframe["keyframe_frames"] == [0]

    index = json.loads((root / "recording_clip_index.json").read_text())
    assert index["mode"] == "materialized_stream_copy"
    assert index["clip_count"] == 3
    assert index["clips"][0]["status"] == "materialized"

    csv_rows = list(csv.DictReader((root / "recording_clip_index.csv").open("r", encoding="utf-8")))
    assert len(csv_rows) == 3
    assert csv_rows[0]["clip_id"] == "clip_000000"


def test_materialize_clip_plan_refuses_existing_index_without_overwrite(tmp_path: Path) -> None:
    video, metadata, keyframe = _write_source_bundle(tmp_path)
    plan = build_clip_plan(
        video_path=video,
        metadata_csv=metadata,
        keyframe_json=keyframe,
        target_duration_minutes=2 / 60,
        snap_direction="next",
        recording_id="rec",
    )
    (tmp_path / "source" / "recording_clip_index.json").write_text("{}", encoding="utf-8")

    def fake_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError("ffmpeg runner should not be called after preflight failure")

    try:
        materialize_clip_plan(
            plan,
            options=MaterializeOptions(output_recording_dir=tmp_path / "source", apply=True),
            runner=fake_runner,
        )
    except FileExistsError as exc:
        assert "recording_clip_index.json" in str(exc)
    else:
        raise AssertionError("Expected FileExistsError")
