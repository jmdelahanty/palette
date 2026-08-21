import json
import os
from pathlib import Path

import pytest

from fisheye.utils import consolidate_external_ipc_rolling_recordings as consolidate

SESSION_ID = "2026_08_06_19_13_35"
CAMERAS = ("2010093", "2010094")
CLIPS = ("clip_000000", "clip_000001")


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _write_json(path: Path, value: object) -> Path:
    return _write(path, json.dumps(value))


def _full_clock(first: int, last: int) -> str:
    return (
        "recording_frame_id,timestamp,gop_index,frame_index_within_gop,bytes\n"
        + "".join(
            f"{frame},{frame * 100},0,{frame - first},1000\n"
            for frame in range(first, last + 1)
        )
    )


def _crop_clock(first: int, last: int) -> str:
    return "recording_frame_id,timestamp,crop_video_frame_index,crop_state\n" + "".join(
        f"{frame},{frame * 100},{frame - first},detected_crop\n"
        for frame in range(first, last + 1)
    )


def _make_accidental_import(tmp_path: Path) -> tuple[Path, Path]:
    recordings_root = tmp_path / "recordings"
    staging_dir = tmp_path / "staging" / SESSION_ID
    recordings_root.mkdir(parents=True)
    staging_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []

    for clip_index, clip_id in enumerate(CLIPS):
        first = clip_index * 2 + 1
        last = first + 1
        _write_json(
            staging_dir
            / "external_recorder"
            / "clips"
            / clip_id
            / "clip_manifest.json",
            {"clip_id": clip_id, "source": "orange"},
        )
        _write(
            staging_dir / "external_recorder" / "clips" / clip_id / "clip_status.txt",
            "completed\n",
        )
        for camera in CAMERAS:
            recording_dir = recordings_root / f"{SESSION_ID}_{clip_id}_Cam{camera}"
            base = f"Cam{camera}_{SESSION_ID}_{clip_id}"
            full_video = _write(recording_dir / "cams" / f"{base}.mp4", "full-video")
            # Reproduce the historical organizer bug: this is frame-clock CSV
            # content carrying a summary-shaped JSON filename.
            full_metadata = _write(
                recording_dir / "cams" / f"{base}_external_summary.json",
                _full_clock(first, last),
            )
            full_keyframes = _write_json(
                recording_dir / "cams" / f"{base}_keyframe.json",
                {"total_frames": 2, "keyframe_frames": [0]},
            )
            crop_video = _write(
                recording_dir
                / "derived"
                / "external_crop_recorder"
                / f"{base}_crop.mp4",
                "crop-video",
            )
            crop_metadata = _write(
                recording_dir
                / "derived"
                / "external_crop_recorder"
                / f"{base}_crop_meta.csv",
                _crop_clock(first, last),
            )
            crop_keyframes = _write_json(
                recording_dir
                / "derived"
                / "external_crop_recorder"
                / f"{base}_crop_keyframe.json",
                {"total_frames": 2, "keyframe_frames": [0]},
            )
            _write_json(
                recording_dir / "raw" / "recording_session.json",
                {
                    "session_id": SESSION_ID,
                    "created_at_utc": "2026-08-06T23:13:35Z",
                    "producer": "orange_gui_external_ipc",
                    "recording": {"started_at_utc": "2026-08-06T23:13:35Z"},
                },
            )
            _write(recording_dir / "raw" / "session.events.jsonl", "{}\n")
            manifest = {
                "recording_name": recording_dir.name,
                "orange_session_id": SESSION_ID,
                "camera_id": camera,
                "video_streams": {
                    "streams": {
                        "full": {
                            "video": str(full_video.relative_to(recording_dir)),
                            "frame_clock_metadata": str(
                                crop_metadata.relative_to(recording_dir)
                            ),
                            "summary": str(full_metadata.relative_to(recording_dir)),
                            "keyframes": str(full_keyframes.relative_to(recording_dir)),
                        },
                        "crop": {
                            "video": str(crop_video.relative_to(recording_dir)),
                            "metadata": str(crop_metadata.relative_to(recording_dir)),
                            "keyframes": str(crop_keyframes.relative_to(recording_dir)),
                        },
                    }
                },
            }
            _write_json(recording_dir / "recording_manifest.json", manifest)
            rows.append(
                {
                    "session_id": SESSION_ID,
                    "clip_id": clip_id,
                    "clip_index": clip_index,
                    "camera_serial": camera,
                    "status": "completed",
                    "frame_count": 2,
                    "first_recording_frame_id": first,
                    "last_recording_frame_id": last,
                    "recording_frame_id_gaps": 0,
                    "actual_duration_s": 2 / 30,
                    "video": f"/stale/orange/{clip_id}/Cam{camera}.mp4",
                    "metadata": f"/stale/orange/{clip_id}/Cam{camera}_meta.csv",
                    "keyframes": f"/stale/orange/{clip_id}/Cam{camera}_keyframes.json",
                }
            )

    _write_json(
        staging_dir / consolidate.INDEX_NAME,
        {
            "session_id": SESSION_ID,
            "columns": list(rows[0]),
            "rows": rows,
        },
    )
    _write(staging_dir / consolidate.INDEX_CSV_NAME, "original,index\n")
    _write_json(staging_dir / "recording_session.json", {"session_id": SESSION_ID})
    return recordings_root, staging_dir


def test_build_plan_dry_run_recovers_complete_clip_camera_grid(tmp_path: Path) -> None:
    recordings_root, staging_dir = _make_accidental_import(tmp_path)

    plan = consolidate.build_plan(
        recordings_root=recordings_root,
        staging_dir=staging_dir,
        session_id=SESSION_ID,
        camera_serial=CAMERAS[0],
    )
    summary = consolidate.plan_summary(plan)

    assert summary["status"] == "ready"
    assert summary["recording_id"] == f"{SESSION_ID}_cam{CAMERAS[0]}"
    assert summary["source_recording_count"] == 2
    assert summary["clip_count"] == 2
    assert summary["camera_count"] == 1
    assert summary["index_row_count"] == 2
    assert summary["source_directories_will_be_deleted"] == 0
    assert not plan.destination.exists()
    metadata = {
        artifact.source
        for artifact in plan.artifacts
        if artifact.role == "authoritative_full_metadata"
    }
    assert all(path.suffix == ".json" for path in metadata)


def test_apply_publishes_parent_with_hardlinks_and_preserves_sources(
    tmp_path: Path,
) -> None:
    recordings_root, staging_dir = _make_accidental_import(tmp_path)
    plan = consolidate.build_plan(
        recordings_root=recordings_root,
        staging_dir=staging_dir,
        session_id=SESSION_ID,
        camera_serial=CAMERAS[0],
    )
    full_video_artifact = next(
        artifact
        for artifact in plan.artifacts
        if artifact.role == "authoritative_full_video"
    )

    receipt = consolidate.apply_plan(plan, scan_metadata=True)

    assert receipt["status"] == "published"
    assert receipt["recording_id"] == f"{SESSION_ID}_cam{CAMERAS[0]}"
    assert receipt["source_recording_count"] == 2
    assert receipt["source_directories_deleted"] == 0
    assert receipt["validation"]["status"] == "pass"
    assert receipt["validation"]["metadata_scans"] == 2
    assert receipt["validation"]["crop_metadata_scans"] == 2
    assert plan.destination.is_dir()
    assert not plan.work_dir.exists()
    assert all(split.root.is_dir() for split in plan.split_recordings.values())

    linked_video = plan.destination / full_video_artifact.relative_path
    source_stat = os.stat(full_video_artifact.source)
    linked_stat = os.stat(linked_video)
    assert (linked_stat.st_dev, linked_stat.st_ino) == (
        source_stat.st_dev,
        source_stat.st_ino,
    )

    index = json.loads(
        (plan.destination / consolidate.INDEX_NAME).read_text(encoding="utf-8")
    )
    assert index["recording_id"] == f"{SESSION_ID}_cam{CAMERAS[0]}"
    assert index["row_count"] == 2
    assert index["clip_count"] == 2
    assert {row["camera_serial"] for row in index["rows"]} == {CAMERAS[0]}
    assert all(not Path(row["video_path"]).is_absolute() for row in index["rows"])
    manifest = json.loads(
        (plan.destination / "recording_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_schema_id"] == consolidate.ARTIFACT_SCHEMA_ID
    assert manifest["camera_id"] == CAMERAS[0]
    assert manifest["rolling_clip_streams"]["clip_count"] == 2

    destination_inodes = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in plan.destination.rglob("*")
        if path.is_file()
    }
    for split in plan.split_recordings.values():
        for source in split.root.rglob("*"):
            if source.is_file():
                stat = source.stat()
                assert (stat.st_dev, stat.st_ino) in destination_inodes


def test_build_plan_fails_closed_on_missing_split_recording(tmp_path: Path) -> None:
    recordings_root, staging_dir = _make_accidental_import(tmp_path)
    missing = recordings_root / f"{SESSION_ID}_{CLIPS[-1]}_Cam{CAMERAS[-1]}"
    for path in sorted(missing.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    missing.rmdir()

    with pytest.raises(FileNotFoundError, match="Missing accidental split recording"):
        consolidate.build_plan(
            recordings_root=recordings_root,
            staging_dir=staging_dir,
            session_id=SESSION_ID,
            camera_serial=CAMERAS[-1],
        )


def test_build_plan_rejects_crop_metadata_as_full_clock(tmp_path: Path) -> None:
    recordings_root, staging_dir = _make_accidental_import(tmp_path)
    split = recordings_root / f"{SESSION_ID}_{CLIPS[0]}_Cam{CAMERAS[0]}"
    manifest_path = split / "recording_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["video_streams"]["streams"]["full"].pop("summary")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="recover full-frame metadata"):
        consolidate.build_plan(
            recordings_root=recordings_root,
            staging_dir=staging_dir,
            session_id=SESSION_ID,
            camera_serial=CAMERAS[0],
        )
