from pathlib import Path
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import organize_recordings


def test_build_video_only_plan_defaults_and_renames_cam_file(tmp_path: Path) -> None:
    video_path = tmp_path / "Cam2010093.mp4"
    video_path.write_bytes(b"video")

    plan = organize_recordings._build_video_only_plan(
        {
            "source_video": str(video_path),
            "session_uuid": "2026-03-09_colleague_set_001",
            "recording_name": "Colleague Set 001",
            "dish_design": "cedar",
        },
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )

    assert plan.name == "Colleague_Set_001"
    assert plan.camera_id == "2010093"
    assert plan.meta["recording_type"] == "behavior"
    assert plan.meta["recording_subtype"] == "free"
    assert plan.meta["behavior_mode"] == "free"
    assert plan.meta["artifact_schema_id"] == "video_only_v1"
    assert plan.cam_files[0].dest_name == "Cam2010093_2026-03-09_colleague_set_001.mp4"


def test_main_video_only_apply_moves_file_and_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "staging"
    source_root.mkdir()
    video_path = source_root / "Cam2010093.mp4"
    video_path.write_bytes(b"video")

    metadata_csv = tmp_path / "video_only_metadata.csv"
    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_video",
                "session_uuid",
                "recording_name",
                "dish_design",
                "rig_id",
                "arena_id",
                "camera_id",
                "protocol_name",
                "num_dishes",
                "fish_per_dish",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_video": "Cam2010093.mp4",
                "session_uuid": "2026-03-09_colleague_set_001",
                "recording_name": "Colleague Set 001",
                "dish_design": "cedar",
                "rig_id": "omnifin0",
                "arena_id": "arena_1",
                "camera_id": "2010093",
                "protocol_name": "ManualProtocol",
                "num_dishes": "1",
                "fish_per_dish": "1",
            }
        )

    monkeypatch.setattr(
        organize_recordings,
        "check_hevc_keyframe_flags",
        lambda _: {
            "codec": "hevc",
            "has_stss": True,
            "needs_fix": False,
            "message": "ok",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "organize_recordings.py",
            str(source_root),
            "--video-only",
            "--metadata-csv",
            str(metadata_csv),
            "--dest-root",
            str(tmp_path / "recordings"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--apply",
            "--write-manifest",
        ],
    )

    rc = organize_recordings.main()
    assert rc == 0

    dest_dir = tmp_path / "recordings" / "Colleague_Set_001"
    moved_video = dest_dir / "cams" / "Cam2010093_2026-03-09_colleague_set_001.mp4"
    assert moved_video.exists()
    assert not video_path.exists()

    manifest_path = dest_dir / "recording_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["artifact_schema_id"] == "video_only_v1"
    assert payload["recording_type"] == "behavior"
    assert payload["recording_subtype"] == "free"
    assert payload["behavior_mode"] == "free"
    assert payload["dish_design"] == "cedar"
    assert payload["protocol_name_from_definition"] == "ManualProtocol"
    assert payload["num_dishes"] == "1"
    assert payload["fish_per_dish"] == "1"
    assert payload["files"]["cams"] == ["cams/Cam2010093_2026-03-09_colleague_set_001.mp4"]
