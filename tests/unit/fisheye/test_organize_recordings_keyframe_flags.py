from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import organize_recordings


def test_apply_plan_records_keyframe_flags_in_manifest(tmp_path: Path, monkeypatch) -> None:
    src_mp4 = tmp_path / "Cam1.mp4"
    src_mp4.write_bytes(b"abc")

    plan = organize_recordings.RecordingPlan(
        name="recording_001",
        source_dir=tmp_path,
        dest_dir=tmp_path / "organized" / "recording_001",
        raw_files=[],
        cam_files=[organize_recordings.PlannedFile(source=src_mp4, dest_name="Cam1.mp4")],
        derived_files=[],
        camera_id="1",
        meta={"session_uuid": "session_1"},
    )

    monkeypatch.setattr(
        organize_recordings,
        "check_hevc_keyframe_flags",
        lambda _: {
            "codec": "hevc",
            "has_stss": True,
            "needs_fix": False,
            "message": "HEVC stream has sync sample table (stss) in moov.",
        },
    )

    warnings = organize_recordings._apply_plan(
        [plan],
        create_empty=False,
        write_manifest=True,
        snapshot=None,
        snapshot_mode="split",
        logger=None,
        run_id="run_1",
        log_path=None,
    )

    assert warnings == []

    manifest_path = plan.dest_dir / "recording_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = payload.get("hevc_keyframe_flags")
    assert isinstance(checks, dict)
    assert "cams/Cam1.mp4" in checks

    check_payload = checks["cams/Cam1.mp4"]
    assert check_payload["codec"] == "hevc"
    assert check_payload["has_stss"] is True
    assert check_payload["needs_fix"] is False
    assert check_payload["file_size_bytes"] == 3
    assert isinstance(check_payload["file_mtime_ns"], int)
    assert "checked_at_utc" in check_payload


def test_apply_plan_warns_when_hevc_keyframe_flags_need_fix(tmp_path: Path, monkeypatch) -> None:
    src_mp4 = tmp_path / "Cam2.mp4"
    src_mp4.write_bytes(b"xyz")

    plan = organize_recordings.RecordingPlan(
        name="recording_002",
        source_dir=tmp_path,
        dest_dir=tmp_path / "organized" / "recording_002",
        raw_files=[],
        cam_files=[organize_recordings.PlannedFile(source=src_mp4, dest_name="Cam2.mp4")],
        derived_files=[],
        camera_id="2",
        meta={"session_uuid": "session_2"},
    )

    monkeypatch.setattr(
        organize_recordings,
        "check_hevc_keyframe_flags",
        lambda _: {
            "codec": "hevc",
            "has_stss": False,
            "needs_fix": True,
            "message": "HEVC stream is missing sync sample table (stss); seeking may be unreliable.",
        },
    )

    warnings = organize_recordings._apply_plan(
        [plan],
        create_empty=False,
        write_manifest=False,
        snapshot=None,
        snapshot_mode="split",
        logger=None,
        run_id="run_2",
        log_path=None,
    )

    assert len(warnings) == 1
    assert "HEVC keyframe flags issue" in warnings[0]
    assert "stss" in warnings[0]

    check_payload = plan.keyframe_checks.get("cams/Cam2.mp4")
    assert isinstance(check_payload, dict)
    assert check_payload["needs_fix"] is True
