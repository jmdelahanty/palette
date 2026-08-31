from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import organize_recordings
from fisheye.diagnostics.video import container
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
)


def _identity(recording_id: str, session_uuid: str, camera_id: str) -> dict[str, str]:
    return {
        SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: SOURCE_RECORDING_IDENTITY_PROFILE,
        "recording_id": recording_id,
        "session_uuid": session_uuid,
        "camera_id": camera_id,
    }


def _atom(atom_type: bytes, payload: bytes = b"") -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + atom_type + payload


def _all_sync_mp4() -> bytes:
    hdlr = _atom(b"hdlr", b"\x00" * 8 + b"vide" + b"\x00" * 12)
    stbl = _atom(b"stbl", _atom(b"stts"))
    mdia = _atom(b"mdia", hdlr + _atom(b"minf", stbl))
    return _atom(b"ftyp", b"isom0000") + _atom(b"moov", _atom(b"trak", mdia))


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
        meta=_identity("recording_001", "session_1", "1"),
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


def test_apply_plan_warns_on_orange_sync_evidence_contradiction(
    tmp_path: Path, monkeypatch
) -> None:
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
        meta=_identity("recording_002", "session_2", "2"),
    )

    monkeypatch.setattr(
        organize_recordings,
        "check_hevc_keyframe_flags",
        lambda _: {
            "codec": "hevc",
            "has_stss": False,
            "sync_sample_semantics": "all_samples_sync",
            "sync_sample_proof": "orange_idr_sidecar_contradiction",
            "needs_fix": True,
            "message": "Orange evidence is contradictory.",
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
    assert "Orange sync-sample evidence contradiction" in warnings[0]
    assert "re-encod" not in warnings[0].lower()

    check_payload = plan.keyframe_checks.get("cams/Cam2.mp4")
    assert isinstance(check_payload, dict)
    assert check_payload["needs_fix"] is True


def test_apply_plan_does_not_warn_for_hevc_all_samples_sync(
    tmp_path: Path, monkeypatch
) -> None:
    src_mp4 = tmp_path / "Cam3.mp4"
    src_mp4.write_bytes(b"all-i")
    plan = organize_recordings.RecordingPlan(
        name="recording_003",
        source_dir=tmp_path,
        dest_dir=tmp_path / "organized" / "recording_003",
        raw_files=[],
        cam_files=[organize_recordings.PlannedFile(source=src_mp4, dest_name="Cam3.mp4")],
        derived_files=[],
        camera_id="3",
        meta=_identity("recording_003", "session_3", "3"),
    )
    monkeypatch.setattr(
        organize_recordings,
        "check_hevc_keyframe_flags",
        lambda _: {
            "codec": "hevc",
            "has_stss": False,
            "sync_sample_semantics": "all_samples_sync",
            "sync_sample_proof": "container_declared",
            "needs_fix": False,
            "message": "ISO BMFF declares every sample to be a sync sample.",
        },
    )

    warnings = organize_recordings._apply_plan(
        [plan],
        create_empty=False,
        write_manifest=False,
        snapshot=None,
        snapshot_mode="split",
        logger=None,
        run_id="run_3",
        log_path=None,
    )

    assert warnings == []
    assert plan.keyframe_checks["cams/Cam3.mp4"]["needs_fix"] is False


def test_apply_plan_verifies_orange_gop1_crop_without_reencoding(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    video = source / "Cam4_crop_external.mp4"
    summary = source / "Cam4_crop_external_summary.json"
    keyframes = source / "Cam4_crop_external_keyframe.json"
    original_video_bytes = _all_sync_mp4()
    video.write_bytes(original_video_bytes)
    summary.write_text(
        json.dumps(
            {
                "output_kind": "crop",
                "stream_kind": "crop",
                "tuning": "lossless",
                "resolved_gop_length": 1,
                "frames_encoded": 4,
            }
        ),
        encoding="utf-8",
    )
    keyframes.write_text(
        json.dumps({"total_frames": 4, "keyframe_frames": [0, 1, 2, 3]}),
        encoding="utf-8",
    )
    base = "external_crop_recorder/Cam4_crop_external"
    meta = _identity("recording_004", "session_4", "4")
    meta["video_streams"] = {
        "schema_id": "orange_runtime_video_streams_v1",
        "streams": {
            "crop": {
                "source": "orange_external_ipc",
                "output_kind": "crop",
                "stream_kind": "crop",
                "tuning": "lossless",
                "frame_count": 4,
                "packet_count": 4,
                "video": f"derived/{base}.mp4",
                "summary": f"derived/{base}_summary.json",
                "keyframes": f"derived/{base}_keyframe.json",
            }
        },
    }
    plan = organize_recordings.RecordingPlan(
        name="recording_004",
        source_dir=source,
        dest_dir=tmp_path / "organized" / "recording_004",
        raw_files=[],
        cam_files=[],
        derived_files=[
            organize_recordings.PlannedFile(video, f"{base}.mp4"),
            organize_recordings.PlannedFile(summary, f"{base}_summary.json"),
            organize_recordings.PlannedFile(keyframes, f"{base}_keyframe.json"),
        ],
        camera_id="4",
        meta=meta,
    )
    monkeypatch.setattr(container, "_probe_codec_name", lambda _: "hevc")

    warnings = organize_recordings._apply_plan(
        [plan],
        create_empty=False,
        write_manifest=False,
        snapshot=None,
        snapshot_mode="split",
        logger=None,
        run_id="run_4",
        log_path=None,
    )

    check = plan.keyframe_checks[f"derived/{base}.mp4"]
    assert warnings == []
    assert check["sync_sample_semantics"] == "all_samples_sync"
    assert check["sync_sample_proof"] == "orange_idr_sidecar_verified"
    assert check["needs_fix"] is False
    assert (plan.dest_dir / f"derived/{base}.mp4").read_bytes() == original_video_bytes
