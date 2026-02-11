from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import backfill_hevc_keyframe_flags as backfill


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_backfill_hevc_keyframe_flags_apply_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    recording_dir = tmp_path / "recording_1"
    (recording_dir / "raw").mkdir(parents=True)
    (recording_dir / "cams").mkdir(parents=True)
    (recording_dir / "raw" / "rec.mp4").write_bytes(b"raw")
    (recording_dir / "cams" / "cam.mp4").write_bytes(b"cams")

    manifest_path = recording_dir / "recording_manifest.json"
    _write_manifest(
        manifest_path,
        {
            "recording_name": "recording_1",
            "files": {
                "raw": ["raw/rec.mp4"],
                "cams": ["cams/cam.mp4"],
                "derived": [],
            },
        },
    )

    def fake_check(path: Path) -> dict:
        if path.name == "cam.mp4":
            return {
                "codec": "hevc",
                "has_stss": True,
                "needs_fix": False,
                "message": "ok",
            }
        return {
            "codec": "hevc",
            "has_stss": False,
            "needs_fix": True,
            "message": "missing stss",
        }

    monkeypatch.setattr(backfill, "check_hevc_keyframe_flags", fake_check)

    rc = backfill.main([str(recording_dir), "--apply"])
    assert rc == 0

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = updated.get("hevc_keyframe_flags")
    assert isinstance(checks, dict)
    assert sorted(checks.keys()) == ["cams/cam.mp4", "raw/rec.mp4"]
    assert checks["cams/cam.mp4"]["needs_fix"] is False
    assert checks["raw/rec.mp4"]["needs_fix"] is True
    assert isinstance(checks["cams/cam.mp4"]["file_size_bytes"], int)
    assert isinstance(checks["cams/cam.mp4"]["file_mtime_ns"], int)
    assert "checked_at_utc" in checks["cams/cam.mp4"]


def test_backfill_hevc_keyframe_flags_dry_run_does_not_write(tmp_path: Path, monkeypatch) -> None:
    recording_dir = tmp_path / "recording_2"
    (recording_dir / "cams").mkdir(parents=True)
    (recording_dir / "cams" / "cam.mp4").write_bytes(b"cams")

    manifest_path = recording_dir / "recording_manifest.json"
    _write_manifest(
        manifest_path,
        {
            "recording_name": "recording_2",
            "files": {"raw": [], "cams": ["cams/cam.mp4"], "derived": []},
        },
    )

    monkeypatch.setattr(
        backfill,
        "check_hevc_keyframe_flags",
        lambda _: {
            "codec": "hevc",
            "has_stss": True,
            "needs_fix": False,
            "message": "ok",
        },
    )

    rc = backfill.main([str(recording_dir)])
    assert rc == 0

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "hevc_keyframe_flags" not in payload


def test_backfill_hevc_keyframe_flags_overwrite_updates_existing_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recording_dir = tmp_path / "recording_3"
    (recording_dir / "cams").mkdir(parents=True)
    (recording_dir / "cams" / "cam.mp4").write_bytes(b"cams")

    manifest_path = recording_dir / "recording_manifest.json"
    _write_manifest(
        manifest_path,
        {
            "recording_name": "recording_3",
            "files": {"raw": [], "cams": ["cams/cam.mp4"], "derived": []},
            "hevc_keyframe_flags": {
                "cams/cam.mp4": {
                    "codec": "hevc",
                    "has_stss": False,
                    "needs_fix": True,
                    "message": "old",
                }
            },
        },
    )

    monkeypatch.setattr(
        backfill,
        "check_hevc_keyframe_flags",
        lambda _: {
            "codec": "hevc",
            "has_stss": True,
            "needs_fix": False,
            "message": "new",
        },
    )

    rc_skip = backfill.main([str(recording_dir), "--apply"])
    assert rc_skip == 0
    payload_skip = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload_skip["hevc_keyframe_flags"]["cams/cam.mp4"]["message"] == "old"

    rc_overwrite = backfill.main([str(recording_dir), "--apply", "--overwrite"])
    assert rc_overwrite == 0
    payload_overwrite = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload_overwrite["hevc_keyframe_flags"]["cams/cam.mp4"]["message"] == "new"
    assert payload_overwrite["hevc_keyframe_flags"]["cams/cam.mp4"]["needs_fix"] is False
