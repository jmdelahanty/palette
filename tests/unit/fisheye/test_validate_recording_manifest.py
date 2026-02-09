"""Unit tests for recording manifest validation utility."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils.validate_recording_manifest import main as validate_manifest_main


def test_validate_recording_manifest_apply_defaults_patches_missing_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "recording_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "session_uuid": "2026-01-01T00-00-00Z_arena_1",
                "files": {
                    "raw": [],
                    "cams": [],
                },
            }
        ),
        encoding="utf-8",
    )

    rc = validate_manifest_main([str(manifest_path), "--apply-defaults", "--no-rich"])
    assert rc == 0

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["recording_type"] == "behavior"
    assert payload["recording_subtype"] == "free"
    assert payload["behavior_mode"] == "free"
    assert payload["artifact_schema_id"] == "behavior_v1"


def test_validate_recording_manifest_fails_without_defaults_for_missing_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "recording_manifest.json"
    manifest_path.write_text(json.dumps({"session_uuid": "session_x"}), encoding="utf-8")

    rc = validate_manifest_main([str(manifest_path), "--no-rich"])
    assert rc == 2
