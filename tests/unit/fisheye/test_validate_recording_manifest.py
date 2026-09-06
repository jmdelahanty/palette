"""Unit tests for recording manifest validation utility."""

from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils.validate_recording_manifest import main as validate_manifest_main
from fisheye.shared.recording_manifest_context import (
    recording_manifest_context_issues,
    validate_recording_manifest_context,
)


@pytest.mark.parametrize("recording_type,subtype,mode", [
    ("behavior", "free", "free"), ("behavior", "embedded", "embedded"),
    ("microscopy", "lightsheet", "none"), ("microscopy", "confocal", "none"),
    ("microscopy", "2p", "none"), ("histology", "section", "none"),
    ("histology", "wholemount", "none"),
])
def test_manifest_context_preserves_product_vocabulary_without_mutation(
    tmp_path: Path, recording_type: str, subtype: str, mode: str,
) -> None:
    payload = {
        "recording_type": recording_type, "recording_subtype": subtype,
        "behavior_mode": mode, "artifact_schema_id": "producer_schema_v1",
    }
    before = json.dumps(payload, sort_keys=True)
    validate_recording_manifest_context(payload)
    assert json.dumps(payload, sort_keys=True) == before
    manifest_path = tmp_path / "recording_manifest.json"
    manifest_path.write_text(before)
    assert validate_manifest_main([str(manifest_path), "--no-rich"]) == 0
    assert manifest_path.read_text() == before


def test_manifest_context_retains_explicit_registry_vocabulary() -> None:
    payload = {
        "recording_type": "custom", "recording_subtype": "custom_subtype",
        "behavior_mode": "none", "artifact_schema_id": "custom_v1",
    }
    assert recording_manifest_context_issues(
        payload, allowed_types={"custom"}, allowed_subtypes={"custom": {"custom_subtype"}},
    ) == []


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
