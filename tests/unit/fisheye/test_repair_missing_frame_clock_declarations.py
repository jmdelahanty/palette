from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from fisheye.utils import repair_missing_frame_clock_declarations as repair


def _write_manifest(recording_dir: Path, *, declaration: str) -> Path:
    recording_dir.mkdir(parents=True)
    manifest_path = recording_dir / "recording_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "recording_name": recording_dir.name,
                "video_streams": {
                    "streams": {
                        "full": {
                            "video": "cams/Cam1.mp4",
                            "frame_clock_metadata": declaration,
                        }
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_dry_run_reports_missing_declaration_without_mutation(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    manifest_path = _write_manifest(
        recording_dir,
        declaration="cams/Cam1_meta.csv",
    )
    before = manifest_path.read_bytes()

    result = repair.repair_recording_manifest(
        recording_dir,
        apply=False,
        repair_id="repair_test",
        reason="test",
    )

    assert result["status"] == "repair_planned"
    assert result["repaired_stream_count"] == 1
    assert result["streams"][0]["fallback"] == {
        "kind": "none",
        "acquisition_frame_clock_status": "unavailable_no_camera_clock_source",
    }
    assert manifest_path.read_bytes() == before


def test_apply_removes_only_missing_pointer_and_records_audit(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    manifest_path = _write_manifest(
        recording_dir,
        declaration="cams/missing_meta.csv",
    )
    before_digest = sha256(manifest_path.read_bytes()).hexdigest()

    result = repair.repair_recording_manifest(
        recording_dir,
        apply=True,
        repair_id="repair_test",
        reason="transferred without CSV",
    )

    assert result["status"] == "repaired"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    full = manifest["video_streams"]["streams"]["full"]
    assert "frame_clock_metadata" not in full
    audit = manifest["metadata_repairs"][0]
    assert audit["repair_type"] == repair.REPAIR_TYPE
    assert audit["repair_id"] == "repair_test"
    assert audit["manifest_sha256_before"] == before_digest
    assert audit["streams"][0]["removed_frame_clock_metadata"] == "cams/missing_meta.csv"


def test_existing_declared_csv_is_unchanged(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    manifest_path = _write_manifest(
        recording_dir,
        declaration="cams/Cam1_meta.csv",
    )
    csv_path = recording_dir / "cams" / "Cam1_meta.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(
        "recording_frame_id,timestamp,timestamp_sys\n1,10,20\n",
        encoding="utf-8",
    )
    before = manifest_path.read_bytes()

    result = repair.repair_recording_manifest(
        recording_dir,
        apply=True,
        repair_id="repair_test",
        reason="test",
    )

    assert result["status"] == "unchanged_no_missing_declarations"
    assert manifest_path.read_bytes() == before


def test_removal_exposes_real_conventional_csv_fallback(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    _write_manifest(recording_dir, declaration="cams/stale.csv")
    conventional = recording_dir / "cams" / "Cam1_meta.csv"
    conventional.parent.mkdir(parents=True)
    conventional.write_text(
        "recording_frame_id,timestamp,timestamp_sys\n1,10,20\n",
        encoding="utf-8",
    )

    result = repair.repair_recording_manifest(
        recording_dir,
        apply=False,
        repair_id="repair_test",
        reason="test",
    )

    assert result["streams"][0]["fallback"] == {
        "kind": "conventional_camera_metadata_csv",
        "path": "cams/Cam1_meta.csv",
    }
