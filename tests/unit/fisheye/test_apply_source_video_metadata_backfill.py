from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

from fisheye.shared.source_video_metadata import (
    SOURCE_VIDEO_METADATA_SCHEMA_ID,
    build_source_video_metadata_v2,
)
from fisheye.utils import apply_source_video_metadata_backfill as apply_mod
from fisheye.utils.preflight_source_video_metadata_backfill import (
    build_preflight_report,
    select_registry_datasets,
)


def _write_v3_attrs(group_path: Path, attrs: dict[str, object]) -> None:
    group_path.mkdir(parents=True, exist_ok=True)
    (group_path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attrs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _create_registry(registry: Path, *, recording: Path, zarr_path: Path) -> None:
    conn = sqlite3.connect(registry)
    try:
        conn.executescript(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                recording_id TEXT,
                zarr_path TEXT NOT NULL,
                status TEXT,
                zarr_use TEXT,
                artifact_kind TEXT,
                source_layout TEXT
            );
            CREATE TABLE recordings (
                recording_id TEXT PRIMARY KEY,
                recording_path TEXT,
                recording_name TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO recordings VALUES (?, ?, ?)",
            (recording.name, str(recording), recording.name),
        )
        conn.execute(
            "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "dataset-1",
                recording.name,
                str(zarr_path),
                "active",
                "analysis",
                "source_recording",
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _build_fixture(tmp_path: Path, *, versioned: bool = False) -> dict[str, Path | str]:
    recording = tmp_path / "recording_GoodCopBadCop"
    video = recording / "cams" / "source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    zarr_path = recording / "zarr" / "recording_analysis.zarr"
    metadata: dict[str, object] = {
        "source_path": str(video),
        "source_video": video.name,
        "width": 4512,
        "height": 4512,
        "fps": 100.0,
        "total_frames": 100,
    }
    if versioned:
        metadata = build_source_video_metadata_v2(
            metadata,
            recording_path=recording,
        )
    _write_v3_attrs(
        zarr_path,
        {
            "recording_path": str(recording),
            "source_video": video.name,
            "source_video_path": str(video),
            "source_path": str(video),
            "source_video_metadata": metadata,
            "width": 4512,
            "height": 4512,
            "fps": 100.0,
            "total_frames": 100,
        },
    )
    _write_v3_attrs(
        zarr_path / "raw_video",
        {"source_video": video.name, "source_path": str(video)},
    )
    registry = tmp_path / "registry.sqlite"
    _create_registry(registry, recording=recording, zarr_path=zarr_path)
    datasets = select_registry_datasets(registry, path_contains="GoodCopBadCop")
    report = build_preflight_report(
        datasets,
        registry_path=registry,
        path_contains="GoodCopBadCop",
        expected_count=1,
        required_storage_root=tmp_path,
    )
    report_path = tmp_path / "preflight.json"
    report_bytes = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    report_path.write_bytes(report_bytes)
    return {
        "zarr_path": zarr_path,
        "root_metadata": zarr_path / "zarr.json",
        "raw_metadata": zarr_path / "raw_video" / "zarr.json",
        "report": report_path,
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "backup": tmp_path / "backup",
        "receipt": tmp_path / "receipt.json",
    }


def _apply(fixture: dict[str, Path | str]) -> dict[str, object]:
    return apply_mod.apply_reviewed_report(
        Path(fixture["report"]),
        expected_sha256=str(fixture["report_sha256"]),
        backup_dir=Path(fixture["backup"]),
        receipt_path=Path(fixture["receipt"]),
    )


def test_apply_backfills_v2_metadata_and_validates(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)

    receipt = _apply(fixture)

    root_payload = json.loads(Path(fixture["root_metadata"]).read_text(encoding="utf-8"))
    metadata = root_payload["attributes"]["source_video_metadata"]
    assert metadata["schema_id"] == SOURCE_VIDEO_METADATA_SCHEMA_ID
    assert metadata["locator"] == {
        "kind": "recording_relative",
        "relative_path": "cams/source.mp4",
    }
    assert receipt["status"] == "complete"
    assert receipt["target_count"] == 1
    assert receipt["metadata_file_count_backed_up"] == 2
    assert receipt["metadata_file_count_changed"] == 1
    assert Path(fixture["receipt"]).is_file()
    assert (Path(fixture["backup"]) / "backup_manifest.json").is_file()


def test_apply_refuses_metadata_precondition_drift_before_backup(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    root_metadata = Path(fixture["root_metadata"])
    stat = root_metadata.stat()
    os.utime(root_metadata, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    with pytest.raises(apply_mod.SourceVideoMetadataApplyError, match="Precondition drift"):
        _apply(fixture)

    assert not Path(fixture["backup"]).exists()
    payload = json.loads(root_metadata.read_text(encoding="utf-8"))
    assert "schema_id" not in payload["attributes"]["source_video_metadata"]


def test_apply_rolls_back_all_changed_metadata_on_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    root_metadata = Path(fixture["root_metadata"])
    original = root_metadata.read_bytes()

    def _fail_validation(_target):
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(apply_mod, "_validate_applied_target", _fail_validation)

    with pytest.raises(apply_mod.SourceVideoMetadataApplyError, match="rolled back"):
        _apply(fixture)

    assert root_metadata.read_bytes() == original
    receipt = json.loads(Path(fixture["receipt"]).read_text(encoding="utf-8"))
    assert receipt["status"] == "rolled_back"
    assert receipt["restored_files"] == [str(root_metadata)]


def test_apply_is_noop_for_reviewed_already_v2_archive(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, versioned=True)

    receipt = _apply(fixture)

    assert receipt["status"] == "complete"
    assert receipt["metadata_file_count_changed"] == 0
    assert receipt["validations"][0]["schema_id"] == SOURCE_VIDEO_METADATA_SCHEMA_ID


def test_apply_refuses_wrong_report_hash(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)

    with pytest.raises(apply_mod.SourceVideoMetadataApplyError, match="SHA-256 mismatch"):
        apply_mod.apply_reviewed_report(
            Path(fixture["report"]),
            expected_sha256="0" * 64,
            backup_dir=Path(fixture["backup"]),
            receipt_path=Path(fixture["receipt"]),
        )

    assert not Path(fixture["backup"]).exists()
