"""Registry migration 078: derived recording orientation status (palette#377 §10)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fisheye.registry.db import Registry
from fisheye.registry.extractors.orientation_status import (
    extract_recording_orientation_status,
)
from fisheye.registry.migrations import ADDITIVE_MIGRATIONS, MIGRATION_METHODS
from fisheye.shared.unified_h5 import orientation_status as resolver

from tests.unit.fisheye.test_unified_h5_declared_orientation import _omnifin0
from tests.unit.fisheye.test_unified_h5_orientation_status import (  # noqa: F401
    SESSION,
    _write,
    listed,
)

POST_CUTOFF = "2026-11-16T10-00-00Z"
H5_RELATIVE = "raw/acquisition/citrus/citsess_x.h5"
ORANGE_SESSION = "2026_11_16_10_00_00"
CAMERA = "2010093"


def _recording(tmp_path: Path, *, h5_relative: str | None = H5_RELATIVE) -> Path:
    recording = tmp_path / "2026_11_16_10_00_00_Cam2010093"
    (recording / "raw/acquisition/citrus").mkdir(parents=True)
    manifest = {"recording_id": "rec-1", "session_uuid": ORANGE_SESSION}
    if h5_relative is not None:
        manifest["h5_relative_path"] = h5_relative
    (recording / "recording_manifest.json").write_text(json.dumps(manifest))
    return recording


def _registry(tmp_path: Path) -> Registry:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.conn.execute(
        "INSERT INTO recordings (recording_id, session_uuid) VALUES ('rec-1', ?);",
        (ORANGE_SESSION,),
    )
    registry.conn.commit()
    return registry


def _rows(registry: Registry) -> list[sqlite3.Row]:
    return registry.conn.execute(
        "SELECT * FROM recording_orientation_status ORDER BY recording_id;"
    ).fetchall()


def test_migration_078_is_registered_and_additive() -> None:
    assert (
        78,
        "recording_orientation_status",
        "_migration_078_recording_orientation_status",
    ) in MIGRATION_METHODS
    assert 78 in ADDITIVE_MIGRATIONS


def test_post_cutoff_citrus_check_projects_verified(tmp_path: Path) -> None:
    recording = _recording(tmp_path)
    _write(recording / H5_RELATIVE, record=_omnifin0(), session_start=POST_CUTOFF)
    row = extract_recording_orientation_status(recording, session_uuid=ORANGE_SESSION, camera_id=CAMERA)
    assert row["orientation_status"] == resolver.VERIFIED
    assert row["basis"] == resolver.BASIS_CITRUS_CHECK
    assert row["verified_link"] == "camera_to_canvas"
    assert row["camera_to_dish_parity"] == "declared"
    assert row["h5_relative_path"] == H5_RELATIVE
    assert (row["session_uuid"], row["citrus_session_uuid"]) == (ORANGE_SESSION, SESSION)
    assert (row["rig_id"], row["arena_id"], row["camera_serial"]) == (
        "omnifin0",
        "arena_1",
        "2010093",
    )
    status = json.loads(row["status_json"])
    assert status["recorded_space_parity_basis"]["viewing_side"] == "dorsal"

    registry = _registry(tmp_path)
    try:
        registry.replace_recording_orientation_status("rec-1", row)
        registry.replace_recording_orientation_status("rec-1", row)
        (stored,) = _rows(registry)
        assert stored["orientation_status"] == "verified"
        assert stored["status_schema_id"] == resolver.SCHEMA_ID
        assert stored["status_schema_version"] == resolver.SCHEMA_VERSION
        assert stored["updated_utc"]
    finally:
        registry.close()


def test_listed_recording_projects_the_evidence_basis(tmp_path: Path, listed) -> None:
    recording = _recording(tmp_path)
    _write(recording / H5_RELATIVE)
    row = extract_recording_orientation_status(recording, session_uuid=ORANGE_SESSION, camera_id=CAMERA)
    assert row["orientation_status"] == resolver.VERIFIED
    assert row["basis"] == "omnifin0_20261115_static_dots_nir_filters_off"
    assert row["evidence_sha256"].startswith("sha256:")


def test_unlisted_pre_cutoff_recording_projects_unverified(tmp_path: Path) -> None:
    recording = _recording(tmp_path)
    _write(recording / H5_RELATIVE)
    row = extract_recording_orientation_status(recording, session_uuid=ORANGE_SESSION, camera_id=CAMERA)
    assert row["orientation_status"] == resolver.UNVERIFIED
    assert row["reason"] == "no_orientation_evidence"
    assert row["basis"] is None and row["verified_link"] is None


@pytest.mark.parametrize(
    ("h5_relative", "write", "reason"),
    [
        (None, None, "no_manifest_citrus_h5"),
        ("raw/acquisition/citrus/missing.h5", None, "citrus_h5_unresolved"),
        ("../outside.h5", None, "citrus_h5_unresolved"),
        (H5_RELATIVE, b"not an hdf5 file", "citrus_h5_unreadable"),
    ],
)
def test_unreadable_sources_project_unverified_not_raise(
    tmp_path: Path, h5_relative: str | None, write: bytes | None, reason: str
) -> None:
    recording = _recording(tmp_path, h5_relative=h5_relative)
    if write is not None:
        (recording / H5_RELATIVE).write_bytes(write)
    row = extract_recording_orientation_status(recording, session_uuid=ORANGE_SESSION, camera_id=CAMERA)
    assert row["orientation_status"] == resolver.UNVERIFIED
    assert row["reason"] == reason
    assert (row["session_uuid"], row["citrus_session_uuid"]) == (ORANGE_SESSION, None)
    registry = _registry(tmp_path)
    try:
        registry.replace_recording_orientation_status("rec-1", row)
        assert _rows(registry)[0]["reason"] == reason
    finally:
        registry.close()


def test_missing_manifest_projects_unverified(tmp_path: Path) -> None:
    row = extract_recording_orientation_status(tmp_path, session_uuid=ORANGE_SESSION, camera_id=CAMERA)
    assert row["reason"] == "recording_manifest_unreadable"


def test_verified_h5_of_another_camera_is_downgraded(tmp_path: Path) -> None:
    recording = _recording(tmp_path)
    _write(recording / H5_RELATIVE, record=_omnifin0(), session_start=POST_CUTOFF)
    row = extract_recording_orientation_status(
        recording, session_uuid=ORANGE_SESSION, camera_id="2010094"
    )
    assert row["orientation_status"] == resolver.UNVERIFIED
    assert row["reason"] == "registry_camera_mismatch"
    assert row["camera_serial"] == CAMERA
    assert json.loads(row["status_json"])["reason"] == "registry_camera_mismatch"


def test_table_refuses_inconsistent_rows_and_cascades(tmp_path: Path) -> None:
    recording = _recording(tmp_path)
    _write(recording / H5_RELATIVE)
    row = extract_recording_orientation_status(recording, session_uuid=ORANGE_SESSION, camera_id=CAMERA)
    registry = _registry(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            registry.replace_recording_orientation_status(
                "rec-1", {**row, "orientation_status": "verified"}
            )
        with pytest.raises(sqlite3.IntegrityError):
            registry.replace_recording_orientation_status(
                "rec-1", {**row, "orientation_status": "maybe"}
            )
        with pytest.raises(sqlite3.IntegrityError):
            registry.replace_recording_orientation_status("rec-unknown", row)
        registry.replace_recording_orientation_status("rec-1", row)
        registry.conn.execute("DELETE FROM recordings WHERE recording_id = 'rec-1';")
        assert _rows(registry) == []
    finally:
        registry.close()
