from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fisheye.registry.db import Registry
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_CLAIM_SCHEMA_ID,
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
    SourceRecordingIdentity,
    SourceRecordingIdentityClaim,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_bytes


EVIDENCE_SCHEMA_ID = SOURCE_RECORDING_IDENTITY_CLAIM_SCHEMA_ID
REVISION_SCHEMA_ID = "palette.registry.recording_identity_revision.v1"


def _seed_identity_chain(
    registry: Registry,
    *,
    dataset_id: str,
    recording_id: str,
    session_uuid: str,
    identity_scope_id: str,
    identity_snapshot_id: str,
) -> None:
    conn = registry.conn
    conn.execute(
        """
        INSERT INTO recordings(recording_id, session_uuid)
        VALUES (?, ?);
        """,
        (recording_id, session_uuid),
    )
    conn.execute(
        """
        INSERT INTO datasets(
            dataset_id, session_uuid, zarr_path, recording_id
        ) VALUES (?, ?, ?, ?);
        """,
        (dataset_id, session_uuid, f"/data/{dataset_id}.zarr", recording_id),
    )
    identity_claim = SourceRecordingIdentityClaim.create(
        SourceRecordingIdentity.from_mapping(
            {
                SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: SOURCE_RECORDING_IDENTITY_PROFILE,
                "recording_id": recording_id,
                "session_uuid": session_uuid,
                "camera_id": f"camera-{recording_id}",
            }
        )
    )
    evidence_digest = identity_claim.claim_sha256
    evidence_json = canonical_json_bytes(identity_claim.as_dict()).decode("utf-8")
    conn.execute(
        """
        INSERT INTO recording_identity_evidence(
            evidence_digest, schema_id, evidence_json
        ) VALUES (?, ?, ?);
        """,
        (evidence_digest, EVIDENCE_SCHEMA_ID, evidence_json),
    )
    conn.execute(
        """
        INSERT INTO recording_identity_revisions(
            identity_snapshot_id, identity_scope_id, recording_id,
            session_uuid, identity_revision,
            supersedes_identity_snapshot_id, schema_id, revision_kind,
            decided_by, decided_at_utc, correction_reason,
            evidence_digest, initiating_dataset_id, registry_schema_version
        ) VALUES (?, ?, ?, ?, 1, NULL, ?, 'initial', 'pytest',
                  '2026-08-25T12:00:00Z', NULL, ?, ?, 72);
        """,
        (
            identity_snapshot_id,
            identity_scope_id,
            recording_id,
            session_uuid,
            REVISION_SCHEMA_ID,
            evidence_digest,
            dataset_id,
        ),
    )
    conn.commit()


def _insert_binding(
    registry: Registry,
    *,
    receipt_sha256: str = "a" * 64,
    dataset_id: str = "dataset-a",
    identity_scope_id: str = "11111111-1111-4111-8111-111111111111",
    identity_snapshot_id: str = "22222222-2222-4222-8222-222222222222",
) -> None:
    registry.conn.execute(
        """
        INSERT INTO recording_import_receipt_bindings(
            receipt_sha256, dataset_id, identity_scope_id,
            identity_snapshot_id, bound_by, bound_at_utc,
            registry_schema_version
        ) VALUES (?, ?, ?, ?, 'pytest', '2026-08-25T12:01:00Z', 72);
        """,
        (
            receipt_sha256,
            dataset_id,
            identity_scope_id,
            identity_snapshot_id,
        ),
    )
    registry.conn.commit()


def _make_registry_with_identity_rows(tmp_path: Path) -> Registry:
    registry = Registry(tmp_path / "registry.sqlite")
    _seed_identity_chain(
        registry,
        dataset_id="dataset-a",
        recording_id="recording-a",
        session_uuid="session-a",
        identity_scope_id="11111111-1111-4111-8111-111111111111",
        identity_snapshot_id="22222222-2222-4222-8222-222222222222",
    )
    _seed_identity_chain(
        registry,
        dataset_id="dataset-b",
        recording_id="recording-b",
        session_uuid="session-b",
        identity_scope_id="33333333-3333-4333-8333-333333333333",
        identity_snapshot_id="44444444-4444-4444-8444-444444444444",
    )
    return registry


def test_migration_72_creates_empty_receipt_binding_table(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        columns = {
            str(row["name"])
            for row in registry.conn.execute(
                "PRAGMA table_info(recording_import_receipt_bindings);"
            ).fetchall()
        }
        assert columns == {
            "receipt_sha256",
            "dataset_id",
            "identity_scope_id",
            "identity_snapshot_id",
            "bound_by",
            "bound_at_utc",
            "registry_schema_version",
        }
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recording_import_receipt_bindings;"
        ).fetchone()[0] == 0
        assert registry._current_schema_version() == 73
    finally:
        registry.close()


def test_receipt_binding_accepts_exact_row_and_rejects_foreign_keys(
    tmp_path: Path,
) -> None:
    registry = _make_registry_with_identity_rows(tmp_path)
    try:
        _insert_binding(registry)
        row = registry.conn.execute(
            "SELECT * FROM recording_import_receipt_bindings;"
        ).fetchone()
        assert tuple(row) == (
            "a" * 64,
            "dataset-a",
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
            "pytest",
            "2026-08-25T12:01:00Z",
            72,
        )

        with pytest.raises(sqlite3.IntegrityError):
            _insert_binding(registry, receipt_sha256="b" * 64, dataset_id="missing")
        with pytest.raises(sqlite3.IntegrityError):
            _insert_binding(
                registry,
                receipt_sha256="c" * 64,
                identity_scope_id="55555555-5555-4555-8555-555555555555",
                identity_snapshot_id="66666666-6666-4666-8666-666666666666",
            )
    finally:
        registry.close()


def test_receipt_binding_rejects_uppercase_digest_and_duplicate_tuple_or_digest(
    tmp_path: Path,
) -> None:
    registry = _make_registry_with_identity_rows(tmp_path)
    try:
        _insert_binding(registry)

        with pytest.raises(sqlite3.IntegrityError):
            _insert_binding(registry, receipt_sha256="A" * 64)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_binding(registry, receipt_sha256="b" * 64)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_binding(
                registry,
                receipt_sha256="a" * 64,
                dataset_id="dataset-b",
                identity_scope_id="33333333-3333-4333-8333-333333333333",
                identity_snapshot_id="44444444-4444-4444-8444-444444444444",
            )
    finally:
        registry.close()


def test_receipt_binding_rejects_update_delete_and_insert_or_replace(
    tmp_path: Path,
) -> None:
    registry = _make_registry_with_identity_rows(tmp_path)
    try:
        _insert_binding(registry)
        with pytest.raises(sqlite3.IntegrityError):
            registry.conn.execute(
                "UPDATE recording_import_receipt_bindings SET bound_by = 'other';"
            )
        with pytest.raises(sqlite3.IntegrityError):
            registry.conn.execute(
                "DELETE FROM recording_import_receipt_bindings WHERE receipt_sha256 = ?;",
                ("a" * 64,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            registry.conn.execute(
                """
                INSERT OR REPLACE INTO recording_import_receipt_bindings(
                    receipt_sha256, dataset_id, identity_scope_id,
                    identity_snapshot_id, bound_by, bound_at_utc,
                    registry_schema_version
                ) VALUES (?, 'dataset-a', ?, ?, 'other',
                          '2026-08-25T12:02:00Z', 72);
                """,
                (
                    "a" * 64,
                    "11111111-1111-4111-8111-111111111111",
                    "22222222-2222-4222-8222-222222222222",
                ),
            )
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recording_import_receipt_bindings;"
        ).fetchone()[0] == 1
    finally:
        registry.close()
