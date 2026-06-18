from __future__ import annotations

import sqlite3
from pathlib import Path

from fisheye.registry.dedupe import apply_registry_dataset_dedupe, plan_registry_dataset_dedupe


def _init_test_registry(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(
        """
        CREATE TABLE datasets (
            dataset_id TEXT PRIMARY KEY,
            session_uuid TEXT,
            zarr_path TEXT NOT NULL,
            recording_id TEXT,
            artifact_kind TEXT,
            zarr_origin TEXT,
            zarr_use TEXT,
            source_layout TEXT,
            source_frame_index_path TEXT,
            source_recording_frame_index_path TEXT,
            source_frame_index_schema TEXT,
            path_hash TEXT,
            created_utc TEXT,
            last_seen_utc TEXT,
            status TEXT
        );

        CREATE TABLE provenance (
            dataset_id TEXT PRIMARY KEY,
            camera_id TEXT,
            FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        );

        CREATE TABLE recording_step_status (
            dataset_id TEXT NOT NULL,
            step_name TEXT NOT NULL,
            status TEXT NOT NULL,
            updated_utc TEXT,
            PRIMARY KEY (dataset_id, step_name),
            FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        );

        CREATE TABLE recording_step_status_history (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            step_name TEXT NOT NULL,
            status TEXT NOT NULL,
            recorded_utc TEXT NOT NULL,
            FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        );

        CREATE TABLE dataset_lineage (
            child_dataset_id TEXT NOT NULL,
            parent_dataset_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            PRIMARY KEY (child_dataset_id, parent_dataset_id, relationship_type),
            FOREIGN KEY(child_dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,
            FOREIGN KEY(parent_dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        );
        """
    )
    return conn


def _insert_dataset(
    conn: sqlite3.Connection,
    dataset_id: str,
    *,
    session_uuid: str,
    zarr_path: str,
    path_hash: str,
    zarr_use: str = "training",
    status: str = "active",
) -> None:
    conn.execute(
        """
        INSERT INTO datasets (
            dataset_id, session_uuid, zarr_path, zarr_use, path_hash,
            created_utc, last_seen_utc, status
        )
        VALUES (?, ?, ?, ?, ?, '2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z', ?);
        """,
        (dataset_id, session_uuid, zarr_path, zarr_use, path_hash, status),
    )


def test_plan_registry_dataset_dedupe_reports_conflicting_step_status(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    conn = _init_test_registry(registry)
    try:
        _insert_dataset(
            conn,
            "rec_a",
            session_uuid="rec_a",
            zarr_path="/data/rec_a_training.zarr",
            path_hash="hash-a",
        )
        _insert_dataset(
            conn,
            "rec_a:zhash",
            session_uuid="rec_a",
            zarr_path="/data/rec_a_training.zarr",
            path_hash="hash-a",
        )
        conn.execute("INSERT INTO provenance (dataset_id, camera_id) VALUES ('rec_a', '2010093');")
        conn.execute(
            "INSERT INTO recording_step_status (dataset_id, step_name, status, updated_utc) "
            "VALUES ('rec_a', 'detect', 'ok', '2026-01-02T00:00:00Z');"
        )
        conn.execute(
            "INSERT INTO recording_step_status (dataset_id, step_name, status, updated_utc) "
            "VALUES ('rec_a:zhash', 'detect', 'missing', '2026-01-02T00:00:00Z');"
        )
        conn.execute(
            "INSERT INTO recording_step_status_history (dataset_id, step_name, status, recorded_utc) "
            "VALUES ('rec_a:zhash', 'detect', 'missing', '2026-01-02T00:00:00Z');"
        )
        conn.commit()
    finally:
        conn.close()

    report = plan_registry_dataset_dedupe(registry)

    assert report["status"] == "conflicts"
    assert report["duplicate_group_count"] == 1
    group = report["groups"][0]
    assert group["canonical_dataset_id"] == "rec_a"
    assert group["rows_to_repoint"] == 2
    assert group["conflicting_rows"] == 1
    duplicate = group["duplicates"][0]
    assert duplicate["dataset"]["dataset_id"] == "rec_a:zhash"
    refs = {(item["table"], item["column"]): item for item in duplicate["reference_updates"]}
    assert refs[("recording_step_status", "dataset_id")]["rows_to_repoint"] == 1
    assert refs[("recording_step_status", "dataset_id")]["conflicts"][0]["constraint"] == "primary_key"
    assert refs[("recording_step_status_history", "dataset_id")]["rows_to_repoint"] == 1
    assert refs[("recording_step_status_history", "dataset_id")]["conflicts"] == []


def test_plan_registry_dataset_dedupe_scopes_by_zarr_use_and_path(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    conn = _init_test_registry(registry)
    try:
        _insert_dataset(
            conn,
            "rec_a",
            session_uuid="rec_a",
            zarr_path="/data/rec_a_training.zarr",
            path_hash="hash-a",
            zarr_use="training",
        )
        _insert_dataset(
            conn,
            "rec_a:zhash",
            session_uuid="rec_a",
            zarr_path="/data/rec_a_training.zarr",
            path_hash="hash-a",
            zarr_use="training",
        )
        _insert_dataset(
            conn,
            "rec_b",
            session_uuid="rec_b",
            zarr_path="/data/rec_b_analysis.zarr",
            path_hash="hash-b",
            zarr_use="analysis",
        )
        _insert_dataset(
            conn,
            "rec_b:zhash",
            session_uuid="rec_b",
            zarr_path="/data/rec_b_analysis.zarr",
            path_hash="hash-b",
            zarr_use="analysis",
        )
        conn.commit()
    finally:
        conn.close()

    report = plan_registry_dataset_dedupe(
        registry,
        zarr_use="training",
        path_contains="rec_a",
    )

    assert report["status"] == "ok"
    assert report["duplicate_group_count"] == 1
    assert report["groups"][0]["zarr_path"] == "/data/rec_a_training.zarr"
    assert report["groups"][0]["canonical_dataset_id"] == "rec_a"


def test_plan_registry_dataset_dedupe_flags_dataset_lineage_self_edge(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    conn = _init_test_registry(registry)
    try:
        _insert_dataset(
            conn,
            "rec_a",
            session_uuid="rec_a",
            zarr_path="/data/rec_a_training.zarr",
            path_hash="hash-a",
        )
        _insert_dataset(
            conn,
            "rec_a:zhash",
            session_uuid="rec_a",
            zarr_path="/data/rec_a_training.zarr",
            path_hash="hash-a",
        )
        conn.execute(
            "INSERT INTO dataset_lineage (child_dataset_id, parent_dataset_id, relationship_type) "
            "VALUES ('rec_a:zhash', 'rec_a', 'duplicate_of');"
        )
        conn.commit()
    finally:
        conn.close()

    report = plan_registry_dataset_dedupe(registry)

    assert report["status"] == "conflicts"
    lineage_update = report["groups"][0]["duplicates"][0]["reference_updates"][0]
    assert lineage_update["table"] == "dataset_lineage"
    assert lineage_update["column"] == "child_dataset_id"
    constraints = {item["constraint"] for item in lineage_update["conflicts"]}
    assert "dataset_lineage_no_self_edge" in constraints


def test_apply_registry_dataset_dedupe_merges_duplicate_rows(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    conn = _init_test_registry(registry)
    try:
        _insert_dataset(
            conn,
            "rec_a",
            session_uuid="rec_a",
            zarr_path="/data/rec_a_training.zarr",
            path_hash="hash-a",
        )
        _insert_dataset(
            conn,
            "rec_a:zhash",
            session_uuid="rec_a",
            zarr_path="/data/rec_a_training.zarr",
            path_hash="hash-a",
        )
        conn.execute("INSERT INTO provenance (dataset_id, camera_id) VALUES ('rec_a', '2010093');")
        conn.execute(
            "INSERT INTO recording_step_status (dataset_id, step_name, status, updated_utc) "
            "VALUES ('rec_a', 'detect', 'ok', '2026-01-02T00:00:00Z');"
        )
        conn.execute(
            "INSERT INTO recording_step_status (dataset_id, step_name, status, updated_utc) "
            "VALUES ('rec_a:zhash', 'detect', 'missing', '2026-01-02T00:00:00Z');"
        )
        conn.execute(
            "INSERT INTO recording_step_status_history (dataset_id, step_name, status, recorded_utc) "
            "VALUES ('rec_a:zhash', 'detect', 'missing', '2026-01-02T00:00:00Z');"
        )
        conn.commit()
    finally:
        conn.close()

    report = apply_registry_dataset_dedupe(registry)

    assert report["status"] == "ok"
    assert report["dataset_rows_deleted"] == 1
    assert report["conflict_rows_deleted"] == 1
    assert report["rows_repointed"] == 1

    conn = sqlite3.connect(str(registry))
    try:
        assert conn.execute("SELECT dataset_id FROM datasets;").fetchall() == [("rec_a",)]
        assert conn.execute(
            "SELECT dataset_id, step_name, status FROM recording_step_status;"
        ).fetchall() == [("rec_a", "detect", "ok")]
        assert conn.execute(
            "SELECT dataset_id, step_name, status FROM recording_step_status_history;"
        ).fetchall() == [("rec_a", "detect", "missing")]
        assert conn.execute("PRAGMA foreign_key_check;").fetchall() == []
    finally:
        conn.close()


def test_apply_registry_dataset_dedupe_removes_lineage_self_edges(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    conn = _init_test_registry(registry)
    try:
        _insert_dataset(
            conn,
            "rec_a",
            session_uuid="rec_a",
            zarr_path="/data/rec_a_training.zarr",
            path_hash="hash-a",
        )
        _insert_dataset(
            conn,
            "rec_a:zhash",
            session_uuid="rec_a",
            zarr_path="/data/rec_a_training.zarr",
            path_hash="hash-a",
        )
        _insert_dataset(
            conn,
            "parent",
            session_uuid="parent",
            zarr_path="/data/parent_training.zarr",
            path_hash="hash-parent",
        )
        conn.execute(
            "INSERT INTO dataset_lineage (child_dataset_id, parent_dataset_id, relationship_type) "
            "VALUES ('rec_a:zhash', 'rec_a', 'duplicate_of');"
        )
        conn.execute(
            "INSERT INTO dataset_lineage (child_dataset_id, parent_dataset_id, relationship_type) "
            "VALUES ('rec_a:zhash', 'parent', 'source');"
        )
        conn.commit()
    finally:
        conn.close()

    report = apply_registry_dataset_dedupe(registry)

    assert report["status"] == "ok"
    assert report["self_edge_rows_deleted"] == 1
    assert report["rows_repointed"] == 1

    conn = sqlite3.connect(str(registry))
    try:
        assert conn.execute(
            "SELECT child_dataset_id, parent_dataset_id, relationship_type "
            "FROM dataset_lineage ORDER BY relationship_type;"
        ).fetchall() == [("rec_a", "parent", "source")]
        assert conn.execute("PRAGMA foreign_key_check;").fetchall() == []
    finally:
        conn.close()
