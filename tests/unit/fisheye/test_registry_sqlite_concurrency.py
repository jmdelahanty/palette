import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import zarr

from fisheye.registry.db import SQLITE_BUSY_TIMEOUT_MS, Registry
from fisheye.registry.dedupe import _connect as dedupe_connect


def _table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = [
        str(row["name"])
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
            """
        ).fetchall()
    ]
    return {table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}";').fetchone()[0]) for table in tables}


def _minimal_zarr(path: Path) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["session_uuid"] = "session_atomic"
    root.attrs["recording_id"] = "recording_atomic"
    root.attrs["zarr_use"] = "analysis"
    return root


def test_registry_connections_set_busy_timeout_and_keep_rollback_journal(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        timeout_ms = int(registry.conn.execute("PRAGMA busy_timeout;").fetchone()[0])
        journal_mode = str(registry.conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower()
        assert timeout_ms == SQLITE_BUSY_TIMEOUT_MS
        assert journal_mode != "wal"
    finally:
        registry.close()


def test_dedupe_connection_sets_busy_timeout_and_keeps_rollback_journal(tmp_path: Path) -> None:
    conn = dedupe_connect(tmp_path / "registry.sqlite")
    try:
        timeout_ms = int(conn.execute("PRAGMA busy_timeout;").fetchone()[0])
        journal_mode = str(conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower()
        assert timeout_ms == SQLITE_BUSY_TIMEOUT_MS
        assert journal_mode != "wal"
    finally:
        conn.close()


def test_migration_version_is_rechecked_after_write_lock() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE schema_version(version INTEGER PRIMARY KEY, name TEXT, applied_utc TEXT);"
    )
    conn.execute(
        "INSERT INTO schema_version VALUES (72, 'already_applied', '2026-08-25T12:00:00Z');"
    )
    conn.commit()
    observed_versions = iter((71, 72))
    migration_called = False

    def stale_then_locked_version() -> int:
        return next(observed_versions)

    def should_not_replay() -> None:
        nonlocal migration_called
        migration_called = True

    fixture = SimpleNamespace(
        conn=conn,
        _schema_migrations=lambda: [(72, "fixture", should_not_replay)],
        _current_schema_version=stale_then_locked_version,
        _has_legacy_schema=lambda: False,
    )
    try:
        Registry._apply_schema_migrations(fixture)  # type: ignore[arg-type]
        assert migration_called is False
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_register_from_root_rolls_back_when_late_replace_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recording_analysis.zarr"
    _minimal_zarr(zarr_path)
    before = _table_counts(registry.conn)

    def fail_late_replace(self: Registry, dataset_id: str, records: object) -> None:
        raise RuntimeError("late replace failed")

    monkeypatch.setattr(Registry, "replace_subject_mask_component_quality", fail_late_replace)
    with pytest.raises(RuntimeError, match="late replace failed"):
        registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)

    assert registry._managed_transaction_depth == 0
    assert _table_counts(registry.conn) == before

    registry.upsert_dataset(
        "standalone_ds",
        session_uuid=None,
        zarr_path=tmp_path / "standalone.zarr",
        recording_id="standalone_rec",
        zarr_use="analysis",
    )
    registry.replace_detection_sources(
        "standalone_ds",
        [
            {
                "refined_run": "refined_detect_runs/refined_1",
                "source_type": "detect",
                "counts": {"total_detections": 1},
            }
        ],
    )
    row_count = registry.conn.execute(
        "SELECT COUNT(*) FROM detection_sources WHERE dataset_id = 'standalone_ds';"
    ).fetchone()[0]
    assert int(row_count) == 1
    registry.close()


def test_managed_transaction_suppresses_inner_standalone_commit(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    observer = sqlite3.connect(str(tmp_path / "registry.sqlite"))
    observer.row_factory = sqlite3.Row
    observer.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS};")
    try:
        with registry._transaction_context():
            registry.upsert_dataset(
                "nested_ds",
                session_uuid=None,
                zarr_path=tmp_path / "nested.zarr",
                recording_id="nested_rec",
                zarr_use="analysis",
            )
            visible_during_transaction = observer.execute(
                "SELECT COUNT(*) FROM datasets WHERE dataset_id = 'nested_ds';"
            ).fetchone()[0]
            assert int(visible_during_transaction) == 0

        visible_after_commit = observer.execute(
            "SELECT COUNT(*) FROM datasets WHERE dataset_id = 'nested_ds';"
        ).fetchone()[0]
        assert int(visible_after_commit) == 1
    finally:
        observer.close()
        registry.close()
