"""Schema v9: the database itself enforces task, checkpoint, and receipt state domains."""

from __future__ import annotations

import re
import sqlite3

import pytest

from fisheye.labeling import assignment_store as store_mod
from fisheye.labeling import checkpoint_store
from fisheye.labeling.assignment_store import TASK_STATES, LabelingStore

from .labeling_store_legacy import as_v8 as _as_v8

from .labeling_store_legacy import STATE_TABLES


def _seed(store: LabelingStore) -> None:
    store.assign_recording(recording_id="rec", assignee_user="u", assigned_by="u")
    store.upsert_task(recording_id="rec", task_id="t1", workflow_kind="keypoints")
    lease = store.create_session(task_id="t1", user="u")
    store.upsert_session_checkpoint(
        session_id=lease.session_id, task_id="t1", recording_id="rec", user="u",
        workflow_kind="keypoints", target_run_path="refined_keypoints_runs/r",
        target_edit_revision=0, source_rowset_path=None, roi_idx=0,
        component_name="keypoints", payload={},
    )


def _table_sql(path, table):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT sql FROM sqlite_master WHERE name = ?;", (table,)).fetchone()[0]
    finally:
        conn.close()


def test_fresh_store_rejects_unknown_states_in_every_state_column(tmp_path):
    store = LabelingStore(tmp_path / "s.sqlite")
    _seed(store)
    conn = store.conn
    for table, column in (
        ("labeling_tasks", "state"),
        ("labeling_session_checkpoints", "state"),
        ("labeling_checkpoint_apply_receipts", "state"),
        ("labeling_checkpoint_apply_receipts", "secondary_effects_state"),
    ):
        assert "CHECK" in _table_sql(tmp_path / "s.sqlite", table)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE labeling_tasks SET state = 'done' WHERE task_id = 't1';")
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE labeling_session_checkpoints SET state = 'lost';")
    conn.rollback()
    for state in TASK_STATES:  # Every allowed value stays writable.
        conn.execute("UPDATE labeling_tasks SET state = ? WHERE task_id = 't1';", (state,))
    for state in checkpoint_store.CHECKPOINT_STATES:
        conn.execute("UPDATE labeling_session_checkpoints SET state = ?;", (state,))
    conn.rollback()
    store.close()


def test_v8_store_upgrades_in_place_preserving_rows_and_indexes(tmp_path):
    path = tmp_path / "s.sqlite"
    store = LabelingStore(path)
    _seed(store)
    store.close()
    _as_v8(path)
    conn = sqlite3.connect(path)
    rows_before = {t: conn.execute(f'SELECT * FROM "{t}" ORDER BY rowid').fetchall() for t in STATE_TABLES}
    indexes_before = sorted(r[0] for r in conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"))
    conn.close()
    assert all("CHECK" not in _table_sql(path, t) for t in STATE_TABLES)

    upgraded = LabelingStore(path)
    upgraded.initialize()
    upgraded.close()
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT value FROM labeling_schema_meta WHERE key='schema_version'").fetchone()[0] == str(store_mod.SCHEMA_VERSION)
    for table in STATE_TABLES:
        assert "CHECK" in _table_sql(path, table)
        assert conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall() == rows_before[table]
    assert sorted(r[0] for r in conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")) == indexes_before
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_upgrade_refuses_out_of_domain_rows_and_changes_nothing(tmp_path):
    path = tmp_path / "s.sqlite"
    store = LabelingStore(path)
    _seed(store)
    store.close()
    _as_v8(path)
    conn = sqlite3.connect(path)
    conn.execute("UPDATE labeling_tasks SET state = 'typo' WHERE task_id = 't1';")
    conn.commit()
    conn.close()
    before = {t: _table_sql(path, t) for t in STATE_TABLES}
    with pytest.raises(RuntimeError, match="labeling_tasks.state"):
        LabelingStore(path).initialize()
    assert {t: _table_sql(path, t) for t in STATE_TABLES} == before
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT value FROM labeling_schema_meta WHERE key='schema_version'").fetchone()[0] == "8"
    conn.close()


def test_failed_rebuild_rolls_back_every_table(tmp_path, monkeypatch):
    path = tmp_path / "s.sqlite"
    store = LabelingStore(path)
    _seed(store)
    store.close()
    _as_v8(path)
    before = {t: _table_sql(path, t) for t in STATE_TABLES}
    real = store_mod._state_checks

    def broken_last_table():
        checks = dict(real())
        checks["labeling_checkpoint_apply_receipts"] = ("CHECK (no_such_column IN ('x'))",)
        return checks

    monkeypatch.setattr(store_mod, "_state_checks", broken_last_table)
    with pytest.raises(sqlite3.OperationalError):
        LabelingStore(path).initialize()
    assert {t: _table_sql(path, t) for t in STATE_TABLES} == before
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE name LIKE '%__v9'")} == set()
    conn.close()


def test_backup_validator_reports_out_of_domain_states_in_legacy_stores(tmp_path):
    from fisheye.labeling.store_backup import validate_labeling_sqlite

    path = tmp_path / "s.sqlite"
    store = LabelingStore(path)
    _seed(store)
    store.close()
    assert validate_labeling_sqlite(path)["state_violations"] == {}
    _as_v8(path)
    conn = sqlite3.connect(path)
    conn.execute("UPDATE labeling_tasks SET state = 'typo' WHERE task_id = 't1';")
    conn.commit()
    conn.close()
    assert validate_labeling_sqlite(path)["state_violations"] == {"labeling_tasks": {"typo": 1}}
