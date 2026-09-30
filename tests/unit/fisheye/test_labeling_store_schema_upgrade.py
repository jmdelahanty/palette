"""Opening an older labeling store upgrades every table to the fresh schema.

A long-lived store reported schema_version 9 but its receipts table predated
released_checkpoint_count, which was declared only in CREATE TABLE: every Apply
on that store failed with "no such column". These tests pin the upgrade.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fisheye.labeling.assignment_store import LabelingStore

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "labeling_store_schema_v9_pre_released_count.sql"


def _columns(path) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(path)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")]
        # name, declared type, not-null, default; column order is not compared
        # because ALTER TABLE ADD COLUMN always appends.
        return {t: sorted((r[1], r[2], r[3], r[4]) for r in conn.execute(f'PRAGMA table_info("{t}");'))
                for t in tables}
    finally:
        conn.close()


def _open(path) -> None:
    store = LabelingStore(path)
    try:
        store.initialize()
    finally:
        store.close()


def test_historical_store_schema_upgrades_to_the_fresh_schema(tmp_path):
    fresh = tmp_path / "fresh.sqlite"
    _open(fresh)
    old = tmp_path / "old.sqlite"
    conn = sqlite3.connect(old)
    conn.executescript(FIXTURE.read_text())
    conn.execute("INSERT INTO labeling_schema_meta (key, value) VALUES ('schema_version', '9');")
    conn.commit()
    conn.close()
    assert "released_checkpoint_count" not in {c[0] for c in _columns(old)["labeling_checkpoint_apply_receipts"]}
    _open(old)
    upgraded, expected = _columns(old), _columns(fresh)
    # Known, accepted difference: snapshot_row_sha256 was added by ALTER, which
    # cannot add a NOT NULL column without a default, so upgraded stores keep it
    # nullable and the upgrade backfills every row instead.
    relaxed = ("snapshot_row_sha256", "TEXT", 0, None)
    strict = ("snapshot_row_sha256", "TEXT", 1, None)
    checkpoints = upgraded["labeling_session_checkpoints"]
    assert relaxed in checkpoints
    upgraded["labeling_session_checkpoints"] = sorted(strict if c == relaxed else c for c in checkpoints)
    assert upgraded == expected


def test_receipts_without_released_count_gain_it_and_apply_bookkeeping_works(tmp_path):
    path = tmp_path / "store.sqlite"
    store = LabelingStore(path)
    store.initialize()
    store.assign_recording(recording_id="rec", assignee_user="u", assigned_by="u")
    store.upsert_task(recording_id="rec", task_id="t1", workflow_kind="keypoints")
    lease = store.create_session(task_id="t1", user="u")
    store.upsert_session_checkpoint(
        session_id=lease.session_id, task_id="t1", recording_id="rec", user="u",
        workflow_kind="keypoints", target_run_path="refined_keypoints_runs/r",
        target_edit_revision=0, source_rowset_path=None, roi_idx=0,
        component_name="keypoints", payload={},
    )
    claimed = store.claim_session_checkpoints_for_apply(task_id="t1", component_name="keypoints", apply_id="a1")
    assert len(claimed) == 1
    store.close()

    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE labeling_checkpoint_apply_receipts DROP COLUMN released_checkpoint_count;")
    conn.commit()
    conn.close()

    reopened = LabelingStore(path)
    try:
        reopened.initialize()
        row = reopened.conn.execute(
            "SELECT released_checkpoint_count FROM labeling_checkpoint_apply_receipts WHERE apply_id = 'a1';"
        ).fetchone()
        assert row[0] == 0
        # Releasing the claim reads and writes the column (it raised "no such column").
        assert reopened.release_session_checkpoints_apply(task_id="t1", apply_id="a1") == 1
        assert reopened.count_unapplied_session_checkpoints(task_id="t1") == 1
    finally:
        reopened.close()

