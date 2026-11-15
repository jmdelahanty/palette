"""Opening a store backfills receipts only for apply ids that lack one.

Every start used to rebuild and re-encode a receipt for every apply id, then
let INSERT OR IGNORE discard it; on a store with large payloads that was ~22 s.
"""

from __future__ import annotations

import json
import sqlite3

from fisheye.labeling import checkpoint_store
from fisheye.labeling.assignment_store import LabelingStore


def _claimed_store(path) -> None:
    store = LabelingStore(path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec", assignee_user="u", assigned_by="u")
        for task_id, apply_id in (("t1", "has-receipt"), ("t2", "legacy")):
            store.upsert_task(recording_id="rec", task_id=task_id, workflow_kind="keypoints")
            lease = store.create_session(task_id=task_id, user="u")
            store.upsert_session_checkpoint(
                session_id=lease.session_id, task_id=task_id, recording_id="rec", user="u",
                workflow_kind="keypoints", target_run_path="refined_keypoints_runs/r",
                target_edit_revision=0, source_rowset_path=None, roi_idx=0,
                component_name="keypoints", payload={"points": [[1.0, 2.0]]},
            )
            claimed = store.claim_session_checkpoints_for_apply(
                task_id=task_id, component_name="keypoints", apply_id=apply_id
            )
            assert len(claimed) == 1
    finally:
        store.close()


def _receipt(conn, apply_id):
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM labeling_checkpoint_apply_receipts WHERE apply_id = ?;", (apply_id,)
    ).fetchone()
    return dict(row) if row is not None else None


def test_initialize_backfills_missing_receipts_and_skips_existing_ones(tmp_path, monkeypatch):
    path = tmp_path / "store.sqlite"
    _claimed_store(path)
    conn = sqlite3.connect(path)
    existing = _receipt(conn, "has-receipt")
    # A pre-v7 store: the checkpoints were claimed but no receipt was written.
    conn.execute("DELETE FROM labeling_checkpoint_apply_receipts WHERE apply_id = 'legacy';")
    conn.commit()
    conn.close()
    assert existing is not None

    read_apply_ids = []
    original_checkpoint_row = checkpoint_store._checkpoint_row

    def recording_checkpoint_row(row):
        read_apply_ids.append(row["apply_id"])
        return original_checkpoint_row(row)

    monkeypatch.setattr(checkpoint_store, "_checkpoint_row", recording_checkpoint_row)
    store = LabelingStore(path)
    try:
        store.initialize()
    finally:
        store.close()

    assert read_apply_ids == ["legacy"]
    conn = sqlite3.connect(path)
    try:
        assert _receipt(conn, "has-receipt") == existing
        backfilled = _receipt(conn, "legacy")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM labeling_session_checkpoints WHERE apply_id = 'legacy';"
        ).fetchall()
    finally:
        conn.close()
    assert backfilled is not None
    assert (backfilled["task_id"], backfilled["component_name"], backfilled["state"]) == (
        "t2", "keypoints", "applying",
    )
    assert backfilled["checkpoint_count"] == 1
    assert backfilled["secondary_effects_state"] == "not_ready"
    assert json.loads(backfilled["checkpoints_json"]) == json.loads(
        checkpoint_store._json_dumps([original_checkpoint_row(row) for row in rows])
    )
