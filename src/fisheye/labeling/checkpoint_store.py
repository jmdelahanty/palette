"""Transactional SQLite mechanics for browser labeling checkpoints."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Mapping, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_loads(value: object) -> object:
    if value is None or isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _snapshot_row(row: sqlite3.Row | Mapping[str, object]) -> dict[str, object]:
    values = dict(row)
    payload = values.get("payload")
    if "payload_json" in values:
        payload = _json_loads(values["payload_json"])
    metadata = values.get("metadata")
    if "metadata_json" in values:
        metadata = _json_loads(values["metadata_json"])
    if not isinstance(payload, Mapping):
        raise RuntimeError("Checkpoint payload must be a JSON object.")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Checkpoint metadata must be a JSON object.")
    return {
        "schema": "palette.labeling_session_checkpoint_snapshot_row.v1",
        "checkpoint_id": str(values.get("checkpoint_id") or ""),
        "task_id": str(values.get("task_id") or ""),
        "recording_id": str(values.get("recording_id") or ""),
        "user": str(values.get("user") or ""),
        "workflow_kind": str(values.get("workflow_kind") or ""),
        "target_run_path": str(values.get("target_run_path") or ""),
        "target_edit_revision": int(values.get("target_edit_revision") or 0),
        "source_rowset_path": str(values.get("source_rowset_path") or ""),
        "roi_idx": int(values.get("roi_idx") or 0),
        "component_name": str(values.get("component_name") or ""),
        "payload": dict(payload),
        "metadata": dict(metadata),
    }


def _snapshot_row_digest(
    row: sqlite3.Row | Mapping[str, object],
    *,
    allow_nan: bool,
) -> str:
    snapshot = _snapshot_row(row)
    canonical = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=allow_nan,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def snapshot_row_sha256(row: sqlite3.Row | Mapping[str, object]) -> str:
    """Hash one checkpoint using the strict v1 canonical snapshot grammar."""

    return _snapshot_row_digest(row, allow_nan=False)


def compatible_snapshot_row_sha256(
    row: sqlite3.Row | Mapping[str, object],
) -> str:
    try:
        return snapshot_row_sha256(row)
    except ValueError as exc:
        if "Out of range float values" not in str(exc):
            raise
        # Generic checkpoint saves historically use json.dumps' allow_nan=True
        # grammar. Preserve it here; scientific consumers still validate full
        # checkpoint content before applying it to canonical data.
        return _snapshot_row_digest(row, allow_nan=True)


def _expected_snapshot_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    digest = str(value).strip()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("checkpoint_snapshot_sha256 must be a lowercase SHA-256 digest.")
    return digest


def _checkpoint_row(row: sqlite3.Row | Mapping[str, object]) -> dict[str, object]:
    out = dict(row)
    for key in ("payload_json", "metadata_json"):
        if key in out:
            out[key[:-5]] = _json_loads(out.pop(key, None))
    expected_digest = compatible_snapshot_row_sha256(out)
    stored_digest = out.get("snapshot_row_sha256")
    if stored_digest not in (None, "") and str(stored_digest) != expected_digest:
        raise RuntimeError("Checkpoint snapshot digest does not match its stored content.")
    out["snapshot_row_sha256"] = expected_digest
    return out


def checkpoint_row(row: sqlite3.Row | Mapping[str, object]) -> dict[str, object]:
    """Decode and verify one persisted checkpoint row."""

    return _checkpoint_row(row)


def backfill_checkpoint_snapshot_digests(conn: sqlite3.Connection) -> None:
    """Populate the v8 digest column without rewriting checkpoint JSON bytes."""

    cursor = conn.execute(
        """
        SELECT * FROM labeling_session_checkpoints
        WHERE snapshot_row_sha256 IS NULL OR snapshot_row_sha256 = ''
        ORDER BY checkpoint_id ASC;
        """
    )
    while True:
        rows = cursor.fetchmany(32)
        if not rows:
            break
        updates = [
            (
                str(_checkpoint_row(row)["snapshot_row_sha256"]),
                str(row["checkpoint_id"]),
            )
            for row in rows
        ]
        conn.executemany(
            """
            UPDATE labeling_session_checkpoints
            SET snapshot_row_sha256 = ?
            WHERE checkpoint_id = ?
              AND (snapshot_row_sha256 IS NULL OR snapshot_row_sha256 = '');
            """,
            updates,
        )


def _receipt_snapshots(receipt: sqlite3.Row) -> list[dict[str, object]]:
    snapshots = _json_loads(receipt["checkpoints_json"])
    if not isinstance(snapshots, list) or not all(isinstance(row, Mapping) for row in snapshots):
        raise RuntimeError("Checkpoint apply receipt contains an invalid snapshot.")
    out = [_checkpoint_row(row) for row in snapshots]
    checkpoint_ids = [str(row.get("checkpoint_id") or "") for row in out]
    if (
        len(out) != int(receipt["checkpoint_count"])
        or any(not value for value in checkpoint_ids)
        or len(set(checkpoint_ids)) != len(checkpoint_ids)
        or any(
            str(row.get("task_id") or "") != str(receipt["task_id"])
            or str(row.get("component_name") or "") != str(receipt["component_name"])
            or str(row.get("apply_id") or "") != str(receipt["apply_id"])
            or str(row.get("state") or "") != str(receipt["state"])
            for row in out
        )
    ):
        raise RuntimeError("Checkpoint apply receipt snapshot binding is invalid.")
    for row in out:
        row["checkpoint_snapshot_sha256"] = receipt[
            "checkpoint_snapshot_sha256"
        ]
        row["secondary_effects_state"] = str(receipt["secondary_effects_state"])
        row["secondary_effects_completed_at_utc"] = receipt[
            "secondary_effects_completed_at_utc"
        ]
    return out


def backfill_legacy_apply_receipts(conn: sqlite3.Connection) -> None:
    """Preserve pre-v7 applying/applied rows before their latest-row slot is reused."""

    apply_rows = conn.execute(
        """
        SELECT DISTINCT apply_id
        FROM labeling_session_checkpoints
        WHERE apply_id IS NOT NULL AND state IN ('applying', 'applied');
        """
    ).fetchall()
    for apply_row in apply_rows:
        apply_id = str(apply_row["apply_id"])
        rows = conn.execute(
            """
            SELECT * FROM labeling_session_checkpoints
            WHERE apply_id = ?
            ORDER BY updated_at_utc ASC, roi_idx ASC, checkpoint_id ASC;
            """,
            (apply_id,),
        ).fetchall()
        bindings = {
            (str(row["task_id"]), str(row["component_name"]), str(row["state"]))
            for row in rows
        }
        if len(bindings) != 1:
            continue
        task_id, component_name, state = next(iter(bindings))
        snapshots = [_checkpoint_row(row) for row in rows]
        applied_values = [
            str(row["applied_at_utc"])
            for row in rows
            if row["applied_at_utc"] is not None
        ]
        conn.execute(
            """
            INSERT OR IGNORE INTO labeling_checkpoint_apply_receipts (
                apply_id, task_id, component_name, state, checkpoint_count,
                checkpoints_json, claimed_at_utc, applied_at_utc,
                edit_revision_before, edit_revision_after,
                secondary_effects_state, secondary_effects_completed_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                apply_id,
                task_id,
                component_name,
                state,
                len(snapshots),
                _json_dumps(snapshots),
                min(str(row["updated_at_utc"]) for row in rows),
                min(applied_values) if applied_values else None,
                rows[0]["edit_revision_before"],
                rows[0]["edit_revision_after"],
                "complete" if state == "applied" else "not_ready",
                None,
            ),
        )


def upsert_checkpoint(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    recording_id: str,
    user: str,
    workflow_kind: str,
    target_run_path: str,
    target_edit_revision: int,
    source_rowset_path: str | None,
    roi_idx: int,
    component_name: str,
    payload: Mapping[str, object],
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    now = _utc_now()
    conn.execute("BEGIN IMMEDIATE;")
    try:
        existing = conn.execute(
            """
            SELECT checkpoint_id, state FROM labeling_session_checkpoints
            WHERE task_id = ? AND roi_idx = ? AND component_name = ?;
            """,
            (str(task_id), int(roi_idx), str(component_name)),
        ).fetchone()
        if existing is not None and str(existing["state"]) == "applying":
            raise RuntimeError(
                "This row already has a checkpoint being applied to Zarr. "
                "Wait for that apply to finish before saving this same row again."
            )
        checkpoint_id = (
            str(existing["checkpoint_id"])
            if existing is not None
            else str(uuid.uuid4())
        )
        payload_value = dict(payload)
        metadata_value = dict(metadata or {})
        digest = compatible_snapshot_row_sha256(
            {
                "checkpoint_id": checkpoint_id,
                "task_id": str(task_id),
                "recording_id": str(recording_id),
                "user": str(user),
                "workflow_kind": str(workflow_kind),
                "target_run_path": str(target_run_path),
                "target_edit_revision": int(target_edit_revision),
                "source_rowset_path": (
                    str(source_rowset_path).strip() if source_rowset_path else None
                ),
                "roi_idx": int(roi_idx),
                "component_name": str(component_name),
                "payload": payload_value,
                "metadata": metadata_value,
            }
        )
        cur = conn.execute(
            """
            INSERT INTO labeling_session_checkpoints (
                checkpoint_id, session_id, task_id, recording_id, user, workflow_kind,
                target_run_path, target_edit_revision, source_rowset_path,
                roi_idx, component_name, payload_json, metadata_json,
                snapshot_row_sha256, state,
                created_at_utc, updated_at_utc, applied_at_utc, apply_id,
                edit_revision_before, edit_revision_after
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL, NULL, NULL, NULL)
            ON CONFLICT(task_id, roi_idx, component_name) DO UPDATE SET
                session_id = excluded.session_id,
                recording_id = excluded.recording_id,
                user = excluded.user,
                workflow_kind = excluded.workflow_kind,
                target_run_path = excluded.target_run_path,
                target_edit_revision = excluded.target_edit_revision,
                source_rowset_path = excluded.source_rowset_path,
                payload_json = excluded.payload_json,
                metadata_json = excluded.metadata_json,
                snapshot_row_sha256 = excluded.snapshot_row_sha256,
                state = 'active',
                updated_at_utc = excluded.updated_at_utc,
                applied_at_utc = NULL,
                apply_id = NULL,
                edit_revision_before = NULL,
                edit_revision_after = NULL
            WHERE labeling_session_checkpoints.state != 'applying';
            """,
            (
                checkpoint_id,
                str(session_id),
                str(task_id),
                str(recording_id),
                str(user),
                str(workflow_kind),
                str(target_run_path),
                int(target_edit_revision),
                str(source_rowset_path).strip() if source_rowset_path else None,
                int(roi_idx),
                str(component_name),
                _json_dumps(payload_value),
                _json_dumps(metadata_value),
                digest,
                now,
                now,
            ),
        )
        if int(cur.rowcount or 0) != 1:
            raise RuntimeError(
                "This row already has a checkpoint being applied to Zarr. "
                "Wait for that apply to finish before saving this same row again."
            )
        row = conn.execute(
            """
            SELECT * FROM labeling_session_checkpoints
            WHERE task_id = ? AND roi_idx = ? AND component_name = ? AND state = 'active';
            """,
            (str(task_id), int(roi_idx), str(component_name)),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to persist labeling session checkpoint.")
        out = _checkpoint_row(row)
        conn.commit()
        return out
    except Exception:
        conn.rollback()
        raise


def list_checkpoint_snapshot_descriptors(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    state: str,
    component_name: str,
    limit: int,
) -> list[dict[str, object]]:
    """Read a bounded apply-preview window without loading checkpoint JSON."""

    rows = conn.execute(
        """
        SELECT checkpoint_id, roi_idx, updated_at_utc, state, apply_id,
               snapshot_row_sha256
        FROM labeling_session_checkpoints
        WHERE task_id = ? AND state = ? AND component_name = ?
        ORDER BY updated_at_utc ASC, roi_idx ASC, checkpoint_id ASC
        LIMIT ?;
        """,
        (
            str(task_id),
            str(state),
            str(component_name),
            max(1, int(limit)),
        ),
    ).fetchall()
    descriptors = [dict(row) for row in rows]
    if any(not str(row.get("snapshot_row_sha256") or "") for row in descriptors):
        raise RuntimeError("Checkpoint snapshot descriptor is missing its persisted digest.")
    return descriptors


def count_checkpoints(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    state: str,
    component_name: str,
) -> int:
    """Count every checkpoint in one exact state/component binding."""

    row = conn.execute(
        """
        SELECT COUNT(*) AS n FROM labeling_session_checkpoints
        WHERE task_id = ? AND state = ? AND component_name = ?;
        """,
        (str(task_id), str(state), str(component_name)),
    ).fetchone()
    return int(row["n"] if row is not None else 0)


def claim_checkpoints(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    component_name: str,
    apply_id: str,
    limit: int,
    checkpoint_snapshot_sha256: str | None,
) -> list[dict[str, object]]:
    task_id = str(task_id)
    component_name = str(component_name)
    apply_id = str(apply_id)
    expected_snapshot_sha256 = _expected_snapshot_sha256(
        checkpoint_snapshot_sha256
    )
    conn.execute("BEGIN IMMEDIATE;")
    try:
        receipt = conn.execute(
            "SELECT * FROM labeling_checkpoint_apply_receipts WHERE apply_id = ?;",
            (apply_id,),
        ).fetchone()
        if receipt is not None:
            if str(receipt["task_id"]) != task_id or str(receipt["component_name"]) != component_name:
                raise RuntimeError(
                    "apply_id is already bound to a different task or checkpoint component."
                )
            stored_snapshot_sha256 = receipt["checkpoint_snapshot_sha256"]
            if (
                expected_snapshot_sha256 is not None
                and stored_snapshot_sha256 is not None
                and str(stored_snapshot_sha256) != expected_snapshot_sha256
            ):
                raise RuntimeError(
                    "apply_id is already bound to a different checkpoint snapshot digest."
                )
            if expected_snapshot_sha256 is not None and stored_snapshot_sha256 is None:
                if str(receipt["state"]) != "applying":
                    raise RuntimeError(
                        "Applied checkpoint receipt has no expected snapshot digest binding."
                    )
                conn.execute(
                    """
                    UPDATE labeling_checkpoint_apply_receipts
                    SET checkpoint_snapshot_sha256 = ?
                    WHERE apply_id = ? AND state = 'applying'
                      AND checkpoint_snapshot_sha256 IS NULL;
                    """,
                    (expected_snapshot_sha256, apply_id),
                )
            snapshots = _receipt_snapshots(receipt)
            if str(receipt["state"]) == "applied":
                conn.rollback()
                return []
            if str(receipt["state"]) != "applying":
                raise RuntimeError("Checkpoint apply receipt has an invalid state.")
            ids = [str(row.get("checkpoint_id") or "") for row in snapshots]
            if not ids or any(not value for value in ids):
                raise RuntimeError("Checkpoint apply receipt contains an invalid snapshot.")
            placeholders = ", ".join("?" for _ in ids)
            current = conn.execute(
                f"""
                SELECT * FROM labeling_session_checkpoints
                WHERE checkpoint_id IN ({placeholders})
                  AND task_id = ? AND component_name = ?
                  AND apply_id = ? AND state = 'applying'
                ORDER BY updated_at_utc ASC, roi_idx ASC, checkpoint_id ASC;
                """,
                [*ids, task_id, component_name, apply_id],
            ).fetchall()
            if len(current) != len(ids):
                raise RuntimeError("Checkpoint apply snapshot lost ownership and requires reconciliation.")
            current_by_id = {
                str(row["checkpoint_id"]): _checkpoint_row(row) for row in current
            }
            out = [current_by_id[checkpoint_id] for checkpoint_id in ids]
            conn.commit()
            return out

        legacy_rows = conn.execute(
            """
            SELECT * FROM labeling_session_checkpoints
            WHERE apply_id = ?
            ORDER BY updated_at_utc ASC, roi_idx ASC, checkpoint_id ASC;
            """,
            (apply_id,),
        ).fetchall()
        if legacy_rows:
            bindings = {
                (str(row["task_id"]), str(row["component_name"]), str(row["state"]))
                for row in legacy_rows
            }
            if len(bindings) != 1 or next(iter(bindings))[:2] != (task_id, component_name):
                raise RuntimeError(
                    "apply_id has an inconsistent or conflicting legacy checkpoint binding."
                )
            backfill_legacy_apply_receipts(conn)
            receipt = conn.execute(
                "SELECT * FROM labeling_checkpoint_apply_receipts WHERE apply_id = ?;",
                (apply_id,),
            ).fetchone()
            if receipt is None:
                raise RuntimeError("Legacy checkpoint apply receipt migration failed.")
            state = str(receipt["state"])
            if expected_snapshot_sha256 is not None:
                if state != "applying":
                    raise RuntimeError(
                        "Applied checkpoint receipt has no expected snapshot digest binding."
                    )
                conn.execute(
                    """
                    UPDATE labeling_checkpoint_apply_receipts
                    SET checkpoint_snapshot_sha256 = ?
                    WHERE apply_id = ? AND state = 'applying'
                      AND checkpoint_snapshot_sha256 IS NULL;
                    """,
                    (expected_snapshot_sha256, apply_id),
                )
            conn.commit()
            return [_checkpoint_row(row) for row in legacy_rows] if state == "applying" else []

        if conn.execute(
            """
            SELECT 1 FROM labeling_session_checkpoints
            WHERE task_id = ? AND component_name = ? AND state = 'applying' LIMIT 1;
            """,
            (task_id, component_name),
        ).fetchone() is not None:
            conn.rollback()
            return []

        id_rows = conn.execute(
            """
            SELECT checkpoint_id FROM labeling_session_checkpoints
            WHERE task_id = ? AND component_name = ? AND state = 'active'
            ORDER BY updated_at_utc ASC, roi_idx ASC, checkpoint_id ASC LIMIT ?;
            """,
            (task_id, component_name, max(1, int(limit))),
        ).fetchall()
        ids = [str(row["checkpoint_id"]) for row in id_rows]
        if not ids:
            conn.rollback()
            return []
        placeholders = ", ".join("?" for _ in ids)
        cur = conn.execute(
            f"""
            UPDATE labeling_session_checkpoints
            SET state = 'applying', apply_id = ?
            WHERE checkpoint_id IN ({placeholders})
              AND task_id = ? AND component_name = ? AND state = 'active';
            """,
            [apply_id, *ids, task_id, component_name],
        )
        if int(cur.rowcount or 0) != len(ids):
            raise RuntimeError("Failed to claim the complete checkpoint snapshot.")
        rows = conn.execute(
            f"""
            SELECT * FROM labeling_session_checkpoints
            WHERE checkpoint_id IN ({placeholders})
              AND task_id = ? AND component_name = ?
              AND apply_id = ? AND state = 'applying'
            ORDER BY updated_at_utc ASC, roi_idx ASC, checkpoint_id ASC;
            """,
            [*ids, task_id, component_name, apply_id],
        ).fetchall()
        if len(rows) != len(ids):
            raise RuntimeError("Claimed checkpoint snapshot could not be read back completely.")
        rows_by_id = {
            str(row["checkpoint_id"]): _checkpoint_row(row) for row in rows
        }
        snapshots = [rows_by_id[checkpoint_id] for checkpoint_id in ids]
        now = _utc_now()
        conn.execute(
            """
            INSERT INTO labeling_checkpoint_apply_receipts (
                apply_id, task_id, component_name, state, checkpoint_count,
                checkpoints_json, claimed_at_utc, applied_at_utc,
                edit_revision_before, edit_revision_after,
                checkpoint_snapshot_sha256, secondary_effects_state,
                secondary_effects_completed_at_utc
            ) VALUES (?, ?, ?, 'applying', ?, ?, ?, NULL, NULL, NULL, ?, 'not_ready', NULL);
            """,
            (
                apply_id,
                task_id,
                component_name,
                len(snapshots),
                _json_dumps(snapshots),
                now,
                expected_snapshot_sha256,
            ),
        )
        conn.commit()
        return snapshots
    except Exception:
        conn.rollback()
        raise


def release_checkpoints(conn: sqlite3.Connection, *, task_id: str, apply_id: str) -> int:
    now = _utc_now()
    task_id = str(task_id)
    apply_id = str(apply_id)
    conn.execute("BEGIN IMMEDIATE;")
    try:
        receipt = conn.execute(
            "SELECT * FROM labeling_checkpoint_apply_receipts WHERE apply_id = ?;",
            (apply_id,),
        ).fetchone()
        if receipt is not None and str(receipt["task_id"]) != task_id:
            raise RuntimeError("apply_id is bound to a different checkpoint task.")
        if receipt is not None and str(receipt["state"]) == "applied":
            still_applying = conn.execute(
                """
                SELECT COUNT(*) AS n FROM labeling_session_checkpoints
                WHERE task_id = ? AND apply_id = ? AND state = 'applying';
                """,
                (task_id, apply_id),
            ).fetchone()
            if int(still_applying["n"] or 0) != 0:
                raise RuntimeError(
                    "Applied checkpoint receipt still has applying rows and requires reconciliation."
                )
            conn.commit()
            return int(receipt["released_checkpoint_count"] or 0)
        cur = conn.execute(
            """
            UPDATE labeling_session_checkpoints
            SET state = 'active', apply_id = NULL, updated_at_utc = ?
            WHERE task_id = ? AND apply_id = ? AND state = 'applying';
            """,
            (now, task_id, apply_id),
        )
        released = int(cur.rowcount or 0)
        if receipt is not None and str(receipt["state"]) == "applying":
            if released != int(receipt["checkpoint_count"]):
                raise RuntimeError(
                    "Checkpoint apply snapshot lost ownership and cannot be safely released."
                )
            conn.execute(
                "DELETE FROM labeling_checkpoint_apply_receipts WHERE apply_id = ?;",
                (apply_id,),
            )
        conn.commit()
        return released
    except Exception:
        conn.rollback()
        raise


def get_applied_checkpoints(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    apply_id: str,
) -> list[dict[str, object]]:
    receipt = conn.execute(
        "SELECT * FROM labeling_checkpoint_apply_receipts WHERE apply_id = ?;",
        (str(apply_id),),
    ).fetchone()
    if receipt is not None:
        if str(receipt["task_id"]) != str(task_id) or str(receipt["state"]) != "applied":
            return []
        return _receipt_snapshots(receipt)
    rows = conn.execute(
        """
        SELECT * FROM labeling_session_checkpoints
        WHERE task_id = ? AND apply_id = ? AND state = 'applied'
        ORDER BY applied_at_utc ASC, roi_idx ASC, component_name ASC;
        """,
        (str(task_id), str(apply_id)),
    ).fetchall()
    return [_checkpoint_row(row) for row in rows]


def list_pending_apply_effects(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    component_name: str,
    limit: int,
) -> list[dict[str, object]]:
    """List bounded applied-receipt summaries awaiting audit/registry effects."""

    rows = conn.execute(
        """
        SELECT apply_id, task_id, component_name, checkpoint_count,
               claimed_at_utc, applied_at_utc, edit_revision_before,
               edit_revision_after, released_checkpoint_count,
               checkpoint_snapshot_sha256,
               secondary_effects_state, secondary_effects_completed_at_utc
        FROM labeling_checkpoint_apply_receipts
        WHERE task_id = ? AND component_name = ? AND state = 'applied'
          AND secondary_effects_state = 'pending'
        ORDER BY applied_at_utc ASC, apply_id ASC
        LIMIT ?;
        """,
        (str(task_id), str(component_name), max(1, int(limit))),
    ).fetchall()
    return [dict(row) for row in rows]


def count_pending_apply_effects(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    component_name: str | None,
) -> int:
    sql = [
        """
        SELECT COUNT(*) AS n FROM labeling_checkpoint_apply_receipts
        WHERE task_id = ? AND state = 'applied'
          AND secondary_effects_state = 'pending'
        """
    ]
    params: list[object] = [str(task_id)]
    if component_name is not None:
        sql.append("AND component_name = ?")
        params.append(str(component_name))
    row = conn.execute(" ".join(sql), params).fetchone()
    return int(row["n"] if row is not None else 0)


def mark_apply_effects_complete(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    component_name: str,
    apply_id: str,
) -> bool:
    """Strictly and idempotently finalize one applied receipt's side effects."""

    task_id = str(task_id)
    component_name = str(component_name)
    apply_id = str(apply_id)
    conn.execute("BEGIN IMMEDIATE;")
    try:
        receipt = conn.execute(
            "SELECT * FROM labeling_checkpoint_apply_receipts WHERE apply_id = ?;",
            (apply_id,),
        ).fetchone()
        if receipt is None:
            raise RuntimeError("Checkpoint apply receipt does not exist.")
        if str(receipt["task_id"]) != task_id:
            raise RuntimeError("apply_id is bound to a different checkpoint task.")
        if str(receipt["component_name"]) != component_name:
            raise RuntimeError("apply_id is bound to a different checkpoint component.")
        if str(receipt["state"]) != "applied":
            raise RuntimeError("Checkpoint apply receipt is not applied.")
        effects_state = str(receipt["secondary_effects_state"])
        if effects_state == "complete":
            conn.commit()
            return False
        if effects_state != "pending":
            raise RuntimeError("Checkpoint apply receipt effects are not pending.")
        cur = conn.execute(
            """
            UPDATE labeling_checkpoint_apply_receipts
            SET secondary_effects_state = 'complete',
                secondary_effects_completed_at_utc = ?
            WHERE apply_id = ? AND task_id = ? AND component_name = ?
              AND state = 'applied' AND secondary_effects_state = 'pending';
            """,
            (_utc_now(), apply_id, task_id, component_name),
        )
        if int(cur.rowcount or 0) != 1:
            raise RuntimeError("Checkpoint apply receipt effects ownership changed.")
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def mark_checkpoints_applied(
    conn: sqlite3.Connection,
    *,
    checkpoint_ids: Sequence[str],
    apply_id: str,
    edit_revision_before: int,
    edit_revision_after: int,
    require_secondary_effects: bool,
) -> int:
    ids = list(dict.fromkeys(str(value) for value in checkpoint_ids if str(value)))
    if not ids:
        return 0
    now = _utc_now()
    secondary_effects_state = "pending" if require_secondary_effects else "complete"
    secondary_effects_completed_at_utc = None if require_secondary_effects else now
    apply_id = str(apply_id)
    placeholders = ", ".join("?" for _ in ids)
    conn.execute("BEGIN IMMEDIATE;")
    try:
        receipt = conn.execute(
            "SELECT * FROM labeling_checkpoint_apply_receipts WHERE apply_id = ?;",
            (apply_id,),
        ).fetchone()
        if receipt is None or str(receipt["state"]) != "applying":
            raise RuntimeError(
                "Cannot finalize checkpoints without an owned applying checkpoint receipt."
            )
        rows = conn.execute(
            f"""
            SELECT * FROM labeling_session_checkpoints
            WHERE checkpoint_id IN ({placeholders}) AND apply_id = ? AND state = 'applying'
            ORDER BY roi_idx ASC, component_name ASC, checkpoint_id ASC;
            """,
            [*ids, apply_id],
        ).fetchall()
        if len(rows) != len(ids):
            raise RuntimeError(
                "Cannot finalize checkpoints: all requested owned applying checkpoints are required."
            )
        if any(
            str(row["task_id"]) != str(receipt["task_id"])
            or str(row["component_name"]) != str(receipt["component_name"])
            for row in rows
        ):
            raise RuntimeError(
                "Cannot finalize checkpoints outside the apply receipt task/component binding."
            )
        receipt_ids = {
            str(row.get("checkpoint_id") or "") for row in _receipt_snapshots(receipt)
        }
        if any(value not in receipt_ids for value in ids):
            raise RuntimeError("Cannot finalize checkpoints outside the claimed apply snapshot.")
        released_ids = sorted(receipt_ids.difference(ids))
        finalized = []
        for row in rows:
            snapshot = _checkpoint_row(row)
            snapshot.update(
                state="applied",
                apply_id=apply_id,
                edit_revision_before=int(edit_revision_before),
                edit_revision_after=int(edit_revision_after),
                applied_at_utc=now,
                updated_at_utc=now,
            )
            finalized.append(snapshot)
        receipt_cur = conn.execute(
            """
            UPDATE labeling_checkpoint_apply_receipts
            SET state = 'applied', checkpoint_count = ?, checkpoints_json = ?,
                applied_at_utc = ?, edit_revision_before = ?, edit_revision_after = ?,
                released_checkpoint_count = ?, secondary_effects_state = ?,
                secondary_effects_completed_at_utc = ?
            WHERE apply_id = ? AND state = 'applying';
            """,
            (
                len(finalized),
                _json_dumps(finalized),
                now,
                int(edit_revision_before),
                int(edit_revision_after),
                len(released_ids),
                secondary_effects_state,
                secondary_effects_completed_at_utc,
                apply_id,
            ),
        )
        if int(receipt_cur.rowcount or 0) != 1:
            raise RuntimeError("Checkpoint apply receipt ownership changed during finalization.")
        cur = conn.execute(
            f"""
            UPDATE labeling_session_checkpoints
            SET state = 'applied', edit_revision_before = ?, edit_revision_after = ?,
                applied_at_utc = ?, updated_at_utc = ?
            WHERE checkpoint_id IN ({placeholders}) AND apply_id = ? AND state = 'applying';
            """,
            [int(edit_revision_before), int(edit_revision_after), now, now, *ids, apply_id],
        )
        if int(cur.rowcount or 0) != len(ids):
            raise RuntimeError(
                "Cannot finalize checkpoints: all requested owned applying checkpoints are required."
            )
        if released_ids:
            released_placeholders = ", ".join("?" for _ in released_ids)
            released_cur = conn.execute(
                f"""
                UPDATE labeling_session_checkpoints
                SET state = 'active', apply_id = NULL, updated_at_utc = ?
                WHERE checkpoint_id IN ({released_placeholders})
                  AND apply_id = ? AND state = 'applying';
                """,
                [now, *released_ids, apply_id],
            )
            if int(released_cur.rowcount or 0) != len(released_ids):
                raise RuntimeError(
                    "Cannot release every unapplied row from the claimed checkpoint snapshot."
                )
        conn.commit()
        return len(ids)
    except Exception:
        conn.rollback()
        raise
