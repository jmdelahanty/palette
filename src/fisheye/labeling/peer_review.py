"""Read-only peer review of applied labels, and advisory frame flags.

Every active labeling user may look at any labeler's applied mask and keypoint
rows and flag a frame with a note. A row is visible to peers once it has been
applied at least once. A later save resets the row's checkpoint to ``active``
and clears its apply fields, so the history is read from the applied apply
receipts as well as from the current checkpoints. The label shown is the
training Zarr's, which is still the applied value; a saved, unapplied payload is
never returned.

Reads use admin inspect's read-only machinery: a ``mode=ro`` SQLite connection
and Zarr opened with ``mode="r"``. Flags are written through the store's normal
connection. They change no label, checkpoint, receipt, or Apply state, and they
are not part of any digest, export, or publication.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .admin_inspect import InspectError, _connect, _keypoint_sources, _mask_sources, _task

PEER_TASK_KINDS = ("subject_mask_component", "keypoints")
FLAG_STATES = ("open", "resolved")
FLAG_NOTE_MAX_CHARS = 1000

FRAME_FLAGS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS labeling_frame_flags (
    flag_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES labeling_tasks(task_id) ON DELETE CASCADE,
    recording_id TEXT NOT NULL,
    roi_idx INTEGER NOT NULL,
    flagged_by TEXT NOT NULL,
    note TEXT NOT NULL,
    applied_checkpoint_id TEXT,
    apply_id TEXT,
    state TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('open', 'resolved')),
    created_at_utc TEXT NOT NULL,
    resolved_by TEXT,
    resolved_at_utc TEXT
);
CREATE INDEX IF NOT EXISTS idx_labeling_frame_flags_task
    ON labeling_frame_flags(task_id, state, roi_idx);
"""


class FlagError(ValueError):
    """The flag request is invalid."""


class FlagForbidden(PermissionError):
    """The user may not change this flag."""


def _flags_table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'labeling_frame_flags';"
    ).fetchone()
    return row is not None


def _peer_task(conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    task = _task(conn, task_id)
    if task["workflow_kind"] not in PEER_TASK_KINDS:
        raise InspectError(f"Peer review supports mask and keypoint tasks, not {task['workflow_kind']!r}.")
    return task


def _applied_versions(conn: sqlite3.Connection, task_id: str, roi_idx: int | None = None) -> dict[int, dict[str, Any]]:
    """The latest applied version of each row that has ever been applied."""

    versions: dict[int, dict[str, Any]] = {}

    def keep(roi: object, version: dict[str, Any]) -> None:
        if roi is None:
            return
        roi = int(roi)
        current = versions.get(roi)
        if current is None or str(version["applied_at_utc"] or "") >= str(current["applied_at_utc"] or ""):
            versions[roi] = version

    roi_filter = "" if roi_idx is None else " AND json_extract(j.value, '$.roi_idx') = ?"
    receipt_params: tuple = (task_id,) if roi_idx is None else (task_id, int(roi_idx))
    for row in conn.execute(
        f"""
        SELECT json_extract(j.value, '$.roi_idx') AS roi_idx,
               json_extract(j.value, '$.checkpoint_id') AS checkpoint_id,
               json_extract(j.value, '$.user') AS labeler,
               r.apply_id, r.applied_at_utc
        FROM labeling_checkpoint_apply_receipts r, json_each(r.checkpoints_json) j
        WHERE r.task_id = ? AND r.state = 'applied'{roi_filter};
        """,
        receipt_params,
    ):
        keep(row["roi_idx"], {
            "applied_checkpoint_id": row["checkpoint_id"], "apply_id": row["apply_id"],
            "applied_at_utc": row["applied_at_utc"], "labeler": row["labeler"],
        })
    checkpoint_filter = "" if roi_idx is None else " AND roi_idx = ?"
    for row in conn.execute(
        f"""
        SELECT roi_idx, checkpoint_id, user, apply_id, applied_at_utc FROM labeling_session_checkpoints
        WHERE task_id = ? AND state = 'applied'{checkpoint_filter};
        """,
        receipt_params,
    ):
        keep(row["roi_idx"], {
            "applied_checkpoint_id": row["checkpoint_id"], "apply_id": row["apply_id"],
            "applied_at_utc": row["applied_at_utc"], "labeler": row["user"],
        })
    return versions


def _flag_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _flags(conn: sqlite3.Connection, task_id: str, roi_idx: int | None = None) -> list[dict[str, Any]]:
    if not _flags_table_exists(conn):
        return []
    sql = "SELECT * FROM labeling_frame_flags WHERE task_id = ?"
    params: list[object] = [task_id]
    if roi_idx is not None:
        sql += " AND roi_idx = ?"
        params.append(int(roi_idx))
    return [_flag_dict(r) for r in conn.execute(sql + " ORDER BY created_at_utc, flag_id;", params)]


def open_flag_counts(store_path: Path, task_ids: list[str] | None = None) -> dict[str, int]:
    """Open flags per task, for the peer task list and the labeler queue."""

    conn = _connect(store_path)
    try:
        if not _flags_table_exists(conn):
            return {}
        sql = "SELECT task_id, COUNT(*) AS n FROM labeling_frame_flags WHERE state = 'open'"
        params: list[object] = []
        if task_ids is not None:
            if not task_ids:
                return {}
            sql += f" AND task_id IN ({', '.join('?' for _ in task_ids)})"
            params.extend(task_ids)
        return {r["task_id"]: int(r["n"]) for r in conn.execute(sql + " GROUP BY task_id;", params)}
    finally:
        conn.close()


def peer_tasks(store_path: Path) -> dict[str, Any]:
    """Mask and keypoint tasks with owner, applied rows, and open flags.

    ``applied_rows`` counts rows whose latest checkpoint is applied (one cheap
    query). The task view also lists rows applied and then edited again, which
    needs the apply receipts.
    """

    flag_counts = open_flag_counts(store_path)
    conn = _connect(store_path)
    try:
        applied = {
            r["task_id"]: int(r["n"])
            for r in conn.execute(
                "SELECT task_id, COUNT(*) AS n FROM labeling_session_checkpoints WHERE state = 'applied' GROUP BY task_id;"
            )
        }
        tasks = []
        for row in conn.execute(
            f"""
            SELECT t.task_id, t.title, t.recording_id, t.workflow_kind, t.component_name, t.state, a.assignee_user
            FROM labeling_tasks t LEFT JOIN recording_assignments a ON a.recording_id = t.recording_id
            WHERE t.workflow_kind IN ({', '.join('?' for _ in PEER_TASK_KINDS)})
            ORDER BY a.assignee_user, t.recording_id, t.component_name;
            """,
            PEER_TASK_KINDS,
        ):
            tasks.append({
                "task_id": row["task_id"],
                "title": row["title"] or row["task_id"],
                "recording_id": row["recording_id"],
                "workflow_kind": row["workflow_kind"],
                "component_name": row["component_name"],
                "state": row["state"],
                "assignee": row["assignee_user"],
                "applied_rows": applied.get(row["task_id"], 0),
                "open_flag_count": flag_counts.get(row["task_id"], 0),
            })
        return {"ok": True, "read_only": True, "tasks": tasks}
    finally:
        conn.close()


def peer_task(store_path: Path, task_id: str) -> dict[str, Any]:
    """The task's rows that have been applied at least once, with open flags."""

    conn = _connect(store_path)
    try:
        task = _peer_task(conn, task_id)
        versions = _applied_versions(conn, task_id)
        open_by_row: dict[int, int] = {}
        for flag in _flags(conn, task_id):
            if flag["state"] == "open":
                open_by_row[int(flag["roi_idx"])] = open_by_row.get(int(flag["roi_idx"]), 0) + 1
        rows = [
            {
                "roi_idx": roi,
                "labeler": version["labeler"],
                "applied_at_utc": version["applied_at_utc"],
                "open_flag_count": open_by_row.get(roi, 0),
            }
            for roi, version in sorted(versions.items())
        ]
        return {
            "ok": True, "read_only": True, "task_id": task_id,
            "recording_id": task["recording_id"], "workflow_kind": task["workflow_kind"],
            "component_name": task.get("component_name"), "owner": task.get("assignee_user"),
            "rows": rows,
        }
    finally:
        conn.close()


def peer_row(store_path: Path, task_id: str, roi_idx: int) -> dict[str, Any]:
    """One applied row's image, applied label from the Zarr, and its flags."""

    from .web_responses import _raw_array_payload

    conn = _connect(store_path)
    try:
        task = _peer_task(conn, task_id)
        roi_idx = int(roi_idx)
        version = _applied_versions(conn, task_id, roi_idx).get(roi_idx)
        if version is None:
            raise InspectError("This row has no applied work to review.")
        result: dict[str, Any] = {
            "ok": True, "read_only": True, "task_id": task_id, "roi_idx": roi_idx,
            "recording_id": task["recording_id"], "workflow_kind": task["workflow_kind"],
            "owner": task.get("assignee_user"), **version,
            "flags": _flags(conn, task_id, roi_idx),
        }
        if task["workflow_kind"] == "subject_mask_component":
            run, comp_idx, images, _run_name, _crop_name = _mask_sources(task)
            if not 0 <= roi_idx < int(run["masks_roi"].shape[0]):
                raise InspectError("roi_idx is out of range.")
            result.update({
                "component_name": task.get("component_name"),
                "image": _raw_array_payload(np.asarray(images[roi_idx], dtype=np.uint8)),
                "applied_mask": _raw_array_payload((np.asarray(run["masks_roi"][roi_idx, comp_idx]) > 0).astype(np.uint8)),
            })
        else:
            refined, crop, _run_name, _crop_name = _keypoint_sources(task)
            if not 0 <= roi_idx < int(refined["keypoints_roi"].shape[0]):
                raise InspectError("roi_idx is out of range.")
            points = np.asarray(refined["keypoints_roi"][roi_idx], dtype=float)
            labels = [str(v) for v in (refined.attrs.get("keypoint_labels") or [])] or [
                str(i + 1) for i in range(int(points.shape[0]))
            ]
            result.update({
                "labels": labels,
                "image": _raw_array_payload(np.asarray(crop["roi_images"][roi_idx], dtype=np.uint8)),
                "applied_points": [[float(x) if np.isfinite(x) else None, float(y) if np.isfinite(y) else None]
                                   for x, y in points],
            })
        return result
    finally:
        conn.close()


def add_frame_flag(store: Any, *, task_id: str, roi_idx: int, flagged_by: str, note: str) -> dict[str, Any]:
    """Flag one applied row with an advisory note."""

    from .assignment_store import utc_now

    text = str(note or "").strip()
    if not text:
        raise FlagError("A flag needs a note.")
    if len(text) > FLAG_NOTE_MAX_CHARS:
        raise FlagError(f"A flag note is limited to {FLAG_NOTE_MAX_CHARS} characters.")
    author = str(flagged_by or "").strip()
    if not author:
        raise FlagError("flagged_by is required.")
    store.initialize()
    conn = _connect(store.path)
    try:
        task = _peer_task(conn, str(task_id))
        version = _applied_versions(conn, str(task_id), int(roi_idx)).get(int(roi_idx))
    finally:
        conn.close()
    if version is None:
        raise FlagError("Only applied rows can be flagged.")
    flag_id = str(uuid.uuid4())
    store.conn.execute(
        """
        INSERT INTO labeling_frame_flags (
            flag_id, task_id, recording_id, roi_idx, flagged_by, note,
            applied_checkpoint_id, apply_id, state, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?);
        """,
        (flag_id, str(task_id), str(task["recording_id"]), int(roi_idx), author, text,
         version["applied_checkpoint_id"], version["apply_id"], utc_now()),
    )
    store.conn.commit()
    return get_frame_flag(store, flag_id)


def get_frame_flag(store: Any, flag_id: str) -> dict[str, Any]:
    row = store.conn.execute("SELECT * FROM labeling_frame_flags WHERE flag_id = ?;", (str(flag_id),)).fetchone()
    if row is None:
        raise KeyError(f"Unknown flag: {flag_id}")
    return _flag_dict(row)


def resolve_frame_flag(store: Any, *, flag_id: str, user: str, is_admin: bool = False) -> dict[str, Any]:
    """Resolve a flag. Allowed for the task owner, the flag's author, or an admin."""

    from .assignment_store import utc_now

    store.initialize()
    flag = get_frame_flag(store, flag_id)
    owner_row = store.conn.execute(
        """
        SELECT a.assignee_user FROM labeling_tasks t
        LEFT JOIN recording_assignments a ON a.recording_id = t.recording_id
        WHERE t.task_id = ?;
        """,
        (flag["task_id"],),
    ).fetchone()
    owner = owner_row["assignee_user"] if owner_row is not None else None
    actor = str(user or "").strip()
    if not (is_admin or (actor and actor in {owner, flag["flagged_by"]})):
        raise FlagForbidden("Only the task owner, the flag's author, or an admin can resolve a flag.")
    if flag["state"] == "resolved":
        return flag
    store.conn.execute(
        """
        UPDATE labeling_frame_flags SET state = 'resolved', resolved_by = ?, resolved_at_utc = ?
        WHERE flag_id = ? AND state = 'open';
        """,
        (actor, utc_now(), str(flag_id)),
    )
    store.conn.commit()
    return get_frame_flag(store, flag_id)


__all__ = [
    "FLAG_NOTE_MAX_CHARS",
    "FLAG_STATES",
    "FRAME_FLAGS_SCHEMA_SQL",
    "FlagError",
    "FlagForbidden",
    "add_frame_flag",
    "get_frame_flag",
    "open_flag_counts",
    "peer_row",
    "peer_task",
    "peer_tasks",
    "resolve_frame_flag",
]
