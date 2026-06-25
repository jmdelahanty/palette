"""SQLite store for recording-level web labeling assignments and tasks."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


STORE_ENV_VAR = "PALETTE_LABELING_STORE_PATH"
DEFAULT_STORE_PATH = "~/.palette/labeling_work.sqlite"
SCHEMA_VERSION = 3
LABELER_START_TASK_STATES = ("pending", "in_progress")
USER_SUMMARY_REDACT_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w])(?:/[^\s,;:'\"<>]+)+")
USER_SUMMARY_REDACT_ZARR_TOKEN_RE = re.compile(r"(?<![\w./-])[\w./-]*\.zarr(?:[/\w.-]*)?")
USER_SUMMARY_SAFE_LOCAL_URL_PATHS = ("/work", "/datasets", "/identity")
USER_SUMMARY_SAFE_LOCAL_URL_PREFIXES = ("/api/sessions/",)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_store_path() -> Path:
    return Path(os.environ.get(STORE_ENV_VAR, DEFAULT_STORE_PATH)).expanduser()


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_loads(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _row_to_dict(row: sqlite3.Row | Mapping[str, object]) -> dict[str, object]:
    out = dict(row)
    for key in ("scope_json", "target_json", "before_json", "after_json"):
        if key in out:
            decoded = _json_loads(out.get(key))
            out[key[:-5] if key.endswith("_json") else key] = decoded
            out.pop(key, None)
    return out


def _is_user_summary_sensitive_key(key: object) -> bool:
    text = str(key).strip().lower()
    return (
        text in {
            "scope",
            "scope_json",
            "zarr_path",
            "registry_path",
            "review_proxy_manifest",
            "analysis_zarr",
            "training_zarr",
            "promote_training_zarr",
        }
        or text.endswith("_path")
        or text.endswith("_zarr")
    )


def _is_user_summary_safe_local_url(value: str) -> bool:
    text = value.strip()
    if text != value:
        return False
    if ".zarr" in text:
        return False
    if any(ch.isspace() for ch in text):
        return False
    if text.startswith(USER_SUMMARY_SAFE_LOCAL_URL_PREFIXES):
        return True
    return any(
        text == path or text.startswith(f"{path}?") or text.startswith(f"{path}#") or text.startswith(f"{path}/")
        for path in USER_SUMMARY_SAFE_LOCAL_URL_PATHS
    )


def _redact_user_summary_value(value: object) -> object:
    if isinstance(value, Mapping):
        redacted_fields: list[str] = []
        out: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_user_summary_sensitive_key(key):
                redacted_fields.append(key_text)
                continue
            out[key_text] = _redact_user_summary_value(item)
        if redacted_fields:
            out["redacted_fields"] = sorted(set(redacted_fields))
        return out
    if isinstance(value, list):
        return [_redact_user_summary_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_user_summary_value(item) for item in value]
    if isinstance(value, str):
        if _is_user_summary_safe_local_url(value):
            return value
        redacted = USER_SUMMARY_REDACT_ABSOLUTE_PATH_RE.sub("[redacted_path]", value)
        redacted = USER_SUMMARY_REDACT_ZARR_TOKEN_RE.sub("[redacted_zarr_path]", redacted)
        return redacted
    return value


def _redact_user_summary_row(row: Mapping[str, object]) -> dict[str, object]:
    redacted = _redact_user_summary_value(row)
    assert isinstance(redacted, dict)
    return redacted


def _no_open_task_summary(recording: Mapping[str, object]) -> tuple[str, str] | None:
    if int(recording.get("startable_task_count") or 0) > 0:
        return None
    total = int(recording.get("total_task_count") or 0)
    complete = int(recording.get("complete_task_count") or 0)
    incomplete = int(recording.get("incomplete_task_count") or 0)
    if total > 0 and complete >= total:
        return (
            "all_tasks_complete",
            "All tasks for this recording are complete. Ask the operator before reopening or continuing work.",
        )
    if incomplete > 0:
        return (
            "non_startable_task_state",
            "This recording has assigned tasks, but none are in a startable labeling state. Ask the operator to inspect task state before labeling.",
        )
    if total > 0:
        return (
            "no_open_tasks_in_current_summary",
            "This recording is assigned to you, but no open tasks are currently available. Ask the operator to inspect the batch if you expected work here.",
        )
    return (
        "tasks_not_generated",
        "This recording is assigned to you, but no browser-labeling tasks have been generated yet. If you expected work here, ask the operator to generate tasks or inspect the batch.",
    )


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class SessionLease:
    session_id: str
    task_id: str
    recording_id: str
    user: str
    expires_at_utc: str
    superseded_session_ids: tuple[str, ...] = ()


class LabelingStore(AbstractContextManager["LabelingStore"]):
    """Small sidecar store for web labeling assignment state.

    This deliberately lives outside the main Palette registry schema for the
    first slice. It can be merged into registry migrations later once the task
    model stabilizes.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_store_path()
        self._conn: sqlite3.Connection | None = None
        self._local = threading.local()
        self._connection_lock = threading.RLock()
        self._connections: list[sqlite3.Connection] = []

    @property
    def conn(self) -> sqlite3.Connection:
        if str(self.path) == ":memory:":
            with self._connection_lock:
                if self._conn is None:
                    self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
                    self._conn.row_factory = sqlite3.Row
                    self._conn.execute("PRAGMA foreign_keys = ON;")
                    self._conn.execute("PRAGMA busy_timeout = 30000;")
                return self._conn
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            with self._connection_lock:
                self._connections.append(conn)
            self._local.conn = conn
        return conn

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def close(self) -> None:
        with self._connection_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            for conn in self._connections:
                conn.close()
            self._connections.clear()
            if hasattr(self._local, "conn"):
                delattr(self._local, "conn")

    def backup_to(self, destination: str | Path, *, overwrite: bool = False) -> dict[str, object]:
        """Write a SQLite-consistent backup of the sidecar store."""

        self.initialize()
        backup_path = Path(destination).expanduser()
        if backup_path.exists() and not overwrite:
            raise FileExistsError(f"Backup destination already exists: {backup_path}")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if backup_path.exists() and overwrite:
            backup_path.unlink()
        with self._connection_lock:
            with sqlite3.connect(str(backup_path)) as backup_conn:
                self.conn.backup(backup_conn)
        return {
            "source_path": str(self.path),
            "backup_path": str(backup_path),
            "overwrite": bool(overwrite),
        }

    def initialize(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS labeling_schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recording_assignments (
                recording_id TEXT PRIMARY KEY,
                assignee_user TEXT NOT NULL,
                assigned_by TEXT,
                assigned_at_utc TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS labeling_tasks (
                task_id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                workflow_kind TEXT NOT NULL,
                dataset_id TEXT,
                zarr_use TEXT,
                stage_group TEXT,
                run_name TEXT,
                component_name TEXT,
                title TEXT,
                scope_json TEXT,
                state TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                completed_at_utc TEXT
            );

            CREATE TABLE IF NOT EXISTS labeling_sessions (
                session_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                user TEXT NOT NULL,
                workflow_kind TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                expires_at_utc TEXT NOT NULL,
                last_seen_at_utc TEXT,
                closed_at_utc TEXT,
                client_label TEXT,
                FOREIGN KEY(task_id) REFERENCES labeling_tasks(task_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS labeling_task_events (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                user TEXT NOT NULL,
                event_type TEXT NOT NULL,
                target_json TEXT,
                before_json TEXT,
                after_json TEXT,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES labeling_tasks(task_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS labeling_assignment_events (
                event_id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                actor_user TEXT,
                event_type TEXT NOT NULL,
                before_json TEXT,
                after_json TEXT,
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS labeling_task_definition_events (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                actor_user TEXT,
                event_type TEXT NOT NULL,
                before_json TEXT,
                after_json TEXT,
                created_at_utc TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_labeling_assignments_assignee
                ON recording_assignments(assignee_user, status);
            CREATE INDEX IF NOT EXISTS idx_labeling_tasks_recording
                ON labeling_tasks(recording_id, state, priority DESC);
            CREATE INDEX IF NOT EXISTS idx_labeling_sessions_task
                ON labeling_sessions(task_id, user, closed_at_utc, expires_at_utc);
            CREATE INDEX IF NOT EXISTS idx_labeling_events_task
                ON labeling_task_events(task_id, created_at_utc);
            CREATE INDEX IF NOT EXISTS idx_labeling_assignment_events_recording
                ON labeling_assignment_events(recording_id, created_at_utc);
            CREATE INDEX IF NOT EXISTS idx_labeling_task_definition_events_task
                ON labeling_task_definition_events(task_id, created_at_utc);
            """
        )
        cur.execute(
            """
            INSERT INTO labeling_schema_meta(key, value)
            VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value;
            """,
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def assignment_schema_integrity(self) -> dict[str, object]:
        """Return structural evidence for recording assignment ownership rules."""

        self.initialize()
        rows = self.conn.execute("PRAGMA table_info(recording_assignments);").fetchall()
        columns = [_row_to_dict(row) for row in rows]
        primary_key_columns = [
            str(column.get("name") or "")
            for column in sorted(columns, key=lambda item: int(item.get("pk") or 0))
            if int(column.get("pk") or 0) > 0
        ]
        recording_id_column = next(
            (
                column
                for column in columns
                if str(column.get("name") or "") == "recording_id"
            ),
            {},
        )
        recording_id_primary_key = primary_key_columns == ["recording_id"]
        return {
            "schema": "palette.web_labeling_assignment_schema_integrity.v1",
            "assignment_table": "recording_assignments",
            "recording_id_primary_key": recording_id_primary_key,
            "schema_enforced_recording_primary_key": recording_id_primary_key,
            "primary_key_columns": primary_key_columns,
            "recording_id_column_present": bool(recording_id_column),
            "recording_id_column_type": str(recording_id_column.get("type") or ""),
            "recording_id_column_notnull": bool(recording_id_column.get("notnull")),
            "one_row_per_recording_enforced": recording_id_primary_key,
        }

    def single_owner_assignment_contract(
        self,
        *,
        recording_id: str | None = None,
    ) -> dict[str, object]:
        """Return read-only evidence for the recording-level single-owner contract.

        This is the stable assignment-store surface for wrappers and web routes that
        need to prove the labeling workflow is assignment-owned without performing
        an assignment mutation.  Browser writes are expected to resolve the writable
        training-Zarr target from the current active assignment on the server side.
        """

        self.initialize()
        recording_id_value = str(recording_id or "").strip()
        assignment = self.get_assignment(recording_id_value) if recording_id_value else None
        assignment_schema_integrity = self.assignment_schema_integrity()
        schema_enforced_recording_primary_key = bool(
            assignment_schema_integrity.get("schema_enforced_recording_primary_key")
        )
        one_row_per_recording_enforced = bool(
            assignment_schema_integrity.get("one_row_per_recording_enforced")
        )
        current_assignee_user = str((assignment or {}).get("assignee_user") or "")
        current_status = str((assignment or {}).get("status") or "")
        current_assignment_row_present = bool(assignment)
        recording_contract_met = (
            bool(recording_id_value)
            and current_assignment_row_present
            and str((assignment or {}).get("recording_id") or "") == recording_id_value
            and bool(current_assignee_user)
            and schema_enforced_recording_primary_key
            and one_row_per_recording_enforced
        )
        structural_contract_met = bool(
            schema_enforced_recording_primary_key and one_row_per_recording_enforced
        )
        single_owner_assignment_contract_met = (
            recording_contract_met if recording_id_value else structural_contract_met
        )
        return {
            "schema": "palette.web_labeling_assignment_single_owner_contract.v1",
            "assignment_table": "recording_assignments",
            "assignment_scope": "recording",
            "assignment_key": "recording_id",
            "recording_assignment_key": "recording_id",
            "recording_id": recording_id_value,
            "current_assignment_row_present": current_assignment_row_present,
            "current_assignee_user": current_assignee_user,
            "current_status": current_status,
            "current_assignment_is_active": current_status == "active",
            "active_owner_status_value": "active",
            "active_status_values": ["active"],
            "current_owner_source": "recording_assignments.assignee_user",
            "recording_id_primary_key": bool(
                assignment_schema_integrity.get("recording_id_primary_key")
            ),
            "schema_enforced_recording_primary_key": schema_enforced_recording_primary_key,
            "one_row_per_recording_enforced": one_row_per_recording_enforced,
            "one_current_assignment_row_per_recording": one_row_per_recording_enforced,
            "one_active_owner_per_recording_enforced": one_row_per_recording_enforced,
            "one_active_owner": True,
            "primary_key_columns": list(
                assignment_schema_integrity.get("primary_key_columns") or []
            ),
            "multiple_labelers_per_recording_allowed": False,
            "assignment_user_match_required_for_mutation": True,
            "browser_mutation_requires_current_assignment_owner": True,
            "browser_mutation_target_resolved_server_side": True,
            "browser_mutation_target_source": "recording_assignments.active_assignment",
            "labelers_mutate_assigned_training_zarrs": True,
            "labelers_mutate_intermediate_csvs": False,
            "assignment_manifests_are_control_plane": True,
            "duplicate_manifest_rows_do_not_create_multiple_owners": True,
            "reassignment_replaces_owner": True,
            "stale_sessions_closed_on_reassignment": True,
            "raw_assignment_change_blocks_open_sessions": True,
            "operator_reassignment_helper": "assign_recording_with_session_closure",
            "structural_contract_met": structural_contract_met,
            "recording_contract_met": recording_contract_met,
            "single_owner_assignment_contract_met": single_owner_assignment_contract_met,
            "ready": single_owner_assignment_contract_met,
        }

    def assign_recording(
        self,
        *,
        recording_id: str,
        assignee_user: str,
        assigned_by: str | None = None,
        status: str = "active",
        notes: str | None = None,
        allow_stale_open_sessions: bool = False,
    ) -> dict[str, object]:
        """Create or update a recording assignment.

        Use assign_recording_with_session_closure for operator-facing assignment
        changes. Direct owner/status changes fail closed while open browser
        sessions exist unless allow_stale_open_sessions is explicitly set for
        diagnostics or tests that intentionally create inconsistent stores.
        """

        self.initialize()
        recording_id = str(recording_id).strip()
        assignee_user = str(assignee_user).strip()
        if not recording_id:
            raise ValueError("recording_id is required.")
        if not assignee_user:
            raise ValueError("assignee_user is required.")
        assigned_by_value = _normalize_optional_text(assigned_by)
        status_value = str(status or "active")
        notes_value = _normalize_optional_text(notes)
        existing = self.get_assignment(recording_id)
        if existing is not None:
            existing_target = {
                "recording_id": existing.get("recording_id"),
                "assignee_user": existing.get("assignee_user"),
                "assigned_by": existing.get("assigned_by"),
                "status": existing.get("status"),
                "notes": existing.get("notes"),
            }
            target = {
                "recording_id": recording_id,
                "assignee_user": assignee_user,
                "assigned_by": assigned_by_value,
                "status": status_value,
                "notes": notes_value,
            }
            if existing_target == target:
                return existing
        else:
            existing_target = None
        changed_owner_or_status = existing is not None and (
            str(existing.get("assignee_user") or "") != assignee_user
            or str(existing.get("status") or "") != status_value
        )
        if changed_owner_or_status and not allow_stale_open_sessions:
            row = self.conn.execute(
                """
                SELECT session_id
                FROM labeling_sessions
                WHERE recording_id = ?
                  AND closed_at_utc IS NULL
                LIMIT 1;
                """,
                (recording_id,),
            ).fetchone()
            if row is not None:
                raise RuntimeError(
                    "Recording assignment owner/status change would leave open browser sessions stale; "
                    "use assign_recording_with_session_closure or pass allow_stale_open_sessions=True "
                    "only for diagnostics."
                )
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO recording_assignments (
                recording_id, assignee_user, assigned_by, assigned_at_utc, status, notes
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(recording_id) DO UPDATE SET
                assignee_user = excluded.assignee_user,
                assigned_by = excluded.assigned_by,
                assigned_at_utc = excluded.assigned_at_utc,
                status = excluded.status,
                notes = excluded.notes;
            """,
            (
                recording_id,
                assignee_user,
                assigned_by_value,
                now,
                status_value,
                notes_value,
            ),
        )
        self.conn.commit()
        row = self.get_assignment(recording_id)
        assert row is not None
        self.record_assignment_event(
            recording_id=recording_id,
            actor_user=assigned_by_value,
            event_type="assignment_created" if existing is None else "assignment_changed",
            before=existing_target,
            after={
                "recording_id": row.get("recording_id"),
                "assignee_user": row.get("assignee_user"),
                "assigned_by": row.get("assigned_by"),
                "assigned_at_utc": row.get("assigned_at_utc"),
                "status": row.get("status"),
                "notes": row.get("notes"),
            },
        )
        return row

    def assign_recording_with_session_closure(
        self,
        *,
        recording_id: str,
        assignee_user: str,
        assigned_by: str | None = None,
        status: str = "active",
        notes: str | None = None,
        session_event_type: str = "session_closed_by_assignment_change",
    ) -> dict[str, object]:
        """Assign a recording and close stale sessions when owner/status changes."""

        self.initialize()
        recording_id = str(recording_id).strip()
        assignee_user = str(assignee_user).strip()
        if not recording_id:
            raise ValueError("recording_id is required.")
        if not assignee_user:
            raise ValueError("assignee_user is required.")
        assigned_by_value = _normalize_optional_text(assigned_by)
        requested_status = str(status or "active")
        notes_value = _normalize_optional_text(notes)
        with self._connection_lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE;")
            try:
                row = conn.execute(
                    "SELECT * FROM recording_assignments WHERE recording_id = ?;",
                    (recording_id,),
                ).fetchone()
                before_assignment = _row_to_dict(row) if row is not None else None
                before_user = str((before_assignment or {}).get("assignee_user") or "")
                before_status = str((before_assignment or {}).get("status") or "")
                existing_target = (
                    {
                        "recording_id": before_assignment.get("recording_id"),
                        "assignee_user": before_assignment.get("assignee_user"),
                        "assigned_by": before_assignment.get("assigned_by"),
                        "status": before_assignment.get("status"),
                        "notes": before_assignment.get("notes"),
                    }
                    if before_assignment is not None
                    else None
                )
                target = {
                    "recording_id": recording_id,
                    "assignee_user": assignee_user,
                    "assigned_by": assigned_by_value,
                    "status": requested_status,
                    "notes": notes_value,
                }
                changed_owner_or_status = before_assignment is not None and (
                    before_user != assignee_user or before_status != requested_status
                )
                if existing_target == target:
                    conn.rollback()
                    assignment = before_assignment
                    closed_sessions: list[dict[str, object]] = []
                    assignment_event_required = False
                else:
                    closed_sessions = []
                    now = utc_now()
                    if changed_owner_or_status:
                        rows = conn.execute(
                            """
                            SELECT s.*, t.title, t.dataset_id, t.component_name, t.state AS task_state
                            FROM labeling_sessions s
                            JOIN labeling_tasks t ON t.task_id = s.task_id
                            WHERE s.recording_id = ?
                              AND s.closed_at_utc IS NULL;
                            """,
                            (recording_id,),
                        ).fetchall()
                        closed_sessions = [_row_to_dict(session_row) for session_row in rows]
                        conn.execute(
                            """
                            UPDATE labeling_sessions
                            SET closed_at_utc = ?
                            WHERE recording_id = ?
                              AND closed_at_utc IS NULL;
                            """,
                            (now, recording_id),
                        )
                    conn.execute(
                        """
                        INSERT INTO recording_assignments (
                            recording_id, assignee_user, assigned_by, assigned_at_utc, status, notes
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(recording_id) DO UPDATE SET
                            assignee_user = excluded.assignee_user,
                            assigned_by = excluded.assigned_by,
                            assigned_at_utc = excluded.assigned_at_utc,
                            status = excluded.status,
                            notes = excluded.notes;
                        """,
                        (
                            recording_id,
                            assignee_user,
                            assigned_by_value,
                            now,
                            requested_status,
                            notes_value,
                        ),
                    )
                    assignment_row = conn.execute(
                        "SELECT * FROM recording_assignments WHERE recording_id = ?;",
                        (recording_id,),
                    ).fetchone()
                    assignment = _row_to_dict(assignment_row)
                    conn.commit()
                    assignment_event_required = True
            except Exception:
                conn.rollback()
                raise
        if assignment_event_required:
            self.record_assignment_event(
                recording_id=recording_id,
                actor_user=assigned_by_value,
                event_type="assignment_created" if before_assignment is None else "assignment_changed",
                before=existing_target,
                after={
                    "recording_id": assignment.get("recording_id"),
                    "assignee_user": assignment.get("assignee_user"),
                    "assigned_by": assignment.get("assigned_by"),
                    "assigned_at_utc": assignment.get("assigned_at_utc"),
                    "status": assignment.get("status"),
                    "notes": assignment.get("notes"),
                },
            )
        closed_at_utc = (
            str(assignment.get("assigned_at_utc") or "")
            if changed_owner_or_status
            else ""
        )
        for session in closed_sessions:
            self.record_event(
                task_id=str(session["task_id"]),
                recording_id=str(session["recording_id"]),
                user=str(assigned_by_value or assignee_user),
                event_type=session_event_type,
                target={"session_id": str(session["session_id"])},
                after={"closed_at_utc": closed_at_utc},
            )
        new_user = str(assignment.get("assignee_user") or assignee_user)
        new_status = str(assignment.get("status") or status or "active")
        assignment_schema_integrity = self.assignment_schema_integrity()
        schema_enforced_recording_primary_key = bool(
            assignment_schema_integrity.get("schema_enforced_recording_primary_key")
        )
        one_row_per_recording_enforced = bool(
            assignment_schema_integrity.get("one_row_per_recording_enforced")
        )
        single_owner_assignment_contract_met = bool(
            assignment
            and str(assignment.get("recording_id") or "") == recording_id
            and str(assignment.get("assignee_user") or "") == new_user
            and schema_enforced_recording_primary_key
            and one_row_per_recording_enforced
        )
        assignment_single_owner_contract = self.single_owner_assignment_contract(
            recording_id=recording_id
        )
        assignment_single_owner_contract.update(
            {
            "schema": "palette.web_labeling_assignment_single_owner_transition.v1",
            "assignment_table": "recording_assignments",
            "assignment_scope": "recording",
            "assignment_key": "recording_id",
            "recording_id": recording_id,
            "current_assignment_row_present": bool(assignment),
            "current_assignee_user": new_user,
            "current_status": new_status,
            "current_assignment_is_active": new_status == "active",
            "active_owner_status_value": "active",
            "current_owner_source": "recording_assignments.assignee_user",
            "recording_id_primary_key": bool(
                assignment_schema_integrity.get("recording_id_primary_key")
            ),
            "schema_enforced_recording_primary_key": schema_enforced_recording_primary_key,
            "one_row_per_recording_enforced": one_row_per_recording_enforced,
            "one_active_owner_per_recording_enforced": one_row_per_recording_enforced,
            "primary_key_columns": list(
                assignment_schema_integrity.get("primary_key_columns") or []
            ),
            "single_owner_assignment_contract_met": single_owner_assignment_contract_met,
            "ready": single_owner_assignment_contract_met,
            }
        )
        return {
            "assignment": assignment,
            "previous_assignment": before_assignment,
            "assignment_transition": {
                "recording_id": str(recording_id),
                "previous_assignee_user": before_user or None,
                "previous_status": before_status or None,
                "new_assignee_user": new_user,
                "new_status": new_status,
                "owner_changed": before_user != new_user,
                "status_changed": before_status != new_status,
                "changed_owner_or_status": changed_owner_or_status,
                "stale_sessions_closed_before_assignment_update": changed_owner_or_status,
                "session_closure_and_assignment_update_atomic": changed_owner_or_status,
                "session_closure_order": (
                    "before_assignment_update"
                    if changed_owner_or_status
                    else "not_required"
                ),
                "single_owner_assignment_contract_met": single_owner_assignment_contract_met,
                "post_assignment_current_assignee_user": new_user,
                "post_assignment_current_status": new_status,
                "post_assignment_current_assignment_is_active": new_status == "active",
            },
            "assignment_single_owner_contract": assignment_single_owner_contract,
            "assignment_schema_integrity": assignment_schema_integrity,
            "closed_session_count": len(closed_sessions),
            "closed_session_ids": [str(session.get("session_id") or "") for session in closed_sessions],
            "closed_sessions": closed_sessions,
        }

    def get_assignment(self, recording_id: str) -> dict[str, object] | None:
        self.initialize()
        row = self.conn.execute(
            "SELECT * FROM recording_assignments WHERE recording_id = ?;",
            (str(recording_id),),
        ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def list_assignments(
        self,
        *,
        assignee_user: str | None = None,
        status: str | None = "active",
    ) -> list[dict[str, object]]:
        self.initialize()
        sql = ["SELECT * FROM recording_assignments WHERE 1=1"]
        params: list[object] = []
        if assignee_user:
            sql.append("AND assignee_user = ?")
            params.append(str(assignee_user))
        if status:
            sql.append("AND status = ?")
            params.append(str(status))
        sql.append("ORDER BY assignee_user, recording_id")
        rows = self.conn.execute(" ".join(sql), params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def record_assignment_event(
        self,
        *,
        recording_id: str,
        actor_user: str | None,
        event_type: str,
        before: Mapping[str, object] | None = None,
        after: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self.initialize()
        event_id = str(uuid.uuid4())
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO labeling_assignment_events (
                event_id, recording_id, actor_user, event_type, before_json, after_json, created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                event_id,
                str(recording_id),
                _normalize_optional_text(actor_user),
                str(event_type),
                _json_dumps(before) if before is not None else None,
                _json_dumps(after) if after is not None else None,
                now,
            ),
        )
        self.conn.commit()
        return {
            "event_id": event_id,
            "recording_id": str(recording_id),
            "actor_user": _normalize_optional_text(actor_user),
            "event_type": str(event_type),
            "created_at_utc": now,
        }

    def list_assignment_events(
        self,
        *,
        recording_id: str | None = None,
        actor_user: str | None = None,
        event_type: str | None = None,
        since_utc: str | None = None,
        until_utc: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        self.initialize()
        sql = ["SELECT * FROM labeling_assignment_events WHERE 1=1"]
        params: list[object] = []
        if recording_id:
            sql.append("AND recording_id = ?")
            params.append(str(recording_id))
        if actor_user:
            sql.append("AND actor_user = ?")
            params.append(str(actor_user))
        if event_type:
            sql.append("AND event_type = ?")
            params.append(str(event_type))
        if since_utc:
            sql.append("AND created_at_utc >= ?")
            params.append(str(since_utc))
        if until_utc:
            sql.append("AND created_at_utc <= ?")
            params.append(str(until_utc))
        sql.append("ORDER BY created_at_utc DESC LIMIT ?")
        params.append(max(1, int(limit)))
        rows = self.conn.execute(" ".join(sql), params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def upsert_task(
        self,
        *,
        recording_id: str,
        workflow_kind: str,
        task_id: str | None = None,
        dataset_id: str | None = None,
        zarr_use: str | None = None,
        stage_group: str | None = None,
        run_name: str | None = None,
        component_name: str | None = None,
        title: str | None = None,
        scope: Mapping[str, object] | Sequence[object] | None = None,
        state: str = "pending",
        priority: int = 0,
        notes: str | None = None,
        actor_user: str | None = None,
    ) -> dict[str, object]:
        self.initialize()
        recording_id = str(recording_id).strip()
        workflow_kind = str(workflow_kind).strip()
        if not recording_id:
            raise ValueError("recording_id is required.")
        if not workflow_kind:
            raise ValueError("workflow_kind is required.")
        resolved_task_id = str(task_id or uuid.uuid4()).strip()
        now = utc_now()
        state_value = str(state or "pending").strip()
        completed_at = now if state_value == "complete" else None
        scope_payload = scope if scope is not None else {}
        scope_json = _json_dumps(scope_payload)
        decoded_scope = _json_loads(scope_json)
        scope_value = decoded_scope if decoded_scope is not None else {}
        target = {
            "task_id": resolved_task_id,
            "recording_id": recording_id,
            "workflow_kind": workflow_kind,
            "dataset_id": _normalize_optional_text(dataset_id),
            "zarr_use": _normalize_optional_text(zarr_use),
            "stage_group": _normalize_optional_text(stage_group),
            "run_name": _normalize_optional_text(run_name),
            "component_name": _normalize_optional_text(component_name),
            "title": _normalize_optional_text(title),
            "scope": scope_value,
            "state": state_value,
            "priority": int(priority),
            "notes": _normalize_optional_text(notes),
        }
        existing = self.get_task(resolved_task_id)
        if existing is not None:
            existing_target = {
                "task_id": existing.get("task_id"),
                "recording_id": existing.get("recording_id"),
                "workflow_kind": existing.get("workflow_kind"),
                "dataset_id": existing.get("dataset_id"),
                "zarr_use": existing.get("zarr_use"),
                "stage_group": existing.get("stage_group"),
                "run_name": existing.get("run_name"),
                "component_name": existing.get("component_name"),
                "title": existing.get("title"),
                "scope": existing.get("scope") or {},
                "state": existing.get("state"),
                "priority": int(existing.get("priority") or 0),
                "notes": existing.get("notes"),
            }
            if existing_target == target:
                return existing
        else:
            existing_target = None
        self.conn.execute(
            """
            INSERT INTO labeling_tasks (
                task_id, recording_id, workflow_kind, dataset_id, zarr_use,
                stage_group, run_name, component_name, title, scope_json,
                state, priority, notes, created_at_utc, updated_at_utc, completed_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                recording_id = excluded.recording_id,
                workflow_kind = excluded.workflow_kind,
                dataset_id = excluded.dataset_id,
                zarr_use = excluded.zarr_use,
                stage_group = excluded.stage_group,
                run_name = excluded.run_name,
                component_name = excluded.component_name,
                title = excluded.title,
                scope_json = excluded.scope_json,
                state = excluded.state,
                priority = excluded.priority,
                notes = excluded.notes,
                updated_at_utc = excluded.updated_at_utc,
                completed_at_utc = excluded.completed_at_utc;
            """,
            (
                resolved_task_id,
                recording_id,
                workflow_kind,
                _normalize_optional_text(dataset_id),
                _normalize_optional_text(zarr_use),
                _normalize_optional_text(stage_group),
                _normalize_optional_text(run_name),
                _normalize_optional_text(component_name),
                _normalize_optional_text(title),
                scope_json,
                state_value,
                int(priority),
                _normalize_optional_text(notes),
                now,
                now,
                completed_at,
            ),
        )
        self.conn.commit()
        row = self.get_task(resolved_task_id)
        assert row is not None
        self.record_task_definition_event(
            task_id=resolved_task_id,
            recording_id=recording_id,
            actor_user=actor_user,
            event_type="task_definition_created" if existing is None else "task_definition_changed",
            before=existing_target,
            after={
                "task_id": row.get("task_id"),
                "recording_id": row.get("recording_id"),
                "workflow_kind": row.get("workflow_kind"),
                "dataset_id": row.get("dataset_id"),
                "zarr_use": row.get("zarr_use"),
                "stage_group": row.get("stage_group"),
                "run_name": row.get("run_name"),
                "component_name": row.get("component_name"),
                "title": row.get("title"),
                "scope": row.get("scope") if row.get("scope") is not None else {},
                "state": row.get("state"),
                "priority": row.get("priority"),
                "notes": row.get("notes"),
                "created_at_utc": row.get("created_at_utc"),
                "updated_at_utc": row.get("updated_at_utc"),
                "completed_at_utc": row.get("completed_at_utc"),
            },
        )
        return row

    def record_task_definition_event(
        self,
        *,
        task_id: str,
        recording_id: str,
        actor_user: str | None,
        event_type: str,
        before: Mapping[str, object] | None = None,
        after: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self.initialize()
        event_id = str(uuid.uuid4())
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO labeling_task_definition_events (
                event_id, task_id, recording_id, actor_user, event_type, before_json, after_json, created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                event_id,
                str(task_id),
                str(recording_id),
                _normalize_optional_text(actor_user),
                str(event_type),
                _json_dumps(before) if before is not None else None,
                _json_dumps(after) if after is not None else None,
                now,
            ),
        )
        self.conn.commit()
        return {
            "event_id": event_id,
            "task_id": str(task_id),
            "recording_id": str(recording_id),
            "actor_user": _normalize_optional_text(actor_user),
            "event_type": str(event_type),
            "created_at_utc": now,
        }

    def list_task_definition_events(
        self,
        *,
        task_id: str | None = None,
        recording_id: str | None = None,
        actor_user: str | None = None,
        event_type: str | None = None,
        since_utc: str | None = None,
        until_utc: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        self.initialize()
        sql = ["SELECT * FROM labeling_task_definition_events WHERE 1=1"]
        params: list[object] = []
        if task_id:
            sql.append("AND task_id = ?")
            params.append(str(task_id))
        if recording_id:
            sql.append("AND recording_id = ?")
            params.append(str(recording_id))
        if actor_user:
            sql.append("AND actor_user = ?")
            params.append(str(actor_user))
        if event_type:
            sql.append("AND event_type = ?")
            params.append(str(event_type))
        if since_utc:
            sql.append("AND created_at_utc >= ?")
            params.append(str(since_utc))
        if until_utc:
            sql.append("AND created_at_utc <= ?")
            params.append(str(until_utc))
        sql.append("ORDER BY created_at_utc DESC LIMIT ?")
        params.append(max(1, int(limit)))
        rows = self.conn.execute(" ".join(sql), params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_task(self, task_id: str) -> dict[str, object] | None:
        self.initialize()
        row = self.conn.execute(
            """
            SELECT t.*, a.assignee_user, a.status AS assignment_status
            FROM labeling_tasks t
            LEFT JOIN recording_assignments a ON a.recording_id = t.recording_id
            WHERE t.task_id = ?;
            """,
            (str(task_id),),
        ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def list_tasks_for_user(
        self,
        user: str,
        *,
        states: Iterable[str] | None = None,
        include_completed: bool = False,
    ) -> list[dict[str, object]]:
        self.initialize()
        user = str(user).strip()
        if not user:
            raise ValueError("user is required.")
        sql = [
            """
            SELECT
                t.*,
                a.assignee_user,
                a.assigned_at_utc,
                a.notes AS assignment_notes,
                a.status AS assignment_status
            FROM labeling_tasks t
            JOIN recording_assignments a ON a.recording_id = t.recording_id
            WHERE a.assignee_user = ?
              AND a.status = 'active'
            """
        ]
        params: list[object] = [user]
        state_values = [str(item) for item in states or [] if str(item).strip()]
        if state_values:
            placeholders = ",".join("?" for _ in state_values)
            sql.append(f"AND t.state IN ({placeholders})")
            params.extend(state_values)
        elif not include_completed:
            sql.append("AND t.state != 'complete'")
        sql.append(
            """
            ORDER BY
                t.priority DESC,
                t.recording_id,
                CASE t.workflow_kind
                    WHEN 'detect_analysis' THEN 10
                    WHEN 'detect_training' THEN 20
                    WHEN 'keypoints' THEN 30
                    WHEN 'subject_mask_component' THEN 40
                    ELSE 100
                END,
                t.created_at_utc
            """
        )
        rows = self.conn.execute(" ".join(sql), params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def list_tasks(
        self,
        *,
        recording_id: str | None = None,
        assignee_user: str | None = None,
        include_completed: bool = True,
    ) -> list[dict[str, object]]:
        self.initialize()
        sql = [
            """
            SELECT t.*, a.assignee_user, a.status AS assignment_status
            FROM labeling_tasks t
            LEFT JOIN recording_assignments a ON a.recording_id = t.recording_id
            WHERE 1=1
            """
        ]
        params: list[object] = []
        if recording_id:
            sql.append("AND t.recording_id = ?")
            params.append(str(recording_id))
        if assignee_user:
            sql.append("AND a.assignee_user = ?")
            params.append(str(assignee_user))
        if not include_completed:
            sql.append("AND t.state != 'complete'")
        sql.append("ORDER BY t.recording_id, t.priority DESC, t.workflow_kind, t.created_at_utc")
        rows = self.conn.execute(" ".join(sql), params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def update_task_state(self, *, task_id: str, state: str, user: str | None = None) -> dict[str, object]:
        self.initialize()
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        now = utc_now()
        state_value = str(state).strip()
        if not state_value:
            raise ValueError("state is required.")
        if str(task.get("state") or "") == state_value:
            return task
        completed_at = now if state_value == "complete" else None
        self.conn.execute(
            """
            UPDATE labeling_tasks
            SET state = ?, updated_at_utc = ?, completed_at_utc = ?
            WHERE task_id = ?;
            """,
            (state_value, now, completed_at, str(task_id)),
        )
        self.conn.commit()
        updated = self.get_task(task_id)
        assert updated is not None
        if state_value == "complete":
            self.close_sessions_for_task(
                task_id=str(task_id),
                user=user,
                event_type="session_closed_by_task_completion",
            )
        if user:
            self.record_event(
                task_id=str(task_id),
                recording_id=str(task["recording_id"]),
                user=str(user),
                event_type="task_state_changed",
                before={"state": task.get("state")},
                after={"state": state_value},
            )
            if state_value == "complete":
                self.record_event(
                    task_id=str(task_id),
                    recording_id=str(task["recording_id"]),
                    user=str(user),
                    event_type="task_completed",
                    before={"state": task.get("state")},
                    after={"state": state_value, "completed_at_utc": completed_at},
                )
            if str(task.get("state") or "") == "complete" and state_value != "complete":
                self.record_event(
                    task_id=str(task_id),
                    recording_id=str(task["recording_id"]),
                    user=str(user),
                    event_type="task_reopened",
                    before={"state": task.get("state"), "completed_at_utc": task.get("completed_at_utc")},
                    after={"state": state_value, "completed_at_utc": None},
                )
        return updated

    def list_current_task_sessions(
        self,
        *,
        task_id: str,
        user: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        self.initialize()
        sql = [
            """
            SELECT s.*, t.title, t.dataset_id, t.component_name, t.state AS task_state
            FROM labeling_sessions s
            JOIN labeling_tasks t ON t.task_id = s.task_id
            WHERE s.task_id = ?
              AND s.closed_at_utc IS NULL
              AND s.expires_at_utc >= ?
            """
        ]
        params: list[object] = [str(task_id), utc_now()]
        if user:
            sql.append("AND s.user = ?")
            params.append(str(user))
        sql.append("ORDER BY s.created_at_utc DESC LIMIT ?")
        params.append(max(1, int(limit)))
        rows = self.conn.execute(" ".join(sql), params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_current_task_session(self, *, task_id: str) -> dict[str, object] | None:
        sessions = self.list_current_task_sessions(task_id=task_id, limit=1)
        return sessions[0] if sessions else None

    def _active_assignment_mismatched_sessions_for_recording(
        self,
        *,
        conn,
        recording_id: str,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        rows = conn.execute(
            """
            SELECT s.*, t.title, t.dataset_id, t.component_name, t.state AS task_state,
                   a.assignee_user, a.status AS assignment_status
            FROM labeling_sessions s
            JOIN labeling_tasks t ON t.task_id = s.task_id
            LEFT JOIN recording_assignments a ON a.recording_id = s.recording_id
            WHERE s.recording_id = ?
              AND s.closed_at_utc IS NULL
              AND (
                a.recording_id IS NULL
                OR a.status != 'active'
                OR s.user != a.assignee_user
              )
            ORDER BY s.created_at_utc DESC
            LIMIT ?;
            """,
            (str(recording_id), max(1, int(limit))),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def active_assignment_mismatched_sessions_for_recording(
        self,
        recording_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        self.initialize()
        recording_id = str(recording_id).strip()
        if not recording_id:
            return []
        return self._active_assignment_mismatched_sessions_for_recording(
            conn=self.conn,
            recording_id=recording_id,
            limit=limit,
        )

    def create_session(
        self,
        *,
        task_id: str,
        user: str,
        ttl_seconds: int = 12 * 60 * 60,
        client_label: str | None = None,
    ) -> SessionLease:
        self.initialize()
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        assignment_user = str(task.get("assignee_user") or "")
        assignment_status = str(task.get("assignment_status") or "")
        if assignment_status != "active" or assignment_user != str(user):
            raise PermissionError("Task is not assigned to the current user.")
        if str(task.get("state") or "") == "complete":
            raise PermissionError("Task is complete and cannot be reopened for labeling.")
        if str(task.get("state") or "") not in LABELER_START_TASK_STATES:
            raise PermissionError("Task is not in a startable labeling state.")
        with self._connection_lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE;")
            try:
                now_dt = datetime.now(timezone.utc)
                now = now_dt.isoformat()
                requested_ttl_seconds = int(ttl_seconds)
                effective_ttl_seconds = (
                    requested_ttl_seconds
                    if requested_ttl_seconds < 0
                    else max(60, requested_ttl_seconds)
                )
                expires_at = now_dt + timedelta(seconds=effective_ttl_seconds)
                session_id = str(uuid.uuid4())
                mismatched_sessions = self._active_assignment_mismatched_sessions_for_recording(
                    conn=conn,
                    recording_id=str(task["recording_id"]),
                    limit=1,
                )
                if mismatched_sessions:
                    raise PermissionError(
                        "Stale previous-owner sessions are still open for this recording; "
                        "close them or re-run assignment with session closure before labeling."
                    )
                rows = conn.execute(
                    """
                    SELECT s.*, t.title, t.dataset_id, t.component_name, t.state AS task_state
                    FROM labeling_sessions s
                    JOIN labeling_tasks t ON t.task_id = s.task_id
                    WHERE s.task_id = ?
                      AND s.closed_at_utc IS NULL
                      AND s.expires_at_utc >= ?
                    ORDER BY s.created_at_utc DESC
                    LIMIT 100;
                    """,
                    (str(task_id), now),
                ).fetchall()
                superseded_sessions = [_row_to_dict(row) for row in rows]
                superseded_session_ids = tuple(str(session["session_id"]) for session in superseded_sessions)
                if superseded_session_ids:
                    conn.execute(
                        """
                        UPDATE labeling_sessions
                        SET closed_at_utc = ?
                        WHERE task_id = ?
                          AND closed_at_utc IS NULL
                          AND expires_at_utc >= ?;
                        """,
                        (now, str(task_id), now),
                    )
                conn.execute(
                    """
                    INSERT INTO labeling_sessions (
                        session_id, task_id, recording_id, user, workflow_kind,
                        created_at_utc, expires_at_utc, last_seen_at_utc, client_label
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        session_id,
                        str(task_id),
                        str(task["recording_id"]),
                        str(user),
                        str(task["workflow_kind"]),
                        now,
                        expires_at.isoformat(),
                        now,
                        _normalize_optional_text(client_label),
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        for session in superseded_sessions:
            self.record_event(
                task_id=str(session["task_id"]),
                recording_id=str(session["recording_id"]),
                user=str(user),
                event_type="session_superseded",
                target={"session_id": str(session["session_id"])},
                before={"expires_at_utc": session.get("expires_at_utc"), "closed_at_utc": session.get("closed_at_utc")},
                after={"closed_at_utc": now, "superseded_by_session_id": session_id},
            )
        self.record_event(
            task_id=str(task_id),
            recording_id=str(task["recording_id"]),
            user=str(user),
            event_type="session_opened",
            after={
                "session_id": session_id,
                "expires_at_utc": expires_at.isoformat(),
                "superseded_session_ids": list(superseded_session_ids),
            },
        )
        return SessionLease(
            session_id=session_id,
            task_id=str(task_id),
            recording_id=str(task["recording_id"]),
            user=str(user),
            expires_at_utc=expires_at.isoformat(),
            superseded_session_ids=superseded_session_ids,
        )

    def get_session(self, session_id: str) -> dict[str, object] | None:
        self.initialize()
        row = self.conn.execute(
            """
            SELECT s.*, t.title, t.dataset_id, t.zarr_use, t.stage_group, t.run_name,
                   t.component_name, t.scope_json, t.state AS task_state
            FROM labeling_sessions s
            JOIN labeling_tasks t ON t.task_id = s.task_id
            WHERE s.session_id = ?;
            """,
            (str(session_id),),
        ).fetchone()
        if row is None:
            return None
        out = _row_to_dict(row)
        self.conn.execute(
            "UPDATE labeling_sessions SET last_seen_at_utc = ? WHERE session_id = ?;",
            (utc_now(), str(session_id)),
        )
        self.conn.commit()
        return out

    def close_session(self, *, session_id: str, user: str | None = None) -> None:
        self.initialize()
        session = self.get_session(session_id)
        now = utc_now()
        self.conn.execute(
            "UPDATE labeling_sessions SET closed_at_utc = ? WHERE session_id = ?;",
            (now, str(session_id)),
        )
        self.conn.commit()
        if session is not None:
            self.record_event(
                task_id=str(session["task_id"]),
                recording_id=str(session["recording_id"]),
                user=str(user or session["user"]),
                event_type="session_closed",
                after={"session_id": str(session_id), "closed_at_utc": now},
            )

    def list_sessions(
        self,
        *,
        assignee_user: str | None = None,
        include_closed: bool = False,
        expired_only: bool = False,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        self.initialize()
        sql = [
            """
            SELECT s.*, t.title, t.dataset_id, t.component_name, t.state AS task_state,
                   a.assignee_user, a.status AS assignment_status
            FROM labeling_sessions s
            JOIN labeling_tasks t ON t.task_id = s.task_id
            LEFT JOIN recording_assignments a ON a.recording_id = s.recording_id
            WHERE 1=1
            """
        ]
        params: list[object] = []
        if assignee_user:
            sql.append("AND a.assignee_user = ?")
            params.append(str(assignee_user))
        if not include_closed:
            sql.append("AND s.closed_at_utc IS NULL")
        if expired_only:
            sql.append("AND s.expires_at_utc < ?")
            params.append(utc_now())
        sql.append("ORDER BY s.created_at_utc DESC LIMIT ?")
        params.append(max(1, int(limit)))
        rows = self.conn.execute(" ".join(sql), params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def cleanup_stale_sessions(self, *, user: str | None = None) -> list[dict[str, object]]:
        self.initialize()
        sessions = self.list_sessions(include_closed=False, expired_only=True, limit=1000)
        now = utc_now()
        cleaned: list[dict[str, object]] = []
        for session in sessions:
            self.conn.execute(
                "UPDATE labeling_sessions SET closed_at_utc = ? WHERE session_id = ? AND closed_at_utc IS NULL;",
                (now, str(session["session_id"])),
            )
            cleaned.append(session)
        self.conn.commit()
        for session in cleaned:
            self.record_event(
                task_id=str(session["task_id"]),
                recording_id=str(session["recording_id"]),
                user=str(user or session["user"]),
                event_type="stale_session_closed",
                target={"session_id": str(session["session_id"])},
                after={"closed_at_utc": now, "expires_at_utc": session.get("expires_at_utc")},
            )
        return cleaned

    def close_sessions_for_task(
        self,
        *,
        task_id: str,
        user: str | None = None,
        event_type: str = "session_closed_by_task_completion",
    ) -> list[dict[str, object]]:
        self.initialize()
        task_id = str(task_id).strip()
        if not task_id:
            raise ValueError("task_id is required.")
        rows = self.conn.execute(
            """
            SELECT s.*, t.title, t.dataset_id, t.component_name, t.state AS task_state
            FROM labeling_sessions s
            JOIN labeling_tasks t ON t.task_id = s.task_id
            WHERE s.task_id = ?
              AND s.closed_at_utc IS NULL;
            """,
            (task_id,),
        ).fetchall()
        sessions = [_row_to_dict(row) for row in rows]
        now = utc_now()
        self.conn.execute(
            """
            UPDATE labeling_sessions
            SET closed_at_utc = ?
            WHERE task_id = ?
              AND closed_at_utc IS NULL;
            """,
            (now, task_id),
        )
        self.conn.commit()
        for session in sessions:
            self.record_event(
                task_id=str(session["task_id"]),
                recording_id=str(session["recording_id"]),
                user=str(user or session["user"]),
                event_type=event_type,
                target={"session_id": str(session["session_id"])},
                after={"closed_at_utc": now},
            )
        return sessions

    def close_sessions_for_recording(
        self,
        *,
        recording_id: str,
        user: str | None = None,
        event_type: str = "session_closed_by_assignment_change",
    ) -> list[dict[str, object]]:
        self.initialize()
        recording_id = str(recording_id).strip()
        if not recording_id:
            raise ValueError("recording_id is required.")
        rows = self.conn.execute(
            """
            SELECT s.*, t.title, t.dataset_id, t.component_name, t.state AS task_state
            FROM labeling_sessions s
            JOIN labeling_tasks t ON t.task_id = s.task_id
            WHERE s.recording_id = ?
              AND s.closed_at_utc IS NULL;
            """,
            (recording_id,),
        ).fetchall()
        sessions = [_row_to_dict(row) for row in rows]
        now = utc_now()
        self.conn.execute(
            """
            UPDATE labeling_sessions
            SET closed_at_utc = ?
            WHERE recording_id = ?
              AND closed_at_utc IS NULL;
            """,
            (now, recording_id),
        )
        self.conn.commit()
        for session in sessions:
            self.record_event(
                task_id=str(session["task_id"]),
                recording_id=str(session["recording_id"]),
                user=str(user or session["user"]),
                event_type=event_type,
                target={"session_id": str(session["session_id"])},
                after={"closed_at_utc": now},
            )
        return sessions

    def close_assignment_mismatched_sessions_for_recording(
        self,
        *,
        recording_id: str,
        user: str | None = None,
        event_type: str = "session_closed_by_reassignment_safety_repair",
    ) -> list[dict[str, object]]:
        self.initialize()
        recording_id = str(recording_id).strip()
        if not recording_id:
            raise ValueError("recording_id is required.")
        with self._connection_lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE;")
            try:
                sessions = self._active_assignment_mismatched_sessions_for_recording(
                    conn=conn,
                    recording_id=recording_id,
                    limit=1000,
                )
                now = utc_now()
                session_ids = [
                    str(session.get("session_id") or "")
                    for session in sessions
                    if str(session.get("session_id") or "")
                ]
                if session_ids:
                    placeholders = ",".join("?" for _ in session_ids)
                    conn.execute(
                        f"""
                        UPDATE labeling_sessions
                        SET closed_at_utc = ?
                        WHERE session_id IN ({placeholders})
                          AND closed_at_utc IS NULL;
                        """,
                        (now, *session_ids),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        for session in sessions:
            self.record_event(
                task_id=str(session["task_id"]),
                recording_id=str(session["recording_id"]),
                user=str(user or session["user"]),
                event_type=event_type,
                target={
                    "session_id": str(session["session_id"]),
                    "repair_reason": "assignment_mismatched_session",
                },
                before={
                    "session_user": str(session.get("user") or ""),
                    "assignment_user": str(session.get("assignee_user") or ""),
                    "assignment_status": str(session.get("assignment_status") or ""),
                },
                after={"closed_at_utc": now},
            )
        return sessions

    def claim_promotion_retry(
        self,
        *,
        failed_event_id: str,
        task_id: str,
        recording_id: str,
        user: str,
    ) -> dict[str, object]:
        """Atomically claim retry execution for a failed promotion event."""

        self.initialize()
        failed_event_id = str(failed_event_id).strip()
        task_id = str(task_id).strip()
        recording_id = str(recording_id).strip()
        user = str(user).strip()
        if not failed_event_id:
            raise ValueError("failed_event_id is required.")
        if not task_id:
            raise ValueError("task_id is required.")
        if not recording_id:
            raise ValueError("recording_id is required.")
        if not user:
            raise ValueError("user is required.")

        with self._connection_lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE;")
            try:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM labeling_task_events
                    WHERE task_id = ?
                      AND event_type IN (
                          'promotion_retry_started',
                          'promotion_retry_abandoned',
                          'promotion_success',
                          'promotion_failed'
                      )
                    ORDER BY created_at_utc DESC;
                    """,
                    (task_id,),
                ).fetchall()
                events = [_row_to_dict(row) for row in rows]
                related_events: list[dict[str, object]] = []
                for event in events:
                    target = event.get("target")
                    if isinstance(target, Mapping) and str(target.get("retry_of_event_id") or "") == failed_event_id:
                        related_events.append(event)

                for event in related_events:
                    if str(event.get("event_type") or "") == "promotion_success":
                        conn.commit()
                        return {"status": "already_succeeded", "event": event}

                latest_start: dict[str, object] | None = None
                latest_outcome: dict[str, object] | None = None
                for event in related_events:
                    event_type = str(event.get("event_type") or "")
                    if event_type == "promotion_retry_started" and latest_start is None:
                        latest_start = event
                    elif event_type in {"promotion_retry_abandoned", "promotion_failed"} and latest_outcome is None:
                        latest_outcome = event

                if latest_start is not None:
                    start_time = str(latest_start.get("created_at_utc") or "")
                    outcome_time = str(latest_outcome.get("created_at_utc") or "") if latest_outcome is not None else ""
                    if not outcome_time or start_time > outcome_time:
                        conn.commit()
                        return {"status": "in_progress", "event": latest_start}

                now = utc_now()
                claim_event_id = str(uuid.uuid4())
                target = {"retry_of_event_id": failed_event_id}
                after = {"status": "claimed"}
                conn.execute(
                    """
                    INSERT INTO labeling_task_events (
                        event_id, task_id, recording_id, user, event_type,
                        target_json, before_json, after_json, created_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        claim_event_id,
                        task_id,
                        recording_id,
                        user,
                        "promotion_retry_started",
                        _json_dumps(target),
                        None,
                        _json_dumps(after),
                        now,
                    ),
                )
                conn.commit()
                return {
                    "status": "claimed",
                    "event": {
                        "event_id": claim_event_id,
                        "task_id": task_id,
                        "recording_id": recording_id,
                        "user": user,
                        "event_type": "promotion_retry_started",
                        "target": target,
                        "before": None,
                        "after": after,
                        "created_at_utc": now,
                    },
                }
            except Exception:
                conn.rollback()
                raise

    def record_event(
        self,
        *,
        task_id: str,
        recording_id: str,
        user: str,
        event_type: str,
        target: Mapping[str, object] | Sequence[object] | None = None,
        before: Mapping[str, object] | Sequence[object] | None = None,
        after: Mapping[str, object] | Sequence[object] | None = None,
    ) -> dict[str, object]:
        self.initialize()
        event_id = str(uuid.uuid4())
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO labeling_task_events (
                event_id, task_id, recording_id, user, event_type,
                target_json, before_json, after_json, created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                event_id,
                str(task_id),
                str(recording_id),
                str(user),
                str(event_type),
                _json_dumps(target or {}),
                _json_dumps(before or {}),
                _json_dumps(after or {}),
                now,
            ),
        )
        self.conn.commit()
        return {
            "event_id": event_id,
            "task_id": str(task_id),
            "recording_id": str(recording_id),
            "user": str(user),
            "event_type": str(event_type),
            "created_at_utc": now,
        }

    def list_events(
        self,
        *,
        task_id: str | None = None,
        event_type: str | None = None,
        recording_id: str | None = None,
        assignee_user: str | None = None,
        actor_user: str | None = None,
        since_utc: str | None = None,
        until_utc: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        self.initialize()
        sql = [
            """
            SELECT e.*, t.workflow_kind, t.dataset_id, t.title, t.component_name,
                   a.assignee_user, a.status AS assignment_status
            FROM labeling_task_events e
            JOIN labeling_tasks t ON t.task_id = e.task_id
            LEFT JOIN recording_assignments a ON a.recording_id = e.recording_id
            WHERE 1=1
            """
        ]
        params: list[object] = []
        if task_id:
            sql.append("AND e.task_id = ?")
            params.append(str(task_id))
        if event_type:
            sql.append("AND e.event_type = ?")
            params.append(str(event_type))
        if recording_id:
            sql.append("AND e.recording_id = ?")
            params.append(str(recording_id))
        if assignee_user:
            sql.append("AND a.assignee_user = ? AND a.status = 'active'")
            params.append(str(assignee_user))
        if actor_user:
            sql.append("AND e.user = ?")
            params.append(str(actor_user))
        if since_utc:
            sql.append("AND e.created_at_utc >= ?")
            params.append(str(since_utc))
        if until_utc:
            sql.append("AND e.created_at_utc <= ?")
            params.append(str(until_utc))
        sql.append("ORDER BY e.created_at_utc DESC LIMIT ?")
        params.append(max(1, int(limit)))
        rows = self.conn.execute(" ".join(sql), params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_session_closure_event(self, session_id: str) -> dict[str, object] | None:
        """Return the latest audit event explaining why a browser session closed."""

        self.initialize()
        session_id = str(session_id).strip()
        if not session_id:
            return None
        closure_event_types = (
            "session_closed_by_assignment_change",
            "session_closed_by_task_completion",
            "session_superseded",
            "stale_session_closed",
            "session_closed",
        )
        placeholders = ", ".join("?" for _ in closure_event_types)
        rows = self.conn.execute(
            f"""
            SELECT e.*, t.workflow_kind, t.dataset_id, t.title, t.component_name,
                   a.assignee_user, a.status AS assignment_status
            FROM labeling_task_events e
            JOIN labeling_tasks t ON t.task_id = e.task_id
            LEFT JOIN recording_assignments a ON a.recording_id = e.recording_id
            WHERE e.event_type IN ({placeholders})
              AND e.target_json LIKE ?
            ORDER BY e.created_at_utc DESC
            LIMIT 50;
            """,
            (*closure_event_types, f"%{json.dumps(session_id, ensure_ascii=True)}%"),
        ).fetchall()
        for row in rows:
            event = _row_to_dict(row)
            target = event.get("target")
            if isinstance(target, Mapping) and str(target.get("session_id") or "") == session_id:
                return event
        return None

    def get_event(self, event_id: str) -> dict[str, object] | None:
        self.initialize()
        row = self.conn.execute(
            """
            SELECT e.*, t.workflow_kind, t.dataset_id, t.title, t.component_name,
                   t.scope_json, a.assignee_user, a.status AS assignment_status
            FROM labeling_task_events e
            JOIN labeling_tasks t ON t.task_id = e.task_id
            LEFT JOIN recording_assignments a ON a.recording_id = e.recording_id
            WHERE e.event_id = ?;
            """,
            (str(event_id),),
        ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def task_summary_for_user(self, user: str, *, include_completed: bool = False) -> dict[str, object]:
        all_tasks = self.list_tasks_for_user(user, include_completed=True)
        tasks = all_tasks if include_completed else [task for task in all_tasks if str(task.get("state") or "") != "complete"]
        recordings: dict[str, dict[str, object]] = {}
        for assignment in self.list_assignments(assignee_user=str(user), status="active"):
            recording_id = str(assignment.get("recording_id") or "")
            if not recording_id:
                continue
            recordings.setdefault(
                recording_id,
                {
                    "recording_id": recording_id,
                    "assignee_user": assignment.get("assignee_user"),
                    "assignment_notes": assignment.get("notes"),
                    "tasks": [],
                    "counts": {},
                    "total_task_count": 0,
                    "incomplete_task_count": 0,
                    "startable_task_count": 0,
                    "non_startable_task_count": 0,
                    "complete_task_count": 0,
                    "workflow_counts": {},
                    "workflow_state_counts": {},
                    "component_counts": {},
                },
            )
        visible_recording_ids = set(recordings)
        for task in all_tasks:
            recording_id = str(task["recording_id"])
            if recording_id not in visible_recording_ids:
                continue
            entry = recordings.setdefault(
                recording_id,
                {
                    "recording_id": recording_id,
                    "assignee_user": task.get("assignee_user"),
                    "assignment_notes": task.get("assignment_notes"),
                    "tasks": [],
                    "counts": {},
                    "total_task_count": 0,
                    "incomplete_task_count": 0,
                    "startable_task_count": 0,
                    "non_startable_task_count": 0,
                    "complete_task_count": 0,
                    "workflow_counts": {},
                    "workflow_state_counts": {},
                    "component_counts": {},
                },
            )
            state = str(task.get("state") or "unknown")
            startable = state in LABELER_START_TASK_STATES
            if include_completed or state != "complete":
                entry["tasks"].append(task)  # type: ignore[index, union-attr]
            counts = entry["counts"]  # type: ignore[assignment]
            assert isinstance(counts, dict)
            counts[state] = int(counts.get(state, 0)) + 1
            entry["total_task_count"] = int(entry["total_task_count"]) + 1
            if state == "complete":
                entry["complete_task_count"] = int(entry["complete_task_count"]) + 1
            else:
                entry["incomplete_task_count"] = int(entry["incomplete_task_count"]) + 1
                if startable:
                    entry["startable_task_count"] = int(entry["startable_task_count"]) + 1
                else:
                    entry["non_startable_task_count"] = int(entry["non_startable_task_count"]) + 1
            workflow_counts = entry["workflow_counts"]  # type: ignore[assignment]
            assert isinstance(workflow_counts, dict)
            workflow_kind = str(task.get("workflow_kind") or "unknown")
            workflow_counts[workflow_kind] = int(workflow_counts.get(workflow_kind, 0)) + 1
            workflow_state_counts = entry["workflow_state_counts"]  # type: ignore[assignment]
            assert isinstance(workflow_state_counts, dict)
            workflow_entry = workflow_state_counts.setdefault(
                workflow_kind,
                {"total": 0, "incomplete": 0, "startable": 0, "non_startable": 0, "complete": 0, "states": {}},
            )
            assert isinstance(workflow_entry, dict)
            workflow_entry["total"] = int(workflow_entry.get("total", 0)) + 1
            workflow_entry["complete" if state == "complete" else "incomplete"] = int(
                workflow_entry.get("complete" if state == "complete" else "incomplete", 0)
            ) + 1
            if state != "complete":
                workflow_entry["startable" if startable else "non_startable"] = int(
                    workflow_entry.get("startable" if startable else "non_startable", 0)
                ) + 1
            workflow_states = workflow_entry.setdefault("states", {})
            assert isinstance(workflow_states, dict)
            workflow_states[state] = int(workflow_states.get(state, 0)) + 1
            component = str(task.get("component_name") or "").strip()
            if component:
                component_counts = entry["component_counts"]  # type: ignore[assignment]
                assert isinstance(component_counts, dict)
                component_counts[component] = int(component_counts.get(component, 0)) + 1
            if include_completed or state != "complete":
                entry_tasks = entry["tasks"]  # type: ignore[assignment]
                assert isinstance(entry_tasks, list)
                entry_tasks[-1] = _redact_user_summary_row(task)
        for entry in recordings.values():
            no_open_task_summary = _no_open_task_summary(entry)
            if no_open_task_summary is None:
                continue
            reason, message = no_open_task_summary
            entry["no_open_task_reason"] = reason
            entry["no_open_task_message"] = message
        failed_promotion_events = self.list_events(
            event_type="promotion_failed",
            assignee_user=str(user),
            limit=200,
        )
        promotion_success_events = self.list_events(
            event_type="promotion_success",
            assignee_user=str(user),
            limit=500,
        )
        resolved_failed_event_ids: set[str] = set()
        for event in promotion_success_events:
            target = event.get("target")
            if isinstance(target, Mapping):
                retry_of = str(target.get("retry_of_event_id") or "").strip()
                if retry_of:
                    resolved_failed_event_ids.add(retry_of)
        failed_promotions = [
            event
            for event in failed_promotion_events
            if str(event.get("event_id") or "") not in resolved_failed_event_ids
        ][:50]
        redacted_failed_promotions = [
            _redact_user_summary_row(event)
            for event in failed_promotions
        ]
        return {
            "user": str(user),
            "recording_count": len(recordings),
            "task_count": len(tasks),
            "total_task_count": len(all_tasks),
            "complete_task_count": sum(1 for task in all_tasks if str(task.get("state") or "") == "complete"),
            "incomplete_task_count": sum(1 for task in all_tasks if str(task.get("state") or "") != "complete"),
            "startable_task_count": sum(1 for task in all_tasks if str(task.get("state") or "") in LABELER_START_TASK_STATES),
            "non_startable_task_count": sum(
                1
                for task in all_tasks
                if str(task.get("state") or "") != "complete"
                and str(task.get("state") or "") not in LABELER_START_TASK_STATES
            ),
            "failed_promotion_count": len(redacted_failed_promotions),
            "failed_promotions": redacted_failed_promotions,
            "recordings": list(recordings.values()),
        }
