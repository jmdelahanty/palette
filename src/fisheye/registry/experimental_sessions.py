"""Explicit experimental-session identities for cross-recording grouping.

``session_uuid`` identifies one acquisition/recording surface.  An
``experimental_session_id`` identifies the experimental unit shared by recordings
that were acquired together (for example, simultaneous arena recordings).  This
module intentionally contains no timestamp-, path-, or name-based inference.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping
from uuid import UUID, uuid4


EXPERIMENTAL_SESSION_SCHEMA_ID = "palette.registry.experimental_session.v1"
EXPERIMENTAL_SESSION_ASSIGNMENT_SCHEMA_ID = (
    "palette.registry.experimental_session_assignment.v1"
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+~-]{0,254}$")
_METHOD_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ExperimentalSessionIdentityError(ValueError):
    """Base error for invalid or unavailable experimental-session identity."""


class UnknownExperimentalSessionError(ExperimentalSessionIdentityError):
    """Raised when a requested experimental-session entity does not exist."""


class UnknownRecordingIdentityError(ExperimentalSessionIdentityError):
    """Raised when a requested recording identity does not exist."""


class UnknownDatasetIdentityError(ExperimentalSessionIdentityError):
    """Raised when a requested dataset identity does not exist."""


class MissingExperimentalSessionIdentityError(ExperimentalSessionIdentityError):
    """Raised when an existing recording has no explicit session assignment."""


class ExperimentalSessionAssignmentConflictError(ExperimentalSessionIdentityError):
    """Raised when current assignment authority conflicts with a requested write."""


@dataclass(frozen=True)
class ExperimentalSessionRecord:
    experimental_session_id: str
    session_snapshot_id: str
    schema_id: str
    creation_method: str
    created_by: str
    created_at_utc: str
    evidence: Mapping[str, Any]
    registry_schema_version: int


@dataclass(frozen=True)
class ExperimentalSessionAssignment:
    recording_id: str
    experimental_session_id: str
    assignment_snapshot_id: str
    assignment_revision: int
    supersedes_assignment_snapshot_id: str | None
    assignment_batch_id: str
    schema_id: str
    assignment_method: str
    assigned_by: str
    assigned_at_utc: str
    evidence: Mapping[str, Any]
    registry_schema_version: int


def validate_experimental_session_id(value: str) -> str:
    """Validate and return an exact persisted experimental-session ID."""

    text = str(value)
    if text != text.strip() or not _IDENTIFIER_RE.fullmatch(text):
        raise ExperimentalSessionIdentityError(
            "experimental_session_id must be a 1-255 character identifier using "
            "letters, digits, '.', '_', ':', '@', '/', '+', '~', or '-'; leading "
            "and trailing whitespace are forbidden."
        )
    return text


def _validate_recording_id(value: str) -> str:
    text = str(value)
    if not text or text != text.strip() or any(ord(char) < 32 for char in text):
        raise ExperimentalSessionIdentityError(
            "recording_id must be a non-empty exact identifier without surrounding "
            "whitespace or control characters."
        )
    return text


def _validate_method(value: str, *, field: str) -> str:
    text = str(value)
    if not _METHOD_RE.fullmatch(text):
        raise ExperimentalSessionIdentityError(
            f"{field} must be a lower_snake_case identifier of at most 64 characters."
        )
    return text


def _validate_actor(value: str, *, field: str) -> str:
    text = str(value)
    if not text or text != text.strip() or any(ord(char) < 32 for char in text):
        raise ExperimentalSessionIdentityError(
            f"{field} must be a non-empty exact value without surrounding whitespace "
            "or control characters."
        )
    return text


def _canonical_evidence(
    value: Mapping[str, Any] | None,
) -> tuple[str, Mapping[str, Any]]:
    if value is None:
        payload: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise ExperimentalSessionIdentityError(
            "evidence must be a JSON object mapping."
        )
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentalSessionIdentityError(
            "evidence must be strict JSON without non-finite values."
        ) from exc
    loaded = json.loads(text)
    if not isinstance(loaded, dict):  # defensive: Mapping was normalized above
        raise ExperimentalSessionIdentityError("evidence must encode a JSON object.")
    return text, loaded


def _canonical_utc(value: str | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat()
    text = str(value)
    if text != text.strip() or not text:
        raise ExperimentalSessionIdentityError(
            "UTC timestamp must be a non-empty exact value."
        )
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExperimentalSessionIdentityError(
            f"Invalid ISO-8601 UTC timestamp: {text!r}."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ExperimentalSessionIdentityError(
            "Timestamp must carry an explicit UTC offset."
        )
    return parsed.astimezone(UTC).isoformat()


def _canonical_uuid(value: str | None, *, field: str) -> str:
    if value is None:
        return str(uuid4())
    text = str(value)
    try:
        parsed = UUID(text)
    except ValueError as exc:
        raise ExperimentalSessionIdentityError(
            f"{field} must be a canonical UUID."
        ) from exc
    if str(parsed) != text:
        raise ExperimentalSessionIdentityError(
            f"{field} must be a canonical lowercase UUID."
        )
    return text


def _mapping_from_json(text: str) -> Mapping[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ExperimentalSessionIdentityError(
            "Persisted evidence is not a JSON object."
        )
    return value


def _session_from_row(row: sqlite3.Row) -> ExperimentalSessionRecord:
    return ExperimentalSessionRecord(
        experimental_session_id=str(row["experimental_session_id"]),
        session_snapshot_id=str(row["session_snapshot_id"]),
        schema_id=str(row["schema_id"]),
        creation_method=str(row["creation_method"]),
        created_by=str(row["created_by"]),
        created_at_utc=str(row["created_at_utc"]),
        evidence=_mapping_from_json(str(row["evidence_json"])),
        registry_schema_version=int(row["registry_schema_version"]),
    )


def _assignment_from_row(row: sqlite3.Row) -> ExperimentalSessionAssignment:
    return ExperimentalSessionAssignment(
        recording_id=str(row["recording_id"]),
        experimental_session_id=str(row["experimental_session_id"]),
        assignment_snapshot_id=str(row["assignment_snapshot_id"]),
        assignment_revision=int(row["assignment_revision"]),
        supersedes_assignment_snapshot_id=(
            str(row["supersedes_assignment_snapshot_id"])
            if row["supersedes_assignment_snapshot_id"] is not None
            else None
        ),
        assignment_batch_id=str(row["assignment_batch_id"]),
        schema_id=str(row["schema_id"]),
        assignment_method=str(row["assignment_method"]),
        assigned_by=str(row["assigned_by"]),
        assigned_at_utc=str(row["assigned_at_utc"]),
        evidence=_mapping_from_json(str(row["evidence_json"])),
        registry_schema_version=int(row["registry_schema_version"]),
    )


def create_experimental_session_record(
    conn: sqlite3.Connection,
    *,
    experimental_session_id: str,
    creation_method: str,
    created_by: str,
    evidence: Mapping[str, Any] | None,
    registry_schema_version: int,
    created_at_utc: str | None = None,
    session_snapshot_id: str | None = None,
) -> ExperimentalSessionRecord:
    """Create one explicit session entity; existing IDs fail closed."""

    session_id = validate_experimental_session_id(experimental_session_id)
    method = _validate_method(creation_method, field="creation_method")
    actor = _validate_actor(created_by, field="created_by")
    timestamp = _canonical_utc(created_at_utc)
    snapshot_id = _canonical_uuid(session_snapshot_id, field="session_snapshot_id")
    evidence_json, _ = _canonical_evidence(evidence)
    if int(registry_schema_version) <= 0:
        raise ExperimentalSessionIdentityError(
            "registry_schema_version must be positive."
        )
    existing = conn.execute(
        "SELECT 1 FROM experimental_sessions WHERE experimental_session_id = ?;",
        (session_id,),
    ).fetchone()
    if existing is not None:
        raise ExperimentalSessionAssignmentConflictError(
            f"Experimental session {session_id!r} already exists; session entities are immutable."
        )
    conn.execute(
        """
        INSERT INTO experimental_sessions (
            experimental_session_id, session_snapshot_id, schema_id,
            creation_method, created_by, created_at_utc, evidence_json,
            registry_schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            session_id,
            snapshot_id,
            EXPERIMENTAL_SESSION_SCHEMA_ID,
            method,
            actor,
            timestamp,
            evidence_json,
            int(registry_schema_version),
        ),
    )
    return get_experimental_session_record(conn, session_id)


def get_experimental_session_record(
    conn: sqlite3.Connection,
    experimental_session_id: str,
) -> ExperimentalSessionRecord:
    session_id = validate_experimental_session_id(experimental_session_id)
    row = conn.execute(
        "SELECT * FROM experimental_sessions WHERE experimental_session_id = ?;",
        (session_id,),
    ).fetchone()
    if row is None:
        raise UnknownExperimentalSessionError(
            f"Experimental session {session_id!r} is not registered."
        )
    return _session_from_row(row)


def assign_recordings(
    conn: sqlite3.Connection,
    *,
    experimental_session_id: str,
    recording_ids: Iterable[str],
    assignment_method: str,
    assigned_by: str,
    evidence: Mapping[str, Any] | None,
    registry_schema_version: int,
    assigned_at_utc: str | None = None,
    assignment_batch_id: str | None = None,
) -> tuple[ExperimentalSessionAssignment, ...]:
    """Atomically assign recordings without replacing any existing assignment."""

    session_id = validate_experimental_session_id(experimental_session_id)
    method = _validate_method(assignment_method, field="assignment_method")
    actor = _validate_actor(assigned_by, field="assigned_by")
    timestamp = _canonical_utc(assigned_at_utc)
    batch_id = _canonical_uuid(assignment_batch_id, field="assignment_batch_id")
    evidence_json, _ = _canonical_evidence(evidence)
    if int(registry_schema_version) <= 0:
        raise ExperimentalSessionIdentityError(
            "registry_schema_version must be positive."
        )

    normalized_recordings = tuple(
        _validate_recording_id(value) for value in recording_ids
    )
    if not normalized_recordings:
        raise ExperimentalSessionIdentityError("At least one recording_id is required.")
    if len(set(normalized_recordings)) != len(normalized_recordings):
        raise ExperimentalSessionIdentityError(
            "recording_ids must be unique; duplicate assignments are ambiguous."
        )
    get_experimental_session_record(conn, session_id)

    placeholders = ",".join("?" for _ in normalized_recordings)
    known = {
        str(row["recording_id"])
        for row in conn.execute(
            f"SELECT recording_id FROM recordings WHERE recording_id IN ({placeholders});",
            normalized_recordings,
        ).fetchall()
    }
    missing = sorted(set(normalized_recordings) - known)
    if missing:
        raise UnknownRecordingIdentityError(
            "Cannot assign unknown recording identities: " + ", ".join(missing)
        )

    conflicts = conn.execute(
        f"""
        SELECT current.recording_id, assignment.experimental_session_id
        FROM recording_experimental_session_current current
        JOIN recording_experimental_session_assignments assignment
          ON assignment.assignment_snapshot_id = current.assignment_snapshot_id
        WHERE current.recording_id IN ({placeholders})
        ORDER BY current.recording_id;
        """,
        normalized_recordings,
    ).fetchall()
    if conflicts:
        details = ", ".join(
            f"{row['recording_id']}->{row['experimental_session_id']}"
            for row in conflicts
        )
        raise ExperimentalSessionAssignmentConflictError(
            "Recording already has explicit current experimental-session authority; "
            f"use the audited correction API. Existing assignments: {details}."
        )

    for recording_id in normalized_recordings:
        assignment_snapshot_id = str(uuid4())
        conn.execute(
            """
            INSERT INTO recording_experimental_session_assignments (
                assignment_snapshot_id, recording_id, experimental_session_id,
                assignment_revision, supersedes_assignment_snapshot_id,
                assignment_batch_id, schema_id, assignment_method,
                assigned_by, assigned_at_utc, evidence_json,
                registry_schema_version
            ) VALUES (?, ?, ?, 1, NULL, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                assignment_snapshot_id,
                recording_id,
                session_id,
                batch_id,
                EXPERIMENTAL_SESSION_ASSIGNMENT_SCHEMA_ID,
                method,
                actor,
                timestamp,
                evidence_json,
                int(registry_schema_version),
            ),
        )
        conn.execute(
            """
            INSERT INTO recording_experimental_session_current (
                recording_id, assignment_snapshot_id, updated_at_utc
            ) VALUES (?, ?, ?);
            """,
            (recording_id, assignment_snapshot_id, timestamp),
        )
    return tuple(
        get_recording_assignment(conn, value) for value in normalized_recordings
    )


def correct_recording_assignment(
    conn: sqlite3.Connection,
    *,
    recording_id: str,
    experimental_session_id: str,
    expected_current_assignment_snapshot_id: str,
    assignment_method: str,
    assigned_by: str,
    evidence: Mapping[str, Any] | None,
    registry_schema_version: int,
    assigned_at_utc: str | None = None,
    assignment_batch_id: str | None = None,
    assignment_snapshot_id: str | None = None,
) -> ExperimentalSessionAssignment:
    """Append a correction and atomically move the explicit current pointer.

    The expected snapshot is a compare-and-swap guard.  Stale callers cannot
    overwrite a correction made after they read the assignment.
    """

    normalized_recording_id = _validate_recording_id(recording_id)
    session_id = validate_experimental_session_id(experimental_session_id)
    expected_snapshot_id = _canonical_uuid(
        expected_current_assignment_snapshot_id,
        field="expected_current_assignment_snapshot_id",
    )
    new_snapshot_id = _canonical_uuid(
        assignment_snapshot_id,
        field="assignment_snapshot_id",
    )
    batch_id = _canonical_uuid(assignment_batch_id, field="assignment_batch_id")
    method = _validate_method(assignment_method, field="assignment_method")
    actor = _validate_actor(assigned_by, field="assigned_by")
    timestamp = _canonical_utc(assigned_at_utc)
    evidence_json, _ = _canonical_evidence(evidence)
    if int(registry_schema_version) <= 0:
        raise ExperimentalSessionIdentityError(
            "registry_schema_version must be positive."
        )
    get_experimental_session_record(conn, session_id)

    current = get_recording_assignment(conn, normalized_recording_id)
    assert current is not None  # require_assigned=True above
    if current.assignment_snapshot_id != expected_snapshot_id:
        raise ExperimentalSessionAssignmentConflictError(
            "Experimental-session correction compare-and-swap failed for "
            f"{normalized_recording_id!r}: expected {expected_snapshot_id!r}, "
            f"current is {current.assignment_snapshot_id!r}."
        )
    next_revision = current.assignment_revision + 1
    conn.execute(
        """
        INSERT INTO recording_experimental_session_assignments (
            assignment_snapshot_id, recording_id, experimental_session_id,
            assignment_revision, supersedes_assignment_snapshot_id,
            assignment_batch_id, schema_id, assignment_method, assigned_by,
            assigned_at_utc, evidence_json, registry_schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            new_snapshot_id,
            normalized_recording_id,
            session_id,
            next_revision,
            current.assignment_snapshot_id,
            batch_id,
            EXPERIMENTAL_SESSION_ASSIGNMENT_SCHEMA_ID,
            method,
            actor,
            timestamp,
            evidence_json,
            int(registry_schema_version),
        ),
    )
    updated = conn.execute(
        """
        UPDATE recording_experimental_session_current
        SET assignment_snapshot_id = ?, updated_at_utc = ?
        WHERE recording_id = ? AND assignment_snapshot_id = ?;
        """,
        (
            new_snapshot_id,
            timestamp,
            normalized_recording_id,
            expected_snapshot_id,
        ),
    )
    if updated.rowcount != 1:
        raise ExperimentalSessionAssignmentConflictError(
            "Experimental-session current authority changed during correction."
        )
    corrected = get_recording_assignment(conn, normalized_recording_id)
    assert corrected is not None
    return corrected


def list_recording_assignment_history(
    conn: sqlite3.Connection,
    recording_id: str,
) -> tuple[ExperimentalSessionAssignment, ...]:
    normalized_recording_id = _validate_recording_id(recording_id)
    recording = conn.execute(
        "SELECT 1 FROM recordings WHERE recording_id = ?;",
        (normalized_recording_id,),
    ).fetchone()
    if recording is None:
        raise UnknownRecordingIdentityError(
            f"Recording {normalized_recording_id!r} is not registered."
        )
    rows = conn.execute(
        """
        SELECT * FROM recording_experimental_session_assignments
        WHERE recording_id = ?
        ORDER BY assignment_revision;
        """,
        (normalized_recording_id,),
    ).fetchall()
    return tuple(_assignment_from_row(row) for row in rows)


def get_recording_assignment(
    conn: sqlite3.Connection,
    recording_id: str,
    *,
    require_assigned: bool = True,
) -> ExperimentalSessionAssignment | None:
    normalized_recording_id = _validate_recording_id(recording_id)
    recording = conn.execute(
        "SELECT 1 FROM recordings WHERE recording_id = ?;",
        (normalized_recording_id,),
    ).fetchone()
    if recording is None:
        raise UnknownRecordingIdentityError(
            f"Recording {normalized_recording_id!r} is not registered."
        )
    row = conn.execute(
        """
        SELECT assignment.*
        FROM recording_experimental_session_current current
        JOIN recording_experimental_session_assignments assignment
          ON assignment.assignment_snapshot_id = current.assignment_snapshot_id
        WHERE current.recording_id = ?;
        """,
        (normalized_recording_id,),
    ).fetchone()
    if row is None:
        if require_assigned:
            raise MissingExperimentalSessionIdentityError(
                f"Recording {normalized_recording_id!r} has no explicit "
                "experimental_session_id assignment."
            )
        return None
    return _assignment_from_row(row)


def resolve_dataset_assignment(
    conn: sqlite3.Connection,
    dataset_id: str,
    *,
    require_assigned: bool = True,
) -> ExperimentalSessionAssignment | None:
    normalized_dataset_id = str(dataset_id)
    if (
        not normalized_dataset_id
        or normalized_dataset_id != normalized_dataset_id.strip()
        or any(ord(char) < 32 for char in normalized_dataset_id)
    ):
        raise ExperimentalSessionIdentityError(
            "dataset_id must be a non-empty exact value without surrounding whitespace "
            "or control characters."
        )
    row = conn.execute(
        "SELECT recording_id FROM datasets WHERE dataset_id = ?;",
        (normalized_dataset_id,),
    ).fetchone()
    if row is None:
        raise UnknownDatasetIdentityError(
            f"Dataset {normalized_dataset_id!r} is not registered."
        )
    recording_id = row["recording_id"]
    if recording_id is None or not str(recording_id).strip():
        if require_assigned:
            raise MissingExperimentalSessionIdentityError(
                f"Dataset {normalized_dataset_id!r} has no recording identity and therefore "
                "cannot have an experimental-session assignment."
            )
        return None
    return get_recording_assignment(
        conn,
        str(recording_id),
        require_assigned=require_assigned,
    )


__all__ = [
    "EXPERIMENTAL_SESSION_ASSIGNMENT_SCHEMA_ID",
    "EXPERIMENTAL_SESSION_SCHEMA_ID",
    "ExperimentalSessionAssignment",
    "ExperimentalSessionAssignmentConflictError",
    "ExperimentalSessionIdentityError",
    "ExperimentalSessionRecord",
    "MissingExperimentalSessionIdentityError",
    "UnknownDatasetIdentityError",
    "UnknownExperimentalSessionError",
    "UnknownRecordingIdentityError",
    "assign_recordings",
    "correct_recording_assignment",
    "create_experimental_session_record",
    "get_experimental_session_record",
    "get_recording_assignment",
    "list_recording_assignment_history",
    "resolve_dataset_assignment",
    "validate_experimental_session_id",
]
