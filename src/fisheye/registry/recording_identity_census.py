"""Read-only census of recording identity evidence and registry projections.

This module is deliberately observational.  It does not choose an authority,
repair metadata, instantiate :class:`Registry`, or expose a value suitable for
write-back.  The report preserves disagreements between ``recording_id``,
``session_uuid``, and legacy session labels as separate facts.

The filesystem scan reads metadata files directly.  It never opens a Zarr
array, decodes video, or falls back from direct to consolidated metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fisheye.registry.prune_stale_datasets import connect_read_only


REPORT_SCHEMA_ID = "palette.recording_identity_census.v1"
DEFAULT_MAX_JSON_BYTES = 64 * 1024 * 1024
ROOT_PREFIX_READ_THRESHOLD_BYTES = 8 * 1024 * 1024
OVERSIZED_ROOT_METADATA_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_ZARR_NODES = 10_000
DEFAULT_MAX_OBSERVATIONS_PER_SOURCE = 100_000
DEFAULT_FINDING_SAMPLE_LIMIT = 50
DEFAULT_MAX_PARQUET_DISTINCT_VALUES = 1_024
DEFAULT_MAX_PARQUET_MALFORMED_SAMPLES = 50

IDENTITY_COLUMN_MARKERS = (
    "dataset_id",
    "recording_id",
    "session_uuid",
    "session_id",
)
DIRECT_IDENTITY_FIELDS: dict[str, str] = {
    "dataset_id": "dataset_id",
    "source_dataset_id": "source_dataset_id",
    "recording_id": "recording_id",
    "source_recording_id": "source_recording_id",
    "organizer_recording_id": "organizer_recording_id",
    "session_uuid": "session_uuid",
    "source_session_uuid": "source_session_uuid",
    "session_id": "legacy_session_id",
    "orange_session_id": "orange_session_id",
    "camera_id": "camera_id",
    "camera_serial": "camera_id",
}
LIST_IDENTITY_FIELDS: dict[str, str] = {
    "dataset_ids": "dataset_id",
    "recording_ids": "recording_id",
    "session_uuids": "session_uuid",
    "camera_ids": "camera_id",
    "camera_serials": "camera_id",
}
DONOR_LOCATOR_FIELDS = {
    "copy_analysis_metadata_from",
    "copy_existing_detections_from",
    "analysis_metadata_source_zarr",
    "dish_mask_source_zarr",
    "experiment_setup_source_zarr",
    "copied_detection_runs_from",
}
IDENTITY_CONTAINER_FIELDS = {
    "acquisition_index_mapping",
    "analysis_context_source",
    "camera_artifacts",
    "clips",
    "copied_detection_result",
    "dataset",
    "dataset_context",
    "experiment_setup_record",
    "profile_summary",
    "recording",
    "recording_outputs",
    "rolling_clip_streams",
    "rows",
    "session_context",
    "source",
    "source_dataset",
    "source_video_metadata",
    "subject_metadata_record",
    "subject_metadata_ref",
    "video_streams",
}
CONTRACT_MARKER_FIELDS = (
    "artifact_schema_id",
    "artifact_schema_version",
    "source_layout",
    "source_frame_index_schema",
)
SCALAR_COMPARISON_FACTS = {
    "dataset_id",
    "source_dataset_id",
    "recording_id",
    "source_recording_id",
    "organizer_recording_id",
    "session_uuid",
    "source_session_uuid",
    "legacy_session_id",
    "orange_session_id",
}
RECORDING_SIDECAR_RELATIVE_PATHS = (
    Path("recording_manifest.json"),
    Path("recording_session.json"),
    Path("raw/recording_session.json"),
    Path("recording_clip_index.json"),
    Path("recording_frame_index_manifest.json"),
)


class CensusError(RuntimeError):
    """Base error for a census that could not produce trustworthy evidence."""


class UnstableSnapshotError(CensusError):
    """Raised when a registry or metadata source changes during observation."""


@dataclass(frozen=True)
class FileFence:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str
    binding_kind: str
    bytes_hashed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
            "binding_kind": self.binding_kind,
            "bytes_hashed": self.bytes_hashed,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"value_type": "non_finite_float", "repr": repr(value)}
    if isinstance(value, bytes):
        return {
            "value_type": "bytes",
            "size": len(value),
            "sha256": _sha256_bytes(value),
            "hex_prefix": value[:32].hex(),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return {"value_type": type(value).__name__, "repr": repr(value)}


def _identity_text(value: Any) -> tuple[str | None, str | None]:
    """Return an exact valid identity string and an optional defect code."""

    if value is None:
        return None, "missing"
    if not isinstance(value, str):
        return None, "non_string"
    if not value:
        return None, "empty"
    if value != value.strip():
        return None, "surrounding_whitespace"
    return value, None


def _path_state(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"exists": False}
    return {
        "exists": True,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _registry_source_state(path: Path) -> dict[str, Any]:
    return {
        "database": _path_state(path),
        "journal": _path_state(Path(f"{path}-journal")),
        "shm": _path_state(Path(f"{path}-shm")),
        "wal": _path_state(Path(f"{path}-wal")),
    }


def _stable_read_bytes(path: Path, *, max_bytes: int) -> tuple[bytes, FileFence]:
    before = path.stat()
    if not path.is_file():
        raise CensusError(f"Not a regular metadata file: {path}")
    if before.st_size > max_bytes:
        raise CensusError(
            f"Metadata file exceeds {max_bytes} byte census limit: {path} "
            f"({before.st_size} bytes)"
        )
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        payload = handle.read(max_bytes + 1)
        closed = os.fstat(handle.fileno())
    after = path.stat()
    signatures = {
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
        (closed.st_dev, closed.st_ino, closed.st_size, closed.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
    }
    if len(payload) > max_bytes:
        raise CensusError(f"Metadata file grew beyond census limit while reading: {path}")
    if len(signatures) != 1:
        raise UnstableSnapshotError(f"Metadata source changed while reading: {path}")
    return payload, FileFence(
        path=path,
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=_sha256_bytes(payload),
        binding_kind="full_content_sha256",
        bytes_hashed=len(payload),
    )


def _stable_read_prefix(path: Path, *, prefix_bytes: int) -> tuple[bytes, FileFence]:
    before = path.stat()
    if not path.is_file():
        raise CensusError(f"Not a regular metadata file: {path}")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        payload = handle.read(prefix_bytes)
        closed = os.fstat(handle.fileno())
    after = path.stat()
    signatures = {
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
        (closed.st_dev, closed.st_ino, closed.st_size, closed.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
    }
    if len(signatures) != 1:
        raise UnstableSnapshotError(f"Metadata source changed while reading prefix: {path}")
    return payload, FileFence(
        path=path,
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=_sha256_bytes(payload),
        binding_kind="prefix_sha256",
        bytes_hashed=len(payload),
    )


def _zarr_root_prefix_metadata(path: Path, *, max_prefix_bytes: int) -> tuple[Mapping[str, Any], FileFence]:
    """Decode top-level root fields before a potentially huge inline consolidation."""

    payload, fence = _stable_read_prefix(path, prefix_bytes=max_prefix_bytes)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = item
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON number {token}")

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CensusError(f"Invalid UTF-8 Zarr root metadata {path}: {exc}") from exc
    decoder = json.JSONDecoder(object_pairs_hook=unique_object, parse_constant=reject_constant)

    def skip_space(index: int) -> int:
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    index = skip_space(0)
    if index >= len(text) or text[index] != "{":
        raise CensusError(f"Zarr root metadata is not an object: {path}")
    index += 1
    selected: dict[str, Any] = {}
    seen: set[str] = set()
    try:
        while True:
            index = skip_space(index)
            if index < len(text) and text[index] == "}":
                break
            if index < len(text) and text[index] == ",":
                index = skip_space(index + 1)
            key, index = decoder.raw_decode(text, index)
            if not isinstance(key, str):
                raise ValueError("top-level key is not text")
            if key in seen:
                raise ValueError(f"duplicate top-level key {key!r}")
            seen.add(key)
            index = skip_space(index)
            if index >= len(text) or text[index] != ":":
                raise ValueError(f"missing colon after top-level key {key!r}")
            index = skip_space(index + 1)
            if key == "consolidated_metadata":
                break
            item, index = decoder.raw_decode(text, index)
            if key in {"attributes", "zarr_format", "node_type"}:
                selected[key] = item
    except (json.JSONDecodeError, ValueError) as exc:
        raise CensusError(
            f"Could not decode bounded Zarr root prefix {path}; increase --max-json-bytes "
            f"if root attributes exceed {max_prefix_bytes} bytes: {exc}"
        ) from exc
    if not isinstance(selected.get("attributes"), Mapping):
        raise CensusError(f"Zarr root prefix has no object attributes member: {path}")
    if selected.get("zarr_format") != 3:
        raise CensusError(f"Zarr root prefix does not declare zarr_format=3: {path}")
    selected["_prefix_only"] = True
    return selected, fence


def _stable_read_json(path: Path, *, max_bytes: int) -> tuple[Mapping[str, Any], FileFence]:
    payload, fence = _stable_read_bytes(path, max_bytes=max_bytes)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = item
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON number {token}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CensusError(f"Invalid JSON metadata {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CensusError(f"JSON metadata is not an object: {path}")
    return value, fence


def _verify_file_fence(fence: FileFence, *, max_bytes: int) -> None:
    del max_bytes  # The content digest was computed inside the immediate open/stat fence.
    current = fence.path.stat()
    if (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    ) != (fence.device, fence.inode, fence.size, fence.mtime_ns):
        raise UnstableSnapshotError(f"Metadata source changed during census: {fence.path}")


def _fence_summary(fences: Sequence[FileFence]) -> dict[str, Any]:
    rows = [fence.as_dict() for fence in sorted(fences, key=lambda item: str(item.path))]
    return {
        "file_count": len(rows),
        "inventory_digest": _sha256_bytes(_canonical_json_bytes(rows)),
        "binding_kinds": dict(sorted(Counter(row["binding_kind"] for row in rows).items())),
    }


def _capture_registry_snapshot(
    source_path: Path,
    destination: Path,
    *,
    attempts: int = 3,
) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve(strict=True)
    last_reason = "unknown"
    for attempt in range(1, attempts + 1):
        if destination.exists():
            destination.unlink()
        observer = connect_read_only(source_path)
        source = connect_read_only(source_path)
        try:
            observer.execute("PRAGMA read_uncommitted = OFF;")
            source.execute("PRAGMA read_uncommitted = OFF;")
            data_version_before = int(observer.execute("PRAGMA data_version;").fetchone()[0])
            state_before = _registry_source_state(source_path)
            destination_conn = sqlite3.connect(str(destination))
            try:
                source.backup(destination_conn)
            finally:
                destination_conn.close()
            data_version_after = int(observer.execute("PRAGMA data_version;").fetchone()[0])
            state_after = _registry_source_state(source_path)
        finally:
            source.close()
            observer.close()
        if data_version_before == data_version_after and state_before == state_after:
            return {
                "source_path": str(source_path),
                "capture_attempt": attempt,
                "data_version_before": data_version_before,
                "data_version_after": data_version_after,
                "source_state": state_after,
                "snapshot_size": destination.stat().st_size,
                "snapshot_sha256": _sha256_file(destination),
            }
        last_reason = (
            f"data_version {data_version_before}->{data_version_after}; "
            f"file_state_changed={state_before != state_after}"
        )
    raise UnstableSnapshotError(
        f"Registry changed during {attempts} snapshot attempts: {source_path}; {last_reason}"
    )


def _identity_columns(columns: Iterable[str]) -> list[str]:
    return sorted(
        column
        for column in columns
        if any(marker in column.lower() for marker in IDENTITY_COLUMN_MARKERS)
    )


def _column_stats(conn: sqlite3.Connection, object_name: str, column: str) -> dict[str, Any]:
    obj = _quote_identifier(object_name)
    col = _quote_identifier(column)
    row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_rows,
            COUNT(DISTINCT {col}) AS distinct_non_null
        FROM {obj};
        """
    ).fetchone()
    return {
        "total_rows": int(row[0]),
        "null_rows": int(row[1] or 0),
        "non_null_rows": int(row[0]) - int(row[1] or 0),
        "distinct_non_null": int(row[2]),
    }


def _column_distinct_sample(
    conn: sqlite3.Connection,
    object_name: str,
    column: str,
    *,
    limit: int = DEFAULT_FINDING_SAMPLE_LIMIT,
) -> dict[str, Any]:
    obj = _quote_identifier(object_name)
    col = _quote_identifier(column)
    rows = conn.execute(
        f"SELECT DISTINCT {col} FROM {obj} WHERE {col} IS NOT NULL "
        f"ORDER BY CAST({col} AS TEXT) LIMIT ?;",
        (limit + 1,),
    ).fetchall()
    return {
        "values": [_json_safe(row[0]) for row in rows[:limit]],
        "truncated": len(rows) > limit,
        "limit": limit,
    }


def _foreign_keys(conn: sqlite3.Connection, object_name: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA foreign_key_list({_quote_identifier(object_name)});").fetchall()
    return [
        {
            "id": int(row[0]),
            "sequence": int(row[1]),
            "target_table": str(row[2]),
            "source_column": str(row[3]),
            "target_column": str(row[4]),
            "on_update": str(row[5]),
            "on_delete": str(row[6]),
        }
        for row in rows
    ]


def _schema_inventory(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    objects: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    rows = conn.execute(
        """
        SELECT name, type, sql
        FROM sqlite_master
        WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name;
        """
    ).fetchall()
    for row in rows:
        name = str(row[0])
        object_type = str(row[1])
        schema_sql = row[2]
        try:
            xinfo = conn.execute(f"PRAGMA table_xinfo({_quote_identifier(name)});").fetchall()
            columns = [str(info[1]) for info in xinfo]
            identity_columns = _identity_columns(columns)
            if not identity_columns:
                continue
            primary_key_columns = [
                str(info[1])
                for info in sorted(xinfo, key=lambda item: int(item[5] or 0))
                if int(info[5] or 0) > 0
            ]
            column_details = {
                str(info[1]): {
                    "declared_type": str(info[2] or ""),
                    "not_null": bool(info[3]),
                    "primary_key_position": int(info[5] or 0),
                    "hidden": int(info[6] or 0),
                }
                for info in xinfo
                if str(info[1]) in identity_columns
            }
            row_count = int(
                conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(name)};").fetchone()[0]
            )
            for column in identity_columns:
                column_details[column]["counts"] = _column_stats(conn, name, column)
                column_details[column]["distinct_value_sample"] = _column_distinct_sample(
                    conn,
                    name,
                    column,
                )
            objects.append(
                {
                    "object_name": name,
                    "object_type": object_type,
                    "projection_role": "stored_projection" if object_type == "table" else "derived_view",
                    "identity_columns": identity_columns,
                    "primary_key_columns": primary_key_columns,
                    "foreign_keys": _foreign_keys(conn, name) if object_type == "table" else [],
                    "row_count": row_count,
                    "columns": column_details,
                    "schema_sql": schema_sql,
                }
            )
        except sqlite3.Error as exc:
            errors.append(
                {
                    "code": "object_introspection_error",
                    "severity": "action_required",
                    "object_name": name,
                    "object_type": object_type,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return objects, errors


def _row_mapping(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _root_rows(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [
        _row_mapping(row)
        for row in conn.execute(f"SELECT * FROM {_quote_identifier(table)};").fetchall()
    ]


def _finding(
    code: str,
    *,
    severity: str,
    locator: Mapping[str, Any] | None = None,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "locator": _json_safe(dict(locator or {})),
        "detail": _json_safe(dict(detail or {})),
    }


def _dataset_policy_severity(dataset: Mapping[str, Any]) -> str:
    artifact_kind = dataset.get("artifact_kind")
    zarr_use = dataset.get("zarr_use")
    if artifact_kind == "derived_training_merge":
        return "expected"
    if artifact_kind == "source_recording" or zarr_use == "analysis":
        return "action_required"
    return "unresolved"


def _registry_roots(
    conn: sqlite3.Connection,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = _root_rows(conn, "datasets")
    recordings = _root_rows(conn, "recordings")
    findings: list[dict[str, Any]] = []
    dataset_by_id: dict[str, dict[str, Any]] = {}
    recording_by_id: dict[str, dict[str, Any]] = {}

    for row in datasets:
        dataset_id, defect = _identity_text(row.get("dataset_id"))
        if defect is not None:
            findings.append(
                _finding(
                    "malformed_dataset_id",
                    severity="action_required",
                    locator={"table": "datasets"},
                    detail={"value": row.get("dataset_id"), "defect": defect},
                )
            )
            continue
        assert dataset_id is not None
        dataset_by_id[dataset_id] = row

    for row in recordings:
        recording_id, defect = _identity_text(row.get("recording_id"))
        if defect is not None:
            findings.append(
                _finding(
                    "malformed_recording_id",
                    severity="action_required",
                    locator={"table": "recordings"},
                    detail={"value": row.get("recording_id"), "defect": defect},
                )
            )
            continue
        assert recording_id is not None
        recording_by_id[recording_id] = row

    datasets_by_recording: dict[str, list[str]] = defaultdict(list)
    for dataset_id, row in sorted(dataset_by_id.items()):
        recording_id, recording_defect = _identity_text(row.get("recording_id"))
        session_uuid, session_defect = _identity_text(row.get("session_uuid"))
        severity = _dataset_policy_severity(row)
        if recording_defect is not None:
            findings.append(
                _finding(
                    "dataset_parent_recording_null" if recording_defect == "missing" else "malformed_recording_id",
                    severity=severity,
                    locator={"table": "datasets", "dataset_id": dataset_id},
                    detail={"value": row.get("recording_id"), "defect": recording_defect},
                )
            )
        else:
            assert recording_id is not None
            datasets_by_recording[recording_id].append(dataset_id)
            parent = recording_by_id.get(recording_id)
            if parent is None:
                synthetic = row.get("artifact_kind") == "derived_training_merge" and recording_id == dataset_id
                findings.append(
                    _finding(
                        "synthetic_recording_id" if synthetic else "root_recording_missing",
                        severity="expected" if synthetic else severity,
                        locator={"table": "datasets", "dataset_id": dataset_id},
                        detail={"recording_id": recording_id},
                    )
                )
            else:
                parent_session, parent_session_defect = _identity_text(parent.get("session_uuid"))
                if session_defect is None and parent_session_defect is None and session_uuid != parent_session:
                    findings.append(
                        _finding(
                            "root_session_conflict",
                            severity="action_required",
                            locator={"table": "datasets", "dataset_id": dataset_id},
                            detail={
                                "recording_id": recording_id,
                                "dataset_session_uuid": session_uuid,
                                "recording_session_uuid": parent_session,
                            },
                        )
                    )
        if session_defect is not None:
            findings.append(
                _finding(
                    "root_session_null" if session_defect == "missing" else "malformed_session_uuid",
                    severity=severity,
                    locator={"table": "datasets", "dataset_id": dataset_id},
                    detail={"value": row.get("session_uuid"), "defect": session_defect},
                )
            )
        if session_defect is None and dataset_id == session_uuid:
            findings.append(
                _finding(
                    "legacy_dataset_eq_session",
                    severity="expected",
                    locator={"table": "datasets", "dataset_id": dataset_id},
                    detail={
                        "session_uuid": session_uuid,
                        "note": "Equality is a legacy compatibility observation, not proof that the semantic facts are interchangeable.",
                    },
                )
            )

    for recording_id, row in sorted(recording_by_id.items()):
        _session, defect = _identity_text(row.get("session_uuid"))
        if defect is not None:
            findings.append(
                _finding(
                    "root_session_null" if defect == "missing" else "malformed_session_uuid",
                    severity="action_required",
                    locator={"table": "recordings", "recording_id": recording_id},
                    detail={"value": row.get("session_uuid"), "defect": defect},
                )
            )
        if recording_id not in datasets_by_recording:
            findings.append(
                _finding(
                    "orphan_recording",
                    severity="unresolved",
                    locator={"table": "recordings", "recording_id": recording_id},
                    detail={"recording_path": row.get("recording_path")},
                )
            )

    session_to_recordings: dict[str, set[str]] = defaultdict(set)
    recording_to_sessions: dict[str, set[str]] = defaultdict(set)
    for row in recordings:
        recording_id, recording_defect = _identity_text(row.get("recording_id"))
        session_uuid, session_defect = _identity_text(row.get("session_uuid"))
        if recording_defect is None and session_defect is None:
            assert recording_id is not None and session_uuid is not None
            session_to_recordings[session_uuid].add(recording_id)
            recording_to_sessions[recording_id].add(session_uuid)
    for row in datasets:
        recording_id, recording_defect = _identity_text(row.get("recording_id"))
        session_uuid, session_defect = _identity_text(row.get("session_uuid"))
        if recording_defect is None and session_defect is None:
            assert recording_id is not None and session_uuid is not None
            session_to_recordings[session_uuid].add(recording_id)
            recording_to_sessions[recording_id].add(session_uuid)

    roots = {
        "datasets_total": len(datasets),
        "recordings_total": len(recordings),
        "datasets_with_recording_id": sum(_identity_text(row.get("recording_id"))[1] is None for row in datasets),
        "datasets_with_session_uuid": sum(_identity_text(row.get("session_uuid"))[1] is None for row in datasets),
        "recordings_with_session_uuid": sum(_identity_text(row.get("session_uuid"))[1] is None for row in recordings),
        "orphan_recordings": sum(1 for finding in findings if finding["code"] == "orphan_recording"),
        "session_uuid_to_recording_ids": [
            {"session_uuid": key, "recording_ids": sorted(values), "cardinality": len(values)}
            for key, values in sorted(session_to_recordings.items())
        ],
        "recording_id_to_session_uuids": [
            {"recording_id": key, "session_uuids": sorted(values), "cardinality": len(values)}
            for key, values in sorted(recording_to_sessions.items())
        ],
    }
    return roots, datasets, recordings, findings


def _stable_locator(row: Mapping[str, Any], primary_keys: Sequence[str]) -> dict[str, Any]:
    if primary_keys:
        return {key: _json_safe(row.get(key)) for key in primary_keys}
    return {
        key: _json_safe(row.get(key))
        for key in sorted(row)
        if key in {"dataset_id", "recording_id", "session_uuid"}
    }


def _projection_inventory(
    conn: sqlite3.Connection,
    objects: Sequence[Mapping[str, Any]],
    datasets: Sequence[Mapping[str, Any]],
    recordings: Sequence[Mapping[str, Any]],
    *,
    finding_sample_limit: int,
    scope_dataset_ids: set[str] | None = None,
    scope_recording_ids: set[str] | None = None,
    scope_session_uuids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_by_id = {
        value: row
        for row in datasets
        if (value := _identity_text(row.get("dataset_id"))[0]) is not None
    }
    recording_by_id = {
        value: row
        for row in recordings
        if (value := _identity_text(row.get("recording_id"))[0]) is not None
    }
    summaries: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []

    for obj in objects:
        if obj["object_type"] != "table" or obj["object_name"] in {"datasets", "recordings"}:
            continue
        name = str(obj["object_name"])
        identity_columns = list(obj["identity_columns"])
        primary_keys = list(obj["primary_key_columns"])
        selected = list(dict.fromkeys([*primary_keys, *identity_columns]))
        select_sql = ", ".join(_quote_identifier(column) for column in selected)
        try:
            rows = conn.execute(f"SELECT {select_sql} FROM {_quote_identifier(name)};")
        except sqlite3.Error as exc:
            summaries.append(
                {
                    "table": name,
                    "row_count": obj["row_count"],
                    "scan_status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            findings.append(
                _finding(
                    "projection_scan_error",
                    severity="action_required",
                    locator={"table": name},
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
            continue

        counts: Counter[str] = Counter()
        samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        scanned_rows = 0
        scoped_rows = 0
        for sqlite_row in rows:
            scanned_rows += 1
            row = _row_mapping(sqlite_row)
            if scope_dataset_ids is not None:
                row_dataset_values = {
                    value
                    for column, raw in row.items()
                    if "dataset_id" in column
                    and (value := _identity_text(raw)[0]) is not None
                }
                row_recording_values = {
                    value
                    for column, raw in row.items()
                    if "recording_id" in column
                    and (value := _identity_text(raw)[0]) is not None
                }
                row_session_values = {
                    value
                    for column, raw in row.items()
                    if "session_uuid" in column
                    and (value := _identity_text(raw)[0]) is not None
                }
                if not (
                    row_dataset_values.intersection(scope_dataset_ids)
                    or row_recording_values.intersection(scope_recording_ids or set())
                    or row_session_values.intersection(scope_session_uuids or set())
                ):
                    continue
            scoped_rows += 1
            dataset_id, dataset_defect = _identity_text(row.get("dataset_id")) if "dataset_id" in row else (None, None)
            recording_id, recording_defect = _identity_text(row.get("recording_id")) if "recording_id" in row else (None, None)
            session_uuid, session_defect = _identity_text(row.get("session_uuid")) if "session_uuid" in row else (None, None)
            dataset = dataset_by_id.get(dataset_id) if dataset_id is not None else None
            severity = _dataset_policy_severity(dataset) if dataset is not None else "action_required"
            row_issues: list[tuple[str, str, dict[str, Any]]] = []
            if "dataset_id" in row and dataset_defect is None and dataset is None:
                row_issues.append(("dataset_ref_missing", "action_required", {"dataset_id": dataset_id}))
            if "recording_id" in row and recording_defect is None and recording_id not in recording_by_id:
                synthetic = (
                    dataset is not None
                    and dataset.get("artifact_kind") == "derived_training_merge"
                    and recording_id == dataset_id
                )
                row_issues.append(
                    (
                        "synthetic_recording_id" if synthetic else "recording_ref_missing",
                        "expected" if synthetic else severity,
                        {"recording_id": recording_id, "dataset_id": dataset_id},
                    )
                )
            if "recording_id" in row and dataset is not None and recording_defect is None:
                parent_recording_id, parent_defect = _identity_text(dataset.get("recording_id"))
                if parent_defect is None and recording_id != parent_recording_id:
                    row_issues.append(
                        (
                            "dataset_recording_conflict",
                            "action_required",
                            {
                                "dataset_id": dataset_id,
                                "row_recording_id": recording_id,
                                "dataset_recording_id": parent_recording_id,
                            },
                        )
                    )
            if "session_uuid" in row and dataset is not None and session_defect is None:
                parent_session_uuid, parent_session_defect = _identity_text(dataset.get("session_uuid"))
                if parent_session_defect is None and session_uuid != parent_session_uuid:
                    row_issues.append(
                        (
                            "dataset_session_conflict",
                            "action_required",
                            {
                                "dataset_id": dataset_id,
                                "row_session_uuid": session_uuid,
                                "dataset_session_uuid": parent_session_uuid,
                            },
                        )
                    )
            if (
                "session_uuid" in row
                and "dataset_id" not in row
                and recording_defect is None
                and session_defect is None
                and recording_id in recording_by_id
            ):
                parent_session_uuid, parent_session_defect = _identity_text(
                    recording_by_id[recording_id].get("session_uuid")
                )
                if parent_session_defect is None and session_uuid != parent_session_uuid:
                    row_issues.append(
                        (
                            "recording_session_conflict",
                            "action_required",
                            {
                                "recording_id": recording_id,
                                "row_session_uuid": session_uuid,
                                "recording_session_uuid": parent_session_uuid,
                            },
                        )
                    )
            for code, issue_severity, detail in row_issues:
                counts[code] += 1
                if len(samples[code]) < finding_sample_limit:
                    sample = {
                        "locator": _stable_locator(row, primary_keys),
                        "detail": _json_safe(detail),
                    }
                    samples[code].append(sample)
                    findings.append(
                        _finding(
                            code,
                            severity=issue_severity,
                            locator={"table": name, **sample["locator"]},
                            detail=detail,
                        )
                    )
        summaries.append(
            {
                "table": name,
                "row_count": scanned_rows,
                "rows_in_selected_scope": scoped_rows,
                "identity_columns": identity_columns,
                "scan_status": "complete",
                "issue_counts": dict(sorted(counts.items())),
                "issue_samples": {key: value for key, value in sorted(samples.items())},
                "sample_limit_per_issue": finding_sample_limit,
            }
        )
    return summaries, findings


def _registry_uuid(conn: sqlite3.Connection) -> str | None:
    candidates = (
        "SELECT registry_uuid FROM registry_identity WHERE singleton_id = 1 LIMIT 1;",
        "SELECT registry_uuid FROM registry_metadata LIMIT 1;",
        "SELECT value FROM registry_metadata WHERE key = 'registry_uuid' LIMIT 1;",
    )
    for statement in candidates:
        try:
            row = conn.execute(statement).fetchone()
        except sqlite3.Error:
            continue
        if row is not None and row[0] is not None:
            return str(row[0])
    return None


def _scan_registry(
    snapshot_path: Path,
    *,
    dataset_ids: Sequence[str] | None,
    artifact_scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    conn = connect_read_only(snapshot_path)
    try:
        conn.execute("PRAGMA read_uncommitted = OFF;")
        query_only = int(conn.execute("PRAGMA query_only;").fetchone()[0])
        integrity = [str(row[0]) for row in conn.execute("PRAGMA integrity_check;").fetchall()]
        foreign_key_rows = [list(map(_json_safe, row)) for row in conn.execute("PRAGMA foreign_key_check;").fetchall()]
        objects, introspection_findings = _schema_inventory(conn)
        if integrity != ["ok"]:
            introspection_findings.append(
                _finding(
                    "registry_integrity_failure",
                    severity="action_required",
                    detail={"integrity_check": integrity},
                )
            )
        if foreign_key_rows:
            introspection_findings.append(
                _finding(
                    "registry_foreign_key_failure",
                    severity="action_required",
                    detail={"row_count": len(foreign_key_rows), "rows": foreign_key_rows[:50]},
                )
            )
        roots, datasets, recordings, root_findings = _registry_roots(conn)
        scoped_datasets = _selected_artifact_datasets(
            datasets,
            dataset_ids=dataset_ids,
            artifact_scope=artifact_scope,
        )
        scope_dataset_ids = {
            value
            for row in scoped_datasets
            if (value := _identity_text(row.get("dataset_id"))[0]) is not None
        }
        scope_recording_ids = {
            value
            for row in scoped_datasets
            if (value := _identity_text(row.get("recording_id"))[0]) is not None
        }
        scope_session_uuids = {
            value
            for row in scoped_datasets
            if (value := _identity_text(row.get("session_uuid"))[0]) is not None
        }

        def root_finding_is_in_scope(finding: Mapping[str, Any]) -> bool:
            locator = finding.get("locator")
            if not isinstance(locator, Mapping):
                return True
            if "dataset_id" in locator:
                return str(locator.get("dataset_id")) in scope_dataset_ids
            if "recording_id" in locator:
                return str(locator.get("recording_id")) in scope_recording_ids
            return True

        scoped_root_findings = [
            finding for finding in root_findings if root_finding_is_in_scope(finding)
        ]
        projections, projection_findings = _projection_inventory(
            conn,
            objects,
            datasets,
            recordings,
            finding_sample_limit=DEFAULT_FINDING_SAMPLE_LIMIT,
            scope_dataset_ids=scope_dataset_ids,
            scope_recording_ids=scope_recording_ids,
            scope_session_uuids=scope_session_uuids,
        )
        registry = {
            "access_mode": "sqlite_uri_mode_ro_query_only",
            "query_only": bool(query_only),
            "sqlite_runtime": sqlite3.sqlite_version,
            "user_version": int(conn.execute("PRAGMA user_version;").fetchone()[0]),
            "registry_uuid": _registry_uuid(conn),
            "integrity_check": integrity,
            "foreign_key_check": foreign_key_rows,
            "view_scan_level": "schema_row_count_column_counts_and_bounded_distinct_values",
            "views_are_independent_projections": False,
            "roots": roots,
            "finding_scope": {
                "artifact_scope": artifact_scope,
                "dataset_count": len(scope_dataset_ids),
                "recording_count": len(scope_recording_ids),
                "session_uuid_count": len(scope_session_uuids),
                "unmarked_outside_scope_deferred": artifact_scope
                == "explicit_source_layout",
            },
            "identity_bearing_objects": objects,
            "stored_projections": projections,
        }
        findings = [*introspection_findings, *scoped_root_findings, *projection_findings]
        return registry, datasets, recordings, findings
    finally:
        conn.close()


def _observation(
    *,
    semantic_fact: str,
    source_field: str,
    raw_value: Any,
    source_kind: str,
    source_role: str,
    source_locator: str,
    comparison_domain: str,
    source_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    value, defect = _identity_text(raw_value)
    observation = {
        "semantic_fact": semantic_fact,
        "source_field": source_field,
        "value": value,
        "raw_value": _json_safe(raw_value),
        "valid": defect is None,
        "defect": defect,
        "source_kind": source_kind,
        "source_role": source_role,
        "source_locator": source_locator,
        "comparison_domain": comparison_domain,
        "source_sha256": source_sha256,
    }
    if defect is None or defect == "missing":
        return observation, None
    issue = _finding(
        "malformed_artifact_identity",
        severity="action_required",
        locator={"source": source_locator, "field": source_field},
        detail={"value": raw_value, "defect": defect},
    )
    return observation, issue


def _append_json_identity_observations(
    value: Any,
    *,
    source_kind: str,
    source_role: str,
    source_locator: str,
    source_sha256: str,
    observations: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    max_observations: int,
    path: tuple[str, ...] = (),
    comparison_domain: str = "artifact",
    depth: int = 0,
) -> bool:
    """Extract identity fields while retaining nested child comparison domains.

    Returns ``True`` when the observation cap truncated the source.
    """

    if len(observations) >= max_observations:
        return True
    if depth > 12:
        return True
    if isinstance(value, Mapping):
        for raw_key in sorted(value, key=str):
            if len(observations) >= max_observations:
                return True
            key = str(raw_key)
            child = value[raw_key]
            child_path = (*path, key)
            locator = f"{source_locator}#{'.'.join(child_path)}"
            camera_field_allowed = key not in {"camera_id", "camera_serial"} or (
                not path or comparison_domain.startswith("child:") or comparison_domain.startswith("clip:")
            )
            if key in DIRECT_IDENTITY_FIELDS and camera_field_allowed:
                semantic_fact = DIRECT_IDENTITY_FIELDS[key]
                if path and comparison_domain == "artifact" and key in {
                    "dataset_id",
                    "recording_id",
                    "session_uuid",
                    "session_id",
                    "orange_session_id",
                }:
                    semantic_fact = f"nested_{semantic_fact}"
                item, issue = _observation(
                    semantic_fact=semantic_fact,
                    source_field=key,
                    raw_value=child,
                    source_kind=source_kind,
                    source_role=source_role,
                    source_locator=locator,
                    comparison_domain=comparison_domain,
                    source_sha256=source_sha256,
                )
                observations.append(item)
                if issue is not None:
                    findings.append(issue)
            elif (
                key in LIST_IDENTITY_FIELDS
                and (key not in {"camera_ids", "camera_serials"} or not path or comparison_domain.startswith("child:"))
                and isinstance(child, Sequence)
                and not isinstance(child, (str, bytes))
            ):
                for index, raw_item in enumerate(child):
                    if len(observations) >= max_observations:
                        return True
                    item, issue = _observation(
                        semantic_fact=LIST_IDENTITY_FIELDS[key],
                        source_field=key,
                        raw_value=raw_item,
                        source_kind=source_kind,
                        source_role=source_role,
                        source_locator=f"{locator}[{index}]",
                        comparison_domain=f"{comparison_domain}:set:{key}",
                        source_sha256=source_sha256,
                    )
                    observations.append(item)
                    if issue is not None:
                        findings.append(issue)
            if key in DONOR_LOCATOR_FIELDS and isinstance(child, str) and child:
                if len(observations) >= max_observations:
                    return True
                observations.append(
                    {
                        "semantic_fact": "donor_locator",
                        "source_field": key,
                        "value": child,
                        "raw_value": child,
                        "valid": True,
                        "defect": None,
                        "source_kind": source_kind,
                        "source_role": "donor_declaration",
                        "source_locator": locator,
                        "comparison_domain": comparison_domain,
                        "source_sha256": source_sha256,
                    }
                )
            nested_domain = comparison_domain
            if key in {"clips", "rows", "camera_artifacts", "video_streams", "rolling_clip_streams"}:
                nested_domain = f"child:{'.'.join(child_path)}"
            if key in IDENTITY_CONTAINER_FIELDS and isinstance(child, (Mapping, list, tuple)):
                truncated = _append_json_identity_observations(
                    child,
                    source_kind=source_kind,
                    source_role=source_role,
                    source_locator=source_locator,
                    source_sha256=source_sha256,
                    observations=observations,
                    findings=findings,
                    max_observations=max_observations,
                    path=child_path,
                    comparison_domain=nested_domain,
                    depth=depth + 1,
                )
                if truncated:
                    return True
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            truncated = _append_json_identity_observations(
                child,
                source_kind=source_kind,
                source_role=source_role,
                source_locator=source_locator,
                source_sha256=source_sha256,
                observations=observations,
                findings=findings,
                max_observations=max_observations,
                path=(*path, f"[{index}]"),
                comparison_domain=f"{comparison_domain}:row:{index}",
                depth=depth + 1,
            )
            if truncated:
                return True
    return False


def _append_required_identity_presence(
    value: Mapping[str, Any],
    *,
    required_fields: Sequence[str],
    source_kind: str,
    source_role: str,
    source_locator: str,
    source_sha256: str,
    observations: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    severity: str,
) -> None:
    """Record absent or empty exact fields instead of losing source omission."""

    for field in required_fields:
        raw_value = value.get(field)
        _normalized, defect = _identity_text(raw_value)
        if defect != "missing":
            continue
        # The generic extractor already records an observation for a present
        # field, including an explicit null/empty value.  Add an observation
        # here only when the key itself is absent.
        if field not in value:
            item, _issue = _observation(
                semantic_fact=DIRECT_IDENTITY_FIELDS[field],
                source_field=field,
                raw_value=None,
                source_kind=source_kind,
                source_role=source_role,
                source_locator=f"{source_locator}#{field}",
                comparison_domain="artifact",
                source_sha256=source_sha256,
            )
            observations.append(item)
        findings.append(
            _finding(
                "artifact_identity_field_missing",
                severity=severity,
                locator={"source": source_locator, "field": field},
                detail={
                    "presence": "present_but_empty" if field in value else "absent",
                },
            )
        )


def _metadata_file_for_node(path: Path) -> tuple[str, Path] | None:
    v3 = path / "zarr.json"
    if v3.is_file():
        return "zarr_v3", v3
    v2_group = path / ".zgroup"
    if v2_group.is_file():
        return "zarr_v2", v2_group
    v2_array = path / ".zarray"
    if v2_array.is_file():
        return "zarr_v2_array", v2_array
    return None


def _group_child_names(path: Path) -> tuple[str, ...]:
    names: list[str] = []
    try:
        children = sorted(path.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise CensusError(f"Unable to list Zarr group {path}: {type(exc).__name__}: {exc}") from exc
    for child in children:
        if child.is_symlink():
            continue
        if child.is_dir() and _metadata_file_for_node(child) is not None:
            names.append(child.name)
    return tuple(names)


def _identity_metadata_children(relative_group: str, child_names: Sequence[str]) -> tuple[str, ...]:
    """Bound traversal to identity-bearing group surfaces, never array payload trees."""

    parts = () if relative_group == "." else tuple(Path(relative_group).parts)
    if not parts:
        allowed = {
            name
            for name in child_names
            if name in {"analysis", "analysis_metadata", "raw_video", "clips"}
            or name.endswith("_runs")
            or "profile_runs" in name
        }
    elif parts == ("analysis",):
        allowed = {
            name
            for name in child_names
            if name.endswith("_runs")
            or "profile_runs" in name
            or name
            in {
                "acquisition_camera_frames",
                "acquisition_video_streams",
                "arena_geometry_selection",
                "calibration",
                "coordinate_frames",
            }
        }
    elif (len(parts) == 1 and parts[0].endswith("_runs")) or (
        len(parts) == 2 and parts[0] == "analysis" and parts[1].endswith("_runs")
    ):
        allowed = set(child_names)
    elif parts == ("clips",):
        allowed = set(child_names)
    elif len(parts) == 2 and parts[0] == "clips":
        allowed = {name for name in child_names if name == "cameras"}
    elif len(parts) == 3 and parts[0] == "clips" and parts[2] == "cameras":
        allowed = set(child_names)
    elif len(parts) == 4 and parts[0] == "clips" and parts[2] == "cameras":
        allowed = {name for name in child_names if name == "source"}
    elif len(parts) == 5 and parts[0] == "clips" and parts[4] == "source":
        allowed = {name for name in child_names if name == "frame_map"}
    else:
        allowed = set()
    return tuple(name for name in child_names if name in allowed)


def _identity_group_may_descend(relative_group: str) -> bool:
    parts = () if relative_group == "." else tuple(Path(relative_group).parts)
    if not parts or parts == ("analysis",) or parts == ("clips",):
        return True
    if len(parts) == 1 and parts[0].endswith("_runs"):
        return True
    if len(parts) == 2 and parts[0] == "analysis" and parts[1].endswith("_runs"):
        return True
    if len(parts) >= 2 and parts[0] == "clips":
        return len(parts) <= 5
    return False


def _zarr_source_role(relative_group: str) -> str:
    if relative_group == ".":
        return "artifact_evidence"
    if relative_group in {"raw_video", "analysis", "analysis_metadata"}:
        return "compatibility_mirror"
    if "profile_runs" in relative_group or relative_group.startswith("analysis/"):
        return "derived_projection"
    return "nested_projection"


def _zarr_comparison_domain(relative_group: str) -> str:
    """Keep artifact mirrors, run projections, and clip identities separate."""

    if relative_group in {".", "raw_video", "analysis_metadata"}:
        return "artifact"
    if relative_group.startswith("clips/"):
        parts = relative_group.split("/")
        return "clip:" + "/".join(parts[:4])
    if "_profile_runs/" in relative_group:
        return f"profile:{relative_group}"
    if "_runs/" in relative_group:
        return f"run:{relative_group}"
    return f"nested:{relative_group}"


def _scan_zarr_metadata(
    zarr_path: Path,
    *,
    max_json_bytes: int,
    max_zarr_nodes: int,
    max_observations: int,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    fences: list[FileFence] = []
    group_listings: dict[Path, tuple[str, ...]] = {}
    node_count = 0
    capped = False
    unreadable = False
    root_prefix_only = False
    contract_markers: dict[str, Any] = {}
    root = zarr_path.expanduser().resolve(strict=False)
    stack: list[Path] = [root]

    if not root.is_dir() or _metadata_file_for_node(root) is None:
        return {
            "status": "missing" if not root.exists() else "not_zarr",
            "path": str(root),
            "metadata_view": "direct_unconsolidated",
            "node_count": 0,
            "coverage_complete": False,
            "observations": [],
            "contract_markers": {},
            "findings": [
                _finding(
                    "zarr_metadata_missing",
                    severity="action_required",
                    locator={"zarr_path": str(root)},
                )
            ],
            "source_fences": _fence_summary([]),
        }

    try:
        while stack:
            node_path = stack.pop()
            if node_count >= max_zarr_nodes:
                capped = True
                break
            descriptor = _metadata_file_for_node(node_path)
            if descriptor is None:
                continue
            metadata_kind, metadata_path = descriptor
            node_count += 1
            relative = str(node_path.relative_to(root)) if node_path != root else "."
            try:
                if (
                    node_path == root
                    and metadata_kind == "zarr_v3"
                    and metadata_path.stat().st_size > ROOT_PREFIX_READ_THRESHOLD_BYTES
                ):
                    metadata, fence = _zarr_root_prefix_metadata(
                        metadata_path,
                        max_prefix_bytes=min(max_json_bytes, ROOT_PREFIX_READ_THRESHOLD_BYTES),
                    )
                    root_prefix_only = True
                else:
                    metadata, fence = _stable_read_json(metadata_path, max_bytes=max_json_bytes)
            except UnstableSnapshotError:
                raise
            except (CensusError, OSError) as exc:
                unreadable = True
                findings.append(
                    _finding(
                        "zarr_metadata_unreadable",
                        severity="action_required",
                        locator={"path": str(metadata_path)},
                        detail={"error": f"{type(exc).__name__}: {exc}"},
                    )
                )
                continue
            fences.append(fence)
            if node_path == root and metadata_path.stat().st_size > OVERSIZED_ROOT_METADATA_BYTES:
                findings.append(
                    _finding(
                        "oversized_inline_root_metadata",
                        severity="action_required",
                        locator={"path": str(metadata_path)},
                        detail={
                            "size_bytes": metadata_path.stat().st_size,
                            "threshold_bytes": OVERSIZED_ROOT_METADATA_BYTES,
                            "note": "Root identity attrs were read from a stable bounded prefix; inline consolidated metadata was not expanded.",
                        },
                    )
                )
            attrs_fence = fence
            attrs_locator = metadata_path
            if metadata_kind == "zarr_v3":
                node_type = metadata.get("node_type")
                attrs = metadata.get("attributes", {})
                is_group = node_type == "group" or (node_path == root and metadata.get("_prefix_only"))
                if node_type not in {"group", "array"} and not (
                    node_path == root and metadata.get("_prefix_only")
                ):
                    findings.append(
                        _finding(
                            "unsupported_zarr_node",
                            severity="unresolved",
                            locator={"path": str(metadata_path)},
                            detail={"node_type": node_type},
                        )
                    )
            elif metadata_kind == "zarr_v2":
                is_group = True
                attrs_path = node_path / ".zattrs"
                attrs = {}
                if attrs_path.is_file():
                    try:
                        attrs, attrs_fence = _stable_read_json(attrs_path, max_bytes=max_json_bytes)
                    except UnstableSnapshotError:
                        raise
                    except (CensusError, OSError) as exc:
                        unreadable = True
                        findings.append(
                            _finding(
                                "zarr_metadata_unreadable",
                                severity="action_required",
                                locator={"path": str(attrs_path)},
                                detail={"error": f"{type(exc).__name__}: {exc}"},
                            )
                        )
                        attrs = {}
                    else:
                        fences.append(attrs_fence)
                        attrs_locator = attrs_path
            else:
                is_group = False
                attrs = {}
            if not isinstance(attrs, Mapping):
                findings.append(
                    _finding(
                        "zarr_attributes_malformed",
                        severity="action_required",
                        locator={"path": str(metadata_path)},
                    )
                )
                attrs = {}
            if relative == ".":
                contract_markers = {
                    field: _json_safe(attrs.get(field))
                    for field in CONTRACT_MARKER_FIELDS
                    if attrs.get(field) is not None
                }
            if attrs:
                source_kind = f"{metadata_kind}_group_attrs"
                source_role = _zarr_source_role(relative)
                truncated = _append_json_identity_observations(
                    attrs,
                    source_kind=source_kind,
                    source_role=source_role,
                    source_locator=f"{attrs_locator}#attributes",
                    source_sha256=attrs_fence.sha256,
                    observations=observations,
                    findings=findings,
                    max_observations=max_observations,
                    comparison_domain=_zarr_comparison_domain(relative),
                )
                capped = capped or truncated
            else:
                source_kind = f"{metadata_kind}_group_attrs"
                source_role = _zarr_source_role(relative)
            if relative == ".":
                _append_required_identity_presence(
                    attrs,
                    required_fields=("recording_id", "session_uuid"),
                    source_kind=source_kind,
                    source_role=source_role,
                    source_locator=f"{attrs_locator}#attributes",
                    source_sha256=attrs_fence.sha256,
                    observations=observations,
                    findings=findings,
                    severity="action_required",
                )
            if is_group and _identity_group_may_descend(relative):
                names = _group_child_names(node_path)
                group_listings[node_path] = names
                selected_names = _identity_metadata_children(relative, names)
                for name in reversed(selected_names):
                    stack.append(node_path / name)

        for group_path, before_names in group_listings.items():
            if _group_child_names(group_path) != before_names:
                raise UnstableSnapshotError(f"Zarr group inventory changed during census: {group_path}")
        for fence in fences:
            _verify_file_fence(fence, max_bytes=max_json_bytes)
    except UnstableSnapshotError as exc:
        findings.append(
            _finding(
                "unstable_filesystem_snapshot",
                severity="action_required",
                locator={"zarr_path": str(root)},
                detail={"error": str(exc)},
            )
        )
        return {
            "status": "unstable",
            "path": str(root),
            "metadata_view": "direct_unconsolidated",
            "node_count": node_count,
            "coverage_complete": False,
            "observations": observations,
            "contract_markers": contract_markers,
            "findings": findings,
            "source_fences": _fence_summary(fences),
            "root_prefix_only": root_prefix_only,
        }
    except (CensusError, OSError) as exc:
        findings.append(
            _finding(
                "zarr_scan_incomplete",
                severity="action_required",
                locator={"zarr_path": str(root)},
                detail={"error": f"{type(exc).__name__}: {exc}"},
            )
        )
        return {
            "status": "incomplete",
            "path": str(root),
            "metadata_view": "direct_unconsolidated",
            "node_count": node_count,
            "coverage_complete": False,
            "observations": observations,
            "contract_markers": contract_markers,
            "findings": findings,
            "source_fences": _fence_summary(fences),
            "root_prefix_only": root_prefix_only,
        }

    if capped:
        findings.append(
            _finding(
                "artifact_coverage_capped",
                severity="unresolved",
                locator={"zarr_path": str(root)},
                detail={
                    "max_zarr_nodes": max_zarr_nodes,
                    "max_observations": max_observations,
                },
            )
        )
    return {
        "status": "incomplete" if unreadable else ("capped" if capped else "complete"),
        "path": str(root),
        "metadata_view": "direct_unconsolidated",
        "node_count": node_count,
        "coverage_complete": not capped and not unreadable,
        "observations": observations,
        "contract_markers": contract_markers,
        "findings": findings,
        "source_fences": _fence_summary(fences),
        "root_prefix_only": root_prefix_only,
        "traversal_policy": "bounded_identity_groups_v1",
    }


def _safe_child_path(recording_dir: Path, raw_path: str) -> Path | None:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = recording_dir / candidate
    resolved_root = recording_dir.resolve(strict=False)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def _declared_clip_manifest_paths(value: Any, recording_dir: Path) -> list[Path]:
    candidates: set[Path] = set()

    def visit(item: Any, depth: int = 0) -> None:
        if depth > 10:
            return
        if isinstance(item, Mapping):
            clip_id = item.get("clip_id")
            if isinstance(clip_id, str) and clip_id and "/" not in clip_id and ".." not in clip_id:
                candidates.add(recording_dir / "clips" / clip_id / "clip_manifest.json")
            for key, child in item.items():
                if isinstance(child, str) and (
                    str(key) in {"clip_manifest", "clip_manifest_path", "manifest_path"}
                    or child.endswith("clip_manifest.json")
                ):
                    safe = _safe_child_path(recording_dir, child)
                    if safe is not None and safe.name == "clip_manifest.json":
                        candidates.add(safe)
                elif isinstance(child, (Mapping, list, tuple)):
                    visit(child, depth + 1)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)

    visit(value)
    return sorted(candidates, key=str)


def _scan_parquet_identity(
    path: Path,
    *,
    max_distinct_values: int = DEFAULT_MAX_PARQUET_DISTINCT_VALUES,
    max_malformed_samples: int = DEFAULT_MAX_PARQUET_MALFORMED_SAMPLES,
) -> dict[str, Any]:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - required project dependency
        return {
            "path": str(path),
            "status": "unsupported",
            "findings": [
                _finding(
                    "parquet_runtime_unavailable",
                    severity="unresolved",
                    locator={"path": str(path)},
                    detail={"error": str(exc)},
                )
            ],
            "observations": [],
        }
    def collect_projection() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        parquet = pq.ParquetFile(path)
        target_columns = [
            name
            for name in ("recording_id", "session_uuid", "session_id", "camera_serial")
            if name in parquet.schema_arrow.names
        ]
        distinct_values: dict[str, set[str]] = {name: set() for name in target_columns}
        distinct_overflow_counts: Counter[str] = Counter()
        null_counts: Counter[str] = Counter()
        malformed: list[dict[str, Any]] = []
        malformed_count = 0

        def record_raw(name: str, raw: Any, row_group_index: int) -> None:
            nonlocal malformed_count
            if raw is None:
                return
            value, defect = _identity_text(raw)
            if defect is None:
                assert value is not None
                values = distinct_values[name]
                if value in values:
                    return
                if len(values) < max_distinct_values:
                    values.add(value)
                else:
                    distinct_overflow_counts[name] += 1
                return
            malformed_count += 1
            if len(malformed) < max_malformed_samples:
                malformed.append(
                    {
                        "column": name,
                        "row_group": row_group_index,
                        "value": _json_safe(raw),
                        "defect": defect,
                    }
                )

        for row_group_index in range(parquet.num_row_groups):
            for name in target_columns:
                schema_index = parquet.schema_arrow.get_field_index(name)
                column_metadata = parquet.metadata.row_group(row_group_index).column(schema_index)
                statistics = column_metadata.statistics
                if statistics is not None:
                    null_counts[name] += int(statistics.null_count or 0)
                if statistics is not None and statistics.has_min_max and statistics.min == statistics.max:
                    record_raw(name, statistics.min, row_group_index)
                else:
                    for batch in parquet.iter_batches(
                        batch_size=65_536,
                        row_groups=[row_group_index],
                        columns=[name],
                    ):
                        column = batch.column(0)
                        if statistics is None:
                            null_counts[name] += int(column.null_count)
                        for raw in pc.unique(column).to_pylist():
                            record_raw(name, raw, row_group_index)
        projection = {
            "row_count": parquet.metadata.num_rows,
            "row_group_count": parquet.num_row_groups,
            "schema": str(parquet.schema_arrow),
            "identity_columns": target_columns,
            "null_counts": dict(sorted(null_counts.items())),
            "distinct_values": {
                key: sorted(values) for key, values in sorted(distinct_values.items())
            },
            "distinct_value_cap_per_column": max_distinct_values,
            "distinct_overflow_counts": dict(sorted(distinct_overflow_counts.items())),
            "malformed_count": malformed_count,
            "malformed_sample_limit": max_malformed_samples,
        }
        return projection, malformed

    before = path.stat()
    projection, malformed = collect_projection()
    middle = path.stat()
    verification_projection, verification_malformed = collect_projection()
    after = path.stat()
    signatures = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, middle, after)
    }
    projection_digest = _sha256_bytes(_canonical_json_bytes(projection))
    verification_digest = _sha256_bytes(_canonical_json_bytes(verification_projection))
    if len(signatures) != 1 or projection_digest != verification_digest or malformed != verification_malformed:
        raise UnstableSnapshotError(f"Parquet identity projection changed during census: {path}")
    observations: list[dict[str, Any]] = []
    findings = [
        _finding(
            "malformed_artifact_identity",
            severity="action_required",
            locator={"path": str(path), "column": item["column"], "row_group": item["row_group"]},
            detail={"value": item["value"], "defect": item["defect"]},
        )
        for item in malformed
    ]
    overflow_counts = projection["distinct_overflow_counts"]
    if overflow_counts:
        findings.append(
            _finding(
                "parquet_identity_cardinality_capped",
                severity="action_required",
                locator={"path": str(path)},
                detail={
                    "distinct_value_cap_per_column": max_distinct_values,
                    "overflow_counts": overflow_counts,
                    "note": "Row count is not a defect; only identity-column cardinality exceeded the bounded evidence contract.",
                },
            )
        )
    distinct_values = projection["distinct_values"]
    for field, values in sorted(distinct_values.items()):
        semantic_fact = DIRECT_IDENTITY_FIELDS.get(field, field)
        for value in sorted(values):
            observations.append(
                {
                    "semantic_fact": semantic_fact,
                    "source_field": field,
                    "value": value,
                    "raw_value": value,
                    "valid": True,
                    "defect": None,
                    "source_kind": "recording_frame_index_parquet",
                    "source_role": "artifact_evidence",
                    "source_locator": f"{path}#column={field}",
                    "comparison_domain": "artifact",
                    "source_sha256": projection_digest,
                }
            )
    return {
        "path": str(path),
        "status": "capped" if overflow_counts else "complete",
        "binding_kind": "identity_projection_digest_with_pre_post_stat_fence",
        "identity_projection_digest": projection_digest,
        "row_count": projection["row_count"],
        "row_group_count": projection["row_group_count"],
        "identity_columns": projection["identity_columns"],
        "null_counts": projection["null_counts"],
        "distinct_counts": {key: len(value) for key, value in sorted(distinct_values.items())},
        "distinct_overflow_counts": overflow_counts,
        "malformed_count": projection["malformed_count"],
        "observations": observations,
        "findings": findings,
    }


def _scan_recording_directory(
    recording_dir: Path,
    *,
    max_json_bytes: int,
    max_observations: int,
    scan_parquet: bool,
) -> dict[str, Any]:
    root = recording_dir.expanduser().resolve(strict=False)
    observations: list[dict[str, Any]] = [
        {
            "semantic_fact": "recording_dir_name",
            "source_field": "directory_basename",
            "value": root.name,
            "raw_value": root.name,
            "valid": bool(root.name),
            "defect": None if root.name else "empty",
            "source_kind": "filesystem_locator",
            "source_role": "weak_context_hint",
            "source_locator": str(root),
            "comparison_domain": "artifact",
            "source_sha256": None,
        }
    ]
    findings: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    fences: list[FileFence] = []
    clip_manifests: set[Path] = set()

    if not root.is_dir():
        return {
            "path": str(root),
            "status": "missing",
            "coverage_complete": False,
            "sources": [],
            "observations": observations,
            "findings": [
                _finding(
                    "recording_directory_missing",
                    severity="action_required",
                    locator={"recording_path": str(root)},
                )
            ],
        }

    standard_sidecar_presence = {
        root / relative: (root / relative).is_file()
        for relative in RECORDING_SIDECAR_RELATIVE_PATHS
    }

    for relative in RECORDING_SIDECAR_RELATIVE_PATHS:
        path = root / relative
        if not path.exists():
            if relative == Path("recording_manifest.json"):
                sources.append({"path": str(path), "status": "required_missing"})
                findings.append(
                    _finding(
                        "recording_manifest_missing",
                        severity="action_required",
                        locator={"path": str(path)},
                    )
                )
            else:
                sources.append({"path": str(path), "status": "optional_missing"})
            continue
        try:
            payload, fence = _stable_read_json(path, max_bytes=max_json_bytes)
        except UnstableSnapshotError as exc:
            sources.append({"path": str(path), "status": "unstable", "error": str(exc)})
            findings.append(
                _finding(
                    "unstable_filesystem_snapshot",
                    severity="action_required",
                    locator={"path": str(path)},
                    detail={"error": str(exc)},
                )
            )
            continue
        except (CensusError, OSError) as exc:
            sources.append({"path": str(path), "status": "unreadable", "error": f"{type(exc).__name__}: {exc}"})
            findings.append(
                _finding(
                    "sidecar_unreadable",
                    severity="action_required",
                    locator={"path": str(path)},
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
            continue
        fences.append(fence)
        source_kind = path.name.removesuffix(".json")
        truncated = _append_json_identity_observations(
            payload,
            source_kind=source_kind,
            source_role="artifact_evidence" if path.name == "recording_manifest.json" else "compatibility_evidence",
            source_locator=str(path),
            source_sha256=fence.sha256,
            observations=observations,
            findings=findings,
            max_observations=max_observations,
        )
        if path.name == "recording_manifest.json":
            _append_required_identity_presence(
                payload,
                required_fields=("recording_id", "session_uuid"),
                source_kind=source_kind,
                source_role="artifact_evidence",
                source_locator=str(path),
                source_sha256=fence.sha256,
                observations=observations,
                findings=findings,
                severity="unresolved",
            )
        if truncated:
            findings.append(
                _finding(
                    "artifact_coverage_capped",
                    severity="unresolved",
                    locator={"path": str(path)},
                    detail={"max_observations": max_observations},
                )
            )
        sources.append({"path": str(path), "status": "complete", "sha256": fence.sha256, "truncated": truncated})
        if path.name == "recording_clip_index.json":
            clip_manifests.update(_declared_clip_manifest_paths(payload, root))

    for path in sorted(clip_manifests, key=str):
        if not path.is_file():
            sources.append({"path": str(path), "status": "declared_missing"})
            findings.append(
                _finding(
                    "declared_clip_manifest_missing",
                    severity="unresolved",
                    locator={"path": str(path)},
                )
            )
            continue
        try:
            payload, fence = _stable_read_json(path, max_bytes=max_json_bytes)
        except UnstableSnapshotError as exc:
            sources.append({"path": str(path), "status": "unstable", "error": str(exc)})
            findings.append(
                _finding(
                    "unstable_filesystem_snapshot",
                    severity="action_required",
                    locator={"path": str(path)},
                    detail={"error": str(exc)},
                )
            )
            continue
        except (CensusError, OSError) as exc:
            sources.append({"path": str(path), "status": "unreadable", "error": f"{type(exc).__name__}: {exc}"})
            findings.append(
                _finding(
                    "sidecar_unreadable",
                    severity="action_required",
                    locator={"path": str(path)},
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
            continue
        fences.append(fence)
        truncated = _append_json_identity_observations(
            payload,
            source_kind="clip_manifest",
            source_role="compatibility_evidence",
            source_locator=str(path),
            source_sha256=fence.sha256,
            observations=observations,
            findings=findings,
            max_observations=max_observations,
            comparison_domain=f"clip:{path.parent.name}",
        )
        if truncated:
            findings.append(
                _finding(
                    "artifact_coverage_capped",
                    severity="unresolved",
                    locator={"path": str(path)},
                    detail={"max_observations": max_observations},
                )
            )
        sources.append(
            {
                "path": str(path),
                "status": "complete",
                "sha256": fence.sha256,
                "truncated": truncated,
            }
        )

    parquet_report: dict[str, Any] | None = None
    parquet_path = root / "recording_frame_index.parquet"
    if parquet_path.is_file():
        if scan_parquet:
            try:
                parquet_report = _scan_parquet_identity(parquet_path)
            except UnstableSnapshotError as exc:
                parquet_report = {
                    "path": str(parquet_path),
                    "status": "unstable",
                    "error": str(exc),
                    "observations": [],
                    "findings": [
                        _finding(
                            "unstable_filesystem_snapshot",
                            severity="action_required",
                            locator={"path": str(parquet_path)},
                            detail={"error": str(exc)},
                        )
                    ],
                }
            except Exception as exc:
                parquet_report = {
                    "path": str(parquet_path),
                    "status": "unreadable",
                    "error": f"{type(exc).__name__}: {exc}",
                    "observations": [],
                    "findings": [
                        _finding(
                            "frame_index_unreadable",
                            severity="action_required",
                            locator={"path": str(parquet_path)},
                            detail={"error": f"{type(exc).__name__}: {exc}"},
                        )
                    ],
                }
            observations.extend(parquet_report.get("observations", []))
            findings.extend(parquet_report.get("findings", []))
        else:
            parquet_report = {
                "path": str(parquet_path),
                "status": "deferred",
                "reason": "Parquet identity scan disabled explicitly",
            }

    try:
        for fence in fences:
            _verify_file_fence(fence, max_bytes=max_json_bytes)
        if any(path.is_file() != present for path, present in standard_sidecar_presence.items()):
            raise UnstableSnapshotError(
                f"Recording sidecar inventory changed during census: {root}"
            )
    except UnstableSnapshotError as exc:
        findings.append(
            _finding(
                "unstable_filesystem_snapshot",
                severity="action_required",
                locator={"recording_path": str(root)},
                detail={"error": str(exc)},
            )
        )

    coverage_complete = not any(
        source.get("status")
        in {"unreadable", "unstable", "declared_missing", "required_missing"}
        or source.get("truncated")
        for source in sources
    ) and not any(finding["code"] == "unstable_filesystem_snapshot" for finding in findings)
    if parquet_report is not None and parquet_report.get("status") != "complete":
        coverage_complete = False
    return {
        "path": str(root),
        "status": "complete" if coverage_complete else "incomplete",
        "coverage_complete": coverage_complete,
        "sources": sources,
        "parquet": parquet_report,
        "observations": observations,
        "findings": findings,
    }


def _registry_observations_for_zarr(
    datasets: Sequence[Mapping[str, Any]],
    recordings_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for dataset in datasets:
        row_payload = {
            "dataset_id": dataset.get("dataset_id"),
            "recording_id": dataset.get("recording_id"),
            "session_uuid": dataset.get("session_uuid"),
            "zarr_path": dataset.get("zarr_path"),
        }
        digest = _sha256_bytes(_canonical_json_bytes(_json_safe(row_payload)))
        for field in ("dataset_id", "recording_id", "session_uuid"):
            item, _issue = _observation(
                semantic_fact=field,
                source_field=field,
                raw_value=dataset.get(field),
                source_kind="registry.datasets",
                source_role="stored_projection",
                source_locator=f"registry.datasets[{dataset.get('dataset_id')!r}]",
                comparison_domain="artifact",
                source_sha256=digest,
            )
            observations.append(item)
        recording_id, defect = _identity_text(dataset.get("recording_id"))
        if defect is not None or recording_id not in recordings_by_id:
            continue
        recording = recordings_by_id[recording_id]
        recording_payload = {
            "recording_id": recording.get("recording_id"),
            "session_uuid": recording.get("session_uuid"),
            "recording_path": recording.get("recording_path"),
        }
        recording_digest = _sha256_bytes(_canonical_json_bytes(_json_safe(recording_payload)))
        for field in ("recording_id", "session_uuid"):
            item, _issue = _observation(
                semantic_fact=field,
                source_field=field,
                raw_value=recording.get(field),
                source_kind="registry.recordings",
                source_role="registry_entity",
                source_locator=f"registry.recordings[{recording_id!r}]",
                comparison_domain="artifact",
                source_sha256=recording_digest,
            )
            observations.append(item)
    return observations


def _classify_observations(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in observations:
        if item.get("comparison_domain") != "artifact":
            continue
        if item.get("semantic_fact") not in SCALAR_COMPARISON_FACTS:
            continue
        grouped[(str(item.get("comparison_domain")), str(item.get("semantic_fact")))].append(item)
    classifications: list[dict[str, Any]] = []
    for (domain, fact), items in sorted(grouped.items()):
        values = sorted(
            {
                str(item["value"])
                for item in items
                if item.get("valid") and item.get("value") is not None
            }
        )
        malformed_count = sum(not item.get("valid", False) and item.get("defect") != "missing" for item in items)
        missing_count = sum(item.get("defect") == "missing" for item in items)
        if len(values) > 1:
            status = "conflict"
        elif len(values) == 1:
            status = "agree"
        else:
            status = "missing"
        classifications.append(
            {
                "comparison_domain": domain,
                "semantic_fact": fact,
                "status": status,
                "distinct_values": values,
                "observation_count": len(items),
                "missing_count": missing_count,
                "malformed_count": malformed_count,
            }
        )
    return classifications


def _normalized_absolute_locator(raw_value: Any) -> tuple[str | None, str | None]:
    """Normalize a declared path without interpreting it relative to process CWD."""

    if not isinstance(raw_value, str):
        return None, "missing" if raw_value is None else "not_string"
    text = raw_value.strip()
    if not text:
        return None, "missing"
    try:
        candidate = Path(text).expanduser()
    except (OSError, RuntimeError, ValueError):
        return None, "invalid_path_syntax"
    if not candidate.is_absolute():
        return None, "relative_path"
    try:
        return str(candidate.resolve(strict=False)), None
    except (OSError, RuntimeError, ValueError):
        return None, "normalization_failed"


def _selected_artifact_datasets(
    datasets: Sequence[Mapping[str, Any]],
    *,
    dataset_ids: Sequence[str] | None,
    artifact_scope: str,
) -> list[Mapping[str, Any]]:
    requested = set(dataset_ids or ())
    selected: list[Mapping[str, Any]] = []
    for row in datasets:
        dataset_id = row.get("dataset_id")
        if requested and dataset_id not in requested:
            continue
        if artifact_scope in {"active_source_analysis", "explicit_source_layout"} and not (
            row.get("status") == "active"
            and row.get("artifact_kind") == "source_recording"
            and row.get("zarr_use") == "analysis"
        ):
            continue
        if artifact_scope == "explicit_source_layout" and not (
            row.get("source_layout") is not None
            or row.get("source_frame_index_schema") is not None
        ):
            continue
        selected.append(row)
    missing_requested = requested - {str(row.get("dataset_id")) for row in selected}
    if missing_requested:
        raise CensusError(
            "Requested dataset IDs were not present in the selected artifact scope: "
            + ", ".join(sorted(missing_requested))
        )
    return sorted(selected, key=lambda row: (str(row.get("zarr_path")), str(row.get("dataset_id"))))


def _artifact_contract_cohort(
    datasets: Sequence[Mapping[str, Any]],
    root_markers: Mapping[str, Any],
) -> str:
    """Classify explicit artifact markers without inferring writer generation."""

    artifact_schema_id = root_markers.get("artifact_schema_id")
    frame_schemas = {
        str(row.get("source_frame_index_schema"))
        for row in datasets
        if row.get("source_frame_index_schema") is not None
    }
    source_layouts = {
        str(row.get("source_layout"))
        for row in datasets
        if row.get("source_layout") is not None
    }
    if artifact_schema_id is not None or frame_schemas:
        return "explicit_version_or_schema_marker"
    if source_layouts:
        return "explicit_layout_unversioned"
    return "legacy_or_unversioned"


def _artifact_census(
    datasets: Sequence[Mapping[str, Any]],
    recordings: Sequence[Mapping[str, Any]],
    *,
    dataset_ids: Sequence[str] | None,
    artifact_scope: str,
    max_json_bytes: int,
    max_zarr_nodes: int,
    max_observations: int,
    scan_parquet: bool,
    progress: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    selected = _selected_artifact_datasets(
        datasets,
        dataset_ids=dataset_ids,
        artifact_scope=artifact_scope,
    )
    recordings_by_id = {
        value: row
        for row in recordings
        if (value := _identity_text(row.get("recording_id"))[0]) is not None
    }
    by_zarr_path: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        raw_path = row.get("zarr_path")
        normalized, locator_defect = _normalized_absolute_locator(raw_path)
        scope_path = (
            normalized
            if normalized is not None
            else f"invalid:{locator_defect}:{_json_safe(raw_path)!r}"
        )
        by_zarr_path[scope_path].append(row)

    zarr_scopes: list[dict[str, Any]] = []
    all_findings: list[dict[str, Any]] = []
    input_roots: list[Path] = []
    recording_dirs: dict[str, list[str]] = defaultdict(list)
    zarr_items = sorted(by_zarr_path.items())
    for zarr_index, (path_text, path_datasets) in enumerate(zarr_items, start=1):
        if progress:
            print(f"zarr {zarr_index}/{len(zarr_items)} {path_text}", file=sys.stderr, flush=True)
        dataset_id_values = sorted(str(row.get("dataset_id")) for row in path_datasets)
        path_recording_ids = sorted(
            {
                value
                for row in path_datasets
                if (value := _identity_text(row.get("recording_id"))[0]) is not None
            }
        )
        if path_text.startswith("invalid:"):
            raw_locator_tokens = sorted(
                {
                    _canonical_json_bytes(_json_safe(row.get("zarr_path"))).decode("utf-8")
                    for row in path_datasets
                }
            )
            raw_locators = [json.loads(token) for token in raw_locator_tokens]
            locator_defects = sorted(
                {
                    str(_normalized_absolute_locator(row.get("zarr_path"))[1])
                    for row in path_datasets
                    if _normalized_absolute_locator(row.get("zarr_path"))[1] is not None
                }
            )
            report = {
                "status": "invalid_locator",
                "path": path_text,
                "metadata_view": "direct_unconsolidated",
                "coverage_complete": False,
                "node_count": 0,
                "observations": [],
                "findings": [
                    _finding(
                        "invalid_zarr_locator",
                        severity="action_required",
                        locator={"dataset_ids": dataset_id_values},
                        detail={
                            "raw_locators": raw_locators,
                            "defects": locator_defects,
                            "relative_paths_are_not_resolved_against_process_cwd": True,
                        },
                    )
                ],
                "source_fences": _fence_summary([]),
            }
        else:
            zarr_path = Path(path_text)
            input_roots.append(zarr_path)
            report = _scan_zarr_metadata(
                zarr_path,
                max_json_bytes=max_json_bytes,
                max_zarr_nodes=max_zarr_nodes,
                max_observations=max_observations,
            )
        if len(path_datasets) > 1 or len(path_recording_ids) > 1:
            report["findings"].append(
                _finding(
                    "ambiguous_zarr_path_binding",
                    severity="action_required",
                    locator={"zarr_path": path_text},
                    detail={
                        "dataset_ids": dataset_id_values,
                        "recording_ids": path_recording_ids,
                    },
                )
            )
            report["status"] = "ambiguous_binding"
            report["coverage_complete"] = False
        registry_contract_markers = {
            field: sorted(
                {
                    str(row.get(field))
                    for row in path_datasets
                    if row.get(field) is not None
                }
            )
            for field in (
                "artifact_kind",
                "zarr_use",
                "status",
                "source_layout",
                "source_frame_index_schema",
            )
        }
        registry_contract_markers = {
            key: values for key, values in registry_contract_markers.items() if values
        }
        contract_cohort = _artifact_contract_cohort(
            path_datasets,
            report.get("contract_markers", {}),
        )
        registry_observations = _registry_observations_for_zarr(path_datasets, recordings_by_id)
        observations = [*registry_observations, *report["observations"]]
        linked_recording_paths: set[str] = set()
        for row in path_datasets:
            recording_id, defect = _identity_text(row.get("recording_id"))
            dataset_locator = str(row.get("dataset_id"))
            if defect is not None:
                report["findings"].append(
                    _finding(
                        "recording_directory_binding_unavailable",
                        severity="action_required",
                        locator={"dataset_id": dataset_locator, "zarr_path": path_text},
                        detail={"reason": f"recording_id_{defect}"},
                    )
                )
                report["coverage_complete"] = False
                if report["status"] == "complete":
                    report["status"] = "incomplete_linkage"
                continue
            if recording_id not in recordings_by_id:
                report["findings"].append(
                    _finding(
                        "recording_directory_binding_unavailable",
                        severity="action_required",
                        locator={
                            "dataset_id": dataset_locator,
                            "recording_id": recording_id,
                            "zarr_path": path_text,
                        },
                        detail={"reason": "recordings_row_missing"},
                    )
                )
                report["coverage_complete"] = False
                if report["status"] == "complete":
                    report["status"] = "incomplete_linkage"
                continue
            raw_recording_path = recordings_by_id[recording_id].get("recording_path")
            normalized_recording, recording_path_defect = _normalized_absolute_locator(
                raw_recording_path
            )
            if normalized_recording is None:
                report["findings"].append(
                    _finding(
                        "recording_directory_binding_unavailable",
                        severity="action_required",
                        locator={
                            "dataset_id": dataset_locator,
                            "recording_id": recording_id,
                            "zarr_path": path_text,
                        },
                        detail={
                            "reason": f"recording_path_{recording_path_defect}",
                            "raw_recording_path": _json_safe(raw_recording_path),
                            "relative_paths_are_not_resolved_against_process_cwd": True,
                        },
                    )
                )
                report["coverage_complete"] = False
                if report["status"] == "complete":
                    report["status"] = "incomplete_linkage"
                continue
            linked_recording_paths.add(normalized_recording)
            recording_dirs[normalized_recording].append(recording_id)
        zarr_scopes.append(
            {
                "scope_key": f"zarr:{path_text}",
                "path": path_text,
                "dataset_ids": dataset_id_values,
                "recording_ids": path_recording_ids,
                "contract_cohort": contract_cohort,
                "root_contract_markers": report.get("contract_markers", {}),
                "registry_contract_markers": registry_contract_markers,
                "linked_recording_paths": sorted(linked_recording_paths),
                "scan_status": report["status"],
                "metadata_view": report["metadata_view"],
                "traversal_policy": report.get("traversal_policy", "bounded_identity_groups_v1"),
                "coverage_complete": report["coverage_complete"],
                "node_count": report["node_count"],
                "root_prefix_only": bool(report.get("root_prefix_only", False)),
                "source_fences": report["source_fences"],
                "observations": sorted(
                    observations,
                    key=lambda item: (
                        str(item.get("semantic_fact")),
                        str(item.get("value")),
                        str(item.get("source_locator")),
                    ),
                ),
                "classifications": [],
                "findings": report["findings"],
            }
        )

    recording_path_cohorts: dict[str, set[str]] = defaultdict(set)
    for scope in zarr_scopes:
        for linked_path in scope["linked_recording_paths"]:
            recording_path_cohorts[linked_path].add(scope["contract_cohort"])

    recording_scopes: list[dict[str, Any]] = []
    recording_items = sorted(recording_dirs.items())
    for recording_index, (path_text, recording_ids) in enumerate(recording_items, start=1):
        if progress:
            print(
                f"recording {recording_index}/{len(recording_items)} {path_text}",
                file=sys.stderr,
                flush=True,
            )
        recording_path = Path(path_text)
        input_roots.append(recording_path)
        report = _scan_recording_directory(
            recording_path,
            max_json_bytes=max_json_bytes,
            max_observations=max_observations,
            scan_parquet=scan_parquet,
        )
        unique_recording_ids = sorted(set(recording_ids))
        if len(unique_recording_ids) > 1:
            report["findings"].append(
                _finding(
                    "ambiguous_recording_path_binding",
                    severity="action_required",
                    locator={"recording_path": path_text},
                    detail={"recording_ids": unique_recording_ids},
                )
            )
            report["status"] = "ambiguous_binding"
            report["coverage_complete"] = False
        registry_observations: list[dict[str, Any]] = []
        for recording_id in unique_recording_ids:
            recording = recordings_by_id[recording_id]
            payload = {
                "recording_id": recording.get("recording_id"),
                "session_uuid": recording.get("session_uuid"),
                "recording_path": recording.get("recording_path"),
            }
            digest = _sha256_bytes(_canonical_json_bytes(_json_safe(payload)))
            for field in ("recording_id", "session_uuid"):
                item, _issue = _observation(
                    semantic_fact=field,
                    source_field=field,
                    raw_value=recording.get(field),
                    source_kind="registry.recordings",
                    source_role="registry_entity",
                    source_locator=f"registry.recordings[{recording_id!r}]",
                    comparison_domain="artifact",
                    source_sha256=digest,
                )
                registry_observations.append(item)
        observations = [*registry_observations, *report["observations"]]
        classifications = _classify_observations(observations)
        for classification in classifications:
            if classification["status"] == "conflict":
                report["findings"].append(
                    _finding(
                        "recording_sidecar_identity_conflict",
                        severity="action_required",
                        locator={"recording_path": path_text, "semantic_fact": classification["semantic_fact"]},
                        detail={"distinct_values": classification["distinct_values"]},
                    )
                )
        recording_scopes.append(
            {
                "scope_key": f"recording_dir:{path_text}",
                "path": path_text,
                "recording_ids": sorted(set(recording_ids)),
                "linked_contract_cohorts": sorted(recording_path_cohorts.get(path_text, set())),
                "scan_status": report["status"],
                "coverage_complete": report["coverage_complete"],
                "sources": report["sources"],
                "parquet": report["parquet"],
                "observations": sorted(
                    observations,
                    key=lambda item: (
                        str(item.get("comparison_domain")),
                        str(item.get("semantic_fact")),
                        str(item.get("value")),
                        str(item.get("source_locator")),
                    ),
                ),
                "classifications": classifications,
                "findings": report["findings"],
            }
        )

    recording_scope_by_path = {scope["path"]: scope for scope in recording_scopes}
    for scope in zarr_scopes:
        cross_artifact_observations = list(scope["observations"])
        for recording_path in scope["linked_recording_paths"]:
            recording_scope = recording_scope_by_path.get(recording_path)
            if recording_scope is None:
                continue
            cross_artifact_observations.extend(
                item
                for item in recording_scope["observations"]
                if item.get("source_kind") != "registry.recordings"
                and item.get("semantic_fact") != "recording_dir_name"
            )
        scope["classifications"] = _classify_observations(cross_artifact_observations)
        for classification in scope["classifications"]:
            if classification["status"] != "conflict":
                continue
            scope["findings"].append(
                _finding(
                    "artifact_identity_conflict",
                    severity="action_required",
                    locator={"zarr_path": scope["path"], "semantic_fact": classification["semantic_fact"]},
                    detail={
                        "distinct_values": classification["distinct_values"],
                        "comparison_includes_linked_recording_sidecars": True,
                    },
                )
            )

    all_findings = [
        finding
        for scope in [*zarr_scopes, *recording_scopes]
        for finding in scope["findings"]
    ]

    donor_findings: list[dict[str, Any]] = []
    scanned_zarr_paths = set(by_zarr_path)
    seen_donor_declarations: set[tuple[str, str, str]] = set()
    self_reference_count = 0
    for scope in zarr_scopes:
        for observation in scope["observations"]:
            if observation.get("semantic_fact") != "donor_locator":
                continue
            donor = str(observation.get("value"))
            declaration_key = (
                scope["path"],
                donor,
                str(observation.get("source_field")),
            )
            if declaration_key in seen_donor_declarations:
                continue
            seen_donor_declarations.add(declaration_key)
            donor_path = Path(donor).expanduser() if donor else Path()
            normalized = str(donor_path.resolve(strict=False)) if donor_path.is_absolute() else None
            if normalized is not None and (
                normalized == scope["path"]
                or _is_relative_to(Path(normalized), Path(scope["path"]))
            ):
                self_reference_count += 1
                continue
            donor_findings.append(
                _finding(
                    "donor_binding_unproven",
                    severity="unresolved",
                    locator={"target_zarr": scope["path"], "donor_zarr": donor},
                    detail={
                        "donor_in_scanned_registry_scope": (
                            normalized in scanned_zarr_paths if normalized is not None else None
                        ),
                        "relative_locator_resolution": "undeclared" if normalized is None else None,
                        "reason": "No digest-bound same-recording/camera/frame-map proof was observed",
                    },
                )
            )
    all_findings.extend(donor_findings)
    zarr_cohort_summary: dict[str, dict[str, Any]] = {}
    for cohort in sorted({scope["contract_cohort"] for scope in zarr_scopes}):
        cohort_scopes = [scope for scope in zarr_scopes if scope["contract_cohort"] == cohort]
        cohort_findings = [finding for scope in cohort_scopes for finding in scope["findings"]]
        zarr_cohort_summary[cohort] = {
            "scope_count": len(cohort_scopes),
            "coverage_complete_count": sum(bool(scope["coverage_complete"]) for scope in cohort_scopes),
            "findings": _finding_summary(cohort_findings),
        }
    artifact = {
        "scope_policy": artifact_scope,
        "scope_semantics": (
            "Explicit source-layout/frame-index registry metadata selects an evidence cohort; "
            "it does not prove which Palette commit or writer generation produced an artifact."
        ),
        "proves_writer_generation": False,
        "selected_dataset_count": len(selected),
        "selected_zarr_scope_count": len(zarr_scopes),
        "selected_recording_directory_count": len(recording_scopes),
        "metadata_view": "direct_unconsolidated",
        "opens_zarr_arrays": False,
        "decodes_video": False,
        "parquet_identity_scan": scan_parquet,
        "zarr_contract_cohorts": zarr_cohort_summary,
        "zarr_scopes": zarr_scopes,
        "recording_directory_scopes": recording_scopes,
        "donor_findings": donor_findings,
        "donor_declaration_summary": {
            "unique_declaration_count": len(seen_donor_declarations),
            "self_reference_count": self_reference_count,
            "unresolved_or_external_count": len(donor_findings),
        },
    }
    return artifact, all_findings, input_roots


def _finding_summary(findings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_code = Counter(str(item.get("code")) for item in findings)
    by_severity = Counter(str(item.get("severity")) for item in findings)
    return {
        "total": len(findings),
        "by_code": dict(sorted(by_code.items())),
        "by_severity": dict(sorted(by_severity.items())),
        "has_action_required": by_severity.get("action_required", 0) > 0,
        "has_unresolved": by_severity.get("unresolved", 0) > 0,
    }


def run_census(
    registry_path: Path,
    *,
    scan_artifacts: bool = True,
    dataset_ids: Sequence[str] | None = None,
    artifact_scope: str = "explicit_source_layout",
    scan_parquet: bool = True,
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
    max_zarr_nodes: int = DEFAULT_MAX_ZARR_NODES,
    max_observations: int = DEFAULT_MAX_OBSERVATIONS_PER_SOURCE,
    progress: bool = False,
) -> dict[str, Any]:
    """Build a stable, non-authoritative identity census report.

    The registry is first copied through SQLite's backup API from a URI-mode
    read-only connection.  All registry queries then run against that local
    snapshot with ``query_only`` enabled.  Filesystem metadata is fenced
    separately; the report never claims a cross-store atomic snapshot.
    """

    if artifact_scope not in {
        "active_source_analysis",
        "explicit_source_layout",
        "all_registered",
    }:
        raise ValueError(f"Unsupported artifact scope: {artifact_scope}")
    registry_path = registry_path.expanduser().resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="palette-recording-identity-census-") as temp_dir:
        snapshot_path = Path(temp_dir) / "registry-snapshot.sqlite"
        snapshot = _capture_registry_snapshot(registry_path, snapshot_path)
        registry, datasets, recordings, registry_findings = _scan_registry(
            snapshot_path,
            dataset_ids=dataset_ids,
            artifact_scope=artifact_scope,
        )
        input_roots: list[Path] = []
        if scan_artifacts:
            artifacts, artifact_findings, input_roots = _artifact_census(
                datasets,
                recordings,
                dataset_ids=dataset_ids,
                artifact_scope=artifact_scope,
                max_json_bytes=max_json_bytes,
                max_zarr_nodes=max_zarr_nodes,
                max_observations=max_observations,
                scan_parquet=scan_parquet,
                progress=progress,
            )
            if not scan_parquet and any(
                scope.get("parquet", {}).get("status") == "deferred"
                for scope in artifacts["recording_directory_scopes"]
                if isinstance(scope.get("parquet"), Mapping)
            ):
                artifact_findings.append(
                    _finding(
                        "parquet_identity_scan_deferred",
                        severity="unresolved",
                        detail={"reason": "Frame-index Parquet identity scan disabled explicitly"},
                    )
                )
        else:
            artifacts = {
                "scope_policy": artifact_scope,
                "status": "deferred",
                "reason": "Filesystem artifact scan disabled explicitly",
                "metadata_view": "direct_unconsolidated",
                "opens_zarr_arrays": False,
                "decodes_video": False,
                "zarr_scopes": [],
                "recording_directory_scopes": [],
                "donor_findings": [],
            }
            artifact_findings = [
                _finding(
                    "artifact_scan_deferred",
                    severity="unresolved",
                    detail={"reason": "Filesystem artifact scan disabled explicitly"},
                )
            ]

    findings = sorted(
        [*registry_findings, *artifact_findings],
        key=lambda item: (
            str(item.get("severity")),
            str(item.get("code")),
            json.dumps(item.get("locator", {}), sort_keys=True, default=str),
        ),
    )
    artifact_scope_complete = bool(
        scan_artifacts
        and all(scope.get("coverage_complete") for scope in artifacts.get("zarr_scopes", []))
        and all(
            scope.get("coverage_complete")
            for scope in artifacts.get("recording_directory_scopes", [])
        )
    )
    body: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "schema_version": 1,
        "read_only": True,
        "authorizes_mutation": False,
        "effective_identity_values_emitted": False,
        "declared_scope_complete": artifact_scope_complete,
        "governing_rule": (
            "This report inventories evidence and projections. It never substitutes "
            "recording_id, session_uuid, or legacy session labels for one another and "
            "never resolves a conflict by precedence."
        ),
        "snapshot_model": {
            "registry": "consistent SQLite backup from read-only source connection",
            "filesystem": "per-file hash/stat fences plus direct Zarr group inventory fences",
            "cross_store_atomic": False,
        },
        "coverage": {
            "registry": (
                "global schema inventory for all SQLite identity-bearing tables/views; "
                "row-level findings restricted to the selected artifact identities"
            ),
            "artifacts": (
                "registered active source-recording analysis Zarrs carrying explicit registry "
                "source-layout or frame-index-schema metadata and their linked recording directories"
                if artifact_scope == "explicit_source_layout"
                else (
                    "registered active source-recording analysis Zarrs and their linked recording "
                    "directories"
                    if artifact_scope == "active_source_analysis"
                    else "all registered datasets"
                )
            ),
            "included": [
                "datasets and recordings roots",
                "all stored registry identity projections",
                "identity-bearing registry views as derived aggregate/cardinality inventory",
                "direct Zarr group metadata without array reads",
                "bounded declared recording JSON sidecars and clip manifests",
                "recording frame-index Parquet identity columns" if scan_parquet else "Parquet location only",
            ],
            "deferred": [
                "HDF5 identity attributes",
                "video payload decoding",
                "Zarr array payloads and frame-map equality",
                "consolidated-metadata comparison (reserved for lifecycle resolver work)",
                "unregistered filesystem discovery outside registry locators",
                "active artifacts without explicit registry source-layout/frame-index markers",
                "exact producer commit/writer generation where artifacts do not bind it",
            ],
        },
        "registry_snapshot": snapshot,
        "registry": registry,
        "artifacts": artifacts,
        "findings": findings,
        "summary": _finding_summary(findings),
        "input_roots": sorted({str(path) for path in input_roots}),
    }
    body["report_digest"] = _sha256_bytes(_canonical_json_bytes(body))
    return {
        "operational": {"generated_at_utc": _utc_now()},
        "census": body,
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def write_report_read_only_safe(
    report: Mapping[str, Any],
    output_path: Path,
    *,
    registry_path: Path,
) -> None:
    """Write a new report while refusing every observed input location."""

    output = output_path.expanduser().resolve(strict=False)
    registry = registry_path.expanduser().resolve(strict=True)
    input_roots = [Path(value).expanduser().resolve(strict=False) for value in report["census"].get("input_roots", [])]
    forbidden_files = {
        registry,
        Path(f"{registry}-wal"),
        Path(f"{registry}-shm"),
        Path(f"{registry}-journal"),
    }
    if output in forbidden_files or any(output == root or _is_relative_to(output, root) for root in input_roots):
        raise CensusError(f"Refusing to write census output over or below an observed input: {output}")
    if output.exists() or output.is_symlink():
        raise CensusError(f"Refusing to overwrite existing census output: {output}")
    if not output.parent.is_dir():
        raise CensusError(f"Census output parent does not exist: {output.parent}")
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    parent_before = output.parent.stat()
    parent_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    parent_opened = os.fstat(parent_descriptor)
    if (parent_before.st_dev, parent_before.st_ino) != (parent_opened.st_dev, parent_opened.st_ino):
        os.close(parent_descriptor)
        raise UnstableSnapshotError(f"Census output parent changed before creation: {output.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    created = False
    try:
        descriptor = os.open(output.name, flags, 0o600, dir_fd=parent_descriptor)
        created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            try:
                os.unlink(output.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(parent_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True, help="Palette SQLite registry to inspect read-only.")
    parser.add_argument("--output", type=Path, help="New JSON report path. Existing files are never overwritten.")
    parser.add_argument("--dataset-id", action="append", dest="dataset_ids", help="Limit artifact scanning to an exact dataset ID; repeatable.")
    parser.add_argument(
        "--artifact-scope",
        choices=("explicit_source_layout", "active_source_analysis", "all_registered"),
        default="explicit_source_layout",
        help=(
            "Filesystem artifact selection policy. The default selects explicit registry "
            "source-layout/frame-index markers and defers unmarked artifacts; registry schema "
            "inventory remains global while row findings follow the selected scope."
        ),
    )
    parser.add_argument("--registry-only", action="store_true", help="Defer filesystem scanning explicitly.")
    parser.add_argument("--no-parquet", action="store_true", help="Record frame-index Parquet locations without reading identity columns.")
    parser.add_argument("--max-json-bytes", type=int, default=DEFAULT_MAX_JSON_BYTES)
    parser.add_argument("--max-zarr-nodes", type=int, default=DEFAULT_MAX_ZARR_NODES)
    parser.add_argument("--max-observations", type=int, default=DEFAULT_MAX_OBSERVATIONS_PER_SOURCE)
    parser.add_argument("--progress", action="store_true", help="Print artifact progress to stderr.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_census(
            args.registry,
            scan_artifacts=not args.registry_only,
            dataset_ids=args.dataset_ids,
            artifact_scope=args.artifact_scope,
            scan_parquet=not args.no_parquet,
            max_json_bytes=args.max_json_bytes,
            max_zarr_nodes=args.max_zarr_nodes,
            max_observations=args.max_observations,
            progress=args.progress,
        )
        if args.output is not None:
            write_report_read_only_safe(report, args.output, registry_path=args.registry)
        else:
            print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    except UnstableSnapshotError as exc:
        print(f"ERROR unstable snapshot: {exc}")
        return 3
    except (CensusError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    summary = report["census"]["summary"]
    return (
        1
        if not report["census"]["declared_scope_complete"]
        or summary["has_action_required"]
        or summary["has_unresolved"]
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
