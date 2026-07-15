"""Consolidate recording aliases into canonical recording-directory identities.

The repair updates only operational ``recording_id`` fields in recording
manifests and Zarr-root metadata.  Session UUIDs and nested historical
provenance remain unchanged.  Registry updates run in one foreign-key-checked
transaction after a SQLite backup has been created.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


REPAIR_ID = "palette.recording_identity_consolidation.v1"


@dataclass(frozen=True)
class MetadataEdit:
    path: Path
    old_recording_id: str | None
    canonical_recording_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class RecordingIdentityPlan:
    recording_dir: Path
    canonical_recording_id: str
    aliases: tuple[str, ...]
    metadata_edits: tuple[MetadataEdit, ...]
    registry_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_dir": str(self.recording_dir),
            "canonical_recording_id": self.canonical_recording_id,
            "aliases": list(self.aliases),
            "metadata_edits": [
                {
                    "path": str(edit.path),
                    "old_recording_id": edit.old_recording_id,
                    "canonical_recording_id": edit.canonical_recording_id,
                }
                for edit in self.metadata_edits
            ],
            "registry_counts": dict(self.registry_counts),
        }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _metadata_edit(path: Path, *, canonical_recording_id: str, zarr_root: bool) -> MetadataEdit | None:
    payload = _load_json(path)
    container = payload.get("attributes") if zarr_root else payload
    if not isinstance(container, dict):
        raise ValueError(f"Expected {'attributes' if zarr_root else 'root'} object in {path}.")
    old_value = container.get("recording_id")
    old_recording_id = str(old_value).strip() if old_value is not None else None
    if old_recording_id == canonical_recording_id:
        return None
    updated = json.loads(json.dumps(payload))
    updated_container = updated["attributes"] if zarr_root else updated
    updated_container["recording_id"] = canonical_recording_id
    return MetadataEdit(
        path=path,
        old_recording_id=old_recording_id,
        canonical_recording_id=canonical_recording_id,
        payload=updated,
    )


def _recording_id_tables(conn: sqlite3.Connection) -> tuple[str, ...]:
    rows = conn.execute(
        """
        SELECT m.name
        FROM sqlite_master AS m
        WHERE m.type = 'table'
          AND m.name NOT LIKE 'sqlite_%'
          AND EXISTS (
              SELECT 1 FROM pragma_table_info(m.name) AS p WHERE p.name = 'recording_id'
          )
        ORDER BY m.name;
        """
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _registry_counts(
    conn: sqlite3.Connection,
    *,
    aliases: Sequence[str],
) -> dict[str, int]:
    if not aliases:
        return {}
    placeholders = ",".join("?" for _ in aliases)
    counts: dict[str, int] = {}
    for table in _recording_id_tables(conn):
        query = (
            f"SELECT COUNT(*) FROM {_quote_identifier(table)} "
            f"WHERE recording_id IN ({placeholders});"
        )
        count = int(conn.execute(query, tuple(aliases)).fetchone()[0])
        if count:
            counts[table] = count
    return counts


def build_plans(
    conn: sqlite3.Connection,
    *,
    recording_dirs: Sequence[Path],
) -> list[RecordingIdentityPlan]:
    plans: list[RecordingIdentityPlan] = []
    for raw_dir in recording_dirs:
        recording_dir = raw_dir.expanduser().resolve()
        canonical = recording_dir.name
        manifest_path = recording_dir / "recording_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Recording manifest not found: {manifest_path}")
        canonical_row = conn.execute(
            "SELECT recording_id FROM recordings WHERE recording_id = ?;",
            (canonical,),
        ).fetchone()
        if canonical_row is None:
            raise ValueError(
                f"Registry has no canonical recordings row {canonical!r}; refusing alias consolidation."
            )
        alias_rows = conn.execute(
            """
            SELECT recording_id
            FROM recordings
            WHERE recording_name = ? AND recording_id <> ?
            ORDER BY recording_id;
            """,
            (canonical, canonical),
        ).fetchall()
        aliases = tuple(str(row[0]) for row in alias_rows)

        edits: list[MetadataEdit] = []
        manifest_edit = _metadata_edit(
            manifest_path,
            canonical_recording_id=canonical,
            zarr_root=False,
        )
        if manifest_edit is not None:
            edits.append(manifest_edit)
        for zarr_root in sorted((recording_dir / "zarr").glob("*.zarr")):
            metadata_path = zarr_root / "zarr.json"
            if not metadata_path.is_file():
                continue
            edit = _metadata_edit(
                metadata_path,
                canonical_recording_id=canonical,
                zarr_root=True,
            )
            if edit is not None:
                edits.append(edit)

        plans.append(
            RecordingIdentityPlan(
                recording_dir=recording_dir,
                canonical_recording_id=canonical,
                aliases=aliases,
                metadata_edits=tuple(edits),
                registry_counts=_registry_counts(conn, aliases=aliases),
            )
        )
    return plans


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temp, path.stat().st_mode)
    os.replace(temp, path)


def _backup_registry(conn: sqlite3.Connection, backup_path: Path) -> None:
    if backup_path.exists():
        raise FileExistsError(f"Registry backup already exists: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    destination = sqlite3.connect(str(backup_path))
    try:
        conn.backup(destination)
    finally:
        destination.close()


def _apply_registry_plans(
    conn: sqlite3.Connection,
    *,
    plans: Sequence[RecordingIdentityPlan],
) -> dict[str, int]:
    table_updates: dict[str, int] = {}
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("BEGIN IMMEDIATE;")
    try:
        tables = tuple(table for table in _recording_id_tables(conn) if table != "recordings")
        for plan in plans:
            canonical = plan.canonical_recording_id
            for alias in plan.aliases:
                for table in tables:
                    cursor = conn.execute(
                        f"UPDATE {_quote_identifier(table)} SET recording_id = ? WHERE recording_id = ?;",
                        (canonical, alias),
                    )
                    if cursor.rowcount:
                        table_updates[table] = table_updates.get(table, 0) + int(cursor.rowcount)
                cursor = conn.execute("DELETE FROM recordings WHERE recording_id = ?;", (alias,))
                if cursor.rowcount:
                    table_updates["recordings_deleted"] = (
                        table_updates.get("recordings_deleted", 0) + int(cursor.rowcount)
                    )

        foreign_key_issues = conn.execute("PRAGMA foreign_key_check;").fetchall()
        if foreign_key_issues:
            raise ValueError(f"foreign_key_check found {len(foreign_key_issues)} issue(s).")
        integrity = str(conn.execute("PRAGMA integrity_check;").fetchone()[0])
        if integrity.lower() != "ok":
            raise ValueError(f"integrity_check failed: {integrity}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return table_updates


def apply_plans(
    conn: sqlite3.Connection,
    *,
    plans: Sequence[RecordingIdentityPlan],
    backup_path: Path,
) -> dict[str, Any]:
    _backup_registry(conn, backup_path)
    originals = {edit.path: edit.path.read_bytes() for plan in plans for edit in plan.metadata_edits}
    try:
        for plan in plans:
            for edit in plan.metadata_edits:
                _atomic_write_json(edit.path, edit.payload)
        table_updates = _apply_registry_plans(conn, plans=plans)
    except Exception:
        for path, payload in originals.items():
            temp = path.with_name(f".{path.name}.{uuid4().hex}.restore")
            temp.write_bytes(payload)
            os.chmod(temp, path.stat().st_mode)
            os.replace(temp, path)
        raise
    return {
        "status": "complete",
        "backup_path": str(backup_path),
        "table_updates": table_updates,
        "metadata_files_updated": len(originals),
    }


def validate_plans(
    conn: sqlite3.Connection,
    *,
    plans: Sequence[RecordingIdentityPlan],
) -> dict[str, Any]:
    issues: list[str] = []
    for plan in plans:
        canonical = plan.canonical_recording_id
        for alias in plan.aliases:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM recordings WHERE recording_id = ?;", (alias,)
            ).fetchone()[0]
            if remaining:
                issues.append(f"recordings_alias_remaining:{alias}")
        dataset_mismatches = conn.execute(
            """
            SELECT COUNT(*)
            FROM datasets
            WHERE zarr_path LIKE ? AND recording_id <> ?;
            """,
            (str(plan.recording_dir / "zarr") + "/%", canonical),
        ).fetchone()[0]
        if dataset_mismatches:
            issues.append(f"dataset_recording_id_mismatch:{canonical}:{dataset_mismatches}")
        for path in (plan.recording_dir / "recording_manifest.json", *sorted((plan.recording_dir / "zarr").glob("*.zarr/zarr.json"))):
            payload = _load_json(path)
            container = payload.get("attributes") if path.name == "zarr.json" else payload
            value = container.get("recording_id") if isinstance(container, dict) else None
            if str(value or "") != canonical:
                issues.append(f"metadata_recording_id_mismatch:{path}")
    fk_issues = conn.execute("PRAGMA foreign_key_check;").fetchall()
    if fk_issues:
        issues.append(f"foreign_key_check:{len(fk_issues)}")
    return {"status": "ok" if not issues else "error", "issues": issues}


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--recording-dir", action="append", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.apply and args.backup is None:
        raise SystemExit("--backup is required with --apply")
    conn = sqlite3.connect(str(args.registry.expanduser()))
    conn.row_factory = sqlite3.Row
    try:
        plans = build_plans(conn, recording_dirs=args.recording_dir)
        payload: dict[str, Any] = {
            "repair_id": REPAIR_ID,
            "apply": bool(args.apply),
            "planned_at_utc": datetime.now(timezone.utc).isoformat(),
            "recordings": [plan.to_dict() for plan in plans],
        }
        if args.apply:
            payload["apply_result"] = apply_plans(
                conn,
                plans=plans,
                backup_path=args.backup.expanduser(),
            )
            payload["validation"] = validate_plans(conn, plans=plans)
            if payload["validation"]["status"] != "ok":
                raise SystemExit("Post-apply recording identity validation failed.")
        else:
            payload["validation"] = {"status": "dry_run"}
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(
                json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, allow_nan=False, sort_keys=True))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
