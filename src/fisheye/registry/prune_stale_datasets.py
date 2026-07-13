"""Dry-run-first cleanup for stale temp-root dataset rows in the registry."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from fisheye.registry.db import SQLITE_BUSY_TIMEOUT_MS


TEMP_ROOT_CLASS_CHOICES = ("pytest-tmp", "tmp", "var-tmp", "dev-shm", "all-temp")


@dataclass(frozen=True)
class DependentTable:
    table: str
    columns: tuple[str, ...]


DEPENDENT_DATASET_TABLES: tuple[DependentTable, ...] = (
    DependentTable("acquisition_video_streams", ("dataset_id",)),
    DependentTable("crop_quality", ("dataset_id",)),
    DependentTable("dataset_lineage", ("child_dataset_id", "parent_dataset_id")),
    DependentTable("detect_performance", ("dataset_id",)),
    DependentTable("detect_quality", ("dataset_id",)),
    DependentTable("detection_data_profile", ("dataset_id",)),
    DependentTable("detection_sources", ("dataset_id",)),
    DependentTable("eye_mask_data_profile", ("dataset_id",)),
    DependentTable("eye_mask_performance", ("dataset_id",)),
    DependentTable("eye_mask_quality", ("dataset_id",)),
    DependentTable("keypoint_data_profile", ("dataset_id",)),
    DependentTable("keypoint_performance", ("dataset_id",)),
    DependentTable("keypoint_quality", ("dataset_id",)),
    DependentTable("provenance", ("dataset_id",)),
    DependentTable("recording_chasers", ("dataset_id",)),
    DependentTable("recording_step_status", ("dataset_id",)),
    DependentTable("recording_step_status_history", ("dataset_id",)),
    DependentTable("recording_stimulus_modes", ("dataset_id",)),
    DependentTable("recording_stimulus_runs", ("dataset_id",)),
    DependentTable("recording_stimulus_steps", ("dataset_id",)),
    DependentTable("recording_subjects", ("dataset_id",)),
    DependentTable("subject_mask_component_quality", ("dataset_id",)),
    DependentTable("subject_mask_data_profile", ("dataset_id",)),
    DependentTable("subject_mask_performance", ("dataset_id",)),
    DependentTable("training_image_profile", ("dataset_id",)),
)


@dataclass(frozen=True)
class DatasetCandidate:
    dataset_id: str
    zarr_path: str
    recording_id: str | None
    artifact_kind: str | None
    zarr_use: str | None
    status: str | None
    path_class: str
    path_exists: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "zarr_path": self.zarr_path,
            "recording_id": self.recording_id,
            "artifact_kind": self.artifact_kind,
            "zarr_use": self.zarr_use,
            "status": self.status,
            "path_class": self.path_class,
            "path_exists": self.path_exists,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _quote_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
        raise ValueError(f"Unsafe SQLite identifier: {name!r}")
    return f'"{name}"'


def _placeholders(values: Sequence[Any]) -> str:
    if not values:
        raise ValueError("Expected at least one SQL value.")
    return ", ".join("?" for _ in values)


def _readonly_uri(path: Path) -> str:
    absolute = path.expanduser().resolve(strict=False)
    return f"file:{quote(str(absolute), safe='/')}?mode=ro"


def connect_read_only(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(_readonly_uri(path), uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA query_only = ON;")
    return conn


def connect_writable(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _resolved_absolute_path(path_text: str) -> Path | None:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        return None
    return path.resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False
    return os.path.commonpath([str(path), str(root)]) == str(root)


def temp_roots() -> dict[str, tuple[Path, ...]]:
    tmp_candidates = {Path(tempfile.gettempdir()).resolve(strict=False), Path("/tmp")}
    return {
        "tmp": tuple(sorted(tmp_candidates, key=str)),
        "var-tmp": (Path("/var/tmp"),),
        "dev-shm": (Path("/dev/shm"),),
    }


def classify_zarr_path(path_text: str) -> str | None:
    path = _resolved_absolute_path(path_text)
    if path is None:
        return None
    for roots in temp_roots().values():
        for root in roots:
            if _is_relative_to(path, root):
                relative_parts = path.relative_to(root).parts
                if relative_parts and relative_parts[0].startswith("pytest-of-"):
                    return "pytest-tmp"
    for class_name, roots in temp_roots().items():
        for root in roots:
            if _is_relative_to(path, root):
                return class_name
    if _is_relative_to(path, Path("/home")):
        return "home-review"
    return None


def _all_dataset_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT dataset_id, zarr_path, recording_id, artifact_kind, zarr_use, status
        FROM datasets
        ORDER BY dataset_id;
        """
    ).fetchall()


def scan_candidates(conn: sqlite3.Connection) -> dict[str, list[DatasetCandidate]]:
    grouped: dict[str, list[DatasetCandidate]] = {
        "pytest-tmp": [],
        "tmp": [],
        "var-tmp": [],
        "dev-shm": [],
        "home-review": [],
        "unowned-analysis-review": [],
    }
    for row in _all_dataset_rows(conn):
        path_class = classify_zarr_path(str(row["zarr_path"]))
        recording_id = str(row["recording_id"] or "").strip() or None
        if (
            path_class is None
            and str(row["zarr_use"] or "").strip().casefold() == "analysis"
            and str(row["status"] or "").strip().casefold() == "active"
            and recording_id is None
        ):
            path_class = "unowned-analysis-review"
        if path_class not in grouped:
            continue
        path = Path(str(row["zarr_path"]))
        grouped[path_class].append(
            DatasetCandidate(
                dataset_id=str(row["dataset_id"]),
                zarr_path=str(row["zarr_path"]),
                recording_id=recording_id,
                artifact_kind=(
                    str(row["artifact_kind"]) if row["artifact_kind"] is not None else None
                ),
                zarr_use=str(row["zarr_use"]) if row["zarr_use"] is not None else None,
                status=str(row["status"]) if row["status"] is not None else None,
                path_class=path_class,
                path_exists=path.exists(),
            )
        )
    return grouped


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%';
        """
    ).fetchall()
    return {str(row["name"]) for row in rows}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote_ident(table)});").fetchall()
    return {str(row["name"]) for row in rows}


def assert_dependent_table_map_covers_schema(conn: sqlite3.Connection) -> None:
    table_names = _table_names(conn)
    mapped = {entry.table for entry in DEPENDENT_DATASET_TABLES}
    missing_tables = sorted(mapped - table_names)
    if missing_tables:
        joined = ", ".join(missing_tables)
        raise RuntimeError(f"Dependent table map references missing table(s): {joined}")

    live_dataset_id_tables = {
        table
        for table in table_names
        if table != "datasets" and "dataset_id" in _table_columns(conn, table)
    }
    uncovered = sorted(live_dataset_id_tables - mapped)
    if uncovered:
        joined = ", ".join(uncovered)
        raise RuntimeError(f"Dependent table map does not cover dataset_id table(s): {joined}")

    for entry in DEPENDENT_DATASET_TABLES:
        live_columns = _table_columns(conn, entry.table)
        missing_columns = sorted(set(entry.columns) - live_columns)
        if missing_columns:
            joined = ", ".join(missing_columns)
            raise RuntimeError(f"Dependent table map has missing column(s) for {entry.table}: {joined}")

    lineage = next(entry for entry in DEPENDENT_DATASET_TABLES if entry.table == "dataset_lineage")
    if set(lineage.columns) != {"child_dataset_id", "parent_dataset_id"}:
        raise RuntimeError("Dependent table map must cover both dataset_lineage dataset references.")


def _count_for_table(
    conn: sqlite3.Connection,
    table: DependentTable,
    dataset_ids: Sequence[str],
) -> int:
    if not dataset_ids:
        return 0
    placeholders = _placeholders(dataset_ids)
    predicates = [
        f"{_quote_ident(column)} IN ({placeholders})"
        for column in table.columns
    ]
    sql = f"SELECT COUNT(*) AS count FROM {_quote_ident(table.table)} WHERE {' OR '.join(predicates)};"
    params: list[str] = []
    for _column in table.columns:
        params.extend(dataset_ids)
    row = conn.execute(sql, params).fetchone()
    return int(row["count"] if row is not None else 0)


def dependent_counts(conn: sqlite3.Connection, dataset_ids: Sequence[str]) -> dict[str, int]:
    return {
        entry.table: _count_for_table(conn, entry, dataset_ids)
        for entry in DEPENDENT_DATASET_TABLES
    }


def _recordings_to_delete(conn: sqlite3.Connection, dataset_ids: Sequence[str]) -> list[str]:
    if not dataset_ids:
        return []
    placeholders = _placeholders(dataset_ids)
    rows = conn.execute(
        f"""
        SELECT DISTINCT d.recording_id
        FROM datasets d
        INNER JOIN recordings r ON r.recording_id = d.recording_id
        WHERE d.dataset_id IN ({placeholders})
          AND d.recording_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM datasets remaining
              WHERE remaining.recording_id = d.recording_id
                AND remaining.dataset_id NOT IN ({placeholders})
          )
        ORDER BY d.recording_id;
        """,
        [*dataset_ids, *dataset_ids],
    ).fetchall()
    return [str(row["recording_id"]) for row in rows]


def _flatten_ids(candidates: Mapping[str, Sequence[DatasetCandidate]], class_names: Iterable[str]) -> list[str]:
    ids: list[str] = []
    for class_name in class_names:
        ids.extend(candidate.dataset_id for candidate in candidates.get(class_name, ()))
    return sorted(set(ids))


def _build_report(
    conn: sqlite3.Connection,
    registry: Path,
    *,
    execute: bool,
    include_temp_root: Sequence[str],
    include_dataset_id: Sequence[str],
) -> dict[str, Any]:
    assert_dependent_table_map_covers_schema(conn)
    candidates = scan_candidates(conn)
    temp_class_names = ("pytest-tmp", "tmp", "var-tmp", "dev-shm")
    temp_ids = _flatten_ids(candidates, temp_class_names)
    home_candidates = candidates["home-review"]
    unowned_candidates = candidates["unowned-analysis-review"]
    review_by_id = {
        candidate.dataset_id: candidate
        for candidate in (*home_candidates, *unowned_candidates)
    }
    opted_review_ids = sorted(set(include_dataset_id) & set(review_by_id))

    selected_temp_classes = _selected_temp_classes(include_temp_root)
    selected_ids = _flatten_ids(candidates, selected_temp_classes)
    selected_ids.extend(opted_review_ids)
    selected_ids = sorted(set(selected_ids))

    dependent_by_class = {
        class_name: dependent_counts(conn, _flatten_ids(candidates, (class_name,)))
        for class_name in temp_class_names
    }
    dependent_by_class["selected"] = dependent_counts(conn, selected_ids)

    recordings_by_class = {
        class_name: len(_recordings_to_delete(conn, _flatten_ids(candidates, (class_name,))))
        for class_name in temp_class_names
    }
    recordings_by_class["selected"] = len(_recordings_to_delete(conn, selected_ids))

    return {
        "generated_at_utc": _utc_now(),
        "registry": str(registry),
        "mode": "execute" if execute else "dry-run",
        "read_only": not execute,
        "temp_roots": {
            class_name: [str(root) for root in roots]
            for class_name, roots in temp_roots().items()
        },
        "selection": {
            "include_temp_root": list(include_temp_root),
            "include_dataset_id": list(include_dataset_id),
            "selected_dataset_count": len(selected_ids),
        },
        "summary": {
            "temp_root_dataset_count": len(temp_ids),
            "home_review_dataset_count": len(home_candidates),
            "unowned_analysis_review_dataset_count": len(unowned_candidates),
            "selected_dataset_count": len(selected_ids),
            "selected_recording_count": recordings_by_class["selected"],
        },
        "classes": {
            class_name: {
                "dataset_count": len(candidates[class_name]),
                "dependent_row_counts": dependent_by_class[class_name],
                "recordings_that_would_be_deleted": recordings_by_class[class_name],
                "datasets": [candidate.as_dict() for candidate in candidates[class_name]],
            }
            for class_name in temp_class_names
        },
        "needs_maintainer_review": {
            "path_class": "home-review",
            "dataset_count": len(home_candidates),
            "included_dataset_count": len(
                set(opted_review_ids) & {row.dataset_id for row in home_candidates}
            ),
            "datasets": [candidate.as_dict() for candidate in home_candidates],
        },
        "unowned_analysis_review": {
            "path_class": "unowned-analysis-review",
            "dataset_count": len(unowned_candidates),
            "included_dataset_count": len(
                set(opted_review_ids) & {row.dataset_id for row in unowned_candidates}
            ),
            "datasets": [candidate.as_dict() for candidate in unowned_candidates],
        },
        "selected": {
            "dataset_ids": selected_ids,
            "dependent_row_counts": dependent_by_class["selected"],
            "recordings_that_would_be_deleted": recordings_by_class["selected"],
        },
        "dependent_table_map": [
            {"table": entry.table, "columns": list(entry.columns)}
            for entry in DEPENDENT_DATASET_TABLES
        ],
    }


def _selected_temp_classes(include_temp_root: Sequence[str]) -> tuple[str, ...]:
    requested = set(include_temp_root)
    if "all-temp" in requested:
        return ("pytest-tmp", "tmp", "var-tmp", "dev-shm")
    return tuple(class_name for class_name in ("pytest-tmp", "tmp", "var-tmp", "dev-shm") if class_name in requested)


def _write_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_report(report: Mapping[str, Any]) -> None:
    print(f"Registry: {report['registry']}")
    print(f"Mode: {report['mode']}")
    summary = report["summary"]
    print(
        "Candidates: "
        f"temp-root={summary['temp_root_dataset_count']} "
        f"home-review={summary['home_review_dataset_count']} "
        f"selected={summary['selected_dataset_count']}"
    )
    print("")
    print("Temp-root classes:")
    for class_name, payload in report["classes"].items():
        print(
            f"  {class_name}: "
            f"{payload['dataset_count']} dataset(s), "
            f"{payload['recordings_that_would_be_deleted']} recording(s)"
        )
        nonzero = {
            table: count
            for table, count in payload["dependent_row_counts"].items()
            if int(count) > 0
        }
        if nonzero:
            for table, count in nonzero.items():
                print(f"    {table}: {count}")
    review = report["needs_maintainer_review"]
    print("")
    print(f"NEEDS-MAINTAINER-REVIEW /home rows: {review['dataset_count']}")
    for candidate in review["datasets"]:
        print(f"  {candidate['dataset_id']}  {candidate['zarr_path']}")
    unowned = report["unowned_analysis_review"]
    print("")
    print(f"NEEDS-MAINTAINER-REVIEW unowned analysis rows: {unowned['dataset_count']}")
    for candidate in unowned["datasets"]:
        print(f"  {candidate['dataset_id']}  {candidate['zarr_path']}")


def create_backup(source: Path, backup: Path) -> None:
    backup.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source)) as source_conn:
        with sqlite3.connect(str(backup)) as backup_conn:
            source_conn.backup(backup_conn)


def _delete_from_table(conn: sqlite3.Connection, table: DependentTable, dataset_ids: Sequence[str]) -> int:
    if not dataset_ids:
        return 0
    placeholders = _placeholders(dataset_ids)
    predicates = [
        f"{_quote_ident(column)} IN ({placeholders})"
        for column in table.columns
    ]
    sql = f"DELETE FROM {_quote_ident(table.table)} WHERE {' OR '.join(predicates)};"
    params: list[str] = []
    for _column in table.columns:
        params.extend(dataset_ids)
    cursor = conn.execute(sql, params)
    return int(cursor.rowcount if cursor.rowcount is not None else 0)


def _delete_datasets(conn: sqlite3.Connection, dataset_ids: Sequence[str]) -> int:
    if not dataset_ids:
        return 0
    placeholders = _placeholders(dataset_ids)
    cursor = conn.execute(f"DELETE FROM datasets WHERE dataset_id IN ({placeholders});", list(dataset_ids))
    return int(cursor.rowcount if cursor.rowcount is not None else 0)


def _delete_recordings(conn: sqlite3.Connection, recording_ids: Sequence[str]) -> int:
    if not recording_ids:
        return 0
    placeholders = _placeholders(recording_ids)
    cursor = conn.execute(
        f"DELETE FROM recordings WHERE recording_id IN ({placeholders});",
        list(recording_ids),
    )
    return int(cursor.rowcount if cursor.rowcount is not None else 0)


def _remaining_temp_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    rows = _all_dataset_rows(conn)
    remaining = []
    for row in rows:
        path_class = classify_zarr_path(str(row["zarr_path"]))
        if path_class in {"pytest-tmp", "tmp", "var-tmp", "dev-shm"}:
            remaining.append(row)
    return remaining


def _verify_after_execute(conn: sqlite3.Connection) -> dict[str, Any]:
    integrity_rows = conn.execute("PRAGMA integrity_check;").fetchall()
    integrity = [str(row[0]) for row in integrity_rows]
    if integrity != ["ok"]:
        raise RuntimeError(f"PRAGMA integrity_check failed: {integrity}")

    fk_rows = conn.execute("PRAGMA foreign_key_check;").fetchall()
    if fk_rows:
        raise RuntimeError(f"PRAGMA foreign_key_check failed: {[tuple(row) for row in fk_rows]}")

    remaining = _remaining_temp_rows(conn)
    if remaining:
        examples = ", ".join(str(row["dataset_id"]) for row in remaining[:10])
        raise RuntimeError(f"Temp-root dataset rows remain after prune: {len(remaining)} ({examples})")

    return {
        "integrity_check": integrity,
        "foreign_key_check": [],
        "remaining_temp_root_dataset_count": 0,
    }


def execute_prune(registry: Path, backup: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    selected_ids = [str(dataset_id) for dataset_id in report["selected"]["dataset_ids"]]
    create_backup(registry, backup)
    conn = connect_writable(registry)
    try:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            assert_dependent_table_map_covers_schema(conn)
            recording_ids = _recordings_to_delete(conn, selected_ids)
            deleted_dependents = {
                entry.table: _delete_from_table(conn, entry, selected_ids)
                for entry in DEPENDENT_DATASET_TABLES
            }
            deleted_datasets = _delete_datasets(conn, selected_ids)
            deleted_recordings = _delete_recordings(conn, recording_ids)
            verification = _verify_after_execute(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()
    return {
        "backup": str(backup),
        "deleted_dataset_count": deleted_datasets,
        "deleted_recording_count": deleted_recordings,
        "deleted_dependent_row_counts": deleted_dependents,
        "verification": verification,
    }


def _validate_execute_args(args: argparse.Namespace, report: Mapping[str, Any]) -> None:
    if args.backup is None:
        raise SystemExit("--execute requires --backup <path>.")
    if "all-temp" not in set(args.include_temp_root):
        raise SystemExit("--execute requires --include-temp-root all-temp.")
    review_ids = {
        str(candidate["dataset_id"])
        for candidate in report["needs_maintainer_review"]["datasets"]
    }
    review_ids.update(
        str(candidate["dataset_id"])
        for candidate in report["unowned_analysis_review"]["datasets"]
    )
    invalid_includes = sorted(set(args.include_dataset_id) - review_ids)
    if invalid_includes:
        joined = ", ".join(invalid_includes)
        raise SystemExit(
            "--include-dataset-id is only allowed for /home or unowned-analysis "
            f"review rows: {joined}"
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run-first cleanup of registry dataset rows under temp roots.",
    )
    parser.add_argument("--registry", required=True, type=Path, help="Path to the registry SQLite file.")
    parser.add_argument("--dry-run", action="store_true", help="Preview only; this is the default.")
    parser.add_argument("--execute", action="store_true", help="Apply the selected stale-row prune.")
    parser.add_argument("--json", type=Path, help="Write machine-readable report JSON to this path.")
    parser.add_argument("--backup", type=Path, help="Required with --execute; created via sqlite3 backup API.")
    parser.add_argument(
        "--include-temp-root",
        action="append",
        choices=TEMP_ROOT_CLASS_CHOICES,
        default=[],
        help="Temp-root class to prune on execute. Use all-temp for the full stale temp-root cleanup.",
    )
    parser.add_argument(
        "--include-dataset-id",
        action="append",
        default=[],
        help=(
            "Repeatable exact /home or active unowned-analysis review dataset_id "
            "to include during execute."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute and args.dry_run:
        raise SystemExit("--dry-run and --execute are mutually exclusive.")

    execute = bool(args.execute)
    if execute:
        conn = connect_writable(args.registry)
    else:
        conn = connect_read_only(args.registry)
    try:
        report = _build_report(
            conn,
            args.registry,
            execute=execute,
            include_temp_root=args.include_temp_root,
            include_dataset_id=args.include_dataset_id,
        )
    finally:
        conn.close()

    result: dict[str, Any] | None = None
    if execute:
        _validate_execute_args(args, report)
        result = execute_prune(args.registry, args.backup, report)
        report = {**report, "execute_result": result}

    if args.json is not None:
        _write_json(args.json, report)
    _print_report(report)
    if result is not None:
        print("")
        print(f"Deleted datasets: {result['deleted_dataset_count']}")
        print(f"Backup: {result['backup']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
