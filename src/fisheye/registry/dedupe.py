"""Duplicate dataset-row analysis and repair for the Palette registry.

By default this utility does not mutate the registry. It identifies active
dataset rows that point at the same physical Zarr path, chooses a proposed
canonical row, and reports what would need to move before duplicate rows can be
removed. With ``--apply`` it merges duplicate rows into the chosen canonical
rows.

The intended workflow is:

1. run this dry-run report;
2. inspect dependent row counts and conflicts;
3. run ``--apply`` against a backed-up registry.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fisheye.registry.db import SQLITE_BUSY_TIMEOUT_MS, RegistryPaths


DATASET_REF_COLUMN_NAMES = {
    "dataset_id",
    "child_dataset_id",
    "parent_dataset_id",
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _quote_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
        raise ValueError(f"Unsafe SQLite identifier: {name!r}")
    return f'"{name}"'


def _placeholders(values: Sequence[Any]) -> str:
    if not values:
        raise ValueError("Expected at least one SQL value.")
    return ", ".join("?" for _ in values)


@dataclass(frozen=True)
class DatasetRef:
    table: str
    column: str
    pk_columns: tuple[str, ...]
    unique_indexes: tuple[tuple[str, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "column": self.column,
            "pk_columns": list(self.pk_columns),
            "unique_indexes": [list(index) for index in self.unique_indexes],
        }


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
        """
    ).fetchall()
    return [str(row["name"]) for row in rows]


def _table_columns(conn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    return conn.execute(f"PRAGMA table_info({_quote_ident(table)});").fetchall()


def _primary_key_columns(columns: Sequence[sqlite3.Row]) -> tuple[str, ...]:
    keyed = sorted(
        ((int(row["pk"]), str(row["name"])) for row in columns if int(row["pk"] or 0) > 0),
        key=lambda item: item[0],
    )
    return tuple(name for _position, name in keyed)


def _unique_indexes(conn: sqlite3.Connection, table: str) -> tuple[tuple[str, ...], ...]:
    indexes: list[tuple[str, ...]] = []
    for index_row in conn.execute(f"PRAGMA index_list({_quote_ident(table)});").fetchall():
        if int(index_row["unique"] or 0) != 1:
            continue
        index_name = str(index_row["name"])
        cols = [
            str(info["name"])
            for info in conn.execute(f"PRAGMA index_info({_quote_ident(index_name)});").fetchall()
            if info["name"] is not None
        ]
        if cols:
            indexes.append(tuple(cols))
    return tuple(indexes)


def _foreign_key_dataset_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    refs: set[str] = set()
    for row in conn.execute(f"PRAGMA foreign_key_list({_quote_ident(table)});").fetchall():
        if str(row["table"]) == "datasets" and str(row["to"]) == "dataset_id":
            refs.add(str(row["from"]))
    return refs


def discover_dataset_refs(conn: sqlite3.Connection) -> list[DatasetRef]:
    refs: list[DatasetRef] = []
    for table in _table_names(conn):
        if table == "datasets":
            continue
        columns = _table_columns(conn, table)
        column_names = {str(row["name"]) for row in columns}
        ref_columns = _foreign_key_dataset_columns(conn, table)
        ref_columns.update(column_names & DATASET_REF_COLUMN_NAMES)
        if not ref_columns:
            continue
        pk_columns = _primary_key_columns(columns)
        unique_indexes = _unique_indexes(conn, table)
        for column in sorted(ref_columns):
            refs.append(
                DatasetRef(
                    table=table,
                    column=column,
                    pk_columns=pk_columns,
                    unique_indexes=unique_indexes,
                )
            )
    return refs


def _dependent_count(conn: sqlite3.Connection, ref: DatasetRef, dataset_id: str) -> int:
    table = _quote_ident(ref.table)
    column = _quote_ident(ref.column)
    row = conn.execute(
        f"SELECT COUNT(*) AS count FROM {table} WHERE {column} = ?;",
        (str(dataset_id),),
    ).fetchone()
    return int(row["count"] if row is not None else 0)


def _total_dependent_count(conn: sqlite3.Connection, refs: Sequence[DatasetRef], dataset_id: str) -> int:
    return sum(_dependent_count(conn, ref, dataset_id) for ref in refs)


def _is_canonical_dataset_id(row: Mapping[str, Any]) -> bool:
    dataset_id = str(row.get("dataset_id") or "")
    session_uuid = str(row.get("session_uuid") or "")
    return bool(session_uuid and dataset_id.startswith(f"{session_uuid}:z"))


def _is_legacy_preferred(row: Mapping[str, Any]) -> bool:
    dataset_id = str(row.get("dataset_id") or "")
    session_uuid = str(row.get("session_uuid") or "")
    return bool(session_uuid and dataset_id == session_uuid) or (
        ":z" not in dataset_id and not dataset_id.startswith("path-")
    )


def _choose_canonical(
    conn: sqlite3.Connection,
    refs: Sequence[DatasetRef],
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    def score(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
        dep_count = _total_dependent_count(conn, refs, str(row["dataset_id"]))
        last_seen = str(row.get("last_seen_utc") or "")
        return (
            1 if _is_canonical_dataset_id(row) else 0,
            dep_count,
            last_seen,
            str(row["dataset_id"]),
        )

    return sorted(rows, key=score, reverse=True)[0]


def _candidate_duplicate_groups(
    conn: sqlite3.Connection,
    *,
    active_only: bool,
    zarr_use: str | None,
    path_contains: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    clauses = ["zarr_path IS NOT NULL", "trim(zarr_path) <> ''"]
    params: list[Any] = []
    if active_only:
        clauses.append("COALESCE(status, 'active') = 'active'")
    if zarr_use:
        clauses.append("zarr_use = ?")
        params.append(str(zarr_use))
    if path_contains:
        clauses.append("zarr_path LIKE ?")
        params.append(f"%{path_contains}%")
    where = " AND ".join(clauses)
    limit_sql = f" LIMIT {int(limit)}" if limit is not None and int(limit) > 0 else ""
    rows = conn.execute(
        f"""
        SELECT zarr_path, COALESCE(path_hash, '') AS path_hash, COUNT(*) AS row_count
        FROM datasets
        WHERE {where}
        GROUP BY zarr_path, COALESCE(path_hash, '')
        HAVING COUNT(*) > 1
        ORDER BY zarr_path
        {limit_sql};
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _load_group_rows(conn: sqlite3.Connection, zarr_path: str, path_hash: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT dataset_id, session_uuid, zarr_path, recording_id, artifact_kind,
               zarr_origin, zarr_use, source_layout, source_frame_index_path,
               source_recording_frame_index_path, source_frame_index_schema,
               path_hash, created_utc, last_seen_utc, status
        FROM datasets
        WHERE zarr_path = ?
          AND COALESCE(path_hash, '') = ?
        ORDER BY dataset_id;
        """,
        (str(zarr_path), str(path_hash or "")),
    ).fetchall()
    return [dict(row) for row in rows]


def _constraint_sets_for_ref(ref: DatasetRef) -> list[tuple[str, tuple[str, ...]]]:
    constraints: list[tuple[str, tuple[str, ...]]] = []
    if ref.column in ref.pk_columns:
        constraints.append(("primary_key", ref.pk_columns))
    for columns in ref.unique_indexes:
        if ref.pk_columns and tuple(columns) == tuple(ref.pk_columns):
            continue
        if ref.column in columns:
            constraints.append(("unique_index", columns))
    # Deduplicate while preserving order. Some SQLite autoindexes duplicate the PK.
    seen: set[tuple[str, tuple[str, ...]]] = set()
    unique: list[tuple[str, tuple[str, ...]]] = []
    for item in constraints:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _pk_conflict_count(
    conn: sqlite3.Connection,
    ref: DatasetRef,
    *,
    duplicate_id: str,
    canonical_id: str,
    constraint_columns: Sequence[str],
) -> int:
    other_columns = [column for column in constraint_columns if column != ref.column]
    table = _quote_ident(ref.table)
    ref_column = _quote_ident(ref.column)
    if not other_columns:
        duplicate_exists = conn.execute(
            f"SELECT 1 FROM {table} WHERE {ref_column} = ? LIMIT 1;",
            (str(duplicate_id),),
        ).fetchone()
        canonical_exists = conn.execute(
            f"SELECT 1 FROM {table} WHERE {ref_column} = ? LIMIT 1;",
            (str(canonical_id),),
        ).fetchone()
        return int(bool(duplicate_exists and canonical_exists))

    join_terms = " AND ".join(
        f"d.{_quote_ident(column)} IS c.{_quote_ident(column)}"
        for column in other_columns
    )
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM {table} d
        JOIN {table} c
          ON {join_terms}
        WHERE d.{ref_column} = ?
          AND c.{ref_column} = ?;
        """,
        (str(duplicate_id), str(canonical_id)),
    ).fetchone()
    return int(row["count"] if row is not None else 0)


def _self_edge_count(
    conn: sqlite3.Connection,
    ref: DatasetRef,
    *,
    duplicate_id: str,
    canonical_id: str,
) -> int:
    if ref.table != "dataset_lineage" or ref.column not in {"child_dataset_id", "parent_dataset_id"}:
        return 0
    other = "parent_dataset_id" if ref.column == "child_dataset_id" else "child_dataset_id"
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM dataset_lineage
        WHERE {_quote_ident(ref.column)} = ?
          AND {_quote_ident(other)} = ?;
        """,
        (str(duplicate_id), str(canonical_id)),
    ).fetchone()
    return int(row["count"] if row is not None else 0)


def _delete_constraint_conflicts(
    conn: sqlite3.Connection,
    ref: DatasetRef,
    *,
    duplicate_id: str,
    canonical_id: str,
    constraint_columns: Sequence[str],
) -> int:
    other_columns = [column for column in constraint_columns if column != ref.column]
    table = _quote_ident(ref.table)
    ref_column = _quote_ident(ref.column)
    if not other_columns:
        canonical_exists = conn.execute(
            f"SELECT 1 FROM {table} WHERE {ref_column} = ? LIMIT 1;",
            (str(canonical_id),),
        ).fetchone()
        if canonical_exists is None:
            return 0
        cursor = conn.execute(
            f"DELETE FROM {table} WHERE {ref_column} = ?;",
            (str(duplicate_id),),
        )
        return int(cursor.rowcount if cursor.rowcount is not None else 0)

    join_terms = " AND ".join(
        f"d.{_quote_ident(column)} IS c.{_quote_ident(column)}"
        for column in other_columns
    )
    cursor = conn.execute(
        f"""
        DELETE FROM {table}
        WHERE rowid IN (
            SELECT d.rowid
            FROM {table} d
            JOIN {table} c
              ON {join_terms}
            WHERE d.{ref_column} = ?
              AND c.{ref_column} = ?
        );
        """,
        (str(duplicate_id), str(canonical_id)),
    )
    return int(cursor.rowcount if cursor.rowcount is not None else 0)


def _delete_dataset_lineage_self_edges(
    conn: sqlite3.Connection,
    *,
    duplicate_id: str,
    canonical_id: str,
) -> int:
    cursor = conn.execute(
        """
        DELETE FROM dataset_lineage
        WHERE (child_dataset_id = ? OR parent_dataset_id = ?)
          AND child_dataset_id IN (?, ?)
          AND parent_dataset_id IN (?, ?);
        """,
        (
            str(duplicate_id),
            str(duplicate_id),
            str(duplicate_id),
            str(canonical_id),
            str(duplicate_id),
            str(canonical_id),
        ),
    )
    return int(cursor.rowcount if cursor.rowcount is not None else 0)


def _update_ref_to_canonical(
    conn: sqlite3.Connection,
    ref: DatasetRef,
    *,
    duplicate_id: str,
    canonical_id: str,
) -> int:
    table = _quote_ident(ref.table)
    column = _quote_ident(ref.column)
    cursor = conn.execute(
        f"UPDATE {table} SET {column} = ? WHERE {column} = ?;",
        (str(canonical_id), str(duplicate_id)),
    )
    return int(cursor.rowcount if cursor.rowcount is not None else 0)


def _ref_report(
    conn: sqlite3.Connection,
    ref: DatasetRef,
    *,
    duplicate_id: str,
    canonical_id: str,
) -> dict[str, Any] | None:
    row_count = _dependent_count(conn, ref, duplicate_id)
    if row_count == 0:
        return None
    conflicts = []
    for constraint_kind, columns in _constraint_sets_for_ref(ref):
        count = _pk_conflict_count(
            conn,
            ref,
            duplicate_id=duplicate_id,
            canonical_id=canonical_id,
            constraint_columns=columns,
        )
        if count:
            conflicts.append(
                {
                    "constraint": constraint_kind,
                    "columns": list(columns),
                    "conflicting_rows": int(count),
                }
            )
    self_edges = _self_edge_count(conn, ref, duplicate_id=duplicate_id, canonical_id=canonical_id)
    if self_edges:
        conflicts.append(
            {
                "constraint": "dataset_lineage_no_self_edge",
                "columns": [ref.column],
                "conflicting_rows": int(self_edges),
            }
        )
    return {
        "table": ref.table,
        "column": ref.column,
        "rows_to_repoint": int(row_count),
        "conflicts": conflicts,
    }


def _dataset_report(
    conn: sqlite3.Connection,
    refs: Sequence[DatasetRef],
    *,
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    dependent_counts = []
    total = 0
    for ref in refs:
        count = _dependent_count(conn, ref, str(dataset["dataset_id"]))
        if count:
            total += count
            dependent_counts.append({"table": ref.table, "column": ref.column, "rows": int(count)})
    return {
        **{key: _json_safe(value) for key, value in dataset.items()},
        "legacy_preferred": _is_legacy_preferred(dataset),
        "dependent_row_count": int(total),
        "dependent_counts": dependent_counts,
    }


def plan_registry_dataset_dedupe(
    registry_path: str | Path,
    *,
    active_only: bool = True,
    zarr_use: str | None = None,
    path_contains: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    registry = Path(registry_path).expanduser().resolve()
    conn = _connect(registry)
    try:
        refs = discover_dataset_refs(conn)
        groups = _candidate_duplicate_groups(
            conn,
            active_only=bool(active_only),
            zarr_use=zarr_use,
            path_contains=path_contains,
            limit=limit,
        )
        out_groups: list[dict[str, Any]] = []
        duplicate_dataset_count = 0
        conflict_group_count = 0
        total_rows_to_repoint = 0
        total_conflicting_rows = 0
        for group in groups:
            rows = _load_group_rows(conn, str(group["zarr_path"]), str(group.get("path_hash") or ""))
            if len(rows) < 2:
                continue
            canonical = _choose_canonical(conn, refs, rows)
            canonical_id = str(canonical["dataset_id"])
            duplicates = [row for row in rows if str(row["dataset_id"]) != canonical_id]
            duplicate_dataset_count += len(duplicates)
            duplicate_reports = []
            group_conflict_rows = 0
            group_rows_to_repoint = 0
            for duplicate in duplicates:
                duplicate_id = str(duplicate["dataset_id"])
                ref_reports = []
                for ref in refs:
                    report = _ref_report(conn, ref, duplicate_id=duplicate_id, canonical_id=canonical_id)
                    if report is None:
                        continue
                    ref_reports.append(report)
                    group_rows_to_repoint += int(report["rows_to_repoint"])
                    for conflict in report["conflicts"]:
                        group_conflict_rows += int(conflict["conflicting_rows"])
                duplicate_reports.append(
                    {
                        "dataset": _dataset_report(conn, refs, dataset=duplicate),
                        "reference_updates": ref_reports,
                        "rows_to_repoint": int(
                            sum(int(item["rows_to_repoint"]) for item in ref_reports)
                        ),
                        "conflicting_rows": int(
                            sum(
                                int(conflict["conflicting_rows"])
                                for item in ref_reports
                                for conflict in item["conflicts"]
                            )
                        ),
                    }
                )
            if group_conflict_rows:
                conflict_group_count += 1
            total_rows_to_repoint += group_rows_to_repoint
            total_conflicting_rows += group_conflict_rows
            out_groups.append(
                {
                    "zarr_path": str(group["zarr_path"]),
                    "path_hash": str(group.get("path_hash") or ""),
                    "row_count": int(len(rows)),
                    "canonical_dataset_id": canonical_id,
                    "canonical": _dataset_report(conn, refs, dataset=canonical),
                    "duplicates": duplicate_reports,
                    "rows_to_repoint": int(group_rows_to_repoint),
                    "conflicting_rows": int(group_conflict_rows),
                    "safe_to_auto_apply": bool(group_rows_to_repoint >= 0 and group_conflict_rows == 0),
                }
            )
        status = "ok"
        if total_conflicting_rows:
            status = "conflicts"
        return {
            "schema_version": "palette.registry_dataset_dedupe_plan.v1",
            "status": status,
            "dry_run": True,
            "registry_path": str(registry),
            "active_only": bool(active_only),
            "zarr_use": zarr_use,
            "path_contains": path_contains,
            "limit": limit,
            "reference_columns": [ref.as_dict() for ref in refs],
            "duplicate_group_count": int(len(out_groups)),
            "duplicate_dataset_count": int(duplicate_dataset_count),
            "conflict_group_count": int(conflict_group_count),
            "rows_to_repoint": int(total_rows_to_repoint),
            "conflicting_rows": int(total_conflicting_rows),
            "groups": out_groups,
        }
    finally:
        conn.close()


def apply_registry_dataset_dedupe(
    registry_path: str | Path,
    *,
    active_only: bool = True,
    zarr_use: str | None = None,
    path_contains: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    registry = Path(registry_path).expanduser().resolve()
    plan = plan_registry_dataset_dedupe(
        registry,
        active_only=active_only,
        zarr_use=zarr_use,
        path_contains=path_contains,
        limit=limit,
    )
    conn = _connect(registry)
    try:
        refs = discover_dataset_refs(conn)
        ref_by_key = {(ref.table, ref.column): ref for ref in refs}
        rows_repointed = 0
        conflict_rows_deleted = 0
        self_edge_rows_deleted = 0
        dataset_rows_deleted = 0

        with conn:
            for group in plan["groups"]:
                canonical_id = str(group["canonical_dataset_id"])
                for duplicate_report in group["duplicates"]:
                    duplicate_id = str(duplicate_report["dataset"]["dataset_id"])
                    if duplicate_id == canonical_id:
                        continue
                    if any(
                        key in ref_by_key
                        for key in (
                            ("dataset_lineage", "child_dataset_id"),
                            ("dataset_lineage", "parent_dataset_id"),
                        )
                    ):
                        self_edge_rows_deleted += _delete_dataset_lineage_self_edges(
                            conn,
                            duplicate_id=duplicate_id,
                            canonical_id=canonical_id,
                        )
                    for ref_report in duplicate_report["reference_updates"]:
                        ref = ref_by_key.get((str(ref_report["table"]), str(ref_report["column"])))
                        if ref is None:
                            continue
                        for _constraint_kind, columns in _constraint_sets_for_ref(ref):
                            conflict_rows_deleted += _delete_constraint_conflicts(
                                conn,
                                ref,
                                duplicate_id=duplicate_id,
                                canonical_id=canonical_id,
                                constraint_columns=columns,
                            )
                        rows_repointed += _update_ref_to_canonical(
                            conn,
                            ref,
                            duplicate_id=duplicate_id,
                            canonical_id=canonical_id,
                        )
                    cursor = conn.execute(
                        "DELETE FROM datasets WHERE dataset_id = ?;",
                        (duplicate_id,),
                    )
                    dataset_rows_deleted += int(cursor.rowcount if cursor.rowcount is not None else 0)

        remaining_groups = _candidate_duplicate_groups(
            conn,
            active_only=active_only,
            zarr_use=zarr_use,
            path_contains=path_contains,
            limit=limit,
        )
        return {
            "schema_version": "palette.registry_dataset_dedupe_apply.v1",
            "status": "ok" if not remaining_groups else "remaining_duplicates",
            "dry_run": False,
            "registry_path": str(registry),
            "active_only": bool(active_only),
            "zarr_use": zarr_use,
            "path_contains": path_contains,
            "limit": limit,
            "planned_duplicate_group_count": int(plan["duplicate_group_count"]),
            "planned_duplicate_dataset_count": int(plan["duplicate_dataset_count"]),
            "planned_rows_to_repoint": int(plan["rows_to_repoint"]),
            "planned_conflicting_rows": int(plan["conflicting_rows"]),
            "rows_repointed": int(rows_repointed),
            "conflict_rows_deleted": int(conflict_rows_deleted),
            "self_edge_rows_deleted": int(self_edge_rows_deleted),
            "dataset_rows_deleted": int(dataset_rows_deleted),
            "remaining_duplicate_group_count": int(len(remaining_groups)),
        }
    finally:
        conn.close()


def _default_registry_path() -> Path:
    return RegistryPaths.from_env(Path.cwd()).path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to palette_registry.sqlite. Defaults to PALETTE_REGISTRY_PATH/config/default.",
    )
    parser.add_argument("--zarr-use", help="Only analyze duplicate rows with this zarr_use.")
    parser.add_argument("--path-contains", help="Only analyze duplicate paths containing this substring.")
    parser.add_argument("--limit", type=int, help="Maximum duplicate path groups to report.")
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include non-active dataset rows. Default analyzes active rows only.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Merge duplicate active dataset rows into the chosen canonical rows. "
            "Back up the registry before using this."
        ),
    )
    return parser


def _print_summary(report: Mapping[str, Any]) -> None:
    print(f"status: {report['status']}")
    print(f"registry_path: {report['registry_path']}")
    if not report.get("dry_run", True):
        print(f"planned_duplicate_group_count: {report['planned_duplicate_group_count']}")
        print(f"planned_duplicate_dataset_count: {report['planned_duplicate_dataset_count']}")
        print(f"planned_rows_to_repoint: {report['planned_rows_to_repoint']}")
        print(f"planned_conflicting_rows: {report['planned_conflicting_rows']}")
        print(f"rows_repointed: {report['rows_repointed']}")
        print(f"conflict_rows_deleted: {report['conflict_rows_deleted']}")
        print(f"self_edge_rows_deleted: {report['self_edge_rows_deleted']}")
        print(f"dataset_rows_deleted: {report['dataset_rows_deleted']}")
        print(f"remaining_duplicate_group_count: {report['remaining_duplicate_group_count']}")
        return
    print(f"duplicate_group_count: {report['duplicate_group_count']}")
    print(f"duplicate_dataset_count: {report['duplicate_dataset_count']}")
    print(f"rows_to_repoint: {report['rows_to_repoint']}")
    print(f"conflict_group_count: {report['conflict_group_count']}")
    print(f"conflicting_rows: {report['conflicting_rows']}")
    for group in report["groups"]:
        print(
            f"- {group['zarr_path']}: canonical={group['canonical_dataset_id']} "
            f"duplicates={len(group['duplicates'])} rows_to_repoint={group['rows_to_repoint']} "
            f"conflicting_rows={group['conflicting_rows']}"
        )


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    registry = args.registry.expanduser() if args.registry is not None else _default_registry_path()
    kwargs = {
        "active_only": not bool(args.include_inactive),
        "zarr_use": args.zarr_use,
        "path_contains": args.path_contains,
        "limit": args.limit,
    }
    if args.apply:
        report = apply_registry_dataset_dedupe(registry, **kwargs)
    else:
        report = plan_registry_dataset_dedupe(registry, **kwargs)
    text = json.dumps(_json_safe(report), indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
    else:
        _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
