"""Query indexed analytics exports from the Palette registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from fisheye.registry.db import Registry, RegistryPaths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List Palette analytics exports indexed in the registry.",
    )
    parser.add_argument("--registry", type=Path, help="Palette registry SQLite path.")
    parser.add_argument("--collection-id", help="Filter by collection_id.")
    parser.add_argument("--collection-manifest-sha256", help="Filter by collection manifest SHA-256.")
    parser.add_argument("--export-run-id", help="Filter by export_run_id.")
    parser.add_argument("--table", help="Require/export a specific table name.")
    parser.add_argument("--status", default="active", help="Filter by export status; use 'any' to disable.")
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--limit", type=int, default=100)
    return parser


def _query_rows(registry: Registry, args: argparse.Namespace) -> list[dict[str, Any]]:
    params: list[Any] = []
    select_cols = [
        "ae.export_run_id",
        "ae.status",
        "ae.collection_id",
        "ae.collection_manifest_sha256",
        "ae.collection_name",
        "ae.export_manifest_path",
        "ae.output_root",
        "ae.created_at_utc",
        "ae.source_recording_count",
        "ae.table_count",
        "ae.diagnostics_count",
        "ae.row_counts_json",
        "ae.tables_json",
    ]
    sql: list[str] = [
        f"SELECT {', '.join(select_cols)}",
        "FROM analytics_export_overview ae",
    ]
    if args.table:
        sql.append("JOIN analytics_export_tables aet ON aet.export_run_id = ae.export_run_id")
    sql.append("WHERE 1=1")

    if args.status != "any":
        sql.append("AND ae.status = ?")
        params.append(args.status)
    if args.collection_id:
        sql.append("AND ae.collection_id = ?")
        params.append(args.collection_id)
    if args.collection_manifest_sha256:
        sql.append("AND ae.collection_manifest_sha256 = ?")
        params.append(args.collection_manifest_sha256)
    if args.export_run_id:
        sql.append("AND ae.export_run_id = ?")
        params.append(args.export_run_id)
    if args.table:
        sql.append("AND aet.table_name = ?")
        params.append(args.table)

    sql.append("ORDER BY COALESCE(ae.created_at_utc, ae.indexed_utc) DESC, ae.export_run_id DESC")
    if args.limit and int(args.limit) > 0:
        sql.append("LIMIT ?")
        params.append(int(args.limit))

    rows = registry.conn.execute("\n".join(sql), params).fetchall()
    return [dict(row) for row in rows]


def _print_table(rows: Iterable[dict[str, Any]]) -> None:
    headers = [
        "export_run_id",
        "status",
        "collection_id",
        "source_recording_count",
        "table_count",
        "diagnostics_count",
        "export_manifest_path",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join("" if row.get(key) is None else str(row.get(key)) for key in headers))


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    registry_path = (
        args.registry.expanduser().resolve()
        if args.registry is not None
        else RegistryPaths.from_env(Path.cwd()).path
    )
    registry = Registry(registry_path)
    try:
        rows = _query_rows(registry, args)
    finally:
        registry.close()

    if args.format == "json":
        print(json.dumps(rows, sort_keys=True))
    else:
        _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
