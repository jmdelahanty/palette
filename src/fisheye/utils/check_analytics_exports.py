"""Inventory and validate indexed Palette analytics exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from fisheye.registry.db import Registry, RegistryPaths


def _json_list(value: Any) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _query_rows(registry: Registry, args: argparse.Namespace) -> list[dict[str, Any]]:
    params: list[Any] = []
    sql = [
        """
        SELECT
            ae.export_run_id,
            ae.status,
            ae.collection_id,
            ae.collection_manifest_sha256,
            ae.collection_name,
            ae.export_manifest_path,
            ae.output_root,
            ae.created_at_utc,
            ae.source_recording_count,
            ae.table_count,
            ae.diagnostics_count,
            aet.table_name,
            aet.table_path,
            aet.row_count,
            aet.part_count,
            aet.part_files_json
        FROM analytics_export_overview ae
        JOIN analytics_export_tables aet
          ON aet.export_run_id = ae.export_run_id
        WHERE 1=1
        """
    ]
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
    sql.append(
        "ORDER BY COALESCE(ae.created_at_utc, ae.export_run_id) DESC, "
        "ae.export_run_id DESC, aet.table_name ASC"
    )
    if args.limit and int(args.limit) > 0:
        sql.append("LIMIT ?")
        params.append(int(args.limit))
    return [dict(row) for row in registry.conn.execute("\n".join(sql), params).fetchall()]


def _check_row_files(row: dict[str, Any]) -> dict[str, Any]:
    table_path_raw = row.get("table_path")
    part_files = _json_list(row.get("part_files_json"))
    row["part_files"] = part_files
    row["listed_part_count"] = len(part_files)

    if not table_path_raw:
        row.update(
            {
                "check_status": "missing_table_path",
                "table_dir_exists": False,
                "actual_parquet_count": 0,
                "missing_part_count": len(part_files),
                "missing_part_files": part_files,
                "unlisted_part_count": 0,
                "unlisted_part_files": [],
            }
        )
        return row

    table_path = Path(str(table_path_raw))
    table_dir_exists = table_path.is_dir()
    actual_parts = tuple(sorted(table_path.glob("*.parquet"))) if table_dir_exists else ()
    actual_parquet_count = len(actual_parts)
    missing_part_files = [path for path in part_files if not Path(path).is_file()]
    listed_resolved = {Path(path).resolve() for path in part_files}
    unlisted_part_files = [
        str(path)
        for path in actual_parts
        if path.resolve() not in listed_resolved
    ]
    expected_part_count = row.get("part_count")
    count_mismatch = (
        expected_part_count is not None
        and part_files
        and len(part_files) != int(expected_part_count)
    )
    actual_count_mismatch = (
        expected_part_count is not None
        and not part_files
        and actual_parquet_count != int(expected_part_count)
    )

    if not table_dir_exists:
        check_status = "missing_table_dir"
    elif missing_part_files:
        check_status = "missing_part_files"
    elif unlisted_part_files:
        check_status = "unlisted_part_files"
    elif count_mismatch or actual_count_mismatch:
        check_status = "part_count_mismatch"
    else:
        check_status = "ok"

    row.update(
        {
            "check_status": check_status,
            "table_dir_exists": table_dir_exists,
            "actual_parquet_count": actual_parquet_count,
            "missing_part_count": len(missing_part_files),
            "missing_part_files": missing_part_files,
            "unlisted_part_count": len(unlisted_part_files),
            "unlisted_part_files": unlisted_part_files,
        }
    )
    return row


def _annotate_rows(rows: list[dict[str, Any]], *, check_files: bool) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        if check_files:
            annotated.append(_check_row_files(row))
        else:
            row["part_files"] = _json_list(row.get("part_files_json"))
            row["listed_part_count"] = len(row["part_files"])
            row["check_status"] = "not_checked"
            row["table_dir_exists"] = None
            row["actual_parquet_count"] = None
            row["missing_part_count"] = None
            row["missing_part_files"] = []
            row["unlisted_part_count"] = None
            row["unlisted_part_files"] = []
            annotated.append(row)
    return annotated


def _print_table(rows: Iterable[dict[str, Any]]) -> None:
    headers = [
        "check_status",
        "collection_id",
        "export_run_id",
        "table_name",
        "row_count",
        "part_count",
        "actual_parquet_count",
        "missing_part_count",
        "table_path",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join("" if row.get(key) is None else str(row.get(key)) for key in headers))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory indexed Palette analytics exports and optionally validate files.",
    )
    parser.add_argument("--registry", type=Path, help="Palette registry SQLite path.")
    parser.add_argument("--collection-id", help="Filter by collection id.")
    parser.add_argument("--collection-manifest-sha256", help="Filter by collection manifest SHA-256.")
    parser.add_argument("--export-run-id", help="Filter by export run id.")
    parser.add_argument("--table", help="Filter by analytics table name.")
    parser.add_argument("--status", default="active", help="Registry export status filter; use 'any' to disable.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum rows to list; <=0 disables limiting.")
    parser.add_argument("--check-files", action="store_true", help="Verify table directories and indexed Parquet parts exist.")
    parser.add_argument("--format", choices=("table", "json"), default="table")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
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

    rows = _annotate_rows(rows, check_files=bool(args.check_files))
    if args.format == "json":
        print(json.dumps(rows, sort_keys=True))
    else:
        _print_table(rows)

    if args.check_files and any(row["check_status"] != "ok" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
