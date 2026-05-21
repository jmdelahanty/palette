"""Report shadow-mode stage-array validation telemetry from the registry."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Optional, Sequence

from fisheye.registry.db import RegistryPaths

DEFAULT_VALIDATION_STATUSES = ("invalid",)
KNOWN_VALIDATION_STATUSES = ("ok", "invalid", "no_spec")


def _parse_details_json(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    payload: object = value
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", "ignore")
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {"_parse_error": "invalid_json", "_raw": text}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _registry_path_from_arg(value: Optional[str]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()


def _object_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE name = ? AND type IN ('table', 'view')
        LIMIT 1;
        """,
        (name,),
    ).fetchone()
    return row is not None


def _source_relation(conn: sqlite3.Connection) -> str:
    if _object_exists(conn, "recording_step_status_latest"):
        return "recording_step_status_latest"
    if _object_exists(conn, "recording_step_status"):
        return "recording_step_status"
    raise RuntimeError("registry is missing recording_step_status_latest/recording_step_status")


def _relation_columns(conn: sqlite3.Connection, relation: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({relation});").fetchall()
    return {str(row["name"]) for row in rows}


def _select_expr(columns: set[str], name: str) -> str:
    if name in columns:
        return name
    return f"NULL AS {name}"


def _iter_status_rows(
    conn: sqlite3.Connection,
    *,
    relation: str,
    step_name: Optional[str],
) -> Iterable[sqlite3.Row]:
    columns = _relation_columns(conn, relation)
    select_cols = [
        _select_expr(columns, "dataset_id"),
        _select_expr(columns, "recording_id"),
        _select_expr(columns, "step_name"),
        _select_expr(columns, "status"),
        _select_expr(columns, "run_name"),
        _select_expr(columns, "method"),
        _select_expr(columns, "source"),
        _select_expr(columns, "updated_utc"),
        _select_expr(columns, "details_json"),
    ]
    sql = f"SELECT {', '.join(select_cols)} FROM {relation}"
    params: list[object] = []
    if step_name:
        sql += " WHERE step_name = ?"
        params.append(step_name)
    sql += " ORDER BY step_name, dataset_id, run_name"
    yield from conn.execute(sql, params)


def build_shadow_validation_report(
    registry_path: Path,
    *,
    validation_statuses: Sequence[str] = DEFAULT_VALIDATION_STATUSES,
    step_name: Optional[str] = None,
    limit: int = 50,
) -> dict[str, Any]:
    statuses = tuple(dict.fromkeys(str(status).strip() for status in validation_statuses if str(status).strip()))
    status_set = set(statuses)
    if not statuses:
        raise ValueError("At least one validation status must be requested.")

    conn = sqlite3.connect(str(registry_path))
    conn.row_factory = sqlite3.Row
    try:
        relation = _source_relation(conn)
        rows: list[dict[str, Any]] = []
        all_status_counts: Counter[str] = Counter()
        matched_status_counts: Counter[str] = Counter()
        matched_stage_counts: Counter[str] = Counter()

        for row in _iter_status_rows(conn, relation=relation, step_name=step_name):
            details = _parse_details_json(row["details_json"])
            validation_status = details.get("stage_array_validation_status")
            if not isinstance(validation_status, str):
                continue
            validation_status = validation_status.strip()
            if not validation_status:
                continue
            all_status_counts[validation_status] += 1
            if validation_status not in status_set:
                continue

            step = str(row["step_name"] or "")
            matched_status_counts[validation_status] += 1
            matched_stage_counts[step] += 1
            if len(rows) < limit:
                rows.append(
                    {
                        "dataset_id": row["dataset_id"],
                        "recording_id": row["recording_id"],
                        "step_name": step,
                        "status": row["status"],
                        "run_name": row["run_name"],
                        "method": row["method"],
                        "source": row["source"],
                        "updated_utc": row["updated_utc"],
                        "stage_array_validation_status": validation_status,
                        "stage_array_validation_stage": details.get("stage_array_validation_stage"),
                        "stage_array_validation_enforced": details.get("stage_array_validation_enforced"),
                        "stage_array_validation_errors": details.get("stage_array_validation_errors") or [],
                        "stage_array_validation_warnings": details.get("stage_array_validation_warnings") or [],
                    }
                )

        return {
            "schema_version": "palette.stage_array_validation_shadow_report.v1",
            "registry_path": str(registry_path),
            "source_relation": relation,
            "validation_status_filter": list(statuses),
            "step_name_filter": step_name,
            "matched_row_count": int(sum(matched_status_counts.values())),
            "returned_row_count": len(rows),
            "all_validation_status_counts": dict(sorted(all_status_counts.items())),
            "matched_validation_status_counts": dict(sorted(matched_status_counts.items())),
            "matched_stage_counts": dict(sorted(matched_stage_counts.items())),
            "rows": rows,
        }
    finally:
        conn.close()


def _format_row(row: Mapping[str, Any]) -> str:
    errors = row.get("stage_array_validation_errors")
    warnings = row.get("stage_array_validation_warnings")
    reason = ""
    if isinstance(errors, list) and errors:
        reason = f" errors={'; '.join(str(item) for item in errors[:2])}"
    elif isinstance(warnings, list) and warnings:
        reason = f" warnings={'; '.join(str(item) for item in warnings[:2])}"
    return (
        f"  {row.get('stage_array_validation_status', '-'):<8} "
        f"{row.get('step_name', '-'):<24} "
        f"dataset={row.get('dataset_id') or '-'} "
        f"run={row.get('run_name') or '-'}"
        f"{reason}"
    )


def print_text_report(report: Mapping[str, Any]) -> None:
    print("stage_array_validation_shadow_report")
    print(f"registry: {report['registry_path']}")
    print(f"source_relation: {report['source_relation']}")
    print(f"validation_status_filter: {report['validation_status_filter']}")
    if report.get("step_name_filter"):
        print(f"step_name_filter: {report['step_name_filter']}")
    print(f"matched_row_count: {report['matched_row_count']}")
    print(f"all_validation_status_counts: {report['all_validation_status_counts']}")
    print(f"matched_validation_status_counts: {report['matched_validation_status_counts']}")
    print(f"matched_stage_counts: {report['matched_stage_counts']}")
    rows = report.get("rows")
    if isinstance(rows, list) and rows:
        print()
        print("sample_rows:")
        for row in rows:
            if isinstance(row, Mapping):
                print(_format_row(row))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report stage-array validation telemetry recorded while validation "
            "is running in shadow-mode."
        )
    )
    parser.add_argument(
        "--registry",
        help="Registry SQLite path. Defaults to RegistryPaths.from_env(Path.cwd()).",
    )
    parser.add_argument("--step-name", help="Optional recording step filter.")
    parser.add_argument(
        "--validation-status",
        action="append",
        choices=KNOWN_VALIDATION_STATUSES,
        help="Validation status to include. Repeatable. Default: invalid.",
    )
    parser.add_argument(
        "--include-no-spec",
        action="store_true",
        help="Also include rows where no StageSpec existed for the step.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include ok, invalid, and no_spec validation rows.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Maximum sample rows to print/include.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    parser.add_argument("--output-json", help="Optional path to write the JSON report.")
    parser.add_argument(
        "--fail-on-match",
        action="store_true",
        help="Exit nonzero when the selected validation status filter matches any rows.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.all:
        statuses = KNOWN_VALIDATION_STATUSES
    else:
        statuses = tuple(args.validation_status or DEFAULT_VALIDATION_STATUSES)
        if args.include_no_spec and "no_spec" not in statuses:
            statuses = (*statuses, "no_spec")

    report = build_shadow_validation_report(
        _registry_path_from_arg(args.registry),
        validation_statuses=statuses,
        step_name=args.step_name,
        limit=max(int(args.limit), 0),
    )

    if args.output_json:
        Path(args.output_json).expanduser().resolve().write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)
    return 1 if args.fail_on_match and int(report["matched_row_count"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
