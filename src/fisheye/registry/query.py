"""Query CLI for dataset curation and registry inspection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .db import Registry, RegistryPaths


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the Palette registry for curated dataset selection.",
    )
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table", help="Output format.")
    parser.add_argument("--output", type=Path, help="Optional output file for json/csv formats.")
    parser.add_argument("--output-file-list", type=Path, help="Write matched zarr paths to file.")
    parser.add_argument("--list-ids", action="store_true", help="Print only dataset IDs (one per line).")

    parser.add_argument("--dataset-id", type=str, help="Exact dataset_id match.")
    parser.add_argument("--status", choices=["active", "missing"], help="Filter dataset status.")
    parser.add_argument("--path-contains", type=str, help="Substring match for zarr_path.")

    parser.add_argument("--dish-id", type=str, help="Exact dish_id match.")
    parser.add_argument("--cross-id", type=str, help="Exact cross_id match.")
    parser.add_argument(
        "--fish-id",
        type=str,
        help="Exact subject_id match; also matches legacy fish_id compatibility aliases.",
    )
    parser.add_argument("--strain", type=str, help="Substring match for line_strain.")
    parser.add_argument("--protocol", type=str, help="Exact protocol_name match.")
    parser.add_argument("--protocol-hash", type=str, help="Exact protocol_hash match.")

    parser.add_argument("--dpf", type=int, help="Exact dpf_at_acquisition match.")
    parser.add_argument("--dpf-min", type=int, help="Minimum dpf_at_acquisition.")
    parser.add_argument("--dpf-max", type=int, help="Maximum dpf_at_acquisition.")
    parser.add_argument(
        "--since",
        type=str,
        help="Only datasets created on or after this date (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--provenance",
        choices=["any", "complete", "partial", "missing"],
        default="any",
        help="Provenance completeness filter.",
    )
    parser.add_argument(
        "--missing",
        action="store_true",
        help="Legacy alias for --provenance missing.",
    )
    parser.add_argument("--rig-id", type=str, help="Exact rig_id match.")
    parser.add_argument("--arena-id", type=str, help="Exact arena_id match.")
    parser.add_argument("--camera-id", type=str, help="Exact camera_id match.")
    parser.add_argument("--canvas-name", type=str, help="Exact canvas_name match.")
    parser.add_argument(
        "--model-input",
        choices=["gray", "rgb"],
        help="Filter by required downsample modality for training input.",
    )
    parser.add_argument("--require-context", action="store_true", help="Require full context fields.")
    parser.add_argument("--missing-context", action="store_true", help="Require any missing context field.")

    parser.add_argument("--include-training", action="store_true", help="Include latest training run metadata.")
    parser.add_argument("--trained-only", action="store_true", help="Only datasets with at least one training run.")
    parser.add_argument("--set-id", type=str, help="Only datasets that belong to this training set ID.")

    parser.add_argument("--where", type=str, help="Raw SQL WHERE clause suffix (advanced).")
    parser.add_argument("--limit", type=int, default=200, help="Maximum rows to return (0 = no limit).")
    parser.add_argument("--offset", type=int, default=0, help="Offset rows.")
    return parser.parse_args(argv)


def _build_query(args: argparse.Namespace) -> Tuple[str, List[Any]]:
    needs_training_join = bool(args.include_training or args.trained_only or args.set_id)
    sql: List[str] = []
    params: List[Any] = []

    if needs_training_join:
        sql.extend(
            [
                "WITH set_members AS (",
                "  SELECT ts.set_id, ts.name AS set_name, je.value AS dataset_id",
                "  FROM training_sets ts, json_each(COALESCE(ts.dataset_ids_json, '[]')) je",
                "),",
                "latest_runs AS (",
                "  SELECT",
                "    sm.dataset_id,",
                "    tr.run_id,",
                "    tr.set_id AS run_set_id,",
                "    tr.created_utc AS run_created_utc,",
                "    tr.model_path AS run_model_path,",
                "    tr.metrics_path AS run_metrics_path,",
                "    tr.manifest_path AS run_manifest_path,",
                "    ROW_NUMBER() OVER (PARTITION BY sm.dataset_id ORDER BY tr.created_utc DESC, tr.run_id DESC) AS rn",
                "  FROM set_members sm",
                "  JOIN training_runs tr ON tr.set_id = sm.set_id",
                ")",
            ]
        )

    select_cols = [
        "d.dataset_id",
        "d.session_uuid",
        "d.zarr_path",
        "d.status",
        "dcc.dish_id",
        "dcc.cross_id",
        "COALESCE(dcc.subject_id, dcc.legacy_fish_id) AS fish_id",
        "dcc.line_strain",
        "dcc.genotype",
        "dcc.dpf_at_acquisition",
        "dcc.protocol_name",
        "dcc.protocol_hash",
        "dcc.snapshot_status",
        "dcc.rig_id",
        "dcc.arena_id",
        "dcc.camera_id",
        "dcc.canvas_name",
    ]
    if needs_training_join:
        select_cols.extend(
            [
                "sm.set_id AS training_set_id",
                "sm.set_name AS training_set_name",
                "lr.run_id AS latest_run_id",
                "lr.run_created_utc AS latest_run_created_utc",
                "lr.run_model_path AS latest_run_model_path",
                "lr.run_metrics_path AS latest_run_metrics_path",
                "lr.run_manifest_path AS latest_run_manifest_path",
            ]
        )

    sql.append(f"SELECT {', '.join(select_cols)}")
    sql.append("FROM datasets d")
    sql.append("LEFT JOIN dataset_context_current dcc ON d.dataset_id = dcc.dataset_id")
    sql.append("LEFT JOIN provenance p ON d.dataset_id = p.dataset_id")
    if needs_training_join:
        sql.append("LEFT JOIN set_members sm ON sm.dataset_id = d.dataset_id")
        sql.append("LEFT JOIN latest_runs lr ON lr.dataset_id = d.dataset_id AND lr.rn = 1")

    sql.append("WHERE 1=1")

    def add_clause(clause: str, value: Any) -> None:
        sql.append(clause)
        params.append(value)

    def add_exact_or_json_membership(
        *,
        scalar_column: str,
        json_column: str,
        value: Any,
        cast_type: str = "TEXT",
    ) -> None:
        sql.append(
            f"AND ({scalar_column} = ? OR EXISTS ("
            f"SELECT 1 FROM json_each(COALESCE({json_column}, '[]')) "
            f"WHERE CAST(json_each.value AS {cast_type}) = ?"
            "))"
        )
        params.extend([value, value])

    def add_like_or_json_membership(*, scalar_column: str, json_column: str, pattern: str) -> None:
        sql.append(
            f"AND ({scalar_column} LIKE ? OR EXISTS ("
            f"SELECT 1 FROM json_each(COALESCE({json_column}, '[]')) "
            f"WHERE CAST(json_each.value AS TEXT) LIKE ?"
            "))"
        )
        params.extend([pattern, pattern])

    def add_numeric_or_json_range(
        *,
        scalar_column: str,
        json_column: str,
        operator: str,
        value: Any,
    ) -> None:
        sql.append(
            f"AND ({scalar_column} {operator} ? OR EXISTS ("
            f"SELECT 1 FROM json_each(COALESCE({json_column}, '[]')) "
            f"WHERE CAST(json_each.value AS INTEGER) {operator} ?"
            "))"
        )
        params.extend([value, value])

    if args.dataset_id:
        add_clause("AND d.dataset_id = ?", args.dataset_id)
    if args.status:
        add_clause("AND d.status = ?", args.status)
    if args.path_contains:
        add_clause("AND d.zarr_path LIKE ?", f"%{args.path_contains}%")

    if args.dish_id:
        add_exact_or_json_membership(
            scalar_column="dcc.dish_id",
            json_column="dcc.dish_ids_json",
            value=args.dish_id,
        )
    if args.cross_id:
        add_exact_or_json_membership(
            scalar_column="dcc.cross_id",
            json_column="dcc.cross_ids_json",
            value=args.cross_id,
        )
    if args.fish_id:
        add_exact_or_json_membership(
            scalar_column="COALESCE(dcc.subject_id, dcc.legacy_fish_id)",
            json_column="dcc.subject_ids_json",
            value=args.fish_id,
        )
    if args.strain:
        add_like_or_json_membership(
            scalar_column="dcc.line_strain",
            json_column="dcc.line_strains_json",
            pattern=f"%{args.strain}%",
        )
    if args.protocol:
        add_clause("AND dcc.protocol_name = ?", args.protocol)
    if args.protocol_hash:
        add_clause("AND dcc.protocol_hash = ?", args.protocol_hash)

    if args.dpf is not None:
        add_exact_or_json_membership(
            scalar_column="dcc.dpf_at_acquisition",
            json_column="dcc.dpf_values_json",
            value=args.dpf,
            cast_type="INTEGER",
        )
    if args.dpf_min is not None:
        add_numeric_or_json_range(
            scalar_column="dcc.dpf_at_acquisition",
            json_column="dcc.dpf_values_json",
            operator=">=",
            value=args.dpf_min,
        )
    if args.dpf_max is not None:
        add_numeric_or_json_range(
            scalar_column="dcc.dpf_at_acquisition",
            json_column="dcc.dpf_values_json",
            operator="<=",
            value=args.dpf_max,
        )
    if args.since:
        add_clause("AND d.created_utc >= ?", args.since)

    provenance_mode = "missing" if args.missing else args.provenance
    if provenance_mode == "complete":
        sql.append("AND dcc.snapshot_status = 'complete'")
        sql.append("AND dcc.protocol_hash IS NOT NULL AND dcc.protocol_hash != ''")
        sql.append("AND dcc.dpf_at_acquisition IS NOT NULL")
    elif provenance_mode == "partial":
        sql.append("AND dcc.snapshot_status = 'partial'")
    elif provenance_mode == "missing":
        sql.append(
            "AND (p.dataset_id IS NULL OR dcc.snapshot_status IS NULL OR dcc.snapshot_status = '' "
            "OR dcc.protocol_hash IS NULL OR dcc.protocol_hash = '' OR dcc.dpf_at_acquisition IS NULL)"
        )

    if args.rig_id:
        add_clause("AND dcc.rig_id = ?", args.rig_id)
    if args.arena_id:
        add_clause("AND dcc.arena_id = ?", args.arena_id)
    if args.camera_id:
        add_clause("AND dcc.camera_id = ?", args.camera_id)
    if args.canvas_name:
        add_clause("AND dcc.canvas_name = ?", args.canvas_name)
    if args.model_input == "gray":
        sql.append(
            "AND (COALESCE(dcc.has_images_ds, 0) = 1 OR dcc.downsample_formats_json LIKE '%\"gray\"%')"
        )
    elif args.model_input == "rgb":
        sql.append(
            "AND (COALESCE(dcc.has_images_ds_rgb, 0) = 1 OR dcc.downsample_formats_json LIKE '%\"rgb\"%')"
        )

    if args.require_context:
        sql.append(
            "AND dcc.rig_id IS NOT NULL AND dcc.rig_id != '' "
            "AND dcc.arena_id IS NOT NULL AND dcc.arena_id != '' "
            "AND dcc.camera_id IS NOT NULL AND dcc.camera_id != '' "
            "AND dcc.canvas_name IS NOT NULL AND dcc.canvas_name != ''"
        )
    if args.missing_context:
        sql.append(
            "AND (dcc.rig_id IS NULL OR dcc.rig_id = '' "
            "OR dcc.arena_id IS NULL OR dcc.arena_id = '' "
            "OR dcc.camera_id IS NULL OR dcc.camera_id = '' "
            "OR dcc.canvas_name IS NULL OR dcc.canvas_name = '')"
        )

    if args.set_id:
        add_clause("AND sm.set_id = ?", args.set_id)
    if args.trained_only:
        sql.append("AND lr.run_id IS NOT NULL")

    if args.where:
        sql.append(f"AND ({args.where})")

    sql.append("ORDER BY d.dataset_id")
    if args.limit and args.limit > 0:
        add_clause("LIMIT ?", int(args.limit))
    if args.offset and args.offset > 0:
        add_clause("OFFSET ?", int(args.offset))

    return " ".join(sql), params


def _rows_to_dicts(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def _print_table(rows: List[Dict[str, Any]], include_training: bool) -> None:
    if not rows:
        print("No datasets matched.")
        return

    columns = [
        "dataset_id",
        "status",
        "dpf_at_acquisition",
        "snapshot_status",
        "protocol_name",
        "line_strain",
        "rig_id",
        "arena_id",
        "camera_id",
        "zarr_path",
    ]
    if include_training:
        columns.extend(["training_set_id", "latest_run_id", "latest_run_model_path"])

    print("\t".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None or value == "":
                value = "-"
            values.append(str(value))
        print("\t".join(values))


def _write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _emit_json(rows: List[Dict[str, Any]], output: Optional[Path]) -> None:
    content = json.dumps(rows, indent=2)
    if output:
        _write_output(output, content + "\n")
        print(f"Wrote JSON output to {output}")
    else:
        print(content)


def _emit_csv(rows: List[Dict[str, Any]], output: Optional[Path]) -> None:
    fieldnames: List[str] = sorted({key for row in rows for key in row.keys()})
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV output to {output}")
        return

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)

    query, params = _build_query(args)
    try:
        rows = registry.conn.execute(query, params).fetchall()
    finally:
        registry.close()

    payload = _rows_to_dicts(rows)

    if args.list_ids:
        for row in payload:
            print(row.get("dataset_id", ""))
    elif args.format == "json":
        _emit_json(payload, args.output)
    elif args.format == "csv":
        _emit_csv(payload, args.output)
    else:
        _print_table(payload, include_training=bool(args.include_training or args.trained_only or args.set_id))

    if args.output_file_list:
        args.output_file_list.parent.mkdir(parents=True, exist_ok=True)
        args.output_file_list.write_text(
            "\n".join(row.get("zarr_path", "") for row in payload if row.get("zarr_path"))
            + ("\n" if payload else ""),
            encoding="utf-8",
        )
        print(f"Wrote {len(payload)} paths to {args.output_file_list}")


if __name__ == "__main__":
    main()
