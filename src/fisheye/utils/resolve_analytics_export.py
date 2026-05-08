"""Resolve indexed Palette analytics export table paths from the registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fisheye.utils.analytics_export_resolution import (
    AnalyticsExportTableResolution,
    resolve_latest_export_table,
)


def _resolution_payload(resolution: AnalyticsExportTableResolution) -> dict[str, Any]:
    return {
        "registry_path": str(resolution.registry_path),
        "export_run_id": resolution.export_run_id,
        "table_name": resolution.table_name,
        "table_path": str(resolution.table_path),
        "collection_id": resolution.collection_id,
        "collection_manifest_sha256": resolution.collection_manifest_sha256,
        "collection_name": resolution.collection_name,
        "status": resolution.status,
        "output_root": str(resolution.output_root) if resolution.output_root is not None else None,
        "export_manifest_path": (
            str(resolution.export_manifest_path)
            if resolution.export_manifest_path is not None
            else None
        ),
        "created_at_utc": resolution.created_at_utc,
        "row_count": resolution.row_count,
        "part_count": resolution.part_count,
        "part_files": list(resolution.part_files),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve an indexed Palette analytics export table from the registry.",
    )
    parser.add_argument("--registry", type=Path, help="Palette registry SQLite path.")
    parser.add_argument("--collection-id", help="Filter by collection id.")
    parser.add_argument("--collection-manifest-sha256", help="Filter by collection manifest SHA-256.")
    parser.add_argument("--export-run-id", help="Filter by explicit export run id.")
    parser.add_argument("--table", required=True, help="Analytics table name, e.g. swim_bout_metrics.")
    parser.add_argument("--status", default="active", help="Registry export status filter; use 'any' to disable.")
    parser.add_argument(
        "--format",
        choices=("json", "path"),
        default="json",
        help="Output JSON metadata or only the resolved table path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    resolution = resolve_latest_export_table(
        registry_path=args.registry,
        collection_id=args.collection_id,
        collection_manifest_sha256=args.collection_manifest_sha256,
        export_run_id=args.export_run_id,
        table_name=args.table,
        status=str(args.status),
    )
    if args.format == "path":
        print(resolution.table_path)
    else:
        print(json.dumps(_resolution_payload(resolution), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
