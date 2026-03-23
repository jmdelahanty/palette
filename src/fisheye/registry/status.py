"""Report registry coverage and provenance completeness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from .db import Registry, RegistryPaths


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize registry coverage and missing provenance fields.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry SQLite path.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List datasets (limited).",
    )
    parser.add_argument(
        "--list-issues",
        action="store_true",
        help="List datasets with missing/partial provenance (limited).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum rows to list (0 = no limit).",
    )
    return parser.parse_args(argv)


def _fetch_count(cursor, query: str) -> int:
    row = cursor.execute(query).fetchone()
    return int(row[0]) if row else 0


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    cursor = registry.conn.cursor()

    total = _fetch_count(cursor, "SELECT COUNT(*) FROM datasets;")
    missing_prov = _fetch_count(
        cursor,
        """
        SELECT COUNT(*) FROM datasets d
        LEFT JOIN provenance p ON d.dataset_id = p.dataset_id
        WHERE p.dataset_id IS NULL;
        """,
    )
    partial_snapshot = _fetch_count(
        cursor,
        """
        SELECT COUNT(*) FROM provenance
        WHERE snapshot_status = 'partial';
        """,
    )
    missing_snapshot = _fetch_count(
        cursor,
        """
        SELECT COUNT(*) FROM provenance
        WHERE snapshot_status IS NULL OR snapshot_status = '';
        """,
    )
    missing_protocol = _fetch_count(
        cursor,
        """
        SELECT COUNT(*) FROM provenance
        WHERE protocol_hash IS NULL OR protocol_hash = '';
        """,
    )
    missing_dpf = _fetch_count(
        cursor,
        """
        SELECT COUNT(*) FROM dataset_context_current
        WHERE dpf_at_acquisition IS NULL;
        """,
    )
    missing_context = _fetch_count(
        cursor,
        """
        SELECT COUNT(*) FROM dataset_context_current
        WHERE rig_id IS NULL OR TRIM(rig_id) = ''
           OR arena_id IS NULL OR TRIM(arena_id) = ''
           OR camera_id IS NULL OR TRIM(camera_id) = ''
           OR canvas_name IS NULL OR TRIM(canvas_name) = '';
        """,
    )

    _print_header("Registry Summary")
    print(f"Registry path: {registry_path}")
    print(f"Datasets: {total}")
    print(f"Missing provenance rows: {missing_prov}")
    print(f"Snapshot partial: {partial_snapshot}")
    print(f"Snapshot missing: {missing_snapshot}")
    print(f"Protocol hash missing: {missing_protocol}")
    print(f"DPF missing: {missing_dpf}")
    print(f"Context missing: {missing_context}")

    if args.list or args.list_issues:
        _print_header("Dataset Listing")
        limit_clause = ""
        params = ()
        if args.limit > 0:
            limit_clause = "LIMIT ?"
            params = (args.limit,)

        where = ""
        if args.list_issues:
            where = (
                "WHERE p.snapshot_status IS NULL OR p.snapshot_status != 'complete' "
                "OR p.protocol_hash IS NULL OR dcc.dpf_at_acquisition IS NULL "
                "OR dcc.rig_id IS NULL OR TRIM(dcc.rig_id) = '' "
                "OR dcc.arena_id IS NULL OR TRIM(dcc.arena_id) = '' "
                "OR dcc.camera_id IS NULL OR TRIM(dcc.camera_id) = '' "
                "OR dcc.canvas_name IS NULL OR TRIM(dcc.canvas_name) = ''"
            )

        rows = cursor.execute(
            f"""
            SELECT d.dataset_id, d.zarr_path, d.status,
                   p.snapshot_status, p.snapshot_missing_json,
                   dcc.protocol_name, dcc.dpf_at_acquisition
            FROM datasets d
            LEFT JOIN provenance p ON d.dataset_id = p.dataset_id
            LEFT JOIN dataset_context_current dcc ON d.dataset_id = dcc.dataset_id
            {where}
            ORDER BY d.dataset_id
            {limit_clause};
            """,
            params,
        ).fetchall()

        if not rows:
            print("No datasets matched.")
        else:
            for row in rows:
                missing = row["snapshot_missing_json"]
                missing = missing if missing else "-"
                print(
                    f"{row['dataset_id']} | {row['status']} | snapshot={row['snapshot_status'] or '-'} "
                    f"| dpf={row['dpf_at_acquisition'] if row['dpf_at_acquisition'] is not None else '-'} "
                    f"| protocol={row['protocol_name'] or '-'} | missing={missing}"
                )
                print(f"  {row['zarr_path']}")

    registry.close()


if __name__ == "__main__":
    main()
