"""Stub CLI for querying the registry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from .db import Registry, RegistryPaths


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the Palette registry (stub).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry SQLite path.",
    )
    parser.add_argument(
        "--where",
        type=str,
        help="Raw SQL WHERE clause (advanced).",
    )
    parser.add_argument(
        "--dpf",
        type=int,
        help="Filter by dpf_at_acquisition.",
    )
    parser.add_argument(
        "--strain",
        type=str,
        help="Filter by line_strain substring.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        help="Filter by protocol_name.",
    )
    parser.add_argument(
        "--cross-id",
        type=str,
        help="Filter by cross_id.",
    )
    parser.add_argument(
        "--dish-id",
        type=str,
        help="Filter by dish_id.",
    )
    parser.add_argument(
        "--missing",
        action="store_true",
        help="Show datasets with missing or partial provenance.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum rows to display (0 = no limit).",
    )
    return parser.parse_args(argv)


def _build_query(args: argparse.Namespace) -> tuple[str, list]:
    clauses = []
    params: list = []
    if args.dpf is not None:
        clauses.append("p.dpf_at_acquisition = ?")
        params.append(args.dpf)
    if args.strain:
        clauses.append("p.line_strain LIKE ?")
        params.append(f"%{args.strain}%")
    if args.protocol:
        clauses.append("p.protocol_name = ?")
        params.append(args.protocol)
    if args.cross_id:
        clauses.append("p.cross_id = ?")
        params.append(args.cross_id)
    if args.dish_id:
        clauses.append("p.dish_id = ?")
        params.append(args.dish_id)
    if args.missing:
        clauses.append("(p.snapshot_status IS NULL OR p.snapshot_status != 'complete')")
    if args.where:
        clauses.append(f"({args.where})")

    where = ""
    if clauses:
        where = "WHERE " + " AND ".join(clauses)
    return where, params


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    cursor = registry.conn.cursor()

    where, params = _build_query(args)
    limit_clause = ""
    if args.limit > 0:
        limit_clause = "LIMIT ?"
        params = list(params) + [args.limit]

    rows = cursor.execute(
        f"""
        SELECT d.dataset_id, d.zarr_path, d.status,
               p.dish_id, p.cross_id, p.line_strain,
               p.dpf_at_acquisition, p.protocol_name, p.snapshot_status
        FROM datasets d
        LEFT JOIN provenance p ON d.dataset_id = p.dataset_id
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
            print(
                f"{row['dataset_id']} | {row['status']} | dpf={row['dpf_at_acquisition'] or '-'} | "
                f"protocol={row['protocol_name'] or '-'} | strain={row['line_strain'] or '-'} | "
                f"snapshot={row['snapshot_status'] or '-'}"
            )
            print(f"  {row['zarr_path']}")

    registry.close()


if __name__ == "__main__":
    main()
