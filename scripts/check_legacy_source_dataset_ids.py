#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count/list legacy source-recording dataset rows where dataset_id=session_uuid."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/nvme1/palette_registry.sqlite"),
        help="Path to registry sqlite file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many example rows to print (0 disables row listing).",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(str(args.registry))
    conn.row_factory = sqlite3.Row
    try:
        count_row = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM datasets
            WHERE artifact_kind='source_recording'
              AND session_uuid IS NOT NULL
              AND dataset_id=session_uuid
              AND status='active';
            """
        ).fetchone()
        count = int(count_row["n"]) if count_row is not None else 0
        print(f"legacy_active_source_rows={count}")

        if args.limit > 0 and count > 0:
            rows = conn.execute(
                """
                SELECT dataset_id, zarr_path, last_seen_utc
                FROM datasets
                WHERE artifact_kind='source_recording'
                  AND session_uuid IS NOT NULL
                  AND dataset_id=session_uuid
                  AND status='active'
                ORDER BY last_seen_utc DESC
                LIMIT ?;
                """,
                (int(args.limit),),
            ).fetchall()
            for row in rows:
                print(
                    f"{row['dataset_id']}\t{row['zarr_path']}\t{row['last_seen_utc']}"
                )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
