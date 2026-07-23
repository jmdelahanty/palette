#!/usr/bin/env python
"""Preflight the pre/post GoodCopBadCop trajectory figure inputs.

This command currently fails closed because the figure requires behavior-role,
epoch, and smoothed-kinematics bindings that are not protected by the canonical
chaser-distance publication seal. Republish those semantic authorities before
restoring the renderer; raw protocol attrs and direct run navigation are not
acceptable recovery paths.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from fisheye.analysis.chaser_distance_io import load_chaser_distance_run
from fisheye.analysis.goodcopbadcop_common import registry_db
from fisheye.shared.zarr_io import open_zarr_root


DEFAULT_RID = "2026-06-14T21-50-10Z_arena_3"


def resolve_zarr(recording_like: str) -> tuple[str, str]:
    """Return one active analysis-zarr locator from the canonical registry."""

    conn = sqlite3.connect(registry_db())
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT recording_id, zarr_path FROM dataset_context_current "
            "WHERE recording_id LIKE ? AND zarr_use='analysis' "
            "AND dataset_status='active' ORDER BY recording_id LIMIT 1",
            (f"%{recording_like}%",),
        ).fetchone()
    finally:
        conn.close()
    if row is None or not Path(row["zarr_path"]).is_dir():
        raise SystemExit(
            f"No reachable active analysis zarr for recording like {recording_like!r}."
        )
    return str(row["recording_id"]), str(row["zarr_path"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", default=DEFAULT_RID)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Reserved output path; no output is written until preflight succeeds.",
    )
    args = parser.parse_args(argv)

    _recording_id, zarr_path = resolve_zarr(args.recording_id)
    root = open_zarr_root(zarr_path, mode="r")
    distance = load_chaser_distance_run(root)
    distance.require_behavior_authority()


if __name__ == "__main__":
    raise SystemExit(main())
