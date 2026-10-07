#!/usr/bin/env python3
"""Ingest one completed Citrus transfer (transfer-v2) inside an LSF job.

Transfer-v2 parent intake is the only way new recordings enter Palette. This
command is the compatibility entry point the launcher's job script runs; it
calls :func:`fisheye.intake.import_delivery`, which organizes the exact parent
recordings, imports their analysis Zarrs in-process (routing unified H5s to
the native profile) and retires verified staging. It never writes the
registry: registration happens on the writer host through
``python -m fisheye.intake register-delivery`` (job-mode registration is
retired). It does not run detect, refine, crops, keypoints, or masks.

Exit codes follow fisheye.intake: 0 done, 65 refused, 75 held by another
live job, 1 retryable failure; 2 for a usage error.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from fisheye.intake.delivery import DEFAULT_DESTINATION_ROOT

DEFAULT_DEST_ROOT = DEFAULT_DESTINATION_ROOT
RETIRED_REGISTRATION = (
    "job-mode registration is retired: the LSF import never writes the registry; "
    "the writer host registers with `python -m fisheye.intake register-delivery`"
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path, help="Completed Citrus transfer session directory.")
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path(os.environ.get("PALETTE_RECORDINGS_ROOT", DEFAULT_DEST_ROOT)),
        help=f"Organized recordings destination root (default: $PALETTE_RECORDINGS_ROOT or {DEFAULT_DEST_ROOT}).",
    )
    parser.add_argument("--run-dir", type=Path, help="New directory for this attempt's logs/status.")
    parser.add_argument("--apply", action="store_true", help="Apply organization/import writes.")
    parser.add_argument("--dry-run", action="store_true", help="Plan organization/import without writes.")
    parser.add_argument("--register", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--registry", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--status-json", type=Path, help="Optional path for final status JSON.")
    parser.add_argument("--resume-transfer-plan", type=Path, help="Exact saved organization plan for retry, including interrupted retirement.")

    args = parser.parse_args(argv)
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive.")
    if args.register or args.registry is not None:
        parser.error(RETIRED_REGISTRATION)
    if not args.apply:
        args.dry_run = True
    from fisheye.utils.citrus_transfer_parent_workflow import run_transfer_parent_workflow

    return run_transfer_parent_workflow(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
