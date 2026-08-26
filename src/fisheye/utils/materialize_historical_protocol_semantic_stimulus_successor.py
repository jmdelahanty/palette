#!/usr/bin/env python3
"""Plan or publish one selector-ineligible historical semantic stimulus run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.historical_protocol_semantic_stimulus_successor import (
    plan_historical_protocol_semantic_stimulus_successor,
    publish_historical_protocol_semantic_stimulus_successor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--source-run-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--raw-h5", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument(
        "--copy-backend",
        choices=("python", "rsync"),
        default="python",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish the immutable selector-ineligible run (default: no-write plan).",
    )
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = plan_historical_protocol_semantic_stimulus_successor(
        args.analysis_zarr,
        source_run_name=args.source_run_name,
        run_name=args.run_name,
        raw_h5=args.raw_h5,
    )
    receipt = (
        publish_historical_protocol_semantic_stimulus_successor(
            plan,
            scratch_root=args.scratch_root,
            copy_backend=args.copy_backend,
        )
        if args.apply
        else {"mode": "no_write", **plan.receipt()}
    )
    text = json.dumps(receipt, indent=2, sort_keys=True)
    print(text)
    if args.receipt is not None:
        target = args.receipt.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
