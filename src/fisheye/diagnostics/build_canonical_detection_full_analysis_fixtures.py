#!/usr/bin/env python3
"""Plan or publish the paired Crimson canonical-detection full-analysis fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.zarr.canonical_detection_full_analysis_fixture import (
    load_full_analysis_fixture_spec,
    plan_full_analysis_fixture_pair,
    publish_full_analysis_fixture_pair,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        help="Existing node-local scratch directory; required by apply mode.",
    )
    parser.add_argument(
        "--expected-palette-commit",
        help=(
            "Require this full Palette commit. Apply mode requires the option and "
            "also requires a clean worktree."
        ),
    )
    parser.add_argument(
        "--pair-copy-mode",
        choices=("auto", "copy", "reflink"),
        default="auto",
        help=(
            "How to derive hybrid.zarr from the independent incomplete regular "
            "benchmark base. auto uses reflink only after an isolation probe."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform payload copies and atomic publication; default is read-only plan mode.",
    )
    args = parser.parse_args(argv)
    if args.apply and not args.expected_palette_commit:
        parser.error("--apply requires --expected-palette-commit")
    if args.apply and args.scratch_root is None:
        parser.error("--apply requires --scratch-root")

    spec = load_full_analysis_fixture_spec(args.spec)
    kwargs = {
        "spec": spec,
        "destination": args.destination,
        "benchmark_root": args.benchmark_root,
        "pair_copy_mode": args.pair_copy_mode,
        "expected_palette_commit": args.expected_palette_commit,
        "scratch_root": args.scratch_root,
    }
    result = (
        publish_full_analysis_fixture_pair(**kwargs)
        if args.apply
        else plan_full_analysis_fixture_pair(**kwargs)
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
