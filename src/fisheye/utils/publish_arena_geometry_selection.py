"""Publish one reviewed arena-geometry candidate as the operational selection."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.materializers.arena_geometry_selection import (
    build_arena_geometry_selection_plan,
    publish_arena_geometry_selection,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr", type=Path)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--selected-by", required=True)
    parser.add_argument("--decision-reason", required=True)
    parser.add_argument("--decision-source", default="manual_review")
    parser.add_argument("--expected-selection-run", default=None)
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_arena_geometry_selection_plan(
        args.zarr,
        candidate_run=args.candidate_run,
        selected_by=args.selected_by,
        decision_reason=args.decision_reason,
        decision_source=args.decision_source,
    )
    if (
        args.expected_selection_run is not None
        and args.expected_selection_run != plan.selection_id
    ):
        raise ValueError(
            "Computed selection identity does not match --expected-selection-run: "
            f"{plan.selection_id!r} != {args.expected_selection_run!r}."
        )
    result = {
        "status": "planned",
        "selection_id": plan.selection_id,
        "candidate_run": plan.candidate_run,
        "candidate_record_sha256": plan.candidate_record_sha256,
        "selection_record_sha256": plan.selection_record_sha256,
        "target_run_path": str(plan.target_run_path),
    }
    if args.apply:
        scratch = args.scratch_root or Path(tempfile.gettempdir()) / "palette_geometry"
        result = publish_arena_geometry_selection(
            plan,
            scratch_root=scratch,
            copy_backend=args.copy_backend,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
