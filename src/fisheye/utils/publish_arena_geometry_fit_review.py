"""Import a frozen arena-geometry review package into its analysis Zarr."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.materializers.arena_geometry_fit_review import (
    build_arena_geometry_fit_review_plan,
    publish_arena_geometry_fit_review,
)
from fisheye.shared.json_safety import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--review-package-dir", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_arena_geometry_fit_review_plan(
        args.zarr, review_package_dir=args.review_package_dir
    )
    result = {
        "status": "planned",
        "fit_review_run": plan.run_name,
        "review_record_sha256": plan.review_record_sha256,
        "target_run_path": str(plan.target_run_path),
        "human_review_required": True,
        "candidate_published": False,
        "selection_performed": False,
        "detection_gate_applied": False,
    }
    if args.apply:
        scratch = args.scratch_root or Path(tempfile.gettempdir()) / "palette_geometry"
        result = {
            **result,
            **publish_arena_geometry_fit_review(
                plan, scratch_root=scratch, copy_backend=args.copy_backend
            ),
        }
    if args.result_json is not None:
        write_json_atomic(args.result_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
