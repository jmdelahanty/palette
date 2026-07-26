"""Plan or publish one reviewed, image-derived arena-geometry candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
    plan_reviewed_palette_geometry_candidate,
    publish_arena_geometry_candidate,
    validate_arena_geometry_candidate_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--review-montage", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--reviewed-at-utc",
        required=True,
        help="Immutable RFC3339 review time supplied by the review record.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp"))
        / "palette-arena-geometry-candidates",
    )
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish the pointerless candidate. Default is a read-only dry-run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = plan_reviewed_palette_geometry_candidate(
        source_zarr=args.zarr,
        fit_report_path=args.fit_report,
        montage_path=args.review_montage,
        reviewer=args.reviewer,
        reviewed_at_utc=args.reviewed_at_utc,
    )
    existing = None
    if plan.target_run_path.exists():
        existing = validate_arena_geometry_candidate_run(
            plan.target_run_path,
            expected_plan=plan,
            require_complete=True,
            require_eligible=True,
        )
        if not existing["valid"]:
            raise RuntimeError(
                f"Existing candidate is not the planned immutable run: {existing}"
            )
    if args.apply:
        result = publish_arena_geometry_candidate(
            plan,
            scratch_root=args.scratch_root,
            copy_backend=args.copy_backend,
        )
        status = result["status"]
        published = result["published"]
    else:
        status = "already_complete" if existing is not None else "dry_run_validated"
        published = False
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry_run",
                "candidate_id": plan.candidate_id,
                "candidate_kind": plan.candidate_kind,
                "candidate_record_sha256": plan.candidate_record_sha256,
                "target_run_path": str(plan.target_run_path),
                "status": status,
                "published": bool(published),
                "operationally_selected": False,
                "detection_gate_applied": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
