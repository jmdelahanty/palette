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
from fisheye.shared.json_safety import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path)
    parser.add_argument("--review-montage", type=Path)
    parser.add_argument(
        "--fit-review-run",
        help=(
            "Immutable analysis/arena_geometry_fit_runs run containing the fit, "
            "reveal, panels, and montage. Preferred over external staging files."
        ),
    )
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
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Optional immutable summary receipt for DAG orchestration.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    external_supplied = args.fit_report is not None or args.review_montage is not None
    if args.fit_review_run is not None and external_supplied:
        raise ValueError(
            "Choose --fit-review-run or the external fit-report/montage pair, not both."
        )
    if args.fit_review_run is None and (
        args.fit_report is None or args.review_montage is None
    ):
        raise ValueError(
            "Supply --fit-review-run or both --fit-report and --review-montage."
        )
    plan = plan_reviewed_palette_geometry_candidate(
        source_zarr=args.zarr,
        fit_report_path=args.fit_report,
        montage_path=args.review_montage,
        fit_review_run=args.fit_review_run,
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
    payload = {
        "schema_id": "palette.reviewed_arena_geometry_candidate_publication",
        "schema_version": 1,
        "mode": "apply" if args.apply else "dry_run",
        "candidate_id": plan.candidate_id,
        "candidate_kind": plan.candidate_kind,
        "candidate_record_sha256": plan.candidate_record_sha256,
        "target_run_path": str(plan.target_run_path),
        "status": status,
        "published": bool(published),
        "operationally_selected": False,
        "detection_gate_applied": False,
        "registry_updated": False,
    }
    if args.result_json is not None:
        write_json_atomic(args.result_json.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
