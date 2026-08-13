"""Plan or publish one immutable arena-geometry comparison artifact."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.materializers.arena_geometry_comparison import (
    CORROBORATED_ACQUISITION_POLICY_ID,
    MANUAL_REVIEW_POLICY_ID,
    SEMANTIC_COMPATIBILITY_STATES,
    build_arena_geometry_comparison_plan,
    publish_arena_geometry_comparison,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr", type=Path)
    parser.add_argument("--acquisition-candidate-run", required=True)
    parser.add_argument("--palette-candidate-run", required=True)
    parser.add_argument(
        "--semantic-compatibility",
        choices=tuple(sorted(SEMANTIC_COMPATIBILITY_STATES)),
        required=True,
    )
    parser.add_argument(
        "--policy-id",
        choices=(MANUAL_REVIEW_POLICY_ID, CORROBORATED_ACQUISITION_POLICY_ID),
        default=MANUAL_REVIEW_POLICY_ID,
    )
    parser.add_argument("--semantic-reviewer")
    parser.add_argument("--semantic-reviewed-at-utc")
    parser.add_argument("--semantic-evidence-reason")
    parser.add_argument("--detect-source-group")
    parser.add_argument("--expected-comparison-run")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    review_values = (
        args.semantic_reviewer,
        args.semantic_reviewed_at_utc,
        args.semantic_evidence_reason,
    )
    if any(value is not None for value in review_values):
        if not all(str(value or "").strip() for value in review_values):
            raise ValueError("Semantic review fields must be supplied together.")
        semantic_review = {
            "reviewer": args.semantic_reviewer,
            "reviewed_at_utc": args.semantic_reviewed_at_utc,
            "evidence_reason": args.semantic_evidence_reason,
        }
    else:
        semantic_review = None
    plan = build_arena_geometry_comparison_plan(
        args.zarr,
        acquisition_candidate_run=args.acquisition_candidate_run,
        palette_candidate_run=args.palette_candidate_run,
        semantic_compatibility=args.semantic_compatibility,
        policy_id=args.policy_id,
        semantic_review=semantic_review,
        detect_source_group_path=args.detect_source_group,
    )
    if (
        args.expected_comparison_run is not None
        and args.expected_comparison_run != plan.comparison_id
    ):
        raise ValueError(
            "Computed comparison identity does not match --expected-comparison-run: "
            f"{plan.comparison_id!r} != {args.expected_comparison_run!r}."
        )
    result = {
        "status": "planned",
        "comparison_id": plan.comparison_id,
        "comparison_record_sha256": plan.comparison_record_sha256,
        "target_run_path": str(plan.target_run_path),
        "decision": plan.comparison_record["decision"],
        "policy": plan.comparison_record["policy"],
    }
    if args.apply:
        scratch = args.scratch_root or Path(tempfile.gettempdir()) / "palette_geometry"
        result = publish_arena_geometry_comparison(
            plan,
            scratch_root=scratch,
            copy_backend=args.copy_backend,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
