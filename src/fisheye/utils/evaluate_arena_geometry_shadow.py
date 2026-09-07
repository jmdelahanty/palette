"""Evaluate a frozen dish-rim policy in shadow mode without archive writes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.materializers.arena_geometry_comparison import (
    build_arena_geometry_shadow_evaluation,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr", type=Path)
    parser.add_argument("--acquisition-candidate-run", required=True)
    parser.add_argument("--fit-review-run", required=True)
    parser.add_argument("--detect-source-group", required=True)
    parser.add_argument(
        "--policy-json",
        type=Path,
        help="Exact shadow policy JSON; omission leaves thresholds inactive.",
    )
    args = parser.parse_args(argv)
    policy = None
    if args.policy_json is not None:
        policy = json.loads(args.policy_json.read_text(encoding="utf-8"))
        if not isinstance(policy, dict):
            raise ValueError("--policy-json must contain one versioned policy object.")
    result = build_arena_geometry_shadow_evaluation(
        args.zarr,
        acquisition_candidate_run=args.acquisition_candidate_run,
        fit_review_run=args.fit_review_run,
        detect_source_group_path=args.detect_source_group,
        policy=policy,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
