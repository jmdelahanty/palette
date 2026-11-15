"""Plan or publish one log-only Citrus-vs-Palette blind-fit geometry cross-check.

The cross-check compares the dish geometry Citrus declared/applied in the
unified H5 (an ``acquisition_registered_dish`` candidate with source kind
``citrus_h5``) against a frozen, unreviewed ``top_rim_preferred_v1`` Palette
blind fit (an ``arena_geometry_fit_runs`` package).  It records raw
disagreement only: no pass/fail flag, no threshold, no selection.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.materializers.arena_geometry_comparison import (
    build_arena_geometry_citrus_crosscheck_plan,
    publish_arena_geometry_citrus_crosscheck,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr", type=Path)
    parser.add_argument("--acquisition-candidate-run", required=True)
    parser.add_argument("--fit-review-run", required=True)
    parser.add_argument("--detect-source-group")
    parser.add_argument("--expected-comparison-run")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_arena_geometry_citrus_crosscheck_plan(
        args.zarr,
        acquisition_candidate_run=args.acquisition_candidate_run,
        fit_review_run=args.fit_review_run,
        detect_source_group_path=args.detect_source_group,
    )
    if (
        args.expected_comparison_run is not None
        and args.expected_comparison_run != plan.comparison_id
    ):
        raise ValueError(
            "Computed cross-check identity does not match --expected-comparison-run: "
            f"{plan.comparison_id!r} != {args.expected_comparison_run!r}."
        )
    record = plan.comparison_record
    result = {
        "status": "planned",
        "comparison_id": plan.comparison_id,
        "comparison_record_sha256": plan.comparison_record_sha256,
        "target_run_path": str(plan.target_run_path),
        "citrus_geometry": record["citrus_geometry"],
        "disagreement_metrics": record["disagreement_metrics"],
        "gate_disagreement": {
            key: value
            for key, value in record["gate_disagreement"].items()
            if key != "source_signature_payload"
        },
        "decision": record["decision"],
    }
    if args.apply:
        scratch = args.scratch_root or Path(tempfile.gettempdir()) / "palette_geometry"
        result = publish_arena_geometry_citrus_crosscheck(
            plan,
            scratch_root=scratch,
            copy_backend=args.copy_backend,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
