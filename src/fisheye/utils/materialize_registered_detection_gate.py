"""Build and atomically publish one selected-geometry detection gate table."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.materializers.registered_detection_gate import (
    build_registered_detection_gate_plan,
    publish_registered_detection_gate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr", type=Path)
    parser.add_argument("--source-group", required=True)
    parser.add_argument("--selection-run", required=True)
    parser.add_argument("--output-run", default=None)
    parser.add_argument("--inner-rows", type=int, default=16_384)
    parser.add_argument("--shard-rows", type=int, default=131_072)
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_registered_detection_gate_plan(
        args.zarr,
        source_group_path=args.source_group,
        selection_run=args.selection_run,
        output_run=args.output_run,
        inner_rows=args.inner_rows,
        shard_rows=args.shard_rows,
    )
    result = {
        "status": "planned",
        "source_group_path": plan.source_group_path,
        "selection_run": plan.selection_run,
        "output_run": plan.output_run,
        "target_run_path": str(plan.target_run_path),
        "row_count": plan.row_count,
        "frame_count": plan.frame_count,
        "inner_rows": plan.inner_rows,
        "shard_rows": plan.shard_rows,
    }
    if args.apply:
        scratch = args.scratch_root or Path(tempfile.gettempdir()) / "palette_geometry"
        result = publish_registered_detection_gate(
            plan,
            scratch_root=scratch,
            copy_backend=args.copy_backend,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
