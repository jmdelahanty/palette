#!/usr/bin/env python3
"""Combine compatible canonical detection benchmark workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.zarr.detection_benchmark_reduction import (
    reduce_detection_benchmark_workflows,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow-root",
        action="append",
        required=True,
        type=Path,
    )
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = reduce_detection_benchmark_workflows(
        workflow_roots=args.workflow_root,
        benchmark_root=args.benchmark_root,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.expanduser().resolve()),
                "summary": result["summary"],
                "selection": result["selection"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
