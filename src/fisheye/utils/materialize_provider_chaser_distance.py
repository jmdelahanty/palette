"""Plan or publish one receipt-backed provider chaser-distance run.

The command is a read-only plan by default.  ``--apply`` is required before
any Zarr group is created or atomically published.  It never updates a
registry or a selector pointer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fisheye.analysis_workflows.provider_chaser_distance_publication import (
    build_provider_chaser_distance_publication_plan,
    publish_provider_chaser_distance_run,
)
from fisheye.shared.json_safety import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--receipt-json", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-recording-id")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument(
        "--copy-backend",
        choices=("python", "rsync"),
        default="python",
        help="Atomic publication copy backend (default: python).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish the immutable selector-ineligible run; default is dry-run.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optionally write the plan/result receipt atomically.",
    )
    return parser


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read candidate-chain receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Candidate-chain receipt must be one JSON object.")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = _read_receipt(args.receipt_json)
    plan = build_provider_chaser_distance_publication_plan(
        args.analysis_zarr,
        receipt=receipt,
        run_name=args.run_name,
        expected_recording_id=args.expected_recording_id,
    )
    if not args.apply:
        result = plan.to_json()
    else:
        result = publish_provider_chaser_distance_run(
            plan,
            scratch_root=args.scratch_root,
            copy_backend=args.copy_backend,
        )
    if args.output_json is not None:
        write_json_atomic(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
