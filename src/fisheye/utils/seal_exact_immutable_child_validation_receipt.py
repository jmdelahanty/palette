"""Seal or exactly reuse one immutable exact-child validation receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.exact_immutable_child_validation_receipt import (
    ensure_exact_immutable_child_validation_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--run-path", required=True)
    parser.add_argument("--manifest-attr", required=True)
    parser.add_argument("--manifest-digest-attr", required=True)
    parser.add_argument("--palette-commit", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-recording-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = ensure_exact_immutable_child_validation_receipt(
        args.analysis_zarr,
        run_path=args.run_path,
        manifest_attr=args.manifest_attr,
        manifest_digest_attr=args.manifest_digest_attr,
        palette_commit=args.palette_commit,
        output_json=args.output_json,
        expected_recording_id=args.expected_recording_id,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
