"""Seal or exactly reuse a bounded chaser relative-frame validation receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    ensure_chaser_relative_frame_validation_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--palette-commit", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-recording-id")
    parser.add_argument("--expected-manifest-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = ensure_chaser_relative_frame_validation_receipt(
        args.analysis_zarr,
        run_name=args.run_name,
        palette_commit=args.palette_commit,
        output_json=args.output_json,
        expected_recording_id=args.expected_recording_id,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
