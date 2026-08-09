"""Publish strict keypoint bases plus an instance-key delta review surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.training_review_artifact_publication import (
    publish_training_keypoint_review_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--terminal-keypoint-run", required=True)
    parser.add_argument("--raw-keypoint-run", required=True)
    parser.add_argument("--quality-run", required=True)
    parser.add_argument("--refined-keypoint-run", required=True)
    parser.add_argument("--body-frame-run", required=True)
    parser.add_argument("--keypoint-delta-run", required=True)
    parser.add_argument("--keypoint-delta-generation", default="generation_000001")
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--result-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = publish_training_keypoint_review_artifact(
        source_archive=args.source_archive,
        destination=args.destination,
        scratch_root=args.scratch_root,
        crop_run_id=args.crop_run,
        terminal_keypoint_run_id=args.terminal_keypoint_run,
        raw_keypoint_run_id=args.raw_keypoint_run,
        quality_run_id=args.quality_run,
        refined_keypoint_run_id=args.refined_keypoint_run,
        body_frame_run_id=args.body_frame_run,
        keypoint_delta_run_id=args.keypoint_delta_run,
        keypoint_delta_generation=args.keypoint_delta_generation,
        created_by=args.created_by,
    )
    if args.result_json is not None:
        write_json_atomic(args.result_json.expanduser().resolve(), result)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
