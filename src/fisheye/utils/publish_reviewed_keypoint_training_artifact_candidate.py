"""Publish compacted reviewed keypoints as an immutable task-specific source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from fisheye.training.training_review_compaction_publication import (
    publish_reviewed_keypoint_training_artifact_candidate,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_review_archive", type=Path)
    parser.add_argument("--compacted-keypoint-archive", required=True, type=Path)
    parser.add_argument("--compacted-keypoint-run", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--created-by", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = publish_reviewed_keypoint_training_artifact_candidate(
        source_review_archive=args.source_review_archive,
        compacted_keypoint_archive=args.compacted_keypoint_archive,
        compacted_keypoint_run_id=args.compacted_keypoint_run,
        destination=args.destination,
        scratch_root=args.scratch_root,
        created_by=args.created_by,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
