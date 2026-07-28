"""Publish the paired refined-detection physical-profile canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.zarr.refined_detection_profile_canary import (
    publish_refined_detection_profile_canary,
)
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-group", type=Path, required=True)
    parser.add_argument("--source-fixture-manifest", type=Path, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--canary-id", required=True)
    parser.add_argument("--crimson-implementation-commit", required=True)
    parser.add_argument("--crimson-evidence-commit", required=True)
    parser.add_argument("--crimson-evidence-sha256", required=True)
    parser.add_argument(
        "--promoted-default",
        action="store_true",
        help=(
            "Write the named promoted detection profile instead of its frozen "
            "physically identical benchmark candidate."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = publish_refined_detection_profile_canary(
        source_group_path=args.source_group,
        source_fixture_manifest_path=args.source_fixture_manifest,
        source_run_id=args.source_run_id,
        recording_identity=args.recording_identity,
        destination=args.destination,
        scratch_root=args.scratch_root,
        canary_id=args.canary_id,
        crimson_implementation_commit=args.crimson_implementation_commit,
        crimson_evidence_commit=args.crimson_evidence_commit,
        crimson_evidence_sha256=args.crimson_evidence_sha256,
        **(
            {"access_aware_profile": DETECTION_PUBLISHED_ACCESS_AWARE_V1}
            if args.promoted_default
            else {}
        ),
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
