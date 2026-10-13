"""Inspect, publish, or activate one canonical-v3 detection successor.

Without ``--selector-eligible``/``--activate`` this publishes the historical
selector-ineligible legacy-conversion successor.  ``--selector-eligible`` seals
``stage_selector_eligible=true`` in the new successor manifest but leaves
selectors unchanged.  ``--activate`` publishes such a successor when absent,
validates it against the currently selected legacy run, and makes it the
active canonical-v3 raw-detection authority.  Every mode is a read-only dry
run unless ``--apply`` is given; the registry is never updated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.detection_snapshot_publication import (
    CANONICAL_DETECTION_SUCCESSOR_ACTIVATION_SCHEMA_ID,
    CANONICAL_DETECTION_SUCCESSOR_PUBLICATION_SCHEMA_ID,
    CanonicalDetectionActivationRefused,
    activate_canonical_detection_successor,
    inspect_canonical_detection_successor_source,
    publish_canonical_detection_successor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--source-detect-group", required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--successor-run", required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument(
        "--copy-backend",
        choices=("python", "rsync"),
        default="python",
    )
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument(
        "--selector-eligible",
        action="store_true",
        help=(
            "Seal stage_selector_eligible=true in the published successor "
            "manifest without changing selectors."
        ),
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help=(
            "Publish a selector-eligible successor if absent, then activate it "
            "as the canonical-v3 detection authority (implies "
            "--selector-eligible). Refuses on selector drift, source drift, "
            "or content mismatch with an existing successor."
        ),
    )
    parser.add_argument(
        "--repair-latest-complete",
        action="store_true",
        help=(
            "With --activate: when detect_runs latest names exactly the legacy "
            "source and latest_complete is unset, set latest_complete to that "
            "run (only if it is strictly complete) under the archive lock "
            "before activation. Refuses if latest_complete names another run "
            "or the legacy run is not strictly complete."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Publish the successor. Without this flag the command is a "
            "read-only source inspection."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repair_latest_complete and not args.activate:
        build_parser().error("--repair-latest-complete requires --activate.")
    if args.activate:
        return _run_activation(args)
    try:
        if args.apply:
            if args.scratch_root is None:
                raise ValueError("--apply requires --scratch-root.")
            result = publish_canonical_detection_successor(
                analysis_zarr=args.analysis_zarr,
                source_detect_group_path=args.source_detect_group,
                recording_identity=args.recording_identity,
                successor_run_id=args.successor_run,
                scratch_root=args.scratch_root,
                copy_backend=args.copy_backend,
                keep_scratch=args.keep_scratch,
                result_json=args.result_json,
                selector_eligible=args.selector_eligible,
            )
        else:
            result = inspect_canonical_detection_successor_source(
                analysis_zarr=args.analysis_zarr,
                source_detect_group_path=args.source_detect_group,
                recording_identity=args.recording_identity,
                successor_run_id=args.successor_run,
            )
            result = {
                **result,
                "mode": "dry_run",
                "zarr_writes": False,
                "manifest_selector_eligible": args.selector_eligible,
                "next_action": "rerun_with_apply_after_review",
            }
            write_json_atomic(args.result_json, result)
    except Exception as exc:
        result = {
            "schema_id": CANONICAL_DETECTION_SUCCESSOR_PUBLICATION_SCHEMA_ID,
            "schema_version": 1,
            "status": "failed",
            "mode": "apply" if args.apply else "dry_run",
            "analysis_zarr": str(args.analysis_zarr),
            "source_group_path": args.source_detect_group,
            "successor_run_id": args.successor_run,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_activation(args: argparse.Namespace) -> int:
    base = {
        "schema_id": CANONICAL_DETECTION_SUCCESSOR_ACTIVATION_SCHEMA_ID,
        "schema_version": 1,
        "mode": "apply" if args.apply else "dry_run",
        "analysis_zarr": str(args.analysis_zarr),
        "source_group_path": args.source_detect_group,
        "successor_run_id": args.successor_run,
    }
    try:
        result = activate_canonical_detection_successor(
            analysis_zarr=args.analysis_zarr,
            source_detect_group_path=args.source_detect_group,
            recording_identity=args.recording_identity,
            successor_run_id=args.successor_run,
            apply=args.apply,
            scratch_root=args.scratch_root,
            copy_backend=args.copy_backend,
            keep_scratch=args.keep_scratch,
            result_json=args.result_json,
            repair_latest_complete=args.repair_latest_complete,
        )
    except CanonicalDetectionActivationRefused as exc:
        result = {**base, **exc.receipt, "mode": base["mode"]}
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    except Exception as exc:
        result = {
            **base,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
