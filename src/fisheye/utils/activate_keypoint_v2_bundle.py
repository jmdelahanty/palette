"""Plan or apply one atomic keypoint-v2 four-surface authority selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.keypoint_bundle_activation import (
    activate_keypoint_bundle_from_plan,
    build_keypoint_bundle_activation_plan,
)


def _read_plan(path: Path) -> Mapping[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, Mapping):
        raise ValueError("Activation plan must be one JSON object.")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively plan a keypoint-v2 bundle activation. The default is "
            "read-only; --apply requires the exact reviewed plan file."
        )
    )
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--raw-run", required=True)
    parser.add_argument("--quality-run", required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--body-frame-run", required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply --reviewed-plan after revalidating it against live state.",
    )
    parser.add_argument(
        "--reviewed-plan",
        type=Path,
        help="Exact dry-run plan required when --apply is supplied.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.apply:
        if args.reviewed_plan is None:
            raise ValueError("--apply requires --reviewed-plan.")
        plan = _read_plan(args.reviewed_plan.expanduser().resolve())
        payload = plan.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("Reviewed activation plan lacks its payload.")
        requested = {
            "analysis_zarr": str(args.analysis_zarr.expanduser().resolve()),
            "crop": args.crop_run,
            "raw_keypoints": args.raw_run,
            "keypoint_quality": args.quality_run,
            "refined_keypoints": args.refined_run,
            "body_frame": args.body_frame_run,
        }
        planned = {
            "analysis_zarr": payload.get("analysis_zarr"),
            "crop": (
                payload.get("crop", {}).get("run_id")
                if isinstance(payload.get("crop"), Mapping)
                else None
            ),
            **{
                role: (
                    payload.get("members", {}).get(role, {}).get("run_id")
                    if isinstance(payload.get("members"), Mapping)
                    and isinstance(payload["members"].get(role), Mapping)
                    else None
                )
                for role in (
                    "raw_keypoints",
                    "keypoint_quality",
                    "refined_keypoints",
                    "body_frame",
                )
            },
        }
        if planned != requested:
            raise ValueError("CLI candidate IDs differ from the reviewed plan.")
        result = activate_keypoint_bundle_from_plan(plan)
    else:
        if args.reviewed_plan is not None:
            raise ValueError("--reviewed-plan is only valid with --apply.")
        result = build_keypoint_bundle_activation_plan(
            args.analysis_zarr,
            crop_run_id=args.crop_run,
            raw_run_id=args.raw_run,
            quality_run_id=args.quality_run,
            refined_run_id=args.refined_run,
            body_frame_run_id=args.body_frame_run,
        )
    write_json_atomic(args.result_json.expanduser().resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
