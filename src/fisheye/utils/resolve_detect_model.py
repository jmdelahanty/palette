#!/usr/bin/env python3
"""Resolve a detection model from registry by recording metadata similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.registry.model_resolution import (
    Candidate,
    TargetProfile,
    load_candidates,
    load_target_profile,
    resolve_recording_id,
)

_load_candidates = load_candidates
_load_target_profile = load_target_profile
_resolve_recording_id = resolve_recording_id

__all__ = [
    "Candidate",
    "TargetProfile",
    "_load_candidates",
    "_load_target_profile",
    "_resolve_recording_id",
    "load_candidates",
    "load_target_profile",
    "main",
    "resolve_recording_id",
]


def _print_target(target: TargetProfile) -> None:
    print("Target recording profile")
    print(f"  recording_id: {target.recording_id}")
    for key in (
        "recording_type",
        "recording_subtype",
        "behavior_mode",
        "rig_id",
        "arena_id",
        "camera_id",
        "canvas_name",
        "protocol_name",
        "dish_design",
        "cross_id",
        "genotype",
        "dpf_at_acquisition",
    ):
        print(f"  {key}: {getattr(target, key)}")


def _print_candidates(candidates: list[Candidate], limit: int) -> None:
    print("Candidate models (ranked)")
    head = candidates[: max(0, limit)]
    if not head:
        print("  none")
        return
    for idx, item in enumerate(head, start=1):
        print(
            f"  {idx}. run_id={item.run_id} set_id={item.set_id} "
            f"score={item.weighted_score:.4f} datasets={item.dataset_count} "
            f"status={item.status or '-'}"
        )
        print(f"     model_path={item.model_path}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--recording-id", type=str, help="Recording ID to match.")
    parser.add_argument("--recording-dir", type=Path, help="Recording directory path to resolve recording ID.")
    parser.add_argument(
        "--task",
        choices=("detect", "pose", "eye_masks", "any"),
        default="detect",
        help="Model task family to resolve (default: detect).",
    )
    parser.add_argument("--set-id", type=str, help="Optional set_id filter.")
    parser.add_argument("--include-non-success", action="store_true", help="Include non-success model runs.")
    parser.add_argument("--require-unique", action="store_true", help="Fail if top score is tied.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of ranked candidates to print.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)

    if not args.recording_id and not args.recording_dir:
        raise SystemExit("Provide --recording-id or --recording-dir.")

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    try:
        rec_id = resolve_recording_id(
            registry,
            recording_id=args.recording_id,
            recording_dir=args.recording_dir,
        )
        target = load_target_profile(registry, rec_id)
        candidates = load_candidates(
            registry,
            target=target,
            task=str(args.task),
            set_id_filter=args.set_id,
            include_non_success=bool(args.include_non_success),
        )
    finally:
        registry.close()

    if not candidates:
        raise SystemExit("No candidate detection models found.")

    best = candidates[0]
    if args.require_unique and len(candidates) > 1:
        if abs(candidates[0].weighted_score - candidates[1].weighted_score) < 1e-12:
            raise SystemExit(
                "Top model score is tied. Re-run with --set-id to select deterministically."
            )

    payload = {
        "registry": str(registry_path),
        "task": str(args.task),
        "target": {
            "recording_id": target.recording_id,
            "recording_type": target.recording_type,
            "recording_subtype": target.recording_subtype,
            "behavior_mode": target.behavior_mode,
            "rig_id": target.rig_id,
            "arena_id": target.arena_id,
            "camera_id": target.camera_id,
            "canvas_name": target.canvas_name,
            "protocol_name": target.protocol_name,
            "dish_design": target.dish_design,
            "cross_id": target.cross_id,
            "genotype": target.genotype,
            "dpf_at_acquisition": target.dpf_at_acquisition,
        },
        "best": {
            "run_id": best.run_id,
            "set_id": best.set_id,
            "model_path": best.model_path,
            "score": best.weighted_score,
            "created_utc": best.created_utc,
            "status": best.status,
            "dataset_count": best.dataset_count,
            "feature_match_counts": best.feature_match_counts,
            "feature_weights_used": best.feature_weights_used,
        },
        "candidates": [
            {
                "run_id": item.run_id,
                "set_id": item.set_id,
                "model_path": item.model_path,
                "score": item.weighted_score,
                "created_utc": item.created_utc,
                "status": item.status,
                "dataset_count": item.dataset_count,
                "feature_match_counts": item.feature_match_counts,
                "feature_weights_used": item.feature_weights_used,
            }
            for item in candidates[: max(0, int(args.top_k))]
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    _print_target(target)
    print("")
    _print_candidates(candidates, limit=int(args.top_k))
    print("")
    print("Recommended model")
    print(f"  run_id: {best.run_id}")
    print(f"  set_id: {best.set_id}")
    print(f"  score: {best.weighted_score:.4f}")
    print(f"  model_path: {best.model_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
