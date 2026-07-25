#!/usr/bin/env python3
"""Resolve and content-verify the newest successful registered model."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.registry.model_resolution import TargetProfile, load_candidates
from fisheye.shared.artifact_fingerprint import fingerprint_artifact
from fisheye.shared.json_safety import write_json_atomic


SCHEMA_ID = "palette.latest_registered_model_resolution.v1"
SELECTION_POLICY = "newest_successful_registered_model_v1"


def resolve_latest_registered_model(
    registry_path: str | Path,
    *,
    task: str,
    set_id: str | None = None,
    verify_content: bool = True,
) -> dict[str, Any]:
    registry_path = Path(registry_path).expanduser().resolve()
    registry = Registry(registry_path)
    try:
        target = TargetProfile(
            recording_id="registry_latest_model_resolution",
            recording_type=None,
            recording_subtype=None,
            behavior_mode=None,
            rig_id=None,
            arena_id=None,
            camera_id=None,
            canvas_name=None,
            protocol_name=None,
            dish_design=None,
            cross_id=None,
            genotype=None,
            dpf_at_acquisition=None,
        )
        candidates = load_candidates(
            registry,
            target=target,
            task=task,
            set_id_filter=set_id,
            include_non_success=False,
        )
    finally:
        registry.close()
    if not candidates:
        raise RuntimeError(f"No successful registered {task!r} models were found.")
    # The synthetic target has no matching features, so every candidate has
    # score zero and load_candidates' deterministic date/run ordering is the
    # desired global recency order.
    if any(candidate.weighted_score != 0.0 for candidate in candidates):
        raise RuntimeError("Synthetic newest-model resolution unexpectedly received a profile score.")
    selected = candidates[0]
    model_path = Path(selected.model_path).expanduser().resolve()
    model_sha256 = str(selected.model_sha256 or "").strip().lower()
    if not model_path.is_file():
        raise FileNotFoundError(f"Registered model artifact is missing: {model_path}")
    if not re.fullmatch(r"[0-9a-f]{64}", model_sha256):
        raise RuntimeError(
            f"Registered model {selected.run_id!r} has no canonical SHA-256."
        )
    verification: dict[str, Any] | None = None
    if verify_content:
        verification = fingerprint_artifact(
            model_path,
            role=f"{task}_model",
            registry_hash=model_sha256,
        )
        actual = str(verification.get("sha256") or "").lower()
        if actual != model_sha256 or verification.get("mismatch") is True:
            raise RuntimeError(
                "Newest registered model content differs from its registry digest: "
                f"expected={model_sha256}, actual={actual or None}."
            )
    selected_payload = asdict(selected)
    selected_payload["model_path"] = str(model_path)
    selected_payload["model_sha256"] = model_sha256
    return {
        "schema_id": SCHEMA_ID,
        "status": "ok",
        "selection_policy": SELECTION_POLICY,
        "registry_path": str(registry_path),
        "task": task,
        "set_id_filter": set_id,
        "candidate_count": len(candidates),
        "selected": selected_payload,
        "content_verification": verification,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Registry SQLite path (default: Palette registry environment).",
    )
    parser.add_argument("--task", default="detect")
    parser.add_argument("--set-id")
    parser.add_argument("--no-verify-content", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    registry = (
        args.registry or RegistryPaths.from_env(Path.cwd()).path
    ).expanduser().resolve()
    try:
        result = resolve_latest_registered_model(
            registry,
            task=args.task,
            set_id=args.set_id,
            verify_content=not args.no_verify_content,
        )
    except Exception as exc:
        result = {
            "schema_id": SCHEMA_ID,
            "status": "failed",
            "registry_path": str(registry),
            "task": args.task,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.output_json:
            write_json_atomic(args.output_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    if args.output_json:
        write_json_atomic(args.output_json, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
