#!/usr/bin/env python3
"""Publish and hand off only a canonical-v3 Crimson benchmark companion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.utils.finalize_crimson_canonical_v3_companion import (
    finalize_crimson_canonical_v3_companion,
)
from fisheye.utils.finalize_recording_canonical_detection_benchmark_adapter import (
    finalize_recording_canonical_detection_benchmark_adapter,
)


def publish_crimson_canonical_v3_companion(
    *,
    analysis_zarr: Path,
    detection_plan: Path,
    recording_frame_index: Path,
    recording_identity: str,
    expected_model_sha256: str,
    expected_n_frames: int,
    expected_n_instances: int,
    destination: Path,
    benchmark_root: Path,
    run_id: str,
    scratch_parent: Path,
    canonical_result: Path,
    base_handoff: Path,
    expected_base_handoff_sha256: str,
    crimson_validation_commit: str,
    expected_palette_commit: str,
    handoff_output: Path,
) -> dict[str, object]:
    """Build v3, persist its result, then issue the replacement full handoff."""

    result_path = canonical_result.expanduser().resolve()
    handoff_path = handoff_output.expanduser().resolve()
    if result_path.exists() or handoff_path.exists():
        raise FileExistsError("Canonical result and handoff outputs must be new.")
    result = finalize_recording_canonical_detection_benchmark_adapter(
        analysis_zarr=analysis_zarr,
        detection_plan_path=detection_plan,
        recording_frame_index=recording_frame_index,
        recording_identity=recording_identity,
        expected_model_sha256=expected_model_sha256,
        expected_n_frames=expected_n_frames,
        expected_n_instances=expected_n_instances,
        destination=destination,
        benchmark_root=benchmark_root,
        run_id=run_id,
        scratch_parent=scratch_parent,
        coordinate_catalog=True,
    )
    write_json_atomic(result_path, result)
    handoff = finalize_crimson_canonical_v3_companion(
        base_handoff_path=base_handoff,
        expected_base_handoff_sha256=expected_base_handoff_sha256,
        canonical_result_path=result_path,
        canonical_archive=destination,
        canonical_run=run_id,
        crimson_validation_commit=crimson_validation_commit,
        expected_palette_commit=expected_palette_commit,
        output=handoff_path,
    )
    return {
        "status": "complete",
        "canonical_result": str(result_path),
        "canonical_manifest_digest": result["run_manifest_digest"],
        "handoff": str(handoff_path),
        "handoff_payload_digest": handoff["payload_digest"],
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--detection-plan", type=Path, required=True)
    parser.add_argument("--recording-frame-index", type=Path, required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-n-frames", type=int, required=True)
    parser.add_argument("--expected-n-instances", type=int, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scratch-parent", type=Path)
    parser.add_argument("--canonical-result", type=Path, required=True)
    parser.add_argument("--base-handoff", type=Path, required=True)
    parser.add_argument("--expected-base-handoff-sha256", required=True)
    parser.add_argument("--crimson-validation-commit", required=True)
    parser.add_argument("--expected-palette-commit", required=True)
    parser.add_argument("--handoff-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scratch_parent = args.scratch_parent
    if scratch_parent is None:
        base = Path(os.environ.get("TMPDIR", "/tmp")).expanduser().resolve()
        scratch_parent = base / f"palette_{os.getuid()}"
        scratch_parent.mkdir(parents=True, exist_ok=True)
    result = publish_crimson_canonical_v3_companion(
        analysis_zarr=args.analysis_zarr,
        detection_plan=args.detection_plan,
        recording_frame_index=args.recording_frame_index,
        recording_identity=args.recording_identity,
        expected_model_sha256=args.expected_model_sha256,
        expected_n_frames=args.expected_n_frames,
        expected_n_instances=args.expected_n_instances,
        destination=args.destination,
        benchmark_root=args.benchmark_root,
        run_id=args.run_id,
        scratch_parent=scratch_parent,
        canonical_result=args.canonical_result,
        base_handoff=args.base_handoff,
        expected_base_handoff_sha256=args.expected_base_handoff_sha256,
        crimson_validation_commit=args.crimson_validation_commit,
        expected_palette_commit=args.expected_palette_commit,
        handoff_output=args.handoff_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["publish_crimson_canonical_v3_companion"]
