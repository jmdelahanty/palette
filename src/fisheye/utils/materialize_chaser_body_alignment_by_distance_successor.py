"""Plan or publish one exact anatomical alignment-by-distance successor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.chaser_body_alignment_by_distance_successor import (
    DISTANCE_BIN_WIDTH_MM,
    _REQUIRED_BASE_ARRAYS,
    _REQUIRED_BODY_ARRAYS,
    prepare_chaser_body_alignment_by_distance_successor_from_handles,
)
from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    load_chaser_relative_frame_source_handle,
)
from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    load_chaser_relative_frame_targeted_source_handle,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    build_composable_chaser_successor_publication_plan,
    publish_composable_chaser_successor_run,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
    load_protocol_semantic_chaser_selection_source_handle,
)


def run(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    relative_frame_run: str,
    semantic_selection_run: str,
    expected_recording_id: str,
    relative_frame_receipt: str | Path | None = None,
    semantic_selection_receipt: str | Path | None = None,
    distance_bin_width_mm: float = DISTANCE_BIN_WIDTH_MM,
    apply: bool = False,
    scratch_root: str | Path | None = None,
    copy_backend: str = "python",
) -> dict[str, object]:
    archive = Path(analysis_zarr).expanduser().resolve()
    if relative_frame_receipt is None:
        relative = load_chaser_relative_frame_source_handle(
            archive,
            run_name=relative_frame_run,
            expected_recording_id=expected_recording_id,
            use_consolidated=True,
        )
    else:
        relative = load_chaser_relative_frame_targeted_source_handle(
            relative_frame_receipt,
            required_base_arrays=_REQUIRED_BASE_ARRAYS,
            required_body_arrays=_REQUIRED_BODY_ARRAYS,
            collapsed_frame_arrays=(),
            expected_analysis_zarr=archive,
            expected_recording_id=expected_recording_id,
            expected_run_name=relative_frame_run,
        )
    semantic = load_protocol_semantic_chaser_selection_source_handle(
        archive,
        run_name=semantic_selection_run,
        expected_recording_id=expected_recording_id,
        use_consolidated=True,
        deep_audit=True,
        direct_validation_receipt=semantic_selection_receipt,
    )
    prepared = prepare_chaser_body_alignment_by_distance_successor_from_handles(
        relative,
        semantic,
        distance_bin_width_mm=distance_bin_width_mm,
    )
    plan = build_composable_chaser_successor_publication_plan(
        archive,
        run_name=run_name,
        prepared=prepared,
    )
    result: dict[str, object] = {
        "status": "dry_run_plan",
        "successor_kind": plan.successor_kind,
        "run_path": plan.run_path,
        "recording_id": plan.recording_id,
        "scientific_payload_sha256": prepared.payload_digest,
        "dimensions": dict(prepared.manifest["dimensions"]),
        "distance_bin_recipe": dict(prepared.manifest["distance_bin_recipe"]),
        "relative_frame_verification": {
            "run_path": relative.run_path,
            "verification_mode": relative.verification_mode,
            "validation_receipt_sha256": getattr(relative, "receipt_digest", None),
        },
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
    }
    if apply:
        result = publish_composable_chaser_successor_run(
            plan,
            scratch_root=scratch_root,
            copy_backend=copy_backend,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--relative-frame-run", required=True)
    parser.add_argument("--semantic-selection-run", required=True)
    parser.add_argument("--expected-recording-id", required=True)
    parser.add_argument("--relative-frame-receipt")
    parser.add_argument("--semantic-selection-receipt")
    parser.add_argument(
        "--distance-bin-width-mm", type=float, default=DISTANCE_BIN_WIDTH_MM
    )
    parser.add_argument("--scratch-root")
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        args.analysis_zarr,
        run_name=args.run_name,
        relative_frame_run=args.relative_frame_run,
        semantic_selection_run=args.semantic_selection_run,
        expected_recording_id=args.expected_recording_id,
        relative_frame_receipt=args.relative_frame_receipt,
        semantic_selection_receipt=args.semantic_selection_receipt,
        distance_bin_width_mm=args.distance_bin_width_mm,
        apply=args.apply,
        scratch_root=args.scratch_root,
        copy_backend=args.copy_backend,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
