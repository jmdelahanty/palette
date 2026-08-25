"""Plan or publish one paired-provider exact-epoch spatial occupancy run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    load_chaser_relative_frame_source_handle,
)
from fisheye.analysis_workflows.chaser_spatial_occupancy_successor import (
    prepare_chaser_spatial_occupancy_successor_from_handles,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    build_composable_chaser_successor_publication_plan,
    load_composable_chaser_successor_source_handle,
    publish_composable_chaser_successor_run,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
    load_protocol_semantic_chaser_selection_source_handle,
)


def run(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    keypoint_relative_frame_run: str,
    detection_relative_frame_run: str,
    semantic_selection_run: str,
    keypoint_radial_run: str,
    detection_radial_run: str,
    expected_recording_id: str,
    bin_width_mm: float = 2.0,
    apply: bool = False,
    scratch_root: str | Path | None = None,
    copy_backend: str = "python",
) -> dict[str, object]:
    archive = Path(analysis_zarr).expanduser().resolve()
    relative_keypoint = load_chaser_relative_frame_source_handle(
        archive,
        run_name=keypoint_relative_frame_run,
        expected_recording_id=expected_recording_id,
        use_consolidated=True,
    )
    relative_detection = load_chaser_relative_frame_source_handle(
        archive,
        run_name=detection_relative_frame_run,
        expected_recording_id=expected_recording_id,
        use_consolidated=True,
    )
    semantic = load_protocol_semantic_chaser_selection_source_handle(
        archive,
        run_name=semantic_selection_run,
        expected_recording_id=expected_recording_id,
        use_consolidated=True,
        deep_audit=True,
    )
    radial_keypoint = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_radial_near_field",
        run_name=keypoint_radial_run,
        expected_recording_id=expected_recording_id,
        use_consolidated=True,
        deep_audit=True,
    )
    radial_detection = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_radial_near_field",
        run_name=detection_radial_run,
        expected_recording_id=expected_recording_id,
        use_consolidated=True,
        deep_audit=True,
    )
    prepared = prepare_chaser_spatial_occupancy_successor_from_handles(
        relative_keypoint,
        relative_detection,
        semantic,
        radial_keypoint,
        radial_detection,
        bin_width_mm=bin_width_mm,
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
        "position_providers": [
            {
                "provider_role": str(record["provider_role"]),
                "provider_id": str(record["provider_id"]),
                "provider_digest": str(record["provider_digest"]),
            }
            for record in prepared.manifest["sources"]["position_providers"]
        ],
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
    parser.add_argument("--keypoint-relative-frame-run", required=True)
    parser.add_argument("--detection-relative-frame-run", required=True)
    parser.add_argument("--semantic-selection-run", required=True)
    parser.add_argument("--keypoint-radial-run", required=True)
    parser.add_argument("--detection-radial-run", required=True)
    parser.add_argument("--expected-recording-id", required=True)
    parser.add_argument("--bin-width-mm", type=float, default=2.0)
    parser.add_argument("--scratch-root")
    parser.add_argument(
        "--copy-backend", choices=("python", "rsync"), default="python"
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        args.analysis_zarr,
        run_name=args.run_name,
        keypoint_relative_frame_run=args.keypoint_relative_frame_run,
        detection_relative_frame_run=args.detection_relative_frame_run,
        semantic_selection_run=args.semantic_selection_run,
        keypoint_radial_run=args.keypoint_radial_run,
        detection_radial_run=args.detection_radial_run,
        expected_recording_id=args.expected_recording_id,
        bin_width_mm=args.bin_width_mm,
        apply=args.apply,
        scratch_root=args.scratch_root,
        copy_backend=args.copy_backend,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
