#!/usr/bin/env python3
"""Publish crop-v2 from an explicit clipped refined-detection candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.crop_defaults import DEFAULT_ZEBRAFISH_CROP_SIZE_PX
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.crop_pixel_authority import (
    bind_external_video_crop_pixel_authority,
)
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.shared.zarr.crop_snapshot_publication import (
    publish_crop_geometry_from_explicit_refined_candidate,
)
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionBoundClipEvidence,
)


def publish_clipped_crop_geometry_v2(
    *,
    analysis_zarr: Path,
    refined_archive: Path,
    refined_run_id: str,
    clip_archives: Sequence[Path],
    clip_run_ids: Sequence[str],
    destination: Path,
    safe_root: Path,
    run_id: str,
    purpose: str,
    roi_width: int,
    roi_height: int,
    camera_id: str,
) -> dict[str, object]:
    if not clip_archives or len(clip_archives) != len(clip_run_ids):
        raise ValueError("--clip-archive and --clip-run require equal nonzero counts.")
    evidence: list[RefinedDetectionBoundClipEvidence] = []
    for index, (archive, clip_run) in enumerate(
        zip(clip_archives, clip_run_ids, strict=True)
    ):
        source = bind_refined_detection_crop_source(
            archive,
            run_id=clip_run,
            allow_selector_ineligible_benchmark=True,
        )
        evidence.append(
            RefinedDetectionBoundClipEvidence(
                clip_index=index,
                manifest=source.manifest,
                arrays=source.arrays,
                parent_manifest=source.parent_manifest,
                parent_arrays=source.parent_arrays,
            )
        )
    refined = bind_refined_detection_crop_source(
        refined_archive,
        run_id=refined_run_id,
        allow_selector_ineligible_benchmark=True,
        clipped_source_evidence=tuple(evidence),
    )
    lineage = refined.manifest["payload"]["snapshot_lineage"]
    recording_identity = lineage["manual_instance_key_allocator"]["recording_identity"]
    pixels = bind_external_video_crop_pixel_authority(
        analysis_zarr,
        expected_recording_identity=recording_identity,
        expected_camera_identity=camera_id,
        expected_n_frames=refined.dimensions.n_frames,
        expected_source_width=refined.dimensions.source_width,
        expected_source_height=refined.dimensions.source_height,
    )
    crop = publish_crop_geometry_from_explicit_refined_candidate(
        refined_archive=refined_archive,
        refined_run_id=refined_run_id,
        pixel_authority=pixels,
        policy=CropGeometryPolicy(
            purpose=purpose,
            size_mode=CropSizeMode.FIXED_PER_RUN,
            fixed_size_wh=(roi_width, roi_height),
            padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
        ),
        destination=destination,
        run_id=run_id,
        safe_root=safe_root,
        clipped_source_evidence=tuple(evidence),
    )
    return {
        "schema_id": "palette.clipped_crop_geometry.publication",
        "schema_version": 1,
        "status": "complete",
        "analysis_zarr": str(analysis_zarr.expanduser().resolve()),
        "refined_archive": str(refined_archive.expanduser().resolve()),
        "refined_run_id": refined_run_id,
        "output_archive": str(crop.output_path),
        "output_run_id": crop.run_id,
        "run_manifest_digest": crop.manifest["payload_digest"],
        "logical_content_digest": crop.receipt["logical_content_digest"],
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--refined-archive", type=Path, required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--clip-archive", type=Path, action="append", required=True)
    parser.add_argument("--clip-run", action="append", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument(
        "--roi-width",
        type=int,
        default=DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
    )
    parser.add_argument(
        "--roi-height",
        type=int,
        default=DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
    )
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = publish_clipped_crop_geometry_v2(
            analysis_zarr=args.analysis_zarr,
            refined_archive=args.refined_archive,
            refined_run_id=args.refined_run,
            clip_archives=args.clip_archive,
            clip_run_ids=args.clip_run,
            destination=args.destination,
            safe_root=args.safe_root,
            run_id=args.run_id,
            purpose=args.purpose,
            roi_width=args.roi_width,
            roi_height=args.roi_height,
            camera_id=args.camera_id,
        )
    except Exception as exc:
        result = {
            "schema_id": "palette.clipped_crop_geometry.publication",
            "schema_version": 1,
            "status": "failed",
            "output_archive": str(args.destination),
            "output_run_id": args.run_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    write_json_atomic(args.result_json, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
