"""Atomically enrich sampled detection-review training data with crop pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.zarr.training_crop_materialization_publication import (
    create_sampled_acquisition_crop_training_artifact,
    create_sampled_images_full_training_crop_artifact,
    create_training_crop_artifact,
    enrich_sampled_training_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        type=Path,
        help="Existing sampled detection-review training Zarr, or new artifact path.",
    )
    parser.add_argument(
        "--create-artifact",
        action="store_true",
        help=(
            "Copy a sampled detection-review training Zarr, enrich it, and "
            "publish the complete new artifact atomically."
        ),
    )
    parser.add_argument(
        "--base-training-zarr",
        type=Path,
        help="Required with --create-artifact; must contain sampled frames and detection review.",
    )
    parser.add_argument("--source-zarr", type=Path)
    parser.add_argument("--source-crop-run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    provider = parser.add_mutually_exclusive_group(required=True)
    provider.add_argument("--video-path", type=Path)
    provider.add_argument("--roi-cache-manifest", type=Path)
    provider.add_argument(
        "--sampled-images-full",
        action="store_true",
        help=(
            "Materialize reviewed positive detections from the copied "
            "raw_video/images_full surface. Requires --create-artifact and "
            "--refined-detect-run."
        ),
    )
    provider.add_argument(
        "--acquisition-crop-video",
        action="store_true",
        help=(
            "Use native lossless acquisition crop-video pixels, with an "
            "explicit reviewed full-frame fallback where required. Requires "
            "--create-artifact, --recording-dir, and --refined-detect-run."
        ),
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=348,
        help="Square sampled-images-full crop extent in pixels (default: 348).",
    )
    parser.add_argument("--recording-dir", type=Path)
    parser.add_argument("--crop-video-path", type=Path)
    parser.add_argument("--crop-metadata-path", type=Path)
    parser.add_argument("--crop-summary-path", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--forbid-full-frame-fallback",
        action="store_true",
        help="Fail if any reviewed row cannot use its recorded acquisition crop.",
    )
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--cache-copy-batch-rows", type=int, default=1024)
    parser.add_argument(
        "--decode-mode", choices=("auto", "sequential", "indexed"), default="auto"
    )
    parser.add_argument("--decode-chunk-frames", type=int, default=1)
    parser.add_argument(
        "--source-instance-keys",
        help="Optional comma-separated stable instance_key selection.",
    )
    parser.add_argument("--detect-run", help="Explicit base detect review run.")
    parser.add_argument(
        "--refined-detect-run",
        help="Explicit base refined-detect review run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_instance_keys = (
        [
            int(token.strip())
            for token in args.source_instance_keys.split(",")
            if token.strip()
        ]
        if args.source_instance_keys is not None
        else None
    )
    if bool(args.create_artifact) != (args.base_training_zarr is not None):
        raise SystemExit(
            "--create-artifact and --base-training-zarr must be supplied together."
        )
    if args.sampled_images_full:
        if not args.create_artifact:
            raise SystemExit(
                "--sampled-images-full requires --create-artifact and publishes a new whole artifact."
            )
        if args.refined_detect_run is None:
            raise SystemExit("--sampled-images-full requires --refined-detect-run.")
        if args.source_zarr is not None or args.source_crop_run is not None:
            raise SystemExit(
                "--sampled-images-full derives directly from the base artifact; "
                "do not pass --source-zarr or --source-crop-run."
            )
        if source_instance_keys is not None:
            raise SystemExit(
                "--sampled-images-full always materializes the complete reviewed positive rowset."
            )
        result = create_sampled_images_full_training_crop_artifact(
            destination=args.destination,
            base_training_zarr=args.base_training_zarr,
            run_id=args.run_id,
            refined_run_id=args.refined_detect_run,
            scratch_root=args.scratch_root,
            roi_size_wh=(int(args.roi_size), int(args.roi_size)),
            copy_backend=args.copy_backend,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.acquisition_crop_video:
        if not args.create_artifact:
            raise SystemExit("--acquisition-crop-video requires --create-artifact.")
        if args.refined_detect_run is None or args.recording_dir is None:
            raise SystemExit(
                "--acquisition-crop-video requires --refined-detect-run and --recording-dir."
            )
        if args.source_zarr is not None or args.source_crop_run is not None:
            raise SystemExit(
                "--acquisition-crop-video binds the reviewed base and recording sidecar; "
                "do not pass --source-zarr or --source-crop-run."
            )
        if source_instance_keys is not None:
            raise SystemExit(
                "--acquisition-crop-video always materializes the complete reviewed rowset."
            )
        result = create_sampled_acquisition_crop_training_artifact(
            destination=args.destination,
            base_training_zarr=args.base_training_zarr,
            run_id=args.run_id,
            refined_run_id=args.refined_detect_run,
            scratch_root=args.scratch_root,
            recording_dir=args.recording_dir,
            crop_video_path=args.crop_video_path,
            crop_metadata_path=args.crop_metadata_path,
            crop_summary_path=args.crop_summary_path,
            gpu_id=args.gpu_id,
            allow_full_frame_fallback=not args.forbid_full_frame_fallback,
            copy_backend=args.copy_backend,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.source_zarr is None or args.source_crop_run is None:
        raise SystemExit(
            "Video/cache providers require --source-zarr and --source-crop-run."
        )
    publisher = (
        create_training_crop_artifact
        if args.create_artifact
        else enrich_sampled_training_dataset
    )
    kwargs = dict(
        destination=args.destination,
        source_zarr=args.source_zarr,
        source_crop_run=args.source_crop_run,
        run_id=args.run_id,
        scratch_root=args.scratch_root,
        video_path=args.video_path,
        roi_cache_manifest=args.roi_cache_manifest,
        copy_backend=args.copy_backend,
        cache_copy_batch_rows=args.cache_copy_batch_rows,
        decode_mode=args.decode_mode,
        decode_chunk_frames=args.decode_chunk_frames,
        source_instance_keys=source_instance_keys,
        detect_run_id=args.detect_run,
        refined_run_id=args.refined_detect_run,
    )
    if args.create_artifact:
        kwargs["base_training_zarr"] = args.base_training_zarr
    result = publisher(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
