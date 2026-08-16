"""Run one split subject-mask stage with optional node-local ROI-cache staging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.flat_roi_cache import (
    cleanup_staged_flat_roi_cache,
    stage_flat_roi_cache_manifest,
)
from fisheye.utils import run_subject_mask_batch_pipeline as pipeline

def _stage_flat_roi_cache_manifest(
    manifest_path: Path,
    *,
    staging_dir: Path | None,
) -> tuple[Path, dict[str, object]]:
    if staging_dir is None:
        raise ValueError("Subject-mask cache staging requires --roi-cache-staging-dir.")
    return stage_flat_roi_cache_manifest(
        manifest_path,
        staging_dir=staging_dir,
    )


def _pipeline_args(args: argparse.Namespace, *, cache_manifest: Path | None) -> list[str]:
    raw_worker_run = str(
        getattr(args, "raw_worker_run", None)
        or f"subject_masks_unet_registry_{args.run_label}"
    )
    refined_draft_run = str(
        getattr(args, "refined_draft_run", None)
        or f"refined_subject_masks_smart_finalizer_{args.run_label}"
    )
    legacy_crop_run = getattr(args, "crop_run", None)
    pixel_crop_run = getattr(args, "pixel_crop_run", None) or legacy_crop_run
    geometry_crop_run = getattr(args, "geometry_crop_run", None) or legacy_crop_run
    stage_crop_run = pixel_crop_run if args.stage == "inference" else geometry_crop_run
    if not stage_crop_run:
        raise ValueError(
            f"Subject-mask {args.stage} requires its exact crop authority."
        )
    command = [
        str(args.analysis_zarr),
        "--apply",
        "--registry",
        str(args.registry),
        "--run-label",
        args.run_label,
        "--subject-run-name",
        raw_worker_run,
        "--refined-run-name",
        refined_draft_run,
        "--workflow-stage",
        args.stage,
        "--subject-output-parent",
        "subject_mask_shard_runs",
        "--crop-run",
        str(stage_crop_run),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--mask-probs-dtype",
        "uint8",
        "--mask-probs-chunk-rois",
        "32",
        "--mask-probs-shard-rois",
        str(args.mask_probs_shard_rois),
        "--output-queue-size",
        "2",
        "--model-coverage-class",
        args.model_coverage_class,
        "--model-component-coverage-key",
        args.model_component_coverage_key,
        "--model-label-schema-id",
        args.model_label_schema_id,
        "--model-top-k",
        str(args.model_top_k),
        "--model-require-unique",
        "--roi-cache-policy",
        "never",
        "--profile-timings",
        "--stage-output-to-scratch",
        "--defer-registry-status",
        "--require-production-proof",
        "--metric-level",
        "cheap",
        "--mask-storage",
        "dense_uint8",
        "--mask-rle-validation-mode",
        "invariants",
        "--finalize-chunk-size",
        "256",
        "--finalize-dense-mask-row-chunk",
        "256",
        "--finalize-execution-backend",
        "process_shards",
        "--finalize-num-workers",
        str(args.finalize_num_workers),
        "--finalize-postcompute-backend",
        "process_shards",
        "--write-eye-geometry",
        "--no-write-component-contours",
        "--write-sampled-component-contours",
    ]
    model_set_id = getattr(args, "model_set_id", None)
    model_run_id = getattr(args, "model_run_id", None)
    if bool(model_set_id) != bool(model_run_id):
        raise ValueError(
            "Exact subject-mask model selection requires both --model-set-id "
            "and --model-run-id."
        )
    if model_set_id is not None and model_run_id is not None:
        command.extend(["--model-set-id", str(model_set_id)])
        command.extend(["--model-run-id", str(model_run_id)])
    model_input_size = getattr(args, "model_input_size", None)
    if model_input_size is not None:
        command.extend(["--model-input-size", str(int(model_input_size))])
    command.extend(
        [
            "--model-input-transform",
            str(getattr(args, "model_input_transform", "auto")),
        ]
    )
    if args.progress_dir is not None:
        command.extend(["--progress-dir", str(args.progress_dir)])
    if args.handoff_package_dir is not None:
        command.extend(["--handoff-package-dir", str(args.handoff_package_dir)])
    if args.stage == "inference":
        command.append("--no-assignment-keypoints")
        if geometry_crop_run != pixel_crop_run:
            command.extend(["--geometry-crop-run", str(geometry_crop_run)])
        if cache_manifest is None:
            raise ValueError("Inference requires an ROI-cache manifest.")
        command.extend(["--roi-cache-manifest", str(cache_manifest)])
    else:
        command.extend(
            [
                "--assignment-keypoint-group",
                "refined_keypoints_runs",
                "--assignment-keypoints-run",
                args.refined_keypoint_run,
                "--stage-finalization-input-to-scratch",
            ]
        )
    return command


def _cleanup_staged_cache(staged_manifest: Path) -> None:
    cleanup_staged_flat_roi_cache(staged_manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inference", "finalization"))
    parser.add_argument("--analysis-zarr", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--raw-worker-run")
    parser.add_argument("--refined-draft-run")
    parser.add_argument(
        "--crop-run",
        help=(
            "Legacy common crop authority. Prefer --pixel-crop-run plus "
            "--geometry-crop-run when cached pixels and canonical geometry differ."
        ),
    )
    parser.add_argument(
        "--pixel-crop-run",
        help="Exact crop run bound by the inference ROI-cache manifest.",
    )
    parser.add_argument(
        "--geometry-crop-run",
        help="Exact strict crop-v2 authority used for finalization/publication.",
    )
    parser.add_argument("--refined-keypoint-run")
    parser.add_argument("--roi-cache-manifest", type=Path)
    parser.add_argument("--roi-cache-staging-dir", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mask-probs-shard-rois", type=int, default=2048)
    parser.add_argument("--finalize-num-workers", type=int, default=16)
    parser.add_argument("--model-coverage-class", default="dense_all_components")
    parser.add_argument(
        "--model-component-coverage-key",
        default="body+eyes+swim_bladder",
    )
    parser.add_argument("--model-label-schema-id", default="subject_v1_union")
    parser.add_argument("--model-top-k", type=int, default=5)
    parser.add_argument("--model-set-id")
    parser.add_argument("--model-run-id")
    parser.add_argument("--model-input-size", type=int)
    parser.add_argument(
        "--model-input-transform",
        choices=("auto", "identity", "pad_to_size"),
        default="auto",
    )
    parser.add_argument("--progress-dir", type=Path)
    parser.add_argument("--handoff-package-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.pixel_crop_run or args.crop_run):
        raise ValueError("Inference pixel authority requires --pixel-crop-run.")
    if not (args.geometry_crop_run or args.crop_run):
        raise ValueError("Finalization geometry authority requires --geometry-crop-run.")
    if bool(args.model_set_id) != bool(args.model_run_id):
        raise ValueError(
            "--model-set-id and --model-run-id must be provided together."
        )
    if args.stage == "inference" and args.roi_cache_manifest is None:
        raise ValueError("Inference requires --roi-cache-manifest.")
    if args.stage == "finalization" and not args.refined_keypoint_run:
        raise ValueError("Finalization requires --refined-keypoint-run.")

    staged_manifest: Path | None = None
    staging_details: dict[str, object] = {}
    try:
        if args.stage == "inference":
            assert args.roi_cache_manifest is not None
            staged_manifest, staging_details = _stage_flat_roi_cache_manifest(
                args.roi_cache_manifest,
                staging_dir=args.roi_cache_staging_dir,
            )
        command = _pipeline_args(args, cache_manifest=staged_manifest)
        status = int(pipeline.main(command))
        if args.json:
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "status": status,
                        "analysis_zarr": str(args.analysis_zarr),
                        "run_label": args.run_label,
                        "pixel_crop_run": args.pixel_crop_run or args.crop_run,
                        "geometry_crop_run": args.geometry_crop_run or args.crop_run,
                        "refined_keypoint_run": args.refined_keypoint_run,
                        "roi_cache_staging": staging_details,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return status
    finally:
        if staged_manifest is not None:
            _cleanup_staged_cache(staged_manifest)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
