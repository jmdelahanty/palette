"""Bootstrap reviewable prediction surfaces for a sampled training zarr.

This is intentionally a workflow wrapper, not a new data contract. It chains
existing stage writers with deterministic run names so cluster submission
scripts do not need shell snippets to discover intermediate ``latest`` runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console

from fisheye.detection.detect_keypoints_yolo import detect_keypoints_yolo
from fisheye.refinement.finalize_subject_masks import finalize_subject_masks
from fisheye.refinement.refine_keypoints import create_refined_keypoint_run
from fisheye.segmentation import infer_unet_subject_masks


@dataclass(frozen=True)
class BootstrapReviewSurfaceResult:
    zarr_path: str
    crop_run: str
    keypoints_run: str
    refined_keypoints_run: str
    subject_masks_run: str
    refined_subject_masks_run: str
    finalize_summary: dict[str, object]


def _default_run_name(prefix: str, run_id: str) -> str:
    cleaned = str(run_id).strip().replace("/", "_")
    if not cleaned:
        raise ValueError("run_id must not be empty")
    return f"{prefix}_{cleaned}"


def bootstrap_training_review_surfaces(
    *,
    zarr_path: Path,
    crop_run: str,
    pose_model: Path,
    registry: Path,
    run_id: str,
    keypoints_run: Optional[str] = None,
    refined_keypoints_run: Optional[str] = None,
    subject_masks_run: Optional[str] = None,
    refined_subject_masks_run: Optional[str] = None,
    pose_schema: str = "traditional_v2",
    keypoint_batch_size: int = 256,
    keypoint_imgsz: int = 512,
    keypoint_device: str = "0",
    keypoint_conf: float = 0.25,
    keypoint_iou: float = 0.5,
    keypoint_max_det: int = 1,
    keypoint_input_mode: str = "tensor",
    model_input_transform: str = "auto",
    refine_scheduler: str = "threads",
    refine_num_workers: int = 4,
    subject_batch_size: int = 128,
    subject_device: str = "cuda:0",
    subject_model_input_size: int = 512,
    mask_probs_chunk_rois: int = 64,
    mask_probs_dtype: str = "uint8",
    dense_mask_row_chunk: int = 128,
    progress_jsonl: Optional[Path] = None,
    overwrite_subject_masks: bool = True,
    overwrite_refined_subject_masks: bool = True,
    console: Optional[Console] = None,
) -> BootstrapReviewSurfaceResult:
    console = console or Console()
    zarr_path = zarr_path.expanduser().resolve()
    pose_model = pose_model.expanduser().resolve()
    registry = registry.expanduser().resolve()
    keypoints_run = keypoints_run or _default_run_name("keypoints_training_review", run_id)
    refined_keypoints_run = refined_keypoints_run or _default_run_name("refined_keypoints_training_review", run_id)
    subject_masks_run = subject_masks_run or _default_run_name("subject_masks_training_review", run_id)
    refined_subject_masks_run = refined_subject_masks_run or _default_run_name(
        "refined_subject_masks_training_review",
        run_id,
    )

    console.rule("[bold cyan]Training Review Surface Bootstrap[/bold cyan]")
    console.print(f"Zarr: [cyan]{zarr_path}[/cyan]")
    console.print(f"Crop run: [cyan]{crop_run}[/cyan]")
    console.print(f"Run id: [cyan]{run_id}[/cyan]")

    console.print("[bold]1/4 Keypoint inference[/bold]")
    created_keypoints_run = detect_keypoints_yolo(
        zarr_path=str(zarr_path),
        model_path=str(pose_model),
        run_name=keypoints_run,
        crop_run=crop_run,
        pose_schema=pose_schema,
        batch_size=int(keypoint_batch_size),
        device=keypoint_device,
        imgsz=int(keypoint_imgsz),
        conf=float(keypoint_conf),
        iou=float(keypoint_iou),
        max_det=int(keypoint_max_det),
        input_mode=keypoint_input_mode,
        model_input_transform_mode=model_input_transform,
        profile_timings=True,
        registry=registry,
        console=console,
    )
    if created_keypoints_run != keypoints_run:
        raise RuntimeError(f"Expected keypoints run {keypoints_run!r}, got {created_keypoints_run!r}")

    console.print("[bold]2/4 Keypoint refinement[/bold]")
    refine_config = {
        "refine_keypoints": {
            "scheduler": refine_scheduler,
            "num_workers": int(refine_num_workers),
            "post_refinement_audit": False,
        }
    }
    created_refined_keypoints_run = create_refined_keypoint_run(
        str(zarr_path),
        keypoint_run=keypoints_run,
        config=refine_config,
        console=console,
        run_name=refined_keypoints_run,
    )
    if created_refined_keypoints_run != refined_keypoints_run:
        raise RuntimeError(
            f"Expected refined keypoints run {refined_keypoints_run!r}, got {created_refined_keypoints_run!r}"
        )

    console.print("[bold]3/4 Subject-mask inference[/bold]")
    subject_args = [
        str(zarr_path),
        "--resolve-model-from-registry",
        "--registry",
        str(registry),
        "--crop-run",
        crop_run,
        "--run-name",
        subject_masks_run,
        "--batch-size",
        str(int(subject_batch_size)),
        "--device",
        subject_device,
        "--model-input-size",
        str(int(subject_model_input_size)),
        "--model-input-transform",
        model_input_transform,
        "--mask-probs-chunk-rois",
        str(int(mask_probs_chunk_rois)),
        "--mask-probs-dtype",
        mask_probs_dtype,
        "--write-masks-roi",
        "--assignment-keypoint-group",
        "refined_keypoints_runs",
        "--assignment-keypoint-run",
        refined_keypoints_run,
        "--profile-timings",
    ]
    if overwrite_subject_masks:
        subject_args.append("--overwrite")
    infer_unet_subject_masks.main(subject_args)

    console.print("[bold]4/4 Refined subject-mask finalization[/bold]")
    finalize_summary = finalize_subject_masks(
        zarr_path,
        subject_run=subject_masks_run,
        refined_run=refined_subject_masks_run,
        components=("subject_body", "swim_bladder", "eyes_union"),
        metric_level="cheap",
        write_eye_geometry=True,
        write_component_contours=True,
        mask_storage="dense_uint8",
        dense_mask_row_chunk=int(dense_mask_row_chunk),
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoints_run=refined_keypoints_run,
        registry=registry,
        overwrite=bool(overwrite_refined_subject_masks),
        progress_jsonl=progress_jsonl,
        console=console,
    )

    result = BootstrapReviewSurfaceResult(
        zarr_path=str(zarr_path),
        crop_run=crop_run,
        keypoints_run=keypoints_run,
        refined_keypoints_run=refined_keypoints_run,
        subject_masks_run=subject_masks_run,
        refined_subject_masks_run=refined_subject_masks_run,
        finalize_summary=finalize_summary,
    )
    console.print("[green]✓[/green] Training review surface bootstrap complete")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--pose-model", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--keypoints-run")
    parser.add_argument("--refined-keypoints-run")
    parser.add_argument("--subject-masks-run")
    parser.add_argument("--refined-subject-masks-run")
    parser.add_argument("--pose-schema", default="traditional_v2")
    parser.add_argument("--keypoint-batch-size", type=int, default=256)
    parser.add_argument("--keypoint-imgsz", type=int, default=512)
    parser.add_argument("--keypoint-device", default="0")
    parser.add_argument("--keypoint-conf", type=float, default=0.25)
    parser.add_argument("--keypoint-iou", type=float, default=0.5)
    parser.add_argument("--keypoint-max-det", type=int, default=1)
    parser.add_argument("--keypoint-input-mode", default="tensor", choices=("numpy-list", "tensor", "auto"))
    parser.add_argument("--model-input-transform", default="auto", choices=("auto", "identity", "pad_to_size"))
    parser.add_argument("--refine-scheduler", default="threads", choices=("processes", "threads", "distributed"))
    parser.add_argument("--refine-num-workers", type=int, default=4)
    parser.add_argument("--subject-batch-size", type=int, default=128)
    parser.add_argument("--subject-device", default="cuda:0")
    parser.add_argument("--subject-model-input-size", type=int, default=512)
    parser.add_argument("--mask-probs-chunk-rois", type=int, default=64)
    parser.add_argument("--mask-probs-dtype", default="uint8", choices=("float16", "uint8"))
    parser.add_argument("--dense-mask-row-chunk", type=int, default=128)
    parser.add_argument("--progress-jsonl", type=Path)
    parser.add_argument("--no-overwrite-subject-masks", action="store_true")
    parser.add_argument("--no-overwrite-refined-subject-masks", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = bootstrap_training_review_surfaces(
        zarr_path=args.zarr_path,
        crop_run=args.crop_run,
        pose_model=args.pose_model,
        registry=args.registry,
        run_id=args.run_id,
        keypoints_run=args.keypoints_run,
        refined_keypoints_run=args.refined_keypoints_run,
        subject_masks_run=args.subject_masks_run,
        refined_subject_masks_run=args.refined_subject_masks_run,
        pose_schema=args.pose_schema,
        keypoint_batch_size=args.keypoint_batch_size,
        keypoint_imgsz=args.keypoint_imgsz,
        keypoint_device=args.keypoint_device,
        keypoint_conf=args.keypoint_conf,
        keypoint_iou=args.keypoint_iou,
        keypoint_max_det=args.keypoint_max_det,
        keypoint_input_mode=args.keypoint_input_mode,
        model_input_transform=args.model_input_transform,
        refine_scheduler=args.refine_scheduler,
        refine_num_workers=args.refine_num_workers,
        subject_batch_size=args.subject_batch_size,
        subject_device=args.subject_device,
        subject_model_input_size=args.subject_model_input_size,
        mask_probs_chunk_rois=args.mask_probs_chunk_rois,
        mask_probs_dtype=args.mask_probs_dtype,
        dense_mask_row_chunk=args.dense_mask_row_chunk,
        progress_jsonl=args.progress_jsonl,
        overwrite_subject_masks=not bool(args.no_overwrite_subject_masks),
        overwrite_refined_subject_masks=not bool(args.no_overwrite_refined_subject_masks),
    )
    if args.json:
        print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
