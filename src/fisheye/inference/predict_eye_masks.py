"""CLI wrapper for YOLO eye-mask inference on FishEye Zarr crops."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from rich.console import Console

from ..segmentation.eye_segmentation_yolo import segment_eye_masks_yolo


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO eye segmentation on FishEye Zarr crops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m fisheye.inference.predict_eye_masks --model weights/best.pt --zarr video.zarr
  python -m fisheye.inference.predict_eye_masks --model best.pt --zarr video.zarr --batch-size 128
""",
    )
    parser.add_argument("--model", required=True, help="Path to trained YOLO segmentation model (.pt)")
    parser.add_argument("--zarr", required=True, help="Path to FishEye Zarr archive")
    parser.add_argument("--run-name", help="Optional custom run name inside eye_masks_runs")
    parser.add_argument("--crop-run", help="Optional crop run name to use (defaults to latest)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--device", default=None, help="Torch device string (e.g. cuda:0, cpu)")
    parser.add_argument("--imgsz", type=int, default=None, help="Override inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=4, help="Max detections per ROI")
    parser.add_argument("--mask-threshold", type=float, default=0.3, help="Minimum probability floor for binarization")
    parser.add_argument(
        "--proto-upsample-factor",
        type=int,
        default=1,
        help="Optional upsampling factor applied to YOLO mask prototypes prior to reconstruction",
    )
    parser.add_argument(
        "--no-retina-masks",
        dest="use_retina_masks",
        action="store_false",
        help="Disable Ultralytics retina mask outputs (use prototype reconstruction instead)",
    )
    parser.add_argument(
        "--legacy-masks",
        action="store_true",
        help="Disable retina/soft-mask hooks and use only Ultralytics binary masks (compatibility mode)",
    )
    parser.set_defaults(use_retina_masks=True, legacy_masks=False)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ultralytics output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    console = Console()
    segment_eye_masks_yolo(
        zarr_path=args.zarr,
        model_path=args.model,
        run_name=args.run_name,
        crop_run=args.crop_run,
        batch_size=args.batch_size,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        mask_threshold=args.mask_threshold,
        use_retina_masks=args.use_retina_masks if not args.legacy_masks else False,
        proto_upsample_factor=args.proto_upsample_factor,
        legacy_masks=args.legacy_masks,
        verbose=args.verbose,
        console=console,
    )


if __name__ == "__main__":
    main()
