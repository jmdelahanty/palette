"""CLI wrapper for YOLO pose inference on FishEye Zarr crops."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from rich.console import Console

from ..detection.detect_keypoints_yolo import detect_keypoints_yolo


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO pose inference on FishEye Zarr crops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m fisheye.inference.predict_pose --model weights/best.pt --zarr video.zarr
  python -m fisheye.inference.predict_pose --model best.pt --zarr video.zarr --batch-size 128
""",
    )
    parser.add_argument("--model", required=True, help="Path to trained YOLO pose model (.pt)")
    parser.add_argument("--zarr", required=True, help="Path to FishEye Zarr archive")
    parser.add_argument("--run-name", help="Optional custom run name inside keypoints_runs")
    parser.add_argument("--crop-run", help="Optional crop run name to use (defaults to latest)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--device", default=None, help="Torch device string (e.g. cuda:0, cpu)")
    parser.add_argument("--imgsz", type=int, default=None, help="Override inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=1, help="Max detections per ROI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ultralytics output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    console = Console()
    detect_keypoints_yolo(
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
        verbose=args.verbose,
        console=console,
    )


if __name__ == "__main__":
    main()

