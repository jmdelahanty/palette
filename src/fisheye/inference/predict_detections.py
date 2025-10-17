"""YOLO detection inference helper.

Wraps :func:`fisheye.detection.detect_yolo.detect_yolo` to match the
`fisheye.inference` CLI layout so both detection and pose inference live in
one namespace.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console

from ..detection.detect_yolo import detect_yolo as run_detect_yolo


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO detection inference and write a detect_runs entry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal (uses config defaults)
  python -m fisheye.inference.predict_detections --video video.mp4

  # Specify model and output Zarr
  python -m fisheye.inference.predict_detections --video video.mp4 --model runs/detect/best.pt --output detections.zarr

  # Override thresholds and batch size
  python -m fisheye.inference.predict_detections --video video.mp4 --model best.pt --conf 0.35 --batch-size 64

  # Force CPU inference
  python -m fisheye.inference.predict_detections --video video.mp4 --cpu
""",
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--model", dest="model_path", help="YOLO detection model (.pt)")
    parser.add_argument("--output", dest="output_zarr", help="Output Zarr path (optional)")
    parser.add_argument("--config", dest="config_path", help="YAML config with defaults")
    parser.add_argument("--conf", dest="conf_threshold", type=float, help="Confidence threshold override")
    parser.add_argument("--iou", dest="iou_threshold", type=float, help="IoU threshold override")
    parser.add_argument("--max-det", dest="max_det", type=int, help="Max detections per frame override")
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--gpu", action="store_true", help="Force GPU inference")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    console = Console()

    if args.cpu and args.gpu:
        raise ValueError("--cpu and --gpu are mutually exclusive")
    use_gpu: Optional[bool]
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = None

    run_detect_yolo(
        video_path=str(video_path),
        model_path=args.model_path,
        output_zarr=args.output_zarr,
        config_path=args.config_path,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        max_det=args.max_det,
        batch_size=args.batch_size,
        console=console,
        use_gpu=use_gpu,
    )


if __name__ == "__main__":
    main()

