#!/usr/bin/env python3
"""
Lightweight helper to run a trained Ultralytics YOLO detection model on a video.

Example:
    python scripts/predict_with_ultralytics.py \
        --model runs/detect/multi_zarr_train20/weights/best.pt \
        --video /path/to/video.mp4 \
        --output runs/predict/video_eval \
        --conf 0.4
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from rich.console import Console
from ultralytics import YOLO


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Ultralytics YOLO detection inference on a video file and collect summary stats."
    )
    parser.add_argument("--model", required=True, help="Path to trained YOLO weights (e.g. runs/.../best.pt).")
    parser.add_argument("--video", required=True, help="Path to an input video file.")
    parser.add_argument(
        "--output",
        default="runs/ultralytics_predict",
        help="Directory to store Ultralytics outputs (defaults to runs/ultralytics_predict).",
    )
    parser.add_argument("--run-name", help="Optional run folder name inside --output (defaults to video stem + timestamp).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--imgsz", type=str, default="640", help="imgsz override (int or 'height,width').")
    parser.add_argument("--batch-size", type=int, default=32, help="Frames per inference batch.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed from the video.")
    parser.add_argument("--device", default=None, help="Device string passed to Ultralytics (e.g. '0', 'cpu').")
    parser.add_argument("--max-det", type=int, default=None, help="Max detections per frame.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--half", action="store_true", help="Use half precision (FP16) during inference where supported.")
    parser.add_argument("--save-frames", action="store_true", help="Save per-frame renderings produced by Ultralytics.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing run directory.")
    parser.add_argument("--verbose", action="store_true", help="Print Ultralytics per-frame logs.")
    return parser


def _validate_paths(model_path: Path, video_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")


def _parse_imgsz(value: str) -> Union[int, Tuple[int, int]]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return 640
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError("imgsz accepts either a single int or 'height,width'.")


def _frame_batches(
    video_path: Path,
    target_shape: Tuple[int, int],
    *,
    stride: int,
    batch_size: int,
    max_frames: Optional[int],
) -> Iterator[Tuple[List[np.ndarray], List[int]]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    frame_idx = 0
    yielded = 0
    try:
        batch: List[np.ndarray] = []
        indices: List[int] = []
        while True:
            if max_frames is not None and yielded >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if stride > 1 and frame_idx % stride != 0:
                frame_idx += 1
                continue
            resized = cv2.resize(frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            batch.append(resized)
            indices.append(frame_idx)
            yielded += 1
            frame_idx += 1
            if len(batch) >= batch_size:
                yield batch, indices
                batch, indices = [], []
        if batch:
            yield batch, indices
    finally:
        cap.release()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    console = Console()

    model_path = Path(args.model).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    _validate_paths(model_path, video_path)

    project_dir = Path(args.output).expanduser().resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_name = f"{video_path.stem}_{timestamp}"
    run_name = args.run_name or default_name

    console.print(f"[bold]Loading model:[/bold] {model_path}")
    model = YOLO(str(model_path))

    imgsz = _parse_imgsz(args.imgsz)
    if isinstance(imgsz, int):
        target_shape = (imgsz, imgsz)
    else:
        target_shape = (imgsz[0], imgsz[1])

    run_dir = project_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_detections = 0
    batch_iter = _frame_batches(
        video_path,
        target_shape,
        stride=max(1, args.vid_stride),
        batch_size=max(1, args.batch_size),
        max_frames=args.max_frames,
    )

    for batch_frames, batch_indices in batch_iter:
        results = model.predict(
            source=batch_frames,
            stream=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=imgsz,
            half=args.half,
            verbose=args.verbose,
            device=args.device,
            max_det=args.max_det,
        )
        for result, frame_idx in zip(results, batch_indices):
            total_frames += 1
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                total_detections += len(boxes)
            if args.save_frames:
                annotated = result.plot()
                out_path = run_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), annotated)

    console.print("\n[bold green]Inference complete[/bold green]")
    console.print(f"Frames processed : {total_frames}")
    console.print(f"Total detections  : {total_detections}")
    console.print(f"Output directory  : {run_dir}")


if __name__ == "__main__":
    main()
