#!/usr/bin/env python3
"""
Run a trained Ultralytics YOLO detector directly on frames stored in a Palette Zarr.

We stream frames from the downsampled arrays so the model sees exactly the same
preprocessing it was trained on (rectangular shapes, no extra letterbox step).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
import zarr
from tqdm import tqdm
from ultralytics import YOLO

from fisheye.utils.zarr_metadata import get_downsample_array_path, get_downsample_formats


def parse_imgsz(value: Optional[str]) -> Optional[Union[int, Tuple[int, int]]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError("imgsz accepts either a single int or 'height,width'.")


def frame_stream(
    images: zarr.Array,
    *,
    fmt: str,
    max_frames: Optional[int],
    progress: bool = True,
) -> Iterator[np.ndarray]:
    total = images.shape[0] if max_frames is None else min(max_frames, images.shape[0])
    iterator: Iterable[int] = range(total)
    if progress:
        iterator = tqdm(iterator, desc="Streaming frames")

    for idx in iterator:
        frame = images[idx]
        if fmt == "rgb":
            yield cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            yield cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def run_inference(
    model_path: Path,
    zarr_path: Path,
    output_dir: Path,
    *,
    confidence: float,
    max_frames: Optional[int],
    input_format: str,
    imgsz: Optional[Union[int, Tuple[int, int]]],
    save_annotated: bool,
) -> bool:
    try:
        root = zarr.open(str(zarr_path), mode="r")
        available_formats = get_downsample_formats(root)
        fmt = input_format.lower()
        if fmt not in {"gray", "rgb", "auto"}:
            fmt = "gray"
        if fmt == "auto":
            fmt = "rgb" if "rgb" in available_formats else "gray"
        array_path = get_downsample_array_path(root, format_hint=fmt)
        if array_path is None:
            print(
                f"No downsampled frames for '{fmt}' "
                f"(available: {available_formats or 'none'})."
            )
            return False
        images = root[array_path]
    except Exception as exc:
        print(f"Failed to open Zarr dataset: {exc}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model from {model_path}")
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        print(f"Unable to load model: {exc}")
        return False

    total_frames = images.shape[0] if max_frames is None else min(max_frames, images.shape[0])
    print(f"Using '{fmt}' frames ({images.shape[1]}×{images.shape[2]}). "
          f"Processing {total_frames} frames.")

    summary: List[dict] = []

    chunk_size = 32
    frame_iter = frame_stream(images, fmt=fmt, max_frames=max_frames)
    annotated_written = 0
    frames_processed = 0
    summaries: List[dict] = []

    def process_batch(batch_frames: List[np.ndarray], batch_indices: List[int]) -> None:
        nonlocal annotated_written, frames_processed
        if not batch_frames:
            return
        results_iter = model.predict(
            source=batch_frames,
            stream=True,
            conf=confidence,
            verbose=False,
            imgsz=imgsz,
        )
        for result, frame_idx in zip(results_iter, batch_indices):
            frames_processed += 1
            boxes = []
            if result.boxes is not None:
                for box in result.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    boxes.append(
                        {
                            "bbox": coords.tolist(),
                            "confidence": float(box.conf[0]),
                            "class": int(box.cls[0]),
                        }
                    )
            entry = {"frame_index": frame_idx, "boxes": boxes}
            summaries.append(entry)
            if save_annotated and boxes:
                annotated_img = result.plot()
                out_path = output_dir / f"annotated_frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), annotated_img)
                annotated_written += 1

    batch_frames: List[np.ndarray] = []
    batch_indices: List[int] = []
    for idx, frame in enumerate(frame_iter):
        batch_frames.append(frame)
        batch_indices.append(idx)
        if len(batch_frames) >= chunk_size:
            process_batch(batch_frames, batch_indices)
            batch_frames.clear()
            batch_indices.clear()

    process_batch(batch_frames, batch_indices)

    detections = sum(len(entry["boxes"]) for entry in summaries)
    frames_with_det = sum(1 for entry in summaries if entry["boxes"])

    print("\nDetection Summary")
    print(f"Frames processed : {len(summaries)}")
    print(f"Total detections  : {detections}")
    print(
        f"Images with detections: {frames_with_det}/{len(summaries)} "
        f"({frames_with_det / max(1, len(summaries)) * 100:.1f}%)"
    )
    if detections:
        confidences = [
            box["confidence"] for entry in summaries for box in entry["boxes"]
        ]
        print(
            f"Confidence scores - Avg: {np.mean(confidences):.3f}, "
            f"Min: {np.min(confidences):.3f}, Max: {np.max(confidences):.3f}"
        )

    report_path = output_dir / "detection_results.txt"
    with report_path.open("w") as fh:
        fh.write("Detection Results Summary\n" + "=" * 50 + "\n\n")
        for entry in summaries:
            fh.write(
                f"Frame: {entry['frame_index']}\n"
                f"Detections: {len(entry['boxes'])}\n"
            )
            for idx, box in enumerate(entry["boxes"], start=1):
                x1, y1, x2, y2 = box["bbox"]
                fh.write(
                    f"  - Detection {idx}: bbox=({x1:.1f}, {y1:.1f}, "
                    f"{x2:.1f}, {y2:.1f}), conf={box['confidence']:.3f}\n"
                )
            fh.write("\n")

    print(f"Detailed results saved to: {report_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model on Zarr frames.")
    parser.add_argument("model_path", type=Path, help="Path to trained weights (.pt).")
    parser.add_argument("zarr_path", type=Path, help="Path to the Zarr dataset.")
    parser.add_argument("output_dir", type=Path, help="Directory for outputs.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed.")
    parser.add_argument(
        "--input-format",
        choices=["auto", "gray", "rgb"],
        default="gray",
        help="Which downsampled frames to read.",
    )
    parser.add_argument(
        "--imgsz",
        type=str,
        default=None,
        help="imgsz override (int or 'height,width').",
    )
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Disable saving annotated frames.",
    )
    args = parser.parse_args()

    imgsz = parse_imgsz(args.imgsz)

    success = run_inference(
        model_path=args.model_path,
        zarr_path=args.zarr_path,
        output_dir=args.output_dir,
        confidence=args.confidence,
        max_frames=args.max_frames,
        input_format=args.input_format,
        imgsz=imgsz,
        save_annotated=not args.no_annotated,
    )

    if success:
        print("\nModel testing completed!")
        print(f"Results available in: {args.output_dir}")
    else:
        print("\nModel testing failed.")


if __name__ == "__main__":
    main()
