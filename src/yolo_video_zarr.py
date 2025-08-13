#!/usr/bin/env python3
"""
Process videos with YOLO model and save detections to zarr format.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import zarr
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import json
from datetime import datetime


def get_video_info(video_path):
    """Get video properties."""
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frame_count, fps, width, height


def process_video_to_zarr(video_path, model_path, output_zarr_path, 
                         batch_size=1, conf_threshold=0.25, 
                         max_detections=100, device='cuda:0'):
    """
    Process video with YOLO model and save detections to zarr.
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLO model weights
        output_zarr_path: Path for output zarr file
        batch_size: Batch size for inference
        conf_threshold: Confidence threshold for detections
        max_detections: Maximum number of detections per frame
        device: Device for inference ('cuda:0' or 'cpu')
    """
    
    # Load model
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Get video info
    frame_count, fps, width, height = get_video_info(video_path)
    print(f"Video info: {frame_count} frames, {fps} fps, {width}x{height}")
    
    # Create zarr store
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    # Create datasets
    # Bounding boxes: [frame, detection_id, [x1, y1, x2, y2]]
    bboxes = root.create_dataset(
        'bboxes',
        shape=(frame_count, max_detections, 4),
        chunks=(100, max_detections, 4),
        dtype='float32',
        fill_value=-1
    )
    
    # Confidence scores: [frame, detection_id]
    scores = root.create_dataset(
        'scores',
        shape=(frame_count, max_detections),
        chunks=(100, max_detections),
        dtype='float32',
        fill_value=-1
    )
    
    # Class IDs: [frame, detection_id]
    class_ids = root.create_dataset(
        'class_ids',
        shape=(frame_count, max_detections),
        chunks=(100, max_detections),
        dtype='int32',
        fill_value=-1
    )
    
    # Number of detections per frame
    n_detections = root.create_dataset(
        'n_detections',
        shape=(frame_count,),
        chunks=(1000,),
        dtype='int32',
        fill_value=0
    )
    
    # Store metadata
    root.attrs['video_path'] = str(video_path)
    root.attrs['model_path'] = str(model_path)
    root.attrs['frame_count'] = frame_count
    root.attrs['fps'] = fps
    root.attrs['width'] = width
    root.attrs['height'] = height
    root.attrs['max_detections'] = max_detections
    root.attrs['conf_threshold'] = conf_threshold
    root.attrs['processed_date'] = datetime.now().isoformat()
    
    # Process video
    cap = cv2.VideoCapture(str(video_path))
    
    print(f"Processing {frame_count} frames...")
    with tqdm(total=frame_count) as pbar:
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = model(frame, conf=conf_threshold, device=device, verbose=False)
            
            # Extract detections
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                n_dets = min(len(boxes), max_detections)
                
                if n_dets > 0:
                    # Store bounding boxes (xyxy format)
                    bboxes[frame_idx, :n_dets] = boxes.xyxy.cpu().numpy()[:n_dets]
                    
                    # Store confidence scores
                    scores[frame_idx, :n_dets] = boxes.conf.cpu().numpy()[:n_dets]
                    
                    # Store class IDs
                    class_ids[frame_idx, :n_dets] = boxes.cls.cpu().numpy()[:n_dets].astype(int)
                    
                    # Store number of detections
                    n_detections[frame_idx] = n_dets
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    
    # Store class names if available
    if hasattr(model.model, 'names'):
        root.attrs['class_names'] = json.dumps(model.model.names)
    
    print(f"Saved detections to {output_zarr_path}")
    
    # Print summary statistics
    total_detections = n_detections[:].sum()
    frames_with_detections = (n_detections[:] > 0).sum()
    print(f"\nSummary:")
    print(f"  Total detections: {total_detections}")
    print(f"  Frames with detections: {frames_with_detections}/{frame_count}")
    print(f"  Average detections per frame: {total_detections/frame_count:.2f}")
    
    return root


def main():
    parser = argparse.ArgumentParser(description='Process video with YOLO and save to zarr')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLO model weights (.pt, .onnx, or .engine)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path for output zarr file (default: based on video name)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--max-detections', type=int, default=100,
                       help='Maximum detections per frame (default: 100)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for inference (default: cuda:0)')
    parser.add_argument('--h5-base', type=str, default=None,
                       help='Base .h5 file to derive zarr name from')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    elif args.h5_base:
        # Use h5 file base name for zarr
        h5_path = Path(args.h5_base)
        output_path = h5_path.stem + '_detections.zarr'
    else:
        # Use video file base name
        video_path = Path(args.video)
        output_path = video_path.stem + '_detections.zarr'
    
    # Process video
    process_video_to_zarr(
        video_path=args.video,
        model_path=args.model,
        output_zarr_path=output_path,
        conf_threshold=args.conf,
        max_detections=args.max_detections,
        device=args.device
    )


if __name__ == '__main__':
    main()