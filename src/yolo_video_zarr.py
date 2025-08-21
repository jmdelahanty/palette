#!/usr/bin/env python3
"""
Process videos with YOLO model and save detections to zarr format.
Includes top-k filtering based on confidence scores.
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
                         max_detections=100, top_k=None, device='cuda:0'):
    """
    Process video with YOLO model and save detections to zarr.
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLO model weights
        output_zarr_path: Path for output zarr file
        batch_size: Batch size for inference
        conf_threshold: Confidence threshold for detections
        max_detections: Maximum number of detections per frame (hard limit)
        top_k: If specified, keep only top-k detections by confidence per frame
        device: Device for inference ('cuda:0' or 'cpu')
    """
    
    # Load model
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Get video info
    frame_count, fps, width, height = get_video_info(video_path)
    print(f"Video info: {frame_count} frames, {fps} fps, {width}x{height}")
    
    # Determine effective max detections
    effective_max_dets = min(max_detections, top_k) if top_k else max_detections
    
    # Create zarr store
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    # Create datasets
    # Bounding boxes: [frame, detection_id, [x1, y1, x2, y2]]
    bboxes = root.create_dataset(
        'bboxes',
        shape=(frame_count, effective_max_dets, 4),
        chunks=(100, effective_max_dets, 4),
        dtype='float32',
        fill_value=-1
    )
    
    # Confidence scores: [frame, detection_id]
    scores = root.create_dataset(
        'scores',
        shape=(frame_count, effective_max_dets),
        chunks=(100, effective_max_dets),
        dtype='float32',
        fill_value=-1
    )
    
    # Class IDs: [frame, detection_id]
    class_ids = root.create_dataset(
        'class_ids',
        shape=(frame_count, effective_max_dets),
        chunks=(100, effective_max_dets),
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
    root.attrs['max_detections'] = effective_max_dets
    root.attrs['conf_threshold'] = conf_threshold
    root.attrs['top_k'] = top_k if top_k else 'None'
    root.attrs['processed_date'] = datetime.now().isoformat()
    
    # Process video
    cap = cv2.VideoCapture(str(video_path))
    
    # Statistics tracking
    total_raw_detections = 0
    total_kept_detections = 0
    frames_with_raw_detections = 0
    frames_with_kept_detections = 0
    
    print(f"Processing {frame_count} frames...")
    if top_k:
        print(f"Using top-{top_k} filtering per frame")
    
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
                raw_n_dets = len(boxes)
                
                if raw_n_dets > 0:
                    frames_with_raw_detections += 1
                    total_raw_detections += raw_n_dets
                    
                    # Get detection data
                    det_bboxes = boxes.xyxy.cpu().numpy()
                    det_scores = boxes.conf.cpu().numpy()
                    det_classes = boxes.cls.cpu().numpy().astype(int)
                    
                    # Apply top-k filtering if specified
                    if top_k and raw_n_dets > top_k:
                        # Sort by confidence scores (descending)
                        sorted_indices = np.argsort(det_scores)[::-1][:top_k]
                        
                        # Keep only top-k detections
                        det_bboxes = det_bboxes[sorted_indices]
                        det_scores = det_scores[sorted_indices]
                        det_classes = det_classes[sorted_indices]
                        n_dets = top_k
                    else:
                        n_dets = min(raw_n_dets, effective_max_dets)
                    
                    # Store the detections
                    bboxes[frame_idx, :n_dets] = det_bboxes[:n_dets]
                    scores[frame_idx, :n_dets] = det_scores[:n_dets]
                    class_ids[frame_idx, :n_dets] = det_classes[:n_dets]
                    n_detections[frame_idx] = n_dets
                    
                    if n_dets > 0:
                        frames_with_kept_detections += 1
                        total_kept_detections += n_dets
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    
    # Store class names if available
    if hasattr(model.model, 'names'):
        root.attrs['class_names'] = json.dumps(model.model.names)
    
    # Store filtering statistics
    filtering_stats = {
        'total_raw_detections': total_raw_detections,
        'total_kept_detections': total_kept_detections,
        'frames_with_raw_detections': frames_with_raw_detections,
        'frames_with_kept_detections': frames_with_kept_detections,
        'filtering_ratio': total_kept_detections / total_raw_detections if total_raw_detections > 0 else 0
    }
    root.attrs['filtering_statistics'] = json.dumps(filtering_stats)
    
    print(f"Saved detections to {output_zarr_path}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total raw detections: {total_raw_detections}")
    print(f"  Total kept detections: {total_kept_detections}")
    print(f"  Frames with raw detections: {frames_with_raw_detections}/{frame_count} "
          f"({frames_with_raw_detections/frame_count*100:.1f}%)")
    print(f"  Frames with kept detections: {frames_with_kept_detections}/{frame_count} "
          f"({frames_with_kept_detections/frame_count*100:.1f}%)")
    
    if top_k and total_raw_detections > total_kept_detections:
        print(f"  Filtering: Kept {total_kept_detections}/{total_raw_detections} detections "
              f"({total_kept_detections/total_raw_detections*100:.1f}%)")
    
    # Calculate average detections per frame
    avg_raw_per_frame = total_raw_detections / frames_with_raw_detections if frames_with_raw_detections > 0 else 0
    avg_kept_per_frame = total_kept_detections / frames_with_kept_detections if frames_with_kept_detections > 0 else 0
    
    print(f"  Avg raw detections per frame (with detections): {avg_raw_per_frame:.2f}")
    print(f"  Avg kept detections per frame (with detections): {avg_kept_per_frame:.2f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Process video with YOLO model and save detections to zarr format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s video.mp4 yolov8n.pt output.zarr
  
  # With top-k filtering (keep only top 5 detections per frame)
  %(prog)s video.mp4 model.pt output.zarr --top-k 5
  
  # Custom confidence threshold and top-k
  %(prog)s video.mp4 model.pt output.zarr --conf 0.5 --top-k 3
  
  # Specify device
  %(prog)s video.mp4 model.pt output.zarr --device cpu
        """
    )
    
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('model_path', type=str, help='Path to YOLO model weights')
    parser.add_argument('output_zarr_path', type=str, help='Path for output zarr file')
    
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for inference (default: 1)')
    parser.add_argument('--conf', '--conf-threshold', type=float, default=0.25,
                      help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--max-detections', type=int, default=100,
                      help='Maximum number of detections per frame - hard limit (default: 100)')
    parser.add_argument('--top-k', type=int, default=None,
                      help='Keep only top-k detections by confidence per frame (default: no filtering)')
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Device for inference (default: cuda:0)')
    
    args = parser.parse_args()
    
    # Validate inputs
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1
    
    # Validate top-k
    if args.top_k is not None and args.top_k < 1:
        print(f"Error: top-k must be >= 1")
        return 1
    
    # Process video
    success = process_video_to_zarr(
        video_path=args.video_path,
        model_path=args.model_path,
        output_zarr_path=args.output_zarr_path,
        batch_size=args.batch_size,
        conf_threshold=args.conf,
        max_detections=args.max_detections,
        top_k=args.top_k,
        device=args.device
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())