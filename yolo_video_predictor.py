#!/usr/bin/env python3
"""
YOLO Fish Detection Video Predictor
Uses a trained YOLO model to detect fish in raw video and create annotated output.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import time
from ultralytics import YOLO
from tqdm import tqdm


def predict_on_video(input_video_path, model_path, output_video_path, 
                    confidence_threshold=0.25, target_size=640, 
                    start_frame=0, end_frame=None, frame_skip=1):
    """
    Apply YOLO fish detection to a video and create annotated output.
    
    Args:
        input_video_path: Path to input video file
        model_path: Path to trained YOLO model (.pt file)
        output_video_path: Path for annotated output video
        confidence_threshold: Minimum confidence for detections
        target_size: Resize frames to this size (your model was trained on 640x640)
        start_frame: Starting frame index
        end_frame: Ending frame index (None for all)
        frame_skip: Process every Nth frame (1 = all frames)
    """
    
    print(f"🎯 Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"📹 Opening input video: {input_video_path}")
    cap = cv2.VideoCapture(str(input_video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {input_video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video info: {original_width}x{original_height}, {total_frames} frames, {fps:.2f} FPS")
    
    # Determine frame range
    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(end_frame, total_frames)
    
    frames_to_process = list(range(start_frame, end_frame, frame_skip))
    print(f"🎬 Processing {len(frames_to_process)} frames ({start_frame} to {end_frame}, skip={frame_skip})")
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = fps / frame_skip  # Adjust FPS based on frame skipping
    
    # Output video at target size for consistency
    output_width, output_height = target_size, target_size
    out = cv2.VideoWriter(str(output_video_path), fourcc, out_fps, (output_width, output_height))
    
    if not out.isOpened():
        print(f"❌ Error: Could not create output video: {output_video_path}")
        cap.release()
        return
    
    print(f"💾 Output video: {output_width}x{output_height}, {out_fps:.2f} FPS")
    
    # Detection statistics
    stats = {
        'total_frames': 0,
        'frames_with_detections': 0,
        'total_detections': 0,
        'confidence_scores': []
    }
    
    # Process frames
    frame_count = 0
    for target_frame_idx in tqdm(frames_to_process, desc="🐟 Detecting fish"):
        # Seek to target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠️  Warning: Could not read frame {target_frame_idx}")
            continue
        
        # Convert BGR to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to target size (what model was trained on)
        frame_resized = cv2.resize(frame_rgb, (target_size, target_size))
        
        # Run YOLO prediction
        results = model.predict(frame_resized, conf=confidence_threshold, verbose=False)
        
        # Create annotated frame
        annotated_frame = frame_resized.copy()
        
        # Process detections
        stats['total_frames'] += 1
        frame_detections = 0
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Draw confidence score and label
                label = f"Fish {confidence:.3f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background for text
                cv2.rectangle(annotated_frame, 
                            (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)), 
                            (0, 255, 0), -1)
                
                # Text
                cv2.putText(annotated_frame, label, 
                          (int(x1), int(y1) - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Update statistics
                frame_detections += 1
                stats['confidence_scores'].append(confidence)
        
        if frame_detections > 0:
            stats['frames_with_detections'] += 1
            stats['total_detections'] += frame_detections
        
        # Add frame info overlay
        info_text = f"Frame: {target_frame_idx} | Detections: {frame_detections}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Convert back to BGR for video writing
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        # Write frame to output video
        out.write(annotated_frame_bgr)
        frame_count += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    # Print statistics
    print(f"\n📊 Detection Results:")
    print(f"   Total frames processed: {stats['total_frames']}")
    print(f"   Frames with fish detected: {stats['frames_with_detections']}")
    print(f"   Detection rate: {stats['frames_with_detections']/stats['total_frames']*100:.1f}%")
    print(f"   Total fish detections: {stats['total_detections']}")
    print(f"   Average detections per frame: {stats['total_detections']/stats['total_frames']:.2f}")
    
    if stats['confidence_scores']:
        conf_scores = np.array(stats['confidence_scores'])
        print(f"   Confidence scores - Mean: {np.mean(conf_scores):.3f}, "
              f"Min: {np.min(conf_scores):.3f}, Max: {np.max(conf_scores):.3f}")
    
    print(f"✅ Annotated video saved to: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply trained YOLO fish detection model to video",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument("model_path", type=str, help="Path to trained YOLO model (.pt file)")
    parser.add_argument("output_video", type=str, help="Path for annotated output video")
    
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Confidence threshold for detections (default: 0.25)")
    parser.add_argument("--size", type=int, default=640,
                       help="Target image size for detection (default: 640)")
    parser.add_argument("--start-frame", type=int, default=0,
                       help="Starting frame index (default: 0)")
    parser.add_argument("--end-frame", type=int, default=None,
                       help="Ending frame index (default: all frames)")
    parser.add_argument("--frame-skip", type=int, default=1,
                       help="Process every Nth frame (default: 1 = all frames)")
    
    args = parser.parse_args()
    
    # Validate input files
    input_path = Path(args.input_video)
    model_path = Path(args.model_path)
    output_path = Path(args.output_video)
    
    if not input_path.exists():
        print(f"❌ Error: Input video not found: {input_path}")
        return
    
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        return
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting YOLO fish detection...")
    print(f"   Input: {input_path}")
    print(f"   Model: {model_path}")
    print(f"   Output: {output_path}")
    print(f"   Confidence threshold: {args.confidence}")
    
    start_time = time.time()
    
    predict_on_video(
        input_path, model_path, output_path,
        confidence_threshold=args.confidence,
        target_size=args.size,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_skip=args.frame_skip
    )
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()