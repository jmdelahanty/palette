#!/usr/bin/env python3
"""
Interactive Prediction Viewer
An interactive dashboard with a slider to compare ground truth Zarr labels
with a trained YOLO model's annotated prediction images.
"""

import zarr
import numpy as np
import cv2
import argparse
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads for reproducibility
cv2.setNumThreads(0)  # Limit OpenCV threads for reproducibility

# Global variable to store the current frame index from the slider
frame_idx = 0

def draw_yolo_bbox(image, bbox, label, color, thickness=2):
    """Draws a single YOLO format bounding box on an image."""
    h, w = image.shape[:2]
    center_x_norm, center_y_norm, width_norm, height_norm = bbox
    
    center_x, center_y = int(center_x_norm * w), int(center_y_norm * h)
    box_w, box_h = int(width_norm * w), int(height_norm * h)
    
    x1, y1 = int(center_x - box_w / 2), int(center_y - box_h / 2)
    x2, y2 = int(x1 + box_w), int(y1 + box_h)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def on_trackbar_change(val):
    """Callback function for the slider."""
    global frame_idx
    frame_idx = val

def create_comparison_dashboard(zarr_root, annotated_frames_map, column_map):
    """Creates the visual dashboard for a given frame."""
    global frame_idx
    
    # --- 1. Get Ground Truth Data ---
    image_ds = zarr_root['raw_video/images_ds'][frame_idx]
    tracking_data = zarr_root['tracking/tracking_results'][frame_idx]
    
    # Prepare the base image (grayscale to BGR)
    base_image = np.stack([image_ds] * 3, axis=-1)
    
    # Get ground truth bbox
    gt_bbox = [
        tracking_data[column_map['bbox_x_norm_ds']],
        tracking_data[column_map['bbox_y_norm_ds']],
        tracking_data[column_map['bbox_width_norm_ds']],
        tracking_data[column_map['bbox_height_norm_ds']]
    ]
    
    # --- 2. Create Visualization Panels ---
    h, w = base_image.shape[:2]
    panel_size = (w, h)
    
    # Panel 1: Ground Truth
    gt_panel = base_image.copy()
    if not np.isnan(gt_bbox).any():
        gt_panel = draw_yolo_bbox(gt_panel, gt_bbox, "Ground Truth", (0, 255, 0))
    cv2.putText(gt_panel, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Panel 2: Model Prediction
    pred_panel = base_image.copy()
    annotated_path = annotated_frames_map.get(frame_idx)
    if annotated_path:
        # Load the pre-annotated image if it exists
        pred_panel = cv2.imread(str(annotated_path))
        if pred_panel.shape[:2] != panel_size:
            pred_panel = cv2.resize(pred_panel, panel_size)
    else:
        # Otherwise, indicate no detection was made
        cv2.putText(pred_panel, "No Detection", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(pred_panel, "Model Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
    
    # Panel 3: Overlay
    overlay_panel = base_image.copy()
    if not np.isnan(gt_bbox).any():
        overlay_panel = draw_yolo_bbox(overlay_panel, gt_bbox, "GT", (0, 255, 0))
    if annotated_path:
        # We need to read the prediction boxes to draw them
        # (For simplicity, we'll re-draw the GT box and assume the pred box is similar for vis)
        # A more advanced version would parse the prediction log file.
        temp_pred_img = cv2.imread(str(annotated_path))
        overlay_panel = cv2.addWeighted(overlay_panel, 0.7, temp_pred_img, 0.3, 0)
        
    cv2.putText(overlay_panel, "Overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Assemble the dashboard
    top_row = np.hstack((gt_panel, pred_panel))
    # Create an info panel for the bottom
    info_panel = np.zeros((100, top_row.shape[1], 3), dtype=np.uint8)
    cv2.putText(info_panel, f"Frame: {frame_idx}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Stack top row and a resized overlay panel
    final_view = np.hstack((gt_panel, pred_panel, overlay_panel))
    
    return final_view

def main(args):
    global frame_idx
    frame_idx = args.start_frame

    # --- Load Data ---
    try:
        zarr_root = zarr.open(args.zarr_path, mode='r')
        total_frames = zarr_root['raw_video/images_ds'].shape[0]
        column_names = zarr_root['tracking/tracking_results'].attrs['column_names']
        column_map = {name: i for i, name in enumerate(column_names)}
    except Exception as e:
        print(f"❌ Error loading Zarr file: {e}")
        return

    # --- Map Annotated Frames ---
    annotated_dir = Path(args.annotated_dir)
    if not annotated_dir.exists():
        print(f"❌ Annotated frames directory not found: {annotated_dir}")
        return
        
    print(f"🖼️  Mapping annotated frames from: {annotated_dir}")
    annotated_frames_map = {int(p.stem.split('_')[-1]): p for p in annotated_dir.glob("frame_*.jpg")}
    print(f"✅ Found {len(annotated_frames_map)} annotated frames.")

    # --- Create UI ---
    window_name = "Interactive Prediction Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Frame", window_name, frame_idx, total_frames - 1, on_trackbar_change)

    print("\n🚀 Starting interactive viewer. Press 'q' or Esc to quit.")
    while True:
        dashboard = create_comparison_dashboard(zarr_root, annotated_frames_map, column_map)
        cv2.imshow(window_name, dashboard)
        
        # Update the slider position to match the current frame_idx
        cv2.setTrackbarPos("Frame", window_name, frame_idx)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27: # Quit
            break
        elif key == 83: # Right arrow
            frame_idx = min(total_frames - 1, frame_idx + 1)
        elif key == 81: # Left arrow
            frame_idx = max(0, frame_idx - 1)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively compare Zarr ground truth with YOLO predictions.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file.")
    parser.add_argument("annotated_dir", type=str, help="Directory containing the annotated prediction images.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame to start on.")
    args = parser.parse_args()
    main(args)