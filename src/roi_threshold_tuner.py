#!/usr/bin/env python3
"""
ROI Threshold Tuner - Interactive tool for finding optimal tracking parameters
for individual ROI images from the crop stage.
"""

import cv2
import numpy as np
import zarr
import argparse
from pathlib import Path
import yaml
from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops

# Global variables for trackbar values
current_roi_idx = 0
roi_thresh = 115
se1_radius = 1
se2_radius = 2
current_frame = 1
current_detection = 0
use_difference = 0  # Toggle for using difference image

def update_use_difference(val):
    global use_difference
    use_difference = val

def update_roi_idx(val):
    global current_roi_idx
    current_roi_idx = val

def update_roi_thresh(val):
    global roi_thresh
    roi_thresh = val

def update_se1(val):
    global se1_radius
    se1_radius = val

def update_se2(val):
    global se2_radius  
    se2_radius = val

def update_frame(val):
    global current_frame
    current_frame = val

def update_detection(val):
    global current_detection
    current_detection = val

def process_roi_with_params(roi_image, thresh, se1_rad, se2_rad, use_diff=False, background_roi=None):
    """
    Process an ROI image with given parameters to detect the fish.
    Returns the processed image and detected regions.
    
    Args:
        roi_image: The ROI image
        thresh: Threshold value
        se1_rad: Erosion radius
        se2_rad: Dilation radius
        use_diff: If True, use difference from background
        background_roi: Background ROI for difference calculation
    """
    if roi_image is None or roi_image.size == 0:
        return None, []
    
    # Use difference image if background is available
    if use_diff and background_roi is not None:
        # This mimics what the actual tracking does
        process_image = np.clip(background_roi.astype(np.int16) - roi_image.astype(np.int16), 0, 255).astype(np.uint8)
    else:
        process_image = roi_image
    
    # Apply threshold
    _, binary = cv2.threshold(process_image, thresh, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations (matching tracking algorithm)
    se1 = disk(max(1, se1_rad))
    se2 = disk(max(1, se2_rad))
    
    # Match the tracking algorithm: erosion(dilation(erosion(...)))
    processed = erosion(dilation(erosion(binary, se1), se2), se1)
    
    # Find regions
    labeled = label(processed)
    regions = [r for r in regionprops(labeled) if r.area > 5]  # Match tracking filter
    
    return processed, regions

def create_dashboard(roi_image, background_roi, roi_thresh, se1_radius, se2_radius, roi_idx, frame_num, det_num, use_diff):
    """
    Create a visualization dashboard showing the effects of parameters.
    Optimized for finding exactly 3 blobs for keypoint detection.
    """
    if roi_image is None or roi_image.size == 0:
        # Create empty dashboard
        dashboard = np.zeros((600, 1200, 3), dtype=np.uint8)
        cv2.putText(dashboard, "No ROI data available", (400, 300),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return dashboard
    
    # Process the ROI
    processed, regions = process_roi_with_params(roi_image, roi_thresh, se1_radius, se2_radius, 
                                                 use_diff, background_roi)
    
    # Create panels
    display_size = (400, 400)
    
    # Panel 1: Original ROI or Difference Image
    if use_diff and background_roi is not None:
        diff_image = np.clip(background_roi.astype(np.int16) - roi_image.astype(np.int16), 0, 255).astype(np.uint8)
        panel1 = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2BGR)
        title1 = "Difference Image"
    else:
        panel1 = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        title1 = "Original ROI"
    panel1_resized = cv2.resize(panel1, display_size, interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel1_resized, title1, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Panel 2: Binary threshold result
    process_img = diff_image if (use_diff and background_roi is not None) else roi_image
    _, binary = cv2.threshold(process_img, roi_thresh, 255, cv2.THRESH_BINARY)
    panel2 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    panel2_resized = cv2.resize(panel2, display_size, interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel2_resized, f"Threshold: {roi_thresh}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Panel 3: After morphological operations
    if processed is not None:
        panel3 = cv2.cvtColor(processed.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        panel3 = np.zeros_like(panel2)
    panel3_resized = cv2.resize(panel3, display_size, interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel3_resized, f"Morph (SE1:{se1_radius}, SE2:{se2_radius})", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Panel 4: Detected regions overlay - COLOR CODED FOR 3-BLOB DETECTION
    panel4 = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    
    # Sort regions by area (largest first)
    sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:3]  # Take top 3
    
    # Special coloring for keypoint detection scenario
    if len(sorted_regions) == 3:
        # Perfect! We have exactly 3 blobs
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
        labels = ["Largest", "Middle", "Smallest"]
        thickness = 2
    else:
        # Not ideal for keypoint detection
        colors = [(128, 128, 128)] * len(sorted_regions)
        labels = [f"Blob {i+1}" for i in range(len(sorted_regions))]
        thickness = 1
    
    for i, region in enumerate(sorted_regions):
        color = colors[i] if i < len(colors) else (128, 128, 128)
        label = labels[i] if i < len(labels) else f"Blob {i+1}"
        
        # Draw bounding box
        minr, minc, maxr, maxc = region.bbox
        cv2.rectangle(panel4, (minc, minr), (maxc, maxr), color, thickness)
        
        # Draw centroid
        cy, cx = region.centroid
        cv2.circle(panel4, (int(cx), int(cy)), 3, color, -1)
        
        # Add label with area
        cv2.putText(panel4, f"{label}: {region.area}", (minc, minr-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Add status indicator
    status_color = (0, 255, 0) if len(sorted_regions) == 3 else (0, 0, 255)
    status_text = "✓ 3 BLOBS!" if len(sorted_regions) == 3 else f"✗ {len(regions)} blobs"
    
    panel4_resized = cv2.resize(panel4, display_size, interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel4_resized, status_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Panel 5: Statistics with keypoint detection focus
    stats_panel = np.zeros((display_size[0], display_size[1], 3), dtype=np.uint8)
    y_offset = 40
    line_height = 25
    
    stats_text = [
        f"Frame: {frame_num}",
        f"Detection: {det_num + 1}",
        f"ROI Index: {roi_idx}",
        "",
        f"Threshold: {roi_thresh}",
        f"SE1 Radius: {se1_radius}",
        f"SE2 Radius: {se2_radius}",
        f"Use Difference: {'Yes' if use_diff else 'No'}",
        "",
        f"Total Regions: {len(regions)}",
        f"Top 3 Areas: {[r.area for r in sorted_regions]}",
    ]
    
    # Add keypoint detection status
    if len(sorted_regions) == 3:
        stats_text.extend([
            "",
            "KEYPOINT READY ✓",
            "Can identify:",
            "- Swim bladder",
            "- Left eye",
            "- Right eye"
        ])
        text_color = (0, 255, 0)
    else:
        stats_text.extend([
            "",
            f"NEED 3 BLOBS (have {len(regions)})",
            "Adjust parameters to get",
            "exactly 3 distinct blobs"
        ])
        text_color = (0, 128, 255)
    
    for i, text in enumerate(stats_text):
        color = text_color if "KEYPOINT" in text or "NEED" in text else (255, 255, 255)
        cv2.putText(stats_panel, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += line_height
    
    # Panel 6: Histogram
    hist_panel = np.zeros((display_size[0], display_size[1], 3), dtype=np.uint8)
    hist = cv2.calcHist([process_img], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist = hist / hist.max() * (display_size[0] - 40)  # Normalize
    
    for i in range(256):
        cv2.line(hist_panel, 
                (i * display_size[1] // 256, display_size[0] - 20),
                (i * display_size[1] // 256, display_size[0] - 20 - int(hist[i])),
                (128, 128, 128), 1)
    
    # Draw threshold line
    thresh_x = roi_thresh * display_size[1] // 256
    cv2.line(hist_panel, (thresh_x, 0), (thresh_x, display_size[0]), (0, 255, 255), 2)
    cv2.putText(hist_panel, "Intensity Histogram", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Combine panels
    top_row = np.hstack([panel1_resized, panel2_resized, panel3_resized])
    bottom_row = np.hstack([panel4_resized, stats_panel, hist_panel])
    dashboard = np.vstack([top_row, bottom_row])
    
    return dashboard

def main(zarr_path, start_frame=1):
    global current_roi_idx, roi_thresh, se1_radius, se2_radius, current_frame, current_detection, use_difference
    
    current_frame = start_frame
    
    # Load config
    config_path = Path("src/pipeline_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            track_params = config.get('track', {})
            roi_thresh = track_params.get('roi_thresh', roi_thresh)
            se1_radius = track_params.get('se1_radius', se1_radius)
            se2_radius = track_params.get('se2_radius', se2_radius)
            print(f"Loaded initial parameters from {config_path}")
    
    # Open zarr file
    try:
        zarr_root = zarr.open_group(zarr_path, mode='r')
    except Exception as e:
        print(f"Error opening Zarr file: {e}")
        return
    
    # Get latest runs
    if 'crop_runs' not in zarr_root:
        print("Error: No crop runs found. Please run crop stage first.")
        return
    
    latest_crop_run = zarr_root['crop_runs'].attrs['latest']
    latest_detect_run = zarr_root[f'crop_runs/{latest_crop_run}'].attrs.get('source_detect_run')
    if not latest_detect_run:
        latest_detect_run = zarr_root['detect_runs'].attrs['latest']
    
    print(f"Using crop run: {latest_crop_run}")
    print(f"Using detect run: {latest_detect_run}")
    
    # Load background for difference calculation
    latest_bg_run = zarr_root['background_runs'].attrs['latest']
    bg_group = zarr_root[f'background_runs/{latest_bg_run}']
    
    # Get background (prefer full resolution)
    if 'background_full' in bg_group:
        background = bg_group['background_full'][:]
        print("Using full resolution background")
    elif 'background' in bg_group:
        background = bg_group['background'][:]
        print("Using background")
    elif 'background_ds' in bg_group:
        background_ds = bg_group['background_ds'][:]
        # Upscale if needed
        full_shape = zarr_root['raw_video/images_full'].shape[1:]
        background = cv2.resize(background_ds, (full_shape[1], full_shape[0]), 
                               interpolation=cv2.INTER_LINEAR)
        print("Using upscaled downsampled background")
    else:
        background = None
        print("Warning: No background found - difference mode will be disabled")
    
    # Load data
    roi_images = zarr_root[f'crop_runs/{latest_crop_run}/roi_images']
    roi_coordinates = zarr_root[f'crop_runs/{latest_crop_run}/roi_coordinates_full']
    n_detections = zarr_root[f'detect_runs/{latest_detect_run}/n_detections'][:]
    
    total_rois = len(roi_images)
    num_frames = len(n_detections)
    max_detections_per_frame = int(n_detections.max()) if n_detections.max() > 0 else 1
    
    print(f"\nData summary:")
    print(f"  Total ROIs: {total_rois}")
    print(f"  Total frames: {num_frames}")
    print(f"  Max detections per frame: {max_detections_per_frame}")
    
    # Create window
    window_name = "ROI Threshold Tuner - 3-Blob Keypoint Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Create trackbars
    cv2.createTrackbar("Frame", window_name, current_frame, num_frames, update_frame)
    if max_detections_per_frame > 1:
        cv2.createTrackbar("Detection", window_name, 0, max_detections_per_frame - 1, update_detection)
    cv2.createTrackbar("Threshold", window_name, roi_thresh, 255, update_roi_thresh)
    cv2.createTrackbar("SE1 Radius", window_name, se1_radius, 10, update_se1)
    cv2.createTrackbar("SE2 Radius", window_name, se2_radius, 10, update_se2)
    if background is not None:
        cv2.createTrackbar("Use Diff", window_name, 0, 1, update_use_difference)
    
    print("\n🎮 Controls:")
    print("  - Frame: Navigate through frames")
    print("  - Detection: Select which detection in multi-fish frames")
    print("  - Threshold: Adjust binary threshold")
    print("  - SE1 Radius: Erosion (noise removal)")
    print("  - SE2 Radius: Dilation (blob enhancement)")
    if background is not None:
        print("  - Use Diff: Toggle difference from background (mimics actual tracking)")
    print("  - Press 's' to save parameters")
    print("  - Press 'q' or Esc to quit")
    print("\n🎯 Goal: Get EXACTLY 3 blobs")
    print("  ✓ Green panel = 3 blobs detected (ready for keypoints)")
    print("  ✗ Red text = Wrong number of blobs")
    print("\nThe 3 blobs should correspond to:")
    print("  1. Swim bladder (largest)")
    print("  2. Left eye")
    print("  3. Right eye")
    
    while True:
        frame_idx = current_frame - 1
        n_dets_frame = n_detections[frame_idx] if frame_idx < num_frames else 0
        
        background_roi = None
        if n_dets_frame > 0:
            # Calculate ROI index for this frame/detection
            cumulative_dets = np.cumsum(np.insert(n_detections[:frame_idx+1], 0, 0))
            det_idx = min(current_detection, n_dets_frame - 1)
            roi_idx = cumulative_dets[frame_idx] + det_idx
            
            if roi_idx < total_rois:
                roi_image = roi_images[roi_idx]
                roi_coords = roi_coordinates[roi_idx]
                
                # Extract background ROI if available
                if background is not None and roi_coords[0] != -1:
                    roi_y1 = int(roi_coords[1])
                    roi_x1 = int(roi_coords[0])
                    roi_y2 = min(roi_y1 + roi_image.shape[0], background.shape[0])
                    roi_x2 = min(roi_x1 + roi_image.shape[1], background.shape[1])
                    background_roi = background[roi_y1:roi_y2, roi_x1:roi_x2]
                    
                    # Ensure same shape
                    if background_roi.shape != roi_image.shape:
                        background_roi = None
            else:
                roi_image = None
        else:
            roi_image = None
            roi_idx = -1
        
        # Create and display dashboard
        dashboard = create_dashboard(roi_image, background_roi, roi_thresh, se1_radius, se2_radius,
                                    roi_idx, current_frame, current_detection, use_difference)
        
        cv2.imshow(window_name, dashboard)
        
        # Update trackbar limits
        if n_dets_frame > 1 and max_detections_per_frame > 1:
            cv2.setTrackbarMax("Detection", window_name, n_dets_frame - 1)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            # Save parameters
            try:
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    config = {}
                
                if 'track' not in config:
                    config['track'] = {}
                
                config['track']['roi_thresh'] = roi_thresh
                config['track']['se1_radius'] = se1_radius
                config['track']['se2_radius'] = se2_radius
                
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                print(f"\n✅ Parameters saved to {config_path}")
                print(f"  roi_thresh: {roi_thresh}")
                print(f"  se1_radius: {se1_radius}")
                print(f"  se2_radius: {se2_radius}")
            except Exception as e:
                print(f"❌ Error saving parameters: {e}")
        elif key == 83:  # Right arrow
            current_frame = min(num_frames, current_frame + 1)
            cv2.setTrackbarPos("Frame", window_name, current_frame)
        elif key == 81:  # Left arrow
            current_frame = max(1, current_frame - 1)
            cv2.setTrackbarPos("Frame", window_name, current_frame)
        elif key == ord('d'):  # Toggle difference mode
            use_difference = 1 - use_difference
            if background is not None:
                cv2.setTrackbarPos("Use Diff", window_name, use_difference)
    
    cv2.destroyAllWindows()
    print("\nTuner closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROI Threshold Tuner for 3-blob keypoint detection")
    parser.add_argument("zarr_path", type=str, help="Path to Zarr file")
    parser.add_argument("start_frame", type=int, nargs='?', default=1,
                       help="Starting frame (default: 1)")
    args = parser.parse_args()
    
    main(args.zarr_path, args.start_frame)