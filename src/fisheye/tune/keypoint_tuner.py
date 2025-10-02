#!/usr/bin/env python3
"""
Keypoint Detection Tuner - Interactive tool for optimizing anatomical keypoint detection.

This tuner helps find optimal parameters for detecting the swim bladder and eyes
in fish ROI images. The goal is to consistently detect exactly 3 blobs that can be
identified as bladder, left eye, and right eye.

Usage:
    python keypoint_tuner.py data.zarr [start_frame]
    
Controls:
    - Arrow keys: Navigate frames
    - Trackbars: Adjust detection parameters
    - 's': Save parameters to config
    - 'd': Toggle difference image
    - 'g': Toggle geometry visualization
    - 'q' or ESC: Quit
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
current_frame = 1
current_detection = 0
roi_thresh = 50
se1_radius = 1
se2_radius = 2
min_area = 5
adaptive_steps = 5
thresh_decrement = 5
use_difference = 1  # Default to using difference (matches actual pipeline)
show_geometry = 1   # Show triangle geometry analysis

def update_frame(val):
    global current_frame
    current_frame = val

def update_detection(val):
    global current_detection
    current_detection = val

def update_roi_thresh(val):
    global roi_thresh
    roi_thresh = val

def update_se1(val):
    global se1_radius
    se1_radius = max(1, val)

def update_se2(val):
    global se2_radius
    se2_radius = max(1, val)

def update_min_area(val):
    global min_area
    min_area = max(1, val)

def update_adaptive_steps(val):
    global adaptive_steps
    adaptive_steps = max(1, val)

def update_thresh_decrement(val):
    global thresh_decrement
    thresh_decrement = max(1, val)

def update_use_difference(val):
    global use_difference
    use_difference = val

def update_show_geometry(val):
    global show_geometry
    show_geometry = val


def calculate_triangle_angles(p1, p2, p3):
    """Calculate angles at each vertex of a triangle."""
    # Calculate side lengths
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    
    angles = np.zeros(3)
    
    if b * c > 0:
        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
        angles[0] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    if a * c > 0:
        cos_angle = (a**2 + c**2 - b**2) / (2 * a * c)
        angles[1] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    if a * b > 0:
        cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
        angles[2] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.rad2deg(angles)


def identify_keypoints_by_geometry(keypoint_stats):
    """
    Identify which blob is the bladder and which are eyes.
    Returns indices in order: [bladder_idx, eye_left_idx, eye_right_idx]
    """
    if len(keypoint_stats) != 3:
        return None
    
    # Get centroid positions (y, x) -> (x, y)
    pts = np.array([s.centroid[::-1] for s in keypoint_stats])
    
    # Calculate triangle angles
    angles = calculate_triangle_angles(pts[0], pts[1], pts[2])
    kp_idx = np.argsort(angles)
    
    # Smallest angle vertex is opposite the largest angle = bladder
    bladder_idx = kp_idx[0]
    eye_indices = kp_idx[1:3]
    
    # Calculate heading
    eye_mean = np.mean(pts[eye_indices], axis=0)
    head_vec = eye_mean - pts[bladder_idx]
    heading = np.rad2deg(np.arctan2(-head_vec[1], head_vec[0]))
    
    # Rotate to determine left/right
    angle_rad = np.deg2rad(heading)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_pts = (pts - eye_mean) @ R.T
    
    if rotated_pts[eye_indices[0], 1] > rotated_pts[eye_indices[1], 1]:
        eye_r_idx, eye_l_idx = eye_indices[0], eye_indices[1]
    else:
        eye_r_idx, eye_l_idx = eye_indices[1], eye_indices[0]
    
    return {
        'bladder_idx': bladder_idx,
        'eye_l_idx': eye_l_idx,
        'eye_r_idx': eye_r_idx,
        'heading': heading,
        'angles': angles,
        'kp_idx': kp_idx
    }


def detect_keypoints_with_params(roi_image, background_roi, params):
    """
    Mimics the actual keypoint detection algorithm.
    Returns processed image, all regions, and identified keypoints.
    """
    if roi_image is None or roi_image.size == 0:
        return None, [], None
    
    # Use difference image (this is what the pipeline does)
    if params['use_diff'] and background_roi is not None:
        diff_roi = np.clip(
            background_roi.astype(np.int16) - roi_image.astype(np.int16), 
            0, 255
        ).astype(np.uint8)
    else:
        diff_roi = roi_image
    
    # Adaptive thresholding
    se1 = disk(params['se1_radius'])
    se2 = disk(params['se2_radius'])
    
    current_thresh = params['roi_thresh']
    keypoint_stats = []
    effective_thresh = current_thresh
    
    for step in range(params['adaptive_steps']):
        # Apply morphological operations (matching pipeline)
        im_roi = erosion(dilation(erosion(
            diff_roi >= current_thresh, se1), se2), se1
        )
        
        # Find regions
        roi_stat = [r for r in regionprops(label(im_roi)) 
                    if r.area > params['min_area']]
        
        if len(roi_stat) >= 3:
            keypoint_stats = sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]
            effective_thresh = current_thresh
            break
        
        current_thresh -= params['thresh_decrement']
    
    # Try to identify keypoints if we have 3 blobs
    keypoint_id = None
    if len(keypoint_stats) == 3:
        keypoint_id = identify_keypoints_by_geometry(keypoint_stats)
        if keypoint_id:
            keypoint_id['effective_thresh'] = effective_thresh
    
    # Return final processed image
    final_processed = erosion(dilation(erosion(
        diff_roi >= effective_thresh, se1), se2), se1
    )
    
    return final_processed, keypoint_stats, keypoint_id


def create_keypoint_dashboard(roi_image, background_roi, params, frame_num, det_num, roi_idx):
    """
    Create comprehensive visualization for keypoint detection tuning.
    """
    if roi_image is None or roi_image.size == 0:
        dashboard = np.zeros((800, 1600, 3), dtype=np.uint8)
        cv2.putText(dashboard, "No ROI data available", (600, 400),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return dashboard
    
    # Detect keypoints
    processed, keypoint_stats, keypoint_id = detect_keypoints_with_params(
        roi_image, background_roi, params
    )
    
    display_size = (400, 400)
    
    # ========== Panel 1: Original or Difference ==========
    if params['use_diff'] and background_roi is not None:
        diff_image = np.clip(background_roi.astype(np.int16) - roi_image.astype(np.int16), 
                            0, 255).astype(np.uint8)
        panel1 = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2BGR)
        title1 = "Difference Image"
    else:
        panel1 = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        title1 = "Original ROI"
    
    panel1 = cv2.resize(panel1, display_size, interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel1, title1, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # ========== Panel 2: After Morphological Operations ==========
    if processed is not None:
        panel2 = cv2.cvtColor(processed.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    else:
        panel2 = np.zeros((display_size[0], display_size[1], 3), dtype=np.uint8)
    
    panel2 = cv2.resize(panel2, display_size, interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel2, f"Processed (thresh={params['roi_thresh']})", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # ========== Panel 3: Keypoint Identification ==========
    panel3 = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    
    success = keypoint_id is not None and len(keypoint_stats) == 3
    
    if success:
        # Draw identified keypoints with colors
        colors = {
            'bladder': (0, 255, 0),    # Green
            'eye_l': (255, 0, 0),      # Blue  
            'eye_r': (0, 0, 255)       # Red
        }
        
        bladder = keypoint_stats[keypoint_id['bladder_idx']]
        eye_l = keypoint_stats[keypoint_id['eye_l_idx']]
        eye_r = keypoint_stats[keypoint_id['eye_r_idx']]
        
        for kp, color, name in [(bladder, colors['bladder'], 'Bladder'),
                                 (eye_l, colors['eye_l'], 'L Eye'),
                                 (eye_r, colors['eye_r'], 'R Eye')]:
            minr, minc, maxr, maxc = kp.bbox
            cy, cx = kp.centroid
            
            cv2.rectangle(panel3, (minc, minr), (maxc, maxr), color, 2)
            cv2.circle(panel3, (int(cx), int(cy)), 4, color, -1)
            cv2.putText(panel3, f"{name}", (minc, minr-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw heading vector
        bladder_pos = np.array(bladder.centroid[::-1])
        eye_mean = (np.array(eye_l.centroid[::-1]) + np.array(eye_r.centroid[::-1])) / 2
        head_vec = (eye_mean - bladder_pos) * 0.8  # Scale for visibility
        
        cv2.arrowedLine(panel3, 
                       tuple(bladder_pos.astype(int)),
                       tuple((bladder_pos + head_vec).astype(int)),
                       (255, 255, 0), 2, tipLength=0.3)
        
        status_text = f"SUCCESS! Heading: {keypoint_id['heading']:.1f}°"
        status_color = (0, 255, 0)
    else:
        # Draw all detected blobs
        for i, region in enumerate(keypoint_stats):
            minr, minc, maxr, maxc = region.bbox
            cv2.rectangle(panel3, (minc, minr), (maxc, maxr), (128, 128, 128), 1)
        
        status_text = f"FAILED: {len(keypoint_stats)} blobs"
        status_color = (0, 0, 255)
    
    panel3 = cv2.resize(panel3, display_size, interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel3, status_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # ========== Panel 4: Geometry Analysis ==========
    panel4 = np.zeros((display_size[0], display_size[1], 3), dtype=np.uint8)
    
    if success and params['show_geometry']:
        # Draw triangle geometry
        panel4_img = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        
        bladder_pos = np.array(bladder.centroid[::-1])
        eye_l_pos = np.array(eye_l.centroid[::-1])
        eye_r_pos = np.array(eye_r.centroid[::-1])
        
        # Draw triangle
        pts = np.array([bladder_pos, eye_l_pos, eye_r_pos], dtype=np.int32)
        cv2.polylines(panel4_img, [pts], True, (255, 255, 0), 2)
        
        # Draw angle arcs at each vertex
        for i, (pt, angle) in enumerate(zip([bladder_pos, eye_l_pos, eye_r_pos],
                                            keypoint_id['angles'])):
            cv2.putText(panel4_img, f"{angle:.1f}°", 
                       tuple((pt + [10, -10]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        panel4 = cv2.resize(panel4_img, display_size, interpolation=cv2.INTER_NEAREST)
        cv2.putText(panel4, "Triangle Geometry", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(panel4, "Geometry unavailable", (80, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
    
    # ========== Panel 5: Statistics ==========
    stats_panel = np.zeros((display_size[0], display_size[1], 3), dtype=np.uint8)
    y_offset = 40
    line_height = 25
    
    stats_text = [
        f"Frame: {frame_num}",
        f"Detection: {det_num + 1}",
        f"ROI Index: {roi_idx}",
        "",
        "Parameters:",
        f"  Threshold: {params['roi_thresh']}",
        f"  SE1 Radius: {params['se1_radius']}",
        f"  SE2 Radius: {params['se2_radius']}",
        f"  Min Area: {params['min_area']}",
        f"  Adaptive Steps: {params['adaptive_steps']}",
        f"  Thresh Dec: {params['thresh_decrement']}",
        "",
        f"Blobs Found: {len(keypoint_stats)}",
    ]
    
    if success:
        stats_text.extend([
            "",
            "KEYPOINT DETECTION ✓",
            f"Heading: {keypoint_id['heading']:.1f}°",
            f"Effective Thresh: {keypoint_id['effective_thresh']}",
            "",
            "Areas:",
            f"  Bladder: {bladder.area}",
            f"  Left Eye: {eye_l.area}",
            f"  Right Eye: {eye_r.area}",
        ])
    else:
        stats_text.extend([
            "",
            f"NEED 3 BLOBS!",
            f"(currently: {len(keypoint_stats)})",
            "",
            "Adjust parameters to",
            "get exactly 3 distinct",
            "high-contrast blobs"
        ])
    
    for text in stats_text:
        color = (0, 255, 0) if "✓" in text else (255, 255, 255)
        cv2.putText(stats_panel, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += line_height
    
    # ========== Panel 6: Histogram ==========
    hist_panel = np.zeros((display_size[0], display_size[1], 3), dtype=np.uint8)
    
    source_img = diff_image if (params['use_diff'] and background_roi is not None) else roi_image
    hist = cv2.calcHist([source_img], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist = hist / hist.max() * (display_size[0] - 40)
    
    for i in range(256):
        cv2.line(hist_panel,
                (i * display_size[1] // 256, display_size[0] - 20),
                (i * display_size[1] // 256, display_size[0] - 20 - int(hist[i])),
                (128, 128, 128), 1)
    
    # Draw threshold line
    thresh_x = params['roi_thresh'] * display_size[1] // 256
    cv2.line(hist_panel, (thresh_x, 0), (thresh_x, display_size[0]), (0, 255, 255), 2)
    
    if success:
        effective_x = keypoint_id['effective_thresh'] * display_size[1] // 256
        cv2.line(hist_panel, (effective_x, 0), (effective_x, display_size[0]), (0, 255, 0), 1)
    
    cv2.putText(hist_panel, "Intensity Histogram", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # ========== Combine panels ==========
    top_row = np.hstack([panel1, panel2, panel3, panel4])
    bottom_row = np.hstack([stats_panel, hist_panel, 
                           np.zeros((display_size[0], display_size[1] * 2, 3), dtype=np.uint8)])
    dashboard = np.vstack([top_row, bottom_row])
    
    return dashboard


def main(zarr_path, start_frame=1):
    global current_frame, roi_thresh, se1_radius, se2_radius
    global min_area, adaptive_steps, thresh_decrement, use_difference
    
    current_frame = start_frame
    
    # Load config
    config_path = Path("configs/fisheye/default.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            kp_params = config.get('keypoints', {})
            roi_thresh = kp_params.get('roi_thresh', roi_thresh)
            se1_radius = kp_params.get('se1_radius', se1_radius)
            se2_radius = kp_params.get('se2_radius', se2_radius)
            min_area = kp_params.get('min_area', min_area)
            adaptive_steps = kp_params.get('adaptive_steps', adaptive_steps)
            thresh_decrement = kp_params.get('thresh_decrement', thresh_decrement)
            print(f"Loaded parameters from {config_path}")
    
    # Open zarr
    try:
        zarr_root = zarr.open_group(zarr_path, mode='r')
    except Exception as e:
        print(f"Error opening Zarr: {e}")
        return
    
    # Check prerequisites
    if 'crop_runs' not in zarr_root:
        print("Error: Run crop stage first")
        return
    if 'background_runs' not in zarr_root:
        print("Error: Run background stage first")
        return
    
    # Get latest runs
    latest_crop = zarr_root['crop_runs'].attrs['latest']
    latest_bg = zarr_root['background_runs'].attrs['latest']
    latest_detect = zarr_root[f'crop_runs/{latest_crop}'].attrs.get('source_detect_run')
    
    print(f"Using crop: {latest_crop}")
    print(f"Using background: {latest_bg}")
    
    # Load background
    bg_group = zarr_root[f'background_runs/{latest_bg}']
    if 'background_full' in bg_group:
        background = bg_group['background_full'][:]
    else:
        print("Warning: No full background found")
        background = None
    
    # Load data
    roi_images = zarr_root[f'crop_runs/{latest_crop}/roi_images']
    roi_coords = zarr_root[f'crop_runs/{latest_crop}/roi_coordinates_full']
    n_detections = zarr_root[f'detect_runs/{latest_detect}/n_detections'][:]
    
    total_rois = len(roi_images)
    num_frames = len(n_detections)
    max_dets = int(n_detections.max()) if n_detections.max() > 0 else 1
    
    print(f"\nData: {total_rois} ROIs, {num_frames} frames")
    
    # Create window
    window_name = "Keypoint Detection Tuner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 800)
    
    # Create trackbars
    cv2.createTrackbar("Frame", window_name, current_frame, num_frames, update_frame)
    cv2.createTrackbar("Detection", window_name, 0, max(1, max_dets - 1), update_detection)
    cv2.createTrackbar("Threshold", window_name, roi_thresh, 255, update_roi_thresh)
    cv2.createTrackbar("SE1 Radius", window_name, se1_radius, 10, update_se1)
    cv2.createTrackbar("SE2 Radius", window_name, se2_radius, 10, update_se2)
    cv2.createTrackbar("Min Area", window_name, min_area, 50, update_min_area)
    cv2.createTrackbar("Adaptive Steps", window_name, adaptive_steps, 10, update_adaptive_steps)
    cv2.createTrackbar("Thresh Dec", window_name, thresh_decrement, 20, update_thresh_decrement)
    if background is not None:
        cv2.createTrackbar("Use Diff", window_name, use_difference, 1, update_use_difference)
    cv2.createTrackbar("Show Geometry", window_name, show_geometry, 1, update_show_geometry)
    
    print("\nControls:")
    print("  Arrow keys: Navigate frames")
    print("  s: Save parameters")
    print("  d: Toggle difference mode")
    print("  g: Toggle geometry display")
    print("  q/ESC: Quit")
    
    while True:
        frame_idx = current_frame - 1
        n_dets_frame = n_detections[frame_idx] if frame_idx < num_frames else 0
        
        if n_dets_frame > 0:
            cumulative_dets = np.cumsum(np.insert(n_detections[:frame_idx+1], 0, 0))
            det_idx = min(current_detection, n_dets_frame - 1)
            roi_idx = cumulative_dets[frame_idx] + det_idx
            
            if roi_idx < total_rois:
                roi_image = roi_images[roi_idx]
                roi_coord = roi_coords[roi_idx]
                
                # Extract background ROI
                background_roi = None
                if background is not None and roi_coord[0] != -1:
                    y1, x1 = int(roi_coord[1]), int(roi_coord[0])
                    y2, x2 = y1 + roi_image.shape[0], x1 + roi_image.shape[1]
                    background_roi = background[y1:y2, x1:x2]
                    if background_roi.shape != roi_image.shape:
                        background_roi = None
            else:
                roi_image = None
                background_roi = None
        else:
            roi_image = None
            background_roi = None
            roi_idx = -1
        
        # Create dashboard
        params = {
            'roi_thresh': roi_thresh,
            'se1_radius': se1_radius,
            'se2_radius': se2_radius,
            'min_area': min_area,
            'adaptive_steps': adaptive_steps,
            'thresh_decrement': thresh_decrement,
            'use_diff': use_difference,
            'show_geometry': show_geometry
        }
        
        dashboard = create_keypoint_dashboard(
            roi_image, background_roi, params,
            current_frame, current_detection, roi_idx
        )
        
        cv2.imshow(window_name, dashboard)
        
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
                
                if 'keypoints' not in config:
                    config['keypoints'] = {}
                
                config['keypoints'].update({
                    'roi_thresh': roi_thresh,
                    'se1_radius': se1_radius,
                    'se2_radius': se2_radius,
                    'min_area': min_area,
                    'adaptive_steps': adaptive_steps,
                    'thresh_decrement': thresh_decrement
                })
                
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                print(f"\n✓ Saved to {config_path}")
                for k, v in config['keypoints'].items():
                    print(f"  {k}: {v}")
            except Exception as e:
                print(f"Error saving: {e}")
        elif key == ord('d'):
            use_difference = 1 - use_difference
            if background is not None:
                cv2.setTrackbarPos("Use Diff", window_name, use_difference)
        elif key == ord('g'):
            show_geometry = 1 - show_geometry
            cv2.setTrackbarPos("Show Geometry", window_name, show_geometry)
        elif key == 83:  # Right arrow
            current_frame = min(num_frames, current_frame + 1)
            cv2.setTrackbarPos("Frame", window_name, current_frame)
        elif key == 81:  # Left arrow
            current_frame = max(1, current_frame - 1)
            cv2.setTrackbarPos("Frame", window_name, current_frame)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keypoint Detection Tuner")
    parser.add_argument("zarr_path", help="Path to Zarr archive")
    parser.add_argument("start_frame", type=int, nargs='?', default=1,
                       help="Starting frame (default: 1)")
    args = parser.parse_args()
    
    main(args.zarr_path, args.start_frame)