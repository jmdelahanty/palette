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
    - 's': Save parameters to Zarr metadata
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
from datetime import datetime, timezone
from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops

# Global variables for trackbar values
current_frame = 1
min_valid_angle = 10
min_triangle_area = 100
current_detection = 0
roi_thresh = 50
se1_radius = 1
se2_radius = 2
min_area = 5
adaptive_steps = 5
se2_decrement = 1
use_difference = 1  # Default to using difference (matches actual pipeline)
show_geometry = 1   # Show triangle geometry analysis

MIN_AREA_SLIDER_MAX = 1000
MIN_TRI_AREA_SLIDER_MAX = 2000

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

def update_se2_decrement(val):
    global se2_decrement
    se2_decrement = max(1, val)

def update_use_difference(val):
    global use_difference
    use_difference = val

def update_show_geometry(val):
    global show_geometry
    show_geometry = val

def update_min_valid_angle(val):
    global min_valid_angle
    min_valid_angle = max(1, val)

def update_min_triangle_area(val):
    global min_triangle_area
    min_triangle_area = max(1, val)

def calculate_triangle_area(p1, p2, p3):
    """Calculate area of triangle using cross product method."""
    # Vectors from p1 to p2 and p1 to p3
    v1 = p2 - p1
    v2 = p3 - p1
    # Area = 0.5 * |cross product|
    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return area

def process_roi_for_keypoints(roi_image, background_roi, params):
    """
    Process an ROI image to detect keypoints using adaptive thresholding.
    Returns processed image, all regions, and identified keypoints.
    """
    if roi_image is None or roi_image.size == 0:
        return None, [], None
    
    # Use difference image
    if params['use_diff'] and background_roi is not None:
        diff_roi = np.clip(
            background_roi.astype(np.int16) - roi_image.astype(np.int16), 
            0, 255
        ).astype(np.uint8)
    else:
        diff_roi = roi_image
    
    # Adaptive thresholding with geometry validation
    se1 = disk(params['se1_radius'])
    
    base_thresh = params['roi_thresh']
    current_se2_radius = params['se2_radius']
    se2_step = params.get('se2_decrement', 1)
    keypoint_stats = []
    effective_se2_radius = current_se2_radius
    min_angle_threshold = params.get('min_valid_angle', 10.0)
    min_triangle_area = params.get('min_triangle_area', 100.0)  # NEW
    
    for step in range(params['adaptive_steps']):
        se2_radius_int = max(1, int(round(current_se2_radius)))
        se2 = disk(se2_radius_int)
        # Apply morphological operations
        im_roi = erosion(dilation(erosion(
            diff_roi >= base_thresh, se1), se2), se1
        )
        
        # Find regions
        roi_stat = [r for r in regionprops(label(im_roi)) 
                    if r.area > params['min_area']]
        
        if len(roi_stat) >= 3:
            # Take top 3 by area
            candidate_stats = sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]
            
            # Calculate triangle geometry for validation
            centroids = np.array([r.centroid for r in candidate_stats])
            angles = calculate_triangle_angles(centroids[0], centroids[1], centroids[2])
            tri_area = calculate_triangle_area(centroids[0], centroids[1], centroids[2])
            
            # Check if angles form a valid triangle AND triangle is large enough
            if np.all(angles >= min_angle_threshold) and tri_area >= min_triangle_area:
                keypoint_stats = candidate_stats
                effective_se2_radius = se2_radius_int
                break
            # Otherwise continue adaptive search
        
        current_se2_radius = max(1, current_se2_radius - se2_step)
    
    # Try to identify keypoints if we have valid 3 blobs
    keypoint_id = None
    if len(keypoint_stats) == 3:
        keypoint_id = identify_keypoints_by_geometry(keypoint_stats)
        if keypoint_id:
            keypoint_id['effective_thresh'] = base_thresh
            keypoint_id['effective_se2_radius'] = effective_se2_radius
    
    # Return final processed image
    final_se2 = disk(max(1, int(round(effective_se2_radius))))
    final_processed = erosion(dilation(erosion(
        diff_roi >= base_thresh, se1), final_se2), se1
    )
    
    return final_processed, keypoint_stats, keypoint_id

# Add this function after process_roi_for_keypoints():

def run_adaptive_threshold_demo(roi_image, background_roi, params, window_name="Adaptive Demo"):
    """
    Run adaptive threshold step-by-step with visualization.
    Shows each threshold iteration until valid keypoints are found.
    """
    if roi_image is None or roi_image.size == 0:
        print("No ROI available for demo")
        return
    
    # Use difference image
    if params['use_diff'] and background_roi is not None:
        diff_roi = np.clip(
            background_roi.astype(np.int16) - roi_image.astype(np.int16), 
            0, 255
        ).astype(np.uint8)
    else:
        diff_roi = roi_image
    
    se1 = disk(params['se1_radius'])
    
    base_thresh = params['roi_thresh']
    current_se2_radius = params['se2_radius']
    se2_step = params.get('se2_decrement', 1)
    min_angle = params.get('min_valid_angle', 10.0)
    
    print("\n=== ADAPTIVE THRESHOLD DEMO ===")
    print(f"Starting SE2 radius: {current_se2_radius}")
    print(f"Threshold held at: {base_thresh}")
    print(f"Min valid angle: {min_angle}°")
    print(f"Max steps: {params['adaptive_steps']}")
    print("Press any key to advance to next step...")
    
    for step in range(params['adaptive_steps']):
        se2_radius_int = max(1, int(round(current_se2_radius)))
        se2 = disk(se2_radius_int)
        # Apply morphological operations
        im_roi = erosion(dilation(erosion(
            diff_roi >= base_thresh, se1), se2), se1
        )
        
        # Find regions
        roi_stat = [r for r in regionprops(label(im_roi)) 
                    if r.area > params['min_area']]
        
        # Visualize this step
        vis_panel = cv2.cvtColor((im_roi * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Draw detected blobs
        for i, region in enumerate(sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]):
            y, x = map(int, region.centroid)
            color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][i] if i < 3 else (128, 128, 128)
            cv2.circle(vis_panel, (x, y), 5, color, -1)
            cv2.putText(vis_panel, f"A:{region.area}", (x+8, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Status text
        status = (
            f"Step {step+1}/{params['adaptive_steps']} | "
            f"SE2 radius: {se2_radius_int} | Blobs: {len(roi_stat)}"
        )
        cv2.putText(vis_panel, status, (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        
        # Check if we have valid keypoints
        valid = False
        if len(roi_stat) >= 3:
            candidate_stats = sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]
            centroids = np.array([r.centroid for r in candidate_stats])
            angles = calculate_triangle_angles(centroids[0], centroids[1], centroids[2])
            tri_area = calculate_triangle_area(centroids[0], centroids[1], centroids[2])
            
            if np.all(angles >= min_angle) and tri_area >= params.get('min_triangle_area', 100):
                valid = True
                msg = (
                    f"VALID! SE2:{se2_radius_int} A:{tri_area:.1f} "
                    f"Ang:{angles[0]:.1f},{angles[1]:.1f},{angles[2]:.1f}"
                )
                color = (0, 255, 0)
            else:
                msg = f"Invalid angles: {angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}"
                color = (0, 165, 255)
        else:
            msg = f"Need 3 blobs (have {len(roi_stat)})"
            color = (0, 0, 255)
        
        cv2.putText(vis_panel, msg, (5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Show
        cv2.imshow(window_name, cv2.resize(vis_panel, (600, 600)))
        print(
            f"  Step {step+1}: SE2 radius={se2_radius_int}, "
            f"blobs={len(roi_stat)}, {msg}"
        )
        
        if valid:
            print("  ✓ Valid keypoints found!")
            cv2.waitKey(2000)  # Show for 2 seconds
            break
        
        cv2.waitKey(0)  # Wait for keypress
        current_se2_radius = max(1, current_se2_radius - se2_step)
    
    cv2.destroyWindow(window_name)
    print("=== DEMO COMPLETE ===\n")

def save_keypoint_params(zarr_path, params):
    """
    Save keypoint detection parameters to Zarr metadata ONLY.
    
    Config file remains as a template - tuned parameters go in zarr.
    
    Args:
        zarr_path: Path to zarr file
        params: Dictionary of parameters including roi_thresh, se1_radius, etc.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Save to Zarr analysis_metadata
        root = zarr.open_group(zarr_path, mode='r+')
        
        if 'analysis_metadata' not in root:
            analysis_meta = root.create_group('analysis_metadata')
        else:
            analysis_meta = root['analysis_metadata']
        
        # Get existing attrs or create new dict
        metadata = dict(analysis_meta.attrs) if analysis_meta.attrs else {}
        
        # Add/update keypoint tuning data
        metadata['keypoint_tuning'] = {
            'method': 'adaptive_threshold_morphology',
            'version': '1.0',
            'tuned_timestamp': datetime.now(timezone.utc).isoformat(),
            'tuned_parameters': {
                'roi_thresh': params['roi_thresh'],
                'se1_radius': params['se1_radius'],
                'se2_radius': params['se2_radius'],
                'min_area': params['min_area'],
                'adaptive_steps': params['adaptive_steps'],
                'se2_decrement': params['se2_decrement'],
                'min_valid_angle': params.get('min_valid_angle', 10),
                'min_triangle_area': params.get('min_triangle_area', 100),
            },
            'tuned_on_frame': params.get('frame_index', None),
            'tuned_on_detection': params.get('detection_index', None)
        }
        
        # Save back to attrs
        analysis_meta.attrs.update(metadata)
        
        print(f"\n✓ Parameters saved to zarr: {zarr_path}")
        print(f"   Location: analysis_metadata/attrs['keypoint_tuning']")
        print(f"   roi_thresh: {params['roi_thresh']}")
        print(f"   se1_radius: {params['se1_radius']}")
        print(f"   se2_radius: {params['se2_radius']}")
        print(f"   min_area: {params['min_area']}")
        print(f"   min_triangle_area: {params['min_triangle_area']}")
        print(f"   adaptive_steps: {params['adaptive_steps']}")
        print(f"   se2_decrement: {params['se2_decrement']}")
        print(f"   Tuned on frame: {params.get('frame_index', 'N/A')}")
        print(f"   Tuned on detection: {params.get('detection_index', 'N/A')}")
        
        return True, "Parameters saved to zarr"
        
    except Exception as e:
        return False, f"Error saving parameters: {e}"


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
    
    Uses geometric criteria:
    - Bladder should be at vertex with largest angle (obtuse)
    - Eyes should form acute angles with bladder
    """
    if len(keypoint_stats) != 3:
        return None
    
    # Get centroids
    centroids = np.array([r.centroid for r in keypoint_stats])
    
    # Calculate angles at each vertex
    angles = calculate_triangle_angles(centroids[0], centroids[1], centroids[2])
    
    # Bladder should be at the vertex with the largest angle (typically obtuse)
    bladder_idx = np.argmin(angles)
    eye_indices = [i for i in range(3) if i != bladder_idx]
    
    # Determine left/right eye based on x-coordinate
    eye_x = [centroids[i][1] for i in eye_indices]
    if eye_x[0] < eye_x[1]:
        left_eye_idx, right_eye_idx = eye_indices[0], eye_indices[1]
    else:
        left_eye_idx, right_eye_idx = eye_indices[1], eye_indices[0]
    
    return {
        'bladder': keypoint_stats[bladder_idx],
        'left_eye': keypoint_stats[left_eye_idx],
        'right_eye': keypoint_stats[right_eye_idx],
        'bladder_idx': bladder_idx,
        'left_eye_idx': left_eye_idx,
        'right_eye_idx': right_eye_idx,
        'angles': angles
    }


def create_keypoint_dashboard(roi_image, background_roi, params, frame_num, det_num, roi_idx):
    """
    Create comprehensive visualization for keypoint detection tuning.
    """
    display_size = (400, 400)
    
    if roi_image is None:
        # Create blank dashboard
        blank = np.zeros((display_size[1] * 2, display_size[0] * 2, 3), dtype=np.uint8)
        cv2.putText(blank, "No detection at this frame", (50, display_size[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return blank
    
    # Process the ROI
    processed, keypoint_stats, keypoint_id = process_roi_for_keypoints(
        roi_image, background_roi, params
    )
    
    # Panel 1: Original ROI
    panel1 = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel1, "Original ROI", (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Panel 2: Background ROI (if available)
    if background_roi is not None:
        panel2 = cv2.cvtColor(background_roi, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel2, "Background ROI", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        panel2 = np.zeros_like(panel1)
        cv2.putText(panel2, "No Background", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    # Panel 3: Difference image (if using diff mode)
    if params['use_diff'] and background_roi is not None:
        diff_roi = np.clip(
            background_roi.astype(np.int16) - roi_image.astype(np.int16), 
            0, 255
        ).astype(np.uint8)
        panel3 = cv2.cvtColor(diff_roi, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel3, "Difference (BG - ROI)", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        panel3 = np.zeros_like(panel1)
        cv2.putText(panel3, "Diff mode OFF", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    # Panel 4: Processed with keypoint detection
    panel4 = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # Panel 5 placeholder (rotated view)
    panel5 = np.zeros_like(panel1)
    
    # Draw detected keypoints
    color_map = {
        'bladder': (0, 255, 0),      # Green
        'left_eye': (255, 0, 0),     # Blue
        'right_eye': (0, 0, 255)     # Red
    }
    
    if keypoint_id:
        # Draw labeled keypoints
        for key, stat in [('bladder', keypoint_id['bladder']),
                         ('left_eye', keypoint_id['left_eye']),
                         ('right_eye', keypoint_id['right_eye'])]:
            y, x = map(int, stat.centroid)
            color = color_map[key]
            cv2.circle(panel4, (x, y), 5, color, -1)
            cv2.putText(panel4, key.split('_')[0][:4].upper(), (x + 8, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw triangle if geometry is enabled
        if params['show_geometry']:
            pts = np.array([
                [int(keypoint_id['bladder'].centroid[1]), int(keypoint_id['bladder'].centroid[0])],
                [int(keypoint_id['left_eye'].centroid[1]), int(keypoint_id['left_eye'].centroid[0])],
                [int(keypoint_id['right_eye'].centroid[1]), int(keypoint_id['right_eye'].centroid[0])]
            ])
            cv2.polylines(panel4, [pts], True, (255, 255, 0), 1)
            
            # Display angles
            angles = keypoint_id['angles']
            y_offset = roi_image.shape[0] - 40
            cv2.putText(panel4, f"Angles: {angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}", 
                       (5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        effective_se2 = keypoint_id.get('effective_se2_radius', params['se2_radius'])
        status = "IDENTIFIED"
        status_color = (0, 255, 0)

        # Build rotated fish frame visualization
        heading_deg = float(keypoint_id.get('heading', 0.0))
        if np.isfinite(heading_deg):
            center = (roi_image.shape[1] / 2.0, roi_image.shape[0] / 2.0)
            rot_mat = cv2.getRotationMatrix2D(center, heading_deg, 1.0)
            rotated_roi = cv2.warpAffine(
                roi_image,
                rot_mat,
                (roi_image.shape[1], roi_image.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            panel5 = cv2.cvtColor(rotated_roi.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            def rotate_point(stat):
                y, x = stat.centroid
                vec = np.array([x, y, 1.0])
                rx, ry = rot_mat @ vec
                return int(round(rx)), int(round(ry))

            for key, stat in [('bladder', keypoint_id['bladder']),
                              ('left_eye', keypoint_id['left_eye']),
                              ('right_eye', keypoint_id['right_eye'])]:
                rx, ry = rotate_point(stat)
                color = color_map[key]
                cv2.circle(panel5, (rx, ry), 5, color, -1)
                cv2.putText(panel5, key.split('_')[0][:4].upper(), (rx + 8, ry),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.putText(panel5, "Rotated ROI", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(panel5, "Rotated ROI (heading NA)", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        # Draw unidentified blobs
        for stat in keypoint_stats:
            y, x = map(int, stat.centroid)
            cv2.circle(panel4, (x, y), 5, (128, 128, 128), -1)
        
        effective_se2 = params['se2_radius']
        status = f"Found {len(keypoint_stats)} blobs (need 3)"
        status_color = (0, 165, 255)
        cv2.putText(panel5, "Rotated ROI (unavailable)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    cv2.putText(panel4, status, (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
    
    slider_info = (
        f"Thresh:{params['roi_thresh']} | SE1:{params['se1_radius']} "
        f"| SE2 slider:{params['se2_radius']} | SE2 eff:{effective_se2} "
        f"| MinArea:{params['min_area']} | MinTri:{params['min_triangle_area']}"
    )
    cv2.putText(panel4, slider_info, (10, 38),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Resize all panels
    panel1 = cv2.resize(panel1, display_size)
    panel2 = cv2.resize(panel2, display_size)
    panel3 = cv2.resize(panel3, display_size)
    panel4 = cv2.resize(panel4, display_size)
    panel5 = cv2.resize(panel5, display_size)
    blank_panel = np.zeros_like(panel1)

    # Add frame info to panel 1
    info_text = f"Frame: {frame_num}, Det: {det_num}, ROI: {roi_idx}"
    cv2.putText(panel1, info_text, (10, panel1.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Combine into 2x3 grid
    top_row = np.hstack((panel1, panel2, panel5))
    bottom_row = np.hstack((panel3, panel4, blank_panel))
    dashboard = np.vstack((top_row, bottom_row))
    
    return dashboard


def main(zarr_path, start_frame=1):
    global current_frame, roi_thresh, se1_radius, se2_radius
    global min_area, min_triangle_area, adaptive_steps, se2_decrement, use_difference, show_geometry
    
    current_frame = start_frame
    
    # Load config for initial values
    config_path = Path("configs/fisheye/default.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            kp_params = config.get('keypoints', {})
            roi_thresh = kp_params.get('roi_thresh', roi_thresh)
            se1_radius = kp_params.get('se1_radius', se1_radius)
            se2_radius = kp_params.get('se2_radius', se2_radius)
            min_area = kp_params.get('min_area', min_area)
            min_triangle_area = kp_params.get('min_triangle_area', min_triangle_area)
            adaptive_steps = kp_params.get('adaptive_steps', adaptive_steps)
            se2_decrement = kp_params.get('se2_decrement', se2_decrement)
            print(f"✓ Loaded initial parameters from {config_path}")
    
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

    # Get detect run - should be from detect_runs, not crop_runs
    if 'detect_runs' not in zarr_root:
        print("Error: No detect_runs found in zarr")
        return
    latest_detect = zarr_root['detect_runs'].attrs.get('latest')
    if not latest_detect:
        print("Error: No latest detect_run found")
        return
        
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
    # Load per-frame detection counts (n_detections or frame_counts)
    if 'n_detections' in zarr_root[f'detect_runs/{latest_detect}']:
        n_detections = zarr_root[f'detect_runs/{latest_detect}/n_detections'][:]
    elif 'frame_counts' in zarr_root[f'detect_runs/{latest_detect}']:
        n_detections = zarr_root[f'detect_runs/{latest_detect}/frame_counts'][:]
    else:
        print("Error: Neither 'n_detections' nor 'frame_counts' found in detect run")
        return
    
    total_rois = roi_images.shape[0]
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
    # Clamp globals to slider limits before creating trackbars
    min_area = int(min(min_area, MIN_AREA_SLIDER_MAX))
    min_triangle_area = int(min(min_triangle_area, MIN_TRI_AREA_SLIDER_MAX))

    cv2.createTrackbar("Min Area", window_name, int(min_area), MIN_AREA_SLIDER_MAX, update_min_area)
    cv2.createTrackbar("Adaptive Steps", window_name, adaptive_steps, 10, update_adaptive_steps)
    cv2.createTrackbar("SE2 Dec", window_name, se2_decrement, 10, update_se2_decrement)
    cv2.createTrackbar("Min Angle", window_name, min_valid_angle, 90, update_min_valid_angle)
    cv2.createTrackbar("Min Tri Area", window_name, int(min_triangle_area), MIN_TRI_AREA_SLIDER_MAX, update_min_triangle_area)
    if background is not None:
        cv2.createTrackbar("Use Diff", window_name, use_difference, 1, update_use_difference)
    cv2.createTrackbar("Show Geometry", window_name, show_geometry, 1, update_show_geometry)
    
    print("\nControls:")
    print("  Arrow keys: Navigate frames")
    print("  s: Save parameters to Zarr metadata")
    print("  d: Toggle difference mode")
    print("  g: Toggle geometry display")
    print("  a: Run adaptive threshold demo")
    print("  q/ESC: Quit")
    
    while True:
        frame_idx = current_frame - 1
        n_dets_frame = n_detections[frame_idx] if frame_idx < num_frames else 0
        
        if n_dets_frame > 0:
            # Calculate ROI index
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
            'se2_decrement': se2_decrement,
            'min_valid_angle': min_valid_angle,
            'use_diff': use_difference,
            'show_geometry': show_geometry,
            'min_triangle_area': min_triangle_area
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
            # Save parameters to Zarr
            try:
                params = {
                    'roi_thresh': roi_thresh,
                    'se1_radius': se1_radius,
                    'se2_radius': se2_radius,
                    'min_area': min_area,
                    'adaptive_steps': adaptive_steps,
                    'se2_decrement': se2_decrement,
                    'min_valid_angle': min_valid_angle,
                    'min_triangle_area': min_triangle_area,
                    'frame_index': current_frame,
                    'detection_index': current_detection
                }
                
                success, message = save_keypoint_params(zarr_path, params)
                
                if success:
                    print(f"✓ {message}")
                else:
                    print(f"✗ {message}")
                    
            except Exception as e:
                print(f"✗ Error saving: {e}")
        elif key == ord('d'):
            use_difference = 1 - use_difference
            if background is not None:
                cv2.setTrackbarPos("Use Diff", window_name, use_difference)
        elif key == ord('g'):
            show_geometry = 1 - show_geometry
            cv2.setTrackbarPos("Show Geometry", window_name, show_geometry)
        elif key == ord('a'):  # 'a' for adaptive demo
            print("\n[Starting adaptive threshold demo...]")
            run_adaptive_threshold_demo(roi_image, background_roi, params, 
                                        window_name="Adaptive Threshold Demo")
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
