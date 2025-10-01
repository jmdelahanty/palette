"""
Traditional computer vision-based keypoint detection for fish tracking.
Uses morphological operations and blob detection to identify swim bladder and eyes.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops
import cv2


def detect_keypoints_traditional(
    roi: np.ndarray,
    background_roi: np.ndarray,
    roi_thresh: int = 50,
    se1_radius: int = 1,
    se2_radius: int = 2,
    min_area: int = 5,
    adaptive_steps: int = 5,
    thresh_decrement: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Detect keypoints (swim bladder and eyes) in a fish ROI using traditional CV methods.
    
    Args:
        roi: Grayscale ROI image containing the fish
        background_roi: Background model for the same ROI region
        roi_thresh: Initial threshold for blob detection
        se1_radius: Radius for first morphological structuring element
        se2_radius: Radius for second morphological structuring element
        min_area: Minimum area for valid blobs
        adaptive_steps: Number of adaptive threshold steps to try
        thresh_decrement: Amount to decrease threshold each adaptive step
    
    Returns:
        Dictionary with keypoint positions and metadata, or None if detection fails
    """
    if roi.shape != background_roi.shape:
        return None
    
    # Create structuring elements for morphological operations
    se1 = disk(se1_radius)
    se2 = disk(se2_radius)
    
    # Calculate difference image
    diff_roi = np.clip(
        background_roi.astype(np.int16) - roi.astype(np.int16), 
        0, 255
    ).astype(np.uint8)
    
    # Adaptive thresholding to find exactly 3 keypoints
    current_thresh = roi_thresh
    keypoint_stats = []
    
    for _ in range(adaptive_steps):
        # Apply morphological operations
        im_roi = erosion(dilation(erosion(
            diff_roi >= current_thresh, se1), se2), se1
        )
        
        # Find and filter blobs
        roi_stat = [r for r in regionprops(label(im_roi)) if r.area > min_area]
        
        if len(roi_stat) >= 3:
            # Take the 3 largest blobs
            keypoint_stats = sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]
            break
            
        current_thresh -= thresh_decrement
    
    if len(keypoint_stats) != 3:
        return None
    
    # Extract positions and identify keypoints
    keypoints = identify_keypoints_by_geometry(keypoint_stats)
    if keypoints is None:
        return None
    
    # Add detection metadata
    keypoints['confidence'] = calculate_confidence(keypoint_stats)
    keypoints['effective_threshold'] = current_thresh
    keypoints['num_blobs_found'] = len(keypoint_stats)
    
    return keypoints


def identify_keypoints_by_geometry(
    keypoint_stats: List[Any]
) -> Optional[Dict[str, np.ndarray]]:
    """
    Identify which blob is the swim bladder and which are the eyes based on geometry.
    
    Args:
        keypoint_stats: List of exactly 3 regionprops objects
        
    Returns:
        Dictionary with 'bladder', 'eye_left', 'eye_right' positions and 'heading'
    """
    if len(keypoint_stats) != 3:
        return None
    
    # Get centroid positions (convert from (y,x) to (x,y))
    pts = np.array([s.centroid[::-1] for s in keypoint_stats])
    
    # Calculate triangle angles to identify the bladder (vertex with largest angle)
    angles, _ = calculate_triangle_metrics(pts[0], pts[1], pts[2])
    kp_idx = np.argsort(angles)
    
    bladder_idx = kp_idx[0]  # Vertex with smallest angle (bladder is opposite to largest angle)
    eye_indices = kp_idx[1:3]
    
    # Calculate heading based on bladder-to-eyes vector
    eye_mean = np.mean(pts[eye_indices], axis=0)
    head_vec = eye_mean - pts[bladder_idx]
    heading = np.rad2deg(np.arctan2(-head_vec[1], head_vec[0]))
    
    # Determine left/right eyes by rotating to heading-aligned coordinates
    R = rotation_matrix_2d(heading)
    rotated_pts = (pts - eye_mean) @ R.T
    
    # In rotated coordinates, left eye has negative y, right eye has positive y
    if rotated_pts[eye_indices[0], 1] > rotated_pts[eye_indices[1], 1]:
        eye_r_idx, eye_l_idx = eye_indices[0], eye_indices[1]
    else:
        eye_r_idx, eye_l_idx = eye_indices[1], eye_indices[0]
    
    return {
        'bladder': pts[bladder_idx],
        'eye_left': pts[eye_l_idx],
        'eye_right': pts[eye_r_idx],
        'heading': heading,
        'bladder_stats': keypoint_stats[bladder_idx],
        'eye_left_stats': keypoint_stats[eye_l_idx],
        'eye_right_stats': keypoint_stats[eye_r_idx]
    }


def calculate_triangle_metrics(
    p1: np.ndarray, 
    p2: np.ndarray, 
    p3: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Calculate angles and area of a triangle formed by three points.
    
    Args:
        p1, p2, p3: 2D points as numpy arrays
        
    Returns:
        Tuple of (angles in radians, triangle area)
    """
    # Calculate side lengths
    a = np.linalg.norm(p2 - p3)  # Side opposite to p1
    b = np.linalg.norm(p1 - p3)  # Side opposite to p2
    c = np.linalg.norm(p1 - p2)  # Side opposite to p3
    
    # Calculate angles using law of cosines
    angles = np.zeros(3)
    
    # Angle at p1
    if b * c > 0:
        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
        angles[0] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Angle at p2
    if a * c > 0:
        cos_angle = (a**2 + c**2 - b**2) / (2 * a * c)
        angles[1] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Angle at p3
    if a * b > 0:
        cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
        angles[2] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Calculate area using Heron's formula
    s = (a + b + c) / 2  # Semi-perimeter
    area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
    
    return angles, area


def rotation_matrix_2d(angle_degrees: float) -> np.ndarray:
    """
    Create a 2D rotation matrix for the given angle.
    
    Args:
        angle_degrees: Rotation angle in degrees
        
    Returns:
        2x2 rotation matrix
    """
    angle_rad = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])


def calculate_confidence(keypoint_stats: List[Any]) -> float:
    """
    Calculate confidence score based on keypoint properties.
    
    Args:
        keypoint_stats: List of regionprops objects for detected keypoints
        
    Returns:
        Confidence score between 0 and 1
    """
    if not keypoint_stats:
        return 0.0
    
    # Base confidence on average area (larger = more confident)
    mean_area = np.mean([s.area for s in keypoint_stats])
    
    # Normalize to 0-1 range (assuming typical area range)
    confidence = min(1.0, mean_area / 100.0)
    
    # Additional factors could be added here:
    # - Consistency of blob shapes (eccentricity)
    # - Distance ratios between keypoints
    # - Intensity consistency
    
    return confidence


def calculate_bounding_box(
    keypoints: Dict[str, np.ndarray],
    roi_shape: Tuple[int, int],
    margin_factor: float = 1.5,
    min_bbox_fraction: float = 0.05
) -> Dict[str, np.ndarray]:
    """
    Calculate a bounding box around detected keypoints.
    
    Args:
        keypoints: Dictionary with 'bladder', 'eye_left', 'eye_right' positions
        roi_shape: Shape of the ROI (height, width)
        margin_factor: Factor to expand bbox beyond keypoints
        min_bbox_fraction: Minimum bbox size as fraction of ROI
        
    Returns:
        Dictionary with bbox center and extent in normalized coordinates
    """
    # Collect all keypoint positions
    positions = np.array([
        keypoints['bladder'],
        keypoints['eye_left'],
        keypoints['eye_right']
    ])
    
    # Find bounding box of keypoints
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    
    # Calculate center and extent
    center_px = (min_pos + max_pos) / 2.0
    keypoint_extent_px = max_pos - min_pos
    
    # Add margin
    margin_px = keypoint_extent_px * (margin_factor - 1.0)
    bbox_extent_px = keypoint_extent_px + margin_px
    
    # Enforce minimum size
    min_size_px = np.array(roi_shape[::-1]) * min_bbox_fraction
    bbox_extent_px = np.maximum(bbox_extent_px, min_size_px)
    
    # Convert to normalized coordinates
    roi_size = np.array(roi_shape[::-1])  # (width, height)
    center_norm = center_px / roi_size
    extent_norm = bbox_extent_px / roi_size
    
    # Clip to valid range
    extent_norm = np.minimum(extent_norm, [1.0, 1.0])
    
    return {
        'center_norm': center_norm,
        'extent_norm': extent_norm,
        'center_px': center_px,
        'extent_px': bbox_extent_px
    }


def transform_keypoints_to_image_coords(
    keypoints: Dict[str, np.ndarray],
    roi_coords: Tuple[int, int],
    roi_shape: Tuple[int, int],
    image_shape: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """
    Transform keypoint coordinates from ROI space to full image space.
    
    Args:
        keypoints: Keypoint positions in ROI coordinates
        roi_coords: Top-left corner of ROI in image (x, y)
        roi_shape: Shape of the ROI (height, width)
        image_shape: Shape of the full image (height, width)
        
    Returns:
        Dictionary with keypoints in both pixel and normalized image coordinates
    """
    result = {}
    
    for key in ['bladder', 'eye_left', 'eye_right']:
        if key in keypoints:
            # Convert to image coordinates
            kp_roi = keypoints[key]
            kp_img_px = np.array(roi_coords) + kp_roi
            
            # Normalize to image dimensions
            kp_img_norm = kp_img_px / np.array(image_shape[::-1])
            
            result[f'{key}_img_px'] = kp_img_px
            result[f'{key}_img_norm'] = kp_img_norm
            result[f'{key}_roi_norm'] = kp_roi / np.array(roi_shape[::-1])
    
    if 'heading' in keypoints:
        result['heading'] = keypoints['heading']
    
    return result


# Convenience function that combines all steps
def process_roi_keypoints(
    roi: np.ndarray,
    background_roi: np.ndarray,
    roi_coords_full: Tuple[int, int],
    roi_shape: Tuple[int, int],
    full_img_shape: Tuple[int, int],
    detection_params: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Complete keypoint detection pipeline for a single ROI.
    
    Args:
        roi: ROI image
        background_roi: Background model for ROI region
        roi_coords_full: Top-left corner of ROI in full image
        roi_shape: Shape of the ROI
        full_img_shape: Shape of the full image
        detection_params: Optional detection parameters
        
    Returns:
        Complete keypoint detection results or None if failed
    """
    if detection_params is None:
        detection_params = {}
    
    # Detect keypoints
    keypoints = detect_keypoints_traditional(
        roi, 
        background_roi,
        roi_thresh=detection_params.get('roi_thresh', 50),
        se1_radius=detection_params.get('se1_radius', 1),
        se2_radius=detection_params.get('se2_radius', 2)
    )
    
    if keypoints is None:
        return None
    
    # Calculate bounding box
    bbox = calculate_bounding_box(keypoints, roi_shape)
    
    # Transform to image coordinates
    transformed = transform_keypoints_to_image_coords(
        keypoints, roi_coords_full, roi_shape, full_img_shape
    )
    
    # Combine all results
    result = {
        **transformed,
        'bbox_center_norm': bbox['center_norm'],
        'bbox_extent_norm': bbox['extent_norm'],
        'confidence': keypoints['confidence'],
        'effective_threshold': keypoints['effective_threshold']
    }
    
    return result