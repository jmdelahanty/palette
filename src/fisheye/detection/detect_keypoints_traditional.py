"""
Traditional computer vision-based keypoint detection for fish tracking.
Uses morphological operations and blob detection to identify swim bladder and eyes.

Zarr-first implementation that reads from crop_runs and writes directly to keypoint_runs.
"""

import numpy as np
import zarr
import time
import os
import sys
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime, timezone
from pathlib import Path
import yaml
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops

import dask
from dask import delayed
from dask.diagnostics import ProgressBar

from ..shared.zarr.schema import get_run_group
from ..pose.schema import schema_from_package

# Optional distributed
try:
    from dask.distributed import Client, LocalCluster
    HAVE_DISTRIBUTED = True  # Enable distributed
except Exception:  # pragma: no cover - optional dependency
    HAVE_DISTRIBUTED = False

from ..utils.system import get_environment_info


TRADITIONAL_POSE_SCHEMA = schema_from_package("traditional_v1")


# ========== Core Detection Functions ==========

def detect_keypoints_traditional(
    roi: np.ndarray,
    background_roi: np.ndarray,
    roi_thresh: int = 50,
    se1_radius: int = 1,
    se2_radius: int = 2,
    min_area: int = 5,
    min_valid_angle: float = 10.0,
    max_valid_angle: float = 90.0,
    min_triangle_area: float = 100.0,
    max_triangle_area: Optional[float] = None
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
        min_valid_angle: Minimum angle in degrees for valid triangle
        max_valid_angle: Maximum angle in degrees for valid triangle
        min_triangle_area: Minimum triangle area in pixels^2
        max_triangle_area: Maximum triangle area in pixels^2 (None to disable)
    
    Returns:
        Dictionary with keypoint positions and metadata, or None if detection fails
    """
    if roi.shape != background_roi.shape:
        return None
    
    se1 = disk(se1_radius)
    
    diff_roi = np.clip(
        background_roi.astype(np.int16) - roi.astype(np.int16),
        0, 255
    ).astype(np.uint8)
    
    current_se2 = se2_radius
    keypoint_stats = []
    effective_se2 = se2_radius
    triangle_angles_raw = np.full(3, np.nan, dtype=np.float64)
    triangle_area = float("nan")
    
    se2 = disk(max(1, int(round(current_se2))))
    im_roi = erosion(dilation(erosion(
        diff_roi >= roi_thresh, se1), se2), se1
    )

    roi_stat = [r for r in regionprops(label(im_roi)) if r.area > min_area]

    if len(roi_stat) >= 3:
        # Take top 3 by area
        candidate_stats = sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]

        # Calculate triangle geometry for validation
        centroids = np.array([r.centroid for r in candidate_stats])
        angles, tri_area = calculate_triangle_metrics(centroids[0], centroids[1], centroids[2])

        # Check if angles form a valid triangle AND triangle is large enough
        max_ok = max_triangle_area is None or max_triangle_area <= 0 or tri_area <= max_triangle_area
        if (
            np.all(angles >= np.deg2rad(min_valid_angle))
            and np.all(angles <= np.deg2rad(max_valid_angle))
            and tri_area >= min_triangle_area
            and max_ok
        ):
            keypoint_stats = candidate_stats
            effective_se2 = max(1, int(round(current_se2)))
            # Store triangle metrics for later
            triangle_angles_raw = np.rad2deg(angles)  # Candidate-order angles (degrees)
            triangle_area = tri_area

    if len(keypoint_stats) != 3:
        return None

    keypoints = identify_keypoints_by_geometry(keypoint_stats)
    if keypoints is None:
        return None

    keypoints['confidence'] = calculate_confidence(keypoint_stats)
    keypoints['effective_threshold'] = roi_thresh
    keypoints['effective_se2_radius'] = effective_se2
    keypoints['num_blobs_found'] = len(keypoint_stats)

    # Triangle geometry metrics
    canonical_angles, _ = calculate_triangle_metrics(
        keypoints['bladder'], keypoints['eye_left'], keypoints['eye_right']
    )
    keypoints['triangle_angles'] = np.rad2deg(canonical_angles)  # (3,) array in degrees
    keypoints['triangle_angles_raw'] = triangle_angles_raw  # Candidate-order angles (degrees)
    keypoints['triangle_area'] = triangle_area      # scalar in pixels^2

    # Per-keypoint confidence from blob areas (normalized)
    blob_areas = np.array([
        keypoints['bladder_stats'].area,
        keypoints['eye_left_stats'].area,
        keypoints['eye_right_stats'].area
    ])
    # Normalize: larger area = higher confidence, cap at 1.0
    keypoints['keypoint_confidences'] = np.clip(blob_areas / 100.0, 0.0, 1.0)

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
    
    pts = np.array([s.centroid[::-1] for s in keypoint_stats])
    
    angles, _ = calculate_triangle_metrics(pts[0], pts[1], pts[2])
    kp_idx = np.argsort(angles)
    
    # Bladder at smallest angle (furthest from eye pair)
    bladder_idx = kp_idx[0]
    eye_indices = kp_idx[1:3]
    
    eye_mean = np.mean(pts[eye_indices], axis=0)
    head_vec = eye_mean - pts[bladder_idx]
    heading = np.rad2deg(np.arctan2(-head_vec[1], head_vec[0]))
    
    R = rotation_matrix_2d(heading)
    rotated_pts = (pts - eye_mean) @ R.T
    
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
    """Calculate angles and area of a triangle formed by three points."""
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
    
    # Calculate area using Heron's formula
    s = (a + b + c) / 2
    area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
    
    return angles, area


def rotation_matrix_2d(angle_degrees: float) -> np.ndarray:
    """Create a 2D rotation matrix for the given angle."""
    angle_rad = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])


def calculate_confidence(keypoint_stats: List[Any]) -> float:
    """Calculate confidence score based on keypoint properties."""
    if not keypoint_stats:
        return 0.0
    
    mean_area = np.mean([s.area for s in keypoint_stats])
    confidence = min(1.0, mean_area / 100.0)
    
    return confidence


# ========== Zarr-First Processing Functions ==========

@delayed
def process_keypoint_chunk_delayed(
    zarr_path: str,
    roi_indices: np.ndarray,
    detection_params: Dict[str, Any]
) -> Tuple[int, int]:
    """
    Process a chunk of ROIs for keypoint detection and write results directly to zarr.
    
    Args:
        zarr_path: Path to zarr archive
        roi_indices: Array of ROI indices to process in this chunk
        detection_params: Detection parameters
        
    Returns:
        Tuple of (results array, num_successful, num_failed)
    """
    root = zarr.open(zarr_path, mode='a')
    
    # Get keypoint group path from root attrs
    keypoint_group_path = root.attrs.get('current_keypoint_group_path')
    if keypoint_group_path is None:
        raise RuntimeError("Missing 'current_keypoint_group_path' in root attrs")
    
    keypoint_group = root[keypoint_group_path]
    
    # Get latest crop and background runs
    latest_crop = root['crop_runs'].attrs['latest']
    latest_background = root['background_runs'].attrs['latest']
    
    crop_group = root[f'crop_runs/{latest_crop}']
    roi_images = crop_group['roi_images']
    roi_coords_full = crop_group['roi_coordinates_full']
    
    background_full = root[f'background_runs/{latest_background}/background_full'][:]
    full_img_shape = root['raw_video/images_full'].shape[1:]  # (H, W)
    roi_shape = roi_images.shape[1:]  # (H, W) of each ROI
    
    chunk_len = len(roi_indices)
    keypoints_roi = np.full((chunk_len, 3, 2), np.nan, dtype='f8')
    keypoints_img = np.full((chunk_len, 3, 2), np.nan, dtype='f8')
    keypoints_norm = np.full((chunk_len, 3, 2), np.nan, dtype='f8')
    heading = np.full(chunk_len, np.nan, dtype='f8')
    confidence = np.full(chunk_len, np.nan, dtype='f8')
    thresholds = np.full(chunk_len, np.nan, dtype='f8')
    se2_radii = np.full(chunk_len, np.nan, dtype='f8')
    success_mask = np.zeros(chunk_len, dtype='bool')

    # NEW: Per-keypoint confidence and triangle metrics
    keypoint_confidences = np.full((chunk_len, 3), np.nan, dtype='f8')
    triangle_angles = np.full((chunk_len, 3), np.nan, dtype='f8')
    triangle_angles_raw = np.full((chunk_len, 3), np.nan, dtype='f8')
    triangle_area = np.full(chunk_len, np.nan, dtype='f8')

    num_successful = 0
    num_failed = 0
    
    for i, roi_idx in enumerate(roi_indices):
        roi_img = roi_images[roi_idx]
        roi_coord = roi_coords_full[roi_idx]  # (x, y) top-left in full image
        
        # Extract background ROI
        x1, y1 = roi_coord
        x2, y2 = x1 + roi_shape[1], y1 + roi_shape[0]
        
        # Bounds check
        if x1 < 0 or y1 < 0 or x2 > full_img_shape[1] or y2 > full_img_shape[0]:
            num_failed += 1
            continue
        
        background_roi = background_full[y1:y2, x1:x2]
        
        # Detect keypoints
        keypoints = detect_keypoints_traditional(
            roi_img,
            background_roi,
            roi_thresh=detection_params.get('roi_thresh', 50),
            se1_radius=detection_params.get('se1_radius', 1),
            se2_radius=detection_params.get('se2_radius', 2),
            min_area=detection_params.get('min_area', 5),
            min_valid_angle=detection_params.get('min_valid_angle', 10.0),
            max_valid_angle=detection_params.get('max_valid_angle', 90.0),
            min_triangle_area=detection_params.get('min_triangle_area', 100.0),
            max_triangle_area=detection_params.get('max_triangle_area')
        )
        
        if keypoints is None:
            num_failed += 1
            continue

        success_mask[i] = True
        
        # ROI coordinates
        keypoints_roi[i] = np.array([
            keypoints['bladder'],
            keypoints['eye_left'],
            keypoints['eye_right']
        ])
        
        # Image coordinates (pixel space)
        bladder_img = np.array(roi_coord) + keypoints['bladder']
        eye_l_img = np.array(roi_coord) + keypoints['eye_left']
        eye_r_img = np.array(roi_coord) + keypoints['eye_right']
        
        keypoints_img[i] = np.array([bladder_img, eye_l_img, eye_r_img])
        
        # Normalized coordinates (0-1 relative to full image)
        norm_factor = np.array(full_img_shape[::-1], dtype='f8')
        keypoints_norm[i] = keypoints_img[i] / norm_factor
        
        # Metadata
        heading[i] = keypoints['heading']
        confidence[i] = keypoints['confidence']
        thresholds[i] = keypoints['effective_threshold']
        se2_radii[i] = keypoints['effective_se2_radius']

        # NEW: Store per-keypoint and triangle metrics
        keypoint_confidences[i] = keypoints['keypoint_confidences']
        triangle_angles[i] = keypoints['triangle_angles']
        triangle_angles_raw[i] = keypoints['triangle_angles_raw']
        triangle_area[i] = keypoints['triangle_area']

        num_successful += 1
    
    # Write results to zarr
    start_idx = int(roi_indices[0])
    end_idx = int(roi_indices[-1]) + 1
    keypoint_group['keypoints_roi'][start_idx:end_idx] = keypoints_roi
    keypoint_group['keypoints_img'][start_idx:end_idx] = keypoints_img
    keypoint_group['keypoints_norm'][start_idx:end_idx] = keypoints_norm
    keypoint_group['heading'][start_idx:end_idx] = heading
    keypoint_group['confidence'][start_idx:end_idx] = confidence
    keypoint_group['effective_threshold'][start_idx:end_idx] = thresholds
    keypoint_group['effective_se2_radius'][start_idx:end_idx] = se2_radii
    keypoint_group['detection_success'][start_idx:end_idx] = success_mask

    # NEW: Write additional metrics
    keypoint_group['keypoint_confidences'][start_idx:end_idx] = keypoint_confidences
    keypoint_group['triangle_angles'][start_idx:end_idx] = triangle_angles
    keypoint_group['triangle_angles_raw'][start_idx:end_idx] = triangle_angles_raw
    keypoint_group['triangle_area'][start_idx:end_idx] = triangle_area

    return num_successful, num_failed


def get_keypoint_parameters(root: zarr.Group, config: Dict[str, Any], console: Optional[Console] = None) -> Tuple[Dict[str, Any], str]:
    """
    Get keypoint detection parameters with priority order:
    1. Zarr analysis_metadata (if keypoint tuning exists)
    2. Config file defaults
    """
    # Start with config defaults
    keypoint_params = config.get('keypoints', {}).copy()
    keypoint_params.setdefault('roi_thresh', 50)
    keypoint_params.setdefault('se1_radius', 1)
    keypoint_params.setdefault('se2_radius', 2)
    keypoint_params.setdefault('min_area', 5)
    keypoint_params.setdefault('min_valid_angle', 10.0)
    keypoint_params.setdefault('max_valid_angle', 90.0)
    keypoint_params.setdefault('min_triangle_area', 100.0)
    keypoint_params.setdefault('max_triangle_area', None)
    
    param_source = 'config_default'
    
    # Check for tuned parameters in zarr
    if 'analysis_metadata' in root:
        analysis_meta = root['analysis_metadata']
        
        if 'keypoint_tuning' in analysis_meta.attrs:
            tuning_data = analysis_meta.attrs['keypoint_tuning']
            tuned_params = tuning_data.get('tuned_parameters', {})
            if tuned_params:
                keypoint_params.update(tuned_params)
                param_source = 'zarr_tuned'
                if console:
                    console.print(f"[green]✓ Using tuned keypoint parameters from zarr[/green]")
                    console.print(f"  Tuned on: {tuning_data.get('tuned_timestamp', 'unknown')}")

    keypoint_params.pop('adaptive_steps', None)
    keypoint_params.pop('se2_decrement', None)
    return keypoint_params, param_source


def detect_keypoints(
    zarr_path: str,
    config: Dict[str, Any],
    scheduler: str = None,
    num_workers: Optional[int] = None,
    console: Optional[Console] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Main function to detect keypoints in cropped ROIs.
    
    Args:
        zarr_path: Path to zarr archive
        config: Pipeline configuration dictionary
        scheduler: Dask scheduler ('processes', 'threads', 'single-threaded', 'distributed')
        num_workers: Number of workers
        console: Rich console for output
        show_progress: Whether to show per-chunk progress (default: True)
        
    Returns:
        Dictionary with summary statistics
    """
    if console is None:
        console = Console()
    
    console.rule("[bold]Stage: Keypoint Detection[/bold]")
    start_time = time.perf_counter()
    
    root = zarr.open(zarr_path, mode='a')
    
    # Check prerequisites
    if 'crop_runs' not in root:
        raise ValueError("Crop stage not run. Run crop before keypoints.")
    if 'background_runs' not in root:
        raise ValueError("Background stage not run. Run background before keypoints.")
    
    # Get parameters
    keypoints_params, param_source = get_keypoint_parameters(root, config, console)
    
    if scheduler is None:
        scheduler = keypoints_params.get('scheduler', 'processes')
    if num_workers is None:
        num_workers = keypoints_params.get('num_workers', None)
    
    chunk_size = config.get('import', {}).get('chunk_size', 32)
    
    # Distributed-specific settings
    threads_per_worker = keypoints_params.get('threads_per_worker', 1)
    memory_limit = keypoints_params.get('memory_limit', '4GB')
    
    console.print(f"Scheduler: {scheduler}, Workers: {num_workers or 'default'}")
    console.print(f"Chunk size: {chunk_size} ROIs per task")
    console.print(f"Parameter source: {param_source}")
    
    # Create run group
    keypoint_group, run_group_name = get_run_group(root, 'keypoints', console)
    
    # Store metadata
    latest_crop = root['crop_runs'].attrs['latest']
    latest_background = root['background_runs'].attrs['latest']
    crop_group = root[f'crop_runs/{latest_crop}']
    source_detect_run = crop_group.attrs.get('source_detect_run')
    source_refined_run = crop_group.attrs.get('source_refined_run')

    metadata_dict = {
        'keypoints_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'dask_scheduler': scheduler,
        'dask_num_workers': num_workers or os.cpu_count(),
        'parameters': keypoints_params,
        'parameter_source': param_source,
        'source_crop_run': latest_crop,
        'source_background_run': latest_background,
        'source_detect_run': source_detect_run or 'unknown',
        'method': 'traditional_pose',
    }
    if source_refined_run:
        metadata_dict['source_refined_run'] = source_refined_run
    
    # Add distributed-specific metadata if using distributed
    use_distributed = (scheduler == 'distributed') and HAVE_DISTRIBUTED
    
    if use_distributed:
        metadata_dict['dask_threads_per_worker'] = threads_per_worker
        metadata_dict['dask_memory_limit'] = memory_limit
    
    keypoint_group.attrs.update(metadata_dict)
    
    # Get ROI count
    total_rois = crop_group['roi_images'].shape[0]
    
    if total_rois == 0:
        console.print("[yellow]No ROIs found. Nothing to process.[/yellow]")
        return {
            'total_rois': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'success_rate_percent': 0.0
        }
    
    console.print(f"Total ROIs to process: [green]{total_rois}[/green]")
    
    # Precompute frame indices and counts (align with detection outputs)
    frame_indices = crop_group['frame_indices'][:].astype('i4', copy=False)
    frame_chunks = (min(chunk_size, total_rois),) if total_rois > 0 else None
    keypoint_group.create_array(
        'frame_indices',
        data=frame_indices,
        chunks=frame_chunks,
        overwrite=True
    )
    
    frame_counts_total = (
        np.bincount(frame_indices, minlength=root['raw_video/images_full'].shape[0]).astype('i4', copy=False)
        if total_rois > 0 else np.zeros(root['raw_video/images_full'].shape[0], dtype='i4')
    )
    count_chunks = (min(chunk_size, len(frame_counts_total)),) if len(frame_counts_total) > 0 else None
    keypoint_group.create_array(
        'n_rois',
        data=frame_counts_total,
        chunks=count_chunks,
        overwrite=True
    )
    keypoint_group.create_array(
        'frame_counts',
        data=frame_counts_total,
        chunks=count_chunks,
        overwrite=True
    )

    if 'detection_indices' in crop_group:
        detection_indices = crop_group['detection_indices'][:].astype('i4', copy=False)
        det_chunks = (min(chunk_size, detection_indices.shape[0]),) if detection_indices.size > 0 else None
        keypoint_group.create_array(
            'detection_indices',
            data=detection_indices,
            chunks=det_chunks,
            overwrite=True
        )
    else:
        console.print("[yellow]Crop run missing 'detection_indices'; keypoint run will omit them.[/yellow]")
    
    # Create output arrays matching detection-style layout
    chunk_len = max(1, min(chunk_size, total_rois if total_rois > 0 else 1))
    data_chunk = (chunk_len, 3, 2)
    scalar_chunk = (chunk_len,)
    
    keypoint_group.create_array(
        'keypoints_roi',
        shape=(total_rois, 3, 2),
        chunks=data_chunk,
        dtype='f8',
        fill_value=np.nan,
        overwrite=True
    )
    keypoint_group.create_array(
        'keypoints_img',
        shape=(total_rois, 3, 2),
        chunks=data_chunk,
        dtype='f8',
        fill_value=np.nan,
        overwrite=True
    )
    keypoint_group.create_array(
        'keypoints_norm',
        shape=(total_rois, 3, 2),
        chunks=data_chunk,
        dtype='f8',
        fill_value=np.nan,
        overwrite=True
    )
    keypoint_group.create_array(
        'heading',
        shape=(total_rois,),
        chunks=scalar_chunk,
        dtype='f8',
        fill_value=np.nan,
        overwrite=True
    )
    keypoint_group.create_array(
        'confidence',
        shape=(total_rois,),
        chunks=scalar_chunk,
        dtype='f8',
        fill_value=np.nan,
        overwrite=True
    )
    keypoint_group.create_array(
        'effective_threshold',
        shape=(total_rois,),
        chunks=scalar_chunk,
        dtype='f8',
        fill_value=np.nan,
        overwrite=True
    )
    keypoint_group.create_array(
        'effective_se2_radius',
        shape=(total_rois,),
        chunks=scalar_chunk,
        dtype='f8',
        fill_value=np.nan,
        overwrite=True
    )
    keypoint_group.create_array(
        'detection_success',
        shape=(total_rois,),
        chunks=scalar_chunk,
        dtype='bool',
        fill_value=False,
        overwrite=True
    )

    if 'detection_source' in crop_group:
        detection_source_values = crop_group['detection_source'][:].astype('i1', copy=False)
        if detection_source_values.shape[0] != total_rois:
            raise ValueError(
                f"Crop run detection_source length {detection_source_values.shape[0]} does not match ROI count {total_rois}"
            )
    else:
        detection_source_values = np.zeros(total_rois, dtype='i1')
    keypoint_group.create_array(
        'detection_source',
        data=detection_source_values,
        chunks=scalar_chunk,
        overwrite=True
    )
    keypoint_group.create_array(
        'heading_valid',
        shape=(total_rois,),
        chunks=scalar_chunk,
        dtype='bool',
        fill_value=False,
        overwrite=True
    )

    # NEW: Per-keypoint confidence (N, 3) for [bladder, eye_left, eye_right]
    keypoint_group.create_array(
        'keypoint_confidences',
        shape=(total_rois, 3),
        chunks=(chunk_len, 3),
        dtype='f8',
        fill_value=np.nan,
        overwrite=True
    )

    # NEW: Triangle angles at each vertex (N, 3) in degrees (bladder, left, right)
    keypoint_group.create_array(
        'triangle_angles',
        shape=(total_rois, 3),
        chunks=(chunk_len, 3),
        dtype='f8',
        fill_value=np.nan,
        overwrite=True
    )
    # NEW: Triangle angles in candidate order (largest -> smallest blob by area)
    keypoint_group.create_array(
        'triangle_angles_raw',
        shape=(total_rois, 3),
        chunks=(chunk_len, 3),
        dtype='f8',
        fill_value=np.nan,
        overwrite=True
    )

    # NEW: Triangle area (N,) in pixels^2
    keypoint_group.create_array(
        'triangle_area',
        shape=(total_rois,),
        chunks=scalar_chunk,
        dtype='f8',
        fill_value=np.nan,
        overwrite=True
    )

    keypoint_group.attrs['keypoint_labels'] = ['bladder', 'eye_left', 'eye_right']
    keypoint_group.attrs['keypoint_confidence_labels'] = ['bladder', 'eye_left', 'eye_right']
    keypoint_group.attrs['triangle_angle_order'] = [
        'angle at bladder',
        'angle at eye_left',
        'angle at eye_right'
    ]
    keypoint_group.attrs['triangle_angle_raw_order'] = [
        'angle at vertex 0 (largest blob by area)',
        'angle at vertex 1 (2nd largest blob)',
        'angle at vertex 2 (3rd largest blob)'
    ]

    # Mark group path for workers
    root.attrs['current_keypoint_group_path'] = keypoint_group.path
    
    # Create chunks of ROI indices
    roi_indices = np.arange(total_rois)
    chunks = [roi_indices[i:i+chunk_size] for i in range(0, total_rois, chunk_size)]
    
    console.print(f"Creating [yellow]{len(chunks)}[/yellow] Dask tasks...")
    
    # Setup distributed cluster if requested
    client = None
    cluster = None
    
    if use_distributed:
        console.print("[yellow]Setting up distributed LocalCluster...[/yellow]")
        cluster = LocalCluster(
            n_workers=num_workers or None,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            processes=True
        )
        client = Client(cluster)
        console.print(f"Dashboard: [link={client.dashboard_link}]{client.dashboard_link}[/link]")
        console.print(f"Workers: {len(client.scheduler_info()['workers'])}")
    else:
        # Configure local Dask
        dask.config.set(scheduler=scheduler)
        if num_workers:
            dask.config.set(num_workers=num_workers)
    
    # Create delayed tasks
    delayed_tasks = [
        process_keypoint_chunk_delayed(zarr_path, chunk, keypoints_params)
        for chunk in chunks
    ]
    
    # Execute with progress
    console.print("Processing keypoints...")

    chunk_stats = []
    try:
        if show_progress and console is not None:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Detecting keypoints ({scheduler})...", total=len(delayed_tasks)
                )
                if use_distributed:
                    futures = client.compute(delayed_tasks)
                    for future in futures:
                        result = future.result()
                        chunk_stats.append(result)
                        progress.update(task, advance=1)
                else:
                    for dt in delayed_tasks:
                        result = dt.compute()
                        chunk_stats.append(result)
                        progress.update(task, advance=1)
        else:
            if use_distributed:
                futures = client.compute(delayed_tasks)
                for future in futures:
                    chunk_stats.append(future.result())
            else:
                for dt in delayed_tasks:
                    chunk_stats.append(dt.compute())
    finally:
        # Cleanup distributed resources
        if use_distributed and client is not None:
            client.close()
            if cluster is not None:
                cluster.close()
    
    # Aggregate statistics
    total_successful = sum(r[0] for r in chunk_stats)
    total_failed = sum(r[1] for r in chunk_stats)
    success_rate = (total_successful / total_rois * 100) if total_rois > 0 else 0

    # Store per-frame success counts
    detection_success = keypoint_group['detection_success'][:]
    if total_rois > 0 and np.any(detection_success):
        success_counts = np.bincount(frame_indices[detection_success],
                                     minlength=len(frame_counts_total)).astype('i4', copy=False)
    else:
        success_counts = np.zeros_like(frame_counts_total, dtype='i4')
    keypoint_group.create_array(
        'n_keypoints',
        data=success_counts,
        chunks=count_chunks,
        overwrite=True
    )

    heading_valid = np.logical_and(detection_success, detection_source_values == 0)
    keypoint_group['heading_valid'][:] = heading_valid
    frames_with_success = int(np.sum(success_counts > 0))
    
    # Store summary
    duration = time.perf_counter() - start_time
    
    summary_stats = {
        'total_rois': int(total_rois),
        'successful_detections': int(total_successful),
        'failed_detections': int(total_failed),
        'success_rate_percent': round(success_rate, 2),
        'successful_keypoint_detections': int(total_successful),
        'failed_keypoint_detections': int(total_failed),
        'frames_with_keypoints': frames_with_success
    }
    
    keypoint_group.attrs['summary_statistics'] = summary_stats
    keypoint_group.attrs['duration_seconds'] = duration
    keypoint_group.attrs['keypoints_processed'] = int(total_rois)
    keypoint_group.attrs['success_rate'] = round(success_rate, 2)
    
    # Environment info + provenance snapshot
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    git_info = env_info.get('git', {})
    platform_info = env_info.get('platform', {})

    keypoint_group.attrs.update({
        'git_commit': git_info.get('commit_hash', 'unknown'),
        'git_branch': git_info.get('branch', 'unknown'),
        'hostname': platform_info.get('hostname', 'unknown')
    })
    provenance = {
        'stage': 'keypoints_detect',
        'method': 'traditional_pose',
        'command': ' '.join(sys.argv),
        'created_at_utc': metadata_dict.get('keypoints_timestamp_utc'),
        'git': git_info,
        'environment': env_info.get('environment'),
        'platform': {
            'hostname': platform_info.get('hostname'),
            'system': platform_info.get('system'),
            'release': platform_info.get('release'),
            'python_version': platform_info.get('python_version'),
        },
        'parameters': keypoints_params,
        'parameter_source': param_source,
        'inputs': {
            'source_crop_run': latest_crop,
            'source_background_run': latest_background,
            'source_detect_run': source_detect_run or 'unknown',
            'source_refined_run': source_refined_run,
            'frame_source': crop_group.attrs.get('video_source_type', 'zarr'),
            'source_video_path': crop_group.attrs.get('video_source_path'),
        },
        'scheduler': {
            'dask_scheduler': scheduler,
            'dask_num_workers': num_workers or os.cpu_count(),
            'threads_per_worker': threads_per_worker if use_distributed else None,
            'memory_limit': memory_limit if use_distributed else None,
        },
    }
    provenance = {k: v for k, v in provenance.items() if v is not None}
    keypoint_group.attrs['provenance'] = provenance
    keypoint_group.attrs['pose_schema'] = {
        'name': TRADITIONAL_POSE_SCHEMA.name,
        'nodes': TRADITIONAL_POSE_SCHEMA.node_names,
        'edges': TRADITIONAL_POSE_SCHEMA.edges,
        'metadata': TRADITIONAL_POSE_SCHEMA.metadata,
        'source': 'configs/fisheye/pose_schemas/traditional_v1.json'
    }
    
    # Completion panel
    completion_text = f"""[green]✓[/green] Keypoint detection completed

[bold]Performance:[/bold]
  Time: {duration:.1f}s ({duration/60:.1f} min)
  ROIs/sec: {total_rois/duration:.1f}

[bold]Results:[/bold]
  Successful ROIs: {total_successful}/{total_rois} ({success_rate:.1f}%)
  Failed ROIs: {total_failed}
  Frames with keypoints: {frames_with_success}

[bold]Output:[/bold]
  Path: {zarr_path}
  Arrays:
    • keypoints_runs/{run_group_name}/keypoints_roi ({total_rois}, 3, 2)
    • keypoints_runs/{run_group_name}/n_keypoints ({len(frame_counts_total)})"""
    
    panel = Panel(
        completion_text,
        title="[bold]Keypoint Detection Complete[/bold]",
        border_style="green",
        padding=(1, 2)
    )
    
    console.print("\n")
    console.print(panel)
    
    return summary_stats


def _load_config(config_path: Optional[str], console: Console) -> Dict[str, Any]:
    """Load YAML config if available; return empty dict otherwise."""
    if not config_path:
        return {}
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        console.print(f"[yellow]Config file not found at {cfg_path}. Using defaults.[/yellow]")
        return {}
    try:
        with cfg_path.open() as fh:
            data = yaml.safe_load(fh) or {}
        console.print(f"[dim]Loaded config from {cfg_path}[/dim]")
        return data
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[red]Failed to parse config {cfg_path}: {exc}[/red]")
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run traditional keypoint detection on a Palette Zarr archive."
    )
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr archive.")
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline_config.yaml",
        help="Path to pipeline config file (default: pipeline_config.yaml).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="Dask scheduler to use (overrides config).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of Dask workers (overrides config).",
    )
    args = parser.parse_args()

    console = Console()
    config = _load_config(args.config, console)

    detect_keypoints(
        zarr_path=args.zarr_path,
        config=config,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        console=console,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
