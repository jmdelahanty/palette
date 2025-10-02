"""
Traditional computer vision-based keypoint detection for fish tracking.
Uses morphological operations and blob detection to identify swim bladder and eyes.

Zarr-first implementation that reads from crop_runs and writes directly to keypoint_runs.
"""

import numpy as np
import zarr
import time
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops

import dask
from dask import delayed
from dask.diagnostics import ProgressBar

# Optional distributed
try:
    from dask.distributed import Client, LocalCluster
    HAVE_DISTRIBUTED = False  # Disabled for keypoints (memory concerns)
except:
    HAVE_DISTRIBUTED = False

from ..utils.system import get_environment_info


# ========== Core Detection Functions ==========

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
    
    se1 = disk(se1_radius)
    se2 = disk(se2_radius)
    
    diff_roi = np.clip(
        background_roi.astype(np.int16) - roi.astype(np.int16), 
        0, 255
    ).astype(np.uint8)
    
    current_thresh = roi_thresh
    keypoint_stats = []
    
    for _ in range(adaptive_steps):
        im_roi = erosion(dilation(erosion(
            diff_roi >= current_thresh, se1), se2), se1
        )
        
        roi_stat = [r for r in regionprops(label(im_roi)) if r.area > min_area]
        
        if len(roi_stat) >= 3:
            keypoint_stats = sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]
            break
            
        current_thresh -= thresh_decrement
    
    if len(keypoint_stats) != 3:
        return None
    
    keypoints = identify_keypoints_by_geometry(keypoint_stats)
    if keypoints is None:
        return None
    
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
    
    pts = np.array([s.centroid[::-1] for s in keypoint_stats])
    
    angles, _ = calculate_triangle_metrics(pts[0], pts[1], pts[2])
    kp_idx = np.argsort(angles)
    
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
) -> Tuple[np.ndarray, int, int]:
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
    
    # Result buffer: 21 columns per ROI
    # [bladder_x_roi, bladder_y_roi, eye_l_x_roi, eye_l_y_roi, eye_r_x_roi, eye_r_y_roi,
    #  bladder_x_img, bladder_y_img, eye_l_x_img, eye_l_y_img, eye_r_x_img, eye_r_y_img,
    #  bladder_x_norm, bladder_y_norm, eye_l_x_norm, eye_l_y_norm, eye_r_x_norm, eye_r_y_norm,
    #  heading, confidence, effective_threshold]
    results = np.full((len(roi_indices), 21), np.nan, dtype='f8')
    
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
            adaptive_steps=detection_params.get('adaptive_steps', 5),
            thresh_decrement=detection_params.get('thresh_decrement', 5)
        )
        
        if keypoints is None:
            num_failed += 1
            continue
        
        # Pack results: ROI coordinates
        results[i, 0:2] = keypoints['bladder']
        results[i, 2:4] = keypoints['eye_left']
        results[i, 4:6] = keypoints['eye_right']
        
        # Image coordinates (pixel space)
        bladder_img = np.array(roi_coord) + keypoints['bladder']
        eye_l_img = np.array(roi_coord) + keypoints['eye_left']
        eye_r_img = np.array(roi_coord) + keypoints['eye_right']
        
        results[i, 6:8] = bladder_img
        results[i, 8:10] = eye_l_img
        results[i, 10:12] = eye_r_img
        
        # Normalized coordinates (0-1 relative to full image)
        results[i, 12:14] = bladder_img / np.array(full_img_shape[::-1])
        results[i, 14:16] = eye_l_img / np.array(full_img_shape[::-1])
        results[i, 16:18] = eye_r_img / np.array(full_img_shape[::-1])
        
        # Metadata
        results[i, 18] = keypoints['heading']
        results[i, 19] = keypoints['confidence']
        results[i, 20] = keypoints['effective_threshold']
        
        num_successful += 1
    
    # Write results to zarr
    start_idx = int(roi_indices[0])
    end_idx = int(roi_indices[-1]) + 1
    keypoint_group['keypoint_results'][start_idx:end_idx] = results
    
    return results, num_successful, num_failed


def detect_keypoints(
    zarr_path: str,
    config: Dict[str, Any],
    scheduler: str = None,
    num_workers: Optional[int] = None,
    console: Optional[Console] = None
) -> Dict[str, Any]:
    """
    Main function to detect keypoints in cropped ROIs.
    
    Args:
        zarr_path: Path to zarr archive
        config: Pipeline configuration dictionary
        scheduler: Dask scheduler ('processes', 'threads', 'single-threaded')
        num_workers: Number of workers
        console: Rich console for output
        
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
    keypoints_params = config.get('keypoints', {})
    
    if scheduler is None:
        scheduler = keypoints_params.get('scheduler', 'processes')
    if num_workers is None:
        num_workers = keypoints_params.get('num_workers', None)
    
    chunk_size = config.get('import', {}).get('chunk_size', 32)
    
    console.print(f"Scheduler: {scheduler}, Workers: {num_workers or 'default'}")
    console.print(f"Chunk size: {chunk_size} ROIs per task")
    
    # Create run group
    from ..shared.zarr import get_run_group
    keypoint_group, run_group_name = get_run_group(root, 'keypoints', console)
    
    # Store metadata
    latest_crop = root['crop_runs'].attrs['latest']
    latest_background = root['background_runs'].attrs['latest']
    
    keypoint_group.attrs.update({
        'keypoints_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'dask_scheduler': scheduler,
        'dask_num_workers': num_workers or dask.system.CPU_COUNT,
        'parameters': keypoints_params,
        'source_crop_run': latest_crop,
        'source_background_run': latest_background
    })
    
    # Get ROI count
    crop_group = root[f'crop_runs/{latest_crop}']
    total_rois = crop_group['roi_images'].shape[0]
    
    if total_rois == 0:
        console.print("[yellow]No ROIs found. Nothing to process.[/yellow]")
        return {'total_rois': 0, 'successful': 0}
    
    console.print(f"Total ROIs to process: [green]{total_rois}[/green]")
    
    # Create output array
    keypoint_results = keypoint_group.create_array(
        'keypoint_results',
        shape=(total_rois, 21),
        chunks=(min(chunk_size * 4, total_rois), 21),
        dtype='f8',
        fill_value=np.nan,
        overwrite=True
    )
    
    # Column documentation
    keypoint_results.attrs['columns'] = [
        'bladder_x_roi', 'bladder_y_roi',
        'eye_left_x_roi', 'eye_left_y_roi',
        'eye_right_x_roi', 'eye_right_y_roi',
        'bladder_x_img', 'bladder_y_img',
        'eye_left_x_img', 'eye_left_y_img',
        'eye_right_x_img', 'eye_right_y_img',
        'bladder_x_norm', 'bladder_y_norm',
        'eye_left_x_norm', 'eye_left_y_norm',
        'eye_right_x_norm', 'eye_right_y_norm',
        'heading', 'confidence', 'effective_threshold'
    ]
    
    # Mark group path for workers
    root.attrs['current_keypoint_group_path'] = keypoint_group.path
    
    # Create chunks of ROI indices
    roi_indices = np.arange(total_rois)
    chunks = [roi_indices[i:i+chunk_size] for i in range(0, total_rois, chunk_size)]
    
    console.print(f"Creating [yellow]{len(chunks)}[/yellow] Dask tasks...")
    
    # Configure Dask
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
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Detecting keypoints ({scheduler})...", total=len(delayed_tasks))
        
        results = []
        for dt in delayed_tasks:
            result = dt.compute()
            results.append(result)
            progress.update(task, advance=1)
    
    # Aggregate statistics
    total_successful = sum(r[1] for r in results)
    total_failed = sum(r[2] for r in results)
    success_rate = (total_successful / total_rois * 100) if total_rois > 0 else 0
    
    # Store summary
    duration = time.perf_counter() - start_time
    
    summary_stats = {
        'total_rois': int(total_rois),
        'successful_detections': int(total_successful),
        'failed_detections': int(total_failed),
        'success_rate_percent': round(success_rate, 2)
    }
    
    keypoint_group.attrs['summary_statistics'] = summary_stats
    keypoint_group.attrs['duration_seconds'] = duration
    
    # Environment info
    env_info = get_environment_info()
    keypoint_group.attrs.update({
        'git_commit': env_info['git'].get('commit_hash', 'unknown'),
        'git_branch': env_info['git'].get('branch', 'unknown'),
        'hostname': env_info['platform']['hostname']
    })
    
    # Completion panel
    completion_text = f"""[green]✓[/green] Keypoint detection completed

[bold]Performance:[/bold]
  Time: {duration:.1f}s ({duration/60:.1f} min)
  ROIs/sec: {total_rois/duration:.1f}

[bold]Results:[/bold]
  Successful: {total_successful}/{total_rois} ({success_rate:.1f}%)
  Failed: {total_failed}

[bold]Output:[/bold]
  Path: {zarr_path}
  Array: keypoint_runs/{run_group_name}/keypoint_results
  Shape: ({total_rois}, 21)"""
    
    panel = Panel(
        completion_text,
        title="[bold]Keypoint Detection Complete[/bold]",
        border_style="green",
        padding=(1, 2)
    )
    
    console.print("\n")
    console.print(panel)
    
    return summary_stats