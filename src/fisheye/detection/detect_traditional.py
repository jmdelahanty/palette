"""
Fish detection module using blob detection on background-subtracted frames.
"""

import time
import numpy as np
import zarr
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List, Any
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn
from tqdm import tqdm

import skimage
from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops
import dask
from dask import delayed
from dask.diagnostics import ProgressBar

from ..utils.system import get_git_info, get_platform_info


def create_dish_mask(mask_params: Dict, img_shape: Tuple[int, int], console: Optional[Console] = None) -> Optional[np.ndarray]:
    """
    Create a dish mask based on parameters from config or analysis_metadata.
    
    Args:
        mask_params: Dictionary containing mask parameters
        img_shape: Tuple of (height, width) for the image
        console: Rich console for output
    
    Returns:
        numpy array mask or None if no mask defined
    """
    import cv2
    
    if not mask_params:
        return None
    
    dish_shape = mask_params.get('shape', 'circle')
    mask = None
    
    if dish_shape == 'rectangle' and 'roi' in mask_params:
        x, y, w, h = mask_params['roi']
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        if console:
            console.print(f"  Using rectangular mask: x={x}, y={y}, w={w}, h={h}")
            
    elif dish_shape == 'circle':
        center = mask_params.get('center')
        radius = mask_params.get('radius')
        
        if center and radius:
            mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.circle(mask, tuple(center), radius, 255, -1)
            if console:
                console.print(f"  Using circular mask: center={center}, radius={radius}")
    
    return mask


@delayed
def detect_chunk_delayed(zarr_path: str, chunk_slice: slice, detect_params: Dict, 
                         dish_mask: Optional[np.ndarray], latest_bg_run: str) -> Tuple:
    """
    Detects fish in a chunk of downsampled frames.
    
    Args:
        zarr_path: Path to zarr archive
        chunk_slice: Slice of frames to process
        detect_params: Detection parameters
        dish_mask: Optional mask for dish region
        latest_bg_run: Name of background run to use
    
    Returns:
        Tuple of (chunk_slice, frame_detection_counts, all_bbox_norms)
    """
    se1 = disk(detect_params['se1_radius'])
    se4 = disk(detect_params['se4_radius'])

    root = zarr.open(zarr_path, mode='r')
    images_ds_chunk = root['raw_video/images_ds'][chunk_slice]
    background_ds = root[f'background_runs/{latest_bg_run}/background_ds'][:]
    ds_img_shape = images_ds_chunk.shape[1:]

    chunk_len = images_ds_chunk.shape[0]
    all_bbox_norms = []
    frame_detection_counts = np.zeros(chunk_len, dtype='i4')

    for i in range(chunk_len):
        diff_ds = np.clip(background_ds.astype(np.int16) - images_ds_chunk[i].astype(np.int16), 0, 255).astype(np.uint8)
        if dish_mask is not None:
            diff_ds[dish_mask == 0] = 0

        current_thresh = detect_params['ds_thresh']
        valid_blobs = []
        
        # Adaptive thresholding - try up to 5 threshold values
        for _ in range(5):
            im_ds = erosion(dilation(erosion(diff_ds >= current_thresh, se1), se4), se1)
            all_blobs = regionprops(label(im_ds))
            valid_blobs = [r for r in all_blobs if detect_params['min_area'] <= r.area <= detect_params['max_area']]
            if valid_blobs:
                break
            current_thresh -= 5
        
        if not valid_blobs:
            continue

        # Keep only top N fish by area
        sorted_blobs = sorted(valid_blobs, key=lambda r: r.area, reverse=True)[:detect_params.get('max_fish', 20)]
        frame_detection_counts[i] = len(sorted_blobs)

        for blob in sorted_blobs:
            min_r, min_c, max_r, max_c = blob.bbox
            center_y, center_x = (min_r + max_r) / 2, (min_c + max_c) / 2
            height, width = max_r - min_r, max_c - min_c
            
            # Normalized coordinates
            center_norm = np.array([center_x / ds_img_shape[1], center_y / ds_img_shape[0]])
            size_norm = np.array([width / ds_img_shape[1], height / ds_img_shape[0]])
            all_bbox_norms.append([*center_norm, *size_norm])
            
    return chunk_slice, frame_detection_counts, all_bbox_norms


def detect_fish(
    zarr_path: str,
    config_path: Optional[str] = "pipeline_config.yaml",
    scheduler: str = "processes",
    num_workers: Optional[int] = None,
    console: Optional[Console] = None
) -> Dict[str, Any]:
    """
    Detect fish in video frames using blob detection.
    
    Args:
        zarr_path: Path to zarr archive
        config_path: Path to pipeline config
        scheduler: Dask scheduler type
        num_workers: Number of workers for parallel processing
        console: Rich console for output
    
    Returns:
        Dictionary with detection results and metadata
    """
    if console is None:
        console = Console()
    
    console.rule("[bold]Fish Detection[/bold]")
    start_time = time.perf_counter()
    
    # Load config
    import yaml
    from pathlib import Path
    
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    detect_params = config.get('detect', {})
    
    # Set defaults if not in config
    detect_params.setdefault('ds_thresh', 55)
    detect_params.setdefault('se1_radius', 1)
    detect_params.setdefault('se4_radius', 4)
    detect_params.setdefault('min_area', 50)
    detect_params.setdefault('max_area', 500)
    detect_params.setdefault('max_fish', 20)
    
    # Get version info
    git_info = get_git_info()
    platform_info = get_platform_info(collect_ip=False, disk_path=zarr_path)
    
    # Open zarr
    root = zarr.open_group(zarr_path, mode='r+')
    
    if 'background_runs' not in root:
        raise ValueError("Background stage not run. Run background computation first.")
    
    latest_bg_run = root['background_runs'].attrs['latest']
    console.print(f"Using background: [cyan]{latest_bg_run}[/cyan]")
    
    # Create detect runs group
    parent_group = root.require_group('detect_runs')
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"detect_{timestamp}"
    detect_group = parent_group.create_group(run_name)
    parent_group.attrs['latest'] = run_name
    
    console.print(f"Created new run: [cyan]detect_runs/{run_name}[/cyan]")
    
    # Load or create dish mask
    mask_params = detect_params.get('dish_mask', {})
    
    # Check analysis_metadata for mask from tuner
    if 'analysis_metadata' in root and 'dish_mask' in root['analysis_metadata'].attrs:
        tuned_mask = root['analysis_metadata'].attrs['dish_mask']
        if 'detected_circle' in tuned_mask:
            mask_params = {
                'shape': 'circle',
                'center': tuned_mask['detected_circle']['center'],
                'radius': tuned_mask['detected_circle']['radius']
            }
            console.print("Using mask from mask tuner")
    
    # Create the mask
    background_ds = root[f'background_runs/{latest_bg_run}/background_ds'][:]
    ds_img_shape = background_ds.shape
    mask = create_dish_mask(mask_params, ds_img_shape, console)
    
    # Get video info
    num_images = root['raw_video/images_ds'].shape[0]
    chunk_size = root['raw_video/images_ds'].chunks[0]
    
    console.print(f"Processing {num_images} frames in chunks of {chunk_size}")
    
    # Create output arrays
    n_detections = detect_group.create_array(
        'n_detections', 
        shape=(num_images,), 
        chunks=(chunk_size * 4,), 
        dtype='i4', 
        overwrite=True
    )
    
    # Start with small array, will resize as needed
    bbox_norm_coords = detect_group.create_array(
        'bbox_norm_coords', 
        shape=(1, 4), 
        chunks=(chunk_size * 4, 4), 
        dtype='f8', 
        overwrite=True
    )
    
    # Create detection tasks
    chunk_slices = [slice(i, min(i + chunk_size, num_images)) for i in range(0, num_images, chunk_size)]
    console.print(f"Creating [yellow]{len(chunk_slices)}[/yellow] detection tasks...")
    
    # Configure dask
    import os
    dask.config.set(scheduler=scheduler, num_workers=num_workers or os.cpu_count())
    
    # Create delayed tasks
    delayed_tasks = [
        detect_chunk_delayed(zarr_path, s, detect_params, mask, latest_bg_run) 
        for s in chunk_slices
    ]
    
    # Run detection
    with ProgressBar():
        results = dask.compute(*delayed_tasks)
    
    console.print("Writing detection results...")
    
    # A list to hold all the bounding box arrays from each chunk
    all_chunk_bboxes = []
    total_detections = 0
    
    # This loop just collects results, no writing to bbox_norm_coords yet
    for slc, counts, bboxes in tqdm(results, desc="Processing chunks"):
        n_detections[slc] = counts # This write is fine, as the array is pre-sized
        if bboxes: # Check if the list of bboxes is not empty
            all_chunk_bboxes.append(np.array(bboxes))
            total_detections += len(bboxes)

    console.print(f"Aggregating {total_detections} total detections...")

    # Now, resize and write just ONCE after the loop
    if total_detections > 0:
        # Resize the Zarr array to the final correct shape, passed as a TUPLE
        bbox_norm_coords.resize((total_detections, 4)) 
        
        # Concatenate all collected bounding boxes into one large array
        final_bboxes = np.concatenate(all_chunk_bboxes, axis=0)
        
        # Write all the data in a single operation
        bbox_norm_coords[:] = final_bboxes
    else:
        # If no detections were found, resize to an empty array, passed as a TUPLE
        bbox_norm_coords.resize((0, 4))

    # Calculate statistics (this part remains the same)
    frames_with_detections = np.sum(n_detections[:] > 0)
    percent_detected = (frames_with_detections / num_images) * 100 if num_images > 0 else 0
    
    # Store metadata
    duration = time.perf_counter() - start_time
    
    detect_group.attrs.update({
        'detect_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'duration_seconds': duration,
        'parameters': detect_params,
        'source_background_run': latest_bg_run,
        'dask_scheduler': scheduler,
        'summary_statistics': {
            'total_frames': num_images,
            'frames_with_detections': int(frames_with_detections),
            'total_detections': int(total_detections),
            'percent_frames_with_detections': round(percent_detected, 2)
        },
        'code_version': {
            'git_commit': git_info['commit_hash'],
            'git_short': git_info['short_hash'],
            'git_branch': git_info['branch'],
            'git_dirty': git_info['is_dirty'],
            'hostname': platform_info['hostname'],
            'system': platform_info['system'],
            'numpy_version': np.__version__,
            'scikit_image_version': skimage.__version__,
            'dask_version': dask.__version__,
            'lsf_job_id': platform_info.get('lsf', {}).get('job_id'),
            'slurm_job_id': platform_info.get('slurm', {}).get('job_id'),
        }
    })
    
    console.print(f"[green]Detection completed in {duration:.2f} seconds[/green]")
    console.print(f"Found [green]{total_detections}[/green] fish in [green]{frames_with_detections}/{num_images}[/green] frames ([cyan]{percent_detected:.2f}%[/cyan])")
    
    return {
        'duration_seconds': duration,
        'total_detections': total_detections,
        'frames_with_detections': frames_with_detections,
        'run_name': run_name
    }