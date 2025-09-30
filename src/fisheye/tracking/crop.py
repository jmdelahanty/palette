"""
Crop ROIs from full-resolution frames based on detection results.
Part of the FishEye tracking pipeline.
"""

import time
import zarr
from zarr.codecs import Blosc
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from rich.console import Console
from tqdm import tqdm

# Dask imports
import dask
from dask import delayed
from dask.diagnostics import ProgressBar

from ..utils.system import get_environment_info


@delayed
def crop_chunk_from_bbox_delayed(
    zarr_path: str,
    chunk_slice: slice,
    roi_sz: Tuple[int, int]
) -> Tuple[slice, List[np.ndarray], List[Tuple[int, int]]]:
    """
    Crops ROIs from full-resolution frames based on pre-computed bounding boxes.
    
    Args:
        zarr_path: Path to zarr archive
        chunk_slice: Slice of frames to process
        roi_sz: Size of ROI to extract (height, width)
    
    Returns:
        Tuple of (chunk_slice, roi_images, roi_coordinates_full)
    """
    with zarr.open(zarr_path, mode='r') as root:
        images_full_chunk = root['raw_video/images_full'][chunk_slice]
        full_img_shape = images_full_chunk.shape[1:]
        
        # Load detection data for this chunk
        latest_detect_run = root['detect_runs'].attrs['latest']
        n_detections_per_frame = root[f'detect_runs/{latest_detect_run}/n_detections'][chunk_slice]
        
        # Calculate detection indices for this chunk
        start_detection_idx = np.sum(root[f'detect_runs/{latest_detect_run}/n_detections'][:chunk_slice.start])
        end_detection_idx = start_detection_idx + np.sum(n_detections_per_frame)
        detection_slice = slice(int(start_detection_idx), int(end_detection_idx))
        
        bbox_coords_chunk = root[f'detect_runs/{latest_detect_run}/bbox_norm_coords'][detection_slice]

    all_rois = []
    all_coords_full = []
    bbox_cursor = 0

    for i in range(len(images_full_chunk)):
        num_detections_in_frame = int(n_detections_per_frame[i])
        if num_detections_in_frame == 0:
            continue

        for _ in range(num_detections_in_frame):
            # Get normalized center coordinates
            center_norm = bbox_coords_chunk[bbox_cursor][:2]
            
            # Convert to pixel coordinates
            full_centroid_px = np.round(center_norm * np.array(full_img_shape)[::-1]).astype(int)
            
            # Calculate ROI bounds
            roi_x1_full = full_centroid_px[0] - roi_sz[1] // 2
            roi_y1_full = full_centroid_px[1] - roi_sz[0] // 2
            
            # Extract ROI
            roi = images_full_chunk[i][roi_y1_full:roi_y1_full+roi_sz[0], 
                                       roi_x1_full:roi_x1_full+roi_sz[1]]
            
            # Handle edge cases with padding
            if roi.shape != tuple(roi_sz):
                padded_roi = np.zeros(roi_sz, dtype='uint8')
                # Calculate valid region
                valid_y1 = max(0, roi_y1_full)
                valid_y2 = min(full_img_shape[0], roi_y1_full + roi_sz[0])
                valid_x1 = max(0, roi_x1_full)
                valid_x2 = min(full_img_shape[1], roi_x1_full + roi_sz[1])
                
                # Calculate positions in padded array
                pad_y1 = max(0, -roi_y1_full)
                pad_y2 = pad_y1 + (valid_y2 - valid_y1)
                pad_x1 = max(0, -roi_x1_full)
                pad_x2 = pad_x1 + (valid_x2 - valid_x1)
                
                # Copy valid region
                if pad_y2 > pad_y1 and pad_x2 > pad_x1:
                    padded_roi[pad_y1:pad_y2, pad_x1:pad_x2] = images_full_chunk[i][valid_y1:valid_y2, valid_x1:valid_x2]
                
                roi = padded_roi
            
            all_rois.append(roi)
            all_coords_full.append((roi_x1_full, roi_y1_full))
            bbox_cursor += 1

    return chunk_slice, all_rois, all_coords_full


def crop_detections(
    zarr_path: str,
    config: Dict[str, Any],
    scheduler: str = 'processes',
    num_workers: Optional[int] = None,
    console: Optional[Console] = None
) -> Dict[str, Any]:
    """
    Main function to crop ROIs from full-resolution frames based on detections.
    
    Args:
        zarr_path: Path to zarr archive
        config: Configuration dictionary with crop parameters
        scheduler: Dask scheduler to use
        num_workers: Number of workers for parallel processing
        console: Rich console for output
        
    Returns:
        Dictionary with cropping results and statistics
    """
    if console is None:
        console = Console()
    
    console.rule("[bold]Stage: Cropping ROIs from Detections[/bold]")
    start_time = time.perf_counter()
    
    # Open zarr archive
    root = zarr.open_group(zarr_path, mode='a')
    
    # Check prerequisites
    if 'detect_runs' not in root:
        raise ValueError("Detection stage has not been run. Please run detection first.")
    
    # Get crop parameters
    crop_params = config.get('crop', {})
    roi_sz = tuple(crop_params.get('roi_sz', [320, 320]))
    chunk_size = config.get('import', {}).get('chunk_size', 32)
    
    console.print(f"ROI size: {roi_sz[0]}×{roi_sz[1]} pixels")
    console.print(f"Chunk size: {chunk_size} frames")
    
    # Create new run group for this crop
    crop_group = get_run_group(root, 'crop', console)
    latest_detect_run = root['detect_runs'].attrs['latest']
    
    # Store metadata
    crop_group.attrs.update({
        'crop_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'dask_scheduler': scheduler,
        'parameters': crop_params,
        'source_detect_run': latest_detect_run,
        'roi_size': roi_sz
    })
    
    # Get detection information
    detect_group = root[f'detect_runs/{latest_detect_run}']
    n_detections = detect_group['n_detections'][:]
    total_detections = int(n_detections.sum())
    num_images = len(n_detections)
    
    if total_detections == 0:
        console.print("[yellow]Warning: No detections found. Nothing to crop.[/yellow]")
        crop_group.attrs['summary_statistics'] = {
            'total_frames': num_images,
            'frames_with_crops': 0,
            'total_rois_cropped': 0,
            'percent_frames_with_crops': 0.0
        }
        return {
            'total_crops': 0,
            'frames_with_crops': 0
        }
    
    console.print(f"Total detections to crop: [green]{total_detections}[/green]")
    
    # Create Zarr arrays for cropped data
    compressor = Blosc(cname='lz4', clevel=1, shuffle='bitshuffle')
    
    roi_images = crop_group.create_array(
        'roi_images',
        shape=(total_detections, *roi_sz),
        chunks=(min(chunk_size * 4, total_detections), None, None),
        dtype='uint8',
        overwrite=True,
        compressors=compressor
    )

    roi_coordinates_full = crop_group.create_array(
        'roi_coordinates_full',
        shape=(total_detections, 2),
        chunks=(min(chunk_size * 8, total_detections), None),
        dtype='i4',
        overwrite=True
    )
    
    # Also store downsampled coordinates for convenience
    ds_img_shape = root['raw_video/images_ds'].shape[1:]
    full_img_shape = root['raw_video/images_full'].shape[1:]
    scale_factor = ds_img_shape[0] / full_img_shape[0]
    
    roi_coordinates_ds = crop_group.create_array(
        'roi_coordinates_ds',
        shape=(total_detections, 2),
        chunks=(min(chunk_size * 8, total_detections), None),
        dtype='i4',
        overwrite=True
    )
    
    # Create chunk slices for parallel processing
    chunk_slices = [slice(i, min(i + chunk_size, num_images)) 
                    for i in range(0, num_images, chunk_size)]
    
    console.print(f"Creating [yellow]{len(chunk_slices)}[/yellow] Dask tasks for cropping...")
    
    # Set up Dask
    dask.config.set(scheduler=scheduler, num_workers=num_workers or os.cpu_count())
    
    # Create delayed tasks
    delayed_tasks = [
        crop_chunk_from_bbox_delayed(zarr_path, s, roi_sz)
        for s in chunk_slices
    ]
    
    # Execute tasks with progress bar
    console.print("Processing chunks...")
    with ProgressBar():
        results = dask.compute(*delayed_tasks)
    
    # Write results to Zarr
    console.print("Writing cropped ROIs to Zarr...")
    
    # Calculate cumulative detections for indexing
    cumulative_detections = np.cumsum(np.insert(n_detections, 0, 0))
    
    for slc, rois, coords_full in tqdm(results, desc="Writing crop chunks"):
        # Calculate indices for this chunk
        start_idx = int(cumulative_detections[slc.start])
        end_idx = int(cumulative_detections[slc.stop])
        
        if end_idx > start_idx and rois:
            # Write ROI images
            roi_images[start_idx:end_idx] = rois
            
            # Write full resolution coordinates
            roi_coordinates_full[start_idx:end_idx] = coords_full
            
            # Calculate and write downsampled coordinates
            coords_ds = [(int(x * scale_factor), int(y * scale_factor)) 
                         for x, y in coords_full]
            roi_coordinates_ds[start_idx:end_idx] = coords_ds
    
    # Calculate summary statistics
    frames_with_crops = int(np.sum(n_detections > 0))
    percent_cropped = (frames_with_crops / num_images) * 100 if num_images > 0 else 0
    
    summary_stats = {
        'total_frames': num_images,
        'frames_with_crops': frames_with_crops,
        'total_rois_cropped': total_detections,
        'percent_frames_with_crops': round(percent_cropped, 2),
        'roi_size': list(roi_sz),
        'scale_factor': scale_factor
    }
    
    # Store summary and duration
    crop_group.attrs['summary_statistics'] = summary_stats
    duration = time.perf_counter() - start_time
    crop_group.attrs['duration_seconds'] = duration
    
    # Add environment info
    env_info = get_environment_info()
    crop_group.attrs.update({
        'git_commit': env_info['git'].get('commit_hash', 'unknown'),
        'git_branch': env_info['git'].get('branch', 'unknown'),
        'hostname': env_info['platform']['hostname']
    })
    
    console.print(f"[green]✓[/green] Cropping completed in [green]{duration:.2f}[/green] seconds")
    console.print(f"  Cropped [green]{total_detections}[/green] ROIs from [green]{frames_with_crops}/{num_images}[/green] frames ([cyan]{percent_cropped:.2f}%[/cyan])")
    
    return {
        'total_crops': total_detections,
        'frames_with_crops': frames_with_crops,
        'percent_cropped': percent_cropped,
        'duration': duration
    }


def get_run_group(root: zarr.Group, stage_name: str, console: Console) -> zarr.Group:
    """
    Create a new timestamped run group for a pipeline stage.
    
    Args:
        root: Zarr root group
        stage_name: Name of the pipeline stage
        console: Rich console for output
    
    Returns:
        New zarr group for this run
    """
    parent_group_name = f"{stage_name}_runs"
    parent_group = root.require_group(parent_group_name)
    
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    run_group_name = f"{stage_name}_{timestamp}"
    run_group = parent_group.create_group(run_group_name)
    
    # Update latest pointer
    parent_group.attrs['latest'] = run_group_name
    
    console.print(f"Created new run group: [cyan]{run_group.path}[/cyan]")
    return run_group


# For command-line usage
def main():
    """Command-line interface for crop stage."""
    import argparse
    import yaml
    import os
    
    parser = argparse.ArgumentParser(description="Crop ROIs from detected fish")
    parser.add_argument("zarr_path", help="Path to zarr archive")
    parser.add_argument("--config", default="configs/fisheye/default.yaml", 
                       help="Configuration file")
    parser.add_argument("--scheduler", default="processes",
                       choices=["processes", "threads", "single-threaded"],
                       help="Dask scheduler to use")
    parser.add_argument("--num-workers", type=int, 
                       help="Number of workers (default: CPU count)")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    console = Console()
    
    try:
        results = crop_detections(
            zarr_path=args.zarr_path,
            config=config,
            scheduler=args.scheduler,
            num_workers=args.num_workers,
            console=console
        )
        
        console.print(f"\n[green]Cropping complete![/green]")
        console.print(f"Total ROIs cropped: {results['total_crops']}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())