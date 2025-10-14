"""
Crop ROIs from full-resolution frames based on detection results.
Part of the FishEye tracking pipeline.

This version supports multiple detection sources:
- 'detect': Original blob/YOLO detections
- 'filtered': Refined detections with jumps removed
- 'interpolated': Refined detections with gaps filled

Streams work with Dask and writes directly from workers to Zarr
to avoid accumulating large results in driver memory.
"""

import time
import zarr
import os
from zarr.codecs import BloscCodec
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn

# Dask imports
import dask
from dask import delayed
from dask.diagnostics import ProgressBar

# Optional distributed scheduler
try:
    from dask.distributed import Client, LocalCluster, as_completed
    HAVE_DISTRIBUTED = True
except Exception:
    HAVE_DISTRIBUTED = False

from ..utils.system import get_environment_info


def get_detection_source_info(
    root: zarr.Group,
    source_type: str = 'detect',
    console: Optional[Console] = None
) -> Tuple[str, zarr.Group, Optional[np.ndarray]]:
    """
    Get information about the detection source to use for cropping.
    
    Args:
        root: Zarr root group
        source_type: 'detect', 'filtered', or 'interpolated'
        console: Optional Rich console for output
        
    Returns:
        Tuple of (source_path, source_group, detection_source_array)
        - source_path: Path string like 'detect_runs/latest' or 'refined_runs/latest/filtered'
        - source_group: Zarr group containing the detection data
        - detection_source_array: Array indicating real (0) vs interpolated (1), or None
    """
    if source_type == 'detect':
        # Use original detections
        if 'detect_runs' not in root:
            raise ValueError("No detect_runs found in zarr file")
        
        latest = root['detect_runs'].attrs.get('latest')
        if latest is None:
            raise ValueError("No latest detect run found")
        
        source_path = f'detect_runs/{latest}'
        source_group = root[source_path]
        detection_source = None
        
        if console:
            console.print(f"[cyan]Using original detections:[/cyan] {latest}")
        
    elif source_type in ['filtered', 'interpolated']:
        # Use refined detections
        if 'refined_runs' not in root:
            raise ValueError("No refined_runs found. Run refinement pipeline first.")
        
        latest_refined = root['refined_runs'].attrs.get('latest')
        if latest_refined is None:
            raise ValueError("No latest refined run found")
        
        refined_group = root[f'refined_runs/{latest_refined}']
        
        if source_type not in refined_group:
            raise ValueError(f"Stage '{source_type}' not found in refined run {latest_refined}")
        
        source_path = f'refined_runs/{latest_refined}/{source_type}'
        source_group = refined_group[source_type]
        
        # Get detection source array for interpolated data
        detection_source = None
        if source_type == 'interpolated' and 'detection_source' in source_group:
            detection_source = source_group['detection_source'][:]
        
        if console:
            console.print(f"[cyan]Using refined detections ({source_type}):[/cyan] {latest_refined}")
            if detection_source is not None:
                n_real = np.sum(detection_source == 0)
                n_interp = np.sum(detection_source == 1)
                console.print(f"  Real detections: {n_real}, Interpolated: {n_interp}")
    
    else:
        raise ValueError(f"Invalid source_type: {source_type}. Must be 'detect', 'filtered', or 'interpolated'")
    
    return source_path, source_group, detection_source


def get_crop_parameters(
    root: zarr.Group,
    config: Dict[str, Any],
    console: Optional[Console] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Get crop parameters with zarr-first resolution.
    
    Priority order:
    1. Zarr analysis_metadata (if crop tuning exists)
    2. Config file defaults
    """
    # Start with config defaults
    crop_params = config.get('crop', {}).copy()
    crop_params.setdefault('roi_sz', [256, 256])
    
    param_source = 'config_default'
    
    # Check for tuned parameters in zarr (future: crop tuning)
    if 'analysis_metadata' in root:
        analysis_meta = root['analysis_metadata']
        
        # Future: if we add crop parameter tuning
        if 'crop_tuning' in analysis_meta.attrs:
            tuning_data = analysis_meta.attrs['crop_tuning']
            tuned_params = tuning_data.get('tuned_parameters', {})
            if tuned_params:
                crop_params.update(tuned_params)
                param_source = 'zarr_tuned'
                if console:
                    console.print(f"[green]✓ Using tuned crop parameters from zarr[/green]")
        
        # Check for mask tuning (this is the main one)
        if 'dish_mask' in analysis_meta.attrs:
            mask_data = analysis_meta.attrs['dish_mask']
            if 'detected_circle' in mask_data:
                if 'dish_mask' not in crop_params:
                    crop_params['dish_mask'] = {}
                crop_params['dish_mask'].update({
                    'shape': 'circle',
                    'center': mask_data['detected_circle']['center'],
                    'radius': mask_data['detected_circle']['radius']
                })
                if console:
                    console.print(f"[green]✓ Using tuned dish mask from zarr[/green]")
    
    return crop_params, param_source


# -------- Worker task: compute + WRITE directly into Zarr -------- #

@delayed
def crop_and_store_chunk_delayed(
    zarr_path: str,
    chunk_slice: slice,
    out_slice: Tuple[int, int],
    roi_sz: Tuple[int, int],
    scale_factor: float,
    source_path: str
) -> Dict[str, int]:
    """
    Crops ROIs for a chunk and writes them directly into the precreated Zarr arrays.

    Args:
        zarr_path: path to zarr archive
        chunk_slice: frames [start:stop] to process
        out_slice: (start_det, end_det) in the flattened detection space for this chunk
        roi_sz: (H, W) of the crop
        scale_factor: ds/full scale for coordinates_ds
        source_path: Path to detection source (e.g., 'detect_runs/latest' or 'refined_runs/latest/filtered')

    Returns:
        Tiny dict with counts/indices for bookkeeping.
    """
    root = zarr.open(zarr_path, mode='a')
    
    # Find the target crop group via root attrs (set by driver before dispatch)
    crop_group_path = root.attrs.get('current_crop_group_path')
    if crop_group_path is None:
        raise RuntimeError("Root attrs missing 'current_crop_group_path' for worker writes.")
    crop_group = root[crop_group_path]

    # Load full-resolution images
    images_full_chunk = root['raw_video/images_full'][chunk_slice]
    full_img_shape = images_full_chunk.shape[1:]  # (H, W)

    # Load detection data from specified source
    source_group = root[source_path]
    
    # Handle different data structures for detect vs refined
    if 'frame_mapping' in source_group:
        # Refined data: uses frame_mapping to connect detections to frames
        frame_mapping = source_group['frame_mapping'][:]
        bbox_coords = source_group['bbox_norm_coords'][:]
        
        # Find which detections correspond to this chunk's frames
        chunk_frames = np.arange(chunk_slice.start, chunk_slice.stop)
        mask = np.isin(frame_mapping, chunk_frames)
        
        detection_indices = np.where(mask)[0]
        bbox_coords_chunk = bbox_coords[detection_indices]
        frames_for_detections = frame_mapping[detection_indices]
        
        # Build n_per_frame for this chunk
        n_per_frame = np.zeros(len(chunk_frames), dtype=int)
        for i, frame in enumerate(chunk_frames):
            n_per_frame[i] = np.sum(frames_for_detections == frame)
        
    else:
        # Original detect data: uses n_detections array directly
        n_per_frame = source_group['n_detections'][chunk_slice]
        
        # Compute detection index range within detect_runs for this chunk
        start_detection_idx = int(np.sum(source_group['n_detections'][:chunk_slice.start]))
        end_detection_idx = start_detection_idx + int(np.sum(n_per_frame))
        bbox_coords_chunk = source_group['bbox_norm_coords'][start_detection_idx:end_detection_idx]
        
        # For detect runs, frame index corresponds to position in chunk
        frames_for_detections = None  # Will use sequential indexing

    start_det_out, end_det_out = out_slice
    count = end_det_out - start_det_out
    if count == 0:
        return {"frames": int(np.sum(n_per_frame)), "start": start_det_out, "end": end_det_out}

    # Allocate local buffers (live only within the worker)
    rois_buf = np.zeros((count, *roi_sz), dtype='uint8')
    coords_full_buf = np.zeros((count, 2), dtype='i4')
    coords_ds_buf = np.zeros((count, 2), dtype='i4')

    cursor_in = 0
    cursor_out = 0
    H, W = full_img_shape

    for i in range(len(images_full_chunk)):
        nd = int(n_per_frame[i])
        if nd == 0:
            continue

        img = images_full_chunk[i]
        
        for _ in range(nd):
            center_norm = bbox_coords_chunk[cursor_in][:2]  # (cx_norm, cy_norm)
            # Note: bbox coords are normalized w.r.t (W, H), so multiply in (W, H) order
            full_centroid_px = np.round(center_norm * np.array([W, H])).astype(int)

            x1 = int(full_centroid_px[0] - roi_sz[1] // 2)
            y1 = int(full_centroid_px[1] - roi_sz[0] // 2)

            # Extract ROI with padding if needed
            y2 = y1 + roi_sz[0]
            x2 = x1 + roi_sz[1]

            # Compute valid region within img
            vy1 = max(0, y1); vy2 = min(H, y2)
            vx1 = max(0, x1); vx2 = min(W, x2)

            if (vy2 - vy1) == roi_sz[0] and (vx2 - vx1) == roi_sz[1] and 0 <= y1 < H and 0 <= x1 < W:
                roi = img[vy1:vy2, vx1:vx2]
            else:
                # Pad when ROI extends outside edges
                roi = np.zeros(roi_sz, dtype='uint8')
                if vy2 > vy1 and vx2 > vx1:
                    py1 = max(0, -y1)
                    px1 = max(0, -x1)
                    py2 = py1 + (vy2 - vy1)
                    px2 = px1 + (vx2 - vx1)
                    roi[py1:py2, px1:px2] = img[vy1:vy2, vx1:vx2]

            rois_buf[cursor_out] = roi
            coords_full_buf[cursor_out] = (x1, y1)

            # Downsampled coords (integer)
            dx = int(x1 * scale_factor)
            dy = int(y1 * scale_factor)
            coords_ds_buf[cursor_out] = (dx, dy)

            cursor_in += 1
            cursor_out += 1

    # Single write per array per worker (targeting non-overlapping slices)
    crop_group['roi_images'][start_det_out:end_det_out] = rois_buf
    crop_group['roi_coordinates_full'][start_det_out:end_det_out] = coords_full_buf
    crop_group['roi_coordinates_ds'][start_det_out:end_det_out] = coords_ds_buf

    return {"frames": int(np.sum(n_per_frame)), "start": start_det_out, "end": end_det_out}


def crop_detections(
    zarr_path: str,
    config: Dict[str, Any],
    source_type: str = 'detect',
    scheduler: str = None,
    num_workers: Optional[int] = None,
    console: Optional[Console] = None
) -> Dict[str, Any]:
    """
    Main function to crop ROIs from full-resolution frames based on detections.
    
    Args:
        zarr_path: Path to zarr file
        config: Configuration dictionary
        source_type: Detection source - 'detect', 'filtered', or 'interpolated'
        scheduler: Dask scheduler ('processes', 'threads', or 'distributed')
        num_workers: Number of workers (None = auto)
        console: Optional Rich console for output
    
    Returns:
        Dictionary with cropping statistics
    """
    if console is None:
        console = Console()

    console.rule("[bold]Stage: Cropping ROIs from Detections[/bold]")
    start_time = time.perf_counter()

    root = zarr.open_group(zarr_path, mode='a')

    # Get detection source information
    source_path, source_group, detection_source = get_detection_source_info(
        root, source_type, console
    )

    # Get crop parameters including scheduler settings
    crop_params, param_source = get_crop_parameters(root, config, console)
    
    # Use config values if not explicitly provided
    if scheduler is None:
        scheduler = crop_params.get('scheduler', 'processes')
    if num_workers is None:
        num_workers = crop_params.get('num_workers', None)
    
    # Determine if we'll use distributed BEFORE building metadata
    use_distributed = (scheduler == "distributed") and HAVE_DISTRIBUTED
    
    roi_sz = tuple(crop_params.get('roi_sz', [256, 256]))
    chunk_size = config.get('import', {}).get('chunk_size', 32)

    console.print(f"ROI size: {roi_sz[0]}×{roi_sz[1]} pixels")
    console.print(f"Chunk size: {chunk_size} frames")
    console.print(f"Scheduler: {scheduler}, Workers: {num_workers or 'default'}")

    # Create run group
    from ..shared.zarr.schema import get_run_group
    crop_group, run_group_name = get_run_group(root, 'crop', console)

    # Build initial metadata dictionary
    metadata_dict = {
        'crop_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'dask_scheduler': scheduler,
        'dask_num_workers': num_workers or os.cpu_count(),
        'parameters': crop_params,
        'parameter_source': param_source,
        'detection_source_type': source_type,
        'detection_source_path': source_path,
        'roi_size': roi_sz
    }

    # Store initial metadata
    crop_group.attrs.update(metadata_dict)

    # Get detection counts
    if 'frame_mapping' in source_group:
        # Refined data
        n_detections_array = source_group['n_detections'][:]
        bbox_coords = source_group['bbox_norm_coords'][:]
        total_detections = len(bbox_coords)
    else:
        # Original detect data
        n_detections_array = source_group['n_detections'][:]
        total_detections = int(n_detections_array.sum())
    
    num_images = len(n_detections_array)

    if total_detections == 0:
        console.print("[yellow]Warning: No detections found. Nothing to crop.[/yellow]")
        return {'total_crops': 0}

    console.print(f"Total detections to crop: {total_detections:,}")
    console.print(f"Total frames: {num_images:,}")

    # Get video dimensions and scale factor
    ds_img_shape = root['raw_video/images_ds'].shape[1:]
    full_img_shape = root['raw_video/images_full'].shape[1:]
    scale_factor = ds_img_shape[0] / full_img_shape[0]

    # Compressor
    compressor = BloscCodec(typesize=1, cname='lz4', clevel=1, shuffle="bitshuffle")

    # Create output arrays in crop group
    roi_images = crop_group.create_array(
        'roi_images',
        shape=(total_detections, *roi_sz),
        chunks=(min(chunk_size * 4, total_detections), roi_sz[0], roi_sz[1]),
        dtype='uint8',
        overwrite=True,
        compressors=compressor
    )
    
    roi_coordinates_full = crop_group.create_array(
        'roi_coordinates_full',
        shape=(total_detections, 2),
        chunks=(min(chunk_size * 8, total_detections), 2),
        dtype='i4',
        overwrite=True
    )
    
    roi_coordinates_ds = crop_group.create_array(
        'roi_coordinates_ds',
        shape=(total_detections, 2),
        chunks=(min(chunk_size * 8, total_detections), 2),
        dtype='i4',
        overwrite=True
    )
    
    # If using interpolated data, save the detection source array
    if detection_source is not None:
        crop_group.create_array(
            'detection_source',
            data=detection_source,
            chunks=(min(chunk_size * 8, len(detection_source)),),
            dtype='i1',
            overwrite=True
        )
        crop_group.attrs['includes_interpolated'] = True
        crop_group.attrs['n_real_detections'] = int(np.sum(detection_source == 0))
        crop_group.attrs['n_interpolated_detections'] = int(np.sum(detection_source == 1))
    else:
        crop_group.attrs['includes_interpolated'] = False

    # Store path to this crop group in root for workers to find
    root.attrs['current_crop_group_path'] = crop_group.path

    # Build chunk frame slices
    chunk_slices = [slice(i, min(i + chunk_size, num_images))
                    for i in range(0, num_images, chunk_size)]
    console.print(f"Creating [yellow]{len(chunk_slices)}[/yellow] Dask tasks for cropping...")

    # Precompute cumulative detection offsets ONCE on driver
    cumulative_detections = np.cumsum(np.insert(n_detections_array, 0, 0))
    
    # Build chunks with output slices
    chunks = []
    for chunk_slice in chunk_slices:
        start_det = int(cumulative_detections[chunk_slice.start])
        end_det = int(cumulative_detections[chunk_slice.stop])
        
        if end_det > start_det:  # Only add if chunk has detections
            chunks.append((chunk_slice, (start_det, end_det)))
    
    frames_with_crops = int(np.sum(n_detections_array > 0))

    # Create delayed tasks
    delayed_tasks = [
        crop_and_store_chunk_delayed(
            zarr_path, frame_slice, out_slice, roi_sz, scale_factor, source_path
        )
        for frame_slice, out_slice in chunks
    ]
    
    # Dask config and scheduler
    dask.config.set({
        "distributed.worker.memory.target": 0.65,
        "distributed.worker.memory.spill": 0.75,
        "distributed.worker.memory.pause": 0.90,
        "distributed.worker.memory.terminate": 0.98,
    })
    
    client = None

    # Execute based on scheduler
    if use_distributed:
        # Distributed execution with Rich progress bar
        client = Client()
        console.print(f"[green]Dask distributed dashboard:[/green] {client.dashboard_link}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Cropping chunks (distributed)...", total=len(delayed_tasks))
            
            futures = client.compute(delayed_tasks)
            for future in as_completed(futures):
                _ = future.result()
                progress.update(task, advance=1)
        
        client.close()
    
    else:
        # Local execution with Rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Cropping chunks ({scheduler})...", total=len(delayed_tasks))
            
            for d in delayed_tasks:
                _ = d.compute()
                progress.update(task, advance=1)

    # Summary stats and attrs
    percent_cropped = (frames_with_crops / num_images) * 100 if num_images > 0 else 0
    summary_stats = {
        'total_frames': num_images,
        'frames_with_crops': frames_with_crops,
        'total_rois_cropped': total_detections,
        'percent_frames_with_crops': round(percent_cropped, 2),
        'roi_size': list(roi_sz),
        'scale_factor': float(scale_factor)
    }
    crop_group.attrs['summary_statistics'] = summary_stats
    duration = time.perf_counter() - start_time
    crop_group.attrs['duration_seconds'] = duration

    # Environment info
    env_info = get_environment_info()
    crop_group.attrs.update({
        'git_commit': env_info['git'].get('commit_hash', 'unknown'),
        'git_branch': env_info['git'].get('branch', 'unknown'),
        'hostname': env_info['platform']['hostname']
    })

    # Clean up root attrs
    if 'current_crop_group_path' in root.attrs:
        del root.attrs['current_crop_group_path']

    # Create completion panel
    source_info = f"{source_type}"
    if detection_source is not None:
        n_real = int(np.sum(detection_source == 0))
        n_interp = int(np.sum(detection_source == 1))
        source_info += f" ({n_real} real + {n_interp} interpolated)"
    
    completion_text = f"""[green]✓[/green] Cropping completed successfully

[bold]Performance:[/bold]
  Time: {duration:.1f}s ({duration/60:.1f} min)
  ROIs/sec: {total_detections/duration:.1f}
  Throughput: {(total_detections * roi_sz[0] * roi_sz[1]) / (1024*1024*duration):.2f} MP/s

[bold]Output:[/bold]
  Path: {zarr_path}
  Detection source: {source_info}

[bold]Arrays created:[/bold]
  - crop_runs/{run_group_name}/roi_images: ({total_detections}, {roi_sz[0]}, {roi_sz[1]})
  - crop_runs/{run_group_name}/roi_coordinates_full: ({total_detections}, 2)
  - crop_runs/{run_group_name}/roi_coordinates_ds: ({total_detections}, 2)"""
    
    if detection_source is not None:
        completion_text += f"\n  - crop_runs/{run_group_name}/detection_source: ({total_detections},)"

    console.print(Panel(
        Align.center(completion_text),
        title="[bold green]Cropping Complete[/bold green]",
        border_style="green"
    ))

    return {
        'total_crops': total_detections,
        'frames_with_crops': frames_with_crops,
        'percent_cropped': percent_cropped,
        'duration_seconds': duration,
        'detection_source_type': source_type
    }


def main():
    """CLI entry point."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Crop ROIs from detections")
    parser.add_argument("zarr_path", type=str, help="Path to zarr file")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--source-type", type=str, default='detect',
                       choices=['detect', 'filtered', 'interpolated'],
                       help="Detection source to use")
    parser.add_argument("--scheduler", type=str, default=None,
                       choices=['processes', 'threads', 'distributed'],
                       help="Dask scheduler type")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of workers")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    console = Console()
    
    # Warn if distributed requested but not available
    if args.scheduler == "distributed" and not HAVE_DISTRIBUTED:
        console.print("[yellow]Warning: distributed scheduler not available, falling back to processes[/yellow]")
        args.scheduler = "processes"

    try:
        results = crop_detections(
            zarr_path=args.zarr_path,
            config=config,
            source_type=args.source_type,
            scheduler=args.scheduler,
            num_workers=args.num_workers,
            console=console
        )
        console.print(f"\n[green]Cropping complete![/green]")
        console.print(f"Total ROIs cropped: {results['total_crops']}")
        console.print(f"Detection source: {results['detection_source_type']}")
        return 0
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())