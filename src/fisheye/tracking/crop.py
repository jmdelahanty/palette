"""
Crop ROIs from full-resolution frames based on detection results.
Part of the FishEye tracking pipeline.

This version streams work with Dask and writes directly from workers to Zarr
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


# -------- Worker task: compute + WRITE directly into Zarr -------- #

@delayed
def crop_and_store_chunk_delayed(
    zarr_path: str,
    chunk_slice: slice,
    out_slice: Tuple[int, int],
    roi_sz: Tuple[int, int],
    scale_factor: float
) -> Dict[str, int]:
    """
    Crops ROIs for a chunk and writes them directly into the precreated Zarr arrays.

    Args:
        zarr_path: path to zarr archive
        chunk_slice: frames [start:stop] to process
        out_slice: (start_det, end_det) in the flattened detection space for this chunk
        roi_sz: (H, W) of the crop
        scale_factor: ds/full scale for coordinates_ds

    Returns:
        Tiny dict with counts/indices for bookkeeping.
    """
    root = zarr.open(zarr_path, mode='a')
    # Find the target crop group via root attrs (set by driver before dispatch)
    crop_group_path = root.attrs.get('current_crop_group_path')
    if crop_group_path is None:
        raise RuntimeError("Root attrs missing 'current_crop_group_path' for worker writes.")
    crop_group = root[crop_group_path]

    images_full_chunk = root['raw_video/images_full'][chunk_slice]
    full_img_shape = images_full_chunk.shape[1:]  # (H, W)

    latest = root['detect_runs'].attrs['latest']
    n_per_frame = root[f'detect_runs/{latest}/n_detections'][chunk_slice]

    # Compute detection index range within detect_runs for this chunk
    start_detection_idx = int(np.sum(root[f'detect_runs/{latest}/n_detections'][:chunk_slice.start]))
    end_detection_idx = start_detection_idx + int(np.sum(n_per_frame))
    bbox_coords_chunk = root[f'detect_runs/{latest}/bbox_norm_coords'][start_detection_idx:end_detection_idx]

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
    scheduler: str = None,
    num_workers: Optional[int] = None,
    console: Optional[Console] = None
) -> Dict[str, Any]:
    """
    Main function to crop ROIs from full-resolution frames based on detections.
    """
    if console is None:
        console = Console()

    console.rule("[bold]Stage: Cropping ROIs from Detections[/bold]")
    start_time = time.perf_counter()

    root = zarr.open_group(zarr_path, mode='a')

    # Get crop parameters including scheduler settings
    crop_params = config.get('crop', {})
    
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
    crop_group, run_group_name = get_run_group(root, 'crop', console)
    latest_detect_run = root['detect_runs'].attrs['latest']

    # Build initial metadata dictionary
    metadata_dict = {
        'crop_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'dask_scheduler': scheduler,
        'dask_num_workers': num_workers or os.cpu_count(),
        'parameters': crop_params,
        'source_detect_run': latest_detect_run,
        'roi_size': roi_sz
    }

    # Store initial metadata
    crop_group.attrs.update(metadata_dict)

    # Detection info
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
        return {'total_crops': 0, 'frames_with_crops': 0}

    console.print(f"Total detections to crop: [green]{total_detections}[/green]")

    # Compressor
    compressor = BloscCodec(typesize=1, cname='lz4', clevel=1, shuffle="bitshuffle")

    # Create output arrays
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

    ds_img_shape = root['raw_video/images_ds'].shape[1:]
    full_img_shape = root['raw_video/images_full'].shape[1:]
    scale_factor = ds_img_shape[0] / full_img_shape[0]

    roi_coordinates_ds = crop_group.create_array(
        'roi_coordinates_ds',
        shape=(total_detections, 2),
        chunks=(min(chunk_size * 8, total_detections), 2),
        dtype='i4',
        overwrite=True
    )

    # Build chunk frame slices
    chunk_slices = [slice(i, min(i + chunk_size, num_images))
                    for i in range(0, num_images, chunk_size)]
    console.print(f"Creating [yellow]{len(chunk_slices)}[/yellow] Dask tasks for cropping...")

    # Precompute cumulative detection offsets ONCE on driver
    cumulative_detections = np.cumsum(np.insert(n_detections, 0, 0))

    # Mark the crop group path so workers can reopen it
    root.attrs['current_crop_group_path'] = crop_group.path

    # Dask config and scheduler
    dask.config.set({
        "distributed.worker.memory.target": 0.65,
        "distributed.worker.memory.spill": 0.75,
        "distributed.worker.memory.pause": 0.90,
        "distributed.worker.memory.terminate": 0.98,
    })

    client = None
    if use_distributed:
        # Processes are best for NumPy/Zarr workloads
        cluster = LocalCluster(
            processes=True,
            n_workers=num_workers or os.cpu_count(),
            threads_per_worker=2,
            memory_limit="10GiB",
            local_directory=os.environ.get("DASK_DISTRIBUTED__WORKER__LOCAL_DIRECTORY", None),
        )
        client = Client(cluster)
        console.print(f"Dask dashboard: {client.dashboard_link}")
        
        # NOW update metadata with distributed config
        crop_group.attrs['distributed_config'] = {
            'processes': True,
            'threads_per_worker': 2,
            'memory_limit': '10GiB',
            'local_directory': os.environ.get("DASK_DISTRIBUTED__WORKER__LOCAL_DIRECTORY", None),
            'dashboard_link': client.dashboard_link,
            'memory_target': 0.65,
            'memory_spill': 0.75,
            'memory_pause': 0.90,
            'memory_terminate': 0.98
        }
    else:
        # Fall back to local schedulers
        if scheduler not in {"processes", "threads", "single-threaded"}:
            console.print("[yellow]Unknown scheduler; using 'processes'.[/yellow]")
            scheduler = "processes"
        dask.config.set(scheduler=scheduler,
                        num_workers=num_workers or os.cpu_count())
        # For local schedulers, capture the actual number used
        crop_group.attrs['actual_num_workers'] = dask.config.get('num_workers', os.cpu_count())

    # Create delayed tasks
    delayed_tasks = []
    for slc in chunk_slices:
        start_idx = int(cumulative_detections[slc.start])
        end_idx = int(cumulative_detections[slc.stop])
        if end_idx == start_idx:
            continue
        d = crop_and_store_chunk_delayed(
            zarr_path, slc, (start_idx, end_idx), roi_sz, float(scale_factor)
        )
        delayed_tasks.append(d)

    frames_with_crops = int(np.sum(n_detections > 0))
    # Execute with streaming to avoid materializing all results
    console.print("Processing chunks (streaming writes from workers)...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    ) as progress:
        
        if use_distributed:
            futures = client.compute(delayed_tasks)
            n_workers_actual = len(client.cluster.workers) if client.cluster.workers else num_workers or os.cpu_count()
            task = progress.add_task(f"[cyan]Cropping chunks (distributed, {n_workers_actual} workers)...", total=len(futures))
            
            for fut in as_completed(futures):
                _ = fut.result()
                progress.update(task, advance=1)
        else:
            # Use Rich for local schedulers too
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

    # Create completion panel
    completion_text = f"""[green]✓[/green] Cropping completed successfully

[bold]Performance:[/bold]
  Time: {duration:.1f}s ({duration/60:.1f} min)
  ROIs/sec: {total_detections/duration:.1f}
  Throughput: {(total_detections * roi_sz[0] * roi_sz[1]) / (1024*1024*duration):.2f} MP/s

[bold]Output:[/bold]
  Path: {zarr_path}

[bold]Arrays created:[/bold]
  - crop_runs/{run_group_name}/roi_images: ({total_detections}, {roi_sz[0]}, {roi_sz[1]})
  - crop_runs/{run_group_name}/roi_coordinates_full: ({total_detections}, 2)
  - crop_runs/{run_group_name}/roi_coordinates_ds: ({total_detections}, 2)

[bold]Statistics:[/bold]
  Total ROIs: {total_detections:,}
  Frames with crops: {frames_with_crops:,}/{num_images:,} ({percent_cropped:.1f}%)"""

    # Add scheduler info
    if use_distributed:
        completion_text += f"\n  Scheduler: distributed ({num_workers or os.cpu_count()} workers)"
    else:
        completion_text += f"\n  Scheduler: {scheduler} ({dask.config.get('num_workers', os.cpu_count())} workers)"

    panel = Panel(
        completion_text,
        title="[bold]Crop Complete[/bold]",
        border_style="green",
        padding=(1, 2)
    )

    console.print("\n")
    console.print(panel)

    # Cleanup distributed client/cluster
    if use_distributed and client is not None:
        client.close()
        cluster.close()

    return {
        'total_crops': total_detections,
        'frames_with_crops': frames_with_crops,
        'percent_cropped': percent_cropped,
        'duration': duration,
        'run_name': run_group_name
    }


def get_run_group(root: zarr.Group, stage_name: str, console: Console) -> Tuple[zarr.Group, str]:
    parent_group_name = f"{stage_name}_runs"
    parent_group = root.require_group(parent_group_name)
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    run_group_name = f"{stage_name}_{timestamp}"
    run_group = parent_group.create_group(run_group_name)
    parent_group.attrs['latest'] = run_group_name
    console.print(f"Created new run group: [cyan]{run_group.path}[/cyan]")
    return run_group, run_group_name


# ---- CLI ---- #

def main():
    """Command-line interface for crop stage."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Crop ROIs from detected fish")
    parser.add_argument("zarr_path", help="Path to zarr archive")
    parser.add_argument("--config", default="configs/fisheye/default.yaml",
                       help="Configuration file")
    parser.add_argument("--scheduler", default="processes",
                       choices=["processes", "threads", "single-threaded", "distributed"],
                       help="Dask scheduler to use")
    parser.add_argument("--num-workers", type=int,
                       help="Number of workers (default: CPU count)")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    console = Console()

    # Friendly hint if they asked for distributed but it's not installed
    if args.scheduler == "distributed" and not HAVE_DISTRIBUTED:
        console.print("[yellow]dask.distributed not found; falling back to 'processes'.[/yellow]")
        args.scheduler = "processes"

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
        return 0
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())