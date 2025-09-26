"""
Improved video import functionality for FishEye using Zarr.
GPU-accelerated decoding with parallel I/O.
"""

import zarr
import numpy as np
import torch
import torch.nn.functional as F
import decord
import imageio.v3 as iio
import cv2
import queue
import threading
import time
from math import lcm
from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel

from ..utils.system import get_git_info, get_platform_info, get_gpu_info, get_environment_info
from ..shared.zarr.schema import create_palette_zarr, update_import_duration


def import_video(
    video_path: str,
    zarr_path: str,
    config: Dict[str, Any],
    cli_args: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
    use_gpu: bool = True,
    force_cpu: bool = False
) -> zarr.Group:
    """
    Import video into Zarr format with GPU acceleration and parallel I/O.
    
    Args:
        video_path: Path to input video file
        zarr_path: Path to output zarr file
        config: Pipeline configuration dict
        cli_args: Optional command line arguments to store
        console: Optional Rich console for output
        use_gpu: Whether to attempt GPU decoding
        force_cpu: Force CPU processing even if GPU available
        
    Returns:
        Root zarr group
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be decoded
    """
    if console is None:
        console = Console()
    
    # Validate inputs
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    console.rule("[bold]Stage 1: Importing Video[/bold]")
    start_time = time.perf_counter()
    
    # Setup video decoding
    device, vr = _setup_video_reader(video_path, use_gpu, force_cpu, console)
    
    # Get video properties
    n_frames = len(vr)
    if n_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Get first frame to determine dimensions
    first_frame = vr[0]
    if device == 'cuda:0':
        full_height, full_width = first_frame.shape[0], first_frame.shape[1]
    else:
        full_height, full_width = first_frame.shape[0], first_frame.shape[1]
    
    # Get import parameters from config
    import_params = config.get('import', {})
    ds_size = tuple(import_params.get('downsample_size', [640, 640]))
    chunk_size = import_params.get('chunk_size', 32)
    batch_size = import_params.get('batch_size', 16)
    
    # Calculate optimal chunk size (matching tracker.py logic)
    optimal_chunk_size = min(64, chunk_size * 2)
    
    # Get comprehensive video metadata
    video_metadata = _get_video_metadata(video_path, vr, full_width, full_height, n_frames)
    
    # Create Zarr hierarchy with optimal chunk size
    config_with_optimal = config.copy()
    config_with_optimal['import']['chunk_size'] = optimal_chunk_size
    
    root = create_palette_zarr(
        str(zarr_path),
        video_metadata,
        config_with_optimal,
        cli_args=cli_args
    )
    
    # Get the raw_video group and arrays
    raw_video_group = root['raw_video']
    images_full = raw_video_group['images_full']
    images_ds = raw_video_group['images_ds']

    # --- Shard alignment ---------------------------------------------------------
    # Read frames-per-shard we recorded in root attrs during schema creation.
    sharding_info = raw_video_group.attrs.get('sharding', {})
    sf_full = int(sharding_info.get('images_full', {}).get('frames_per_shard', 1))
    sf_ds   = int(sharding_info.get('images_ds',   {}).get('frames_per_shard', 1))
    write_quantum = lcm(max(sf_full, 1), max(sf_ds, 1))

    target = batch_size * 4
    max_shards_per_batch = 3
    k = max(1, min(max_shards_per_batch, target // write_quantum))
    io_batch_size = write_quantum * k


    raw_video_group.attrs['io_alignment'] = {
        'frames_per_shard_full': sf_full,
        'frames_per_shard_ds': sf_ds,
        'write_quantum_frames': write_quantum,
        'io_batch_size_aligned': io_batch_size,
        'max_shards_per_batch': max_shards_per_batch,
    }

    raw_video_group.attrs.update({
        'actual_chunk_size': optimal_chunk_size,
        'batch_size': batch_size,
        'io_batch_size': io_batch_size,   # aligned
        'decoding_device': device
    })

    # Console output
    console.print(f"Video: [cyan]{video_path.name}[/cyan]")
    console.print(f"Frames: [yellow]{n_frames}[/yellow]")
    console.print(f"Original: {full_height}×{full_width}")
    console.print(f"Downsampled: {ds_size[0]}×{ds_size[1]}")
    console.print(f"Chunk size: {optimal_chunk_size}")
    console.print(f"Device: [green]{device}[/green]")
    console.print(f"I/O batch (aligned): {io_batch_size} frames "
              f"(quantum={write_quantum}, shard_full={sf_full}, shard_ds={sf_ds})")

    
    # Determine max writers based on CPU cores
    max_writers = min(8, os.cpu_count() or 8)

    # Process video with parallel I/O
    _process_video_parallel(
        vr, n_frames, batch_size, ds_size, device,
        images_full, images_ds, console,
        io_batch_size=io_batch_size,
        write_quantum=write_quantum,
        max_writers=max_writers
    )
    
    # Record performance metrics
    duration = time.perf_counter() - start_time
    update_import_duration(root, duration)
    
    # Calculate and display throughput
    throughput = n_frames / duration
    console.print(Panel(
        f"✓ Import completed\n"
        f"Time: [bold yellow]{duration:.1f}s[/bold yellow] ({duration/60:.1f} min)\n"
        f"Throughput: [bold green]{throughput:.1f} fps[/bold green]\n"
        f"Output: {zarr_path}",
        title="Import Performance",
        expand=False
    ))
    
    return root


def _setup_video_reader(
    video_path: Path,
    use_gpu: bool,
    force_cpu: bool,
    console: Console
) -> Tuple[str, decord.VideoReader]:
    """Setup video reader with appropriate backend."""
    
    if force_cpu:
        decord.bridge.set_bridge('numpy')
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu())
        console.print("[yellow]Forced CPU decoding[/yellow]")
        return 'cpu', vr
    
    if use_gpu and torch.cuda.is_available():
        try:
            decord.bridge.set_bridge('torch')
            vr = decord.VideoReader(str(video_path), ctx=decord.gpu(0))
            console.print("[green]Using GPU acceleration[/green]")
            return 'cuda:0', vr
        except Exception as e:
            console.print(f"[yellow]GPU init failed: {e}[/yellow]")
            console.print("[yellow]Falling back to CPU[/yellow]")
    
    # CPU fallback
    decord.bridge.set_bridge('numpy')
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu())
    console.print("Using CPU decoding")
    return 'cpu', vr


def _get_video_metadata(
    video_path: Path,
    vr: decord.VideoReader,
    width: int,
    height: int,
    n_frames: int
) -> Dict[str, Any]:
    """Extract comprehensive video metadata."""
    
    # Get metadata from imageio
    try:
        iio_meta = iio.immeta(str(video_path))
    except Exception:
        iio_meta = {}
    
    # Combine sources
    metadata = {
        'source_video': str(video_path.name),
        'source_path': str(video_path),
        'width': width,
        'height': height,
        'total_frames': n_frames,
        'fps': vr.get_avg_fps(),
        'duration_seconds': n_frames / vr.get_avg_fps() if vr.get_avg_fps() > 0 else 0,
    }
    
    # Add imageio metadata if available
    if iio_meta:
        metadata['codec'] = iio_meta.get('codec', 'unknown')
        metadata['pix_fmt'] = iio_meta.get('pix_fmt', 'unknown')
        # Store full imageio metadata separately
        metadata['imageio_metadata'] = iio_meta
    
    return metadata


def _process_video_parallel(
    vr, n_frames, batch_size, ds_size, device,
    images_full, images_ds, console,
    io_batch_size=None,
    write_quantum: int = 1,
    max_writers: int = 4,
) -> None:
    if io_batch_size is None:
        io_batch_size = batch_size * 4

    # queue holds shard-sized tasks
    data_queue = queue.Queue(maxsize=write_quantum * 2)

    def write_slice(start_idx, end_idx, full_data, ds_data):
        # one full shard write (end_idx - start_idx == write_quantum for all but tail)
        images_full[start_idx:end_idx] = full_data
        images_ds[start_idx:end_idx] = ds_data

    # small pool: one shard per task, disjoint shards in parallel
    executor = ThreadPoolExecutor(max_workers=max_writers)

    def writer_drain():
        # pull tasks and submit to pool
        while True:
            item = data_queue.get()
            if item is None:
                break
            s, e, fblk, dblk = item
            executor.submit(write_slice, s, e, fblk, dblk)
            data_queue.task_done()

    wt = threading.Thread(target=writer_drain, daemon=True)
    wt.start()

    try:
        if device == 'cuda:0':
            gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device='cuda:0')
            _process_video_gpu(
                vr, n_frames, io_batch_size, batch_size,
                ds_size, gray_weights, data_queue, write_quantum
            )
        else:
            gray_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
            _process_video_cpu(
                vr, n_frames, io_batch_size, batch_size,
                ds_size, gray_weights, data_queue, write_quantum
            )
    finally:
        data_queue.put(None)
        wt.join()
        executor.shutdown(wait=True)



def _process_video_gpu(
    vr,
    n_frames: int,
    io_batch_size: int,
    batch_size: int,
    ds_size: Tuple[int, int],
    gray_weights: torch.Tensor,          # shape (3,), on GPU
    data_queue: "queue.Queue",
    write_quantum: int,
):
    """
    Decode + grayscale + downsample on GPU, enqueue shard-sized slices.
    Ensures NCHW for interpolate to avoid massive allocations.
    """
    # weights as fp16 on GPU
    w = gray_weights.to(device="cuda", dtype=torch.float16).view(1, 1, 1, 3)

    with torch.no_grad():
        for i in tqdm(range(0, n_frames, io_batch_size), desc="GPU Import", unit="batch", ascii=True, ncols=100):
            io_batch_end = min(i + io_batch_size, n_frames)
            n_this_batch = io_batch_end - i

            full_parts: list[np.ndarray] = []
            ds_parts:   list[np.ndarray] = []

            j = i
            while j < io_batch_end:
                sub_end = min(j + batch_size, io_batch_end)
                idx = list(range(j, sub_end))
                if not idx:
                    break

                # Decode to GPU: (N, H, W, 3) uint8
                batch_tensor = vr.get_batch(idx)

                # -> fp16, compute gray in NHWC, then convert to NCHW
                x    = batch_tensor.to(torch.float16, non_blocking=True)          # (N,H,W,3)
                gray = (x * w).sum(dim=-1)                                        # (N,H,W)
                gray = gray.unsqueeze(1)                                           # (N,1,H,W)

                # Downsample on GPU (expects NCHW)
                ds = F.interpolate(gray, size=ds_size, mode="bilinear", align_corners=False)  # (N,1,dsH,dsW)

                # Move to CPU uint8
                full_parts.append(gray.squeeze(1).to(torch.uint8).cpu().numpy())  # (N,H,W)
                ds_parts.append(  ds.squeeze(1).to(torch.uint8).cpu().numpy())    # (N,dsH,dsW)

                del batch_tensor, x, gray, ds
                j = sub_end

            if not full_parts:
                continue

            full_combined = np.concatenate(full_parts, axis=0)
            ds_combined   = np.concatenate(ds_parts,   axis=0)
            assert full_combined.shape[0] == n_this_batch == ds_combined.shape[0]

            # shard-aligned enqueue
            s = i
            while s < io_batch_end:
                e  = min(s + write_quantum, io_batch_end)
                lo = s - i
                hi = e - i
                data_queue.put((s, e, full_combined[lo:hi], ds_combined[lo:hi]))
                s = e

            torch.cuda.empty_cache()




def _process_video_cpu(vr, n_frames, io_batch_size, batch_size, ds_size, gray_weights, data_queue, write_quantum):
    ds_h, ds_w = ds_size
    for i in tqdm(range(0, n_frames, io_batch_size), desc="CPU Import", unit="batch", ascii=True, ncols=100):
        io_batch_end = min(i + io_batch_size, n_frames)
        full_batch_data, ds_batch_data = [], []

        for j in range(i, io_batch_end, batch_size):
            sub_batch_end = min(j + batch_size, io_batch_end)
            idx = list(range(j, sub_batch_end))
            if not idx:
                continue

            batch = vr.get_batch(idx)                    # numpy (N,H,W,3)
            gray  = np.dot(batch, gray_weights).astype(np.uint8)  # (N,H,W)

            ds = np.empty((gray.shape[0], ds_size[0], ds_size[1]), dtype=np.uint8)
            for k, frame in enumerate(gray):
                ds[k] = cv2.resize(frame, (ds_w, ds_h), interpolation=cv2.INTER_LINEAR)

            full_batch_data.append(gray)
            ds_batch_data.append(ds)

        if not full_batch_data:
            continue

        full_combined = np.concatenate(full_batch_data, axis=0)
        ds_combined   = np.concatenate(ds_batch_data,   axis=0)

        for s in range(i, io_batch_end, write_quantum):
            e  = min(s + write_quantum, io_batch_end)
            lo = s - i
            hi = e - i
            data_queue.put((s, e, full_combined[lo:hi], ds_combined[lo:hi]))


# Additional utility functions

def validate_import(zarr_path: str, expected_frames: int) -> bool:
    """
    Validate that import completed successfully.
    
    Args:
        zarr_path: Path to zarr file
        expected_frames: Expected number of frames
        
    Returns:
        True if valid, False otherwise
    """
    try:
        root = zarr.open_group(zarr_path, mode='r')
        
        # Check structure
        if 'raw_video' not in root:
            return False
        
        raw_video = root['raw_video']
        
        # Check arrays exist and have correct shape
        if 'images_full' not in raw_video or 'images_ds' not in raw_video:
            return False
        
        # Check frame count
        actual_frames = raw_video['images_full'].shape[0]
        if actual_frames != expected_frames:
            print(f"Frame count mismatch: expected {expected_frames}, got {actual_frames}")
            return False
        
        # Check metadata
        required_attrs = ['import_timestamp_utc', 'duration_seconds']
        for attr in required_attrs:
            if attr not in raw_video.attrs:
                print(f"Missing attribute: {attr}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def get_import_stats(zarr_path: str) -> Dict[str, Any]:
    """
    Get statistics about an imported video.
    
    Args:
        zarr_path: Path to zarr file
        
    Returns:
        Dict with import statistics
    """
    root = zarr.open_group(zarr_path, mode='r')
    raw_video = root['raw_video']
    
    stats = {
        'total_frames': raw_video['images_full'].shape[0],
        'full_resolution': raw_video['images_full'].shape[1:],
        'ds_resolution': raw_video['images_ds'].shape[1:],
        'import_duration': raw_video.attrs.get('duration_seconds', 0),
        'fps': root.attrs.get('source_video_metadata', {}).get('fps', 0),
        'chunk_size': raw_video.attrs.get('actual_chunk_size', 0),
        'device': raw_video.attrs.get('decoding_device', 'unknown')
    }
    
    if stats['import_duration'] > 0:
        stats['throughput_fps'] = stats['total_frames'] / stats['import_duration']
    
    return stats