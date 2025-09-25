"""
Video import functionality for FishEye using Zarr.
GPU-accelerated decoding with parallel I/O.
"""

import zarr
import numpy as np
import torch
import torch.nn.functional as F
import decord
import imageio.v3 as iio
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel

from ..utils.system import get_git_info, get_platform_info, get_gpu_info, get_environment_info
from ..shared.zarr.schema import create_palette_zarr, get_run_group

def import_video(
    video_path: str,
    zarr_path: str,
    config: Dict[str, Any],
    cli_args: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
    use_gpu: bool = True
) -> zarr.Group:
    """
    Import video into Zarr format with GPU acceleration and parallel I/O.
    
    Args:
        video_path: Path to input video file
        zarr_path: Path to output zarr file
        config: Pipeline configuration dict
        cli_args: Optional command line arguments to store
        console: Optional Rich console for output
        use_gpu: Whether to use GPU for decoding
        
    Returns:
        Root zarr group
    """
    if console is None:
        console = Console()
    
    console.rule("[bold]Stage 1: Importing Video[/bold]")
    start_time = time.perf_counter()
    
    # Setup video decoding
    if use_gpu and torch.cuda.is_available():
        decord.bridge.set_bridge('torch')
        vr = decord.VideoReader(video_path, ctx=decord.gpu(0))
        console.print("Using GPU context for video decoding")
        device = 'cuda:0'
    else:
        decord.bridge.set_bridge('numpy')
        vr = decord.VideoReader(video_path, ctx=decord.cpu())
        console.print("Using CPU context for video decoding")
        device = 'cpu'
    
    # Get video metadata
    n_frames = len(vr)
    full_height, full_width = vr[0].shape[0], vr[0].shape[1]
    
    # Get import parameters from config
    import_params = config['import']
    ds_size = tuple(import_params['downsample_size'])
    chunk_size = import_params['chunk_size']
    batch_size = import_params['batch_size']
    
    # Grayscale conversion weights
    if device == 'cuda:0':
        gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device)
    else:
        gray_weights = np.array([0.2989, 0.5870, 0.1140])
    
    # Create Zarr hierarchy
    video_metadata = {
        'fps': vr.get_avg_fps(),
        'width': full_width,
        'height': full_height,
        'total_frames': n_frames,
        'source_video': str(Path(video_path).name)
    }
    
    # Create root with Zarr format
    root = create_palette_zarr(
        zarr_path,
        video_metadata,
        config,
        use_sharding=True
    )
    
    # Store comprehensive environment information
    # Use the improved get_environment_info that checks disk space at zarr location
    env_info = get_environment_info(
        disk_path=str(Path(zarr_path).parent),  # Check disk space where zarr will be written
        collect_ip=False  # Don't risk network hang on HPC
    )
    
    # Store all metadata
    root.attrs.update({
        'command_line_args': cli_args or {},
        'environment': env_info,  # Complete environment info
        'source_video_metadata': iio.immeta(video_path),
    })
    
    # Store pipeline parameters
    param_group = root.create_group('pipeline_params')
    for stage, stage_params in config.items():
        param_group.attrs[stage] = stage_params
    
    # Get the raw_video group
    raw_video_group = root['raw_video']
    
    # Add import-specific metadata
    raw_video_group.attrs.update({
        'import_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'original_resolution': (full_height, full_width),
        'downsampled_resolution': ds_size,
        'decoding_device': 'GPU' if use_gpu and torch.cuda.is_available() else 'CPU',
    })
    
    # Get the arrays
    images_full = raw_video_group['images_full']
    images_ds = raw_video_group['images_ds']
    
    console.print(f"Shard size: {chunk_size} frames per shard")
    console.print(f"Full resolution: {full_height}x{full_width}")
    console.print(f"Downsampled resolution: {ds_size}")
    
    # Log environment details
    if env_info.get('platform', {}).get('lsf'):
        lsf_info = env_info['platform']['lsf']
        console.print(f"LSF Job ID: {lsf_info.get('job_id', 'N/A')}")
        console.print(f"Queue: {lsf_info.get('queue', 'N/A')}")
    
    if env_info.get('platform', {}).get('disk'):
        disk_info = env_info['platform']['disk']
        console.print(f"Disk available: {disk_info.get('available_gb', 'N/A')} GB")
    
    # Setup parallel I/O
    data_queue = queue.Queue(maxsize=4)
    
    def writer_task(q, zarr_full, zarr_ds):
        """Writer thread for parallel I/O."""
        while True:
            item = q.get()
            if item is None:
                break
            start_idx, end_idx, full_data, ds_data = item
            zarr_full[start_idx:end_idx] = full_data
            zarr_ds[start_idx:end_idx] = ds_data
            q.task_done()
    
    writer_thread = threading.Thread(
        target=writer_task,
        args=(data_queue, images_full, images_ds),
        daemon=True
    )
    writer_thread.start()
    console.print("Writer thread started...")
    
    # Process video in batches
    io_batch_size = batch_size * 4
    console.print(f"Importing {n_frames} frames")
    
    if device == 'cuda:0':
        _process_video_gpu(
            vr, n_frames, io_batch_size, batch_size,
            ds_size, gray_weights, data_queue
        )
    else:
        _process_video_cpu(
            vr, n_frames, io_batch_size, batch_size,
            ds_size, gray_weights, data_queue
        )
    
    # Signal writer thread to stop
    data_queue.put(None)
    writer_thread.join()
    console.print("Writer thread finished. All data saved to Zarr.")
    
    # Record performance metrics
    duration = time.perf_counter() - start_time
    raw_video_group.attrs['duration_seconds'] = duration
    
    console.print(Panel(
        f"Total time: [bold yellow]{duration:.1f}s[/bold yellow] "
        f"([cyan]{duration/60:.1f}[/cyan] minutes)\n"
        f"Overall throughput: [bold green]{n_frames/duration:.1f} fps[/bold green]",
        title="Import Performance Summary",
        expand=False
    ))
    
    return root

def _process_video_gpu(vr, n_frames, io_batch_size, batch_size, ds_size, gray_weights, data_queue):
    """Process video using GPU acceleration."""
    for i in tqdm(range(0, n_frames, io_batch_size), desc="GPU Video Import"):
        io_batch_end = min(i + io_batch_size, n_frames)
        full_batch_data = []
        ds_batch_data = []
        
        for j in range(i, io_batch_end, batch_size):
            sub_batch_end = min(j + batch_size, io_batch_end)
            indices = list(range(j, sub_batch_end))
            if not indices:
                continue
            
            # GPU decoding and processing
            batch_tensor = vr.get_batch(indices)
            
            # Convert to grayscale on GPU
            gray_batch_float = torch.matmul(batch_tensor.float(), gray_weights).unsqueeze(1)
            
            # Downsample on GPU
            ds_batch_float = F.interpolate(
                gray_batch_float,
                size=ds_size,
                mode='bilinear',
                align_corners=False
            )
            
            # Move to CPU and convert to numpy
            full_batch_data.append(gray_batch_float.squeeze(1).byte().cpu().numpy())
            ds_batch_data.append(ds_batch_float.squeeze(1).byte().cpu().numpy())
            
            # Clean up GPU memory
            del batch_tensor, gray_batch_float, ds_batch_float
            torch.cuda.empty_cache()  # More aggressive cleanup
        
        if not full_batch_data:
            continue
        
        # Combine sub-batches
        full_combined = np.concatenate(full_batch_data, axis=0)
        ds_combined = np.concatenate(ds_batch_data, axis=0)
        
        # Queue for writing
        data_queue.put((i, io_batch_end, full_combined, ds_combined))

def _process_video_cpu(vr, n_frames, io_batch_size, batch_size, ds_size, gray_weights, data_queue):
    """Process video using CPU (fallback)."""
    for i in tqdm(range(0, n_frames, io_batch_size), desc="CPU Video Import"):
        io_batch_end = min(i + io_batch_size, n_frames)
        full_batch_data = []
        ds_batch_data = []
        
        for j in range(i, io_batch_end, batch_size):
            sub_batch_end = min(j + batch_size, io_batch_end)
            indices = list(range(j, sub_batch_end))
            if not indices:
                continue
            
            # CPU decoding
            batch = vr.get_batch(indices)
            
            # Convert to grayscale
            gray_batch = np.dot(batch, gray_weights).astype(np.uint8)
            
            # Downsample
            ds_batch = np.array([
                cv2.resize(frame, ds_size, interpolation=cv2.INTER_LINEAR)
                for frame in gray_batch
            ])
            
            full_batch_data.append(gray_batch)
            ds_batch_data.append(ds_batch)
        
        if not full_batch_data:
            continue
        
        # Combine sub-batches
        full_combined = np.concatenate(full_batch_data, axis=0)
        ds_combined = np.concatenate(ds_batch_data, axis=0)
        
        # Queue for writing
        data_queue.put((i, io_batch_end, full_combined, ds_combined))