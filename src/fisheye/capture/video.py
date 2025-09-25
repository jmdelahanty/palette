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
    
    # Update with actual parameters used
    raw_video_group.attrs.update({
        'actual_chunk_size': optimal_chunk_size,
        'batch_size': batch_size,
        'io_batch_size': batch_size * 4,
        'decoding_device': device
    })
    
    # Console output
    console.print(f"Video: [cyan]{video_path.name}[/cyan]")
    console.print(f"Frames: [yellow]{n_frames}[/yellow]")
    console.print(f"Original: {full_height}×{full_width}")
    console.print(f"Downsampled: {ds_size[0]}×{ds_size[1]}")
    console.print(f"Chunk size: {optimal_chunk_size}")
    console.print(f"Device: [green]{device}[/green]")
    
    # Process video with parallel I/O
    _process_video_parallel(
        vr, n_frames, batch_size, ds_size, device,
        images_full, images_ds, console
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
    vr: decord.VideoReader,
    n_frames: int,
    batch_size: int,
    ds_size: Tuple[int, int],
    device: str,
    images_full: zarr.Array,
    images_ds: zarr.Array,
    console: Console
) -> None:
    """Process video with parallel I/O pipeline."""
    
    # Setup processing pipeline
    io_batch_size = batch_size * 4
    data_queue = queue.Queue(maxsize=4)
    
    # Writer thread
    def writer_thread():
        while True:
            item = data_queue.get()
            if item is None:
                break
            start_idx, end_idx, full_data, ds_data = item
            images_full[start_idx:end_idx] = full_data
            images_ds[start_idx:end_idx] = ds_data
            data_queue.task_done()
    
    writer = threading.Thread(target=writer_thread, daemon=True)
    writer.start()
    
    # Process based on device
    try:
        if device == 'cuda:0':
            _process_gpu_batches(
                vr, n_frames, io_batch_size, batch_size,
                ds_size, data_queue, console
            )
        else:
            _process_cpu_batches(
                vr, n_frames, io_batch_size, batch_size,
                ds_size, data_queue, console
            )
    finally:
        # Ensure writer thread completes
        data_queue.put(None)
        writer.join()


def _process_gpu_batches(
    vr: decord.VideoReader,
    n_frames: int,
    io_batch_size: int,
    batch_size: int,
    ds_size: Tuple[int, int],
    data_queue: queue.Queue,
    console: Console
) -> None:
    """Process video batches on GPU."""
    
    # Grayscale weights on GPU
    gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device='cuda:0')
    
    with tqdm(total=n_frames, desc="GPU Import", unit="frames") as pbar:
        for i in range(0, n_frames, io_batch_size):
            io_batch_end = min(i + io_batch_size, n_frames)
            full_batch_data = []
            ds_batch_data = []
            
            for j in range(i, io_batch_end, batch_size):
                sub_batch_end = min(j + batch_size, io_batch_end)
                indices = list(range(j, sub_batch_end))
                if not indices:
                    continue
                
                # Decode batch
                batch_tensor = vr.get_batch(indices)
                
                # Convert to grayscale
                gray_batch = torch.matmul(
                    batch_tensor.float(), gray_weights
                ).unsqueeze(1)
                
                # Downsample
                ds_batch = F.interpolate(
                    gray_batch,
                    size=ds_size,
                    mode='bilinear',
                    align_corners=False
                )
                
                # Transfer to CPU
                full_batch_data.append(
                    gray_batch.squeeze(1).byte().cpu().numpy()
                )
                ds_batch_data.append(
                    ds_batch.squeeze(1).byte().cpu().numpy()
                )
                
                # Clean GPU memory
                del batch_tensor, gray_batch, ds_batch
                if j % (batch_size * 8) == 0:  # Periodic cleanup
                    torch.cuda.empty_cache()
                
                pbar.update(len(indices))
            
            if full_batch_data:
                # Combine and queue
                full_combined = np.concatenate(full_batch_data, axis=0)
                ds_combined = np.concatenate(ds_batch_data, axis=0)
                data_queue.put((i, io_batch_end, full_combined, ds_combined))


def _process_cpu_batches(
    vr: decord.VideoReader,
    n_frames: int,
    io_batch_size: int,
    batch_size: int,
    ds_size: Tuple[int, int],
    data_queue: queue.Queue,
    console: Console
) -> None:
    """Process video batches on CPU."""
    
    # Grayscale weights for CPU
    gray_weights = np.array([0.2989, 0.5870, 0.1140])
    
    with tqdm(total=n_frames, desc="CPU Import", unit="frames") as pbar:
        for i in range(0, n_frames, io_batch_size):
            io_batch_end = min(i + io_batch_size, n_frames)
            full_batch_data = []
            ds_batch_data = []
            
            for j in range(i, io_batch_end, batch_size):
                sub_batch_end = min(j + batch_size, io_batch_end)
                indices = list(range(j, sub_batch_end))
                if not indices:
                    continue
                
                # Decode batch
                batch = vr.get_batch(indices)
                
                # Convert to grayscale
                gray_batch = np.dot(batch, gray_weights).astype(np.uint8)
                
                # Downsample each frame
                ds_batch = np.zeros(
                    (len(indices), ds_size[0], ds_size[1]),
                    dtype=np.uint8
                )
                for idx, frame in enumerate(gray_batch):
                    ds_batch[idx] = cv2.resize(
                        frame,
                        (ds_size[1], ds_size[0]),  # cv2 uses (width, height)
                        interpolation=cv2.INTER_LINEAR
                    )
                
                full_batch_data.append(gray_batch)
                ds_batch_data.append(ds_batch)
                pbar.update(len(indices))
            
            if full_batch_data:
                # Combine and queue
                full_combined = np.concatenate(full_batch_data, axis=0)
                ds_combined = np.concatenate(ds_batch_data, axis=0)
                data_queue.put((i, io_batch_end, full_combined, ds_combined))


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