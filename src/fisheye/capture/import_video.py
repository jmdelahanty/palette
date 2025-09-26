"""
Video import functionality for FishEye using Zarr.
GPU-accelerated decoding with shard-aligned writes.
Optimized for maximum throughput by focusing on single array writes.
"""

import os
os.environ.setdefault("BLOSC_NTHREADS", "4")

import zarr
import numpy as np
import torch
import decord
import imageio.v3 as iio
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from concurrent.futures import ThreadPoolExecutor
from collections import deque

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
    Import video into Zarr format (full resolution only).
    GPU-accelerated decoding with shard-aligned parallel writes.
    """
    if console is None:
        console = Console()

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    console.rule("[bold]Video Import (Full Resolution)[/bold]")
    start_time = time.perf_counter()

    # ---- Setup decoder -------------------------------------------------------
    device, vr = _setup_video_reader(video_path, use_gpu, force_cpu, console)
    n_frames = len(vr)
    if n_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    first = vr[0]
    full_h, full_w = int(first.shape[0]), int(first.shape[1])

    # ---- Configuration -------------------------------------------------------
    ip = config.get("import", {})
    chunk_size = int(ip.get("chunk_size", 32))
    frames_per_shard = int(ip.get("shard_size", 64))
    batch_size = int(ip.get("batch_size", 32))
    gpu_fp16 = bool(ip.get("gpu_fp16", True))
    max_writers = int(ip.get("max_writers", min(4, (os.cpu_count() or 4))))
    max_inflight = int(ip.get("max_inflight", max_writers * 2))
    
    
    # Adjust chunk size for better shard alignment
    if frames_per_shard % chunk_size != 0:
        # Find closest divisor
        divisors = [i for i in range(1, frames_per_shard + 1) 
                   if frames_per_shard % i == 0]
        chunk_size = min(divisors, key=lambda x: abs(x - chunk_size))
        console.print(f"[yellow]Adjusted chunk_size to {chunk_size} for shard alignment[/yellow]")

    # IO batch size = exact shard multiple for perfect writes
    io_batch_size = frames_per_shard
    
    # Adjust Blosc threads based on writers
    blosc_threads = max(1, (os.cpu_count() or 4) // max_writers)
    os.environ["BLOSC_NTHREADS"] = str(min(4, blosc_threads))

    # ---- Create Zarr structure -----------------------------------------------
    vid_meta = _get_video_metadata(video_path, vr, full_w, full_h, n_frames)
    
    # Update config with import settings
    cfg2 = dict(config)
    cfg2.setdefault("import", dict(ip))
    cfg2["import"].update({
        "chunk_size": chunk_size,
        "shard_size": frames_per_shard,
        "import_stage": "full_only",
        "downsampling": "deferred"  # Will be done in separate stage
    })

    root = create_palette_zarr(str(zarr_path), vid_meta, cfg2, cli_args=cli_args)
    raw = root["raw_video"]
    arr_full = raw["images_full"]
    
    # Store import metadata
    raw.attrs.update({
        "import_config": {
            "chunk_size": chunk_size,
            "shard_size": frames_per_shard,
            "batch_size": batch_size,
            "io_batch_size": io_batch_size,
            "device": device,
            "max_writers": max_writers,
            "blosc_threads": int(os.environ["BLOSC_NTHREADS"]),
        },
        "import_stage": "full_resolution",
        "downsampled": False,  # Flag for downstream processing
    })

    # ---- Console info --------------------------------------------------------
    console.print(Panel.fit(
        f"[cyan]Video:[/cyan] {video_path.name}\n"
        f"[cyan]Frames:[/cyan] {n_frames}\n"
        f"[cyan]Resolution:[/cyan] {full_h}×{full_w}\n"
        f"[cyan]Device:[/cyan] {device}\n"
        f"[cyan]Chunk size:[/cyan] {chunk_size} frames\n"
        f"[cyan]Shard size:[/cyan] {frames_per_shard} frames\n"
        f"[cyan]Batch size:[/cyan] {batch_size} frames\n"
        f"[cyan]Writers:[/cyan] {max_writers} threads",
        title="Import Configuration"
    ))

    # ---- Setup writer queue --------------------------------------------------
    q = queue.Queue(maxsize=10) 
    executor = ThreadPoolExecutor(max_workers=max_writers)
    write_times = deque(maxlen=100)
    print_lock = threading.Lock()

    def write_shard(start_idx: int, end_idx: int, data: np.ndarray):
        """Write a complete shard with timing."""
        size_mb = data.nbytes / (1024 * 1024)
        
        t0 = time.perf_counter()
        arr_full[start_idx:end_idx] = data
        dt = time.perf_counter() - t0
        
        throughput = size_mb / max(dt, 1e-9)
        write_times.append(throughput)
        
        with print_lock:
            avg_throughput = np.mean(write_times) if write_times else throughput
            print(f"Wrote [{start_idx:6d}:{end_idx:6d}] "
                  f"{size_mb:6.1f} MB in {dt*1000:6.0f} ms "
                  f"({throughput:6.1f} MB/s, avg: {avg_throughput:6.1f} MB/s)")

    def writer():
        """Writer thread that processes queue."""
        inflight = deque()
        while True:
            item = q.get()
            if item is None:
                break
                
            start, end, data = item
            fut = executor.submit(write_shard, start, end, data)
            inflight.append(fut)
            
            # Limit inflight writes
            while len(inflight) >= max_inflight:
                inflight.popleft().result()
            
            q.task_done()
        
        # Wait for remaining
        while inflight:
            inflight.popleft().result()
        executor.shutdown(wait=True)

    # Start writer thread
    writer_thread = threading.Thread(target=writer, daemon=True)
    writer_thread.start()

    # ---- Process video -------------------------------------------------------
    try:
        if device == "cuda:0":
            _process_video_gpu(
                vr, n_frames, io_batch_size, batch_size,
                gpu_fp16, q, (full_h, full_w)
            )
        else:
            _process_video_cpu(
                vr, n_frames, io_batch_size, batch_size,
                q, (full_h, full_w)
            )
    finally:
        q.put(None)
        writer_thread.join()

    # ---- Final statistics ----------------------------------------------------
    duration = time.perf_counter() - start_time
    update_import_duration(root, duration)
    
    fps = n_frames / max(duration, 1e-9)
    gb_processed = (n_frames * full_h * full_w) / (1024**3)
    throughput_gbps = gb_processed / max(duration, 1e-9)
    
    console.print(Panel(
        f"[green]✓ Import completed successfully[/green]\n\n"
        f"[yellow]Performance:[/yellow]\n"
        f"  Time: {duration:.1f}s ({duration/60:.1f} min)\n"
        f"  FPS: {fps:.1f}\n"
        f"  Throughput: {throughput_gbps:.2f} GB/s\n\n"
        f"[yellow]Output:[/yellow]\n"
        f"  Path: {zarr_path}\n"
        f"  Array: raw_video/images_full\n"
        f"  Shape: ({n_frames}, {full_h}, {full_w})",
        title="Import Complete",
        expand=False
    ))
    
    return root


def _process_video_gpu(
    vr,
    n_frames: int,
    io_batch_size: int,
    batch_size: int,
    use_fp16: bool,
    q: queue.Queue,
    full_shape: Tuple[int, int]
):
    """
    GPU processing with double buffering to hide write latency.
    """
    full_h, full_w = full_shape
    frames_per_shard = io_batch_size
    
    # Double buffering - alternate between two pinned buffers
    pinned_buffers = [
        torch.empty((frames_per_shard, full_h, full_w), dtype=torch.uint8, pin_memory=True)
        for _ in range(2)
    ]
    buffer_idx = 0
    
    # GPU buffer (single, reused)
    gpu_shard_buffer = torch.empty(
        (frames_per_shard, full_h, full_w),
        device='cuda',
        dtype=torch.uint8
    )
    
    # Grayscale weights
    weights = torch.tensor(
        [0.2989, 0.5870, 0.1140],
        device='cuda',
        dtype=torch.float16 if use_fp16 else torch.float32
    ).view(1, 1, 1, 3)
    
    # Stream for async transfers
    stream = torch.cuda.Stream()
    
    torch.cuda.synchronize()
    
    with torch.no_grad():
        for shard_idx in tqdm(
            range(0, n_frames, frames_per_shard),
            desc="Importing",
            unit="shard",
            ascii=True,
            ncols=100
        ):
            shard_end = min(shard_idx + frames_per_shard, n_frames)
            actual_shard_size = shard_end - shard_idx
            
            # Process shard on GPU
            shard_position = 0
            for batch_start in range(shard_idx, shard_end, batch_size):
                batch_end = min(batch_start + batch_size, shard_end)
                batch_indices = list(range(batch_start, batch_end))
                batch_size_actual = len(batch_indices)
                
                # Decode and process
                frames = vr.get_batch(batch_indices)
                x = frames.half() if use_fp16 else frames.float()
                gray = (x * weights).sum(dim=-1)
                gray_uint8 = gray.to(torch.uint8)
                
                # Store in GPU buffer
                gpu_shard_buffer[shard_position:shard_position + batch_size_actual] = gray_uint8
                shard_position += batch_size_actual
                
                del frames, x, gray, gray_uint8
            
            # Get current buffer
            current_buffer = pinned_buffers[buffer_idx]
            
            # Start async copy to pinned memory
            with torch.cuda.stream(stream):
                shard_data_gpu = gpu_shard_buffer[:actual_shard_size]
                current_buffer[:actual_shard_size].copy_(shard_data_gpu, non_blocking=True)
            
            # Wait for transfer to complete
            stream.synchronize()
            
            # Make contiguous copy and queue
            shard_numpy = current_buffer[:actual_shard_size].numpy()
            shard_copy = np.ascontiguousarray(shard_numpy)
            
            # This will block if queue is full (backpressure)
            q.put((shard_idx, shard_end, shard_copy))
            
            # Switch buffers for next iteration
            buffer_idx = 1 - buffer_idx
            
            # Less aggressive cleanup - only every 20 shards
            if shard_idx % (frames_per_shard * 20) == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()


def _process_video_cpu(
    vr,
    n_frames: int,
    io_batch_size: int,
    batch_size: int,
    q: queue.Queue,
    full_shape: Tuple[int, int]
):
    """
    CPU video processing - decode and convert to grayscale.
    """
    grayscale_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    
    for i in tqdm(
        range(0, n_frames, io_batch_size),
        desc="Importing",
        unit="shard",
        ascii=True,
        ncols=100
    ):
        end_idx = min(i + io_batch_size, n_frames)
        
        # Process in smaller batches
        parts = []
        for j in range(i, end_idx, batch_size):
            batch_end = min(j + batch_size, end_idx)
            indices = list(range(j, batch_end))
            
            # Decode batch
            frames = vr.get_batch(indices)  # (N,H,W,3)
            
            # Convert to grayscale
            gray = np.dot(frames, grayscale_weights).astype(np.uint8)
            parts.append(gray)
            
            del frames
        
        # Combine and enqueue
        if parts:
            shard_data = np.concatenate(parts, axis=0)
            shard_copy = np.ascontiguousarray(shard_data)
            q.put((i, end_idx, shard_copy))
            del shard_data, parts


def _setup_video_reader(
    video_path: Path,
    use_gpu: bool,
    force_cpu: bool,
    console: Console
) -> Tuple[str, decord.VideoReader]:
    """Setup video reader with GPU if available."""
    if force_cpu:
        decord.bridge.set_bridge("numpy")
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu())
        console.print("[yellow]Using CPU (forced)[/yellow]")
        return "cpu", vr

    if use_gpu and torch.cuda.is_available():
        try:
            decord.bridge.set_bridge("torch")
            vr = decord.VideoReader(str(video_path), ctx=decord.gpu(0))
            console.print("[green]Using GPU acceleration[/green]")
            return "cuda:0", vr
        except Exception as e:
            console.print(f"[yellow]GPU failed: {e}, falling back to CPU[/yellow]")

    decord.bridge.set_bridge("numpy")
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu())
    console.print("Using CPU")
    return "cpu", vr


def _get_video_metadata(
    video_path: Path,
    vr: decord.VideoReader,
    width: int,
    height: int,
    n_frames: int
) -> Dict[str, Any]:
    """Extract video metadata."""
    try:
        iio_meta = iio.immeta(str(video_path))
    except Exception:
        iio_meta = {}

    meta = {
        "source_video": str(video_path.name),
        "source_path": str(video_path),
        "width": width,
        "height": height,
        "total_frames": n_frames,
        "fps": vr.get_avg_fps(),
        "duration_seconds": n_frames / vr.get_avg_fps() if vr.get_avg_fps() > 0 else 0,
    }
    
    if iio_meta:
        meta["codec"] = iio_meta.get("codec", "unknown")
        meta["pix_fmt"] = iio_meta.get("pix_fmt", "unknown")
        meta["imageio_metadata"] = iio_meta
        
    return meta


# ========================= VALIDATION =========================================

def validate_import(zarr_path: str, expected_frames: int) -> bool:
    """Validate that import completed successfully."""
    try:
        root = zarr.open_group(zarr_path, mode='r')
        if 'raw_video' not in root:
            return False
            
        raw = root['raw_video']
        if 'images_full' not in raw:
            return False
            
        actual_frames = raw['images_full'].shape[0]
        if actual_frames != expected_frames:
            print(f"Frame count mismatch: expected {expected_frames}, got {actual_frames}")
            return False
            
        # Check metadata
        if raw.attrs.get('import_stage') != 'full_resolution':
            print("Import stage not marked as complete")
            return False
            
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def get_import_stats(zarr_path: str) -> Dict[str, Any]:
    """Get statistics about imported video."""
    root = zarr.open_group(zarr_path, mode='r')
    raw = root['raw_video']
    
    stats = {
        'total_frames': raw['images_full'].shape[0],
        'resolution': raw['images_full'].shape[1:],
        'import_duration': raw.attrs.get('duration_seconds', 0),
        'import_config': raw.attrs.get('import_config', {}),
        'downsampled': raw.attrs.get('downsampled', False),
    }
    
    if stats['import_duration'] > 0:
        stats['throughput_fps'] = stats['total_frames'] / stats['import_duration']
        
    # Calculate data size
    dtype_size = raw['images_full'].dtype.itemsize
    stats['data_size_gb'] = (stats['total_frames'] * 
                             stats['resolution'][0] * 
                             stats['resolution'][1] * 
                             dtype_size) / (1024**3)
    
    return stats