"""
Video import functionality for FishEye using Zarr.
Supports CPU or GPU-accelerated decoding, with optional GPUDirect Storage writes.
"""

import os
os.environ.setdefault("BLOSC_NTHREADS", "4")
# Force kvikIO to use GDS mode, not compatibility mode
os.environ["KVIKIO_COMPAT_MODE"] = "OFF"

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

# Optional deps
try:
    import cupy as cp
    _HAVE_CUPY = True
except Exception:
    _HAVE_CUPY = False

try:
    import kvikio
    import kvikio.zarr
    _HAVE_KVIKIO = True
except Exception:
    _HAVE_KVIKIO = False

from ..utils.system import get_git_info, get_platform_info, get_gpu_info, get_environment_info
from ..shared.zarr.schema import create_palette_zarr, update_import_duration


# ---------------- GDS support helpers ---------------- #

def _probe_gds(console: Console) -> Tuple[bool, Optional[str]]:
    """Check if GDS is available"""
    if not _HAVE_KVIKIO:
        return False, "kvikio not importable"
    if not _HAVE_CUPY:
        return False, "cupy not importable"
    
    # Check if GDS is actually available
    try:
        import kvikio.defaults as defaults

        if not defaults.compat_mode():
            return True, None
        else:
            return False, "kvikio in compatibility mode (no GDS)"
    except Exception as e:
        return False, str(e)

def _process_video_gpu_kvikio(
    vr,
    n_frames: int,
    io_batch_size: int,
    batch_size: int,
    use_fp16: bool,
    full_shape: Tuple[int, int],
    zarr_array,
    console: Console
):
    import cupy as cp
    
    full_h, full_w = full_shape
    
    # Clear cache before allocating
    torch.cuda.empty_cache()
    if _HAVE_CUPY:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    
    # Determine write size based on whether we're using sharding
    if hasattr(zarr_array, 'shards') and zarr_array.shards:
        frames_per_write = zarr_array.shards[0]
        required_gb = frames_per_write * full_h * full_w / (1024**3)
        console.print(f"[yellow]Shard buffer requires {required_gb:.2f} GB[/yellow]")
        
        # Check if we have enough memory
        available = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
        if required_gb > available * 0.9:  # Leave 10% buffer
            console.print(f"[red]Not enough GPU memory! Required: {required_gb:.2f} GB, Available: {available:.2f} GB[/red]")
            console.print("[yellow]Falling back to chunk-based writes[/yellow]")
            frames_per_write = io_batch_size
    else:
        frames_per_write = io_batch_size
    
    # Allocate buffer for the write size
    gpu_shard_buffer = torch.empty((frames_per_write, full_h, full_w), device='cuda', dtype=torch.uint8)
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device='cuda',
                           dtype=torch.float16 if use_fp16 else torch.float32).view(1, 1, 1, 3)

    write_times = []
    torch.cuda.synchronize()
    
    with torch.no_grad():
        # Loop through the video in write-sized increments
        pbar = tqdm(range(0, n_frames, frames_per_write),
                    desc="GPU→Zarr GDS", unit="write", ascii=True, ncols=100)
        
        for write_idx in pbar:
            write_end = min(write_idx + frames_per_write, n_frames)
            actual_write_size = write_end - write_idx
            buffer_position = 0
            
            # Fill GPU buffer with grayscale frames
            for batch_start in range(write_idx, write_end, batch_size):
                batch_end = min(batch_start + batch_size, write_end)
                frames = vr.get_batch(list(range(batch_start, batch_end)))
                x = frames.half() if use_fp16 else frames.float()
                gray = (x * weights).sum(dim=-1)
                gray_uint8 = gray.to(torch.uint8)
                gpu_shard_buffer[buffer_position:buffer_position+len(frames)] = gray_uint8
                buffer_position += len(frames)
                del frames, x, gray, gray_uint8

            # Convert to CuPy and write
            t0 = time.perf_counter()
            
            # Convert PyTorch to CuPy (zero-copy)
            cupy_shard = cp.from_dlpack(
                gpu_shard_buffer[:actual_write_size].contiguous()
            )
            
            # Direct CuPy write (should trigger GDS)
            zarr_array[write_idx:write_end] = cupy_shard
            cp.cuda.Stream.null.synchronize()
            
            dt = time.perf_counter() - t0
            write_times.append(dt)
            
            # Update progress
            size_mb = actual_write_size * full_h * full_w / (1024 * 1024)
            speed_mbps = size_mb / max(dt, 1e-9)
            avg_speed = sum(size_mb / max(t, 1e-9) for t in write_times) / len(write_times)
            
            pbar.set_postfix({
                'MB': f'{size_mb:.1f}',
                'MB/s': f'{speed_mbps:.1f}',
                'Avg': f'{avg_speed:.1f}'
            })

            if write_idx % (frames_per_write * 20) == 0:
                torch.cuda.empty_cache()
                import gc; gc.collect()
    
    return write_times

class _GDSShardWriter:
    def __init__(self, out_dir: Path, require_gds: bool = False, console: Optional[Console]=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.require_gds = require_gds
        self.console = console or Console()
        if require_gds:
            os.environ["KVIKIO_COMPAT_MODE"] = "OFF"

    def write_gpu_shard(self, start_idx: int, end_idx: int, cupy_array: "cp.ndarray"):
        from time import perf_counter
        shard_path = self.out_dir / f"shard_{start_idx:08d}_{end_idx:08d}.bin"
        t0 = perf_counter()
        with kvikio.CuFile(str(shard_path), "w") as f:
            nbytes = f.write(cupy_array)
        dt = perf_counter() - t0
        mb = int(cupy_array.nbytes) / (1024*1024)
        self.console.print(
            f"[green]GDS[/green] wrote {shard_path.name} "
            f"{mb:.1f} MB in {dt*1e3:.0f} ms ({mb/max(dt,1e-9):.1f} MB/s)"
        )
        return shard_path, nbytes


# ---------------- Main import ---------------- #

def import_video(
    video_path: str,
    zarr_path: str,
    config: Dict[str, Any],
    cli_args: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
    use_gpu: bool = True,
    force_cpu: bool = False
) -> zarr.Group:
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
    chunk_size = int(ip.get("chunk_size", 64))
    batch_size = int(ip.get("batch_size", 32))
    gpu_fp16 = bool(ip.get("gpu_fp16", True))
    max_writers = int(ip.get("max_writers", min(4, (os.cpu_count() or 4))))
    max_inflight = int(ip.get("max_inflight", max_writers * 2))
    
    # kvikIO and sharding configuration
    use_kvikio_zarr = bool(ip.get("use_kvikio_zarr", False))
    use_sharding = bool(ip.get("use_sharding", False))
    
    # Determine io_batch_size based on sharding
    if use_sharding:
        chunks_per_shard = int(ip.get("chunks_per_shard", 10))
        io_batch_size = chunk_size * chunks_per_shard  # e.g., 640 frames
    else:
        io_batch_size = chunk_size  # e.g., 64 frames

    # GDS config (for binary shards)
    gds_enable = bool(ip.get("import_gds_enable", False))
    gds_dir = ip.get("gds_dir", None)
    gds_require = bool(ip.get("gds_require", False))
    
    # Check if we should use kvikIO Zarr
    kvikio_available = False
    if use_kvikio_zarr and device == "cuda:0":
        try:
            import kvikio.zarr
            import cupy as cp
            kvikio_available = True
            console.print("[green]✓ kvikIO available for direct GPU→Zarr writes[/green]")
            gds_enable = False  # Disable binary GDS if using kvikIO Zarr
        except ImportError as e:
            console.print(f"[yellow]kvikIO not available: {e}, using standard path[/yellow]")
            use_kvikio_zarr = False
    
    # Original GDS probe for binary shards (only if not using kvikIO)
    if not use_kvikio_zarr and gds_enable:
        gds_ok, gds_reason = _probe_gds(console)
        if device == "cuda:0":
            if not gds_dir:
                console.print("[yellow]GDS enabled but no gds_dir provided; disabling GDS[/yellow]")
                gds_enable = False
            elif not gds_ok:
                msg = f"GDS probe failed ({gds_reason}); "
                if gds_require:
                    raise RuntimeError(msg + "and gds_require=True")
                console.print(f"[yellow]{msg}falling back to CPU/Zarr path[/yellow]")
                gds_enable = False

    blosc_threads = max(1, (os.cpu_count() or 4) // max_writers)
    os.environ["BLOSC_NTHREADS"] = str(min(4, blosc_threads))

    # ---- Create Zarr structure -----------------------------------------------
    vid_meta = _get_video_metadata(video_path, vr, full_w, full_h, n_frames)
    
    if kvikio_available and use_kvikio_zarr:
        import kvikio.zarr
        import zarr
        import cupy as cp
        
        # Check if GDS is actually active (updated API)
        try:
            import kvikio
            # Try the newer API
            if hasattr(kvikio, 'compat_mode'):
                compat_mode = kvikio.compat_mode()
            else:
                # Assume GDS is active if kvikio imported successfully
                compat_mode = False
            
            console.print(f"[yellow]kvikIO GDS active: {not compat_mode}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Could not check kvikIO mode: {e}[/yellow]")
        
        # Enable GPU support
        zarr.config.enable_gpu()

        console.print(f"[cyan]Zarr GPU support enabled[/cyan]")
        
        # Create directory structure
        zarr_path_obj = Path(zarr_path)
        zarr_path_obj.mkdir(parents=True, exist_ok=True)
        (zarr_path_obj / "raw_video").mkdir(exist_ok=True)
        images_full_path = zarr_path_obj / "raw_video" / "images_full"
        
        # Create the GDS store
        store = kvikio.zarr.GDSStore(str(images_full_path))
        
        if use_sharding:
            console.print(f"[cyan]Creating kvikIO GDSStore with sharding[/cyan]")
            console.print(f"[cyan]Chunks: {chunk_size} frames, Shards: {io_batch_size} frames ({chunks_per_shard} chunks/shard)[/cyan]")
            
            # Create sharded array
            arr_full = zarr.create_array(
                store=store,
                shape=(n_frames, full_h, full_w),
                chunks=(chunk_size, full_h, full_w),
                shards=(io_batch_size, full_h, full_w),  # Shard size
                dtype="uint8",
                fill_value=0,
                compressors=None,  # No compression for GDS
                zarr_format=3,
                overwrite=True
            )
            console.print(f"[green]✓ Created sharded array: ~{n_frames//io_batch_size} shards vs {n_frames//chunk_size} individual chunks[/green]")
        else:
            console.print(f"[cyan]Creating kvikIO GDSStore without sharding[/cyan]")
            console.print(f"[cyan]Chunks: {io_batch_size} frames[/cyan]")
            
            # Create non-sharded array
            arr_full = zarr.create_array(
                store=store,
                shape=(n_frames, full_h, full_w),
                chunks=(io_batch_size, full_h, full_w),  # Just chunks, no shards
                dtype="uint8",
                fill_value=0,
                compressors=None,
                zarr_format=3,
                overwrite=True
            )
            console.print(f"[green]✓ Created standard array: {n_frames//io_batch_size} chunks[/green]")
        
        # Create root and metadata structure
        root = zarr.open_group(str(zarr_path), mode='w')
        raw = root.create_group("raw_video", overwrite=True)
        
        # Build metadata conditionally
        metadata = {
            # Core info always present
            "chunk_size": chunk_size,
            "io_batch_size": io_batch_size,
            "batch_size": batch_size,
            "device": device,
            "import_method": "kvikio_zarr",
            "import_stage": "full_resolution",
            "downsampled": False,
            "original_resolution": (full_h, full_w),
            "fps": vid_meta.get("fps", 30),
            "total_frames": n_frames,
            "source_video": str(video_path.name),
            "source_path": str(video_path.absolute()),
            "import_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Add sharding info only if used
        if use_sharding:
            metadata["sharding_enabled"] = True
            metadata["chunks_per_shard"] = chunks_per_shard
            metadata["shards_total"] = n_frames // io_batch_size
        
        # Add GPU-specific info only if on GPU
        if device == "cuda:0" and gpu_fp16:
            metadata["gpu_fp16"] = True
        
        # kvikIO is always uncompressed
        metadata["compression"] = "none"
        
        raw.attrs.update(metadata)
    else:
        # Original Zarr creation path
        cfg2 = dict(config)
        cfg2.setdefault("import", dict(ip))
        cfg2["import"].update({
            "chunk_size": chunk_size,
            "io_batch_size": io_batch_size,
            "import_stage": "full_only",
            "downsampling": "deferred"
        })
        
        root = create_palette_zarr(str(zarr_path), vid_meta, cfg2, cli_args=cli_args)
        raw = root["raw_video"]
        arr_full = raw["images_full"]
        
        # Build metadata conditionally
        metadata = {
            # Core info always present
            "chunk_size": chunk_size,
            "io_batch_size": io_batch_size,
            "batch_size": batch_size,
            "device": device,
            "import_stage": "full_resolution",
            "downsampled": False,
            "original_resolution": (full_h, full_w),
            "fps": vid_meta.get("fps", 30),
            "total_frames": n_frames,
            "source_video": str(video_path.name),
            "source_path": str(video_path.absolute()),
            "import_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Determine method
        if gds_enable:
            metadata["import_method"] = "gds_binary"
            metadata["gds_dir"] = gds_dir
            if gds_require:
                metadata["gds_require"] = True
        else:
            metadata["import_method"] = "standard_zarr"
            # Add compression info for standard path
            compression_type = ip.get("compression", "none")
            metadata["compression"] = compression_type
            if compression_type != "none":
                metadata["compression_level"] = ip.get("compression_level", 1)
            
            # Threading info for CPU path
            metadata["max_writers"] = max_writers
            metadata["max_inflight"] = max_inflight
            metadata["blosc_threads"] = int(os.environ.get("BLOSC_NTHREADS", 4))
        
        # Add GPU-specific info only if on GPU
        if device == "cuda:0" and gpu_fp16:
            metadata["gpu_fp16"] = True
        
        raw.attrs.update(metadata)
    # ---- Console info --------------------------------------------------------
    method = "kvikIO GPU→Zarr" if (kvikio_available and use_kvikio_zarr) else \
             "GDS Binary" if gds_enable else "Standard Zarr"
    
    write_mode = "sharded" if use_sharding else "chunked"
    
    console.print(Panel.fit(
        f"[cyan]Video:[/cyan] {video_path.name}\n"
        f"[cyan]Frames:[/cyan] {n_frames}\n"
        f"[cyan]Resolution:[/cyan] {full_h}×{full_w}\n"
        f"[cyan]Device:[/cyan] {device}\n"
        f"[cyan]Chunk size:[/cyan] {chunk_size} frames\n"
        f"[cyan]Write size:[/cyan] {io_batch_size} frames ({write_mode})\n"
        f"[cyan]Batch size:[/cyan] {batch_size} frames\n"
        f"[cyan]Method:[/cyan] {method}",
        title="Import Configuration"
    ))

    # ---- Process based on method --------------------------------------------
    if kvikio_available and use_kvikio_zarr:
        console.print("[green]Using kvikIO direct GPU→Zarr writes[/green]")
        _process_video_gpu_kvikio(
            vr, n_frames, io_batch_size, batch_size,
            gpu_fp16, (full_h, full_w), arr_full, console
        )
    else:
        # Original path with queue/threads
        q = queue.Queue(maxsize=10)
        executor = ThreadPoolExecutor(max_workers=max_writers)
        write_times = deque(maxlen=100)
        print_lock = threading.Lock()
        gds_writer = _GDSShardWriter(gds_dir, require_gds=gds_require, console=console) if gds_enable else None

        def _write_zarr_slice(start_idx: int, end_idx: int, data: np.ndarray):
            size_mb = data.nbytes / (1024 * 1024)
            t0 = time.perf_counter()
            arr_full[start_idx:end_idx] = data
            dt = time.perf_counter() - t0
            thr = size_mb / max(dt, 1e-9)
            write_times.append(thr)
            with print_lock:
                avg = np.mean(write_times) if write_times else thr
                print(f"ZARR wrote [{start_idx:6d}:{end_idx:6d}] "
                      f"{size_mb:6.1f} MB in {dt*1e3:6.0f} ms "
                      f"({thr:6.1f} MB/s, avg: {avg:6.1f} MB/s)")

        def writer():
            inflight = deque()
            while True:
                item = q.get()
                if item is None: break
                if item["mode"] == "zarr":
                    fut = executor.submit(_write_zarr_slice, item["start"], item["end"], item["data"])
                elif item["mode"] == "gds":
                    fut = executor.submit(gds_writer.write_gpu_shard, item["start"], item["end"], item["cupy"])
                else:
                    fut = None
                if fut: inflight.append(fut)
                while len(inflight) >= max_inflight:
                    inflight.popleft().result()
                q.task_done()
            while inflight:
                inflight.popleft().result()
            executor.shutdown(wait=True)

        writer_thread = threading.Thread(target=writer, daemon=True)
        writer_thread.start()

        # Process video
        try:
            if device == "cuda:0":
                _process_video_gpu(
                    vr, n_frames, io_batch_size, batch_size,
                    gpu_fp16, q, (full_h, full_w), use_gds=gds_enable
                )
            else:
                _process_video_cpu(vr, n_frames, io_batch_size, batch_size, q, (full_h, full_w))
        finally:
            q.put(None)
            writer_thread.join()

    # ---- Final statistics and metadata ------------------------------------------
    duration = time.perf_counter() - start_time
    update_import_duration(root, duration)
    fps = n_frames / max(duration, 1e-9)
    gb_processed = (n_frames * full_h * full_w) / (1024**3)
    throughput_gbps = gb_processed / max(duration, 1e-9)

    # Store performance and video metadata
    performance_metadata = {
        "import_duration_seconds": duration,
        "throughput_gbps": throughput_gbps,
        "frames_per_second": fps,
        "data_size_gb": gb_processed,
    }
    raw.attrs.update(performance_metadata)

    # Store video metadata (from vid_meta)
    video_metadata = {
        "video_width": full_w,
        "video_height": full_h,
        "video_codec": vid_meta.get("codec", "unknown"),
        "video_pix_fmt": vid_meta.get("pix_fmt", "unknown"),
        "video_duration_seconds": vid_meta.get("duration_seconds", 0),
    }
    raw.attrs.update(video_metadata)

    # Optionally add system info
    include_system_info = bool(ip.get("include_system_info", False))
    if include_system_info:
        try:
            env_info = get_environment_info(
                include_all_packages=False,
                disk_path=str(zarr_path),
                collect_ip=False
            )
            
            system_metadata = {
                "system_hostname": env_info['platform']['hostname'],
                "system_os": env_info['platform']['os'],
                "system_python_version": env_info['platform']['python_version'],
                "git_hash": env_info.get('git', {}).get('hash', 'unknown'),
                "git_branch": env_info.get('git', {}).get('branch', 'unknown'),
            }
            
            # Add GPU info if used
            if device == "cuda:0" and env_info.get('gpu', {}).get('devices'):
                gpu_device = env_info['gpu']['devices'][0]
                system_metadata["gpu_name"] = gpu_device.get('name', 'unknown')
                system_metadata["gpu_memory_gb"] = gpu_device.get('memory_total', 0) / 1024**3
            
            raw.attrs.update(system_metadata)
        except Exception as e:
            console.print(f"[yellow]Could not collect system info: {e}[/yellow]")

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
    full_shape: Tuple[int, int],
    use_gds: bool = False,
    zarr_array = None  # Pass the zarr array directly
):
    full_h, full_w = full_shape
    frames_per_shard = io_batch_size
    gpu_shard_buffer = torch.empty((frames_per_shard, full_h, full_w), device='cuda', dtype=torch.uint8)
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device='cuda',
                           dtype=torch.float16 if use_fp16 else torch.float32).view(1, 1, 1, 3)

    torch.cuda.synchronize()
    with torch.no_grad():
        for shard_idx in tqdm(range(0, n_frames, frames_per_shard),
                              desc="Importing", unit="shard", ascii=True, ncols=100):
            shard_end = min(shard_idx + frames_per_shard, n_frames)
            actual_shard_size = shard_end - shard_idx
            shard_position = 0
            
            # Process frames into GPU buffer
            for batch_start in range(shard_idx, shard_end, batch_size):
                batch_end = min(batch_start + batch_size, shard_end)
                frames = vr.get_batch(list(range(batch_start, batch_end)))
                x = frames.half() if use_fp16 else frames.float()
                gray = (x * weights).sum(dim=-1)
                gray_uint8 = gray.to(torch.uint8)
                gpu_shard_buffer[shard_position:shard_position+len(frames)] = gray_uint8
                shard_position += len(frames)
                del frames, x, gray, gray_uint8

            if use_gds and zarr_array is not None:
                # Direct GPU → Zarr write using kvikIO
                import cupy as cp
                cupy_view = cp.from_dlpack(
                    gpu_shard_buffer[:actual_shard_size].contiguous()
                )
                
                # Write directly to Zarr array (no queue needed!)
                zarr_array[shard_idx:shard_end] = cupy_view
                
                # Optional: print performance
                size_mb = cupy_view.nbytes / (1024 * 1024)
                console.print(f"[green]GDS→Zarr[/green] wrote [{shard_idx}:{shard_end}] {size_mb:.1f} MB")
                
            elif use_gds:
                # Original binary shard approach (fallback)
                cupy_view = cp.fromDlpack(torch.utils.dlpack.to_dlpack(
                    gpu_shard_buffer[:actual_shard_size].contiguous()
                ))
                q.put({"mode": "gds", "start": shard_idx, "end": shard_end, "cupy": cupy_view})
            else:
                # CPU path
                shard_cpu = gpu_shard_buffer[:actual_shard_size].cpu().numpy()
                shard_copy = np.ascontiguousarray(shard_cpu)
                q.put({"mode": "zarr", "start": shard_idx, "end": shard_end, "data": shard_copy})

            if shard_idx % (frames_per_shard * 20) == 0:
                torch.cuda.empty_cache()
                import gc; gc.collect()


def _process_video_cpu(vr, n_frames, io_batch_size, batch_size, q, full_shape: Tuple[int, int]):
    grayscale_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    for i in tqdm(range(0, n_frames, io_batch_size),
                  desc="Importing", unit="shard", ascii=True, ncols=100):
        end_idx = min(i + io_batch_size, n_frames)
        parts = []
        for j in range(i, end_idx, batch_size):
            frames = vr.get_batch(list(range(j, min(j+batch_size, end_idx))))
            gray = np.dot(frames, grayscale_weights).astype(np.uint8)
            parts.append(gray)
            del frames
        if parts:
            shard_data = np.concatenate(parts, axis=0)
            shard_copy = np.ascontiguousarray(shard_data)
            q.put({"mode": "zarr", "start": i, "end": end_idx, "data": shard_copy})


def _setup_video_reader(video_path: Path, use_gpu: bool, force_cpu: bool, console: Console) -> Tuple[str, decord.VideoReader]:
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


def _get_video_metadata(video_path: Path, vr: decord.VideoReader, width: int, height: int, n_frames: int) -> Dict[str, Any]:
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


# ---------------- Validation ---------------- #

def validate_import(zarr_path: str, expected_frames: int) -> bool:
    try:
        root = zarr.open_group(zarr_path, mode='r')
        raw = root.get('raw_video')
        if raw is None or 'images_full' not in raw:
            return False
        actual_frames = raw['images_full'].shape[0]
        if actual_frames != expected_frames:
            print(f"Frame count mismatch: expected {expected_frames}, got {actual_frames}")
            return False
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
    if 'raw_video' not in root or 'images_full' not in root['raw_video']:
        raise ValueError("Zarr does not contain raw_video/images_full")

    raw = root['raw_video']
    arr = raw['images_full']

    stats: Dict[str, Any] = {
        'total_frames': int(arr.shape[0]),
        'resolution': tuple(arr.shape[1:]),
        'dtype': str(arr.dtype),
        'chunks': tuple(arr.chunks) if hasattr(arr, "chunks") else None,
        'import_duration': float(raw.attrs.get('duration_seconds', 0.0)),
        'import_config': dict(raw.attrs.get('import_config', {})),
        'downsampled': bool(raw.attrs.get('downsampled', False)),
    }

    if stats['import_duration'] > 0:
        stats['throughput_fps'] = stats['total_frames'] / stats['import_duration']

    # Approx data size in GiB (uncompressed)
    dtype_size = arr.dtype.itemsize
    stats['data_size_gib'] = (
        stats['total_frames'] * stats['resolution'][0] * stats['resolution'][1] * dtype_size
    ) / (1024 ** 3)

    return stats

