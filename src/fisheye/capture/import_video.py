"""
Video import functionality for FishEye using Zarr.
Supports CPU or GPU-accelerated decoding, with optional GPUDirect Storage writes.
Uses process isolation to avoid segmentation faults during cleanup.
"""

import os
import argparse
import shutil
import subprocess
from os import fork, waitpid, WIFEXITED, WEXITSTATUS, _exit
os.environ.setdefault("BLOSC_NTHREADS", "4")
# Force kvikIO to use GDS mode, not compatibility mode
os.environ["KVIKIO_COMPAT_MODE"] = "OFF"

import json
import copy
import zarr
import torch
import decord
import imageio.v3 as iio
import time
import cupy as cp
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import yaml
from zarr.storage import LocalStore
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn

from ..utils.encoder_tags import parse_encoder_comment

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

from ..utils.system import get_git_info, get_environment_info, get_gds_config
from ..shared.zarr.schema import create_palette_zarr, update_import_duration


# ---------------- GDS support helpers ---------------- #

def _compute_frame_indices(
    total_frames: int,
    frame_step: Optional[int],
    *,
    skip_tail_frames: int = 0,
) -> list[int]:
    """
    Compute which frame indices to import based on sampling strategy.

    Args:
        total_frames: Total number of frames in the video
        frame_step: If provided, sample every Nth frame. If None, import all frames.

    Returns:
        List of frame indices to import
    """
    if skip_tail_frames < 0:
        skip_tail_frames = 0

    effective_total = max(total_frames - skip_tail_frames, 0)

    if frame_step is None or frame_step == 1:
        # Import all frames
        return list(range(effective_total))
    else:
        # Uniform sampling: [0, step, 2*step, ...]
        return list(range(0, effective_total, frame_step))


def _default_import_config() -> Dict[str, Any]:
    """Baseline configuration used when no YAML file is supplied."""
    return {
        "import": {
            "resolutions": "both",
            "chunk_size": 16,
            "batch_size": 32,
            "gpu_fp16": True,
            "use_kvikio_zarr": True,
            "use_sharding": True,
            "chunks_per_shard": 2,
            "include_system_info": True,
            "full": {},
            "downsampled": {
                "size": [640, 640],
                "method": "area",
                "chunk_size": 16,
                "preserve_aspect": False,
            },
        }
    }


def _merge_import_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user overrides into the import section while preserving defaults."""
    merged = dict(base)
    for key, value in override.items():
        if key in {"downsampled", "full"} and isinstance(value, dict):
            nested = dict(base.get(key, {}))
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _normalize_downsample_formats(raw_formats: Optional[Any]) -> List[str]:
    """
    Normalize the requested downsampled formats.

    Always returns at least ['gray'] so downstream stages (detection, background)
    can rely on a luminance channel being present.
    """
    if raw_formats is None:
        return ["gray"]

    if isinstance(raw_formats, str):
        candidates = [raw_formats]
    else:
        candidates = list(raw_formats)

    normalized: List[str] = []
    for item in candidates:
        val = str(item).strip().lower()
        if val in {"gray", "grey", "grayscale"}:
            val = "gray"
        elif val in {"rgb", "color", "colour", "colors"}:
            val = "rgb"
        else:
            continue
        if val not in normalized:
            normalized.append(val)

    if "gray" not in normalized:
        normalized.insert(0, "gray")
    return normalized


def _load_import_config(config_path: Optional[Path], console: Console) -> Dict[str, Any]:
    """Load configuration from YAML, falling back to defaults."""
    base = _default_import_config()
    config = copy.deepcopy(base)

    if config_path is None:
        return config

    path = Path(config_path).expanduser()
    if not path.exists():
        console.print(f"[yellow]Config file not found: {path}. Using defaults.[/yellow]")
        return config

    with path.open("r") as fh:
        try:
            data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse config file '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Config file '{path}' must contain a mapping at the top level.")

    for key, value in data.items():
        if key == "import" and isinstance(value, dict):
            config["import"] = _merge_import_config(base["import"], value)
        else:
            config[key] = value
    return config


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
    arrays: Dict[str, Any],
    config: Dict[str, Any],
    console: Console,
    frame_indices: Optional[list[int]] = None
):
    import torch.nn.functional as F

    def _compute_letterbox_dims(
        source_h: int,
        source_w: int,
        target_h: int,
        target_w: int,
    ) -> Tuple[int, int, int, int, int, int]:
        scale = min(target_h / source_h, target_w / source_w)
        resized_h = max(1, min(target_h, int(round(source_h * scale))))
        resized_w = max(1, min(target_w, int(round(source_w * scale))))
        pad_top = (target_h - resized_h) // 2
        pad_bottom = target_h - resized_h - pad_top
        pad_left = (target_w - resized_w) // 2
        pad_right = target_w - resized_w - pad_left
        return resized_h, resized_w, pad_top, pad_bottom, pad_left, pad_right

    def _resize_batch(
        batch_bchw: torch.Tensor,
        target_h: int,
        target_w: int,
        method: str,
        method_align_corners: Optional[bool],
        preserve_aspect: bool,
    ) -> torch.Tensor:
        if not preserve_aspect:
            return F.interpolate(
                batch_bchw,
                size=(target_h, target_w),
                mode=method,
                align_corners=method_align_corners,
            )

        source_h = int(batch_bchw.shape[-2])
        source_w = int(batch_bchw.shape[-1])
        (
            resized_h,
            resized_w,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
        ) = _compute_letterbox_dims(source_h, source_w, target_h, target_w)

        resized = F.interpolate(
            batch_bchw,
            size=(resized_h, resized_w),
            mode=method,
            align_corners=method_align_corners,
        )

        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return resized

        return F.pad(
            resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

    # If frame_indices not provided, import all frames sequentially
    if frame_indices is None:
        frame_indices = list(range(n_frames))

    # Number of frames we're actually importing
    n_import_frames = len(frame_indices)

    # Clear cache before allocating
    torch.cuda.empty_cache()
    if _HAVE_CUPY:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

    # Get shapes and configs for each resolution
    shapes: Dict[str, Tuple[int, ...]] = {}
    buffers: Dict[str, torch.Tensor] = {}
    frames_per_write = io_batch_size  # Default

    for key, arr in arrays.items():
        spatial_shape = tuple(int(dim) for dim in arr.shape[1:])
        shapes[key] = spatial_shape

        # Check sharding for write size
        if hasattr(arr, 'shards') and arr.shards:
            frames_per_write = max(frames_per_write, arr.shards[0])
    
    # Allocate buffers for each resolution
    if 'full' in arrays:
        full_h, full_w = shapes['full'][:2]
        buffers['full'] = torch.empty((frames_per_write, full_h, full_w), 
                                      device='cuda', dtype=torch.uint8)
        console.print(f"[cyan]Full resolution buffer: {frames_per_write}×{full_h}×{full_w}[/cyan]")
    
    if 'downsampled_gray' in arrays:
        down_h, down_w = shapes['downsampled_gray'][:2]
        buffers['downsampled_gray'] = torch.empty((frames_per_write, down_h, down_w),
                                                  device='cuda', dtype=torch.uint8)
        console.print(f"[cyan]Downsampled (gray) buffer: {frames_per_write}×{down_h}×{down_w}[/cyan]")
    
    if 'downsampled_rgb' in arrays:
        down_h, down_w, _ = shapes['downsampled_rgb']
        buffers['downsampled_rgb'] = torch.empty((frames_per_write, down_h, down_w, 3),
                                                 device='cuda', dtype=torch.uint8)
        console.print(f"[cyan]Downsampled (RGB) buffer: {frames_per_write}×{down_h}×{down_w}×3[/cyan]")
    
    # Grayscale conversion weights
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device='cuda',
                          dtype=torch.float16 if use_fp16 else torch.float32).view(1, 1, 1, 3)
    
    # Get downsample method if needed
    down_method = config['import'].get('downsampled', {}).get('method', 'area')
    down_preserve_aspect = bool(config['import'].get('downsampled', {}).get('preserve_aspect', False))
    align_corners = False if down_method in ['bilinear', 'bicubic'] else None
    
    write_times = []
    torch.cuda.synchronize()

    # Calculate total writes based on actual frames to import
    total_writes = (n_import_frames + frames_per_write - 1) // frames_per_write

    with torch.no_grad():
        with Progress(
            TextColumn("[bold blue]GPU → Zarr GDS"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[mb]:.1f} MB"),
            TextColumn("•"),
            TextColumn("[green]{task.fields[speed]:.1f} MB/s"),
            TextColumn("•"),
            TextColumn("[yellow]Avg: {task.fields[avg_speed]:.1f} MB/s"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=10
        ) as progress:

            task = progress.add_task(
                f"Processing ({', '.join(arrays.keys())})",
                total=total_writes,
                mb=0.0,
                speed=0.0,
                avg_speed=0.0
            )

            # Iterate over frame_indices in chunks
            for write_idx_start in range(0, n_import_frames, frames_per_write):
                write_idx_end = min(write_idx_start + frames_per_write, n_import_frames)
                actual_write_size = write_idx_end - write_idx_start
                buffer_position = 0

                # Fill buffers with frames (single decode pass)
                for batch_idx_start in range(write_idx_start, write_idx_end, batch_size):
                    batch_idx_end = min(batch_idx_start + batch_size, write_idx_end)
                    batch_size_actual = batch_idx_end - batch_idx_start

                    # Get the actual video frame indices for this batch
                    batch_frame_indices = frame_indices[batch_idx_start:batch_idx_end]

                    # Decode frames once
                    frames = vr.get_batch(batch_frame_indices)
                    frames_f32 = frames.float()
                    work_tensor = frames_f32.half() if use_fp16 else frames_f32
                    gray = (work_tensor * weights).sum(dim=-1)
                    gray_uint8 = gray.clamp(0, 255).to(torch.uint8)

                    # Store full resolution if needed
                    if 'full' in buffers:
                        buffers['full'][buffer_position:buffer_position+batch_size_actual] = gray_uint8

                    # Generate and store downsampled if needed
                    if 'downsampled_gray' in buffers:
                        down_h, down_w = shapes['downsampled_gray'][:2]
                        # Add batch and channel dims for interpolation
                        gray_for_interp = gray_uint8.unsqueeze(1).float()  # [B, 1, H, W]

                        gray_down = _resize_batch(
                            gray_for_interp,
                            target_h=down_h,
                            target_w=down_w,
                            method=down_method,
                            method_align_corners=align_corners,
                            preserve_aspect=down_preserve_aspect,
                        ).squeeze(1).to(torch.uint8)  # Remove channel dim

                        buffers['downsampled_gray'][buffer_position:buffer_position+batch_size_actual] = gray_down
                        del gray_down, gray_for_interp

                    if 'downsampled_rgb' in buffers:
                        down_h, down_w, _ = shapes['downsampled_rgb']
                        rgb_for_interp = frames_f32.permute(0, 3, 1, 2)  # [B, 3, H, W]
                        rgb_down = _resize_batch(
                            rgb_for_interp,
                            target_h=down_h,
                            target_w=down_w,
                            method=down_method,
                            method_align_corners=align_corners,
                            preserve_aspect=down_preserve_aspect,
                        ).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
                        buffers['downsampled_rgb'][buffer_position:buffer_position+batch_size_actual] = rgb_down
                        del rgb_down, rgb_for_interp

                    buffer_position += batch_size_actual
                    del frames, frames_f32, work_tensor, gray, gray_uint8
                
                # Write all buffers to their arrays
                t0 = time.perf_counter()

                total_bytes_written = 0
                for key, buffer in buffers.items():
                    # Convert PyTorch to CuPy (zero-copy)
                    cupy_data = cp.from_dlpack(
                        buffer[:actual_write_size].contiguous()
                    )

                    # Direct CuPy write (triggers GDS)
                    # Write to sequential indices in zarr (0, 1, 2, ...)
                    # even though we may have read sparse frames from video
                    arrays[key][write_idx_start:write_idx_end] = cupy_data

                    # Track data size
                    elem_per_frame = int(np.prod(shapes[key]))
                    total_bytes_written += actual_write_size * elem_per_frame
                
                cp.cuda.Stream.null.synchronize()
                dt = time.perf_counter() - t0
                write_times.append(dt)
                
                # Calculate metrics
                size_mb = total_bytes_written / (1024 * 1024)
                speed_mbps = size_mb / max(dt, 1e-9)
                avg_speed = sum(total_bytes_written / (1024 * 1024) / max(t, 1e-9) 
                              for t in write_times) / len(write_times)
                
                # Update progress
                progress.update(
                    task,
                    advance=1,
                    mb=size_mb,
                    speed=speed_mbps,
                    avg_speed=avg_speed
                )
                
                # Periodically free CUDA/CPU caches every 20 chunk writes
                if write_idx_start > 0 and write_idx_start % (frames_per_write * 20) == 0:
                    torch.cuda.empty_cache()
                    import gc; gc.collect()
    
    return write_times


def _finalize_kvikio_zarr_metadata(
    zarr_path: Path,
    *,
    n_frames: int,
    full_shape: Optional[Tuple[int, int]],
    downsample_specs: Dict[str, Tuple[int, ...]],
    create_full: bool,
    chunk_size: int,
    io_batch_size: int,
    down_chunk_size: int,
    down_io_batch_size: int,
    use_sharding: bool,
) -> None:
    """
    Ensure that kvikIO-written archives include the Zarr v3 manifest files.

    kvikIO's GDSStore skips emitting metadata, so we patch it in afterwards
    using the LocalStore view of the archive.
    """
    store = LocalStore(str(zarr_path))
    root = zarr.open_group(store=store, mode='r+')
    raw = root.require_group("raw_video")

    if create_full and full_shape:
        full_h, full_w = full_shape
        array_kwargs = {
            "shape": (n_frames, full_h, full_w),
            "dtype": "uint8",
            "fill_value": 0,
            "compressors": [],
        }
        if use_sharding:
            array_kwargs.update({
                "chunks": (chunk_size, full_h, full_w),
                "shards": (io_batch_size, full_h, full_w),
            })
        else:
            array_kwargs["chunks"] = (io_batch_size, full_h, full_w)
        raw.require_array("images_full", **array_kwargs)

    for name, spatial_shape in downsample_specs.items():
        array_kwargs = {
            "shape": (n_frames, *spatial_shape),
            "dtype": "uint8",
            "fill_value": 0,
            "compressors": [],
        }
        chunk_shape = (down_chunk_size, *spatial_shape)
        if use_sharding:
            array_kwargs.update({
                "chunks": chunk_shape,
                "shards": (down_io_batch_size, *spatial_shape),
            })
        else:
            array_kwargs["chunks"] = chunk_shape
        raw.require_array(name, **array_kwargs)

    raw.require_array(
        "timestamps",
        shape=(n_frames,),
        chunks=(min(1000, n_frames),),
        dtype="float64",
        fill_value=float("nan"),
        compressors=[],
    )

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
    stream_codec = None
    stream_pix_fmt = None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format_tags=title,comment,encoder:stream=codec_name,codec_tag_string,pix_fmt",
                "-of",
                "json",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout:
            payload = json.loads(result.stdout)
            tags = payload.get("format", {}).get("tags", {})
            if isinstance(tags, dict) and tags:
                meta["format_tags"] = tags
                encoder_fields = parse_encoder_comment(tags.get("comment"))
                if encoder_fields:
                    meta["encoder_fields"] = encoder_fields
            streams = payload.get("streams", [])
            if isinstance(streams, list) and streams:
                stream = streams[0]
                if isinstance(stream, dict):
                    stream_codec = stream.get("codec_name") or stream.get("codec_tag_string")
                    stream_pix_fmt = stream.get("pix_fmt")
    except Exception:
        pass
    if meta.get("codec") in (None, "", "unknown") and stream_codec:
        meta["codec"] = stream_codec
    if meta.get("pix_fmt") in (None, "", "unknown") and stream_pix_fmt:
        meta["pix_fmt"] = stream_pix_fmt
    return meta


# ---------------- Main import with process isolation ---------------- #

def import_video(
    video_path: str,
    zarr_path: str,
    config: Dict[str, Any],
    cli_args: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
    use_gpu: bool = True,
    force_cpu: bool = False
) -> Path:
    """
    Import video to Zarr format with process isolation to avoid segmentation faults.
    
    This function forks a child process to perform the actual import work, then
    uses os._exit(0) to terminate the child without running cleanup handlers,
    avoiding conflicts between CUDA/GPU libraries during shutdown.
    """
    if console is None:
        console = Console()


    # Collect git info in parent process before forking
    # This avoids issues with git commands in the child process
    repo_root = Path(__file__).parent.parent.parent
    parent_git_info = get_git_info(repo_path=repo_root)

    # If git info still failed, try finding from module location
    if parent_git_info.get('commit_hash') == 'N/A':
        try:
            import inspect
            current_file = Path(inspect.getfile(import_video)).resolve()
            search_path = current_file.parent
            
            # Walk up to find .git directory
            for _ in range(10):
                if (search_path / '.git').exists():
                    parent_git_info = get_git_info(repo_path=search_path)
                    if parent_git_info.get('commit_hash') != 'N/A':
                        console.print(f"[dim]Git repository found at: {search_path}[/dim]")
                    break
                if search_path.parent == search_path:
                    break
                search_path = search_path.parent
        except Exception:
            pass  # Keep the original failed info

    # Fork the current process to create a child process
    pid = fork()

    if pid == 0:
        # === CHILD PROCESS ===
        # This is where all the actual import work happens
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # ---- Configuration -------------------------------------------------------
            ip = config.get("import", {})
            chunk_size = int(ip.get("chunk_size", 64))
            batch_size = int(ip.get("batch_size", 32))
            gpu_fp16 = bool(ip.get("gpu_fp16", True))
            
            # kvikIO and sharding configuration
            use_kvikio_zarr = bool(ip.get("use_kvikio_zarr", True))  # Default to True for GPU
            use_sharding = bool(ip.get("use_sharding", False))
            
            resolutions_mode = ip.get("resolutions", "full")
            full_config = ip.get("full", {})
            down_config = ip.get("downsampled", {})
            down_formats = _normalize_downsample_formats(down_config.get("formats"))
            store_gray = "gray" in down_formats
            store_rgb = "rgb" in down_formats
            
            create_full = resolutions_mode in ["full", "both"]
            create_down = resolutions_mode in ["downsampled", "both"]

            mode_text = {
                "full": "Full Resolution",
                "downsampled": "Downsampled Only", 
                "both": "Full + Downsampled"
            }.get(resolutions_mode, "Unknown")

            console.rule(f"[bold]Video Import ({mode_text}) | Worker Process[/bold]")

            start_time = time.perf_counter()

            # ---- Setup decoder -------------------------------------------------------
            device, vr = _setup_video_reader(video_path, use_gpu, force_cpu, console)
            n_frames = len(vr)
            if n_frames == 0:
                raise ValueError(f"Video has no frames: {video_path}")

            first = vr[0]
            full_h, full_w = int(first.shape[0]), int(first.shape[1])

            # ---- Frame sampling configuration ----------------------------------------
            # Check if this is a training data import with frame sampling
            training_data_mode = cli_args.get('training_data', False) if cli_args else False
            frame_step = cli_args.get('frame_step', None) if cli_args else None
            skip_tail_frames = int(cli_args.get('skip_tail_frames', 0) or 0) if cli_args else 0

            if skip_tail_frames >= n_frames:
                raise ValueError(
                    f"skip_tail_frames ({skip_tail_frames}) must be less than total frames ({n_frames})"
                )

            if training_data_mode and frame_step:
                # Compute frame indices for sampled import
                frame_indices_to_import = _compute_frame_indices(
                    n_frames,
                    frame_step,
                    skip_tail_frames=skip_tail_frames,
                )
                n_import_frames = len(frame_indices_to_import)
                if n_import_frames == 0:
                    raise ValueError("No frames to import after applying frame_step/skip_tail_frames")
                console.print(f"[yellow]Training data mode:[/yellow] Importing every {frame_step}th frame")
                if skip_tail_frames:
                    console.print(f"[yellow]Skipping tail:[/yellow] last {skip_tail_frames} frame(s)")
                effective_total = n_frames - skip_tail_frames
                console.print(
                    f"[yellow]Frames:[/yellow] {n_import_frames} of {effective_total} "
                    f"({100*n_import_frames/effective_total:.1f}%)"
                )
            else:
                # Standard full import
                if skip_tail_frames:
                    frame_indices_to_import = _compute_frame_indices(
                        n_frames,
                        None,
                        skip_tail_frames=skip_tail_frames,
                    )
                    n_import_frames = len(frame_indices_to_import)
                    console.print(f"[yellow]Skipping tail:[/yellow] last {skip_tail_frames} frame(s)")
                else:
                    frame_indices_to_import = None
                    n_import_frames = n_frames


            # Determine io_batch_size based on sharding
            if use_sharding:
                chunks_per_shard = int(ip.get("chunks_per_shard", 10))
                io_batch_size = chunk_size * chunks_per_shard
            else:
                io_batch_size = chunk_size

            down_target = down_config.get("size", [640, 640])
            down_h, down_w = int(down_target[0]), int(down_target[1])
            down_preserve_aspect = bool(down_config.get("preserve_aspect", False))
            down_chunk_size = int(down_config.get("chunk_size", chunk_size))
            if use_sharding:
                down_io_batch = down_chunk_size * chunks_per_shard
            else:
                down_io_batch = down_chunk_size

            downsample_specs: Dict[str, Tuple[int, ...]] = {}
            if create_down and store_gray:
                downsample_specs["images_ds"] = (down_h, down_w)
            if create_down and store_rgb:
                downsample_specs["images_ds_rgb"] = (down_h, down_w, 3)

            # Check if we should use kvikIO Zarr
            kvikio_available = False
            if use_kvikio_zarr and device == "cuda:0":
                try:
                    import kvikio.zarr
                    import cupy as cp
                    kvikio_available = True
                    console.print("[green]kvikIO available for direct GPU --> Zarr writes[/green]")
                except ImportError as e:
                    console.print(f"[yellow]kvikIO not available: {e}, using standard path[/yellow]")
                    use_kvikio_zarr = False

            # ---- Create Zarr structure -----------------------------------------------
            vid_meta = _get_video_metadata(video_path, vr, full_w, full_h, n_frames)

            if kvikio_available and use_kvikio_zarr:
                import kvikio.zarr
                import kvikio.defaults
                import cupy as cp

                kvikio_config = {
                    "num_threads": 8,
                    "task_size": 32 * 1024 * 1024,  # 32MB tasks for your ~20MB frames
                    "bounce_buffer_size": 64 * 1024 * 1024,  # 64MB bounce buffer (currently 16MB)
                    "gds_threshold": 1024,  # Use GDS for all I/O > 1KB
                }
                
                kvikio.defaults.set(kvikio_config)
                
                # Verify configuration took effect
                console.print(f"[cyan]kvikIO configured:[/cyan]")
                console.print(f"  Threads: {kvikio.defaults.get('num_threads')}")
                console.print(f"  Task size: {kvikio.defaults.get('task_size')/(1024*1024):.1f} MB")
                console.print(f"  Bounce buffer: {kvikio.defaults.get('bounce_buffer_size')/(1024*1024):.1f} MB")
                console.print(f"  GDS mode: {not kvikio.defaults.get('compat_mode')}")

                # Enable GPU support in Zarr
                zarr.config.enable_gpu()
                console.print(f"[cyan]Zarr GPU support enabled[/cyan]")

                # Create GDS store and root group
                store = kvikio.zarr.GDSStore(str(zarr_path))
                root = zarr.open_group(store, mode='w', zarr_format=3)
                raw = root.create_group("raw_video", overwrite=True)

                                # Create arrays based on resolution mode
                arrays = {}

                if create_full:
                    if use_sharding:
                        arrays['full'] = raw.create_array(
                            name='images_full',
                            shape=(n_import_frames, full_h, full_w),
                            chunks=(chunk_size, full_h, full_w),
                            shards=(io_batch_size, full_h, full_w),
                            dtype="uint8",
                            fill_value=0,
                            compressors=None,
                            overwrite=True
                        )
                    else:
                        arrays['full'] = raw.create_array(
                            name='images_full',
                            shape=(n_import_frames, full_h, full_w),
                            chunks=(io_batch_size, full_h, full_w),
                            dtype="uint8",
                            fill_value=0,
                            compressors=None,
                            overwrite=True
                        )

                if create_down and store_gray:
                    if use_sharding:
                        arrays['downsampled_gray'] = raw.create_array(
                            name='images_ds',
                            shape=(n_import_frames, down_h, down_w),
                            chunks=(down_chunk_size, down_h, down_w),
                            shards=(down_io_batch, down_h, down_w),
                            dtype="uint8",
                            fill_value=0,
                            compressors=None,
                            overwrite=True
                        )
                    else:
                        arrays['downsampled_gray'] = raw.create_array(
                            name='images_ds',
                            shape=(n_import_frames, down_h, down_w),
                            chunks=(down_chunk_size, down_h, down_w),
                            dtype="uint8",
                            fill_value=0,
                            compressors=None,
                            overwrite=True
                        )

                if create_down and store_rgb:
                    rgb_shape = (n_import_frames, down_h, down_w, 3)
                    if use_sharding:
                        arrays['downsampled_rgb'] = raw.create_array(
                            name='images_ds_rgb',
                            shape=rgb_shape,
                            chunks=(down_chunk_size, down_h, down_w, 3),
                            shards=(down_io_batch, down_h, down_w, 3),
                            dtype="uint8",
                            fill_value=0,
                            compressors=None,
                            overwrite=True
                        )
                    else:
                        arrays['downsampled_rgb'] = raw.create_array(
                            name='images_ds_rgb',
                            shape=rgb_shape,
                            chunks=(down_chunk_size, down_h, down_w, 3),
                            dtype="uint8",
                            fill_value=0,
                            compressors=None,
                            overwrite=True
                        )

                if 'downsampled_gray' in arrays:
                    arrays['downsampled_gray'].attrs['format'] = 'gray'
                    arrays['downsampled_gray'].attrs['resolution'] = [down_h, down_w]
                if 'downsampled_rgb' in arrays:
                    arrays['downsampled_rgb'].attrs['format'] = 'rgb'
                    arrays['downsampled_rgb'].attrs['resolution'] = [down_h, down_w, 3]

                # Build metadata
                metadata = {
                    "import_method": "kvikio_zarr",
                    "resolutions_mode": resolutions_mode,
                    "device": device,
                    "chunk_size": chunk_size,
                    "io_batch_size": io_batch_size,
                    "batch_size": batch_size,
                    "import_stage": "complete",
                    "original_resolution": [full_h, full_w],
                    "fps": vid_meta.get("fps", 30),
                    "total_frames": n_import_frames,
                    "source_video": str(video_path.name),
                    "source_path": str(video_path.absolute()),
                    "import_timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # Add training data mode metadata if applicable
                if training_data_mode and frame_step:
                    metadata["import_mode"] = "sampled"
                    metadata["frame_step"] = frame_step
                    metadata["original_video_length"] = n_frames
                    metadata["imported_frame_count"] = n_import_frames
                    metadata["import_purpose"] = "training_data"
                else:
                    metadata["import_mode"] = "full"
                
                if create_full:
                    metadata["has_full_resolution"] = True
                
                if create_down:
                    metadata["has_downsampled"] = True
                    metadata["downsampled_resolution"] = list(down_target)
                    metadata["downsample_method"] = down_config.get("method", "area")
                    metadata["downsample_preserve_aspect"] = down_preserve_aspect
                    metadata["downsample_formats"] = down_formats
                    if down_preserve_aspect:
                        scale = min(down_h / full_h, down_w / full_w)
                        resized_h = max(1, min(down_h, int(round(full_h * scale))))
                        resized_w = max(1, min(down_w, int(round(full_w * scale))))
                        pad_top = (down_h - resized_h) // 2
                        pad_bottom = down_h - resized_h - pad_top
                        pad_left = (down_w - resized_w) // 2
                        pad_right = down_w - resized_w - pad_left
                        metadata["downsample_resized_shape"] = [resized_h, resized_w]
                        metadata["downsample_padding"] = {
                            "top": pad_top,
                            "bottom": pad_bottom,
                            "left": pad_left,
                            "right": pad_right,
                        }
                    if downsample_specs:
                        metadata["downsampled_shapes"] = {k: list(v) for k, v in downsample_specs.items()}
                
                if use_sharding:
                    metadata["sharding_enabled"] = True
                    metadata["chunks_per_shard"] = chunks_per_shard
                
                if device == "cuda:0" and gpu_fp16:
                    metadata["gpu_fp16"] = True
                
                raw.attrs.update(metadata)
                if training_data_mode and frame_step:
                    root.attrs["zarr_purpose"] = "training"
                else:
                    root.attrs.setdefault("zarr_purpose", "analysis")

            else:
                # Standard Zarr creation path (CPU or no kvikIO)
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
                
                metadata = {
                    "import_method": "standard_zarr",
                    "device": device,
                    "chunk_size": chunk_size,
                    "io_batch_size": io_batch_size,
                    "batch_size": batch_size,
                    "import_stage": "full_resolution",
                    "downsampled": False,
                    "original_resolution": (full_h, full_w),
                    "fps": vid_meta.get("fps", 30),
                    "total_frames": n_frames,
                    "source_video": str(video_path.name),
                    "source_path": str(video_path.absolute()),
                    "import_timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # Add training data mode metadata if applicable
                if training_data_mode and frame_step:
                    metadata["import_mode"] = "sampled"
                    metadata["frame_step"] = frame_step
                    metadata["original_video_length"] = n_frames
                    metadata["imported_frame_count"] = n_import_frames
                    metadata["import_purpose"] = "training_data"
                else:
                    metadata["import_mode"] = "full"

                if skip_tail_frames:
                    metadata["skip_tail_frames"] = skip_tail_frames
                    metadata["effective_video_length"] = n_frames - skip_tail_frames

                raw.attrs.update(metadata)
                if training_data_mode and frame_step:
                    root.attrs["zarr_purpose"] = "training"
                else:
                    root.attrs.setdefault("zarr_purpose", "analysis")

            # ---- Console info --------------------------------------------------------
            method = "kvikIO GPU --> Zarr" if (kvikio_available and use_kvikio_zarr) else "Standard Zarr"
            write_mode = "sharded" if use_sharding else "chunked"
            
            # Build resolution info for display
            res_info = []
            if create_full:
                res_info.append(f"Full: {full_h}×{full_w}")
            if create_down:
                target_size = down_config.get("size", [640, 640])
                fmt_label = "/".join(down_formats)
                aspect_note = " (preserve_aspect)" if down_preserve_aspect else ""
                res_info.append(f"Downsampled ({fmt_label}): {target_size[0]}×{target_size[1]}{aspect_note}")

            frames_info = f"{n_import_frames}"
            if training_data_mode and frame_step:
                sample_total = n_frames - skip_tail_frames
                frames_info += f" (sampled from {sample_total}, every {frame_step}th)"
            elif skip_tail_frames:
                frames_info += f" (skipped last {skip_tail_frames} of {n_frames})"

            console.print(Panel.fit(
                f"[cyan]Video:[/cyan] {video_path.name}\n"
                f"[cyan]Frames:[/cyan] {frames_info}\n"
                f"[cyan]Resolutions:[/cyan] {', '.join(res_info)}\n"
                f"[cyan]Device:[/cyan] {device}\n"
                f"[cyan]Chunk size:[/cyan] {chunk_size} frames\n"
                f"[cyan]Write size:[/cyan] {io_batch_size} frames ({write_mode})\n"
                f"[cyan]Batch size:[/cyan] {batch_size} frames\n"
                f"[cyan]Method:[/cyan] {method}",
                title="Import Configuration"
            ))

            # ---- Process video -------------------------------------------------------
            if kvikio_available and use_kvikio_zarr:
                console.print("[green]Using kvikIO direct GPU --> Zarr writes[/green]")
                _process_video_gpu_kvikio(
                    vr, n_frames, io_batch_size, batch_size,
                    gpu_fp16, arrays, config, console,
                    frame_indices=frame_indices_to_import
                )
                _finalize_kvikio_zarr_metadata(
                    Path(zarr_path),
                    n_frames=n_import_frames,
                    full_shape=(full_h, full_w) if create_full else None,
                    downsample_specs=downsample_specs,
                    create_full=create_full,
                    chunk_size=chunk_size,
                    io_batch_size=io_batch_size,
                    down_chunk_size=down_chunk_size,
                    down_io_batch_size=down_io_batch,
                    use_sharding=use_sharding,
                )

                # Store original frame indices if sampled import
                if training_data_mode and frame_step and frame_indices_to_import:
                    console.print("[cyan]Creating original_frame_indices array...[/cyan]")
                    # Reopen with LocalStore to write the frame indices array
                    store_local = LocalStore(str(zarr_path))
                    root_local = zarr.open_group(store=store_local, mode='r+')
                    raw_local = root_local['raw_video']
                    raw_local.create_array(
                        'original_frame_indices',
                        data=np.array(frame_indices_to_import, dtype=np.int32),
                        chunks=(min(1000, n_import_frames),),
                        overwrite=True
                    )
                    console.print(f"[green]✓ Stored mapping of {n_import_frames} imported frames to original video indices[/green]")
            else:
                # TODO: Add CPU processing path
                console.print("[red]CPU processing not implemented in this example[/red]")
                raise NotImplementedError("CPU processing path not shown")

            # ---- Final statistics and metadata ---------------------------------------
            duration = time.perf_counter() - start_time
            update_import_duration(root, duration)

            total_gb = 0
            if 'full' in arrays:
                total_gb += (n_import_frames * full_h * full_w) / (1024**3)
            if 'downsampled_gray' in arrays:
                down_h, down_w = down_config.get("size", [640, 640])
                total_gb += (n_import_frames * down_h * down_w) / (1024**3)
            if 'downsampled_rgb' in arrays:
                down_h, down_w = down_config.get("size", [640, 640])
                total_gb += (n_import_frames * down_h * down_w * 3) / (1024**3)

            throughput_gbps = total_gb / max(duration, 1e-9)

            fps = n_import_frames / max(duration, 1e-9)
            eq_fps = None
            if training_data_mode and frame_step:
                eq_fps = fps * frame_step

            # Store performance metadata
            performance_metadata = {
                "import_duration_seconds": duration,
                "throughput_gbps": throughput_gbps,
                "frames_per_second": fps,
                "data_size_gb": total_gb
            }
            raw.attrs.update(performance_metadata)

            # Store video metadata
            format_tags = vid_meta.get("format_tags") or {}
            encoder_fields = vid_meta.get("encoder_fields") or {}
            video_metadata = {
                "video_width": full_w,
                "video_height": full_h,
                "video_codec": vid_meta.get("codec", "unknown"),
                "video_pix_fmt": vid_meta.get("pix_fmt", "unknown"),
                "video_duration_seconds": vid_meta.get("duration_seconds", 0),
                "format_title": format_tags.get("title"),
                "format_comment": format_tags.get("comment"),
                "format_encoder": format_tags.get("encoder"),
                "format_tags": format_tags if format_tags else None,
            }
            if isinstance(encoder_fields, dict) and encoder_fields:
                video_metadata.update(encoder_fields)
            raw.attrs.update(video_metadata)

            # Add comprehensive system info (default to True for HPC tracking)
            include_system_info = bool(ip.get("include_system_info", True))
            if include_system_info:
                try:
                    env_info = get_environment_info(
                        include_all_packages=False,
                        disk_path=str(zarr_path),
                        collect_ip=False
                    )

                    # Replace git info with the one collected in parent process
                    env_info['git'] = parent_git_info
                    
                    # Store full environment info as nested structure
                    system_metadata = {
                        # Platform info
                        "system_hostname": env_info['platform']['hostname'],
                        "system_fqdn": env_info['platform']['fqdn'],
                        "system_os": env_info['platform']['system'],
                        "system_os_release": env_info['platform']['release'],
                        "system_machine": env_info['platform']['machine'],
                        "system_python_version": env_info['platform']['python_version'],
                        "system_username": env_info['platform']['username'],
                        "system_cpu_cores": env_info['platform']['cpu_cores'],
                        
                        # Git info for reproducibility
                        "git_commit_hash": env_info['git'].get('commit_hash', 'unknown'),
                        "git_short_hash": env_info['git'].get('short_hash', 'unknown'),
                        "git_branch": env_info['git'].get('branch', 'unknown'),
                        "git_is_dirty": env_info['git'].get('is_dirty', False),
                        "git_remote_url": env_info['git'].get('remote_url', 'unknown'),
                    }

                    # Add GDS configuration if using GPU
                    if device == "cuda:0":
                        gds_config = get_gds_config()
                        env_info['gds_config'] = gds_config
                        
                        # Store key GDS settings in metadata
                        if 'cufile_json' in gds_config:
                            # Store critical cufile settings
                            cufile = gds_config['cufile_json']
                            system_metadata['gds_cufile_config'] = json.dumps({
                                'max_direct_io_size': cufile.get('max_direct_io_size'),
                                'max_device_cache_size': cufile.get('max_device_cache_size'),
                                'poll_mode': cufile.get('poll_mode'),
                                'nvtx': cufile.get('nvtx'),
                            })
                        
                        if 'kvikio' in gds_config:
                            kvikio = gds_config['kvikio']
                            system_metadata['gds_compat_mode'] = kvikio.get('compat_mode', True)
                            system_metadata['gds_enabled'] = kvikio.get('gds_enabled', False)
                            system_metadata['kvikio_threads'] = kvikio.get('thread_pool_size', 0)
                    
                    
                    # Add CPU details if available
                    if 'cpu_details' in env_info['platform']:
                        cpu = env_info['platform']['cpu_details']
                        system_metadata.update({
                            "cpu_model": cpu.get('model', 'unknown'),
                            "cpu_arch": cpu.get('arch', 'unknown'),
                        })
                    
                    # Add memory and disk info if available
                    if 'memory' in env_info['platform']:
                        mem = env_info['platform']['memory']
                        system_metadata.update({
                            "memory_total_gb": mem.get('total_gb', 0),
                            "memory_available_gb": mem.get('available_gb', 0),
                            "memory_percent_used": mem.get('percent_used', 0),
                        })
                    
                    if 'disk' in env_info['platform']:
                        disk = env_info['platform']['disk']
                        system_metadata.update({
                            "disk_path": disk.get('path', str(zarr_path)),
                            "disk_total_gb": disk.get('total_gb', 0),
                            "disk_available_gb": disk.get('available_gb', 0),
                            "disk_percent_used": disk.get('percent_used', 0),
                        })
                    
                    # Add HPC scheduler info (LSF at Janelia/SLURM) if present
                    if 'lsf' in env_info['platform']:
                        lsf = env_info['platform']['lsf']
                        system_metadata.update({
                            "hpc_scheduler": "LSF",
                            "lsf_job_id": lsf.get('job_id', 'unknown'),
                            "lsf_job_name": lsf.get('job_name', 'unknown'),
                            "lsf_queue": lsf.get('queue', 'unknown'),
                            "lsf_hosts": lsf.get('hosts', 'unknown'),
                        })
                    elif 'slurm' in env_info['platform']:
                        slurm = env_info['platform']['slurm']
                        system_metadata.update({
                            "hpc_scheduler": "SLURM",
                            "slurm_job_id": slurm.get('job_id', 'unknown'),
                            "slurm_job_name": slurm.get('job_name', 'unknown'),
                            "slurm_node_list": slurm.get('node_list', 'unknown'),
                        })
                    
                    # Add GPU info if available
                    if env_info.get('gpu', {}).get('available'):
                        gpu_info = env_info['gpu']
                        system_metadata.update({
                            "gpu_available": True,
                            "gpu_backend": gpu_info.get('backend', 'unknown'),
                            "gpu_count": len(gpu_info.get('devices', [])),
                        })
                        
                        # If CUDA, add version
                        if 'cuda_version' in gpu_info:
                            system_metadata["cuda_version"] = gpu_info['cuda_version']
                        
                        # Add details for the primary GPU used
                        if gpu_info.get('devices'):
                            primary_gpu = gpu_info['devices'][0]
                            system_metadata.update({
                                "gpu_name": primary_gpu.get('name', 'unknown'),
                                "gpu_compute_capability": primary_gpu.get('compute_capability', 'unknown'),
                                "gpu_memory_total_gb": primary_gpu.get('total_memory_gb', 0),
                            })
                            
                            # Add runtime telemetry if available
                            if 'temperature_c' in primary_gpu:
                                system_metadata["gpu_temperature_c"] = primary_gpu['temperature_c']
                            if 'power_draw_w' in primary_gpu:
                                system_metadata["gpu_power_draw_w"] = primary_gpu['power_draw_w']
                            if 'utilization_percent' in primary_gpu:
                                system_metadata["gpu_utilization_percent"] = primary_gpu['utilization_percent']
                    
                    # Add environment summary
                    env_summary = env_info.get('environment', {})
                    if env_summary:
                        system_metadata.update({
                            "environment_type": env_summary.get('environment_type', 'unknown'),
                            "environment_name": env_summary.get('environment_name', 'unknown'),
                            "total_packages": env_summary.get('total_packages', 0),
                        })
                        
                        # Add deep learning framework info
                        if 'deep_learning_framework' in env_summary:
                            system_metadata["deep_learning_framework"] = env_summary['deep_learning_framework']
                        
                        # Store key packages as a separate attribute for easy access
                        if 'key_packages' in env_summary:
                            # Store as JSON string for Zarr compatibility
                            system_metadata["key_packages_json"] = json.dumps(env_summary['key_packages'])
                    
                    raw.attrs.update(system_metadata)
                    
                    # Also store the complete environment info as a JSON attribute for full reproducibility
                    raw.attrs["_full_environment_info"] = json.dumps(env_info, default=str)
                    
                    console.print(f"[green]✓ System metadata collected and stored[/green]")
                    
                except Exception as e:
                    console.print(f"[yellow]Could not collect full system info: {e}[/yellow]")
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")

            # ---- Display completion info ---------------------------------------------
            output_arrays = []
            if 'full' in arrays:
                output_arrays.append(f"  - raw_video/images_full: ({n_import_frames}, {full_h}, {full_w})")
            if 'downsampled_gray' in arrays:
                _, h, w = arrays['downsampled_gray'].shape
                output_arrays.append(f"  - raw_video/images_ds: ({n_import_frames}, {h}, {w})")
            if 'downsampled_rgb' in arrays:
                _, h, w, c = arrays['downsampled_rgb'].shape
                output_arrays.append(f"  - raw_video/images_ds_rgb: ({n_import_frames}, {h}, {w}, {c})")
            if training_data_mode and frame_step:
                output_arrays.append(f"  - raw_video/original_frame_indices: ({n_import_frames},)")

            perf_lines = [
                f"  Time: {duration:.1f}s ({duration/60:.1f} min)",
                f"  FPS (saved frames): {fps:.1f}",
            ]
            if eq_fps is not None:
                perf_lines.append(f"  FPS (equivalent source): {eq_fps:.1f}")
                perf_lines.append("  Note: sampled import; FPS reflects saved frames, not full decode")
            perf_lines.append(f"  Throughput: {throughput_gbps:.2f} GB/s")

            console.print(Panel(
                f"[green]✓ Import completed successfully[/green]\n\n"
                f"[yellow]Performance:[/yellow]\n"
                + "\n".join(perf_lines)
                + "\n\n"
                f"[yellow]Output:[/yellow]\n"
                f"  Path: {zarr_path}\n"
                f"[yellow]Arrays created:[/yellow]\n" + "\n".join(output_arrays),
                title="Import Complete",
                expand=False
            ))

            # ---- Explicit cleanup before hard exit ----------------------------------
            console.print("[dim]Performing pre-exit cleanup...[/dim]")
            
            # Close the Zarr store explicitly
            root.store.close()
            
            # This terminates the child process immediately WITHOUT running
            # any atexit handlers, avoiding the segmentation fault
            console.print("[dim]Worker process performing hard exit...[/dim]")
            _exit(0)

        except Exception as e:
            # If any error occurs, print it and exit with error code
            console.print(f"[bold red]Error in import worker process:[/bold red]")
            console.print_exception()
            _exit(1)

    else:
        # === PARENT PROCESS ===
        # The parent's only job is to wait for the child to finish
        console.print(f"[dim]Spawned import worker process with PID: {pid}[/dim]")
        
        # Wait for the child process to exit and get its status
        _, status = waitpid(pid, 0)
        
        # Check the exit status
        if WIFEXITED(status):
            exit_code = WEXITSTATUS(status)
            if exit_code == 0:
                console.print(f"[green]✓ Worker process {pid} completed successfully[/green]")
                return Path(zarr_path)
            else:
                raise RuntimeError(f"Import worker process failed with exit code {exit_code}")
        else:
            # Process was terminated by a signal (shouldn't happen with our hard exit)
            raise RuntimeError(f"Import worker process was terminated unexpectedly (status: {status})")


# ---------------- Validation and utility functions ---------------- #

def validate_import(zarr_path: str, expected_frames: int) -> bool:
    """Validate that the import completed successfully."""
    try:
        root = zarr.open_group(zarr_path, mode='r')
        raw = root.get('raw_video')
        if raw is None:
            return False
        
        # Check what was imported based on metadata
        resolutions_mode = raw.attrs.get('resolutions_mode', 'full')
        
        if resolutions_mode in ['full', 'both'] and 'images_full' in raw:
            actual_frames = raw['images_full'].shape[0]
            if actual_frames != expected_frames:
                print(f"Full resolution frame count mismatch: expected {expected_frames}, got {actual_frames}")
                return False
                
        if resolutions_mode in ['downsampled', 'both'] and 'images_ds' in raw:
            actual_frames = raw['images_ds'].shape[0]
            if actual_frames != expected_frames:
                print(f"Downsampled frame count mismatch: expected {expected_frames}, got {actual_frames}")
                return False
        
        if raw.attrs.get('import_stage') != 'complete':
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
        'import_duration': float(raw.attrs.get('import_duration_seconds', 0.0)),
        'import_method': raw.attrs.get('import_method', 'unknown'),
        'device': raw.attrs.get('device', 'unknown'),
        'downsampled': bool(raw.attrs.get('downsampled', False)),
    }

    if stats['import_duration'] > 0:
        stats['throughput_fps'] = stats['total_frames'] / stats['import_duration']
        stats['throughput_gbps'] = raw.attrs.get('throughput_gbps', 0.0)

    # Approximate data size in GiB (uncompressed)
    dtype_size = arr.dtype.itemsize
    stats['data_size_gib'] = (
        stats['total_frames'] * stats['resolution'][0] * stats['resolution'][1] * dtype_size
    ) / (1024 ** 3)

    return stats


# ---------------- CLI entrypoint ---------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a video into a Palette Zarr archive.")
    parser.add_argument("video_path", type=Path, help="Path to the source video file.")
    parser.add_argument(
        "--zarr-path",
        type=Path,
        help="Destination Zarr path (default: <video_stem>.zarr next to the video).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML configuration with pipeline parameters.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing Zarr path before importing.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable GPU decoding and force CPU mode.",
    )
    parser.add_argument(
        "--training-data",
        action="store_true",
        help="Enable sampled import mode for training data collection (requires --frame-step).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        metavar="N",
        help="Import every Nth frame (requires --training-data flag). Example: --frame-step 100 imports frames 0, 100, 200, ...",
    )
    parser.add_argument(
        "--skip-tail-frames",
        type=int,
        metavar="N",
        default=0,
        help="Skip the last N frames to avoid EOF/GOP issues during decoding.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    console = Console()

    # Validate training data arguments
    if args.frame_step is not None and not args.training_data:
        console.print("[bold red]Error:[/bold red] --frame-step requires --training-data flag")
        return 1

    if args.training_data and args.frame_step is None:
        console.print("[bold red]Error:[/bold red] --training-data requires --frame-step N")
        return 1

    if args.frame_step is not None and args.frame_step < 1:
        console.print(f"[bold red]Error:[/bold red] --frame-step must be >= 1, got {args.frame_step}")
        return 1
    if args.skip_tail_frames is not None and args.skip_tail_frames < 0:
        console.print(f"[bold red]Error:[/bold red] --skip-tail-frames must be >= 0, got {args.skip_tail_frames}")
        return 1

    video_path = Path(args.video_path).expanduser()
    if not video_path.exists():
        console.print(f"[bold red]Video file not found:[/bold red] {video_path}")
        return 1

    if args.zarr_path:
        zarr_path = Path(args.zarr_path).expanduser()
    else:
        zarr_path = video_path.with_suffix(".zarr")

    if zarr_path.exists():
        if not args.overwrite:
            console.print(
                f"[bold red]Destination already exists:[/bold red] {zarr_path}\n"
                "Use --overwrite to remove it or specify a different --zarr-path."
            )
            return 1
        console.print(f"[yellow]Overwriting existing Zarr directory: {zarr_path}[/yellow]")
        try:
            if zarr_path.is_file():
                zarr_path.unlink()
            else:
                shutil.rmtree(zarr_path)
        except OSError as exc:
            console.print(f"[bold red]Failed to remove existing path:[/bold red] {exc}")
            return 1

    config = _load_import_config(args.config, console)

    cli_args = {
        "video_path": str(video_path),
        "zarr_path": str(zarr_path),
        "config_path": str(args.config) if args.config else None,
        "cpu_only": bool(args.cpu_only),
        "overwrite": bool(args.overwrite),
        "training_data": bool(args.training_data),
        "frame_step": int(args.frame_step) if args.frame_step else None,
        "skip_tail_frames": int(args.skip_tail_frames) if args.skip_tail_frames else 0,
    }

    try:
        import_video(
            video_path=str(video_path),
            zarr_path=str(zarr_path),
            config=config,
            cli_args=cli_args,
            console=console,
            use_gpu=not args.cpu_only,
            force_cpu=bool(args.cpu_only),
        )
    except Exception as exc:
        console.print(f"[bold red]Import failed:[/bold red] {exc}")
        return 1

    console.print(f"[green]Import complete.[/green] Zarr saved to {zarr_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
