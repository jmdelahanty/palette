#!/usr/bin/env python3
"""
Direct video inference with YOLO - no import required.

For when you have a trained model and just want detections,
not training data or full video storage.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "65536")

# Try to import decord for faster video decoding (GPU or CPU) before other FFmpeg users.
try:
    import decord  # type: ignore
    from decord import VideoReader, cpu, gpu  # type: ignore
    _DECORD_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    decord = None  # type: ignore
    VideoReader = None  # type: ignore
    cpu = None  # type: ignore
    gpu = None  # type: ignore
    _DECORD_IMPORT_ERROR = exc

import sys
import time
import yaml
import zarr
import numpy as np
import torch
import cv2
import imageio.v3 as iio
from datetime import datetime, timezone
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.markup import escape
from ultralytics import YOLO

from fisheye.shared.zarr.schema import get_run_group
from fisheye.utils.system import get_environment_info, get_git_info


def _decord_available() -> bool:
    return decord is not None and VideoReader is not None and cpu is not None


def _init_decord_reader(video_path: Path, prefer_gpu: bool, console: Console) -> Optional[Dict[str, Any]]:
    """Initialise a Decord VideoReader with GPU preference and graceful fallback."""
    if not _decord_available():
        if _DECORD_IMPORT_ERROR:
            console.print(f"[yellow]Decord unavailable: {escape(str(_DECORD_IMPORT_ERROR))}[/yellow]")
        return None

    if prefer_gpu and torch.cuda.is_available():
        try:
            decord.bridge.set_bridge('torch')
            vr = VideoReader(str(video_path), ctx=gpu(0))
            first = vr[0]
            height, width = int(first.shape[0]), int(first.shape[1])
            fps = vr.get_avg_fps()
            console.print(f"[green]✓[/green] Using Decord GPU decoder")
            return {
                'reader': vr,
                'type': 'decord_gpu',
                'on_gpu': True,
                'width': width,
                'height': height,
                'fps': fps,
            }
        except Exception as exc:
            console.print(f"[yellow]Decord GPU decoder failed ({escape(str(exc))}); retrying on CPU[/yellow]")

    try:
        decord.bridge.set_bridge('native')
        vr = VideoReader(str(video_path), ctx=cpu())
        first = vr[0]
        height, width = int(first.shape[0]), int(first.shape[1])
        fps = vr.get_avg_fps()
        console.print(f"[green]✓[/green] Using Decord CPU decoder")
        return {
            'reader': vr,
            'type': 'decord_cpu',
            'on_gpu': False,
            'width': width,
            'height': height,
            'fps': fps,
        }
    except Exception as exc:
        console.print(f"[yellow]Decord CPU decoder failed ({escape(str(exc))}); falling back to OpenCV[/yellow]")
        return None


def get_video_metadata(video_path: Path, cap: Optional[cv2.VideoCapture], width: int, height: int, n_frames: int, fps: float) -> Dict[str, Any]:
    """
    Get comprehensive video metadata similar to import_video.py.
    Uses both cv2 and imageio for maximum compatibility.
    """
    cap_owner = False
    if cap is None:
        cap = cv2.VideoCapture(str(video_path))
        cap_owner = True
        if not cap.isOpened():
            cap = None
    # Try to get codec info from imageio
    try:
        iio_meta = iio.immeta(str(video_path))
    except Exception:
        iio_meta = {}
    
    # Build metadata dictionary matching import_video structure
    meta = {
        "source_video": str(video_path.name),
        "source_path": str(video_path.absolute()),
        "width": width,
        "height": height,
        "total_frames": n_frames,
        "fps": fps,
        "duration_seconds": n_frames / fps if fps > 0 else 0,
    }
    
    # Add codec information if available
    if iio_meta:
        meta["codec"] = iio_meta.get("codec", "unknown")
        meta["pix_fmt"] = iio_meta.get("pix_fmt", "unknown")
        meta["imageio_metadata"] = iio_meta
    else:
        # Fallback to cv2 fourcc
        if cap is not None:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            if fourcc > 0:
                codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                meta["codec"] = codec_str
            else:
                meta["codec"] = "unknown"
        else:
            meta["codec"] = "unknown"
        meta["pix_fmt"] = "unknown"
    
    if cap_owner and cap is not None:
        cap.release()
    return meta


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    
    if config_path is None:
        # Try default locations
        default_paths = [
            Path('yolo_detect_config.yaml'),  # Current directory
            Path('configs/fisheye/yolo_detect_config.yaml'),  # Standard location
            Path(__file__).parent.parent.parent / 'configs/fisheye/yolo_detect_config.yaml',  # Relative to module
            Path.home() / 'gitrepos/palette/configs/fisheye/yolo_detect_config.yaml',  # Absolute
            Path('src/fisheye/yolo_detect_config.yaml'),  # Old location
            Path(__file__).parent / 'yolo_detect_config.yaml',  # Same dir as this script
        ]
        
        console = Console()
        console.print("[dim]Searching for config file...[/dim]")
        
        for path in default_paths:
            console.print(f"[dim]  Checking: {path}[/dim]")
            if path.exists():
                console.print(f"[green]  ✓ Found config: {path}[/green]")
                config_path = path
                break
        
        if config_path is None:
            console.print("[yellow]  No config file found, using defaults[/yellow]")
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    return {}


def detect_yolo(
    video_path: str,
    model_path: Optional[str] = None,
    output_zarr: str = None,
    config_path: Optional[str] = None,
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    max_det: Optional[int] = None,
    batch_size: Optional[int] = None,
    console: Optional[Console] = None,
    use_gpu: Optional[bool] = None
) -> str:
    """
    Run YOLO inference directly on video file, creating minimal zarr output.
    
    This is the INFERENCE pathway - for getting detections from a trained model.
    Does NOT import full video, only saves detection results.
    
    Args:
        video_path: Path to input video file
        model_path: Path to trained YOLO model (.pt) - optional if in config
        output_zarr: Path for output zarr - optional, will auto-generate if None
        config_path: Path to YAML config file (optional)
        conf_threshold: Confidence threshold (overrides config)
        iou_threshold: IoU threshold for NMS (overrides config)
        max_det: Max detections per frame (overrides config)
        batch_size: Frames to process at once (overrides config)
        console: Rich console
        use_gpu: Use GPU for inference (overrides config)
        
    Returns:
        Name of detect_runs group
    """
    if console is None:
        console = Console()
    
    console.rule("[bold]YOLO Video Inference[/bold]")
    
    # Load config
    config = load_config(config_path)
    
    # Get parameters from config with CLI overrides
    model_path = model_path or config.get('model', {}).get('path')
    if model_path is None:
        raise ValueError("model_path required (via argument or config file)")
    
    # Auto-generate output path if not provided
    video_path = Path(video_path)
    if output_zarr is None:
        output_zarr = video_path.parent / f"{video_path.stem}_detections.zarr"
    
    # Get detection parameters with CLI overrides taking precedence
    detect_config = config.get('detection', {})
    conf_threshold = conf_threshold if conf_threshold is not None else detect_config.get('conf_threshold', 0.25)
    iou_threshold = iou_threshold if iou_threshold is not None else detect_config.get('iou_threshold', 0.45)
    max_det = max_det if max_det is not None else detect_config.get('max_det', 20)
    batch_size = batch_size if batch_size is not None else detect_config.get('batch_size', 32)
    
    # Get video processing parameters
    video_config = config.get('video', {})
    resize_dims = video_config.get('resize', None)  # e.g., [640, 640] or None
    
    # GPU/device configuration
    device_config = config.get('model', {}).get('device', 'auto')
    if use_gpu is None:
        use_gpu = device_config != 'cpu'
    
    # Validate inputs
    model_path = Path(model_path).expanduser()
    output_zarr = Path(output_zarr)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if output_zarr.exists():
        console.print(f"[yellow]Warning: {output_zarr} already exists, will overwrite[/yellow]")
    
    console.print(f"Video: [cyan]{video_path}[/cyan]")
    console.print(f"Model: [cyan]{model_path}[/cyan]")
    console.print(f"Output: [cyan]{output_zarr}[/cyan]")
    
    # Print parameters
    console.print(f"\n[bold]Detection Parameters:[/bold]")
    console.print(f"  Confidence threshold: {conf_threshold}")
    console.print(f"  IoU threshold: {iou_threshold}")
    console.print(f"  Max detections: {max_det}")
    console.print(f"  Batch size: {batch_size}")
    if resize_dims:
        console.print(f"  Resize to: {resize_dims[0]}×{resize_dims[1]}")
    else:
        console.print(f"  Resize: None (use original)")
    
    # Load model
    console.print("\n[bold]Loading model...[/bold]")
    model = YOLO(str(model_path))
    try:
        model.fuse()
    except AttributeError:
        pass  # Older versions may not expose fuse; Ultralytics will handle it.
    
    # Check device and move model
    model_fp16 = False
    
    if not use_gpu:
        model.to('cpu')
        console.print(f"[green]✓[/green] Model loaded on CPU")
    else:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            model.to('cuda')
            model.model = model.model.to(memory_format=torch.channels_last)
            model.half()
            model_fp16 = True
            console.print(f"[green]✓[/green] Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            console.print(f"[cyan]  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB[/cyan]")
        else:
            console.print(f"[yellow]⚠[/yellow]  CUDA not available, using CPU")
            use_gpu = False
    
    # Open video to get metadata
    console.print("\n[bold]Opening video...[/bold]")
    
    decord_info = _init_decord_reader(video_path, prefer_gpu=bool(use_gpu), console=console)
    vr = decord_info['reader'] if decord_info else None
    cap = None
    
    if decord_info:
        n_frames = len(vr)
        width = decord_info['width']
        height = decord_info['height']
        fps = decord_info['fps']
        video_reader_type = decord_info['type']
        use_decord = True
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        console.print(f"[green]✓[/green] Using OpenCV decoder")
        video_reader_type = 'opencv'
        use_decord = False
    
    # Determine dimensions for normalization (actual or resized)
    if resize_dims:
        inference_width, inference_height = resize_dims
        console.print(f"[green]✓[/green] Video: {n_frames} frames, {fps:.1f} fps, {width}×{height}")
        console.print(f"[cyan]  Will resize to {inference_width}×{inference_height} for inference[/cyan]")
    else:
        inference_width, inference_height = width, height
        console.print(f"[green]✓[/green] Video: {n_frames} frames, {fps:.1f} fps, {width}×{height}")
    
    # Create minimal zarr structure (NO raw_video!)
    console.print("\n[bold]Creating zarr structure...[/bold]")
    root = zarr.open_group(str(output_zarr), mode='w')
    
    # Get comprehensive video metadata
    vid_meta = get_video_metadata(video_path, cap, width, height, n_frames, fps)
    
    # Get git info for reproducibility (matching import_video.py)
    git_info = get_git_info()
    
    # Get full environment info for provenance
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(output_zarr),
        collect_ip=False
    )
    
    # Store basic video metadata (matching import_video structure)
    root.attrs.update({
        # Video source info
        'source_video': vid_meta['source_video'],
        'source_video_path': vid_meta['source_path'],
        'source_path': vid_meta['source_path'],  # Alias for compatibility
        
        # Video properties
        'video_width': width,
        'video_height': height,
        'width': width,  # Alias
        'height': height,  # Alias
        'fps': fps,
        'n_frames': n_frames,
        'total_frames': n_frames,  # Alias
        'duration_seconds': vid_meta['duration_seconds'],
        
        # Codec info
        'video_codec': vid_meta.get('codec', 'unknown'),
        'video_pix_fmt': vid_meta.get('pix_fmt', 'unknown'),
        
        # Pipeline info - CLEAR DISTINCTION
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'pipeline_type': 'yolo_inference',  # Changed from 'inference_only'
        'has_raw_video': False,  # Flag that we don't store video
        'detection_method': 'yolo',  # Will be 'blob' for traditional
        
        # Model info
        'model_path': str(model_path.absolute()),
        'model_name': model_path.name,
        
        # Processing info
        'inference_width': inference_width,
        'inference_height': inference_height,
        'resized_for_inference': resize_dims is not None,
        
        # Git provenance (matching import_video)
        'git_commit_hash': git_info.get('commit_hash', 'unknown'),
        'git_short_hash': git_info.get('short_hash', 'unknown'),
        'git_branch': git_info.get('branch', 'unknown'),
        'git_is_dirty': git_info.get('is_dirty', False),
        'git_remote_url': git_info.get('remote_url', 'unknown'),
        
        # System provenance (matching import_video)
        'system_hostname': env_info['platform']['hostname'],
        'system_fqdn': env_info['platform']['fqdn'],
        'system_os': env_info['platform']['system'],
        'system_os_release': env_info['platform']['release'],
        'system_machine': env_info['platform']['machine'],
        'system_python_version': env_info['platform']['python_version'],
        'system_username': env_info['platform']['username'],
        'system_cpu_cores': env_info['platform']['cpu_cores'],
    })
    
    # Add optional metadata if available
    if 'imageio_metadata' in vid_meta:
        root.attrs['imageio_metadata'] = vid_meta['imageio_metadata']
    
    # Add CPU details if available
    if 'cpu_details' in env_info['platform']:
        cpu = env_info['platform']['cpu_details']
        root.attrs.update({
            'cpu_model': cpu.get('model', 'unknown'),
            'cpu_arch': cpu.get('arch', 'unknown'),
        })
    
    # Add memory info if available
    if 'memory' in env_info['platform']:
        mem = env_info['platform']['memory']
        root.attrs.update({
            'memory_total_gb': mem.get('total_gb', 0),
            'memory_available_gb': mem.get('available_gb', 0),
            'memory_percent_used': mem.get('percent_used', 0),
        })
    
    # Add disk info if available
    if 'disk' in env_info['platform']:
        disk = env_info['platform']['disk']
        root.attrs.update({
            'disk_path': disk.get('path', str(output_zarr)),
            'disk_total_gb': disk.get('total_gb', 0),
            'disk_available_gb': disk.get('available_gb', 0),
            'disk_percent_used': disk.get('percent_used', 0),
        })
    
    # Add HPC scheduler info if present (matching import_video)
    if 'lsf' in env_info['platform']:
        lsf = env_info['platform']['lsf']
        root.attrs.update({
            'hpc_scheduler': 'LSF',
            'lsf_job_id': lsf.get('job_id', 'unknown'),
            'lsf_job_name': lsf.get('job_name', 'unknown'),
            'lsf_queue': lsf.get('queue', 'unknown'),
            'lsf_hosts': lsf.get('hosts', 'unknown'),
        })
    elif 'slurm' in env_info['platform']:
        slurm = env_info['platform']['slurm']
        root.attrs.update({
            'hpc_scheduler': 'SLURM',
            'slurm_job_id': slurm.get('job_id', 'unknown'),
            'slurm_job_name': slurm.get('job_name', 'unknown'),
            'slurm_node_list': slurm.get('node_list', 'unknown'),
        })
    
    # Add GPU info if available
    if env_info.get('gpu', {}).get('available'):
        gpu_info = env_info['gpu']
        root.attrs.update({
            'gpu_available': True,
            'gpu_backend': gpu_info.get('backend', 'unknown'),
            'gpu_count': len(gpu_info.get('devices', [])),
        })
        
        if 'cuda_version' in gpu_info:
            root.attrs['cuda_version'] = gpu_info['cuda_version']
        
        if gpu_info.get('devices'):
            primary_gpu = gpu_info['devices'][0]
            root.attrs.update({
                'gpu_name': primary_gpu.get('name', 'unknown'),
                'gpu_compute_capability': primary_gpu.get('compute_capability', 'unknown'),
                'gpu_memory_total_gb': primary_gpu.get('total_memory_gb', 0),
            })
    
    # Add environment summary (matching import_video)
    env_summary = env_info.get('environment', {})
    if env_summary:
        root.attrs.update({
            'environment_type': env_summary.get('environment_type', 'unknown'),
            'environment_name': env_summary.get('environment_name', 'unknown'),
            'total_packages': env_summary.get('total_packages', 0),
        })
        
        if 'deep_learning_framework' in env_summary:
            root.attrs['deep_learning_framework'] = env_summary['deep_learning_framework']
        
        if 'key_packages' in env_summary:
            import json
            root.attrs['key_packages_json'] = json.dumps(env_summary['key_packages'])
    
    # Store complete environment info for full reproducibility (matching import_video)
    import json
    root.attrs['_full_environment_info'] = json.dumps(env_info, default=str)
    
    # Create detect_runs group
    root.create_group('detect_runs')
    detect_group, run_name = get_run_group(root, 'detect', console, create_new=True)
    
    console.print(f"[green]✓[/green] Zarr structure created")
    
    # Storage for detections
    all_frame_indices = []
    all_bboxes = []
    all_scores = []
    frame_counts = np.zeros(n_frames, dtype=np.int32)
    
    # Process video in batches
    frame_idx = 0
    batch_frames = []
    batch_indices = []
    
    console.print("\n[bold]Running inference...[/bold]")
    console.print(f"[cyan]Decoder: {video_reader_type}[/cyan]")
    
    # Performance tracking
    inference_times = []
    read_times = []
    processing_start = time.time()
    
    decord_on_gpu = bool(decord_info and decord_info.get('on_gpu'))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[fps]:.1f} fps"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing frames", total=n_frames, fps=0.0)
        
        batch_count = 0
        
        if use_decord:
            # Double-buffered decode: prefetch next batch while current inference runs.
            batch_starts = list(range(0, n_frames, batch_size))
            prefetched = None
            
            for idx, batch_start in enumerate(batch_starts):
                batch_end = min(batch_start + batch_size, n_frames)
                indices = list(range(batch_start, batch_end))
                
                if prefetched is None:
                    read_start = time.time()
                    current_batch = vr.get_batch(indices)
                    read_times.append(time.time() - read_start)
                else:
                    current_batch = prefetched
                
                next_indices = None
                prefetched = None
                if idx + 1 < len(batch_starts):
                    next_start = batch_starts[idx + 1]
                    next_end = min(next_start + batch_size, n_frames)
                    next_indices = list(range(next_start, next_end))
                    if not decord_on_gpu or torch.cuda.is_available():
                        prefetch_start = time.time()
                        prefetched = vr.get_batch(next_indices)
                        read_times.append(time.time() - prefetch_start)
                
                if decord_on_gpu:
                    import torch.nn.functional as F
                    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                    dtype = torch.float16 if model_fp16 else torch.float32
                    frames_chw = current_batch.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W] uint8
                    
                    total = frames_chw.shape[0]
                    chunk_size = total
                    results = []
                    start = 0
                    
                    while start < total:
                        end = min(start + chunk_size, total)
                        chunk = frames_chw[start:end]
                        try:
                            chunk = chunk.to(device=device, dtype=dtype, non_blocking=True).contiguous(memory_format=torch.channels_last)
                            if resize_dims:
                                chunk = F.interpolate(
                                    chunk,
                                    size=resize_dims,
                                    mode='bilinear',
                                    align_corners=False
                                )
                            chunk = chunk.mul_(1.0 / 255.0)
                            
                            inference_start = time.time()
                            preds = model.predict(
                                chunk,
                                conf=conf_threshold,
                                iou=iou_threshold,
                                max_det=max_det,
                                verbose=False,
                                device='cuda' if use_gpu else 'cpu',
                                half=model_fp16
                            )
                            inference_times.append(time.time() - inference_start)
                            results.extend(preds)
                            start = end
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            if chunk_size == 1:
                                raise
                            chunk_size = max(1, chunk_size // 2)
                            continue
                        finally:
                            del chunk
                    
                    del frames_chw
                else:
                    frames_nd = current_batch.asnumpy() if hasattr(current_batch, "asnumpy") else np.asarray(current_batch)
                    if resize_dims:
                        batch_frames_np = [
                            cv2.resize(frame, tuple(resize_dims)) for frame in frames_nd
                        ]
                    else:
                        batch_frames_np = [np.asarray(frame) for frame in frames_nd]
                    del frames_nd
                
                    inference_start = time.time()
                    results = model.predict(
                        batch_frames_np,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        max_det=max_det,
                        verbose=False,
                        device='cuda' if use_gpu else 'cpu',
                        half=model_fp16
                    )
                    inference_times.append(time.time() - inference_start)
                
                for batch_i, result in enumerate(results):
                    global_frame_idx = indices[batch_i]
                    
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) / 2 / inference_width
                        cy = (y1 + y2) / 2 / inference_height
                        w = (x2 - x1) / inference_width
                        h = (y2 - y1) / inference_height
                        
                        all_frame_indices.append(global_frame_idx)
                        all_bboxes.append([cx, cy, w, h])
                        all_scores.append(score)
                        frame_counts[global_frame_idx] += 1
                
                del current_batch
                if not decord_on_gpu:
                    del batch_frames_np, frames_nd
                
                frame_idx += len(indices)
                batch_count += 1
                
                elapsed = time.time() - processing_start
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                progress.update(task, advance=len(indices), fps=current_fps)
                
                if batch_count % 100 == 0:
                    avg_inference = np.mean(inference_times[-100:])
                    avg_read = np.mean(read_times[-100:])
                    console.print(f"[dim]Batch {batch_count}: read={avg_read*1000:.1f}ms, "
                                f"inference={avg_inference*1000:.1f}ms, fps={current_fps:.1f}[/dim]")
            
            if prefetched is not None:
                del prefetched
        
        else:
            # OpenCV frame-by-frame processing
            while True:
                # Time frame reading
                read_start = time.time()
                ret, frame = cap.read()
                read_times.append(time.time() - read_start)
                
                if not ret:
                    break
                
                # Resize if specified
                if resize_dims:
                    frame = cv2.resize(frame, tuple(resize_dims))
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch_frames.append(frame_rgb)
                batch_indices.append(frame_idx)
                frame_idx += 1
                
                # Process batch when full or at end
                if len(batch_frames) == batch_size or frame_idx == n_frames:
                    # Time inference
                    inference_start = time.time()
                    
                    # Run inference
                    results = model.predict(
                        batch_frames,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        max_det=max_det,
                        verbose=False,
                        device='cuda' if use_gpu else 'cpu',
                        half=model_fp16
                    )
                    
                    inference_time = time.time() - inference_start
                    inference_times.append(inference_time)
                    
                    # Calculate FPS
                    elapsed = time.time() - processing_start
                    current_fps = frame_idx / elapsed if elapsed > 0 else 0
                    
                    # Extract detections
                    for batch_i, result in enumerate(results):
                        global_frame_idx = batch_indices[batch_i]
                        
                        if result.boxes is None or len(result.boxes) == 0:
                            continue
                        
                        boxes = result.boxes.xyxy.cpu().numpy()
                        scores = result.boxes.conf.cpu().numpy()
                        
                        # Convert to normalized center format [cx, cy, w, h]
                        for box, score in zip(boxes, scores):
                            x1, y1, x2, y2 = box
                            # Normalize using inference dimensions
                            cx = (x1 + x2) / 2 / inference_width
                            cy = (y1 + y2) / 2 / inference_height
                            w = (x2 - x1) / inference_width
                            h = (y2 - y1) / inference_height
                            
                            all_frame_indices.append(global_frame_idx)
                            all_bboxes.append([cx, cy, w, h])
                            all_scores.append(score)
                            frame_counts[global_frame_idx] += 1
                    
                    batch_size_actual = len(batch_frames)
                    
                    # Clear batch
                    batch_frames = []
                    batch_indices = []
                    batch_count += 1
                    
                    # Update progress
                    progress.update(task, advance=batch_size_actual, fps=current_fps)
                    
                    # Print diagnostics every 100 batches
                    if batch_count % 100 == 0:
                        avg_inference = np.mean(inference_times[-100:]) if len(inference_times) > 0 else 0
                        avg_read = np.mean(read_times[-100:]) if len(read_times) > 0 else 0
                        console.print(f"[dim]Batch {batch_count}: inference={avg_inference*1000:.1f}ms, "
                                    f"read={avg_read*1000:.1f}ms, fps={current_fps:.1f}[/dim]")
            
            cap.release()

    if use_decord and vr is not None:
        del vr
        if decord_on_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_time = time.time() - processing_start
    console.print(f"[green]✓[/green] Inference complete")
    console.print(f"[cyan]  Total time: {total_time:.1f}s ({total_time/60:.1f} min)[/cyan]")
    console.print(f"[cyan]  Average FPS: {n_frames/total_time:.1f}[/cyan]")
    
    if len(inference_times) > 0:
        avg_inference = np.mean(inference_times)
        console.print(f"[cyan]  Avg inference time per batch: {avg_inference*1000:.1f}ms[/cyan]")
        console.print(f"[cyan]  Avg read time per batch: {np.mean(read_times)*1000:.1f}ms[/cyan]")
    
    # Convert to arrays
    console.print("\n[bold]Saving detections to zarr...[/bold]")
    frame_indices = np.array(all_frame_indices, dtype=np.int32)
    bbox_coords = np.array(all_bboxes, dtype=np.float64)
    scores = np.array(all_scores, dtype=np.float32)
    
    # Save to zarr
    detect_group.create_array('frame_indices', data=frame_indices, chunks=(1000,))
    detect_group.create_array('bbox_norm_coords', data=bbox_coords, chunks=(1000, 4))
    detect_group.create_array('scores', data=scores, chunks=(1000,))
    detect_group.create_array('n_detections', data=frame_counts, chunks=(10000,))
    detect_group.create_array('frame_counts', data=frame_counts, chunks=(10000,))
    
    # Calculate statistics
    total_detections = len(frame_indices)
    frames_with_detections = np.sum(frame_counts > 0)
    coverage_percent = (frames_with_detections / n_frames) * 100
    
    stats = {
        'total_detections': int(total_detections),
        'frames_with_detections': int(frames_with_detections),
        'percent_frames_with_detections': float(coverage_percent),
        'frames_with_zero_detections': int(np.sum(frame_counts == 0)),
        'frames_with_multiple_detections': int(np.sum(frame_counts > 1)),
        'mean_detections_per_frame': float(total_detections / n_frames),
        'mean_confidence': float(np.mean(scores)) if len(scores) > 0 else 0.0,
        'min_confidence': float(np.min(scores)) if len(scores) > 0 else 0.0,
        'max_confidence': float(np.max(scores)) if len(scores) > 0 else 0.0,
    }
    
    # Store metadata
    detect_group.attrs.update({
        'detect_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'detection_method': 'yolo',  # 'blob' for traditional, 'yolo' for neural net
        'detection_source': 'video_file',  # vs 'zarr_import' for traditional
        'model_type': 'yolo_object_detection',
        'model_path': str(model_path.absolute()),
        'model_name': model_path.name,
        'parameters': {
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'max_det': max_det,
            'batch_size': batch_size,
            'resize_dims': resize_dims,
        },
        'summary_statistics': stats,
        'git_commit': git_info.get('commit_hash', 'unknown'),
        'git_branch': git_info.get('branch', 'unknown'),
        'hostname': env_info['platform']['hostname']
    })
    
    # Mark as latest
    root['detect_runs'].attrs['latest'] = run_name
    
    console.print(f"[green]✓[/green] Detections saved")
    
    # Calculate storage savings
    zarr_size_mb = (total_detections * 32) / 1024 / 1024  # Rough estimate
    video_size_mb = (n_frames * width * height) / 1024 / 1024  # If we stored grayscale
    
    # Print summary
    summary_text = f"""[green]✓[/green] Inference complete!

[bold]Results:[/bold]
  Detections: {total_detections:,}
  Coverage: {coverage_percent:.1f}% ({frames_with_detections:,}/{n_frames:,} frames)
  Mean confidence: {stats['mean_confidence']:.3f}

[bold]Storage:[/bold]
  Zarr size: ~{zarr_size_mb:.1f} MB (detections only)
  Saved vs full import: ~{video_size_mb:.1f} MB ({video_size_mb/zarr_size_mb:.0f}× smaller)

[bold]Output:[/bold]
  {output_zarr}
  
[bold]Next steps:[/bold]
  # Refine detections
  python -m fisheye.refinement.refine_detect {output_zarr}
  
  # Assign IDs
  python -m fisheye.tracking.assign_ids {output_zarr}
  
  # Visualize
  python -m fisheye.visualization.detection_visualizer {output_zarr}
"""
    
    panel = Panel(
        summary_text,
        title="[bold green]Detection Summary[/bold green]",
        border_style="green"
    )
    console.print("\n")
    console.print(panel)
    
    return run_name


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on video without importing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (model path in config)
  python -m fisheye.detection.detect_yolo video.mp4
  python -m fisheye.detection.detect_yolo video.mp4 --output test.zarr
  
  # Explicit model and output
  python -m fisheye.detection.detect_yolo video.mp4 --model model.pt --output test.zarr
  
  # With custom thresholds (overrides config)
  python -m fisheye.detection.detect_yolo video.mp4 --conf 0.35 --batch-size 64
  
  # Force CPU
  python -m fisheye.detection.detect_yolo video.mp4 --cpu
  
  # Then run downstream analysis
  python -m fisheye.refinement.refine_detect output.zarr
  python -m fisheye.tracking.assign_ids output.zarr
        """
    )
    
    parser.add_argument('video_path', help='Input video file')
    parser.add_argument('--model', '--model-path', dest='model_path', default=None,
                       help='Trained YOLO model (.pt) - optional if in config')
    parser.add_argument('--output', '--output-zarr', dest='output_zarr', default=None,
                       help='Output zarr path - optional, auto-generated if not provided')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    parser.add_argument('--conf', type=float, default=None, 
                       help='Confidence threshold (overrides config)')
    parser.add_argument('--iou', type=float, default=None, 
                       help='IoU threshold for NMS (overrides config)')
    parser.add_argument('--max-det', type=int, default=None, 
                       help='Max detections per frame (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None, 
                       help='Inference batch size (overrides config)')
    parser.add_argument('--cpu', action='store_true', 
                       help='Force CPU inference')
    
    args = parser.parse_args()
    
    try:
        detect_yolo(
            video_path=args.video_path,
            model_path=args.model_path,
            output_zarr=args.output_zarr,
            config_path=args.config,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_det=args.max_det,
            batch_size=args.batch_size,
            use_gpu=not args.cpu if args.cpu else None
        )
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
