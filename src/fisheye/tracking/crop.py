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
import sys
from queue import Queue
from zarr.codecs import BloscCodec
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn

# Metadata helpers
from ..utils.metadata import has_raw_video, get_video_source_path, get_total_frames, get_detection_method

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"

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

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import decord
    from decord import VideoReader, cpu, gpu
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False

import cv2  # For CPU fallback in gpu decoding

from ..utils.system import get_environment_info


def check_gpu_crop_available() -> Tuple[bool, str]:
    """
    Check if GPU-accelerated cropping is available.
    
    Returns:
        Tuple of (available, reason)
    """
    if not _TORCH_AVAILABLE:
        return False, "PyTorch not installed"
    
    if not _DECORD_AVAILABLE:
        return False, "Decord not installed"
    
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    return True, "GPU cropping available"

def get_video_source(root: zarr.Group, console: Console) -> Tuple[str, Optional[str]]:
    """
    Determine video source - zarr or external file.
    
    Args:
        root: Zarr root group
        console: Rich console for output
    
    Returns:
        Tuple of (source_type, video_path)
        - source_type: 'zarr' or 'external'
        - video_path: Path to video file (None if zarr)
    """
    from ..utils.metadata import has_raw_video, get_video_source_path
    
    # Try zarr first (preferred for speed)
    if has_raw_video(root):
        console.print("[green]✓[/green] Video source: zarr (raw_video)")
        return 'zarr', None
    
    # Try external video
    video_path = get_video_source_path(root)
    if video_path:
        video_path = Path(video_path)
        if video_path.exists():
            console.print(f"[cyan]✓[/cyan] Video source: external file")
            console.print(f"[dim]  Path: {video_path}[/dim]")
            return 'external', str(video_path)
        else:
            console.print(f"[yellow]⚠[/yellow] Video path in metadata not found: {video_path}")
    
    # No video source found
    raise ValueError(
        "No video source found. Need either:\n"
        "  1. raw_video in zarr (run import stage), OR\n"
        "  2. source_video_path in metadata pointing to valid video file"
    )


def create_crop_batches(
    frame_indices: np.ndarray,
    max_frames_per_batch: int = 32
) -> List[np.ndarray]:
    """
    Group detections into batches where frames are close together.
    
    Args:
        frame_indices: Frame index for each detection
        max_frames_per_batch: Maximum unique frames to decode at once
    
    Returns:
        List of detection index arrays, one per batch
    """
    # Sort detections by frame
    sorted_idx = np.argsort(frame_indices)
    
    batches = []
    current_batch = []
    unique_frames_in_batch = set()
    
    for det_idx in sorted_idx:
        frame = frame_indices[det_idx]
        
        # Start new batch if too many unique frames
        if len(unique_frames_in_batch) >= max_frames_per_batch:
            batches.append(np.array(current_batch))
            current_batch = []
            unique_frames_in_batch = set()
        
        current_batch.append(det_idx)
        unique_frames_in_batch.add(frame)
    
    if current_batch:
        batches.append(np.array(current_batch))
    
    return batches


def crop_batch_gpu(
    video_reader: Any,
    frame_indices: np.ndarray,
    bbox_coords: np.ndarray,
    roi_sz: Tuple[int, int],
    video_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop a batch of detections using GPU acceleration.
    
    Args:
        video_reader: Decord VideoReader with GPU context
        frame_indices: Frame index for each detection in batch
        bbox_coords: Normalized bbox coordinates [N, 4]
        roi_sz: (height, width) of ROI
        video_shape: (height, width) of video
    
    Returns:
        Tuple of (crops, coordinates)
    """
    # Get unique frames and create mapping
    unique_frames = np.unique(frame_indices)
    frame_to_idx = {int(f): i for i, f in enumerate(unique_frames)}
    
    console.print(
        f"[debug] crop_batch_gpu: decoding {len(unique_frames)} frames starting at {int(unique_frames[0]) if len(unique_frames) else 'N/A'}"
    )
    decode_start = time.perf_counter()
    # Decode frames on GPU
    frames_gpu = video_reader.get_batch(unique_frames.tolist())  # [N, H, W, C]
    decode_time = time.perf_counter() - decode_start
    console.print(
        f"[debug] crop_batch_gpu: decode completed in {decode_time*1000:.1f} ms"
    )
    
    # Convert to grayscale on GPU
    frames_gray = frames_gpu.to(torch.float32).mean(dim=-1).to(torch.uint8)  # [N, H, W]
    
    # Prepare outputs
    num_crops = len(frame_indices)
    roi_h, roi_w = roi_sz
    H, W = video_shape
    
    crops_gpu = torch.zeros((num_crops, roi_h, roi_w), dtype=torch.uint8, device='cuda')
    coords_cpu = np.zeros((num_crops, 2), dtype=np.int32)
    
    # Extract each crop
    for i, (frame_idx, bbox) in enumerate(zip(frame_indices, bbox_coords)):
        frame_batch_idx = frame_to_idx[int(frame_idx)]
        frame_tensor = frames_gray[frame_batch_idx]
        
        # Calculate crop coordinates
        cx_norm, cy_norm = bbox[:2]
        cx_px = int(cx_norm * W)
        cy_px = int(cy_norm * H)
        
        x1 = cx_px - roi_w // 2
        y1 = cy_px - roi_h // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        
        coords_cpu[i] = (x1, y1)
        
        # Extract with bounds checking
        if 0 <= x1 and x2 <= W and 0 <= y1 and y2 <= H:
            crops_gpu[i] = frame_tensor[y1:y2, x1:x2]
        else:
            vy1 = max(0, y1); vy2 = min(H, y2)
            vx1 = max(0, x1); vx2 = min(W, x2)
            
            if vy2 > vy1 and vx2 > vx1:
                py1 = max(0, -y1)
                px1 = max(0, -x1)
                py2 = py1 + (vy2 - vy1)
                px2 = px1 + (vx2 - vx1)
                
                crops_gpu[i, py1:py2, px1:px2] = frame_tensor[vy1:vy2, vx1:vx2]
    
    # Copy to CPU
    crops_cpu = crops_gpu.cpu().numpy()
    
    # Cleanup
    del frames_gpu, frames_gray, crops_gpu
    torch.cuda.empty_cache()
    
    return crops_cpu, coords_cpu


def crop_batch_cpu(
    video_path: str,
    frame_indices: np.ndarray,
    bbox_coords: np.ndarray,
    roi_sz: Tuple[int, int],
    video_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop a batch of detections using CPU (OpenCV).
    """
    cap = cv2.VideoCapture(str(video_path))
    
    # Get unique frames and decode
    unique_frames = np.unique(frame_indices)
    frame_cache = {}
    
    for frame_idx in unique_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_cache[int(frame_idx)] = gray
    
    cap.release()
    
    # Extract crops
    num_crops = len(frame_indices)
    roi_h, roi_w = roi_sz
    H, W = video_shape
    
    crops = np.zeros((num_crops, roi_h, roi_w), dtype=np.uint8)
    coords = np.zeros((num_crops, 2), dtype=np.int32)
    
    for i, (frame_idx, bbox) in enumerate(zip(frame_indices, bbox_coords)):
        frame = frame_cache.get(int(frame_idx))
        if frame is None:
            continue
        
        cx_norm, cy_norm = bbox[:2]
        cx_px = int(cx_norm * W)
        cy_px = int(cy_norm * H)
        
        x1 = cx_px - roi_w // 2
        y1 = cy_px - roi_h // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        
        coords[i] = (x1, y1)
        
        if 0 <= x1 and x2 <= W and 0 <= y1 and y2 <= H:
            crops[i] = frame[y1:y2, x1:x2]
        else:
            vy1 = max(0, y1); vy2 = min(H, y2)
            vx1 = max(0, x1); vx2 = min(W, x2)
            
            if vy2 > vy1 and vx2 > vx1:
                py1 = max(0, -y1)
                px1 = max(0, -x1)
                py2 = py1 + (vy2 - vy1)
                px2 = px1 + (vx2 - vx1)
                
                crops[i, py1:py2, px1:px2] = frame[vy1:vy2, vx1:vx2]
    
    return crops, coords


def _process_chunk_gpu(
    chunk_idx: int,
    chunk_frames: List[int],
    frames_gpu: "torch.Tensor",
    frame_to_det: Dict[int, List[int]],
    bbox_coords: np.ndarray,
    roi_sz: Tuple[int, int],
    video_shape: Tuple[int, int]
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    """Process a contiguous chunk of frames on GPU and return crops/co-ordinates."""
    start = time.perf_counter()
    frames_gray = frames_gpu.to(torch.float32).mean(dim=-1).to(torch.uint8)
    H, W = video_shape

    chunk_det_indices: List[int] = []
    chunk_coords_full: List[Tuple[int, int]] = []
    chunk_rois: List[torch.Tensor] = []

    for local_idx, frame_idx in enumerate(chunk_frames):
        det_list = frame_to_det.get(int(frame_idx))
        if not det_list:
            continue

        frame_tensor = frames_gray[local_idx]
        for det_idx in det_list:
            bbox = bbox_coords[det_idx]
            cx_norm, cy_norm = bbox[:2]
            cx_px = int(cx_norm * W)
            cy_px = int(cy_norm * H)

            x1 = cx_px - roi_sz[1] // 2
            y1 = cy_px - roi_sz[0] // 2
            x2 = x1 + roi_sz[1]
            y2 = y1 + roi_sz[0]

            chunk_det_indices.append(det_idx)
            chunk_coords_full.append((x1, y1))

            if 0 <= x1 and x2 <= W and 0 <= y1 and y2 <= H:
                roi_tensor = frame_tensor[y1:y2, x1:x2].clone()
            else:
                roi_tensor = torch.zeros((roi_sz[0], roi_sz[1]), dtype=torch.uint8, device=frame_tensor.device)
                vy1 = max(0, y1); vy2 = min(H, y2)
                vx1 = max(0, x1); vx2 = min(W, x2)
                if vy2 > vy1 and vx2 > vx1:
                    py1 = max(0, -y1); py2 = py1 + (vy2 - vy1)
                    px1 = max(0, -x1); px2 = px1 + (vx2 - vx1)
                    roi_tensor[py1:py2, px1:px2] = frame_tensor[vy1:vy2, vx1:vx2]

            chunk_rois.append(roi_tensor)

    chunk_time = time.perf_counter() - start

    if not chunk_rois:
        return (
            chunk_idx,
            np.array([], dtype=np.int64),
            np.empty((0, roi_sz[0], roi_sz[1]), dtype=np.uint8),
            np.empty((0, 2), dtype=np.int32),
            chunk_time
        )

    crops_gpu = torch.stack(chunk_rois, dim=0)
    crops_cpu = crops_gpu.cpu().numpy()
    coords_full_cpu = np.array(chunk_coords_full, dtype=np.int32)
    det_indices_np = np.array(chunk_det_indices, dtype=np.int64)

    return chunk_idx, det_indices_np, crops_cpu, coords_full_cpu, chunk_time


def crop_from_external_video(
    zarr_path: str,
    video_path: str,
    source_path: str,
    source_group: zarr.Group,
    detection_source: Optional[np.ndarray],
    roi_sz: Tuple[int, int],
    use_gpu: bool,
    console: Console,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Crop detections from external video file (GPU or CPU).
    """
    start_time = time.perf_counter()
    
    root = zarr.open(zarr_path, mode='a')
    
    video_reader = None
    crop_group: Optional[zarr.Group] = None
    run_name: Optional[str] = None
    success = False
    error_message: Optional[str] = None
    started_at = datetime.now(timezone.utc).isoformat()
    
    actual_use_gpu = use_gpu
    try:
        # Load detection data
        frame_indices = source_group['frame_indices'][:]
        bbox_coords = source_group['bbox_norm_coords'][:]
        total_detections = frame_indices.shape[0]
        
        if total_detections == 0:
            console.print("[yellow]No detections to crop[/yellow]")
            return {'total_crops': 0}
        
        # Get video dimensions from metadata
        video_height = root.attrs['height']
        video_width = root.attrs['width']
        video_shape = (video_height, video_width)
        num_frames = root.attrs['total_frames']
        
        console.print(f"Total detections: {total_detections:,}")
        console.print(f"Video dimensions: {video_width}x{video_height}")
        console.print(f"ROI size: {roi_sz[1]}x{roi_sz[0]}")
        
        # Create crop group
        from ..shared.zarr.schema import get_run_group
        crop_group, run_name = get_run_group(root, 'crop', console)
        
        # Create output arrays
        compressor = BloscCodec(typesize=1, cname='lz4', clevel=1, shuffle="bitshuffle")
        
        crop_group.create_array(
            'roi_images',
            shape=(total_detections, *roi_sz),
            chunks=(min(1024, total_detections), roi_sz[0], roi_sz[1]),
            dtype='uint8',
            overwrite=True,
            compressors=compressor
        )
        
        crop_group.create_array(
            'roi_coordinates_full',
            shape=(total_detections, 2),
            chunks=(min(2048, total_detections), 2),
            dtype='i4',
            overwrite=True
        )
        
        # Get environment info for complete metadata
        env_info = get_environment_info(
            include_all_packages=False,
            disk_path=str(zarr_path),
            collect_ip=False
        )

        # Store comprehensive metadata following unified spec
        src_type_label = source_path.split('/')[-1] if '/' in source_path else 'detect'

        crop_group.attrs.update({
            # === Core Identifiers ===
            'created_at_utc': started_at,
            'started_at_utc': started_at,
            'stage': 'crop',
            'pipeline_type': 'fisheye_tracking',
            'status': 'running',
            
            # === Video Source ===
            'video_source_type': 'external',
            'video_source_path': str(video_path),
            'source_video_path': root.attrs.get('source_video_path', str(video_path)),
            'total_frames': num_frames,
            'width': video_width,
            'height': video_height,
            
            # === Detection Source ===
            'detection_source_type': src_type_label,
            'detection_source_path': source_path,
            'detection_method': get_detection_method(source_group),
            'total_detections': total_detections,
            
            # === Crop Configuration ===
            'roi_size': list(roi_sz),
            'parameter_source': 'config',  # External video uses config params
            
            # === Acceleration ===
            'acceleration': 'gpu' if actual_use_gpu else 'cpu',
            'device': 'cuda:0' if actual_use_gpu else 'cpu',
            
            # === Git Provenance ===
            'git_commit_hash': env_info['git'].get('commit_hash', 'unknown'),
            'git_short_hash': env_info['git'].get('short_hash', 'unknown'),
            'git_branch': env_info['git'].get('branch', 'unknown'),
            'git_is_dirty': env_info['git'].get('is_dirty', False),
            'git_remote_url': env_info['git'].get('remote_url', 'unknown'),
            
            # === Platform Info ===
            'hostname': env_info['platform']['hostname'],
            'system_os': env_info['platform']['system'],
            'system_machine': env_info['platform']['machine'],
            'python_version': env_info['platform']['python_version'],
            'cpu_cores': env_info['platform']['cpu_cores'],
            
            # === GPU Info ===
            'gpu_available': env_info['gpu']['available'],
            
            # === Environment ===
            'environment_type': env_info['environment']['environment_type'],
            'environment_name': env_info['environment']['environment_name'],
            
            # === Execution ===
            'command': ' '.join(sys.argv),
        })

        # Add detailed GPU info if using GPU
        if actual_use_gpu and env_info['gpu']['available'] and env_info['gpu'].get('devices'):
            primary_gpu = env_info['gpu']['devices'][0]
            crop_group.attrs.update({
                'gpu_name': primary_gpu.get('name', 'unknown'),
                'gpu_memory_total_mb': primary_gpu.get('memory_total_mb', 0),
                'gpu_compute_capability': primary_gpu.get('compute_capability', 'unknown'),
                'gpu_driver_version': env_info['gpu'].get('driver_version', 'unknown'),
                'cuda_version': env_info['gpu'].get('cuda_version', 'unknown'),
            })
            
        # Copy source metadata
        save_crop_metadata(
            crop_group=crop_group,
            source_group=source_group,
            source_path=source_path,
            source_type=src_type_label,
            detection_source=detection_source,
            total_detections=total_detections,
            num_frames=num_frames
        )
        
        # Initialize video reader
        if actual_use_gpu:
            console.print("[green]Initializing GPU video decoder...[/green]")
            try:
                decord.bridge.set_bridge('torch')
                video_reader = VideoReader(str(video_path), ctx=gpu(0))
                console.print(f"[green]✓[/green] GPU decoder ready")
            except Exception as gpu_exc:
                console.print(f"[yellow]GPU decoder failed ({gpu_exc}); falling back to CPU[/yellow]")
                actual_use_gpu = False
                crop_group.attrs['acceleration'] = 'cpu'
                crop_group.attrs['device'] = 'cpu'
                if _DECORD_AVAILABLE:
                    decord.bridge.set_bridge('native')
        if not actual_use_gpu:
            console.print("[cyan]Using CPU video decoder...[/cyan]")
        
        total_processed = 0
        batch_times: List[float] = []

        if actual_use_gpu:
            chunk_size = 96
            frame_to_det: Dict[int, List[int]] = {}
            for det_idx, frame_idx in enumerate(frame_indices):
                frame_to_det.setdefault(int(frame_idx), []).append(int(det_idx))

            frame_min = int(frame_indices.min())
            frame_max = int(frame_indices.max())
            chunk_starts = list(range(frame_min, frame_max + 1, chunk_size))
            total_chunks = len(chunk_starts)

            if verbose:
                console.print(f"Processing {total_chunks} contiguous chunks... (use_gpu=True)")
                iterator = enumerate(chunk_starts)
            else:
                from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("•"),
                    TextColumn("[cyan]{task.fields[rate]:.1f} crops/s"),
                    TimeRemainingColumn(),
                    console=console
                )
                progress_task = progress.add_task("Cropping ROIs (GPU)", total=total_detections, rate=0.0)
                progress.__enter__()
                iterator = enumerate(chunk_starts)

            try:
                for chunk_idx, chunk_start in iterator:
                    chunk_end = min(chunk_start + chunk_size, frame_max + 1)
                    chunk_frames = list(range(chunk_start, chunk_end))
                    if not any(frame in frame_to_det for frame in chunk_frames):
                        continue

                    if verbose:
                        console.print(
                            f"[debug] Chunk {chunk_idx + 1}/{total_chunks}: frames {chunk_frames[0]}-{chunk_frames[-1]} ({len(chunk_frames)} frames)"
                        )

                    chunk_start_time = time.perf_counter()
                    frames_gpu = video_reader.get_batch(chunk_frames)
                    _, det_ids, crops_cpu, coords_full_cpu, chunk_time = _process_chunk_gpu(
                        chunk_idx,
                        chunk_frames,
                        frames_gpu,
                        frame_to_det,
                        bbox_coords,
                        roi_sz,
                        (video_height, video_width)
                    )
                    del frames_gpu
                    batch_times.append(chunk_time)

                    processed = int(det_ids.size)
                    if processed:
                        crop_group['roi_images'][det_ids] = crops_cpu
                        crop_group['roi_coordinates_full'][det_ids] = coords_full_cpu
                        total_processed += processed

                    if verbose:
                        console.print(
                            f"[debug] <- Chunk {chunk_idx + 1}/{total_chunks}: {processed} detections in {chunk_time*1000:.1f} ms"
                        )
                    else:
                        elapsed_total = time.perf_counter() - start_time
                        rate = total_processed / elapsed_total if elapsed_total > 0 else 0
                        progress.update(progress_task, advance=processed, rate=rate)
            finally:
                if not verbose:
                    progress.__exit__(None, None, None)

        else:
            max_frames = 32
            batches = create_crop_batches(frame_indices, max_frames_per_batch=max_frames)
            console.print(f"Processing {len(batches)} batches... (use_gpu=False)")
            total_batches = len(batches)
            progress_every = max(1, total_batches // 20)

            for batch_idx, det_indices in enumerate(batches):
                batch_start = time.perf_counter()

                batch_frames = frame_indices[det_indices]
                batch_bboxes = bbox_coords[det_indices]

                crops, coords = crop_batch_cpu(
                    video_path, batch_frames, batch_bboxes,
                    roi_sz, video_shape
                )

                crop_group['roi_images'][det_indices] = crops
                crop_group['roi_coordinates_full'][det_indices] = coords

                batch_time = time.perf_counter() - batch_start
                batch_times.append(batch_time)
                total_processed += len(det_indices)

                if verbose:
                    console.print(
                        f"[debug] Batch {batch_idx + 1}/{total_batches} completed in {batch_time*1000:.1f} ms"
                    )
                else:
                    if (batch_idx + 1) % progress_every == 0 or batch_idx == total_batches - 1:
                        console.print(
                            f"[cyan]Batch {batch_idx + 1}/{total_batches}: {total_processed} detections processed[/cyan]"
                        )

        duration = time.perf_counter() - start_time

        # Add summary statistics
        console.print("[dim]Calculating summary statistics...[/dim]")
        if verbose:
            console.print("[debug] Reading frame_counts for summary computation")
        frames_with_crops = int(np.sum(crop_group['frame_counts'][:] > 0))
        percent_cropped = (frames_with_crops / num_frames) * 100 if num_frames > 0 else 0

        crop_group.attrs['summary_statistics'] = {
            'total_frames': num_frames,
            'frames_with_crops': frames_with_crops,
            'total_rois_cropped': total_detections,
            'percent_frames_with_crops': round(percent_cropped, 2),
            'roi_size': list(roi_sz)
        }
        crop_group.attrs['duration_seconds'] = duration
        crop_group.attrs['avg_batch_ms'] = float(np.mean(batch_times)) if batch_times else 0.0

        if verbose:
            avg_batch = (np.mean(batch_times) * 1000) if batch_times else 0.0
            console.print(
                f"[debug] Completed {total_processed} crops in {duration:.2f}s "
                f"({total_processed/duration if duration > 0 else 0:.1f} crops/s). "
                f"Average batch time: {avg_batch:.1f} ms"
            )
            if actual_use_gpu and _TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    mem_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
                    mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                    console.print(
                        f"[debug] GPU memory - allocated: {mem_alloc:.1f} MB, reserved: {mem_reserved:.1f} MB"
                    )
                except Exception:
                    pass

        console.print(f"[green]✓[/green] Cropping complete")
        console.print(f"[cyan]  Time: {duration:.1f}s[/cyan]")
        console.print(f"[cyan]  Rate: {total_detections/duration:.1f} crops/s[/cyan]")
        console.print("[dim]Finalizing crop metadata...[/dim]")

        success = True
        return {
            'total_crops': total_detections,
            'frames_with_crops': frames_with_crops,
            'percent_cropped': percent_cropped,
            'duration_seconds': duration,
            'detection_source_type': src_type_label
        }
    except KeyboardInterrupt:
        error_message = "Interrupted by user"
        console.print("[red]Cropping interrupted by user[/red]")
        raise
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if actual_use_gpu:
            # Dereference the video_reader to allow the garbage collector
            # to release the underlying CUDA resources.
            video_reader = None
            
            # Clear PyTorch's cache of unused memory
            if _TORCH_AVAILABLE:
                if verbose:
                    console.print("[debug] Clearing CUDA cache...")
                torch.cuda.empty_cache()
                if verbose:
                    console.print("[debug] CUDA cache cleared")

        if crop_group is not None:
            timestamp = datetime.now(timezone.utc).isoformat()
            if success:
                crop_group.attrs['status'] = 'completed'
                crop_group.attrs['completed_at_utc'] = timestamp
                crop_group.attrs.pop('error_message', None)
                crop_group.attrs.pop('failed_at_utc', None)
                if verbose:
                    console.print("[debug] Crop run marked as completed")
            else:
                crop_group.attrs['status'] = 'failed'
                crop_group.attrs['failed_at_utc'] = timestamp
                if error_message:
                    crop_group.attrs['error_message'] = error_message
                if verbose:
                    console.print("[debug] Crop run marked as failed")

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
        - source_path: Path string like 'detect_runs/latest' or 'refined_detect_runs/latest/filtered'
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
        if REFINED_DETECT_GROUP in root:
            refined_root = root[REFINED_DETECT_GROUP]
        elif LEGACY_REFINED_DETECT_GROUP in root:
            refined_root = root[LEGACY_REFINED_DETECT_GROUP]
        else:
            raise ValueError("No refined detection runs found. Run refinement pipeline first.")

        latest_refined = refined_root.attrs.get('latest')
        if latest_refined is None:
            raise ValueError("No latest refined detection run found")

        refined_group = refined_root[latest_refined]

        if source_type not in refined_group:
            raise ValueError(f"Stage '{source_type}' not found in refined detection run {latest_refined}")

        source_group = refined_group[source_type]
        source_path = f"{refined_root.path}/{latest_refined}/{source_type}"
        
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
    crop_params.setdefault('acceleration', 'auto')
    crop_params.setdefault('gpu_min_detections', 200)
    
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

def save_crop_metadata(
    crop_group: zarr.Group,
    source_group: zarr.Group,
    source_path: str,
    source_type: str,
    detection_source: Optional[np.ndarray],
    total_detections: int,
    num_frames: int
) -> None:
    """
    Copy coordinates and metadata from source to make crops self-contained.
    
    Args:
        crop_group: Target crop group
        source_group: Source detection group
        source_path: Path to source (for provenance)
        source_type: 'detect', 'filtered', or 'interpolated'
        detection_source: Array indicating real (0) vs interpolated (1), or None
        total_detections: Total number of detections
        num_frames: Total number of frames in video  # ADD THIS
    """
    # Copy bbox coordinates
    bbox_coords = source_group['bbox_norm_coords'][:]
    crop_group.create_array(
        'bbox_norm_coords',
        chunks=(min(1000, len(bbox_coords)), 4),
        data=bbox_coords,
        overwrite=True
    )
    
    # Copy frame_indices directly
    frame_indices = source_group['frame_indices'][:]
    crop_group.create_array(
        'frame_indices',
        chunks=(min(1000, len(frame_indices)),),
        data=frame_indices,
        overwrite=True
    )
    
    # Compute and store frame_counts for visualization
    frame_counts = np.bincount(frame_indices, minlength=num_frames)
    crop_group.create_array(
        'frame_counts',
        chunks=(min(10000, num_frames),),
        data=frame_counts,
        overwrite=True
    )
    
    # Copy detection_source if it exists (interpolated data)
    if detection_source is not None:
        crop_group.create_array(
            'detection_source',
            chunks=(min(1000, len(detection_source)),),
            data=detection_source,
            overwrite=True
        )
        
        n_real = int(np.sum(detection_source == 0))
        n_interp = int(np.sum(detection_source == 1))
        
        crop_group.attrs['includes_interpolated'] = True
        crop_group.attrs['n_real_detections'] = n_real
        crop_group.attrs['n_interpolated_detections'] = n_interp
    else:
        crop_group.attrs['includes_interpolated'] = False
        crop_group.attrs['n_real_detections'] = total_detections
        crop_group.attrs['n_interpolated_detections'] = 0
    
    # Store provenance
    crop_group.attrs['source_coords_path'] = source_path
    crop_group.attrs['detection_source_type'] = source_type

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
        source_path: Path to detection source (e.g., 'detect_runs/latest' or 'refined_detect_runs/latest/filtered')

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
    
    # Load frame indices and bbox coordinates
    # Load only the slice of frame indices/bboxes we need
    frame_indices = source_group['frame_indices'][:]
    bbox_coords = source_group['bbox_norm_coords'][:]
    chunk_frames = np.arange(chunk_slice.start, chunk_slice.stop)
    
    # Determine which detections fall into this chunk
    mask = np.isin(frame_indices, chunk_frames)
    detection_indices = np.where(mask)[0]
    
    bbox_coords_chunk = bbox_coords[detection_indices]
    frames_for_detections = frame_indices[detection_indices]
    
    # Build n_per_frame for this chunk
    n_per_frame = np.zeros(len(chunk_frames), dtype=int)
    for i, frame in enumerate(chunk_frames):
        n_per_frame[i] = np.sum(frames_for_detections == frame)

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
    console: Optional[Console] = None,
    acceleration: Optional[str] = None,
    use_gpu_allowed: bool = True,
    force_cpu: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Main function to crop ROIs from full-resolution frames based on detections.
    
    Supports both:
    - Zarr video (traditional workflow)
    - External video files (YOLO workflow with GPU/CPU acceleration)
    
    Args:
        zarr_path: Path to zarr file
        config: Configuration dictionary
        source_type: Detection source - 'detect', 'filtered', or 'interpolated'
        scheduler: Dask scheduler ('processes', 'threads', or 'distributed')
        num_workers: Number of workers (None = auto)
        console: Optional Rich console for output

        acceleration: 'auto', 'gpu', or 'cpu' to override external video mode selection
        use_gpu_allowed: Whether GPU usage is permitted globally
        force_cpu: Force CPU processing regardless of availability
        verbose: Enable additional logging and disable progress bars
    
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

    # Get crop parameters
    crop_params, param_source = get_crop_parameters(root, config, console)
    roi_sz = tuple(crop_params.get('roi_sz', [256, 256]))
    
    # Determine video source
    video_source_type, video_path = get_video_source(root, console)
    
    # Route to appropriate implementation
    if video_source_type == 'external':
        total_detections = int(source_group['frame_indices'].shape[0])
        accel_choice = (acceleration or crop_params.get('acceleration', 'auto') or 'auto').lower()
        gpu_threshold = int(crop_params.get('gpu_min_detections', 200))
        valid_accels = {'auto', 'gpu', 'cpu'}
        if accel_choice not in valid_accels:
            console.print(f"[yellow]Unknown crop acceleration '{accel_choice}', defaulting to auto[/yellow]")
            accel_choice = 'auto'

        use_gpu = False
        decision_note = ""
        gpu_available = False
        gpu_reason = "GPU disabled"

        if force_cpu:
            decision_note = "forced CPU via --force-cpu"
        elif not use_gpu_allowed and accel_choice != 'gpu':
            decision_note = "GPU disabled via --no-gpu flag"
        elif accel_choice == 'cpu':
            decision_note = "crop acceleration configured as CPU"
        else:
            gpu_available, gpu_reason = check_gpu_crop_available()
            if not gpu_available:
                decision_note = f"GPU unavailable ({gpu_reason})"
            elif not use_gpu_allowed:
                decision_note = "GPU globally disabled"
            elif accel_choice == 'gpu':
                use_gpu = True
                decision_note = "crop acceleration configured as GPU"
            else:  # auto
                use_gpu = total_detections >= gpu_threshold
                if use_gpu:
                    decision_note = f"auto mode threshold met ({total_detections:,} >= {gpu_threshold})"
                else:
                    decision_note = f"auto mode threshold not met ({total_detections:,} < {gpu_threshold})"

        if use_gpu:
            console.print(f"[green]✓ Using GPU-accelerated cropping[/green]")
            console.print(f"[dim]  {decision_note}[/dim]")
            console.print(f"[dim]  Device: {torch.cuda.get_device_name(0)}[/dim]")
        else:
            console.print(f"[cyan]Using CPU ({decision_note})[/cyan]")
            if accel_choice == 'gpu' and not force_cpu and use_gpu_allowed and not gpu_available:
                console.print(f"[yellow]Warning: GPU requested but unavailable ({gpu_reason})[/yellow]")
        
        # Use external video cropping
        return crop_from_external_video(
            zarr_path, video_path, source_path, source_group,
            detection_source, roi_sz, use_gpu, console,
            verbose=verbose
        )
    
    # Otherwise, use zarr-based cropping
    
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

    num_images = get_total_frames(root, source_group)
    if num_images is None:
        # Fallback to video shape if helper can't determine
        if 'raw_video/images_ds' in root:
            num_images = root['raw_video/images_ds'].shape[0]
        else:
            raise ValueError("Cannot determine total frames")

    # Get detection info using frame_indices (BEFORE metadata collection)
    frame_indices = source_group['frame_indices'][:]
    bbox_coords = source_group['bbox_norm_coords'][:]
    total_detections = len(frame_indices)
    
    if total_detections == 0:
        console.print("[yellow]Warning: No detections found. Nothing to crop.[/yellow]")
        return {'total_crops': 0}
    
    console.print(f"Total detections to crop: {total_detections:,}")
    console.print(f"Total frames: {num_images:,}")

    # Get environment info
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False
    )

    # Determine if GPU will be used
    use_distributed = (scheduler == "distributed") and HAVE_DISTRIBUTED

    # Build comprehensive metadata following unified spec
    crop_group.attrs.update({
        # === Core Identifiers ===
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'crop',
        'pipeline_type': 'fisheye_tracking',
        
        # === Video Source ===
        'video_source_type': 'zarr',
        'video_source_path': None,
        'source_video_path': root.attrs.get('source_video_path', 'unknown'),
        'total_frames': num_images,
        'width': root.attrs.get('width', 0),
        'height': root.attrs.get('height', 0),
        
        # === Detection Source ===
        'detection_source_type': source_type,
        'detection_source_path': source_path,
        'detection_method': get_detection_method(source_group),
        'total_detections': total_detections,
        
        # === Crop Configuration ===
        'roi_size': list(roi_sz),
        'parameter_source': param_source,
        'parameters': crop_params,
        
        # === Dask Configuration ===
        'scheduler': scheduler,
        'num_workers': num_workers or os.cpu_count(),
        'use_distributed': use_distributed,
        
        # === Git Provenance ===
        'git_commit_hash': env_info['git'].get('commit_hash', 'unknown'),
        'git_short_hash': env_info['git'].get('short_hash', 'unknown'),
        'git_branch': env_info['git'].get('branch', 'unknown'),
        'git_is_dirty': env_info['git'].get('is_dirty', False),
        'git_remote_url': env_info['git'].get('remote_url', 'unknown'),
        
        # === Platform Info ===
        'hostname': env_info['platform']['hostname'],
        'system_os': env_info['platform']['system'],
        'system_machine': env_info['platform']['machine'],
        'python_version': env_info['platform']['python_version'],
        'cpu_cores': env_info['platform']['cpu_cores'],
        
        # === GPU Info (CPU workflow but still track GPU availability) ===
        'gpu_available': env_info['gpu']['available'],
        
        # === Environment ===
        'environment_type': env_info['environment']['environment_type'],
        'environment_name': env_info['environment']['environment_name'],
        
        # === Execution ===
        'command': ' '.join(sys.argv),
    })

    # Add GPU details if available (even though zarr workflow uses CPU)
    if env_info['gpu']['available'] and env_info['gpu'].get('devices'):
        primary_gpu = env_info['gpu']['devices'][0]
        crop_group.attrs.update({
            'gpu_name': primary_gpu.get('name', 'unknown'),
            'gpu_memory_total_mb': primary_gpu.get('memory_total_mb', 0),
            'gpu_compute_capability': primary_gpu.get('compute_capability', 'unknown'),
            'gpu_driver_version': env_info['gpu'].get('driver_version', 'unknown'),
            'cuda_version': env_info['gpu'].get('cuda_version', 'unknown'),
        })

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
    
    # Copy source coordinates to make crops self-contained
    console.print("[cyan]Copying source coordinates to crop group...[/cyan]")
    save_crop_metadata(
        crop_group=crop_group,
        source_group=source_group,
        source_path=source_path,
        source_type=source_type,
        detection_source=detection_source,
        total_detections=total_detections,
        num_frames=num_images
    )

    # Store path to this crop group in root for workers to find
    root.attrs['current_crop_group_path'] = crop_group.path

    # Build chunk frame slices
    chunk_slices = [slice(i, min(i + chunk_size, num_images))
                    for i in range(0, num_images, chunk_size)]
    console.print(f"Creating [yellow]{len(chunk_slices)}[/yellow] Dask tasks for cropping...")

    # Build detection counts per frame from frame_indices
    n_detections_per_frame = np.zeros(num_images, dtype='i4')
    unique_frames, counts = np.unique(frame_indices, return_counts=True)
    n_detections_per_frame[unique_frames] = counts
    
    # Precompute cumulative detection offsets ONCE on driver
    cumulative_detections = np.cumsum(np.insert(n_detections_per_frame, 0, 0))
    
    # Build chunks with output slices
    chunks = []
    for chunk_slice in chunk_slices:
        start_det = int(cumulative_detections[chunk_slice.start])
        end_det = int(cumulative_detections[chunk_slice.stop])
        
        if end_det > start_det:  # Only add if chunk has detections
            chunks.append((chunk_slice, (start_det, end_det)))
    
    frames_with_crops = int(np.sum(n_detections_per_frame > 0))

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

    # Clean up root attrs
    if 'current_crop_group_path' in root.attrs:
        del root.attrs['current_crop_group_path']

    # Create completion panel
    source_info = f"{source_type}"
    if 'includes_interpolated' in crop_group.attrs and crop_group.attrs['includes_interpolated']:
        n_real = crop_group.attrs['n_real_detections']
        n_interp = crop_group.attrs['n_interpolated_detections']
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
  - crop_runs/{run_group_name}/roi_coordinates_ds: ({total_detections}, 2)
  - crop_runs/{run_group_name}/bbox_norm_coords: ({total_detections}, 4)
  - crop_runs/{run_group_name}/frame_indices: ({total_detections},)
  - crop_runs/{run_group_name}/frame_counts: ({num_images},)"""
    
    if 'detection_source' in crop_group:
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
    parser.add_argument("--acceleration", type=str, default=None,
                       choices=['auto', 'gpu', 'cpu'],
                       help="Acceleration mode for external video cropping")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    parser.add_argument("--force-cpu", action="store_true",
                       help="Force CPU processing even if GPU available")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
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
            console=console,
            acceleration=args.acceleration,
            use_gpu_allowed=not args.no_gpu,
            force_cpu=args.force_cpu,
            verbose=args.verbose
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
