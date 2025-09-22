# src/tracker.py

import os
import zarr
import imageio.v3 as iio
import numpy as np
import cv2
from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops
import random
import time
import sys
from datetime import datetime, timezone
import platform
import socket
import skimage
import argparse
import subprocess
import decord
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import queue
import threading

# Use rich for console output
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False

# Dask imports
import dask
from dask import delayed
from dask.diagnostics import ProgressBar

# GPU acceleration imports (if available)
try:
    import cupy as cp
    GPU_AVAILABLE = True
    if __name__ == "__main__":
        console = Console()
        device_id = cp.cuda.get_device_id()
        device_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode('utf-8')
        console.print(f"[bold green]GPU acceleration available:[/bold green] {device_name}")
except ImportError:
    GPU_AVAILABLE = False
    cp = None

os.environ['OMP_NUM_THREADS'] = '1'
cv2.setNumThreads(0)

# --- Utility Functions (No changes here) ---

def get_cpu_info():
    """Gathers detailed CPU information using py-cpuinfo."""
    if not CPUINFO_AVAILABLE:
        return {"error": "py-cpuinfo not installed"}
    
    try:
        info = cpuinfo.get_cpu_info()
        return {
            'model': info.get('brand_raw', 'N/A'),
            'arch': info.get('arch_string_raw', 'N/A'),
            'hz_advertised': info.get('hz_advertised_friendly', 'N/A'),
            'l2_cache_size': info.get('l2_cache_size', 'N/A'),
        }
    except Exception as e:
        return {"error": f"Could not retrieve CPU info: {e}"}

def get_gpu_info():
    """Gets the name of the active CUDA device."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "N/A"

def get_platform_info():
    """Gathers detailed platform and CPU information."""
    info = {
        'system': platform.system(),
        'release': platform.release(),
        'machine': platform.machine(),
        'hostname': socket.gethostname(),
        'cpu_cores': os.cpu_count()
    }
    info['cpu_details'] = get_cpu_info()
    return info

def get_git_info():
    try:
        script_path = os.path.dirname(os.path.realpath(__file__))
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=script_path, stderr=subprocess.DEVNULL).strip().decode('utf-8')
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=script_path, stderr=subprocess.DEVNULL).strip().decode('utf-8')
        return {'commit_hash': commit_hash, 'is_dirty': bool(status)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit_hash': 'N/A', 'is_dirty': True, 'error': 'Not a git repository or git not found'}

def get_run_group(root, stage_name, console):
    """Creates a new, timestamped group for a pipeline stage run."""
    parent_group_name = f"{stage_name}_runs"
    parent_group = root.require_group(parent_group_name)
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    run_group_name = f"{stage_name}_{timestamp}"
    run_group = parent_group.create_group(run_group_name)
    parent_group.attrs['latest'] = run_group_name
    console.print(f"Created new run group: [cyan]{run_group.path}[/cyan]")
    return run_group

def fast_mode_bincount(arr):
    moved_axis_arr = np.moveaxis(arr, 0, -1)
    mode_map = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=moved_axis_arr)
    return mode_map.astype(np.uint8)

def triangle_calculations(p1, p2, p3):
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    angles = np.zeros(3)
    if (2 * b * c) > 0: angles[0] = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0)) * 180 / np.pi
    if (2 * a * c) > 0: angles[1] = np.arccos(np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1.0, 1.0)) * 180 / np.pi
    if (2 * a * b) > 0: angles[2] = np.arccos(np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)) * 180 / np.pi
    return angles, np.array([a, b, c])

def calculate_multi_scale_bounding_boxes(keypoint_stats, roi_sz, margin_factor=3, min_bbox_size=0.5):
    if len(keypoint_stats) < 3: return None
    all_positions = np.array([s.centroid[::-1] for s in keypoint_stats])
    min_pos, max_pos = np.min(all_positions, axis=0), np.max(all_positions, axis=0)
    center_roi_px = (min_pos + max_pos) / 2.0
    center_roi_norm = center_roi_px / np.array(roi_sz[::-1])
    keypoint_extent_px = max_pos - min_pos
    margin_px = keypoint_extent_px * (margin_factor - 1.0)
    tight_extent_px = keypoint_extent_px + margin_px
    min_size_px = np.array(roi_sz[::-1]) * min_bbox_size
    tight_extent_px = np.maximum(tight_extent_px, min_size_px)
    extent_roi_norm = tight_extent_px / np.array(roi_sz[::-1])
    extent_roi_norm = np.minimum(extent_roi_norm, [1.0, 1.0])
    return {
        'center_roi_norm': center_roi_norm, 
        'extent_roi_norm': extent_roi_norm,
        'bladder_roi_norm': all_positions[0] / np.array(roi_sz[::-1]),
        'eye_l_roi_norm': all_positions[1] / np.array(roi_sz[::-1]),
        'eye_r_roi_norm': all_positions[2] / np.array(roi_sz[::-1]),
        'keypoint_count': len(keypoint_stats)
    }

def transform_bbox_to_image_scales(bbox_data, roi_coords_full, roi_coords_ds, roi_sz,
                                 full_img_shape=(4512, 4512), ds_img_shape=(640, 640)):
    if bbox_data is None: return None
    center_roi_norm, extent_roi_norm = bbox_data['center_roi_norm'], bbox_data['extent_roi_norm']
    center_full_norm, extent_full_norm = [np.nan, np.nan], [np.nan, np.nan]
    if roi_coords_full[0] != -1:
        center_full_px = np.array(roi_coords_full) + center_roi_norm * np.array(roi_sz[::-1])
        extent_full_px = extent_roi_norm * np.array(roi_sz[::-1])
        center_full_norm = center_full_px / np.array(full_img_shape[::-1])
        extent_full_norm = extent_full_px / np.array(full_img_shape[::-1])
    center_ds_norm, extent_ds_norm = [np.nan, np.nan], [np.nan, np.nan]
    if roi_coords_ds[0] != -1:
        roi_sz_ds = np.array(roi_sz) * (ds_img_shape[0] / full_img_shape[0])
        center_ds_px = np.array(roi_coords_ds) + center_roi_norm * roi_sz_ds[::-1]
        extent_ds_px = extent_roi_norm * roi_sz_ds[::-1]
        center_ds_norm = center_ds_px / np.array(ds_img_shape[::-1])
        extent_ds_norm = extent_ds_px / np.array(ds_img_shape[::-1])
    return {
        'full_scale': {'center_norm': center_full_norm, 'extent_norm': extent_full_norm},
        'ds_scale': {'center_norm': center_ds_norm, 'extent_norm': extent_ds_norm},
        'roi_scale': {'center_norm': center_roi_norm, 'extent_norm': extent_roi_norm}
    }
def run_assign_ids_stage(zarr_path, params, console):
    """
    Assigns an ID to each detection based on pre-defined sub-dish ROIs.
    """
    console.rule("[bold]Stage 4: Assigning Detection IDs[/bold]")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'detect_runs' not in root: raise ValueError("Detect stage not run.")
    
    assign_params = params.get('assign_ids')
    if not assign_params or 'sub_dish_rois' not in assign_params:
        console.print("[yellow]Warning: No 'sub_dish_rois' found in config. Skipping ID assignment.[/yellow]")
        return

    assign_group = get_run_group(root, 'id_assignments', console)
    latest_detect_run = root['detect_runs'].attrs['latest']
    
    assign_group.attrs.update({
        'assignment_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'parameters': assign_params,
        'source_detect_run': latest_detect_run
    })

    detect_group = root[f'detect_runs/{latest_detect_run}']
    
    # --- APPLYING THE FIX TO BOTH VARIABLES ---
    # 1. Get references to the Zarr array objects
    bbox_coords_zarr = detect_group['bbox_norm_coords']
    n_detections_zarr = detect_group['n_detections']
    
    # 2. Now, load the data into NumPy arrays for calculations
    bbox_coords_numpy = bbox_coords_zarr[:]
    n_detections_numpy = n_detections_zarr[:]
    
    ds_img_shape = root['raw_video/images_ds'].shape[1:]
    rois = assign_params['sub_dish_rois']
    
    console.print(f"Assigning IDs based on [yellow]{len(rois)}[/yellow] sub-dish ROIs...")
    
    bbox_coords_numpy[:, 0] *= ds_img_shape[1] # center_x
    bbox_coords_numpy[:, 1] *= ds_img_shape[0] # center_y

    detection_ids = np.full(bbox_coords_numpy.shape[0], -1, dtype='i4')
    
    for roi in rois:
        roi_id = roi['id']
        x, y, w, h = roi['roi_pixels']
        
        in_roi_mask = (
            (bbox_coords_numpy[:, 0] >= x) & (bbox_coords_numpy[:, 0] < x + w) &
            (bbox_coords_numpy[:, 1] >= y) & (bbox_coords_numpy[:, 1] < y + h)
        )
        detection_ids[in_roi_mask] = roi_id

    # 3. Use the Zarr array objects to access the .chunks attribute
    assign_group.create_dataset('detection_ids', data=detection_ids, chunks=(bbox_coords_zarr.chunks[0],), overwrite=True)
    
    n_frames = len(n_detections_numpy)
    n_rois = len(rois)
    per_roi_counts = np.zeros((n_frames, n_rois), dtype='i4')
    
    cumulative_detections = np.cumsum(np.insert(n_detections_numpy, 0, 0))
    for i in range(n_frames):
        start_idx = cumulative_detections[i]
        end_idx = cumulative_detections[i+1]
        frame_ids = detection_ids[start_idx:end_idx]
        for roi_id in range(n_rois):
            per_roi_counts[i, roi_id] = np.sum(frame_ids == roi_id)
            
    # Use the second Zarr object's chunks attribute here
    assign_group.create_dataset('n_detections_per_roi', data=per_roi_counts, chunks=(n_detections_zarr.chunks[0], None), overwrite=True)
    
    unassigned_count = np.sum(detection_ids == -1)
    console.print(f"Assigned IDs to [green]{len(detection_ids) - unassigned_count}[/green] detections.")
    if unassigned_count > 0:
        console.print(f"[yellow]{unassigned_count}[/yellow] detections were outside any defined ROI.")
        
    assign_group.attrs['duration_seconds'] = time.perf_counter() - start_time
    console.print(f"ID Assignment stage completed in [green]{assign_group.attrs['duration_seconds']:.2f}[/green] seconds.")

# --- Stage-Specific Functions ---

def run_import_stage_parallel_io(video_path, zarr_path, params, cli_args, console):
    """
    An improved import stage that uses a separate thread for writing to disk,
    allowing GPU decoding and disk I/O to happen in parallel.
    """
    console.rule("[bold]Stage 1: Importing Video[/bold]")
    start_time = time.perf_counter()
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(video_path, ctx=decord.gpu(0))
    console.print("Using GPU context for video decoding")
    n_frames, full_height, full_width = len(vr), vr[0].shape[0], vr[0].shape[1]
    import_params = params['import']
    ds_size, chunk_size, batch_size = tuple(import_params['downsample_size']), import_params['chunk_size'], import_params['batch_size']
    gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device='cuda:0')
    root = zarr.open_group(zarr_path, mode='w')
    param_group = root.create_group('pipeline_params')
    for stage, stage_params in params.items():
        param_group.attrs[stage] = stage_params
    root.attrs.update({
        'command_line_args': cli_args, 'git_info': get_git_info(),
        'source_video_metadata': iio.immeta(video_path), 'platform_info': get_platform_info(),
        'software_versions': {
            'python': platform.python_version(), 'numpy': np.__version__, 'zarr': zarr.__version__,
            'scikit-image': skimage.__version__, 'opencv-python': cv2.__version__, 'torch': torch.__version__,
            'decord': decord.__version__
        }
    })
    raw_video_group = root.create_group('raw_video')
    compressor_details = {'cname': 'lz4', 'clevel': 1, 'shuffle': 'bit'}
    compressor = zarr.Blosc(cname=compressor_details['cname'], clevel=compressor_details['clevel'], shuffle=zarr.Blosc.BITSHUFFLE)
    raw_video_group.attrs.update({
        'import_timestamp_utc': datetime.now(timezone.utc).isoformat(), 'original_resolution': (full_height, full_width),
        'downsampled_resolution': ds_size, 'decoding_device': get_gpu_info(), 'compressor': compressor_details
    })
    optimal_chunk_size = min(64, chunk_size * 2)
    images_full = raw_video_group.create_dataset('images_full', shape=(n_frames, full_height, full_width), chunks=(optimal_chunk_size, None, None), dtype=np.uint8, compressor=compressor, write_empty_chunks=False)
    images_ds = raw_video_group.create_dataset('images_ds', shape=(n_frames, ds_size[0], ds_size[1]), chunks=(optimal_chunk_size, None, None), dtype=np.uint8, compressor=compressor, write_empty_chunks=False)
    console.print(f"Compression: [cyan]{compressor_details['cname']}[/cyan]")
    console.print(f"Chunk size: {optimal_chunk_size}")
    data_queue = queue.Queue(maxsize=4)
    def writer_task(q, zarr_full, zarr_ds):
        while True:
            item = q.get()
            if item is None: break
            start_idx, end_idx, full_data, ds_data = item
            zarr_full[start_idx:end_idx] = full_data
            zarr_ds[start_idx:end_idx] = ds_data
            q.task_done()
    writer_thread = threading.Thread(target=writer_task, args=(data_queue, images_full, images_ds), daemon=True)
    writer_thread.start()
    console.print("Writer thread started...")
    io_batch_size = batch_size * 4  
    console.print(f"Importing {n_frames} frames")
    for i in tqdm(range(0, n_frames, io_batch_size), desc="GPU Video Import"):
        io_batch_end = min(i + io_batch_size, n_frames)
        full_batch_data, ds_batch_data = [], []
        for j in range(i, io_batch_end, batch_size):
            sub_batch_end = min(j + batch_size, io_batch_end)
            indices = list(range(j, sub_batch_end))
            if not indices: continue
            batch_tensor = vr.get_batch(indices)
            gray_batch_float = torch.matmul(batch_tensor.float(), gray_weights).unsqueeze(1)
            ds_batch_float = F.interpolate(gray_batch_float, size=ds_size, mode='bilinear', align_corners=False)
            full_batch_data.append(gray_batch_float.squeeze(1).byte().cpu().numpy())
            ds_batch_data.append(ds_batch_float.squeeze(1).byte().cpu().numpy())
            del batch_tensor, gray_batch_float, ds_batch_float
        if not full_batch_data: continue
        full_combined, ds_combined = np.concatenate(full_batch_data, axis=0), np.concatenate(ds_batch_data, axis=0)
        data_queue.put((i, io_batch_end, full_combined, ds_combined))
    data_queue.put(None)
    writer_thread.join()
    console.print("Writer thread finished. All data saved to Zarr.")
    duration = time.perf_counter() - start_time
    raw_video_group.attrs['duration_seconds'] = duration
    console.print(Panel(f"Total time: [bold yellow]{duration:.1f}s[/bold yellow] ([cyan]{duration/60:.1f}[/cyan] minutes)\nOverall throughput: [bold green]{n_frames/duration:.1f} fps[/bold green]", title="Import Performance Summary", expand=False))

def run_background_stage(zarr_path, params, console):
    console.rule("[bold]Stage 2: Calculating Background[/bold]")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'raw_video' not in root: raise ValueError("Import stage not run.")
    bg_params = params['background']
    bg_group = get_run_group(root, 'background', console)
    bg_group.attrs.update({'background_timestamp_utc': datetime.now(timezone.utc).isoformat(), 'parameters': bg_params})
    num_images = root['raw_video/images_full'].shape[0]
    
    # Use -1 to signify using all frames ---
    sample_size = bg_params.get('sample_size', 100) # Default to 100 if not specified
    if sample_size == -1:
        console.print(f"Using [yellow]all {num_images}[/yellow] frames for background model.")
        random_indices = list(range(num_images))
    else:
        # Use the original random sampling method
        random.seed(bg_params.get('seed', 42))
        sample_count = min(int(sample_size), num_images)
        console.print(f"Randomly sampling [yellow]{sample_count}[/yellow] frames for background model.")
        random_indices = random.sample(range(num_images), sample_count)

    console.print("Calculating background modes (this may take a while for large samples)...")
    bg_group.create_dataset('background_full', data=fast_mode_bincount(root['raw_video/images_full'].get_orthogonal_selection((random_indices, slice(None), slice(None)))), overwrite=True)
    bg_group.create_dataset('background_ds', data=fast_mode_bincount(root['raw_video/images_ds'].get_orthogonal_selection((random_indices, slice(None), slice(None)))), overwrite=True)
    
    bg_group.attrs['source_frame_indices'] = 'all' if sample_size == -1 else random_indices
    bg_group.attrs['duration_seconds'] = time.perf_counter() - start_time
    console.print(f"Background stage completed in [green]{bg_group.attrs['duration_seconds']:.2f}[/green] seconds.")

@delayed
def detect_chunk_delayed(zarr_path, chunk_slice, detect_params, dish_mask):
    """
    Detects fish in a chunk of downsampled frames and returns their bounding boxes.
    """
    se1 = disk(detect_params['se1_radius'])
    se4 = disk(detect_params['se4_radius'])
    
    with zarr.open(zarr_path, mode='r') as root:
        images_ds_chunk = root['raw_video/images_ds'][chunk_slice]
        latest_bg_run = root['background_runs'].attrs['latest']
        background_ds = root[f'background_runs/{latest_bg_run}/background_ds'][:]
        ds_img_shape = images_ds_chunk.shape[1:]

    chunk_len = images_ds_chunk.shape[0]
    all_bbox_norms = []
    frame_detection_counts = np.zeros(chunk_len, dtype='i4')

    for i in range(chunk_len):
        diff_ds = np.clip(background_ds.astype(np.int16) - images_ds_chunk[i].astype(np.int16), 0, 255).astype(np.uint8)
        if dish_mask is not None:
            diff_ds[dish_mask == 0] = 0

        current_thresh = detect_params['ds_thresh']
        valid_blobs = []
        # Adaptive thresholding
        for _ in range(5):
            im_ds = erosion(dilation(erosion(diff_ds >= current_thresh, se1), se4), se1)
            all_blobs = regionprops(label(im_ds))
            valid_blobs = [r for r in all_blobs if detect_params['min_area'] <= r.area <= detect_params['max_area']]
            if valid_blobs:
                break
            current_thresh -= 5
        
        if not valid_blobs:
            continue

        sorted_blobs = sorted(valid_blobs, key=lambda r: r.area, reverse=True)[:detect_params['max_fish']]
        frame_detection_counts[i] = len(sorted_blobs)

        for blob in sorted_blobs:
            min_r, min_c, max_r, max_c = blob.bbox
            center_y, center_x = (min_r + max_r) / 2, (min_c + max_c) / 2
            height, width = max_r - min_r, max_c - min_c
            
            center_norm = np.array([center_x / ds_img_shape[1], center_y / ds_img_shape[0]])
            size_norm = np.array([width / ds_img_shape[1], height / ds_img_shape[0]])
            all_bbox_norms.append([*center_norm, *size_norm])
            
    return chunk_slice, frame_detection_counts, all_bbox_norms

def run_detect_stage(zarr_path, scheduler_name, params, console):
    console.rule(f"[bold]Stage 3: Detecting Fish (Dask {scheduler_name.title()} Scheduler)[/bold]")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'background_runs' not in root: 
        raise ValueError("Background stage not run.")
    
    detect_params = params['detect']
    detect_group = get_run_group(root, 'detect', console)
    latest_bg_run = root['background_runs'].attrs['latest']
    
    detect_group.attrs.update({
        'detect_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'dask_scheduler': scheduler_name,
        'parameters': detect_params,
        'source_background_run': latest_bg_run
    })
    
    # Create dish mask using the new function that supports both rectangle and circle
    background_ds = root[f'background_runs/{latest_bg_run}/background_ds'][:]
    ds_img_shape = background_ds.shape
    mask_params = detect_params.get('dish_mask', {})
    mask = create_dish_mask(mask_params, ds_img_shape, console)
    
    num_images = root['raw_video/images_ds'].shape[0]
    chunk_size = params['import']['chunk_size']
    
    # Create Zarr arrays to store detection results
    n_detections = detect_group.create_dataset('n_detections', shape=(num_images,), chunks=(chunk_size * 4,), dtype='i4', overwrite=True)
    bbox_norm_coords = detect_group.create_dataset('bbox_norm_coords', shape=(1, 4), chunks=(chunk_size * 4, None), dtype='f8', overwrite=True)

    chunk_slices = [slice(i, min(i + chunk_size, num_images)) for i in range(0, num_images, chunk_size)]
    console.print(f"Creating [yellow]{len(chunk_slices)}[/yellow] Dask tasks for detection...")
    
    delayed_tasks = [detect_chunk_delayed(zarr_path, s, detect_params, mask) for s in chunk_slices]
    
    with ProgressBar(): results = dask.compute(*delayed_tasks)
    console.print("Writing detection results to Zarr...")
    
    total_detections = 0
    for slc, counts, bboxes in tqdm(results, desc="Writing Detection Chunks"):
        n_detections[slc] = counts
        num_in_chunk = sum(counts)
        if num_in_chunk > 0:
            start_idx = total_detections
            end_idx = total_detections + num_in_chunk
            bbox_norm_coords.resize(end_idx, bbox_norm_coords.shape[1])
            bbox_norm_coords[start_idx:end_idx] = bboxes
            total_detections += num_in_chunk
            
    bbox_norm_coords.resize(total_detections, bbox_norm_coords.shape[1])

    frames_with_detections = np.sum(n_detections[:] > 0)
    percent_detected = (frames_with_detections / num_images) * 100 if num_images > 0 else 0
    
    summary_stats = {
        'total_frames': num_images, 
        'frames_with_detections': int(frames_with_detections), 
        'total_detections': int(total_detections), 
        'percent_frames_with_detections': round(percent_detected, 2)
    }
    detect_group.attrs['summary_statistics'] = summary_stats
    detect_group.attrs['duration_seconds'] = time.perf_counter() - start_time
    
    console.print(f"Detection stage completed in [green]{detect_group.attrs['duration_seconds']:.2f}[/green] seconds.")
    console.print(f"Found [green]{total_detections}[/green] total fish in [green]{frames_with_detections}/{num_images}[/green] frames ([cyan]{percent_detected:.2f}%[/cyan]).")


# MODIFIED: Dask task for the new, simpler crop stage
@delayed
def crop_chunk_from_bbox_delayed(zarr_path, chunk_slice, roi_sz):
    """
    Crops ROIs from full-resolution frames based on pre-computed bounding boxes.
    """
    with zarr.open(zarr_path, mode='r') as root:
        images_full_chunk = root['raw_video/images_full'][chunk_slice]
        full_img_shape = images_full_chunk.shape[1:]
        
        # Load detection data for this chunk
        latest_detect_run = root['detect_runs'].attrs['latest']
        n_detections_per_frame = root[f'detect_runs/{latest_detect_run}/n_detections'][chunk_slice]
        
        start_detection_idx = np.sum(root[f'detect_runs/{latest_detect_run}/n_detections'][:chunk_slice.start])
        end_detection_idx = start_detection_idx + np.sum(n_detections_per_frame)
        detection_slice = slice(start_detection_idx, end_detection_idx)
        
        bbox_coords_chunk = root[f'detect_runs/{latest_detect_run}/bbox_norm_coords'][detection_slice]

    all_rois, all_coords_full, all_coords_ds = [], [], []
    bbox_cursor = 0

    for i in range(len(images_full_chunk)):
        num_detections_in_frame = n_detections_per_frame[i]
        if num_detections_in_frame == 0:
            continue

        for _ in range(num_detections_in_frame):
            center_norm = bbox_coords_chunk[bbox_cursor][:2]
            
            full_centroid_px = np.round(center_norm * np.array(full_img_shape)[::-1]).astype(int)
            roi_x1_full, roi_y1_full = full_centroid_px[0] - roi_sz[1] // 2, full_centroid_px[1] - roi_sz[0] // 2
            roi = images_full_chunk[i][roi_y1_full:roi_y1_full+roi_sz[0], roi_x1_full:roi_x1_full+roi_sz[1]]
            
            if roi.shape != tuple(roi_sz):
                padded_roi = np.zeros(roi_sz, dtype='uint8')
                padded_roi[:roi.shape[0], :roi.shape[1]] = roi
                roi = padded_roi
            
            all_rois.append(roi)
            all_coords_full.append((roi_x1_full, roi_y1_full))
            bbox_cursor += 1

    return chunk_slice, all_rois, all_coords_full

def run_crop_stage(zarr_path, scheduler_name, params, console):
    console.rule(f"[bold]Stage 4: Cropping ROIs from Detections (Dask {scheduler_name.title()} Scheduler)[/bold]")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'detect_runs' not in root: 
        raise ValueError("Detect stage not run.")
    
    crop_params = params['crop']
    crop_group = get_run_group(root, 'crop', console)
    latest_detect_run = root['detect_runs'].attrs['latest']
    
    crop_group.attrs.update({
        'crop_timestamp_utc': datetime.now(timezone.utc).isoformat(), 
        'dask_scheduler': scheduler_name, 
        'parameters': crop_params,
        'source_detect_run': latest_detect_run
    })
    
    num_images = root['raw_video/images_ds'].shape[0]
    chunk_size = params['import']['chunk_size']
    
    # Re-use n_detections from the detect stage
    n_detections = root[f'detect_runs/{latest_detect_run}/n_detections'][:]
    total_detections = int(n_detections.sum())
    
    # Create Zarr arrays for cropped data
    roi_images = crop_group.create_dataset(
        'roi_images', 
        shape=(total_detections, *crop_params['roi_sz']), 
        chunks=(chunk_size, None, None), 
        dtype='uint8', 
        overwrite=True, 
        compressor=zarr.Blosc(cname='lz4', clevel=1, shuffle=zarr.Blosc.BITSHUFFLE)
    )
    roi_coordinates_full = crop_group.create_dataset(
        'roi_coordinates_full', 
        shape=(total_detections, 2), 
        chunks=(chunk_size * 4, None), 
        dtype='i4', 
        overwrite=True
    )
    
    chunk_slices = [slice(i, min(i + chunk_size, num_images)) for i in range(0, num_images, chunk_size)]
    console.print(f"Creating [yellow]{len(chunk_slices)}[/yellow] Dask tasks for cropping...")

    delayed_tasks = [crop_chunk_from_bbox_delayed(zarr_path, s, tuple(crop_params['roi_sz'])) for s in chunk_slices]
    
    with ProgressBar(): results = dask.compute(*delayed_tasks)
    console.print("Writing cropped ROIs to Zarr...")
    
    # To correctly place the results, we need to know the cumulative count of detections
    cumulative_detections = np.cumsum(np.insert(n_detections, 0, 0))
    
    for slc, rois, coords_full in tqdm(results, desc="Writing Crop Chunks"):
        start_idx = cumulative_detections[slc.start]
        end_idx = cumulative_detections[slc.stop]
        
        if end_idx > start_idx:
            roi_images[start_idx:end_idx] = rois
            roi_coordinates_full[start_idx:end_idx] = coords_full

    frames_with_crops = np.sum(n_detections > 0)
    percent_cropped = (frames_with_crops / num_images) * 100 if num_images > 0 else 0
    
    summary_stats = {
        'total_frames': num_images, 
        'frames_with_crops': int(frames_with_crops), 
        'total_rois_cropped': int(total_detections), 
        'percent_frames_with_crops': round(percent_cropped, 2)
    }
    crop_group.attrs['summary_statistics'] = summary_stats
    crop_group.attrs['duration_seconds'] = time.perf_counter() - start_time
    
    console.print(f"Cropping stage completed in [green]{crop_group.attrs['duration_seconds']:.2f}[/green] seconds.")
    console.print(f"  Found [green]{total_detections}[/green] cropped ROIs in [green]{frames_with_crops}/{num_images}[/green] frames ([cyan]{percent_cropped:.2f}%[/cyan]).")

def triangle_calculations(p1, p2, p3):
    """Calculate angles and area of a triangle formed by three points."""
    # Calculate distances between points
    d12 = np.linalg.norm(p2 - p1)
    d23 = np.linalg.norm(p3 - p2)
    d31 = np.linalg.norm(p1 - p3)
    
    # Calculate angles using law of cosines
    # Angle at p1
    cos_angle1 = (d12**2 + d31**2 - d23**2) / (2 * d12 * d31)
    angle1 = np.arccos(np.clip(cos_angle1, -1, 1))
    
    # Angle at p2
    cos_angle2 = (d12**2 + d23**2 - d31**2) / (2 * d12 * d23)
    angle2 = np.arccos(np.clip(cos_angle2, -1, 1))
    
    # Angle at p3
    cos_angle3 = (d23**2 + d31**2 - d12**2) / (2 * d23 * d31)
    angle3 = np.arccos(np.clip(cos_angle3, -1, 1))
    
    angles = np.array([angle1, angle2, angle3])
    
    # Calculate area using Heron's formula
    s = (d12 + d23 + d31) / 2  # semi-perimeter
    area = np.sqrt(s * (s - d12) * (s - d23) * (s - d31))
    
    return angles, area

def calculate_multi_scale_bounding_boxes(keypoint_stats, roi_sz, margin_factor=1.5, min_bbox_size=0.05):
    """
    Calculate bounding boxes from keypoints at multiple scales.
    
    Args:
        keypoint_stats: List of 3 regionprops objects [bladder, eye_l, eye_r]
        roi_sz: Tuple of ROI dimensions (height, width)
        margin_factor: Factor to expand bbox beyond keypoints
        min_bbox_size: Minimum bbox size as fraction of ROI
    
    Returns:
        Dictionary with bbox data and keypoint positions
    """
    if len(keypoint_stats) != 3:
        return None
    
    all_positions = np.array([s.centroid[::-1] for s in keypoint_stats])
    min_pos, max_pos = np.min(all_positions, axis=0), np.max(all_positions, axis=0)
    center_roi_px = (min_pos + max_pos) / 2.0
    center_roi_norm = center_roi_px / np.array(roi_sz[::-1])
    
    keypoint_extent_px = max_pos - min_pos
    margin_px = keypoint_extent_px * (margin_factor - 1.0)
    tight_extent_px = keypoint_extent_px + margin_px
    min_size_px = np.array(roi_sz[::-1]) * min_bbox_size
    tight_extent_px = np.maximum(tight_extent_px, min_size_px)
    extent_roi_norm = tight_extent_px / np.array(roi_sz[::-1])
    extent_roi_norm = np.minimum(extent_roi_norm, [1.0, 1.0])
    
    return {
        'center_roi_norm': center_roi_norm, 
        'extent_roi_norm': extent_roi_norm,
        'bladder_roi_norm': all_positions[0] / np.array(roi_sz[::-1]),
        'eye_l_roi_norm': all_positions[1] / np.array(roi_sz[::-1]),
        'eye_r_roi_norm': all_positions[2] / np.array(roi_sz[::-1]),
        'keypoint_count': len(keypoint_stats)
    }

def transform_bbox_to_image_scales(bbox_data, roi_coords_full, roi_coords_ds, roi_sz,
                                   full_img_shape=(4512, 4512), ds_img_shape=(640, 640)):
    """Transform bounding box coordinates to multiple image scales."""
    if bbox_data is None:
        return None
    
    center_roi_norm, extent_roi_norm = bbox_data['center_roi_norm'], bbox_data['extent_roi_norm']
    
    # Full resolution scale
    center_full_norm, extent_full_norm = [np.nan, np.nan], [np.nan, np.nan]
    if roi_coords_full[0] != -1:
        center_full_px = np.array(roi_coords_full) + center_roi_norm * np.array(roi_sz[::-1])
        extent_full_px = extent_roi_norm * np.array(roi_sz[::-1])
        center_full_norm = center_full_px / np.array(full_img_shape[::-1])
        extent_full_norm = extent_full_px / np.array(full_img_shape[::-1])
    
    # Downsampled scale
    center_ds_norm, extent_ds_norm = [np.nan, np.nan], [np.nan, np.nan]
    if roi_coords_ds[0] != -1:
        roi_sz_ds = np.array(roi_sz) * (ds_img_shape[0] / full_img_shape[0])
        center_ds_px = np.array(roi_coords_ds) + center_roi_norm * roi_sz_ds[::-1]
        extent_ds_px = extent_roi_norm * roi_sz_ds[::-1]
        center_ds_norm = center_ds_px / np.array(ds_img_shape[::-1])
        extent_ds_norm = extent_ds_px / np.array(ds_img_shape[::-1])
    
    return {
        'full_scale': {'center_norm': center_full_norm, 'extent_norm': extent_full_norm},
        'ds_scale': {'center_norm': center_ds_norm, 'extent_norm': extent_ds_norm},
        'roi_scale': {'center_norm': center_roi_norm, 'extent_norm': extent_roi_norm}
    }

@delayed
def track_chunk_delayed(zarr_path, chunk_slice, roi_sz, roi_thresh, se1_radius, se2_radius):
    """
    Process tracking for a chunk of ROIs using keypoint detection.
    Detects swim bladder and eyes, calculates heading, and creates multi-scale bounding boxes.
    """
    se1, se2 = disk(se1_radius), disk(se2_radius)
    
    with zarr.open(zarr_path, mode='r') as root:
        # Get the latest runs
        latest_crop_run = root['crop_runs'].attrs['latest']
        latest_detect_run = root[f'crop_runs/{latest_crop_run}'].attrs.get('source_detect_run')
        if not latest_detect_run:
            latest_detect_run = root['detect_runs'].attrs['latest']
        
        # Get detection counts for this chunk
        n_detections_per_frame = root[f'detect_runs/{latest_detect_run}/n_detections'][chunk_slice]
        
        # Calculate the detection indices for this chunk
        start_detection_idx = int(np.sum(root[f'detect_runs/{latest_detect_run}/n_detections'][:chunk_slice.start]))
        end_detection_idx = start_detection_idx + int(np.sum(n_detections_per_frame))
        detection_slice = slice(start_detection_idx, end_detection_idx)
        
        # Load ROI images and coordinates
        rois_chunk = root[f'crop_runs/{latest_crop_run}/roi_images'][detection_slice]
        coords_full_chunk = root[f'crop_runs/{latest_crop_run}/roi_coordinates_full'][detection_slice]
        
        # Calculate downsampled coordinates
        ds_img_shape = root['raw_video/images_ds'].shape[1:]
        full_img_shape = root['raw_video/images_full'].shape[1:]
        scale_factor = ds_img_shape[0] / full_img_shape[0]
        coords_ds_chunk = coords_full_chunk * scale_factor
        
        # Get background for difference calculation
        latest_bg_run = root['background_runs'].attrs['latest']
        bg_group = root[f'background_runs/{latest_bg_run}']
        
        # Check what background arrays are available and use the appropriate one
        if 'background_full' in bg_group:
            background = bg_group['background_full'][:]
            use_full_background = True
        elif 'background' in bg_group:
            background = bg_group['background'][:]
            use_full_background = True
        elif 'background_ds' in bg_group:
            # Only downsampled background available - we'll need to work with downsampled ROIs
            background_ds = bg_group['background_ds'][:]
            use_full_background = False
            import cv2
            # Upscale to full resolution for ROI extraction
            background = cv2.resize(background_ds, (full_img_shape[1], full_img_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
        else:
            raise KeyError(f"No background array found in {latest_bg_run}")
        
        # Get initial bounding boxes from detect stage
        bbox_coords_chunk = root[f'detect_runs/{latest_detect_run}/bbox_norm_coords'][detection_slice]
    
    chunk_len = len(rois_chunk)
    chunk_results = np.full((chunk_len, 21), np.nan, dtype='f8')
    
    for i in range(chunk_len):
        try:
            # Start with the detection bbox as fallback
            initial_bbox = bbox_coords_chunk[i]
            if not np.isnan(initial_bbox[0]):
                # Store initial bbox in downsampled coordinates (columns 7-10)
                chunk_results[i, 7:11] = initial_bbox
            
            # Now attempt keypoint detection for more precise tracking
            roi = rois_chunk[i]
            coords_full = coords_full_chunk[i]
            coords_ds = coords_ds_chunk[i]
            
            if np.all(roi == 0) or coords_full[0] == -1:
                continue
            
            # Extract background ROI
            roi_y1 = int(coords_full[1])
            roi_x1 = int(coords_full[0])
            roi_y2 = roi_y1 + roi.shape[0]
            roi_x2 = roi_x1 + roi.shape[1]
            
            # Ensure coordinates are within bounds
            roi_y1 = max(0, roi_y1)
            roi_x1 = max(0, roi_x1)
            roi_y2 = min(background_full.shape[0], roi_y2)
            roi_x2 = min(background_full.shape[1], roi_x2)
            
            background_roi = background[roi_y1:roi_y2, roi_x1:roi_x2]
            
            if background_roi.shape != roi.shape:
                continue
            
            # Calculate difference image
            diff_roi = np.clip(background_roi.astype(np.int16) - roi.astype(np.int16), 0, 255).astype(np.uint8)
            
            # Adaptive thresholding to find keypoints
            current_thresh = roi_thresh
            roi_stat = []
            for _ in range(5):
                im_roi = erosion(dilation(erosion(diff_roi >= current_thresh, se1), se2), se1)
                roi_stat = [r for r in regionprops(label(im_roi)) if r.area > 5]
                if len(roi_stat) >= 3:
                    break
                current_thresh -= 5
            
            # If we found 3+ keypoints, process them
            if len(roi_stat) >= 3:
                keypoint_stats = sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]
                pts = np.array([s.centroid[::-1] for s in keypoint_stats])
                
                # Use triangle geometry to identify keypoints
                angles, _ = triangle_calculations(pts[0], pts[1], pts[2])
                kp_idx = np.argsort(angles)
                
                # The point with smallest angle is the swim bladder
                # The other two are eyes
                eye_mean = np.mean(pts[kp_idx[1:3]], axis=0)
                head_vec = eye_mean - pts[kp_idx[0]]
                heading = np.rad2deg(np.arctan2(-head_vec[1], head_vec[0]))
                
                # Determine which eye is left/right using rotation
                R = np.array([[np.cos(np.deg2rad(heading)), -np.sin(np.deg2rad(heading))], 
                             [np.sin(np.deg2rad(heading)), np.cos(np.deg2rad(heading))]])
                rotpts = (pts - eye_mean) @ R.T
                eye_r_idx, eye_l_idx = (kp_idx[1], kp_idx[2]) if rotpts[kp_idx[1], 1] > rotpts[kp_idx[2], 1] else (kp_idx[2], kp_idx[1])
                ordered_stats = [keypoint_stats[kp_idx[0]], keypoint_stats[eye_l_idx], keypoint_stats[eye_r_idx]]
                
                # Calculate bounding boxes at multiple scales
                bbox_data = calculate_multi_scale_bounding_boxes(ordered_stats, roi_sz)
                if bbox_data is None:
                    continue
                
                multi_scale_data = transform_bbox_to_image_scales(
                    bbox_data, coords_full, coords_ds, tuple(roi_sz),
                    full_img_shape, ds_img_shape
                )
                if multi_scale_data is None:
                    continue
                
                # Calculate confidence based on keypoint area
                confidence = min(1.0, np.mean([s.area for s in ordered_stats]) / 100.0)
                
                # Store all tracking data
                chunk_results[i, :] = [
                    heading,
                    *bbox_data['bladder_roi_norm'],
                    *bbox_data['eye_l_roi_norm'],
                    *bbox_data['eye_r_roi_norm'],
                    *multi_scale_data['ds_scale']['center_norm'],
                    *multi_scale_data['ds_scale']['extent_norm'],
                    *multi_scale_data['full_scale']['center_norm'],
                    *multi_scale_data['full_scale']['extent_norm'],
                    *coords_full,
                    *coords_ds,
                    confidence,
                    current_thresh
                ]
        except Exception as e:
            # Silent failure - keep the initial bbox data
            continue
    
    # Count successful tracks per frame
    frame_counts = np.zeros(len(n_detections_per_frame), dtype='i4')
    detection_idx = 0
    for frame_idx, n_dets in enumerate(n_detections_per_frame):
        if n_dets > 0:
            frame_results = chunk_results[detection_idx:detection_idx+n_dets]
            frame_counts[frame_idx] = np.sum(~np.isnan(frame_results[:, 0]))
            detection_idx += n_dets
    
    return detection_slice, chunk_results, frame_counts

def run_tracking_stage(zarr_path, scheduler_name, params, console):
    console.rule(f"[bold]Stage 5: Tracking with Multi-Fish Data (Dask {scheduler_name.title()} Scheduler)[/bold]")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'crop_runs' not in root: 
        raise ValueError("Crop stage not run.")
    
    track_params = params['track']
    track_group = get_run_group(root, 'tracking', console)
    
    source_run_name = root['crop_runs'].attrs['latest']
    source_type = 'crop'
    console.print(f"  Using cropped data from run: [cyan]{source_run_name}[/cyan]")
    
    # Get the detect run that the crop was based on
    source_detect_run = root[f'crop_runs/{source_run_name}'].attrs.get('source_detect_run')
    if not source_detect_run:
        # Fallback to latest detect run if not stored
        source_detect_run = root['detect_runs'].attrs['latest']
    console.print(f"  Using detections from run: [cyan]{source_detect_run}[/cyan]")

    track_group.attrs.update({
        'tracking_timestamp_utc': datetime.now(timezone.utc).isoformat(), 
        'dask_scheduler': scheduler_name,
        'parameters': track_params,
        f'source_{source_type}_run': source_run_name,
        'source_detect_run': source_detect_run
    })

    num_images = root['raw_video/images_ds'].shape[0]
    
    # Get n_detections from the detect stage, not crop stage
    n_detections_array = root[f'detect_runs/{source_detect_run}/n_detections']
    temp_n_detections = n_detections_array[:]
    total_detections = int(temp_n_detections.sum())
    
    if total_detections == 0:
        console.print(f"  [yellow]⚠[/yellow] No detections found in detect stage. Skipping tracking.")
        track_group.attrs['summary_statistics'] = {
            'total_frames': num_images,
            'frames_with_tracks': 0,
            'total_successful_tracks': 0,
            'percent_frames_tracked': 0.0
        }
        track_group.attrs['duration_seconds'] = time.perf_counter() - start_time
        return
    
    # Get chunk size from crop stage
    track_chunk_size = root[f"crop_runs/{source_run_name}/roi_images"].chunks[0]

    # Create new arrays for tracking results
    n_tracked_detections = track_group.create_dataset(
        'n_detections', 
        shape=(num_images,), 
        chunks=(track_chunk_size * 4,), 
        dtype='i4', 
        overwrite=True
    )
    
    tracking_results = track_group.create_dataset(
        'tracking_results', 
        shape=(total_detections, 21), 
        chunks=(track_chunk_size * 4, None), 
        dtype='f8', 
        overwrite=True
    )
    tracking_results[:] = np.nan
    tracking_results.attrs['column_names'] = [
        'heading_degrees', 'bladder_x_roi_norm', 'bladder_y_roi_norm', 
        'eye_l_x_roi_norm', 'eye_l_y_roi_norm', 'eye_r_x_roi_norm', 'eye_r_y_roi_norm',
        'bbox_x_norm_ds', 'bbox_y_norm_ds', 'bbox_width_norm_ds', 'bbox_height_norm_ds',
        'bbox_x_norm_full', 'bbox_y_norm_full', 'bbox_width_norm_full', 'bbox_height_norm_full',
        'roi_x1_full', 'roi_y1_full', 'roi_x1_ds', 'roi_y1_ds', 
        'confidence_score', 'effective_threshold'
    ]
    
    chunk_slices = [slice(i, min(i + track_chunk_size, num_images)) for i in range(0, num_images, track_chunk_size)]
    console.print(f"  Creating [yellow]{len(chunk_slices)}[/yellow] tracking tasks...")
    
    delayed_tasks = [
        track_chunk_delayed(
            zarr_path, s, 
            roi_sz=tuple(track_params['roi_sz']), 
            roi_thresh=track_params['roi_thresh'], 
            se1_radius=track_params['se1_radius'], 
            se2_radius=track_params['se2_radius']
        ) for s in chunk_slices
    ]
    
    with ProgressBar(): 
        results = dask.compute(*delayed_tasks)
    
    console.print("  Writing tracking results to Zarr...")
    for det_slc, chunk_res, counts in tqdm(results, desc="Writing Tracking Chunks"):
        tracking_results[det_slc] = chunk_res
        
    # Recalculate detection counts based on successful tracks
    cumulative_counts = np.cumsum(np.insert(temp_n_detections, 0, 0))
    
    for i in range(num_images):
        start_idx = cumulative_counts[i]
        end_idx = cumulative_counts[i+1]
        if end_idx > start_idx:
            frame_results = tracking_results[start_idx:end_idx]
            n_tracked_detections[i] = np.sum(~np.isnan(frame_results[:, 0]))
        else:
            n_tracked_detections[i] = 0

    successful_tracks = int(n_tracked_detections[:].sum())
    frames_with_tracks = int(np.sum(n_tracked_detections[:] > 0))
    percent_tracked = (frames_with_tracks / num_images) * 100 if num_images > 0 else 0
    
    summary_stats = {
        'total_frames': num_images, 
        'frames_with_tracks': frames_with_tracks, 
        'total_successful_tracks': successful_tracks, 
        'percent_frames_tracked': round(percent_tracked, 2)
    }
    
    if successful_tracks > 0:
        valid_indices = np.where(~np.isnan(tracking_results[:, 0]))[0]
        summary_stats['confidence_stats'] = {
            'mean': float(np.nanmean(tracking_results[valid_indices, 19])), 
            'std': float(np.nanstd(tracking_results[valid_indices, 19])), 
            'min': float(np.nanmin(tracking_results[valid_indices, 19])), 
            'max': float(np.nanmax(tracking_results[valid_indices, 19]))
        }
        
    track_group.attrs['summary_statistics'] = summary_stats
    track_group.attrs['duration_seconds'] = time.perf_counter() - start_time
    
    console.print(f"  [bold green]✓[/bold green] Tracking completed in [green]{track_group.attrs['duration_seconds']:.2f}[/green] seconds.")
    console.print(f"  Found [green]{successful_tracks}[/green] successful tracks in [green]{frames_with_tracks}/{num_images}[/green] frames ([cyan]{percent_tracked:.2f}%[/cyan]).")
    
    parent_group = root['tracking_runs']
    if 'best' not in parent_group.attrs or percent_tracked > parent_group.attrs['best']['percent_frames_tracked']:
        console.print(f"  [bold green] New best tracking run! Success rate: {percent_tracked:.2f}% of frames[/bold green]")
        parent_group.attrs['best'] = {'run_name': track_group.name, **summary_stats}

def run_refine_stage(zarr_path, params, console):
    """
    Refines the output of the crop stage by removing temporal jumps.
    NOTE: This stage is less critical in a multi-fish pipeline and may be removed in the future.
    """
    console.rule("[bold]Stage 3.5: Refining Crop Detections (Not Recommended for Multi-Fish)[/bold]")
    console.print("[yellow]Warning: The refinement stage is designed for single-fish tracking and may produce suboptimal results for multiple fish.[/yellow]")
    # For now, this stage will be a placeholder.
    # A proper multi-fish refinement would require an object tracking algorithm (e.g., Kalman filter, SORT).
    time.sleep(2) # Simulate work
    console.print("Skipping refinement stage for multi-fish workflow.")


def create_dish_mask(mask_params, img_shape, console=None):
    """
    Create a dish mask based on the shape specified in mask_params.
    Supports both 'rectangle' and 'circle' shapes.
    
    Args:
        mask_params: Dictionary containing mask parameters
        img_shape: Tuple of (height, width) for the image
        console: Rich Console object for formatted printing (optional)
    
    Returns:
        numpy array mask or None if no mask defined
    """
    if not mask_params:
        return None
    
    dish_shape = mask_params.get('shape', 'circle')
    mask = None
    
    if dish_shape == 'rectangle' and 'roi' in mask_params:
        x, y, w, h = mask_params['roi']
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        if console:
            console.print(f"  [green]✓[/green] Using rectangular mask from config.")
        
    elif dish_shape == 'circle' and 'center' in mask_params and 'radius' in mask_params:
        center = mask_params['center']
        radius = mask_params['radius']
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.circle(mask, tuple(center), radius, 255, -1)
        if console:
            console.print(f"  [green]✓[/green] Using circular mask from config (center={center}, radius={radius}).")
    
    return mask


def main():
    console = Console()
    parser = argparse.ArgumentParser(description="Fish tracking pipeline with YOLO-ready zarr data generation.")
    parser.add_argument("zarr_path", type=str, help="Path to the output Zarr file.")
    parser.add_argument("--video-path", type=str, help="Path to the input video file (required for 'import' or 'all' stages).")
    parser.add_argument("--config", type=str, default="src/pipeline_config.yaml", help="Path to the pipeline configuration YAML file.")
    parser.add_argument(
        "--stage", 
        required=True, 
        nargs='+', 
        choices=['import', 'background', 'detect', 'crop', 'track', 'assign_ids', 'all'], 
        help="One or more processing stages to run in order (e.g., import background detect)."
    )
    parser.add_argument("--scheduler", default='processes', choices=['processes', 'threads', 'single-thread'], help="Dask scheduler.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of workers.")
    parser.add_argument("--roi-thresh", type=int, help="Override roi_thresh in the config file.")
    args = parser.parse_args()

    console.print(Panel("[bold cyan]Multi-Fish Tracking Pipeline[/bold cyan]", subtitle="[yellow]Powered by Dask & Zarr![/yellow]", expand=False))

    if ('import' in args.stage or 'all' in args.stage) and not args.video_path:
        parser.error("--video-path is required when running the 'import' or 'all' stages.")

    try:
        with open(args.config, 'r') as f:
            pipeline_params = yaml.safe_load(f)
        console.print(f"Loaded pipeline parameters from: [green]{args.config}[/green]")
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Config file not found at {args.config}"); return
    if args.roi_thresh is not None:
        pipeline_params['track']['roi_thresh'] = args.roi_thresh
        console.print(f"Overriding [magenta]track:roi_thresh[/magenta] with CLI value: [yellow]{args.roi_thresh}[/yellow]")
    
    dask.config.set(scheduler=args.scheduler, num_workers=args.num_workers or os.cpu_count())
    console.print(f"Using Dask '[yellow]{dask.config.get('scheduler')}[/yellow]' scheduler with [yellow]{dask.config.get('num_workers')}[/yellow] workers.")
    
    overall_start_time = time.perf_counter()
    cli_args_dict = vars(args)

    # Logic to run stages based on list membership
    if 'import' in args.stage or 'all' in args.stage: run_import_stage_parallel_io(args.video_path, args.zarr_path, pipeline_params, cli_args_dict, console)
    if 'background' in args.stage or 'all' in args.stage: run_background_stage(args.zarr_path, pipeline_params, console)
    if 'detect' in args.stage or 'all' in args.stage: run_detect_stage(args.zarr_path, dask.config.get('scheduler'), pipeline_params, console)
    if 'assign_ids' in args.stage or 'all' in args.stage: run_assign_ids_stage(args.zarr_path, pipeline_params, console)
    if 'crop' in args.stage or 'all' in args.stage: run_crop_stage(args.zarr_path, dask.config.get('scheduler'), pipeline_params, console)
    if 'track' in args.stage or 'all' in args.stage: run_tracking_stage(args.zarr_path, dask.config.get('scheduler'), pipeline_params, console)    
    
    # Check if 'all' was specified for the final summary
    if 'all' in args.stage:
        root = zarr.open_group(args.zarr_path, mode='a')
        root.attrs.update({'total_pipeline_duration_seconds': time.perf_counter() - overall_start_time, 'pipeline_version': '2.0-multi-fish'})
        latest_track_run = root['tracking_runs'].attrs['latest']
        successful_tracks = root[f'tracking_runs/{latest_track_run}'].attrs['summary_statistics']['total_successful_tracks']
        root.attrs['yolo_ready'] = successful_tracks > 0
        
        console.rule("[bold]Pipeline Complete[/bold]")
        if successful_tracks > 0:
            console.print(f"Pipeline completed! Total time: [bold green]{root.attrs['total_pipeline_duration_seconds']:.2f}[/bold green] seconds.")
            console.print(f"[cyan]Data is ready for YOLO training. Next step:[/] python zarr_yolo_dataset_loader.py {args.zarr_path}")
        else:
            console.print(f"[bold yellow]Pipeline completed but no fish were successfully tracked![/bold yellow]")
    
    console.print("\n[bold]Tracking complete![/bold]")

if __name__ == "__main__":
    main()