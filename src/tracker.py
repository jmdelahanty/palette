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

# --- Utility Functions ---

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

def calculate_multi_scale_bounding_boxes(keypoint_stats, roi_sz, margin_factor=1.5, min_bbox_size=0.05):
    if len(keypoint_stats) < 3: return None
    all_positions = np.array([s.centroid[::-1] for s in keypoint_stats])
    min_pos, max_pos = np.min(all_positions, axis=0), np.max(all_positions, axis=0)
    center_roi_px = (min_pos + max_pos) / 2.0
    extent_roi_px = np.maximum((max_pos - min_pos) * margin_factor, min_bbox_size * np.array(roi_sz[::-1]))
    center_roi_norm = center_roi_px / np.array(roi_sz[::-1])
    extent_roi_norm = extent_roi_px / np.array(roi_sz[::-1])
    return {
        'center_roi_norm': center_roi_norm, 'extent_roi_norm': extent_roi_norm,
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

# --- Stage-Specific Functions ---

def run_import_stage_parallel_io(video_path, zarr_path, params, cli_args, console):
    """
    An improved import stage that uses a separate thread for writing to disk,
    allowing GPU decoding and disk I/O to happen in parallel.
    """
    console.rule("[bold]Stage 1: Importing Video[/bold]")
    start_time = time.perf_counter()

    # --- GPU and Video Setup ---
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(video_path, ctx=decord.gpu(0))
    console.print("Using GPU context for video decoding")

    n_frames, full_height, full_width = len(vr), vr[0].shape[0], vr[0].shape[1]
    import_params = params['import']
    ds_size, chunk_size, batch_size = tuple(import_params['downsample_size']), import_params['chunk_size'], import_params['batch_size']
    gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device='cuda:0')

    # --- Zarr File and Metadata Setup ---
    root = zarr.open_group(zarr_path, mode='w')
    param_group = root.create_group('pipeline_params')
    for stage, stage_params in params.items():
        param_group.attrs[stage] = stage_params

    root.attrs.update({
        'command_line_args': cli_args,
        'git_info': get_git_info(),
        'source_video_metadata': iio.immeta(video_path),
        'platform_info': get_platform_info(),
        'software_versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'zarr': zarr.__version__,
            'scikit-image': skimage.__version__,
            'opencv-python': cv2.__version__,
            'torch': torch.__version__,
            'decord': decord.__version__
        }
    })

    raw_video_group = root.create_group('raw_video')
    
    compressor_details = {'cname': 'lz4', 'clevel': 1, 'shuffle': 'bit'}
    compressor = zarr.Blosc(cname=compressor_details['cname'], 
                            clevel=compressor_details['clevel'], 
                            shuffle=zarr.Blosc.BITSHUFFLE)

    raw_video_group.attrs.update({
        'import_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'original_resolution': (full_height, full_width),
        'downsampled_resolution': ds_size,
        'decoding_device': get_gpu_info(),
        'compressor': compressor_details
    })

    optimal_chunk_size = min(64, chunk_size * 2)
    
    images_full = raw_video_group.create_dataset(
        'images_full',
        shape=(n_frames, full_height, full_width),
        chunks=(optimal_chunk_size, None, None),
        dtype=np.uint8,
        compressor=compressor,
        write_empty_chunks=False
    )

    images_ds = raw_video_group.create_dataset(
        'images_ds',
        shape=(n_frames, ds_size[0], ds_size[1]),
        chunks=(optimal_chunk_size, None, None),
        dtype=np.uint8,
        compressor=compressor,
        write_empty_chunks=False
    )
    console.print(f"Compression: [cyan]{compressor_details['cname']}[/cyan]")
    console.print(f"Chunk size: {optimal_chunk_size}")

    data_queue = queue.Queue(maxsize=4)

    def writer_task(q, zarr_full, zarr_ds):
        while True:
            item = q.get()
            if item is None:
                break
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
        
        full_batch_data = []
        ds_batch_data = []

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
        full_combined = np.concatenate(full_batch_data, axis=0)
        ds_combined = np.concatenate(ds_batch_data, axis=0)
        
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
    random.seed(bg_params['seed'])
    random_indices = random.sample(range(num_images), min(bg_params['sample_size'], num_images))
    console.print("Calculating background modes...")
    bg_group.create_dataset('background_full', data=fast_mode_bincount(root['raw_video/images_full'].get_orthogonal_selection((random_indices, slice(None), slice(None)))), overwrite=True)
    bg_group.create_dataset('background_ds', data=fast_mode_bincount(root['raw_video/images_ds'].get_orthogonal_selection((random_indices, slice(None), slice(None)))), overwrite=True)
    bg_group.attrs['source_frame_indices'] = random_indices
    bg_group.attrs['duration_seconds'] = time.perf_counter() - start_time
    console.print(f"Background stage completed in [green]{bg_group.attrs['duration_seconds']:.2f}[/green] seconds.")

@delayed
def crop_chunk_delayed(zarr_path, chunk_slice, roi_sz, ds_thresh, se1_radius, se4_radius, dish_mask, max_distance_norm):
    se1, se4 = disk(se1_radius), disk(se4_radius)
    with zarr.open(zarr_path, mode='r') as root:
        images_ds_chunk = root['raw_video/images_ds'][chunk_slice]
        images_full_chunk = root['raw_video/images_full'][chunk_slice]
        latest_bg_run = root['background_runs'].attrs['latest']
        background_ds = root[f'background_runs/{latest_bg_run}/background_ds'][:]
        full_img_shape, ds_img_shape = images_full_chunk.shape[1:], images_ds_chunk.shape[1:]
    
    chunk_len = images_ds_chunk.shape[0]
    chunk_rois = np.zeros((chunk_len, *roi_sz), dtype='uint8')
    chunk_coords_full = np.full((chunk_len, 2), -1, dtype='i4')
    chunk_coords_ds = np.full((chunk_len, 2), -1, dtype='i4')
    chunk_bbox_norms = np.full((chunk_len, 2), np.nan, dtype='f8')
    chunk_thresholds = np.full(chunk_len, -1, dtype='i2')

    # Variable to track the last known position within this chunk
    last_known_centroid_norm = None

    for i in range(chunk_len):
        try:
            diff_ds = np.clip(background_ds.astype(np.int16) - images_ds_chunk[i].astype(np.int16), 0, 255).astype(np.uint8)
            
            # Apply the pre-calculated dish mask
            if dish_mask is not None:
                diff_ds[dish_mask == 0] = 0

            # Adaptive Thresholding Loop
            current_thresh = ds_thresh
            ds_stat = []
            for _ in range(5):
                im_ds = erosion(dilation(erosion(diff_ds >= current_thresh, se1), se4), se1)
                ds_stat = regionprops(label(im_ds))
                if ds_stat:
                    chunk_thresholds[i] = current_thresh
                    break
                current_thresh -= 5
            
            if not ds_stat: continue

            # Proximity Check Logic
            chosen_blob = max(ds_stat, key=lambda r: r.area) # Default to largest
            if last_known_centroid_norm is not None and len(ds_stat) > 1:
                centroids_norm = [np.array(s.centroid)[::-1] / np.array(ds_img_shape) for s in ds_stat]
                distances = [np.linalg.norm(c - last_known_centroid_norm) for c in centroids_norm]
                closest_idx = np.argmin(distances)
                
                if distances[closest_idx] < max_distance_norm:
                    chosen_blob = ds_stat[closest_idx]

            ds_centroid_norm = np.array(chosen_blob.centroid)[::-1] / np.array(ds_img_shape)
            last_known_centroid_norm = ds_centroid_norm # Update for next frame

            # ROI Calculation (uses the chosen blob's centroid)
            full_centroid_px = np.round(ds_centroid_norm * np.array(full_img_shape)).astype(int)
            roi_x1_full, roi_y1_full = full_centroid_px[0] - roi_sz[1] // 2, full_centroid_px[1] - roi_sz[0] // 2
            roi_size_ds = np.array(roi_sz) * (ds_img_shape[0] / full_img_shape[0])
            roi_x1_ds, roi_y1_ds = int(full_centroid_px[0] * (ds_img_shape[1]/full_img_shape[1])) - int(roi_size_ds[1] // 2), int(full_centroid_px[1] * (ds_img_shape[0]/full_img_shape[0])) - int(roi_size_ds[0] // 2)
            roi = images_full_chunk[i][roi_y1_full:roi_y1_full+roi_sz[0], roi_x1_full:roi_x1_full+roi_sz[1]]
            
            if roi.shape != tuple(roi_sz): continue
            
            chunk_rois[i], chunk_coords_full[i], chunk_coords_ds[i], chunk_bbox_norms[i] = roi, (roi_x1_full, roi_y1_full), (roi_x1_ds, roi_y1_ds), ds_centroid_norm
        except Exception: 
            continue
    
    return chunk_slice, chunk_rois, chunk_coords_full, chunk_coords_ds, chunk_bbox_norms, chunk_thresholds


def run_crop_stage(zarr_path, scheduler_name, params, console):
    console.rule(f"[bold]Stage 3: Cropping with Advanced Logic (Dask {scheduler_name.title()} Scheduler)[/bold]")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'background_runs' not in root: raise ValueError("Background stage not run.")
    
    crop_params = params['crop']
    crop_group = get_run_group(root, 'crop', console)
    # --- MODIFICATION: Save all parameters used for this run ---
    crop_group.attrs.update({
        'crop_timestamp_utc': datetime.now(timezone.utc).isoformat(), 
        'dask_scheduler': scheduler_name, 
        'parameters': crop_params
    })
    
    # --- Automatically detect the dish mask ---
    console.print("Detecting dish from background model...")
    latest_bg_run = root['background_runs'].attrs['latest']
    background_ds = root[f'background_runs/{latest_bg_run}/background_ds'][:]
    
    mask_params = crop_params.get('dish_mask', {})
    dish_shape = mask_params.get('shape', 'circle')
    ds_img_shape = background_ds.shape
    mask = None

    if dish_shape == 'circle':
        circles = cv2.HoughCircles(background_ds, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=mask_params.get('hough_param1', 50),
                                   param2=mask_params.get('hough_param2', 30),
                                   minRadius=0, maxRadius=0)
        if circles is not None:
            circle = np.uint16(np.around(circles[0, 0, :]))
            # Apply radius adjustment from config
            radius = circle[2] + mask_params.get('radius_adj', 0)
            mask = np.zeros(ds_img_shape, dtype=np.uint8)
            cv2.circle(mask, (circle[0], circle[1]), radius, 255, -1)
            console.print(f"  [green]✓[/green] Detected circular dish and applied radius adjustment.")
    
    elif dish_shape == 'rectangle':
        _, thresh = cv2.threshold(background_ds, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            mask = np.zeros(ds_img_shape, dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            console.print(f"  [green]✓[/green] Detected rectangular dish.")

    if mask is None:
        console.print(f"  [yellow]⚠️[/yellow]  Could not detect a {dish_shape} dish. Proceeding without a mask.")
        mask = np.ones(ds_img_shape, dtype=np.uint8) * 255

    # --- Setup for Dask tasks ---
    num_images = root['raw_video/images_ds'].shape[0]
    crop_chunk_size = min(params['import']['chunk_size'], num_images)
    
    # Create Zarr datasets for the output
    roi_images = crop_group.create_dataset('roi_images', shape=(num_images, *crop_params['roi_sz']), chunks=(crop_chunk_size, None, None), dtype='uint8', overwrite=True)
    roi_coordinates_full = crop_group.create_dataset('roi_coordinates_full', shape=(num_images, 2), chunks=(crop_chunk_size * 4, None), dtype='i4', overwrite=True)
    roi_coordinates_ds = crop_group.create_dataset('roi_coordinates_ds', shape=(num_images, 2), chunks=(crop_chunk_size * 4, None), dtype='i4', overwrite=True)
    bbox_norm_coords = crop_group.create_dataset('bbox_norm_coords', shape=(num_images, 2), chunks=(crop_chunk_size * 4, None), dtype='f8', overwrite=True)
    effective_thresholds = crop_group.create_dataset('effective_thresholds', shape=(num_images,), chunks=(crop_chunk_size * 4,), dtype='i2', overwrite=True)
    
    roi_coordinates_full[:], roi_coordinates_ds[:], bbox_norm_coords[:] = -1, -1, np.nan
    chunk_slices = [slice(i, min(i + crop_chunk_size, num_images)) for i in range(0, num_images, crop_chunk_size)]
    
    max_dist = crop_params.get('proximity_check', {}).get('max_distance_norm', 0.15)
    console.print(f"Creating [yellow]{len(chunk_slices)}[/yellow] Dask tasks with proximity checks and dish masking...")

    delayed_tasks = [crop_chunk_delayed(zarr_path, s, 
                                        roi_sz=tuple(crop_params['roi_sz']), 
                                        ds_thresh=crop_params['ds_thresh'], 
                                        se1_radius=crop_params['se1_radius'], 
                                        se4_radius=crop_params['se4_radius'],
                                        dish_mask=mask,
                                        max_distance_norm=max_dist) for s in chunk_slices]
    
    with ProgressBar(): results = dask.compute(*delayed_tasks)
    console.print("Writing results to Zarr...")
    
    for slc, rois, coords_full, coords_ds, bboxes, thresholds in tqdm(results, desc="Writing Chunks"):
        roi_images[slc], roi_coordinates_full[slc], roi_coordinates_ds[slc], bbox_norm_coords[slc] = rois, coords_full, coords_ds, bboxes
        effective_thresholds[slc] = thresholds

    # --- Final Summary and "Best" Run Logic ---
    successful_crops = np.sum(effective_thresholds[:] > -1)
    percent_cropped = (successful_crops / num_images) * 100 if num_images > 0 else 0
    summary_stats = {'total_frames': num_images, 'frames_cropped': successful_crops, 'percent_cropped': round(percent_cropped, 2)}
    crop_group.attrs['summary_statistics'] = summary_stats
    
    parent_group = root['crop_runs']
    if 'best' not in parent_group.attrs or percent_cropped > parent_group.attrs['best']['percent_cropped']:
        console.print(f"[bold green]New best cropping run found! Success rate: {percent_cropped:.2f}%[/bold green]")
        parent_group.attrs['best'] = {'run_name': crop_group.name, **summary_stats}
    
    crop_group.attrs['duration_seconds'] = time.perf_counter() - start_time
    
    console.print(f"Cropping stage completed in [green]{crop_group.attrs['duration_seconds']:.2f}[/green] seconds.")
    console.print(f"Successfully cropped [green]{successful_crops}/{num_images}[/green] frames ([cyan]{percent_cropped:.2f}%[/cyan]).")

@delayed
def track_chunk_delayed(zarr_path, chunk_slice, roi_sz, roi_thresh, se1_radius, se2_radius):
    se1, se2 = disk(se1_radius), disk(se2_radius)
    with zarr.open(zarr_path, mode='r') as root:
        latest_crop_run = root['crop_runs'].attrs['latest']
        latest_bg_run = root['background_runs'].attrs['latest']
        rois_chunk = root[f'crop_runs/{latest_crop_run}/roi_images'][chunk_slice]
        coords_full_chunk = root[f'crop_runs/{latest_crop_run}/roi_coordinates_full'][chunk_slice]
        coords_ds_chunk = root[f'crop_runs/{latest_crop_run}/roi_coordinates_ds'][chunk_slice]
        background_full = root[f'background_runs/{latest_bg_run}/background_full'][:]
    chunk_len = rois_chunk.shape[0]
    chunk_results = np.full((chunk_len, 20), np.nan, dtype='f8')
    for i in range(chunk_len):
        try:
            roi = rois_chunk[i]
            if np.all(roi == 0): continue
            coords_full, coords_ds = coords_full_chunk[i], coords_ds_chunk[i]
            if coords_full[0] == -1: continue
            background_roi = background_full[coords_full[1]:coords_full[1]+roi.shape[0], coords_full[0]:coords_full[0]+roi.shape[1]]
            if background_roi.shape != roi.shape: continue
            diff_roi = np.clip(background_roi.astype(np.int16) - roi.astype(np.int16), 0, 255).astype(np.uint8)
            im_roi = erosion(dilation(erosion(diff_roi >= roi_thresh, se1), se2), se1)
            roi_stat = [r for r in regionprops(label(im_roi)) if r.area > 5]
            if len(roi_stat) < 3: continue
            keypoint_stats = sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]
            pts = np.array([s.centroid[::-1] for s in keypoint_stats])
            angles, _ = triangle_calculations(pts[0], pts[1], pts[2])
            kp_idx = np.argsort(angles)
            eye_mean = np.mean(pts[kp_idx[1:3]], axis=0)
            head_vec = eye_mean - pts[kp_idx[0]]
            heading = np.rad2deg(np.arctan2(-head_vec[1], head_vec[0]))
            R = np.array([[np.cos(np.deg2rad(heading)), -np.sin(np.deg2rad(heading))], [np.sin(np.deg2rad(heading)), np.cos(np.deg2rad(heading))]])
            rotpts = (pts - eye_mean) @ R.T
            eye_r_idx, eye_l_idx = (kp_idx[1], kp_idx[2]) if rotpts[kp_idx[1], 1] > rotpts[kp_idx[2], 1] else (kp_idx[2], kp_idx[1])
            ordered_stats = [keypoint_stats[kp_idx[0]], keypoint_stats[eye_l_idx], keypoint_stats[eye_r_idx]]
            bbox_data = calculate_multi_scale_bounding_boxes(ordered_stats, roi_sz)
            if bbox_data is None: continue
            multi_scale_data = transform_bbox_to_image_scales(bbox_data, coords_full, coords_ds, tuple(roi_sz))
            if multi_scale_data is None: continue
            confidence = min(1.0, np.mean([s.area for s in ordered_stats]) / 100.0)
            chunk_results[i, :] = [heading, *bbox_data['bladder_roi_norm'], *bbox_data['eye_l_roi_norm'], *bbox_data['eye_r_roi_norm'], *multi_scale_data['ds_scale']['center_norm'], *multi_scale_data['ds_scale']['extent_norm'], *multi_scale_data['full_scale']['center_norm'], *multi_scale_data['full_scale']['extent_norm'], *coords_full, *coords_ds, confidence]
        except Exception: continue
    return chunk_slice, chunk_results

def run_tracking_stage(zarr_path, scheduler_name, params, console):
    console.rule(f"[bold]Stage 4: Tracking with Multi-Scale Data (Dask {scheduler_name.title()} Scheduler)[/bold]")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'crop_runs' not in root: raise ValueError("Crop stage not run.")
    track_params = params['track']
    track_group = get_run_group(root, 'tracking', console)
    track_group.attrs.update({'tracking_timestamp_utc': datetime.now(timezone.utc).isoformat(), 'dask_scheduler': scheduler_name, 'parameters': track_params})
    num_images = root['raw_video/images_ds'].shape[0]
    track_chunk_size = root[f"crop_runs/{root['crop_runs'].attrs['latest']}/roi_images"].chunks[0]
    tracking_results = track_group.create_dataset('tracking_results', shape=(num_images, 20), chunks=(track_chunk_size * 4, None), dtype='f8', overwrite=True)
    tracking_results[:] = np.nan
    tracking_results.attrs['column_names'] = ['heading_degrees', 'bladder_x_roi_norm', 'bladder_y_roi_norm', 'eye_l_x_roi_norm', 'eye_l_y_roi_norm', 'eye_r_x_roi_norm', 'eye_r_y_roi_norm', 'bbox_x_norm_ds', 'bbox_y_norm_ds', 'bbox_width_norm_ds', 'bbox_height_norm_ds', 'bbox_x_norm_full', 'bbox_y_norm_full', 'bbox_width_norm_full', 'bbox_height_norm_full', 'roi_x1_full', 'roi_y1_full', 'roi_x1_ds', 'roi_y1_ds', 'confidence_score']
    coord_systems = track_group.create_group('coordinate_systems')
    coord_systems.attrs.update({
        'roi_normalized': {'description': 'Coordinates normalized to ROI size (e.g. 320x320)', 'range': [0.0, 1.0], 'columns': ['bladder_x_roi_norm', 'bladder_y_roi_norm', 'eye_l_x_roi_norm', 'eye_l_y_roi_norm', 'eye_r_x_roi_norm', 'eye_r_y_roi_norm']},
        'downsampled_normalized': {'description': 'Coordinates normalized to downsampled image (640x640) - YOLO ready', 'range': [0.0, 1.0], 'columns': ['bbox_x_norm_ds', 'bbox_y_norm_ds', 'bbox_width_norm_ds', 'bbox_height_norm_ds']},
        'full_normalized': {'description': 'Coordinates normalized to full resolution image (4512x4512)', 'range': [0.0, 1.0], 'columns': ['bbox_x_norm_full', 'bbox_y_norm_full', 'bbox_width_norm_full', 'bbox_height_norm_full']},
        'pixel_coordinates': {'description': 'ROI positions in pixel coordinates', 'columns': ['roi_x1_full', 'roi_y1_full', 'roi_x1_ds', 'roi_y1_ds']}
    })
    chunk_slices = [slice(i, min(i + track_chunk_size, num_images)) for i in range(0, num_images, track_chunk_size)]
    console.print(f"Creating [yellow]{len(chunk_slices)}[/yellow] tasks...")
    delayed_tasks = [track_chunk_delayed(zarr_path, s, roi_sz=tuple(track_params['roi_sz']), roi_thresh=track_params['roi_thresh'], se1_radius=track_params['se1_radius'], se2_radius=track_params['se2_radius']) for s in chunk_slices]
    with ProgressBar(): results = dask.compute(*delayed_tasks)
    console.print("Writing tracking results to Zarr...")
    for slc, chunk_res in tqdm(results, desc="Writing Tracking Chunks"):
        tracking_results[slc] = chunk_res
    successful_tracks = np.sum(~np.isnan(tracking_results[:, 0]))
    percent_tracked = (successful_tracks / num_images) * 100 if num_images > 0 else 0
    summary_stats = {'total_frames': num_images, 'frames_tracked': successful_tracks, 'percent_tracked': round(percent_tracked, 2)}
    if successful_tracks > 0:
        valid_indices = np.where(~np.isnan(tracking_results[:, 0]))[0]
        summary_stats['confidence_stats'] = {'mean': float(np.nanmean(tracking_results[valid_indices, 19])), 'std': float(np.nanstd(tracking_results[valid_indices, 19])), 'min': float(np.nanmin(tracking_results[valid_indices, 19])), 'max': float(np.nanmax(tracking_results[valid_indices, 19]))}
    track_group.attrs['summary_statistics'] = summary_stats
    track_group.attrs['duration_seconds'] = time.perf_counter() - start_time
    console.print(f"Tracking: [green]{successful_tracks}/{num_images}[/green] frames tracked ([cyan]{percent_tracked:.2f}%[/cyan]).")
    parent_group = root['tracking_runs']
    if 'best' not in parent_group.attrs or percent_tracked > parent_group.attrs['best']['percent_tracked']:
        console.print(f"[bold green]New best tracking run found! Success rate: {percent_tracked:.2f}%[/bold green]")
        parent_group.attrs['best'] = {'run_name': track_group.name, **summary_stats}

def main():
    # Instantiate Console
    console = Console()

    parser = argparse.ArgumentParser(description="Fish tracking pipeline with YOLO-ready zarr data generation.")
    parser.add_argument("zarr_path",
                        type=str,
                        help="Path to the output Zarr file."
                        )
    parser.add_argument("--video-path",
                        type=str,
                        help="Path to the input video file (required for 'import' or 'all' stages)."
                        )
    parser.add_argument("--config",
                        type=str,
                        default="src/pipeline_config.yaml",
                        help="Path to the pipeline configuration YAML file."
                        )
    parser.add_argument("--stage",
                        required=True,
                        choices=['import', 'background', 'crop', 'track', 'all'],
                        help="Processing stage to run. Choose from: 'import', 'background', 'crop', 'track', or 'all'."
                        )
    parser.add_argument("--scheduler",
                        default='processes',
                        choices=['processes', 'threads', 'single-thread'],
                        help="Dask scheduler."
                        )
    parser.add_argument("--num-workers",
                        type=int,
                        default=None,
                        help="Number of workers."
                        )
    parser.add_argument("--roi-thresh",
                        type=int,
                        help="Override roi_thresh in the config file."
                        )
    args = parser.parse_args()

    console.print(Panel("[bold cyan]Simple Fish Tracking Pipeline[/bold cyan]", 
                        subtitle="[yellow]Powered by Dask & Zarr![/yellow]", expand=False))

    if args.stage in ['import', 'all'] and not args.video_path:
        parser.error("--video-path is required for the 'import' and 'all' stages.")

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

    # Pass console object to all stages for rich outputs
    if args.stage in ['import', 'all']:
        run_import_stage_parallel_io(args.video_path, args.zarr_path, pipeline_params, cli_args_dict, console)
    if args.stage in ['background', 'all']:
        run_background_stage(args.zarr_path, pipeline_params, console)
    if args.stage in ['crop', 'all']:
        run_crop_stage(args.zarr_path, dask.config.get('scheduler'), pipeline_params, console)
    if args.stage in ['track', 'all']:
        run_tracking_stage(args.zarr_path, dask.config.get('scheduler'), pipeline_params, console)
        
    if args.stage == 'all':
        root = zarr.open_group(args.zarr_path, mode='a')
        root.attrs.update({'total_pipeline_duration_seconds': time.perf_counter() - overall_start_time, 'pipeline_version': '1.6'})
        latest_track_run = root['tracking_runs'].attrs['latest']
        successful_tracks = root[f'tracking_runs/{latest_track_run}'].attrs['summary_statistics']['frames_tracked']
        root.attrs['yolo_ready'] = successful_tracks > 0
        
        # Final Summary
        console.rule("[bold]Pipeline Complete[/bold]")
        if successful_tracks > 0:
            console.print(f"Pipeline completed! Total time: [bold green]{root.attrs['total_pipeline_duration_seconds']:.2f}[/bold green] seconds.")
            console.print(f"[cyan]Data is ready for YOLO training. Next step:[/] python zarr_yolo_dataset_bbox.py {args.zarr_path}")
        else:
            console.print(f"[bold yellow]Pipeline completed but no fish were tracked![/bold yellow]")
    
    console.print("\n[bold]Tracking complete![/bold]")

if __name__ == "__main__":
    main()