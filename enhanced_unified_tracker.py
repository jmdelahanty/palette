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
from datetime import datetime
import platform
import socket
import skimage
import argparse
import subprocess
import decord
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Dask imports
import dask
import dask.array as da
from dask import delayed
from dask.diagnostics import ProgressBar

# GPU acceleration imports (optional)
try:
    import cupy as cp
    from cupyx.scipy.ndimage import binary_erosion, binary_dilation
    GPU_AVAILABLE = True
    # Only print GPU info in main process
    if __name__ == "__main__":
        device_id = cp.cuda.get_device_id()
        device_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode('utf-8')
        print(f"GPU acceleration available: {device_name}")
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# Prevent conflicts between threading libraries
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP conflicts
cv2.setNumThreads(0)  # Let scheduler handle OpenCV threading

# --- Utility Functions (unchanged) ---
def get_git_info():
    try:
        script_path = os.path.dirname(os.path.realpath(__file__))
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=script_path, stderr=subprocess.DEVNULL).strip().decode('utf-8')
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=script_path, stderr=subprocess.DEVNULL).strip().decode('utf-8')
        return {'commit_hash': commit_hash, 'is_dirty': bool(status)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit_hash': 'N/A', 'is_dirty': True, 'error': 'Not a git repository or git not found'}

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
    """
    Calculate bounding boxes from keypoint statistics with comprehensive coordinate systems.
    
    Args:
        keypoint_stats: List of regionprops stats for [bladder, eye_l, eye_r]
        roi_sz: ROI size tuple (height, width)
        margin_factor: Multiplier for keypoint spread
        min_bbox_size: Minimum bbox dimension
        
    Returns:
        dict: Comprehensive bounding box data
    """
    if len(keypoint_stats) < 3:
        return None
        
    # Extract keypoint positions in ROI pixel coordinates
    bladder_pos = np.array(keypoint_stats[0].centroid[::-1])  # (x, y)
    eye_l_pos = np.array(keypoint_stats[1].centroid[::-1])
    eye_r_pos = np.array(keypoint_stats[2].centroid[::-1])
    
    # Calculate fish center and extent in ROI pixels
    all_positions = np.array([bladder_pos, eye_l_pos, eye_r_pos])
    min_pos = np.min(all_positions, axis=0)
    max_pos = np.max(all_positions, axis=0)
    
    # Fish center in ROI pixels
    center_roi_px = (min_pos + max_pos) / 2.0
    
    # Fish extent with margin in ROI pixels
    extent_roi_px = (max_pos - min_pos) * margin_factor
    extent_roi_px = np.maximum(extent_roi_px, min_bbox_size * np.array(roi_sz[::-1]))  # Ensure minimum size
    
    # Convert to normalized ROI coordinates
    center_roi_norm = center_roi_px / np.array(roi_sz[::-1])  # (x/width, y/height)
    extent_roi_norm = extent_roi_px / np.array(roi_sz[::-1])
    
    # Individual keypoint normalized coordinates
    bladder_norm = bladder_pos / np.array(roi_sz[::-1])
    eye_l_norm = eye_l_pos / np.array(roi_sz[::-1])
    eye_r_norm = eye_r_pos / np.array(roi_sz[::-1])
    
    return {
        'center_roi_norm': center_roi_norm,      # (x, y) in ROI normalized coords
        'extent_roi_norm': extent_roi_norm,      # (width, height) in ROI normalized coords
        'bladder_roi_norm': bladder_norm,        # Individual keypoints
        'eye_l_roi_norm': eye_l_norm,
        'eye_r_roi_norm': eye_r_norm,
        'keypoint_count': len(keypoint_stats)
    }

def transform_bbox_to_image_scales(bbox_data, roi_coords_full, roi_coords_ds, roi_sz, 
                                 full_img_shape=(4512, 4512), ds_img_shape=(640, 640)):
    """
    Transform bounding box from ROI coordinates to multiple image scales.
    
    Returns:
        dict: Bounding box coordinates in all scales
    """
    if bbox_data is None:
        return None
        
    center_roi_norm = bbox_data['center_roi_norm']
    extent_roi_norm = bbox_data['extent_roi_norm']
    
    # Transform to full resolution (4512x4512)
    roi_x1_full, roi_y1_full = roi_coords_full
    if roi_x1_full != -1:  # Valid ROI
        center_full_px = np.array([roi_x1_full, roi_y1_full]) + center_roi_norm * np.array(roi_sz[::-1])
        extent_full_px = extent_roi_norm * np.array(roi_sz[::-1])
        
        center_full_norm = center_full_px / np.array(full_img_shape[::-1])
        extent_full_norm = extent_full_px / np.array(full_img_shape[::-1])
    else:
        center_full_norm = np.array([np.nan, np.nan])
        extent_full_norm = np.array([np.nan, np.nan])
    
    # Transform to downsampled resolution (640x640) 
    roi_x1_ds, roi_y1_ds = roi_coords_ds
    if roi_x1_ds != -1:  # Valid ROI
        # Scale ROI size to downsampled image
        scale_factor = ds_img_shape[0] / full_img_shape[0]  # 640/4512
        roi_sz_ds = np.array(roi_sz) * scale_factor
        
        center_ds_px = np.array([roi_x1_ds, roi_y1_ds]) + center_roi_norm * roi_sz_ds[::-1]
        extent_ds_px = extent_roi_norm * roi_sz_ds[::-1]
        
        center_ds_norm = center_ds_px / np.array(ds_img_shape[::-1])
        extent_ds_norm = extent_ds_px / np.array(ds_img_shape[::-1])
    else:
        center_ds_norm = np.array([np.nan, np.nan])
        extent_ds_norm = np.array([np.nan, np.nan])
    
    return {
        'full_scale': {
            'center_norm': center_full_norm,  # (x, y) normalized to 4512x4512
            'extent_norm': extent_full_norm   # (width, height) normalized to 4512x4512
        },
        'ds_scale': {
            'center_norm': center_ds_norm,    # (x, y) normalized to 640x640  
            'extent_norm': extent_ds_norm     # (width, height) normalized to 640x640
        },
        'roi_scale': {
            'center_norm': center_roi_norm,   # (x, y) normalized to ROI
            'extent_norm': extent_roi_norm    # (width, height) normalized to ROI
        }
    }

# YOLO dataset generation removed - use separate generate_yolo_dataset.py script for flexible dataset creation

# --- Stage-Specific Functions (Import and Background unchanged) ---

def run_import_stage(video_path, zarr_path):
    """Imports video, saving both full-res and downsampled versions with rich metadata."""
    print("--- Stage 1: Importing Video ---")
    start_time = time.perf_counter()
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(video_path, ctx=decord.gpu(0))
    n_frames, full_height, full_width = len(vr), vr[0].shape[0], vr[0].shape[1]
    ds_size = (640, 640)
    gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device='cuda:0')

    root = zarr.open_group(zarr_path, mode='w')
    root.attrs['import_command'] = " ".join(sys.argv)
    root.attrs['git_info'] = get_git_info()
    root.attrs['source_video_metadata'] = iio.immeta(video_path)
    root.attrs['platform_info'] = {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine(), 'hostname': socket.gethostname()}
    root.attrs['software_versions'] = {'python': platform.python_version(), 'numpy': np.__version__, 'zarr': zarr.__version__, 'scikit-image': skimage.__version__, 'opencv-python': cv2.__version__, 'torch': torch.__version__, 'decord': decord.__version__}

    raw_video_group = root.create_group('raw_video')
    raw_video_group.attrs['import_timestamp_utc'] = datetime.utcnow().isoformat()
    raw_video_group.attrs['gpu_info'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    raw_video_group.attrs['original_resolution'] = (full_height, full_width)
    raw_video_group.attrs['downsampled_resolution'] = ds_size
    
    # Use consistent smaller chunk sizes throughout pipeline for optimal performance
    chunk_size = 32
    
    images_full = raw_video_group.create_dataset('images_full', shape=(n_frames, full_height, full_width), 
                                                chunks=(chunk_size, None, None), dtype=np.uint8)
    images_ds = raw_video_group.create_dataset('images_ds', shape=(n_frames, ds_size[0], ds_size[1]), 
                                             chunks=(chunk_size, None, None), dtype=np.uint8)
    
    print(f"Importing {n_frames} frames and downsampling...")
    print(f"Using chunk size: {chunk_size}")
    
    batch_size = 32  # Keep original batch size to avoid GPU memory issues
    for i in tqdm(range(0, n_frames, batch_size), desc="Importing Video"):
        batch_gpu = vr.get_batch(range(i, min(i + batch_size, n_frames)))
        gray_batch_float_gpu = (batch_gpu.float() @ gray_weights).unsqueeze(1)
        images_full[i:i + len(batch_gpu)] = gray_batch_float_gpu.squeeze(1).byte().cpu().numpy()
        ds_batch_gpu = F.interpolate(gray_batch_float_gpu, size=ds_size, mode='bilinear', align_corners=False)
        images_ds[i:i + len(batch_gpu)] = ds_batch_gpu.squeeze(1).byte().cpu().numpy()

    end_time = time.perf_counter()
    raw_video_group.attrs['duration_seconds'] = end_time - start_time
    raw_video_group.attrs['chunk_size'] = chunk_size
    print(f"Import stage completed in {end_time - start_time:.2f} seconds.")

def run_background_stage(zarr_path):
    """Calculates and saves background models."""
    print("--- Stage 2: Calculating Background ---")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'raw_video' not in root: raise ValueError("Import stage not run.")
    
    num_images = root['raw_video/images_full'].shape[0]
    bg_group = root.require_group('background_models')
    bg_group.attrs['background_timestamp_utc'] = datetime.utcnow().isoformat()
    
    random.seed(42)
    random_indices = random.sample(range(num_images), min(100, num_images))
    
    print("Calculating background modes...")
    bg_group.create_dataset('background_full', data=fast_mode_bincount(root['raw_video/images_full'].get_orthogonal_selection((random_indices, slice(None), slice(None)))), overwrite=True)
    bg_group.create_dataset('background_ds', data=fast_mode_bincount(root['raw_video/images_ds'].get_orthogonal_selection((random_indices, slice(None), slice(None)))), overwrite=True)
    bg_group.attrs['source_frame_indices'] = random_indices
    
    end_time = time.perf_counter()
    bg_group.attrs['duration_seconds'] = end_time - start_time
    print(f"Background stage completed in {end_time - start_time:.2f} seconds.")

@delayed
def enhanced_crop_chunk_delayed(zarr_path, chunk_slice, roi_sz, ds_thresh, se1, se4, use_gpu=False):
    """Enhanced crop processing with multi-scale coordinate tracking."""
    root = zarr.open_group(zarr_path, mode='r')
    
    # Read data
    images_ds_chunk = root['raw_video/images_ds'][chunk_slice]
    images_full_chunk = root['raw_video/images_full'][chunk_slice]
    background_ds = root['background_models/background_ds'][:]
    
    # Get image shapes
    full_img_shape = images_full_chunk.shape[1:]  # (4512, 4512)
    ds_img_shape = images_ds_chunk.shape[1:]      # (640, 640)
    
    chunk_len = images_ds_chunk.shape[0]
    
    # Result arrays
    chunk_rois = np.zeros((chunk_len, roi_sz[0], roi_sz[1]), dtype='uint8')
    chunk_coords_full = np.full((chunk_len, 2), -1, dtype='i4')    # ROI position in full image
    chunk_coords_ds = np.full((chunk_len, 2), -1, dtype='i4')      # ROI position in downsampled image
    chunk_bbox_norms = np.full((chunk_len, 2), np.nan, dtype='f8') # Fish center in downsampled normalized
    
    # Process each frame (simplified CPU version for clarity)
    for i in range(chunk_len):
        try:
            img_ds = images_ds_chunk[i]
            img_full = images_full_chunk[i]
            
            # Fish detection in downsampled image
            diff_ds = np.clip(background_ds.astype(np.int16) - img_ds.astype(np.int16), 0, 255).astype(np.uint8)
            im_ds = erosion(diff_ds >= ds_thresh, se1)
            im_ds = dilation(im_ds, se4)
            
            ds_stat = regionprops(label(im_ds))
            if not ds_stat: continue
            
            # Fish center in downsampled image
            ds_centroid_norm = np.array(ds_stat[0].centroid)[::-1] / np.array(ds_img_shape)
            
            # Calculate ROI positions for both scales
            # Full image ROI position
            full_centroid_px = np.round(ds_centroid_norm * np.array(full_img_shape)).astype(int)
            roi_x1_full = full_centroid_px[0] - roi_sz[1] // 2
            roi_y1_full = full_centroid_px[1] - roi_sz[0] // 2
            
            # Downsampled image ROI position (scaled)
            ds_centroid_px = np.round(ds_centroid_norm * np.array(ds_img_shape)).astype(int)
            roi_size_ds = np.array(roi_sz) * (ds_img_shape[0] / full_img_shape[0])  # Scale ROI size
            roi_x1_ds = ds_centroid_px[0] - roi_size_ds[1] // 2
            roi_y1_ds = ds_centroid_px[1] - roi_size_ds[0] // 2
            
            # Extract ROI from full image
            roi = img_full[roi_y1_full:roi_y1_full+roi_sz[0], roi_x1_full:roi_x1_full+roi_sz[1]]
            if roi.shape != roi_sz: continue
            
            # Store results
            chunk_rois[i] = roi
            chunk_coords_full[i] = (roi_x1_full, roi_y1_full)
            chunk_coords_ds[i] = (roi_x1_ds, roi_y1_ds)
            chunk_bbox_norms[i] = ds_centroid_norm
            
        except Exception:
            continue
            
    return chunk_slice, chunk_rois, chunk_coords_full, chunk_coords_ds, chunk_bbox_norms

def run_enhanced_crop_stage(zarr_path, scheduler_name, use_gpu=False):
    """Enhanced crop stage with multi-scale coordinate tracking."""
    gpu_info = f" + GPU" if use_gpu and GPU_AVAILABLE else ""
    print(f"--- Stage 3: Enhanced Cropping with Multi-Scale Coordinates (Dask {scheduler_name.title()} Scheduler{gpu_info}) ---")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'background_models' not in root: raise ValueError("Background stage not run.")

    images_ds = root['raw_video/images_ds']
    num_images = images_ds.shape[0]
    
    crop_chunk_size = min(32, num_images)
    
    crop_group = root.require_group('crop_data')
    crop_group.attrs['crop_timestamp_utc'] = datetime.utcnow().isoformat()
    crop_group.attrs['dask_scheduler'] = scheduler_name
    crop_group.attrs['gpu_acceleration'] = use_gpu and GPU_AVAILABLE
    crop_group.attrs['processing_chunk_size'] = crop_chunk_size
    crop_group.attrs['enhanced_features'] = ['multi_scale_coordinates', 'yolo_ready_data']
    
    params = {'roi_sz': (320, 320), 'ds_thresh': 55, 'se1': disk(1), 'se4': disk(4)}
    serializable_params = {k: v for k, v in params.items() if not isinstance(v, np.ndarray)}
    serializable_params['morphology_disk_radii'] = {'se1': 1, 'se4': 4}
    crop_group.attrs['crop_parameters'] = serializable_params

    # Create enhanced output datasets
    roi_images = crop_group.create_dataset('roi_images', shape=(num_images, *params['roi_sz']), 
                                         chunks=(crop_chunk_size, None, None), dtype='uint8', overwrite=True)
    
    # Multi-scale coordinate storage
    roi_coordinates_full = crop_group.create_dataset('roi_coordinates_full', shape=(num_images, 2), 
                                                   chunks=(crop_chunk_size * 4, None), dtype='i4', overwrite=True)
    roi_coordinates_ds = crop_group.create_dataset('roi_coordinates_ds', shape=(num_images, 2), 
                                                 chunks=(crop_chunk_size * 4, None), dtype='i4', overwrite=True)
    bbox_norm_coords = crop_group.create_dataset('bbox_norm_coords', shape=(num_images, 2), 
                                                chunks=(crop_chunk_size * 4, None), dtype='f8', overwrite=True)
    
    # Initialize with invalid markers
    roi_coordinates_full[:] = -1
    roi_coordinates_ds[:] = -1
    bbox_norm_coords[:] = np.nan

    # Create chunk slices
    chunk_slices = [slice(i, min(i + crop_chunk_size, num_images)) for i in range(0, num_images, crop_chunk_size)]
    
    print(f"Creating {len(chunk_slices)} enhanced Dask tasks...")
    
    # Create delayed tasks
    delayed_tasks = [enhanced_crop_chunk_delayed(zarr_path, chunk_slice, use_gpu=use_gpu, **params) for chunk_slice in chunk_slices]
    
    # Compute with progress bar
    processing_mode = "GPU-accelerated" if use_gpu and GPU_AVAILABLE else "CPU"
    print(f"Executing enhanced Dask computation ({processing_mode})...")
    with ProgressBar():
        results = dask.compute(*delayed_tasks)
    
    # Write results back to zarr
    print("Writing enhanced results to Zarr...")
    for slc, rois, coords_full, coords_ds, bboxes in tqdm(results, desc="Writing Enhanced Chunks"):
        roi_images[slc] = rois
        roi_coordinates_full[slc] = coords_full
        roi_coordinates_ds[slc] = coords_ds
        bbox_norm_coords[slc] = bboxes
    
    end_time = time.perf_counter()
    crop_group.attrs['duration_seconds'] = end_time - start_time
    print(f"Enhanced crop stage completed in {end_time - start_time:.2f} seconds.")

@delayed
def enhanced_track_chunk_delayed(zarr_path, chunk_slice, roi_sz, roi_thresh, se1, se2, use_gpu=False):
    """Enhanced tracking with comprehensive coordinate systems."""
    root = zarr.open_group(zarr_path, mode='r')
    
    # Read data
    rois_chunk = root['crop_data/roi_images'][chunk_slice]
    coords_full_chunk = root['crop_data/roi_coordinates_full'][chunk_slice]
    coords_ds_chunk = root['crop_data/roi_coordinates_ds'][chunk_slice]
    background_full = root['background_models/background_full'][:]
    
    chunk_len = rois_chunk.shape[0]
    
    # Enhanced result array - more columns for comprehensive data
    # Format: [heading, bladder_x_roi, bladder_y_roi, eye_l_x_roi, eye_l_y_roi, eye_r_x_roi, eye_r_y_roi,
    #          bbox_x_ds, bbox_y_ds, bbox_w_ds, bbox_h_ds, bbox_x_full, bbox_y_full, bbox_w_full, bbox_h_full,
    #          roi_x1_full, roi_y1_full, roi_x1_ds, roi_y1_ds, confidence]
    chunk_results = np.full((chunk_len, 20), np.nan, dtype='f8')
    
    # Process each frame
    for i in range(chunk_len):
        try:
            roi = rois_chunk[i]
            if np.all(roi == 0): continue  # Skip empty ROIs
            
            coords_full = coords_full_chunk[i]
            coords_ds = coords_ds_chunk[i]
            
            if coords_full[0] == -1: continue  # Skip invalid ROIs
            
            # Get background region
            x1_full, y1_full = coords_full
            background_roi = background_full[y1_full:y1_full+roi.shape[0], x1_full:x1_full+roi.shape[1]]
            if background_roi.shape != roi.shape: continue
            
            # Fish detection in ROI
            diff_roi = np.clip(background_roi.astype(np.int16) - roi.astype(np.int16), 0, 255).astype(np.uint8)
            im_roi = erosion(diff_roi >= roi_thresh, se1)
            im_roi = dilation(im_roi, se2)
            im_roi = erosion(im_roi, se1)
            
            # Find keypoints
            L = label(im_roi)
            roi_stat = [r for r in regionprops(L) if r.area > 5]
            if len(roi_stat) < 3: continue

            # Sort by area and take top 3
            keypoint_stats = sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]
            pts = np.array([s.centroid[::-1] for s in keypoint_stats])
            angles, _ = triangle_calculations(pts[0], pts[1], pts[2])
            kp_idx = np.argsort(angles)

            # Calculate heading
            eye_mean = np.mean(pts[kp_idx[1:3]], axis=0)
            head_vec = eye_mean - pts[kp_idx[0]]
            heading = np.rad2deg(np.arctan2(-head_vec[1], head_vec[0]))
            
            # Determine eye assignments
            R = np.array([[np.cos(np.deg2rad(heading)), -np.sin(np.deg2rad(heading))], 
                         [np.sin(np.deg2rad(heading)), np.cos(np.deg2rad(heading))]])
            rotpts = (pts - eye_mean) @ R.T
            
            bladder_orig_idx, eye1_orig_idx, eye2_orig_idx = kp_idx[0], kp_idx[1], kp_idx[2]
            eye_r_orig_idx, eye_l_orig_idx = (eye1_orig_idx, eye2_orig_idx) if rotpts[eye1_orig_idx, 1] > rotpts[eye2_orig_idx, 1] else (eye2_orig_idx, eye1_orig_idx)

            # Get ordered keypoint stats
            ordered_stats = [keypoint_stats[bladder_orig_idx], keypoint_stats[eye_l_orig_idx], keypoint_stats[eye_r_orig_idx]]
            
            # Calculate comprehensive bounding box data
            bbox_data = calculate_multi_scale_bounding_boxes(ordered_stats, roi_sz)
            if bbox_data is None: continue
            
            # Transform to image scales
            multi_scale_data = transform_bbox_to_image_scales(
                bbox_data, coords_full, coords_ds, roi_sz,
                full_img_shape=(4512, 4512), ds_img_shape=(640, 640)
            )
            if multi_scale_data is None: continue
            
            # Confidence score based on keypoint quality
            areas = [s.area for s in ordered_stats]
            confidence = min(1.0, np.mean(areas) / 100.0)  # Simple confidence metric
            
            # Store comprehensive results
            chunk_results[i, :] = [
                heading,                                                    # 0: heading_degrees
                bbox_data['bladder_roi_norm'][0], bbox_data['bladder_roi_norm'][1],  # 1-2: bladder_x/y_roi_norm
                bbox_data['eye_l_roi_norm'][0], bbox_data['eye_l_roi_norm'][1],      # 3-4: eye_l_x/y_roi_norm
                bbox_data['eye_r_roi_norm'][0], bbox_data['eye_r_roi_norm'][1],      # 5-6: eye_r_x/y_roi_norm
                multi_scale_data['ds_scale']['center_norm'][0], multi_scale_data['ds_scale']['center_norm'][1],      # 7-8: bbox_x/y_norm_ds
                multi_scale_data['ds_scale']['extent_norm'][0], multi_scale_data['ds_scale']['extent_norm'][1],      # 9-10: bbox_w/h_norm_ds
                multi_scale_data['full_scale']['center_norm'][0], multi_scale_data['full_scale']['center_norm'][1],  # 11-12: bbox_x/y_norm_full
                multi_scale_data['full_scale']['extent_norm'][0], multi_scale_data['full_scale']['extent_norm'][1],  # 13-14: bbox_w/h_norm_full
                coords_full[0], coords_full[1],                             # 15-16: roi_x1/y1_full
                coords_ds[0], coords_ds[1],                                 # 17-18: roi_x1/y1_ds
                confidence                                                  # 19: confidence_score
            ]
            
        except Exception:
            continue
            
    return chunk_slice, chunk_results

def run_enhanced_tracking_stage(zarr_path, scheduler_name, use_gpu=False):
    """Enhanced tracking with comprehensive coordinate systems and YOLO-ready data."""
    gpu_info = f" + GPU" if use_gpu and GPU_AVAILABLE else ""
    print(f"--- Stage 4: Enhanced Tracking with Multi-Scale Data (Dask {scheduler_name.title()} Scheduler{gpu_info}) ---")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'crop_data' not in root: raise ValueError("Crop stage not run.")
    
    roi_images = root['crop_data/roi_images']
    num_images = roi_images.shape[0]
    
    track_chunk_size = roi_images.chunks[0] if roi_images.chunks else min(32, num_images)
    
    track_group = root.require_group('tracking')
    track_group.attrs['tracking_timestamp_utc'] = datetime.utcnow().isoformat()
    track_group.attrs['dask_scheduler'] = scheduler_name
    track_group.attrs['gpu_acceleration'] = use_gpu and GPU_AVAILABLE
    track_group.attrs['processing_chunk_size'] = track_chunk_size
    track_group.attrs['enhanced_features'] = ['multi_scale_coordinates', 'yolo_ready_data', 'comprehensive_provenance']
    
    params = {'roi_sz': (320, 320), 'roi_thresh': 115, 'se1': disk(1), 'se2': disk(2)}
    serializable_params = {k: v for k, v in params.items() if not isinstance(v, np.ndarray)}
    serializable_params['morphology_disk_radii'] = {'se1': 1, 'se2': 2}
    track_group.attrs['tracking_parameters'] = serializable_params

    # Enhanced results with comprehensive column structure
    results_cols = 20
    tracking_results = track_group.create_dataset('tracking_results', shape=(num_images, results_cols), 
                                                chunks=(track_chunk_size * 4, None), dtype='f8', overwrite=True)
    tracking_results[:] = np.nan
    
    # Comprehensive column names
    tracking_results.attrs['column_names'] = [
        'heading_degrees',                           # 0
        'bladder_x_roi_norm', 'bladder_y_roi_norm', # 1-2: keypoints in ROI
        'eye_l_x_roi_norm', 'eye_l_y_roi_norm',     # 3-4
        'eye_r_x_roi_norm', 'eye_r_y_roi_norm',     # 5-6
        'bbox_x_norm_ds', 'bbox_y_norm_ds',         # 7-8: YOLO-ready 640x640 coords
        'bbox_width_norm_ds', 'bbox_height_norm_ds', # 9-10: YOLO-ready 640x640 dims
        'bbox_x_norm_full', 'bbox_y_norm_full',     # 11-12: full resolution coords
        'bbox_width_norm_full', 'bbox_height_norm_full', # 13-14: full resolution dims
        'roi_x1_full', 'roi_y1_full',               # 15-16: ROI position in full image
        'roi_x1_ds', 'roi_y1_ds',                   # 17-18: ROI position in downsampled
        'confidence_score'                          # 19: detection confidence
    ]
    
    # Additional metadata for coordinate systems
    coord_systems = track_group.create_group('coordinate_systems')
    coord_systems.attrs['roi_normalized'] = {
        'description': 'Coordinates normalized to ROI size (320x320)',
        'range': [0.0, 1.0],
        'columns': ['bladder_x_roi_norm', 'bladder_y_roi_norm', 'eye_l_x_roi_norm', 'eye_l_y_roi_norm', 'eye_r_x_roi_norm', 'eye_r_y_roi_norm']
    }
    coord_systems.attrs['downsampled_normalized'] = {
        'description': 'Coordinates normalized to downsampled image (640x640) - YOLO ready',
        'range': [0.0, 1.0],
        'columns': ['bbox_x_norm_ds', 'bbox_y_norm_ds', 'bbox_width_norm_ds', 'bbox_height_norm_ds']
    }
    coord_systems.attrs['full_normalized'] = {
        'description': 'Coordinates normalized to full resolution image (4512x4512)',
        'range': [0.0, 1.0],
        'columns': ['bbox_x_norm_full', 'bbox_y_norm_full', 'bbox_width_norm_full', 'bbox_height_norm_full']
    }
    coord_systems.attrs['pixel_coordinates'] = {
        'description': 'ROI positions in pixel coordinates',
        'columns': ['roi_x1_full', 'roi_y1_full', 'roi_x1_ds', 'roi_y1_ds']
    }

    # Create chunk slices
    chunk_slices = [slice(i, min(i + track_chunk_size, num_images)) for i in range(0, num_images, track_chunk_size)]
    
    print(f"Creating {len(chunk_slices)} enhanced tracking tasks...")
    
    # Create delayed tasks
    delayed_tasks = [enhanced_track_chunk_delayed(zarr_path, chunk_slice, use_gpu=use_gpu, **params) for chunk_slice in chunk_slices]

    # Compute with progress bar
    processing_mode = "GPU-accelerated" if use_gpu and GPU_AVAILABLE else "CPU"
    print(f"Executing enhanced tracking computation ({processing_mode})...")
    with ProgressBar():
        results = dask.compute(*delayed_tasks)

    # Write results back to zarr
    print("Writing enhanced tracking results to Zarr...")
    for slc, chunk_results in tqdm(results, desc="Writing Enhanced Tracking"):
        tracking_results[slc] = chunk_results

    # Calculate comprehensive statistics
    valid_frames = ~np.isnan(tracking_results[:, 0])  # Non-NaN heading
    valid_indices = np.where(valid_frames)[0]
    
    successful_tracks = len(valid_indices)
    percent_tracked = (successful_tracks / num_images) * 100
    
    # Enhanced statistics
    track_group.attrs['summary_statistics'] = {
        'total_frames': num_images, 
        'frames_tracked': successful_tracks, 
        'percent_tracked': round(percent_tracked, 2),
        'coordinate_systems': 4,  # ROI, DS, Full, Pixel
        'keypoints_per_frame': 3,
        'confidence_stats': {
            'mean': float(np.nanmean(tracking_results[valid_indices, 19])),
            'std': float(np.nanstd(tracking_results[valid_indices, 19])),
            'min': float(np.nanmin(tracking_results[valid_indices, 19])),
            'max': float(np.nanmax(tracking_results[valid_indices, 19]))
        }
    }
    
    end_time = time.perf_counter()
    track_group.attrs['duration_seconds'] = end_time - start_time
    print(f"Enhanced tracking: {successful_tracks}/{num_images} frames tracked ({percent_tracked:.2f}%).")
    print(f"Enhanced tracking stage completed in {end_time - start_time:.2f} seconds.")
    print(f"💡 Use generate_yolo_dataset.py to create flexible training datasets from this tracking data")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced fish tracking pipeline with comprehensive YOLO-ready data generation.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Enhanced Features:
  • Multi-scale coordinate systems (ROI, 640x640, 4512x4512)
  • Comprehensive bounding box data with proper transformations
  • Complete provenance tracking for all coordinate transformations
  • Ready for flexible YOLO dataset generation (use separate script)
  • Enhanced metadata and audit trails
        """)
    
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("zarr_path", type=str, help="Path to the output Zarr file.")
    parser.add_argument("--stage", required=True, choices=['import', 'background', 'crop', 'track', 'all'], 
                       help="Processing stage to run.")
    parser.add_argument("--scheduler", default='processes', 
                       choices=['processes', 'threads', 'single-thread'],
                       help="Dask scheduler to use (default: processes)")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of workers (default: number of CPU cores)")
    parser.add_argument("--gpu", action='store_true',
                       help="Use GPU acceleration for image processing (requires CuPy)")
    
    args = parser.parse_args()

    # Check GPU requirements
    if args.gpu and not GPU_AVAILABLE:
        print("Warning: GPU acceleration requested but CuPy is not available. Install with: pip install cupy")
        print("Falling back to CPU processing.")
        args.gpu = False

    # Configure Dask scheduler
    scheduler_config = {'scheduler': args.scheduler}
    if args.num_workers is not None:
        scheduler_config['num_workers'] = args.num_workers
    
    dask.config.set(**scheduler_config)
    
    # Print enhanced info
    active_scheduler = dask.config.get('scheduler', 'threads')
    num_workers = dask.config.get('num_workers', os.cpu_count())
    gpu_status = " + GPU acceleration" if args.gpu and GPU_AVAILABLE else ""
    print(f"🚀 Enhanced Fish Tracking Pipeline")
    print(f"Using Dask '{active_scheduler}' scheduler with {num_workers} workers{gpu_status}")
    print(f"Features: Multi-scale coordinates, comprehensive data, flexible YOLO compatibility")

    overall_start_time = time.perf_counter()

    if args.stage == 'import':
        run_import_stage(args.video_path, args.zarr_path)
    elif args.stage == 'background':
        run_background_stage(args.zarr_path)
    elif args.stage == 'crop':
        run_enhanced_crop_stage(args.zarr_path, active_scheduler, args.gpu)
    elif args.stage == 'track':
        run_enhanced_tracking_stage(args.zarr_path, active_scheduler, args.gpu)
    elif args.stage == 'all':
        run_import_stage(args.video_path, args.zarr_path)
        run_background_stage(args.zarr_path)
        run_enhanced_crop_stage(args.zarr_path, active_scheduler, args.gpu)
        run_enhanced_tracking_stage(args.zarr_path, active_scheduler, args.gpu)
        
        overall_end_time = time.perf_counter()
        total_elapsed = overall_end_time - overall_start_time
        root = zarr.open_group(args.zarr_path, mode='a')
        root.attrs['total_pipeline_duration_seconds'] = total_elapsed
        root.attrs['enhanced_pipeline_version'] = '2.1'  # Updated version
        root.attrs['yolo_ready'] = True  # Data is ready for YOLO, but not pre-generated
        print(f"\n🎉 Enhanced pipeline completed! Total time: {total_elapsed:.2f} seconds.")
        print(f"✅ Data is ready for YOLO training (use separate generate_yolo_dataset.py script)")
        print(f"💡 Next step: python generate_yolo_dataset.py {args.zarr_path} --split 0.8")

    print("\n✨ Enhanced tracking complete!")

if __name__ == "__main__":
    main()