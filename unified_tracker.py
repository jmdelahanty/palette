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

# --- Stage-Specific Functions ---

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
def crop_chunk_delayed(zarr_path, chunk_slice, roi_sz, ds_thresh, se1, se4, use_gpu=False):
    """Dask delayed function to process a chunk of frames for cropping."""
    # Open zarr group within the delayed function
    root = zarr.open_group(zarr_path, mode='r')
    
    # Read all necessary data for the chunk in one go
    images_ds_chunk = root['raw_video/images_ds'][chunk_slice]
    images_full_chunk = root['raw_video/images_full'][chunk_slice]
    background_ds = root['background_models/background_ds'][:]
    
    chunk_len = images_ds_chunk.shape[0]
    # Prepare result arrays for this chunk
    chunk_rois = np.zeros((chunk_len, roi_sz[0], roi_sz[1]), dtype='uint8')
    chunk_coords = np.full((chunk_len, 2), -1, dtype='i4')
    chunk_bbox_norms = np.full((chunk_len, 2), np.nan, dtype='f8')

    if use_gpu and GPU_AVAILABLE:
        # GPU BATCH PROCESSING - process all frames in chunk at once
        try:
            # Move entire chunk to GPU
            images_ds_gpu = cp.asarray(images_ds_chunk)  # Shape: (chunk_len, H, W)
            background_ds_gpu = cp.asarray(background_ds)  # Shape: (H, W)
            
            # Broadcast background subtraction across all frames in chunk
            diff_ds_gpu = cp.clip(
                background_ds_gpu[None, :, :].astype(cp.int16) - images_ds_gpu.astype(cp.int16), 
                0, 255
            ).astype(cp.uint8)  # Shape: (chunk_len, H, W)
            
            # Create GPU morphological elements
            se1_gpu = cp.asarray(se1)
            se4_gpu = cp.asarray(se4)
            
            # Batch morphological operations on GPU
            # Process all frames simultaneously
            thresh_mask_gpu = diff_ds_gpu >= ds_thresh  # Shape: (chunk_len, H, W)
            
            # Apply erosion to all frames at once
            eroded_gpu = cp.zeros_like(thresh_mask_gpu)
            for i in range(chunk_len):
                eroded_gpu[i] = binary_erosion(thresh_mask_gpu[i], se1_gpu)
            
            # Apply dilation to all frames at once  
            dilated_gpu = cp.zeros_like(eroded_gpu)
            for i in range(chunk_len):
                dilated_gpu[i] = binary_dilation(eroded_gpu[i], se4_gpu)
            
            # Move results back to CPU for regionprops
            im_ds_batch = cp.asnumpy(dilated_gpu)
            
            # Process each frame for regionprops (CPU-only operation)
            for i in range(chunk_len):
                try:
                    ds_stat = regionprops(label(im_ds_batch[i]))
                    if not ds_stat: continue

                    ds_centroid_norm = np.array(ds_stat[0].centroid)[::-1] / np.array(images_ds_chunk.shape[1:])
                    full_centroid_px = np.round(ds_centroid_norm * np.array(images_full_chunk.shape[1:])).astype(int)
                    
                    x1 = full_centroid_px[0] - roi_sz[1] // 2
                    y1 = full_centroid_px[1] - roi_sz[0] // 2
                    
                    roi = images_full_chunk[i, y1:y1+roi_sz[0], x1:x1+roi_sz[1]]
                    if roi.shape != roi_sz: continue
                    
                    chunk_rois[i] = roi
                    chunk_coords[i] = (x1, y1)
                    chunk_bbox_norms[i] = ds_centroid_norm
                except Exception:
                    continue
                    
        except Exception as e:
            # Fallback to CPU if GPU processing fails
            print(f"GPU processing failed, falling back to CPU: {e}")
            use_gpu = False
    
    if not use_gpu:
        # CPU processing - simple and efficient frame by frame
        for i in range(chunk_len):
            try:
                img_ds = images_ds_chunk[i]
                diff_ds = np.clip(background_ds.astype(np.int16) - img_ds.astype(np.int16), 0, 255).astype(np.uint8)
                im_ds = erosion(diff_ds >= ds_thresh, se1)
                im_ds = dilation(im_ds, se4)
                
                ds_stat = regionprops(label(im_ds))
                if not ds_stat: continue

                ds_centroid_norm = np.array(ds_stat[0].centroid)[::-1] / np.array(images_ds_chunk.shape[1:])
                full_centroid_px = np.round(ds_centroid_norm * np.array(images_full_chunk.shape[1:])).astype(int)
                
                x1 = full_centroid_px[0] - roi_sz[1] // 2
                y1 = full_centroid_px[1] - roi_sz[0] // 2
                
                roi = images_full_chunk[i, y1:y1+roi_sz[0], x1:x1+roi_sz[1]]
                if roi.shape != roi_sz: continue
                
                chunk_rois[i] = roi
                chunk_coords[i] = (x1, y1)
                chunk_bbox_norms[i] = ds_centroid_norm
            except Exception:
                continue  # Skip frame on error
            
    return chunk_slice, chunk_rois, chunk_coords, chunk_bbox_norms

def run_crop_stage(zarr_path, scheduler_name, use_gpu=False):
    """Finds and saves ROIs using Dask with configurable scheduler."""
    gpu_info = f" + GPU" if use_gpu and GPU_AVAILABLE else ""
    print(f"--- Stage 3: Cropping ROIs (Dask {scheduler_name.title()} Scheduler{gpu_info}) ---")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'background_models' not in root: raise ValueError("Background stage not run.")

    images_ds = root['raw_video/images_ds']
    num_images = images_ds.shape[0]
    
    # Use larger chunk size for crop stage since we're processing smaller downsampled images
    # and writing smaller ROI outputs
    crop_chunk_size = min(32, num_images)  # Larger chunks for crop stage efficiency
    
    crop_group = root.require_group('crop_data')
    crop_group.attrs['crop_timestamp_utc'] = datetime.utcnow().isoformat()
    crop_group.attrs['dask_scheduler'] = scheduler_name
    crop_group.attrs['gpu_acceleration'] = use_gpu and GPU_AVAILABLE
    crop_group.attrs['processing_chunk_size'] = crop_chunk_size
    
    params = {'roi_sz': (320, 320), 'ds_thresh': 55, 'se1': disk(1), 'se4': disk(4)}
    serializable_params = {k: v for k, v in params.items() if not isinstance(v, np.ndarray)}
    serializable_params['morphology_disk_radii'] = {'se1': 1, 'se4': 4}
    crop_group.attrs['crop_parameters'] = serializable_params

    # Create output datasets with larger chunks for better performance
    roi_images = crop_group.create_dataset('roi_images', shape=(num_images, *params['roi_sz']), 
                                         chunks=(crop_chunk_size, None, None), dtype='uint8', overwrite=True)
    roi_coords = crop_group.create_dataset('roi_coordinates', shape=(num_images, 2), 
                                         chunks=(crop_chunk_size * 4, None), dtype='i4', overwrite=True)
    roi_coords[:] = -1  # Use -1 to indicate not found
    bbox_norm_coords = crop_group.create_dataset('bbox_norm_coords', shape=(num_images, 2), 
                                                chunks=(crop_chunk_size * 4, None), dtype='f8', overwrite=True)
    bbox_norm_coords[:] = np.nan

    # Create chunk slices based on optimal crop chunk size, not original video chunk size
    chunk_slices = [slice(i, min(i + crop_chunk_size, num_images)) for i in range(0, num_images, crop_chunk_size)]
    
    print(f"Creating {len(chunk_slices)} Dask tasks for cropping {num_images} frames (chunk size: {crop_chunk_size})...")
    
    # Create delayed tasks with GPU flag
    delayed_tasks = [crop_chunk_delayed(zarr_path, chunk_slice, use_gpu=use_gpu, **params) for chunk_slice in chunk_slices]
    
    # Compute with progress bar
    processing_mode = "GPU-accelerated" if use_gpu and GPU_AVAILABLE else "CPU"
    print(f"Executing Dask computation graph using '{scheduler_name}' scheduler ({processing_mode})...")
    with ProgressBar():
        results = dask.compute(*delayed_tasks)
    
    # Write results back to zarr
    print("Writing results to Zarr...")
    for slc, rois, coords, bboxes in tqdm(results, desc="Writing Chunks"):
        roi_images[slc] = rois
        roi_coords[slc] = coords
        bbox_norm_coords[slc] = bboxes
    
    end_time = time.perf_counter()
    crop_group.attrs['duration_seconds'] = end_time - start_time
    print(f"Crop stage completed in {end_time - start_time:.2f} seconds.")

@delayed
def track_chunk_delayed(zarr_path, chunk_slice, roi_sz, roi_thresh, se1, se2, use_gpu=False):
    """Dask delayed function to process a chunk of ROIs for tracking."""
    # Open zarr group within the delayed function
    root = zarr.open_group(zarr_path, mode='r')
    
    # Read all necessary data for the chunk in one go
    rois_chunk = root['crop_data/roi_images'][chunk_slice]
    coords_chunk = root['crop_data/roi_coordinates'][chunk_slice]
    background_full = root['background_models/background_full'][:]
    
    chunk_len = rois_chunk.shape[0]
    # Prepare result array for keypoint data for this chunk
    chunk_keypoints = np.full((chunk_len, 7), np.nan, dtype='f8')

    if use_gpu and GPU_AVAILABLE:
        # GPU BATCH PROCESSING - collect valid ROIs for batch processing
        try:
            valid_indices = []
            valid_rois = []
            valid_backgrounds = []
            
            # Collect valid ROIs and their background regions
            for i in range(chunk_len):
                roi = rois_chunk[i]
                if np.all(roi == 0): continue
                x1, y1 = coords_chunk[i]
                if x1 == -1: continue
                
                background_roi = background_full[y1:y1+roi.shape[0], x1:x1+roi.shape[1]]
                if background_roi.shape == roi.shape:
                    valid_indices.append(i)
                    valid_rois.append(roi)
                    valid_backgrounds.append(background_roi)
            
            if valid_indices:
                # Stack valid ROIs into batch arrays
                rois_batch = np.stack(valid_rois)  # Shape: (batch_size, H, W)
                backgrounds_batch = np.stack(valid_backgrounds)  # Shape: (batch_size, H, W)
                
                # Move to GPU for batch processing
                rois_gpu = cp.asarray(rois_batch)
                backgrounds_gpu = cp.asarray(backgrounds_batch)
                
                # Batch background subtraction
                diff_batch_gpu = cp.clip(
                    backgrounds_gpu.astype(cp.int16) - rois_gpu.astype(cp.int16), 
                    0, 255
                ).astype(cp.uint8)
                
                # Create GPU morphological elements
                se1_gpu = cp.asarray(se1)
                se2_gpu = cp.asarray(se2)
                
                # Batch morphological operations
                thresh_mask_gpu = diff_batch_gpu >= roi_thresh
                
                # Apply morphological operations to all ROIs in batch
                batch_size = len(valid_indices)
                eroded_gpu = cp.zeros_like(thresh_mask_gpu)
                dilated_gpu = cp.zeros_like(thresh_mask_gpu)
                final_gpu = cp.zeros_like(thresh_mask_gpu)
                
                for i in range(batch_size):
                    eroded_gpu[i] = binary_erosion(thresh_mask_gpu[i], se1_gpu)
                    dilated_gpu[i] = binary_dilation(eroded_gpu[i], se2_gpu)
                    final_gpu[i] = binary_erosion(dilated_gpu[i], se1_gpu)
                
                # Move results back to CPU for regionprops
                im_roi_batch = cp.asnumpy(final_gpu)
                
                # Process each valid ROI for keypoint detection (CPU-only operation)
                for batch_idx, chunk_idx in enumerate(valid_indices):
                    try:
                        im_roi = im_roi_batch[batch_idx]
                        
                        L = label(im_roi)
                        roi_stat = [r for r in regionprops(L) if r.area > 5]
                        if len(roi_stat) < 3: continue

                        # Sort by area and take top 3
                        keypoint_stats = sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]
                        pts = np.array([s.centroid[::-1] for s in keypoint_stats])
                        angles, _ = triangle_calculations(pts[0], pts[1], pts[2])
                        kp_idx = np.argsort(angles)

                        eye_mean = np.mean(pts[kp_idx[1:3]], axis=0)
                        head_vec = eye_mean - pts[kp_idx[0]]
                        heading = np.rad2deg(np.arctan2(-head_vec[1], head_vec[0]))
                        
                        # This logic is restored from the original combined script
                        R = np.array([[np.cos(np.deg2rad(heading)), -np.sin(np.deg2rad(heading))], [np.sin(np.deg2rad(heading)), np.cos(np.deg2rad(heading))]])
                        rotpts = (pts - eye_mean) @ R.T
                        
                        bladder_orig_idx, eye1_orig_idx, eye2_orig_idx = kp_idx[0], kp_idx[1], kp_idx[2]
                        eye_r_orig_idx, eye_l_orig_idx = (eye1_orig_idx, eye2_orig_idx) if rotpts[eye1_orig_idx, 1] > rotpts[eye2_orig_idx, 1] else (eye2_orig_idx, eye1_orig_idx)

                        bladder_pt_norm = keypoint_stats[bladder_orig_idx].centroid[::-1] / np.array(roi_sz)
                        eye_l_pt_norm = keypoint_stats[eye_l_orig_idx].centroid[::-1] / np.array(roi_sz)
                        eye_r_pt_norm = keypoint_stats[eye_r_orig_idx].centroid[::-1] / np.array(roi_sz)
                        
                        # Store all the keypoint data for this frame
                        chunk_keypoints[chunk_idx, :] = [
                            heading,
                            bladder_pt_norm[0], bladder_pt_norm[1],
                            eye_l_pt_norm[0], eye_l_pt_norm[1],
                            eye_r_pt_norm[0], eye_r_pt_norm[1],
                        ]
                    except Exception:
                        continue
                        
        except Exception as e:
            # Fallback to CPU if GPU processing fails
            print(f"GPU batch processing failed, falling back to CPU: {e}")
            use_gpu = False

    if not use_gpu:
        # CPU processing - simple and efficient frame by frame
        for i in range(chunk_len):
            try:
                roi = rois_chunk[i]
                if np.all(roi == 0): continue  # Skip if ROI is empty from crop stage

                x1, y1 = coords_chunk[i]
                if x1 == -1: continue  # Skip if ROI was not found

                background_roi = background_full[y1:y1+roi.shape[0], x1:x1+roi.shape[1]]
                diff_roi = np.clip(background_roi.astype(np.int16) - roi.astype(np.int16), 0, 255).astype(np.uint8)
                
                im_roi = erosion(diff_roi >= roi_thresh, se1)
                im_roi = dilation(im_roi, se2)
                im_roi = erosion(im_roi, se1)
                
                L = label(im_roi)
                roi_stat = [r for r in regionprops(L) if r.area > 5]
                if len(roi_stat) < 3: continue

                # Sort by area and take top 3
                keypoint_stats = sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]
                pts = np.array([s.centroid[::-1] for s in keypoint_stats])
                angles, _ = triangle_calculations(pts[0], pts[1], pts[2])
                kp_idx = np.argsort(angles)

                eye_mean = np.mean(pts[kp_idx[1:3]], axis=0)
                head_vec = eye_mean - pts[kp_idx[0]]
                heading = np.rad2deg(np.arctan2(-head_vec[1], head_vec[0]))
                
                # This logic is restored from the original combined script
                R = np.array([[np.cos(np.deg2rad(heading)), -np.sin(np.deg2rad(heading))], [np.sin(np.deg2rad(heading)), np.cos(np.deg2rad(heading))]])
                rotpts = (pts - eye_mean) @ R.T
                
                bladder_orig_idx, eye1_orig_idx, eye2_orig_idx = kp_idx[0], kp_idx[1], kp_idx[2]
                eye_r_orig_idx, eye_l_orig_idx = (eye1_orig_idx, eye2_orig_idx) if rotpts[eye1_orig_idx, 1] > rotpts[eye2_orig_idx, 1] else (eye2_orig_idx, eye1_orig_idx)

                bladder_pt_norm = keypoint_stats[bladder_orig_idx].centroid[::-1] / np.array(roi_sz)
                eye_l_pt_norm = keypoint_stats[eye_l_orig_idx].centroid[::-1] / np.array(roi_sz)
                eye_r_pt_norm = keypoint_stats[eye_r_orig_idx].centroid[::-1] / np.array(roi_sz)
                
                # Store all the keypoint data for this frame
                chunk_keypoints[i, :] = [
                    heading,
                    bladder_pt_norm[0], bladder_pt_norm[1],
                    eye_l_pt_norm[0], eye_l_pt_norm[1],
                    eye_r_pt_norm[0], eye_r_pt_norm[1],
                ]
            except Exception:
                continue
            
    return chunk_slice, chunk_keypoints

def run_tracking_stage(zarr_path, scheduler_name, use_gpu=False):
    """Runs keypoint tracking using Dask with configurable scheduler."""
    gpu_info = f" + GPU" if use_gpu and GPU_AVAILABLE else ""
    print(f"--- Stage 4: Tracking Keypoints (Dask {scheduler_name.title()} Scheduler{gpu_info}) ---")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'crop_data' not in root: raise ValueError("Crop stage not run.")
    
    roi_images = root['crop_data/roi_images']
    num_images = roi_images.shape[0]
    
    # Use the same chunk size as the crop stage for consistency
    track_chunk_size = roi_images.chunks[0] if roi_images.chunks else min(32, num_images)
    
    track_group = root.require_group('tracking')
    track_group.attrs['tracking_timestamp_utc'] = datetime.utcnow().isoformat()
    track_group.attrs['dask_scheduler'] = scheduler_name
    track_group.attrs['gpu_acceleration'] = use_gpu and GPU_AVAILABLE
    track_group.attrs['processing_chunk_size'] = track_chunk_size
    
    params = {'roi_sz': (320, 320), 'roi_thresh': 115, 'se1': disk(1), 'se2': disk(2)}
    serializable_params = {k: v for k, v in params.items() if not isinstance(v, np.ndarray)}
    serializable_params['morphology_disk_radii'] = {'se1': 1, 'se2': 2}
    track_group.attrs['tracking_parameters'] = serializable_params

    results_cols = 9
    tracking_results = track_group.create_dataset('tracking_results', shape=(num_images, results_cols), 
                                                chunks=(track_chunk_size * 4, None), dtype='f8', overwrite=True)
    tracking_results[:] = np.nan
    tracking_results.attrs['column_names'] = [
        'heading_degrees', 'bbox_x_norm', 'bbox_y_norm', 'bladder_x_roi_norm', 'bladder_y_roi_norm',
        'eye_l_x_roi_norm', 'eye_l_y_roi_norm', 'eye_r_x_roi_norm', 'eye_r_y_roi_norm'
    ]
    
    # Pre-load all bbox coordinates since it's a small array
    bbox_coords = root['crop_data/bbox_norm_coords'][:]

    # Create chunk slices based on the ROI chunk size
    chunk_slices = [slice(i, min(i + track_chunk_size, num_images)) for i in range(0, num_images, track_chunk_size)]
    
    print(f"Creating {len(chunk_slices)} Dask tasks for tracking {num_images} ROIs (chunk size: {track_chunk_size})...")
    
    # Create delayed tasks with GPU flag
    delayed_tasks = [track_chunk_delayed(zarr_path, chunk_slice, use_gpu=use_gpu, **params) for chunk_slice in chunk_slices]

    # Compute with progress bar
    processing_mode = "GPU-accelerated" if use_gpu and GPU_AVAILABLE else "CPU"
    print(f"Executing Dask computation graph using '{scheduler_name}' scheduler ({processing_mode})...")
    with ProgressBar():
        results = dask.compute(*delayed_tasks)

    # Write results back to zarr
    print("Writing results to Zarr...")
    for slc, keypoint_data in tqdm(results, desc="Writing Chunks"):
        # Combine the bbox data with the keypoint data from the worker
        combined_data = np.hstack([keypoint_data[:, 0:1], bbox_coords[slc], keypoint_data[:, 1:]])
        tracking_results[slc] = combined_data

    successful_tracks = np.count_nonzero(~np.isnan(tracking_results[:, 0]))
    percent_tracked = (successful_tracks / num_images) * 100
    track_group.attrs['summary_statistics'] = {'total_frames': num_images, 'frames_tracked': successful_tracks, 'percent_tracked': round(percent_tracked, 2)}
    
    end_time = time.perf_counter()
    track_group.attrs['duration_seconds'] = end_time - start_time
    print(f"Summary: {successful_tracks}/{num_images} frames tracked ({percent_tracked:.2f}%).")
    print(f"Tracking stage completed in {end_time - start_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(
        description="Modular, multi-stage fish tracking pipeline with Dask scheduler options.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Scheduler Options:
  processes     - Use multiprocessing for maximum CPU utilization (best for CPU-bound tasks)
  threads       - Use threading for better memory efficiency (best when I/O bound or limited RAM)
  single-thread - No parallelism, useful for debugging
  
Performance Guide:
  • Use 'processes' for best speed when you have plenty of RAM
  • Use 'threads' for better memory management on large datasets
  • Use 'single-thread' for debugging or profiling
  • Add --gpu for GPU acceleration of image processing operations
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

    # Configure Dask scheduler based on user choice
    scheduler_config = {'scheduler': args.scheduler}
    if args.num_workers is not None:
        scheduler_config['num_workers'] = args.num_workers
    
    # Apply the scheduler configuration
    dask.config.set(**scheduler_config)
    
    # Print scheduler info
    active_scheduler = dask.config.get('scheduler', 'threads')
    num_workers = dask.config.get('num_workers', os.cpu_count())
    gpu_status = " + GPU acceleration" if args.gpu and GPU_AVAILABLE else ""
    print(f"Using Dask '{active_scheduler}' scheduler with {num_workers} workers{gpu_status}")

    overall_start_time = time.perf_counter()

    if args.stage == 'import':
        run_import_stage(args.video_path, args.zarr_path)
    elif args.stage == 'background':
        run_background_stage(args.zarr_path)
    elif args.stage == 'crop':
        run_crop_stage(args.zarr_path, active_scheduler, args.gpu)
    elif args.stage == 'track':
        run_tracking_stage(args.zarr_path, active_scheduler, args.gpu)
    elif args.stage == 'all':
        run_import_stage(args.video_path, args.zarr_path)
        run_background_stage(args.zarr_path)
        run_crop_stage(args.zarr_path, active_scheduler, args.gpu)
        run_tracking_stage(args.zarr_path, active_scheduler, args.gpu)
        
        overall_end_time = time.perf_counter()
        total_elapsed = overall_end_time - overall_start_time
        root = zarr.open_group(args.zarr_path, mode='a')
        root.attrs['total_pipeline_duration_seconds'] = total_elapsed
        print(f"\nAll stages completed. Total pipeline time: {total_elapsed:.2f} seconds.")

    print("\nDone.")

if __name__ == "__main__":
    main()