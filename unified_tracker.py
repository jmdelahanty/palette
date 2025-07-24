import os
import zarr
import imageio.v3 as iio
import numpy as np
import cv2
from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops
import random
import multiprocessing
from functools import partial
from tqdm import tqdm
import argparse
import subprocess
import decord
import torch
import torch.nn.functional as F
import time
import sys
from datetime import datetime
import platform
import socket
import skimage

# Prevent deadlocks from nested parallelism
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
cv2.setNumThreads(0)

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

# --- Worker Initializer ---
def init_worker(zarr_path):
    """Initializes the multiprocessing worker with a read-only Zarr group."""
    global worker_root
    worker_root = zarr.open_group(zarr_path, mode='r')

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
    
    # Using larger chunks for the frame axis to improve read performance later
    images_full = raw_video_group.create_dataset('images_full', shape=(n_frames, full_height, full_width), chunks=(32, None, None), dtype=np.uint8)
    images_ds = raw_video_group.create_dataset('images_ds', shape=(n_frames, ds_size[0], ds_size[1]), chunks=(32, None, None), dtype=np.uint8)
    
    print(f"Importing {n_frames} frames and downsampling...")
    batch_size = 32 
    for i in tqdm(range(0, n_frames, batch_size), desc="Importing Video"):
        batch_gpu = vr.get_batch(range(i, min(i + batch_size, n_frames)))
        gray_batch_float_gpu = (batch_gpu.float() @ gray_weights).unsqueeze(1)
        images_full[i:i + len(batch_gpu)] = gray_batch_float_gpu.squeeze(1).byte().cpu().numpy()
        ds_batch_gpu = F.interpolate(gray_batch_float_gpu, size=ds_size, mode='bilinear', align_corners=False)
        images_ds[i:i + len(batch_gpu)] = ds_batch_gpu.squeeze(1).byte().cpu().numpy()

    end_time = time.perf_counter()
    raw_video_group.attrs['duration_seconds'] = end_time - start_time
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

def crop_chunk_worker(chunk_slice, roi_sz, ds_thresh, se1, se4):
    """Worker to find and crop ROIs for an entire chunk of frames."""
    # Read all necessary data for the chunk in one go
    images_ds_chunk = worker_root['raw_video/images_ds'][chunk_slice]
    images_full_chunk = worker_root['raw_video/images_full'][chunk_slice]
    background_ds = worker_root['background_models/background_ds'][:]
    
    chunk_len = images_ds_chunk.shape[0]
    # Prepare result arrays for this chunk
    chunk_rois = np.zeros((chunk_len, roi_sz[0], roi_sz[1]), dtype='uint8')
    chunk_coords = np.full((chunk_len, 2), -1, dtype='i4')
    chunk_bbox_norms = np.full((chunk_len, 2), np.nan, dtype='f8')

    # Loop internally over frames in the chunk
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

def run_crop_stage(zarr_path):
    """Finds and saves ROIs using chunk-aware parallel processing."""
    print("--- Stage 3: Cropping ROIs (Chunk-Aware) ---")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'background_models' not in root: raise ValueError("Background stage not run.")

    images_ds = root['raw_video/images_ds']
    num_images, chunk_size = images_ds.shape[0], images_ds.chunks[0]
    
    crop_group = root.require_group('crop_data')
    crop_group.attrs['crop_timestamp_utc'] = datetime.utcnow().isoformat()
    
    params = {'roi_sz': (320, 320), 'ds_thresh': 55, 'se1': disk(1), 'se4': disk(4)}
    serializable_params = {k: v for k, v in params.items() if not isinstance(v, np.ndarray)}
    serializable_params['morphology_disk_radii'] = {'se1': 1, 'se4': 4}
    crop_group.attrs['crop_parameters'] = serializable_params

    roi_images = crop_group.create_dataset('roi_images', shape=(num_images, *params['roi_sz']), chunks=(chunk_size, None, None), dtype='uint8', overwrite=True)
    roi_coords = crop_group.create_dataset('roi_coordinates', shape=(num_images, 2), chunks=(chunk_size * 4, None), dtype='i4', overwrite=True)
    roi_coords[:] = -1  # Use -1 to indicate not found
    
    # NEW: Dataset for normalized bounding box coordinates from the downsampled image
    bbox_norm_coords = crop_group.create_dataset('bbox_norm_coords', shape=(num_images, 2), chunks=(chunk_size * 4, None), dtype='f8', overwrite=True)
    bbox_norm_coords[:] = np.nan

    # Create a list of chunk slices to distribute to workers
    chunk_slices = [slice(i, min(i + chunk_size, num_images)) for i in range(0, num_images, chunk_size)]
    
    worker_func = partial(crop_chunk_worker, **params)
    print(f"Cropping {num_images} frames in {len(chunk_slices)} chunks using {os.cpu_count()} CPU cores...")
    with multiprocessing.Pool(initializer=init_worker, initargs=(zarr_path,)) as pool:
        pbar = tqdm(pool.imap_unordered(worker_func, chunk_slices), total=len(chunk_slices), desc="Cropping Chunks")
        for slc, rois, coords, bboxes in pbar:
            # Write the entire chunk of results at once
            roi_images[slc] = rois
            roi_coords[slc] = coords
            bbox_norm_coords[slc] = bboxes
    
    end_time = time.perf_counter()
    crop_group.attrs['duration_seconds'] = end_time - start_time
    print(f"Crop stage completed in {end_time - start_time:.2f} seconds.")

def track_chunk_worker(chunk_slice, roi_sz, roi_thresh, se1, se2):
    """Worker to find keypoints for an entire chunk of ROIs."""
    # Read all necessary data for the chunk in one go
    rois_chunk = worker_root['crop_data/roi_images'][chunk_slice]
    coords_chunk = worker_root['crop_data/roi_coordinates'][chunk_slice]
    background_full = worker_root['background_models/background_full'][:]
    
    chunk_len = rois_chunk.shape[0]
    # Prepare result array for keypoint data for this chunk
    chunk_keypoints = np.full((chunk_len, 7), np.nan, dtype='f8')

    # Loop internally over frames in the chunk
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

def run_tracking_stage(zarr_path):
    """Runs keypoint tracking using chunk-aware parallel processing."""
    print("--- Stage 4: Tracking Keypoints (Chunk-Aware) ---")
    start_time = time.perf_counter()
    root = zarr.open_group(zarr_path, mode='a')
    if 'crop_data' not in root: raise ValueError("Crop stage not run.")
    
    roi_images = root['crop_data/roi_images']
    num_images, chunk_size = roi_images.shape[0], roi_images.chunks[0]
    
    track_group = root.require_group('tracking')
    track_group.attrs['tracking_timestamp_utc'] = datetime.utcnow().isoformat()
    
    params = {'roi_sz': (320, 320), 'roi_thresh': 115, 'se1': disk(1), 'se2': disk(2)}
    serializable_params = {k: v for k, v in params.items() if not isinstance(v, np.ndarray)}
    serializable_params['morphology_disk_radii'] = {'se1': 1, 'se2': 2}
    track_group.attrs['tracking_parameters'] = serializable_params

    results_cols = 9
    tracking_results = track_group.create_dataset('tracking_results', shape=(num_images, results_cols), chunks=(chunk_size * 4, None), dtype='f8', overwrite=True)
    tracking_results[:] = np.nan
    tracking_results.attrs['column_names'] = [
        'heading_degrees', 'bbox_x_norm', 'bbox_y_norm', 'bladder_x_roi_norm', 'bladder_y_roi_norm',
        'eye_l_x_roi_norm', 'eye_l_y_roi_norm', 'eye_r_x_roi_norm', 'eye_r_y_roi_norm'
    ]
    
    # Pre-load all bbox coordinates since it's a small array
    bbox_coords = root['crop_data/bbox_norm_coords'][:]

    # Create a list of chunk slices to distribute to workers
    chunk_slices = [slice(i, min(i + chunk_size, num_images)) for i in range(0, num_images, chunk_size)]

    worker_func = partial(track_chunk_worker, **params)
    print(f"Tracking {num_images} ROIs in {len(chunk_slices)} chunks using {os.cpu_count()} CPU cores...")
    with multiprocessing.Pool(initializer=init_worker, initargs=(zarr_path,)) as pool:
        pbar = tqdm(pool.imap_unordered(worker_func, chunk_slices), total=len(chunk_slices), desc="Tracking Chunks")
        for slc, keypoint_data in pbar:
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
    parser = argparse.ArgumentParser(description="Modular, multi-stage fish tracking pipeline.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("zarr_path", type=str, help="Path to the output Zarr file.")
    parser.add_argument("--stage", required=True, choices=['import', 'background', 'crop', 'track', 'all'], help="Processing stage to run.")
    args = parser.parse_args()

    overall_start_time = time.perf_counter()

    if args.stage == 'import':
        run_import_stage(args.video_path, args.zarr_path)
    elif args.stage == 'background':
        run_background_stage(args.zarr_path)
    elif args.stage == 'crop':
        run_crop_stage(args.zarr_path)
    elif args.stage == 'track':
        run_tracking_stage(args.zarr_path)
    elif args.stage == 'all':
        run_import_stage(args.video_path, args.zarr_path)
        run_background_stage(args.zarr_path)
        run_crop_stage(args.zarr_path)
        run_tracking_stage(args.zarr_path)
        
        overall_end_time = time.perf_counter()
        total_elapsed = overall_end_time - overall_start_time
        root = zarr.open_group(args.zarr_path, mode='a')
        root.attrs['total_pipeline_duration_seconds'] = total_elapsed
        print(f"\nAll stages completed. Total pipeline time: {total_elapsed:.2f} seconds.")

    print("\nDone.")

if __name__ == "__main__":
    main()