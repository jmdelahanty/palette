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

# --- Stage-Specific Functions ---

def run_import_stage(video_path, zarr_path):
    """Imports video, saving both full-res and downsampled versions with rich metadata."""
    print("--- Stage: Importing Video ---")
    start_time = time.perf_counter()
    
    print("Initializing GPU-accelerated video pipeline...")
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(video_path, ctx=decord.gpu(0))

    # Correctly get video dimensions by reading the shape of the first frame
    n_frames = len(vr)
    first_frame_shape = vr[0].shape
    full_height, full_width = first_frame_shape[0], first_frame_shape[1]
    
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
    
    images_full = raw_video_group.create_dataset('images_full', shape=(n_frames, full_height, full_width), chunks=(32, None, None), dtype=np.uint8)
    images_ds = raw_video_group.create_dataset('images_ds', shape=(n_frames, ds_size[0], ds_size[1]), chunks=(128, None, None), dtype=np.uint8)
    
    print(f"Importing video at {full_height}x{full_width} and downsampling to {ds_size[0]}x{ds_size[1]}...")
    batch_size = 32
    for i in tqdm(range(0, n_frames, batch_size), desc="Importing Video"):
        batch_gpu = vr.get_batch(range(i, min(i + batch_size, n_frames)))
        gray_batch_float_gpu = (batch_gpu.float() @ gray_weights).unsqueeze(1)
        full_res_batch_cpu = gray_batch_float_gpu.squeeze(1).byte().cpu().numpy()
        images_full[i:i + len(full_res_batch_cpu)] = full_res_batch_cpu
        ds_batch_gpu = F.interpolate(gray_batch_float_gpu, size=ds_size, mode='bilinear', align_corners=False)
        ds_batch_cpu = ds_batch_gpu.squeeze(1).byte().cpu().numpy()
        images_ds[i:i + len(ds_batch_cpu)] = ds_batch_cpu

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    raw_video_group.attrs['duration_seconds'] = elapsed_time
    print(f"Import stage completed in {elapsed_time:.2f} seconds.")

def run_background_stage(zarr_path):
    """Calculates and saves background models with metadata."""
    print("--- Stage 2: Calculating Background ---")
    start_time = time.perf_counter()
    
    root = zarr.open_group(zarr_path, mode='a')
    if 'raw_video/images_full' not in root:
        raise ValueError("Cannot find image datasets. Please run the 'import' stage first.")
    
    images_full = root['raw_video/images_full']
    images_ds = root['raw_video/images_ds']
    num_images = images_full.shape[0]
    
    bg_group = root.require_group('background_models')
    bg_group.attrs['background_command'] = " ".join(sys.argv)
    bg_group.attrs['background_timestamp_utc'] = datetime.utcnow().isoformat()
    
    random.seed(42)
    random_indices = random.sample(range(num_images), min(100, num_images))
    
    print("Calculating background modes...")
    rand_imgs_full = images_full.get_orthogonal_selection((random_indices, slice(None), slice(None)))
    background_full = fast_mode_bincount(rand_imgs_full)
    
    rand_imgs_ds = images_ds.get_orthogonal_selection((random_indices, slice(None), slice(None)))
    background_ds = fast_mode_bincount(rand_imgs_ds)
    
    bg_group.create_dataset('background_full', data=background_full, overwrite=True)
    bg_group.create_dataset('background_ds', data=background_ds, overwrite=True)
    bg_group.attrs['source_frame_indices'] = random_indices
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    bg_group.attrs['duration_seconds'] = elapsed_time
    print(f"Background models saved. Stage completed in {elapsed_time:.2f} seconds.")

def init_track_worker(zarr_path):
    """Initializes the multiprocessing worker for the tracking stage."""
    global worker_root
    worker_root = zarr.open_group(zarr_path, mode='r')

def track_frame_worker(frame_index, roi_sz, ds_thresh, roi_thresh, se1, se2, se4):
    """Performs tracking on a single frame."""
    try:
        images_full = worker_root['raw_video/images_full']
        images_ds = worker_root['raw_video/images_ds']
        background_full = worker_root['background_models/background_full'][:]
        background_ds = worker_root['background_models/background_ds'][:]
        
        img_ds = images_ds[frame_index]
        diff_ds = np.clip(background_ds.astype(np.int16) - img_ds.astype(np.int16), 0, 255).astype(np.uint8)
        
        im_ds = diff_ds >= ds_thresh
        im_ds = erosion(im_ds, se1)
        im_ds = dilation(im_ds, se4)
        ds_stat = regionprops(label(im_ds))
        if not ds_stat: return None

        ds_centroid = np.array(ds_stat[0].centroid)[::-1]
        ds_centroid_norm = ds_centroid / np.array(images_ds.shape[1:])
        
        full_centroid_px = np.round(ds_centroid_norm * np.array(images_full.shape[1:])).astype(int)
        x1, y1 = full_centroid_px[0] - roi_sz[1]//2, full_centroid_px[1] - roi_sz[0]//2
        roi = images_full[frame_index, y1:y1+roi_sz[0], x1:x1+roi_sz[1]]
        if roi.shape != roi_sz: return None

        background_roi = background_full[y1:y1+roi_sz[0], x1:x1+roi_sz[1]]
        diff_roi = np.clip(background_roi.astype(np.int16) - roi.astype(np.int16), 0, 255).astype(np.uint8)
        im_roi = diff_roi >= roi_thresh
        im_roi = erosion(im_roi, se1)
        im_roi = dilation(im_roi, se2)
        im_roi = erosion(im_roi, se1)
        
        L = label(im_roi)
        roi_stat = [r for r in regionprops(L) if r.area > 5]
        if len(roi_stat) < 3: return None

        areas = [stat.area for stat in roi_stat]
        top_indices = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)[:3]
        keypoint_stats = [roi_stat[i] for i in top_indices]
        pts = np.array([s.centroid[::-1] for s in keypoint_stats])
        angles, _ = triangle_calculations(pts[0], pts[1], pts[2])
        kp_idx = np.argsort(angles)
        
        eye_mean = np.mean(pts[kp_idx[1:3]], axis=0)
        head_vec = eye_mean - pts[kp_idx[0]]
        heading = np.rad2deg(np.arctan2(-head_vec[1], head_vec[0]))
        R = np.array([[np.cos(np.deg2rad(heading)), -np.sin(np.deg2rad(heading))], [np.sin(np.deg2rad(heading)), np.cos(np.deg2rad(heading))]])
        rotpts = (pts - eye_mean) @ R.T
        
        bladder_orig_idx, eye1_orig_idx, eye2_orig_idx = kp_idx[0], kp_idx[1], kp_idx[2]
        eye_r_orig_idx, eye_l_orig_idx = (eye1_orig_idx, eye2_orig_idx) if rotpts[eye1_orig_idx, 1] > rotpts[eye2_orig_idx, 1] else (eye2_orig_idx, eye1_orig_idx)

        bladder_pt_norm = keypoint_stats[bladder_orig_idx].centroid[::-1] / np.array(roi_sz)
        eye_l_pt_norm = keypoint_stats[eye_l_orig_idx].centroid[::-1] / np.array(roi_sz)
        eye_r_pt_norm = keypoint_stats[eye_r_orig_idx].centroid[::-1] / np.array(roi_sz)
        
        result_data = [
            heading, ds_centroid_norm[0], ds_centroid_norm[1],
            bladder_pt_norm[0], bladder_pt_norm[1],
            eye_l_pt_norm[0], eye_l_pt_norm[1],
            eye_r_pt_norm[0], eye_r_pt_norm[1],
        ]
        
        return frame_index, roi, result_data
    except Exception:
        return None

def run_tracking_stage(zarr_path):
    """Runs parallel tracking and saves results with rich metadata and parameters."""
    print("--- Stage 3: Tracking Objects ---")
    start_time = time.perf_counter()
    
    root = zarr.open_group(zarr_path, mode='a')
    if 'background_models' not in root:
        raise ValueError("Missing '/background_models'. Please run 'background' stage.")
    
    num_images = root['raw_video/images_full'].shape[0]
    
    track_group = root.require_group('tracking')
    track_group.attrs['tracking_command'] = " ".join(sys.argv)
    track_group.attrs['tracking_timestamp_utc'] = datetime.utcnow().isoformat()
    
    # Define and save tracking parameters
    params = {'roi_sz': (320, 320), 'ds_thresh': 55, 'roi_thresh': 115, 'se1': disk(1), 'se2': disk(2), 'se4': disk(4)}
    serializable_params = {k: v for k, v in params.items() if not isinstance(v, np.ndarray)}
    serializable_params['morphology_disk_radii'] = {'se1': 1, 'se2': 2, 'se4': 4}
    track_group.attrs['tracking_parameters'] = serializable_params

    roi_images = track_group.create_dataset('roi_images', shape=(num_images, params['roi_sz'][0], params['roi_sz'][1]), chunks=(128, None, None), dtype='uint8', overwrite=True)
    
    results_cols = 9
    tracking_results = track_group.create_dataset('tracking_results', shape=(num_images, results_cols), chunks=(4096, None), dtype='f8', overwrite=True)
    tracking_results[:] = np.nan
    tracking_results.attrs['column_names'] = [
        'heading_degrees', 'bbox_x_norm', 'bbox_y_norm', 'bladder_x_roi_norm', 'bladder_y_roi_norm',
        'eye_l_x_roi_norm', 'eye_l_y_roi_norm', 'eye_r_x_roi_norm', 'eye_r_y_roi_norm'
    ]
    
    worker_func = partial(track_frame_worker, **params)
    
    print(f"Processing {num_images} frames using {os.cpu_count()} CPU cores...")
    with multiprocessing.Pool(initializer=init_track_worker, initargs=(zarr_path,)) as pool:
        pbar = tqdm(pool.imap_unordered(worker_func, range(num_images)), total=num_images, desc="Tracking Frames")
        for result in pbar:
            if result:
                frame_idx, roi, data = result
                roi_images[frame_idx] = roi
                tracking_results[frame_idx] = data

    # --- Add Summary Statistics ---
    print("Calculating summary statistics...")
    successful_tracks = np.count_nonzero(~np.isnan(tracking_results[:, 0]))
    percent_tracked = (successful_tracks / num_images) * 100
    track_group.attrs['summary_statistics'] = {'total_frames': num_images, 'frames_tracked': successful_tracks, 'percent_tracked': round(percent_tracked, 2)}
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    track_group.attrs['duration_seconds'] = elapsed_time
    print(f"Summary: {successful_tracks}/{num_images} frames tracked ({percent_tracked:.2f}%).")
    print(f"Tracking stage complete. Total time: {elapsed_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description="Unified, multi-stage fish tracking pipeline with rich metadata.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("zarr_path", type=str, help="Path to the output Zarr file.")
    parser.add_argument("--stage", required=True, choices=['import', 'background', 'track', 'all'], help="Processing stage to run.")
    args = parser.parse_args()

    overall_start_time = time.perf_counter()

    if args.stage == 'import':
        run_import_stage(args.video_path, args.zarr_path)
    elif args.stage == 'background':
        run_background_stage(args.zarr_path)
    elif args.stage == 'track':
        run_tracking_stage(args.zarr_path)
    elif args.stage == 'all':
        run_import_stage(args.video_path, args.zarr_path)
        run_background_stage(args.zarr_path)
        run_tracking_stage(args.zarr_path)
        
        # Save total pipeline time for 'all' stage
        overall_end_time = time.perf_counter()
        total_elapsed = overall_end_time - overall_start_time
        root = zarr.open_group(args.zarr_path, mode='a')
        root.attrs['total_pipeline_duration_seconds'] = total_elapsed
        print(f"\nAll stages completed. Total pipeline time: {total_elapsed:.2f} seconds.")

    print("\nDone.")

if __name__ == "__main__":
    main()