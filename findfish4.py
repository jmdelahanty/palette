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
from datetime import datetime
import subprocess
import decord
import torch
import torch.nn.functional as F

# Prevent deadlocks from nested parallelism
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
cv2.setNumThreads(0)

def get_git_info():
    """
    Captures the Git commit hash and repository status.
    Returns a dictionary with 'commit_hash' and 'is_dirty' status.
    """
    try:
        # Get the path of the current script
        script_path = os.path.dirname(os.path.realpath(__file__))

        # Get the full commit hash
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            cwd=script_path, 
            stderr=subprocess.DEVNULL
        ).strip().decode('utf-8')

        # Check for uncommitted changes
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'], 
            cwd=script_path, 
            stderr=subprocess.DEVNULL
        ).strip().decode('utf-8')

        is_dirty = bool(status)

        return {'commit_hash': commit_hash, 'is_dirty': is_dirty}

    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit_hash': 'N/A', 'is_dirty': True, 'error': 'Not a git repository or git not found'}


def create_zarr_from_video(video_path, zarr_path, target_size=None):
    """
    Converts a video to a Zarr array using a fully GPU-accelerated pipeline.
    """
    print("Initializing fully GPU-accelerated video pipeline...")
    decord.bridge.set_bridge('torch')
    
    try:
        vr = decord.VideoReader(video_path, ctx=decord.gpu(0))
        n_frames = len(vr)
        
        gray_weights = torch.tensor([0.2989, 0.5870, 0.1140]).to('cuda:0')

        # Process the first frame to determine final dimensions
        test_frame_gpu = vr[0]
        
        # Convert to float and keep it as a float for processing
        gray_test_float_gpu = (test_frame_gpu.float() @ gray_weights).unsqueeze(0).unsqueeze(0)
        
        if target_size:
            resized_test_gpu = F.interpolate(gray_test_float_gpu, size=target_size, mode='bilinear', align_corners=False)
            height, width = resized_test_gpu.shape[2], resized_test_gpu.shape[3]
        else:
            height, width = gray_test_float_gpu.shape[2], gray_test_float_gpu.shape[3]

        dtype = np.uint8

    except Exception as e:
        raise RuntimeError(f"Error initializing GPU pipeline: {e}")

    root_group = zarr.open_group(zarr_path, mode='w')
    image_array = root_group.create_dataset('images', shape=(n_frames, height, width), chunks=(128, None, None), dtype=dtype)
    
    status = f"downsampling to {height}x{width}" if target_size else f"at full resolution ({height}x{width})"
    print(f"Converting video to Zarr array {status} on GPU...")

    batch_size = 64
    for i in tqdm(range(0, n_frames, batch_size), desc="Converting Video"):
        batch_gpu = vr.get_batch(range(i, min(i + batch_size, n_frames)))
        
        # 1. Convert to float and then to grayscale on GPU
        gray_batch_float_gpu = (batch_gpu.float() @ gray_weights).unsqueeze(1)

        # 2. Resize on GPU (if needed) while still a float
        if target_size:
            processed_batch_float_gpu = F.interpolate(gray_batch_float_gpu, size=target_size, mode='bilinear', align_corners=False)
        else:
            processed_batch_float_gpu = gray_batch_float_gpu
        
        # 3. Convert back to byte, move to CPU, and save
        final_batch_cpu = processed_batch_float_gpu.squeeze(1).byte().cpu().numpy()
        image_array[i:i + len(final_batch_cpu)] = final_batch_cpu

    print(f"Zarr array creation complete! Total frames: {n_frames}")


def init_worker(bg_full, bg_ds, zarr_path):
    """Initializes worker with background images and a handle to the Zarr Group."""
    global worker_bg_full, worker_bg_ds, worker_zarr_root
    worker_bg_full = bg_full
    worker_bg_ds = bg_ds
    worker_zarr_root = zarr.open_group(zarr_path, mode='a')


def process_frame(frame_index, full_img_sz, ds_img_sz, ds_thresh, roi_thresh, se1, se2, se4, bbox_width, bbox_height, roi_sz):
    """Worker function that reads from and writes all results to the global Zarr Group."""
    try:
        images_array = worker_zarr_root['images']
        roi_images_array = worker_zarr_root['roi_images']
        results_array = worker_zarr_root['tracking_results']
        background_full = worker_bg_full
        background_ds = worker_bg_ds

        img_from_zarr = images_array[frame_index]
        
        # The processing pipeline still uses a 640x640 version for initial detection for speed
        img_ds = cv2.resize(img_from_zarr, ds_img_sz, interpolation=cv2.INTER_LINEAR)
        
        diff_ds = np.clip(background_ds.astype(np.int16) - img_ds.astype(np.int16), 0, 255).astype(np.uint8)
        thresh = ds_thresh
        ds_stat = []
        while len(ds_stat) < 1:
            im_ds = diff_ds >= thresh
            im_ds = erosion(im_ds, se1)
            im_ds = dilation(im_ds, se4)
            labeled_ds = label(im_ds)
            ds_stat = regionprops(labeled_ds)
            if len(ds_stat) < 1 and thresh > 5:
                thresh -= 5
            else:
                break
        if not ds_stat: return

        while len(ds_stat) > 1:
            im_ds = dilation(im_ds, se1)
            labeled_ds = label(im_ds)
            ds_stat = regionprops(labeled_ds)
        
        ds_centroid = np.array(ds_stat[0].centroid)[::-1]
        ds_centroid_norm = ds_centroid / np.array(ds_img_sz)
        full_centroid_px = np.round(ds_centroid_norm * np.array(full_img_sz)).astype(int)
        
        roi_halfwidth = (roi_sz[0] // 2, roi_sz[1] // 2)
        x1, y1 = full_centroid_px[0] - roi_halfwidth[1], full_centroid_px[1] - roi_halfwidth[0]
        x2, y2 = x1 + roi_sz[1], y1 + roi_sz[0]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(full_img_sz[1], x2), min(full_img_sz[0], y2)
        roi = img_from_zarr[y1:y2, x1:x2]
        
        if roi.shape == roi_sz:
            roi_images_array[frame_index] = roi

        background_roi = background_full[y1:y2, x1:x2]
        diff_roi = np.clip(background_roi.astype(np.int16) - roi.astype(np.int16), 0, 255).astype(np.uint8)
        
        im_roi = diff_roi >= roi_thresh
        im_roi = erosion(im_roi, se1)
        im_roi = dilation(im_roi, se2)
        im_roi = erosion(im_roi, se1)
        L = label(im_roi)
        roi_stat = regionprops(L)
        if len(roi_stat) < 3: return

        areas = [stat.area for stat in roi_stat]
        sorted_indices = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)[:3]
        im = np.zeros_like(im_roi)
        for idx in sorted_indices: im[L == (idx + 1)] = True
        L3 = label(im)
        roi_stat = regionprops(L3)
        if len(roi_stat) < 3: return

        pts = np.array([stat.centroid[::-1] for stat in roi_stat])
        angles, _ = triangle_calculations(pts[0], pts[1], pts[2])
        kp_idx = np.argsort(angles)
        eye_mean = np.mean(pts[kp_idx[1:3]], axis=0)
        head_vec = eye_mean - pts[kp_idx[0]]
        heading = np.rad2deg(np.arctan2(-head_vec[1], head_vec[0]))
        
        R = np.array([[np.cos(np.deg2rad(heading)), -np.sin(np.deg2rad(heading))], [np.sin(np.deg2rad(heading)), np.cos(np.deg2rad(heading))]])
        rotpts = (pts - eye_mean) @ R.T
        cc_idx = np.zeros(3, dtype=int)
        cc_idx[0] = kp_idx[0]
        eye1, eye2 = kp_idx[1], kp_idx[2]
        cc_idx[1], cc_idx[2] = (eye1, eye2) if rotpts[eye1, 1] > 0 else (eye2, eye1)

        bladder_norm = np.array(roi_stat[cc_idx[0]].centroid[::-1]) / np.array(roi_sz)
        eye_l_norm = np.array(roi_stat[cc_idx[1]].centroid[::-1]) / np.array(roi_sz)
        eye_r_norm = np.array(roi_stat[cc_idx[2]].centroid[::-1]) / np.array(roi_sz)
        
        result_data = [
            heading, ds_centroid_norm[0], ds_centroid_norm[1], bbox_width, bbox_height,
            bladder_norm[0], bladder_norm[1],
            eye_l_norm[0], eye_l_norm[1],
            eye_r_norm[0], eye_r_norm[1]
        ]
        results_array[frame_index] = result_data

    except Exception as e:
        print(f"Error processing frame {frame_index}: {e}")


def fast_mode_bincount(arr):
    moved_axis_arr = np.moveaxis(arr, 0, -1)
    mode_map = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=moved_axis_arr)
    return mode_map.astype(np.uint8)


def triangle_calculations(p1, p2, p3):
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    angles = np.zeros(3)
    # Check for collinear points to avoid division by zero
    if (2 * b * c) > 0: angles[0] = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0)) * 180 / np.pi
    if (2 * a * c) > 0: angles[1] = np.arccos(np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1.0, 1.0)) * 180 / np.pi
    if (2 * a * b) > 0: angles[2] = np.arccos(np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)) * 180 / np.pi
    return angles, np.array([a, b, c])


def main():
    parser = argparse.ArgumentParser(description="Process fish tracking video into a Zarr dataset.")
    parser.add_argument(
        '--full-resolution', 
        action='store_true', 
        help="Store full-resolution images in Zarr. Default is to downsample to 640x640."
    )
    args = parser.parse_args()

    base_folder = r'/home/delahantyj@hhmi.org/Desktop/concentricOMR3/'
    video_path = os.path.join(base_folder, 'concentric_omr_example.mp4')
    zarr_path = os.path.join(base_folder, 'video.zarr')

    if not os.path.exists(zarr_path):
        target_size = None if args.full_resolution else (640, 640)
        create_zarr_from_video(video_path, zarr_path, target_size=target_size)

    zarr_group = zarr.open_group(zarr_path, mode='a')
    image_array = zarr_group['images']
    num_images, zarr_height, zarr_width = image_array.shape
    roi_sz = (320, 320)
    
    print(f"Opened Zarr group. Found {num_images} frames with resolution {zarr_width}x{zarr_height}.")
    
    # Store all available metadata if it hasn't been stored already
    if 'source_video_metadata' not in zarr_group.attrs:
        print("Reading and saving all available video metadata...")
        try:
            video_metadata = iio.immeta(video_path)
            zarr_group.attrs['source_video_metadata'] = video_metadata
            print("Metadata saved.")
        except Exception as e:
            print(f"Could not read video metadata: {e}")
            zarr_group.attrs['source_video_metadata'] = {'error': 'failed to read metadata'}

    if 'roi_images' not in zarr_group:
        zarr_group.create_dataset('roi_images', shape=(num_images, roi_sz[0], roi_sz[1]), chunks=(128, None, None), dtype='uint8')
        
    if 'tracking_results' in zarr_group:
        del zarr_group['tracking_results']
    results_array = zarr_group.create_dataset('tracking_results', shape=(num_images, 11), chunks=(4096, None), dtype='f8')
    results_array.attrs['column_names'] = [
        'heading_degrees', 'bbox_x_norm', 'bbox_y_norm', 'bbox_w_norm', 'bbox_h_norm',
        'bladder_x_norm', 'bladder_y_norm', 'eye_l_x_norm', 'eye_l_y_norm',
        'eye_r_x_norm', 'eye_r_y_norm'
    ]
    results_array.attrs['analysis_parameters'] = {
        'downsample_threshold': 55, 'roi_threshold': 115, 'roi_size': roi_sz
    }

    if 'software_provenance' not in zarr_group.attrs:
        print("Saving software provenance (Git commit hash)...")
        git_info = get_git_info()
        zarr_group.attrs['software_provenance'] = git_info
        if git_info.get('is_dirty'):
            print("Warning: Uncommitted changes found in the repository.")

    random.seed(42)
    random_indices = random.sample(range(num_images), min(100, num_images))
    rand_imgs_from_zarr = image_array.get_orthogonal_selection((random_indices, slice(None), slice(None)))
    
    # Full image size for analysis is always the size stored in the zarr array
    full_img_sz = (zarr_height, zarr_width)
    ds_img_sz = (640, 640) # Detection is always done on 640x640 for speed
    
    # The rand_imgs need to be resized to the detection size for background calculation
    rand_imgs_ds = np.array([cv2.resize(img, ds_img_sz, interpolation=cv2.INTER_LINEAR) for img in rand_imgs_from_zarr])
    
    print("Calculating background mode...")
    background_full = fast_mode_bincount(rand_imgs_from_zarr)
    background_ds = fast_mode_bincount(rand_imgs_ds)
    print("Background calculation complete")

    # Save background image and indices to Zarr store
    if 'background' not in zarr_group:
        print("Saving background image to Zarr store...")
        bg_dset = zarr_group.create_dataset('background', data=background_full, chunks=(512, 512), overwrite=True)
        bg_dset.attrs['source_frame_indices'] = random_indices
        print("Background and source frame indices saved.")

    ds_thresh, roi_thresh = 55, 115
    se1, se2, se4 = disk(1), disk(2), disk(4)
    bbox_width, bbox_height = 0.0171875, 0.0171875

    worker = partial(process_frame,
                     full_img_sz=full_img_sz, ds_img_sz=ds_img_sz,
                     ds_thresh=ds_thresh, roi_thresh=roi_thresh,
                     se1=se1, se2=se2, se4=se4,
                     bbox_width=bbox_width, bbox_height=bbox_height, roi_sz=roi_sz)

    frame_iterator = range(num_images)
    
    print(f"Processing {num_images} frames using {os.cpu_count()} CPU cores...")
    initargs = (background_full, background_ds, zarr_path)
    with multiprocessing.Pool(initializer=init_worker, initargs=initargs) as pool:
        list(tqdm(pool.imap_unordered(worker, frame_iterator), total=num_images, desc="Processing Frames"))

    print("Processing complete! All data saved within video.zarr.")


if __name__ == "__main__":
    main()