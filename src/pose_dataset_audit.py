#!/usr/bin/env python3
"""
Comprehensive Pose Dataset Audit
Deep dive into the pose dataset creation logic to find frames with invalid keypoints.
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def audit_pose_dataset_selection(zarr_path, split_ratio=0.8, random_seed=42):
    """
    Comprehensive audit of pose dataset selection logic.
    """
    print(f"🔍 COMPREHENSIVE POSE DATASET AUDIT")
    print(f"📁 Zarr path: {zarr_path}")
    print("=" * 80)
    
    try:
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"❌ Error opening Zarr file: {e}")
        return

    # Load all necessary data
    latest_tracking_run = root['tracking_runs'].attrs['latest']
    tracking_results = root[f'tracking_runs/{latest_tracking_run}/tracking_results']
    column_names = tracking_results.attrs.get('column_names', [])
    data = tracking_results[:]
    total_frames = data.shape[0]

    # Use the same source coordinate logic as the dataset
    if 'refine_runs' in root and 'latest' in root['refine_runs'].attrs:
        source_coords = root[f"refine_runs/{root['refine_runs'].attrs['latest']}/refined_bbox_norm_coords"][:]
    else:
        source_coords = root[f"crop_runs/{root['crop_runs'].attrs['latest']}/bbox_norm_coords"][:]

    print(f"📊 Total frames: {total_frames}")
    
    col_map = {name: i for i, name in enumerate(column_names)}
    
    print("\n🔬 STEP 1: Reproducing Pose Dataset Selection Logic")
    print("-" * 60)
    
    # --- This mirrors the exact logic in GlobalIndexManager._get_valid_indices for pose ---
    
    # Base filter: valid crop/refine coordinates
    base_valid_mask = ~np.isnan(source_coords[:, 0])
    print(f"✅ Frames with valid crop/refine coords: {np.sum(base_valid_mask)}/{total_frames}")

    # Pose-specific filter: requires all keypoints to be valid
    kpt_cols = ['bladder_x_roi_norm', 'bladder_y_roi_norm', 'eye_l_x_roi_norm', 
                'eye_l_y_roi_norm', 'eye_r_x_roi_norm', 'eye_r_y_roi_norm']
    kpt_indices = [col_map.get(col) for col in kpt_cols]

    if not all(idx is not None for idx in kpt_indices):
        print(f"❌ Error: Missing one or more required keypoint columns: {kpt_cols}")
        return

    keypoints_valid_mask = ~np.any(np.isnan(data[:, kpt_indices]), axis=1)
    print(f"✅ Frames with all valid keypoints: {np.sum(keypoints_valid_mask)}/{total_frames}")

    # Final combined filter for pose
    final_valid_mask = base_valid_mask & keypoints_valid_mask
    final_valid_indices = np.where(final_valid_mask)[0]
    
    print(f"🎯 Final selected frames for pose task: {len(final_valid_indices)}/{total_frames}")
    print()

    # Step 2: Simulate what the dataset's _get_pose_data method would return
    print("🔬 STEP 2: Simulating _get_pose_data() Method")
    print("-" * 50)

    def simulate_get_pose_data(frame_data):
        """Simulates the logic of _get_pose_data to check for issues."""
        try:
            # Check for NaN in any of the required fields
            bbox_fields = [col_map['bbox_x_norm_ds'], col_map['bbox_y_norm_ds'], col_map['bbox_width_norm_ds'], col_map['bbox_height_norm_ds']]
            if np.isnan(frame_data[bbox_fields]).any() or np.isnan(frame_data[kpt_indices]).any():
                return None, "NaN value in required bbox or keypoint field"

            # If all checks pass, simulate data creation
            bbox_x, bbox_y, bbox_w, bbox_h = frame_data[bbox_fields]
            kpts = frame_data[kpt_indices]
            
            kpts_with_visibility = np.array([kpts[0], kpts[1], 2, kpts[2], kpts[3], 2, kpts[4], kpts[5], 2]).reshape(1, -1)

            return {
                "cls": np.array([0]),
                "bboxes": np.array([[bbox_x, bbox_y, bbox_w, bbox_h]]),
                "keypoints": kpts_with_visibility
            }, "OK"
        except Exception as e:
            return None, f"Exception: {e}"

    issues_found = []
    sample_to_check = final_valid_indices[:min(1000, len(final_valid_indices))]

    print(f"Simulating data retrieval for {len(sample_to_check)} selected frames...")
    for i, frame_idx in enumerate(sample_to_check):
        frame_data = data[frame_idx]
        pose_data, status = simulate_get_pose_data(frame_data)

        if pose_data is None:
            issues_found.append((frame_idx, status))
            if len(issues_found) < 10:
                print(f"  ❌ Frame {frame_idx}: {status}")
        elif i < 5:
            print(f"  ✅ Frame {frame_idx}: OK (cls: {pose_data['cls'].shape}, bbox: {pose_data['bboxes'].shape}, kpts: {pose_data['keypoints'].shape})")

    if issues_found:
        print(f"\n⚠️  Found {len(issues_found)} frames that would fail data retrieval!")
    else:
        print("\n✅ All checked frames appear to produce valid pose data.")
    print()


    # Final summary
    print("🎯 POSE AUDIT SUMMARY")
    print("-" * 25)
    print(f"📊 Total frames in dataset: {total_frames}")
    print(f"✅ Frames passing pose filter: {len(final_valid_indices)}")
    
    train_indices, val_indices = train_test_split(
        final_valid_indices, train_size=split_ratio, random_state=random_seed, shuffle=True)
    
    print(f"🚂 Training samples: {len(train_indices)}")
    print(f"✅ Validation samples: {len(val_indices)}")
    print(f"❌ Frames that would fail data retrieval: {len(issues_found)}/{len(sample_to_check)} checked")
    print()
    
    if len(issues_found) > 0:
        print("🚨 CRITICAL FINDING:")
        print(f"   {len(issues_found)} frames passed the initial filter but would fail during data loading.")
        print("   This means the dataset will return empty labels for these frames, which could lead to training issues like the 'pose_loss: 0' problem.")
        print("\n🔧 RECOMMENDED FIX:")
        print("   The filtering logic in `GlobalIndexManager._get_valid_indices` seems correct, but double-check your Zarr data for unexpected NaN values in tracked frames.")
        return False
    else:
        print("✅ AUDIT PASSED:")
        print("   The dataset filtering and data retrieval logic appear to be consistent for the pose task.")
        return True

def main():
    parser = argparse.ArgumentParser(description="Comprehensive audit for the pose dataset.")
    parser.add_argument("zarr_path", type=str, help="Path to the video.zarr file.")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=42)
    
    args = parser.parse_args()
    
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"❌ Error: Zarr file not found: {zarr_path}")
        return
    
    audit_pose_dataset_selection(args.zarr_path, args.split_ratio, args.random_seed)

if __name__ == "__main__":
    main()