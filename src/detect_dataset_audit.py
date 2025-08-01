#!/usr/bin/env python3
"""
Comprehensive Dataset Audit
Deep dive into the dataset creation logic to find any frames without proper detections.
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def audit_dataset_selection(zarr_path, split_ratio=0.8, random_seed=42):
    """
    Comprehensive audit of dataset selection logic to find any problematic frames.
    """
    print(f"🔍 COMPREHENSIVE DATASET AUDIT")
    print(f"📁 Zarr path: {zarr_path}")
    print("=" * 80)
    
    try:
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"❌ Error opening Zarr file: {e}")
        return
    
    # Load tracking results
    tracking_results = root['tracking/tracking_results']
    column_names = tracking_results.attrs.get('column_names', [])
    data = tracking_results[:]
    total_frames = data.shape[0]
    
    print(f"📊 Total frames: {total_frames}")
    print(f"📋 Columns: {column_names}")
    print()
    
    # Create column mapping
    col_map = {name: i for i, name in enumerate(column_names)}
    
    # Step 1: Analyze the EXACT logic used in ZarrYOLODataset
    print("🔬 STEP 1: Reproducing ZarrYOLODataset Selection Logic")
    print("-" * 60)
    
    # This mirrors the exact logic in ZarrYOLODataset.__init__()
    data_format = 'enhanced' if 'bbox_x_norm_ds' in column_names else 'original'
    print(f"📊 Detected format: {data_format}")
    
    if data_format == 'enhanced':
        col_mappings = {
            'heading': col_map['heading_degrees'],
            'bbox_x': col_map['bbox_x_norm_ds'],
            'bbox_y': col_map['bbox_y_norm_ds'],
            'bbox_width': col_map['bbox_width_norm_ds'],
            'bbox_height': col_map['bbox_height_norm_ds'],
        }
    else:
        col_mappings = {
            'heading': col_map['heading_degrees'],
            'bbox_x': col_map['bbox_x_norm'],
            'bbox_y': col_map['bbox_y_norm'],
        }
    
    print(f"🎯 Key columns being checked:")
    for key, idx in col_mappings.items():
        print(f"   {key}: column {idx} ({column_names[idx]})")
    print()
    
    # Reproduce the exact filtering logic
    print("🎯 STEP 2: Applying Dataset Filtering Logic")
    print("-" * 50)
    
    # Step 1: Valid heading (this is always the first filter)
    valid_heading = ~np.isnan(data[:, col_mappings['heading']])
    heading_count = np.sum(valid_heading)
    print(f"✅ Frames with valid heading: {heading_count}/{total_frames}")
    
    # Step 2: For enhanced format, also check bbox data (this is the second filter)
    if data_format == 'enhanced':
        bbox_valid = (~np.isnan(data[:, col_mappings['bbox_x']]) & 
                     ~np.isnan(data[:, col_mappings['bbox_y']]))
        bbox_count = np.sum(bbox_valid)
        print(f"✅ Frames with valid bbox coords: {bbox_count}/{total_frames}")
        
        # The combined filter (this is what's actually used)
        current_valid_mask = valid_heading & bbox_valid
        combined_count = np.sum(current_valid_mask)
        print(f"🎯 Combined filter (heading AND bbox): {combined_count}/{total_frames}")
    else:
        current_valid_mask = valid_heading
        combined_count = heading_count
        print(f"🎯 Using heading-only filter: {combined_count}/{total_frames}")
    
    current_valid_indices = np.where(current_valid_mask)[0]
    print(f"📊 Selected frame indices: {len(current_valid_indices)} frames")
    print()
    
    # Step 3: Deep dive into the selected frames
    print("🔬 STEP 3: Deep Analysis of Selected Frames")
    print("-" * 45)
    
    # Check for potential issues in selected frames
    issues_found = []
    
    print("Checking first 20 selected frames for data quality...")
    for i, frame_idx in enumerate(current_valid_indices[:20]):
        frame_data = data[frame_idx]
        
        # Check heading
        heading = frame_data[col_mappings['heading']]
        bbox_x = frame_data[col_mappings['bbox_x']]
        bbox_y = frame_data[col_mappings['bbox_y']]
        
        issues = []
        
        # These should never happen since we filtered on them, but let's double-check
        if np.isnan(heading):
            issues.append("NaN heading")
        if np.isnan(bbox_x):
            issues.append("NaN bbox_x")
        if np.isnan(bbox_y):
            issues.append("NaN bbox_y")
            
        # Check additional fields that might cause problems
        if data_format == 'enhanced':
            bbox_width = frame_data[col_mappings['bbox_width']]
            bbox_height = frame_data[col_mappings['bbox_height']]
            
            if np.isnan(bbox_width):
                issues.append("NaN bbox_width")
            if np.isnan(bbox_height):
                issues.append("NaN bbox_height")
            
            # Check for invalid bbox dimensions
            if not np.isnan(bbox_width) and bbox_width <= 0:
                issues.append(f"Invalid bbox_width: {bbox_width}")
            if not np.isnan(bbox_height) and bbox_height <= 0:
                issues.append(f"Invalid bbox_height: {bbox_height}")
        
        # Check keypoint data (these aren't filtered on but are used in _get_bbox_data)
        keypoint_cols = []
        for kp_name in ['bladder_x_roi_norm', 'bladder_y_roi_norm', 'eye_l_x_roi_norm', 
                       'eye_l_y_roi_norm', 'eye_r_x_roi_norm', 'eye_r_y_roi_norm']:
            if kp_name in col_map:
                keypoint_cols.append(col_map[kp_name])
        
        missing_keypoints = 0
        for kp_idx in keypoint_cols:
            if np.isnan(frame_data[kp_idx]):
                missing_keypoints += 1
        
        if missing_keypoints > 0:
            issues.append(f"{missing_keypoints}/{len(keypoint_cols)} keypoints are NaN")
        
        if issues:
            issues_found.append((frame_idx, i, issues))
            print(f"  ❌ Frame {frame_idx} (selected #{i}): {', '.join(issues)}")
        elif i < 5:  # Show first few good ones
            print(f"  ✅ Frame {frame_idx} (selected #{i}): heading={heading:.2f}, bbox=({bbox_x:.3f}, {bbox_y:.3f})")
    
    if issues_found:
        print(f"\n⚠️  Found {len(issues_found)} problematic frames in the first 20 selected!")
    else:
        print(f"\n✅ First 20 selected frames look good")
    
    print()
    
    # Step 4: Check what _get_bbox_data would return for problematic cases
    print("🔬 STEP 4: Simulating _get_bbox_data() Method")
    print("-" * 45)
    
    def simulate_get_bbox_data(frame_idx, data_format, col_mappings, frame_data):
        """Simulate the _get_bbox_data method logic"""
        try:
            heading = frame_data[col_mappings['heading']]
            bbox_x = frame_data[col_mappings['bbox_x']]
            bbox_y = frame_data[col_mappings['bbox_y']]
            
            if np.isnan(heading) or np.isnan(bbox_x) or np.isnan(bbox_y):
                return None, "Basic validation failed"
            
            if data_format == 'enhanced':
                bbox_width = frame_data[col_mappings['bbox_width']]
                bbox_height = frame_data[col_mappings['bbox_height']]
                
                if np.isnan(bbox_width) or np.isnan(bbox_height):
                    return None, "Enhanced bbox dimensions missing"
                
                confidence = 1.0  # Default
                if 'confidence_score' in col_map:
                    conf_val = frame_data[col_map['confidence_score']]
                    if not np.isnan(conf_val):
                        confidence = conf_val
                
                bbox_data = np.array([0, bbox_x, bbox_y, bbox_width, bbox_height, confidence], dtype=np.float32)
            else:
                # Original format
                bbox_width = 0.05
                bbox_height = 0.05
                confidence = 1.0
                bbox_data = np.array([0, bbox_x, bbox_y, bbox_width, bbox_height, confidence], dtype=np.float32)
            
            return bbox_data, "OK"
            
        except Exception as e:
            return None, f"Exception: {e}"
    
    # Test _get_bbox_data on all selected frames (or a large sample)
    test_indices = current_valid_indices[:100] if len(current_valid_indices) > 100 else current_valid_indices
    bbox_issues = []
    
    print(f"Testing _get_bbox_data simulation on {len(test_indices)} frames...")
    
    for i, frame_idx in enumerate(test_indices):
        frame_data = data[frame_idx]
        bbox_data, status = simulate_get_bbox_data(frame_idx, data_format, col_mappings, frame_data)
        
        if bbox_data is None:
            bbox_issues.append((frame_idx, i, status))
            if len(bbox_issues) <= 10:  # Show first 10 issues
                print(f"  ❌ Frame {frame_idx}: {status}")
        elif i < 5:  # Show first few successes
            label = bbox_data[:5].reshape(1, -1)
            cls_labels = label[:, 0].astype(np.float32)
            bbox_coords = label[:, 1:5].astype(np.float32)
            print(f"  ✅ Frame {frame_idx}: cls={cls_labels}, bbox_shape={bbox_coords.shape}")
    
    if bbox_issues:
        print(f"\n⚠️  _get_bbox_data would fail for {len(bbox_issues)}/{len(test_indices)} frames!")
        print("This means these frames would return None and use fallback data")
    else:
        print(f"\n✅ _get_bbox_data simulation successful for all {len(test_indices)} frames")
    
    print()
    
    # Step 5: Simulate the train/val split
    print("🔬 STEP 5: Analyzing Train/Val Split")
    print("-" * 40)
    
    train_indices, val_indices = train_test_split(
        current_valid_indices,
        train_size=split_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"🚂 Training indices: {len(train_indices)} frames")
    print(f"✅ Validation indices: {len(val_indices)} frames")
    
    # Check for issues in the actual train/val splits
    def check_split_issues(indices, split_name):
        split_issues = []
        print(f"\nChecking {split_name} split:")
        
        for i, frame_idx in enumerate(indices[:20]):  # Check first 20
            frame_data = data[frame_idx]
            bbox_data, status = simulate_get_bbox_data(frame_idx, data_format, col_mappings, frame_data)
            
            if bbox_data is None:
                split_issues.append((frame_idx, status))
                if len(split_issues) <= 5:
                    print(f"  ❌ {split_name} frame {frame_idx}: {status}")
            elif i < 3:
                print(f"  ✅ {split_name} frame {frame_idx}: Valid bbox data")
        
        return split_issues
    
    train_issues = check_split_issues(train_indices, "Training")
    val_issues = check_split_issues(val_indices, "Validation")
    
    # Final summary
    print()
    print("🎯 AUDIT SUMMARY")
    print("-" * 20)
    print(f"📊 Total frames in dataset: {total_frames}")
    print(f"✅ Frames passing filter: {len(current_valid_indices)}")
    print(f"🚂 Training samples: {len(train_indices)}")
    print(f"✅ Validation samples: {len(val_indices)}")
    print(f"❌ Issues in first 20 selected: {len(issues_found)}")
    print(f"❌ _get_bbox_data failures: {len(bbox_issues)}/{len(test_indices)}")
    print(f"❌ Training split issues: {len(train_issues)}")
    print(f"❌ Validation split issues: {len(val_issues)}")
    print()
    
    if len(bbox_issues) > 0:
        print("🚨 CRITICAL FINDING:")
        print(f"   {len(bbox_issues)} selected frames would fail _get_bbox_data()")
        print("   These frames would return fallback data: [0, 0.5, 0.5, 0.1, 0.1]")
        print("   This could cause training instability!")
        print()
        print("🔧 RECOMMENDED FIX:")
        print("   Add stricter filtering in ZarrYOLODataset to exclude these frames")
        return False
    else:
        print("✅ AUDIT PASSED:")
        print("   All selected frames should produce valid YOLO training data")
        return True

def main():
    parser = argparse.ArgumentParser(description="Comprehensive dataset audit")
    parser.add_argument("zarr_path", type=str, help="Path to the video.zarr file")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=42)
    
    args = parser.parse_args()
    
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"❌ Error: Zarr file not found: {zarr_path}")
        return
    
    audit_passed = audit_dataset_selection(args.zarr_path, args.split_ratio, args.random_seed)
    
    if not audit_passed:
        print("💡 Consider running with stricter dataset filtering!")

if __name__ == "__main__":
    main()