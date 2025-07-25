#!/usr/bin/env python3
"""
Exhaustive Frame Validation Script
Test EVERY SINGLE selected frame from the raw Zarr data to find any problematic ones.
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def validate_every_selected_frame(zarr_path, split_ratio=0.8, random_seed=42):
    """
    Test every single frame that would be selected by the dataset filtering logic.
    """
    print(f"🔍 EXHAUSTIVE FRAME VALIDATION")
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
    
    print(f"📊 Total frames in Zarr: {total_frames}")
    print(f"📋 Columns: {len(column_names)}")
    print()
    
    # Create column mapping (exact replica of ZarrYOLODataset logic)
    col_map = {name: i for i, name in enumerate(column_names)}
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
        # Additional columns for comprehensive validation
        extra_cols = {
            'bladder_x_roi_norm': col_map.get('bladder_x_roi_norm'),
            'bladder_y_roi_norm': col_map.get('bladder_y_roi_norm'),
            'eye_l_x_roi_norm': col_map.get('eye_l_x_roi_norm'),
            'eye_l_y_roi_norm': col_map.get('eye_l_y_roi_norm'),
            'eye_r_x_roi_norm': col_map.get('eye_r_x_roi_norm'),
            'eye_r_y_roi_norm': col_map.get('eye_r_y_roi_norm'),
            'confidence_score': col_map.get('confidence_score'),
        }
    else:
        col_mappings = {
            'heading': col_map['heading_degrees'],
            'bbox_x': col_map['bbox_x_norm'],
            'bbox_y': col_map['bbox_y_norm'],
        }
        extra_cols = {}
    
    print(f"🎯 Core columns: {list(col_mappings.keys())}")
    print(f"📋 Extra columns: {[k for k, v in extra_cols.items() if v is not None]}")
    print()
    
    # Reproduce EXACT filtering logic from ZarrYOLODataset
    print("🔬 STEP 1: Applying Dataset Filtering Logic")
    print("-" * 50)
    
    valid_heading = ~np.isnan(data[:, col_mappings['heading']])
    heading_count = np.sum(valid_heading)
    print(f"✅ Frames with valid heading: {heading_count}/{total_frames}")
    
    if data_format == 'enhanced':
        bbox_valid = (~np.isnan(data[:, col_mappings['bbox_x']]) & 
                     ~np.isnan(data[:, col_mappings['bbox_y']]))
        bbox_count = np.sum(bbox_valid)
        print(f"✅ Frames with valid bbox coords: {bbox_count}/{total_frames}")
        
        current_valid_mask = valid_heading & bbox_valid
        combined_count = np.sum(current_valid_mask)
        print(f"🎯 Combined filter: {combined_count}/{total_frames}")
    else:
        current_valid_mask = valid_heading
        combined_count = heading_count
        print(f"🎯 Heading-only filter: {combined_count}/{total_frames}")
    
    current_valid_indices = np.where(current_valid_mask)[0]
    print(f"📊 Selected frame indices: {len(current_valid_indices)} frames")
    print()
    
    # Create train/val split (exact replica of dataset logic)
    train_indices, val_indices = train_test_split(
        current_valid_indices,
        train_size=split_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"🚂 Training frames: {len(train_indices)}")
    print(f"✅ Validation frames: {len(val_indices)}")
    print()
    
    # Now test EVERY SINGLE selected frame
    print("🔬 STEP 2: Exhaustive Frame-by-Frame Validation")
    print("-" * 55)
    
    def validate_single_frame(frame_idx, frame_data, is_train=True):
        """Validate a single frame and simulate what the dataset would return."""
        split_name = "TRAIN" if is_train else "VAL"
        issues = []
        
        try:
            # Check core fields
            heading = frame_data[col_mappings['heading']]
            bbox_x = frame_data[col_mappings['bbox_x']]
            bbox_y = frame_data[col_mappings['bbox_y']]
            
            # These should never be NaN since we filtered on them
            if np.isnan(heading):
                issues.append("NaN heading (filter failed)")
            if np.isnan(bbox_x):
                issues.append("NaN bbox_x (filter failed)")
            if np.isnan(bbox_y):
                issues.append("NaN bbox_y (filter failed)")
            
            # Check enhanced format fields (these might be NaN and cause _get_bbox_data to fail)
            if data_format == 'enhanced':
                bbox_width = frame_data[col_mappings['bbox_width']]
                bbox_height = frame_data[col_mappings['bbox_height']]
                
                if np.isnan(bbox_width):
                    issues.append("NaN bbox_width (would cause _get_bbox_data to fail)")
                if np.isnan(bbox_height):
                    issues.append("NaN bbox_height (would cause _get_bbox_data to fail)")
                
                # Check for invalid dimensions
                if not np.isnan(bbox_width) and bbox_width <= 0:
                    issues.append(f"Invalid bbox_width: {bbox_width}")
                if not np.isnan(bbox_height) and bbox_height <= 0:
                    issues.append(f"Invalid bbox_height: {bbox_height}")
                
                # Check confidence score
                if extra_cols['confidence_score'] is not None:
                    confidence = frame_data[extra_cols['confidence_score']]
                    if np.isnan(confidence):
                        issues.append("NaN confidence_score")
            
            # Check keypoint data (used in _get_bbox_data but not filtered on)
            keypoint_issues = []
            for kp_name, col_idx in extra_cols.items():
                if col_idx is not None and 'roi_norm' in kp_name:
                    kp_value = frame_data[col_idx]
                    if np.isnan(kp_value):
                        keypoint_issues.append(kp_name)
            
            if keypoint_issues:
                issues.append(f"NaN keypoints: {', '.join(keypoint_issues)}")
            
            # Simulate _get_bbox_data logic
            bbox_data_result = simulate_get_bbox_data(frame_data, data_format, col_mappings, extra_cols)
            
            if bbox_data_result is None:
                issues.append("_get_bbox_data would return None (fallback data used)")
            else:
                # Simulate dataset __getitem__ logic
                try:
                    label = bbox_data_result[:5].reshape(1, -1)
                    cls_labels = label[:, 0].astype(np.float32)
                    bbox_coords = label[:, 1:5].astype(np.float32)
                    
                    # Check tensor properties
                    if cls_labels.ndim != 1:
                        issues.append(f"cls would have wrong ndim: {cls_labels.ndim}")
                    if len(cls_labels) != 1:
                        issues.append(f"cls would have wrong length: {len(cls_labels)}")
                    if bbox_coords.ndim != 2:
                        issues.append(f"bbox would have wrong ndim: {bbox_coords.ndim}")
                    if bbox_coords.shape != (1, 4):
                        issues.append(f"bbox would have wrong shape: {bbox_coords.shape}")
                    
                    # Check for NaN in final data
                    if np.any(np.isnan(cls_labels)):
                        issues.append("Final cls would contain NaN")
                    if np.any(np.isnan(bbox_coords)):
                        issues.append("Final bbox would contain NaN")
                    
                except Exception as e:
                    issues.append(f"Dataset processing would fail: {e}")
            
            return issues
            
        except Exception as e:
            return [f"Frame validation exception: {e}"]
    
    def simulate_get_bbox_data(frame_data, data_format, col_mappings, extra_cols):
        """Simulate the _get_bbox_data method exactly"""
        try:
            heading = frame_data[col_mappings['heading']]
            bbox_x = frame_data[col_mappings['bbox_x']]
            bbox_y = frame_data[col_mappings['bbox_y']]
            
            if np.isnan(heading) or np.isnan(bbox_x) or np.isnan(bbox_y):
                return None
            
            if data_format == 'enhanced':
                bbox_width = frame_data[col_mappings['bbox_width']]
                bbox_height = frame_data[col_mappings['bbox_height']]
                
                if np.isnan(bbox_width) or np.isnan(bbox_height):
                    return None
                
                confidence = 1.0
                if extra_cols['confidence_score'] is not None:
                    conf_val = frame_data[extra_cols['confidence_score']]
                    if not np.isnan(conf_val):
                        confidence = conf_val
                
                return np.array([0, bbox_x, bbox_y, bbox_width, bbox_height, confidence], dtype=np.float32)
            else:
                return np.array([0, bbox_x, bbox_y, 0.05, 0.05, 1.0], dtype=np.float32)
                
        except Exception:
            return None
    
    # Test ALL training frames
    print("🚂 Testing ALL training frames...")
    train_issues = {}
    train_problematic = []
    
    for i, frame_idx in enumerate(tqdm(train_indices, desc="Validating train frames")):
        frame_data = data[frame_idx]
        issues = validate_single_frame(frame_idx, frame_data, is_train=True)
        
        if issues:
            train_issues[frame_idx] = issues
            train_problematic.append(frame_idx)
            
            # Show first 10 problematic frames
            if len(train_problematic) <= 10:
                print(f"  ❌ TRAIN frame {frame_idx} (#{i}): {'; '.join(issues)}")
    
    print(f"🚂 Training validation complete: {len(train_problematic)}/{len(train_indices)} problematic frames")
    print()
    
    # Test ALL validation frames
    print("✅ Testing ALL validation frames...")
    val_issues = {}
    val_problematic = []
    
    for i, frame_idx in enumerate(tqdm(val_indices, desc="Validating val frames")):
        frame_data = data[frame_idx]
        issues = validate_single_frame(frame_idx, frame_data, is_train=False)
        
        if issues:
            val_issues[frame_idx] = issues
            val_problematic.append(frame_idx)
            
            # Show first 10 problematic frames
            if len(val_problematic) <= 10:
                print(f"  ❌ VAL frame {frame_idx} (#{i}): {'; '.join(issues)}")
    
    print(f"✅ Validation validation complete: {len(val_problematic)}/{len(val_indices)} problematic frames")
    print()
    
    # Comprehensive summary
    print("🎯 EXHAUSTIVE VALIDATION SUMMARY")
    print("-" * 40)
    print(f"📊 Total frames in dataset: {total_frames}")
    print(f"✅ Frames passing filter: {len(current_valid_indices)}")
    print(f"🚂 Training samples: {len(train_indices)}")
    print(f"✅ Validation samples: {len(val_indices)}")
    print()
    print(f"❌ Problematic training frames: {len(train_problematic)}/{len(train_indices)} ({len(train_problematic)/len(train_indices)*100:.1f}%)")
    print(f"❌ Problematic validation frames: {len(val_problematic)}/{len(val_indices)} ({len(val_problematic)/len(val_indices)*100:.1f}%)")
    print()
    
    if train_problematic or val_problematic:
        print("🚨 CRITICAL FINDINGS:")
        
        # Analyze types of issues
        all_issues = list(train_issues.values()) + list(val_issues.values())
        issue_types = {}
        
        for frame_issues in all_issues:
            for issue in frame_issues:
                issue_type = issue.split('(')[0].strip()  # Extract main issue type
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        print("📋 Issue breakdown:")
        for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {issue_type}: {count} frames")
        
        print()
        print("🔧 RECOMMENDED ACTIONS:")
        
        if any('_get_bbox_data would return None' in str(issues) for issues in all_issues):
            print("   1. Add stricter filtering in ZarrYOLODataset to exclude frames that would fail _get_bbox_data")
        
        if any('NaN' in str(issues) for issues in all_issues):
            print("   2. Enhance the filtering logic to check ALL required fields, not just heading and bbox_x/y")
        
        if any('wrong ndim' in str(issues) or 'wrong shape' in str(issues) for issues in all_issues):
            print("   3. Fix tensor shape issues in the dataset __getitem__ method")
        
        print()
        print("💡 These problematic frames are likely causing the 0-d tensor error!")
        
        return False, train_problematic, val_problematic
    else:
        print("✅ EXHAUSTIVE VALIDATION PASSED:")
        print("   All selected frames should produce valid YOLO training data")
        print("   The 0-d tensor error might be caused by something else")
        
        return True, [], []

def main():
    parser = argparse.ArgumentParser(description="Exhaustively validate every selected frame")
    parser.add_argument("zarr_path", type=str, help="Path to the video.zarr file")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=42)
    
    args = parser.parse_args()
    
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"❌ Error: Zarr file not found: {zarr_path}")
        return
    
    validation_passed, train_issues, val_issues = validate_every_selected_frame(
        args.zarr_path, args.split_ratio, args.random_seed
    )
    
    if not validation_passed:
        print(f"\n🚨 FOUND THE ROOT CAUSE!")
        print(f"   Problematic training frames: {len(train_issues)}")
        print(f"   Problematic validation frames: {len(val_issues)}")
        print(f"   These frames need to be filtered out of the dataset!")

if __name__ == "__main__":
    main()