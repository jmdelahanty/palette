#!/usr/bin/env python3
"""
Zarr Bounding Box Data Inspector
Inspects the video.zarr file to check what bounding box data is available
and whether we have proper center, width, height information for YOLO training.
"""

import zarr
import numpy as np
import argparse
from pathlib import Path

def inspect_bbox_data(zarr_path):
    """Inspect the bounding box data in the zarr file."""
    print(f"🔍 Inspecting bounding box data in: {zarr_path}")
    print("=" * 60)
    
    try:
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"❌ Error opening Zarr file: {e}")
        return
    
    # Check if tracking results exist
    if 'tracking/tracking_results' not in root:
        print("❌ No tracking results found in the Zarr file")
        return
    
    tracking_results = root['tracking/tracking_results']
    column_names = tracking_results.attrs.get('column_names', [])
    
    print(f"📊 Tracking Results Shape: {tracking_results.shape}")
    print(f"📋 Column Names: {column_names}")
    print()
    
    # Map column names to indices for easier access
    col_map = {name: i for i, name in enumerate(column_names)}
    
    # Check what bounding box data we have
    print("🎯 Available Bounding Box Related Data:")
    bbox_cols = [col for col in column_names if 'bbox' in col.lower()]
    for col in bbox_cols:
        print(f"   - {col} (index {col_map[col]})")
    
    print()
    print("📍 Available Keypoint Data:")
    keypoint_cols = [col for col in column_names if any(kw in col.lower() for kw in ['bladder', 'eye', 'x_roi', 'y_roi'])]
    for col in keypoint_cols:
        print(f"   - {col} (index {col_map[col]})")
    
    print()
    
    # Load some sample data to analyze
    print("📈 Sample Data Analysis:")
    print("-" * 40)
    
    # Find frames with valid tracking data (non-NaN)
    valid_mask = ~np.isnan(tracking_results[:, 0])  # Check first column (heading)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print("❌ No valid tracking data found")
        return
    
    print(f"✅ Found {len(valid_indices)} frames with valid tracking data")
    print(f"📊 Total frames: {tracking_results.shape[0]}")
    print(f"📈 Success rate: {len(valid_indices)/tracking_results.shape[0]*100:.1f}%")
    print()
    
    # Analyze first few valid frames
    sample_indices = valid_indices[:min(5, len(valid_indices))]
    print(f"🔬 Sample data from first {len(sample_indices)} valid frames:")
    print()
    
    for i, frame_idx in enumerate(sample_indices):
        data = tracking_results[frame_idx]
        print(f"Frame {frame_idx}:")
        for col_name, col_idx in col_map.items():
            value = data[col_idx]
            print(f"  {col_name:20}: {value:.6f}" if not np.isnan(value) else f"  {col_name:20}: NaN")
        print()
    
    # Check if we have proper bounding box data
    print("🎯 Bounding Box Data Assessment:")
    print("-" * 40)
    
    required_for_yolo = ['bbox_x_norm', 'bbox_y_norm']
    missing_data = []
    
    for req_col in required_for_yolo:
        if req_col not in col_map:
            missing_data.append(req_col)
        else:
            col_idx = col_map[req_col]
            valid_data = ~np.isnan(tracking_results[valid_indices, col_idx])
            print(f"✅ {req_col}: Available ({np.sum(valid_data)}/{len(valid_indices)} valid values)")
    
    if missing_data:
        print(f"❌ Missing required columns: {missing_data}")
    
    # Check if we have keypoint data to calculate bbox dimensions
    keypoint_data_cols = ['bladder_x_roi_norm', 'bladder_y_roi_norm', 
                         'eye_l_x_roi_norm', 'eye_l_y_roi_norm',
                         'eye_r_x_roi_norm', 'eye_r_y_roi_norm']
    
    print()
    print("🔑 Keypoint Data for Bbox Calculation:")
    all_keypoints_available = True
    for kp_col in keypoint_data_cols:
        if kp_col in col_map:
            col_idx = col_map[kp_col]
            valid_data = ~np.isnan(tracking_results[valid_indices, col_idx])
            print(f"✅ {kp_col}: Available ({np.sum(valid_data)}/{len(valid_indices)} valid values)")
        else:
            print(f"❌ {kp_col}: Missing")
            all_keypoints_available = False
    
    # Calculate sample bounding box dimensions if we have keypoint data
    if all_keypoints_available and len(valid_indices) > 0:
        print()
        print("📏 Sample Bounding Box Dimension Calculations:")
        print("-" * 50)
        
        # Calculate bbox dimensions for first few valid frames
        for i, frame_idx in enumerate(sample_indices):
            data = tracking_results[frame_idx]
            
            # Get keypoint coordinates
            bladder_x = data[col_map['bladder_x_roi_norm']]
            bladder_y = data[col_map['bladder_y_roi_norm']]
            eye_l_x = data[col_map['eye_l_x_roi_norm']]
            eye_l_y = data[col_map['eye_l_y_roi_norm']]
            eye_r_x = data[col_map['eye_r_x_roi_norm']]
            eye_r_y = data[col_map['eye_r_y_roi_norm']]
            
            # Check if all keypoints are valid
            keypoints = [bladder_x, bladder_y, eye_l_x, eye_l_y, eye_r_x, eye_r_y]
            if not any(np.isnan(keypoints)):
                # Calculate bbox dimensions from keypoint spread
                x_coords = [bladder_x, eye_l_x, eye_r_x]
                y_coords = [bladder_y, eye_l_y, eye_r_y]
                
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                # Calculate bbox with some margin
                margin_factor = 1.5
                bbox_width = (max_x - min_x) * margin_factor
                bbox_height = (max_y - min_y) * margin_factor
                
                # Center coordinates (already available)
                center_x = data[col_map['bbox_x_norm']]
                center_y = data[col_map['bbox_y_norm']]
                
                print(f"Frame {frame_idx}:")
                print(f"  Keypoint spread: x=[{min_x:.3f}, {max_x:.3f}], y=[{min_y:.3f}, {max_y:.3f}]")
                print(f"  Calculated bbox: center=({center_x:.3f}, {center_y:.3f}), size=({bbox_width:.3f}, {bbox_height:.3f})")
                print(f"  YOLO format: [0, {center_x:.3f}, {center_y:.3f}, {bbox_width:.3f}, {bbox_height:.3f}]")
                print()
    
    # Provide recommendations
    print("💡 Recommendations:")
    print("-" * 20)
    
    if not missing_data and all_keypoints_available:
        print("✅ Your Zarr file has all the necessary data for YOLO bounding box training!")
        print("   - Center coordinates: bbox_x_norm, bbox_y_norm")
        print("   - Keypoint data available to calculate width/height")
        print("   - You can proceed with the YOLO dataset implementation")
    else:
        print("⚠️  Some data is missing or needs to be added:")
        if missing_data:
            print(f"   - Missing bbox center data: {missing_data}")
        if not all_keypoints_available:
            print("   - Missing some keypoint data for bbox dimension calculation")
        print("   - You may need to re-run the tracking pipeline or modify the data structure")
    
    return root, col_map, valid_indices

def main():
    parser = argparse.ArgumentParser(
        description="Inspect bounding box data in a Zarr file for YOLO training compatibility"
    )
    parser.add_argument("zarr_path", type=str, help="Path to the video.zarr file")
    args = parser.parse_args()
    
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"❌ Error: File not found: {zarr_path}")
        return
    
    inspect_bbox_data(zarr_path)

if __name__ == "__main__":
    main()