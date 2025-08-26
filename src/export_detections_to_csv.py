#!/usr/bin/env python3
"""
Export fish tracking detections to CSV files with multiple coordinate systems.
Exports one CSV per fish ID with both normalized and pixel coordinates.
"""

import zarr
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import yaml
from datetime import datetime
from typing import Dict, List, Tuple

def load_config(config_path: str) -> Dict:
    """Load pipeline configuration if available."""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def export_detections_with_coordinates(zarr_path: str, output_dir: str, 
                                      include_interpolated: bool = True,
                                      config_path: str = "src/pipeline_config.yaml"):
    """
    Export detections with multiple coordinate systems for each fish ID.
    
    Args:
        zarr_path: Path to the zarr file
        output_dir: Directory to save CSV files
        include_interpolated: Whether to include interpolated detections
        config_path: Path to pipeline config for additional metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Loading data from: {zarr_path}")
    
    try:
        root = zarr.open(zarr_path, mode='r')
        
        # Get image dimensions
        img_width = root.attrs.get('width', 4512)
        img_height = root.attrs.get('height', 4512)
        fps = root.attrs.get('fps', 60.0)
        ds_width = 640
        ds_height = 640
        
        # Load detection data
        detect_group = root['detect_runs']
        latest_detect = detect_group.attrs['latest']
        n_detections = detect_group[latest_detect]['n_detections'][:]
        bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
        
        # Load ID assignments
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in root else 'id_assignments'
        id_group = root[id_key]
        latest_id = id_group.attrs['latest']
        detection_ids = id_group[latest_id]['detection_ids'][:]
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        
        # Get sub-dish ROI information if available
        subdish_rois = {}
        if 'parameters' in id_group[latest_id].attrs:
            params = id_group[latest_id].attrs['parameters']
            if 'sub_dish_rois' in params:
                for roi in params['sub_dish_rois']:
                    subdish_rois[roi['id']] = roi['roi_pixels']
        
        # Save background model
        latest_bg_run = root['background_runs'].attrs['latest']
        background_ds = root[f'background_runs/{latest_bg_run}/background_ds'][:]
        bg_output_path = output_dir / "background_model.png"
        cv2.imwrite(str(bg_output_path), background_ds)
        print(f"🖼️  Background model saved to: {bg_output_path}")
        
    except Exception as e:
        print(f"❌ Error loading data from Zarr: {e}")
        return
    
    print("📊 Processing detection data...")
    
    # Process original detections
    data_by_id = {}
    cumulative_idx = 0
    
    for frame_idx in tqdm(range(len(n_detections)), desc="Processing detections"):
        frame_det_count = int(n_detections[frame_idx])
        
        if frame_det_count > 0:
            frame_detection_ids = detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
            frame_bboxes = bbox_coords[cumulative_idx:cumulative_idx + frame_det_count]
            
            for i in range(frame_det_count):
                assigned_id = int(frame_detection_ids[i])
                if assigned_id == -1:
                    continue
                
                if assigned_id not in data_by_id:
                    data_by_id[assigned_id] = []
                
                bbox = frame_bboxes[i]
                
                # bbox format: [center_x_norm, center_y_norm, width_norm, height_norm]
                # Normalized to 640x640 downsampled space
                center_x_norm = bbox[0]
                center_y_norm = bbox[1]
                width_norm = bbox[2]
                height_norm = bbox[3]
                
                # Convert to pixel coordinates
                center_x_ds = center_x_norm * ds_width
                center_y_ds = center_y_norm * ds_height
                width_ds = width_norm * ds_width
                height_ds = height_norm * ds_height
                
                # Convert to full resolution
                scale_factor = img_width / ds_width
                center_x_full = center_x_ds * scale_factor
                center_y_full = center_y_ds * scale_factor
                width_full = width_ds * scale_factor
                height_full = height_ds * scale_factor
                
                # Calculate time
                time_sec = frame_idx / fps
                
                data_by_id[assigned_id].append({
                    'frame': frame_idx,
                    'time_sec': time_sec,
                    'detection_type': 'original',
                    # Normalized coordinates (0-1)
                    'center_x_norm': center_x_norm,
                    'center_y_norm': center_y_norm,
                    'width_norm': width_norm,
                    'height_norm': height_norm,
                    # 640x640 pixel coordinates
                    'center_x_640': center_x_ds,
                    'center_y_640': center_y_ds,
                    'width_640': width_ds,
                    'height_640': height_ds,
                    # Full resolution coordinates
                    'center_x_full': center_x_full,
                    'center_y_full': center_y_full,
                    'width_full': width_full,
                    'height_full': height_full,
                    # Bounding box corners (640x640)
                    'x1_640': center_x_ds - width_ds/2,
                    'y1_640': center_y_ds - height_ds/2,
                    'x2_640': center_x_ds + width_ds/2,
                    'y2_640': center_y_ds + height_ds/2,
                })
        
        cumulative_idx += frame_det_count
    
    # Add interpolated detections if available
    if include_interpolated and 'interpolated_detections' in root:
        print("📈 Adding interpolated detections...")
        interp_group = root['interpolated_detections']
        if 'latest' in interp_group.attrs:
            latest_interp = interp_group.attrs['latest']
            interp_data = interp_group[latest_interp]
            
            frame_indices = interp_data['frame_indices'][:]
            roi_ids = interp_data['roi_ids'][:]
            bboxes = interp_data['bboxes'][:]
            
            for i in range(len(frame_indices)):
                frame_idx = int(frame_indices[i])
                assigned_id = int(roi_ids[i])
                bbox = bboxes[i]
                
                if assigned_id not in data_by_id:
                    data_by_id[assigned_id] = []
                
                # Same coordinate conversion
                center_x_norm = bbox[0]
                center_y_norm = bbox[1]
                width_norm = bbox[2]
                height_norm = bbox[3]
                
                center_x_ds = center_x_norm * ds_width
                center_y_ds = center_y_norm * ds_height
                width_ds = width_norm * ds_width
                height_ds = height_norm * ds_height
                
                scale_factor = img_width / ds_width
                center_x_full = center_x_ds * scale_factor
                center_y_full = center_y_ds * scale_factor
                width_full = width_ds * scale_factor
                height_full = height_ds * scale_factor
                
                time_sec = frame_idx / fps
                
                data_by_id[assigned_id].append({
                    'frame': frame_idx,
                    'time_sec': time_sec,
                    'detection_type': 'interpolated',
                    'center_x_norm': center_x_norm,
                    'center_y_norm': center_y_norm,
                    'width_norm': width_norm,
                    'height_norm': height_norm,
                    'center_x_640': center_x_ds,
                    'center_y_640': center_y_ds,
                    'width_640': width_ds,
                    'height_640': height_ds,
                    'center_x_full': center_x_full,
                    'center_y_full': center_y_full,
                    'width_full': width_full,
                    'height_full': height_full,
                    'x1_640': center_x_ds - width_ds/2,
                    'y1_640': center_y_ds - height_ds/2,
                    'x2_640': center_x_ds + width_ds/2,
                    'y2_640': center_y_ds + height_ds/2,
                })
    
    # Export to CSV files
    print(f"\n✅ Found {len(data_by_id)} unique fish IDs. Exporting to CSV files...")
    
    # Create metadata file
    metadata = {
        'export_timestamp': datetime.now().isoformat(),
        'source_zarr': str(zarr_path),
        'image_dimensions': {'width': img_width, 'height': img_height},
        'downsampled_dimensions': {'width': ds_width, 'height': ds_height},
        'fps': fps,
        'total_frames': len(n_detections),
        'fish_ids': sorted(data_by_id.keys()),
        'subdish_rois': subdish_rois,
        'include_interpolated': include_interpolated
    }
    
    # Save metadata
    metadata_path = output_dir / 'export_metadata.yaml'
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    # Export individual CSV files
    for fish_id in tqdm(sorted(data_by_id.keys()), desc="Writing CSV files"):
        df = pd.DataFrame(data_by_id[fish_id])
        df.sort_values(by=['frame'], inplace=True)
        
        # Add subdish information if available
        if fish_id in subdish_rois:
            x, y, w, h = subdish_rois[fish_id]
            df['subdish_x'] = x
            df['subdish_y'] = y
            df['subdish_w'] = w
            df['subdish_h'] = h
        
        output_path = output_dir / f"fish_{fish_id:02d}_tracking.csv"
        df.to_csv(output_path, index=False, float_format='%.6f')
        
        # Print summary for this fish
        orig_count = len(df[df['detection_type'] == 'original'])
        interp_count = len(df[df['detection_type'] == 'interpolated'])
        print(f"  Fish {fish_id}: {orig_count} original + {interp_count} interpolated = {len(df)} total")
    
    # Create summary statistics
    summary_df = []
    for fish_id, data in data_by_id.items():
        df = pd.DataFrame(data)
        orig_df = df[df['detection_type'] == 'original']
        summary_df.append({
            'fish_id': fish_id,
            'total_detections': len(df),
            'original_detections': len(orig_df),
            'interpolated_detections': len(df) - len(orig_df),
            'first_frame': df['frame'].min(),
            'last_frame': df['frame'].max(),
            'coverage_frames': df['frame'].max() - df['frame'].min() + 1,
            'detection_rate': len(orig_df) / (df['frame'].max() - df['frame'].min() + 1) if len(orig_df) > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    
    print(f"\n🎉 Export complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Files created:")
    print(f"   - {len(data_by_id)} individual fish CSV files")
    print(f"   - background_model.png")
    print(f"   - export_metadata.yaml")
    print(f"   - summary_statistics.csv")
    

def main():
    parser = argparse.ArgumentParser(
        description="Export fish tracking detections to CSV files with multiple coordinate systems"
    )
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file")
    parser.add_argument("output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--no-interpolated", action="store_true", 
                       help="Exclude interpolated detections")
    parser.add_argument("--config", type=str, default="src/pipeline_config.yaml",
                       help="Path to pipeline config file")
    
    args = parser.parse_args()
    
    export_detections_with_coordinates(
        args.zarr_path, 
        args.output_dir,
        include_interpolated=not args.no_interpolated,
        config_path=args.config
    )

if __name__ == "__main__":
    main()