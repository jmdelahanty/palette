# src/export_detections_to_csv.py

import zarr
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2 # NEW: Imported OpenCV for image writing

def export_detections(zarr_path, output_dir):
    """
    Exports bounding box positions for each assigned ID to separate CSV files
    and saves the background model image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Loading data from: {zarr_path}")
    
    try:
        root = zarr.open(zarr_path, mode='r')
        
        # Load detection data
        latest_detect_run = root['detect_runs'].attrs['latest']
        detect_group = root[f'detect_runs/{latest_detect_run}']
        n_detections = detect_group['n_detections'][:]
        bbox_coords = detect_group['bbox_norm_coords'][:]
        
        # Load ID assignment data
        latest_id_run = root['id_assignments_runs'].attrs['latest']
        id_group = root[f'id_assignments_runs/{latest_id_run}']
        detection_ids = id_group['detection_ids'][:]

        # --- NEW: Load and save the background model ---
        latest_bg_run = root['background_runs'].attrs['latest']
        background_ds = root[f'background_runs/{latest_bg_run}/background_ds'][:]
        bg_output_path = output_dir / "background_model.png"
        cv2.imwrite(str(bg_output_path), background_ds)
        print(f"🖼️  Background model saved to: {bg_output_path}")
        # --- End of new section ---

    except Exception as e:
        print(f"❌ Error loading data from Zarr file: {e}")
        return

    print("📊 Processing and structuring data...")
    
    # ... (rest of the data processing and CSV writing logic is unchanged) ...
    data_list = []
    cumulative_detections = np.cumsum(np.insert(n_detections, 0, 0))

    for frame_idx in tqdm(range(len(n_detections)), desc="Processing frames"):
        num_dets = n_detections[frame_idx]
        if num_dets > 0:
            start_idx = cumulative_detections[frame_idx]
            end_idx = cumulative_detections[frame_idx + 1]
            
            frame_bboxes = bbox_coords[start_idx:end_idx]
            frame_ids = detection_ids[start_idx:end_idx]
            
            for i in range(num_dets):
                bbox = frame_bboxes[i]
                assigned_id = frame_ids[i]
                
                if assigned_id != -1:
                    data_list.append({
                        'frame': frame_idx,
                        'id': assigned_id,
                        'center_x_norm': bbox[0],
                        'center_y_norm': bbox[1],
                        'width_norm': bbox[2],
                        'height_norm': bbox[3]
                    })
    
    if not data_list:
        print("⚠️ No assigned detections found to export.")
        return
        
    df = pd.DataFrame(data_list)
    unique_ids = sorted(df['id'].unique())
    
    print(f"\n✅ Found {len(unique_ids)} unique fish IDs. Exporting to separate CSV files...")
    
    for fish_id in tqdm(unique_ids, desc="Exporting CSVs"):
        id_df = df[df['id'] == fish_id].copy()
        id_df.sort_values(by='frame', inplace=True)
        output_path = output_dir / f"fish_id_{fish_id}.csv"
        id_df.to_csv(output_path, index=False)
        
    print(f"\n🎉 Export complete! {len(unique_ids)} CSV files and background image saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export assigned YOLO detections to CSV files and save the background model.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output files.")
    args = parser.parse_args()
    
    export_detections(args.zarr_path, args.output_dir)

if __name__ == "__main__":
    main()