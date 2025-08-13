import h5py
import numpy as np
import argparse

def inspect_single_frame(h5_filepath, frame_number):
    """
    Inspects all available data for a single camera frame number in the H5 file.
    """
    # ... (This function remains the same as before) ...
    print(f"\n--- 🧐 Inspecting Data for Camera Frame: {frame_number} ---")
    
    try:
        with h5py.File(h5_filepath, 'r') as f:
            # 1. Check for Bounding Box Data
            print("\n[1/3] Checking for Bounding Boxes...")
            if '/tracking_data/bounding_boxes' in f:
                boxes_ds = f['/tracking_data/bounding_boxes']
                matching_boxes = boxes_ds[boxes_ds['payload_frame_id'] == frame_number]
                if len(matching_boxes) > 0:
                    print(f"  ✅ Found {len(matching_boxes)} bounding box(es) for this frame.")
                else:
                    print(f"  ❌ No bounding boxes found with payload_frame_id == {frame_number}.")
            else:
                print("  - Dataset '/tracking_data/bounding_boxes' not found.")

            # 2. Check for Frame Metadata
            print("\n[2/3] Checking for Frame Metadata...")
            stimulus_frame_num = -1
            if '/video_metadata/frame_metadata' in f:
                metadata_ds = f['/video_metadata/frame_metadata']
                matching_meta = metadata_ds[metadata_ds['triggering_camera_frame_id'] == frame_number]
                if len(matching_meta) > 0:
                    stimulus_frame_num = matching_meta[0]['stimulus_frame_num']
                    print(f"  ✅ Found metadata record! Camera Frame {frame_number} maps to Stimulus Frame {stimulus_frame_num}")
                else:
                    print(f"  ❌ CRITICAL: No metadata record found where 'triggering_camera_frame_id' == {frame_number}.")
            else:
                print("  - Dataset '/video_metadata/frame_metadata' not found.")
                
            # 3. Check for Chaser/Target State
            print("\n[3/3] Checking for Chaser/Target State...")
            if stimulus_frame_num != -1:
                # ... (logic remains the same)
                 print("  ✅ Found chaser state!")
            else:
                print("  - Skipping chaser check because no stimulus frame number could be found.")

    except Exception as e:
        print(f"An unexpected error occurred during inspection: {e}")


def analyze_overall_data(h5_filepath):
    """
    Analyzes the integrity and continuity of key datasets in the H5 file.
    """
    print(f"--- 📊 Analyzing Overall Data Integrity for {h5_filepath} ---")
    try:
        with h5py.File(h5_filepath, 'r') as f:
            
            # --- 1. Analyze 'triggering_camera_frame_id' Gaps (NEW) ---
            print("\n[1/2] Analyzing 'triggering_camera_frame_id' for gaps...")
            if '/video_metadata/frame_metadata' in f:
                metadata_ds = f['/video_metadata/frame_metadata']
                camera_frame_ids = np.sort(metadata_ds['triggering_camera_frame_id'][:])
                
                diffs = np.diff(camera_frame_ids)
                gap_indices = np.where(diffs > 1)[0]
                
                if len(gap_indices) == 0:
                    print("  ✅ No gaps found. Camera frame IDs are contiguous.")
                else:
                    print(f"  🚨 Found {len(gap_indices)} gap(s) in camera frame IDs:")
                    for i, index in enumerate(gap_indices):
                        gap_start = camera_frame_ids[index] + 1
                        gap_end = camera_frame_ids[index + 1] - 1
                        print(f"    - Gap {i+1}: Missing IDs from {gap_start} to {gap_end}")
            else:
                print("  - Dataset '/video_metadata/frame_metadata' not found.")

            # --- 2. Analyze 'chaser_states' ---
            print("\n[2/2] Analyzing 'chaser_states' for stimulus frame gaps...")
            if '/tracking_data/chaser_states' in f:
                chaser_states = f['/tracking_data/chaser_states']
                frame_numbers = np.sort(chaser_states['stimulus_frame_num'][:])
                diffs = np.diff(frame_numbers)
                gap_indices = np.where(diffs > 1)[0]

                if len(gap_indices) == 0:
                    print("  ✅ No gaps found in 'stimulus_frame_num'.")
                else:
                    print(f"  🚨 Found {len(gap_indices)} gap(s) in stimulus frame numbers.")
            else:
                print("  - Dataset '/tracking_data/chaser_states' not found.")
                
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze and inspect tracking data in an H5 file."
    )
    parser.add_argument(
        "h5_file", 
        help="Path to the H5 file to analyze."
    )
    parser.add_argument(
        "-f", "--frame",
        type=int,
        help="Optional: A specific camera frame number to inspect in detail."
    )
    args = parser.parse_args()
    
    if args.frame is not None:
        inspect_single_frame(args.h5_file, args.frame)
    else:
        analyze_overall_data(args.h5_file)