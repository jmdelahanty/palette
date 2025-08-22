import h5py
import numpy as np
import argparse
from collections import Counter, defaultdict

def inspect_single_frame(h5_filepath, frame_number):
    """
    Inspects all available data for a single camera frame number in the H5 file.
    """
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
                if '/tracking_data/chaser_states' in f:
                    chaser_ds = f['/tracking_data/chaser_states']
                    matching_chaser = chaser_ds[chaser_ds['stimulus_frame_num'] == stimulus_frame_num]
                    if len(matching_chaser) > 0:
                        print(f"  ✅ Found {len(matching_chaser)} chaser state(s) for stimulus frame {stimulus_frame_num}!")
                    else:
                        print(f"  ❌ No chaser states found for stimulus frame {stimulus_frame_num}.")
                else:
                    print("  - Dataset '/tracking_data/chaser_states' not found.")
            else:
                print("  - Skipping chaser check because no stimulus frame number could be found.")

    except Exception as e:
        print(f"An unexpected error occurred during inspection: {e}")


def analyze_bounding_box_gaps(h5_filepath):
    """
    Analyzes gaps in bounding box payload_frame_id values.
    """
    print("\n--- 🔍 Analyzing Bounding Box Frame ID Gaps ---")
    
    try:
        with h5py.File(h5_filepath, 'r') as f:
            if '/tracking_data/bounding_boxes' not in f:
                print("  - Dataset '/tracking_data/bounding_boxes' not found.")
                return
            
            boxes_ds = f['/tracking_data/bounding_boxes']
            
            # Get unique frame IDs from bounding boxes
            all_frame_ids = boxes_ds['payload_frame_id'][:]
            unique_frame_ids = np.unique(all_frame_ids)
            unique_frame_ids_sorted = np.sort(unique_frame_ids)
            
            print(f"\n  📊 Bounding Box Frame Statistics:")
            print(f"  - Total bounding boxes: {len(all_frame_ids)}")
            print(f"  - Unique frame IDs: {len(unique_frame_ids)}")
            print(f"  - Frame ID range: {unique_frame_ids_sorted[0]} to {unique_frame_ids_sorted[-1]}")
            
            # Analyze gaps
            expected_frames = unique_frame_ids_sorted[-1] - unique_frame_ids_sorted[0] + 1
            actual_frames = len(unique_frame_ids)
            missing_frames = expected_frames - actual_frames
            
            print(f"  - Expected frames in range: {expected_frames}")
            print(f"  - Actual frames present: {actual_frames}")
            print(f"  - Missing frames: {missing_frames}")
            
            # Find specific gaps
            diffs = np.diff(unique_frame_ids_sorted)
            gap_indices = np.where(diffs > 1)[0]
            
            if len(gap_indices) == 0:
                print("\n  ✅ No gaps found in bounding box frame IDs!")
            else:
                print(f"\n  🚨 Found {len(gap_indices)} gap(s) in bounding box frame IDs:")
                
                # Analyze gap patterns
                gap_sizes = []
                gap_positions = []
                
                for i, index in enumerate(gap_indices):
                    gap_start = unique_frame_ids_sorted[index]
                    gap_end = unique_frame_ids_sorted[index + 1]
                    gap_size = gap_end - gap_start - 1
                    gap_sizes.append(gap_size)
                    gap_positions.append(gap_start)
                    
                    if i < 20:  # Show first 20 gaps
                        if gap_size == 1:
                            print(f"    - Gap {i+1}: Missing frame {gap_start + 1} (after frame {gap_start})")
                        else:
                            print(f"    - Gap {i+1}: Missing frames {gap_start + 1} to {gap_end - 1} ({gap_size} frames)")
                
                if len(gap_indices) > 20:
                    print(f"    ... and {len(gap_indices) - 20} more gaps")
                
                # Analyze gap patterns
                print(f"\n  📈 Gap Pattern Analysis:")
                gap_size_counter = Counter(gap_sizes)
                for size, count in sorted(gap_size_counter.items()):
                    print(f"    - Gaps of size {size}: {count} occurrences")
                
                # Check for periodic gaps (every 60 frames)
                print(f"\n  🔄 Checking for Periodic Gaps (every 60 frames):")
                gap_intervals = np.diff(gap_positions)
                
                # Check if gaps occur at regular intervals
                interval_counter = Counter(gap_intervals)
                most_common_interval = interval_counter.most_common(1)[0] if interval_counter else (0, 0)
                
                if most_common_interval[1] > 1:
                    print(f"    - Most common interval between gaps: {most_common_interval[0]} frames ({most_common_interval[1]} occurrences)")
                
                # Check specifically for 60-frame intervals
                sixty_frame_gaps = sum(1 for interval in gap_intervals if 58 <= interval <= 62)  # Allow small variation
                if sixty_frame_gaps > 0:
                    print(f"    - Gaps occurring approximately every 60 frames: {sixty_frame_gaps}")
                    
                    # Show examples
                    print(f"\n    Examples of ~60-frame periodic gaps:")
                    for i in range(min(5, len(gap_intervals))):
                        if 58 <= gap_intervals[i] <= 62:
                            print(f"      - Gap at frame {gap_positions[i]} -> next gap at frame {gap_positions[i+1]} (interval: {gap_intervals[i]})")
                
                # Check if all gaps are single-frame gaps
                if all(size == 1 for size in gap_sizes):
                    print(f"\n  ⚠️  All gaps are single-frame gaps (exactly 1 frame missing each time)")
                    
                    # Check if they follow a pattern
                    if len(set(gap_intervals)) == 1:
                        print(f"  🎯 Perfect periodicity detected! Gaps occur exactly every {gap_intervals[0]} frames")
                    elif len(gap_intervals) > 0:
                        avg_interval = np.mean(gap_intervals)
                        std_interval = np.std(gap_intervals)
                        print(f"  📊 Average interval between gaps: {avg_interval:.1f} ± {std_interval:.1f} frames")
            
            # Compare with frame_metadata gaps
            print("\n  🔗 Comparing with Frame Metadata Gaps:")
            if '/video_metadata/frame_metadata' in f:
                metadata_ds = f['/video_metadata/frame_metadata']
                camera_frame_ids_meta = np.unique(metadata_ds['triggering_camera_frame_id'][:])
                
                # Find frames that are in metadata but not in bounding boxes
                in_meta_not_boxes = set(camera_frame_ids_meta) - set(unique_frame_ids)
                in_boxes_not_meta = set(unique_frame_ids) - set(camera_frame_ids_meta)
                
                if in_meta_not_boxes:
                    print(f"    - Frames in metadata but NOT in bounding boxes: {len(in_meta_not_boxes)}")
                    print(f"      Examples (first 5): {sorted(list(in_meta_not_boxes))[:5]}")
                
                if in_boxes_not_meta:
                    print(f"    - Frames in bounding boxes but NOT in metadata: {len(in_boxes_not_meta)}")
                    print(f"      Examples (first 5): {sorted(list(in_boxes_not_meta))[:5]}")
                
                if not in_meta_not_boxes and not in_boxes_not_meta:
                    print(f"    - ✅ Bounding box and metadata frame IDs match perfectly!")
                    
    except Exception as e:
        print(f"An unexpected error occurred during bounding box gap analysis: {e}")


def analyze_frame_reuse(h5_filepath):
    """
    Analyzes how often stimulus frames reuse the same camera frame.
    """
    print("\n--- 🔄 Analyzing Camera Frame Reuse for Stimulus Frames ---")
    
    try:
        with h5py.File(h5_filepath, 'r') as f:
            if '/video_metadata/frame_metadata' not in f:
                print("  - Dataset '/video_metadata/frame_metadata' not found.")
                return
            
            metadata_ds = f['/video_metadata/frame_metadata']
            
            # Create mapping of stimulus_frame -> camera_frame
            stimulus_to_camera = defaultdict(list)
            for record in metadata_ds:
                stimulus_frame = record['stimulus_frame_num']
                camera_frame = record['triggering_camera_frame_id']
                stimulus_to_camera[stimulus_frame].append(camera_frame)
            
            # Analyze reuse patterns
            reuse_counts = Counter()
            multiple_camera_frames = []
            
            for stim_frame, cam_frames in stimulus_to_camera.items():
                unique_cam_frames = set(cam_frames)
                num_unique = len(unique_cam_frames)
                reuse_counts[num_unique] += 1
                
                if num_unique > 1:
                    multiple_camera_frames.append((stim_frame, unique_cam_frames))
            
            # Print statistics
            print(f"\n  📊 Frame Reuse Statistics:")
            print(f"  - Total stimulus frames: {len(stimulus_to_camera)}")
            
            for count, frequency in sorted(reuse_counts.items()):
                percentage = (frequency / len(stimulus_to_camera)) * 100
                if count == 1:
                    print(f"  - Stimulus frames using 1 unique camera frame: {frequency} ({percentage:.1f}%)")
                else:
                    print(f"  - Stimulus frames using {count} unique camera frames: {frequency} ({percentage:.1f}%)")
            
            # Show examples of multiple camera frame usage
            if multiple_camera_frames:
                print(f"\n  ⚠️  Found {len(multiple_camera_frames)} stimulus frames using multiple camera frames!")
                print("  Examples (first 5):")
                for stim_frame, cam_frames in multiple_camera_frames[:5]:
                    cam_list = sorted(list(cam_frames))
                    print(f"    - Stimulus frame {stim_frame}: uses camera frames {cam_list}")
            
            # Analyze consecutive camera frame usage
            print("\n  🔍 Analyzing Consecutive Camera Frame Usage:")
            camera_frame_usage = Counter()
            
            # Sort by stimulus frame number to analyze in order
            sorted_mapping = sorted(stimulus_to_camera.items())
            
            consecutive_reuse = 0
            last_camera_frame = None
            max_consecutive = 0
            consecutive_sequences = []
            current_sequence_start = None
            
            for stim_frame, cam_frames in sorted_mapping:
                # Assuming single camera frame per stimulus frame for this analysis
                if len(set(cam_frames)) == 1:
                    current_camera = list(set(cam_frames))[0]
                    camera_frame_usage[current_camera] += 1
                    
                    if current_camera == last_camera_frame:
                        if consecutive_reuse == 0:
                            current_sequence_start = stim_frame - 1
                        consecutive_reuse += 1
                        max_consecutive = max(max_consecutive, consecutive_reuse + 1)
                    else:
                        if consecutive_reuse > 0:
                            consecutive_sequences.append({
                                'start_stim': current_sequence_start,
                                'end_stim': stim_frame - 1,
                                'camera_frame': last_camera_frame,
                                'count': consecutive_reuse + 1
                            })
                        consecutive_reuse = 0
                        current_sequence_start = None
                    
                    last_camera_frame = current_camera
            
            # Check for final sequence
            if consecutive_reuse > 0:
                consecutive_sequences.append({
                    'start_stim': current_sequence_start,
                    'end_stim': sorted_mapping[-1][0],
                    'camera_frame': last_camera_frame,
                    'count': consecutive_reuse + 1
                })
            
            # Print camera frame usage statistics
            usage_distribution = Counter(camera_frame_usage.values())
            print(f"  - Camera frames used exactly once: {sum(1 for count in camera_frame_usage.values() if count == 1)}")
            print(f"  - Camera frames used multiple times: {sum(1 for count in camera_frame_usage.values() if count > 1)}")
            
            if max_consecutive > 1:
                print(f"  - Maximum consecutive stimulus frames using same camera frame: {max_consecutive}")
                
                # Show examples of consecutive reuse
                if consecutive_sequences:
                    print(f"\n  📍 Found {len(consecutive_sequences)} sequences of camera frame reuse:")
                    # Sort by count to show longest sequences first
                    sorted_sequences = sorted(consecutive_sequences, key=lambda x: x['count'], reverse=True)
                    for i, seq in enumerate(sorted_sequences[:5]):
                        print(f"    - Camera frame {seq['camera_frame']} used {seq['count']} times consecutively")
                        print(f"      (stimulus frames {seq['start_stim']} to {seq['end_stim']})")
            
            # Distribution of how many times each camera frame is used
            print(f"\n  📈 Camera Frame Usage Distribution:")
            for times_used, count in sorted(usage_distribution.items()):
                if times_used == 1:
                    print(f"  - Camera frames used exactly 1 time: {count}")
                else:
                    print(f"  - Camera frames used {times_used} times: {count}")
                    
    except Exception as e:
        print(f"An unexpected error occurred during frame reuse analysis: {e}")


def analyze_overall_data(h5_filepath):
    """
    Analyzes the integrity and continuity of key datasets in the H5 file.
    """
    print(f"\n--- 📊 Analyzing Overall Data Integrity for {h5_filepath} ---")
    try:
        with h5py.File(h5_filepath, 'r') as f:
            
            # --- 1. Analyze 'triggering_camera_frame_id' Gaps ---
            print("\n[1/4] Analyzing 'triggering_camera_frame_id' for gaps...")
            if '/video_metadata/frame_metadata' in f:
                metadata_ds = f['/video_metadata/frame_metadata']
                camera_frame_ids = np.sort(metadata_ds['triggering_camera_frame_id'][:])
                
                diffs = np.diff(camera_frame_ids)
                gap_indices = np.where(diffs > 1)[0]
                
                if len(gap_indices) == 0:
                    print("  ✅ No gaps found. Camera frame IDs are contiguous.")
                else:
                    print(f"  🚨 Found {len(gap_indices)} gap(s) in camera frame IDs:")
                    for i, index in enumerate(gap_indices[:20]):  # Limit to first 20 gaps
                        gap_start = camera_frame_ids[index] + 1
                        gap_end = camera_frame_ids[index + 1] - 1
                        if gap_start == gap_end:
                            print(f"    - Gap {i+1}: Missing ID {gap_start}")
                        else:
                            print(f"    - Gap {i+1}: Missing IDs from {gap_start} to {gap_end}")
                    if len(gap_indices) > 20:
                        print(f"    ... and {len(gap_indices) - 20} more gaps")
            else:
                print("  - Dataset '/video_metadata/frame_metadata' not found.")

            # --- 2. Analyze Bounding Box Gaps (NEW) ---
            analyze_bounding_box_gaps(h5_filepath)

            # --- 3. Analyze 'chaser_states' ---
            print("\n[3/4] Analyzing 'chaser_states' for stimulus frame gaps...")
            if '/tracking_data/chaser_states' in f:
                chaser_states = f['/tracking_data/chaser_states']
                frame_numbers = np.sort(chaser_states['stimulus_frame_num'][:])
                diffs = np.diff(frame_numbers)
                gap_indices = np.where(diffs > 1)[0]

                if len(gap_indices) == 0:
                    print("  ✅ No gaps found in 'stimulus_frame_num'.")
                else:
                    print(f"  🚨 Found {len(gap_indices)} gap(s) in stimulus frame numbers.")
                    for i, index in enumerate(gap_indices[:10]):  # Show first 10
                        gap_start = frame_numbers[index] + 1
                        gap_end = frame_numbers[index + 1] - 1
                        if gap_start == gap_end:
                            print(f"    - Gap {i+1}: Missing stimulus frame {gap_start}")
                        else:
                            print(f"    - Gap {i+1}: Missing stimulus frames {gap_start} to {gap_end}")
            else:
                print("  - Dataset '/tracking_data/chaser_states' not found.")
            
            # --- 4. Analyze Frame Reuse ---
            analyze_frame_reuse(h5_filepath)
                
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