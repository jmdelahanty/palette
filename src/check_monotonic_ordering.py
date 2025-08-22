#!/usr/bin/env python3
"""
Check if triggering_camera_frame_id is monotonically increasing
and properly ordered with respect to stimulus_frame_num.
"""

import h5py
import numpy as np
import sys

def check_monotonic_ordering(h5_path):
    """
    Check ordering issues in frame metadata.
    """
    print(f"\n🔍 Checking frame ordering in: {h5_path}")
    print("=" * 70)
    
    with h5py.File(h5_path, 'r') as f:
        metadata = f['/video_metadata/frame_metadata'][:]
        
        # Get the arrays
        stim_frames = metadata['stimulus_frame_num']
        camera_frames = metadata['triggering_camera_frame_id']
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total records: {len(metadata)}")
        print(f"  Unique stimulus frames: {len(np.unique(stim_frames))}")
        print(f"  Unique camera frames: {len(np.unique(camera_frames))}")
        
        # Check if arrays are sorted
        print(f"\n🔄 Ordering Checks:")
        
        # Check stimulus frames
        stim_sorted = np.all(stim_frames[:-1] <= stim_frames[1:])
        print(f"  Stimulus frames monotonically increasing: {stim_sorted}")
        
        if not stim_sorted:
            # Find where it's not sorted
            bad_indices = np.where(stim_frames[:-1] > stim_frames[1:])[0]
            print(f"    ⚠️  Found {len(bad_indices)} places where order decreases")
            for idx in bad_indices[:5]:
                print(f"      Index {idx}: stim {stim_frames[idx]} -> {stim_frames[idx+1]}")
        
        # Check camera frames (should be non-decreasing within same stimulus frame)
        print(f"\n  Camera frame ordering:")
        
        # Group by stimulus frame and check camera ordering within each
        issues = []
        stim_to_cameras = {}
        
        for i, (stim, cam) in enumerate(zip(stim_frames, camera_frames)):
            if stim not in stim_to_cameras:
                stim_to_cameras[stim] = []
            stim_to_cameras[stim].append((i, cam))
        
        # Check for duplicate stimulus frames mapping to different camera frames
        multi_camera_stims = []
        for stim, cameras in stim_to_cameras.items():
            unique_cams = set([c[1] for c in cameras])
            if len(unique_cams) > 1:
                multi_camera_stims.append((stim, unique_cams))
        
        if multi_camera_stims:
            print(f"    ⚠️  Found {len(multi_camera_stims)} stimulus frames with multiple camera frames!")
            print(f"       This might indicate duplicate records")
            for stim, cams in multi_camera_stims[:5]:
                print(f"      Stimulus {stim}: camera frames {sorted(cams)}")
        
        # Check specific problem area around frames 816-818
        print(f"\n🔍 Detailed Check Around Frames 816-818:")
        print("-" * 50)
        
        # Find all records with camera frames 814-820
        mask = (camera_frames >= 814) & (camera_frames <= 820)
        problem_records = metadata[mask]
        
        print(f"Records with camera frames 814-820:")
        print(f"{'Index':<8} {'Stim Frame':<12} {'Camera Frame':<12}")
        print("-" * 35)
        
        indices = np.where(mask)[0]
        for idx, record in zip(indices, problem_records):
            mark = " <--" if record['triggering_camera_frame_id'] in [816, 817, 818] else ""
            print(f"{idx:<8} {record['stimulus_frame_num']:<12} {record['triggering_camera_frame_id']:<12}{mark}")
        
        # Check if there are duplicates
        print(f"\n🔍 Checking for Duplicate Records:")
        print("-" * 50)
        
        # Count occurrences of each (stim, camera) pair
        from collections import Counter
        pairs = [(s, c) for s, c in zip(stim_frames, camera_frames)]
        pair_counts = Counter(pairs)
        
        duplicates = [(pair, count) for pair, count in pair_counts.items() if count > 1]
        
        if duplicates:
            print(f"  ⚠️  Found {len(duplicates)} duplicate (stimulus, camera) pairs!")
            for (stim, cam), count in duplicates[:10]:
                print(f"    Stimulus {stim}, Camera {cam}: appears {count} times")
        else:
            print(f"  ✅ No duplicate (stimulus, camera) pairs found")
        
        # Check interpolation mask for these frames
        if '/analysis/interpolation_mask' in f:
            mask = f['/analysis/interpolation_mask'][:]
            
            print(f"\n🎭 Interpolation Status for Problem Frames:")
            print("-" * 50)
            
            for cam_frame in [816, 817, 818]:
                cam_mask = camera_frames == cam_frame
                if np.any(cam_mask):
                    idx = np.where(cam_mask)[0][0]
                    is_original = mask[idx]
                    stim = stim_frames[idx]
                    print(f"  Camera {cam_frame} (idx {idx}, stim {stim}): "
                          f"{'ORIGINAL' if is_original else 'INTERPOLATED'}")
        
        # Final check: Are stimulus frames unique?
        print(f"\n⚠️  CRITICAL CHECK: Stimulus Frame Uniqueness")
        print("-" * 50)
        
        unique_stims = np.unique(stim_frames)
        if len(unique_stims) != len(stim_frames):
            print(f"  ❌ DUPLICATE STIMULUS FRAMES FOUND!")
            print(f"     Total records: {len(stim_frames)}")
            print(f"     Unique stimulus frames: {len(unique_stims)}")
            print(f"     Duplicates: {len(stim_frames) - len(unique_stims)}")
            
            # Find which stimulus frames are duplicated
            from collections import Counter
            stim_counts = Counter(stim_frames)
            dups = [(s, c) for s, c in stim_counts.items() if c > 1]
            print(f"\n     First 10 duplicated stimulus frames:")
            for stim, count in sorted(dups)[:10]:
                # Find camera frames for this stimulus
                cam_frames_for_stim = camera_frames[stim_frames == stim]
                print(f"       Stim {stim}: appears {count} times, camera frames: {sorted(set(cam_frames_for_stim))}")
        else:
            print(f"  ✅ All stimulus frames are unique")

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_monotonic_ordering.py <h5_file>")
        sys.exit(1)
    
    check_monotonic_ordering(sys.argv[1])

if __name__ == '__main__':
    main()