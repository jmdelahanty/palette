#!/usr/bin/env python3
"""
Frame-rate aware monotonicity check that understands 120Hz stimulus / 60Hz camera mismatch.
"""

import h5py
import numpy as np
from pathlib import Path
import argparse

def check_frame_rate_aware_monotonicity(h5_path: str):
    """
    Check monotonicity with understanding of frame rate differences.
    """
    h5_path = Path(h5_path)
    
    print(f"\n📊 Frame Rate-Aware Analysis: {h5_path.name}")
    print("=" * 70)
    
    with h5py.File(h5_path, 'r') as f:
        # Get root attributes for frame rates if available
        stimulus_rate = 120  # Hz (default)
        camera_rate = 60     # FPS (default)
        
        if 'stimulus_frame_rate' in f.attrs:
            stimulus_rate = f.attrs['stimulus_frame_rate']
        if 'camera_frame_rate' in f.attrs:
            camera_rate = f.attrs['camera_frame_rate']
        
        print(f"\n⚙️  System Configuration:")
        print(f"  Stimulus rate: {stimulus_rate} Hz")
        print(f"  Camera rate: {camera_rate} FPS")
        print(f"  Expected ratio: {stimulus_rate/camera_rate:.1f} stimulus frames per camera frame")
        
        # Load frame metadata
        metadata = f['/video_metadata/frame_metadata'][:]
        camera_frames = metadata['triggering_camera_frame_id']
        stimulus_frames = metadata['stimulus_frame_num']
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total records: {len(metadata):,}")
        print(f"  Unique camera frames: {len(np.unique(camera_frames)):,}")
        print(f"  Unique stimulus frames: {len(np.unique(stimulus_frames)):,}")
        print(f"  Actual ratio: {len(metadata) / len(np.unique(camera_frames)):.2f} records per camera frame")
        
        # Analyze camera frame progression (removing duplicates)
        unique_camera_frames = []
        prev_frame = -1
        for frame in camera_frames:
            if frame != prev_frame:
                unique_camera_frames.append(frame)
                prev_frame = frame
        
        unique_camera_frames = np.array(unique_camera_frames)
        
        print(f"\n✅ Camera Frame Monotonicity (after deduplication):")
        is_monotonic = np.all(np.diff(unique_camera_frames) > 0)
        print(f"  Strictly monotonic: {'✅ Yes' if is_monotonic else '❌ No'}")
        
        if not is_monotonic:
            backwards = np.where(np.diff(unique_camera_frames) <= 0)[0]
            print(f"  Found {len(backwards)} non-monotonic transitions")
            for idx in backwards[:5]:
                print(f"    {unique_camera_frames[idx]} → {unique_camera_frames[idx+1]}")
        
        # Analyze frame drops (gaps in camera frames)
        gaps = np.diff(unique_camera_frames)
        expected_gap = 1
        
        frame_drops = gaps[gaps > expected_gap]
        if len(frame_drops) > 0:
            print(f"\n⚠️  Camera Frame Drops Detected:")
            print(f"  Total drops: {len(frame_drops)}")
            print(f"  Total frames lost: {np.sum(frame_drops - expected_gap)}")
            print(f"  Largest gap: {np.max(frame_drops)} frames")
            print(f"  Average gap: {np.mean(frame_drops):.1f} frames")
        else:
            print(f"\n✅ No camera frame drops detected")
        
        # Analyze stimulus-camera mapping consistency
        print(f"\n🔗 Stimulus-Camera Mapping Analysis:")
        
        # Group by camera frame
        from collections import defaultdict
        camera_to_stimulus = defaultdict(list)
        for cam, stim in zip(camera_frames, stimulus_frames):
            camera_to_stimulus[cam].append(stim)
        
        # Analyze mapping ratios
        mapping_ratios = [len(stims) for stims in camera_to_stimulus.values()]
        ratio_counts = {}
        for ratio in mapping_ratios:
            ratio_counts[ratio] = ratio_counts.get(ratio, 0) + 1
        
        print(f"  Mapping distribution:")
        for ratio, count in sorted(ratio_counts.items()):
            percentage = (count / len(camera_to_stimulus)) * 100
            expected = "← Expected" if ratio == round(stimulus_rate/camera_rate) else ""
            print(f"    {ratio} stimulus/camera: {count:5d} frames ({percentage:5.1f}%) {expected}")
        
        # Check for stimulus frame continuity within each camera frame
        discontinuities = 0
        for cam_frame, stim_list in camera_to_stimulus.items():
            stim_array = np.array(sorted(stim_list))
            if len(stim_array) > 1:
                gaps = np.diff(stim_array)
                if not np.all(gaps == 1):
                    discontinuities += 1
        
        if discontinuities > 0:
            print(f"\n  ⚠️  Found {discontinuities} camera frames with non-continuous stimulus frames")
        else:
            print(f"\n  ✅ All stimulus frames are continuous within each camera frame")
        
        # Summary
        print(f"\n" + "=" * 70)
        print(f"SUMMARY:")
        
        if is_monotonic and len(frame_drops) == 0 and discontinuities == 0:
            print(f"✅ Data structure is healthy!")
            print(f"   - Camera frames are monotonic (when deduplicated)")
            print(f"   - No unexpected frame drops")
            print(f"   - Stimulus-camera mapping matches expected {stimulus_rate}Hz/{camera_rate}FPS ratio")
        else:
            print(f"⚠️  Some issues detected:")
            if not is_monotonic:
                print(f"   - Camera frames have non-monotonic transitions")
            if len(frame_drops) > 0:
                print(f"   - {len(frame_drops)} frame drops detected")
            if discontinuities > 0:
                print(f"   - {discontinuities} camera frames have non-continuous stimulus mappings")
        
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Frame rate-aware monotonicity check for 120Hz/60FPS systems"
    )
    parser.add_argument(
        'h5_file',
        type=str,
        help='Path to the analysis.h5 file'
    )
    
    args = parser.parse_args()
    
    try:
        check_frame_rate_aware_monotonicity(args.h5_file)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())