#!/usr/bin/env python3
"""
Detailed analysis of frame ID issues in analysis.h5 files.
Focuses on understanding backwards steps and duplicate mappings.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import Counter, defaultdict

def analyze_frame_issues(h5_path: str, verbose: bool = True):
    """
    Perform detailed analysis of frame ID issues.
    """
    h5_path = Path(h5_path)
    
    print(f"\n🔍 Analyzing frame issues in: {h5_path.name}")
    print("=" * 70)
    
    with h5py.File(h5_path, 'r') as f:
        # Load frame metadata
        metadata = f['/video_metadata/frame_metadata'][:]
        
        camera_frames = metadata['triggering_camera_frame_id']
        stimulus_frames = metadata['stimulus_frame_num']
        
        print(f"\n📊 Dataset Overview:")
        print(f"  Total records: {len(metadata):,}")
        print(f"  Camera frame range: {camera_frames.min():,} to {camera_frames.max():,}")
        print(f"  Stimulus frame range: {stimulus_frames.min():,} to {stimulus_frames.max():,}")
        
        # Analyze backwards steps
        print(f"\n🔄 Backwards Steps Analysis:")
        backwards_indices = np.where(np.diff(camera_frames) < 0)[0]
        
        if len(backwards_indices) > 0:
            print(f"  Found {len(backwards_indices)} backwards steps")
            
            # Analyze the pattern of backwards steps
            step_sizes = []
            for idx in backwards_indices:
                step = camera_frames[idx + 1] - camera_frames[idx]
                step_sizes.append(step)
            
            step_counter = Counter(step_sizes)
            print(f"\n  Step size distribution:")
            for step_size, count in sorted(step_counter.items())[:10]:
                print(f"    Step {step_size:4d}: {count:5d} occurrences")
            
            # Show examples of backwards steps with context
            print(f"\n  Examples of backwards steps (showing index, camera_frame, stimulus_frame):")
            for i, idx in enumerate(backwards_indices[:5]):
                print(f"\n    Example {i+1}:")
                for j in range(max(0, idx-2), min(len(camera_frames), idx+3)):
                    marker = " <--" if j == idx else "    "
                    if j == idx + 1:
                        marker = " <-- backwards to here"
                    print(f"      [{j:5d}] Camera: {camera_frames[j]:5d}, Stimulus: {stimulus_frames[j]:5d}{marker}")
        else:
            print(f"  No backwards steps found")
        
        # Analyze duplicates
        print(f"\n📋 Duplicate Analysis:")
        
        # Camera frame duplicates
        cam_counter = Counter(camera_frames)
        cam_duplicates = {k: v for k, v in cam_counter.items() if v > 1}
        
        if cam_duplicates:
            print(f"  Camera frames with duplicates: {len(cam_duplicates):,}")
            print(f"  Total duplicate entries: {sum(v for v in cam_duplicates.values()) - len(cam_duplicates):,}")
            
            # Show distribution of duplication counts
            dup_count_dist = Counter(cam_duplicates.values())
            print(f"\n  Duplication frequency:")
            for dup_count, num_frames in sorted(dup_count_dist.items())[:10]:
                print(f"    {num_frames:5d} frames appear {dup_count} times")
            
            # Analyze stimulus frames for duplicated camera frames
            print(f"\n  Analyzing stimulus frame mapping for duplicated camera frames...")
            
            # Sample a few duplicated camera frames
            sample_cam_frames = sorted(cam_duplicates.keys())[:5]
            for cam_frame in sample_cam_frames:
                indices = np.where(camera_frames == cam_frame)[0]
                stim_frames_for_cam = stimulus_frames[indices]
                unique_stims = np.unique(stim_frames_for_cam)
                
                print(f"\n    Camera frame {cam_frame}:")
                print(f"      Appears {len(indices)} times")
                print(f"      Maps to stimulus frames: {unique_stims[:10].tolist()}")
                if len(unique_stims) > 10:
                    print(f"      ... and {len(unique_stims) - 10} more")
        
        # Analyze stimulus frame duplicates
        stim_counter = Counter(stimulus_frames)
        stim_duplicates = {k: v for k, v in stim_counter.items() if v > 1}
        
        if stim_duplicates:
            print(f"\n  Stimulus frames with duplicates: {len(stim_duplicates):,}")
            print(f"  Total duplicate entries: {sum(v for v in stim_duplicates.values()) - len(stim_duplicates):,}")
            
            # Sample analysis
            sample_stim_frames = sorted(stim_duplicates.keys())[:5]
            for stim_frame in sample_stim_frames:
                indices = np.where(stimulus_frames == stim_frame)[0]
                cam_frames_for_stim = camera_frames[indices]
                unique_cams = np.unique(cam_frames_for_stim)
                
                print(f"\n    Stimulus frame {stim_frame}:")
                print(f"      Appears {len(indices)} times")
                print(f"      Maps to camera frames: {unique_cams[:10].tolist()}")
                if len(unique_cams) > 10:
                    print(f"      ... and {len(unique_cams) - 10} more")
        
        # Analyze the relationship between duplicates and backwards steps
        print(f"\n🔗 Relationship Analysis:")
        
        # Check if backwards steps occur at duplicate boundaries
        backwards_at_duplicates = 0
        for idx in backwards_indices:
            if camera_frames[idx] in cam_duplicates or camera_frames[idx + 1] in cam_duplicates:
                backwards_at_duplicates += 1
        
        if len(backwards_indices) > 0:
            pct = (backwards_at_duplicates / len(backwards_indices)) * 100
            print(f"  {backwards_at_duplicates}/{len(backwards_indices)} ({pct:.1f}%) backwards steps involve duplicate frames")
        
        # Analyze continuous sequences
        print(f"\n📈 Sequence Analysis:")
        
        # Find continuous increasing sequences
        diff = np.diff(camera_frames)
        sequence_starts = [0]  # First element starts a sequence
        sequence_starts.extend(np.where(diff <= 0)[0] + 1)
        sequence_ends = list(np.where(diff <= 0)[0])
        sequence_ends.append(len(camera_frames) - 1)
        
        sequences = []
        for start, end in zip(sequence_starts, sequence_ends):
            if end >= start:
                sequences.append((start, end, end - start + 1))
        
        sequences.sort(key=lambda x: x[2], reverse=True)
        
        print(f"  Found {len(sequences)} continuous sequences")
        print(f"\n  Longest sequences:")
        for i, (start, end, length) in enumerate(sequences[:5]):
            print(f"    {i+1}. Length {length:5d}: indices [{start:5d}:{end:5d}], "
                  f"camera frames [{camera_frames[start]:5d} to {camera_frames[end]:5d}]")
        
        # Check if there's a pattern to where breaks occur
        if len(sequences) > 1:
            print(f"\n  Sequence break analysis:")
            for i in range(min(3, len(sequences) - 1)):
                end_idx = sequences[i][1]
                next_start_idx = sequences[i + 1][0]
                
                print(f"\n    Break {i+1}:")
                print(f"      Sequence {i+1} ends at index {end_idx}, camera frame {camera_frames[end_idx]}")
                print(f"      Sequence {i+2} starts at index {next_start_idx}, camera frame {camera_frames[next_start_idx]}")
                
                if next_start_idx > 0:
                    jump = camera_frames[next_start_idx] - camera_frames[next_start_idx - 1]
                    print(f"      Jump: {jump}")
        
        return {
            'total_records': len(metadata),
            'backwards_steps': len(backwards_indices),
            'camera_duplicates': len(cam_duplicates),
            'stimulus_duplicates': len(stim_duplicates),
            'sequences': len(sequences)
        }


def visualize_frame_patterns(h5_path: str):
    """
    Create visualizations to understand frame ID patterns.
    """
    with h5py.File(h5_path, 'r') as f:
        metadata = f['/video_metadata/frame_metadata'][:]
        
        camera_frames = metadata['triggering_camera_frame_id']
        stimulus_frames = metadata['stimulus_frame_num']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Frame ID Pattern Analysis\n{Path(h5_path).name}', fontsize=14)
        
        # Plot 1: Camera frames over index (zoomed to see patterns)
        ax = axes[0, 0]
        sample_size = min(1000, len(camera_frames))
        ax.plot(camera_frames[:sample_size], 'b-', linewidth=0.5)
        ax.set_xlabel('Index')
        ax.set_ylabel('Camera Frame ID')
        ax.set_title(f'Camera Frame IDs (first {sample_size} records)')
        ax.grid(True, alpha=0.3)
        
        # Mark backwards steps
        backwards = np.where(np.diff(camera_frames[:sample_size]) < 0)[0]
        if len(backwards) > 0:
            ax.plot(backwards, camera_frames[backwards], 'ro', markersize=4, label='Backwards steps')
            ax.legend()
        
        # Plot 2: Frame ID differences
        ax = axes[0, 1]
        diffs = np.diff(camera_frames)
        # Clip extreme values for visualization
        diffs_clipped = np.clip(diffs, -10, 10)
        ax.hist(diffs_clipped, bins=50, edgecolor='black')
        ax.set_xlabel('Frame ID Difference')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Camera Frame ID Differences\n(clipped to [-10, 10])')
        ax.axvline(x=0, color='r', linestyle='--', label='No change')
        ax.axvline(x=1, color='g', linestyle='--', label='Normal increment')
        ax.legend()
        
        # Plot 3: Stimulus vs Camera frame relationship
        ax = axes[1, 0]
        # Sample for visibility
        sample_indices = np.linspace(0, len(camera_frames)-1, min(5000, len(camera_frames)), dtype=int)
        ax.scatter(stimulus_frames[sample_indices], camera_frames[sample_indices], 
                  s=1, alpha=0.5)
        ax.set_xlabel('Stimulus Frame')
        ax.set_ylabel('Camera Frame ID')
        ax.set_title('Stimulus vs Camera Frame Mapping (sampled)')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Duplicate frequency over time
        ax = axes[1, 1]
        window_size = 100
        duplicate_density = []
        
        for i in range(0, len(camera_frames) - window_size, window_size):
            window = camera_frames[i:i+window_size]
            unique_count = len(np.unique(window))
            duplicate_ratio = (window_size - unique_count) / window_size
            duplicate_density.append(duplicate_ratio)
        
        ax.plot(duplicate_density)
        ax.set_xlabel(f'Window Index (window size = {window_size})')
        ax.set_ylabel('Duplicate Ratio')
        ax.set_title('Duplicate Density Over Time')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze frame ID issues in detail"
    )
    parser.add_argument(
        'h5_file',
        type=str,
        help='Path to the analysis.h5 file'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Show visualization plots'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        results = analyze_frame_issues(args.h5_file, verbose=not args.quiet)
        
        if args.plot:
            visualize_frame_patterns(args.h5_file)
        
        # Summary
        print(f"\n" + "=" * 70)
        print(f"SUMMARY:")
        print(f"  Total records: {results['total_records']:,}")
        print(f"  Backwards steps: {results['backwards_steps']}")
        print(f"  Unique camera frames with duplicates: {results['camera_duplicates']:,}")
        print(f"  Unique stimulus frames with duplicates: {results['stimulus_duplicates']:,}")
        print(f"  Continuous sequences: {results['sequences']}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())