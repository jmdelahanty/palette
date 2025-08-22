#!/usr/bin/env python3
"""
H5 Camera Frame ID Mapping Checker
Checks the mapping between triggering_camera_frame_id and stimulus_frame_num.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


class CameraFrameMapper:
    """Check camera frame ID mappings."""
    
    def __init__(self, h5_path: str):
        """Initialize mapper with H5 file path."""
        self.h5_path = Path(h5_path)
        self.h5_file = None
        
    def __enter__(self):
        """Context manager entry."""
        self.h5_file = h5py.File(self.h5_path, 'r')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.h5_file:
            self.h5_file.close()
    
    def analyze_camera_frame_mapping(self):
        """Analyze the triggering_camera_frame_id mapping."""
        if '/video_metadata/frame_metadata' not in self.h5_file:
            print("No frame_metadata found")
            return
            
        metadata = self.h5_file['/video_metadata/frame_metadata'][:]
        df = pd.DataFrame(metadata)
        
        print(f"\n{'='*80}")
        print("CAMERA FRAME ID MAPPING ANALYSIS")
        print(f"{'='*80}")
        
        # Basic statistics
        print(f"\nTotal metadata entries: {len(df)}")
        print(f"Camera frame ID range: {df['triggering_camera_frame_id'].min()} - {df['triggering_camera_frame_id'].max()}")
        print(f"Stimulus frame range: {df['stimulus_frame_num'].min()} - {df['stimulus_frame_num'].max()}")
        
        # Check for duplicates
        dup_camera = df['triggering_camera_frame_id'].duplicated().sum()
        dup_stimulus = df['stimulus_frame_num'].duplicated().sum()
        print(f"\nDuplicate camera frame IDs: {dup_camera}")
        print(f"Duplicate stimulus frame nums: {dup_stimulus}")
        
        # Check for gaps in camera frame IDs
        camera_frames = np.sort(df['triggering_camera_frame_id'].values)
        expected_range = np.arange(camera_frames.min(), camera_frames.max() + 1)
        missing_camera_frames = set(expected_range) - set(camera_frames)
        
        print(f"\nExpected camera frames: {len(expected_range)}")
        print(f"Actual camera frames: {len(camera_frames)}")
        print(f"Missing camera frames: {len(missing_camera_frames)}")
        
        if missing_camera_frames:
            missing_list = sorted(list(missing_camera_frames))
            print(f"  First 100 missing: {missing_list[:100]}")
            
            # Analyze gap patterns
            gaps = []
            if len(camera_frames) > 1:
                diffs = np.diff(camera_frames)
                gap_indices = np.where(diffs > 1)[0]
                
                for idx in gap_indices:
                    gap_start = camera_frames[idx]
                    gap_end = camera_frames[idx + 1]
                    gap_size = gap_end - gap_start - 1
                    gaps.append({
                        'start': int(gap_start),
                        'end': int(gap_end),
                        'size': int(gap_size)
                    })
            
            if gaps:
                print(f"\n  Gap analysis:")
                print(f"    Number of gaps: {len(gaps)}")
                gap_sizes = [g['size'] for g in gaps]
                print(f"    Gap sizes: min={min(gap_sizes)}, max={max(gap_sizes)}, mean={np.mean(gap_sizes):.1f}")
                print(f"\n    First 10 gaps:")
                for i, gap in enumerate(gaps[:10]):
                    print(f"      Gap {i+1}: {gap['start']} -> {gap['end']} (size: {gap['size']})")
        
        # Check mapping consistency
        print(f"\n{'='*40}")
        print("MAPPING CONSISTENCY:")
        
        # Check if mapping is 1:1 or 2:1 (camera frames to stimulus frames)
        camera_to_stimulus = df.groupby('triggering_camera_frame_id')['stimulus_frame_num'].nunique()
        stimulus_to_camera = df.groupby('stimulus_frame_num')['triggering_camera_frame_id'].nunique()
        
        print(f"Camera frames mapping to multiple stimulus frames: {(camera_to_stimulus > 1).sum()}")
        print(f"Stimulus frames with multiple camera frames: {(stimulus_to_camera > 1).sum()}")
        
        # Check the ratio
        if len(df) > 0:
            ratio = len(df['stimulus_frame_num'].unique()) / len(df['triggering_camera_frame_id'].unique())
            print(f"Ratio of stimulus frames to camera frames: {ratio:.2f}")
            
            # Common ratios suggest:
            # 2.0 = 120Hz stimulus, 60Hz camera
            # 1.0 = same frame rate
            # 0.5 = 60Hz stimulus, 120Hz camera
            if abs(ratio - 2.0) < 0.1:
                print("  -> Likely 120Hz stimulus with 60Hz camera recording")
            elif abs(ratio - 1.0) < 0.1:
                print("  -> Likely same frame rate for stimulus and camera")
            elif abs(ratio - 0.5) < 0.1:
                print("  -> Likely 60Hz stimulus with 120Hz camera recording")
    
    def check_specific_frames(self, frame_list: list):
        """Check if specific camera frames have metadata."""
        if '/video_metadata/frame_metadata' not in self.h5_file:
            print("No frame_metadata found")
            return
            
        metadata = self.h5_file['/video_metadata/frame_metadata'][:]
        df = pd.DataFrame(metadata)
        
        print(f"\n{'='*40}")
        print("SPECIFIC FRAME CHECK:")
        
        camera_frame_set = set(df['triggering_camera_frame_id'].values)
        
        for frame in frame_list:
            if frame in camera_frame_set:
                row = df[df['triggering_camera_frame_id'] == frame].iloc[0]
                print(f"  Frame {frame}: ✓ Maps to stimulus frame {row['stimulus_frame_num']}")
            else:
                print(f"  Frame {frame}: ✗ NO METADATA")
    
    def plot_mapping(self, output_path: str = None):
        """Plot the camera frame to stimulus frame mapping."""
        if '/video_metadata/frame_metadata' not in self.h5_file:
            print("No frame_metadata found")
            return
            
        metadata = self.h5_file['/video_metadata/frame_metadata'][:]
        df = pd.DataFrame(metadata)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Camera frame IDs over index
        ax = axes[0, 0]
        ax.plot(df.index, df['triggering_camera_frame_id'], 'b-', linewidth=0.5)
        ax.set_xlabel('Index')
        ax.set_ylabel('Camera Frame ID')
        ax.set_title('Camera Frame IDs (should be continuous if no gaps)')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Stimulus frame nums over index
        ax = axes[0, 1]
        ax.plot(df.index, df['stimulus_frame_num'], 'r-', linewidth=0.5)
        ax.set_xlabel('Index')
        ax.set_ylabel('Stimulus Frame Num')
        ax.set_title('Stimulus Frame Numbers')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mapping relationship
        ax = axes[1, 0]
        ax.scatter(df['triggering_camera_frame_id'], df['stimulus_frame_num'], 
                  s=1, alpha=0.5, c='green')
        ax.set_xlabel('Camera Frame ID')
        ax.set_ylabel('Stimulus Frame Num')
        ax.set_title('Camera to Stimulus Mapping')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Gaps visualization
        ax = axes[1, 1]
        camera_frames = np.sort(df['triggering_camera_frame_id'].values)
        if len(camera_frames) > 1:
            diffs = np.diff(camera_frames)
            gap_positions = np.where(diffs > 1)[0]
            
            # Plot frame differences
            ax.plot(camera_frames[:-1], diffs, 'b-', linewidth=0.5, alpha=0.5)
            
            # Highlight gaps
            if len(gap_positions) > 0:
                ax.scatter(camera_frames[gap_positions], diffs[gap_positions], 
                          c='red', s=20, zorder=5, label=f'{len(gap_positions)} gaps')
            
            ax.set_xlabel('Camera Frame ID')
            ax.set_ylabel('Frame ID Difference')
            ax.set_title('Frame ID Gaps (should be 1 everywhere if continuous)')
            ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Expected (1)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {output_path}")
        else:
            plt.show()
    
    def find_blinking_pattern(self):
        """Try to identify patterns in missing frames that could cause blinking."""
        if '/video_metadata/frame_metadata' not in self.h5_file:
            print("No frame_metadata found")
            return
            
        metadata = self.h5_file['/video_metadata/frame_metadata'][:]
        df = pd.DataFrame(metadata)
        
        print(f"\n{'='*40}")
        print("BLINKING PATTERN ANALYSIS:")
        
        camera_frames = np.sort(df['triggering_camera_frame_id'].values)
        min_frame = camera_frames.min()
        max_frame = camera_frames.max()
        
        # Create a boolean mask for which frames exist
        frame_exists = np.zeros(max_frame - min_frame + 1, dtype=bool)
        frame_exists[camera_frames - min_frame] = True
        
        # Look for periodic patterns
        missing_frames = np.where(~frame_exists)[0] + min_frame
        
        if len(missing_frames) > 1:
            # Check if missing frames follow a pattern
            missing_diffs = np.diff(missing_frames)
            
            print(f"\nMissing frame analysis:")
            print(f"  Total missing: {len(missing_frames)}")
            print(f"  Missing frame spacing:")
            unique_diffs, counts = np.unique(missing_diffs, return_counts=True)
            for diff, count in zip(unique_diffs[:10], counts[:10]):  # Show top 10
                print(f"    Spacing of {diff}: occurs {count} times")
            
            # Check for regular intervals
            if len(unique_diffs) == 1:
                print(f"\n  ⚠️  Missing frames occur at REGULAR intervals of {unique_diffs[0]}")
                print(f"  This would cause periodic blinking!")
            elif len(unique_diffs) < 5:
                print(f"\n  ⚠️  Missing frames follow a simple pattern with {len(unique_diffs)} different spacings")
                print(f"  This could cause semi-regular blinking")
            
            # Show example missing frames
            print(f"\n  First 20 missing camera frame IDs:")
            print(f"  {missing_frames[:20].tolist()}")


def main():
    parser = argparse.ArgumentParser(description='Check camera frame ID mappings in H5 files')
    parser.add_argument('h5_file', help='Path to H5 file')
    parser.add_argument('--check-frames', nargs='+', type=int,
                       help='Check specific camera frame IDs')
    parser.add_argument('--plot', action='store_true',
                       help='Generate mapping visualization plots')
    parser.add_argument('--plot-output', help='Save plot to file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.h5_file).exists():
        print(f"Error: File {args.h5_file} not found")
        return 1
    
    # Run analysis
    with CameraFrameMapper(args.h5_file) as mapper:
        mapper.analyze_camera_frame_mapping()
        mapper.find_blinking_pattern()
        
        if args.check_frames:
            mapper.check_specific_frames(args.check_frames)
        
        if args.plot:
            mapper.plot_mapping(args.plot_output)
    
    return 0


if __name__ == '__main__':
    exit(main())