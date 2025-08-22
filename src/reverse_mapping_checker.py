#!/usr/bin/env python3
"""
H5 Reverse Mapping Checker
Investigates stimulus frames that map to multiple camera frames.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import Counter


class ReverseMappingChecker:
    """Check stimulus frame to camera frame reverse mapping."""
    
    def __init__(self, h5_path: str):
        """Initialize checker with H5 file path."""
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
    
    def analyze_reverse_mapping(self):
        """Analyze stimulus frames that map to multiple camera frames."""
        if '/video_metadata/frame_metadata' not in self.h5_file:
            print("No frame_metadata found")
            return None
            
        metadata = self.h5_file['/video_metadata/frame_metadata'][:]
        df = pd.DataFrame(metadata)
        
        print(f"\n{'='*80}")
        print("STIMULUS → CAMERA REVERSE MAPPING ANALYSIS")
        print(f"{'='*80}")
        
        # Basic info
        print(f"\nBasic Statistics:")
        print(f"  Total metadata entries: {len(df)}")
        print(f"  Unique camera frame IDs: {df['triggering_camera_frame_id'].nunique()}")
        print(f"  Unique stimulus frames: {df['stimulus_frame_num'].nunique()}")
        print(f"  Difference: {len(df) - df['stimulus_frame_num'].nunique()} extra entries")
        
        # Group by stimulus frame
        stim_grouped = df.groupby('stimulus_frame_num')
        
        # Find stimulus frames with multiple camera frames
        multi_camera = stim_grouped.filter(lambda x: len(x) > 1)
        single_camera = stim_grouped.filter(lambda x: len(x) == 1)
        
        print(f"\n{'='*40}")
        print("STIMULUS FRAME MAPPING:")
        print(f"  Stimulus frames mapping to 1 camera frame: {stim_grouped.ngroups - multi_camera['stimulus_frame_num'].nunique()}")
        print(f"  Stimulus frames mapping to >1 camera frame: {multi_camera['stimulus_frame_num'].nunique()}")
        
        # Distribution of mappings
        mapping_counts = stim_grouped.size()
        distribution = Counter(mapping_counts.values)
        
        print(f"\nMapping distribution:")
        for count, freq in sorted(distribution.items()):
            print(f"  {freq} stimulus frames → {count} camera frame(s)")
        
        # Check for patterns in multi-mapped stimulus frames
        if len(multi_camera) > 0:
            print(f"\n{'='*40}")
            print("MULTI-MAPPED STIMULUS FRAMES:")
            
            # Sample some
            sample_stim_frames = multi_camera['stimulus_frame_num'].unique()[:10]
            for stim_frame in sample_stim_frames:
                group = df[df['stimulus_frame_num'] == stim_frame]
                cam_frames = group['triggering_camera_frame_id'].values
                print(f"\n  Stimulus frame {stim_frame}:")
                print(f"    Maps from camera frames: {cam_frames}")
                print(f"    Camera frame spacing: {np.diff(sorted(cam_frames)) if len(cam_frames) > 1 else 'N/A'}")
        
        # Check the order of entries
        print(f"\n{'='*40}")
        print("ENTRY ORDER ANALYSIS:")
        
        # Are entries sorted by camera frame or stimulus frame?
        cam_sorted = df['triggering_camera_frame_id'].is_monotonic_increasing
        stim_sorted = df['stimulus_frame_num'].is_monotonic_increasing
        
        print(f"  Entries sorted by camera frame ID: {cam_sorted}")
        print(f"  Entries sorted by stimulus frame: {stim_sorted}")
        
        if not cam_sorted and not stim_sorted:
            print(f"  ⚠️  Entries are not sorted by either field!")
            print(f"     This could affect lookup behavior")
        
        return df
    
    def check_lookup_behavior(self, df: pd.DataFrame):
        """Simulate how different lookup strategies would behave."""
        print(f"\n{'='*40}")
        print("LOOKUP BEHAVIOR SIMULATION:")
        
        # Group by camera frame
        cam_grouped = df.groupby('triggering_camera_frame_id')
        
        # For each camera frame with duplicates, check which stimulus frame comes first
        first_stim_frames = []
        all_stim_frames = []
        
        for cam_frame, group in cam_grouped:
            if len(group) > 1:
                # Get the first stimulus frame (what the C++ code would return)
                first_stim = group.iloc[0]['stimulus_frame_num']
                all_stims = group['stimulus_frame_num'].values
                
                first_stim_frames.append(first_stim)
                all_stim_frames.extend(all_stims)
        
        # Check if we're consistently getting odd or even frames
        first_even = sum(1 for f in first_stim_frames if f % 2 == 0)
        first_odd = len(first_stim_frames) - first_even
        
        all_even = sum(1 for f in all_stim_frames if f % 2 == 0)
        all_odd = len(all_stim_frames) - all_even
        
        print(f"\n  For camera frames with multiple stimulus frames:")
        print(f"    First returned stimulus frames: {len(first_stim_frames)}")
        print(f"      Even: {first_even} ({first_even/len(first_stim_frames)*100:.1f}%)")
        print(f"      Odd: {first_odd} ({first_odd/len(first_stim_frames)*100:.1f}%)")
        
        print(f"\n    All possible stimulus frames: {len(all_stim_frames)}")
        print(f"      Even: {all_even} ({all_even/len(all_stim_frames)*100:.1f}%)")
        print(f"      Odd: {all_odd} ({all_odd/len(all_stim_frames)*100:.1f}%)")
        
        if abs(first_even/len(first_stim_frames) - 0.5) > 0.4:
            print(f"\n  ⚠️  BIASED SELECTION DETECTED!")
            if first_even > first_odd:
                print(f"     The lookup predominantly returns EVEN stimulus frames")
            else:
                print(f"     The lookup predominantly returns ODD stimulus frames")
            print(f"     This could cause systematic rendering issues")
    
    def analyze_index_ordering(self, df: pd.DataFrame):
        """Analyze the index ordering to understand lookup behavior."""
        print(f"\n{'='*40}")
        print("INDEX ORDERING ANALYSIS:")
        
        # Check if indices correlate with camera or stimulus frames
        cam_correlation = np.corrcoef(df.index, df['triggering_camera_frame_id'])[0, 1]
        stim_correlation = np.corrcoef(df.index, df['stimulus_frame_num'])[0, 1]
        
        print(f"  Index correlation with camera frame ID: {cam_correlation:.3f}")
        print(f"  Index correlation with stimulus frame: {stim_correlation:.3f}")
        
        # For duplicated camera frames, check index patterns
        cam_grouped = df.groupby('triggering_camera_frame_id')
        
        consecutive_indices = 0
        non_consecutive = 0
        
        for cam_frame, group in cam_grouped:
            if len(group) > 1:
                indices = group.index.values
                if all(np.diff(indices) == 1):
                    consecutive_indices += 1
                else:
                    non_consecutive += 1
        
        print(f"\n  Camera frames with multiple entries:")
        print(f"    Consecutive indices: {consecutive_indices}")
        print(f"    Non-consecutive indices: {non_consecutive}")
        
        if non_consecutive > 0:
            print(f"\n  ⚠️  Non-consecutive indices detected!")
            print(f"     This means duplicate entries are scattered throughout the file")
            print(f"     Linear search would find different entries than expected")
    
    def plot_mapping_analysis(self, df: pd.DataFrame, output_path: str = None):
        """Visualize the mapping patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Index vs Camera Frame ID
        ax = axes[0, 0]
        ax.scatter(df.index, df['triggering_camera_frame_id'], s=1, alpha=0.5)
        ax.set_xlabel('DataFrame Index')
        ax.set_ylabel('Camera Frame ID')
        ax.set_title('Index vs Camera Frame ID (should be monotonic if well-ordered)')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Index vs Stimulus Frame
        ax = axes[0, 1]
        ax.scatter(df.index, df['stimulus_frame_num'], s=1, alpha=0.5, c='red')
        ax.set_xlabel('DataFrame Index')
        ax.set_ylabel('Stimulus Frame Number')
        ax.set_title('Index vs Stimulus Frame (shows ordering)')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Histogram of camera frames per stimulus frame
        ax = axes[1, 0]
        stim_grouped = df.groupby('stimulus_frame_num').size()
        ax.hist(stim_grouped.values, bins=range(1, max(stim_grouped.values) + 2), 
                edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Camera Frames')
        ax.set_ylabel('Count of Stimulus Frames')
        ax.set_title('How Many Camera Frames per Stimulus Frame')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: First vs second stimulus frame for duplicated camera frames
        ax = axes[1, 1]
        cam_grouped = df.groupby('triggering_camera_frame_id')
        
        first_frames = []
        second_frames = []
        
        for cam_frame, group in cam_grouped:
            if len(group) >= 2:
                sorted_group = group.sort_index()  # Sort by dataframe index
                first_frames.append(sorted_group.iloc[0]['stimulus_frame_num'])
                second_frames.append(sorted_group.iloc[1]['stimulus_frame_num'])
        
        if first_frames:
            ax.scatter(first_frames, second_frames, s=2, alpha=0.5)
            ax.set_xlabel('First Stimulus Frame (returned by lookup)')
            ax.set_ylabel('Second Stimulus Frame (ignored by lookup)')
            ax.set_title('First vs Second Stimulus Frame for Duplicated Camera Frames')
            
            # Add diagonal line
            min_val = min(min(first_frames), min(second_frames))
            max_val = max(max(first_frames), max(second_frames))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.3)
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {output_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Check reverse mapping in H5 files')
    parser.add_argument('h5_file', help='Path to H5 file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate mapping visualization')
    parser.add_argument('--plot-output', help='Save plot to file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.h5_file).exists():
        print(f"Error: File {args.h5_file} not found")
        return 1
    
    # Run analysis
    with ReverseMappingChecker(args.h5_file) as checker:
        df = checker.analyze_reverse_mapping()
        
        if df is not None:
            checker.check_lookup_behavior(df)
            checker.analyze_index_ordering(df)
            
            if args.plot:
                checker.plot_mapping_analysis(df, args.plot_output)
    
    return 0


if __name__ == '__main__':
    exit(main())