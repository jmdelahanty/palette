#!/usr/bin/env python3
"""
H5 Duplicate Frame ID Analyzer
Investigates duplicate triggering_camera_frame_id entries.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import Counter


class DuplicateFrameAnalyzer:
    """Analyze duplicate camera frame IDs in H5 files."""
    
    def __init__(self, h5_path: str):
        """Initialize analyzer with H5 file path."""
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
    
    def analyze_duplicates(self):
        """Analyze duplicate camera frame IDs and their impact."""
        if '/video_metadata/frame_metadata' not in self.h5_file:
            print("No frame_metadata found")
            return None
            
        metadata = self.h5_file['/video_metadata/frame_metadata'][:]
        df = pd.DataFrame(metadata)
        
        print(f"\n{'='*80}")
        print("DUPLICATE CAMERA FRAME ID ANALYSIS")
        print(f"{'='*80}")
        
        # Group by camera frame ID
        grouped = df.groupby('triggering_camera_frame_id')
        
        # Find duplicates
        duplicate_groups = grouped.filter(lambda x: len(x) > 1)
        unique_groups = grouped.filter(lambda x: len(x) == 1)
        
        print(f"\nTotal metadata entries: {len(df)}")
        print(f"Unique camera frame IDs: {grouped.ngroups}")
        print(f"Camera frames with duplicates: {len(duplicate_groups['triggering_camera_frame_id'].unique())}")
        print(f"Camera frames without duplicates: {len(unique_groups['triggering_camera_frame_id'].unique())}")
        
        # Analyze duplicate patterns
        dup_counts = grouped.size()
        dup_distribution = Counter(dup_counts.values)
        
        print(f"\nDuplicate distribution:")
        for count, freq in sorted(dup_distribution.items()):
            if count == 1:
                print(f"  {count} entry per camera frame: {freq} frames")
            else:
                print(f"  {count} entries per camera frame: {freq} frames")
        
        # Check the pattern of stimulus frames for duplicated camera frames
        print(f"\n{'='*40}")
        print("STIMULUS FRAME PATTERNS FOR DUPLICATES:")
        
        # Sample some duplicate groups
        sample_size = min(10, len(duplicate_groups['triggering_camera_frame_id'].unique()))
        sample_camera_frames = duplicate_groups['triggering_camera_frame_id'].unique()[:sample_size]
        
        for cam_frame in sample_camera_frames:
            group = df[df['triggering_camera_frame_id'] == cam_frame]
            stim_frames = group['stimulus_frame_num'].values
            print(f"\n  Camera frame {cam_frame}:")
            print(f"    Maps to stimulus frames: {stim_frames}")
            print(f"    Spacing: {np.diff(stim_frames) if len(stim_frames) > 1 else 'N/A'}")
        
        # Check if there's a consistent pattern
        if len(duplicate_groups) > 0:
            print(f"\n{'='*40}")
            print("CHECKING FOR CONSISTENT PATTERN:")
            
            all_spacings = []
            for cam_frame in duplicate_groups['triggering_camera_frame_id'].unique()[:100]:  # Check first 100
                group = df[df['triggering_camera_frame_id'] == cam_frame]
                stim_frames = group['stimulus_frame_num'].values
                if len(stim_frames) > 1:
                    spacings = np.diff(sorted(stim_frames))
                    all_spacings.extend(spacings)
            
            if all_spacings:
                unique_spacings = np.unique(all_spacings)
                print(f"  Unique spacings between stimulus frames: {unique_spacings}")
                
                if len(unique_spacings) == 1:
                    print(f"  ✓ Consistent spacing of {unique_spacings[0]} between stimulus frames")
                    print(f"  This suggests a regular 2:1 frame rate ratio")
                else:
                    print(f"  ⚠️  Inconsistent spacing patterns detected")
        
        return df
    
    def check_chaser_states_alignment(self, df: pd.DataFrame):
        """Check how duplicate frames align with chaser states."""
        if '/tracking_data/chaser_states' not in self.h5_file:
            print("\nNo chaser_states data to check")
            return
            
        states = self.h5_file['/tracking_data/chaser_states'][:]
        chaser_df = pd.DataFrame(states)
        
        print(f"\n{'='*40}")
        print("CHASER STATE ALIGNMENT CHECK:")
        
        # For each duplicated camera frame, check if all stimulus frames have chaser data
        duplicate_groups = df.groupby('triggering_camera_frame_id').filter(lambda x: len(x) > 1)
        
        missing_chaser_count = 0
        partial_chaser_count = 0
        full_chaser_count = 0
        
        for cam_frame in duplicate_groups['triggering_camera_frame_id'].unique()[:100]:  # Check first 100
            group = df[df['triggering_camera_frame_id'] == cam_frame]
            stim_frames = group['stimulus_frame_num'].values
            
            # Check which stimulus frames have chaser data
            has_chaser = [sf in chaser_df['stimulus_frame_num'].values for sf in stim_frames]
            
            if all(has_chaser):
                full_chaser_count += 1
            elif any(has_chaser):
                partial_chaser_count += 1
            else:
                missing_chaser_count += 1
        
        print(f"  Checked first 100 duplicate camera frames:")
        print(f"    All stimulus frames have chaser data: {full_chaser_count}")
        print(f"    Some stimulus frames have chaser data: {partial_chaser_count}")
        print(f"    No stimulus frames have chaser data: {missing_chaser_count}")
        
        if partial_chaser_count > 0:
            print(f"\n  ⚠️  POTENTIAL ISSUE: {partial_chaser_count} camera frames have")
            print(f"     inconsistent chaser data across their stimulus frames!")
            print(f"     This could cause blinking as only one stimulus frame gets rendered")
    
    def simulate_frame_lookup(self, df: pd.DataFrame, test_frames: list = None):
        """Simulate what getFrameMetadataByCameraID would return."""
        print(f"\n{'='*40}")
        print("SIMULATING FRAME LOOKUP BEHAVIOR:")
        
        if test_frames is None:
            # Test a range of frames
            camera_frames = df['triggering_camera_frame_id'].unique()
            test_frames = np.random.choice(camera_frames, min(20, len(camera_frames)), replace=False)
        
        for cam_frame in sorted(test_frames)[:10]:
            matches = df[df['triggering_camera_frame_id'] == cam_frame]
            
            if len(matches) == 0:
                print(f"\n  Camera frame {cam_frame}: NO MATCH (would cause blank)")
            elif len(matches) == 1:
                stim_frame = matches.iloc[0]['stimulus_frame_num']
                print(f"\n  Camera frame {cam_frame}: Single match -> stimulus frame {stim_frame}")
            else:
                print(f"\n  Camera frame {cam_frame}: {len(matches)} MATCHES")
                print(f"    First match -> stimulus frame {matches.iloc[0]['stimulus_frame_num']}")
                print(f"    All matches -> stimulus frames {matches['stimulus_frame_num'].values}")
                print(f"    ⚠️  Only first would be used, others ignored!")
    
    def plot_duplicate_analysis(self, df: pd.DataFrame, output_path: str = None):
        """Visualize duplicate patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Entries per camera frame
        ax = axes[0, 0]
        grouped = df.groupby('triggering_camera_frame_id').size()
        ax.hist(grouped.values, bins=range(1, max(grouped.values) + 2), edgecolor='black')
        ax.set_xlabel('Number of entries per camera frame')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Entries per Camera Frame ID')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Camera frames with duplicates over time
        ax = axes[0, 1]
        dup_frames = grouped[grouped > 1].index
        ax.scatter(dup_frames, [1]*len(dup_frames), s=1, alpha=0.5, c='red', label='Duplicates')
        unique_frames = grouped[grouped == 1].index
        ax.scatter(unique_frames, [0]*len(unique_frames), s=1, alpha=0.5, c='blue', label='Unique')
        ax.set_xlabel('Camera Frame ID')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Unique', 'Duplicate'])
        ax.set_title('Duplicate vs Unique Camera Frame IDs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Stimulus frame mapping for duplicates
        ax = axes[1, 0]
        # Sample some duplicates to show pattern
        sample_dups = df[df['triggering_camera_frame_id'].isin(dup_frames[:100])]
        ax.scatter(sample_dups['triggering_camera_frame_id'], 
                  sample_dups['stimulus_frame_num'], 
                  s=5, alpha=0.5, c='orange')
        ax.set_xlabel('Camera Frame ID')
        ax.set_ylabel('Stimulus Frame Num')
        ax.set_title('Stimulus Frame Mapping for First 100 Duplicate Camera Frames')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Index position of duplicates
        ax = axes[1, 1]
        for cam_frame in dup_frames[:50]:  # Show first 50 duplicate camera frames
            group = df[df['triggering_camera_frame_id'] == cam_frame]
            indices = group.index.values
            ax.scatter([cam_frame]*len(indices), indices, s=2, alpha=0.5)
        ax.set_xlabel('Camera Frame ID')
        ax.set_ylabel('Index in DataFrame')
        ax.set_title('Index Positions of Duplicate Entries (First 50)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {output_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze duplicate camera frame IDs in H5 files')
    parser.add_argument('h5_file', help='Path to H5 file')
    parser.add_argument('--test-frames', nargs='+', type=int,
                       help='Test specific camera frame IDs for lookup behavior')
    parser.add_argument('--plot', action='store_true',
                       help='Generate duplicate analysis plots')
    parser.add_argument('--plot-output', help='Save plot to file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.h5_file).exists():
        print(f"Error: File {args.h5_file} not found")
        return 1
    
    # Run analysis
    with DuplicateFrameAnalyzer(args.h5_file) as analyzer:
        df = analyzer.analyze_duplicates()
        
        if df is not None:
            analyzer.check_chaser_states_alignment(df)
            analyzer.simulate_frame_lookup(df, args.test_frames)
            
            if args.plot:
                analyzer.plot_duplicate_analysis(df, args.plot_output)
    
    return 0


if __name__ == '__main__':
    exit(main())