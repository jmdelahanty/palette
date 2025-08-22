#!/usr/bin/env python3
"""
H5 Stimulus Frame Coverage Checker
Checks if every stimulus frame has associated chaser/target data.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


class StimulusCoverageChecker:
    """Check stimulus frame coverage for chaser/target data."""
    
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
    
    def analyze_stimulus_coverage(self):
        """Check if all stimulus frames have chaser/target data."""
        print(f"\n{'='*80}")
        print("STIMULUS FRAME COVERAGE ANALYSIS")
        print(f"{'='*80}")
        
        # Load frame metadata
        if '/video_metadata/frame_metadata' not in self.h5_file:
            print("No frame_metadata found")
            return
        
        metadata = self.h5_file['/video_metadata/frame_metadata'][:]
        meta_df = pd.DataFrame(metadata)
        
        # Load chaser states
        if '/tracking_data/chaser_states' not in self.h5_file:
            print("No chaser_states found")
            return
            
        states = self.h5_file['/tracking_data/chaser_states'][:]
        chaser_df = pd.DataFrame(states)
        
        # Get unique stimulus frames from both sources
        meta_stimulus_frames = set(meta_df['stimulus_frame_num'].unique())
        chaser_stimulus_frames = set(chaser_df['stimulus_frame_num'].unique())
        
        print(f"\nMetadata:")
        print(f"  Total entries: {len(meta_df)}")
        print(f"  Unique stimulus frames: {len(meta_stimulus_frames)}")
        print(f"  Stimulus frame range: {min(meta_stimulus_frames)} - {max(meta_stimulus_frames)}")
        
        print(f"\nChaser States:")
        print(f"  Total entries: {len(chaser_df)}")
        print(f"  Unique stimulus frames: {len(chaser_stimulus_frames)}")
        print(f"  Stimulus frame range: {min(chaser_stimulus_frames)} - {max(chaser_stimulus_frames)}")
        
        # Find gaps
        frames_without_chaser = meta_stimulus_frames - chaser_stimulus_frames
        frames_without_metadata = chaser_stimulus_frames - meta_stimulus_frames
        frames_with_both = meta_stimulus_frames & chaser_stimulus_frames
        
        print(f"\n{'='*40}")
        print("COVERAGE ANALYSIS:")
        print(f"  Stimulus frames with BOTH metadata and chaser data: {len(frames_with_both)}")
        print(f"  Stimulus frames with metadata but NO chaser data: {len(frames_without_chaser)}")
        print(f"  Stimulus frames with chaser data but NO metadata: {len(frames_without_metadata)}")
        
        if frames_without_chaser:
            print(f"\n  ⚠️  WARNING: {len(frames_without_chaser)} stimulus frames lack chaser data!")
            print(f"     These frames will appear blank when displayed")
            frames_list = sorted(list(frames_without_chaser))[:50]
            print(f"     First 50: {frames_list}")
            
            # Check for patterns
            if len(frames_without_chaser) > 1:
                frames_array = np.array(sorted(list(frames_without_chaser)))
                gaps = np.diff(frames_array)
                unique_gaps, counts = np.unique(gaps, return_counts=True)
                
                print(f"\n  Gap pattern analysis:")
                for gap, count in zip(unique_gaps[:10], counts[:10]):
                    print(f"    Gap of {gap}: occurs {count} times")
                
                # Check if these are even or odd frames
                even_count = sum(1 for f in frames_without_chaser if f % 2 == 0)
                odd_count = len(frames_without_chaser) - even_count
                print(f"\n  Even/Odd distribution:")
                print(f"    Even frames: {even_count}")
                print(f"    Odd frames: {odd_count}")
                
                if even_count == 0:
                    print(f"    → All missing frames are ODD")
                elif odd_count == 0:
                    print(f"    → All missing frames are EVEN")
        
        # Analyze camera frame mapping for problematic stimulus frames
        if frames_without_chaser:
            print(f"\n{'='*40}")
            print("CAMERA FRAME MAPPING FOR PROBLEMATIC STIMULUS FRAMES:")
            
            # Find which camera frames map to stimulus frames without chaser data
            problematic_camera_frames = []
            for stim_frame in list(frames_without_chaser)[:100]:  # Check first 100
                camera_frames = meta_df[meta_df['stimulus_frame_num'] == stim_frame]['triggering_camera_frame_id'].values
                problematic_camera_frames.extend(camera_frames)
            
            print(f"  Camera frames that map to stimulus frames without chaser data: {len(set(problematic_camera_frames))}")
            print(f"  First 20 problematic camera frames: {sorted(set(problematic_camera_frames))[:20]}")
        
        return meta_df, chaser_df, frames_without_chaser
    
    def check_alternating_pattern(self, meta_df, chaser_df):
        """Check if chaser data alternates with stimulus frames (120Hz/60Hz issue)."""
        print(f"\n{'='*40}")
        print("CHECKING FOR ALTERNATING PATTERN:")
        
        # For camera frames with duplicates, check which stimulus frames have chaser data
        grouped = meta_df.groupby('triggering_camera_frame_id')
        
        alternating_count = 0
        both_have_chaser = 0
        neither_have_chaser = 0
        
        chaser_stimulus_set = set(chaser_df['stimulus_frame_num'].values)
        
        for cam_frame, group in grouped:
            if len(group) == 2:  # Only check frames with exactly 2 stimulus frames
                stim_frames = group['stimulus_frame_num'].values
                has_chaser = [sf in chaser_stimulus_set for sf in stim_frames]
                
                if has_chaser == [True, False] or has_chaser == [False, True]:
                    alternating_count += 1
                elif all(has_chaser):
                    both_have_chaser += 1
                elif not any(has_chaser):
                    neither_have_chaser += 1
        
        total_pairs = alternating_count + both_have_chaser + neither_have_chaser
        
        print(f"  Camera frames with 2 stimulus frames: {total_pairs}")
        print(f"    Only ONE has chaser data (alternating): {alternating_count} ({alternating_count/total_pairs*100:.1f}%)")
        print(f"    BOTH have chaser data: {both_have_chaser} ({both_have_chaser/total_pairs*100:.1f}%)")
        print(f"    NEITHER has chaser data: {neither_have_chaser} ({neither_have_chaser/total_pairs*100:.1f}%)")
        
        if alternating_count > total_pairs * 0.8:
            print(f"\n  ⚠️  ALTERNATING PATTERN DETECTED!")
            print(f"     This causes blinking because getFrameMetadataByCameraID")
            print(f"     returns the first stimulus frame, but chaser data might")
            print(f"     only exist for the second stimulus frame!")
    
    def plot_coverage(self, meta_df, chaser_df, frames_without_chaser, output_path=None):
        """Visualize coverage patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Stimulus frames with/without chaser data
        ax = axes[0, 0]
        all_stim_frames = sorted(set(meta_df['stimulus_frame_num'].unique()))
        chaser_stim_frames = set(chaser_df['stimulus_frame_num'].unique())
        
        has_chaser = [1 if sf in chaser_stim_frames else 0 for sf in all_stim_frames]
        
        # Create blocks of continuous coverage
        blocks = []
        current_block = {'start': 0, 'has_chaser': has_chaser[0]}
        
        for i in range(1, len(has_chaser)):
            if has_chaser[i] != current_block['has_chaser']:
                current_block['end'] = i - 1
                blocks.append(current_block)
                current_block = {'start': i, 'has_chaser': has_chaser[i]}
        current_block['end'] = len(has_chaser) - 1
        blocks.append(current_block)
        
        for block in blocks:
            color = 'green' if block['has_chaser'] else 'red'
            ax.axvspan(all_stim_frames[block['start']], 
                      all_stim_frames[block['end']], 
                      alpha=0.3, color=color)
        
        ax.set_xlabel('Stimulus Frame Number')
        ax.set_ylabel('Has Chaser Data')
        ax.set_title('Stimulus Frames With/Without Chaser Data')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Camera frames and their stimulus frame coverage
        ax = axes[0, 1]
        grouped = meta_df.groupby('triggering_camera_frame_id')
        chaser_stimulus_set = set(chaser_df['stimulus_frame_num'].values)
        
        camera_frames = []
        coverage_status = []  # 0=none, 1=partial, 2=full
        
        for cam_frame, group in grouped:
            stim_frames = group['stimulus_frame_num'].values
            has_chaser = [sf in chaser_stimulus_set for sf in stim_frames]
            
            camera_frames.append(cam_frame)
            if all(has_chaser):
                coverage_status.append(2)
            elif any(has_chaser):
                coverage_status.append(1)
            else:
                coverage_status.append(0)
        
        colors = ['red', 'orange', 'green']
        for status in [0, 1, 2]:
            mask = np.array(coverage_status) == status
            label = ['No chaser', 'Partial chaser', 'Full chaser'][status]
            ax.scatter(np.array(camera_frames)[mask], 
                      np.ones(sum(mask)) * status, 
                      c=colors[status], s=1, label=label, alpha=0.5)
        
        ax.set_xlabel('Camera Frame ID')
        ax.set_ylabel('Chaser Data Coverage')
        ax.set_title('Camera Frame Chaser Data Coverage')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['None', 'Partial', 'Full'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Distribution of missing frames
        ax = axes[1, 0]
        if frames_without_chaser:
            missing_array = np.array(sorted(list(frames_without_chaser)))
            ax.hist(missing_array % 2, bins=[0, 0.5, 1, 1.5, 2], 
                   edgecolor='black', alpha=0.7)
            ax.set_xlabel('Frame Number Modulo 2')
            ax.set_ylabel('Count')
            ax.set_title('Even/Odd Distribution of Missing Chaser Data')
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels(['Even', 'Odd'])
        else:
            ax.text(0.5, 0.5, 'No missing frames', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('No Missing Chaser Data')
        
        # Plot 4: Gap analysis
        ax = axes[1, 1]
        if frames_without_chaser and len(frames_without_chaser) > 1:
            missing_array = np.array(sorted(list(frames_without_chaser)))
            gaps = np.diff(missing_array)
            unique_gaps, counts = np.unique(gaps, return_counts=True)
            
            ax.bar(unique_gaps[:20], counts[:20], edgecolor='black')
            ax.set_xlabel('Gap Size (frames)')
            ax.set_ylabel('Frequency')
            ax.set_title('Gap Sizes Between Missing Chaser Data')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Not enough missing frames for gap analysis', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {output_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Check stimulus frame coverage in H5 files')
    parser.add_argument('h5_file', help='Path to H5 file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate coverage visualization')
    parser.add_argument('--plot-output', help='Save plot to file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.h5_file).exists():
        print(f"Error: File {args.h5_file} not found")
        return 1
    
    # Run analysis
    with StimulusCoverageChecker(args.h5_file) as checker:
        result = checker.analyze_stimulus_coverage()
        
        if result:
            meta_df, chaser_df, frames_without_chaser = result
            checker.check_alternating_pattern(meta_df, chaser_df)
            
            if args.plot:
                checker.plot_coverage(meta_df, chaser_df, frames_without_chaser, args.plot_output)
    
    return 0


if __name__ == '__main__':
    exit(main())