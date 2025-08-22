#!/usr/bin/env python3
"""
H5 Position Frame-by-Frame Debugger
Investigates position values frame-by-frame to identify rendering issues.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Tuple, Optional


class PositionDebugger:
    """Debug position values and potential rendering issues."""
    
    def __init__(self, h5_path: str):
        """Initialize debugger with H5 file path."""
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
    
    def analyze_position_patterns(self, start_frame: int = 0, num_frames: int = 100):
        """Analyze position patterns for a range of frames."""
        if '/tracking_data/chaser_states' not in self.h5_file:
            print("No chaser_states found")
            return
            
        states = self.h5_file['/tracking_data/chaser_states'][:]
        df = pd.DataFrame(states)
        
        # Filter to requested range
        df = df[(df['stimulus_frame_num'] >= start_frame) & 
                (df['stimulus_frame_num'] < start_frame + num_frames)]
        
        if df.empty:
            print(f"No data found for frames {start_frame} to {start_frame + num_frames}")
            return
            
        print(f"\n{'='*80}")
        print(f"FRAME-BY-FRAME POSITION ANALYSIS (Frames {start_frame}-{start_frame + num_frames})")
        print(f"{'='*80}")
        
        # Analyze each frame
        issues_found = []
        
        for idx, row in df.iterrows():
            frame = row['stimulus_frame_num']
            
            # Check for various issues
            issues = []
            
            # Check if positions are at origin (0,0)
            if row['chaser_pos_x'] == 0 and row['chaser_pos_y'] == 0:
                issues.append("CHASER_AT_ORIGIN")
            if row['target_pos_x'] == 0 and row['target_pos_y'] == 0:
                issues.append("TARGET_AT_ORIGIN")
                
            # Check if positions are identical
            if (row['chaser_pos_x'] == row['target_pos_x'] and 
                row['chaser_pos_y'] == row['target_pos_y']):
                issues.append("IDENTICAL_POSITIONS")
                
            # Check for NaN or inf
            if np.isnan(row['chaser_pos_x']) or np.isnan(row['chaser_pos_y']):
                issues.append("CHASER_NAN")
            if np.isnan(row['target_pos_x']) or np.isnan(row['target_pos_y']):
                issues.append("TARGET_NAN")
            if np.isinf(row['chaser_pos_x']) or np.isinf(row['chaser_pos_y']):
                issues.append("CHASER_INF")
            if np.isinf(row['target_pos_x']) or np.isinf(row['target_pos_y']):
                issues.append("TARGET_INF")
                
            # Check for out-of-bounds (assuming typical arena size)
            ARENA_MIN = -2000
            ARENA_MAX = 2000
            if not (ARENA_MIN <= row['chaser_pos_x'] <= ARENA_MAX and 
                    ARENA_MIN <= row['chaser_pos_y'] <= ARENA_MAX):
                issues.append("CHASER_OUT_OF_BOUNDS")
            if not (ARENA_MIN <= row['target_pos_x'] <= ARENA_MAX and 
                    ARENA_MIN <= row['target_pos_y'] <= ARENA_MAX):
                issues.append("TARGET_OUT_OF_BOUNDS")
                
            # Print frame info if issues found or in verbose mode
            if issues or idx < 10:  # Show first 10 frames regardless
                status = "⚠️ " if issues else "✓"
                print(f"\n{status} Frame {int(frame)}:")
                print(f"  Chaser: ({row['chaser_pos_x']:.2f}, {row['chaser_pos_y']:.2f})")
                print(f"  Target: ({row['target_pos_x']:.2f}, {row['target_pos_y']:.2f})")
                print(f"  Is Chasing: {row['is_chasing']}")
                if issues:
                    print(f"  Issues: {', '.join(issues)}")
                    issues_found.append((int(frame), issues))
        
        # Summary
        print(f"\n{'='*40}")
        print(f"SUMMARY:")
        print(f"  Total frames analyzed: {len(df)}")
        print(f"  Frames with issues: {len(issues_found)}")
        
        if issues_found:
            print(f"\n  Issue breakdown:")
            issue_counts = {}
            for frame, frame_issues in issues_found:
                for issue in frame_issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
            for issue, count in sorted(issue_counts.items()):
                print(f"    {issue}: {count} frames")
    
    def find_zero_positions(self, max_results: int = 100):
        """Find all frames where positions are at (0,0)."""
        if '/tracking_data/chaser_states' not in self.h5_file:
            print("No chaser_states found")
            return
            
        states = self.h5_file['/tracking_data/chaser_states'][:]
        df = pd.DataFrame(states)
        
        # Find frames with zero positions
        chaser_zeros = df[(df['chaser_pos_x'] == 0) & (df['chaser_pos_y'] == 0)]
        target_zeros = df[(df['target_pos_x'] == 0) & (df['target_pos_y'] == 0)]
        
        print(f"\n{'='*60}")
        print("ZERO POSITION ANALYSIS")
        print(f"{'='*60}")
        
        print(f"\nChaser at (0,0): {len(chaser_zeros)} frames")
        if len(chaser_zeros) > 0:
            print(f"  First {min(max_results, len(chaser_zeros))} frames:")
            for idx, row in chaser_zeros.head(max_results).iterrows():
                print(f"    Frame {int(row['stimulus_frame_num'])}")
                
        print(f"\nTarget at (0,0): {len(target_zeros)} frames")
        if len(target_zeros) > 0:
            print(f"  First {min(max_results, len(target_zeros))} frames:")
            for idx, row in target_zeros.head(max_results).iterrows():
                print(f"    Frame {int(row['stimulus_frame_num'])}")
    
    def analyze_position_distribution(self):
        """Analyze the distribution of positions."""
        if '/tracking_data/chaser_states' not in self.h5_file:
            print("No chaser_states found")
            return
            
        states = self.h5_file['/tracking_data/chaser_states'][:]
        df = pd.DataFrame(states)
        
        print(f"\n{'='*60}")
        print("POSITION DISTRIBUTION ANALYSIS")
        print(f"{'='*60}")
        
        # Statistics for chaser
        print("\nChaser Position Statistics:")
        print(f"  X: min={df['chaser_pos_x'].min():.2f}, "
              f"max={df['chaser_pos_x'].max():.2f}, "
              f"mean={df['chaser_pos_x'].mean():.2f}, "
              f"std={df['chaser_pos_x'].std():.2f}")
        print(f"  Y: min={df['chaser_pos_y'].min():.2f}, "
              f"max={df['chaser_pos_y'].max():.2f}, "
              f"mean={df['chaser_pos_y'].mean():.2f}, "
              f"std={df['chaser_pos_y'].std():.2f}")
        
        # Statistics for target
        print("\nTarget Position Statistics:")
        print(f"  X: min={df['target_pos_x'].min():.2f}, "
              f"max={df['target_pos_x'].max():.2f}, "
              f"mean={df['target_pos_x'].mean():.2f}, "
              f"std={df['target_pos_x'].std():.2f}")
        print(f"  Y: min={df['target_pos_y'].min():.2f}, "
              f"max={df['target_pos_y'].max():.2f}, "
              f"mean={df['target_pos_y'].mean():.2f}, "
              f"std={df['target_pos_y'].std():.2f}")
        
        # Check for static positions (no movement)
        chaser_static = df.groupby(['chaser_pos_x', 'chaser_pos_y']).size()
        target_static = df.groupby(['target_pos_x', 'target_pos_y']).size()
        
        print(f"\nUnique Positions:")
        print(f"  Chaser: {len(chaser_static)} unique positions")
        print(f"  Target: {len(target_static)} unique positions")
        
        # Find most common positions
        if len(chaser_static) > 0:
            top_chaser = chaser_static.nlargest(5)
            print(f"\n  Top 5 most common chaser positions:")
            for (x, y), count in top_chaser.items():
                percentage = (count / len(df)) * 100
                print(f"    ({x:.2f}, {y:.2f}): {count} frames ({percentage:.1f}%)")
                
        if len(target_static) > 0:
            top_target = target_static.nlargest(5)
            print(f"\n  Top 5 most common target positions:")
            for (x, y), count in top_target.items():
                percentage = (count / len(df)) * 100
                print(f"    ({x:.2f}, {y:.2f}): {count} frames ({percentage:.1f}%)")
    
    def check_interpolation_alignment(self):
        """Check if interpolated frames correspond to position issues."""
        if '/analysis/interpolation_mask' not in self.h5_file:
            print("No interpolation mask found")
            return
            
        if '/tracking_data/chaser_states' not in self.h5_file:
            print("No chaser_states found")
            return
            
        mask = self.h5_file['/analysis/interpolation_mask'][:]
        states = self.h5_file['/tracking_data/chaser_states'][:]
        df = pd.DataFrame(states)
        
        print(f"\n{'='*60}")
        print("INTERPOLATION vs POSITION ANALYSIS")
        print(f"{'='*60}")
        
        # Check positions for interpolated frames
        interpolated_frames = []
        original_frames = []
        
        for frame_idx, is_original in enumerate(mask):
            if frame_idx + 1 in df['stimulus_frame_num'].values:
                frame_data = df[df['stimulus_frame_num'] == frame_idx + 1].iloc[0]
                
                # Check if position is at origin
                at_origin = (frame_data['chaser_pos_x'] == 0 and 
                           frame_data['chaser_pos_y'] == 0) or \
                          (frame_data['target_pos_x'] == 0 and 
                           frame_data['target_pos_y'] == 0)
                
                if not is_original:  # Interpolated
                    interpolated_frames.append({
                        'frame': frame_idx + 1,
                        'at_origin': at_origin,
                        'chaser_x': frame_data['chaser_pos_x'],
                        'chaser_y': frame_data['chaser_pos_y']
                    })
                else:  # Original
                    original_frames.append({
                        'frame': frame_idx + 1,
                        'at_origin': at_origin,
                        'chaser_x': frame_data['chaser_pos_x'],
                        'chaser_y': frame_data['chaser_pos_y']
                    })
        
        # Statistics
        interp_at_origin = sum(1 for f in interpolated_frames if f['at_origin'])
        orig_at_origin = sum(1 for f in original_frames if f['at_origin'])
        
        print(f"\nInterpolated frames with position at origin: {interp_at_origin}/{len(interpolated_frames)}")
        print(f"Original frames with position at origin: {orig_at_origin}/{len(original_frames)}")
        
        # Show some examples
        if interp_at_origin > 0:
            print(f"\nFirst 10 interpolated frames at origin:")
            count = 0
            for frame in interpolated_frames:
                if frame['at_origin'] and count < 10:
                    print(f"  Frame {frame['frame']}: "
                          f"Chaser ({frame['chaser_x']:.2f}, {frame['chaser_y']:.2f})")
                    count += 1
    
    def plot_position_timeline(self, output_path: Optional[str] = None):
        """Plot positions over time to visualize patterns."""
        if '/tracking_data/chaser_states' not in self.h5_file:
            print("No chaser_states found")
            return
            
        states = self.h5_file['/tracking_data/chaser_states'][:]
        df = pd.DataFrame(states)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot X positions over time
        ax = axes[0, 0]
        ax.plot(df['stimulus_frame_num'], df['chaser_pos_x'], 'b-', 
                alpha=0.5, linewidth=0.5, label='Chaser')
        ax.plot(df['stimulus_frame_num'], df['target_pos_x'], 'r-', 
                alpha=0.5, linewidth=0.5, label='Target')
        ax.set_xlabel('Frame')
        ax.set_ylabel('X Position')
        ax.set_title('X Position Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Y positions over time
        ax = axes[0, 1]
        ax.plot(df['stimulus_frame_num'], df['chaser_pos_y'], 'b-', 
                alpha=0.5, linewidth=0.5, label='Chaser')
        ax.plot(df['stimulus_frame_num'], df['target_pos_y'], 'r-', 
                alpha=0.5, linewidth=0.5, label='Target')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Y Position')
        ax.set_title('Y Position Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot distance between chaser and target
        ax = axes[1, 0]
        distance = np.sqrt((df['chaser_pos_x'] - df['target_pos_x'])**2 + 
                          (df['chaser_pos_y'] - df['target_pos_y'])**2)
        ax.plot(df['stimulus_frame_num'], distance, 'g-', alpha=0.7, linewidth=0.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Distance')
        ax.set_title('Chaser-Target Distance Over Time')
        ax.grid(True, alpha=0.3)
        
        # Highlight frames at origin
        ax = axes[1, 1]
        at_origin_chaser = (df['chaser_pos_x'] == 0) & (df['chaser_pos_y'] == 0)
        at_origin_target = (df['target_pos_x'] == 0) & (df['target_pos_y'] == 0)
        
        ax.scatter(df.loc[at_origin_chaser, 'stimulus_frame_num'], 
                  np.ones(at_origin_chaser.sum()), 
                  c='blue', s=1, label=f'Chaser at origin ({at_origin_chaser.sum()} frames)')
        ax.scatter(df.loc[at_origin_target, 'stimulus_frame_num'], 
                  np.ones(at_origin_target.sum()) * 2, 
                  c='red', s=1, label=f'Target at origin ({at_origin_target.sum()} frames)')
        
        ax.set_xlabel('Frame')
        ax.set_yticks([1, 2])
        ax.set_yticklabels(['Chaser', 'Target'])
        ax.set_title('Frames with Position at Origin (0,0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Debug position values in H5 files')
    parser.add_argument('h5_file', help='Path to H5 file')
    parser.add_argument('--start-frame', type=int, default=0, 
                       help='Start frame for detailed analysis')
    parser.add_argument('--num-frames', type=int, default=100,
                       help='Number of frames to analyze in detail')
    parser.add_argument('--find-zeros', action='store_true',
                       help='Find all frames with zero positions')
    parser.add_argument('--plot', action='store_true',
                       help='Generate position timeline plots')
    parser.add_argument('--plot-output', help='Save plot to file')
    parser.add_argument('--check-interpolation', action='store_true',
                       help='Check interpolation alignment with positions')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.h5_file).exists():
        print(f"Error: File {args.h5_file} not found")
        return 1
    
    # Run analysis
    with PositionDebugger(args.h5_file) as debugger:
        # Always run basic analyses
        debugger.analyze_position_distribution()
        debugger.analyze_position_patterns(args.start_frame, args.num_frames)
        
        if args.find_zeros:
            debugger.find_zero_positions()
            
        if args.check_interpolation:
            debugger.check_interpolation_alignment()
            
        if args.plot:
            debugger.plot_position_timeline(args.plot_output)
    
    return 0


if __name__ == '__main__':
    exit(main())