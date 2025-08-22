#!/usr/bin/env python3
"""
H5 Chaser/Target Position Gap Analyzer
Investigates chaser/target positions and frame numbers to identify gaps and discontinuities.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import argparse


class ChaserTargetAnalyzer:
    """Analyze chaser/target positions and gaps in H5 files."""
    
    def __init__(self, h5_path: str):
        """Initialize analyzer with H5 file path."""
        self.h5_path = Path(h5_path)
        self.h5_file = None
        self.chaser_states = None
        self.frame_metadata = None
        
    def __enter__(self):
        """Context manager entry."""
        self.h5_file = h5py.File(self.h5_path, 'r')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.h5_file:
            self.h5_file.close()
            
    def load_chaser_states(self) -> pd.DataFrame:
        """Load chaser states from H5 file."""
        if '/tracking_data/chaser_states' not in self.h5_file:
            print("Warning: No chaser_states found in H5 file")
            return pd.DataFrame()
            
        states = self.h5_file['/tracking_data/chaser_states'][:]
        
        # Convert to pandas DataFrame for easier analysis
        df = pd.DataFrame({
            'stimulus_frame_num': states['stimulus_frame_num'],
            'timestamp_ns': states['timestamp_ns_session'],
            'chaser_index': states['chaser_index'],
            'is_chasing': states['is_chasing'],
            'chaser_pos_x': states['chaser_pos_x'],
            'chaser_pos_y': states['chaser_pos_y'],
            'target_pos_x': states['target_pos_x'],
            'target_pos_y': states['target_pos_y']
        })
        
        self.chaser_states = df
        return df
    
    def load_frame_metadata(self) -> pd.DataFrame:
        """Load frame metadata from H5 file."""
        if '/video_metadata/frame_metadata' not in self.h5_file:
            print("Warning: No frame_metadata found in H5 file")
            return pd.DataFrame()
            
        metadata = self.h5_file['/video_metadata/frame_metadata'][:]
        
        df = pd.DataFrame({
            'stimulus_frame_num': metadata['stimulus_frame_num'],
            'camera_frame_id': metadata['triggering_camera_frame_id'],
            'timestamp_ns': metadata['timestamp_ns']
        })
        
        self.frame_metadata = df
        return df
    
    def analyze_frame_gaps(self) -> Dict:
        """Analyze gaps in frame numbers."""
        if self.chaser_states is None:
            self.load_chaser_states()
            
        if self.chaser_states.empty:
            return {}
            
        # Get unique frame numbers
        frames = np.sort(self.chaser_states['stimulus_frame_num'].unique())
        
        # Find gaps
        gaps = []
        if len(frames) > 1:
            frame_diffs = np.diff(frames)
            gap_indices = np.where(frame_diffs > 1)[0]
            
            for idx in gap_indices:
                gap_start = frames[idx]
                gap_end = frames[idx + 1]
                gap_size = gap_end - gap_start - 1
                gaps.append({
                    'start_frame': int(gap_start),
                    'end_frame': int(gap_end),
                    'gap_size': int(gap_size),
                    'missing_frames': list(range(int(gap_start + 1), int(gap_end)))
                })
        
        # Calculate statistics
        total_frames = int(frames[-1] - frames[0] + 1) if len(frames) > 0 else 0
        actual_frames = len(frames)
        missing_frames = total_frames - actual_frames
        coverage = (actual_frames / total_frames * 100) if total_frames > 0 else 0
        
        return {
            'frame_range': [int(frames[0]), int(frames[-1])] if len(frames) > 0 else [],
            'total_expected_frames': total_frames,
            'actual_frames': actual_frames,
            'missing_frames': missing_frames,
            'coverage_percent': coverage,
            'num_gaps': len(gaps),
            'gaps': gaps,
            'largest_gap': max([g['gap_size'] for g in gaps]) if gaps else 0
        }
    
    def analyze_position_discontinuities(self, max_jump: float = 100.0) -> Dict:
        """Analyze position discontinuities/jumps."""
        if self.chaser_states is None:
            self.load_chaser_states()
            
        if self.chaser_states.empty:
            return {}
            
        # Sort by frame number
        df = self.chaser_states.sort_values('stimulus_frame_num')
        
        # Calculate position changes
        df['chaser_dx'] = df['chaser_pos_x'].diff()
        df['chaser_dy'] = df['chaser_pos_y'].diff()
        df['chaser_dist'] = np.sqrt(df['chaser_dx']**2 + df['chaser_dy']**2)
        
        df['target_dx'] = df['target_pos_x'].diff()
        df['target_dy'] = df['target_pos_y'].diff()
        df['target_dist'] = np.sqrt(df['target_dx']**2 + df['target_dy']**2)
        
        # Find jumps
        chaser_jumps = df[df['chaser_dist'] > max_jump].copy()
        target_jumps = df[df['target_dist'] > max_jump].copy()
        
        # Check for NaN positions
        nan_positions = df[
            df['chaser_pos_x'].isna() | df['chaser_pos_y'].isna() |
            df['target_pos_x'].isna() | df['target_pos_y'].isna()
        ]
        
        return {
            'chaser_jumps': {
                'count': len(chaser_jumps),
                'frames': chaser_jumps['stimulus_frame_num'].tolist(),
                'distances': chaser_jumps['chaser_dist'].tolist(),
                'max_jump': float(df['chaser_dist'].max()) if not df['chaser_dist'].isna().all() else 0
            },
            'target_jumps': {
                'count': len(target_jumps),
                'frames': target_jumps['stimulus_frame_num'].tolist(),
                'distances': target_jumps['target_dist'].tolist(),
                'max_jump': float(df['target_dist'].max()) if not df['target_dist'].isna().all() else 0
            },
            'nan_positions': {
                'count': len(nan_positions),
                'frames': nan_positions['stimulus_frame_num'].tolist()
            },
            'mean_chaser_movement': float(df['chaser_dist'].mean()) if not df['chaser_dist'].isna().all() else 0,
            'mean_target_movement': float(df['target_dist'].mean()) if not df['target_dist'].isna().all() else 0
        }
    
    def analyze_chasing_states(self) -> Dict:
        """Analyze chasing state transitions."""
        if self.chaser_states is None:
            self.load_chaser_states()
            
        if self.chaser_states.empty:
            return {}
            
        df = self.chaser_states.sort_values('stimulus_frame_num')
        
        # Find state transitions
        df['state_change'] = df['is_chasing'].diff() != 0
        transitions = df[df['state_change']].copy()
        
        # Calculate chasing periods
        chasing_periods = []
        chase_start = None
        
        for _, row in df.iterrows():
            if row['is_chasing'] and chase_start is None:
                chase_start = row['stimulus_frame_num']
            elif not row['is_chasing'] and chase_start is not None:
                chasing_periods.append({
                    'start_frame': int(chase_start),
                    'end_frame': int(row['stimulus_frame_num'] - 1),
                    'duration_frames': int(row['stimulus_frame_num'] - chase_start)
                })
                chase_start = None
        
        # Handle case where chasing continues to end
        if chase_start is not None:
            chasing_periods.append({
                'start_frame': int(chase_start),
                'end_frame': int(df.iloc[-1]['stimulus_frame_num']),
                'duration_frames': int(df.iloc[-1]['stimulus_frame_num'] - chase_start + 1)
            })
        
        total_frames = len(df)
        chasing_frames = df['is_chasing'].sum()
        
        return {
            'total_frames': total_frames,
            'chasing_frames': int(chasing_frames),
            'not_chasing_frames': int(total_frames - chasing_frames),
            'chasing_percentage': (chasing_frames / total_frames * 100) if total_frames > 0 else 0,
            'num_chasing_periods': len(chasing_periods),
            'chasing_periods': chasing_periods,
            'num_transitions': len(transitions),
            'transition_frames': transitions['stimulus_frame_num'].tolist()
        }
    
    def analyze_camera_frame_mapping(self) -> Dict:
        """Analyze mapping between camera frames and stimulus frames."""
        if self.frame_metadata is None:
            self.load_frame_metadata()
        if self.chaser_states is None:
            self.load_chaser_states()
            
        if self.frame_metadata.empty or self.chaser_states.empty:
            return {}
            
        # Get unique stimulus frames from both sources
        metadata_stimulus_frames = set(self.frame_metadata['stimulus_frame_num'].unique())
        chaser_stimulus_frames = set(self.chaser_states['stimulus_frame_num'].unique())
        
        # Find camera frames and their stimulus frame mappings
        camera_frames = self.frame_metadata['camera_frame_id'].values
        camera_frame_range = [int(camera_frames.min()), int(camera_frames.max())]
        expected_camera_frames = set(range(camera_frame_range[0], camera_frame_range[1] + 1))
        actual_camera_frames = set(camera_frames)
        
        # Find missing camera frames
        missing_camera_frames = expected_camera_frames - actual_camera_frames
        
        # Find stimulus frames that have metadata but no chaser states
        stimulus_frames_without_chaser = metadata_stimulus_frames - chaser_stimulus_frames
        
        # Find which camera frames map to stimulus frames without chaser states
        camera_frames_without_chaser = []
        if stimulus_frames_without_chaser:
            for stim_frame in stimulus_frames_without_chaser:
                cam_frames = self.frame_metadata[
                    self.frame_metadata['stimulus_frame_num'] == stim_frame
                ]['camera_frame_id'].tolist()
                camera_frames_without_chaser.extend(cam_frames)
        
        # Analyze gaps in camera frame sequence
        camera_frame_gaps = []
        if len(camera_frames) > 1:
            sorted_cam_frames = np.sort(camera_frames)
            cam_diffs = np.diff(sorted_cam_frames)
            gap_indices = np.where(cam_diffs > 1)[0]
            
            for idx in gap_indices:
                gap_start = sorted_cam_frames[idx]
                gap_end = sorted_cam_frames[idx + 1]
                gap_size = gap_end - gap_start - 1
                camera_frame_gaps.append({
                    'start': int(gap_start),
                    'end': int(gap_end),
                    'size': int(gap_size)
                })
        
        return {
            'camera_frame_range': camera_frame_range,
            'total_camera_frames_expected': len(expected_camera_frames),
            'total_camera_frames_actual': len(actual_camera_frames),
            'missing_camera_frames': len(missing_camera_frames),
            'missing_camera_frame_list': sorted(list(missing_camera_frames))[:100],  # First 100
            'camera_frame_gaps': camera_frame_gaps,
            'num_camera_frame_gaps': len(camera_frame_gaps),
            'stimulus_frames_in_metadata': len(metadata_stimulus_frames),
            'stimulus_frames_in_chaser': len(chaser_stimulus_frames),
            'stimulus_frames_without_chaser': len(stimulus_frames_without_chaser),
            'stimulus_frames_without_chaser_list': sorted(list(stimulus_frames_without_chaser))[:100],
            'camera_frames_without_chaser': len(camera_frames_without_chaser),
            'camera_frames_without_chaser_list': sorted(camera_frames_without_chaser)[:100]
        }
    
    def check_interpolation_mask(self) -> Optional[Dict]:
        """Check for interpolation mask in analysis files."""
        if '/analysis/interpolation_mask' not in self.h5_file:
            return None
            
        mask = self.h5_file['/analysis/interpolation_mask'][:]
        
        # True = original, False = interpolated
        original_count = np.sum(mask)
        interpolated_count = len(mask) - original_count
        
        # Find interpolated segments
        interpolated_segments = []
        in_segment = False
        segment_start = None
        
        for i, is_original in enumerate(mask):
            if not is_original and not in_segment:
                segment_start = i
                in_segment = True
            elif is_original and in_segment:
                interpolated_segments.append({
                    'start_frame': segment_start,
                    'end_frame': i - 1,
                    'length': i - segment_start
                })
                in_segment = False
        
        # Handle segment that extends to end
        if in_segment:
            interpolated_segments.append({
                'start_frame': segment_start,
                'end_frame': len(mask) - 1,
                'length': len(mask) - segment_start
            })
        
        return {
            'total_frames': len(mask),
            'original_frames': int(original_count),
            'interpolated_frames': int(interpolated_count),
            'interpolation_percentage': (interpolated_count / len(mask) * 100) if len(mask) > 0 else 0,
            'num_interpolated_segments': len(interpolated_segments),
            'interpolated_segments': interpolated_segments,
            'longest_interpolated_segment': max([s['length'] for s in interpolated_segments]) if interpolated_segments else 0
        }
    
    def plot_trajectories(self, output_path: Optional[str] = None):
        """Plot chaser and target trajectories with gap visualization."""
        if self.chaser_states is None:
            self.load_chaser_states()
            
        if self.chaser_states.empty:
            print("No data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        df = self.chaser_states.sort_values('stimulus_frame_num')
        
        # Plot 1: Chaser trajectory
        ax = axes[0, 0]
        scatter = ax.scatter(df['chaser_pos_x'], df['chaser_pos_y'], 
                           c=df['stimulus_frame_num'], cmap='viridis', s=2)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Chaser Trajectory (colored by frame number)')
        plt.colorbar(scatter, ax=ax)
        ax.set_aspect('equal')
        
        # Plot 2: Target trajectory
        ax = axes[0, 1]
        scatter = ax.scatter(df['target_pos_x'], df['target_pos_y'], 
                           c=df['stimulus_frame_num'], cmap='plasma', s=2)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Target Trajectory (colored by frame number)')
        plt.colorbar(scatter, ax=ax)
        ax.set_aspect('equal')
        
        # Plot 3: Frame coverage
        ax = axes[1, 0]
        frames = df['stimulus_frame_num'].values
        ax.plot(frames, np.ones_like(frames), 'b.', markersize=1)
        
        # Highlight gaps
        if len(frames) > 1:
            frame_diffs = np.diff(frames)
            gap_indices = np.where(frame_diffs > 1)[0]
            for idx in gap_indices:
                gap_start = frames[idx]
                gap_end = frames[idx + 1]
                ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')
        
        ax.set_xlabel('Frame Number')
        ax.set_yticks([])
        ax.set_title('Frame Coverage (red = gaps)')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Position changes over time
        ax = axes[1, 1]
        
        # Calculate distances
        df['chaser_dist'] = np.sqrt(
            df['chaser_pos_x'].diff()**2 + df['chaser_pos_y'].diff()**2
        )
        df['target_dist'] = np.sqrt(
            df['target_pos_x'].diff()**2 + df['target_pos_y'].diff()**2
        )
        
        ax.plot(df['stimulus_frame_num'], df['chaser_dist'], 
                'b-', label='Chaser', alpha=0.7, linewidth=0.5)
        ax.plot(df['stimulus_frame_num'], df['target_dist'], 
                'r-', label='Target', alpha=0.7, linewidth=0.5)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Movement Distance')
        ax.set_title('Frame-to-Frame Movement')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        print("Analyzing chaser/target data...")
        
        # Load data
        self.load_chaser_states()
        self.load_frame_metadata()
        
        report = {
            'file': str(self.h5_path),
            'has_chaser_states': self.chaser_states is not None and not self.chaser_states.empty,
            'has_frame_metadata': self.frame_metadata is not None and not self.frame_metadata.empty
        }
        
        if report['has_chaser_states']:
            print("  - Analyzing frame gaps...")
            report['frame_gaps'] = self.analyze_frame_gaps()
            
            print("  - Analyzing position discontinuities...")
            report['position_discontinuities'] = self.analyze_position_discontinuities()
            
            print("  - Analyzing chasing states...")
            report['chasing_states'] = self.analyze_chasing_states()
        
        if report['has_frame_metadata'] and report['has_chaser_states']:
            print("  - Analyzing camera frame mapping...")
            report['camera_frame_mapping'] = self.analyze_camera_frame_mapping()
        
        print("  - Checking for interpolation data...")
        report['interpolation'] = self.check_interpolation_mask()
        
        return report
    
    def print_summary(self, report: Dict):
        """Print human-readable summary of analysis."""
        print("\n" + "="*60)
        print(f"CHASER/TARGET ANALYSIS REPORT")
        print(f"File: {report['file']}")
        print("="*60)
        
        if not report['has_chaser_states']:
            print("\n⚠️  No chaser states data found in file!")
            return
        
        # Frame coverage
        if 'frame_gaps' in report:
            gaps = report['frame_gaps']
            print(f"\n📊 FRAME COVERAGE:")
            print(f"  • Frame range: {gaps['frame_range'][0]} - {gaps['frame_range'][1]}")
            print(f"  • Coverage: {gaps['actual_frames']}/{gaps['total_expected_frames']} frames ({gaps['coverage_percent']:.1f}%)")
            print(f"  • Missing frames: {gaps['missing_frames']}")
            print(f"  • Number of gaps: {gaps['num_gaps']}")
            if gaps['num_gaps'] > 0:
                print(f"  • Largest gap: {gaps['largest_gap']} frames")
                
                # Show first few gaps
                print(f"\n  Gap details (showing first 5):")
                for i, gap in enumerate(gaps['gaps'][:5]):
                    print(f"    Gap {i+1}: frames {gap['start_frame']}-{gap['end_frame']} ({gap['gap_size']} frames)")
        
        # Camera frame mapping
        if 'camera_frame_mapping' in report:
            mapping = report['camera_frame_mapping']
            print(f"\n📹 CAMERA FRAME MAPPING:")
            print(f"  • Camera frame range: {mapping['camera_frame_range'][0]} - {mapping['camera_frame_range'][1]}")
            print(f"  • Expected camera frames: {mapping['total_camera_frames_expected']}")
            print(f"  • Actual camera frames: {mapping['total_camera_frames_actual']}")
            print(f"  • Missing camera frames: {mapping['missing_camera_frames']}")
            
            if mapping['num_camera_frame_gaps'] > 0:
                print(f"  • Camera frame gaps: {mapping['num_camera_frame_gaps']}")
                for i, gap in enumerate(mapping['camera_frame_gaps'][:5]):
                    print(f"    Gap {i+1}: frames {gap['start']}-{gap['end']} ({gap['size']} frames)")
            
            print(f"\n  • Stimulus frames in metadata: {mapping['stimulus_frames_in_metadata']}")
            print(f"  • Stimulus frames with chaser states: {mapping['stimulus_frames_in_chaser']}")
            print(f"  • Stimulus frames WITHOUT chaser states: {mapping['stimulus_frames_without_chaser']}")
            
            if mapping['stimulus_frames_without_chaser'] > 0:
                print(f"    ⚠️  These stimulus frames have no chaser data (first 20):")
                print(f"    {mapping['stimulus_frames_without_chaser_list'][:20]}")
            
            if mapping['camera_frames_without_chaser'] > 0:
                print(f"\n  • Camera frames WITHOUT chaser states: {mapping['camera_frames_without_chaser']}")
                print(f"    ⚠️  These camera frames map to stimulus frames with no chaser data (first 20):")
                print(f"    {mapping['camera_frames_without_chaser_list'][:20]}")
        
        # Position discontinuities
        if 'position_discontinuities' in report:
            disc = report['position_discontinuities']
            print(f"\n🎯 POSITION CONTINUITY:")
            print(f"  • Mean chaser movement: {disc['mean_chaser_movement']:.2f} pixels/frame")
            print(f"  • Mean target movement: {disc['mean_target_movement']:.2f} pixels/frame")
            print(f"  • Chaser jumps (>100px): {disc['chaser_jumps']['count']}")
            if disc['chaser_jumps']['count'] > 0:
                print(f"    - Max jump: {disc['chaser_jumps']['max_jump']:.1f} pixels")
            print(f"  • Target jumps (>100px): {disc['target_jumps']['count']}")
            if disc['target_jumps']['count'] > 0:
                print(f"    - Max jump: {disc['target_jumps']['max_jump']:.1f} pixels")
            print(f"  • Frames with NaN positions: {disc['nan_positions']['count']}")
        
        # Chasing states
        if 'chasing_states' in report:
            states = report['chasing_states']
            print(f"\n🏃 CHASING BEHAVIOR:")
            print(f"  • Total frames: {states['total_frames']}")
            print(f"  • Chasing: {states['chasing_frames']} frames ({states['chasing_percentage']:.1f}%)")
            print(f"  • Not chasing: {states['not_chasing_frames']} frames")
            print(f"  • Number of chase periods: {states['num_chasing_periods']}")
            print(f"  • State transitions: {states['num_transitions']}")
            
            if states['chasing_periods']:
                durations = [p['duration_frames'] for p in states['chasing_periods']]
                print(f"  • Chase duration: min={min(durations)}, max={max(durations)}, mean={np.mean(durations):.1f} frames")
        
        # Interpolation info
        if report['interpolation']:
            interp = report['interpolation']
            print(f"\n🔄 INTERPOLATION DATA:")
            print(f"  • Original frames: {interp['original_frames']} ({100 - interp['interpolation_percentage']:.1f}%)")
            print(f"  • Interpolated frames: {interp['interpolated_frames']} ({interp['interpolation_percentage']:.1f}%)")
            print(f"  • Interpolated segments: {interp['num_interpolated_segments']}")
            if interp['longest_interpolated_segment'] > 0:
                print(f"  • Longest interpolated segment: {interp['longest_interpolated_segment']} frames")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Analyze chaser/target positions and gaps in H5 files')
    parser.add_argument('h5_file', help='Path to H5 file')
    parser.add_argument('--plot', action='store_true', help='Generate trajectory plots')
    parser.add_argument('--plot-output', help='Save plot to file instead of displaying')
    parser.add_argument('--json', help='Save report as JSON file')
    parser.add_argument('--max-jump', type=float, default=100.0, 
                       help='Maximum position jump threshold in pixels (default: 100)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.h5_file).exists():
        print(f"Error: File {args.h5_file} not found")
        return 1
    
    # Run analysis
    with ChaserTargetAnalyzer(args.h5_file) as analyzer:
        report = analyzer.generate_report()
        analyzer.print_summary(report)
        
        # Save JSON report if requested
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nReport saved to {args.json}")
        
        # Generate plots if requested
        if args.plot:
            analyzer.plot_trajectories(args.plot_output)
    
    return 0


if __name__ == '__main__':
    exit(main())