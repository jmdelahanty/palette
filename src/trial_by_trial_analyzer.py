#!/usr/bin/env python3
"""
Trial-by-Trial Chase Response Analyzer

Analyzes fish behavioral responses to individual chase trials, integrating:
- YOLO detections from zarr files
- Chase events and chaser positions from H5 files
- Frame alignment between 60Hz camera and 120Hz stimulus

This analyzer extracts metrics for each chase trial including:
- Fish-chaser distance over time
- Escape responses and velocities
- Spatial distribution changes
- Response latencies
"""

import zarr
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# Event type mappings
EVENT_TYPES = {
    24: "CHASER_PRE_PERIOD_START",
    25: "CHASER_TRAINING_START", 
    26: "CHASER_POST_PERIOD_START",
    27: "CHASER_CHASE_SEQUENCE_START",
    28: "CHASER_CHASE_SEQUENCE_END"
}

@dataclass
class TrialMetrics:
    """Metrics for a single chase trial."""
    trial_number: int
    start_time_s: float
    duration_s: float
    phase: str  # 'pre', 'training', 'post'
    
    # Distance metrics
    mean_distance_px: float
    min_distance_px: float
    max_distance_px: float
    initial_distance_px: float
    final_distance_px: float
    
    # Movement metrics
    fish_total_distance_px: float
    fish_mean_speed_px_per_s: float
    fish_max_speed_px_per_s: float
    chaser_total_distance_px: float
    
    # Response metrics
    escape_detected: bool
    escape_latency_s: Optional[float]
    escape_speed_px_per_s: Optional[float]
    approach_events: int
    
    # Coverage
    frames_with_detection: int
    total_frames: int
    detection_rate: float


class TrialByTrialAnalyzer:
    """Analyzes fish responses to individual chase trials."""
    
    def __init__(self, 
                 zarr_path: str,
                 h5_path: str,
                 escape_threshold_px_per_s: float = 500,
                 approach_threshold_px: float = 100,
                 verbose: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            zarr_path: Path to zarr file with YOLO detections
            h5_path: Path to H5 file with chase events and chaser positions
            escape_threshold_px_per_s: Speed threshold for escape detection
            approach_threshold_px: Distance threshold for approach detection
            verbose: Print progress messages
        """
        self.zarr_path = Path(zarr_path)
        self.h5_path = Path(h5_path)
        self.escape_threshold = escape_threshold_px_per_s
        self.approach_threshold = approach_threshold_px
        self.verbose = verbose
        
        # Camera to texture scaling (from your system)
        self.camera_width = 4512
        self.camera_height = 4512
        self.texture_width = 358
        self.texture_height = 358
        self.texture_to_camera_scale = self.camera_width / self.texture_width
        
        # Load data
        self.load_data()
        
        # Extract trials
        self.extract_trials()
    
    def load_data(self):
        """Load data from zarr and H5 files."""
        if self.verbose:
            print("Loading data...")
        
        # Load zarr data (YOLO detections)
        root = zarr.open(self.zarr_path, mode='r')
        
        # Try to get best available data (preprocessed or raw)
        if 'preprocessing' in root and root['preprocessing'].attrs.get('latest'):
            latest = root['preprocessing'].attrs['latest']
            data = root['preprocessing'][latest]
            self.bboxes = data['bboxes'][:]
            self.n_detections = data['n_detections'][:]
            if self.verbose:
                print(f"  Using preprocessed data: {latest}")
        else:
            self.bboxes = root['bboxes'][:]
            self.n_detections = root['n_detections'][:]
            if self.verbose:
                print("  Using raw detection data")
        
        # Load H5 data
        with h5py.File(self.h5_path, 'r') as f:
            # Events
            self.events = f['/events'][:]
            
            # Chaser states
            self.chaser_states = f['/tracking_data/chaser_states'][:]
            
            # Frame metadata for alignment
            self.frame_metadata = f['/video_metadata/frame_metadata'][:]
            
            # Bounding boxes from H5 (for frame alignment)
            self.h5_bboxes = f['/tracking_data/bounding_boxes'][:]
        
        if self.verbose:
            print(f"  Loaded {len(self.bboxes)} frames of detections")
            print(f"  Loaded {len(self.events)} events")
            print(f"  Loaded {len(self.chaser_states)} chaser states")
    
    def extract_trials(self):
        """Extract individual chase trials from events."""
        self.trials = []
        
        # Find chase sequences
        chase_starts = []
        chase_ends = []
        
        for event in self.events:
            if event['event_type_id'] == 27:  # CHASE_START
                chase_starts.append(event)
            elif event['event_type_id'] == 28:  # CHASE_END
                chase_ends.append(event)
        
        # Determine training phases
        training_start_time = None
        post_start_time = None
        
        for event in self.events:
            if event['event_type_id'] == 25:  # TRAINING_START
                training_start_time = event['timestamp_ns_session']
            elif event['event_type_id'] == 26:  # POST_START
                post_start_time = event['timestamp_ns_session']
        
        # Process each chase
        for i, (start_event, end_event) in enumerate(zip(chase_starts, chase_ends)):
            # Determine phase
            start_time = start_event['timestamp_ns_session']
            if training_start_time and start_time < training_start_time:
                phase = 'pre'
            elif post_start_time and start_time >= post_start_time:
                phase = 'post'
            else:
                phase = 'training'
            
            trial = {
                'number': i + 1,
                'phase': phase,
                'start_event': start_event,
                'end_event': end_event,
                'start_time_ns': start_time,
                'end_time_ns': end_event['timestamp_ns_session'],
                'duration_s': (end_event['timestamp_ns_session'] - start_time) / 1e9
            }
            
            self.trials.append(trial)
        
        if self.verbose:
            print(f"\nExtracted {len(self.trials)} trials:")
            phase_counts = {'pre': 0, 'training': 0, 'post': 0}
            for trial in self.trials:
                phase_counts[trial['phase']] += 1
            for phase, count in phase_counts.items():
                print(f"  {phase}: {count} trials")
    
    def get_trial_data(self, trial: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get fish positions, chaser positions, and timestamps for a trial.
        
        Returns:
            fish_positions: Array of (x, y) positions in camera space
            chaser_positions: Array of (x, y) positions in camera space
            timestamps: Array of timestamps in seconds
        """
        # Get chaser states for this trial
        time_mask = (self.chaser_states['timestamp_ns_session'] >= trial['start_time_ns']) & \
                   (self.chaser_states['timestamp_ns_session'] <= trial['end_time_ns'])
        trial_chaser = self.chaser_states[time_mask]
        
        if len(trial_chaser) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Get stimulus frame range
        stim_frames = trial_chaser['stimulus_frame_num']
        min_stim = stim_frames.min()
        max_stim = stim_frames.max()
        
        # Get camera frames for these stimulus frames
        meta_mask = (self.frame_metadata['stimulus_frame_num'] >= min_stim) & \
                   (self.frame_metadata['stimulus_frame_num'] <= max_stim)
        trial_metadata = self.frame_metadata[meta_mask]
        
        # Get unique camera frames
        camera_frames = np.unique(trial_metadata['triggering_camera_frame_id'])
        
        # Initialize arrays
        fish_positions = []
        chaser_positions = []
        timestamps = []
        
        for cam_frame in camera_frames:
            # Get fish position from zarr (YOLO detections)
            # Camera frames in zarr are 0-indexed, H5 frames may have offset
            zarr_frame_idx = cam_frame - camera_frames[0]  # Relative indexing
            
            if zarr_frame_idx < len(self.bboxes) and self.n_detections[zarr_frame_idx] > 0:
                bbox = self.bboxes[zarr_frame_idx, 0]  # First detection
                fish_x = (bbox[0] + bbox[2]) / 2  # Center x
                fish_y = (bbox[1] + bbox[3]) / 2  # Center y
                fish_positions.append([fish_x, fish_y])
            else:
                fish_positions.append([np.nan, np.nan])
            
            # Get corresponding chaser position
            # Find stimulus frame for this camera frame
            meta_row = trial_metadata[trial_metadata['triggering_camera_frame_id'] == cam_frame][0]
            stim_frame = meta_row['stimulus_frame_num']
            
            # Get chaser state for this stimulus frame
            chaser_mask = trial_chaser['stimulus_frame_num'] == stim_frame
            if np.any(chaser_mask):
                chaser_state = trial_chaser[chaser_mask][0]
                # Convert texture space to camera space
                chaser_x = chaser_state['chaser_pos_x'] * self.texture_to_camera_scale
                chaser_y = chaser_state['chaser_pos_y'] * self.texture_to_camera_scale
                chaser_positions.append([chaser_x, chaser_y])
            else:
                chaser_positions.append([np.nan, np.nan])
            
            # Get timestamp
            timestamps.append(meta_row['timestamp_ns'] / 1e9)
        
        return np.array(fish_positions), np.array(chaser_positions), np.array(timestamps)
    
    def calculate_trial_metrics(self, trial: Dict) -> TrialMetrics:
        """Calculate metrics for a single trial."""
        # Get trial data
        fish_pos, chaser_pos, timestamps = self.get_trial_data(trial)
        
        if len(fish_pos) == 0:
            # Return empty metrics if no data
            return TrialMetrics(
                trial_number=trial['number'],
                start_time_s=trial['start_time_ns'] / 1e9,
                duration_s=trial['duration_s'],
                phase=trial['phase'],
                mean_distance_px=np.nan,
                min_distance_px=np.nan,
                max_distance_px=np.nan,
                initial_distance_px=np.nan,
                final_distance_px=np.nan,
                fish_total_distance_px=0,
                fish_mean_speed_px_per_s=0,
                fish_max_speed_px_per_s=0,
                chaser_total_distance_px=0,
                escape_detected=False,
                escape_latency_s=None,
                escape_speed_px_per_s=None,
                approach_events=0,
                frames_with_detection=0,
                total_frames=len(fish_pos),
                detection_rate=0
            )
        
        # Calculate distances
        valid_mask = ~np.isnan(fish_pos[:, 0]) & ~np.isnan(chaser_pos[:, 0])
        distances = np.sqrt(np.sum((fish_pos - chaser_pos)**2, axis=1))
        valid_distances = distances[valid_mask]
        
        # Distance metrics
        if len(valid_distances) > 0:
            mean_distance = np.mean(valid_distances)
            min_distance = np.min(valid_distances)
            max_distance = np.max(valid_distances)
            initial_distance = valid_distances[0] if len(valid_distances) > 0 else np.nan
            final_distance = valid_distances[-1] if len(valid_distances) > 0 else np.nan
        else:
            mean_distance = min_distance = max_distance = initial_distance = final_distance = np.nan
        
        # Fish movement metrics
        valid_fish = fish_pos[valid_mask]
        if len(valid_fish) > 1:
            fish_displacements = np.sqrt(np.sum(np.diff(valid_fish, axis=0)**2, axis=1))
            fish_total_distance = np.sum(fish_displacements)
            
            # Calculate speeds (pixels per second)
            time_diffs = np.diff(timestamps[valid_mask])
            fish_speeds = fish_displacements / np.maximum(time_diffs, 0.001)  # Avoid division by zero
            fish_mean_speed = np.mean(fish_speeds)
            fish_max_speed = np.max(fish_speeds)
            
            # Detect escape response
            escape_detected = np.any(fish_speeds > self.escape_threshold)
            if escape_detected:
                escape_idx = np.argmax(fish_speeds > self.escape_threshold)
                escape_latency = timestamps[valid_mask][escape_idx + 1] - timestamps[0]
                escape_speed = fish_speeds[escape_idx]
            else:
                escape_latency = None
                escape_speed = None
        else:
            fish_total_distance = 0
            fish_mean_speed = 0
            fish_max_speed = 0
            escape_detected = False
            escape_latency = None
            escape_speed = None
        
        # Chaser movement
        valid_chaser = chaser_pos[valid_mask]
        if len(valid_chaser) > 1:
            chaser_displacements = np.sqrt(np.sum(np.diff(valid_chaser, axis=0)**2, axis=1))
            chaser_total_distance = np.sum(chaser_displacements)
        else:
            chaser_total_distance = 0
        
        # Approach events (when distance drops below threshold)
        approach_events = 0
        if len(valid_distances) > 1:
            below_threshold = valid_distances < self.approach_threshold
            # Count transitions from above to below threshold
            approach_events = np.sum(np.diff(below_threshold.astype(int)) > 0)
        
        return TrialMetrics(
            trial_number=trial['number'],
            start_time_s=trial['start_time_ns'] / 1e9,
            duration_s=trial['duration_s'],
            phase=trial['phase'],
            mean_distance_px=mean_distance,
            min_distance_px=min_distance,
            max_distance_px=max_distance,
            initial_distance_px=initial_distance,
            final_distance_px=final_distance,
            fish_total_distance_px=fish_total_distance,
            fish_mean_speed_px_per_s=fish_mean_speed,
            fish_max_speed_px_per_s=fish_max_speed,
            chaser_total_distance_px=chaser_total_distance,
            escape_detected=escape_detected,
            escape_latency_s=escape_latency,
            escape_speed_px_per_s=escape_speed,
            approach_events=approach_events,
            frames_with_detection=np.sum(valid_mask),
            total_frames=len(fish_pos),
            detection_rate=np.sum(valid_mask) / len(fish_pos) if len(fish_pos) > 0 else 0
        )
    
    def analyze_all_trials(self) -> pd.DataFrame:
        """Analyze all trials and return results as DataFrame."""
        if self.verbose:
            print("\nAnalyzing trials...")
        
        metrics_list = []
        for trial in self.trials:
            metrics = self.calculate_trial_metrics(trial)
            metrics_list.append(asdict(metrics))
            
            if self.verbose and trial['number'] % 5 == 0:
                print(f"  Processed trial {trial['number']}/{len(self.trials)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_list)
        
        # Add pixel to mm conversion if available
        pixel_to_mm = 0.019605  # From your calibration
        for col in df.columns:
            if '_px' in col:
                mm_col = col.replace('_px', '_mm')
                df[mm_col] = df[col] * pixel_to_mm
        
        if self.verbose:
            print(f"\nAnalysis complete! Processed {len(df)} trials")
        
        return df
    
    def plot_trial_summary(self, trial_num: int):
        """Plot detailed summary for a single trial."""
        trial = self.trials[trial_num - 1]
        fish_pos, chaser_pos, timestamps = self.get_trial_data(trial)
        
        if len(fish_pos) == 0:
            print(f"No data available for trial {trial_num}")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # 1. Trajectory plot
        ax1 = fig.add_subplot(gs[0, :2])
        valid = ~np.isnan(fish_pos[:, 0])
        ax1.plot(fish_pos[valid, 0], fish_pos[valid, 1], 'b-', alpha=0.7, label='Fish', linewidth=2)
        ax1.plot(chaser_pos[valid, 0], chaser_pos[valid, 1], 'r-', alpha=0.7, label='Chaser', linewidth=2)
        ax1.scatter(fish_pos[valid, 0][0], fish_pos[valid, 1][0], c='blue', s=100, marker='o', label='Fish start')
        ax1.scatter(fish_pos[valid, 0][-1], fish_pos[valid, 1][-1], c='blue', s=100, marker='s', label='Fish end')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_title(f'Trial {trial_num} ({trial["phase"].capitalize()}) - Trajectories')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distance over time
        ax2 = fig.add_subplot(gs[1, :2])
        distances = np.sqrt(np.sum((fish_pos - chaser_pos)**2, axis=1))
        valid_times = timestamps - timestamps[0]
        ax2.plot(valid_times[valid], distances[valid], 'g-', linewidth=2)
        ax2.axhline(y=self.approach_threshold, color='r', linestyle='--', alpha=0.5, label='Approach threshold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Distance (pixels)')
        ax2.set_title('Fish-Chaser Distance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Speed profile
        ax3 = fig.add_subplot(gs[0, 2])
        if np.sum(valid) > 1:
            time_diffs = np.diff(timestamps[valid])
            fish_displacements = np.sqrt(np.sum(np.diff(fish_pos[valid], axis=0)**2, axis=1))
            speeds = fish_displacements / np.maximum(time_diffs, 0.001)
            speed_times = valid_times[valid][1:]
            ax3.plot(speed_times, speeds, 'b-', linewidth=1)
            ax3.axhline(y=self.escape_threshold, color='r', linestyle='--', alpha=0.5, label='Escape threshold')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Speed (px/s)')
            ax3.set_title('Fish Speed')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Metrics summary
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        metrics = self.calculate_trial_metrics(trial)
        
        text = f"Trial {trial_num} Summary\n"
        text += f"{'='*20}\n"
        text += f"Phase: {trial['phase'].capitalize()}\n"
        text += f"Duration: {metrics.duration_s:.2f} s\n"
        text += f"Detection rate: {metrics.detection_rate:.1%}\n\n"
        text += f"Distance Metrics:\n"
        text += f"  Mean: {metrics.mean_distance_px:.0f} px\n"
        text += f"  Min: {metrics.min_distance_px:.0f} px\n"
        text += f"  Max: {metrics.max_distance_px:.0f} px\n\n"
        text += f"Movement Metrics:\n"
        text += f"  Fish speed: {metrics.fish_mean_speed_px_per_s:.0f} px/s\n"
        text += f"  Max speed: {metrics.fish_max_speed_px_per_s:.0f} px/s\n"
        text += f"  Escape: {'Yes' if metrics.escape_detected else 'No'}\n"
        if metrics.escape_detected:
            text += f"  Latency: {metrics.escape_latency_s:.3f} s\n"
        
        ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.suptitle(f'Trial {trial_num} Analysis', fontsize=14, y=1.02)
        plt.show()
    
    def plot_phase_comparison(self, df: pd.DataFrame):
        """Plot comparison of metrics across phases."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Mean distance by phase
        ax = axes[0, 0]
        sns.boxplot(data=df, x='phase', y='mean_distance_px', ax=ax, order=['pre', 'training', 'post'])
        ax.set_title('Mean Fish-Chaser Distance')
        ax.set_ylabel('Distance (pixels)')
        
        # 2. Fish speed by phase
        ax = axes[0, 1]
        sns.boxplot(data=df, x='phase', y='fish_mean_speed_px_per_s', ax=ax, order=['pre', 'training', 'post'])
        ax.set_title('Mean Fish Speed')
        ax.set_ylabel('Speed (px/s)')
        
        # 3. Escape probability by phase
        ax = axes[0, 2]
        escape_rates = df.groupby('phase')['escape_detected'].mean()
        escape_rates = escape_rates.reindex(['pre', 'training', 'post'])
        ax.bar(escape_rates.index, escape_rates.values)
        ax.set_title('Escape Probability')
        ax.set_ylabel('Proportion of Trials')
        ax.set_ylim([0, 1])
        
        # 4. Escape latency by phase (only for trials with escapes)
        ax = axes[1, 0]
        escape_df = df[df['escape_detected']].copy()
        if len(escape_df) > 0:
            sns.boxplot(data=escape_df, x='phase', y='escape_latency_s', ax=ax, order=['pre', 'training', 'post'])
        ax.set_title('Escape Latency (when escape occurs)')
        ax.set_ylabel('Latency (s)')
        
        # 5. Min distance by phase
        ax = axes[1, 1]
        sns.boxplot(data=df, x='phase', y='min_distance_px', ax=ax, order=['pre', 'training', 'post'])
        ax.set_title('Minimum Fish-Chaser Distance')
        ax.set_ylabel('Distance (pixels)')
        
        # 6. Detection rate by phase
        ax = axes[1, 2]
        sns.boxplot(data=df, x='phase', y='detection_rate', ax=ax, order=['pre', 'training', 'post'])
        ax.set_title('Detection Rate')
        ax.set_ylabel('Proportion of Frames')
        
        plt.tight_layout()
        plt.suptitle('Behavioral Metrics Across Training Phases', fontsize=14, y=1.02)
        plt.show()
    
    def save_results(self, df: pd.DataFrame, output_path: Optional[str] = None):
        """Save analysis results to CSV."""
        if output_path is None:
            output_path = self.h5_path.with_suffix('.trial_metrics.csv')
        
        df.to_csv(output_path, index=False)
        if self.verbose:
            print(f"\nResults saved to: {output_path}")
        
        # Also save summary statistics
        summary_path = Path(output_path).with_suffix('.summary.json')
        
        summary = {
            'n_trials': len(df),
            'phases': {
                phase: {
                    'n_trials': int(np.sum(df['phase'] == phase)),
                    'mean_distance_px': float(df[df['phase'] == phase]['mean_distance_px'].mean()),
                    'mean_speed_px_per_s': float(df[df['phase'] == phase]['fish_mean_speed_px_per_s'].mean()),
                    'escape_rate': float(df[df['phase'] == phase]['escape_detected'].mean()),
                    'detection_rate': float(df[df['phase'] == phase]['detection_rate'].mean())
                }
                for phase in ['pre', 'training', 'post']
            },
            'analysis_date': datetime.now().isoformat(),
            'parameters': {
                'escape_threshold_px_per_s': self.escape_threshold,
                'approach_threshold_px': self.approach_threshold
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.verbose:
            print(f"Summary saved to: {summary_path}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Analyze fish responses to individual chase trials',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This analyzer extracts behavioral metrics for each chase trial including:
- Fish-chaser distances
- Movement speeds and escape responses  
- Comparison across training phases (pre/training/post)

Examples:
  %(prog)s detections.zarr analysis.h5
  %(prog)s detections.zarr analysis.h5 --plot-trial 5
  %(prog)s detections.zarr analysis.h5 --plot-comparison --save
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file with YOLO detections')
    parser.add_argument('h5_path', help='Path to H5 analysis file')
    parser.add_argument('--plot-trial', type=int, help='Plot detailed view of specific trial')
    parser.add_argument('--plot-comparison', action='store_true', 
                       help='Plot comparison across phases')
    parser.add_argument('--save', action='store_true', 
                       help='Save results to CSV')
    parser.add_argument('--output', help='Output path for CSV (default: auto-generated)')
    parser.add_argument('--escape-threshold', type=float, default=500,
                       help='Speed threshold for escape detection (px/s)')
    parser.add_argument('--approach-threshold', type=float, default=100,
                       help='Distance threshold for approach detection (px)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TrialByTrialAnalyzer(
        zarr_path=args.zarr_path,
        h5_path=args.h5_path,
        escape_threshold_px_per_s=args.escape_threshold,
        approach_threshold_px=args.approach_threshold,
        verbose=not args.quiet
    )
    
    # Run analysis
    df = analyzer.analyze_all_trials()
    
    # Print summary
    if not args.quiet:
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        for phase in ['pre', 'training', 'post']:
            phase_df = df[df['phase'] == phase]
            if len(phase_df) > 0:
                print(f"\n{phase.upper()} Phase ({len(phase_df)} trials):")
                print(f"  Mean distance: {phase_df['mean_distance_px'].mean():.0f} px")
                print(f"  Mean speed: {phase_df['fish_mean_speed_px_per_s'].mean():.0f} px/s")
                print(f"  Escape rate: {phase_df['escape_detected'].mean():.1%}")
                print(f"  Detection rate: {phase_df['detection_rate'].mean():.1%}")
    
    # Plot if requested
    if args.plot_trial:
        analyzer.plot_trial_summary(args.plot_trial)
    
    if args.plot_comparison:
        analyzer.plot_phase_comparison(df)
    
    # Save if requested
    if args.save:
        analyzer.save_results(df, args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())