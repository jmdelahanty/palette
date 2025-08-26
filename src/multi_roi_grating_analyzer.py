#!/usr/bin/env python3
"""
Multi-ROI Moving Grating Trial Analyzer

Analyzes behavioral responses of multiple fish to moving grating stimuli, integrating:
- YOLO detections from zarr files (multiple ROIs with corrected coordinates)
- Stimulus events from H5 files (moving gratings, not chaser)
- Trial structure based on step start/end events
- Speed and movement analysis during different stimulus phases
"""

import zarr
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
from scipy.ndimage import gaussian_filter1d
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import warnings
warnings.filterwarnings('ignore')

console = Console()
sns.set_palette("husl")

# Event type mappings for moving grating experiments
EVENT_TYPES = {
    0: "PROTOCOL_START",
    1: "PROTOCOL_STOP",
    2: "PROTOCOL_PAUSE",
    3: "PROTOCOL_RESUME",
    4: "PROTOCOL_FINISH",
    11: "STEP_START",
    12: "STEP_END",
    13: "ITI_START",
    14: "ITI_END"
}

# Stimulus mode types
STIMULUS_MODES = {
    3: "MOVING_GRATING",
    4: "SOLID_BLACK",
    5: "SOLID_WHITE",
    99: "NONE"
}

@dataclass
class TrialMetrics:
    """Metrics for a single fish during a stimulus trial."""
    roi_id: int
    trial_number: int
    trial_type: str  # 'grating', 'black', 'white', 'iti'
    start_frame: int
    end_frame: int
    duration_s: float
    
    # Movement metrics
    total_distance_px: float
    mean_speed_px_per_s: float
    median_speed_px_per_s: float
    max_speed_px_per_s: float
    std_speed_px_per_s: float
    
    # Activity metrics
    active_frames: int  # Frames with movement > threshold
    activity_ratio: float
    bout_count: int  # Number of movement bouts
    mean_bout_duration_s: float
    
    # Coverage metrics
    frames_with_detection: int
    total_frames: int
    detection_rate: float
    
    # Spatial metrics
    mean_x_position: float
    mean_y_position: float
    spatial_variance: float  # How spread out the positions are


class MultiROIGratingAnalyzer:
    """Analyzes multiple fish responses to moving grating trials."""
    
    def __init__(self, 
                 zarr_path: str,
                 h5_path: str,
                 activity_threshold_px_per_s: float = 10.0,
                 use_interpolated: bool = True,
                 verbose: bool = True):
        """
        Initialize analyzer for moving grating experiments.
        
        Args:
            zarr_path: Path to zarr file with detections
            h5_path: Path to H5 file with stimulus data
            activity_threshold_px_per_s: Speed threshold for active vs inactive
            use_interpolated: Include interpolated detections
            verbose: Print detailed output
        """
        self.zarr_path = Path(zarr_path)
        self.h5_path = Path(h5_path)
        self.activity_threshold = activity_threshold_px_per_s
        self.use_interpolated = use_interpolated
        self.verbose = verbose
        
        # Load zarr data first to get dimensions and fps
        self.root = zarr.open_group(self.zarr_path, mode='r')
        
        # Get dimensions and fps BEFORE loading H5 data
        self.camera_width = self.root.attrs.get('width', 4512)
        self.camera_height = self.root.attrs.get('height', 4512)
        self.fps = self.root.attrs.get('fps', 60.0)

        self.load_roi_boundaries_from_config()
        
        # Check for calibration
        self.pixel_to_mm = None
        if 'calibration' in self.root:
            self.pixel_to_mm = self.root['calibration'].attrs.get('pixel_to_mm', None)
        
        # Now load H5 and extract trials (which needs fps)
        self.load_h5_data()
        self.load_roi_data()
        self.extract_trials()
        
        if verbose:
            console.print(f"[cyan]Loaded {self.num_rois} ROIs and {len(self.trials)} trials[/cyan]")
            self.print_trial_summary()
    
    def load_h5_data(self):
        """Load events from H5 file (simplified structure without frame IDs)."""
        with h5py.File(self.h5_path, 'r') as f:
            # Load events
            events_data = f['/events'][:]
            
            # Create structured array with available fields
            self.events = []
            for event in events_data:
                event_dict = {
                    'timestamp_ns_epoch': int(event['timestamp_ns_epoch']),
                    'timestamp_ns_session': int(event['timestamp_ns_session']),
                    'event_type_id': int(event['event_type_id']),
                    'stimulus_mode_id': int(event['stimulus_mode_id']) if 'stimulus_mode_id' in event.dtype.names else -1,
                    'name_or_context': event['name_or_context'].decode('utf-8') if isinstance(event['name_or_context'], bytes) else str(event['name_or_context'])
                }
                self.events.append(event_dict)
            
            # Get session start time for frame estimation
            self.session_start_ns = f.attrs.get('session_start_ns_epoch', 0)
    
    def estimate_frame_from_timestamp(self, timestamp_ns_session: int) -> int:
        """Estimate video frame number from session timestamp."""
        # Convert nanoseconds to seconds and multiply by FPS
        time_s = timestamp_ns_session / 1e9
        return int(time_s * self.fps)
    
    def load_roi_data(self):
        """Load detection data for all ROIs with corrected coordinates."""
        # Get detection data
        detect_group = self.root['detect_runs']
        latest_detect = detect_group.attrs['latest']
        self.n_detections = detect_group[latest_detect]['n_detections'][:]
        self.bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
        
        # Get ID assignments
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        self.detection_ids = id_group[latest_id]['detection_ids'][:]
        self.n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        self.num_rois = self.n_detections_per_roi.shape[1]
        
        # Load interpolated if available
        self.interpolated_data = None
        if self.use_interpolated and 'interpolated_detections' in self.root:
            interp_group = self.root['interpolated_detections']
            if 'latest' in interp_group.attrs:
                latest_interp = interp_group.attrs['latest']
                interp_data = interp_group[latest_interp]
                self.interpolated_data = {
                    'frame_indices': interp_data['frame_indices'][:],
                    'roi_ids': interp_data['roi_ids'][:],
                    'bboxes': interp_data['bboxes'][:]
                }
    
    def get_roi_positions_for_frames(self, roi_id: int, start_frame: int, end_frame: int) -> np.ndarray:
        """
        Get positions for specific ROI during frame range with CORRECTED coordinates.
        
        Returns:
            Array of [x, y] positions in camera pixels (NaN for missing frames)
        """
        frame_range = range(start_frame, min(end_frame, len(self.n_detections)))
        positions = np.full((len(frame_range), 2), np.nan)
        
        # Get original detections
        cumulative_idx = 0
        for frame_idx in range(len(self.n_detections)):
            frame_det_count = int(self.n_detections[frame_idx])
            
            if frame_idx in frame_range:
                i = frame_idx - start_frame
                
                if frame_det_count > 0 and self.n_detections_per_roi[frame_idx, roi_id] > 0:
                    frame_detection_ids = self.detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
                    roi_mask = frame_detection_ids == roi_id
                    
                    if np.any(roi_mask):
                        roi_idx = np.where(roi_mask)[0][0]
                        bbox = self.bbox_coords[cumulative_idx + roi_idx]
                        
                        # CORRECTED: bbox[0] and bbox[1] are already centers!
                        center_x_norm = bbox[0]
                        center_y_norm = bbox[1]
                        
                        # Convert to camera pixels (640 -> 4512)
                        centroid_x_ds = center_x_norm * 640
                        centroid_y_ds = center_y_norm * 640
                        scale = self.camera_width / 640
                        
                        positions[i, 0] = centroid_x_ds * scale
                        positions[i, 1] = centroid_y_ds * scale
            
            cumulative_idx += frame_det_count
        
        # Add interpolated positions
        if self.interpolated_data is not None:
            for j in range(len(self.interpolated_data['frame_indices'])):
                frame_idx = int(self.interpolated_data['frame_indices'][j])
                if start_frame <= frame_idx < end_frame and int(self.interpolated_data['roi_ids'][j]) == roi_id:
                    i = frame_idx - start_frame
                    if i < len(positions) and np.isnan(positions[i, 0]):  # Only use if no original detection
                        bbox = self.interpolated_data['bboxes'][j]
                        center_x_norm = bbox[0]
                        center_y_norm = bbox[1]
                        
                        centroid_x_ds = center_x_norm * 640
                        centroid_y_ds = center_y_norm * 640
                        scale = self.camera_width / 640
                        
                        positions[i, 0] = centroid_x_ds * scale
                        positions[i, 1] = centroid_y_ds * scale
        
        return positions
    
    def extract_trials(self):
        """Extract trial information from events (step start/end pairs)."""
        self.trials = []
        
        # Find step start and end events
        step_starts = []
        step_ends = []
        
        for event in self.events:
            if event['event_type_id'] == 11:  # STEP_START
                step_starts.append(event)
            elif event['event_type_id'] == 12:  # STEP_END
                step_ends.append(event)
        
        # Match starts with ends
        for i, (start_event, end_event) in enumerate(zip(step_starts, step_ends)):
            # Determine trial type from stimulus mode
            stim_mode = start_event.get('stimulus_mode_id', -1)
            if stim_mode == 3:
                trial_type = 'grating'
            elif stim_mode == 4:
                trial_type = 'black'
            elif stim_mode == 5:
                trial_type = 'white'
            else:
                trial_type = 'unknown'
            
            # Estimate frames from timestamps
            start_frame = self.estimate_frame_from_timestamp(start_event['timestamp_ns_session'])
            end_frame = self.estimate_frame_from_timestamp(end_event['timestamp_ns_session'])
            
            trial = {
                'number': i + 1,
                'type': trial_type,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration_s': (end_event['timestamp_ns_session'] - start_event['timestamp_ns_session']) / 1e9,
                'name': start_event.get('name_or_context', f'Trial {i+1}')
            }
            
            self.trials.append(trial)
    
    def calculate_speed_for_positions(self, positions: np.ndarray) -> Dict:
        """Calculate speed metrics from position array."""
        valid_mask = ~np.isnan(positions[:, 0])
        
        if np.sum(valid_mask) < 2:
            return {
                'speeds': np.array([]),
                'mean_speed': 0,
                'median_speed': 0,
                'max_speed': 0,
                'std_speed': 0,
                'total_distance': 0
            }
        
        # Calculate frame-to-frame distances
        valid_pos = positions[valid_mask]
        distances = np.sqrt(np.sum(np.diff(valid_pos, axis=0)**2, axis=1))
        speeds = distances * self.fps
        
        return {
            'speeds': speeds,
            'mean_speed': np.mean(speeds),
            'median_speed': np.median(speeds),
            'max_speed': np.max(speeds),
            'std_speed': np.std(speeds),
            'total_distance': np.sum(distances)
        }
    
    def plot_temporal_analysis(self, df: pd.DataFrame, bin_minutes: float = 1.0, 
                            save_path: Optional[str] = None):
        """Create temporal analysis plots showing activity over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert frame numbers to minutes
        df['time_minutes'] = df['start_frame'] / (self.fps * 60)
        
        # Create time bins
        max_time = df['time_minutes'].max()
        time_bins = np.arange(0, max_time + bin_minutes, bin_minutes)
        df['time_bin'] = pd.cut(df['time_minutes'], bins=time_bins, 
                                labels=time_bins[:-1], include_lowest=True)
        
        stimulus_colors = {
            'grating': '#FF6B6B',
            'black': '#4ECDC4', 
            'white': '#95E77E',
            'unknown': '#FFE66D'
        }
        
        # 1. Mean speed over time (all fish averaged)
        ax = axes[0, 0]
        time_speed = df.groupby(['time_bin', 'trial_type'])['mean_speed_px_per_s'].mean().reset_index()
        for trial_type in time_speed['trial_type'].unique():
            type_data = time_speed[time_speed['trial_type'] == trial_type]
            ax.plot(type_data['time_bin'], type_data['mean_speed_px_per_s'],
                marker='o', label=trial_type, color=stimulus_colors.get(trial_type, 'gray'),
                linewidth=2, markersize=6)
        ax.set_xlabel(f'Time (minutes)')
        ax.set_ylabel('Mean Speed (px/s)')
        ax.set_title('Average Speed Over Time by Stimulus Type', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Activity ratio over time
        ax = axes[0, 1]
        time_activity = df.groupby(['time_bin', 'trial_type'])['activity_ratio'].mean().reset_index()
        for trial_type in time_activity['trial_type'].unique():
            type_data = time_activity[time_activity['trial_type'] == trial_type]
            ax.plot(type_data['time_bin'], type_data['activity_ratio'],
                marker='s', label=trial_type, color=stimulus_colors.get(trial_type, 'gray'),
                linewidth=2, markersize=6)
        ax.set_xlabel(f'Time (minutes)')
        ax.set_ylabel('Activity Ratio')
        ax.set_title('Activity Level Over Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(0.5, time_activity['activity_ratio'].max() * 1.1)])
        
        # 3. Individual fish speeds over time
        ax = axes[1, 0]
        n_fish = df['roi_id'].nunique()
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_fish, 10)))
        
        for idx, roi_id in enumerate(sorted(df['roi_id'].unique())[:10]):  # Limit to 10 fish for clarity
            fish_df = df[df['roi_id'] == roi_id]
            fish_time = fish_df.groupby('time_bin')['mean_speed_px_per_s'].mean().reset_index()
            ax.plot(fish_time['time_bin'], fish_time['mean_speed_px_per_s'],
                alpha=0.6, label=f'Fish {roi_id}', color=colors[idx % 10],
                linewidth=1.5)
        
        ax.set_xlabel(f'Time (minutes)')
        ax.set_ylabel('Mean Speed (px/s)')
        ax.set_title('Individual Fish Speed Profiles', fontweight='bold')
        if n_fish <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
        else:
            ax.text(0.98, 0.98, f'Showing 10/{n_fish} fish', transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, alpha=0.3)
        
        # 4. Cumulative distance over time (population average)
        ax = axes[1, 1]
        
        # Calculate cumulative distance for each fish
        cumulative_data = []
        for roi_id in df['roi_id'].unique():
            fish_df = df[df['roi_id'] == roi_id].sort_values('time_minutes')
            fish_df['cumulative_distance'] = fish_df['total_distance_px'].cumsum()
            cumulative_data.append(fish_df[['time_minutes', 'cumulative_distance', 'roi_id']])
        
        cumulative_df = pd.concat(cumulative_data)
        
        # Plot mean with error bars
        time_points = sorted(df['time_minutes'].unique())
        mean_cumulative = []
        std_cumulative = []
        
        for time_point in time_points:
            time_data = cumulative_df[cumulative_df['time_minutes'] <= time_point]
            by_fish = time_data.groupby('roi_id')['cumulative_distance'].max()
            mean_cumulative.append(by_fish.mean())
            std_cumulative.append(by_fish.std())
        
        mean_cumulative = np.array(mean_cumulative)
        std_cumulative = np.array(std_cumulative)
        
        ax.plot(time_points, mean_cumulative, 'b-', linewidth=2, label='Mean')
        ax.fill_between(time_points, 
                        mean_cumulative - std_cumulative,
                        mean_cumulative + std_cumulative,
                        alpha=0.3, color='blue', label='±1 SD')
        
        ax.set_xlabel(f'Time (minutes)')
        ax.set_ylabel('Cumulative Distance (px)')
        ax.set_title('Population Cumulative Distance', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add stimulus timing bars at the top
        for ax_row in axes:
            for ax in ax_row:
                # Add colored bars for stimulus periods
                y_top = ax.get_ylim()[1]
                for trial in self.trials:
                    trial_start_min = trial['start_frame'] / (self.fps * 60)
                    trial_end_min = trial['end_frame'] / (self.fps * 60)
                    color = stimulus_colors.get(trial['type'], 'gray')
                    ax.axvspan(trial_start_min, trial_end_min, 
                            ymin=0.98, ymax=1.0, 
                            color=color, alpha=0.5, zorder=10)
        
        plt.suptitle(f'Temporal Analysis (bin size: {bin_minutes} min)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Temporal analysis plot saved to:[/green] {save_path}")
        
        plt.show()

    def plot_heatmap_temporal(self, df: pd.DataFrame, bin_minutes: float = 1.0,
                            save_path: Optional[str] = None):
        """Create temporal heatmap showing speed patterns across time and fish."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                    gridspec_kw={'height_ratios': [1, 3]})
        
        # Convert frame numbers to minutes
        df['time_minutes'] = df['start_frame'] / (self.fps * 60)
        
        # Create time bins
        max_time = df['time_minutes'].max()
        time_bins = np.arange(0, max_time + bin_minutes, bin_minutes)
        n_bins = len(time_bins) - 1
        
        # Create matrix for heatmap (fish x time bins)
        n_fish = df['roi_id'].nunique()
        speed_matrix = np.full((n_fish, n_bins), np.nan)
        activity_matrix = np.full((n_fish, n_bins), np.nan)
        
        for i, roi_id in enumerate(sorted(df['roi_id'].unique())):
            fish_df = df[df['roi_id'] == roi_id]
            for j, (bin_start, bin_end) in enumerate(zip(time_bins[:-1], time_bins[1:])):
                bin_trials = fish_df[(fish_df['time_minutes'] >= bin_start) & 
                                    (fish_df['time_minutes'] < bin_end)]
                if len(bin_trials) > 0:
                    speed_matrix[i, j] = bin_trials['mean_speed_px_per_s'].mean()
                    activity_matrix[i, j] = bin_trials['activity_ratio'].mean()
        
        # Plot 1: Stimulus timeline
        stimulus_colors = {
            'grating': '#FF6B6B',
            'black': '#4ECDC4',
            'white': '#95E77E',
            'unknown': '#FFE66D'
        }
        
        for trial in self.trials:
            trial_start_min = trial['start_frame'] / (self.fps * 60)
            trial_end_min = trial['end_frame'] / (self.fps * 60)
            color = stimulus_colors.get(trial['type'], 'gray')
            ax1.axvspan(trial_start_min, trial_end_min, color=color, alpha=0.7)
            
            # Add text labels for stimulus type
            trial_mid = (trial_start_min + trial_end_min) / 2
            ax1.text(trial_mid, 0.5, trial['type'], ha='center', va='center',
                    fontweight='bold', fontsize=9)
        
        ax1.set_xlim([0, max_time])
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('Stimulus', fontweight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('Experiment Timeline', fontweight='bold')
        
        # Plot 2: Speed heatmap
        im = ax2.imshow(speed_matrix, aspect='auto', cmap='YlOrRd',
                    extent=[0, max_time, n_fish-0.5, -0.5])
        
        ax2.set_xlabel(f'Time (minutes)', fontweight='bold')
        ax2.set_ylabel('Fish ID', fontweight='bold')
        ax2.set_yticks(range(n_fish))
        ax2.set_yticklabels([f'{i}' for i in sorted(df['roi_id'].unique())])
        ax2.set_title('Speed Heatmap Across Time', fontweight='bold')
        
        # Add vertical lines for trial boundaries
        for trial in self.trials:
            trial_start_min = trial['start_frame'] / (self.fps * 60)
            ax2.axvline(trial_start_min, color='white', linewidth=0.5, alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Mean Speed (px/s)', rotation=270, labelpad=20)
        
        # Add summary statistics as text
        overall_mean = np.nanmean(speed_matrix)
        overall_std = np.nanstd(speed_matrix)
        stats_text = f'Overall: {overall_mean:.1f} ± {overall_std:.1f} px/s'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=10, color='white', fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.suptitle(f'Temporal Speed Heatmap (bin size: {bin_minutes} min)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            # If save_path doesn't specify heatmap, add it
            if 'heatmap' not in save_path.lower():
                save_path = save_path.replace('.png', '_heatmap.png').replace('.pdf', '_heatmap.pdf')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Temporal heatmap saved to:[/green] {save_path}")
        
        plt.show()

    def save_results(self, df: pd.DataFrame, output_path: str):
        """Save analysis results to CSV."""
        df.to_csv(output_path, index=False)
        console.print(f"[green]✓ Results saved to:[/green] {output_path}")
        
        # Also save summary statistics
        summary_path = Path(output_path).with_suffix('.summary.json')
        summary = {
            'num_fish': self.num_rois,
            'num_trials': len(self.trials),
            'trial_types': df['trial_type'].value_counts().to_dict(),
            'mean_speed_by_stimulus': df.groupby('trial_type')['mean_speed_px_per_s'].mean().to_dict(),
            'mean_activity_by_stimulus': df.groupby('trial_type')['activity_ratio'].mean().to_dict(),
            'detection_rates': {
                'overall': df['detection_rate'].mean(),
                'by_stimulus': df.groupby('trial_type')['detection_rate'].mean().to_dict()
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        console.print(f"[green]✓ Summary saved to:[/green] {summary_path}")
        
    def calculate_trial_metrics_for_roi(self, roi_id: int, trial: Dict) -> TrialMetrics:
        """Calculate metrics for one ROI during one trial."""
        # Get positions
        positions = self.get_roi_positions_for_frames(roi_id, trial['start_frame'], trial['end_frame'])
        
        # Calculate speed metrics
        speed_info = self.calculate_speed_for_positions(positions)
        
        # Activity analysis
        active_frames = 0
        bout_count = 0
        bout_durations = []
        
        if len(speed_info['speeds']) > 0:
            active_mask = speed_info['speeds'] > self.activity_threshold
            active_frames = np.sum(active_mask)
            
            # Count bouts (transitions from inactive to active)
            if len(active_mask) > 0:
                transitions = np.diff(np.concatenate(([False], active_mask, [False])).astype(int))
                bout_starts = np.where(transitions == 1)[0]
                bout_ends = np.where(transitions == -1)[0]
                bout_count = len(bout_starts)
                bout_durations = (bout_ends - bout_starts) / self.fps
        
        # Spatial metrics
        valid_mask = ~np.isnan(positions[:, 0])
        if np.any(valid_mask):
            valid_pos = positions[valid_mask]
            mean_x = np.mean(valid_pos[:, 0])
            mean_y = np.mean(valid_pos[:, 1])
            spatial_variance = np.mean(np.var(valid_pos, axis=0))
        else:
            mean_x = mean_y = spatial_variance = np.nan
        
        return TrialMetrics(
            roi_id=roi_id,
            trial_number=trial['number'],
            trial_type=trial['type'],
            start_frame=trial['start_frame'],
            end_frame=trial['end_frame'],
            duration_s=trial['duration_s'],
            total_distance_px=speed_info['total_distance'],
            mean_speed_px_per_s=speed_info['mean_speed'],
            median_speed_px_per_s=speed_info['median_speed'],
            max_speed_px_per_s=speed_info['max_speed'],
            std_speed_px_per_s=speed_info['std_speed'],
            active_frames=active_frames,
            activity_ratio=active_frames / len(speed_info['speeds']) if len(speed_info['speeds']) > 0 else 0,
            bout_count=bout_count,
            mean_bout_duration_s=np.mean(bout_durations) if len(bout_durations) > 0 else 0,
            frames_with_detection=np.sum(valid_mask),
            total_frames=len(positions),
            detection_rate=np.sum(valid_mask) / len(positions) if len(positions) > 0 else 0,
            mean_x_position=mean_x,
            mean_y_position=mean_y,
            spatial_variance=spatial_variance
        )
    
    def analyze_all_trials(self) -> pd.DataFrame:
        """Analyze all ROIs across all trials."""
        metrics_list = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            total_analyses = self.num_rois * len(self.trials)
            task = progress.add_task(
                f"[cyan]Analyzing {self.num_rois} fish across {len(self.trials)} trials...", 
                total=total_analyses
            )
            
            for roi_id in range(self.num_rois):
                for trial in self.trials:
                    metrics = self.calculate_trial_metrics_for_roi(roi_id, trial)
                    metrics_list.append(asdict(metrics))
                    progress.advance(task)
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_list)
        
        # Add mm conversions if calibration available
        if self.pixel_to_mm:
            for col in df.columns:
                if '_px' in col and 'px_per_s' not in col:
                    mm_col = col.replace('_px', '_mm')
                    df[mm_col] = df[col] * self.pixel_to_mm
                elif 'px_per_s' in col:
                    mm_col = col.replace('px_per_s', 'mm_per_s')
                    df[mm_col] = df[col] * self.pixel_to_mm
        
        return df
    
    def print_trial_summary(self):
        """Print summary of trials found."""
        trial_types = {}
        for trial in self.trials:
            trial_types[trial['type']] = trial_types.get(trial['type'], 0) + 1
        
        table = Table(title="Trial Summary")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="yellow")
        table.add_column("Mean Duration (s)", style="green")
        
        for trial_type in trial_types:
            type_trials = [t for t in self.trials if t['type'] == trial_type]
            mean_duration = np.mean([t['duration_s'] for t in type_trials])
            table.add_row(trial_type, str(trial_types[trial_type]), f"{mean_duration:.2f}")
        
        console.print(table)
    
    def plot_stimulus_comparison(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create comparison plots across stimulus types."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        stimulus_colors = {
            'grating': '#FF6B6B',
            'black': '#4ECDC4',
            'white': '#95E77E',
            'unknown': '#FFE66D'
        }
        
        # 1. Speed by stimulus type
        ax = axes[0, 0]
        sns.boxplot(data=df, x='trial_type', y='mean_speed_px_per_s', ax=ax,
                   palette=stimulus_colors)
        ax.set_title('Mean Speed by Stimulus', fontweight='bold')
        ax.set_ylabel('Speed (px/s)')
        ax.set_xlabel('Stimulus Type')
        
        # 2. Activity ratio by stimulus
        ax = axes[0, 1]
        sns.boxplot(data=df, x='trial_type', y='activity_ratio', ax=ax,
                   palette=stimulus_colors)
        ax.set_title('Activity Level by Stimulus', fontweight='bold')
        ax.set_ylabel('Activity Ratio')
        ax.set_xlabel('Stimulus Type')
        
        # 3. Bout frequency by stimulus
        ax = axes[0, 2]
        df['bout_frequency'] = df['bout_count'] / df['duration_s']
        sns.boxplot(data=df, x='trial_type', y='bout_frequency', ax=ax,
                   palette=stimulus_colors)
        ax.set_title('Bout Frequency by Stimulus', fontweight='bold')
        ax.set_ylabel('Bouts per Second')
        ax.set_xlabel('Stimulus Type')
        
        # 4. Speed over time (all fish, colored by stimulus)
        ax = axes[1, 0]
        for trial_type in df['trial_type'].unique():
            type_df = df[df['trial_type'] == trial_type]
            ax.scatter(type_df['trial_number'], type_df['mean_speed_px_per_s'],
                      alpha=0.5, label=trial_type, color=stimulus_colors.get(trial_type, 'gray'))
        ax.set_title('Speed Across Trials', fontweight='bold')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Mean Speed (px/s)')
        ax.legend()
        
        # 5. Individual fish responses
        ax = axes[1, 1]
        fish_means = df.groupby(['roi_id', 'trial_type'])['mean_speed_px_per_s'].mean().reset_index()
        fish_pivot = fish_means.pivot(index='roi_id', columns='trial_type', values='mean_speed_px_per_s')
        fish_pivot.plot(kind='bar', ax=ax, color=[stimulus_colors.get(c, 'gray') for c in fish_pivot.columns])
        ax.set_title('Individual Fish Responses', fontweight='bold')
        ax.set_xlabel('Fish ID')
        ax.set_ylabel('Mean Speed (px/s)')
        ax.legend(title='Stimulus')
        
        # 6. Detection quality
        ax = axes[1, 2]
        sns.boxplot(data=df, x='trial_type', y='detection_rate', ax=ax,
                   palette=stimulus_colors)
        ax.set_title('Detection Quality by Stimulus', fontweight='bold')
        ax.set_ylabel('Detection Rate')
        ax.set_xlabel('Stimulus Type')
        ax.set_ylim([0, 1.05])
        
        plt.suptitle('Multi-Fish Response to Moving Grating Stimuli', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Figure saved to:[/green] {save_path}")
        
        plt.show()
    
    def plot_per_fish_distances(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create per-fish plots showing total distance traveled across trials."""
        n_fish = df['roi_id'].nunique()
        
        # Create subplots grid
        n_cols = min(4, n_fish)
        n_rows = int(np.ceil(n_fish / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_fish == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        stimulus_colors = {
            'grating': '#FF6B6B',
            'black': '#4ECDC4',
            'white': '#95E77E',
            'unknown': '#FFE66D'
        }
        
        for idx, roi_id in enumerate(sorted(df['roi_id'].unique())):
            ax = axes[idx]
            fish_df = df[df['roi_id'] == roi_id]
            
            # Plot distance by trial, colored by stimulus type
            for trial_type in fish_df['trial_type'].unique():
                type_df = fish_df[fish_df['trial_type'] == trial_type]
                ax.bar(type_df['trial_number'], type_df['total_distance_px'],
                      color=stimulus_colors.get(trial_type, 'gray'),
                      alpha=0.7, label=trial_type)
            
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('Distance (pixels)')
            ax.set_title(f'Fish {roi_id}')
            ax.grid(True, alpha=0.3)
            
            # Add legend only to first subplot
            if idx == 0:
                ax.legend(loc='upper right')
            
            # Add summary stats as text
            total_dist = fish_df['total_distance_px'].sum()
            mean_dist = fish_df['total_distance_px'].mean()
            ax.text(0.02, 0.98, f'Total: {total_dist:.0f} px\nMean: {mean_dist:.0f} px',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for idx in range(n_fish, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Distance Traveled Per Fish Across Trials', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Per-fish distance plot saved to:[/green] {save_path}")
        
        plt.show()
    
    def plot_cumulative_distances(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot cumulative distance over trials for each fish."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get unique colors for each fish
        n_fish = df['roi_id'].nunique()
        colors = plt.cm.tab20(np.linspace(0, 1, n_fish))
        
        for idx, roi_id in enumerate(sorted(df['roi_id'].unique())):
            fish_df = df[df['roi_id'] == roi_id].sort_values('trial_number')
            cumulative_distance = fish_df['total_distance_px'].cumsum()
            
            ax.plot(fish_df['trial_number'], cumulative_distance,
                marker='o', label=f'Fish {roi_id}', 
                color=colors[idx], linewidth=2, markersize=4)
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Cumulative Distance (pixels)')
        ax.set_title('Cumulative Distance Traveled Over Experiment', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        
        # Add summary statistics
        total_by_fish = df.groupby('roi_id')['total_distance_px'].sum()
        stats_text = f"Range: {total_by_fish.min():.0f} - {total_by_fish.max():.0f} px\n"
        stats_text += f"Mean: {total_by_fish.mean():.0f} px\n"
        stats_text += f"CV: {(total_by_fish.std()/total_by_fish.mean()):.2f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Cumulative distance plot saved to:[/green] {save_path}")
        
        plt.show()

    def load_roi_boundaries_from_config(self):
        """Load predefined ROI boundaries from config."""
        self.roi_boundaries_cache = {}
        
        # Hardcoded from config file - these are in 640x640 space
        sub_dish_rois = [
            {'id': 0, 'roi_pixels': [59, 73, 71, 180]},
            {'id': 1, 'roi_pixels': [139, 74, 71, 178]},
            {'id': 2, 'roi_pixels': [241, 76, 70, 180]},
            {'id': 3, 'roi_pixels': [320, 73, 70, 183]},
            {'id': 4, 'roi_pixels': [425, 76, 71, 183]},
            {'id': 5, 'roi_pixels': [503, 70, 73, 188]},
            {'id': 6, 'roi_pixels': [53, 272, 75, 183]},
            {'id': 7, 'roi_pixels': [137, 275, 71, 181]},
            {'id': 8, 'roi_pixels': [236, 271, 72, 185]},
            {'id': 9, 'roi_pixels': [317, 272, 70, 185]},
            {'id': 10, 'roi_pixels': [421, 273, 73, 185]},
            {'id': 11, 'roi_pixels': [502, 275, 70, 184]}
        ]
        
        # Convert from 640x640 to camera resolution
        scale = self.camera_width / 640
        
        for roi_info in sub_dish_rois:
            roi_id = roi_info['id']
            x, y, width, height = roi_info['roi_pixels']
            
            # Scale to camera resolution
            x_min = x * scale
            y_min = y * scale
            x_max = (x + width) * scale
            y_max = (y + height) * scale
            
            self.roi_boundaries_cache[roi_id] = {
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'y_center': (y_min + y_max) / 2,
                'width': (x_max - x_min),
                'height': (y_max - y_min)
            }

    def get_roi_boundaries(self, roi_id: int) -> Dict[str, float]:
        """Get the boundaries of a specific ROI from preloaded config."""
        if not hasattr(self, 'roi_boundaries_cache'):
            self.load_roi_boundaries_from_config()
        
        return self.roi_boundaries_cache.get(roi_id, None)
    
    def assign_roi_quadrant(self, y_position: float, roi_boundaries: Dict[str, float]) -> int:
        """
        Assign a quadrant based on y-position within the ROI.
        
        Args:
            y_position: Y coordinate in camera pixels
            roi_boundaries: ROI boundary dictionary from get_roi_boundaries
            
        Returns:
            1 for top half of ROI (quadrant 1)
            2 for bottom half of ROI (quadrant 2)
            -1 for invalid/missing position
        """
        if np.isnan(y_position) or roi_boundaries is None:
            return -1
        
        # Check if position is within ROI bounds
        if y_position < roi_boundaries['y_min'] or y_position > roi_boundaries['y_max']:
            return -1  # Outside ROI
        
        # Middle of ROI
        middle_y = roi_boundaries['y_center']
        
        if y_position < middle_y:
            return 1  # Top half of ROI
        else:
            return 2  # Bottom half of ROI

    def get_roi_positions_with_quadrants(self, roi_id: int, start_frame: int, end_frame: int) -> pd.DataFrame:
        """
        Get positions for specific ROI during frame range with ROI-based quadrant assignments.
        
        Returns:
            DataFrame with columns: frame, x, y, quadrant, relative_y
        """
        positions = self.get_roi_positions_for_frames(roi_id, start_frame, end_frame)
        
        # Get ROI boundaries
        roi_boundaries = self.get_roi_boundaries(roi_id)
        
        # Create DataFrame with frame-by-frame data
        frames = np.arange(start_frame, min(end_frame, start_frame + len(positions)))
        
        # Calculate relative position within ROI (0 = bottom, 1 = top)
        relative_y = []
        for y in positions[:, 1]:
            if np.isnan(y) or roi_boundaries is None:
                relative_y.append(np.nan)
            else:
                rel_pos = (y - roi_boundaries['y_min']) / (roi_boundaries['y_max'] - roi_boundaries['y_min'])
                relative_y.append(1 - rel_pos)  # Invert so 1 is top
        
        df = pd.DataFrame({
            'frame': frames,
            'x': positions[:, 0],
            'y': positions[:, 1],
            'quadrant': [self.assign_roi_quadrant(y, roi_boundaries) for y in positions[:, 1]],
            'relative_y': relative_y
        })
        
        return df
    
    def plot_fish_quadrant_frame_by_frame(self, roi_id: int, 
                                     window_frames: int = 300,
                                     show_orientation: bool = True,
                                     save_path: Optional[str] = None):
        """
        Plot frame-by-frame quadrant occupancy for a single fish with optional orientation display.
        
        Args:
            roi_id: Fish/ROI ID to analyze
            window_frames: Rolling window size for smoothing
            show_orientation: Whether to show grating orientations
            save_path: Path to save plot
        """
        # Check if we should show orientation data
        if show_orientation and not any('orientation_degrees' in trial for trial in self.trials):
            self.add_orientation_to_trials()
        
        # Get all positions for this fish across entire recording
        total_frames = len(self.n_detections)
        all_positions = self.get_roi_positions_for_frames(roi_id, 0, total_frames)
        
        # Get ROI boundaries
        roi_boundaries = self.get_roi_boundaries(roi_id)
        if roi_boundaries is None:
            console.print(f"[red]No ROI boundaries for fish {roi_id}[/red]")
            return
        
        # Calculate quadrant for each frame
        frames = np.arange(total_frames)
        quadrants = np.array([self.assign_roi_quadrant(y, roi_boundaries) 
                            for y in all_positions[:, 1]])
        
        # Calculate rolling averages
        in_top = (quadrants == 1).astype(float)
        in_bottom = (quadrants == 2).astype(float)
        valid_frames = (quadrants != -1).astype(float)
        
        from scipy.ndimage import uniform_filter1d
        
        if window_frames > 1:
            in_top_smooth = uniform_filter1d(in_top, size=window_frames, mode='nearest')
            in_bottom_smooth = uniform_filter1d(in_bottom, size=window_frames, mode='nearest')
            valid_smooth = uniform_filter1d(valid_frames, size=window_frames, mode='nearest')
            
            with np.errstate(divide='ignore', invalid='ignore'):
                top_proportion = np.where(valid_smooth > 0, in_top_smooth / valid_smooth, np.nan)
                bottom_proportion = np.where(valid_smooth > 0, in_bottom_smooth / valid_smooth, np.nan)
        else:
            top_proportion = in_top
            bottom_proportion = in_bottom
        
        preference_score = top_proportion - bottom_proportion
        
        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), 
                                gridspec_kw={'height_ratios': [1, 2, 2, 1]})
        
        # Convert frames to time for x-axis
        time_seconds = frames / self.fps
        time_minutes = time_seconds / 60
        
        # Plot 1: Stimulus timeline with orientation arrows for gratings
        ax = axes[0]
        stimulus_colors = {
            'grating': '#FF6B6B',
            'black': '#4ECDC4',
            'white': '#95E77E',
            'unknown': '#FFE66D'
        }
        
        for trial in self.trials:
            trial_start = trial['start_frame'] / self.fps / 60
            trial_end = trial['end_frame'] / self.fps / 60
            color = stimulus_colors.get(trial['type'], 'gray')
            ax.axvspan(trial_start, trial_end, color=color, alpha=0.7)
            
            # Add labels
            trial_mid = (trial_start + trial_end) / 2
            if show_orientation and trial['type'] == 'grating' and trial.get('orientation_degrees') is not None:
                # Show orientation as arrow
                orientation = trial['orientation_degrees']
                
                # Convert orientation to arrow direction (assuming 0° is rightward)
                angle_rad = np.deg2rad(orientation)
                dx = np.cos(angle_rad) * 0.3  # Arrow length
                dy = np.sin(angle_rad) * 0.3
                
                # Draw arrow
                ax.arrow(trial_mid, 0.5, dx * (trial_end - trial_start) * 0.3, dy * 0.4,
                        head_width=0.1, head_length=0.05, fc='black', ec='black', alpha=0.8)
                
                # Add angle text
                ax.text(trial_mid, 0.2, f"{orientation:.0f}°", ha='center', va='center',
                    fontsize=7, fontweight='bold')
            else:
                ax.text(trial_mid, 0.5, trial['type'][:4], ha='center', va='center',
                    fontsize=8, fontweight='bold')
        
        ax.set_xlim([0, time_minutes[-1]])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Stimulus')
        ax.set_title(f'Fish {roi_id} - Frame-by-Frame Quadrant Analysis (smoothing window: {window_frames} frames)', 
                    fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot 2: Stacked area showing proportions
        ax = axes[1]
        
        top_prop_masked = np.where(~np.isnan(top_proportion), top_proportion, 0)
        bottom_prop_masked = np.where(~np.isnan(bottom_proportion), bottom_proportion, 0)
        
        ax.fill_between(time_minutes, 0, top_prop_masked,
                    color='#3498db', alpha=0.7, label='Top Half of ROI')
        ax.fill_between(time_minutes, top_prop_masked, top_prop_masked + bottom_prop_masked,
                    color='#e74c3c', alpha=0.7, label='Bottom Half of ROI')
        
        missing_mask = np.isnan(top_proportion)
        if np.any(missing_mask):
            ax.fill_between(time_minutes, 0, 1, where=missing_mask,
                        color='gray', alpha=0.2, label='No detection')
        
        # Add vertical lines at grating starts with orientation annotations
        if show_orientation:
            for trial in self.trials:
                if trial['type'] == 'grating' and trial.get('orientation_degrees') is not None:
                    trial_start = trial['start_frame'] / self.fps / 60
                    ax.axvline(x=trial_start, color='black', linestyle=':', alpha=0.3, linewidth=0.5)
                    # Add small orientation label at top
                    ax.text(trial_start, 0.95, f"{trial['orientation_degrees']:.0f}°", 
                        rotation=45, ha='left', va='bottom', fontsize=6, alpha=0.7)
        
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlim([0, time_minutes[-1]])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Proportion')
        ax.set_title(f'Quadrant Occupancy (rolling {window_frames/self.fps:.1f}s window)', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add overall statistics
        valid_indices = ~np.isnan(preference_score)
        if np.any(valid_indices):
            overall_pref = np.nanmean(preference_score)
            overall_top = np.nanmean(top_proportion)
            overall_bottom = np.nanmean(bottom_proportion)
            overall_detection = np.mean(valid_frames)
            
            stats_text = f'Overall preference: {overall_pref:.2f}\n'
            stats_text += f'Top: {overall_top:.1%} | Bottom: {overall_bottom:.1%}\n'
            stats_text += f'Detection rate: {overall_detection:.1%}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 3: Preference score
        ax = axes[2]
        
        valid_pref = ~np.isnan(preference_score)
        
        ax.plot(time_minutes[valid_pref], preference_score[valid_pref], 
            'k-', linewidth=1, alpha=0.7)
        ax.fill_between(time_minutes, 0, preference_score,
                    where=(preference_score >= 0) & valid_pref,
                    color='#3498db', alpha=0.3, label='Top preference')
        ax.fill_between(time_minutes, 0, preference_score,
                    where=(preference_score < 0) & valid_pref,
                    color='#e74c3c', alpha=0.3, label='Bottom preference')
        
        # Add vertical lines for grating orientations
        if show_orientation:
            for trial in self.trials:
                if trial['type'] == 'grating' and trial.get('orientation_degrees') is not None:
                    trial_start = trial['start_frame'] / self.fps / 60
                    trial_end = trial['end_frame'] / self.fps / 60
                    
                    # Calculate mean preference during this grating
                    start_idx = int(trial['start_frame'])
                    end_idx = min(int(trial['end_frame']), len(preference_score))
                    if start_idx < end_idx:
                        trial_pref = np.nanmean(preference_score[start_idx:end_idx])
                        
                        # Add shaded region with orientation label
                        ax.axvspan(trial_start, trial_end, alpha=0.1, color='gray')
                        ax.text((trial_start + trial_end) / 2, 0.9, 
                            f"{trial['orientation_degrees']:.0f}°\n({trial_pref:+.2f})",
                            ha='center', va='top', fontsize=7,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.set_xlim([0, time_minutes[-1]])
        ax.set_ylim([-1, 1])
        ax.set_ylabel('Preference Score')
        ax.set_title('Position Preference (-1=Bottom, +1=Top)', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Detection coverage
        ax = axes[3]
        
        detection_rate = uniform_filter1d(valid_frames, size=window_frames, mode='nearest')
        ax.fill_between(time_minutes, 0, detection_rate,
                    color='green', alpha=0.3)
        ax.plot(time_minutes, detection_rate, 'g-', linewidth=1)
        
        ax.set_xlim([0, time_minutes[-1]])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Detection Rate')
        ax.set_title('Tracking Coverage', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Frame-by-frame plot saved to {save_path}[/green]")
        
        plt.show()

    def extract_grating_orientations(self):
        """
        Extract grating orientation information from protocol definition.
        
        Returns:
            dict: Dictionary mapping trial numbers to grating orientations
        """
        grating_info = {}
        
        with h5py.File(self.h5_path, 'r') as f:
            # Check for protocol snapshot
            if '/protocol_snapshot' not in f:
                console.print("[yellow]Warning: No protocol_snapshot found in H5 file[/yellow]")
                return grating_info
            
            protocol_group = f['/protocol_snapshot']
            
            # Check for protocol definition JSON
            if 'protocol_definition_json' not in protocol_group:
                console.print("[yellow]Warning: No protocol_definition_json found[/yellow]")
                return grating_info
            
            # Load and parse the protocol JSON
            protocol_json_data = protocol_group['protocol_definition_json'][()]
            
            # Handle different data types
            if isinstance(protocol_json_data, bytes):
                protocol_json_str = protocol_json_data.decode('utf-8')
            elif isinstance(protocol_json_data, np.ndarray):
                protocol_json_str = protocol_json_data.tobytes().decode('utf-8')
            else:
                protocol_json_str = str(protocol_json_data)
            
            try:
                protocol_data = json.loads(protocol_json_str)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error parsing protocol JSON: {e}[/red]")
                return grating_info
            
            # Extract protocol name and steps
            protocol_name = protocol_data.get('protocol_name', 'Unknown')
            console.print(f"[cyan]Protocol: {protocol_name}[/cyan]")
            
            # Look through steps for grating information
            steps = protocol_data.get('steps', [])
            
            for step_idx, step in enumerate(steps):
                stimulus_mode = step.get('stimulus_mode_str', '')
                step_name = step.get('name', f'Step {step_idx}')
                
                # Check if this is a moving grating step
                if stimulus_mode == 'MOVING_GRATING' or 'grating' in stimulus_mode.lower():
                    parameters = step.get('parameters', {})
                    
                    # Extract grating parameters
                    orientation = parameters.get('orientation_degrees')
                    if orientation is None:
                        # Try alternative field names
                        orientation = parameters.get('angle_degrees')
                        if orientation is None:
                            orientation = parameters.get('grating_orientation')
                    
                    # Extract other useful parameters
                    speed = parameters.get('speed_degrees_per_second')
                    if speed is None:
                        speed = parameters.get('velocity_degrees_per_second')
                    
                    spatial_freq = parameters.get('spatial_frequency_cycles_per_degree')
                    if spatial_freq is None:
                        spatial_freq = parameters.get('spatial_freq')
                    
                    duration = parameters.get('duration_seconds')
                    if duration is None:
                        duration = step.get('duration_seconds')
                    
                    grating_info[step_idx] = {
                        'name': step_name,
                        'orientation_degrees': orientation,
                        'speed_degrees_per_second': speed,
                        'spatial_frequency': spatial_freq,
                        'duration_seconds': duration,
                        'stimulus_mode': stimulus_mode
                    }
                    
                    console.print(f"  Step {step_idx} ({step_name}): Orientation={orientation}°, Speed={speed}°/s")
                
                # Also check for solid colors (black/white)
                elif stimulus_mode in ['SOLID_BLACK', 'SOLID_WHITE', 'SOLID_COLOR']:
                    grating_info[step_idx] = {
                        'name': step_name,
                        'stimulus_mode': stimulus_mode,
                        'duration_seconds': step.get('duration_seconds')
                    }
                    console.print(f"  Step {step_idx} ({step_name}): {stimulus_mode}")
        
        return grating_info
    
    def get_minute_by_minute_proportions(self, bin_minutes=1.0) -> pd.DataFrame:
        """
        Calculate minute-by-minute top proportions for all fish.
        """
        all_data = []
        
        for roi_id in range(self.num_rois):
            # Get all positions for this fish
            total_frames = len(self.n_detections)
            all_positions = self.get_roi_positions_for_frames(roi_id, 0, total_frames)
            roi_boundaries = self.get_roi_boundaries(roi_id)
            
            if roi_boundaries is None:
                continue
            
            # Calculate quadrants
            quadrants = np.array([self.assign_roi_quadrant(y, roi_boundaries) 
                                for y in all_positions[:, 1]])
            
            # Convert frames to minutes
            minutes = np.arange(total_frames) / (self.fps * 60)
            
            # Create bins
            max_time = minutes[-1]
            bins = np.arange(0, max_time + bin_minutes, bin_minutes)
            
            # Calculate proportion for each bin
            for i in range(len(bins) - 1):
                bin_start = bins[i]
                bin_end = bins[i + 1]
                bin_mask = (minutes >= bin_start) & (minutes < bin_end)
                
                bin_quadrants = quadrants[bin_mask]
                valid_mask = bin_quadrants != -1
                
                if np.any(valid_mask):
                    top_count = np.sum(bin_quadrants[valid_mask] == 1)
                    total_valid = np.sum(valid_mask)
                    top_proportion = top_count / total_valid if total_valid > 0 else np.nan
                    
                    all_data.append({
                        'roi_id': roi_id,
                        'minute': (bin_start + bin_end) / 2,
                        'top_proportion': top_proportion,
                        'detection_rate': np.sum(valid_mask) / len(bin_quadrants)
                    })
        
        return pd.DataFrame(all_data)


    def add_orientation_to_trials(self):
        """
        Add grating orientation information to existing trials.
        Updates self.trials with orientation data.
        """
        # Extract orientation info from protocol
        grating_info = self.extract_grating_orientations()
        
        if not grating_info:
            console.print("[yellow]No grating orientation data found in protocol[/yellow]")
            return
        
        # Map step indices to trials
        # Note: This assumes trials are created from steps in order
        for trial in self.trials:
            # Try to match trial to step based on timing or order
            # This is a simple approach - you may need to adjust based on your protocol structure
            trial_idx = trial['number'] - 1  # Assuming 1-indexed trial numbers
            
            # Look for matching step
            if trial_idx in grating_info:
                step_info = grating_info[trial_idx]
                trial['orientation_degrees'] = step_info.get('orientation_degrees')
                trial['grating_speed'] = step_info.get('speed_degrees_per_second')
                trial['spatial_frequency'] = step_info.get('spatial_frequency')
            else:
                trial['orientation_degrees'] = None
                trial['grating_speed'] = None
                trial['spatial_frequency'] = None
        
        # Print summary
        console.print("\n[bold]Trial Orientations:[/bold]")
        for trial in self.trials:
            if trial.get('orientation_degrees') is not None:
                console.print(f"  Trial {trial['number']} ({trial['type']}): {trial['orientation_degrees']}°")
            else:
                console.print(f"  Trial {trial['number']} ({trial['type']}): No orientation data")

    def plot_fish_quadrant_frame_by_frame_with_orientation(self, roi_id: int, 
                                                        window_frames: int = 300,
                                                        save_path: Optional[str] = None):
        """
        Plot frame-by-frame quadrant occupancy with grating orientation information.
        
        Args:
            roi_id: Fish/ROI ID to analyze
            window_frames: Rolling window size for smoothing
            save_path: Path to save plot
        """
        # First ensure we have orientation data
        if not any('orientation_degrees' in trial for trial in self.trials):
            self.add_orientation_to_trials()
        
        # Get all positions for this fish
        total_frames = len(self.n_detections)
        all_positions = self.get_roi_positions_for_frames(roi_id, 0, total_frames)
        
        # Get ROI boundaries
        roi_boundaries = self.get_roi_boundaries(roi_id)
        if roi_boundaries is None:
            console.print(f"[red]No ROI boundaries for fish {roi_id}[/red]")
            return
        
        # Calculate quadrant for each frame
        frames = np.arange(total_frames)
        quadrants = np.array([self.assign_roi_quadrant(y, roi_boundaries) 
                            for y in all_positions[:, 1]])
        
        # Create orientation mapping for each frame
        orientation_per_frame = np.full(total_frames, np.nan)
        stimulus_type_per_frame = ['none'] * total_frames
        
        for trial in self.trials:
            start_f = trial['start_frame']
            end_f = min(trial['end_frame'], total_frames)
            
            if trial['type'] == 'grating' and trial.get('orientation_degrees') is not None:
                orientation_per_frame[start_f:end_f] = trial['orientation_degrees']
            
            stimulus_type_per_frame[start_f:end_f] = [trial['type']] * (end_f - start_f)
        
        # Calculate rolling averages
        in_top = (quadrants == 1).astype(float)
        in_bottom = (quadrants == 2).astype(float)
        valid_frames = (quadrants != -1).astype(float)
        
        from scipy.ndimage import uniform_filter1d
        
        if window_frames > 1:
            in_top_smooth = uniform_filter1d(in_top, size=window_frames, mode='nearest')
            in_bottom_smooth = uniform_filter1d(in_bottom, size=window_frames, mode='nearest')
            valid_smooth = uniform_filter1d(valid_frames, size=window_frames, mode='nearest')
            
            with np.errstate(divide='ignore', invalid='ignore'):
                top_proportion = np.where(valid_smooth > 0, in_top_smooth / valid_smooth, np.nan)
                bottom_proportion = np.where(valid_smooth > 0, in_bottom_smooth / valid_smooth, np.nan)
        else:
            top_proportion = in_top
            bottom_proportion = in_bottom
        
        preference_score = top_proportion - bottom_proportion
        
        # Create figure with 5 subplots
        fig, axes = plt.subplots(5, 1, figsize=(16, 14), 
                                gridspec_kw={'height_ratios': [1, 1, 2, 2, 1]})
        
        time_minutes = frames / self.fps / 60
        
        # Plot 1: Stimulus timeline with orientation labels
        ax = axes[0]
        stimulus_colors = {
            'grating': '#FF6B6B',
            'black': '#4ECDC4',
            'white': '#95E77E',
            'unknown': '#FFE66D'
        }
        
        for trial in self.trials:
            trial_start = trial['start_frame'] / self.fps / 60
            trial_end = trial['end_frame'] / self.fps / 60
            color = stimulus_colors.get(trial['type'], 'gray')
            ax.axvspan(trial_start, trial_end, color=color, alpha=0.7)
            
            # Add orientation labels for gratings
            trial_mid = (trial_start + trial_end) / 2
            if trial['type'] == 'grating' and trial.get('orientation_degrees') is not None:
                label = f"{trial['orientation_degrees']:.0f}°"
            else:
                label = trial['type'][:4]
            ax.text(trial_mid, 0.5, label, ha='center', va='center',
                fontsize=8, fontweight='bold')
        
        ax.set_xlim([0, time_minutes[-1]])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Stimulus')
        ax.set_title(f'Fish {roi_id} - Quadrant Analysis with Grating Orientations', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot 2: Orientation over time (continuous)
        ax = axes[1]
        
        # Get unique orientations and assign colors
        unique_orientations = [o for o in np.unique(orientation_per_frame) if not np.isnan(o)]
        if unique_orientations:
            orientation_colors = plt.cm.hsv(np.linspace(0, 1, len(unique_orientations) + 1)[:-1])
            orient_color_map = dict(zip(unique_orientations, orientation_colors))
            
            # Plot orientation as colored segments
            for orientation in unique_orientations:
                mask = orientation_per_frame == orientation
                ax.fill_between(time_minutes, 0, 1, where=mask,
                            color=orient_color_map[orientation], alpha=0.5,
                            label=f'{orientation:.0f}°')
            
            ax.set_ylabel('Grating\nOrientation')
            ax.set_ylim([0, 1])
            ax.set_xlim([0, time_minutes[-1]])
            ax.legend(loc='upper right', ncol=len(unique_orientations), fontsize=8)
            ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, 'No orientation data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_ylim([0, 1])
            ax.set_xlim([0, time_minutes[-1]])
        
        # Plot 3: Quadrant occupancy
        ax = axes[2]
        
        top_prop_masked = np.where(~np.isnan(top_proportion), top_proportion, 0)
        bottom_prop_masked = np.where(~np.isnan(bottom_proportion), bottom_proportion, 0)
        
        ax.fill_between(time_minutes, 0, top_prop_masked,
                    color='#3498db', alpha=0.7, label='Top Half of ROI')
        ax.fill_between(time_minutes, top_prop_masked, top_prop_masked + bottom_prop_masked,
                    color='#e74c3c', alpha=0.7, label='Bottom Half of ROI')
        
        missing_mask = np.isnan(top_proportion)
        if np.any(missing_mask):
            ax.fill_between(time_minutes, 0, 1, where=missing_mask,
                        color='gray', alpha=0.2, label='No detection')
        
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlim([0, time_minutes[-1]])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Proportion')
        ax.set_title(f'Quadrant Occupancy (rolling {window_frames/self.fps:.1f}s window)', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Preference score with orientation highlighting
        ax = axes[3]
        
        valid_pref = ~np.isnan(preference_score)
        
        # Plot base preference line
        ax.plot(time_minutes[valid_pref], preference_score[valid_pref], 
            'k-', linewidth=0.5, alpha=0.5)
        
        # Overlay colored regions for each orientation
        if unique_orientations:
            for orientation in unique_orientations:
                mask = (orientation_per_frame == orientation) & valid_pref
                if np.any(mask):
                    ax.scatter(time_minutes[mask], preference_score[mask],
                            c=[orient_color_map[orientation]], s=1, alpha=0.6,
                            label=f'{orientation:.0f}°')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.set_xlim([0, time_minutes[-1]])
        ax.set_ylim([-1, 1])
        ax.set_ylabel('Preference Score')
        ax.set_title('Position Preference by Orientation', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Detection coverage
        ax = axes[4]
        
        detection_rate = uniform_filter1d(valid_frames, size=window_frames, mode='nearest')
        ax.fill_between(time_minutes, 0, detection_rate, color='green', alpha=0.3)
        ax.plot(time_minutes, detection_rate, 'g-', linewidth=1)
        
        ax.set_xlim([0, time_minutes[-1]])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Detection Rate')
        ax.set_title('Tracking Coverage', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add orientation-specific statistics
        orientation_stats = []
        for orientation in unique_orientations:
            mask = (orientation_per_frame == orientation) & valid_pref
            if np.any(mask):
                mean_pref = np.mean(preference_score[mask])
                orientation_stats.append(f"{orientation:.0f}°: {mean_pref:+.2f}")
        
        if orientation_stats:
            stats_text = "Mean preference by orientation:\n" + " | ".join(orientation_stats)
            axes[2].text(0.02, 0.98, stats_text, transform=axes[2].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Orientation frame analysis saved to {save_path}[/green]")
        
        plt.show()


    def plot_orientation_preference_summary(self, window_frames: int = 300,
                                        save_path: Optional[str] = None):
        """
        Create summary plot showing preference scores for each orientation across all fish.
        """
        if not any('orientation_degrees' in trial for trial in self.trials):
            self.add_orientation_to_trials()
        
        # Collect data for all fish and orientations
        orientation_preferences = []
        
        for roi_id in range(self.num_rois):
            # Get positions and quadrants
            total_frames = len(self.n_detections)
            all_positions = self.get_roi_positions_for_frames(roi_id, 0, total_frames)
            roi_boundaries = self.get_roi_boundaries(roi_id)
            
            if roi_boundaries is None:
                continue
            
            quadrants = np.array([self.assign_roi_quadrant(y, roi_boundaries) 
                                for y in all_positions[:, 1]])
            
            # Calculate preferences for each trial
            for trial in self.trials:
                if trial['type'] == 'grating' and trial.get('orientation_degrees') is not None:
                    start_f = trial['start_frame']
                    end_f = min(trial['end_frame'], total_frames)
                    
                    trial_quadrants = quadrants[start_f:end_f]
                    valid_mask = trial_quadrants != -1
                    
                    if np.any(valid_mask):
                        top_count = np.sum(trial_quadrants[valid_mask] == 1)
                        bottom_count = np.sum(trial_quadrants[valid_mask] == 2)
                        total_valid = np.sum(valid_mask)
                        
                        if total_valid > 0:
                            preference = (top_count - bottom_count) / total_valid
                            
                            orientation_preferences.append({
                                'roi_id': roi_id,
                                'orientation': trial['orientation_degrees'],
                                'preference_score': preference,
                                'trial_number': trial['number']
                            })
        
        if not orientation_preferences:
            console.print("[yellow]No orientation preference data available[/yellow]")
            return
        
        pref_df = pd.DataFrame(orientation_preferences)
        
        # Create summary plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Average preference by orientation
        ax = axes[0]
        orientation_means = pref_df.groupby('orientation')['preference_score'].agg(['mean', 'std', 'sem']).reset_index()
        
        ax.bar(orientation_means['orientation'], orientation_means['mean'],
            yerr=orientation_means['sem'], capsize=5, alpha=0.7, color='steelblue')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Grating Orientation (degrees)')
        ax.set_ylabel('Mean Preference Score')
        ax.set_title('Population Preference by Orientation', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Individual fish preferences
        ax = axes[1]
        pivot_data = pref_df.pivot_table(values='preference_score', 
                                        index='roi_id', 
                                        columns='orientation', 
                                        aggfunc='mean')
        
        im = ax.imshow(pivot_data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xlabel('Grating Orientation (degrees)')
        ax.set_ylabel('Fish ID')
        ax.set_title('Individual Preferences by Orientation', fontweight='bold')
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels([f'{o:.0f}°' for o in pivot_data.columns])
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index)
        
        plt.colorbar(im, ax=ax, label='Preference Score')
        
        plt.suptitle('Quadrant Preference Analysis by Grating Orientation', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Orientation preference summary saved to {save_path}[/green]")
        
        plt.show()
        
        return pref_df

    def plot_orientation_response(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot fish responses grouped by grating orientation.
        
        Args:
            df: DataFrame with analysis results
            save_path: Optional path to save plot
        """
        # First add orientation data to trials if not done
        if not any('orientation_degrees' in trial for trial in self.trials):
            self.add_orientation_to_trials()
        
        # Add orientation to dataframe
        orientation_map = {trial['number']: trial.get('orientation_degrees') 
                        for trial in self.trials}
        df['orientation_degrees'] = df['trial_number'].map(orientation_map)
        
        # Filter to only grating trials with orientation data
        grating_df = df[(df['trial_type'] == 'grating') & (df['orientation_degrees'].notna())]
        
        if grating_df.empty:
            console.print("[yellow]No grating trials with orientation data found[/yellow]")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Get unique orientations and assign colors
        unique_orientations = sorted(grating_df['orientation_degrees'].unique())
        colors = plt.cm.hsv(np.linspace(0, 1, len(unique_orientations) + 1)[:-1])
        orientation_colors = dict(zip(unique_orientations, colors))
        
        # 1. Speed by orientation
        ax = axes[0, 0]
        for orientation in unique_orientations:
            orient_data = grating_df[grating_df['orientation_degrees'] == orientation]
            ax.boxplot([orient_data['mean_speed_px_per_s']], 
                    positions=[orientation],
                    widths=15,
                    patch_artist=True,
                    boxprops=dict(facecolor=orientation_colors[orientation]))
        ax.set_xlabel('Grating Orientation (degrees)')
        ax.set_ylabel('Mean Speed (px/s)')
        ax.set_title('Speed Response by Grating Orientation', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. Polar plot of responses
        ax = plt.subplot(2, 2, 2, projection='polar')
        for orientation in unique_orientations:
            orient_data = grating_df[grating_df['orientation_degrees'] == orientation]
            mean_speed = orient_data['mean_speed_px_per_s'].mean()
            std_speed = orient_data['mean_speed_px_per_s'].std()
            
            # Convert to radians
            angle_rad = np.deg2rad(orientation)
            
            # Plot mean as bar
            ax.bar(angle_rad, mean_speed, width=np.deg2rad(15),
                color=orientation_colors[orientation], alpha=0.7,
                edgecolor='black', linewidth=1)
            
            # Add error bar
            ax.errorbar(angle_rad, mean_speed, yerr=std_speed,
                    color='black', capsize=3)
        
        ax.set_theta_zero_location('E')  # 0° at right
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_title('Directional Response (Polar)', fontweight='bold', pad=20)
        
        # 3. Individual fish responses by orientation
        ax = axes[1, 0]
        fish_orientation_means = grating_df.groupby(['roi_id', 'orientation_degrees'])['mean_speed_px_per_s'].mean().reset_index()
        
        for roi_id in sorted(fish_orientation_means['roi_id'].unique()):
            fish_data = fish_orientation_means[fish_orientation_means['roi_id'] == roi_id]
            ax.plot(fish_data['orientation_degrees'], fish_data['mean_speed_px_per_s'],
                marker='o', alpha=0.5, label=f'Fish {roi_id}')
        
        ax.set_xlabel('Grating Orientation (degrees)')
        ax.set_ylabel('Mean Speed (px/s)')
        ax.set_title('Individual Fish Orientation Tuning', fontweight='bold')
        ax.grid(True, alpha=0.3)
        if len(fish_orientation_means['roi_id'].unique()) <= 6:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Orientation selectivity index
        ax = axes[1, 1]
        
        # Calculate selectivity index for each fish
        selectivity_scores = []
        for roi_id in grating_df['roi_id'].unique():
            fish_orient = grating_df[grating_df['roi_id'] == roi_id].groupby('orientation_degrees')['mean_speed_px_per_s'].mean()
            if len(fish_orient) > 1:
                # Selectivity = (max - min) / (max + min)
                selectivity = (fish_orient.max() - fish_orient.min()) / (fish_orient.max() + fish_orient.min())
                selectivity_scores.append({'roi_id': roi_id, 'selectivity': selectivity,
                                        'preferred_orientation': fish_orient.idxmax()})
        
        if selectivity_scores:
            selectivity_df = pd.DataFrame(selectivity_scores)
            selectivity_df = selectivity_df.sort_values('selectivity')
            
            bars = ax.barh(range(len(selectivity_df)), selectivity_df['selectivity'],
                        color='steelblue', alpha=0.7)
            ax.set_yticks(range(len(selectivity_df)))
            ax.set_yticklabels([f"Fish {int(roi)}" for roi in selectivity_df['roi_id']])
            ax.set_xlabel('Orientation Selectivity Index')
            ax.set_title('Orientation Selectivity by Fish', fontweight='bold')
            ax.set_xlim([0, 1])
            
            # Add preferred orientation as text
            for i, (idx, row) in enumerate(selectivity_df.iterrows()):
                ax.text(row['selectivity'] + 0.02, i, f"{row['preferred_orientation']:.0f}°",
                    va='center', fontsize=8)
        
        plt.suptitle('Grating Orientation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Orientation analysis saved to {save_path}[/green]")
        
        plt.show()


    def plot_all_fish_frame_by_frame(self, window_frames: int = 300,
                                    show_orientation: bool = True,
                                    save_dir: Optional[str] = None):
        """Generate frame-by-frame quadrant plots for all fish."""
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        for roi_id in range(self.num_rois):
            console.print(f"[cyan]Plotting Fish {roi_id}...[/cyan]")
            
            if save_dir:
                save_file = save_path / f'fish_{roi_id}_frame_analysis.png'
            else:
                save_file = None
                
            self.plot_fish_quadrant_frame_by_frame(roi_id, 
                                                window_frames=window_frames,
                                                show_orientation=show_orientation,
                                                save_path=save_file)

    def calculate_trial_metrics_with_quadrants(self, roi_id: int, trial: Dict) -> Dict:
        """
        Calculate metrics for one ROI during one trial, including ROI-based quadrant occupancy.
        """
        # Get original metrics
        base_metrics = self.calculate_trial_metrics_for_roi(roi_id, trial)
        
        # Get positions with quadrants
        positions_df = self.get_roi_positions_with_quadrants(
            roi_id, trial['start_frame'], trial['end_frame']
        )
        
        # Calculate quadrant-specific metrics
        valid_positions = positions_df[positions_df['quadrant'] != -1]
        
        if len(valid_positions) > 0:
            # Time spent in each quadrant of the ROI
            quadrant_counts = valid_positions['quadrant'].value_counts()
            frames_in_q1 = quadrant_counts.get(1, 0)
            frames_in_q2 = quadrant_counts.get(2, 0)
            total_valid_frames = len(valid_positions)
            
            # Proportion of time in each quadrant
            q1_proportion = frames_in_q1 / total_valid_frames if total_valid_frames > 0 else 0
            q2_proportion = frames_in_q2 / total_valid_frames if total_valid_frames > 0 else 0
            
            # Transitions between quadrants
            transitions = (valid_positions['quadrant'].diff() != 0).sum() - 1  # -1 for the first NaN
            transitions = max(0, transitions)
            
            # Preference score (-1 = always bottom of ROI, 0 = equal, +1 = always top of ROI)
            preference_score = q1_proportion - q2_proportion
            
            # Average relative vertical position (0 = bottom, 1 = top)
            avg_relative_y = valid_positions['relative_y'].mean()
            
        else:
            frames_in_q1 = frames_in_q2 = 0
            q1_proportion = q2_proportion = 0
            transitions = 0
            preference_score = 0
            avg_relative_y = np.nan
        
        # Add quadrant metrics to base metrics
        metrics_dict = asdict(base_metrics)
        metrics_dict.update({
            'frames_in_roi_q1': frames_in_q1,
            'frames_in_roi_q2': frames_in_q2,
            'roi_q1_proportion': q1_proportion,
            'roi_q2_proportion': q2_proportion,
            'roi_quadrant_transitions': transitions,
            'roi_quadrant_preference_score': preference_score,
            'avg_relative_y_position': avg_relative_y
        })
        
        return metrics_dict
    
    def get_trial_top_proportions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract just the top quadrant proportion for each trial.
        
        Returns:
            DataFrame with trial info and top proportion (0.51 = 51% in top)
        """
        if 'roi_q1_proportion' not in df.columns:
            console.print("[red]Error: Quadrant data not available. Run with --with-quadrants flag.[/red]")
            return None
        
        # Create simplified dataframe
        trial_proportions = df[['roi_id', 'trial_number', 'trial_type', 
                            'roi_q1_proportion', 'start_frame']].copy()
        
        # Rename for clarity
        trial_proportions.rename(columns={'roi_q1_proportion': 'top_proportion'}, inplace=True)
        
        # Add time in minutes
        trial_proportions['time_minutes'] = trial_proportions['start_frame'] / (self.fps * 60)
        
        # Add orientation if available
        if hasattr(self, 'trials') and any('orientation_degrees' in t for t in self.trials):
            orientation_map = {t['number']: t.get('orientation_degrees') for t in self.trials}
            trial_proportions['orientation'] = trial_proportions['trial_number'].map(orientation_map)
        
        return trial_proportions
        
    def plot_individual_fish_quadrant_timeseries(self, df: pd.DataFrame, 
                                                bin_minutes: float = 0.5,
                                                save_dir: Optional[str] = None):
        """
        Create individual plots for each fish showing quadrant occupancy over time.
        
        Args:
            df: DataFrame with quadrant analysis results
            bin_minutes: Time bin size for aggregation (default 0.5 minutes)
            save_dir: Directory to save individual plots
        """
        if 'roi_quadrant_preference_score' not in df.columns:
            console.print("[red]Error: Quadrant data not available. Run with --with-quadrants flag.[/red]")
            return
        
        # Create save directory if specified
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Process each fish
        for roi_id in sorted(df['roi_id'].unique()):
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), 
                                    gridspec_kw={'height_ratios': [1, 2, 2]})
            
            fish_df = df[df['roi_id'] == roi_id].sort_values('start_frame')
            
            # Convert to time in minutes
            fish_df['time_minutes'] = fish_df['start_frame'] / (self.fps * 60)
            
            # Create time bins
            max_time = fish_df['time_minutes'].max()
            time_bins = np.arange(0, max_time + bin_minutes, bin_minutes)
            
            # Aggregate data by time bins
            binned_data = []
            for i in range(len(time_bins) - 1):
                bin_start = time_bins[i]
                bin_end = time_bins[i + 1]
                bin_trials = fish_df[(fish_df['time_minutes'] >= bin_start) & 
                                    (fish_df['time_minutes'] < bin_end)]
                
                if len(bin_trials) > 0:
                    binned_data.append({
                        'time': (bin_start + bin_end) / 2,
                        'q1_prop': bin_trials['roi_q1_proportion'].mean(),
                        'q2_prop': bin_trials['roi_q2_proportion'].mean(),
                        'preference': bin_trials['roi_quadrant_preference_score'].mean(),
                        'trial_type': bin_trials['trial_type'].mode()[0] if len(bin_trials) > 0 else 'unknown'
                    })
            
            binned_df = pd.DataFrame(binned_data)
            
            if len(binned_df) == 0:
                console.print(f"[yellow]No data for Fish {roi_id}[/yellow]")
                continue
            
            # Plot 1: Stimulus timeline
            ax = axes[0]
            stimulus_colors = {
                'grating': '#FF6B6B',
                'black': '#4ECDC4',
                'white': '#95E77E',
                'unknown': '#FFE66D'
            }
            
            for trial in self.trials:
                trial_start_min = trial['start_frame'] / (self.fps * 60)
                trial_end_min = trial['end_frame'] / (self.fps * 60)
                color = stimulus_colors.get(trial['type'], 'gray')
                ax.axvspan(trial_start_min, trial_end_min, color=color, alpha=0.7)
                
                # Add text labels
                trial_mid = (trial_start_min + trial_end_min) / 2
                if trial_mid <= max_time:
                    ax.text(trial_mid, 0.5, trial['type'][:4], ha='center', va='center',
                        fontsize=8, fontweight='bold')
            
            ax.set_xlim([0, max_time])
            ax.set_ylim([0, 1])
            ax.set_ylabel('Stimulus')
            ax.set_title(f'Fish {roi_id} - Quadrant Occupancy Over Time', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Plot 2: Stacked area chart of quadrant proportions
            ax = axes[1]
            ax.fill_between(binned_df['time'], 0, binned_df['q1_prop'],
                        color='#3498db', alpha=0.7, label='Top Half of ROI')
            ax.fill_between(binned_df['time'], binned_df['q1_prop'], 
                        binned_df['q1_prop'] + binned_df['q2_prop'],
                        color='#e74c3c', alpha=0.7, label='Bottom Half of ROI')
            
            ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_xlim([0, max_time])
            ax.set_ylim([0, 1])
            ax.set_ylabel('Proportion of Time')
            ax.set_title('Quadrant Occupancy Ratio', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Preference score over time
            ax = axes[2]
            ax.plot(binned_df['time'], binned_df['preference'], 
                'k-', linewidth=2, alpha=0.7)
            ax.fill_between(binned_df['time'], 0, binned_df['preference'],
                        where=(binned_df['preference'] >= 0),
                        color='#3498db', alpha=0.3, label='Top preference')
            ax.fill_between(binned_df['time'], 0, binned_df['preference'],
                        where=(binned_df['preference'] < 0),
                        color='#e74c3c', alpha=0.3, label='Bottom preference')
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            ax.set_xlim([0, max_time])
            ax.set_ylim([-1, 1])
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Preference Score')
            ax.set_title('Position Preference (-1=Bottom, +1=Top)', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add summary statistics
            overall_pref = fish_df['roi_quadrant_preference_score'].mean()
            q1_mean = fish_df['roi_q1_proportion'].mean()
            q2_mean = fish_df['roi_q2_proportion'].mean()
            
            stats_text = f'Overall preference: {overall_pref:.2f}\n'
            stats_text += f'Top half: {q1_mean:.1%}\n'
            stats_text += f'Bottom half: {q2_mean:.1%}'
            
            axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save if requested
            if save_dir:
                save_file = save_path / f'fish_{roi_id}_quadrant_timeseries.png'
                plt.savefig(save_file, dpi=150, bbox_inches='tight')
                console.print(f"[green]✓ Saved plot for Fish {roi_id} to {save_file}[/green]")
            
            plt.show()

    def save_quadrant_metrics_csv(self, df: pd.DataFrame, output_path: str):
        """
        Save detailed quadrant metrics for each fish to CSV files.
        
        Args:
            df: DataFrame with quadrant analysis
            output_path: Base path for saving CSV files
        """
        if 'roi_quadrant_preference_score' not in df.columns:
            console.print("[red]Error: Quadrant data not available.[/red]")
            return
        
        output_dir = Path(output_path).parent
        base_name = Path(output_path).stem
        
        # Save overall summary
        summary_data = []
        for roi_id in sorted(df['roi_id'].unique()):
            fish_df = df[df['roi_id'] == roi_id]
            summary_data.append({
                'fish_id': roi_id,
                'mean_preference_score': fish_df['roi_quadrant_preference_score'].mean(),
                'std_preference_score': fish_df['roi_quadrant_preference_score'].std(),
                'mean_q1_proportion': fish_df['roi_q1_proportion'].mean(),
                'mean_q2_proportion': fish_df['roi_q2_proportion'].mean(),
                'total_transitions': fish_df['roi_quadrant_transitions'].sum(),
                'mean_transitions_per_trial': fish_df['roi_quadrant_transitions'].mean()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / f'{base_name}_quadrant_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        console.print(f"[green]✓ Quadrant summary saved to {summary_path}[/green]")
        
        # Save detailed metrics with time
        detailed_df = df[['roi_id', 'trial_number', 'trial_type', 'start_frame',
                        'roi_q1_proportion', 'roi_q2_proportion', 
                        'roi_quadrant_preference_score', 'roi_quadrant_transitions']].copy()
        detailed_df['time_minutes'] = detailed_df['start_frame'] / (self.fps * 60)
        detailed_df = detailed_df.sort_values(['roi_id', 'start_frame'])
        
        detailed_path = output_dir / f'{base_name}_quadrant_detailed.csv'
        detailed_df.to_csv(detailed_path, index=False)
        console.print(f"[green]✓ Detailed quadrant data saved to {detailed_path}[/green]")

    def analyze_all_trials_with_quadrants(self) -> pd.DataFrame:
        """Analyze all ROIs across all trials including ROI-based quadrant analysis."""
        metrics_list = []

        # Load ROI boundaries from config (instant!)
        console.print("[cyan]Loading ROI boundaries from config...[/cyan]")
        self.load_roi_boundaries_from_config()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            total_analyses = self.num_rois * len(self.trials)
            task = progress.add_task(
                f"[cyan]Analyzing {self.num_rois} fish across {len(self.trials)} trials with ROI quadrants...", 
                total=total_analyses
            )
            
            for roi_id in range(self.num_rois):
                for trial in self.trials:
                    metrics = self.calculate_trial_metrics_with_quadrants(roi_id, trial)
                    metrics_list.append(metrics)
                    progress.advance(task)
                
        # Convert to DataFrame
        df = pd.DataFrame(metrics_list)
        
        # Add mm conversions if calibration available
        if self.pixel_to_mm:
            for col in df.columns:
                if '_px' in col and 'px_per_s' not in col:
                    mm_col = col.replace('_px', '_mm')
                    df[mm_col] = df[col] * self.pixel_to_mm
                elif 'px_per_s' in col:
                    mm_col = col.replace('px_per_s', 'mm_per_s')
                    df[mm_col] = df[col] * self.pixel_to_mm
        
        return df

    def plot_quadrant_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create plots analyzing ROI-based quadrant preferences and behaviors."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        stimulus_colors = {
            'grating': '#FF6B6B',
            'black': '#4ECDC4',
            'white': '#95E77E',
            'unknown': '#FFE66D'
        }
        
        # 1. Overall quadrant preference by stimulus
        ax = axes[0, 0]
        quadrant_data = df.groupby('trial_type')[['roi_q1_proportion', 'roi_q2_proportion']].mean()
        quadrant_data.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
        ax.set_title('Average ROI Quadrant Occupancy by Stimulus', fontweight='bold')
        ax.set_ylabel('Proportion of Time')
        ax.set_xlabel('Stimulus Type')
        ax.legend(['Top Half of ROI', 'Bottom Half of ROI'])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim([0, 1])
        
        # 2. Individual fish preferences within their ROIs
        ax = axes[0, 1]
        fish_prefs = df.groupby('roi_id')['roi_quadrant_preference_score'].mean().sort_values()
        colors_fish = ['#e74c3c' if x < 0 else '#3498db' for x in fish_prefs.values]
        fish_prefs.plot(kind='barh', ax=ax, color=colors_fish)
        ax.set_title('Fish Preferences Within Their ROIs', fontweight='bold')
        ax.set_xlabel('Preference Score (-1=Bottom, +1=Top of ROI)')
        ax.set_ylabel('Fish ID')
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        # 3. Transitions within ROI
        ax = axes[0, 2]
        transitions_by_stim = df.groupby('trial_type')['roi_quadrant_transitions'].mean()
        transitions_by_stim.plot(kind='bar', ax=ax, color=[stimulus_colors.get(x, 'gray') for x in transitions_by_stim.index])
        ax.set_title('Average ROI Quadrant Transitions by Stimulus', fontweight='bold')
        ax.set_ylabel('Transitions per Trial')
        ax.set_xlabel('Stimulus Type')
        
        # 4. Relative vertical position by stimulus
        ax = axes[1, 0]
        sns.boxplot(data=df[df['avg_relative_y_position'].notna()], 
                    x='trial_type', y='avg_relative_y_position', ax=ax,
                    palette=stimulus_colors)
        ax.set_title('Vertical Position Within ROI by Stimulus', fontweight='bold')
        ax.set_ylabel('Relative Y Position (0=Bottom, 1=Top)')
        ax.set_xlabel('Stimulus Type')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # 5. Temporal changes in ROI quadrant preference
        ax = axes[1, 1]
        temporal_prefs = df.groupby(['trial_number', 'trial_type'])['roi_quadrant_preference_score'].mean().reset_index()
        for trial_type in temporal_prefs['trial_type'].unique():
            type_data = temporal_prefs[temporal_prefs['trial_type'] == trial_type]
            ax.plot(type_data['trial_number'], type_data['roi_quadrant_preference_score'],
                marker='o', label=trial_type, color=stimulus_colors.get(trial_type, 'gray'),
                linewidth=2, markersize=6)
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Average ROI Preference Score')
        ax.set_title('ROI Quadrant Preference Over Time', fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Heatmap of fish preferences within ROIs
        ax = axes[1, 2]
        pivot_data = df.pivot_table(values='roi_quadrant_preference_score', 
                                    index='roi_id', columns='trial_number', 
                                    aggfunc='mean')
        im = ax.imshow(pivot_data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Fish ID')
        ax.set_title('ROI Position Preferences Across Trials', fontweight='bold')
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index)
        plt.colorbar(im, ax=ax, label='ROI Preference Score')
        
        plt.suptitle('ROI-Based Quadrant Analysis: Top vs Bottom Half Within Each ROI', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ ROI quadrant analysis plot saved to:[/green] {save_path}")
        
        plt.show()

    def generate_roi_quadrant_trajectory_plot(self, df: pd.DataFrame, roi_id: int, 
                                            save_path: Optional[str] = None):
        """Generate a trajectory plot showing position within ROI for a specific fish."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Get ROI boundaries
        roi_boundaries = self.get_roi_boundaries(roi_id)
        if roi_boundaries is None:
            console.print(f"[red]No data found for ROI {roi_id}[/red]")
            return
        
        # Get all positions for this fish
        all_positions = []
        for trial in self.trials:
            pos_df = self.get_roi_positions_with_quadrants(roi_id, trial['start_frame'], trial['end_frame'])
            pos_df['trial_number'] = trial['number']
            pos_df['trial_type'] = trial['type']
            all_positions.append(pos_df)
        
        full_trajectory = pd.concat(all_positions)
        valid_trajectory = full_trajectory[full_trajectory['quadrant'] != -1]
        
        # Plot 1: Spatial trajectory colored by ROI quadrant
        ax = axes[0]
        q1_data = valid_trajectory[valid_trajectory['quadrant'] == 1]
        q2_data = valid_trajectory[valid_trajectory['quadrant'] == 2]
        
        ax.scatter(q1_data['x'], q1_data['y'], c='#3498db', alpha=0.3, s=1, label='Top Half of ROI')
        ax.scatter(q2_data['x'], q2_data['y'], c='#e74c3c', alpha=0.3, s=1, label='Bottom Half of ROI')
        
        # Draw ROI boundaries
        rect = plt.Rectangle((roi_boundaries['x_min'], roi_boundaries['y_min']),
                            roi_boundaries['x_max'] - roi_boundaries['x_min'],
                            roi_boundaries['y_max'] - roi_boundaries['y_min'],
                            linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
        # Add ROI center line
        ax.axhline(y=roi_boundaries['y_center'], 
                xmin=(roi_boundaries['x_min']/self.camera_width), 
                xmax=(roi_boundaries['x_max']/self.camera_width),
                color='black', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_xlim([roi_boundaries['x_min'] - 50, roi_boundaries['x_max'] + 50])
        ax.set_ylim([roi_boundaries['y_min'] - 50, roi_boundaries['y_max'] + 50])
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(f'Fish {roi_id} Spatial Distribution Within ROI', fontweight='bold')
        ax.legend()
        ax.set_aspect('equal')
        
        # Plot 2: Relative vertical position over time
        ax = axes[1]
        valid_trajectory['time_seconds'] = valid_trajectory['frame'] / self.fps
        
        ax.plot(valid_trajectory['time_seconds'], valid_trajectory['relative_y'], 
            color='black', alpha=0.5, linewidth=0.5)
        ax.fill_between(valid_trajectory['time_seconds'], 0.5, valid_trajectory['relative_y'],
                        where=(valid_trajectory['relative_y'] >= 0.5), alpha=0.3, color='#3498db',
                        label='Top Half')
        ax.fill_between(valid_trajectory['time_seconds'], 0.5, valid_trajectory['relative_y'],
                        where=(valid_trajectory['relative_y'] < 0.5), alpha=0.3, color='#e74c3c',
                        label='Bottom Half')
        
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Relative Position in ROI')
        ax.set_ylim([0, 1])
        ax.set_title(f'Fish {roi_id} Vertical Position Within ROI Over Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Histogram of vertical position
        ax = axes[2]
        ax.hist(valid_trajectory['relative_y'], bins=30, orientation='horizontal', 
            color='gray', alpha=0.7, edgecolor='black')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Relative Position in ROI')
        ax.set_ylim([0, 1])
        ax.set_title(f'Fish {roi_id} Vertical Distribution', fontweight='bold')
        
        # Add text with statistics
        mean_pos = valid_trajectory['relative_y'].mean()
        std_pos = valid_trajectory['relative_y'].std()
        stats_text = f'Mean: {mean_pos:.2f}\nSD: {std_pos:.2f}'
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'ROI-Based Position Analysis for Fish {roi_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ ROI trajectory plot saved to:[/green] {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze multi-fish responses to moving grating stimuli',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file with detections')
    parser.add_argument('h5_path', help='Path to H5 file with stimulus events')
    parser.add_argument('--plot', action='store_true',
                       help='Create comparison plots')
    parser.add_argument('--plot-distances', action='store_true',
                       help='Create per-fish distance plots')
    parser.add_argument('--plot-cumulative', action='store_true',
                       help='Create cumulative distance plot')
    parser.add_argument('--plot-temporal', action='store_true',
                       help='Create temporal (minute-by-minute) analysis plots')
    parser.add_argument('--plot-heatmap', action='store_true',
                       help='Create temporal heatmap of speeds')
    parser.add_argument('--plot-quadrants', action='store_true',
                       help='Create ROI-based quadrant analysis plots')
    parser.add_argument('--plot-trajectory', type=int, metavar='ROI_ID',
                       help='Plot trajectory within ROI for specific fish')
    parser.add_argument('--bin-minutes', type=float, default=1.0,
                       help='Time bin size in minutes for temporal analysis (default: 1.0)')
    parser.add_argument('--save', type=str,
                       help='Save results to CSV file')
    parser.add_argument('--save-fig', type=str,
                       help='Save comparison plot to file')
    parser.add_argument('--save-distances', type=str,
                       help='Save distance plot to file')
    parser.add_argument('--save-cumulative', type=str,
                       help='Save cumulative plot to file')
    parser.add_argument('--save-temporal', type=str,
                       help='Save temporal analysis plot to file')
    parser.add_argument('--save-quadrants', type=str,
                       help='Save ROI quadrant analysis plot to file')
    parser.add_argument('--save-trajectory', type=str,
                       help='Save ROI trajectory plot to file')
    parser.add_argument('--activity-threshold', type=float, default=10.0,
                       help='Speed threshold for activity detection (px/s)')
    parser.add_argument('--no-interpolated', action='store_true',
                       help='Do not use interpolated detections')
    parser.add_argument('--with-quadrants', action='store_true',
                       help='Include ROI-based quadrant analysis in metrics')
    parser.add_argument('--plot-quadrant-timeseries', action='store_true',
                   help='Create individual fish quadrant timeseries plots')
    parser.add_argument('--save-quadrant-plots', type=str,
                    help='Directory to save individual quadrant timeseries plots')
    parser.add_argument('--save-quadrant-metrics', type=str,
                    help='Save quadrant metrics to CSV files')
    parser.add_argument('--plot-frame-analysis', type=int, metavar='ROI_ID',
                   help='Plot frame-by-frame quadrant analysis for specific fish')
    parser.add_argument('--plot-all-frame-analysis', action='store_true',
                    help='Plot frame-by-frame analysis for all fish')
    parser.add_argument('--window-frames', type=int, default=300,
                    help='Rolling window size in frames for smoothing (default: 300)')
    parser.add_argument('--save-frame-plots', type=str,
                    help='Directory to save frame-by-frame plots')
    parser.add_argument('--analyze-orientation', action='store_true',
                    help='Analyze responses by grating orientation')
    parser.add_argument('--save-orientation', type=str,
                    help='Save orientation analysis plot')
    parser.add_argument('--plot-frame-orientation', type=int, metavar='ROI_ID',
                   help='Plot frame-by-frame analysis with orientation data for specific fish')
    parser.add_argument('--plot-orientation-summary', action='store_true',
                    help='Plot summary of preferences by orientation')
    parser.add_argument('--save-orientation-summary', type=str,
                    help='Save orientation preference summary plot')
    
    parser.add_argument('--save-proportions', type=str,
                       help='Save trial top proportions to CSV file')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = MultiROIGratingAnalyzer(
        zarr_path=args.zarr_path,
        h5_path=args.h5_path,
        activity_threshold_px_per_s=args.activity_threshold,
        use_interpolated=not args.no_interpolated
    )
    
    # Run analysis - use quadrant analysis if requested
    console.print("\n[cyan]Running analysis...[/cyan]")
    if args.with_quadrants or args.plot_quadrants or args.plot_trajectory is not None:
        df = analyzer.analyze_all_trials_with_quadrants()
    else:
        df = analyzer.analyze_all_trials()
    
    if args.analyze_orientation:
        analyzer.add_orientation_to_trials()
    
    # Print summary statistics
    console.print("\n[bold]Analysis Results:[/bold]")
    for stimulus_type in df['trial_type'].unique():
        stim_df = df[df['trial_type'] == stimulus_type]
        console.print(f"\n{stimulus_type.upper()} trials (n={len(stim_df)}):")
        console.print(f"  Mean speed: {stim_df['mean_speed_px_per_s'].mean():.1f} ± {stim_df['mean_speed_px_per_s'].std():.1f} px/s")
        console.print(f"  Activity ratio: {stim_df['activity_ratio'].mean():.2f} ± {stim_df['activity_ratio'].std():.2f}")
        console.print(f"  Bout frequency: {(stim_df['bout_count']/stim_df['duration_s']).mean():.2f} bouts/s")
        
        # Add ROI quadrant stats if available
        if 'roi_quadrant_preference_score' in df.columns:
            console.print(f"  ROI position preference: {stim_df['roi_quadrant_preference_score'].mean():.2f} ± {stim_df['roi_quadrant_preference_score'].std():.2f}")
            console.print(f"  ROI transitions: {stim_df['roi_quadrant_transitions'].mean():.1f} ± {stim_df['roi_quadrant_transitions'].std():.1f}")
            # Add top proportion summary
            console.print(f"  Top quadrant proportion: {stim_df['roi_q1_proportion'].mean():.2%} ± {stim_df['roi_q1_proportion'].std():.2%}")
    
    # Print per-fish distance summary
    console.print("\n[bold]Distance Summary by Fish:[/bold]")
    fish_distances = df.groupby('roi_id')['total_distance_px'].sum()
    for roi_id in sorted(fish_distances.index):
        console.print(f"  Fish {roi_id}: {fish_distances[roi_id]:.0f} pixels total")
        
        # Add ROI preference if available
        if 'roi_quadrant_preference_score' in df.columns:
            fish_pref = df[df['roi_id'] == roi_id]['roi_quadrant_preference_score'].mean()
            fish_top_prop = df[df['roi_id'] == roi_id]['roi_q1_proportion'].mean()
            pref_text = "prefers top" if fish_pref > 0.1 else "prefers bottom" if fish_pref < -0.1 else "no preference"
            console.print(f"    ROI position: {pref_text} (score: {fish_pref:.2f}, top: {fish_top_prop:.2%})")
    
    # Save proportions if requested
    if args.save_proportions:
        if 'roi_q1_proportion' in df.columns:
            proportions_df = analyzer.get_trial_top_proportions(df)
            proportions_df.to_csv(args.save_proportions, index=False)
            console.print(f"\n[green]✓ Trial proportions saved to {args.save_proportions}[/green]")
            
            # Print sample
            console.print("\n[bold]Sample Trial Proportions:[/bold]")
            for _, row in proportions_df.head(10).iterrows():
                console.print(f"  Fish {int(row['roi_id'])}, Trial {int(row['trial_number'])}: {row['top_proportion']:.2%}")
        else:
            console.print("[yellow]Warning: Quadrant data not available. Re-run with --with-quadrants flag.[/yellow]")
    
    # All plotting functions follow...
    if args.plot:
        analyzer.plot_stimulus_comparison(df, save_path=args.save_fig)
    
    if args.plot_distances:
        analyzer.plot_per_fish_distances(df, save_path=args.save_distances)
    
    if args.plot_cumulative:
        analyzer.plot_cumulative_distances(df, save_path=args.save_cumulative)
    
    if args.plot_temporal:
        analyzer.plot_temporal_analysis(df, bin_minutes=args.bin_minutes, 
                                       save_path=args.save_temporal)
    
    if args.plot_heatmap:
        analyzer.plot_heatmap_temporal(df, bin_minutes=args.bin_minutes,
                                      save_path=args.save_temporal)
    
    if args.plot_quadrants:
        if 'roi_quadrant_preference_score' not in df.columns:
            console.print("[yellow]Warning: Quadrant analysis not available. Re-run with --with-quadrants flag.[/yellow]")
        else:
            analyzer.plot_quadrant_analysis(df, save_path=args.save_quadrants)
    
    if args.plot_trajectory is not None:
        if 'roi_quadrant_preference_score' not in df.columns:
            console.print("[yellow]Warning: Quadrant analysis not available. Re-run with --with-quadrants flag.[/yellow]")
        else:
            analyzer.generate_roi_quadrant_trajectory_plot(df, args.plot_trajectory, 
                                                          save_path=args.save_trajectory)
    
    if args.plot_quadrant_timeseries:
        if 'roi_quadrant_preference_score' not in df.columns:
            console.print("[yellow]Warning: Quadrant analysis not available. Re-run with --with-quadrants flag.[/yellow]")
        else:
            analyzer.plot_individual_fish_quadrant_timeseries(
                df, 
                bin_minutes=args.bin_minutes,
                save_dir=args.save_quadrant_plots
            )
    
    if args.save_quadrant_metrics:
        if 'roi_quadrant_preference_score' not in df.columns:
            console.print("[yellow]Warning: Quadrant analysis not available. Re-run with --with-quadrants flag.[/yellow]")
        else:
            analyzer.save_quadrant_metrics_csv(df, args.save_quadrant_metrics)
    
    if args.plot_frame_analysis is not None:
        analyzer.plot_fish_quadrant_frame_by_frame(
            args.plot_frame_analysis,
            window_frames=args.window_frames,
            save_path=args.save_frame_plots
        )
    
    if args.plot_all_frame_analysis:
        analyzer.plot_all_fish_frame_by_frame(
            window_frames=args.window_frames,
            save_dir=args.save_frame_plots
        )
    
    if args.plot_frame_orientation is not None:
        analyzer.plot_fish_quadrant_frame_by_frame_with_orientation(
            args.plot_frame_orientation,
            window_frames=args.window_frames,
            save_path=args.save_frame_plots
        )
    
    if args.plot_orientation_summary:
        pref_df = analyzer.plot_orientation_preference_summary(
            window_frames=args.window_frames,
            save_path=args.save_orientation_summary
        )
    
    if args.analyze_orientation:
        analyzer.plot_orientation_response(df, save_path=args.save_orientation)
    
    # Save main results if requested
    if args.save:
        analyzer.save_results(df, args.save)
    
    console.print("\n[green]✓ Analysis complete![/green]")


if __name__ == '__main__':
    main()