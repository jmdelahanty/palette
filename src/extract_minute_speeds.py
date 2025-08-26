#!/usr/bin/env python3
"""
Minute-by-Minute Speed Extractor and Plotter

Extracts and visualizes average swimming speed for each fish binned by minutes.
Similar to quadrant preference analysis but focused on locomotor activity.
"""

import zarr
import h5py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import warnings
warnings.filterwarnings('ignore')

console = Console()
sns.set_style("whitegrid")

@dataclass
class MinuteSpeedData:
    """Speed data for one fish during one minute bin."""
    roi_id: int
    minute: float  # End of minute bin
    minute_start: float
    minute_end: float
    mean_speed_px_per_s: float  # Average speed in pixels/second
    median_speed_px_per_s: float
    max_speed_px_per_s: float
    std_speed_px_per_s: float
    total_distance_px: float  # Total distance traveled in pixels
    detection_rate: float
    n_frames: int
    n_valid: int
    active_ratio: float  # Proportion of time moving > threshold


class MinuteSpeedExtractor:
    """Extract minute-by-minute speed data from zarr files."""
    
    def __init__(self, zarr_path: str, h5_path: str, 
                 bin_minutes: float = 1.0,
                 activity_threshold: float = 10.0):
        """
        Initialize speed extractor.
        
        Args:
            zarr_path: Path to zarr file with detections
            h5_path: Path to H5 file (for metadata and calibration)
            bin_minutes: Size of time bins in minutes
            activity_threshold: Speed threshold for active vs inactive (px/s or cm/s if calibrated)
        """
        self.zarr_path = Path(zarr_path)
        self.h5_path = Path(h5_path)
        self.bin_minutes = bin_minutes
        self.activity_threshold = activity_threshold
        
        # Load zarr data
        self.root = zarr.open_group(self.zarr_path, mode='r')
        
        # Get dimensions and fps
        self.camera_width = self.root.attrs.get('width', 4512)
        self.camera_height = self.root.attrs.get('height', 4512)
        self.fps = self.root.attrs.get('fps', 60.0)
        
        # Check for calibration in zarr first
        self.pixel_to_mm = None
        self.pixel_to_cm = None
        
        if 'calibration' in self.root:
            self.pixel_to_mm = self.root['calibration'].attrs.get('pixel_to_mm', None)
            if self.pixel_to_mm:
                console.print(f"[green]✓ Found calibration in zarr: {self.pixel_to_mm:.4f} mm/pixel[/green]")
        
        # If not in zarr, try to extract from H5
        if self.pixel_to_mm is None:
            console.print("[yellow]No calibration in zarr, checking H5...[/yellow]")
            self.extract_calibration_from_h5()
        
        if self.pixel_to_mm:
            self.pixel_to_cm = self.pixel_to_mm / 10.0  # Convert mm to cm
            console.print(f"[cyan]Using calibration: 1 pixel = {self.pixel_to_mm:.4f} mm = {self.pixel_to_cm:.4f} cm[/cyan]")
            # Adjust threshold if calibrated
            self.activity_threshold_cm = self.activity_threshold * self.pixel_to_cm
            console.print(f"[cyan]Activity threshold: {self.activity_threshold_cm:.2f} cm/s[/cyan]")
        else:
            console.print("[red]No calibration found - speeds will be in pixels[/red]")
        
        console.print(f"[cyan]Camera: {self.camera_width}x{self.camera_height}, {self.fps} fps[/cyan]")
        console.print(f"[cyan]Bin size: {bin_minutes} minutes[/cyan]")
        
        # Load detection data
        self.load_detections()
        self.num_rois = 12  # Fixed for this experiment
    
    def extract_calibration_from_h5(self):
        """Extract calibration data from H5 file."""
        try:
            with h5py.File(self.h5_path, 'r') as hf:
                # Check for calibration_snapshot
                if '/calibration_snapshot' in hf:
                    calib_group = hf['/calibration_snapshot']
                    
                    # Get arena config JSON
                    if 'arena_config_json' in calib_group:
                        arena_json = calib_group['arena_config_json'][()].decode('utf-8')
                        arena_config = json.loads(arena_json)
                        
                        # Extract camera calibration data
                        if 'camera_calibrations' in arena_config:
                            for cam_cal in arena_config['camera_calibrations']:
                                # The pixel_to_mm is usually in millimeters_per_pixel field
                                if 'millimeters_per_pixel' in cam_cal:
                                    self.pixel_to_mm = cam_cal['millimeters_per_pixel']
                                    console.print(f"[green]✓ Found calibration in H5: {self.pixel_to_mm:.4f} mm/pixel[/green]")
                                    break
                                # Sometimes it's stored as pixels_per_millimeter
                                elif 'pixels_per_millimeter' in cam_cal:
                                    self.pixel_to_mm = 1.0 / cam_cal['pixels_per_millimeter']
                                    console.print(f"[green]✓ Found calibration in H5: {self.pixel_to_mm:.4f} mm/pixel[/green]")
                                    break
        except Exception as e:
            console.print(f"[yellow]Could not extract calibration from H5: {e}[/yellow]")
        
    def load_detections(self):
        """Load detection data from zarr."""
        detect_group = self.root['detect_runs']
        latest_detect = detect_group.attrs['latest']
        self.n_detections = detect_group[latest_detect]['n_detections'][:]
        self.bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
        
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        self.detection_ids = id_group[latest_id]['detection_ids'][:]
        self.n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        
        # Load interpolated if available
        self.interpolated_data = None
        if 'interpolated_detections' in self.root:
            interp_group = self.root['interpolated_detections']
            if 'latest' in interp_group.attrs:
                latest_interp = interp_group.attrs['latest']
                interp_data = interp_group[latest_interp]
                self.interpolated_data = {
                    'frame_indices': interp_data['frame_indices'][:],
                    'roi_ids': interp_data['roi_ids'][:],
                    'bboxes': interp_data['bboxes'][:]
                }
        
        self.total_frames = len(self.n_detections)
        self.total_minutes = self.total_frames / (self.fps * 60)
        console.print(f"[cyan]Total: {self.total_frames} frames ({self.total_minutes:.1f} minutes)[/cyan]")
    
    def get_roi_positions(self, roi_id: int) -> np.ndarray:
        """Get all positions for a specific ROI."""
        positions = np.full((self.total_frames, 2), np.nan)
        
        cumulative_idx = 0
        for frame_idx in range(self.total_frames):
            frame_det_count = int(self.n_detections[frame_idx])
            
            if frame_det_count > 0 and self.n_detections_per_roi[frame_idx, roi_id] > 0:
                frame_detection_ids = self.detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
                roi_mask = frame_detection_ids == roi_id
                
                if np.any(roi_mask):
                    roi_idx = np.where(roi_mask)[0][0]
                    bbox = self.bbox_coords[cumulative_idx + roi_idx]
                    
                    center_x_norm = bbox[0]
                    center_y_norm = bbox[1]
                    
                    centroid_x_ds = center_x_norm * 640
                    centroid_y_ds = center_y_norm * 640
                    scale = self.camera_width / 640
                    
                    positions[frame_idx, 0] = centroid_x_ds * scale
                    positions[frame_idx, 1] = centroid_y_ds * scale
            
            cumulative_idx += frame_det_count
        
        # Add interpolated positions
        if self.interpolated_data is not None:
            for j in range(len(self.interpolated_data['frame_indices'])):
                frame_idx = int(self.interpolated_data['frame_indices'][j])
                if frame_idx < self.total_frames and int(self.interpolated_data['roi_ids'][j]) == roi_id:
                    if np.isnan(positions[frame_idx, 0]):
                        bbox = self.interpolated_data['bboxes'][j]
                        center_x_norm = bbox[0]
                        center_y_norm = bbox[1]
                        
                        centroid_x_ds = center_x_norm * 640
                        centroid_y_ds = center_y_norm * 640
                        scale = self.camera_width / 640
                        
                        positions[frame_idx, 0] = centroid_x_ds * scale
                        positions[frame_idx, 1] = centroid_y_ds * scale
        
        return positions
    
    def calculate_speeds_for_bin(self, positions: np.ndarray) -> Dict:
        """Calculate speed metrics for positions in a time bin."""
        valid_mask = ~np.isnan(positions[:, 0])
        
        if np.sum(valid_mask) < 2:
            return {
                'mean_speed': np.nan,
                'median_speed': np.nan,
                'max_speed': np.nan,
                'std_speed': np.nan,
                'total_distance': 0,
                'active_frames': 0,
                'n_valid': np.sum(valid_mask)
            }
        
        # Get valid positions
        valid_pos = positions[valid_mask]
        
        # Calculate frame-to-frame distances
        distances = np.sqrt(np.sum(np.diff(valid_pos, axis=0)**2, axis=1))
        speeds = distances * self.fps  # Convert to pixels per second
        
        # Count active frames
        active_frames = np.sum(speeds > self.activity_threshold)
        
        return {
            'mean_speed': np.mean(speeds),
            'median_speed': np.median(speeds),
            'max_speed': np.max(speeds),
            'std_speed': np.std(speeds),
            'total_distance': np.sum(distances),
            'active_frames': active_frames,
            'n_valid': np.sum(valid_mask)
        }
    
    def extract_minute_speeds(self) -> pd.DataFrame:
        """Extract minute-by-minute speed data for all fish."""
        all_data = []
        
        # Create time bins
        time_bins = np.arange(0, self.total_minutes + self.bin_minutes, self.bin_minutes)
        n_bins = len(time_bins) - 1
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            total_analyses = self.num_rois * n_bins
            task = progress.add_task(
                f"[cyan]Extracting speed data for {self.num_rois} fish...", 
                total=total_analyses
            )
            
            for roi_id in range(self.num_rois):
                # Get all positions for this fish
                positions = self.get_roi_positions(roi_id)
                
                # Process each minute bin
                for i in range(n_bins):
                    minute_start = time_bins[i]
                    minute_end = time_bins[i + 1]
                    
                    # Get frames in this minute
                    frame_start = int(minute_start * 60 * self.fps)
                    frame_end = min(int(minute_end * 60 * self.fps), self.total_frames)
                    
                    if frame_start < frame_end:
                        bin_positions = positions[frame_start:frame_end]
                        
                        # Calculate speed metrics
                        speed_info = self.calculate_speeds_for_bin(bin_positions)
                        
                        n_frames = len(bin_positions)
                        n_valid = speed_info['n_valid']
                        detection_rate = n_valid / n_frames if n_frames > 0 else 0
                        active_ratio = speed_info['active_frames'] / n_valid if n_valid > 0 else 0
                        
                        speed_data = MinuteSpeedData(
                            roi_id=roi_id,
                            minute=minute_end,  # Use end of bin for alignment
                            minute_start=minute_start,
                            minute_end=minute_end,
                            mean_speed_px_per_s=speed_info['mean_speed'],
                            median_speed_px_per_s=speed_info['median_speed'],
                            max_speed_px_per_s=speed_info['max_speed'],
                            std_speed_px_per_s=speed_info['std_speed'],
                            total_distance_px=speed_info['total_distance'],
                            detection_rate=detection_rate,
                            n_frames=n_frames,
                            n_valid=n_valid,
                            active_ratio=active_ratio
                        )
                        
                        all_data.append(speed_data.__dict__)
                    
                    progress.advance(task)
        
        df = pd.DataFrame(all_data)
        
        # Add group labels
        df['group'] = df['roi_id'].apply(lambda x: 1 if x <= 5 else 2)
        
        # Add calibrated columns if available
        if self.pixel_to_cm is not None:
            df['mean_speed_cm_per_s'] = df['mean_speed_px_per_s'] * self.pixel_to_cm
            df['median_speed_cm_per_s'] = df['median_speed_px_per_s'] * self.pixel_to_cm
            df['max_speed_cm_per_s'] = df['max_speed_px_per_s'] * self.pixel_to_cm
            df['std_speed_cm_per_s'] = df['std_speed_px_per_s'] * self.pixel_to_cm
            df['total_distance_cm'] = df['total_distance_px'] * self.pixel_to_cm
            console.print("[green]✓ Added calibrated speed columns (cm/s)[/green]")
        
        return df


def plot_minute_speed_scatter(df: pd.DataFrame, save_dir: Optional[str] = None):
    """Plot minute-by-minute speed data with group comparisons."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # Colors for groups
    group_colors = {1: '#2E86AB', 2: '#A23B72'}
    
    # Check if we have calibrated data
    use_cm = 'mean_speed_cm_per_s' in df.columns
    speed_col = 'mean_speed_cm_per_s' if use_cm else 'mean_speed_px_per_s'
    dist_col = 'total_distance_cm' if use_cm else 'total_distance_px'
    y_label = 'Mean Speed (cm/s)' if use_cm else 'Mean Speed (pixels/s)'
    dist_unit = 'cm' if use_cm else 'pixels'
    
    # Filter valid data
    valid_df = df[df['detection_rate'] > 0.5]  # Only use minutes with >50% detection
    
    # PLOT 1: Speed over time
    # Plot individual points with jitter
    for group in [1, 2]:
        group_data = valid_df[valid_df['group'] == group]
        
        # Add jitter
        time_jitter = np.random.normal(0, 0.02, size=len(group_data))
        
        ax1.scatter(group_data['minute'] + time_jitter, 
                  group_data[speed_col],
                  color=group_colors[group], alpha=0.4, s=30,
                  edgecolor='none', label=f'Group {group} (IDs {0 if group==1 else 6}-{5 if group==1 else 11})')
    
    # Calculate and plot means at each time point
    unique_times = sorted(valid_df['minute'].unique())
    
    for group in [1, 2]:
        group_df = valid_df[valid_df['group'] == group]
        
        means = []
        sems = []
        times = []
        
        for time_point in unique_times:
            time_data = group_df[np.abs(group_df['minute'] - time_point) < 0.01]
            if len(time_data) > 0:
                means.append(time_data[speed_col].mean())
                sems.append(time_data[speed_col].sem())
                times.append(time_point)
        
        # Plot means with lines
        if len(times) > 0:
            ax1.plot(times, means,
                   color=group_colors[group], linewidth=3, alpha=0.9,
                   marker='o', markersize=8, markeredgecolor='black', markeredgewidth=1)
            
            # Add error bars
            ax1.errorbar(times, means, yerr=sems,
                       color=group_colors[group], alpha=0.7,
                       capsize=5, capthick=2, elinewidth=2,
                       fmt='none')
    
    ax1.set_xlabel('Time (minutes)', fontweight='bold', fontsize=12)
    ax1.set_ylabel(y_label, fontweight='bold', fontsize=12)
    ax1.set_title('Swimming Speed Over Time', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # PLOT 2: Cumulative distance (group means)
    # Calculate cumulative distance for each fish
    for group in [1, 2]:
        group_df = valid_df[valid_df['group'] == group]
        
        for roi_id in sorted(group_df['roi_id'].unique()):
            fish_df = group_df[group_df['roi_id'] == roi_id].sort_values('minute')
            cumulative = fish_df[dist_col].cumsum()
            
            # Convert to meters if in cm
            if use_cm:
                cumulative = cumulative / 100  # cm to meters
                dist_display_unit = 'meters'
            else:
                dist_display_unit = dist_unit
            
            ax2.plot(fish_df['minute'].values, cumulative.values,
                   color=group_colors[group], alpha=0.3, linewidth=1)
        
        # Add group mean cumulative distance
        group_mean_dist = []
        for time_point in unique_times:
            # Get all fish data up to this time point
            cumulative_per_fish = []
            for roi_id in group_df['roi_id'].unique():
                fish_data = group_df[(group_df['roi_id'] == roi_id) & 
                                    (group_df['minute'] <= time_point)]
                if len(fish_data) > 0:
                    total = fish_data[dist_col].sum()
                    if use_cm:
                        total = total / 100  # Convert to meters
                    cumulative_per_fish.append(total)
            
            if cumulative_per_fish:
                group_mean_dist.append(np.mean(cumulative_per_fish))
            else:
                group_mean_dist.append(0)
        
        # Plot thick group mean line
        ax2.plot(unique_times, group_mean_dist,
               color=group_colors[group], linewidth=4, alpha=0.9,
               marker='o', markersize=6, markeredgecolor='black', markeredgewidth=1,
               label=f'Group {group} mean')
    
    ax2.set_xlabel('Time (minutes)', fontweight='bold', fontsize=12)
    ax2.set_ylabel(f'Cumulative Distance ({dist_display_unit})', fontweight='bold', fontsize=12)
    ax2.set_title('Cumulative Distance Traveled (Group Means)', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    
    # Set x-axis ticks for both plots
    max_time = valid_df['minute'].max()
    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(0, np.ceil(max_time) + 1, 1))
    
    plt.tight_layout()
    
    if save_dir:
        # Create the directory if it doesn't exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Now save the plot
        save_file = save_path / 'minute_speed_and_distance_plot.png'
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved speed and distance plot to {save_file}[/green]")
    
    plt.show()


def plot_individual_cumulative_distance(df: pd.DataFrame, save_dir: Optional[str] = None):
    """Plot individual fish cumulative distance with clear fish labels."""
    
    # Check if we have calibrated data
    use_cm = 'total_distance_cm' in df.columns
    dist_col = 'total_distance_cm' if use_cm else 'total_distance_px'
    
    # Filter valid data
    valid_df = df[df['detection_rate'] > 0.5]  # Only use minutes with >50% detection
    
    # Create figure with 2 subplots (one for each group)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Color palettes for individual fish
    group1_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 6))
    group2_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 6))
    
    # Process each group
    for group_idx, (ax, group_colors_palette, group_name) in enumerate(
        [(ax1, group1_colors, 'Group 1 (Fish 0-5)'), 
         (ax2, group2_colors, 'Group 2 (Fish 6-11)')], start=1):
        
        group_df = valid_df[valid_df['group'] == group_idx]
        
        # Store data for group mean calculation
        all_fish_cumulative = {}
        unique_times = sorted(group_df['minute'].unique())
        
        # Plot each fish individually
        for fish_idx, roi_id in enumerate(sorted(group_df['roi_id'].unique())):
            fish_df = group_df[group_df['roi_id'] == roi_id].sort_values('minute')
            cumulative = fish_df[dist_col].cumsum()
            
            # Convert to meters if in cm
            if use_cm:
                cumulative = cumulative / 100  # cm to meters
                dist_display_unit = 'meters'
            else:
                dist_display_unit = 'pixels'
            
            # Store for mean calculation
            all_fish_cumulative[roi_id] = {
                'times': fish_df['minute'].values,
                'cumulative': cumulative.values
            }
            
            # Plot individual fish line
            ax.plot(fish_df['minute'].values, cumulative.values,
                   color=group_colors_palette[fish_idx], 
                   alpha=0.7, linewidth=2.5,
                   label=f'Fish {roi_id}',
                   marker='o', markersize=3, markevery=3)
        
        # Calculate and plot group mean
        group_mean_dist = []
        for time_point in unique_times:
            cumulative_at_time = []
            for roi_id, data in all_fish_cumulative.items():
                # Find the cumulative distance at or before this time
                time_mask = data['times'] <= time_point
                if np.any(time_mask):
                    cumulative_at_time.append(data['cumulative'][time_mask][-1])
            
            if cumulative_at_time:
                group_mean_dist.append(np.mean(cumulative_at_time))
            else:
                group_mean_dist.append(0)
        
        # Plot thick group mean line
        ax.plot(unique_times, group_mean_dist,
               color='black', linewidth=4, alpha=0.8,
               linestyle='--',
               marker='s', markersize=8, markevery=2,
               label=f'Group mean', zorder=10)
        
        # Formatting
        ax.set_xlabel('Time (minutes)', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'Cumulative Distance ({dist_display_unit})', fontweight='bold', fontsize=12)
        ax.set_title(group_name, fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.legend(loc='upper left', fontsize=9, ncol=1, 
                 framealpha=0.95, shadow=True)
        
        # Set x-axis ticks
        max_time = valid_df['minute'].max()
        ax.set_xticks(np.arange(0, np.ceil(max_time) + 1, 1))
        
        # Add statistics text box
        final_distances = [data['cumulative'][-1] for data in all_fish_cumulative.values()]
        stats_text = (f'Mean final: {np.mean(final_distances):.1f} {dist_display_unit}\n'
                     f'Std dev: {np.std(final_distances):.1f} {dist_display_unit}\n'
                     f'Range: {np.min(final_distances):.1f} - {np.max(final_distances):.1f}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='gray', alpha=0.95))
    
    # Main title
    fig.suptitle('Individual Fish Cumulative Distance Traveled', 
                fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    
    if save_dir:
        # Create the directory if it doesn't exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        save_file = save_path / 'individual_cumulative_distance_plot.png'
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved individual cumulative distance plot to {save_file}[/green]")
    
    plt.show()


def plot_activity_heatmap(df: pd.DataFrame, save_dir: Optional[str] = None):
    """Create a heatmap showing activity patterns for each fish over time."""
    
    # Check if we have calibrated data
    use_cm = 'mean_speed_cm_per_s' in df.columns
    speed_col = 'mean_speed_cm_per_s' if use_cm else 'mean_speed_px_per_s'
    speed_unit = 'cm/s' if use_cm else 'pixels/s'
    
    # Pivot data for heatmap
    pivot_df = df.pivot(index='roi_id', columns='minute', values=speed_col)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                   gridspec_kw={'height_ratios': [1, 1]})
    
    # Group 1 heatmap
    group1_data = pivot_df.iloc[0:6, :]
    im1 = ax1.imshow(group1_data, aspect='auto', cmap='YlOrRd', 
                    interpolation='nearest', vmin=0)
    ax1.set_yticks(range(6))
    ax1.set_yticklabels([f'Fish {i}' for i in range(6)])
    ax1.set_title(f'Group 1 Activity Heatmap - Speed ({speed_unit})', 
                 fontweight='bold', fontsize=12)
    ax1.set_xlabel('')
    
    # Add colorbar for group 1
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label(f'Speed ({speed_unit})', fontsize=10)
    
    # Group 2 heatmap
    group2_data = pivot_df.iloc[6:12, :]
    im2 = ax2.imshow(group2_data, aspect='auto', cmap='YlGnBu', 
                    interpolation='nearest', vmin=0)
    ax2.set_yticks(range(6))
    ax2.set_yticklabels([f'Fish {i}' for i in range(6, 12)])
    ax2.set_title(f'Group 2 Activity Heatmap - Speed ({speed_unit})', 
                 fontweight='bold', fontsize=12)
    ax2.set_xlabel('Time (minutes)', fontweight='bold', fontsize=11)
    
    # Add colorbar for group 2
    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.set_label(f'Speed ({speed_unit})', fontsize=10)
    
    # Set x-axis for both
    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(0, len(pivot_df.columns), 2))
        ax.set_xticklabels([f'{x:.0f}' for x in pivot_df.columns[::2]], fontsize=9)
    
    fig.suptitle('Swimming Activity Patterns Over Time', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        save_file = save_path / 'activity_heatmap.png'
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved activity heatmap to {save_file}[/green]")
    
    plt.show()

def print_speed_statistics(df: pd.DataFrame):
    """Print summary statistics for speed data."""
    console.print("\n[bold cyan]Speed Statistics:[/bold cyan]")
    
    valid_df = df[df['detection_rate'] > 0.5]
    
    # Check if we have calibrated data
    use_cm = 'mean_speed_cm_per_s' in df.columns
    speed_col = 'mean_speed_cm_per_s' if use_cm else 'mean_speed_px_per_s'
    max_col = 'max_speed_cm_per_s' if use_cm else 'max_speed_px_per_s'
    dist_col = 'total_distance_cm' if use_cm else 'total_distance_px'
    unit = 'cm/s' if use_cm else 'px/s'
    dist_unit = 'cm' if use_cm else 'pixels'
    
    # Overall stats
    console.print(f"\nOverall mean speed: {valid_df[speed_col].mean():.2f} {unit}")
    console.print(f"Overall activity ratio: {valid_df['active_ratio'].mean():.2%}")
    
    # Group comparison
    for group in [1, 2]:
        group_data = valid_df[valid_df['group'] == group]
        console.print(f"\n[bold]Group {group}:[/bold]")
        console.print(f"  Mean speed: {group_data[speed_col].mean():.2f} ± {group_data[speed_col].std():.2f} {unit}")
        console.print(f"  Max speed: {group_data[max_col].max():.2f} {unit}")
        console.print(f"  Activity ratio: {group_data['active_ratio'].mean():.2%}")
        console.print(f"  Total distance: {group_data[dist_col].sum():.1f} {dist_unit}")
        
        # Add distance in meters if using cm
        if use_cm:
            total_m = group_data[dist_col].sum() / 100
            console.print(f"  Total distance: {total_m:.2f} meters")


def plot_cumulative_distance_only(df: pd.DataFrame, save_dir: Optional[str] = None, 
                                  show_error_bars: bool = True,
                                  show_per_minute_means: bool = True):
    """
    Plot cumulative distance with group means, error bars, and optional per-minute statistics.
    
    Args:
        df: DataFrame with speed/distance data
        save_dir: Directory to save plot
        show_error_bars: Whether to show error bars on cumulative distance
        show_per_minute_means: Whether to show per-minute mean distances as secondary plot
    """
    
    # Check if we have calibrated data
    use_cm = 'total_distance_cm' in df.columns
    dist_col = 'total_distance_cm' if use_cm else 'total_distance_px'
    
    # Filter valid data
    valid_df = df[df['detection_rate'] > 0.5]  # Only use minutes with >50% detection
    
    # Create figure - add second subplot if showing per-minute means
    if show_per_minute_means:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [2, 1]},
                                       sharex=True)
        ax = ax1
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors for groups
    group_colors = {1: '#2E86AB', 2: '#A23B72'}
    
    # Get unique times
    unique_times = sorted(valid_df['minute'].unique())
    
    # Store data for per-minute plot
    per_minute_data = {1: [], 2: []}
    
    # Process each group
    for group in [1, 2]:
        group_df = valid_df[valid_df['group'] == group]
        
        # Plot individual fish lines (thin, transparent)
        for roi_id in sorted(group_df['roi_id'].unique()):
            fish_df = group_df[group_df['roi_id'] == roi_id].sort_values('minute')
            cumulative = fish_df[dist_col].cumsum()
            
            # Convert to meters if in cm
            if use_cm:
                cumulative = cumulative / 100  # cm to meters
                dist_display_unit = 'meters'
            else:
                dist_display_unit = 'pixels'
            
            ax.plot(fish_df['minute'].values, cumulative.values,
                   color=group_colors[group], alpha=0.2, linewidth=0.8)
        
        # Calculate group mean and SEM for cumulative distance
        group_mean_dist = []
        group_sem_dist = []
        
        for time_point in unique_times:
            # Get cumulative distance for each fish up to this time point
            cumulative_per_fish = []
            for roi_id in group_df['roi_id'].unique():
                fish_data = group_df[(group_df['roi_id'] == roi_id) & 
                                    (group_df['minute'] <= time_point)]
                if len(fish_data) > 0:
                    total = fish_data[dist_col].sum()
                    if use_cm:
                        total = total / 100  # Convert to meters
                    cumulative_per_fish.append(total)
            
            if cumulative_per_fish:
                mean_val = np.mean(cumulative_per_fish)
                sem_val = np.std(cumulative_per_fish) / np.sqrt(len(cumulative_per_fish))
                group_mean_dist.append(mean_val)
                group_sem_dist.append(sem_val)
            else:
                group_mean_dist.append(0)
                group_sem_dist.append(0)
            
            # Store per-minute distance (difference from previous minute)
            if show_per_minute_means and len(group_mean_dist) > 1:
                per_minute = group_mean_dist[-1] - group_mean_dist[-2]
                per_minute_data[group].append(per_minute)
            elif show_per_minute_means:
                per_minute_data[group].append(group_mean_dist[-1])
        
        group_mean_dist = np.array(group_mean_dist)
        group_sem_dist = np.array(group_sem_dist)
        
        # Plot mean line with error bars
        if show_error_bars:
            # Plot error bars as shaded region
            ax.fill_between(unique_times, 
                          group_mean_dist - group_sem_dist,
                          group_mean_dist + group_sem_dist,
                          color=group_colors[group], alpha=0.2)
            
            # Also add discrete error bars at each point
            ax.errorbar(unique_times, group_mean_dist, yerr=group_sem_dist,
                       color=group_colors[group], alpha=0.7,
                       capsize=4, capthick=2, elinewidth=2,
                       fmt='none', zorder=5)
        
        # Plot thick group mean line
        ax.plot(unique_times, group_mean_dist,
               color=group_colors[group], linewidth=4, alpha=0.9,
               marker='o', markersize=8, markeredgecolor='black', markeredgewidth=1.5,
               label=f'Group {group} mean ± SEM', zorder=10)
    
    # Formatting for main plot
    if not show_per_minute_means:
        ax.set_xlabel('Time (minutes)', fontweight='bold', fontsize=14)
    ax.set_ylabel(f'Cumulative Distance ({dist_display_unit})', fontweight='bold', fontsize=14)
    ax.set_title('Cumulative Distance Traveled with Error Bars', fontweight='bold', fontsize=16)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, shadow=True)
    
    # Set x-axis ticks
    max_time = valid_df['minute'].max()
    ax.set_xticks(np.arange(0, np.ceil(max_time) + 1, 1))
    
    # Make sure y-axis starts at 0
    ax.set_ylim(bottom=0)
    
    # Add per-minute distance plot if requested
    if show_per_minute_means:
        # Calculate per-minute means and SEMs
        for group in [1, 2]:
            group_df = valid_df[valid_df['group'] == group]
            
            minute_means = []
            minute_sems = []
            
            for i, time_point in enumerate(unique_times):
                # Get distance traveled in this specific minute for each fish
                minute_distances = []
                for roi_id in group_df['roi_id'].unique():
                    fish_data = group_df[(group_df['roi_id'] == roi_id) & 
                                        (np.abs(group_df['minute'] - time_point) < 0.01)]
                    if len(fish_data) > 0:
                        dist_this_minute = fish_data[dist_col].values[0]
                        if use_cm:
                            dist_this_minute = dist_this_minute / 100  # Convert to meters
                        minute_distances.append(dist_this_minute)
                
                if minute_distances:
                    minute_means.append(np.mean(minute_distances))
                    minute_sems.append(np.std(minute_distances) / np.sqrt(len(minute_distances)))
                else:
                    minute_means.append(0)
                    minute_sems.append(0)
            
            # Plot bars with error bars
            x_positions = np.array(unique_times) + (group - 1.5) * 0.2  # Offset bars
            ax2.bar(x_positions, minute_means, width=0.35,
                   color=group_colors[group], alpha=0.7,
                   label=f'Group {group}', edgecolor='black', linewidth=1)
            
            # Add error bars
            ax2.errorbar(x_positions, minute_means, yerr=minute_sems,
                        fmt='none', color='black', capsize=3, capthick=1, alpha=0.7)
        
        ax2.set_xlabel('Time (minutes)', fontweight='bold', fontsize=14)
        ax2.set_ylabel(f'Distance per Minute ({dist_display_unit})', fontweight='bold', fontsize=12)
        ax2.set_title('Mean Distance Traveled per Minute', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax2.set_xticks(np.arange(0, np.ceil(max_time) + 1, 1))
    
    plt.tight_layout()
    
    if save_dir:
        # Create the directory if it doesn't exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        save_file = save_path / 'cumulative_distance_with_stats_plot.png'
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved enhanced cumulative distance plot to {save_file}[/green]")
    
    plt.show()


def save_cumulative_distance_csv(df: pd.DataFrame, output_path: str):
    """
    Calculate and save minute-by-minute cumulative distance for all fish to CSV.
    
    The output CSV will have columns:
    - minute: Time point in minutes
    - roi_id: Fish ID (0-11)
    - group: Group number (1 or 2)
    - cumulative_distance_px: Cumulative distance in pixels
    - cumulative_distance_cm: Cumulative distance in cm (if calibrated)
    - cumulative_distance_m: Cumulative distance in meters (if calibrated)
    """
    
    # Check if we have calibrated data
    use_cm = 'total_distance_cm' in df.columns
    dist_col = 'total_distance_cm' if use_cm else 'total_distance_px'
    
    # Filter valid data
    valid_df = df[df['detection_rate'] > 0.5]
    
    # Prepare output data
    output_data = []
    
    # Process each fish
    for roi_id in sorted(valid_df['roi_id'].unique()):
        fish_df = valid_df[valid_df['roi_id'] == roi_id].sort_values('minute')
        
        # Calculate cumulative distance
        cumulative_px = fish_df['total_distance_px'].cumsum() if 'total_distance_px' in fish_df.columns else None
        cumulative_cm = fish_df['total_distance_cm'].cumsum() if 'total_distance_cm' in fish_df.columns else None
        
        # Add each time point
        for idx, (_, row) in enumerate(fish_df.iterrows()):
            record = {
                'minute': row['minute'],
                'roi_id': roi_id,
                'group': row['group']
            }
            
            # Add distance data
            if cumulative_px is not None:
                record['cumulative_distance_px'] = cumulative_px.iloc[idx]
            
            if cumulative_cm is not None:
                record['cumulative_distance_cm'] = cumulative_cm.iloc[idx]
                record['cumulative_distance_m'] = cumulative_cm.iloc[idx] / 100  # Convert to meters
            
            output_data.append(record)
    
    # Create DataFrame and save
    output_df = pd.DataFrame(output_data)
    output_df = output_df.sort_values(['minute', 'roi_id'])
    
    # Save to CSV
    output_df.to_csv(output_path, index=False, float_format='%.3f')
    console.print(f"[green]✓ Saved cumulative distance data to {output_path}[/green]")
    
    # Print summary
    console.print(f"[cyan]Data shape: {len(output_df)} rows x {len(output_df.columns)} columns[/cyan]")
    console.print(f"[cyan]Time points: {output_df['minute'].nunique()} minutes[/cyan]")
    console.print(f"[cyan]Fish tracked: {output_df['roi_id'].nunique()} individuals[/cyan]")
    
    return output_df


def save_group_mean_cumulative_csv(df: pd.DataFrame, output_path: str):
    """
    Calculate and save group mean cumulative distances to CSV.
    
    The output CSV will have columns:
    - minute: Time point in minutes
    - group1_mean_distance: Group 1 mean cumulative distance
    - group1_sem: Group 1 standard error of mean
    - group2_mean_distance: Group 2 mean cumulative distance  
    - group2_sem: Group 2 standard error of mean
    - units: Distance units (pixels or meters)
    """
    
    # Check if we have calibrated data
    use_cm = 'total_distance_cm' in df.columns
    dist_col = 'total_distance_cm' if use_cm else 'total_distance_px'
    
    # Filter valid data
    valid_df = df[df['detection_rate'] > 0.5]
    
    # Get unique times
    unique_times = sorted(valid_df['minute'].unique())
    
    # Prepare output data
    output_data = []
    
    for time_point in unique_times:
        record = {'minute': time_point}
        
        # Calculate for each group
        for group in [1, 2]:
            group_df = valid_df[valid_df['group'] == group]
            
            # Get cumulative distance for each fish up to this time
            cumulative_per_fish = []
            for roi_id in group_df['roi_id'].unique():
                fish_data = group_df[(group_df['roi_id'] == roi_id) & 
                                    (group_df['minute'] <= time_point)]
                if len(fish_data) > 0:
                    total = fish_data[dist_col].sum()
                    if use_cm:
                        total = total / 100  # Convert to meters
                    cumulative_per_fish.append(total)
            
            if cumulative_per_fish:
                mean_dist = np.mean(cumulative_per_fish)
                sem_dist = np.std(cumulative_per_fish) / np.sqrt(len(cumulative_per_fish))
                record[f'group{group}_mean_distance'] = mean_dist
                record[f'group{group}_sem'] = sem_dist
                record[f'group{group}_n_fish'] = len(cumulative_per_fish)
            else:
                record[f'group{group}_mean_distance'] = 0
                record[f'group{group}_sem'] = 0
                record[f'group{group}_n_fish'] = 0
        
        record['units'] = 'meters' if use_cm else 'pixels'
        output_data.append(record)
    
    # Create DataFrame and save
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_path, index=False, float_format='%.3f')
    
    console.print(f"[green]✓ Saved group mean cumulative distances to {output_path}[/green]")
    console.print(f"[cyan]Data shape: {len(output_df)} time points[/cyan]")
    
    return output_df


def save_cumulative_distance_csv(df: pd.DataFrame, output_path: str):
    """
    Calculate and save minute-by-minute cumulative distance for all fish to CSV.
    
    The output CSV will have columns:
    - minute: Time point in minutes
    - roi_id: Fish ID (0-11)
    - group: Group number (1 or 2)
    - cumulative_distance_px: Cumulative distance in pixels
    - cumulative_distance_cm: Cumulative distance in cm (if calibrated)
    - cumulative_distance_m: Cumulative distance in meters (if calibrated)
    """
    
    # Check if we have calibrated data
    use_cm = 'total_distance_cm' in df.columns
    dist_col = 'total_distance_cm' if use_cm else 'total_distance_px'
    
    # Filter valid data
    valid_df = df[df['detection_rate'] > 0.5]
    
    # Prepare output data
    output_data = []
    
    # Process each fish
    for roi_id in sorted(valid_df['roi_id'].unique()):
        fish_df = valid_df[valid_df['roi_id'] == roi_id].sort_values('minute')
        
        # Calculate cumulative distance
        cumulative_px = fish_df['total_distance_px'].cumsum() if 'total_distance_px' in fish_df.columns else None
        cumulative_cm = fish_df['total_distance_cm'].cumsum() if 'total_distance_cm' in fish_df.columns else None
        
        # Add each time point
        for idx, (_, row) in enumerate(fish_df.iterrows()):
            record = {
                'minute': row['minute'],
                'roi_id': roi_id,
                'group': row['group']
            }
            
            # Add distance data
            if cumulative_px is not None:
                record['cumulative_distance_px'] = cumulative_px.iloc[idx]
            
            if cumulative_cm is not None:
                record['cumulative_distance_cm'] = cumulative_cm.iloc[idx]
                record['cumulative_distance_m'] = cumulative_cm.iloc[idx] / 100  # Convert to meters
            
            output_data.append(record)
    
    # Create DataFrame and save
    output_df = pd.DataFrame(output_data)
    output_df = output_df.sort_values(['minute', 'roi_id'])
    
    # Save to CSV
    output_df.to_csv(output_path, index=False, float_format='%.3f')
    console.print(f"[green]✓ Saved cumulative distance data to {output_path}[/green]")
    
    # Print summary
    console.print(f"[cyan]Data shape: {len(output_df)} rows x {len(output_df.columns)} columns[/cyan]")
    console.print(f"[cyan]Time points: {output_df['minute'].nunique()} minutes[/cyan]")
    console.print(f"[cyan]Fish tracked: {output_df['roi_id'].nunique()} individuals[/cyan]")
    
    return output_df


def save_group_mean_cumulative_csv(df: pd.DataFrame, output_path: str):
    """
    Calculate and save group mean cumulative distances to CSV.
    
    The output CSV will have columns:
    - minute: Time point in minutes
    - group1_mean_distance: Group 1 mean cumulative distance
    - group1_sem: Group 1 standard error of mean
    - group2_mean_distance: Group 2 mean cumulative distance  
    - group2_sem: Group 2 standard error of mean
    - units: Distance units (pixels or meters)
    """
    
    # Check if we have calibrated data
    use_cm = 'total_distance_cm' in df.columns
    dist_col = 'total_distance_cm' if use_cm else 'total_distance_px'
    
    # Filter valid data
    valid_df = df[df['detection_rate'] > 0.5]
    
    # Get unique times
    unique_times = sorted(valid_df['minute'].unique())
    
    # Prepare output data
    output_data = []
    
    for time_point in unique_times:
        record = {'minute': time_point}
        
        # Calculate for each group
        for group in [1, 2]:
            group_df = valid_df[valid_df['group'] == group]
            
            # Get cumulative distance for each fish up to this time
            cumulative_per_fish = []
            for roi_id in group_df['roi_id'].unique():
                fish_data = group_df[(group_df['roi_id'] == roi_id) & 
                                    (group_df['minute'] <= time_point)]
                if len(fish_data) > 0:
                    total = fish_data[dist_col].sum()
                    if use_cm:
                        total = total / 100  # Convert to meters
                    cumulative_per_fish.append(total)
            
            if cumulative_per_fish:
                mean_dist = np.mean(cumulative_per_fish)
                sem_dist = np.std(cumulative_per_fish) / np.sqrt(len(cumulative_per_fish))
                record[f'group{group}_mean_distance'] = mean_dist
                record[f'group{group}_sem'] = sem_dist
                record[f'group{group}_n_fish'] = len(cumulative_per_fish)
            else:
                record[f'group{group}_mean_distance'] = 0
                record[f'group{group}_sem'] = 0
                record[f'group{group}_n_fish'] = 0
        
        record['units'] = 'meters' if use_cm else 'pixels'
        output_data.append(record)
    
    # Create DataFrame and save
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_path, index=False, float_format='%.3f')
    
    console.print(f"[green]✓ Saved group mean cumulative distances to {output_path}[/green]")
    console.print(f"[cyan]Data shape: {len(output_df)} time points[/cyan]")
    
    return output_df


def main():
    parser = argparse.ArgumentParser(
        description='Extract and plot minute-by-minute swimming speed',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file with detections')
    parser.add_argument('h5_path', help='Path to H5 file with metadata')
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--bin-minutes', type=float, default=1.0,
                       help='Bin size in minutes (default: 1.0)')
    parser.add_argument('--activity-threshold', type=float, default=10.0,
                       help='Speed threshold for activity (px/s, default: 10.0)')
    parser.add_argument('--plot', action='store_true',
                       help='Create standard plots after extraction')
    parser.add_argument('--plot-individual', action='store_true',
                       help='Create individual cumulative distance plot')
    parser.add_argument('--plot-heatmap', action='store_true',
                       help='Create activity heatmap')
    parser.add_argument('--plot-all', action='store_true',
                       help='Create all available plots')
    parser.add_argument('--plot-cumulative-only', action='store_true',
                       help='Create only the cumulative distance plot')
    parser.add_argument('--save-dir', type=str,
                       help='Directory to save plots')
    parser.add_argument('--save-cumulative-csv', type=str,
                       help='Save cumulative distance data to CSV file')
    parser.add_argument('--save-group-means-csv', type=str,
                       help='Save group mean cumulative distances to CSV file')
    
    args = parser.parse_args()
    
    # Create extractor
    console.print("[bold]Minute-by-Minute Speed Extractor[/bold]")
    
    extractor = MinuteSpeedExtractor(
        zarr_path=args.zarr_path,
        h5_path=args.h5_path,
        bin_minutes=args.bin_minutes,
        activity_threshold=args.activity_threshold
    )
    
    # Extract speed data
    console.print("\n[cyan]Extracting speed data...[/cyan]")
    df = extractor.extract_minute_speeds()
    
    # Save results
    df.to_csv(args.output, index=False)
    console.print(f"[green]✓ Saved speed data to {args.output}[/green]")
    
    # Print statistics
    print_speed_statistics(df)
    
    # Create plots based on flags
    plots_to_create = []
    
    if args.plot_all:
        plots_to_create = ['standard', 'individual', 'heatmap']
    else:
        if args.plot:
            plots_to_create.append('standard')
        if args.plot_individual:
            plots_to_create.append('individual')
        if args.plot_heatmap:
            plots_to_create.append('heatmap')
    
    # Handle cumulative-only plot
    if args.plot_cumulative_only:
        plots_to_create.append('cumulative_only')
    
    # Create requested plots
    if 'standard' in plots_to_create:
        console.print("\n[cyan]Creating speed and distance plots...[/cyan]")
        plot_minute_speed_scatter(df, args.save_dir)
    
    if 'individual' in plots_to_create:
        console.print("\n[cyan]Creating individual cumulative distance plot...[/cyan]")
        plot_individual_cumulative_distance(df, args.save_dir)
    
    if 'heatmap' in plots_to_create:
        console.print("\n[cyan]Creating activity heatmap...[/cyan]")
        plot_activity_heatmap(df, args.save_dir)
    
    if 'cumulative_only' in plots_to_create:
        console.print("\n[cyan]Creating cumulative distance plot...[/cyan]")
        plot_cumulative_distance_only(df, args.save_dir)
    
    # Save cumulative distance CSVs if requested
    if args.save_cumulative_csv:
        console.print("\n[cyan]Saving cumulative distance data to CSV...[/cyan]")
        save_cumulative_distance_csv(df, args.save_cumulative_csv)
    
    if args.save_group_means_csv:
        console.print("\n[cyan]Saving group mean cumulative distances to CSV...[/cyan]")
        save_group_mean_cumulative_csv(df, args.save_group_means_csv)
    
    if plots_to_create:
        console.print(f"\n[green]✓ All requested plots created![/green]")
        if args.save_dir:
            console.print(f"[green]✓ Plots saved to {args.save_dir}/[/green]")
    
    console.print("\n[green]✓ Complete![/green]")


if __name__ == '__main__':
    main()