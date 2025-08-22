#!/usr/bin/env python3
"""
Trial Analyzer with Zarr Detection Data (Enhanced with Smoothed Speed)

Combines trial information from H5 files with cleaned detection data from zarr files.
Now includes smoothed speed calculations to reduce noise in behavioral metrics.
"""

import h5py
import zarr
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from scipy import signal

# Event type mappings
EXPERIMENT_EVENT_TYPE = {
    27: "CHASER_CHASE_SEQUENCE_START",
    28: "CHASER_CHASE_SEQUENCE_END"
}

class TrialZarrAnalyzer:
    """Analyze trials using H5 metadata and zarr detection data with smoothed speeds."""
    
    def __init__(self, h5_path: str, zarr_path: str, use_interpolated: bool = True,
                 speed_window: int = 5, escape_threshold: float = 500.0):
        """
        Initialize analyzer.
        
        Args:
            h5_path: Path to H5 analysis file
            zarr_path: Path to zarr detection file
            use_interpolated: Whether to use interpolated data
            speed_window: Window size for speed smoothing (frames)
            escape_threshold: Speed threshold for escape detection (px/s)
        """
        self.h5_path = Path(h5_path)
        self.zarr_path = Path(zarr_path)
        self.use_interpolated = use_interpolated
        self.speed_window = speed_window
        self.escape_threshold = escape_threshold
        
        # Load zarr data
        self.load_zarr_data()
        
    def load_zarr_data(self):
        """Load detection data from zarr file."""
        print(f"\n{'='*70}")
        print(f"Loading zarr data from: {self.zarr_path}")
        print(f"{'='*70}")
        
        self.zarr_root = zarr.open(str(self.zarr_path), mode='r')
        
        # Determine which data to use
        if self.use_interpolated:
            # Try to use the best preprocessed data available
            if 'preprocessing' in self.zarr_root and 'latest' in self.zarr_root['preprocessing'].attrs:
                latest = self.zarr_root['preprocessing'].attrs['latest']
                self.data = self.zarr_root['preprocessing'][latest]
                self.interpolation_mask = self.data.get('interpolation_mask', None)
                print(f"✓ Using preprocessed data: preprocessing/{latest}")
            elif 'filtered_runs' in self.zarr_root and 'latest' in self.zarr_root['filtered_runs'].attrs:
                latest = self.zarr_root['filtered_runs'].attrs['latest']
                self.data = self.zarr_root['filtered_runs'][latest]
                self.interpolation_mask = self.data.get('interpolation_mask', None)
                print(f"✓ Using filtered data: filtered_runs/{latest}")
            else:
                self.data = self.zarr_root
                self.interpolation_mask = None
                print("⚠ No preprocessed data found, using original detections")
        else:
            self.data = self.zarr_root
            self.interpolation_mask = None
            print("✓ Using original detections (--use-original flag)")
        
        # Load arrays
        self.n_detections = self.data['n_detections'][:]
        self.bboxes = self.data['bboxes'][:]
        self.scores = self.data['scores'][:]
        
        # Get metadata
        self.fps = self.zarr_root.attrs.get('fps', 60.0)
        self.img_width = self.zarr_root.attrs.get('img_width', 2592)
        self.img_height = self.zarr_root.attrs.get('img_height', 1944)
        self.total_frames = len(self.n_detections)
        
        # Calculate coverage
        frames_with_detection = np.sum(self.n_detections > 0)
        self.coverage = frames_with_detection / self.total_frames * 100
        
        # Get calibration if available
        self.pixel_to_mm = None
        if 'calibration' in self.zarr_root:
            self.pixel_to_mm = self.zarr_root['calibration'].attrs.get('pixel_to_mm', None)
            if self.pixel_to_mm:
                print(f"✓ Calibration loaded: 1 pixel = {self.pixel_to_mm:.4f} mm")
        
        print(f"\nData summary:")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Frames with detection: {frames_with_detection} ({self.coverage:.1f}%)")
        print(f"  FPS: {self.fps}")
        print(f"  Image size: {self.img_width}x{self.img_height}")
        
        if self.interpolation_mask is not None:
            interp_frames = np.sum(self.interpolation_mask[:])
            print(f"  Interpolated frames: {interp_frames} ({interp_frames/self.total_frames*100:.1f}%)")
    
    def calculate_centroid(self, bbox):
        """Calculate centroid of bounding box."""
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    
    def calculate_smoothed_speed(self, centroids: np.ndarray, frame_indices: np.ndarray) -> Dict:
        """
        Calculate both instantaneous and smoothed speed from centroid data.
        
        Args:
            centroids: Array of centroid positions
            frame_indices: Array of frame indices for each centroid
            
        Returns:
            Dictionary with speed metrics
        """
        if len(centroids) < 2:
            return {
                'instantaneous_speed': np.array([]),
                'smoothed_speed': np.array([]),
                'mean_speed': 0,
                'max_speed': 0,
                'escape_frames': []
            }
        
        # Calculate frame-to-frame distances
        distances = np.zeros(len(centroids))
        distances[1:] = np.linalg.norm(centroids[1:] - centroids[:-1], axis=1)
        
        # Calculate time gaps between frames
        frame_gaps = np.ones(len(frame_indices))
        frame_gaps[1:] = frame_indices[1:] - frame_indices[:-1]
        
        # Calculate instantaneous speed (px/s)
        instantaneous_speed = np.zeros(len(centroids))
        instantaneous_speed[1:] = distances[1:] * self.fps / frame_gaps[1:]
        
        # Apply reasonable speed threshold
        max_reasonable_speed = 1000.0  # px/s
        instantaneous_speed[instantaneous_speed > max_reasonable_speed] = np.nan
        
        # Calculate smoothed speed
        smoothed_speed = self.smooth_speed(instantaneous_speed)
        
        # Detect escape events (high speed + actual displacement)
        escape_frames = []
        if len(smoothed_speed) > self.speed_window:
            for i in range(self.speed_window, len(smoothed_speed)):
                # Check for high speed
                if smoothed_speed[i] > self.escape_threshold:
                    # Verify actual displacement over window
                    window_start = max(0, i - self.speed_window)
                    displacement = np.linalg.norm(centroids[i] - centroids[window_start])
                    if displacement > 20:  # Minimum displacement threshold
                        escape_frames.append(i)
        
        return {
            'instantaneous_speed': instantaneous_speed,
            'smoothed_speed': smoothed_speed,
            'mean_speed': np.nanmean(smoothed_speed),
            'max_speed': np.nanmax(smoothed_speed) if len(smoothed_speed) > 0 else 0,
            'escape_frames': escape_frames
        }
    
    def smooth_speed(self, speed: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to speed data.
        
        Args:
            speed: Instantaneous speed array
            
        Returns:
            Smoothed speed array
        """
        if len(speed) < self.speed_window:
            return speed
        
        smoothed = np.full_like(speed, np.nan)
        valid_mask = ~np.isnan(speed)
        
        if np.sum(valid_mask) > self.speed_window:
            # Apply moving average to valid data
            valid_indices = np.where(valid_mask)[0]
            for idx in valid_indices:
                window_start = max(0, idx - self.speed_window // 2)
                window_end = min(len(speed), idx + self.speed_window // 2 + 1)
                window_data = speed[window_start:window_end]
                
                valid_in_window = window_data[~np.isnan(window_data)]
                if len(valid_in_window) > 0:
                    smoothed[idx] = np.mean(valid_in_window)
        
        return smoothed
    
    def analyze_trials(self, visualize: bool = False, trial_numbers: List[int] = None) -> List[Dict]:
        """
        Analyze all trials in the H5 file using zarr detection data.
        
        Args:
            visualize: Whether to create visualizations
            
        Returns:
            List of trial analysis results
        """
        print(f"\n{'='*70}")
        print("ANALYZING TRIALS")
        print(f"{'='*70}")
        
        trial_results = []
        
        with h5py.File(self.h5_path, 'r') as h5f:
            # Try different event dataset names
            events_dataset = None
            event_path = None
            
            # Check for different possible event dataset names
            for path in ['/events', '/experiment_events']:
                if path in h5f:
                    events_dataset = h5f[path]
                    event_path = path
                    print(f"✓ Found events at: {path}")
                    break
            
            if events_dataset is None:
                print("⚠ No events dataset found in H5 file")
                return []
            
            # Load events data
            events = events_dataset[:]
            
            # Find the event type field (could be different names)
            event_type_field = None
            for field_name in ['event_type_id', 'event_type', 'type']:
                if field_name in events.dtype.names:
                    event_type_field = field_name
                    print(f"✓ Using event type field: {field_name}")
                    break
            
            if event_type_field is None:
                print("⚠ Could not find event type field in events dataset")
                return []
            
            # Find timestamp field
            timestamp_field = None
            for field_name in ['timestamp_ns_session', 'timestamp_ns', 'time', 'timestamp']:
                if field_name in events.dtype.names:
                    timestamp_field = field_name
                    print(f"✓ Using timestamp field: {field_name}")
                    break
            
            if timestamp_field is None:
                print("⚠ Could not find timestamp field in events dataset")
                return []
            
            # Extract event types and times
            event_types = events[event_type_field]
            event_times_raw = events[timestamp_field]
            
            # Convert timestamps to seconds if they're in nanoseconds
            if 'ns' in timestamp_field:
                event_times = event_times_raw / 1e9  # Convert ns to seconds
            else:
                event_times = event_times_raw
            
            # Find chase sequence events
            chase_starts = np.where(event_types == 27)[0]
            chase_ends = np.where(event_types == 28)[0]
            
            if len(chase_starts) == 0:
                print("⚠ No chase sequences found")
                return []
            
            print(f"\nFound {len(chase_starts)} chase start events")
            print(f"Found {len(chase_ends)} chase end events")
            
            # Handle mismatch between starts and ends
            if len(chase_starts) != len(chase_ends):
                print(f"⚠ Warning: Mismatch between starts ({len(chase_starts)}) and ends ({len(chase_ends)})")
                print("  Will analyze complete pairs only")
                
                # Match starts with ends - only process complete sequences
                matched_pairs = []
                for start_idx in chase_starts:
                    # Find the next end after this start
                    matching_ends = chase_ends[chase_ends > start_idx]
                    if len(matching_ends) > 0:
                        end_idx = matching_ends[0]
                        matched_pairs.append((start_idx, end_idx))
                        # Remove this end from future matching
                        chase_ends = chase_ends[chase_ends != end_idx]
                
                print(f"  Matched {len(matched_pairs)} complete chase sequences")
            else:
                # All starts have corresponding ends
                matched_pairs = list(zip(chase_starts, chase_ends))
            
            if len(matched_pairs) == 0:
                print("⚠ No complete chase sequences to analyze")
                return []
            
            # Process each trial
            for trial_num, (start_idx, end_idx) in enumerate(matched_pairs, 1):
                start_time = event_times[start_idx]
                end_time = event_times[end_idx]
                duration_s = end_time - start_time
                
                # Skip if duration is invalid
                if duration_s <= 0:
                    print(f"⚠ Skipping trial {trial_num}: invalid duration ({duration_s:.2f}s)")
                    continue
                
                # Convert times to frame indices
                start_frame = int(start_time * self.fps)
                end_frame = int(end_time * self.fps)
                trial_frames = end_frame - start_frame + 1
                
                print(f"\n--- Trial {trial_num} ---")
                print(f"  Time: {start_time:.2f}s to {end_time:.2f}s")
                print(f"  Duration: {duration_s:.2f}s ({trial_frames} frames)")
                print(f"  Frame range: {start_frame} to {end_frame}")
                
                # Get detection data for this trial
                trial_n_detections = self.n_detections[start_frame:end_frame+1]
                trial_bboxes = self.bboxes[start_frame:end_frame+1]
                
                # Calculate trial statistics
                frames_with_detection = np.sum(trial_n_detections > 0)
                trial_coverage = frames_with_detection / trial_frames * 100
                
                print(f"  Detection coverage: {trial_coverage:.1f}% ({frames_with_detection}/{trial_frames} frames)")
                
                # Check interpolation if available
                interpolated_frames = 0
                if self.interpolation_mask is not None:
                    trial_interp_mask = self.interpolation_mask[start_frame:end_frame+1]
                    interpolated_frames = np.sum(trial_interp_mask)
                    real_frames = frames_with_detection - interpolated_frames
                    print(f"  Real detections: {real_frames} frames")
                    print(f"  Interpolated: {interpolated_frames} frames")
                
                # Extract centroids and calculate trajectory
                centroids = []
                frame_indices = []
                
                for j in range(len(trial_n_detections)):
                    if trial_n_detections[j] > 0:
                        centroid = self.calculate_centroid(trial_bboxes[j, 0])
                        centroids.append(centroid)
                        frame_indices.append(j)
                
                if len(centroids) > 1:
                    centroids = np.array(centroids)
                    frame_indices = np.array(frame_indices)
                    
                    # Calculate speed metrics
                    speed_metrics = self.calculate_smoothed_speed(centroids, frame_indices)
                    
                    # Calculate total distance
                    distances = np.linalg.norm(centroids[1:] - centroids[:-1], axis=1)
                    total_distance = np.sum(distances)
                    
                    print(f"  Total distance: {total_distance:.0f} px")
                    if self.pixel_to_mm:
                        print(f"                 ({total_distance * self.pixel_to_mm:.1f} mm)")
                    
                    print(f"  Avg speed: {speed_metrics['mean_speed']:.0f} px/s")
                    if self.pixel_to_mm:
                        print(f"           ({speed_metrics['mean_speed'] * self.pixel_to_mm:.1f} mm/s)")
                    
                    print(f"  Max speed: {speed_metrics['max_speed']:.0f} px/s")
                    if len(speed_metrics['escape_frames']) > 0:
                        print(f"  Escape events: {len(speed_metrics['escape_frames'])}")
                    
                    # Store results
                    trial_results.append({
                        'trial_num': trial_num,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration_s': duration_s,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'trial_frames': trial_frames,
                        'coverage': trial_coverage,
                        'interpolated_frames': interpolated_frames,
                        'centroids': centroids,
                        'frame_indices': frame_indices,
                        'n_detections': trial_n_detections,
                        'total_distance': total_distance,
                        'avg_speed': speed_metrics['mean_speed'],
                        'max_speed': speed_metrics['max_speed'],
                        'instantaneous_speed': speed_metrics['instantaneous_speed'],
                        'smoothed_speed': speed_metrics['smoothed_speed'],
                        'escape_frames': speed_metrics['escape_frames']
                    })
                else:
                    print(f"  ⚠ Insufficient detections for trajectory analysis")
        
        # Visualize if requested
        if visualize and trial_results:
            # Filter to specific trials if requested
            if trial_numbers:
                filtered_results = [t for t in trial_results if t['trial_num'] in trial_numbers]
                if filtered_results:
                    print(f"\nVisualizing trial(s): {[t['trial_num'] for t in filtered_results]}")
                    self.visualize_trials(filtered_results)
                else:
                    print(f"⚠ Requested trial(s) {trial_numbers} not found in results")
                    print(f"  Available trials: {[t['trial_num'] for t in trial_results]}")
            else:
                self.visualize_trials(trial_results)
        
        return trial_results
    
    def visualize_trials(self, trial_results: List[Dict]):
        """Create enhanced visualization of trial trajectories with speed profiles."""
        n_trials = len(trial_results)
        
        # For single trial, make a larger, more detailed plot
        if n_trials == 1:
            fig = plt.figure(figsize=(14, 7))
            fig.suptitle(f'Chase Trial {trial_results[0]["trial_num"]} - Detailed Analysis', 
                        fontsize=16, fontweight='bold')
            
            trial = trial_results[0]
            
            # Left: Trajectory
            ax1 = plt.subplot(1, 2, 1)
            
            centroids = trial['centroids']
            smoothed_speed = trial['smoothed_speed']
            
            # Plot trajectory colored by speed
            if len(smoothed_speed) > 0:
                # Create speed colormap
                speeds_for_color = smoothed_speed[~np.isnan(smoothed_speed)]
                if len(speeds_for_color) > 0:
                    vmin, vmax = np.percentile(speeds_for_color, [5, 95])
                    
                    # Plot trajectory segments colored by speed
                    for j in range(len(centroids) - 1):
                        if not np.isnan(smoothed_speed[j]):
                            color_val = np.clip((smoothed_speed[j] - vmin) / (vmax - vmin), 0, 1)
                            ax1.plot(centroids[j:j+2, 0], centroids[j:j+2, 1],
                                   color=plt.cm.viridis(color_val), linewidth=3, alpha=0.7)
                    
                    # Add colorbar
                    sm = plt.cm.ScalarMappable(cmap='viridis', 
                                              norm=plt.Normalize(vmin=vmin, vmax=vmax))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
                    cbar.set_label('Speed (px/s)', fontsize=10)
            
            # Mark start and stop
            ax1.plot(centroids[0, 0], centroids[0, 1], 'g^', 
                   markersize=15, markeredgecolor='darkgreen', 
                   markeredgewidth=2, zorder=3, label='START')
            ax1.plot(centroids[-1, 0], centroids[-1, 1], 'rs', 
                   markersize=15, markeredgecolor='darkred', 
                   markeredgewidth=2, zorder=3, label='STOP')
            
            # Mark escape events
            if len(trial['escape_frames']) > 0:
                escape_positions = centroids[trial['escape_frames']]
                ax1.scatter(escape_positions[:, 0], escape_positions[:, 1],
                          c='red', s=100, marker='*', edgecolor='darkred',
                          linewidth=1, zorder=4, label='Escape', alpha=0.8)
            
            ax1.set_xlim(0, self.img_width)
            ax1.set_ylim(0, self.img_height)
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best', fontsize=10)
            ax1.set_title(f"Trajectory (Duration: {trial['duration_s']:.2f}s)", fontsize=12)
            ax1.set_xlabel('X Position (pixels)', fontsize=11)
            ax1.set_ylabel('Y Position (pixels)', fontsize=11)
            
            # Right: Speed profile
            ax2 = plt.subplot(1, 2, 2)
            
            if len(trial['instantaneous_speed']) > 0:
                time_points = trial['frame_indices'] / self.fps
                
                # Plot instantaneous speed (faint)
                ax2.plot(time_points, trial['instantaneous_speed'],
                        'gray', alpha=0.3, linewidth=1, label='Raw')
                
                # Plot smoothed speed (bold)
                ax2.plot(time_points, trial['smoothed_speed'],
                        'b-', linewidth=2.5, alpha=0.8, label='Smoothed')
                
                # Mark escape threshold
                ax2.axhline(y=self.escape_threshold, color='r', linestyle='--',
                          alpha=0.5, label=f'Escape threshold')
                
                # Mark escape events
                if len(trial['escape_frames']) > 0:
                    escape_times = trial['frame_indices'][trial['escape_frames']] / self.fps
                    escape_speeds = trial['smoothed_speed'][trial['escape_frames']]
                    ax2.scatter(escape_times, escape_speeds,
                              c='red', s=100, marker='*', edgecolor='darkred',
                              linewidth=1, zorder=4, alpha=0.8)
                
                ax2.set_xlabel('Time in trial (s)', fontsize=11)
                ax2.set_ylabel('Speed (px/s)', fontsize=11)
                ax2.set_title(f"Speed Profile (window={self.speed_window} frames)", fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='best', fontsize=10)
                
                # Add detailed statistics text
                stats_text = (f"Mean: {trial['avg_speed']:.0f} px/s\n"
                            f"Max: {trial['max_speed']:.0f} px/s\n"
                            f"Coverage: {trial['coverage']:.1f}%\n"
                            f"Distance: {trial['total_distance']:.0f} px")
                if self.pixel_to_mm:
                    stats_text += (f"\n\nMean: {trial['avg_speed'] * self.pixel_to_mm:.1f} mm/s\n"
                                 f"Max: {trial['max_speed'] * self.pixel_to_mm:.1f} mm/s\n"
                                 f"Distance: {trial['total_distance'] * self.pixel_to_mm:.1f} mm")
                if trial['interpolated_frames'] > 0:
                    stats_text += f"\n\nInterp: {trial['interpolated_frames']} frames"
                if len(trial['escape_frames']) > 0:
                    stats_text += f"\nEscapes: {len(trial['escape_frames'])}"
                
                ax2.text(0.98, 0.98, stats_text,
                        transform=ax2.transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.show()
            return
        
        # Multiple trials - use grid layout
        n_cols = min(2, n_trials)
        n_rows = (n_trials + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(12, 6*n_rows))
        fig.suptitle('Chase Trial Analysis with Smoothed Speed Profiles', 
                     fontsize=16, fontweight='bold')
        
        for i, trial in enumerate(trial_results):
            # Create subplot with 2 panels per trial
            # Left: Trajectory
            ax1 = plt.subplot(n_rows, n_cols*2, i*2 + 1)
            
            centroids = trial['centroids']
            smoothed_speed = trial['smoothed_speed']
            
            # Plot trajectory colored by speed
            if len(smoothed_speed) > 0:
                # Create speed colormap
                speeds_for_color = smoothed_speed[~np.isnan(smoothed_speed)]
                if len(speeds_for_color) > 0:
                    vmin, vmax = np.percentile(speeds_for_color, [5, 95])
                    
                    # Plot trajectory segments colored by speed
                    for j in range(len(centroids) - 1):
                        if not np.isnan(smoothed_speed[j]):
                            color_val = np.clip((smoothed_speed[j] - vmin) / (vmax - vmin), 0, 1)
                            ax1.plot(centroids[j:j+2, 0], centroids[j:j+2, 1],
                                   color=plt.cm.viridis(color_val), linewidth=2, alpha=0.7)
                    
                    # Add colorbar
                    sm = plt.cm.ScalarMappable(cmap='viridis', 
                                              norm=plt.Normalize(vmin=vmin, vmax=vmax))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
                    cbar.set_label('Speed (px/s)', fontsize=8)
            
            # Mark start and stop
            ax1.plot(centroids[0, 0], centroids[0, 1], 'g^', 
                   markersize=10, markeredgecolor='darkgreen', 
                   markeredgewidth=1.5, zorder=3, label='START')
            ax1.plot(centroids[-1, 0], centroids[-1, 1], 'rs', 
                   markersize=10, markeredgecolor='darkred', 
                   markeredgewidth=1.5, zorder=3, label='STOP')
            
            # Mark escape events
            if len(trial['escape_frames']) > 0:
                escape_positions = centroids[trial['escape_frames']]
                ax1.scatter(escape_positions[:, 0], escape_positions[:, 1],
                          c='red', s=50, marker='*', edgecolor='darkred',
                          linewidth=1, zorder=4, label='Escape', alpha=0.8)
            
            ax1.set_xlim(0, self.img_width)
            ax1.set_ylim(0, self.img_height)
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best', fontsize=8)
            ax1.set_title(f"Trial {trial['trial_num']} - Trajectory", fontsize=10)
            ax1.set_xlabel('X Position (pixels)', fontsize=9)
            ax1.set_ylabel('Y Position (pixels)', fontsize=9)
            
            # Right: Speed profile
            ax2 = plt.subplot(n_rows, n_cols*2, i*2 + 2)
            
            if len(trial['instantaneous_speed']) > 0:
                time_points = trial['frame_indices'] / self.fps
                
                # Plot instantaneous speed (faint)
                ax2.plot(time_points, trial['instantaneous_speed'],
                        'gray', alpha=0.3, linewidth=0.5, label='Raw')
                
                # Plot smoothed speed (bold)
                ax2.plot(time_points, trial['smoothed_speed'],
                        'b-', linewidth=2, alpha=0.8, label='Smoothed')
                
                # Mark escape threshold
                ax2.axhline(y=self.escape_threshold, color='r', linestyle='--',
                          alpha=0.5, label=f'Escape threshold')
                
                # Mark escape events
                if len(trial['escape_frames']) > 0:
                    escape_times = trial['frame_indices'][trial['escape_frames']] / self.fps
                    escape_speeds = trial['smoothed_speed'][trial['escape_frames']]
                    ax2.scatter(escape_times, escape_speeds,
                              c='red', s=50, marker='*', edgecolor='darkred',
                              linewidth=1, zorder=4, alpha=0.8)
                
                ax2.set_xlabel('Time in trial (s)', fontsize=9)
                ax2.set_ylabel('Speed (px/s)', fontsize=9)
                ax2.set_title(f"Speed Profile (window={self.speed_window} frames)", fontsize=10)
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='best', fontsize=8)
                
                # Add statistics text
                stats_text = (f"Mean: {trial['avg_speed']:.0f} px/s\n"
                            f"Max: {trial['max_speed']:.0f} px/s\n"
                            f"Coverage: {trial['coverage']:.1f}%")
                if trial['interpolated_frames'] > 0:
                    stats_text += f"\nInterp: {trial['interpolated_frames']} frames"
                
                ax2.text(0.98, 0.98, stats_text,
                        transform=ax2.transAxes,
                        fontsize=8,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Add summary statistics
        summary = self.generate_summary(trial_results)
        fig.text(0.5, 0.02, summary, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.show()
    
    def generate_summary(self, trial_results: List[Dict]) -> str:
        """Generate enhanced summary statistics for all trials."""
        coverages = [t['coverage'] for t in trial_results]
        speeds = [t['avg_speed'] for t in trial_results]
        max_speeds = [t['max_speed'] for t in trial_results]
        distances = [t['total_distance'] for t in trial_results]
        
        # Count trials with escape events
        trials_with_escapes = sum(1 for t in trial_results if len(t['escape_frames']) > 0)
        total_escapes = sum(len(t['escape_frames']) for t in trial_results)
        
        summary = (f"Summary: {len(trial_results)} trials | "
                  f"Avg Coverage: {np.mean(coverages):.1f}% | "
                  f"Avg Speed: {np.mean(speeds):.0f} px/s | "
                  f"Max Speed: {np.mean(max_speeds):.0f} px/s | "
                  f"Avg Distance: {np.mean(distances):.0f} px")
        
        if self.pixel_to_mm:
            summary += (f"\n         {np.mean(speeds) * self.pixel_to_mm:.1f} mm/s | "
                       f"{np.mean(distances) * self.pixel_to_mm:.1f} mm")
        
        if trials_with_escapes > 0:
            summary += (f"\nEscape Events: {total_escapes} total in {trials_with_escapes}/{len(trial_results)} trials "
                       f"(threshold: {self.escape_threshold:.0f} px/s)")
        
        if self.use_interpolated:
            total_interp = sum(t['interpolated_frames'] for t in trial_results)
            if total_interp > 0:
                summary += f"\nTotal Interpolated: {total_interp} frames"
        
        return summary
    
    def export_trial_data(self, trial_results: List[Dict], output_path: str):
        """Export trial analysis results to JSON."""
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = []
        for trial in trial_results:
            trial_copy = {
                'trial_num': trial['trial_num'],
                'start_time': trial['start_time'],
                'end_time': trial['end_time'],
                'duration_s': trial['duration_s'],
                'start_frame': trial['start_frame'],
                'end_frame': trial['end_frame'],
                'trial_frames': trial['trial_frames'],
                'coverage': trial['coverage'],
                'interpolated_frames': trial['interpolated_frames'],
                'total_distance': trial['total_distance'],
                'avg_speed': trial['avg_speed'],
                'max_speed': trial['max_speed'],
                'num_escape_events': len(trial['escape_frames']),
                'escape_frame_indices': trial['escape_frames'],
                # Store first and last positions instead of full trajectory
                'start_position': trial['centroids'][0].tolist() if len(trial['centroids']) > 0 else None,
                'end_position': trial['centroids'][-1].tolist() if len(trial['centroids']) > 0 else None
            }
            
            # Add calibrated values if available
            if self.pixel_to_mm:
                trial_copy['total_distance_mm'] = trial['total_distance'] * self.pixel_to_mm
                trial_copy['avg_speed_mm_s'] = trial['avg_speed'] * self.pixel_to_mm
                trial_copy['max_speed_mm_s'] = trial['max_speed'] * self.pixel_to_mm
            
            export_data.append(trial_copy)
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'h5_file': str(self.h5_path),
                'zarr_file': str(self.zarr_path),
                'use_interpolated': self.use_interpolated,
                'speed_window_frames': self.speed_window,
                'escape_threshold_px_s': self.escape_threshold,
                'total_coverage': self.coverage,
                'fps': self.fps,
                'pixel_to_mm': self.pixel_to_mm,
                'trials': export_data
            }, f, indent=2)
        
        print(f"\n✅ Exported trial data to: {output_path}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Analyze trials using H5 metadata and zarr detection data with smoothed speeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool combines:
- Trial structure and timing from H5 files
- Cleaned/interpolated detection data from zarr files
- Smoothed speed calculations to reduce noise

The zarr data should be the output from:
1. frame_distance_analyzer.py (filtering)
2. gap_interpolator.py (interpolation)

Examples:
  # Analyze using interpolated data with default smoothing
  %(prog)s analysis.h5 detections.zarr --visualize
  
  # Visualize only specific trial(s)
  %(prog)s analysis.h5 detections.zarr --visualize --trial 5
  %(prog)s analysis.h5 detections.zarr --visualize --trial 1 3 5
  
  # Use custom speed smoothing window
  %(prog)s analysis.h5 detections.zarr --speed-window 7 --visualize
  
  # Adjust escape detection threshold
  %(prog)s analysis.h5 detections.zarr --escape-threshold 600 --visualize
  
  # Use original (unprocessed) detections
  %(prog)s analysis.h5 detections.zarr --use-original
  
  # Export results to JSON
  %(prog)s analysis.h5 detections.zarr --export trial_results.json
        """
    )
    
    parser.add_argument('h5_path', help='Path to H5 analysis file')
    parser.add_argument('zarr_path', help='Path to zarr detection file')
    
    parser.add_argument('--use-original', action='store_true',
                       help='Use original detections instead of preprocessed')
    
    parser.add_argument('--speed-window', type=int, default=5,
                       help='Window size for speed smoothing (frames, default: 5)')
    
    parser.add_argument('--escape-threshold', type=float, default=500.0,
                       help='Speed threshold for escape detection (px/s, default: 500)')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize trial trajectories with speed profiles')
    
    parser.add_argument('--trial', type=int, nargs='+',
                       help='Specific trial number(s) to visualize (1-indexed)')
    
    parser.add_argument('--export', type=str,
                       help='Export results to JSON file')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.h5_path).exists():
        print(f"❌ H5 file not found: {args.h5_path}")
        return 1
    
    if not Path(args.zarr_path).exists():
        print(f"❌ Zarr file not found: {args.zarr_path}")
        return 1
    
    # Create analyzer with smoothing parameters
    analyzer = TrialZarrAnalyzer(
        h5_path=args.h5_path,
        zarr_path=args.zarr_path,
        use_interpolated=not args.use_original,
        speed_window=args.speed_window,
        escape_threshold=args.escape_threshold
    )
    
    # Analyze trials
    results = analyzer.analyze_trials(visualize=args.visualize, 
                                     trial_numbers=args.trial)
    
    # Export if requested
    if args.export and results:
        analyzer.export_trial_data(results, args.export)
    
    # Summary
    if results:
        print(f"\n{'='*70}")
        print(f"✅ Successfully analyzed {len(results)} trials")
        print(f"{'='*70}")
    else:
        print(f"\n⚠️  No trials could be analyzed")
    
    return 0


if __name__ == '__main__':
    exit(main())