#!/usr/bin/env python3
"""
Fish Behavior Metrics Analyzer

Calculates behavioral metrics from cleaned detection data:
- Cumulative distance traveled
- Instantaneous and smoothed speed
- Acceleration patterns
- Movement statistics

Stores results in zarr with full provenance tracking.
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from typing import Dict, Tuple, Optional


class FishMetricsAnalyzer:
    """Analyze fish behavior from cleaned detection data."""
    
    def __init__(self, zarr_path: str, source: str = 'latest', verbose: bool = True):
        """
        Initialize analyzer.
        
        Args:
            zarr_path: Path to zarr file with detection data
            source: Which data to use ('latest', 'interpolated', 'filtered', or specific version)
            verbose: Print progress messages
        """
        self.zarr_path = Path(zarr_path)
        self.root = zarr.open(str(self.zarr_path), mode='r+')
        self.verbose = verbose
        
        # Load the appropriate data
        self.data, self.source_info = self._load_data(source)
        
        # Get metadata
        self.fps = self.root.attrs.get('fps', 60.0)
        self.total_frames = len(self.data['n_detections'])
        
        # Load calibration if available
        self.calibration = self._load_calibration()
        self.pixel_to_mm = None
        self.fish_length_mm = 4.0  # Default larval zebrafish length
        
        if self.calibration and 'pixel_to_mm' in self.calibration:
            self.pixel_to_mm = self.calibration['pixel_to_mm']
            if verbose:
                print(f"Calibration loaded: 1 pixel = {self.pixel_to_mm:.4f} mm")
        
        if verbose:
            print(f"Loaded {self.source_info['type']} data: {self.source_info['name']}")
            print(f"Coverage: {self.source_info['coverage']*100:.1f}%")
            print(f"FPS: {self.fps}")
            if self.pixel_to_mm:
                print(f"Real-world units: ENABLED (1 px = {self.pixel_to_mm:.4f} mm)")
            else:
                print(f"Real-world units: DISABLED (no calibration found)")
    
    def _load_calibration(self) -> Optional[Dict]:
        """Load calibration data if available."""
        if 'calibration' not in self.root:
            return None
        
        calib_group = self.root['calibration']
        calibration = dict(calib_group.attrs)
        
        # Load subgroups if needed
        for group_name in ['arena', 'rig_info']:
            if group_name in calib_group:
                calibration[group_name] = dict(calib_group[group_name].attrs)
        
        return calibration
    
    def _load_data(self, source: str) -> Tuple[Dict, Dict]:
        """Load detection data based on source specification."""
        
        if source == 'latest' or source == 'interpolated':
            # Try interpolated first
            if 'preprocessing' in self.root and 'latest' in self.root['preprocessing'].attrs:
                latest = self.root['preprocessing'].attrs['latest']
                data_group = self.root['preprocessing'][latest]
                source_type = 'interpolated'
                source_name = latest
            elif 'filtered_runs' in self.root and 'latest' in self.root['filtered_runs'].attrs:
                latest = self.root['filtered_runs'].attrs['latest']
                data_group = self.root['filtered_runs'][latest]
                source_type = 'filtered'
                source_name = latest
            else:
                data_group = self.root
                source_type = 'original'
                source_name = 'root'
        
        elif source == 'filtered':
            if 'filtered_runs' in self.root and 'latest' in self.root['filtered_runs'].attrs:
                latest = self.root['filtered_runs'].attrs['latest']
                data_group = self.root['filtered_runs'][latest]
                source_type = 'filtered'
                source_name = latest
            else:
                raise ValueError("No filtered data found")
        
        elif source == 'original':
            data_group = self.root
            source_type = 'original'
            source_name = 'root'
        
        else:
            # Try to find specific version
            if 'preprocessing' in self.root and source in self.root['preprocessing']:
                data_group = self.root['preprocessing'][source]
                source_type = 'interpolated'
                source_name = source
            elif 'filtered_runs' in self.root and source in self.root['filtered_runs']:
                data_group = self.root['filtered_runs'][source]
                source_type = 'filtered'
                source_name = source
            else:
                raise ValueError(f"Source '{source}' not found")
        
        # Load arrays
        data = {
            'bboxes': data_group['bboxes'][:],
            'scores': data_group['scores'][:],
            'n_detections': data_group['n_detections'][:],
            'class_ids': data_group['class_ids'][:]
        }
        
        # Add interpolation mask if available
        if 'interpolation_mask' in data_group:
            data['interpolation_mask'] = data_group['interpolation_mask'][:]
        
        # Calculate coverage
        coverage = (data['n_detections'] > 0).sum() / len(data['n_detections'])
        
        source_info = {
            'type': source_type,
            'name': source_name,
            'coverage': coverage
        }
        
        return data, source_info
    
    def calculate_cumulative_distance(self) -> Dict:
        """
        Calculate cumulative distance traveled across frames.
        
        Returns:
            Dictionary with distance metrics
        """
        if self.verbose:
            print("\nCalculating cumulative distance...")
        
        # Extract centroids for valid detections
        centroids = []
        valid_frames = []
        
        for frame_idx in range(self.total_frames):
            if self.data['n_detections'][frame_idx] > 0:
                bbox = self.data['bboxes'][frame_idx, 0]
                centroid = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
                centroids.append(centroid)
                valid_frames.append(frame_idx)
        
        if len(centroids) < 2:
            print("Warning: Not enough detections for distance calculation")
            return None
        
        centroids = np.array(centroids)
        valid_frames = np.array(valid_frames)
        
        # Calculate frame-to-frame distances
        distances = []
        cumulative_distance = [0]
        
        for i in range(1, len(centroids)):
            # Calculate Euclidean distance
            dist = np.linalg.norm(centroids[i] - centroids[i-1])
            
            # Check if frames are consecutive or have a gap
            frame_gap = valid_frames[i] - valid_frames[i-1]
            
            # Only add to cumulative if frames are reasonably close
            # (large gaps might be unreliable)
            if frame_gap <= 10:  # Threshold for acceptable gap
                distances.append(dist)
                cumulative_distance.append(cumulative_distance[-1] + dist)
            else:
                # For large gaps, don't add distance but maintain cumulative
                distances.append(np.nan)
                cumulative_distance.append(cumulative_distance[-1])
        
        # Create full arrays with NaN for missing frames
        full_distances = np.full(self.total_frames, np.nan)
        full_cumulative = np.full(self.total_frames, np.nan)
        
        # Fill in the valid frames
        for i, frame_idx in enumerate(valid_frames[1:]):
            full_distances[frame_idx] = distances[i]
        
        for i, frame_idx in enumerate(valid_frames):
            full_cumulative[frame_idx] = cumulative_distance[i]
        
        # Forward fill cumulative distance for visualization continuity
        last_valid = 0
        for i in range(self.total_frames):
            if not np.isnan(full_cumulative[i]):
                last_valid = full_cumulative[i]
            else:
                full_cumulative[i] = last_valid
        
        results = {
            'cumulative_distance': full_cumulative,  # Always in pixels
            'frame_distances': full_distances,       # Always in pixels
            'total_distance': cumulative_distance[-1],
            'mean_distance_per_frame': np.nanmean(distances),
            'max_single_movement': np.nanmax(distances),
            'valid_frame_indices': valid_frames,
            'centroids': centroids
        }
        
        if self.verbose:
            print(f"  Total distance: {results['total_distance']:.1f} pixels", end="")
            if self.pixel_to_mm:
                print(f" ({results['total_distance'] * self.pixel_to_mm:.1f} mm)")
            else:
                print()
            
            print(f"  Mean movement: {results['mean_distance_per_frame']:.2f} pixels/frame", end="")
            if self.pixel_to_mm:
                print(f" ({results['mean_distance_per_frame'] * self.pixel_to_mm:.3f} mm/frame)")
            else:
                print()
            
            print(f"  Max movement: {results['max_single_movement']:.1f} pixels", end="")
            if self.pixel_to_mm:
                print(f" ({results['max_single_movement'] * self.pixel_to_mm:.2f} mm)")
            else:
                print()
        
        return results
    
    def calculate_speed_and_acceleration(self, window_size: int = 5, max_speed_threshold: float = 1000.0) -> Dict:
        """
        Calculate instantaneous speed and acceleration.
        
        Args:
            window_size: Window for smoothing (frames)
            max_speed_threshold: Maximum reasonable speed (pixels/second) - higher values are capped
        
        Returns:
            Dictionary with speed and acceleration metrics
        """
        if self.verbose:
            print("\nCalculating speed and acceleration...")
        
        distance_metrics = self.calculate_cumulative_distance()
        if distance_metrics is None:
            return None
        
        # Instantaneous speed (pixels per second)
        instantaneous_speed = distance_metrics['frame_distances'] * self.fps
        
        # Cap unrealistic speeds (likely from gaps or jumps)
        instantaneous_speed = np.where(
            instantaneous_speed > max_speed_threshold,
            np.nan,
            instantaneous_speed
        )
        
        # For smoothed speed, we work with the valid data as-is
        # No additional interpolation - respect the preprocessing pipeline
        smoothed_speed = np.full_like(instantaneous_speed, np.nan)
        valid_mask = ~np.isnan(instantaneous_speed)
        
        if np.sum(valid_mask) > window_size:
            # Apply smoothing only to valid data
            valid_indices = np.where(valid_mask)[0]
            for idx in valid_indices:
                # Get window around this point
                window_start = max(0, idx - window_size // 2)
                window_end = min(len(instantaneous_speed), idx + window_size // 2 + 1)
                window_data = instantaneous_speed[window_start:window_end]
                
                # Calculate mean of valid points in window
                valid_in_window = window_data[~np.isnan(window_data)]
                if len(valid_in_window) > 0:
                    smoothed_speed[idx] = np.mean(valid_in_window)
        else:
            smoothed_speed = instantaneous_speed.copy()
        
        # Acceleration (change in smoothed speed per second)
        # Only calculate where we have consecutive valid smoothed speeds
        acceleration = np.full(self.total_frames, np.nan)
        
        for i in range(1, len(smoothed_speed)):
            if not np.isnan(smoothed_speed[i]) and not np.isnan(smoothed_speed[i-1]):
                # Change in speed per frame, converted to per second
                dt = 1.0 / self.fps  # Time between frames in seconds
                acceleration[i] = (smoothed_speed[i] - smoothed_speed[i-1]) / dt
        
        # Calculate statistics, handling potential all-NaN cases
        mean_accel = np.nanmean(acceleration) if not np.all(np.isnan(acceleration)) else 0.0
        accel_std = np.nanstd(acceleration) if not np.all(np.isnan(acceleration)) else 0.0
        
        results = {
            'instantaneous_speed': instantaneous_speed,  # Always in pixels/second
            'smoothed_speed': smoothed_speed,           # Always in pixels/second
            'acceleration': acceleration,                # Always in pixels/second²
            'window_size': window_size,
            'mean_speed': np.nanmean(instantaneous_speed),
            'max_speed': np.nanmax(instantaneous_speed),
            'speed_std': np.nanstd(instantaneous_speed),
            'mean_acceleration': mean_accel,
            'acceleration_std': accel_std
        }
        
        if self.verbose:
            print(f"  Mean speed: {results['mean_speed']:.1f} pixels/second", end="")
            if self.pixel_to_mm:
                mm_per_s = results['mean_speed'] * self.pixel_to_mm
                bl_per_s = mm_per_s / self.fish_length_mm
                print(f" ({mm_per_s:.2f} mm/s, {bl_per_s:.2f} BL/s)")
            else:
                print()
            
            print(f"  Max speed: {results['max_speed']:.1f} pixels/second", end="")
            if self.pixel_to_mm:
                mm_per_s = results['max_speed'] * self.pixel_to_mm
                bl_per_s = mm_per_s / self.fish_length_mm
                print(f" ({mm_per_s:.1f} mm/s, {bl_per_s:.1f} BL/s)")
            else:
                print()
            
            print(f"  Speed std: {results['speed_std']:.1f} pixels/second", end="")
            if self.pixel_to_mm:
                print(f" ({results['speed_std'] * self.pixel_to_mm:.2f} mm/s)")
            else:
                print()
            
            if not np.isnan(mean_accel):
                print(f"  Mean acceleration: {mean_accel:.2f} pixels/second²", end="")
                if self.pixel_to_mm:
                    print(f" ({mean_accel * self.pixel_to_mm:.2f} mm/s²)")
                else:
                    print()
                    
                print(f"  Acceleration std: {accel_std:.2f} pixels/second²", end="")
                if self.pixel_to_mm:
                    print(f" ({accel_std * self.pixel_to_mm:.1f} mm/s²)")
                else:
                    print()
        
        return results
    
    def save_metrics(self, overwrite: bool = False):
        """
        Save calculated metrics to zarr file.
        
        Args:
            overwrite: Whether to overwrite existing metrics
        """
        if self.verbose:
            print("\nSaving metrics to zarr...")
        
        # Check if metrics group exists
        if 'behavior_metrics' in self.root and not overwrite:
            print("Error: behavior_metrics already exists. Use --overwrite to replace.")
            return False
        
        # Create or overwrite metrics group
        if 'behavior_metrics' in self.root:
            del self.root['behavior_metrics']
        
        metrics_group = self.root.create_group('behavior_metrics')
        
        # Add metadata
        metrics_group.attrs['created_at'] = datetime.now().isoformat()
        metrics_group.attrs['source_type'] = self.source_info['type']
        metrics_group.attrs['source_name'] = self.source_info['name']
        metrics_group.attrs['source_coverage'] = float(self.source_info['coverage'])
        metrics_group.attrs['fps'] = self.fps
        metrics_group.attrs['total_frames'] = self.total_frames
        
        # Calculate and save distance metrics
        distance_metrics = self.calculate_cumulative_distance()
        if distance_metrics:
            dist_group = metrics_group.create_group('distance')
            
            # Save arrays
            dist_group.create_dataset('cumulative_distance', 
                                     data=distance_metrics['cumulative_distance'],
                                     chunks=True, compression='gzip')
            dist_group.create_dataset('frame_distances', 
                                     data=distance_metrics['frame_distances'],
                                     chunks=True, compression='gzip')
            dist_group.create_dataset('centroids', 
                                     data=distance_metrics['centroids'],
                                     chunks=True, compression='gzip')
            dist_group.create_dataset('valid_frame_indices', 
                                     data=distance_metrics['valid_frame_indices'],
                                     chunks=True, compression='gzip')
            
            # Save summary statistics
            dist_group.attrs['total_distance'] = float(distance_metrics['total_distance'])
            dist_group.attrs['mean_distance_per_frame'] = float(distance_metrics['mean_distance_per_frame'])
            dist_group.attrs['max_single_movement'] = float(distance_metrics['max_single_movement'])
            dist_group.attrs['units'] = 'pixels'
        
        # Calculate and save speed/acceleration metrics
        speed_metrics = self.calculate_speed_and_acceleration()
        if speed_metrics:
            speed_group = metrics_group.create_group('speed')
            
            # Save arrays
            speed_group.create_dataset('instantaneous_speed', 
                                      data=speed_metrics['instantaneous_speed'],
                                      chunks=True, compression='gzip')
            speed_group.create_dataset('smoothed_speed', 
                                      data=speed_metrics['smoothed_speed'],
                                      chunks=True, compression='gzip')
            speed_group.create_dataset('acceleration', 
                                      data=speed_metrics['acceleration'],
                                      chunks=True, compression='gzip')
            
            # Save summary statistics
            speed_group.attrs['mean_speed'] = float(speed_metrics['mean_speed'])
            speed_group.attrs['max_speed'] = float(speed_metrics['max_speed'])
            speed_group.attrs['speed_std'] = float(speed_metrics['speed_std'])
            speed_group.attrs['window_size'] = speed_metrics['window_size']
            speed_group.attrs['speed_units'] = 'pixels/second'
            speed_group.attrs['acceleration_units'] = 'pixels/second^2'
        
        if self.verbose:
            print(f"  ✓ Metrics saved to {self.zarr_path}/behavior_metrics")
            print(f"  ✓ Source: {self.source_info['name']} ({self.source_info['type']})")
        
        return True
    
    def plot_metrics(self, save_path: Optional[str] = None, show: bool = True):
        """
        Create visualization of behavioral metrics.
        
        Args:
            save_path: Optional path to save figure
            show: Whether to display the plot
        """
        distance_metrics = self.calculate_cumulative_distance()
        speed_metrics = self.calculate_speed_and_acceleration()
        
        if not distance_metrics or not speed_metrics:
            print("Error: Unable to calculate metrics for plotting")
            return
        
        # Determine units for labels
        if self.pixel_to_mm:
            dist_unit = "mm"
            dist_conv = self.pixel_to_mm
            speed_unit = "mm/s"
            speed_conv = self.pixel_to_mm
            accel_unit = "mm/s²"
        else:
            dist_unit = "pixels"
            dist_conv = 1.0
            speed_unit = "pixels/s"
            speed_conv = 1.0
            accel_unit = "pixels/s²"
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Add title with units info
        title = f'Fish Behavior Metrics - {self.source_info["name"]}'
        if self.pixel_to_mm:
            title += f'\n(Calibrated: 1 pixel = {self.pixel_to_mm:.4f} mm)'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Time axis
        time_seconds = np.arange(self.total_frames) / self.fps
        
        # 1. Trajectory
        ax = axes[0, 0]
        centroids = distance_metrics['centroids']
        ax.plot(centroids[:, 0], centroids[:, 1], 'b-', alpha=0.5, linewidth=0.5)
        scatter = ax.scatter(centroids[:, 0], centroids[:, 1], 
                           c=distance_metrics['valid_frame_indices'],
                           cmap='viridis', s=1, alpha=0.7)
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('Movement Trajectory')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Frame')
        
        # 2. Cumulative Distance
        ax = axes[0, 1]
        cumulative_display = distance_metrics['cumulative_distance'] * dist_conv
        ax.plot(time_seconds, cumulative_display, 'g-', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(f'Cumulative Distance ({dist_unit})')
        total_dist = distance_metrics["total_distance"] * dist_conv
        ax.set_title(f'Total Distance Traveled: {total_dist:.1f} {dist_unit}')
        ax.grid(True, alpha=0.3)
        
        # 3. Frame-to-frame Distance
        ax = axes[1, 0]
        frame_dist_display = distance_metrics['frame_distances'] * dist_conv
        valid_mask = ~np.isnan(frame_dist_display)
        ax.scatter(time_seconds[valid_mask], frame_dist_display[valid_mask],
                  alpha=0.5, s=1, c='blue')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(f'Distance ({dist_unit})')
        ax.set_title('Frame-to-Frame Movement Distance')
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_dist = distance_metrics['mean_distance_per_frame'] * dist_conv
        ax.axhline(y=mean_dist, color='r', linestyle='--', alpha=0.5,
                  label=f'Mean: {mean_dist:.2f} {dist_unit}')
        ax.legend()
        
        # 4. Speed over time
        ax = axes[1, 1]
        inst_speed_display = speed_metrics['instantaneous_speed'] * speed_conv
        smooth_speed_display = speed_metrics['smoothed_speed'] * speed_conv
        valid_mask = ~np.isnan(inst_speed_display)
        
        ax.plot(time_seconds[valid_mask], inst_speed_display[valid_mask],
               'b-', alpha=0.3, linewidth=0.5, label='Instantaneous')
        ax.plot(time_seconds[valid_mask], smooth_speed_display[valid_mask],
               'r-', linewidth=2, label='Smoothed')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(f'Speed ({speed_unit})')
        
        mean_speed = speed_metrics["mean_speed"] * speed_conv
        title_str = f'Swimming Speed (Mean: {mean_speed:.1f} {speed_unit}'
        if self.pixel_to_mm:
            bl_per_s = mean_speed / self.fish_length_mm
            title_str += f', {bl_per_s:.1f} BL/s'
        title_str += ')'
        ax.set_title(title_str)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Speed distribution
        ax = axes[2, 0]
        valid_speeds = inst_speed_display[~np.isnan(inst_speed_display)]
        ax.hist(valid_speeds, bins=50, alpha=0.7, color='blue', edgecolor='black')
        
        mean_s = speed_metrics['mean_speed'] * speed_conv
        max_s = speed_metrics['max_speed'] * speed_conv
        ax.axvline(x=mean_s, color='r', linestyle='--',
                  label=f'Mean: {mean_s:.1f} {speed_unit}')
        ax.axvline(x=max_s, color='orange', linestyle='--',
                  label=f'Max: {max_s:.1f} {speed_unit}')
        
        if self.pixel_to_mm:
            # Add body length markers
            for bl in [1, 2, 5, 10]:
                bl_speed = bl * self.fish_length_mm
                if bl_speed < max_s:
                    ax.axvline(x=bl_speed, color='green', linestyle=':', alpha=0.5)
                    ax.text(bl_speed, ax.get_ylim()[1]*0.9, f'{bl} BL/s',
                           rotation=90, va='top', fontsize=8, color='green')
        
        ax.set_xlabel(f'Speed ({speed_unit})')
        ax.set_ylabel('Frequency')
        ax.set_title('Speed Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Acceleration
        ax = axes[2, 1]
        accel_display = speed_metrics['acceleration'] * dist_conv
        valid_mask = ~np.isnan(accel_display)
        if np.sum(valid_mask) > 0:
            accel_data = accel_display[valid_mask]
            time_data = time_seconds[valid_mask]
            
            ax.plot(time_data, accel_data, 'purple', alpha=0.6, linewidth=0.5)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Set reasonable y-limits based on data
            accel_std = np.std(accel_data)
            accel_mean = np.mean(accel_data)
            y_limit = max(abs(accel_mean - 3*accel_std), abs(accel_mean + 3*accel_std))
            if y_limit > 0:
                ax.set_ylim([-y_limit, y_limit])
            
            # Add statistics
            stats_text = f'Mean: {accel_mean:.1f} {accel_unit}\nStd: {accel_std:.1f} {accel_unit}'
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No acceleration data', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(f'Acceleration ({accel_unit})')
        ax.set_title('Acceleration Profile')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Figure saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate fish behavior metrics from detection data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate metrics using latest cleaned data
  %(prog)s detections.zarr
  
  # Use specific data source
  %(prog)s detections.zarr --source v3_interpolated_20250821_141332
  
  # Overwrite existing metrics
  %(prog)s detections.zarr --overwrite
  
  # Save visualization
  %(prog)s detections.zarr --plot --save-plot metrics.png
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    
    parser.add_argument('--source', default='latest',
                       help='Data source: latest, interpolated, filtered, original, or specific version')
    
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing metrics')
    
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    
    parser.add_argument('--save-plot', dest='save_plot',
                       help='Path to save plot')
    
    parser.add_argument('--no-save', action='store_true',
                       help="Don't save metrics to zarr (preview only)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FishMetricsAnalyzer(args.zarr_path, source=args.source)
    
    # Save metrics
    if not args.no_save:
        success = analyzer.save_metrics(overwrite=args.overwrite)
        if not success and not args.overwrite:
            print("\nTip: Use --overwrite to replace existing metrics")
            return 1
    
    # Generate plots if requested
    if args.plot or args.save_plot:
        analyzer.plot_metrics(save_path=args.save_plot, show=args.plot)
    
    return 0


if __name__ == '__main__':
    exit(main())