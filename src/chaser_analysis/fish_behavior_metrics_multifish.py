#!/usr/bin/env python3
"""
Fish Behavior Metrics Calculator for Multi-Fish Tracker Zarr

Calculates behavioral metrics from cleaned multi-fish detection data:
- Cumulative distance traveled
- Instantaneous and smoothed speed
- Acceleration patterns
- Movement statistics

Compatible with multi-fish zarr structure and stores results in zarr.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List


class MultiLookingFishMetricsAnalyzer:
    """Analyze fish behavior from multi-fish tracker zarr data."""
    
    def __init__(self, zarr_path: str, source: str = 'latest', 
                 fish_id: Optional[int] = None, verbose: bool = True):
        """
        Initialize analyzer.
        
        Args:
            zarr_path: Path to multi-fish zarr file
            source: Which data to use ('latest', 'preprocessing', 'filtered', or specific run)
            fish_id: Optional fish ID to analyze
            verbose: Print progress messages
        """
        self.zarr_path = Path(zarr_path)
        self.root = zarr.open(str(self.zarr_path), mode='r+')
        self.verbose = verbose
        self.fish_id = fish_id
        
        # Load the appropriate data
        self.data, self.source_info = self._load_data(source)
        
        # Get metadata
        self.fps = self.root.attrs.get('fps', 60.0)
        self.total_frames = self.source_info['total_frames']
        
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
        
        return calibration
    
    def _load_data(self, source: str) -> Tuple[Dict, Dict]:
        """Load detection data based on source specification."""
        # Determine which data to load
        data_group = None
        source_name = None
        
        if source == 'latest':
            # Priority: preprocessing > filtered > detect
            if 'preprocessing' in self.root and self.root['preprocessing'].attrs.get('latest'):
                source_name = 'preprocessing/' + self.root['preprocessing'].attrs['latest']
            elif 'filtered_runs' in self.root and self.root['filtered_runs'].attrs.get('latest'):
                source_name = 'filtered_runs/' + self.root['filtered_runs'].attrs['latest']
            elif 'detect_runs' in self.root and self.root['detect_runs'].attrs.get('latest'):
                source_name = 'detect_runs/' + self.root['detect_runs'].attrs['latest']
        else:
            # Try to load specific source
            if '/' in source:
                source_name = source
            elif source in self.root:
                if self.root[source].attrs.get('latest'):
                    source_name = source + '/' + self.root[source].attrs['latest']
        
        if source_name is None or source_name not in self.root:
            raise ValueError(f"Could not find data source: {source}")
        
        data_group = self.root[source_name]
        
        # Load data
        n_detections = data_group['n_detections'][:]
        bbox_coords = data_group['bbox_norm_coords'][:]
        
        # Check for interpolation mask
        interp_mask = None
        if 'interpolation_mask' in data_group:
            interp_mask = data_group['interpolation_mask'][:]
        
        # Get dimensions
        if 'raw_video' in self.root:
            width = self.root['raw_video/images_ds'].shape[2]
            height = self.root['raw_video/images_ds'].shape[1]
        else:
            width = 640
            height = 640
        
        # Extract centroids
        centroids = []
        valid_frames = []
        cumulative = np.cumsum(np.insert(n_detections, 0, 0))
        
        for frame_idx in range(len(n_detections)):
            start_idx = cumulative[frame_idx]
            end_idx = cumulative[frame_idx + 1]
            
            if end_idx > start_idx:
                bbox = bbox_coords[start_idx]  # Take first detection
                center_x = bbox[0] * width
                center_y = bbox[1] * height
                centroids.append([center_x, center_y])
                valid_frames.append(frame_idx)
        
        centroids = np.array(centroids) if centroids else np.empty((0, 2))
        valid_frames = np.array(valid_frames)
        
        data = {
            'n_detections': n_detections,
            'bbox_coords': bbox_coords,
            'centroids': centroids,
            'valid_frames': valid_frames,
            'interpolation_mask': interp_mask
        }
        
        source_info = {
            'type': source_name.split('/')[0],
            'name': source_name,
            'coverage': len(valid_frames) / len(n_detections) if len(n_detections) > 0 else 0,
            'total_frames': len(n_detections)
        }
        
        return data, source_info
    
    def calculate_cumulative_distance(self) -> Dict:
        """Calculate cumulative distance traveled."""
        if len(self.data['centroids']) < 2:
            return None
        
        centroids = self.data['centroids']
        valid_frames = self.data['valid_frames']
        
        # Calculate frame-to-frame distances
        frame_distances = np.zeros(self.total_frames)
        frame_distances[:] = np.nan
        
        for i in range(1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[i-1])
            # Place distance at the second frame of the pair
            frame_distances[valid_frames[i]] = dist
        
        # Calculate cumulative distance (ignoring NaNs)
        cumulative_distance = np.nancumsum(np.nan_to_num(frame_distances))
        
        # Summary statistics
        total_distance = np.nansum(frame_distances)
        mean_distance = np.nanmean(frame_distances)
        max_distance = np.nanmax(frame_distances)
        
        if self.verbose:
            print(f"\nDistance Metrics:")
            print(f"  Total distance: {total_distance:.1f} pixels", end="")
            if self.pixel_to_mm:
                print(f" ({total_distance * self.pixel_to_mm:.1f} mm)")
            else:
                print()
            
            print(f"  Mean distance/frame: {mean_distance:.2f} pixels", end="")
            if self.pixel_to_mm:
                print(f" ({mean_distance * self.pixel_to_mm:.2f} mm)")
            else:
                print()
        
        return {
            'cumulative_distance': cumulative_distance,
            'frame_distances': frame_distances,
            'centroids': centroids,
            'valid_frame_indices': valid_frames,
            'total_distance': total_distance,
            'mean_distance_per_frame': mean_distance,
            'max_single_movement': max_distance
        }
    
    def calculate_speed_and_acceleration(self, window_size: int = 5) -> Dict:
        """Calculate instantaneous and smoothed speed and acceleration."""
        if len(self.data['centroids']) < 2:
            return None
        
        centroids = self.data['centroids']
        valid_frames = self.data['valid_frames']
        
        # Calculate instantaneous speed (pixels per second)
        instantaneous_speed = np.zeros(self.total_frames)
        instantaneous_speed[:] = np.nan
        
        for i in range(1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[i-1])
            time_diff = (valid_frames[i] - valid_frames[i-1]) / self.fps
            if time_diff > 0:
                speed = dist / time_diff
                instantaneous_speed[valid_frames[i]] = speed
        
        # Calculate smoothed speed
        smoothed_speed = uniform_filter1d(
            np.nan_to_num(instantaneous_speed), 
            size=window_size, 
            mode='constant'
        )
        smoothed_speed[np.isnan(instantaneous_speed)] = np.nan
        
        # Calculate acceleration
        acceleration = np.zeros(self.total_frames)
        acceleration[:] = np.nan
        
        for i in range(1, len(valid_frames)):
            if valid_frames[i] - valid_frames[i-1] == 1:  # Consecutive frames
                speed_diff = instantaneous_speed[valid_frames[i]] - instantaneous_speed[valid_frames[i-1]]
                acceleration[valid_frames[i]] = speed_diff * self.fps
        
        # Summary statistics
        mean_speed = np.nanmean(instantaneous_speed)
        max_speed = np.nanmax(instantaneous_speed)
        speed_std = np.nanstd(instantaneous_speed)
        
        if self.verbose:
            print(f"\nSpeed Metrics:")
            print(f"  Mean speed: {mean_speed:.1f} pixels/second", end="")
            if self.pixel_to_mm:
                mm_per_s = mean_speed * self.pixel_to_mm
                bl_per_s = mm_per_s / self.fish_length_mm
                print(f" ({mm_per_s:.2f} mm/s, {bl_per_s:.2f} BL/s)")
            else:
                print()
            
            print(f"  Max speed: {max_speed:.1f} pixels/second", end="")
            if self.pixel_to_mm:
                mm_per_s = max_speed * self.pixel_to_mm
                bl_per_s = mm_per_s / self.fish_length_mm
                print(f" ({mm_per_s:.1f} mm/s, {bl_per_s:.1f} BL/s)")
            else:
                print()
        
        return {
            'instantaneous_speed': instantaneous_speed,
            'smoothed_speed': smoothed_speed,
            'acceleration': acceleration,
            'mean_speed': mean_speed,
            'max_speed': max_speed,
            'speed_std': speed_std,
            'window_size': window_size
        }
    
    def save_metrics(self, overwrite: bool = False):
        """Save calculated metrics to zarr file."""
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
        if self.fish_id is not None:
            metrics_group.attrs['fish_id'] = self.fish_id
        
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
        """Create comprehensive visualization of behavioral metrics."""
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
        else:
            dist_unit = "pixels"
            dist_conv = 1.0
            speed_unit = "pixels/s"
            speed_conv = 1.0
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Add title
        title = f'Fish Behavior Metrics - {self.source_info["name"]}'
        if self.fish_id is not None:
            title += f' (Fish ID: {self.fish_id})'
        if self.pixel_to_mm:
            title += f'\n(Calibrated: 1 pixel = {self.pixel_to_mm:.4f} mm)'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Time axis
        time_seconds = np.arange(self.total_frames) / self.fps
        
        # 1. Trajectory
        ax = axes[0, 0]
        centroids = distance_metrics['centroids']
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
        ax.set_title(f'Total Distance: {total_dist:.1f} {dist_unit}')
        ax.grid(True, alpha=0.3)
        
        # 3. Frame-to-frame Distance
        ax = axes[1, 0]
        frame_dist_display = distance_metrics['frame_distances'] * dist_conv
        valid_mask = ~np.isnan(frame_dist_display)
        ax.scatter(time_seconds[valid_mask], frame_dist_display[valid_mask],
                  alpha=0.5, s=1, c='blue')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(f'Distance ({dist_unit})')
        ax.set_title('Frame-to-Frame Movement')
        ax.grid(True, alpha=0.3)
        
        # Add threshold line
        ax.axhline(y=50*dist_conv, color='r', linestyle='--', alpha=0.5,
                  label='Jump threshold')
        mean_dist = distance_metrics['mean_distance_per_frame'] * dist_conv
        ax.axhline(y=mean_dist, color='g', linestyle='--', alpha=0.5,
                  label=f'Mean: {mean_dist:.2f}')
        ax.legend()
        
        # 4. Speed over time
        ax = axes[1, 1]
        inst_speed = speed_metrics['instantaneous_speed'] * speed_conv
        smooth_speed = speed_metrics['smoothed_speed'] * speed_conv
        valid_mask = ~np.isnan(inst_speed)
        
        ax.plot(time_seconds[valid_mask], inst_speed[valid_mask],
               'b-', alpha=0.3, linewidth=0.5, label='Instantaneous')
        ax.plot(time_seconds[valid_mask], smooth_speed[valid_mask],
               'r-', linewidth=2, label='Smoothed')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(f'Speed ({speed_unit})')
        mean_speed = speed_metrics["mean_speed"] * speed_conv
        ax.set_title(f'Swimming Speed (Mean: {mean_speed:.1f} {speed_unit})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Speed distribution
        ax = axes[2, 0]
        valid_speeds = inst_speed[valid_mask]
        ax.hist(valid_speeds, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=mean_speed, color='r', linestyle='--', 
                  label=f'Mean: {mean_speed:.1f}')
        ax.set_xlabel(f'Speed ({speed_unit})')
        ax.set_ylabel('Count')
        ax.set_title('Speed Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Activity heatmap
        ax = axes[2, 1]
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(centroids[:, 0], centroids[:, 1], bins=30)
        H = H.T  # Transpose for correct orientation
        ax.imshow(H, origin='lower', aspect='equal', cmap='hot',
                 extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('Activity Heatmap')
        cbar = plt.colorbar(ax.images[0], ax=ax)
        cbar.set_label('Time spent')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate behavioral metrics from multi-fish tracker zarr'
    )
    parser.add_argument('zarr_path', help='Path to multi-fish zarr file')
    parser.add_argument('--source', type=str, default='latest',
                       help='Data source: latest, preprocessing, filtered, or specific run')
    parser.add_argument('--fish-id', type=int, default=None,
                       help='Analyze specific fish ID')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing metrics')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--output-plot', type=str, default=None,
                       help='Path to save plot')
    parser.add_argument('--save', action='store_true',
                       help='Save metrics to zarr')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MULTI-FISH BEHAVIOR METRICS CALCULATOR")
    print("=" * 60)
    print(f"Zarr file: {args.zarr_path}")
    print(f"Data source: {args.source}")
    
    try:
        # Initialize analyzer
        analyzer = MultiLookingFishMetricsAnalyzer(
            args.zarr_path,
            source=args.source,
            fish_id=args.fish_id,
            verbose=True
        )
        
        # Save metrics if requested
        if args.save:
            success = analyzer.save_metrics(overwrite=args.overwrite)
            if not success and not args.overwrite:
                print("Tip: Use --overwrite to replace existing metrics")
        
        # Generate plots if requested
        if args.plot or args.output_plot:
            analyzer.plot_metrics(save_path=args.output_plot)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
    