#!/usr/bin/env python3
"""
Simple Fish Metrics Calculator

Calculates core behavioral metrics for fish tracking data:
- Speed (instantaneous and smoothed)
- Acceleration
- Cumulative distance traveled

These metrics are added to the existing zarr file for analysis.
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt


class SimpleFishMetrics:
    """
    Calculates simple behavioral metrics from fish tracking data.
    """
    
    def __init__(self, 
                 zarr_path: str,
                 interpolation_run: Optional[str] = None,
                 fps: float = 60.0,
                 pixel_to_mm: float = None,
                 verbose: bool = True):
        """
        Initialize the metrics calculator.
        
        Args:
            zarr_path: Path to zarr file with tracking data
            interpolation_run: Specific interpolation run to use
            fps: Frame rate of the video (Hz)
            pixel_to_mm: Conversion factor from pixels to mm (optional)
            verbose: Print progress messages
        """
        self.zarr_path = Path(zarr_path)
        self.fps = fps
        self.pixel_to_mm = pixel_to_mm
        self.verbose = verbose
        
        # Time between frames
        self.dt = 1.0 / fps
        
        # Open zarr file
        self.root = zarr.open(str(zarr_path), mode='r+')
        
        # Determine interpolation run
        if interpolation_run is None and 'interpolation_runs' in self.root:
            interpolation_run = self.root['interpolation_runs'].attrs.get('latest')
            if self.verbose:
                print(f"Using latest interpolation run: {interpolation_run}")
        self.interpolation_run = interpolation_run
        
        # Load fish positions
        self._load_positions()
    
    def _load_positions(self):
        """Load fish position data from zarr."""
        if self.verbose:
            print("Loading fish position data...")
        
        # Try to load from chaser_comparison first (already computed positions)
        if 'chaser_comparison' in self.root:
            comp_group = self.root['chaser_comparison']
            run_name = self.interpolation_run or comp_group.attrs.get('latest', 'original')
            
            if run_name in comp_group and 'fish_position_camera' in comp_group[run_name]:
                self.positions = comp_group[run_name]['fish_position_camera'][:]
                if self.verbose:
                    print(f"  Loaded positions from chaser_comparison/{run_name}")
                    valid = ~np.isnan(self.positions[:, 0])
                    print(f"  Valid frames: {np.sum(valid)}/{len(self.positions)} ({np.sum(valid)/len(self.positions)*100:.1f}%)")
                return
        
        # Otherwise, calculate from bounding boxes
        if self.interpolation_run and f'interpolation_runs/{self.interpolation_run}' in self.root:
            data_group = self.root[f'interpolation_runs/{self.interpolation_run}']
        else:
            data_group = self.root
        
        bboxes = data_group['bboxes'][:]
        n_detections = data_group['n_detections'][:]
        
        # Calculate center positions from bboxes
        self.positions = np.full((len(bboxes), 2), np.nan)
        
        for i in range(len(bboxes)):
            if n_detections[i] > 0:
                bbox = bboxes[i, 0]  # First detection
                # Center of bounding box
                self.positions[i, 0] = (bbox[0] + bbox[2]) / 2  # x
                self.positions[i, 1] = (bbox[1] + bbox[3]) / 2  # y
        
        if self.verbose:
            valid = ~np.isnan(self.positions[:, 0])
            print(f"  Calculated positions from bboxes")
            print(f"  Valid frames: {np.sum(valid)}/{len(self.positions)} ({np.sum(valid)/len(self.positions)*100:.1f}%)")
    
    def calculate_speed(self, smooth_window: int = 5) -> Dict[str, np.ndarray]:
        """
        Calculate fish speed over time.
        
        Args:
            smooth_window: Window size for smoothing (frames)
            
        Returns:
            Dictionary with speed metrics
        """
        if self.verbose:
            print("\nCalculating speed...")
        
        n_frames = len(self.positions)
        
        # Initialize arrays
        instantaneous_speed = np.full(n_frames, np.nan)
        
        # Calculate frame-to-frame displacement and speed
        for i in range(1, n_frames):
            if not np.isnan(self.positions[i, 0]) and not np.isnan(self.positions[i-1, 0]):
                # Calculate displacement
                dx = self.positions[i, 0] - self.positions[i-1, 0]
                dy = self.positions[i, 1] - self.positions[i-1, 1]
                
                # Speed is displacement magnitude per unit time
                frame_speed = np.sqrt(dx**2 + dy**2) / self.dt
                instantaneous_speed[i] = frame_speed
        
        # Apply smoothing for cleaner signal
        smoothed_speed = self._smooth_signal(instantaneous_speed, smooth_window)
        
        # Convert units if needed
        if self.pixel_to_mm is not None:
            instantaneous_speed *= self.pixel_to_mm
            smoothed_speed *= self.pixel_to_mm
            units = "mm/s"
        else:
            units = "pixels/s"
        
        # Calculate statistics
        valid_speeds = instantaneous_speed[~np.isnan(instantaneous_speed)]
        if len(valid_speeds) > 0 and self.verbose:
            print(f"  Speed statistics ({units}):")
            print(f"    Mean: {np.mean(valid_speeds):.2f}")
            print(f"    Median: {np.median(valid_speeds):.2f}")
            print(f"    Max: {np.max(valid_speeds):.2f}")
            print(f"    95th percentile: {np.percentile(valid_speeds, 95):.2f}")
        
        return {
            'instantaneous_speed': instantaneous_speed,
            'smoothed_speed': smoothed_speed,
            'units': units
        }
    
    def calculate_acceleration(self, speed_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate acceleration from speed data.
        
        Args:
            speed_data: Dictionary from calculate_speed()
            
        Returns:
            Acceleration array
        """
        if self.verbose:
            print("\nCalculating acceleration...")
        
        smoothed_speed = speed_data['smoothed_speed']
        n_frames = len(smoothed_speed)
        acceleration = np.full(n_frames, np.nan)
        
        # Calculate acceleration as rate of change of speed
        for i in range(1, n_frames):
            if not np.isnan(smoothed_speed[i]) and not np.isnan(smoothed_speed[i-1]):
                acceleration[i] = (smoothed_speed[i] - smoothed_speed[i-1]) / self.dt
        
        # Calculate statistics
        valid_acc = acceleration[~np.isnan(acceleration)]
        if len(valid_acc) > 0 and self.verbose:
            print(f"  Acceleration statistics ({speed_data['units']}/s):")
            print(f"    Mean: {np.mean(valid_acc):.2f}")
            print(f"    Std: {np.std(valid_acc):.2f}")
            print(f"    Max positive: {np.max(valid_acc):.2f}")
            print(f"    Max negative: {np.min(valid_acc):.2f}")
        
        return acceleration
    
    def calculate_cumulative_distance(self) -> Dict[str, np.ndarray]:
        """
        Calculate cumulative distance traveled.
        
        Returns:
            Dictionary with distance metrics
        """
        if self.verbose:
            print("\nCalculating cumulative distance...")
        
        n_frames = len(self.positions)
        cumulative_distance = np.zeros(n_frames)
        frame_distances = np.full(n_frames, np.nan)
        
        for i in range(1, n_frames):
            if not np.isnan(self.positions[i, 0]) and not np.isnan(self.positions[i-1, 0]):
                # Calculate frame-to-frame distance
                dist = np.linalg.norm(self.positions[i] - self.positions[i-1])
                frame_distances[i] = dist
                cumulative_distance[i] = cumulative_distance[i-1] + dist
            else:
                # Carry forward the cumulative distance
                cumulative_distance[i] = cumulative_distance[i-1]
        
        # Convert units if needed
        if self.pixel_to_mm is not None:
            cumulative_distance *= self.pixel_to_mm
            frame_distances *= self.pixel_to_mm
            units = "mm"
        else:
            units = "pixels"
        
        if self.verbose:
            print(f"  Total distance traveled: {cumulative_distance[-1]:.1f} {units}")
            valid_dists = frame_distances[~np.isnan(frame_distances)]
            if len(valid_dists) > 0:
                print(f"  Mean step size: {np.mean(valid_dists):.2f} {units}")
        
        return {
            'cumulative_distance': cumulative_distance,
            'frame_distances': frame_distances,
            'units': units
        }
    
    def _smooth_signal(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing to a signal."""
        if window <= 1:
            return signal.copy()
        
        smoothed = np.full_like(signal, np.nan)
        
        # Only smooth where we have valid data
        valid_mask = ~np.isnan(signal)
        if np.sum(valid_mask) > window:
            # Get valid segments
            valid_indices = np.where(valid_mask)[0]
            
            # Apply uniform filter to each continuous segment
            # This prevents smoothing across gaps
            segments = []
            current_segment = [valid_indices[0]]
            
            for i in range(1, len(valid_indices)):
                if valid_indices[i] - valid_indices[i-1] == 1:
                    current_segment.append(valid_indices[i])
                else:
                    segments.append(current_segment)
                    current_segment = [valid_indices[i]]
            segments.append(current_segment)
            
            # Smooth each segment
            for segment in segments:
                if len(segment) >= window:
                    segment_signal = signal[segment]
                    smoothed_segment = uniform_filter1d(segment_signal, size=window, mode='nearest')
                    smoothed[segment] = smoothed_segment
                else:
                    # Segment too short to smooth
                    smoothed[segment] = signal[segment]
        else:
            # Not enough data to smooth
            smoothed = signal.copy()
        
        return smoothed
    
    def save_metrics(self, overwrite: bool = False):
        """
        Save calculated metrics to the zarr file.
        
        Args:
            overwrite: Whether to overwrite existing metrics
        """
        if self.verbose:
            print("\nSaving metrics to zarr...")
        
        # Calculate all metrics
        speed_data = self.calculate_speed()
        acceleration = self.calculate_acceleration(speed_data)
        distance_data = self.calculate_cumulative_distance()
        
        # Create or access fish_metrics group
        if 'fish_metrics' in self.root:
            if overwrite:
                del self.root['fish_metrics']
            else:
                print("  Fish metrics already exist. Use --overwrite to replace.")
                return
        
        metrics_group = self.root.create_group('fish_metrics')
        metrics_group.attrs['created_at'] = datetime.now().isoformat()
        metrics_group.attrs['fps'] = self.fps
        if self.pixel_to_mm:
            metrics_group.attrs['pixel_to_mm'] = self.pixel_to_mm
        if self.interpolation_run:
            metrics_group.attrs['interpolation_run'] = self.interpolation_run
        
        # Save speed
        speed_inst = metrics_group.create_dataset(
            'speed_instantaneous', 
            data=speed_data['instantaneous_speed'], 
            chunks=True
        )
        speed_inst.attrs['units'] = speed_data['units']
        speed_inst.attrs['description'] = 'Frame-to-frame speed'
        
        speed_smooth = metrics_group.create_dataset(
            'speed_smoothed', 
            data=speed_data['smoothed_speed'], 
            chunks=True
        )
        speed_smooth.attrs['units'] = speed_data['units']
        speed_smooth.attrs['description'] = 'Smoothed speed (5-frame window)'
        
        # Save acceleration
        acc_ds = metrics_group.create_dataset(
            'acceleration', 
            data=acceleration, 
            chunks=True
        )
        acc_ds.attrs['units'] = f"{speed_data['units']}/s"
        acc_ds.attrs['description'] = 'Rate of change of smoothed speed'
        
        # Save distance metrics
        cum_dist = metrics_group.create_dataset(
            'cumulative_distance', 
            data=distance_data['cumulative_distance'], 
            chunks=True
        )
        cum_dist.attrs['units'] = distance_data['units']
        cum_dist.attrs['description'] = 'Total path length traveled'
        
        frame_dist = metrics_group.create_dataset(
            'frame_distances', 
            data=distance_data['frame_distances'], 
            chunks=True
        )
        frame_dist.attrs['units'] = distance_data['units']
        frame_dist.attrs['description'] = 'Distance moved per frame'
        
        # Save summary statistics
        valid_speeds = speed_data['instantaneous_speed'][~np.isnan(speed_data['instantaneous_speed'])]
        if len(valid_speeds) > 0:
            metrics_group.attrs['mean_speed'] = float(np.mean(valid_speeds))
            metrics_group.attrs['median_speed'] = float(np.median(valid_speeds))
            metrics_group.attrs['max_speed'] = float(np.max(valid_speeds))
            metrics_group.attrs['speed_95th_percentile'] = float(np.percentile(valid_speeds, 95))
            metrics_group.attrs['total_distance'] = float(distance_data['cumulative_distance'][-1])
        
        if self.verbose:
            print("  ✅ Fish metrics saved successfully!")
    
    def plot_metrics(self, save_path: Optional[str] = None, show_plot: bool = True):
        """Create visualization of the metrics."""
        if 'fish_metrics' not in self.root:
            print("No fish metrics found. Run save_metrics() first.")
            return
        
        metrics = self.root['fish_metrics']
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Time axis
        n_frames = len(self.positions)
        time = np.arange(n_frames) / self.fps
        
        # Plot 1: Speed over time
        ax = axes[0, 0]
        speed_smooth = metrics['speed_smoothed'][:]
        speed_inst = metrics['speed_instantaneous'][:]
        speed_units = metrics['speed_smoothed'].attrs['units']
        
        # Plot instantaneous as light line, smoothed as bold
        ax.plot(time, speed_inst, 'b-', alpha=0.3, linewidth=0.5, label='Instantaneous')
        ax.plot(time, speed_smooth, 'b-', alpha=0.8, linewidth=2, label='Smoothed')
        
        # Add mean line
        valid_speeds = speed_smooth[~np.isnan(speed_smooth)]
        if len(valid_speeds) > 0:
            mean_speed = np.mean(valid_speeds)
            ax.axhline(mean_speed, color='r', linestyle='--', alpha=0.5, 
                      label=f'Mean: {mean_speed:.1f}')
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(f'Speed ({speed_units})', fontsize=11)
        ax.set_title('Fish Swimming Speed', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Acceleration
        ax = axes[0, 1]
        acc = metrics['acceleration'][:]
        acc_units = metrics['acceleration'].attrs['units']
        
        # Color positive and negative acceleration differently
        ax.fill_between(time, 0, acc, where=(acc > 0), color='green', alpha=0.3, label='Accelerating')
        ax.fill_between(time, 0, acc, where=(acc < 0), color='red', alpha=0.3, label='Decelerating')
        ax.plot(time, acc, 'k-', alpha=0.5, linewidth=1)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(f'Acceleration ({acc_units})', fontsize=11)
        ax.set_title('Acceleration', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative distance
        ax = axes[1, 0]
        cum_dist = metrics['cumulative_distance'][:]
        dist_units = metrics['cumulative_distance'].attrs['units']
        
        ax.plot(time, cum_dist, 'm-', alpha=0.8, linewidth=2)
        ax.fill_between(time, 0, cum_dist, alpha=0.2, color='m')
        
        # Add total distance text
        total_dist = cum_dist[-1]
        ax.text(0.05, 0.95, f'Total: {total_dist:.1f} {dist_units}', 
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(f'Distance ({dist_units})', fontsize=11)
        ax.set_title('Cumulative Distance Traveled', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Trajectory colored by speed
        ax = axes[1, 1]
        valid = ~np.isnan(self.positions[:, 0]) & ~np.isnan(speed_smooth)
        
        if np.any(valid):
            scatter = ax.scatter(self.positions[valid, 0], self.positions[valid, 1],
                               c=speed_smooth[valid], cmap='viridis', 
                               s=1, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label=f'Speed ({speed_units})')
            
            # Mark start and end
            first_valid = np.where(valid)[0][0]
            last_valid = np.where(valid)[0][-1]
            ax.plot(self.positions[first_valid, 0], self.positions[first_valid, 1], 
                   'go', markersize=10, label='Start')
            ax.plot(self.positions[last_valid, 0], self.positions[last_valid, 1], 
                   'ro', markersize=10, label='End')
            ax.legend()
        
        ax.set_xlabel('X (pixels)', fontsize=11)
        ax.set_ylabel('Y (pixels)', fontsize=11)
        ax.set_title('Swimming Trajectory (colored by speed)', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert y-axis for image coordinates
        
        plt.suptitle('Fish Behavioral Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate simple fish behavioral metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s detections.zarr
  %(prog)s detections.zarr --fps 60
  %(prog)s detections.zarr --pixel-to-mm 0.1
  %(prog)s detections.zarr --plot --save-plot metrics.png
  %(prog)s detections.zarr --overwrite
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file with tracking data')
    parser.add_argument('--run', dest='interpolation_run',
                       help='Specific interpolation run to use')
    parser.add_argument('--fps', type=float, default=60.0,
                       help='Video frame rate (default: 60 Hz)')
    parser.add_argument('--pixel-to-mm', type=float,
                       help='Conversion factor from pixels to millimeters')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing metrics')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--save-plot', help='Path to save plot')
    parser.add_argument('--no-show', action='store_true',
                       help="Don't display the plot")
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SimpleFishMetrics(
        zarr_path=args.zarr_path,
        interpolation_run=args.interpolation_run,
        fps=args.fps,
        pixel_to_mm=args.pixel_to_mm,
        verbose=not args.quiet
    )
    
    # Save metrics
    analyzer.save_metrics(overwrite=args.overwrite)
    
    # Plot if requested
    if args.plot or args.save_plot:
        analyzer.plot_metrics(
            save_path=args.save_plot,
            show_plot=not args.no_show
        )
    
    return 0


if __name__ == '__main__':
    exit(main())