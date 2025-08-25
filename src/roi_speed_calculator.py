#!/usr/bin/env python3
"""
ROI Speed Calculator

Calculates speed metrics for individual ROIs (fish) including:
- Instantaneous speed
- Smoothed speed with configurable windows
- Speed statistics
- Distance traveled
Works with both original and interpolated detections.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
from datetime import datetime
import json
from rich.console import Console
from rich.table import Table
import warnings
from typing import Optional
warnings.filterwarnings('ignore')

console = Console()


class ROISpeedCalculator:
    """Calculate speed metrics for individual ROIs."""
    
    def __init__(self, zarr_path: str, verbose: bool = True):
        self.zarr_path = Path(zarr_path)
        self.verbose = verbose
        self.root = zarr.open_group(self.zarr_path, mode='r+')
        self.fps = self.root.attrs.get('fps', 60.0)
        
        # Get dimensions
        self.img_width = self.root.attrs.get('width', 4512)
        self.img_height = self.root.attrs.get('height', 4512)
        
        # Check for calibration
        self.pixel_to_mm = None
        if 'calibration' in self.root:
            self.pixel_to_mm = self.root['calibration'].attrs.get('pixel_to_mm', None)
            if self.verbose and self.pixel_to_mm:
                console.print(f"[green]Calibration found:[/green] 1 pixel = {self.pixel_to_mm:.4f} mm")
    
    def get_roi_positions(self, roi_id: int, use_interpolated: bool = True):
        """
        Get all positions for a specific ROI.
        
        Args:
            roi_id: The ROI ID to analyze
            use_interpolated: Whether to include interpolated positions
            
        Returns:
            Dict with frame_idx -> position mapping
        """
        # Load detection data
        detect_group = self.root['detect_runs']
        latest_detect = detect_group.attrs['latest']
        n_detections = detect_group[latest_detect]['n_detections'][:]
        bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
        
        # Load ID assignments
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        detection_ids = id_group[latest_id]['detection_ids'][:]
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        
        # Extract original positions
        positions = {}
        cumulative_idx = 0
        
        for frame_idx in range(len(n_detections)):
            frame_det_count = int(n_detections[frame_idx])
            
            if frame_det_count > 0 and n_detections_per_roi[frame_idx, roi_id] > 0:
                frame_detection_ids = detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
                roi_mask = frame_detection_ids == roi_id
                
                if np.any(roi_mask):
                    roi_idx = np.where(roi_mask)[0][0]
                    bbox = bbox_coords[cumulative_idx + roi_idx]
                    # Store in pixels
                    positions[frame_idx] = np.array([
                        bbox[0] * self.img_width,
                        bbox[1] * self.img_height
                    ])
            
            cumulative_idx += frame_det_count
        
        # Add interpolated positions if available and requested
        if use_interpolated and 'interpolated_detections' in self.root:
            interp_det_group = self.root['interpolated_detections']
            if 'latest' in interp_det_group.attrs:
                latest_det = interp_det_group.attrs['latest']
                det_group = interp_det_group[latest_det]
                
                frame_indices = det_group['frame_indices'][:]
                roi_ids = det_group['roi_ids'][:]
                bboxes = det_group['bboxes'][:]
                
                for i in range(len(frame_indices)):
                    if int(roi_ids[i]) == roi_id:
                        frame_idx = int(frame_indices[i])
                        bbox = bboxes[i]
                        positions[frame_idx] = np.array([
                            bbox[0] * self.img_width,
                            bbox[1] * self.img_height
                        ])
        
        return positions
    
    def calculate_speed(self, roi_id: int, window_size: int = 5,
                       max_speed_threshold: float = 1000.0,
                       use_interpolated: bool = True):
        """
        Calculate speed metrics for a specific ROI.
        
        Args:
            roi_id: The ROI ID to analyze
            window_size: Smoothing window size in frames
            max_speed_threshold: Maximum reasonable speed in pixels/second
            use_interpolated: Whether to use interpolated positions
            
        Returns:
            Dict with speed metrics
        """
        # Get positions
        positions = self.get_roi_positions(roi_id, use_interpolated)
        
        if len(positions) < 2:
            if self.verbose:
                console.print(f"[yellow]ROI {roi_id}: Not enough detections for speed calculation[/yellow]")
            return None
        
        # Sort by frame
        sorted_frames = sorted(positions.keys())
        total_frames = max(sorted_frames) + 1
        
        # Calculate frame-to-frame distances
        frame_distances = np.full(total_frames, np.nan)
        instantaneous_speed = np.full(total_frames, np.nan)
        
        for i in range(1, len(sorted_frames)):
            current_frame = sorted_frames[i]
            prev_frame = sorted_frames[i-1]
            frame_gap = current_frame - prev_frame
            
            # Only calculate speed for reasonable gaps
            if frame_gap <= 10:  # Max 10 frame gap
                dist = np.linalg.norm(positions[current_frame] - positions[prev_frame])
                frame_distances[current_frame] = dist
                
                # Speed = distance * fps / frame_gap
                speed = (dist * self.fps) / frame_gap
                
                # Filter unreasonable speeds
                if speed <= max_speed_threshold:
                    instantaneous_speed[current_frame] = speed
        
        # Calculate smoothed speed
        smoothed_speed = np.full_like(instantaneous_speed, np.nan)
        valid_mask = ~np.isnan(instantaneous_speed)
        
        if np.sum(valid_mask) > window_size:
            valid_indices = np.where(valid_mask)[0]
            for idx in valid_indices:
                window_start = max(0, idx - window_size // 2)
                window_end = min(len(instantaneous_speed), idx + window_size // 2 + 1)
                window_data = instantaneous_speed[window_start:window_end]
                
                valid_in_window = window_data[~np.isnan(window_data)]
                if len(valid_in_window) > 0:
                    smoothed_speed[idx] = np.mean(valid_in_window)
        
        # Calculate statistics
        valid_instant = instantaneous_speed[~np.isnan(instantaneous_speed)]
        valid_smooth = smoothed_speed[~np.isnan(smoothed_speed)]
        
        # Calculate total distance
        valid_distances = frame_distances[~np.isnan(frame_distances)]
        total_distance_px = np.sum(valid_distances)
        
        results = {
            'roi_id': roi_id,
            'instantaneous_speed': instantaneous_speed,
            'smoothed_speed': smoothed_speed,
            'frame_distances': frame_distances,
            'positions': positions,
            'total_frames': total_frames,
            'detected_frames': len(positions),
            'coverage': len(positions) / total_frames * 100,
            'window_size': window_size,
            'statistics': {
                'mean_speed_px_s': float(np.mean(valid_smooth)) if len(valid_smooth) > 0 else 0,
                'median_speed_px_s': float(np.median(valid_smooth)) if len(valid_smooth) > 0 else 0,
                'std_speed_px_s': float(np.std(valid_smooth)) if len(valid_smooth) > 0 else 0,
                'max_speed_px_s': float(np.max(valid_smooth)) if len(valid_smooth) > 0 else 0,
                'min_speed_px_s': float(np.min(valid_smooth)) if len(valid_smooth) > 0 else 0,
                'percentile_25_px_s': float(np.percentile(valid_smooth, 25)) if len(valid_smooth) > 0 else 0,
                'percentile_75_px_s': float(np.percentile(valid_smooth, 75)) if len(valid_smooth) > 0 else 0,
                'total_distance_px': float(total_distance_px),
                'mean_instant_speed_px_s': float(np.mean(valid_instant)) if len(valid_instant) > 0 else 0,
                'max_instant_speed_px_s': float(np.max(valid_instant)) if len(valid_instant) > 0 else 0,
            }
        }
        
        # Add mm conversions if calibration available
        if self.pixel_to_mm:
            mm_stats = {}
            for key, value in results['statistics'].items():
                if 'px' in key:
                    if 'distance' in key:
                        mm_key = key.replace('_px', '_mm')
                        mm_stats[mm_key] = value * self.pixel_to_mm
                    else:  # speed
                        mm_key = key.replace('_px_s', '_mm_s')
                        mm_stats[mm_key] = value * self.pixel_to_mm
            results['statistics'].update(mm_stats)
        
        return results
    
    def calculate_all_rois(self, window_size: int = 5,
                          max_speed_threshold: float = 1000.0,
                          use_interpolated: bool = True):
        """Calculate speed for all ROIs."""
        # Get number of ROIs
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        num_rois = n_detections_per_roi.shape[1]
        
        all_results = {}
        
        for roi_id in range(num_rois):
            if self.verbose:
                console.print(f"Processing ROI {roi_id}...")
            
            results = self.calculate_speed(roi_id, window_size, 
                                          max_speed_threshold, use_interpolated)
            if results:
                all_results[roi_id] = results
        
        return all_results
    
    def save_results(self, results: dict):
        """Save speed results to zarr."""
        if 'roi_speed_metrics' not in self.root:
            self.root.create_group('roi_speed_metrics')
        
        metrics_group = self.root['roi_speed_metrics']
        
        # Create timestamped run
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f'speed_run_{timestamp}'
        run_group = metrics_group.create_group(run_name)
        
        # Save metadata
        run_group.attrs.update({
            'created_at': datetime.now().isoformat(),
            'fps': self.fps,
            'pixel_to_mm': self.pixel_to_mm if self.pixel_to_mm else None,
            'num_rois': len(results)
        })
        
        # Save each ROI's results
        for roi_id, roi_results in results.items():
            roi_group = run_group.create_group(f'roi_{roi_id}')
            
            # Save arrays
            roi_group.create_dataset('instantaneous_speed', 
                                    data=roi_results['instantaneous_speed'],
                                    dtype='float32')
            roi_group.create_dataset('smoothed_speed',
                                    data=roi_results['smoothed_speed'],
                                    dtype='float32')
            roi_group.create_dataset('frame_distances',
                                    data=roi_results['frame_distances'],
                                    dtype='float32')
            
            # Save statistics as attributes
            roi_group.attrs['statistics'] = json.dumps(roi_results['statistics'])
            roi_group.attrs['window_size'] = roi_results['window_size']
            roi_group.attrs['coverage'] = roi_results['coverage']
            roi_group.attrs['detected_frames'] = roi_results['detected_frames']
            roi_group.attrs['total_frames'] = roi_results['total_frames']
        
        # Update latest
        metrics_group.attrs['latest'] = run_name
        
        if self.verbose:
            console.print(f"[green]✓ Speed metrics saved to:[/green] roi_speed_metrics/{run_name}")
    
    def print_summary(self, results: dict):
        """Print summary table of speed metrics."""
        table = Table(title="ROI Speed Metrics Summary")
        
        table.add_column("ROI", style="cyan", no_wrap=True)
        table.add_column("Coverage", style="yellow")
        table.add_column("Mean Speed", style="green")
        table.add_column("Max Speed", style="red")
        table.add_column("Distance", style="magenta")
        
        # Determine units
        if self.pixel_to_mm:
            speed_unit = "mm/s"
            dist_unit = "mm"
            speed_key = "mean_speed_mm_s"
            max_key = "max_speed_mm_s"
            dist_key = "total_distance_mm"
        else:
            speed_unit = "px/s"
            dist_unit = "px"
            speed_key = "mean_speed_px_s"
            max_key = "max_speed_px_s"
            dist_key = "total_distance_px"
        
        for roi_id in sorted(results.keys()):
            roi_result = results[roi_id]
            stats = roi_result['statistics']
            
            table.add_row(
                str(roi_id),
                f"{roi_result['coverage']:.1f}%",
                f"{stats[speed_key]:.1f} {speed_unit}",
                f"{stats[max_key]:.1f} {speed_unit}",
                f"{stats[dist_key]:.1f} {dist_unit}"
            )
        
        console.print(table)
    
    def visualize_roi_speed(self, roi_id: int, results: dict, save_path: Optional[Path] = None):
        """Create visualization for a single ROI's speed."""
        roi_result = results[roi_id]
        
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)
        
        # Time axis
        time_seconds = np.arange(roi_result['total_frames']) / self.fps
        
        # Determine units and keys for statistics
        if self.pixel_to_mm:
            speed_unit = "mm/s"
            speed_factor = self.pixel_to_mm
            # Keys for statistics
            mean_key = "mean_speed_mm_s"
            median_key = "median_speed_mm_s"
            std_key = "std_speed_mm_s"
            max_key = "max_speed_mm_s"
            p25_key = "percentile_25_mm_s"
            p75_key = "percentile_75_mm_s"
            dist_key = "total_distance_mm"
        else:
            speed_unit = "pixels/s"
            speed_factor = 1.0
            # Keys for statistics
            mean_key = "mean_speed_px_s"
            median_key = "median_speed_px_s"
            std_key = "std_speed_px_s"
            max_key = "max_speed_px_s"
            p25_key = "percentile_25_px_s"
            p75_key = "percentile_75_px_s"
            dist_key = "total_distance_px"
        
        # 1. Speed over time
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot instantaneous speed
        inst_speed = roi_result['instantaneous_speed'] * speed_factor
        valid_inst = ~np.isnan(inst_speed)
        ax1.plot(time_seconds[valid_inst], inst_speed[valid_inst],
                'gray', alpha=0.3, linewidth=0.5, label='Instantaneous')
        
        # Plot smoothed speed
        smooth_speed = roi_result['smoothed_speed'] * speed_factor
        valid_smooth = ~np.isnan(smooth_speed)
        ax1.plot(time_seconds[valid_smooth], smooth_speed[valid_smooth],
                'blue', linewidth=2, label=f'Smoothed (window={roi_result["window_size"]})')
        
        # Add mean line
        mean_speed = roi_result['statistics'][mean_key]
        ax1.axhline(y=mean_speed, color='red', linestyle='--', alpha=0.5,
                   label=f'Mean: {mean_speed:.1f} {speed_unit}')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel(f'Speed ({speed_unit})')
        ax1.set_title(f'ROI {roi_id} - Speed Over Time (Coverage: {roi_result["coverage"]:.1f}%)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Trajectory
        ax2 = fig.add_subplot(gs[1, 0])
        positions = roi_result['positions']
        if positions:
            sorted_frames = sorted(positions.keys())
            x_coords = [positions[f][0] for f in sorted_frames]
            y_coords = [positions[f][1] for f in sorted_frames]
            
            scatter = ax2.scatter(x_coords, y_coords, c=sorted_frames, 
                                 cmap='viridis', s=1, alpha=0.6)
            plt.colorbar(scatter, ax=ax2, label='Frame')
            
            # Mark start and end
            ax2.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start')
            ax2.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='End')
            
        ax2.set_xlabel('X Position (pixels)')
        ax2.set_ylabel('Y Position (pixels)')
        ax2.set_title('Trajectory')
        ax2.set_aspect('equal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Speed distribution
        ax3 = fig.add_subplot(gs[1, 1])
        valid_speeds = smooth_speed[valid_smooth]
        if len(valid_speeds) > 0:
            ax3.hist(valid_speeds, bins=50, color='blue', alpha=0.7, edgecolor='black')
            ax3.axvline(x=mean_speed, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_speed:.1f}')
            
            # Add percentiles
            p25 = roi_result['statistics'][p25_key]
            p75 = roi_result['statistics'][p75_key]
            ax3.axvline(x=p25, color='orange', linestyle=':', label=f'25%: {p25:.1f}')
            ax3.axvline(x=p75, color='orange', linestyle=':', label=f'75%: {p75:.1f}')
            
        ax3.set_xlabel(f'Speed ({speed_unit})')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Speed Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistics summary
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        stats = roi_result['statistics']
        if self.pixel_to_mm:
            summary_text = f"""
ROI {roi_id} Speed Statistics
{'='*50}

Coverage:              {roi_result['coverage']:.1f}% ({roi_result['detected_frames']}/{roi_result['total_frames']} frames)

Speed (mm/s):
  Mean:                {stats[mean_key]:.2f}
  Median:              {stats[median_key]:.2f}
  Std Dev:             {stats[std_key]:.2f}
  Maximum:             {stats[max_key]:.2f}
  25th Percentile:     {stats[p25_key]:.2f}
  75th Percentile:     {stats[p75_key]:.2f}

Distance:
  Total Distance:      {stats[dist_key]:.2f} mm ({stats[dist_key]/10:.2f} cm)
  
Speed (pixels/s):
  Mean:                {stats['mean_speed_px_s']:.2f}
  Maximum:             {stats['max_speed_px_s']:.2f}
            """
        else:
            summary_text = f"""
ROI {roi_id} Speed Statistics
{'='*50}

Coverage:              {roi_result['coverage']:.1f}% ({roi_result['detected_frames']}/{roi_result['total_frames']} frames)

Speed (pixels/s):
  Mean:                {stats[mean_key]:.2f}
  Median:              {stats[median_key]:.2f}
  Std Dev:             {stats[std_key]:.2f}
  Maximum:             {stats[max_key]:.2f}
  25th Percentile:     {stats[p25_key]:.2f}
  75th Percentile:     {stats[p75_key]:.2f}

Distance:
  Total Distance:      {stats[dist_key]:.2f} pixels
            """
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle(f'ROI {roi_id} - Speed Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Figure saved to:[/green] {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate speed metrics for individual ROIs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--roi', type=int, help='Specific ROI to analyze (default: all)')
    parser.add_argument('--window', type=int, default=5,
                       help='Smoothing window size in frames (default: 5)')
    parser.add_argument('--max-speed', type=float, default=1000.0,
                       help='Maximum reasonable speed in pixels/second (default: 1000)')
    parser.add_argument('--no-interpolated', action='store_true',
                       help='Do not use interpolated positions')
    parser.add_argument('--save', action='store_true',
                       help='Save results to zarr')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization')
    parser.add_argument('--save-fig', type=str,
                       help='Path to save figure')
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = ROISpeedCalculator(args.zarr_path)
    
    # Calculate speed
    if args.roi is not None:
        # Single ROI
        console.print(f"\n[bold cyan]Calculating speed for ROI {args.roi}[/bold cyan]")
        results = calculator.calculate_speed(
            args.roi, 
            args.window,
            args.max_speed,
            not args.no_interpolated
        )
        
        if results:
            all_results = {args.roi: results}
        else:
            console.print("[red]Failed to calculate speed[/red]")
            return
    else:
        # All ROIs
        console.print(f"\n[bold cyan]Calculating speed for all ROIs[/bold cyan]")
        all_results = calculator.calculate_all_rois(
            args.window,
            args.max_speed,
            not args.no_interpolated
        )
    
    # Print summary
    if all_results:
        calculator.print_summary(all_results)
        
        # Save if requested
        if args.save:
            calculator.save_results(all_results)
        
        # Visualize if requested
        if args.visualize:
            if args.roi is not None:
                save_path = Path(args.save_fig) if args.save_fig else None
                calculator.visualize_roi_speed(args.roi, all_results, save_path)
            else:
                console.print("[yellow]Visualization only available for single ROI (use --roi)[/yellow]")
    else:
        console.print("[red]No results to display[/red]")


if __name__ == "__main__":
    main()