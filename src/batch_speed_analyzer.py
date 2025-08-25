#!/usr/bin/env python3
"""
Batch Speed Analyzer

Calculates speed metrics for all ROIs with comprehensive reporting.
Generates individual and comparative visualizations.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
from datetime import datetime
import json
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import seaborn as sns
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

console = Console()


class BatchSpeedAnalyzer:
    """Batch speed analysis for multiple ROIs."""
    
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
        """Get all positions for a specific ROI."""
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
                    # Calculate centroid in pixels
                    centroid_x = ((bbox[0] + bbox[2]) / 2) * self.img_width
                    centroid_y = ((bbox[1] + bbox[3]) / 2) * self.img_height
                    positions[frame_idx] = np.array([centroid_x, centroid_y])
            
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
                        # Calculate centroid
                        centroid_x = ((bbox[0] + bbox[2]) / 2) * self.img_width
                        centroid_y = ((bbox[1] + bbox[3]) / 2) * self.img_height
                        positions[frame_idx] = np.array([centroid_x, centroid_y])
        
        return positions
    
    def calculate_speed_metrics(self, positions: Dict, window_size: int = 5,
                               max_speed_threshold: float = 1000.0):
        """Calculate speed metrics from positions."""
        if len(positions) < 2:
            return None
        
        # Sort by frame
        sorted_frames = sorted(positions.keys())
        total_frames = max(sorted_frames) + 1
        
        # Calculate frame-to-frame distances and speeds
        frame_distances = np.full(total_frames, np.nan)
        instantaneous_speed = np.full(total_frames, np.nan)
        
        for i in range(1, len(sorted_frames)):
            current_frame = sorted_frames[i]
            prev_frame = sorted_frames[i-1]
            frame_gap = current_frame - prev_frame
            
            # Only calculate speed for reasonable gaps
            if frame_gap <= 10:
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
        valid_smooth = smoothed_speed[~np.isnan(smoothed_speed)]
        valid_distances = frame_distances[~np.isnan(frame_distances)]
        
        if len(valid_smooth) == 0:
            return None
        
        stats = {
            'mean_speed_px_s': float(np.mean(valid_smooth)),
            'median_speed_px_s': float(np.median(valid_smooth)),
            'std_speed_px_s': float(np.std(valid_smooth)),
            'max_speed_px_s': float(np.max(valid_smooth)),
            'min_speed_px_s': float(np.min(valid_smooth)),
            'percentile_25_px_s': float(np.percentile(valid_smooth, 25)),
            'percentile_75_px_s': float(np.percentile(valid_smooth, 75)),
            'percentile_95_px_s': float(np.percentile(valid_smooth, 95)),
            'total_distance_px': float(np.sum(valid_distances))
        }
        
        # Add mm conversions if calibration available
        if self.pixel_to_mm:
            for key in list(stats.keys()):
                if 'px' in key:
                    if 'distance' in key:
                        mm_key = key.replace('_px', '_mm')
                        stats[mm_key] = stats[key] * self.pixel_to_mm
                    else:  # speed
                        mm_key = key.replace('_px_s', '_mm_s')
                        stats[mm_key] = stats[key] * self.pixel_to_mm
        
        return {
            'instantaneous_speed': instantaneous_speed,
            'smoothed_speed': smoothed_speed,
            'frame_distances': frame_distances,
            'statistics': stats,
            'coverage': len(positions) / total_frames * 100,
            'detected_frames': len(positions),
            'total_frames': total_frames
        }
    
    def process_all_rois(self, window_size: int = 5, max_speed_threshold: float = 1000.0,
                        use_interpolated: bool = True):
        """Process speed for all ROIs."""
        # Get number of ROIs
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        num_rois = n_detections_per_roi.shape[1]
        
        all_results = {}
        
        # Process with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"[cyan]Processing {num_rois} ROIs...", total=num_rois)
            
            for roi_id in range(num_rois):
                progress.update(task, description=f"[cyan]Analyzing ROI {roi_id}/{num_rois-1}")
                
                # Get positions
                positions = self.get_roi_positions(roi_id, use_interpolated)
                
                # Calculate metrics
                if len(positions) >= 2:
                    metrics = self.calculate_speed_metrics(positions, window_size, max_speed_threshold)
                    if metrics:
                        all_results[roi_id] = {
                            'roi_id': roi_id,
                            'positions': positions,
                            **metrics
                        }
                
                progress.advance(task)
        
        return all_results
    
    def save_results(self, results: Dict):
        """Save speed results to zarr."""
        if 'batch_speed_metrics' not in self.root:
            self.root.create_group('batch_speed_metrics')
        
        metrics_group = self.root['batch_speed_metrics']
        
        # Create timestamped run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f'speed_batch_{timestamp}'
        run_group = metrics_group.create_group(run_name)
        
        # Save metadata
        run_group.attrs.update({
            'created_at': datetime.now().isoformat(),
            'fps': self.fps,
            'pixel_to_mm': self.pixel_to_mm,
            'num_rois': len(results)
        })
        
        # Aggregate statistics for summary
        all_stats = {}
        
        for roi_id, roi_data in results.items():
            roi_group = run_group.create_group(f'roi_{roi_id}')
            
            # Save arrays
            roi_group.create_dataset('instantaneous_speed', 
                                    data=roi_data['instantaneous_speed'],
                                    dtype='float32')
            roi_group.create_dataset('smoothed_speed',
                                    data=roi_data['smoothed_speed'],
                                    dtype='float32')
            roi_group.create_dataset('frame_distances',
                                    data=roi_data['frame_distances'],
                                    dtype='float32')
            
            # Save statistics
            roi_group.attrs['statistics'] = json.dumps(roi_data['statistics'])
            roi_group.attrs['coverage'] = roi_data['coverage']
            roi_group.attrs['detected_frames'] = roi_data['detected_frames']
            roi_group.attrs['total_frames'] = roi_data['total_frames']
            
            all_stats[str(roi_id)] = roi_data['statistics']
        
        # Save aggregated statistics
        run_group.attrs['all_statistics'] = json.dumps(all_stats)
        
        # Update latest
        metrics_group.attrs['latest'] = run_name
        
        console.print(f"[green]✓ Speed metrics saved to:[/green] batch_speed_metrics/{run_name}")
    
    def print_summary(self, results: Dict):
        """Print summary table."""
        table = Table(title="Speed Analysis Summary")
        
        table.add_column("ROI", style="cyan", no_wrap=True)
        table.add_column("Coverage", style="yellow")
        table.add_column("Mean Speed", style="green")
        table.add_column("Median Speed", style="blue")
        table.add_column("Max Speed", style="red")
        table.add_column("Distance", style="magenta")
        
        # Determine units
        if self.pixel_to_mm:
            speed_unit = "mm/s"
            dist_unit = "mm"
            speed_key = "mean_speed_mm_s"
            median_key = "median_speed_mm_s"
            max_key = "max_speed_mm_s"
            dist_key = "total_distance_mm"
        else:
            speed_unit = "px/s"
            dist_unit = "px"
            speed_key = "mean_speed_px_s"
            median_key = "median_speed_px_s"
            max_key = "max_speed_px_s"
            dist_key = "total_distance_px"
        
        for roi_id in sorted(results.keys()):
            roi_data = results[roi_id]
            stats = roi_data['statistics']
            
            table.add_row(
                str(roi_id),
                f"{roi_data['coverage']:.1f}%",
                f"{stats[speed_key]:.1f}",
                f"{stats[median_key]:.1f}",
                f"{stats[max_key]:.1f}",
                f"{stats[dist_key]:.0f}"
            )
        
        console.print(table)
        
        # Print overall statistics
        all_speeds = []
        all_distances = []
        
        for roi_data in results.values():
            all_speeds.append(roi_data['statistics'][speed_key])
            all_distances.append(roi_data['statistics'][dist_key])
        
        console.print(f"\n[bold cyan]Overall Statistics:[/bold cyan]")
        console.print(f"  Mean speed across all ROIs: {np.mean(all_speeds):.1f} {speed_unit}")
        console.print(f"  Median speed across all ROIs: {np.median(all_speeds):.1f} {speed_unit}")
        console.print(f"  Total distance (all ROIs): {np.sum(all_distances):.0f} {dist_unit}")
        
        if self.pixel_to_mm:
            console.print(f"  Total distance in meters: {np.sum(all_distances)/1000:.2f} m")
    
    def create_comparative_plot(self, results: Dict, save_path: Optional[Path] = None):
        """Create comparative visualization for all ROIs."""
        num_rois = len(results)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)
        
        # Determine units
        if self.pixel_to_mm:
            speed_unit = "mm/s"
            dist_unit = "mm"
            prefix = "mm"
        else:
            speed_unit = "pixels/s"
            dist_unit = "pixels"
            prefix = "px"
        
        # 1. Speed comparison boxplot
        ax1 = fig.add_subplot(gs[0, :])
        
        speed_data = []
        roi_labels = []
        
        for roi_id in sorted(results.keys()):
            speeds = results[roi_id]['smoothed_speed']
            valid_speeds = speeds[~np.isnan(speeds)]
            if self.pixel_to_mm:
                valid_speeds *= self.pixel_to_mm
            speed_data.append(valid_speeds)
            roi_labels.append(f"ROI {roi_id}")
        
        bp = ax1.boxplot(speed_data, labels=roi_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], plt.cm.tab20(np.linspace(0, 1, num_rois))):
            patch.set_facecolor(color)
        
        ax1.set_ylabel(f'Speed ({speed_unit})')
        ax1.set_title('Speed Distribution Comparison')
        ax1.grid(True, alpha=0.3)
        
        # 2. Mean speeds bar chart
        ax2 = fig.add_subplot(gs[1, 0])
        
        roi_ids = sorted(results.keys())
        mean_speeds = [results[roi][f'statistics'][f'mean_speed_{prefix}_s'] for roi in roi_ids]
        
        bars = ax2.bar(roi_ids, mean_speeds, color=plt.cm.tab20(np.linspace(0, 1, num_rois)))
        ax2.set_xlabel('ROI')
        ax2.set_ylabel(f'Mean Speed ({speed_unit})')
        ax2.set_title('Average Speed by ROI')
        ax2.grid(True, alpha=0.3)
        
        # 3. Total distance bar chart
        ax3 = fig.add_subplot(gs[1, 1])
        
        distances = [results[roi]['statistics'][f'total_distance_{prefix}'] for roi in roi_ids]
        
        bars = ax3.bar(roi_ids, distances, color=plt.cm.tab20(np.linspace(0, 1, num_rois)))
        ax3.set_xlabel('ROI')
        ax3.set_ylabel(f'Total Distance ({dist_unit})')
        ax3.set_title('Total Distance Traveled')
        ax3.grid(True, alpha=0.3)
        
        # 4. Speed over time for all ROIs
        ax4 = fig.add_subplot(gs[2, :])
        
        for roi_id in sorted(results.keys()):
            speeds = results[roi_id]['smoothed_speed']
            if self.pixel_to_mm:
                speeds = speeds * self.pixel_to_mm
            
            valid_mask = ~np.isnan(speeds)
            if np.any(valid_mask):
                time_seconds = np.arange(len(speeds)) / self.fps
                ax4.plot(time_seconds[valid_mask], speeds[valid_mask], 
                        alpha=0.6, linewidth=1, label=f'ROI {roi_id}')
        
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel(f'Speed ({speed_unit})')
        ax4.set_title('Speed Profiles Over Time')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Batch Speed Analysis - All ROIs', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Figure saved to:[/green] {save_path}")
        
        plt.show()
    
    def export_to_csv(self, results: Dict, output_path: Path):
        """Export results to CSV for further analysis."""
        data_rows = []
        
        for roi_id, roi_data in results.items():
            stats = roi_data['statistics']
            row = {
                'roi_id': roi_id,
                'coverage_percent': roi_data['coverage'],
                'detected_frames': roi_data['detected_frames'],
                'total_frames': roi_data['total_frames']
            }
            row.update(stats)
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        df.to_csv(output_path, index=False)
        console.print(f"[green]✓ Results exported to:[/green] {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch speed analysis for all ROIs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with visualization
  %(prog)s detections.zarr --visualize
  
  # Custom window size
  %(prog)s detections.zarr --window 10
  
  # Export results to CSV
  %(prog)s detections.zarr --export results.csv
  
  # Save visualization
  %(prog)s detections.zarr --visualize --save-fig speed_comparison.png
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--window', type=int, default=5,
                       help='Smoothing window size in frames (default: 5)')
    parser.add_argument('--max-speed', type=float, default=1000.0,
                       help='Maximum reasonable speed in pixels/second (default: 1000)')
    parser.add_argument('--no-interpolated', action='store_true',
                       help='Do not use interpolated positions')
    parser.add_argument('--save', action='store_true',
                       help='Save results to zarr')
    parser.add_argument('--visualize', action='store_true',
                       help='Create comparative visualization')
    parser.add_argument('--save-fig', type=str,
                       help='Path to save figure')
    parser.add_argument('--export', type=str,
                       help='Export results to CSV file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = BatchSpeedAnalyzer(args.zarr_path, verbose=not args.quiet)
    
    console.print(f"\n[bold cyan]Batch Speed Analysis[/bold cyan]")
    console.print(f"File: {args.zarr_path}")
    console.print(f"Window size: {args.window} frames")
    
    # Process all ROIs
    results = analyzer.process_all_rois(
        window_size=args.window,
        max_speed_threshold=args.max_speed,
        use_interpolated=not args.no_interpolated
    )
    
    if results:
        # Print summary
        analyzer.print_summary(results)
        
        # Save to zarr if requested
        if args.save:
            analyzer.save_results(results)
        
        # Create visualization if requested
        if args.visualize:
            save_path = Path(args.save_fig) if args.save_fig else None
            analyzer.create_comparative_plot(results, save_path)
        
        # Export to CSV if requested
        if args.export:
            analyzer.export_to_csv(results, Path(args.export))
        
        console.print("\n[green]✓ Analysis complete![/green]")
    else:
        console.print("[red]No results to display[/red]")


if __name__ == "__main__":
    main()