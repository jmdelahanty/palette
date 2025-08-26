#!/usr/bin/env python3
"""
Zarr Speed Data Visualizer

Loads and visualizes previously calculated speed metrics from zarr files.
Supports both individual ROI analysis and batch comparisons.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import json
import argparse
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

console = Console()
sns.set_palette("husl")


class ZarrSpeedVisualizer:
    """Load and visualize speed data from zarr files."""
    
    def __init__(self, zarr_path: str, verbose: bool = True):
        self.zarr_path = Path(zarr_path)
        self.verbose = verbose
        self.root = zarr.open_group(self.zarr_path, mode='r')
        
        # Get metadata
        self.fps = self.root.attrs.get('fps', 60.0)
        self.img_width = self.root.attrs.get('width', 4512)
        self.img_height = self.root.attrs.get('height', 4512)
        
        # Check for calibration
        self.pixel_to_mm = None
        if 'calibration' in self.root:
            self.pixel_to_mm = self.root['calibration'].attrs.get('pixel_to_mm', None)
            if self.verbose and self.pixel_to_mm:
                console.print(f"[green]Calibration found:[/green] 1 pixel = {self.pixel_to_mm:.4f} mm")
        
        # Check available speed data
        self.has_batch_speed = 'batch_speed_metrics' in self.root
        self.has_speed_metrics = 'speed_metrics' in self.root
        
        if self.verbose:
            console.print("\n[cyan]Available speed data:[/cyan]")
            if self.has_batch_speed:
                console.print("  ✓ batch_speed_metrics")
            if self.has_speed_metrics:
                console.print("  ✓ speed_metrics")
            if not (self.has_batch_speed or self.has_speed_metrics):
                console.print("  [red]No speed data found![/red]")
    
    def list_available_runs(self) -> Dict[str, List[str]]:
        """List all available speed calculation runs."""
        runs = {}
        
        if self.has_batch_speed:
            batch_group = self.root['batch_speed_metrics']
            runs['batch_speed_metrics'] = list(batch_group.keys())
            if 'latest' in batch_group.attrs:
                runs['batch_latest'] = batch_group.attrs['latest']
        
        if self.has_speed_metrics:
            speed_group = self.root['speed_metrics']
            runs['speed_metrics'] = list(speed_group.keys())
            if 'latest' in speed_group.attrs:
                runs['speed_latest'] = speed_group.attrs['latest']
        
        return runs
    
    def load_batch_speed_data(self, run_name: Optional[str] = None) -> Dict:
        """Load batch speed data from zarr."""
        if not self.has_batch_speed:
            console.print("[red]No batch_speed_metrics found in zarr[/red]")
            return None
        
        batch_group = self.root['batch_speed_metrics']
        
        # Get run to load
        if run_name is None:
            if 'latest' in batch_group.attrs:
                run_name = batch_group.attrs['latest']
            else:
                runs = list(batch_group.keys())
                if not runs:
                    console.print("[red]No batch speed runs found[/red]")
                    return None
                run_name = sorted(runs)[-1]
        
        if run_name not in batch_group:
            console.print(f"[red]Run '{run_name}' not found[/red]")
            return None
        
        run_group = batch_group[run_name]
        console.print(f"[green]Loading batch speed data:[/green] {run_name}")
        
        # Load data for all ROIs
        results = {}
        for roi_key in run_group.keys():
            if roi_key.startswith('roi_'):
                roi_id = int(roi_key.split('_')[1])
                roi_group = run_group[roi_key]
                
                results[roi_id] = {
                    'instantaneous_speed': roi_group['instantaneous_speed'][:],
                    'smoothed_speed': roi_group['smoothed_speed'][:],
                    'frame_distances': roi_group['frame_distances'][:],
                    'statistics': json.loads(roi_group.attrs['statistics']),
                    'coverage': roi_group.attrs['coverage'],
                    'detected_frames': roi_group.attrs['detected_frames'],
                    'total_frames': roi_group.attrs['total_frames']
                }
        
        # Add metadata
        metadata = {
            'run_name': run_name,
            'created_at': run_group.attrs.get('created_at', 'Unknown'),
            'fps': run_group.attrs.get('fps', self.fps),
            'pixel_to_mm': run_group.attrs.get('pixel_to_mm', self.pixel_to_mm),
            'num_rois': len(results)
        }
        
        return {'rois': results, 'metadata': metadata}
    
    def plot_individual_roi_speed(self, roi_id: int, data: Dict,
                                 save_path: Optional[Path] = None):
        """Plot speed analysis for a single ROI."""
        if roi_id not in data['rois']:
            console.print(f"[red]ROI {roi_id} not found in data[/red]")
            return
        
        roi_data = data['rois'][roi_id]
        metadata = data['metadata']
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.25)
        
        # Determine units
        if metadata['pixel_to_mm']:
            speed_unit = 'mm/s'
            speed_conv = metadata['pixel_to_mm']
        else:
            speed_unit = 'pixels/s'
            speed_conv = 1.0
        
        # Time axis
        total_frames = roi_data['total_frames']
        time_seconds = np.arange(total_frames) / metadata['fps']
        
        # Convert speeds
        inst_speed = roi_data['instantaneous_speed'] * speed_conv
        smooth_speed = roi_data['smoothed_speed'] * speed_conv
        
        # 1. Speed over time
        ax1 = fig.add_subplot(gs[0:2, :])
        
        valid_inst = ~np.isnan(inst_speed)
        valid_smooth = ~np.isnan(smooth_speed)
        
        ax1.plot(time_seconds[valid_inst], inst_speed[valid_inst],
                'b-', alpha=0.3, linewidth=0.5, label='Instantaneous')
        ax1.plot(time_seconds[valid_smooth], smooth_speed[valid_smooth],
                'r-', linewidth=2, label='Smoothed')
        
        # Add mean line
        mean_key = f'mean_speed_{"mm" if metadata["pixel_to_mm"] else "px"}_s'
        mean_speed = roi_data['statistics'].get(mean_key, 0)
        ax1.axhline(y=mean_speed, color='green', linestyle='--', alpha=0.5,
                   label=f'Mean: {mean_speed:.2f} {speed_unit}')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel(f'Speed ({speed_unit})')
        ax1.set_title(f'ROI {roi_id} - Speed Over Time')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Speed distribution
        ax2 = fig.add_subplot(gs[2, 0])
        
        valid_speeds = smooth_speed[~np.isnan(smooth_speed)]
        if len(valid_speeds) > 0:
            ax2.hist(valid_speeds, bins=50, alpha=0.7, color='blue', edgecolor='black')
            
            median_key = f'median_speed_{"mm" if metadata["pixel_to_mm"] else "px"}_s'
            median_speed = roi_data['statistics'].get(median_key, 0)
            
            ax2.axvline(x=mean_speed, color='green', linestyle='--', linewidth=2)
            ax2.axvline(x=median_speed, color='orange', linestyle='--', linewidth=2)
            
            ax2.set_xlabel(f'Speed ({speed_unit})')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Speed Distribution')
        
        ax2.grid(True, alpha=0.3)
        
        # 3. Statistics
        ax3 = fig.add_subplot(gs[2, 1])
        ax3.axis('off')
        
        stats = roi_data['statistics']
        prefix = 'mm' if metadata['pixel_to_mm'] else 'px'
        
        stats_text = f"""
Speed Statistics - ROI {roi_id}
━━━━━━━━━━━━━━━━━━━━━━━
Mean:     {stats.get(f'mean_speed_{prefix}_s', 0):.2f} {speed_unit}
Median:   {stats.get(f'median_speed_{prefix}_s', 0):.2f} {speed_unit}
Std Dev:  {stats.get(f'std_speed_{prefix}_s', 0):.2f} {speed_unit}
Maximum:  {stats.get(f'max_speed_{prefix}_s', 0):.2f} {speed_unit}

25th %ile: {stats.get(f'percentile_25_{prefix}_s', 0):.2f} {speed_unit}
75th %ile: {stats.get(f'percentile_75_{prefix}_s', 0):.2f} {speed_unit}
95th %ile: {stats.get(f'percentile_95_{prefix}_s', 0):.2f} {speed_unit}

Coverage:  {roi_data['coverage']:.1f}%
Frames:    {roi_data['detected_frames']} / {roi_data['total_frames']}
Distance:  {stats.get(f'total_distance_{prefix}', 0):.0f} {prefix}
"""
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Speed Analysis - ROI {roi_id} (from {metadata["run_name"]})',
                    fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Figure saved to:[/green] {save_path}")
        
        plt.show()
    
    def plot_batch_comparison(self, data: Dict, save_path: Optional[Path] = None):
        """Create comparative plots for all ROIs."""
        rois = data['rois']
        metadata = data['metadata']
        num_rois = len(rois)
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
        
        # Determine units
        if metadata['pixel_to_mm']:
            speed_unit = 'mm/s'
            dist_unit = 'mm'
            prefix = 'mm'
        else:
            speed_unit = 'pixels/s'
            dist_unit = 'pixels'
            prefix = 'px'
        
        # 1. Speed comparison boxplot
        ax1 = fig.add_subplot(gs[0, :])
        
        speed_data = []
        roi_labels = []
        
        for roi_id in sorted(rois.keys()):
            speeds = rois[roi_id]['smoothed_speed']
            valid_speeds = speeds[~np.isnan(speeds)]
            if metadata['pixel_to_mm']:
                valid_speeds *= metadata['pixel_to_mm']
            speed_data.append(valid_speeds)
            roi_labels.append(f"Fish {roi_id}")
        
        bp = ax1.boxplot(speed_data, labels=roi_labels, patch_artist=True)
        colors = plt.cm.tab20(np.linspace(0, 1, num_rois))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_ylabel(f'Speed ({speed_unit})')
        ax1.set_title('Speed Distribution Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Mean speeds
        ax2 = fig.add_subplot(gs[1, 0])
        
        roi_ids = sorted(rois.keys())
        mean_speeds = [rois[roi]['statistics'][f'mean_speed_{prefix}_s'] for roi in roi_ids]
        
        bars = ax2.bar(roi_ids, mean_speeds, color=colors)
        ax2.set_xlabel('Fish ID')
        ax2.set_ylabel(f'Mean Speed ({speed_unit})')
        ax2.set_title('Average Speed by Fish')
        ax2.grid(True, alpha=0.3)
        
        # 3. Coverage
        ax3 = fig.add_subplot(gs[1, 1])
        
        coverages = [rois[roi]['coverage'] for roi in roi_ids]
        
        bars = ax3.bar(roi_ids, coverages, color=colors)
        ax3.set_xlabel('Fish ID')
        ax3.set_ylabel('Coverage (%)')
        ax3.set_title('Detection Coverage')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 105])
        
        # 4. Total distance
        ax4 = fig.add_subplot(gs[1, 2])
        
        distances = [rois[roi]['statistics'][f'total_distance_{prefix}'] for roi in roi_ids]
        
        bars = ax4.bar(roi_ids, distances, color=colors)
        ax4.set_xlabel('Fish ID')
        ax4.set_ylabel(f'Distance ({dist_unit})')
        ax4.set_title('Total Distance Traveled')
        ax4.grid(True, alpha=0.3)
        
        # 5. Speed over time comparison
        ax5 = fig.add_subplot(gs[2, :])
        
        for i, roi_id in enumerate(sorted(rois.keys())):
            speeds = rois[roi_id]['smoothed_speed']
            if metadata['pixel_to_mm']:
                speeds = speeds * metadata['pixel_to_mm']
            
            valid_mask = ~np.isnan(speeds)
            if np.any(valid_mask):
                time_seconds = np.arange(len(speeds)) / metadata['fps']
                ax5.plot(time_seconds[valid_mask], speeds[valid_mask],
                        alpha=0.6, linewidth=1, label=f'Fish {roi_id}',
                        color=colors[i])
        
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel(f'Speed ({speed_unit})')
        ax5.set_title('Speed Profiles Over Time')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax5.grid(True, alpha=0.3)
        
        # Add overall title with metadata
        plt.suptitle(f'Batch Speed Comparison - {num_rois} Fish\n'
                    f'Data: {metadata["run_name"]} | Created: {metadata["created_at"][:19]}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Figure saved to:[/green] {save_path}")
        
        plt.show()
    
    def print_summary_table(self, data: Dict):
        """Print formatted summary table."""
        rois = data['rois']
        metadata = data['metadata']
        
        table = Table(title=f"Speed Analysis Summary - {metadata['run_name']}")
        
        table.add_column("Fish", style="cyan", no_wrap=True)
        table.add_column("Coverage", style="yellow")
        table.add_column("Mean Speed", style="green")
        table.add_column("Max Speed", style="red")
        table.add_column("Distance", style="magenta")
        table.add_column("95th %ile", style="blue")
        
        # Determine units
        if metadata['pixel_to_mm']:
            speed_unit = "mm/s"
            dist_unit = "mm"
            prefix = "mm"
        else:
            speed_unit = "px/s"
            dist_unit = "px"
            prefix = "px"
        
        for roi_id in sorted(rois.keys()):
            roi_data = rois[roi_id]
            stats = roi_data['statistics']
            
            table.add_row(
                str(roi_id),
                f"{roi_data['coverage']:.1f}%",
                f"{stats[f'mean_speed_{prefix}_s']:.1f} {speed_unit}",
                f"{stats[f'max_speed_{prefix}_s']:.1f} {speed_unit}",
                f"{stats[f'total_distance_{prefix}']:.0f} {dist_unit}",
                f"{stats.get(f'percentile_95_{prefix}_s', 0):.1f} {speed_unit}"
            )
        
        console.print(table)
        
        # Print aggregate statistics
        all_means = [rois[roi]['statistics'][f'mean_speed_{prefix}_s'] for roi in rois]
        all_distances = [rois[roi]['statistics'][f'total_distance_{prefix}'] for roi in rois]
        
        console.print(f"\n[bold cyan]Aggregate Statistics:[/bold cyan]")
        console.print(f"  Mean speed (all fish): {np.mean(all_means):.1f} {speed_unit}")
        console.print(f"  Std dev (across fish): {np.std(all_means):.1f} {speed_unit}")
        console.print(f"  Total distance (sum): {np.sum(all_distances):.0f} {dist_unit}")
        
        if metadata['pixel_to_mm']:
            console.print(f"  Total distance: {np.sum(all_distances)/1000:.2f} meters")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize speed data from zarr files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available speed data
  %(prog)s detections.zarr --list
  
  # Plot latest batch comparison
  %(prog)s detections.zarr --batch
  
  # Plot specific ROI from latest data
  %(prog)s detections.zarr --roi 3
  
  # Load specific run
  %(prog)s detections.zarr --batch --run speed_batch_20240315_143022
  
  # Save figures
  %(prog)s detections.zarr --batch --save batch_comparison.png
  %(prog)s detections.zarr --roi 3 --save roi3_speed.png
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--list', action='store_true',
                       help='List available speed data runs')
    parser.add_argument('--batch', action='store_true',
                       help='Plot batch comparison for all ROIs')
    parser.add_argument('--roi', type=int,
                       help='Plot specific ROI')
    parser.add_argument('--run', type=str,
                       help='Specific run name to load')
    parser.add_argument('--save', type=str,
                       help='Save figure to file')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary table only')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ZarrSpeedVisualizer(args.zarr_path)
    
    # List available runs
    if args.list:
        runs = visualizer.list_available_runs()
        console.print("\n[cyan]Available speed data runs:[/cyan]")
        
        if 'batch_speed_metrics' in runs:
            console.print("\n[bold]Batch Speed Metrics:[/bold]")
            for run in runs['batch_speed_metrics']:
                marker = " [green]← latest[/green]" if run == runs.get('batch_latest') else ""
                console.print(f"  • {run}{marker}")
        
        if 'speed_metrics' in runs:
            console.print("\n[bold]Speed Metrics:[/bold]")
            for run in runs['speed_metrics']:
                marker = " [green]← latest[/green]" if run == runs.get('speed_latest') else ""
                console.print(f"  • {run}{marker}")
        
        return
    
    # Load batch speed data
    if args.batch or args.roi is not None:
        data = visualizer.load_batch_speed_data(args.run)
        
        if data is None:
            console.print("[red]Failed to load speed data[/red]")
            return
        
        # Print summary
        if args.summary or (not args.batch and args.roi is None):
            visualizer.print_summary_table(data)
        
        # Create visualizations
        if args.batch:
            save_path = Path(args.save) if args.save else None
            visualizer.plot_batch_comparison(data, save_path)
        
        if args.roi is not None:
            save_path = Path(args.save) if args.save else None
            visualizer.plot_individual_roi_speed(args.roi, data, save_path)
    
    elif not args.list:
        # Default: show summary
        data = visualizer.load_batch_speed_data()
        if data:
            visualizer.print_summary_table(data)


if __name__ == '__main__':
    main()