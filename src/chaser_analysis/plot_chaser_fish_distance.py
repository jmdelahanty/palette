#!/usr/bin/env python3
"""
Plot Fish-Chaser Distance Analysis Results

Visualizes the fish-chaser distance data that has been calculated
and saved in the zarr file by chaser_fish_analyzer.py
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime


def plot_distance_analysis(zarr_path: str, 
                          interpolation_run: str = None,
                          save_path: str = None,
                          frame_range: tuple = None,
                          show_plot: bool = True):
    """
    Plot fish-chaser distance analysis results.
    
    Args:
        zarr_path: Path to zarr file with analysis results
        interpolation_run: Specific run to plot (default: latest)
        save_path: Optional path to save figure
        frame_range: Optional (start, end) frame indices to plot
        show_plot: Whether to display the plot
    """
    print(f"Loading analysis from: {zarr_path}")
    
    # Open zarr file
    root = zarr.open(zarr_path, mode='r')
    
    # Check if analysis exists
    if 'chaser_comparison' not in root:
        print("❌ No chaser_comparison analysis found in zarr file.")
        print("   Run chaser_fish_analyzer.py first.")
        return
    
    comp_group = root['chaser_comparison']
    
    # Determine which run to plot
    if interpolation_run is None:
        interpolation_run = comp_group.attrs.get('latest', 'original')
        print(f"Using latest run: {interpolation_run}")
    
    if interpolation_run not in comp_group:
        print(f"❌ Run '{interpolation_run}' not found in analysis.")
        print(f"   Available runs: {list(comp_group.keys())}")
        return
    
    # Load data
    results = comp_group[interpolation_run]
    
    # Load distance data
    distances_pixels = results['fish_chaser_distance_pixels'][:]
    
    # Load velocity if available
    velocities = None
    if 'relative_velocity' in results:
        velocities = results['relative_velocity'][:]
    
    # Load positions
    fish_pos_cam = results['fish_position_camera'][:]
    chaser_pos_cam = results['chaser_position_camera'][:]
    
    # Load metadata
    metadata = comp_group['metadata']
    video_width = metadata.attrs.get('video_dimensions', [4512, 4512])[0]
    video_height = metadata.attrs.get('video_dimensions', [4512, 4512])[1]
    fps = metadata.attrs.get('fps_video', 60.0)
    
    # Get summary statistics
    summary = results.attrs.get('summary', {})
    
    # Apply frame range if specified
    if frame_range:
        start, end = frame_range
        distances_pixels = distances_pixels[start:end]
        if velocities is not None:
            velocities = velocities[start:end]
        fish_pos_cam = fish_pos_cam[start:end]
        chaser_pos_cam = chaser_pos_cam[start:end]
        frame_offset = start
        print(f"Plotting frames {start} to {end}")
    else:
        frame_offset = 0
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Create gridspec for better layout control
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========== Plot 1: Distance over time ==========
    ax1 = fig.add_subplot(gs[0, 0])
    frames = np.arange(len(distances_pixels)) + frame_offset
    time_seconds = frames / fps
    
    # Plot distance
    valid_mask = ~np.isnan(distances_pixels)
    ax1.plot(time_seconds[valid_mask], distances_pixels[valid_mask], 
             'b-', alpha=0.7, linewidth=1, label='Fish-Chaser Distance')
    
    # Add mean line
    if np.any(valid_mask):
        mean_dist = np.nanmean(distances_pixels)
        ax1.axhline(y=mean_dist, color='r', linestyle='--', alpha=0.5, 
                   label=f'Mean: {mean_dist:.0f} px')
    
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Distance (pixels)', fontsize=11)
    ax1.set_title('Fish-Chaser Distance Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add statistics text
    stats_text = f"Min: {np.nanmin(distances_pixels):.1f} px\n"
    stats_text += f"Max: {np.nanmax(distances_pixels):.1f} px\n"
    stats_text += f"Coverage: {np.sum(valid_mask)/len(distances_pixels)*100:.1f}%"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========== Plot 2: Velocity over time ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    if velocities is not None and np.any(~np.isnan(velocities)):
        valid_vel = ~np.isnan(velocities)
        ax2.plot(time_seconds[valid_vel], velocities[valid_vel], 
                'g-', alpha=0.7, linewidth=0.8)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Highlight escape events (high positive velocity)
        threshold = np.nanpercentile(velocities, 95)
        escape_mask = velocities > threshold
        if np.any(escape_mask):
            ax2.scatter(time_seconds[escape_mask], velocities[escape_mask], 
                       c='r', s=20, alpha=0.6, label=f'Potential escapes (n={np.sum(escape_mask)})')
        
        ax2.set_xlabel('Time (seconds)', fontsize=11)
        ax2.set_ylabel('Velocity (pixels/second)', fontsize=11)
        ax2.set_title('Relative Velocity (negative = approaching)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add zero line label
        ax2.text(0.02, 0.5, 'approaching ←', transform=ax2.transAxes,
                fontsize=8, verticalalignment='bottom', color='blue')
        ax2.text(0.02, 0.52, '→ escaping', transform=ax2.transAxes,
                fontsize=8, verticalalignment='top', color='red')
    else:
        ax2.text(0.5, 0.5, 'No velocity data available', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Relative Velocity', fontsize=12, fontweight='bold')
    
    # ========== Plot 3: Spatial trajectories ==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    valid_fish = ~np.isnan(fish_pos_cam[:, 0])
    valid_chaser = ~np.isnan(chaser_pos_cam[:, 0])
    
    if np.any(valid_fish) and np.any(valid_chaser):
        # Plot trajectories with gradient color for time
        if np.sum(valid_fish) > 1:
            # Create color gradient for fish trajectory
            fish_colors = plt.cm.Blues(np.linspace(0.3, 1, np.sum(valid_fish)))
            for i in range(np.sum(valid_fish) - 1):
                idx = np.where(valid_fish)[0][i:i+2]
                ax3.plot(fish_pos_cam[idx, 0], fish_pos_cam[idx, 1], 
                        color=fish_colors[i], linewidth=1, alpha=0.7)
        
        if np.sum(valid_chaser) > 1:
            # Create color gradient for chaser trajectory
            chaser_colors = plt.cm.Reds(np.linspace(0.3, 1, np.sum(valid_chaser)))
            for i in range(np.sum(valid_chaser) - 1):
                idx = np.where(valid_chaser)[0][i:i+2]
                ax3.plot(chaser_pos_cam[idx, 0], chaser_pos_cam[idx, 1], 
                        color=chaser_colors[i], linewidth=1, alpha=0.7)
        
        # Mark start and end positions
        if np.any(valid_fish):
            first_fish = np.where(valid_fish)[0][0]
            last_fish = np.where(valid_fish)[0][-1]
            ax3.scatter(fish_pos_cam[first_fish, 0], fish_pos_cam[first_fish, 1], 
                       c='blue', s=100, marker='o', edgecolor='white', linewidth=2,
                       label='Fish start', zorder=5)
            ax3.scatter(fish_pos_cam[last_fish, 0], fish_pos_cam[last_fish, 1], 
                       c='darkblue', s=100, marker='s', edgecolor='white', linewidth=2,
                       label='Fish end', zorder=5)
        
        if np.any(valid_chaser):
            first_chaser = np.where(valid_chaser)[0][0]
            last_chaser = np.where(valid_chaser)[0][-1]
            ax3.scatter(chaser_pos_cam[first_chaser, 0], chaser_pos_cam[first_chaser, 1], 
                       c='red', s=100, marker='o', edgecolor='white', linewidth=2,
                       label='Chaser start', zorder=5)
            ax3.scatter(chaser_pos_cam[last_chaser, 0], chaser_pos_cam[last_chaser, 1], 
                       c='darkred', s=100, marker='s', edgecolor='white', linewidth=2,
                       label='Chaser end', zorder=5)
    
    ax3.set_xlabel('X (pixels)', fontsize=11)
    ax3.set_ylabel('Y (pixels)', fontsize=11)
    ax3.set_title('Spatial Trajectories (Camera Coordinates)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, video_width)
    ax3.set_ylim(0, video_height)
    ax3.invert_yaxis()  # Invert y-axis for image coordinates
    ax3.set_aspect('equal')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Add arena center marker
    ax3.plot(video_width/2, video_height/2, 'k+', markersize=15, 
            markeredgewidth=2, label='Arena center')
    
    # ========== Plot 4: Distance histogram ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    valid_distances = distances_pixels[~np.isnan(distances_pixels)]
    if len(valid_distances) > 0:
        n, bins, patches = ax4.hist(valid_distances, bins=50, alpha=0.7, 
                                    color='purple', edgecolor='black', linewidth=0.5)
        
        # Color bars by distance
        cm = plt.cm.viridis
        norm = plt.Normalize(vmin=valid_distances.min(), vmax=valid_distances.max())
        for i, patch in enumerate(patches):
            patch.set_facecolor(cm(norm(bins[i])))
        
        # Add statistics lines
        mean_val = np.mean(valid_distances)
        median_val = np.median(valid_distances)
        ax4.axvline(mean_val, color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.0f} px')
        ax4.axvline(median_val, color='g', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.0f} px')
        
        ax4.set_xlabel('Distance (pixels)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Distance Distribution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add percentile information
        p25 = np.percentile(valid_distances, 25)
        p75 = np.percentile(valid_distances, 75)
        stats_text = f"25th percentile: {p25:.0f} px\n"
        stats_text += f"75th percentile: {p75:.0f} px\n"
        stats_text += f"Total frames: {len(valid_distances)}"
        ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add overall title with metadata
    title = f'Fish-Chaser Distance Analysis: {interpolation_run}'
    if 'created_at' in results.attrs:
        title += f"\n(Analyzed: {results.attrs['created_at'][:19]})"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot fish-chaser distance analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s detections.zarr
  %(prog)s detections.zarr --save plot.png
  %(prog)s detections.zarr --frames 1000 5000
  %(prog)s detections.zarr --run interp_linear_20240120
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file with analysis results')
    parser.add_argument('--run', dest='interpolation_run', 
                       help='Specific analysis run to plot (default: latest)')
    parser.add_argument('--save', dest='save_path', 
                       help='Path to save figure')
    parser.add_argument('--frames', nargs=2, type=int, metavar=('START', 'END'),
                       help='Frame range to plot')
    parser.add_argument('--no-show', action='store_true',
                       help="Don't display the plot")
    
    args = parser.parse_args()
    
    frame_range = None
    if args.frames:
        frame_range = tuple(args.frames)
    
    plot_distance_analysis(
        zarr_path=args.zarr_path,
        interpolation_run=args.interpolation_run,
        save_path=args.save_path,
        frame_range=frame_range,
        show_plot=not args.no_show
    )
    
    return 0


if __name__ == '__main__':
    exit(main())