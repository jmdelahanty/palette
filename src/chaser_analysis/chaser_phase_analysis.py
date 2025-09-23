#!/usr/bin/env python3
"""
Phase-Based Chaser-Fish Analysis

Analyzes fish behavior separately for pre-training, training, and post-training phases.
Creates heatmaps and calculates distance metrics for each experimental phase.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import seaborn as sns
from scipy.ndimage import gaussian_filter
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import sys
import os

# Add parent directory to path to import the unified analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chaser_fish_distance_analyzer import UnifiedDistanceAnalyzer


def identify_experimental_phases(analyzer) -> Dict:
    """
    Identify the start and end frames for each experimental phase.
    
    Returns:
        Dictionary with phase names and their frame ranges
    """
    phases = {
        'pre_training': {'start': 0, 'end': None, 'duration_s': 300},
        'training': {'start': None, 'end': None, 'duration_s': 150},
        'post_training': {'start': None, 'end': None, 'duration_s': 300}
    }
    
    # Find phase transitions from events
    for event in analyzer.chase_events:
        # These aren't the phase markers, they're chase events
        pass
    
    # Use H5 events to find phase transitions if available
    # For now, use approximate timing based on protocol
    fps = analyzer.aligned_data.metadata['fps']
    
    # Pre-training: 0-300 seconds (frames 0-18000 at 60fps)
    phases['pre_training']['end'] = int(300 * fps)
    
    # Training: 300-450 seconds (frames 18000-27000 at 60fps)
    phases['training']['start'] = phases['pre_training']['end']
    phases['training']['end'] = int(450 * fps)
    
    # Post-training: 450-750 seconds (frames 27000-45000 at 60fps)
    phases['post_training']['start'] = phases['training']['end']
    phases['post_training']['end'] = int(750 * fps)
    
    # Adjust to actual data length
    total_frames = analyzer.aligned_data.metadata['total_frames']
    phases['post_training']['end'] = min(phases['post_training']['end'], total_frames)
    
    return phases


def calculate_phase_metrics(analyzer, phase_frames: Tuple[int, int]) -> Dict:
    """
    Calculate metrics for a specific phase.
    """
    start_frame, end_frame = phase_frames
    data = analyzer.aligned_data
    
    # Extract phase data
    phase_distances = data.distances[start_frame:end_frame]
    phase_fish_x = data.fish_x[start_frame:end_frame]
    phase_fish_y = data.fish_y[start_frame:end_frame]
    phase_chaser_x = data.chaser_x[start_frame:end_frame]
    phase_chaser_y = data.chaser_y[start_frame:end_frame]
    
    valid_mask = ~np.isnan(phase_distances)
    valid_distances = phase_distances[valid_mask]
    
    metrics = {}
    
    if len(valid_distances) > 0:
        metrics['mean_distance'] = np.mean(valid_distances)
        metrics['median_distance'] = np.median(valid_distances)
        metrics['min_distance'] = np.min(valid_distances)
        metrics['max_distance'] = np.max(valid_distances)
        metrics['std_distance'] = np.std(valid_distances)
        metrics['q25_distance'] = np.percentile(valid_distances, 25)
        metrics['q75_distance'] = np.percentile(valid_distances, 75)
        
        # Calculate time spent in different distance zones
        close_threshold = 500  # pixels
        medium_threshold = 1500  # pixels
        
        metrics['time_close_pct'] = np.sum(valid_distances < close_threshold) / len(valid_distances) * 100
        metrics['time_medium_pct'] = np.sum((valid_distances >= close_threshold) & 
                                           (valid_distances < medium_threshold)) / len(valid_distances) * 100
        metrics['time_far_pct'] = np.sum(valid_distances >= medium_threshold) / len(valid_distances) * 100
        
        # Calculate approach/escape dynamics
        if len(valid_distances) > 1:
            velocity = np.diff(phase_distances) * analyzer.fps
            valid_vel = velocity[~np.isnan(velocity)]
            if len(valid_vel) > 0:
                metrics['mean_relative_velocity'] = np.mean(valid_vel)
                metrics['approach_events'] = np.sum(valid_vel < -50)
                metrics['escape_events'] = np.sum(valid_vel > 50)
    else:
        # No valid data
        for key in ['mean_distance', 'median_distance', 'min_distance', 'max_distance',
                   'std_distance', 'q25_distance', 'q75_distance', 'time_close_pct',
                   'time_medium_pct', 'time_far_pct', 'mean_relative_velocity',
                   'approach_events', 'escape_events']:
            metrics[key] = np.nan
    
    # Coverage metrics
    metrics['coverage_pct'] = np.sum(valid_mask) / len(phase_distances) * 100
    metrics['valid_frames'] = np.sum(valid_mask)
    metrics['total_frames'] = len(phase_distances)
    
    return metrics


def create_phase_heatmap(fish_x, fish_y, chaser_x, chaser_y, 
                         arena_size=(4512, 4512), bins=50):
    """
    Create occupancy heatmaps for fish and chaser positions.
    """
    # Filter out NaN values
    valid_fish = ~(np.isnan(fish_x) | np.isnan(fish_y))
    valid_chaser = ~(np.isnan(chaser_x) | np.isnan(chaser_y))
    
    # Create 2D histograms
    fish_heatmap = None
    chaser_heatmap = None
    
    if np.any(valid_fish):
        fish_hist, xedges, yedges = np.histogram2d(
            fish_x[valid_fish], fish_y[valid_fish],
            bins=bins, range=[[0, arena_size[0]], [0, arena_size[1]]]
        )
        fish_heatmap = gaussian_filter(fish_hist.T, sigma=1.5)
    
    if np.any(valid_chaser):
        chaser_hist, _, _ = np.histogram2d(
            chaser_x[valid_chaser], chaser_y[valid_chaser],
            bins=bins, range=[[0, arena_size[0]], [0, arena_size[1]]]
        )
        chaser_heatmap = gaussian_filter(chaser_hist.T, sigma=1.5)
    
    return fish_heatmap, chaser_heatmap, (xedges, yedges)


def plot_phase_analysis(analyzer, phases: Dict, save_path: Optional[str] = None):
    """
    Create comprehensive phase-based analysis plot.
    """
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25,
                          height_ratios=[1, 1, 1, 0.3])
    
    phase_names = ['pre_training', 'training', 'post_training']
    phase_titles = ['Pre-Training (0-5 min)', 'Training (5-7.5 min)', 'Post-Training (7.5-12.5 min)']
    phase_colors = ['blue', 'red', 'green']
    
    all_metrics = {}
    
    for col, (phase_name, phase_title, color) in enumerate(zip(phase_names, phase_titles, phase_colors)):
        phase_info = phases[phase_name]
        start_frame = phase_info['start']
        end_frame = phase_info['end']
        
        # Calculate metrics for this phase
        metrics = calculate_phase_metrics(analyzer, (start_frame, end_frame))
        all_metrics[phase_name] = metrics
        
        # Extract phase data
        data = analyzer.aligned_data
        phase_fish_x = data.fish_x[start_frame:end_frame]
        phase_fish_y = data.fish_y[start_frame:end_frame]
        phase_chaser_x = data.chaser_x[start_frame:end_frame]
        phase_chaser_y = data.chaser_y[start_frame:end_frame]
        phase_distances = data.distances[start_frame:end_frame]
        
        # Row 1: Fish occupancy heatmap
        ax1 = fig.add_subplot(gs[0, col])
        fish_heat, chaser_heat, edges = create_phase_heatmap(
            phase_fish_x, phase_fish_y, phase_chaser_x, phase_chaser_y
        )
        
        if fish_heat is not None:
            im1 = ax1.imshow(fish_heat, origin='lower', aspect='equal', cmap='hot',
                            extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]])
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        ax1.set_title(f'{phase_title}\nFish Occupancy', fontweight='bold')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        
        # Add arena circle if applicable
        circle = Circle((2256, 2256), 2000, fill=False, edgecolor='cyan', 
                       linewidth=2, alpha=0.5)
        ax1.add_patch(circle)
        
        # Row 2: Chaser occupancy heatmap
        ax2 = fig.add_subplot(gs[1, col])
        if chaser_heat is not None:
            im2 = ax2.imshow(chaser_heat, origin='lower', aspect='equal', cmap='cool',
                            extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]])
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        ax2.set_title('Chaser Occupancy', fontweight='bold')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        
        # Add arena circle
        circle = Circle((2256, 2256), 2000, fill=False, edgecolor='cyan',
                       linewidth=2, alpha=0.5)
        ax2.add_patch(circle)
        
        # Row 3: Distance distribution
        ax3 = fig.add_subplot(gs[2, col])
        valid_distances = phase_distances[~np.isnan(phase_distances)]
        
        if len(valid_distances) > 0:
            # Create histogram
            n, bins, patches = ax3.hist(valid_distances, bins=50, alpha=0.7, 
                                       color=color, edgecolor='black', density=True)
            
            # Add KDE curve
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(valid_distances)
            x_range = np.linspace(valid_distances.min(), valid_distances.max(), 200)
            ax3.plot(x_range, kde(x_range), color='black', linewidth=2)
            
            # Add statistics lines
            ax3.axvline(metrics['mean_distance'], color='red', linestyle='--', 
                       linewidth=2, label=f"Mean: {metrics['mean_distance']:.0f}")
            ax3.axvline(metrics['median_distance'], color='green', linestyle='--',
                       linewidth=2, label=f"Median: {metrics['median_distance']:.0f}")
            
            ax3.set_xlabel('Distance (pixels)')
            ax3.set_ylabel('Probability Density')
            ax3.set_title('Distance Distribution')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
    
    # Row 4: Metrics summary table
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis('off')
    
    # Create metrics table
    table_data = []
    headers = ['Metric', 'Pre-Training', 'Training', 'Post-Training']
    
    # Distance metrics
    table_data.append(['Mean Distance (px)', 
                      f"{all_metrics['pre_training']['mean_distance']:.1f}",
                      f"{all_metrics['training']['mean_distance']:.1f}",
                      f"{all_metrics['post_training']['mean_distance']:.1f}"])
    
    table_data.append(['Median Distance (px)',
                      f"{all_metrics['pre_training']['median_distance']:.1f}",
                      f"{all_metrics['training']['median_distance']:.1f}",
                      f"{all_metrics['post_training']['median_distance']:.1f}"])
    
    table_data.append(['Min Distance (px)',
                      f"{all_metrics['pre_training']['min_distance']:.1f}",
                      f"{all_metrics['training']['min_distance']:.1f}",
                      f"{all_metrics['post_training']['min_distance']:.1f}"])
    
    # Time in zones
    table_data.append(['Time Close (<500px)',
                      f"{all_metrics['pre_training']['time_close_pct']:.1f}%",
                      f"{all_metrics['training']['time_close_pct']:.1f}%",
                      f"{all_metrics['post_training']['time_close_pct']:.1f}%"])
    
    table_data.append(['Time Medium (500-1500px)',
                      f"{all_metrics['pre_training']['time_medium_pct']:.1f}%",
                      f"{all_metrics['training']['time_medium_pct']:.1f}%",
                      f"{all_metrics['post_training']['time_medium_pct']:.1f}%"])
    
    table_data.append(['Time Far (>1500px)',
                      f"{all_metrics['pre_training']['time_far_pct']:.1f}%",
                      f"{all_metrics['training']['time_far_pct']:.1f}%",
                      f"{all_metrics['post_training']['time_far_pct']:.1f}%"])
    
    # Dynamic metrics
    table_data.append(['Approach Events',
                      f"{all_metrics['pre_training'].get('approach_events', 0):.0f}",
                      f"{all_metrics['training'].get('approach_events', 0):.0f}",
                      f"{all_metrics['post_training'].get('approach_events', 0):.0f}"])
    
    table_data.append(['Escape Events',
                      f"{all_metrics['pre_training'].get('escape_events', 0):.0f}",
                      f"{all_metrics['training'].get('escape_events', 0):.0f}",
                      f"{all_metrics['post_training'].get('escape_events', 0):.0f}"])
    
    # Coverage
    table_data.append(['Data Coverage',
                      f"{all_metrics['pre_training']['coverage_pct']:.1f}%",
                      f"{all_metrics['training']['coverage_pct']:.1f}%",
                      f"{all_metrics['post_training']['coverage_pct']:.1f}%"])
    
    # Create table
    table = ax_table.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center',
                          colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code the phase columns
    colors = ['#e8f4ff', '#ffe8e8', '#e8ffe8']
    for row in range(1, len(table_data) + 1):
        for col in range(1, 4):
            table[(row, col)].set_facecolor(colors[col-1])
    
    # Main title
    fig.suptitle('Phase-Based Chaser-Fish Analysis: Pre-Training vs Training vs Post-Training',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Phase analysis plot saved to: {save_path}")
    
    plt.show()
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Phase-based analysis of chaser-fish interactions'
    )
    parser.add_argument('zarr_path', help='Path to multi-fish tracker zarr')
    parser.add_argument('h5_path', help='Path to experiment H5 file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the analysis plot')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE-BASED CHASER-FISH ANALYSIS")
    print("=" * 60)
    
    # Load data using the unified analyzer
    analyzer = UnifiedDistanceAnalyzer(
        args.zarr_path,
        args.h5_path,
        verbose=True
    )
    
    # Identify experimental phases
    phases = identify_experimental_phases(analyzer)
    
    print("\nExperimental Phases:")
    for phase_name, phase_info in phases.items():
        duration_frames = phase_info['end'] - phase_info['start'] if phase_info['start'] is not None else phase_info['end']
        duration_s = duration_frames / analyzer.fps
        print(f"  {phase_name}: frames {phase_info['start']}-{phase_info['end']} ({duration_s:.1f} seconds)")
    
    # Create phase analysis plot
    metrics = plot_phase_analysis(analyzer, phases, save_path=args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PHASE COMPARISON SUMMARY")
    print("=" * 60)
    
    # Compare key metrics across phases
    print("\nMean Distance (pixels):")
    for phase in ['pre_training', 'training', 'post_training']:
        print(f"  {phase}: {metrics[phase]['mean_distance']:.1f}")
    
    print("\nTime Spent Close (<500 pixels):")
    for phase in ['pre_training', 'training', 'post_training']:
        print(f"  {phase}: {metrics[phase]['time_close_pct']:.1f}%")
    
    return 0


if __name__ == '__main__':
    exit(main())