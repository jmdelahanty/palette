#!/usr/bin/env python3
"""
Trajectory Visualizer for Multi-Fish Tracker Zarr

Creates visualizations similar to analyze_chaser_target.py but for multi-fish zarr data.
Displays trajectory, frame coverage, and movement patterns for tracked fish.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import pandas as pd


def load_tracking_data(zarr_path: str, source: str = 'latest', 
                       fish_id: Optional[int] = None) -> Dict:
    """
    Load tracking data from multi-fish zarr.
    
    Args:
        zarr_path: Path to multi-fish zarr file
        source: Data source - 'latest', 'preprocessing', 'filtered', or specific run
        fish_id: Optional fish ID to visualize
    
    Returns:
        Dictionary containing tracking data
    """
    root = zarr.open(str(zarr_path), mode='r')
    
    # Determine which data to load based on priority
    data_group = None
    source_name = None
    
    if source == 'latest':
        # Priority: preprocessing > filtered > detect
        if 'preprocessing' in root and root['preprocessing'].attrs.get('latest'):
            source_name = 'preprocessing/' + root['preprocessing'].attrs['latest']
            data_group = root[source_name]
        elif 'filtered_runs' in root and root['filtered_runs'].attrs.get('latest'):
            source_name = 'filtered_runs/' + root['filtered_runs'].attrs['latest']
            data_group = root[source_name]
        elif 'detect_runs' in root and root['detect_runs'].attrs.get('latest'):
            source_name = 'detect_runs/' + root['detect_runs'].attrs['latest']
            data_group = root[source_name]
    else:
        # Try to load specific source
        if source in root:
            if root[source].attrs.get('latest'):
                source_name = source + '/' + root[source].attrs['latest']
                data_group = root[source_name]
        elif '/' in source:
            source_name = source
            data_group = root[source]
    
    if data_group is None:
        raise ValueError(f"Could not find data source: {source}")
    
    print(f"Loading data from: {source_name}")
    
    # Load detection data
    n_detections = data_group['n_detections'][:]
    bbox_coords = data_group['bbox_norm_coords'][:]
    
    # Check for interpolation mask if available
    interp_mask = None
    if 'interpolation_mask' in data_group:
        interp_mask = data_group['interpolation_mask'][:]
    
    # Get metadata
    if 'raw_video' in root:
        width = root['raw_video/images_ds'].shape[2]
        height = root['raw_video/images_ds'].shape[1]
        fps = root.attrs.get('fps', 60.0)
    else:
        width = 640
        height = 640
        fps = 60.0
    
    total_frames = len(n_detections)
    
    # Handle fish ID filtering if specified
    if fish_id is not None and 'id_assignments_runs' in root:
        latest_assign = root['id_assignments_runs'].attrs['latest']
        detection_ids = root[f'id_assignments_runs/{latest_assign}/detection_ids'][:]
        # Filter for specific fish
        fish_mask = detection_ids == fish_id
        print(f"Filtering for fish ID {fish_id}")
    else:
        fish_mask = None
    
    # Process frame by frame to get positions
    positions = []
    cumulative_detections = np.cumsum(np.insert(n_detections, 0, 0))
    
    for frame_idx in range(total_frames):
        start_idx = cumulative_detections[frame_idx]
        end_idx = cumulative_detections[frame_idx + 1]
        
        if end_idx > start_idx:
            frame_bboxes = bbox_coords[start_idx:end_idx]
            
            if fish_mask is not None:
                frame_mask = fish_mask[start_idx:end_idx]
                frame_bboxes = frame_bboxes[frame_mask]
            
            if len(frame_bboxes) > 0:
                # Take first detection (or could take all for multi-fish)
                bbox = frame_bboxes[0]
                center_x = bbox[0] * width
                center_y = bbox[1] * height
                
                is_interpolated = False
                if interp_mask is not None and start_idx < len(interp_mask):
                    is_interpolated = interp_mask[start_idx]
                
                positions.append({
                    'frame': frame_idx,
                    'x': center_x,
                    'y': center_y,
                    'interpolated': is_interpolated
                })
    
    return {
        'positions': positions,
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'source': source_name,
        'fish_id': fish_id
    }


def plot_trajectory_analysis(data: Dict, output_path: Optional[str] = None, 
                            title_prefix: str = "Fish"):
    """
    Create a 2x2 subplot visualization similar to analyze_chaser_target.py.
    
    Args:
        data: Dictionary from load_tracking_data
        output_path: Optional path to save the figure
        title_prefix: Prefix for plot titles (e.g., "Fish", "Fish 0", "Chaser")
    """
    if not data['positions']:
        print("No positions to plot")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data['positions'])
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Trajectory (colored by frame number)
    ax = axes[0, 0]
    scatter = ax.scatter(df['x'], df['y'], 
                        c=df['frame'], cmap='viridis', s=2, alpha=0.7)
    
    # Mark interpolated points if present
    if 'interpolated' in df.columns and df['interpolated'].any():
        interp_df = df[df['interpolated']]
        ax.scatter(interp_df['x'], interp_df['y'], 
                  color='red', s=10, alpha=0.5, marker='x', 
                  label=f'Interpolated ({len(interp_df)} points)')
        ax.legend()
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(f'{title_prefix} Trajectory (colored by frame number)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Frame')
    
    # Add start and end markers
    ax.plot(df.iloc[0]['x'], df.iloc[0]['y'], 'go', markersize=8, 
            markeredgecolor='white', markeredgewidth=1, label='Start')
    ax.plot(df.iloc[-1]['x'], df.iloc[-1]['y'], 'ro', markersize=8,
            markeredgecolor='white', markeredgewidth=1, label='End')
    
    # Plot 2: X-Y position over time
    ax = axes[0, 1]
    time_seconds = df['frame'] / data['fps']
    ax.plot(time_seconds, df['x'], 'b-', alpha=0.7, linewidth=0.5, label='X')
    ax.plot(time_seconds, df['y'], 'r-', alpha=0.7, linewidth=0.5, label='Y')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Position (pixels)')
    ax.set_title('Position Components Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Frame coverage
    ax = axes[1, 0]
    frames = df['frame'].values
    
    # Create coverage visualization
    all_frames = np.arange(data['total_frames'])
    coverage = np.zeros(data['total_frames'])
    coverage[frames] = 1
    
    # Show as image (like in analyze_chaser_target)
    ax.imshow([coverage], aspect='auto', cmap='RdYlGn', 
          extent=[0, data['total_frames'], 0, 1],
          interpolation='nearest', vmin=0, vmax=1)
    
    # Highlight gaps
    if len(frames) > 1:
        frame_diffs = np.diff(frames)
        gap_indices = np.where(frame_diffs > 1)[0]
        for idx in gap_indices:
            gap_start = frames[idx]
            gap_end = frames[idx + 1]
            if gap_end - gap_start > 10:  # Only show significant gaps
                ax.axvspan(gap_start, gap_end, alpha=0.3, color='blue', 
                          ymin=0.2, ymax=0.8)
    
    ax.set_xlabel('Frame Number')
    ax.set_yticks([])
    ax.set_title(f'Frame Coverage (Green = detected, Red = missing, Blue = large gaps)')
    ax.set_xlim(0, data['total_frames'])
    
    # Add coverage statistics
    coverage_pct = len(frames) / data['total_frames'] * 100
    ax.text(0.02, 0.5, f'Coverage: {coverage_pct:.1f}%', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: Frame-to-frame movement distance
    ax = axes[1, 1]
    
    if len(df) > 1:
        # Calculate distances
        df['dx'] = df['x'].diff()
        df['dy'] = df['y'].diff()
        df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
        
        # Account for frame gaps
        df['frame_gap'] = df['frame'].diff()
        df['distance_per_frame'] = df['distance'] / df['frame_gap']
        
        # Plot movement distance
        time_seconds = df['frame'] / data['fps']
        ax.plot(time_seconds[1:], df['distance'][1:], 
                'b-', alpha=0.7, linewidth=0.5, label='Frame-to-frame distance')
        
        # Mark large jumps
        jump_threshold = 50  # pixels
        jumps = df[df['distance'] > jump_threshold]
        if not jumps.empty:
            jump_times = jumps['frame'] / data['fps']
            ax.scatter(jump_times, jumps['distance'], 
                      color='red', s=50, zorder=5, 
                      label=f'Jumps (>{jump_threshold}px)')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Movement Distance (pixels)')
        ax.set_title('Frame-to-Frame Movement')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_dist = df['distance'].mean()
        ax.axhline(y=mean_dist, color='g', linestyle='--', alpha=0.5)
        ax.text(0.02, 0.95, f'Mean: {mean_dist:.1f} px', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add overall title with metadata
    fig.suptitle(f'{title_prefix} Tracking Analysis - {data["source"]}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def plot_multi_fish_comparison(zarr_path: str, max_fish: int = 4, 
                              output_path: Optional[str] = None):
    """
    Create a grid visualization comparing multiple fish trajectories.
    """
    root = zarr.open(str(zarr_path), mode='r')
    
    # Check if we have fish IDs
    if 'id_assignments_runs' not in root:
        print("No fish ID assignments found. Showing single detection.")
        data = load_tracking_data(zarr_path)
        plot_trajectory_analysis(data, output_path)
        return
    
    # Get unique fish IDs
    latest_assign = root['id_assignments_runs'].attrs['latest']
    detection_ids = root[f'id_assignments_runs/{latest_assign}/detection_ids'][:]
    unique_ids = np.unique(detection_ids[detection_ids >= 0])[:max_fish]
    
    if len(unique_ids) == 0:
        print("No valid fish IDs found")
        return
    
    # Create figure with subplots for each fish
    n_fish = len(unique_ids)
    fig = plt.figure(figsize=(6 * n_fish, 10))
    gs = gridspec.GridSpec(2, n_fish, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, fish_id in enumerate(unique_ids):
        # Load data for this fish
        data = load_tracking_data(zarr_path, fish_id=fish_id)
        
        if not data['positions']:
            continue
        
        df = pd.DataFrame(data['positions'])
        
        # Top row: Trajectories
        ax = fig.add_subplot(gs[0, idx])
        scatter = ax.scatter(df['x'], df['y'], 
                           c=df['frame'], cmap='viridis', s=2, alpha=0.7)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(f'Fish {fish_id} Trajectory')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Bottom row: Movement over time
        ax = fig.add_subplot(gs[1, idx])
        if len(df) > 1:
            df['distance'] = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2)
            time_seconds = df['frame'] / data['fps']
            ax.plot(time_seconds[1:], df['distance'][1:], 
                   'b-', alpha=0.7, linewidth=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Movement (px)')
            ax.set_title(f'Fish {fish_id} Movement')
            ax.grid(True, alpha=0.3)
    
    fig.suptitle('Multi-Fish Trajectory Comparison', fontsize=14, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Multi-fish plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize trajectories from multi-fish tracker zarr'
    )
    parser.add_argument('zarr_path', help='Path to multi-fish zarr file')
    parser.add_argument('--source', type=str, default='latest',
                       help='Data source: latest, preprocessing, filtered, or specific run')
    parser.add_argument('--fish-id', type=int, default=None,
                       help='Specific fish ID to visualize')
    parser.add_argument('--multi-fish', action='store_true',
                       help='Create comparison plot for multiple fish')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the plot')
    parser.add_argument('--title', type=str, default='Fish',
                       help='Title prefix for plots')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MULTI-FISH TRAJECTORY VISUALIZER")
    print("=" * 60)
    print(f"Zarr file: {args.zarr_path}")
    print(f"Data source: {args.source}")
    
    try:
        if args.multi_fish:
            # Show multi-fish comparison
            plot_multi_fish_comparison(args.zarr_path, output_path=args.output)
        else:
            # Load and plot single fish or combined data
            data = load_tracking_data(
                args.zarr_path,
                source=args.source,
                fish_id=args.fish_id
            )
            
            # Determine title
            title = args.title
            if args.fish_id is not None:
                title = f"Fish {args.fish_id}"
            
            plot_trajectory_analysis(data, output_path=args.output, 
                                   title_prefix=title)
            
            # Print summary statistics
            if data['positions']:
                df = pd.DataFrame(data['positions'])
                print(f"\nSummary Statistics:")
                print(f"  Total frames: {data['total_frames']}")
                print(f"  Frames with detections: {len(df)} ({len(df)/data['total_frames']*100:.1f}%)")
                
                if 'interpolated' in df.columns:
                    n_interp = df['interpolated'].sum()
                    if n_interp > 0:
                        print(f"  Interpolated frames: {n_interp} ({n_interp/len(df)*100:.1f}%)")
                
                if len(df) > 1:
                    df['distance'] = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2)
                    print(f"  Mean movement: {df['distance'].mean():.1f} pixels/frame")
                    print(f"  Max movement: {df['distance'].max():.1f} pixels")
                    
                    # Calculate total distance
                    total_distance = df['distance'].sum()
                    print(f"  Total distance: {total_distance:.1f} pixels")
                    
                    # If we have fps, calculate speed
                    duration_seconds = data['total_frames'] / data['fps']
                    print(f"  Duration: {duration_seconds:.1f} seconds")
                    print(f"  Mean speed: {total_distance/duration_seconds:.1f} pixels/second")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())