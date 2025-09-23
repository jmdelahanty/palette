#!/usr/bin/env python3
"""
Filtering Comparison Visualizer for Multi-Fish Tracker Zarr

Creates before/after comparison plots showing the effects of jump removal
and filtering on trajectory data. Compatible with multi-fish zarr structure.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from pathlib import Path
import argparse
from typing import Dict, Tuple, List, Optional


def load_before_after_data(zarr_path: str, threshold: float = 200.0, 
                          fish_id: Optional[int] = None) -> Tuple[Dict, Dict]:
    """
    Load original and filtered data for comparison.
    
    Returns:
        Tuple of (before_data, after_data) dictionaries
    """
    root = zarr.open(str(zarr_path), mode='r')
    
    # Load BEFORE data (original detections)
    if 'detect_runs' not in root:
        raise ValueError("No detect_runs found")
    
    latest_detect = root['detect_runs'].attrs['latest']
    detect_group = root[f'detect_runs/{latest_detect}']
    
    # Get dimensions
    if 'raw_video' in root:
        width = root['raw_video/images_ds'].shape[2]
        height = root['raw_video/images_ds'].shape[1]
    else:
        width, height = 640, 640
    
    # Process original detections
    n_detections_orig = detect_group['n_detections'][:]
    bbox_coords_orig = detect_group['bbox_norm_coords'][:]
    
    before_data = process_detections(
        n_detections_orig, bbox_coords_orig, width, height, 
        threshold, fish_id, "original"
    )
    
    # Load AFTER data (filtered or preprocessed)
    after_data = None
    
    # Check for filtered data first
    if 'filtered_runs' in root and root['filtered_runs'].attrs.get('latest'):
        latest_filtered = root['filtered_runs'].attrs['latest']
        filtered_group = root[f'filtered_runs/{latest_filtered}']
        n_detections_filt = filtered_group['n_detections'][:]
        bbox_coords_filt = filtered_group['bbox_norm_coords'][:]
        
        after_data = process_detections(
            n_detections_filt, bbox_coords_filt, width, height,
            threshold, fish_id, "filtered"
        )
    
    # If no filtered data, use original as both
    if after_data is None:
        after_data = before_data.copy()
        after_data['source'] = "original (no filtering applied)"
    
    return before_data, after_data


def process_detections(n_detections: np.ndarray, bbox_coords: np.ndarray,
                       width: float, height: float, threshold: float,
                       fish_id: Optional[int], source: str) -> Dict:
    """
    Process detection data and calculate movement statistics.
    """
    # Extract centroids for frames with detections
    centroids = []
    frame_indices = []
    cumulative = np.cumsum(np.insert(n_detections, 0, 0))
    
    for frame_idx in range(len(n_detections)):
        start_idx = cumulative[frame_idx]
        end_idx = cumulative[frame_idx + 1]
        
        if end_idx > start_idx:
            bbox = bbox_coords[start_idx]  # Take first detection
            center_x = bbox[0] * width
            center_y = bbox[1] * height
            centroids.append([center_x, center_y])
            frame_indices.append(frame_idx)
    
    centroids = np.array(centroids) if centroids else np.empty((0, 2))
    frame_indices = np.array(frame_indices)
    
    # Calculate distances and identify issues
    distances = []
    consecutive_jumps = []
    gap_jumps = []
    islands = []
    short_segments = []
    
    if len(centroids) > 1:
        for i in range(1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[i-1])
            distances.append(dist)
            
            frame_gap = frame_indices[i] - frame_indices[i-1]
            
            # Classify issues
            if dist > threshold:
                if frame_gap == 1:
                    consecutive_jumps.append({
                        'from_idx': i-1,
                        'to_idx': i,
                        'distance': dist,
                        'from_frame': frame_indices[i-1],
                        'to_frame': frame_indices[i]
                    })
                else:
                    gap_jumps.append({
                        'from_idx': i-1,
                        'to_idx': i,
                        'distance': dist,
                        'gap': frame_gap,
                        'from_frame': frame_indices[i-1],
                        'to_frame': frame_indices[i]
                    })
            
            # Check for islands (isolated detections)
            if i > 0 and i < len(centroids) - 1:
                if distances[i-1] > threshold and i < len(distances) and distances[i] > threshold:
                    islands.append(frame_indices[i])
    
    # Find short segments
    segments = []
    if len(frame_indices) > 0:
        current_segment = [frame_indices[0]]
        for i in range(1, len(frame_indices)):
            if frame_indices[i] == frame_indices[i-1] + 1:
                current_segment.append(frame_indices[i])
            else:
                if len(current_segment) > 0:
                    segments.append(current_segment)
                current_segment = [frame_indices[i]]
        if current_segment:
            segments.append(current_segment)
    
    min_segment_length = 10
    for segment in segments:
        if len(segment) < min_segment_length:
            short_segments.extend(segment)
    
    distances = np.array(distances) if distances else np.empty(0)
    
    return {
        'centroids': centroids,
        'frame_indices': frame_indices,
        'distances': distances,
        'threshold': threshold,
        'consecutive_jumps': consecutive_jumps,
        'gap_jumps': gap_jumps,
        'islands': islands,
        'short_segments': short_segments,
        'total_frames': len(n_detections),
        'coverage': len(frame_indices) / len(n_detections) * 100,
        'source': source
    }


def create_comparison_plot(before_data: Dict, after_data: Dict, 
                          save_path: Optional[str] = None,
                          interactive: bool = True):
    """
    Create the before/after filtering comparison visualization.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Before vs After Filtering Comparison', fontsize=16, fontweight='bold')
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8], height_ratios=[1, 1],
                         hspace=0.3, wspace=0.3)
    
    # Plot 1: Movement distances BEFORE
    ax1 = fig.add_subplot(gs[0, 0])
    if len(before_data['distances']) > 0:
        ax1.plot(before_data['frame_indices'][1:], before_data['distances'], 
                'b-', alpha=0.6, linewidth=0.5)
        ax1.axhline(y=before_data['threshold'], color='r', linestyle='--',
                   label=f"Threshold ({before_data['threshold']:.0f} px)")
        
        # Highlight outliers
        outliers = before_data['distances'] > before_data['threshold']
        if np.any(outliers):
            outlier_indices = np.where(outliers)[0]
            ax1.scatter(before_data['frame_indices'][outlier_indices + 1],
                       before_data['distances'][outliers],
                       color='red', s=30, zorder=5)
    
    ax1.set_xlabel('Consecutive Transition Index')
    ax1.set_ylabel('Distance (pixels)')
    ax1.set_title('Movement Distances - BEFORE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Movement distances AFTER
    ax2 = fig.add_subplot(gs[1, 0])
    if len(after_data['distances']) > 0:
        ax2.plot(after_data['frame_indices'][1:], after_data['distances'],
                'g-', alpha=0.6, linewidth=0.5)
        ax2.axhline(y=after_data['threshold'], color='r', linestyle='--',
                   label=f"Threshold ({after_data['threshold']:.0f} px)")
        ax2.set_ylim(0, max(200, np.percentile(after_data['distances'], 99) * 1.1))
    
    ax2.set_xlabel('Consecutive Transition Index')
    ax2.set_ylabel('Distance (pixels)')
    ax2.set_title('Movement Distances - AFTER')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Trajectory BEFORE
    ax3 = fig.add_subplot(gs[0, 1])
    if len(before_data['centroids']) > 0:
        # Plot trajectory colored by time
        scatter = ax3.scatter(before_data['centroids'][:, 0],
                            before_data['centroids'][:, 1],
                            c=before_data['frame_indices'],
                            cmap='viridis', s=2, alpha=0.6)
        
        # Mark jumps
        all_jumps = before_data['consecutive_jumps'] + before_data['gap_jumps']
        for jump in all_jumps:
            ax3.plot(before_data['centroids'][jump['to_idx'], 0],
                    before_data['centroids'][jump['to_idx'], 1],
                    'ro', markersize=6, markeredgecolor='white', 
                    markeredgewidth=0.5, zorder=5)
        
        # Mark islands
        for island_frame in before_data['islands']:
            idx = np.where(before_data['frame_indices'] == island_frame)[0]
            if len(idx) > 0:
                ax3.plot(before_data['centroids'][idx[0], 0],
                        before_data['centroids'][idx[0], 1],
                        'mo', markersize=5, markeredgecolor='white',
                        markeredgewidth=0.5, zorder=4)
        
        # Mark short segments
        for seg_frame in before_data['short_segments']:
            idx = np.where(before_data['frame_indices'] == seg_frame)[0]
            if len(idx) > 0:
                ax3.plot(before_data['centroids'][idx[0], 0],
                        before_data['centroids'][idx[0], 1],
                        'co', markersize=4, markeredgecolor='black',
                        markeredgewidth=0.5, alpha=0.8, zorder=3)
    
    ax3.set_xlabel('X Position (pixels)')
    ax3.set_ylabel('Y Position (pixels)')
    ax3.set_title('Trajectory - BEFORE (Red=Jumps, Magenta=Islands, Cyan=Short)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Trajectory AFTER
    ax4 = fig.add_subplot(gs[1, 1])
    if len(after_data['centroids']) > 0:
        scatter = ax4.scatter(after_data['centroids'][:, 0],
                            after_data['centroids'][:, 1],
                            c=after_data['frame_indices'],
                            cmap='viridis', s=2, alpha=0.6)
        
        # Mark start and stop
        ax4.plot(after_data['centroids'][0, 0], after_data['centroids'][0, 1],
                'g^', markersize=10, label='START', markeredgecolor='white',
                markeredgewidth=1, zorder=6)
        ax4.plot(after_data['centroids'][-1, 0], after_data['centroids'][-1, 1],
                'rs', markersize=10, label='STOP', markeredgecolor='white',
                markeredgewidth=1, zorder=6)
        ax4.legend()
    
    ax4.set_xlabel('X Position (pixels)')
    ax4.set_ylabel('Y Position (pixels)')
    ax4.set_title('Trajectory - AFTER (Filtered with Start/Stop)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    # Stats panels
    ax5 = fig.add_subplot(gs[0, 2])
    ax5.axis('off')
    stats_before = f"""BEFORE FILTERING:

Frames with detections: {len(before_data['frame_indices'])}
Total jumps > {before_data['threshold']:.0f}px: {len(before_data['consecutive_jumps'] + before_data['gap_jumps'])}

Movement (consecutive):
- Mean: {np.mean(before_data['distances']):.2f} px
- Median: {np.median(before_data['distances']):.2f} px
- 95th percentile: {np.percentile(before_data['distances'], 95):.2f} px
- Max: {np.max(before_data['distances']):.2f} px

Detection issues found:
- Single-frame blips: {len(before_data['consecutive_jumps'])}
- Island segments: {len(before_data['islands'])}
- Short segments: {len(set(before_data['short_segments']))}
"""
    ax5.text(0.05, 0.95, stats_before, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Calculate removal statistics
    frames_removed = len(before_data['frame_indices']) - len(after_data['frame_indices'])
    removal_pct = (frames_removed / len(before_data['frame_indices']) * 100) if len(before_data['frame_indices']) > 0 else 0
    
    stats_after = f"""AFTER FILTERING:

Frames with detections: {len(after_data['frame_indices'])}
Frames removed: {frames_removed}
({removal_pct:.1f}% of detected)

Removal breakdown:
- Blips: {len(before_data['consecutive_jumps'])}
- Islands: {len(before_data['islands'])}
- Short Segments: {len(set(before_data['short_segments']))}

Movement (consecutive):
- Mean: {np.mean(after_data['distances']) if len(after_data['distances']) > 0 else 0:.2f} px
- Median: {np.median(after_data['distances']) if len(after_data['distances']) > 0 else 0:.2f} px
- 95th percentile: {np.percentile(after_data['distances'], 95) if len(after_data['distances']) > 0 else 0:.2f} px
- Max: {np.max(after_data['distances']) if len(after_data['distances']) > 0 else 0:.2f} px

Coverage: {before_data['coverage']:.1f}%→{after_data['coverage']:.1f}%
"""
    ax6.text(0.05, 0.95, stats_after, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Add interactive buttons if requested
    if interactive:
        # Add save button
        ax_save = plt.axes([0.4, 0.02, 0.1, 0.03])
        btn_save = Button(ax_save, 'Save Data')
        
        ax_quit = plt.axes([0.51, 0.02, 0.1, 0.03])
        btn_quit = Button(ax_quit, 'Close')
        
        def on_save(event):
            print("Save functionality would go here")
            # This is where you'd trigger saving filtered data back to zarr
        
        def on_quit(event):
            plt.close('all')
        
        btn_save.on_clicked(on_save)
        btn_quit.on_clicked(on_quit)
        
        # Add instruction text
        fig.text(0.5, 0.01, "Press 'W' to save filtered data | 'Q' to quit | Close window to cancel",
                ha='center', fontsize=10, color='blue',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    if not interactive:
        plt.show()
    else:
        return fig, btn_save if interactive else None


def main():
    parser = argparse.ArgumentParser(
        description='Visualize before/after filtering comparison for multi-fish zarr'
    )
    parser.add_argument('zarr_path', help='Path to multi-fish zarr file')
    parser.add_argument('--threshold', type=float, default=200.0,
                       help='Distance threshold for identifying jumps')
    parser.add_argument('--fish-id', type=int, default=None,
                       help='Specific fish ID to analyze')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the plot')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Disable interactive buttons')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FILTERING COMPARISON VISUALIZER")
    print("=" * 60)
    print(f"Zarr file: {args.zarr_path}")
    print(f"Jump threshold: {args.threshold} pixels")
    
    try:
        # Load before and after data
        before_data, after_data = load_before_after_data(
            args.zarr_path,
            threshold=args.threshold,
            fish_id=args.fish_id
        )
        
        print(f"\nBEFORE: {before_data['source']}")
        print(f"  Detections: {len(before_data['frame_indices'])}")
        print(f"  Coverage: {before_data['coverage']:.1f}%")
        
        print(f"\nAFTER: {after_data['source']}")
        print(f"  Detections: {len(after_data['frame_indices'])}")
        print(f"  Coverage: {after_data['coverage']:.1f}%")
        
        # Create comparison plot
        fig, save_btn = create_comparison_plot(
            before_data, after_data,
            save_path=args.output,
            interactive=not args.no_interactive
        )
        
        if not args.no_interactive:
            plt.show()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())