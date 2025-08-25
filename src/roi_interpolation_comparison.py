#!/usr/bin/env python3
"""
ROI Interpolation Before/After Comparison

Visualizes the effect of interpolation on specific ROIs:
- Original detections with gaps
- After interpolation
- Coverage timeline
- Gap analysis
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
from typing import Optional
from rich.console import Console

console = Console()


def extract_roi_trajectory(zarr_path: str, roi_id: int):
    """Extract trajectory data for a specific ROI."""
    root = zarr.open_group(zarr_path, mode='r')
    
    # Load detection data
    detect_group = root['detect_runs']
    latest_detect = detect_group.attrs['latest']
    n_detections = detect_group[latest_detect]['n_detections'][:]
    bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
    
    # Load ID assignments
    id_key = 'id_assignments_runs' if 'id_assignments_runs' in root else 'id_assignments'
    id_group = root[id_key]
    latest_id = id_group.attrs['latest']
    detection_ids = id_group[latest_id]['detection_ids'][:]
    n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
    
    # Get video dimensions
    img_width = root.attrs.get('width', 4512)
    img_height = root.attrs.get('height', 4512)
    fps = root.attrs.get('fps', 60.0)
    
    # Extract original positions
    original_positions = {}
    cumulative_idx = 0
    
    for frame_idx in range(len(n_detections)):
        frame_det_count = int(n_detections[frame_idx])
        
        if frame_det_count > 0 and n_detections_per_roi[frame_idx, roi_id] > 0:
            frame_detection_ids = detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
            roi_mask = frame_detection_ids == roi_id
            
            if np.any(roi_mask):
                roi_idx = np.where(roi_mask)[0][0]
                bbox = bbox_coords[cumulative_idx + roi_idx]
                # Convert to pixel coordinates
                original_positions[frame_idx] = {
                    'x': bbox[0] * img_width,
                    'y': bbox[1] * img_height,
                    'w': bbox[2] * img_width,
                    'h': bbox[3] * img_height
                }
        
        cumulative_idx += frame_det_count
    
    # Load interpolated positions if available
    interpolated_positions = {}
    if 'interpolated_detections' in root:
        interp_det_group = root['interpolated_detections']
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
                    interpolated_positions[frame_idx] = {
                        'x': bbox[0] * img_width,
                        'y': bbox[1] * img_height,
                        'w': bbox[2] * img_width,
                        'h': bbox[3] * img_height
                    }
    
    return {
        'original': original_positions,
        'interpolated': interpolated_positions,
        'total_frames': len(n_detections),
        'fps': fps,
        'img_width': img_width,
        'img_height': img_height,
        'roi_detections': n_detections_per_roi[:, roi_id]
    }


def create_comparison_plot(zarr_path: str, roi_id: int, save_path: Optional[Path] = None):
    """Create before/after comparison plot for ROI interpolation."""
    
    console.print(f"\n[bold cyan]ROI {roi_id} Interpolation Comparison[/bold cyan]")
    
    # Extract data
    data = extract_roi_trajectory(zarr_path, roi_id)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.2,
                          height_ratios=[1.2, 0.4, 0.8, 0.8])
    
    # Color scheme
    color_original = '#2E86AB'  # Blue
    color_interpolated = '#A23B72'  # Purple
    
    # --- Row 1: Trajectories ---
    
    # Original trajectory (left)
    ax1 = fig.add_subplot(gs[0, 0])
    if data['original']:
        frames = sorted(data['original'].keys())
        x_coords = [data['original'][f]['x'] for f in frames]
        y_coords = [data['original'][f]['y'] for f in frames]
        
        # Plot trajectory
        ax1.plot(x_coords, y_coords, color=color_original, alpha=0.3, linewidth=0.5)
        scatter = ax1.scatter(x_coords, y_coords, c=frames, cmap='viridis',
                            s=2, alpha=0.6)
        plt.colorbar(scatter, ax=ax1, label='Frame')
        
        # Mark start and end
        ax1.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
        ax1.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    
    coverage_original = len(data['original']) / data['total_frames'] * 100
    ax1.set_title(f"Original Detections\nCoverage: {len(data['original'])}/{data['total_frames']} ({coverage_original:.1f}%)",
                 fontweight='bold')
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Combined trajectory (right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot original points
    if data['original']:
        frames_orig = sorted(data['original'].keys())
        x_orig = [data['original'][f]['x'] for f in frames_orig]
        y_orig = [data['original'][f]['y'] for f in frames_orig]
        ax2.scatter(x_orig, y_orig, c=color_original, s=2, alpha=0.6, label='Original')
    
    # Plot interpolated points
    if data['interpolated']:
        frames_interp = sorted(data['interpolated'].keys())
        x_interp = [data['interpolated'][f]['x'] for f in frames_interp]
        y_interp = [data['interpolated'][f]['y'] for f in frames_interp]
        ax2.scatter(x_interp, y_interp, c=color_interpolated, s=3, alpha=0.5, label='Interpolated')
    
    # Plot complete trajectory
    all_frames = sorted(set(list(data['original'].keys()) + list(data['interpolated'].keys())))
    if all_frames:
        x_all = []
        y_all = []
        for f in all_frames:
            if f in data['original']:
                x_all.append(data['original'][f]['x'])
                y_all.append(data['original'][f]['y'])
            elif f in data['interpolated']:
                x_all.append(data['interpolated'][f]['x'])
                y_all.append(data['interpolated'][f]['y'])
        
        ax2.plot(x_all, y_all, 'k-', alpha=0.2, linewidth=0.5)
    
    coverage_combined = (len(data['original']) + len(data['interpolated'])) / data['total_frames'] * 100
    ax2.set_title(f"After Interpolation\nCoverage: {len(data['original']) + len(data['interpolated'])}/{data['total_frames']} ({coverage_combined:.1f}%)",
                 fontweight='bold')
    ax2.set_xlabel('X Position (pixels)')
    ax2.set_ylabel('Y Position (pixels)')
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # --- Row 2: Detection Barcode ---
    
    # Original barcode
    ax3 = fig.add_subplot(gs[1, 0])
    detection_mask_orig = np.zeros(data['total_frames'])
    for f in data['original'].keys():
        detection_mask_orig[f] = 1
    
    barcode_data = detection_mask_orig.reshape(1, -1)
    time_axis = np.arange(data['total_frames']) / data['fps']
    
    ax3.imshow(barcode_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
              interpolation='nearest', extent=[0, time_axis[-1], 0, 1])
    ax3.set_xlim([0, time_axis[-1]])
    ax3.set_xlabel('Time (seconds)')
    ax3.set_yticks([])
    ax3.set_title('Detection Presence (Green=Detection, Red=Gap)', fontsize=10)
    
    # Combined barcode
    ax4 = fig.add_subplot(gs[1, 1])
    detection_mask_combined = detection_mask_orig.copy()
    for f in data['interpolated'].keys():
        detection_mask_combined[f] = 0.7  # Different value for interpolated
    
    # Custom colormap for three states
    from matplotlib.colors import ListedColormap
    colors = ['red', 'yellow', 'green']  # gap, interpolated, real
    n_bins = 3
    cmap = ListedColormap(colors)
    
    barcode_combined = detection_mask_combined.reshape(1, -1)
    ax4.imshow(barcode_combined, aspect='auto', cmap=cmap, vmin=0, vmax=1,
              interpolation='nearest', extent=[0, time_axis[-1], 0, 1])
    ax4.set_xlim([0, time_axis[-1]])
    ax4.set_xlabel('Time (seconds)')
    ax4.set_yticks([])
    ax4.set_title('Detection Presence (Green=Real, Yellow=Interpolated, Red=Gap)', fontsize=10)
    
    # --- Row 3: Rolling Coverage ---
    
    window = 100  # frames
    
    # Original coverage
    ax5 = fig.add_subplot(gs[2, 0])
    rolling_orig = np.convolve(detection_mask_orig, np.ones(window)/window, mode='same') * 100
    ax5.fill_between(time_axis, 0, rolling_orig, color=color_original, alpha=0.3)
    ax5.plot(time_axis, rolling_orig, color=color_original, alpha=0.8, linewidth=1)
    ax5.axhline(y=coverage_original, color='black', linestyle='--', alpha=0.5, label=f'Mean: {coverage_original:.1f}%')
    ax5.set_ylim([0, 105])
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Detection Coverage (%)')
    ax5.set_title(f'Rolling Coverage (window={window} frames)', fontsize=10)
    ax5.legend(loc='lower right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Combined coverage
    ax6 = fig.add_subplot(gs[2, 1])
    detection_mask_for_rolling = (detection_mask_combined > 0).astype(float)
    rolling_combined = np.convolve(detection_mask_for_rolling, np.ones(window)/window, mode='same') * 100
    ax6.fill_between(time_axis, 0, rolling_combined, color=color_interpolated, alpha=0.3)
    ax6.plot(time_axis, rolling_combined, color=color_interpolated, alpha=0.8, linewidth=1)
    ax6.axhline(y=coverage_combined, color='black', linestyle='--', alpha=0.5, label=f'Mean: {coverage_combined:.1f}%')
    ax6.set_ylim([0, 105])
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Detection Coverage (%)')
    ax6.set_title(f'Rolling Coverage (window={window} frames)', fontsize=10)
    ax6.legend(loc='lower right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # --- Row 4: Gap Analysis ---
    
    # Find gaps in original
    ax7 = fig.add_subplot(gs[3, 0])
    gaps_orig = []
    in_gap = False
    gap_start = None
    
    for i in range(len(detection_mask_orig)):
        if detection_mask_orig[i] == 0 and not in_gap:
            gap_start = i
            in_gap = True
        elif detection_mask_orig[i] == 1 and in_gap:
            gaps_orig.append(i - gap_start)
            in_gap = False
    if in_gap:
        gaps_orig.append(len(detection_mask_orig) - gap_start)
    
    if gaps_orig:
        max_gap = min(max(gaps_orig), 100)
        bins = np.arange(0, max_gap + 2, 5)
        ax7.hist(gaps_orig, bins=bins, color=color_original, alpha=0.7, edgecolor='black')
        ax7.axvline(x=np.mean(gaps_orig), color='red', linestyle='--', label=f'Mean: {np.mean(gaps_orig):.1f}')
        ax7.axvline(x=np.median(gaps_orig), color='orange', linestyle='--', label=f'Median: {np.median(gaps_orig):.1f}')
        ax7.set_xlabel('Gap Size (frames)')
        ax7.set_ylabel('Count')
        ax7.set_title(f'Gap Distribution | Total: {len(gaps_orig)} gaps', fontsize=10)
        ax7.legend(loc='upper right', fontsize=8)
    else:
        ax7.text(0.5, 0.5, 'No gaps!', ha='center', va='center', fontsize=12, color='green')
        ax7.set_title('Gap Distribution', fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # Gaps after interpolation
    ax8 = fig.add_subplot(gs[3, 1])
    gaps_after = []
    in_gap = False
    gap_start = None
    
    for i in range(len(detection_mask_combined)):
        if detection_mask_combined[i] == 0 and not in_gap:
            gap_start = i
            in_gap = True
        elif detection_mask_combined[i] > 0 and in_gap:
            gaps_after.append(i - gap_start)
            in_gap = False
    if in_gap:
        gaps_after.append(len(detection_mask_combined) - gap_start)
    
    if gaps_after:
        max_gap = min(max(gaps_after), 100)
        bins = np.arange(0, max_gap + 2, 5)
        ax8.hist(gaps_after, bins=bins, color=color_interpolated, alpha=0.7, edgecolor='black')
        ax8.set_xlabel('Gap Size (frames)')
        ax8.set_ylabel('Count')
        ax8.set_title(f'Remaining Gaps | Total: {len(gaps_after)}', fontsize=10)
    else:
        ax8.text(0.5, 0.5, '✓ No gaps remaining!', ha='center', va='center', 
                fontsize=12, color='green', weight='bold')
        ax8.set_title('Gap Distribution After Interpolation', fontsize=10)
    ax8.grid(True, alpha=0.3)
    
    # Overall title
    plt.suptitle(f'ROI {roi_id} - Interpolation Before/After Comparison', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Figure saved to:[/green] {save_path}")
    
    plt.show()
    
    # Print summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Original coverage: {coverage_original:.1f}%")
    console.print(f"  After interpolation: {coverage_combined:.1f}%")
    console.print(f"  Coverage gain: +{coverage_combined - coverage_original:.1f}%")
    console.print(f"  Gaps before: {len(gaps_orig)}")
    console.print(f"  Gaps after: {len(gaps_after)}")
    console.print(f"  Interpolated frames: {len(data['interpolated'])}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare ROI trajectory before and after interpolation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--roi', type=int, default=3,
                       help='ROI ID to analyze (default: 3)')
    parser.add_argument('--save', type=str,
                       help='Path to save figure')
    
    args = parser.parse_args()
    
    save_path = Path(args.save) if args.save else None
    create_comparison_plot(args.zarr_path, args.roi, save_path)


if __name__ == "__main__":
    main()