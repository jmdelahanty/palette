#!/usr/bin/env python3
"""
Speed Over Time Plotter

Calculates and visualizes fish speed over time from tracking data.
Uses corrected coordinate transformations for accurate speed calculations.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def get_roi_positions_corrected(root: zarr.Group, roi_id: int, 
                                use_interpolated: bool = True,
                                use_downsampled: bool = False) -> Dict:
    """
    Get positions with CORRECTED coordinate transformations.
    
    Args:
        root: Zarr root group
        roi_id: ROI/fish ID
        use_interpolated: Include interpolated detections
        use_downsampled: Return 640x640 coords instead of full resolution
    
    Returns:
        Dictionary of frame_idx -> [x, y] positions
    """
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
    
    # Get dimensions
    full_width = root.attrs.get('width', 4512)
    full_height = root.attrs.get('height', 4512)
    ds_width = 640
    ds_height = 640
    
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
                
                # CORRECTED: bbox[0] and bbox[1] are already center coordinates
                center_x_norm = bbox[0]
                center_y_norm = bbox[1]
                
                # Convert to pixel coordinates
                if use_downsampled:
                    centroid_x = center_x_norm * ds_width
                    centroid_y = center_y_norm * ds_height
                else:
                    # First to 640x640, then scale up
                    centroid_x_ds = center_x_norm * ds_width
                    centroid_y_ds = center_y_norm * ds_height
                    scale_factor = full_width / ds_width
                    centroid_x = centroid_x_ds * scale_factor
                    centroid_y = centroid_y_ds * scale_factor
                
                positions[frame_idx] = np.array([centroid_x, centroid_y])
        
        cumulative_idx += frame_det_count
    
    # Add interpolated positions if requested
    if use_interpolated and 'interpolated_detections' in root:
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
                    
                    center_x_norm = bbox[0]
                    center_y_norm = bbox[1]
                    
                    if use_downsampled:
                        centroid_x = center_x_norm * ds_width
                        centroid_y = center_y_norm * ds_height
                    else:
                        centroid_x_ds = center_x_norm * ds_width
                        centroid_y_ds = center_y_norm * ds_height
                        scale_factor = full_width / ds_width
                        centroid_x = centroid_x_ds * scale_factor
                        centroid_y = centroid_y_ds * scale_factor
                    
                    positions[frame_idx] = np.array([centroid_x, centroid_y])
    
    return positions


def calculate_speed_for_roi(positions: Dict, fps: float, 
                           window_size: int = 5,
                           pixel_to_mm: Optional[float] = None) -> Dict:
    """
    Calculate speed metrics from positions.
    
    Args:
        positions: Dictionary of frame_idx -> [x, y] positions
        fps: Frames per second
        window_size: Smoothing window size (frames)
        pixel_to_mm: Optional pixel to mm conversion factor
    
    Returns:
        Dictionary with speed metrics
    """
    if not positions:
        return None
    
    # Sort by frame index
    sorted_frames = sorted(positions.keys())
    n_frames = max(sorted_frames) + 1
    
    # Initialize arrays
    instantaneous_speed = np.full(n_frames, np.nan)
    smoothed_speed = np.full(n_frames, np.nan)
    valid_frames = np.zeros(n_frames, dtype=bool)
    
    # Calculate instantaneous speed
    for i in range(1, len(sorted_frames)):
        curr_frame = sorted_frames[i]
        prev_frame = sorted_frames[i-1]
        
        # Only calculate if frames are consecutive or close
        frame_gap = curr_frame - prev_frame
        if frame_gap <= 3:  # Allow small gaps
            curr_pos = positions[curr_frame]
            prev_pos = positions[prev_frame]
            
            # Distance in pixels
            distance = np.linalg.norm(curr_pos - prev_pos)
            
            # Time in seconds
            time_delta = frame_gap / fps
            
            # Speed in pixels/second
            speed = distance / time_delta if time_delta > 0 else 0
            instantaneous_speed[curr_frame] = speed
            valid_frames[curr_frame] = True
    
    # Apply smoothing
    if window_size > 1:
        # Use gaussian filter for smooth results
        sigma = window_size / 4
        smoothed_speed[valid_frames] = gaussian_filter1d(
            instantaneous_speed[valid_frames], 
            sigma=sigma, 
            mode='nearest'
        )
    else:
        smoothed_speed = instantaneous_speed.copy()
    
    # Calculate statistics
    valid_speeds = instantaneous_speed[valid_frames & ~np.isnan(instantaneous_speed)]
    
    stats = {
        'mean_speed_px_s': np.mean(valid_speeds) if len(valid_speeds) > 0 else 0,
        'median_speed_px_s': np.median(valid_speeds) if len(valid_speeds) > 0 else 0,
        'std_speed_px_s': np.std(valid_speeds) if len(valid_speeds) > 0 else 0,
        'max_speed_px_s': np.max(valid_speeds) if len(valid_speeds) > 0 else 0,
        'min_speed_px_s': np.min(valid_speeds) if len(valid_speeds) > 0 else 0,
        'percentile_25_px_s': np.percentile(valid_speeds, 25) if len(valid_speeds) > 0 else 0,
        'percentile_75_px_s': np.percentile(valid_speeds, 75) if len(valid_speeds) > 0 else 0
    }
    
    # Add mm/s statistics if calibration available
    if pixel_to_mm:
        for key in list(stats.keys()):
            mm_key = key.replace('_px_s', '_mm_s')
            stats[mm_key] = stats[key] * pixel_to_mm
    
    return {
        'instantaneous_speed': instantaneous_speed,
        'smoothed_speed': smoothed_speed,
        'valid_frames': valid_frames,
        'positions': positions,
        'statistics': stats,
        'fps': fps,
        'window_size': window_size,
        'pixel_to_mm': pixel_to_mm,
        'total_frames': n_frames,
        'frames_with_data': len(sorted_frames)
    }


def plot_speed_over_time(results: Dict, roi_id: int,
                        save_path: Optional[str] = None,
                        show_interpolated_marker: bool = True):
    """
    Create comprehensive speed visualization.
    
    Args:
        results: Dictionary from calculate_speed_for_roi
        roi_id: ROI/fish ID for title
        save_path: Optional path to save figure
        show_interpolated_marker: Mark interpolated vs original detections
    """
    if not results:
        print("No results to plot")
        return
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Time axis in seconds
    time_seconds = np.arange(results['total_frames']) / results['fps']
    valid = results['valid_frames']
    
    # Determine units
    if results['pixel_to_mm']:
        speed_unit = 'mm/s'
        speed_data = results['instantaneous_speed'] * results['pixel_to_mm']
        smooth_data = results['smoothed_speed'] * results['pixel_to_mm']
        stats = {k.replace('_px_s', ''): v for k, v in results['statistics'].items() if '_mm_s' in k}
        for k in list(stats.keys()):
            stats[k.replace('_mm_s', '')] = stats[k]
    else:
        speed_unit = 'pixels/s'
        speed_data = results['instantaneous_speed']
        smooth_data = results['smoothed_speed']
        stats = {k.replace('_px_s', ''): v for k, v in results['statistics'].items() if '_px_s' in k}
    
    # Create grid
    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. Speed over time (main plot)
    ax1 = fig.add_subplot(gs[0:2, :])
    
    # Plot instantaneous speed
    ax1.plot(time_seconds[valid], speed_data[valid],
             'b-', alpha=0.3, linewidth=0.5, label='Instantaneous')
    
    # Plot smoothed speed
    ax1.plot(time_seconds[valid], smooth_data[valid],
             'r-', linewidth=2, label=f'Smoothed (window={results["window_size"]} frames)')
    
    # Add mean line
    mean_speed = stats.get('mean_speed', 0)
    ax1.axhline(y=mean_speed, color='green', linestyle='--', alpha=0.5,
               label=f'Mean: {mean_speed:.2f} {speed_unit}')
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel(f'Speed ({speed_unit})', fontsize=12)
    ax1.set_title(f'Fish {roi_id} - Speed Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Set reasonable y-limits
    if np.any(valid):
        valid_speeds = speed_data[valid]
        y_max = np.percentile(valid_speeds[~np.isnan(valid_speeds)], 99) * 1.1
        ax1.set_ylim([0, y_max])
    
    # 2. Speed distribution
    ax2 = fig.add_subplot(gs[2, 0])
    
    valid_speeds = speed_data[valid & ~np.isnan(speed_data)]
    if len(valid_speeds) > 0:
        ax2.hist(valid_speeds, bins=50, alpha=0.7, color='blue', edgecolor='black')
        
        # Add statistical markers
        ax2.axvline(x=stats.get('mean_speed', 0), color='green', 
                   linestyle='--', linewidth=2, label=f'Mean: {stats.get("mean_speed", 0):.1f}')
        ax2.axvline(x=stats.get('median_speed', 0), color='orange', 
                   linestyle='--', linewidth=2, label=f'Median: {stats.get("median_speed", 0):.1f}')
        
        ax2.set_xlabel(f'Speed ({speed_unit})')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Speed Distribution')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    # 3. Statistics box
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.axis('off')
    
    stats_text = f"""
Speed Statistics
━━━━━━━━━━━━━━━━━━━━━━
Mean:     {stats.get('mean_speed', 0):.2f} {speed_unit}
Median:   {stats.get('median_speed', 0):.2f} {speed_unit}
Std Dev:  {stats.get('std_speed', 0):.2f} {speed_unit}
Maximum:  {stats.get('max_speed', 0):.2f} {speed_unit}
Minimum:  {stats.get('min_speed', 0):.2f} {speed_unit}

25th percentile: {stats.get('percentile_25', 0):.2f} {speed_unit}
75th percentile: {stats.get('percentile_75', 0):.2f} {speed_unit}

Coverage
━━━━━━━━━━━━━━━━━━━━━━
Total frames:      {results['total_frames']}
Frames with data:  {results['frames_with_data']}
Detection rate:    {results['frames_with_data']/results['total_frames']*100:.1f}%
"""
    
    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Speed Analysis - Fish {roi_id}', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    plt.show()


def export_speed_to_csv(results: Dict, roi_id: int, output_path: str):
    """Export speed data to CSV file."""
    if not results:
        print("No results to export")
        return
    
    # Create time array
    time_seconds = np.arange(results['total_frames']) / results['fps']
    
    # Build DataFrame
    df_data = {
        'frame': np.arange(results['total_frames']),
        'time_seconds': time_seconds,
        'instantaneous_speed_px_s': results['instantaneous_speed'],
        'smoothed_speed_px_s': results['smoothed_speed'],
        'valid': results['valid_frames'].astype(int)
    }
    
    # Add mm/s columns if calibrated
    if results['pixel_to_mm']:
        df_data['instantaneous_speed_mm_s'] = results['instantaneous_speed'] * results['pixel_to_mm']
        df_data['smoothed_speed_mm_s'] = results['smoothed_speed'] * results['pixel_to_mm']
    
    df = pd.DataFrame(df_data)
    
    # Add metadata as comments
    with open(output_path, 'w') as f:
        f.write(f"# Speed analysis for Fish {roi_id}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# FPS: {results['fps']}\n")
        f.write(f"# Smoothing window: {results['window_size']} frames\n")
        if results['pixel_to_mm']:
            f.write(f"# Pixel to mm: {results['pixel_to_mm']:.6f}\n")
        f.write("#\n")
        df.to_csv(f, index=False)
    
    print(f"✅ Speed data exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate and plot fish speed over time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot speed for ROI 3
  %(prog)s detections.zarr --roi 3
  
  # Use downsampled coordinates (faster)
  %(prog)s detections.zarr --roi 3 --downsampled
  
  # Export data to CSV
  %(prog)s detections.zarr --roi 3 --export speed_roi3.csv
  
  # Custom smoothing window
  %(prog)s detections.zarr --roi 3 --window 10
  
  # Without interpolated detections
  %(prog)s detections.zarr --roi 3 --no-interpolated
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--roi', type=int, required=True,
                       help='ROI/fish ID to analyze')
    parser.add_argument('--window', type=int, default=5,
                       help='Smoothing window size in frames (default: 5)')
    parser.add_argument('--downsampled', action='store_true',
                       help='Use 640x640 coordinates instead of full resolution')
    parser.add_argument('--no-interpolated', action='store_true',
                       help='Exclude interpolated detections')
    parser.add_argument('--export', type=str,
                       help='Export speed data to CSV file')
    parser.add_argument('--save', type=str,
                       help='Save figure to file')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot')
    
    args = parser.parse_args()
    
    # Open zarr file
    root = zarr.open_group(args.zarr_path, mode='r')
    
    # Get metadata
    fps = root.attrs.get('fps', 60.0)
    pixel_to_mm = None
    if 'calibration' in root:
        pixel_to_mm = root['calibration'].attrs.get('pixel_to_mm', None)
    
    resolution = "640×640" if args.downsampled else "4512×4512"
    print(f"\n📊 Speed Analysis for Fish {args.roi}")
    print(f"Resolution: {resolution}")
    print(f"FPS: {fps}")
    if pixel_to_mm:
        print(f"Calibration: 1 pixel = {pixel_to_mm:.4f} mm")
    
    # Get positions
    print("Loading position data...")
    positions = get_roi_positions_corrected(
        root, args.roi, 
        use_interpolated=not args.no_interpolated,
        use_downsampled=args.downsampled
    )
    
    if not positions:
        print(f"❌ No positions found for ROI {args.roi}")
        return
    
    print(f"Found {len(positions)} frames with detections")
    
    # Calculate speed
    print("Calculating speed metrics...")
    results = calculate_speed_for_roi(
        positions, fps, 
        window_size=args.window,
        pixel_to_mm=pixel_to_mm if not args.downsampled else None
    )
    
    # Export if requested
    if args.export:
        export_speed_to_csv(results, args.roi, args.export)
    
    # Plot
    if not args.no_show or args.save:
        print("Generating visualization...")
        plot_speed_over_time(results, args.roi, save_path=args.save)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()