#!/usr/bin/env python3
"""
Simple Fish-Chaser Distance Plotter

A straightforward script to plot the distance between fish and chaser over time,
integrating YOLO detections from zarr files with chaser positions from H5 files.
"""

import zarr
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_and_plot_distance(zarr_path: str, h5_path: str, 
                          start_frame: int = 0, 
                          end_frame: int = None,
                          show_events: bool = True):
    """
    Load data and plot fish-chaser distance over time.
    
    Args:
        zarr_path: Path to zarr file with YOLO detections
        h5_path: Path to H5 file with chaser positions
        start_frame: Starting frame for plot (default: 0)
        end_frame: Ending frame for plot (default: all)
        show_events: Whether to mark chase events on the plot
    """
    print("Loading data...")
    
    # Load zarr data (fish detections)
    root = zarr.open(zarr_path, mode='r')
    
    # Try to get best available data
    if 'preprocessing' in root and root['preprocessing'].attrs.get('latest'):
        latest = root['preprocessing'].attrs['latest']
        data = root['preprocessing'][latest]
        bboxes = data['bboxes'][:]
        n_detections = data['n_detections'][:]
        print(f"  Using preprocessed data: {latest}")
    else:
        bboxes = root['bboxes'][:]
        n_detections = root['n_detections'][:]
        print("  Using raw detection data")
    
    # Load H5 data
    with h5py.File(h5_path, 'r') as f:
        # Frame metadata for alignment
        frame_metadata = f['/video_metadata/frame_metadata'][:]
        
        # Chaser states
        chaser_states = f['/tracking_data/chaser_states'][:]
        
        # Events (if showing)
        if show_events and '/events' in f:
            events = f['/events'][:]
        else:
            events = None
    
    print(f"  Loaded {len(bboxes)} frames of detections")
    print(f"  Loaded {len(chaser_states)} chaser states")
    
    # Set frame range
    if end_frame is None:
        end_frame = min(len(bboxes), 15000)  # Limit for performance
    
    # Camera to texture scaling (from your system)
    texture_to_camera_scale = 4512 / 358  # ~12.6
    
    print(f"\nProcessing frames {start_frame} to {end_frame}...")
    
    # Initialize arrays for storing data
    distances = []
    timestamps = []
    frame_numbers = []
    
    # Get unique camera frames in order
    unique_camera_frames = np.unique(frame_metadata['triggering_camera_frame_id'])
    
    # Process each frame
    for i, cam_frame in enumerate(unique_camera_frames[start_frame:end_frame]):
        # Get frame metadata
        meta_mask = frame_metadata['triggering_camera_frame_id'] == cam_frame
        if not np.any(meta_mask):
            continue
        
        meta = frame_metadata[meta_mask][0]
        stim_frame = meta['stimulus_frame_num']
        timestamp = meta['timestamp_ns'] / 1e9  # Convert to seconds
        
        # Get fish position from zarr (if detected)
        zarr_idx = i  # Assuming aligned indexing
        if zarr_idx < len(bboxes) and n_detections[zarr_idx] > 0:
            bbox = bboxes[zarr_idx, 0]  # First detection
            fish_x = (bbox[0] + bbox[2]) / 2  # Center x in camera space
            fish_y = (bbox[1] + bbox[3]) / 2  # Center y in camera space
        else:
            fish_x = fish_y = np.nan
        
        # Get chaser position for this stimulus frame
        chaser_mask = chaser_states['stimulus_frame_num'] == stim_frame
        if np.any(chaser_mask):
            chaser = chaser_states[chaser_mask][0]
            # Convert from texture space to camera space
            chaser_x = chaser['chaser_pos_x'] * texture_to_camera_scale
            chaser_y = chaser['chaser_pos_y'] * texture_to_camera_scale
        else:
            chaser_x = chaser_y = np.nan
        
        # Calculate distance
        if not np.isnan(fish_x) and not np.isnan(chaser_x):
            distance = np.sqrt((fish_x - chaser_x)**2 + (fish_y - chaser_y)**2)
        else:
            distance = np.nan
        
        distances.append(distance)
        timestamps.append(timestamp)
        frame_numbers.append(cam_frame)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} frames...")
    
    # Convert to arrays
    distances = np.array(distances)
    timestamps = np.array(timestamps)
    frame_numbers = np.array(frame_numbers)
    
    # Convert timestamps to relative time (seconds from start)
    if len(timestamps) > 0:
        timestamps = timestamps - timestamps[0]
    
    print(f"\nPlotting {np.sum(~np.isnan(distances))} valid distance measurements...")
    
    # Calculate fish speed
    print("Calculating fish speed...")
    fish_speeds = []
    fish_positions = []
    
    for i, cam_frame in enumerate(unique_camera_frames[start_frame:end_frame]):
        # Get fish position for this frame
        zarr_idx = i
        if zarr_idx < len(bboxes) and n_detections[zarr_idx] > 0:
            bbox = bboxes[zarr_idx, 0]
            fish_x = (bbox[0] + bbox[2]) / 2
            fish_y = (bbox[1] + bbox[3]) / 2
            fish_positions.append([fish_x, fish_y])
        else:
            fish_positions.append([np.nan, np.nan])
    
    fish_positions = np.array(fish_positions)
    
    # Calculate speed (pixels per second) from frame-to-frame displacement
    for i in range(len(fish_positions)):
        if i == 0:
            fish_speeds.append(np.nan)
        else:
            if not np.isnan(fish_positions[i, 0]) and not np.isnan(fish_positions[i-1, 0]):
                displacement = np.sqrt((fish_positions[i, 0] - fish_positions[i-1, 0])**2 + 
                                     (fish_positions[i, 1] - fish_positions[i-1, 1])**2)
                time_diff = timestamps[i] - timestamps[i-1] if i < len(timestamps) else 1/60  # Assume 60 fps
                if time_diff > 0:
                    speed = displacement / time_diff
                else:
                    speed = np.nan
            else:
                speed = np.nan
            fish_speeds.append(speed)
    
    fish_speeds = np.array(fish_speeds)
    
    # Create the plot with shared x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), 
                                        gridspec_kw={'height_ratios': [3, 2, 1]})
    
    # Main distance plot
    valid_mask = ~np.isnan(distances)
    ax1.plot(timestamps[valid_mask], distances[valid_mask], 
             'b-', linewidth=1, alpha=0.8, label='Fish-Chaser Distance')
    
    # Add rolling average
    window = 60  # 1 second at 60 fps
    if np.sum(valid_mask) > window:
        from scipy.ndimage import uniform_filter1d
        valid_distances = distances[valid_mask]
        smoothed = uniform_filter1d(valid_distances, size=window, mode='nearest')
        ax1.plot(timestamps[valid_mask], smoothed, 
                'r-', linewidth=2, alpha=0.7, label=f'Rolling avg ({window} frames)')
    
    # Mark chase events if available
    if events is not None and show_events:
        chase_starts = events[events['event_type_id'] == 27]  # CHASE_START
        chase_ends = events[events['event_type_id'] == 28]    # CHASE_END
        
        for start, end in zip(chase_starts, chase_ends):
            start_time = start['timestamp_ns_session'] / 1e9
            end_time = end['timestamp_ns_session'] / 1e9
            
            # Shade chase periods
            ax1.axvspan(start_time, end_time, alpha=0.2, color='yellow', label='Chase' if start is chase_starts[0] else '')
            ax1.axvline(x=start_time, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax1.axvline(x=end_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.set_ylabel('Distance (pixels)', fontsize=12)
    ax1.set_title('Fish-Chaser Distance Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add statistics text
    valid_distances = distances[valid_mask]
    if len(valid_distances) > 0:
        stats_text = (f'Mean: {np.mean(valid_distances):.0f} px | '
                     f'Min: {np.min(valid_distances):.0f} px | '
                     f'Max: {np.max(valid_distances):.0f} px | '
                     f'Coverage: {np.sum(valid_mask)/len(distances)*100:.1f}%')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Fish speed plot
    valid_speed_mask = ~np.isnan(fish_speeds)
    ax2.plot(timestamps[valid_speed_mask], fish_speeds[valid_speed_mask], 
             'g-', linewidth=1, alpha=0.6, label='Fish Speed')
    
    # Add smoothed speed
    if np.sum(valid_speed_mask) > window:
        from scipy.ndimage import uniform_filter1d
        valid_speeds = fish_speeds[valid_speed_mask]
        # Clip extreme values for better smoothing
        clipped_speeds = np.clip(valid_speeds, 0, np.percentile(valid_speeds, 99))
        smoothed_speed = uniform_filter1d(clipped_speeds, size=window, mode='nearest')
        ax2.plot(timestamps[valid_speed_mask], smoothed_speed, 
                'darkgreen', linewidth=2, alpha=0.8, label=f'Rolling avg ({window} frames)')
    
    # Mark chase events on speed plot too
    if events is not None and show_events:
        for start, end in zip(chase_starts, chase_ends):
            start_time = start['timestamp_ns_session'] / 1e9
            end_time = end['timestamp_ns_session'] / 1e9
            ax2.axvspan(start_time, end_time, alpha=0.2, color='yellow')
    
    # Add escape threshold line
    escape_threshold = 500  # pixels/second - adjust as needed
    ax2.axhline(y=escape_threshold, color='red', linestyle='--', alpha=0.5, 
                linewidth=1, label='Escape threshold')
    
    ax2.set_ylabel('Speed (pixels/second)', fontsize=12)
    ax2.set_title('Fish Swimming Speed', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_ylim([0, min(2000, np.nanpercentile(fish_speeds, 99.5) * 1.1)])  # Cap y-axis for visibility
    
    # Add speed statistics
    valid_speeds = fish_speeds[valid_speed_mask]
    if len(valid_speeds) > 0:
        speed_stats_text = (f'Mean: {np.mean(valid_speeds):.0f} px/s | '
                           f'Median: {np.median(valid_speeds):.0f} px/s | '
                           f'Max: {np.max(valid_speeds):.0f} px/s')
        ax2.text(0.02, 0.98, speed_stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Detection coverage plot
    ax3.fill_between(timestamps, 0, valid_mask.astype(int), 
                     alpha=0.5, color='gray', step='mid')
    ax3.set_ylabel('Detection', fontsize=10)
    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['No', 'Yes'])
    ax3.grid(True, alpha=0.3)
    
    # Synchronize x-axes
    ax2.set_xlim(ax1.get_xlim())
    ax3.set_xlim(ax1.get_xlim())
    
    plt.tight_layout()
    
    # Add pixel to mm conversion note
    pixel_to_mm = 0.019605  # From your calibration
    fig.text(0.99, 0.01, f'1 pixel = {pixel_to_mm:.4f} mm', 
             ha='right', fontsize=9, style='italic', color='gray')
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    if len(valid_distances) > 0:
        print(f"Mean distance: {np.mean(valid_distances):.1f} pixels ({np.mean(valid_distances)*pixel_to_mm:.2f} mm)")
        print(f"Median distance: {np.median(valid_distances):.1f} pixels ({np.median(valid_distances)*pixel_to_mm:.2f} mm)")
        print(f"Min distance: {np.min(valid_distances):.1f} pixels ({np.min(valid_distances)*pixel_to_mm:.2f} mm)")
        print(f"Max distance: {np.max(valid_distances):.1f} pixels ({np.max(valid_distances)*pixel_to_mm:.2f} mm)")
        print(f"Std deviation: {np.std(valid_distances):.1f} pixels ({np.std(valid_distances)*pixel_to_mm:.2f} mm)")
        print(f"Valid measurements: {len(valid_distances)} / {len(distances)} ({len(valid_distances)/len(distances)*100:.1f}%)")
        
        print(f"\nSpeed statistics:")
        print(f"Mean speed: {np.mean(valid_speeds):.1f} pixels/s ({np.mean(valid_speeds)*pixel_to_mm:.2f} mm/s)")
        print(f"Median speed: {np.median(valid_speeds):.1f} pixels/s ({np.median(valid_speeds)*pixel_to_mm:.2f} mm/s)")
        print(f"Max speed: {np.max(valid_speeds):.1f} pixels/s ({np.max(valid_speeds)*pixel_to_mm:.2f} mm/s)")
        print(f"95th percentile: {np.percentile(valid_speeds, 95):.1f} pixels/s ({np.percentile(valid_speeds, 95)*pixel_to_mm:.2f} mm/s)")
        
        # Escape events (high speed events)
        escape_threshold = 500  # pixels/second
        escape_events = np.sum(valid_speeds > escape_threshold)
        escape_percent = escape_events / len(valid_speeds) * 100
        print(f"\nEscape responses (>{escape_threshold} px/s): {escape_events} events ({escape_percent:.1f}% of time)")
        
        # Calculate time spent at different distances
        close_threshold = 200  # pixels
        medium_threshold = 500  # pixels
        
        close = np.sum(valid_distances < close_threshold) / len(valid_distances) * 100
        medium = np.sum((valid_distances >= close_threshold) & (valid_distances < medium_threshold)) / len(valid_distances) * 100
        far = np.sum(valid_distances >= medium_threshold) / len(valid_distances) * 100
        
        print(f"\nDistance distribution:")
        print(f"  Close (<{close_threshold} px): {close:.1f}%")
        print(f"  Medium ({close_threshold}-{medium_threshold} px): {medium:.1f}%")
        print(f"  Far (>{medium_threshold} px): {far:.1f}%")
    else:
        print("No valid distance measurements found!")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Simple plotter for fish-chaser distance over time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script creates a simple plot showing the distance between fish and chaser
over time, with chase events marked if available.

Examples:
  %(prog)s detections.zarr analysis.h5
  %(prog)s detections.zarr analysis.h5 --start 1000 --end 5000
  %(prog)s detections.zarr analysis.h5 --no-events
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file with YOLO detections')
    parser.add_argument('h5_path', help='Path to H5 analysis file')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting frame (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='Ending frame (default: all or 15000)')
    parser.add_argument('--no-events', action='store_true',
                       help="Don't show chase events on plot")
    
    args = parser.parse_args()
    
    # Create plot
    load_and_plot_distance(
        zarr_path=args.zarr_path,
        h5_path=args.h5_path,
        start_frame=args.start,
        end_frame=args.end,
        show_events=not args.no_events
    )
    
    return 0


if __name__ == '__main__':
    exit(main())