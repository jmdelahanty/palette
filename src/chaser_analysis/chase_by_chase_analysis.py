#!/usr/bin/env python3
"""
Chase-by-Chase Event Analysis

Analyzes fish behavior around individual chase events, examining:
- Distance dynamics 2 seconds before and after chase onset
- Speed and acceleration changes
- Escape responses and latencies
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.signal import savgol_filter
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chaser_fish_distance_analyzer import UnifiedDistanceAnalyzer


def extract_chase_windows(analyzer, window_before_s: float = 2.0, 
                         window_after_s: float = 2.0) -> List[Dict]:
    """
    Extract data windows around each chase event.
    
    Args:
        analyzer: UnifiedDistanceAnalyzer object
        window_before_s: Seconds before chase onset to include
        window_after_s: Seconds after chase onset to include
    
    Returns:
        List of dictionaries containing chase window data
    """
    data = analyzer.aligned_data
    fps = data.metadata['fps']
    
    # Convert time windows to frames
    frames_before = int(window_before_s * fps)
    frames_after = int(window_after_s * fps)
    
    chase_windows = []
    
    # Find chase start events
    chase_starts = [e for e in data.chase_events if e['type'] == 'start']
    chase_ends = [e for e in data.chase_events if e['type'] == 'end']
    
    for i, start_event in enumerate(chase_starts):
        chase_frame = start_event['frame']
        
        # Skip if too close to beginning or end of recording
        if chase_frame < frames_before or chase_frame + frames_after >= len(data.distances):
            continue
        
        # Find corresponding end event
        end_frame = None
        for end_event in chase_ends:
            if end_event['frame'] > chase_frame:
                end_frame = end_event['frame']
                break
        
        # Extract window data
        window_start = chase_frame - frames_before
        window_end = chase_frame + frames_after
        
        # Time axis relative to chase onset
        time_axis = np.arange(-frames_before, frames_after) / fps
        
        # Extract metrics for window
        window_data = {
            'chase_id': i + 1,
            'chase_frame': chase_frame,
            'chase_time': chase_frame / fps,
            'window_frames': np.arange(window_start, window_end),
            'time_relative': time_axis,
            'distances': data.distances[window_start:window_end].copy(),
            'fish_x': data.fish_x[window_start:window_end].copy(),
            'fish_y': data.fish_y[window_start:window_end].copy(),
            'chaser_x': data.chaser_x[window_start:window_end].copy(),
            'chaser_y': data.chaser_y[window_start:window_end].copy(),
            'chase_end_frame': end_frame,
            'chase_duration_s': (end_frame - chase_frame) / fps if end_frame else None
        }
        
        # Calculate derivatives
        window_data['speed'] = calculate_speed(
            window_data['fish_x'], 
            window_data['fish_y'],
            fps
        )
        
        window_data['acceleration'] = calculate_acceleration(
            window_data['speed'],
            fps
        )
        
        window_data['relative_velocity'] = calculate_relative_velocity(
            window_data['distances'],
            fps
        )
        
        # Mark baseline and response periods
        window_data['baseline_mean_distance'] = np.nanmean(
            window_data['distances'][:frames_before]
        )
        window_data['response_mean_distance'] = np.nanmean(
            window_data['distances'][frames_before:frames_before + int(fps)]  # First second after onset
        )
        
        # Calculate escape metrics
        window_data['escape_metrics'] = calculate_escape_response(
            window_data['speed'][frames_before:],  # From chase onset onward
            window_data['distances'][frames_before:],
            fps
        )
        
        chase_windows.append(window_data)
    
    return chase_windows


def calculate_speed(x: np.ndarray, y: np.ndarray, fps: float) -> np.ndarray:
    """Calculate instantaneous speed from positions."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    speed = np.sqrt(dx**2 + dy**2) * fps
    
    # Smooth to reduce noise
    valid_mask = ~np.isnan(speed)
    if np.sum(valid_mask) > 5:
        speed[valid_mask] = savgol_filter(speed[valid_mask], 
                                         window_length=min(5, np.sum(valid_mask)),
                                         polyorder=2)
    return speed


def calculate_acceleration(speed: np.ndarray, fps: float) -> np.ndarray:
    """Calculate acceleration from speed."""
    acceleration = np.gradient(speed) * fps
    
    # Smooth to reduce noise
    valid_mask = ~np.isnan(acceleration)
    if np.sum(valid_mask) > 5:
        acceleration[valid_mask] = savgol_filter(acceleration[valid_mask],
                                                window_length=min(5, np.sum(valid_mask)),
                                                polyorder=2)
    return acceleration


def calculate_relative_velocity(distances: np.ndarray, fps: float) -> np.ndarray:
    """Calculate relative velocity (negative = approaching, positive = escaping)."""
    return np.gradient(distances) * fps


def calculate_escape_response(speed_after_onset: np.ndarray, 
                             distance_after_onset: np.ndarray,
                             fps: float) -> Dict:
    """
    Calculate escape response metrics.
    """
    metrics = {}
    
    # Find peak speed and its timing
    valid_speed = speed_after_onset[~np.isnan(speed_after_onset)]
    if len(valid_speed) > 0:
        peak_speed_idx = np.nanargmax(speed_after_onset)
        metrics['peak_speed'] = speed_after_onset[peak_speed_idx]
        metrics['peak_speed_latency_s'] = peak_speed_idx / fps
        
        # Check if this qualifies as an escape (speed > threshold)
        escape_threshold = 500  # pixels/second
        if metrics['peak_speed'] > escape_threshold:
            # Find first frame above threshold
            escape_frames = np.where(speed_after_onset > escape_threshold)[0]
            if len(escape_frames) > 0:
                metrics['escape_detected'] = True
                metrics['escape_latency_s'] = escape_frames[0] / fps
                metrics['escape_distance'] = distance_after_onset[escape_frames[0]]
            else:
                metrics['escape_detected'] = False
        else:
            metrics['escape_detected'] = False
    else:
        metrics['escape_detected'] = False
        metrics['peak_speed'] = np.nan
        metrics['peak_speed_latency_s'] = np.nan
    
    return metrics


def plot_chase_by_chase_analysis(chase_windows: List[Dict], 
                                 save_path: Optional[str] = None):
    """
    Create comprehensive visualization of chase-by-chase dynamics.
    """
    if not chase_windows:
        print("No chase windows to plot")
        return
    
    n_chases = len(chase_windows)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3,
                          height_ratios=[1, 1, 1, 0.8])
    
    # Color map for individual traces
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_chases))
    
    # 1. Distance dynamics around chase onset
    ax1 = fig.add_subplot(gs[0, :])
    
    all_distances = []
    for i, window in enumerate(chase_windows):
        ax1.plot(window['time_relative'], window['distances'], 
                alpha=0.3, color=colors[i], linewidth=0.8)
        all_distances.append(window['distances'])
    
    # Calculate and plot mean
    all_distances = np.array(all_distances)
    mean_distance = np.nanmean(all_distances, axis=0)
    std_distance = np.nanstd(all_distances, axis=0)
    
    ax1.plot(chase_windows[0]['time_relative'], mean_distance, 
            'k-', linewidth=3, label='Mean')
    ax1.fill_between(chase_windows[0]['time_relative'],
                     mean_distance - std_distance,
                     mean_distance + std_distance,
                     alpha=0.2, color='black')
    
    # Mark chase onset
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhspan(0, ax1.get_ylim()[1], xmin=0.5, xmax=1.0, 
               alpha=0.1, color='red', label='Chase period')
    
    ax1.set_xlabel('Time relative to chase onset (seconds)')
    ax1.set_ylabel('Distance (pixels)')
    ax1.set_title(f'Distance Dynamics Around Chase Events (n={n_chases})', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Speed dynamics
    ax2 = fig.add_subplot(gs[1, :])
    
    all_speeds = []
    for i, window in enumerate(chase_windows):
        ax2.plot(window['time_relative'], window['speed'],
                alpha=0.3, color=colors[i], linewidth=0.8)
        all_speeds.append(window['speed'])
    
    # Mean speed
    all_speeds = np.array(all_speeds)
    mean_speed = np.nanmean(all_speeds, axis=0)
    std_speed = np.nanstd(all_speeds, axis=0)
    
    ax2.plot(chase_windows[0]['time_relative'], mean_speed,
            'k-', linewidth=3, label='Mean')
    ax2.fill_between(chase_windows[0]['time_relative'],
                     mean_speed - std_speed,
                     mean_speed + std_speed,
                     alpha=0.2, color='black')
    
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time relative to chase onset (seconds)')
    ax2.set_ylabel('Speed (pixels/second)')
    ax2.set_title('Swimming Speed Dynamics', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Acceleration dynamics
    ax3 = fig.add_subplot(gs[2, :])
    
    all_accelerations = []
    for i, window in enumerate(chase_windows):
        ax3.plot(window['time_relative'], window['acceleration'],
                alpha=0.3, color=colors[i], linewidth=0.8)
        all_accelerations.append(window['acceleration'])
    
    # Mean acceleration
    all_accelerations = np.array(all_accelerations)
    mean_acceleration = np.nanmean(all_accelerations, axis=0)
    std_acceleration = np.nanstd(all_accelerations, axis=0)
    
    ax3.plot(chase_windows[0]['time_relative'], mean_acceleration,
            'k-', linewidth=3, label='Mean')
    ax3.fill_between(chase_windows[0]['time_relative'],
                     mean_acceleration - std_acceleration,
                     mean_acceleration + std_acceleration,
                     alpha=0.2, color='black')
    
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Time relative to chase onset (seconds)')
    ax3.set_ylabel('Acceleration (pixels/second²)')
    ax3.set_title('Acceleration Dynamics', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Escape response characteristics
    ax4 = fig.add_subplot(gs[3, 0])
    
    escape_latencies = []
    peak_speeds = []
    
    for window in chase_windows:
        if window['escape_metrics']['escape_detected']:
            escape_latencies.append(window['escape_metrics']['escape_latency_s'])
            peak_speeds.append(window['escape_metrics']['peak_speed'])
    
    if escape_latencies:
        ax4.scatter(escape_latencies, peak_speeds, s=50, alpha=0.6, color='red')
        ax4.set_xlabel('Escape Latency (seconds)')
        ax4.set_ylabel('Peak Speed (pixels/second)')
        ax4.set_title(f'Escape Responses ({len(escape_latencies)}/{n_chases} chases)')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No escape responses detected',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Escape Responses')
    
    # 5. Distance change distribution
    ax5 = fig.add_subplot(gs[3, 1])
    
    distance_changes = []
    for window in enumerate(chase_windows):
        baseline = window[1]['baseline_mean_distance']
        response = window[1]['response_mean_distance']
        if not np.isnan(baseline) and not np.isnan(response):
            distance_changes.append(response - baseline)
    
    if distance_changes:
        ax5.hist(distance_changes, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Distance Change (pixels)')
        ax5.set_ylabel('Count')
        ax5.set_title('Distance Change\n(Response - Baseline)')
        
        mean_change = np.mean(distance_changes)
        ax5.axvline(x=mean_change, color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {mean_change:.1f}')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[3, 2])
    ax6.axis('off')
    
    # Calculate summary stats
    n_escapes = sum(1 for w in chase_windows if w['escape_metrics']['escape_detected'])
    escape_rate = n_escapes / n_chases * 100
    
    mean_baseline = np.nanmean([w['baseline_mean_distance'] for w in chase_windows])
    mean_response = np.nanmean([w['response_mean_distance'] for w in chase_windows])
    
    avg_escape_latency = np.nanmean([w['escape_metrics']['escape_latency_s'] 
                                     for w in chase_windows 
                                     if w['escape_metrics']['escape_detected']])
    
    avg_peak_speed = np.nanmean([w['escape_metrics']['peak_speed'] 
                                 for w in chase_windows])
    
    summary_text = f"""CHASE EVENT SUMMARY
    
Total chases analyzed: {n_chases}
Escape responses: {n_escapes} ({escape_rate:.1f}%)

Distance (pixels):
  Baseline mean: {mean_baseline:.1f}
  Response mean: {mean_response:.1f}
  Change: {mean_response - mean_baseline:+.1f}

Speed dynamics:
  Mean peak speed: {avg_peak_speed:.1f} px/s
  
Escape timing:
  Mean latency: {avg_escape_latency:.3f} s
  (when escape detected)
"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Main title
    fig.suptitle('Chase-by-Chase Behavioral Analysis', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chase analysis plot saved to: {save_path}")
    
    plt.show()
    
    return chase_windows


def create_chase_summary_table(chase_windows: List[Dict]) -> pd.DataFrame:
    """
    Create a summary table of all chase events.
    """
    summary_data = []
    
    for window in chase_windows:
        row = {
            'Chase ID': window['chase_id'],
            'Time (s)': window['chase_time'],
            'Duration (s)': window['chase_duration_s'],
            'Baseline Distance': window['baseline_mean_distance'],
            'Response Distance': window['response_mean_distance'],
            'Distance Change': window['response_mean_distance'] - window['baseline_mean_distance'],
            'Peak Speed': window['escape_metrics']['peak_speed'],
            'Escape Detected': window['escape_metrics']['escape_detected'],
            'Escape Latency': window['escape_metrics'].get('escape_latency_s', np.nan)
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Chase-by-chase event analysis'
    )
    parser.add_argument('zarr_path', help='Path to multi-fish tracker zarr')
    parser.add_argument('h5_path', help='Path to experiment H5 file')
    parser.add_argument('--window-before', type=float, default=2.0,
                       help='Seconds before chase onset to analyze')
    parser.add_argument('--window-after', type=float, default=2.0,
                       help='Seconds after chase onset to analyze')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the analysis plot')
    parser.add_argument('--save-table', type=str, default=None,
                       help='Path to save summary table as CSV')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CHASE-BY-CHASE EVENT ANALYSIS")
    print("=" * 60)
    
    # Load data using the unified analyzer
    analyzer = UnifiedDistanceAnalyzer(
        args.zarr_path,
        args.h5_path,
        verbose=True
    )
    
    # Extract chase windows
    print(f"\nExtracting {args.window_before}s before and {args.window_after}s after chase onsets...")
    chase_windows = extract_chase_windows(
        analyzer,
        window_before_s=args.window_before,
        window_after_s=args.window_after
    )
    
    print(f"Found {len(chase_windows)} analyzable chase events")
    
    if not chase_windows:
        print("No chase events found to analyze")
        return 1
    
    # Create visualization
    plot_chase_by_chase_analysis(chase_windows, save_path=args.output)
    
    # Create and save summary table
    summary_df = create_chase_summary_table(chase_windows)
    
    print("\n" + "=" * 60)
    print("CHASE EVENT SUMMARY TABLE")
    print("=" * 60)
    print(summary_df.to_string())
    
    if args.save_table:
        summary_df.to_csv(args.save_table, index=False)
        print(f"\nSummary table saved to: {args.save_table}")
    
    # Print aggregate statistics
    print("\n" + "=" * 60)
    print("AGGREGATE STATISTICS")
    print("=" * 60)
    
    print(f"\nDistance changes (Response - Baseline):")
    print(f"  Mean: {summary_df['Distance Change'].mean():.1f} pixels")
    print(f"  Median: {summary_df['Distance Change'].median():.1f} pixels")
    print(f"  Positive changes: {(summary_df['Distance Change'] > 0).sum()}/{len(summary_df)}")
    
    print(f"\nEscape responses:")
    print(f"  Detection rate: {summary_df['Escape Detected'].sum()}/{len(summary_df)} "
          f"({summary_df['Escape Detected'].mean()*100:.1f}%)")
    
    if summary_df['Escape Detected'].any():
        escape_df = summary_df[summary_df['Escape Detected']]
        print(f"  Mean latency: {escape_df['Escape Latency'].mean():.3f} seconds")
        print(f"  Mean peak speed: {escape_df['Peak Speed'].mean():.1f} pixels/second")
    
    return 0


if __name__ == '__main__':
    exit(main())