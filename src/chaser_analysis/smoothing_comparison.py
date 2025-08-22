#!/usr/bin/env python3
"""
Speed Smoothing Comparison Tool

Helps determine the optimal smoothing window size by comparing multiple options.
Visualizes how different smoothing factors affect speed calculations.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import argparse
from pathlib import Path
import pandas as pd


def calculate_centroid(bbox):
    """Calculate the centroid of a bounding box [x_min, y_min, x_max, y_max]."""
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def load_best_available_data(zarr_path):
    """Load the best available preprocessed data from zarr."""
    root = zarr.open(zarr_path, mode='r')
    
    if 'preprocessing' in root and 'latest' in root['preprocessing'].attrs:
        latest = root['preprocessing'].attrs['latest']
        data = root['preprocessing'][latest]
        source = f"preprocessing/{latest}"
    elif 'filtered_runs' in root and 'latest' in root['filtered_runs'].attrs:
        latest = root['filtered_runs'].attrs['latest']
        data = root['filtered_runs'][latest]
        source = f"filtered_runs/{latest}"
    else:
        data = root
        source = "original"
    
    return root, data, source


def calculate_speed_with_smoothing(zarr_path, window_sizes, max_speed_threshold=1000.0, units='auto'):
    """
    Calculate speed with multiple smoothing window sizes for comparison.
    
    Args:
        zarr_path: Path to zarr file
        window_sizes: List of window sizes to compare
        max_speed_threshold: Maximum reasonable speed in pixels/second
        units: Display units
    
    Returns:
        Dictionary with results for each window size
    """
    print(f"\nLoading data from: {zarr_path}")
    
    # Load data
    root, data, source = load_best_available_data(zarr_path)
    
    # Get metadata
    fps = root.attrs.get('fps', 60.0)
    total_frames = root.attrs.get('frame_count', len(data['n_detections']))
    
    # Get calibration
    pixel_to_mm = None
    if 'calibration' in root:
        pixel_to_mm = root['calibration'].attrs.get('pixel_to_mm', None)
    
    # Determine units
    unit_conversions = {
        'pixels': ('pixels/s', 1.0),
        'mm': ('mm/s', pixel_to_mm if pixel_to_mm else None),
        'cm': ('cm/s', pixel_to_mm / 10.0 if pixel_to_mm else None),
        'm': ('m/s', pixel_to_mm / 1000.0 if pixel_to_mm else None)
    }
    
    if units == 'auto':
        units = 'mm' if pixel_to_mm else 'pixels'
    
    display_unit, conversion_factor = unit_conversions.get(units, ('pixels/s', 1.0))
    if conversion_factor is None:
        units = 'pixels'
        display_unit, conversion_factor = unit_conversions['pixels']
    
    print(f"FPS: {fps}, Units: {display_unit}")
    if pixel_to_mm:
        print(f"Calibration: 1 pixel = {pixel_to_mm:.4f} mm")
    
    # Load detection data
    bboxes = data['bboxes'][:]
    n_detections = data['n_detections'][:]
    
    # Extract centroids
    centroids = []
    valid_frames = []
    
    for frame_idx in range(len(n_detections)):
        if n_detections[frame_idx] > 0:
            centroid = calculate_centroid(bboxes[frame_idx, 0])
            centroids.append(centroid)
            valid_frames.append(frame_idx)
    
    if not centroids:
        print("Error: No detections found")
        return None
    
    centroids = np.array(centroids)
    valid_frames = np.array(valid_frames)
    
    # Calculate frame-to-frame distances
    frame_distances = np.full(total_frames, np.nan)
    
    for i in range(1, len(valid_frames)):
        current_frame = valid_frames[i]
        prev_frame = valid_frames[i-1]
        
        if current_frame - prev_frame <= 10:
            dist = np.linalg.norm(centroids[i] - centroids[i-1])
            frame_distances[current_frame] = dist
    
    # Calculate instantaneous speed
    instantaneous_speed = frame_distances * fps
    instantaneous_speed = np.where(
        instantaneous_speed > max_speed_threshold,
        np.nan,
        instantaneous_speed
    )
    
    # Calculate smoothed speeds for each window size
    results = {
        'instantaneous_speed': instantaneous_speed,
        'fps': fps,
        'total_frames': total_frames,
        'valid_frames': valid_frames,
        'centroids': centroids,
        'display_unit': display_unit,
        'conversion_factor': conversion_factor,
        'source': source,
        'smoothed_speeds': {}
    }
    
    for window_size in window_sizes:
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
        else:
            smoothed_speed = instantaneous_speed.copy()
        
        # Calculate statistics for this smoothing
        valid_smooth = smoothed_speed[~np.isnan(smoothed_speed)]
        
        results['smoothed_speeds'][window_size] = {
            'data': smoothed_speed,
            'mean': np.mean(valid_smooth) if len(valid_smooth) > 0 else 0,
            'std': np.std(valid_smooth) if len(valid_smooth) > 0 else 0,
            'max': np.max(valid_smooth) if len(valid_smooth) > 0 else 0,
            'noise_estimate': estimate_noise_level(smoothed_speed)
        }
    
    return results


def estimate_noise_level(speed_data):
    """
    Estimate noise level in speed data using median absolute deviation.
    Lower values indicate smoother data.
    """
    valid_data = speed_data[~np.isnan(speed_data)]
    if len(valid_data) < 2:
        return np.nan
    
    # Calculate differences between consecutive points
    diffs = np.diff(valid_data)
    
    # Use median absolute deviation as noise estimate
    mad = np.median(np.abs(diffs - np.median(diffs)))
    return mad


def analyze_smoothing_quality(results, time_window_seconds=5.0):
    """
    Analyze the quality of different smoothing window sizes.
    
    Returns metrics to help choose optimal smoothing.
    """
    fps = results['fps']
    window_frames = int(time_window_seconds * fps)
    
    analysis = {}
    
    for window_size, smooth_data in results['smoothed_speeds'].items():
        speed = smooth_data['data']
        valid_speed = speed[~np.isnan(speed)]
        
        if len(valid_speed) < 2:
            continue
        
        # Calculate various metrics
        metrics = {
            'window_size': window_size,
            'window_ms': (window_size / fps) * 1000,
            'mean_speed': smooth_data['mean'],
            'std_speed': smooth_data['std'],
            'cv': smooth_data['std'] / smooth_data['mean'] if smooth_data['mean'] > 0 else np.nan,
            'noise_level': smooth_data['noise_estimate'],
            'max_speed': smooth_data['max'],
            'data_preservation': np.sum(~np.isnan(speed)) / len(speed)
        }
        
        # Calculate autocorrelation to measure smoothness
        if len(valid_speed) > 10:
            autocorr = np.correlate(valid_speed - np.mean(valid_speed), 
                                   valid_speed - np.mean(valid_speed), mode='same')
            autocorr = autocorr / np.max(autocorr)
            # Measure how quickly autocorrelation drops (smoother = slower drop)
            metrics['smoothness'] = np.mean(autocorr[len(autocorr)//2:len(autocorr)//2+5])
        else:
            metrics['smoothness'] = np.nan
        
        analysis[window_size] = metrics
    
    return pd.DataFrame(analysis).T


def visualize_smoothing_comparison(results, save_path=None, show=True):
    """
    Create comprehensive visualization comparing different smoothing window sizes.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
    
    # Time axis
    time_seconds = np.arange(results['total_frames']) / results['fps']
    conv = results['conversion_factor']
    unit = results['display_unit']
    
    # Get window sizes
    window_sizes = sorted(results['smoothed_speeds'].keys())
    
    # Color map for different window sizes
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(window_sizes)))
    
    # Title
    fig.suptitle(f'Smoothing Window Size Comparison - {results["source"]}\n'
                 f'Comparing window sizes: {window_sizes} frames', 
                 fontsize=14, fontweight='bold')
    
    # 1. All smoothed speeds overlaid (top row, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Plot instantaneous speed
    inst_speed = results['instantaneous_speed'] * conv
    valid_mask = ~np.isnan(inst_speed)
    ax1.plot(time_seconds[valid_mask], inst_speed[valid_mask],
             'k-', alpha=0.2, linewidth=0.5, label='Raw')
    
    # Plot each smoothed version
    for i, window_size in enumerate(window_sizes):
        smooth_speed = results['smoothed_speeds'][window_size]['data'] * conv
        valid_mask = ~np.isnan(smooth_speed)
        ax1.plot(time_seconds[valid_mask], smooth_speed[valid_mask],
                color=colors[i], linewidth=1.5, alpha=0.8,
                label=f'{window_size} frames ({window_size/results["fps"]*1000:.0f}ms)')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel(f'Speed ({unit})')
    ax1.set_title('Speed Over Time - All Window Sizes')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Set reasonable y-limits
    all_speeds = []
    for smooth_data in results['smoothed_speeds'].values():
        valid = smooth_data['data'][~np.isnan(smooth_data['data'])]
        if len(valid) > 0:
            all_speeds.extend(valid)
    if all_speeds:
        y_max = np.percentile(all_speeds, 99) * conv * 1.1
        ax1.set_ylim([0, y_max])
    
    # 2. Zoom-in comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Find a representative 10-second window with activity
    window_size_sec = 10
    window_frames = int(window_size_sec * results['fps'])
    
    # Find window with highest variance (most interesting)
    best_start = 0
    best_var = 0
    for start in range(0, len(inst_speed) - window_frames, window_frames//2):
        window_data = inst_speed[start:start+window_frames]
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) > window_frames * 0.5:  # At least 50% valid data
            var = np.var(valid_data)
            if var > best_var:
                best_var = var
                best_start = start
    
    zoom_end = min(best_start + window_frames, len(time_seconds))
    zoom_time = time_seconds[best_start:zoom_end]
    
    # Plot zoomed section
    zoom_inst = inst_speed[best_start:zoom_end]
    valid_mask = ~np.isnan(zoom_inst)
    ax2.plot(zoom_time[valid_mask], zoom_inst[valid_mask],
             'k-', alpha=0.3, linewidth=0.5, label='Raw')
    
    for i, ws in enumerate(window_sizes[:4]):  # Show only first 4 for clarity
        smooth_speed = results['smoothed_speeds'][ws]['data'][best_start:zoom_end] * conv
        valid_mask = ~np.isnan(smooth_speed)
        ax2.plot(zoom_time[valid_mask], smooth_speed[valid_mask],
                color=colors[i], linewidth=2, alpha=0.9,
                label=f'{ws} frames')
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel(f'Speed ({unit})')
    ax2.set_title(f'Zoomed View ({window_size_sec}s window)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Noise level comparison (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    noise_levels = [results['smoothed_speeds'][ws]['noise_estimate'] * conv 
                   for ws in window_sizes]
    ax3.bar(range(len(window_sizes)), noise_levels, color=colors)
    ax3.set_xticks(range(len(window_sizes)))
    ax3.set_xticklabels([f'{ws}' for ws in window_sizes])
    ax3.set_xlabel('Window Size (frames)')
    ax3.set_ylabel(f'Noise Level ({unit})')
    ax3.set_title('Noise Estimate (lower is smoother)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Standard deviation comparison (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    
    stds = [results['smoothed_speeds'][ws]['std'] * conv for ws in window_sizes]
    means = [results['smoothed_speeds'][ws]['mean'] * conv for ws in window_sizes]
    
    ax4.plot(window_sizes, stds, 'b-o', label='Std Dev', markersize=8)
    ax4.set_xlabel('Window Size (frames)')
    ax4.set_ylabel(f'Std Dev ({unit})', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4.grid(True, alpha=0.3)
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(window_sizes, means, 'r-s', label='Mean', markersize=8)
    ax4_twin.set_ylabel(f'Mean Speed ({unit})', color='r')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    
    ax4.set_title('Mean and Std Dev vs Window Size')
    
    # 5. Coefficient of Variation (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    
    cvs = [stds[i]/means[i] if means[i] > 0 else 0 for i in range(len(window_sizes))]
    ax5.plot(window_sizes, cvs, 'g-^', markersize=8)
    ax5.set_xlabel('Window Size (frames)')
    ax5.set_ylabel('Coefficient of Variation')
    ax5.set_title('CV (Std/Mean) - Relative Variability')
    ax5.grid(True, alpha=0.3)
    
    # 6. Speed distribution comparison (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    
    # Show distribution for selected window sizes
    selected_ws = [window_sizes[0], window_sizes[len(window_sizes)//2], window_sizes[-1]]
    for ws in selected_ws:
        speed_data = results['smoothed_speeds'][ws]['data'] * conv
        valid_data = speed_data[~np.isnan(speed_data)]
        ax6.hist(valid_data, bins=30, alpha=0.5, label=f'{ws} frames', density=True)
    
    ax6.set_xlabel(f'Speed ({unit})')
    ax6.set_ylabel('Probability Density')
    ax6.set_title('Speed Distribution Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Autocorrelation decay (bottom center)
    ax7 = fig.add_subplot(gs[2, 1])
    
    for i, ws in enumerate([window_sizes[0], window_sizes[-1]]):
        speed_data = results['smoothed_speeds'][ws]['data']
        valid_data = speed_data[~np.isnan(speed_data)]
        
        if len(valid_data) > 50:
            # Calculate autocorrelation
            lags = np.arange(0, min(30, len(valid_data)//2))
            autocorr = [1.0]
            for lag in lags[1:]:
                c = np.corrcoef(valid_data[:-lag], valid_data[lag:])[0, 1]
                autocorr.append(c if not np.isnan(c) else 0)
            
            lag_times = lags / results['fps']
            ax7.plot(lag_times, autocorr, 'o-', 
                    label=f'{ws} frames', 
                    color=colors[i*(-1)], markersize=4)
    
    ax7.set_xlabel('Lag (seconds)')
    ax7.set_ylabel('Autocorrelation')
    ax7.set_title('Temporal Correlation Structure')
    ax7.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Recommendation box (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Generate recommendation
    analysis_df = analyze_smoothing_quality(results)
    
    # Find optimal based on noise reduction vs signal preservation
    if not analysis_df.empty:
        # Normalize metrics
        norm_noise = 1 - (analysis_df['noise_level'] - analysis_df['noise_level'].min()) / \
                     (analysis_df['noise_level'].max() - analysis_df['noise_level'].min())
        norm_cv = 1 - (analysis_df['cv'] - analysis_df['cv'].min()) / \
                  (analysis_df['cv'].max() - analysis_df['cv'].min())
        
        # Combined score (you can adjust weights)
        score = 0.5 * norm_noise + 0.5 * norm_cv
        optimal_idx = score.idxmax()
        optimal_window = int(optimal_idx)
        
        rec_text = "Smoothing Recommendations\n" + "="*30 + "\n\n"
        rec_text += f"Optimal Window: {optimal_window} frames\n"
        rec_text += f"({optimal_window/results['fps']*1000:.0f} ms)\n\n"
        
        rec_text += "Guidelines:\n"
        rec_text += f"• Behavioral events: 3-7 frames\n"
        rec_text += f"• General movement: 5-15 frames\n"
        rec_text += f"• Long-term trends: 15-30 frames\n\n"
        
        rec_text += f"Your data:\n"
        rec_text += f"• FPS: {results['fps']}\n"
        rec_text += f"• Noise level: {'High' if analysis_df['noise_level'].mean() > 10 else 'Low'}\n"
        rec_text += f"• Suggested: {optimal_window} frames"
    else:
        rec_text = "Unable to generate recommendation"
    
    ax8.text(0.1, 0.9, rec_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig, analysis_df


def main():
    parser = argparse.ArgumentParser(
        description='Compare different smoothing window sizes for speed calculation')
    parser.add_argument('zarr_path', type=str, help='Path to zarr file')
    parser.add_argument('--window-sizes', type=int, nargs='+', 
                       default=[1, 3, 5, 7, 10, 15, 20, 30],
                       help='Window sizes to compare (default: 1 3 5 7 10 15 20 30)')
    parser.add_argument('--units', type=str, default='auto',
                       choices=['auto', 'pixels', 'mm', 'cm', 'm'],
                       help='Display units (default: auto)')
    parser.add_argument('--max-speed', type=float, default=1000.0,
                       help='Maximum reasonable speed in pixels/second')
    parser.add_argument('--save', type=str, help='Path to save figure')
    parser.add_argument('--export-analysis', type=str, 
                       help='Export analysis metrics to CSV')
    
    args = parser.parse_args()
    
    # Validate path
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"Error: Zarr file not found: {zarr_path}")
        return
    
    print("\n" + "="*60)
    print("SMOOTHING WINDOW COMPARISON")
    print("="*60)
    
    # Calculate speeds with different smoothing
    results = calculate_speed_with_smoothing(
        str(zarr_path),
        args.window_sizes,
        args.max_speed,
        args.units
    )
    
    if not results:
        print("Failed to calculate speeds")
        return
    
    # Analyze smoothing quality
    print("\nAnalyzing smoothing quality...")
    analysis_df = analyze_smoothing_quality(results)
    
    # Print analysis table
    print("\n" + "-"*60)
    print("SMOOTHING ANALYSIS METRICS")
    print("-"*60)
    
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(analysis_df[['window_ms', 'mean_speed', 'std_speed', 'cv', 
                       'noise_level', 'data_preservation']])
    
    # Find and report optimal
    if not analysis_df.empty:
        # Simple optimization: minimize noise while keeping CV reasonable
        norm_noise = (analysis_df['noise_level'] - analysis_df['noise_level'].min()) / \
                    (analysis_df['noise_level'].max() - analysis_df['noise_level'].min())
        norm_cv = (analysis_df['cv'] - analysis_df['cv'].min()) / \
                 (analysis_df['cv'].max() - analysis_df['cv'].min())
        
        score = norm_noise + 0.5 * norm_cv  # Weight noise reduction more
        optimal_idx = score.idxmin()
        
        print(f"\n" + "="*60)
        print(f"RECOMMENDATION: Use window size of {int(optimal_idx)} frames")
        print(f"This provides good noise reduction while preserving signal features")
        print(f"="*60)
    
    # Export analysis if requested
    if args.export_analysis:
        analysis_df.to_csv(args.export_analysis)
        print(f"\nAnalysis exported to: {args.export_analysis}")
    
    # Visualize
    print("\nGenerating visualization...")
    fig, _ = visualize_smoothing_comparison(results, save_path=args.save)
    
    print("\nDone!")


if __name__ == '__main__':
    main()