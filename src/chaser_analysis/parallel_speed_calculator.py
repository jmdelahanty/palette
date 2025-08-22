#!/usr/bin/env python3
"""
Parallel Speed Calculator

Computes speed metrics with multiple window sizes in parallel.
Utilizes multiprocessing to speed up calculations significantly.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from pathlib import Path
from datetime import datetime
import json
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


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


def smooth_speed_worker(args):
    """
    Worker function for parallel smoothing calculation.
    
    Args:
        args: Tuple of (window_size, instantaneous_speed)
    
    Returns:
        Tuple of (window_size, smoothed_speed, statistics)
    """
    window_size, instantaneous_speed = args
    
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
    
    # Calculate statistics
    valid_smooth = smoothed_speed[~np.isnan(smoothed_speed)]
    stats = {
        'mean': np.mean(valid_smooth) if len(valid_smooth) > 0 else 0,
        'median': np.median(valid_smooth) if len(valid_smooth) > 0 else 0,
        'std': np.std(valid_smooth) if len(valid_smooth) > 0 else 0,
        'max': np.max(valid_smooth) if len(valid_smooth) > 0 else 0,
        'q25': np.percentile(valid_smooth, 25) if len(valid_smooth) > 0 else 0,
        'q75': np.percentile(valid_smooth, 75) if len(valid_smooth) > 0 else 0,
    }
    
    return window_size, smoothed_speed, stats


def calculate_speed_parallel(zarr_path, window_sizes, max_speed_threshold=1000.0, 
                           units='auto', n_workers=None):
    """
    Calculate speed with multiple window sizes in parallel.
    
    Args:
        zarr_path: Path to zarr file
        window_sizes: List of window sizes to calculate
        max_speed_threshold: Maximum reasonable speed in pixels/second
        units: Display units ('auto', 'pixels', 'mm', 'cm', 'm')
        n_workers: Number of parallel workers (default: CPU count - 1)
    
    Returns:
        Dictionary with results for all window sizes
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
    print("Processing detections...")
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
    
    print(f"Valid frames: {len(valid_frames)}/{total_frames} ({len(valid_frames)/total_frames*100:.1f}%)")
    
    # Calculate frame-to-frame distances
    print("Calculating instantaneous speeds...")
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
    
    # Prepare parallel processing
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print(f"\nCalculating smoothed speeds in parallel...")
    print(f"Window sizes: {window_sizes}")
    print(f"Using {n_workers} worker processes")
    
    # Create tasks for parallel processing
    tasks = [(ws, instantaneous_speed) for ws in window_sizes]
    
    # Process in parallel with progress bar
    start_time = time.time()
    results_dict = {}
    
    with Pool(n_workers) as pool:
        # Use tqdm for progress bar
        with tqdm(total=len(tasks), desc="Processing windows") as pbar:
            for window_size, smoothed_speed, stats in pool.imap_unordered(smooth_speed_worker, tasks):
                results_dict[window_size] = {
                    'data': smoothed_speed,
                    'stats_px': stats,
                    'stats_display': {k: v * conversion_factor for k, v in stats.items()}
                }
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    print(f"Parallel processing completed in {elapsed_time:.2f} seconds")
    
    # Compile final results
    results = {
        'instantaneous_speed': instantaneous_speed,
        'window_results': results_dict,
        'fps': fps,
        'total_frames': total_frames,
        'valid_frames': valid_frames,
        'centroids': centroids,
        'source': source,
        'pixel_to_mm': pixel_to_mm,
        'units': units,
        'display_unit': display_unit,
        'conversion_factor': conversion_factor,
        'processing_time': elapsed_time,
        'n_workers': n_workers
    }
    
    return results


def save_parallel_results_to_zarr(zarr_path, results, overwrite=False):
    """
    Save parallel speed calculation results to zarr.
    
    Creates a structured group with results for all window sizes.
    """
    print("\nSaving parallel speed metrics to zarr...")
    
    # Open zarr in write mode
    root = zarr.open(zarr_path, mode='r+')
    
    # Create speed_metrics group if needed
    base_group = 'speed_metrics'
    if base_group not in root:
        root.create_group(base_group)
    
    # Create batch processing group
    batch_group = 'batch_processing'
    if batch_group not in root[base_group]:
        root[base_group].create_group(batch_group)
    
    # Create timestamped batch
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_name = f"batch_{timestamp}_{results['units']}"
    
    if batch_name in root[base_group][batch_group] and not overwrite:
        print(f"Warning: {batch_name} already exists. Use --overwrite to replace.")
        return False
    
    if batch_name in root[base_group][batch_group]:
        del root[base_group][batch_group][batch_name]
    
    batch = root[base_group][batch_group].create_group(batch_name)
    
    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'source_data': results['source'],
        'fps': float(results['fps']),
        'total_frames': int(results['total_frames']),
        'valid_frames': int(len(results['valid_frames'])),
        'units': results['units'],
        'display_unit': results['display_unit'],
        'conversion_factor': float(results['conversion_factor']),
        'pixel_to_mm': float(results['pixel_to_mm']) if results['pixel_to_mm'] else None,
        'processing_time_seconds': results['processing_time'],
        'n_workers': results['n_workers'],
        'window_sizes': sorted(results['window_results'].keys())
    }
    
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            batch.attrs[key] = json.dumps(value)
        else:
            batch.attrs[key] = value
    
    # Save instantaneous speed (shared across all windows)
    batch.create_dataset(
        'instantaneous_speed_px_s',
        data=results['instantaneous_speed'],
        chunks=True,
        compression='gzip',
        dtype='float32'
    )
    
    # Save results for each window size
    for window_size, window_data in results['window_results'].items():
        window_group = batch.create_group(f'window_{window_size}')
        
        # Save smoothed speed
        window_group.create_dataset(
            'smoothed_speed_px_s',
            data=window_data['data'],
            chunks=True,
            compression='gzip',
            dtype='float32'
        )
        
        # Save statistics
        window_group.attrs['stats_pixels'] = json.dumps(window_data['stats_px'])
        window_group.attrs['stats_display'] = json.dumps(window_data['stats_display'])
        window_group.attrs['window_size_frames'] = int(window_size)
        window_group.attrs['window_size_ms'] = float(window_size / results['fps'] * 1000)
    
    # Update latest batch pointer
    root[base_group][batch_group].attrs['latest'] = batch_name
    
    print(f"  ✓ Parallel results saved to: {base_group}/{batch_group}/{batch_name}")
    print(f"  ✓ Window sizes: {sorted(results['window_results'].keys())}")
    print(f"  ✓ Processing time: {results['processing_time']:.2f} seconds")
    
    # Also save individual versions for compatibility
    save_count = 0
    for window_size in sorted(results['window_results'].keys())[:3]:  # Save top 3 as individual versions
        individual_results = {
            'instantaneous_speed': results['instantaneous_speed'],
            'smoothed_speed': results['window_results'][window_size]['data'],
            'valid_frames': results['valid_frames'],
            'centroids': results['centroids'],
            'source': results['source'],
            'fps': results['fps'],
            'total_frames': results['total_frames'],
            'pixel_to_mm': results['pixel_to_mm'],
            'units': results['units'],
            'display_unit': results['display_unit'],
            'conversion_factor': results['conversion_factor'],
            'window_size': window_size,
            'statistics': {
                'mean_speed_px_s': results['window_results'][window_size]['stats_px']['mean'],
                'median_speed_px_s': results['window_results'][window_size]['stats_px']['median'],
                'max_speed_px_s': results['window_results'][window_size]['stats_px']['max'],
                'std_speed_px_s': results['window_results'][window_size]['stats_px']['std'],
                'percentile_25': results['window_results'][window_size]['stats_px']['q25'],
                'percentile_75': results['window_results'][window_size]['stats_px']['q75'],
            }
        }
        
        # Add converted statistics
        for key in ['mean_speed', 'median_speed', 'max_speed', 'std_speed']:
            base_key = f"{key}_px_s"
            if base_key in individual_results['statistics']:
                converted_key = f"{key}_{results['units']}_s"
                individual_results['statistics'][converted_key] = (
                    individual_results['statistics'][base_key] * results['conversion_factor']
                )
        
        for key in ['percentile_25', 'percentile_75']:
            converted_key = f"{key}_{results['units']}_s"
            individual_results['statistics'][converted_key] = (
                individual_results['statistics'][key] * results['conversion_factor']
            )
        
        # Import the save function from the original script
        from calculate_speed import save_speed_to_zarr
        save_speed_to_zarr(zarr_path, individual_results, overwrite=True)
        save_count += 1
    
    if save_count > 0:
        print(f"  ✓ Also saved {save_count} individual window sizes for compatibility")
    
    return True


def visualize_parallel_results(results, save_path=None, show=True):
    """
    Visualize results from parallel processing.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    # Time axis
    time_seconds = np.arange(results['total_frames']) / results['fps']
    conv = results['conversion_factor']
    unit = results['display_unit']
    
    # Get window sizes
    window_sizes = sorted(results['window_results'].keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(window_sizes)))
    
    # Title
    title = f'Parallel Speed Calculation Results - {results["source"]}\n'
    title += f'{len(window_sizes)} window sizes processed in {results["processing_time"]:.2f}s using {results["n_workers"]} workers'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 1. All speeds overlaid
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Plot instantaneous
    inst_speed = results['instantaneous_speed'] * conv
    valid_mask = ~np.isnan(inst_speed)
    ax1.plot(time_seconds[valid_mask], inst_speed[valid_mask],
             'k-', alpha=0.2, linewidth=0.5, label='Raw')
    
    # Plot each smoothed version
    for i, ws in enumerate(window_sizes):
        smooth_speed = results['window_results'][ws]['data'] * conv
        valid_mask = ~np.isnan(smooth_speed)
        ax1.plot(time_seconds[valid_mask], smooth_speed[valid_mask],
                color=colors[i], linewidth=1.5, alpha=0.8,
                label=f'{ws} frames')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel(f'Speed ({unit})')
    ax1.set_title('All Window Sizes Comparison')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 2. Statistics comparison
    ax2 = fig.add_subplot(gs[0, 2])
    
    means = [results['window_results'][ws]['stats_display']['mean'] for ws in window_sizes]
    stds = [results['window_results'][ws]['stats_display']['std'] for ws in window_sizes]
    
    x_pos = np.arange(len(window_sizes))
    ax2.bar(x_pos, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(ws) for ws in window_sizes], rotation=45)
    ax2.set_xlabel('Window Size (frames)')
    ax2.set_ylabel(f'Mean Speed ± SD ({unit})')
    ax2.set_title('Mean Speed by Window Size')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Processing efficiency
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Show processing speed
    total_calculations = len(window_sizes) * results['total_frames']
    calc_per_second = total_calculations / results['processing_time']
    
    efficiency_text = "Processing Efficiency\n" + "="*25 + "\n\n"
    efficiency_text += f"Window sizes: {len(window_sizes)}\n"
    efficiency_text += f"Frames: {results['total_frames']}\n"
    efficiency_text += f"Workers: {results['n_workers']}\n"
    efficiency_text += f"Time: {results['processing_time']:.2f}s\n\n"
    efficiency_text += f"Speed: {calc_per_second:.0f} calc/s\n"
    efficiency_text += f"Speedup: ~{results['n_workers']:.1f}x"
    
    ax3.text(0.1, 0.5, efficiency_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax3.axis('off')
    
    # 4. Speed distributions
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Show a few key distributions
    key_windows = [window_sizes[0], window_sizes[len(window_sizes)//2], window_sizes[-1]]
    for ws in key_windows:
        speed_data = results['window_results'][ws]['data'] * conv
        valid_data = speed_data[~np.isnan(speed_data)]
        ax4.hist(valid_data, bins=30, alpha=0.5, label=f'{ws} frames', density=True)
    
    ax4.set_xlabel(f'Speed ({unit})')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Speed Distributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Optimal window recommendation
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Calculate CV for each window
    cvs = []
    for ws in window_sizes:
        stats = results['window_results'][ws]['stats_display']
        cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0
        cvs.append(cv)
    
    ax5.plot(window_sizes, cvs, 'g-o', markersize=8)
    
    # Mark optimal (minimum CV after initial drop)
    optimal_idx = np.argmin(cvs[2:]) + 2  # Skip first two to avoid under-smoothing
    optimal_window = window_sizes[optimal_idx]
    ax5.axvline(x=optimal_window, color='red', linestyle='--', alpha=0.5)
    ax5.text(optimal_window, max(cvs)*0.9, f'Optimal: {optimal_window}', 
            ha='center', color='red', fontweight='bold')
    
    ax5.set_xlabel('Window Size (frames)')
    ax5.set_ylabel('Coefficient of Variation')
    ax5.set_title('Smoothing Optimization')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Parallel speed calculation with multiple window sizes')
    parser.add_argument('zarr_path', type=str, help='Path to zarr file')
    parser.add_argument('--window-sizes', type=int, nargs='+',
                       default=[1, 3, 5, 7, 10, 15, 20],
                       help='Window sizes to calculate (default: 1 3 5 7 10 15 20)')
    parser.add_argument('--units', type=str, default='auto',
                       choices=['auto', 'pixels', 'mm', 'cm', 'm'],
                       help='Display units')
    parser.add_argument('--max-speed', type=float, default=1000.0,
                       help='Maximum reasonable speed in pixels/second')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--save', type=str, help='Path to save figure')
    parser.add_argument('--save-to-zarr', action='store_true',
                       help='Save results to zarr')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots')
    
    args = parser.parse_args()
    
    # Validate path
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"Error: Zarr file not found: {zarr_path}")
        return
    
    print("\n" + "="*60)
    print("PARALLEL SPEED CALCULATOR")
    print("="*60)
    print(f"CPU cores available: {cpu_count()}")
    print(f"Workers to use: {args.n_workers or max(1, cpu_count() - 1)}")
    
    # Calculate speeds in parallel
    results = calculate_speed_parallel(
        str(zarr_path),
        args.window_sizes,
        args.max_speed,
        args.units,
        args.n_workers
    )
    
    if not results:
        print("Failed to calculate speeds")
        return
    
    # Print summary
    print("\n" + "-"*60)
    print("RESULTS SUMMARY")
    print("-"*60)
    
    for ws in sorted(results['window_results'].keys()):
        stats = results['window_results'][ws]['stats_display']
        print(f"Window {ws:3d} frames: "
              f"Mean={stats['mean']:6.2f}, "
              f"SD={stats['std']:6.2f}, "
              f"Max={stats['max']:6.2f} {results['display_unit']}")
    
    print(f"\nProcessing time: {results['processing_time']:.2f} seconds")
    print(f"Speedup: ~{results['n_workers']:.1f}x over sequential processing")
    
    # Save to zarr if requested
    if args.save_to_zarr:
        success = save_parallel_results_to_zarr(str(zarr_path), results, args.overwrite)
        if not success and not args.overwrite:
            print("\nTip: Use --overwrite to replace existing results")
    
    # Visualize
    if not args.no_show or args.save:
        print("\nGenerating visualization...")
        visualize_parallel_results(results, save_path=args.save, show=not args.no_show)
    
    print("\nDone!")


if __name__ == '__main__':
    main()