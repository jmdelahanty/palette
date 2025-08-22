#!/usr/bin/env python3
"""
Speed Per Second Calculator and Visualizer

A focused script that calculates and visualizes speed per second from zarr detection data.
Reads bounding box data from zarr files and computes instantaneous and smoothed speeds.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime


def calculate_centroid(bbox):
    """Calculate the centroid of a bounding box [x_min, y_min, x_max, y_max]."""
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def load_best_available_data(zarr_path):
    """
    Load the best available preprocessed data from zarr.
    Priority: preprocessing > filtered_runs > original
    """
    root = zarr.open(zarr_path, mode='r')
    
    # Try preprocessed (interpolated) data first
    if 'preprocessing' in root and 'latest' in root['preprocessing'].attrs:
        latest = root['preprocessing'].attrs['latest']
        data = root['preprocessing'][latest]
        source = f"preprocessing/{latest}"
        print(f"Using preprocessed data: {source}")
    # Try filtered data
    elif 'filtered_runs' in root and 'latest' in root['filtered_runs'].attrs:
        latest = root['filtered_runs'].attrs['latest']
        data = root['filtered_runs'][latest]
        source = f"filtered_runs/{latest}"
        print(f"Using filtered data: {source}")
    # Fall back to original
    else:
        data = root
        source = "original"
        print("Using original data")
    
    return root, data, source


def calculate_speed_per_second(zarr_path, window_size=5, max_speed_threshold=1000.0, units='auto'):
    """
    Calculate speed per second from zarr detection data.
    
    Args:
        zarr_path: Path to zarr file
        window_size: Window size for smoothing (in frames)
        max_speed_threshold: Maximum reasonable speed in pixels/second
        units: Display units ('auto', 'pixels', 'mm', 'cm', 'm')
    
    Returns:
        Dictionary with speed metrics and metadata
    """
    print(f"\nLoading data from: {zarr_path}")
    
    # Load best available data
    root, data, source = load_best_available_data(zarr_path)
    
    # Get metadata
    fps = root.attrs.get('fps', 60.0)
    total_frames = root.attrs.get('frame_count', len(data['n_detections']))
    
    # Get calibration if available
    pixel_to_mm = None
    if 'calibration' in root:
        pixel_to_mm = root['calibration'].attrs.get('pixel_to_mm', None)
    
    # Determine unit conversion based on user preference
    unit_conversions = {
        'pixels': ('pixels/s', 1.0),
        'mm': ('mm/s', pixel_to_mm if pixel_to_mm else None),
        'cm': ('cm/s', pixel_to_mm / 10.0 if pixel_to_mm else None),
        'm': ('m/s', pixel_to_mm / 1000.0 if pixel_to_mm else None)
    }
    
    # Handle 'auto' mode
    if units == 'auto':
        if pixel_to_mm:
            units = 'mm'  # Default to mm if calibration available
        else:
            units = 'pixels'
    
    # Get conversion factor
    if units not in unit_conversions:
        print(f"Warning: Unknown unit '{units}', defaulting to pixels")
        units = 'pixels'
    
    display_unit, conversion_factor = unit_conversions[units]
    
    if conversion_factor is None:
        print(f"Warning: No calibration available for {units}, using pixels instead")
        units = 'pixels'
        display_unit, conversion_factor = unit_conversions['pixels']
    
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Display units: {display_unit}")
    if pixel_to_mm:
        print(f"Calibration: 1 pixel = {pixel_to_mm:.4f} mm")
    
    # Load detection data
    bboxes = data['bboxes'][:]
    n_detections = data['n_detections'][:]
    
    # Extract centroids for frames with detections
    centroids = []
    valid_frames = []
    
    for frame_idx in range(len(n_detections)):
        if n_detections[frame_idx] > 0:
            centroid = calculate_centroid(bboxes[frame_idx, 0])
            centroids.append(centroid)
            valid_frames.append(frame_idx)
    
    if not centroids:
        print("Error: No detections found in data")
        return None
    
    centroids = np.array(centroids)
    valid_frames = np.array(valid_frames)
    
    print(f"Valid frames: {len(valid_frames)}/{total_frames} ({len(valid_frames)/total_frames*100:.1f}%)")
    
    # Calculate frame-to-frame distances
    frame_distances = np.full(total_frames, np.nan)
    
    for i in range(1, len(valid_frames)):
        current_frame = valid_frames[i]
        prev_frame = valid_frames[i-1]
        
        # Only calculate distance if frames are consecutive or close
        if current_frame - prev_frame <= 10:  # Allow small gaps
            dist = np.linalg.norm(centroids[i] - centroids[i-1])
            frame_distances[current_frame] = dist
    
    # Calculate instantaneous speed (pixels per second)
    instantaneous_speed = frame_distances * fps
    
    # Cap unrealistic speeds
    instantaneous_speed = np.where(
        instantaneous_speed > max_speed_threshold,
        np.nan,
        instantaneous_speed
    )
    
    # Calculate smoothed speed
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
    valid_speeds = instantaneous_speed[~np.isnan(instantaneous_speed)]
    
    results = {
        'instantaneous_speed': instantaneous_speed,  # pixels/second
        'smoothed_speed': smoothed_speed,           # pixels/second
        'centroids': centroids,
        'valid_frames': valid_frames,
        'fps': fps,
        'total_frames': total_frames,
        'source': source,
        'pixel_to_mm': pixel_to_mm,
        'units': units,
        'display_unit': display_unit,
        'conversion_factor': conversion_factor,
        'window_size': window_size,
        'statistics': {
            'mean_speed_px_s': np.mean(valid_speeds) if len(valid_speeds) > 0 else 0,
            'median_speed_px_s': np.median(valid_speeds) if len(valid_speeds) > 0 else 0,
            'max_speed_px_s': np.max(valid_speeds) if len(valid_speeds) > 0 else 0,
            'std_speed_px_s': np.std(valid_speeds) if len(valid_speeds) > 0 else 0,
            'percentile_25': np.percentile(valid_speeds, 25) if len(valid_speeds) > 0 else 0,
            'percentile_75': np.percentile(valid_speeds, 75) if len(valid_speeds) > 0 else 0,
        }
    }
    
    # Add converted statistics
    for key in ['mean_speed', 'median_speed', 'max_speed', 'std_speed']:
        base_key = f"{key}_px_s"
        if base_key in results['statistics']:
            converted_key = f"{key}_{units}_s"
            results['statistics'][converted_key] = results['statistics'][base_key] * conversion_factor
    
    # Also convert percentiles
    for key in ['percentile_25', 'percentile_75']:
        if key in results['statistics']:
            converted_key = f"{key}_{units}_s"
            results['statistics'][converted_key] = results['statistics'][key] * conversion_factor
    
    return results


def save_speed_to_zarr(zarr_path, results, overwrite=False):
    """
    Save speed calculation results to zarr with full documentation.
    
    Creates a structured group with all parameters, data, and provenance.
    
    Args:
        zarr_path: Path to zarr file
        results: Results dictionary from calculate_speed_per_second
        overwrite: Whether to overwrite existing speed data
    
    Returns:
        True if successful, False otherwise
    """
    from datetime import datetime
    import json
    
    print("\nSaving speed metrics to zarr...")
    
    # Open zarr in write mode
    root = zarr.open(zarr_path, mode='r+')
    
    # Create speed_metrics group (or version it)
    base_group = 'speed_metrics'
    if base_group not in root:
        root.create_group(base_group)
    
    # Create timestamped version
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_name = f"v_{timestamp}_w{results['window_size']}_u{results['units']}"
    
    # Check if this exact configuration exists
    if version_name in root[base_group] and not overwrite:
        print(f"Warning: {base_group}/{version_name} already exists. Use --overwrite to replace.")
        return False
    
    # Create or overwrite the versioned group
    if version_name in root[base_group]:
        del root[base_group][version_name]
    
    metrics_group = root[base_group].create_group(version_name)
    
    # Save metadata with complete parameters
    metadata = {
        'created_at': datetime.now().isoformat(),
        'source_data': results['source'],  # Which preprocessing version was used
        'source_type': 'preprocessing' if 'preprocessing' in results['source'] else 
                      'filtered' if 'filtered' in results['source'] else 'original',
        'fps': float(results['fps']),
        'total_frames': int(results['total_frames']),
        'valid_frames': int(len(results['valid_frames'])),
        'coverage_percent': float(len(results['valid_frames']) / results['total_frames'] * 100),
        
        # Processing parameters
        'window_size_frames': int(results['window_size']),
        'window_size_ms': float(results['window_size'] / results['fps'] * 1000),
        'units': results['units'],
        'display_unit': results['display_unit'],
        'conversion_factor': float(results['conversion_factor']),
        
        # Calibration info
        'pixel_to_mm': float(results['pixel_to_mm']) if results['pixel_to_mm'] else None,
        
        # Statistics in both pixels and selected units
        'statistics': {
            'pixels': {
                'mean': float(results['statistics']['mean_speed_px_s']),
                'median': float(results['statistics']['median_speed_px_s']),
                'max': float(results['statistics']['max_speed_px_s']),
                'std': float(results['statistics']['std_speed_px_s']),
                'q25': float(results['statistics']['percentile_25']),
                'q75': float(results['statistics']['percentile_75'])
            },
            results['units']: {
                'mean': float(results['statistics'][f'mean_speed_{results["units"]}_s']),
                'median': float(results['statistics'][f'median_speed_{results["units"]}_s']),
                'max': float(results['statistics'][f'max_speed_{results["units"]}_s']),
                'std': float(results['statistics'][f'std_speed_{results["units"]}_s']),
                'q25': float(results['statistics'][f'percentile_25_{results["units"]}_s']),
                'q75': float(results['statistics'][f'percentile_75_{results["units"]}_s'])
            }
        },
        
        # Version info
        'version': version_name,
        'script': 'calculate_speed.py',
        'description': f"Speed calculation with {results['window_size']}-frame smoothing window"
    }
    
    # Store metadata as attributes
    for key, value in metadata.items():
        if isinstance(value, dict):
            metrics_group.attrs[key] = json.dumps(value)
        else:
            metrics_group.attrs[key] = value
    
    # Save arrays
    # Instantaneous speed (always in pixels/second)
    metrics_group.create_dataset(
        'instantaneous_speed_px_s',
        data=results['instantaneous_speed'],
        chunks=True,
        compression='gzip',
        dtype='float32'
    )
    
    # Smoothed speed (always in pixels/second)
    metrics_group.create_dataset(
        'smoothed_speed_px_s',
        data=results['smoothed_speed'],
        chunks=True,
        compression='gzip',
        dtype='float32'
    )
    
    # Valid frame indices
    metrics_group.create_dataset(
        'valid_frames',
        data=results['valid_frames'],
        chunks=True,
        compression='gzip',
        dtype='int32'
    )
    
    # Centroids used for calculation
    metrics_group.create_dataset(
        'centroids',
        data=results['centroids'],
        chunks=True,
        compression='gzip',
        dtype='float32'
    )
    
    # Update the 'latest' pointer to this version
    root[base_group].attrs['latest'] = version_name
    root[base_group].attrs['latest_updated'] = datetime.now().isoformat()
    
    # Add summary of all versions
    versions_summary = {}
    for version_key in root[base_group].keys():
        version_group = root[base_group][version_key]
        versions_summary[version_key] = {
            'created': version_group.attrs.get('created_at', 'unknown'),
            'window_size': version_group.attrs.get('window_size_frames', 'unknown'),
            'units': version_group.attrs.get('units', 'unknown'),
            'source': version_group.attrs.get('source_data', 'unknown')
        }
    
    root[base_group].attrs['versions'] = json.dumps(versions_summary, indent=2)
    
    print(f"  ✓ Speed metrics saved to: {base_group}/{version_name}")
    print(f"  ✓ Source data: {results['source']}")
    print(f"  ✓ Window size: {results['window_size']} frames ({results['window_size']/results['fps']*1000:.0f} ms)")
    print(f"  ✓ Units: {results['display_unit']}")
    print(f"  ✓ Coverage: {len(results['valid_frames'])}/{results['total_frames']} frames ({metadata['coverage_percent']:.1f}%)")
    
    return True


def load_speed_from_zarr(zarr_path, version=None):
    """
    Load previously calculated speed metrics from zarr.
    
    Args:
        zarr_path: Path to zarr file
        version: Specific version to load (default: latest)
    
    Returns:
        Dictionary with speed data and metadata
    """
    import json
    
    root = zarr.open(zarr_path, mode='r')
    
    if 'speed_metrics' not in root:
        print("No speed metrics found in zarr")
        return None
    
    speed_group = root['speed_metrics']
    
    # Get version to load
    if version is None:
        if 'latest' in speed_group.attrs:
            version = speed_group.attrs['latest']
        else:
            versions = list(speed_group.keys())
            if not versions:
                print("No speed metric versions found")
                return None
            version = sorted(versions)[-1]  # Get most recent by name
    
    if version not in speed_group:
        print(f"Version '{version}' not found in speed_metrics")
        print(f"Available versions: {list(speed_group.keys())}")
        return None
    
    metrics = speed_group[version]
    
    # Load data
    result = {
        'instantaneous_speed': metrics['instantaneous_speed_px_s'][:],
        'smoothed_speed': metrics['smoothed_speed_px_s'][:],
        'valid_frames': metrics['valid_frames'][:],
        'centroids': metrics['centroids'][:],
        'version': version
    }
    
    # Load metadata
    for key, value in metrics.attrs.items():
        if key == 'statistics':
            result[key] = json.loads(value)
        else:
            result[key] = value
    
    print(f"Loaded speed metrics: {version}")
    return result
    """
    Create visualization of speed per second.
    
    Args:
        results: Dictionary from calculate_speed_per_second
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    if not results:
        print("No results to visualize")
        return
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Create time axis in seconds
    time_seconds = np.arange(results['total_frames']) / results['fps']
    
    # Use the units from the calculation
    speed_unit = results['display_unit']
    speed_conv = results['conversion_factor']
    
    # Convert speeds to display units
    inst_speed_display = results['instantaneous_speed'] * speed_conv
    smooth_speed_display = results['smoothed_speed'] * speed_conv
    
    # Main title
    title = f"Speed Analysis - {results['source']}"
    if results['pixel_to_mm']:
        title += f" (Units: {speed_unit}, Calibration: 1 px = {results['pixel_to_mm']:.4f} mm)"
    else:
        title += f" (Units: {speed_unit})"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. Speed over time (main plot)
    ax1 = fig.add_subplot(gs[0:2, :])
    valid_mask = ~np.isnan(inst_speed_display)
    
    # Plot instantaneous speed
    ax1.plot(time_seconds[valid_mask], inst_speed_display[valid_mask],
             'b-', alpha=0.3, linewidth=0.5, label='Instantaneous')
    
    # Plot smoothed speed
    ax1.plot(time_seconds[valid_mask], smooth_speed_display[valid_mask],
             'r-', linewidth=2, label=f'Smoothed (window={results["window_size"]} frames)')
    
    # Add mean line
    mean_speed = results['statistics'][f'mean_speed_{results["units"]}_s']
    ax1.axhline(y=mean_speed, color='green', linestyle='--', alpha=0.5,
               label=f'Mean: {mean_speed:.2f} {speed_unit}')
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel(f'Speed ({speed_unit})', fontsize=12)
    ax1.set_title('Speed Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Set reasonable y-limits
    if np.any(valid_mask):
        y_max = np.percentile(inst_speed_display[valid_mask], 99) * 1.1
        ax1.set_ylim([0, y_max])
    
    # 2. Speed distribution histogram
    ax2 = fig.add_subplot(gs[2, 0])
    valid_speeds = inst_speed_display[~np.isnan(inst_speed_display)]
    
    if len(valid_speeds) > 0:
        ax2.hist(valid_speeds, bins=50, alpha=0.7, color='blue', edgecolor='black')
        
        # Add statistical markers
        mean_val = results['statistics'][f'mean_speed_{results["units"]}_s']
        ax2.axvline(x=mean_val, color='green', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.2f}')
        median_val = results['statistics'][f'median_speed_{results["units"]}_s']
        ax2.axvline(x=median_val, color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.2f}')
        
        ax2.set_xlabel(f'Speed ({speed_unit})')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Speed Distribution')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    # 3. Statistics box
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.axis('off')
    
    # Create statistics text
    stats = results['statistics']
    units_key = results['units']
    stats_text = "Speed Statistics\n" + "="*30 + "\n\n"
    
    # Always show values in selected units
    stats_text += f"Mean:     {stats[f'mean_speed_{units_key}_s']:.2f} {speed_unit}\n"
    stats_text += f"Median:   {stats[f'median_speed_{units_key}_s']:.2f} {speed_unit}\n"
    stats_text += f"Maximum:  {stats[f'max_speed_{units_key}_s']:.2f} {speed_unit}\n"
    stats_text += f"Std Dev:  {stats[f'std_speed_{units_key}_s']:.2f} {speed_unit}\n"
    
    stats_text += f"\nQ1 (25%): {stats[f'percentile_25_{units_key}_s']:.2f} {speed_unit}\n"
    stats_text += f"Q3 (75%): {stats[f'percentile_75_{units_key}_s']:.2f} {speed_unit}\n"
    
    # Also show pixels/s in parentheses if using other units
    if units_key != 'pixels':
        stats_text += f"\n(Original: {stats['mean_speed_px_s']:.1f} px/s)"
    
    stats_text += f"\n\nData Coverage: {len(results['valid_frames'])}/{results['total_frames']} frames\n"
    stats_text += f"({len(results['valid_frames'])/results['total_frames']*100:.1f}%)"
    
    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Calculate and visualize speed per second from zarr data')
    parser.add_argument('zarr_path', type=str, help='Path to zarr file')
    parser.add_argument('--units', type=str, default='auto',
                       choices=['auto', 'pixels', 'mm', 'cm', 'm'],
                       help='Display units (default: auto - uses mm if calibrated, otherwise pixels)')
    parser.add_argument('--window-size', type=int, default=5,
                       help='Window size for smoothing (frames, default: 5)')
    parser.add_argument('--max-speed', type=float, default=1000.0,
                       help='Maximum reasonable speed in pixels/second (default: 1000)')
    parser.add_argument('--save', type=str, help='Path to save figure')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot')
    parser.add_argument('--export-csv', type=str, help='Export speed data to CSV file')
    parser.add_argument('--save-to-zarr', action='store_true', 
                       help='Save speed metrics to zarr file with full documentation')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing speed metrics in zarr')
    parser.add_argument('--load-from-zarr', type=str, nargs='?', const='latest',
                       help='Load and visualize previously calculated speed metrics (optionally specify version)')
    
    args = parser.parse_args()
    
    # Validate zarr path
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"Error: Zarr file not found: {zarr_path}")
        return
    
    print("\n" + "="*60)
    print("SPEED PER SECOND CALCULATOR")
    print("="*60)
    
    # Check if loading from zarr
    if args.load_from_zarr:
        print(f"\nLoading speed metrics from zarr...")
        results = load_speed_from_zarr(str(zarr_path), 
                                       version=None if args.load_from_zarr == 'latest' 
                                       else args.load_from_zarr)
        if not results:
            print("Failed to load speed metrics from zarr")
            return
        
        # Reconstruct necessary fields for visualization
        results['display_unit'] = results.get('display_unit', 'pixels/s')
        results['conversion_factor'] = results.get('conversion_factor', 1.0)
        
    else:
        # Calculate new speed metrics
        results = calculate_speed_per_second(
            str(zarr_path),
            window_size=args.window_size,
            max_speed_threshold=args.max_speed,
            units=args.units
        )
        
        if not results:
            print("Failed to calculate speed metrics")
            return
        
        # Save to zarr if requested
        if args.save_to_zarr:
            success = save_speed_to_zarr(str(zarr_path), results, overwrite=args.overwrite)
            if not success and not args.overwrite:
                print("\nTip: Use --overwrite to replace existing metrics")
                print("     or change parameters (window-size, units) for a new version")
    
    # Print summary statistics
    print("\n" + "-"*40)
    print("SPEED STATISTICS")
    print("-"*40)
    stats = results['statistics']
    units_key = results['units']
    display_unit = results['display_unit']
    
    print(f"Mean speed:   {stats[f'mean_speed_{units_key}_s']:.2f} {display_unit}")
    print(f"Median speed: {stats[f'median_speed_{units_key}_s']:.2f} {display_unit}")
    print(f"Max speed:    {stats[f'max_speed_{units_key}_s']:.2f} {display_unit}")
    print(f"Std dev:      {stats[f'std_speed_{units_key}_s']:.2f} {display_unit}")
    
    # Also show in pixels if using other units
    if units_key != 'pixels':
        print(f"\nOriginal values (pixels/second):")
        print(f"Mean speed:   {stats['mean_speed_px_s']:.1f}")
        print(f"Max speed:    {stats['max_speed_px_s']:.1f}")
    
    # Export to CSV if requested
    if args.export_csv:
        import pandas as pd
        
        # Create DataFrame with time and speed data
        time_seconds = np.arange(results['total_frames']) / results['fps']
        df = pd.DataFrame({
            'frame': np.arange(results['total_frames']),
            'time_seconds': time_seconds,
            'instantaneous_speed_px_s': results['instantaneous_speed'],
            'smoothed_speed_px_s': results['smoothed_speed']
        })
        
        # Add converted columns based on selected units
        if results['conversion_factor'] != 1.0:
            unit_suffix = results['units']
            df[f'instantaneous_speed_{unit_suffix}_s'] = df['instantaneous_speed_px_s'] * results['conversion_factor']
            df[f'smoothed_speed_{unit_suffix}_s'] = df['smoothed_speed_px_s'] * results['conversion_factor']
        
        df.to_csv(args.export_csv, index=False)
        print(f"\nSpeed data exported to: {args.export_csv}")
    
    # Visualize
    if not args.no_show or args.save:
        print("\nGenerating visualization...")
        visualize_speed(results, save_path=args.save, show=not args.no_show)
    
    print("\nDone!")


if __name__ == '__main__':
    main()