#!/usr/bin/env python3
"""
Preprocessing Pipeline Visualizer

Visualizes the results of the preprocessing pipeline:
- Original detections
- After jump removal (filtered_runs)
- After gap filling (preprocessing)

Shows trajectories, coverage, and statistics for comparison.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from pathlib import Path
from datetime import datetime


def calculate_centroid(bbox):
    """Calculate the centroid of a bounding box."""
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def load_detection_data(group):
    """Load detection data from a zarr group."""
    bboxes = group['bboxes'][:]
    scores = group['scores'][:]
    n_detections = group['n_detections'][:]
    
    # Calculate centroids
    centroids = []
    valid_frames = []
    
    for frame_idx in range(len(n_detections)):
        if n_detections[frame_idx] > 0:
            centroid = calculate_centroid(bboxes[frame_idx, 0])
            centroids.append(centroid)
            valid_frames.append(frame_idx)
    
    if centroids:
        centroids = np.array(centroids)
    else:
        centroids = np.array([]).reshape(0, 2)
    
    return {
        'bboxes': bboxes,
        'scores': scores,
        'n_detections': n_detections,
        'centroids': centroids,
        'valid_frames': np.array(valid_frames),
        'coverage': np.sum(n_detections > 0) / len(n_detections)
    }


def visualize_preprocessing_pipeline(zarr_path, frame_range=None, save_path=None, 
                                    preprocessing_version=None, filtered_version=None):
    """
    Visualize the complete preprocessing pipeline results.
    
    Args:
        zarr_path: Path to zarr file
        frame_range: Optional (start, end) tuple to limit visualization
        save_path: Optional path to save the figure
        preprocessing_version: Specific preprocessing version to visualize (default: latest)
        filtered_version: Specific filtered version to visualize (default: latest)
    """
    print(f"\n{'='*60}")
    print("PREPROCESSING PIPELINE VISUALIZATION")
    print(f"{'='*60}")
    
    # Normalize path
    zarr_path = str(zarr_path).rstrip('/')
    print(f"Zarr file: {zarr_path}")
    
    # Load zarr
    root = zarr.open(zarr_path, mode='r')
    
    # Load data from each stage
    datasets = {}
    
    # 1. Original data
    print("\nLoading original data...")
    datasets['original'] = load_detection_data(root)
    total_frames = len(datasets['original']['n_detections'])
    
    # Apply frame range if specified
    if frame_range:
        start, end = frame_range
        print(f"Limiting to frames {start}-{end}")
        for key in ['bboxes', 'scores', 'n_detections']:
            datasets['original'][key] = datasets['original'][key][start:end]
        # Recalculate for limited range
        datasets['original'] = load_detection_data({'bboxes': datasets['original']['bboxes'],
                                                    'scores': datasets['original']['scores'],
                                                    'n_detections': datasets['original']['n_detections']})
    
    # 2. Filtered data (if exists)
    if 'filtered_runs' in root:
        if filtered_version:
            # Use specific version
            if filtered_version in root['filtered_runs']:
                print(f"Loading filtered data: {filtered_version}")
                filtered_group = root['filtered_runs'][filtered_version]
                datasets['filtered'] = load_detection_data(filtered_group)
            else:
                print(f"Warning: Filtered version '{filtered_version}' not found")
        elif 'latest' in root['filtered_runs'].attrs:
            # Use latest
            latest_filtered = root['filtered_runs'].attrs['latest']
            print(f"Loading filtered data: {latest_filtered}")
            filtered_group = root['filtered_runs'][latest_filtered]
            datasets['filtered'] = load_detection_data(filtered_group)
        
        if 'filtered' in datasets and frame_range:
            for key in ['bboxes', 'scores', 'n_detections']:
                datasets['filtered'][key] = datasets['filtered'][key][start:end]
            datasets['filtered'] = load_detection_data({'bboxes': datasets['filtered']['bboxes'],
                                                        'scores': datasets['filtered']['scores'],
                                                        'n_detections': datasets['filtered']['n_detections']})
    
    # 3. Interpolated data (if exists)
    if 'preprocessing' in root:
        if preprocessing_version:
            # Use specific version
            if preprocessing_version in root['preprocessing']:
                print(f"Loading preprocessed data: {preprocessing_version}")
                prep_group = root['preprocessing'][preprocessing_version]
                datasets['interpolated'] = load_detection_data(prep_group)
                
                # Get interpolation mask if available
                if 'interpolation_mask' in prep_group:
                    interp_mask = prep_group['interpolation_mask'][:]
                    if frame_range:
                        interp_mask = interp_mask[start:end]
                    datasets['interpolated']['interp_mask'] = interp_mask
            else:
                print(f"Warning: Preprocessing version '{preprocessing_version}' not found")
                print(f"Available versions: {[k for k in root['preprocessing'].keys() if not k.startswith('_')]}")
        elif 'latest' in root['preprocessing'].attrs:
            # Use latest
            latest_prep = root['preprocessing'].attrs['latest']
            print(f"Loading preprocessed data: {latest_prep}")
            prep_group = root['preprocessing'][latest_prep]
            datasets['interpolated'] = load_detection_data(prep_group)
            
            # Get interpolation mask if available
            if 'interpolation_mask' in prep_group:
                interp_mask = prep_group['interpolation_mask'][:]
                if frame_range:
                    interp_mask = interp_mask[start:end]
                datasets['interpolated']['interp_mask'] = interp_mask
        
        if 'interpolated' in datasets and frame_range:
            for key in ['bboxes', 'scores', 'n_detections']:
                datasets['interpolated'][key] = datasets['interpolated'][key][start:end]
            datasets['interpolated'] = load_detection_data({'bboxes': datasets['interpolated']['bboxes'],
                                                           'scores': datasets['interpolated']['scores'],
                                                           'n_detections': datasets['interpolated']['n_detections']})
            if 'interp_mask' in locals():
                datasets['interpolated']['interp_mask'] = interp_mask
    
    # Determine how many stages we have
    n_stages = len(datasets)
    
    if n_stages == 1:
        print("\n⚠️  Only original data found. Run preprocessing steps first!")
        print("   1. python frame_distance_analyzer.py <zarr> --threshold 250 --drop --save")
        print("   2. python gap_interpolator.py <zarr> --save")
    
    # Get fps from zarr attributes early
    fps = root.attrs.get('fps', 60.0)
    
    # Create visualization
    # We'll have trajectory plots on top row, coverage plots in middle rows, gap analysis on bottom
    fig = plt.figure(figsize=(20, 14))
    
    # Create custom grid - 4 rows now
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25, 
                          height_ratios=[1.2, 0.5, 0.8, 0.8])
    
    stage_names = {
        'original': 'Original Detections',
        'filtered': 'After Jump Removal',
        'interpolated': 'After Gap Interpolation'
    }
    
    colors = {
        'original': 'blue',
        'filtered': 'green', 
        'interpolated': 'purple'
    }
    
    # Store axes for coverage plots
    coverage_axes = []
    
    # Plot each stage's trajectory in top row
    for idx, (stage, data) in enumerate(datasets.items()):
        ax = fig.add_subplot(gs[0, idx])
        
        if len(data['centroids']) > 0:
            # Plot trajectory
            ax.plot(data['centroids'][:, 0], data['centroids'][:, 1],
                   color=colors[stage], alpha=0.3, linewidth=0.5)
            
            # Plot points
            scatter = ax.scatter(data['centroids'][:, 0], data['centroids'][:, 1],
                               c=data['valid_frames'], cmap='viridis', 
                               s=2, alpha=0.6)
            
            # If interpolated, mark interpolated points
            if stage == 'interpolated' and 'interp_mask' in data:
                # Find interpolated centroids
                interp_centroids = []
                for frame_idx in range(len(data['n_detections'])):
                    if data['interp_mask'][frame_idx] and data['n_detections'][frame_idx] > 0:
                        centroid = calculate_centroid(data['bboxes'][frame_idx, 0])
                        interp_centroids.append(centroid)
                
                if interp_centroids:
                    interp_centroids = np.array(interp_centroids)
                    ax.scatter(interp_centroids[:, 0], interp_centroids[:, 1],
                             c='red', s=4, alpha=0.5, label='Interpolated')
                    ax.legend()
            
            plt.colorbar(scatter, ax=ax, label='Frame')
        
        # Calculate stats
        n_detections = np.sum(data['n_detections'] > 0)
        total = len(data['n_detections'])
        coverage = n_detections / total * 100 if total > 0 else 0
        
        # Add title with stats
        title = f"{stage_names[stage]}\n"
        title += f"Coverage: {n_detections}/{total} ({coverage:.1f}%)"
        
        if stage == 'filtered' and 'original' in datasets:
            removed = datasets['original']['coverage'] * total - n_detections
            title += f" | Removed: {int(removed)} frames"
        elif stage == 'interpolated' and 'filtered' in datasets:
            added = n_detections - datasets['filtered']['coverage'] * total
            title += f" | Added: {int(added)} frames"
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Plot direct detection presence (barcode view) - NEW ROW 2
    for idx, (stage, data) in enumerate(datasets.items()):
        ax_bar = fig.add_subplot(gs[1, idx])
        
        # Create barcode visualization
        detection_mask = data['n_detections'] > 0
        frames = np.arange(len(detection_mask))
        time_seconds = frames / fps
        
        # Create a binary image for detections
        barcode_data = detection_mask.reshape(1, -1)
        
        # Plot as image
        im = ax_bar.imshow(barcode_data, aspect='auto', cmap='RdYlGn', 
                          vmin=0, vmax=1, interpolation='nearest',
                          extent=[0, time_seconds[-1], 0, 1])
        
        # Mark large gaps with annotations
        gaps = []
        gap_starts = []
        gap_ends = []
        in_gap = False
        gap_start = None
        
        for i, has_det in enumerate(detection_mask):
            if not has_det and not in_gap:
                gap_start = i
                in_gap = True
            elif has_det and in_gap:
                gap_size = i - gap_start
                if gap_size > 30:  # Only annotate large gaps (>30 frames)
                    gap_starts.append(gap_start / fps)
                    gap_ends.append(i / fps)
                    gaps.append((gap_start / fps, i / fps, gap_size))
                in_gap = False
        
        if in_gap:  # Handle gap at end
            gap_size = len(detection_mask) - gap_start
            if gap_size > 30:
                gap_starts.append(gap_start / fps)
                gap_ends.append(len(detection_mask) / fps)
                gaps.append((gap_start / fps, len(detection_mask) / fps, gap_size))
        
        # Annotate large gaps
        for start, end, size in gaps:
            mid = (start + end) / 2
            ax_bar.annotate(f'{size}', xy=(mid, 0.5), xytext=(mid, 1.5),
                          ha='center', va='bottom', fontsize=8,
                          arrowprops=dict(arrowstyle='->', color='red', lw=1))
        
        ax_bar.set_xlim([0, time_seconds[-1]])
        ax_bar.set_ylim([0, 2 if gaps else 1])
        ax_bar.set_xlabel('Time (seconds)')
        ax_bar.set_yticks([])
        ax_bar.set_title(f'Detection Presence (Green=Detection, Red=Gap)', fontsize=10)
        
        # Add coverage percentage
        coverage = np.sum(detection_mask) / len(detection_mask) * 100
        ax_bar.text(0.02, 0.5, f'{coverage:.1f}%', transform=ax_bar.transAxes,
                   fontsize=10, va='center', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot rolling coverage (now ROW 3)
    window = 100  # frames for rolling average
    
    for idx, (stage, data) in enumerate(datasets.items()):
        ax_cov = fig.add_subplot(gs[2, idx])
        coverage_axes.append(ax_cov)
        
        # Calculate rolling coverage
        detection_mask = data['n_detections'] > 0
        rolling_coverage = np.convolve(detection_mask, np.ones(window)/window, mode='same') * 100
        
        frames = np.arange(len(data['n_detections']))
        time_seconds = frames / fps
        
        # Plot coverage
        ax_cov.fill_between(time_seconds, 0, rolling_coverage, 
                           color=colors[stage], alpha=0.3)
        ax_cov.plot(time_seconds, rolling_coverage, 
                   color=colors[stage], alpha=0.8, linewidth=1)
        
        # Mark gaps
        gap_starts = []
        gap_ends = []
        in_gap = False
        
        for i, has_det in enumerate(detection_mask):
            if not has_det and not in_gap:
                gap_starts.append(i / fps)
                in_gap = True
            elif has_det and in_gap:
                gap_ends.append(i / fps)
                in_gap = False
        
        if in_gap:  # Handle gap at end
            gap_ends.append(len(detection_mask) / fps)
        
        # Shade gaps
        for start, end in zip(gap_starts, gap_ends):
            ax_cov.axvspan(start, end, color='red', alpha=0.1)
        
        ax_cov.set_ylim([0, 105])
        ax_cov.set_xlabel('Time (seconds)')
        ax_cov.set_ylabel('Detection Coverage (%)')
        ax_cov.set_title(f'Rolling Coverage (window={window} frames)', fontsize=10)
        ax_cov.grid(True, alpha=0.3)
        
        # Add overall coverage line
        overall_coverage = np.sum(detection_mask) / len(detection_mask) * 100
        ax_cov.axhline(y=overall_coverage, color='black', linestyle='--', 
                      alpha=0.5, linewidth=1, label=f'Mean: {overall_coverage:.1f}%')
        ax_cov.legend(loc='lower right', fontsize=8)
    
    # Plot gap analysis (now ROW 4)
    for idx, (stage, data) in enumerate(datasets.items()):
        ax_gap = fig.add_subplot(gs[3, idx])
        
        # Find gaps
        detection_mask = data['n_detections'] > 0
        gaps = []
        gap_sizes = []
        in_gap = False
        gap_start = None
        
        for i, has_det in enumerate(detection_mask):
            if not has_det and not in_gap:
                gap_start = i
                in_gap = True
            elif has_det and in_gap:
                gaps.append((gap_start, i))
                gap_sizes.append(i - gap_start)
                in_gap = False
        
        if in_gap:  # Handle gap at end
            gaps.append((gap_start, len(detection_mask)))
            gap_sizes.append(len(detection_mask) - gap_start)
        
        if gap_sizes:
            # Histogram of gap sizes
            max_gap = max(gap_sizes)
            bins = np.arange(0, min(max_gap + 2, 50), 1)
            ax_gap.hist(gap_sizes, bins=bins, color=colors[stage], alpha=0.7, edgecolor='black')
            
            # Add statistics
            mean_gap = np.mean(gap_sizes)
            median_gap = np.median(gap_sizes)
            ax_gap.axvline(x=mean_gap, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_gap:.1f}')
            ax_gap.axvline(x=median_gap, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_gap:.1f}')
            
            ax_gap.set_xlabel('Gap Size (frames)')
            ax_gap.set_ylabel('Count')
            ax_gap.set_title(f'Gap Distribution | Total: {len(gaps)} gaps', fontsize=10)
            ax_gap.legend(loc='upper right', fontsize=8)
            ax_gap.grid(True, alpha=0.3)
            
            # Add text with gap statistics
            stats_text = f"≤5 frames: {sum(1 for g in gap_sizes if g <= 5)}\n"
            stats_text += f"6-10 frames: {sum(1 for g in gap_sizes if 5 < g <= 10)}\n"
            stats_text += f">10 frames: {sum(1 for g in gap_sizes if g > 10)}"
            ax_gap.text(0.98, 0.98, stats_text, transform=ax_gap.transAxes,
                       fontsize=8, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax_gap.text(0.5, 0.5, 'No gaps detected!', 
                       ha='center', va='center', transform=ax_gap.transAxes,
                       fontsize=12, color='green', weight='bold')
            ax_gap.set_xlabel('Gap Size (frames)')
            ax_gap.set_ylabel('Count')
            ax_gap.set_title('Gap Distribution', fontsize=10)
            ax_gap.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Preprocessing Pipeline Results - Trajectory, Coverage & Gap Analysis', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {save_path}")
    
    plt.show()
    
    # Print summary to console
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for stage in datasets:
        n_det = np.sum(datasets[stage]['n_detections'] > 0)
        total = len(datasets[stage]['n_detections'])
        coverage = n_det / total * 100 if total > 0 else 0
        print(f"{stage_names[stage]}: {coverage:.1f}% coverage ({n_det}/{total} frames)")
    
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description='Visualize preprocessing pipeline results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all preprocessing stages (latest versions)
  %(prog)s detections.zarr
  
  # List all available versions
  %(prog)s detections.zarr --list-versions
  
  # Visualize specific version
  %(prog)s detections.zarr --version v3_interpolated_20250821_141332
  
  # Focus on specific frame range
  %(prog)s detections.zarr --frames 1000 2000
  
  # Save the visualization
  %(prog)s detections.zarr --save pipeline_viz.png
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    
    parser.add_argument(
        '--frames',
        nargs=2,
        type=int,
        metavar=('START', 'END'),
        help='Frame range to visualize'
    )
    
    parser.add_argument(
        '--save',
        help='Path to save the figure'
    )
    
    parser.add_argument('--version', '--preprocessing-version',
                       dest='preprocessing_version',
                       help='Specific preprocessing version to visualize (e.g., v3_interpolated_20250821_141332)')
    
    parser.add_argument('--filtered-version',
                       help='Specific filtered version to visualize')
    
    parser.add_argument('--list-versions', action='store_true',
                       help='List all available versions and exit')
    
    args = parser.parse_args()

    print(f"DEBUG: list_versions = {args.list_versions}")
    if args.list_versions:
        import zarr
        import json
        
        # Normalize path
        zarr_path = str(args.zarr_path).rstrip('/')
        root = zarr.open(zarr_path, mode='r')
        
        print("\n" + "="*60)
        print("AVAILABLE VERSIONS")
        print("="*60)
        
        if 'filtered_runs' in root:
            print("\nFiltered runs:")
            for key in root['filtered_runs'].keys():
                if not key.startswith('_'):
                    is_latest = root['filtered_runs'].attrs.get('latest', '') == key
                    print(f"  - {key} {'[LATEST]' if is_latest else ''}")
        
        if 'preprocessing' in root:
            print("\nPreprocessing versions:")
            versions_info = []
            
            for key in root['preprocessing'].keys():
                if not key.startswith('_'):
                    is_latest = root['preprocessing'].attrs.get('latest', '') == key
                    
                    # Get coverage for this version
                    n_det = root['preprocessing'][key]['n_detections'][:]
                    coverage = (n_det > 0).sum() / len(n_det) * 100
                    
                    # Get max_gap parameter if available
                    max_gap = "?"
                    gaps_filled = "?"
                    if 'history' in root['preprocessing'][key].attrs:
                        history = json.loads(root['preprocessing'][key].attrs['history'])
                        for step in history:
                            if step['step'] == 'interpolate_gaps':
                                max_gap = step['params'].get('max_gap', '?')
                                gaps_filled = step.get('gaps_filled', '?')
                    
                    versions_info.append({
                        'name': key,
                        'coverage': coverage,
                        'max_gap': max_gap,
                        'gaps_filled': gaps_filled,
                        'is_latest': is_latest
                    })
            
            # Sort by coverage
            versions_info.sort(key=lambda x: x['coverage'], reverse=True)
            
            for v in versions_info:
                latest_tag = '[LATEST]' if v['is_latest'] else ''
                print(f"  - {v['name']} {latest_tag}")
                print(f"      Coverage: {v['coverage']:.2f}%")
                print(f"      Max gap: {v['max_gap']} frames")
                print(f"      Gaps filled: {v['gaps_filled']}")
                print()
        
        print("Usage:")
        print(f"  python src/visualize_preprocessing.py {args.zarr_path} --version <version_name>")
        print("\nExample:")
        if 'preprocessing' in root and versions_info:
            best = versions_info[0]['name']
            print(f"  python src/visualize_preprocessing.py {args.zarr_path} --version {best}")
        
        return 0  # EXIT HERE - don't run visualization
    
    frame_range = None
    if args.frames:
        frame_range = tuple(args.frames)
    
    visualize_preprocessing_pipeline(
        zarr_path=args.zarr_path,
        frame_range=frame_range,
        save_path=args.save
    )
    
    return 0


if __name__ == '__main__':
    exit(main())