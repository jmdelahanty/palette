#!/usr/bin/env python3
"""
Refinement Pipeline Visualizer

Visualizes the results of the refinement pipeline:
- Original detections
- After filtering (filtered/)
- After interpolation (interpolated/)

Shows trajectories, coverage, interpolated vs real detections, and statistics.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"


def load_refined_stage(group, stage_name: str, fps: float) -> Dict:
    """Load data from a refinement stage (filtered or interpolated)."""
    bbox_coords = group['bbox_norm_coords'][:]
    frame_indices = group['frame_indices'][:]
    if 'frame_counts' in group:
        frame_counts = group['frame_counts'][:]
    else:
        frame_counts = np.bincount(frame_indices.astype(np.int64, copy=False))
    total_frames = len(frame_counts)
    frames = frame_indices
    
    # Get detection source if available (only for interpolated)
    detection_source = None
    if 'detection_source' in group:
        detection_source = group['detection_source'][:]
    
    # Calculate centroids (bboxes are normalized [cx, cy, w, h])
    centroids = bbox_coords[:, :2]  # Just center positions
    
    time_seconds = frames / fps
    
    return {
        'bbox_coords': bbox_coords,
        'frame_counts': frame_counts,
        'frame_indices': frame_indices,
        'centroids': centroids,
        'frames': frames,
        'time_seconds': time_seconds,
        'detection_source': detection_source,
        'coverage': float(np.sum(frame_counts > 0) / total_frames) if total_frames else 0.0,
        'total_detections': len(bbox_coords),
        'stage': stage_name
    }


def load_original_detections(detect_group, quality_group, fps: float) -> Dict:
    """Load original detection data."""
    bbox_coords = detect_group['bbox_norm_coords'][:]
    frame_indices = detect_group['frame_indices'][:]
    if 'frame_counts' in detect_group:
        frame_counts = detect_group['frame_counts'][:]
    else:
        frame_counts = np.bincount(frame_indices.astype(np.int64, copy=False))
    total_frames = len(frame_counts)

    detection_quality_labels = quality_group['detection_quality_labels'][:]
    if detection_quality_labels.size != frame_indices.size:
        detection_quality_labels = detection_quality_labels[:frame_indices.size]

    centroids = bbox_coords[:, :2]
    frames = frame_indices
    time_seconds = frames / fps
    
    return {
        'bbox_coords': bbox_coords,
        'frame_counts': frame_counts,
        'frame_indices': frame_indices,
        'detection_quality_labels': detection_quality_labels,
        'centroids': centroids,
        'frames': frames,
        'time_seconds': time_seconds,
        'coverage': float(np.sum(frame_counts > 0) / total_frames) if total_frames else 0.0,
        'total_detections': len(bbox_coords),
        'stage': 'original'
    }


def visualize_refinement_pipeline(zarr_path: str, 
                                 refined_run: Optional[str] = None,
                                 save_path: Optional[str] = None):
    """
    Visualize the refinement pipeline results.
    
    Args:
        zarr_path: Path to zarr file
        refined_run: Specific refined run to visualize (default: latest)
        save_path: Optional path to save the figure
    """
    print(f"\n{'='*70}")
    print("REFINEMENT PIPELINE VISUALIZATION")
    print(f"{'='*70}")
    
    # Load zarr
    root = zarr.open(zarr_path, mode='r')
    fps = root.attrs.get('fps', 60.0)
    
    # Get refined run
    refined_root = None
    if REFINED_DETECT_GROUP in root:
        refined_root = root[REFINED_DETECT_GROUP]
    elif LEGACY_REFINED_DETECT_GROUP in root:
        refined_root = root[LEGACY_REFINED_DETECT_GROUP]

    if refined_run is None:
        if refined_root is None or 'latest' not in refined_root.attrs:
            print("\nError: No refined runs found!")
            print("Run: python -m fisheye.refinement.refine_detect <zarr_path>")
            return
        refined_run = refined_root.attrs['latest']

    if refined_root is None or refined_run not in refined_root:
        print(f"\nError: Refined run '{refined_run}' not found!")
        return

    refined_group = refined_root[refined_run]
    
    # Get source runs
    source_detect = refined_group.attrs['source_detect_run']
    source_quality = refined_group.attrs['source_quality_run']
    
    print(f"\nRefined run: {refined_run}")
    print(f"Source detect: {source_detect}")
    print(f"Source quality: {source_quality}")
    
    # Load metadata
    params = refined_group.attrs['parameters']
    coverage_comparison = refined_group.attrs['coverage_comparison']
    
    print(f"\nParameters:")
    print(f"  Max gap: {params['max_gap']} frames")
    print(f"  Method: {params['interpolation_method']}")
    print(f"  Filters: {params['filters_applied']}")
    
    # Load data from each stage
    detect_group = root[f'detect_runs/{source_detect}']
    quality_group = detect_group[f'quality_reports/{source_quality}']
    
    datasets = {}
    datasets['original'] = load_original_detections(detect_group, quality_group, fps)
    datasets['filtered'] = load_refined_stage(refined_group['filtered'], 'filtered', fps)
    datasets['interpolated'] = load_refined_stage(refined_group['interpolated'], 'interpolated', fps)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25,
                          height_ratios=[1.2, 0.5, 0.8, 0.8])
    
    stage_names = {
        'original': 'Original Detections',
        'filtered': 'Filtered (Jumps Removed)',
        'interpolated': 'Interpolated (Gaps Filled)'
    }
    
    colors = {
        'original': 'blue',
        'filtered': 'green',
        'interpolated': 'purple'
    }
    
    # Row 1: Trajectory plots
    for idx, (stage, data) in enumerate(datasets.items()):
        ax = fig.add_subplot(gs[0, idx])
        
        if len(data['centroids']) > 0:
            # For original, color by quality
            if stage == 'original':
                quality_labels = data['detection_quality_labels']
                # Plot clean detections
                clean_mask = quality_labels == 0
                if np.any(clean_mask):
                    ax.scatter(data['centroids'][clean_mask, 0], 
                             data['centroids'][clean_mask, 1],
                             c='green', s=2, alpha=0.6, label='Clean')
                # Plot jumps
                jump_mask = quality_labels == 3
                if np.any(jump_mask):
                    ax.scatter(data['centroids'][jump_mask, 0],
                             data['centroids'][jump_mask, 1],
                             c='red', s=20, marker='x', alpha=0.8, label='Jumps')
                # Plot blips
                blip_mask = quality_labels == 2
                if np.any(blip_mask):
                    ax.scatter(data['centroids'][blip_mask, 0],
                             data['centroids'][blip_mask, 1],
                             c='orange', s=10, marker='s', alpha=0.6, label='Blips')
                ax.legend(loc='upper right', fontsize=8)
                
            # For interpolated, show real vs synthetic
            elif stage == 'interpolated' and data['detection_source'] is not None:
                real_mask = data['detection_source'] == 0
                interp_mask = data['detection_source'] == 1
                
                # Plot real detections
                if np.any(real_mask):
                    ax.scatter(data['centroids'][real_mask, 0],
                             data['centroids'][real_mask, 1],
                             c=data['frames'][real_mask], cmap='viridis',
                             s=2, alpha=0.6, label='Real')
                
                # Plot interpolated detections
                if np.any(interp_mask):
                    ax.scatter(data['centroids'][interp_mask, 0],
                             data['centroids'][interp_mask, 1],
                             c='red', s=15, marker='o', alpha=0.7,
                             edgecolors='darkred', linewidths=1, label='Interpolated')
                
                ax.legend(loc='upper right', fontsize=8)
                
            # For filtered, just plot all
            else:
                scatter = ax.scatter(data['centroids'][:, 0], data['centroids'][:, 1],
                                   c=data['frames'], cmap='viridis',
                                   s=2, alpha=0.6)
                plt.colorbar(scatter, ax=ax, label='Frame', pad=0.01)
        
        # Title with stats
        cov_info = coverage_comparison[stage]
        total_frames_stage = max(len(datasets[stage]['frame_counts']), 1)
        title = f"{stage_names[stage]}\n"
        title += f"Coverage: {cov_info['frames_with_detections']}/{total_frames_stage} "
        title += f"frames ({cov_info['coverage_percent']:.2f}%)"
        
        if stage == 'filtered':
            title += f"\nRemoved: {cov_info['detections_removed']} ({cov_info['coverage_loss']:.2f}%)"
        elif stage == 'interpolated':
            title += f"\nAdded: {cov_info['detections_added']} ({cov_info['coverage_gain']:.2f}%)"
        
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('X Position (normalized)')
        ax.set_ylabel('Y Position (normalized)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.invert_yaxis()
    
    # Row 2: Detection presence barcode
    for idx, (stage, data) in enumerate(datasets.items()):
        ax = fig.add_subplot(gs[1, idx])
        
        frame_counts = data['frame_counts']
        if frame_counts.size == 0:
            ax.text(0.5, 0.5, 'No frames', ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        detection_mask = frame_counts > 0
        time_seconds = np.arange(len(detection_mask)) / max(fps, 1e-6)
        
        barcode_data = detection_mask.reshape(1, -1)
        extent_end = time_seconds[-1] if time_seconds.size else 0
        ax.imshow(barcode_data, aspect='auto', cmap='RdYlGn',
                 vmin=0, vmax=1, interpolation='nearest',
                 extent=[0, extent_end, 0, 1])
        
        # Mark large gaps
        gaps = []
        in_gap = False
        gap_start = None
        
        for i, has_det in enumerate(detection_mask):
            if not has_det and not in_gap:
                gap_start = i
                in_gap = True
            elif has_det and in_gap:
                gap_size = i - gap_start
                if gap_size > 30:
                    gaps.append((gap_start / fps, i / fps, gap_size))
                in_gap = False
        
        if in_gap:
            gap_size = len(detection_mask) - gap_start
            if gap_size > 30:
                gaps.append((gap_start / fps, len(detection_mask) / fps, gap_size))
        
        for start, end, size in gaps:
            mid = (start + end) / 2
            ax.annotate(f'{size}', xy=(mid, 0.5), xytext=(mid, 1.5),
                       ha='center', va='bottom', fontsize=8,
                       arrowprops=dict(arrowstyle='->', color='red', lw=1))
        
        ax.set_xlim([0, extent_end])
        ax.set_ylim([0, 2 if gaps else 1])
        ax.set_xlabel('Time (seconds)')
        ax.set_yticks([])
        ax.set_title('Detection Presence', fontsize=10)
        
        cov_info_stage = coverage_comparison.get(stage, {})
        coverage_pct = cov_info_stage.get('coverage_percent', 0.0)
        ax.text(0.02, 0.5, f'{coverage_pct:.1f}%', transform=ax.transAxes,
               fontsize=10, va='center', weight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Row 3: Rolling coverage
    window = 100
    for idx, (stage, data) in enumerate(datasets.items()):
        ax = fig.add_subplot(gs[2, idx])
        
        frame_counts = data['frame_counts']
        if frame_counts.size == 0:
            ax.text(0.5, 0.5, 'No frames', ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        detection_mask = frame_counts > 0
        rolling_coverage = np.convolve(detection_mask, np.ones(window)/window, mode='same') * 100
        time_seconds = np.arange(len(detection_mask)) / max(fps, 1e-6)
        
        ax.fill_between(time_seconds, 0, rolling_coverage,
                       color=colors[stage], alpha=0.3)
        ax.plot(time_seconds, rolling_coverage,
               color=colors[stage], alpha=0.8, linewidth=1)
        
        # Mark gaps
        in_gap = False
        for i, has_det in enumerate(detection_mask):
            if not has_det and not in_gap:
                gap_start = i / fps
                in_gap = True
            elif has_det and in_gap:
                ax.axvspan(gap_start, i / fps, color='red', alpha=0.1)
                in_gap = False
        
        if in_gap:
            ax.axvspan(gap_start, len(detection_mask) / fps, color='red', alpha=0.1)
        
        ax.set_ylim([0, 105])
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Coverage (%)')
        ax.set_title(f'Rolling Coverage (window={window} frames)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        mean_cov = np.mean(rolling_coverage)
        ax.axhline(y=mean_cov, color='black', linestyle='--',
                  alpha=0.5, linewidth=1, label=f'Mean: {mean_cov:.1f}%')
        ax.legend(loc='lower right', fontsize=8)
    
    # Row 4: Gap analysis
    for idx, (stage, data) in enumerate(datasets.items()):
        ax = fig.add_subplot(gs[3, idx])
        
        frame_counts = data['frame_counts']
        if frame_counts.size == 0:
            ax.text(0.5, 0.5, 'No frames', ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        detection_mask = frame_counts > 0
        gap_sizes = []
        in_gap = False
        gap_start = None
        
        for i, has_det in enumerate(detection_mask):
            if not has_det and not in_gap:
                gap_start = i
                in_gap = True
            elif has_det and in_gap:
                gap_sizes.append(i - gap_start)
                in_gap = False
        
        if in_gap:
            gap_sizes.append(len(detection_mask) - gap_start)
        
        if gap_sizes:
            max_gap = max(gap_sizes)
            bins = np.arange(0, min(max_gap + 2, 50), 1)
            ax.hist(gap_sizes, bins=bins, color=colors[stage], alpha=0.7, edgecolor='black')
            
            mean_gap = np.mean(gap_sizes)
            median_gap = np.median(gap_sizes)
            ax.axvline(x=mean_gap, color='red', linestyle='--', alpha=0.7,
                      label=f'Mean: {mean_gap:.1f}')
            ax.axvline(x=median_gap, color='orange', linestyle='--', alpha=0.7,
                      label=f'Median: {median_gap:.1f}')
            
            ax.set_xlabel('Gap Size (frames)')
            ax.set_ylabel('Count')
            ax.set_title(f'Gap Distribution | Total: {len(gap_sizes)} gaps', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            stats_text = f"≤5: {sum(1 for g in gap_sizes if g <= 5)}\n"
            stats_text += f"6-10: {sum(1 for g in gap_sizes if 5 < g <= 10)}\n"
            stats_text += f">10: {sum(1 for g in gap_sizes if g > 10)}"
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No gaps!',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='green', weight='bold')
            ax.set_xlabel('Gap Size (frames)')
            ax.set_ylabel('Count')
            ax.set_title('Gap Distribution', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'Refinement Pipeline Results - {refined_run}',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {save_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for stage_key in ['original', 'filtered', 'interpolated']:
        cov = coverage_comparison[stage_key]
        print(f"{stage_names[stage_key]}:")
        print(f"  Coverage: {cov['coverage_percent']:.2f}% ({cov['frames_with_detections']} frames)")
        print(f"  Detections: {cov['total_detections']}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize refinement pipeline results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize latest refined run
  %(prog)s data.zarr
  
  # Visualize specific refined run
  %(prog)s data.zarr --run refined_2025-10-03_21-39-43
  
  # Save visualization
  %(prog)s data.zarr --save refinement_viz.png
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--run', '--refined-run', dest='refined_run',
                       help='Specific refined run to visualize (default: latest)')
    parser.add_argument('--save', help='Path to save the figure')
    
    args = parser.parse_args()
    
    visualize_refinement_pipeline(
        zarr_path=args.zarr_path,
        refined_run=args.refined_run,
        save_path=args.save
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
