# src/fisheye/refinement/visualize_detect_quality.py
"""
Detection Quality Visualization

Visualizes detection quality metrics and artifacts identified by detect_quality.py
Shows trajectories, temporal artifacts, bbox issues, and quality scores.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_quality_report(zarr_path: str, 
                       detect_run: Optional[str] = None,
                       quality_run: Optional[str] = None) -> Tuple[Dict, Dict]:
    """
    Load quality report from within a detect run.
    
    Args:
        zarr_path: Path to zarr file
        detect_run: Specific detect run (default: latest)
        quality_run: Specific quality run within detect run (default: latest)
        
    Returns:
        Tuple of (quality_report_data, detection_data)
    """
    root = zarr.open(zarr_path, mode='r')
    
    # Get detect run
    if detect_run is None:
        detect_run = root['detect_runs'].attrs['latest']
    
    detect_group = root[f'detect_runs/{detect_run}']
    
    # Check for quality reports
    if 'quality_reports' not in detect_group:
        raise ValueError(f"No quality_reports found in {detect_run}. Run detect_quality.py first.")
    
    # Get quality run
    if quality_run is None:
        quality_run = detect_group['quality_reports'].attrs['latest']
    
    quality_group = detect_group[f'quality_reports/{quality_run}']
    
    # Load quality data
    quality_flags = quality_group['quality_flags'][:]
    island_frames = quality_group['island_frames'][:]
    blip_frames = quality_group['blip_frames'][:]
    jump_frames = quality_group['jump_frames'][:]
    
    # Load attributes
    quality_score = dict(quality_group.attrs['quality_score'])
    coverage_stats = dict(quality_group.attrs['coverage_stats'])
    bbox_validation = dict(quality_group.attrs['bbox_validation'])
    
    quality_data = {
        'quality_flags': quality_flags,
        'island_frames': island_frames,
        'blip_frames': blip_frames,
        'jump_frames': jump_frames,
        'quality_score': quality_score,
        'coverage_stats': coverage_stats,
        'bbox_validation': bbox_validation,
        'source_run': detect_run  # Changed: use the detect_run we already have
    }
    
    # Load corresponding detection data - source_run is the detect_run itself
    n_detections = detect_group['n_detections'][:]
    bbox_coords = detect_group['bbox_norm_coords'][:]
    
    # Get dimensions
    if 'raw_video' in root:
        width = root['raw_video/images_ds'].shape[2]
        height = root['raw_video/images_ds'].shape[1]
    else:
        width, height = 640, 640
    
    # Extract centroids and frame indices
    centroids = []
    frame_indices = []
    cumulative = np.cumsum(np.insert(n_detections, 0, 0))
    
    for frame_idx in range(len(n_detections)):
        if n_detections[frame_idx] > 0:
            idx = cumulative[frame_idx]
            center_x = bbox_coords[idx, 0] * width
            center_y = bbox_coords[idx, 1] * height
            centroids.append([center_x, center_y])
            frame_indices.append(frame_idx)
    
    detection_data = {
        'centroids': np.array(centroids),
        'frame_indices': np.array(frame_indices),
        'n_detections': n_detections,
        'bbox_coords': bbox_coords,
        'width': width,
        'height': height
    }
    
    return quality_data, detection_data

def create_quality_visualization(quality_data: Dict,
                                 detection_data: Dict,
                                 save_path: Optional[str] = None):
    """
    Create comprehensive quality visualization.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Detection Quality Analysis', fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Plot 1: Trajectory with artifacts marked
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    centroids = detection_data['centroids']
    frame_indices = detection_data['frame_indices']
    
    if len(centroids) > 0:
        # Base trajectory
        ax1.plot(centroids[:, 0], centroids[:, 1], 
                'b-', alpha=0.3, linewidth=0.5, zorder=1)
        
        # Good detections
        good_mask = quality_data['quality_flags'][frame_indices] == 0
        ax1.scatter(centroids[good_mask, 0], centroids[good_mask, 1],
                   c=frame_indices[good_mask], cmap='viridis', 
                   s=10, alpha=0.6, zorder=2, label='Good')
        
        # Mark islands
        if len(quality_data['island_frames']) > 0:
            island_mask = np.isin(frame_indices, quality_data['island_frames'])
            ax1.scatter(centroids[island_mask, 0], centroids[island_mask, 1],
                       color='red', s=100, marker='x', linewidths=2,
                       zorder=5, label=f"Islands ({len(quality_data['island_frames'])})")
        
        # Mark blips
        if len(quality_data['blip_frames']) > 0:
            blip_mask = np.isin(frame_indices, quality_data['blip_frames'])
            ax1.scatter(centroids[blip_mask, 0], centroids[blip_mask, 1],
                       color='orange', s=80, marker='s', 
                       zorder=4, label=f"Blips ({len(quality_data['blip_frames'])})")
        
        # Mark jumps
        if len(quality_data['jump_frames']) > 0:
            jump_mask = np.isin(frame_indices, quality_data['jump_frames'])
            ax1.scatter(centroids[jump_mask, 0], centroids[jump_mask, 1],
                       color='magenta', s=60, marker='^',
                       zorder=3, label=f"Jumps ({len(quality_data['jump_frames'])})")
        
        # Mark start/end
        ax1.plot(centroids[0, 0], centroids[0, 1], 'g^', 
                markersize=15, markeredgecolor='white', markeredgewidth=2,
                label='Start', zorder=6)
        ax1.plot(centroids[-1, 0], centroids[-1, 1], 'rs',
                markersize=15, markeredgecolor='white', markeredgewidth=2,
                label='End', zorder=6)
    
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title('Trajectory with Quality Flags')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Quality Score Summary
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    score = quality_data['quality_score']
    grade_color = {
        'A': 'green',
        'B': 'lightgreen',
        'C': 'yellow',
        'D': 'orange',
        'F': 'red'
    }
    
    summary_text = f"""QUALITY SCORE

Overall Grade: {score['grade']}
Overall Score: {score['overall_score']:.1f}/100

Component Scores:
- Coverage: {score['coverage_score']:.1f}/100
- Artifacts: {score['artifact_score']:.1f}/100
- Bbox: {score['bbox_score']:.1f}/100

Total Artifacts: {len(quality_data['island_frames']) + len(quality_data['blip_frames']) + len(quality_data['jump_frames'])}
"""
    
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor=grade_color.get(score['grade'], 'white'), 
                     alpha=0.8, edgecolor='black', linewidth=2))
    
    # Plot 3: Coverage Stats
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')
    
    cov = quality_data['coverage_stats']
    
    coverage_text = f"""COVERAGE

Total Frames: {cov['total_frames']}
Detected: {cov['present_frames']}
Missing: {cov['absent_frames']}
Coverage: {cov['coverage_percent']:.1f}%

Multi-detection: {cov['multi_detection_frames']}

Gaps:
- Total: {cov['gaps']['total_count']}
- Longest: {cov['gaps']['longest_gap']} frames
- Mean: {cov['gaps']['mean_gap_size']:.1f} frames
"""
    
    ax3.text(0.05, 0.95, coverage_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 4: Bbox Validation
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.axis('off')
    
    bbox = quality_data['bbox_validation']
    
    bbox_text = f"""BBOX VALIDATION

Total Boxes: {bbox['total_bboxes']}

Issues:
- Out of range: {bbox['out_of_range']}
- Size outliers: {bbox['size_outliers']}
- Malformed: {bbox['malformed']}

Size Stats:
- Mean: {bbox['mean_size']:.3f}
- Std: {bbox['std_size']:.3f}
- CV: {bbox['size_cv']:.3f}
"""
    
    ax4.text(0.05, 0.95, bbox_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Plot 5: Quality flags timeline
    ax5 = fig.add_subplot(gs[2, 0:2])
    
    quality_flags = quality_data['quality_flags']
    
    # Create color map for quality flags
    # 0=good(green), 1=island(red), 2=blip(orange), 3=jump(magenta), 4=multi(yellow)
    flag_colors = np.zeros((len(quality_flags), 3))
    flag_colors[quality_flags == 0] = [0, 1, 0]  # Green
    flag_colors[quality_flags == 1] = [1, 0, 0]  # Red (islands)
    flag_colors[quality_flags == 2] = [1, 0.5, 0]  # Orange (blips)
    flag_colors[quality_flags == 3] = [1, 0, 1]  # Magenta (jumps)
    flag_colors[quality_flags == 4] = [1, 1, 0]  # Yellow (multi)
    
    ax5.imshow([flag_colors], aspect='auto', interpolation='nearest')
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('')
    ax5.set_yticks([])
    ax5.set_title('Quality Flags Timeline (Green=Good, Red=Island, Orange=Blip, Magenta=Jump)')
    
    # Add frame numbers
    n_ticks = 10
    tick_positions = np.linspace(0, len(quality_flags)-1, n_ticks, dtype=int)
    ax5.set_xticks(tick_positions)
    ax5.set_xticklabels([str(i) for i in tick_positions])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize detection quality analysis results'
    )
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--quality-run', help='Specific quality run to visualize (default: latest)')
    parser.add_argument('--output', '-o', help='Save visualization to file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DETECTION QUALITY VISUALIZATION")
    print("="*60)
    
    try:
        # Load quality report
        quality_data, detection_data = load_quality_report(
            args.zarr_path,
            quality_run=args.quality_run
        )
        
        print(f"Quality run: {args.quality_run or 'latest'}")
        print(f"Source detect run: {quality_data['source_run']}")
        print(f"Overall grade: {quality_data['quality_score']['grade']}")
        print(f"Total artifacts: {len(quality_data['island_frames']) + len(quality_data['blip_frames']) + len(quality_data['jump_frames'])}")
        
        # Create visualization
        create_quality_visualization(quality_data, detection_data, args.output)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())