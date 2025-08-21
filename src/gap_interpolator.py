#!/usr/bin/env python3
"""
Gap Interpolator for Detection Data

Fills gaps in detection data using intelligent interpolation.
Builds on cleaned data from jump removal step.

This is Step 2 of the preprocessing pipeline:
1. Remove jumps (frame_distance_analyzer.py)
2. Fill gaps (this script)
3. [Future: smooth trajectories]
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import interpolate
import json


def calculate_centroid(bbox):
    """Calculate the centroid of a bounding box."""
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def find_gaps(n_detections, max_gap=None):
    """
    Find gaps in detection data.
    
    Args:
        n_detections: Array of detection counts per frame
        max_gap: Maximum gap size to consider for interpolation (None = all gaps)
    
    Returns:
        List of gap dictionaries with start, end, and size
    """
    gaps = []
    in_gap = False
    gap_start = None
    
    for frame_idx in range(len(n_detections)):
        has_detection = n_detections[frame_idx] > 0
        
        if not has_detection and not in_gap:
            # Start of a gap
            in_gap = True
            gap_start = frame_idx
        elif has_detection and in_gap:
            # End of a gap
            gap_size = frame_idx - gap_start
            if max_gap is None or gap_size <= max_gap:
                gaps.append({
                    'start': gap_start,
                    'end': frame_idx - 1,
                    'size': gap_size,
                    'before_frame': gap_start - 1 if gap_start > 0 else None,
                    'after_frame': frame_idx
                })
            in_gap = False
    
    # Handle gap at the end
    if in_gap:
        gap_size = len(n_detections) - gap_start
        if max_gap is None or gap_size <= max_gap:
            gaps.append({
                'start': gap_start,
                'end': len(n_detections) - 1,
                'size': gap_size,
                'before_frame': gap_start - 1 if gap_start > 0 else None,
                'after_frame': None
            })
    
    return gaps


def interpolate_gap(bboxes, scores, gap, method='linear', confidence_decay=0.95):
    """
    Interpolate detections for a single gap.
    
    Args:
        bboxes: Full bbox array
        scores: Full scores array
        gap: Gap dictionary with start, end, before_frame, after_frame
        method: Interpolation method ('linear', 'cubic', 'nearest')
        confidence_decay: How much to decay confidence per frame
    
    Returns:
        interpolated_bboxes, interpolated_scores for the gap frames
    """
    if gap['before_frame'] is None or gap['after_frame'] is None:
        # Can't interpolate at boundaries
        return None, None
    
    # Get bounding boxes before and after gap
    bbox_before = bboxes[gap['before_frame'], 0]
    bbox_after = bboxes[gap['after_frame'], 0]
    score_before = scores[gap['before_frame'], 0]
    score_after = scores[gap['after_frame'], 0]
    
    # Calculate how many frames to interpolate
    gap_frames = np.arange(gap['start'], gap['end'] + 1)
    n_frames = len(gap_frames)
    
    # Interpolate each bbox coordinate
    interpolated_bboxes = np.zeros((n_frames, 4))
    
    if method == 'linear':
        # Linear interpolation
        for coord_idx in range(4):
            interpolated_bboxes[:, coord_idx] = np.linspace(
                bbox_before[coord_idx],
                bbox_after[coord_idx],
                n_frames + 2
            )[1:-1]  # Exclude the before/after frames
    
    elif method == 'nearest':
        # Use nearest neighbor (first half uses before, second half uses after)
        mid_point = n_frames // 2
        interpolated_bboxes[:mid_point] = bbox_before
        interpolated_bboxes[mid_point:] = bbox_after
    
    # Handle confidence scores
    interpolated_scores = np.zeros(n_frames)
    base_score = (score_before + score_after) / 2
    
    for i in range(n_frames):
        # Decay confidence based on distance from nearest real detection
        distance_to_nearest = min(i + 1, n_frames - i)
        decay_factor = confidence_decay ** distance_to_nearest
        interpolated_scores[i] = base_score * decay_factor
    
    return interpolated_bboxes, interpolated_scores


def fill_gaps(zarr_path, max_gap=10, method='linear', confidence_decay=0.95,
              min_confidence=0.1, source='latest', visualize=False, save=False):
    """
    Fill gaps in detection data using interpolation.
    
    Args:
        zarr_path: Path to zarr file
        max_gap: Maximum gap size to interpolate (frames)
        method: Interpolation method
        confidence_decay: Confidence decay per frame away from real detection
        min_confidence: Minimum confidence for interpolated detections
        source: Which data to use ('latest', 'original', or specific run name)
        visualize: Show before/after visualization
        save: Save the interpolated data
    """
    print(f"\n{'='*60}")
    print("GAP INTERPOLATION")
    print(f"{'='*60}")
    print(f"Zarr file: {zarr_path}")
    print(f"Max gap size: {max_gap} frames")
    print(f"Method: {method}")
    print(f"Source: {source}")
    print()
    
    # Load zarr
    root = zarr.open(str(zarr_path), mode='r+' if save else 'r')
    
    # Determine which data to use
    if source == 'latest':
        # Try to use latest cleaned data, fall back to original
        if 'preprocessing' in root and 'latest' in root['preprocessing'].attrs:
            source_path = root['preprocessing'].attrs['latest']
            source_group = root['preprocessing'][source_path]
            print(f"Using latest preprocessed data: {source_path}")
        elif 'cleaned_runs' in root and 'latest' in root['cleaned_runs'].attrs:
            source_name = root['cleaned_runs'].attrs['latest']
            source_group = root['cleaned_runs'][source_name]
            print(f"Using latest cleaned data: {source_name}")
        else:
            source_group = root
            print("Using original data")
    elif source == 'original':
        source_group = root
        print("Using original data")
    else:
        # Try to find specific run
        if 'preprocessing' in root and source in root['preprocessing']:
            source_group = root['preprocessing'][source]
        elif 'cleaned_runs' in root and source in root['cleaned_runs']:
            source_group = root['cleaned_runs'][source]
        else:
            print(f"Error: Could not find source '{source}'")
            return None
    
    # Load data
    bboxes = source_group['bboxes'][:]
    scores = source_group['scores'][:]
    class_ids = source_group['class_ids'][:]
    n_detections = source_group['n_detections'][:]
    
    # Get metadata
    fps = root.attrs.get('fps', 30.0)
    total_frames = len(n_detections)
    frames_with_detections = np.sum(n_detections > 0)
    
    print(f"Loaded data: {total_frames} frames, {frames_with_detections} with detections")
    print()
    
    # Find gaps
    all_gaps = find_gaps(n_detections, max_gap=None)
    fillable_gaps = find_gaps(n_detections, max_gap=max_gap)
    
    print(f"Gap Analysis:")
    print(f"  Total gaps: {len(all_gaps)}")
    print(f"  Fillable gaps (≤{max_gap} frames): {len(fillable_gaps)}")
    
    if fillable_gaps:
        gap_sizes = [g['size'] for g in fillable_gaps]
        print(f"  Fillable gap sizes: min={min(gap_sizes)}, max={max(gap_sizes)}, "
              f"mean={np.mean(gap_sizes):.1f}")
        total_frames_to_fill = sum(gap_sizes)
        print(f"  Total frames to fill: {total_frames_to_fill}")
    else:
        print("  No gaps to fill!")
        return None
    
    # Create interpolated data
    interpolated_bboxes = bboxes.copy()
    interpolated_scores = scores.copy()
    interpolated_n_detections = n_detections.copy()
    interpolation_mask = np.zeros(total_frames, dtype=bool)
    
    # Track processing history
    history = []
    if hasattr(source_group, 'attrs') and 'history' in source_group.attrs:
        # Load existing history
        history = json.loads(source_group.attrs['history'])
    
    # Fill each gap
    print(f"\nFilling {len(fillable_gaps)} gaps...")
    filled_count = 0
    
    for gap in fillable_gaps:
        # Interpolate this gap
        gap_bboxes, gap_scores = interpolate_gap(
            bboxes, scores, gap, method=method, confidence_decay=confidence_decay
        )
        
        if gap_bboxes is not None:
            # Apply minimum confidence threshold
            gap_scores = np.maximum(gap_scores, min_confidence)
            
            # Fill the gap
            for i, frame_idx in enumerate(range(gap['start'], gap['end'] + 1)):
                interpolated_bboxes[frame_idx, 0] = gap_bboxes[i]
                interpolated_scores[frame_idx, 0] = gap_scores[i]
                interpolated_n_detections[frame_idx] = 1
                interpolation_mask[frame_idx] = True
                filled_count += 1
    
    print(f"Filled {filled_count} frames")
    new_frames_with_detections = np.sum(interpolated_n_detections > 0)
    print(f"Coverage: {frames_with_detections}/{total_frames} → "
          f"{new_frames_with_detections}/{total_frames} "
          f"({frames_with_detections/total_frames*100:.1f}% → "
          f"{new_frames_with_detections/total_frames*100:.1f}%)")
    
    # Visualize if requested
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Gap Interpolation Results', fontsize=16, fontweight='bold')
        
        # Before trajectory
        ax1 = axes[0, 0]
        before_centroids = []
        for frame_idx in range(total_frames):
            if n_detections[frame_idx] > 0:
                centroid = calculate_centroid(bboxes[frame_idx, 0])
                before_centroids.append(centroid)
            else:
                before_centroids.append([np.nan, np.nan])
        before_centroids = np.array(before_centroids)
        
        valid_mask = ~np.isnan(before_centroids[:, 0])
        ax1.scatter(before_centroids[valid_mask, 0], before_centroids[valid_mask, 1],
                   c=np.where(valid_mask)[0], cmap='viridis', s=2, alpha=0.6)
        ax1.plot(before_centroids[valid_mask, 0], before_centroids[valid_mask, 1],
                'b-', alpha=0.2, linewidth=0.5)
        ax1.set_title(f'BEFORE - {frames_with_detections} detections')
        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Y Position (pixels)')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # After trajectory
        ax2 = axes[0, 1]
        after_centroids = []
        for frame_idx in range(total_frames):
            if interpolated_n_detections[frame_idx] > 0:
                centroid = calculate_centroid(interpolated_bboxes[frame_idx, 0])
                after_centroids.append(centroid)
            else:
                after_centroids.append([np.nan, np.nan])
        after_centroids = np.array(after_centroids)
        
        valid_mask = ~np.isnan(after_centroids[:, 0])
        colors = np.where(valid_mask)[0]
        
        # Color interpolated points differently
        scatter = ax2.scatter(after_centroids[valid_mask, 0], after_centroids[valid_mask, 1],
                            c=colors, cmap='viridis', s=2, alpha=0.6)
        
        # Mark interpolated points
        interp_mask_valid = interpolation_mask[valid_mask]
        if np.any(interp_mask_valid):
            interp_points = after_centroids[interpolation_mask]
            ax2.scatter(interp_points[:, 0], interp_points[:, 1],
                      c='red', s=4, alpha=0.5, label='Interpolated')
        
        ax2.plot(after_centroids[valid_mask, 0], after_centroids[valid_mask, 1],
                'g-', alpha=0.2, linewidth=0.5)
        ax2.set_title(f'AFTER - {new_frames_with_detections} detections')
        ax2.set_xlabel('X Position (pixels)')
        ax2.set_ylabel('Y Position (pixels)')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Statistics
        ax3 = axes[0, 2]
        ax3.axis('off')
        stats_text = f"""INTERPOLATION SUMMARY:

Gaps filled: {len(fillable_gaps)}/{len(all_gaps)}
Frames added: {filled_count}
Coverage improvement: +{(new_frames_with_detections-frames_with_detections)/total_frames*100:.1f}%

Parameters:
• Max gap: {max_gap} frames
• Method: {method}
• Confidence decay: {confidence_decay}
• Min confidence: {min_confidence}

Gap size distribution:
• ≤5 frames: {sum(1 for g in fillable_gaps if g['size'] <= 5)}
• 6-10 frames: {sum(1 for g in fillable_gaps if 5 < g['size'] <= 10)}
• >10 frames: {sum(1 for g in all_gaps if g['size'] > 10)} (not filled)"""
        
        ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Gap visualization
        ax4 = axes[1, 0]
        gap_sizes_all = [g['size'] for g in all_gaps]
        gap_sizes_filled = [g['size'] for g in fillable_gaps]
        
        bins = np.arange(0, max(gap_sizes_all) + 2)
        ax4.hist([gap_sizes_all, gap_sizes_filled], bins=bins, 
                label=['All gaps', 'Filled gaps'], color=['red', 'green'], alpha=0.6)
        ax4.axvline(x=max_gap, color='black', linestyle='--', label=f'Max gap = {max_gap}')
        ax4.set_xlabel('Gap Size (frames)')
        ax4.set_ylabel('Count')
        ax4.set_title('Gap Size Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Coverage timeline
        ax5 = axes[1, 1]
        window = 100  # frames
        coverage_before = np.convolve(n_detections > 0, np.ones(window)/window, mode='same')
        coverage_after = np.convolve(interpolated_n_detections > 0, np.ones(window)/window, mode='same')
        
        frames = np.arange(total_frames)
        ax5.plot(frames, coverage_before * 100, 'b-', alpha=0.6, label='Before')
        ax5.plot(frames, coverage_after * 100, 'g-', alpha=0.6, label='After')
        ax5.set_xlabel('Frame')
        ax5.set_ylabel('Detection Coverage (%)')
        ax5.set_title(f'Rolling Coverage (window={window} frames)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 105])
        
        # Confidence distribution
        ax6 = axes[1, 2]
        original_scores = scores[n_detections > 0, 0]
        interpolated_only_scores = interpolated_scores[interpolation_mask, 0]
        
        ax6.hist([original_scores, interpolated_only_scores], bins=30,
                label=['Original', 'Interpolated'], color=['blue', 'red'], alpha=0.6)
        ax6.set_xlabel('Confidence Score')
        ax6.set_ylabel('Count')
        ax6.set_title('Confidence Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add save instruction
        if save:
            fig.text(0.5, 0.01,
                    "Press 'S' to save | Press 'Q' to quit without saving",
                    ha='center', fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            def on_key(event):
                if event.key == 's':
                    save_interpolated_data(
                        root, interpolated_bboxes, interpolated_scores,
                        class_ids, interpolated_n_detections, interpolation_mask,
                        history, max_gap, method, confidence_decay,
                        min_confidence, filled_count, len(fillable_gaps)
                    )
                    plt.close('all')
                elif event.key == 'q':
                    print("\n✗ Cancelled - no data saved")
                    plt.close('all')
            
            fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.show()
    
    elif save:
        # Save without visualization
        save_interpolated_data(
            root, interpolated_bboxes, interpolated_scores,
            class_ids, interpolated_n_detections, interpolation_mask,
            history, max_gap, method, confidence_decay,
            min_confidence, filled_count, len(fillable_gaps)
        )
    
    return {
        'filled_gaps': len(fillable_gaps),
        'frames_added': filled_count,
        'coverage_before': frames_with_detections / total_frames,
        'coverage_after': new_frames_with_detections / total_frames
    }


def save_interpolated_data(root, bboxes, scores, class_ids, n_detections,
                           interpolation_mask, history, max_gap, method,
                           confidence_decay, min_confidence, filled_count, gaps_filled):
    """Save interpolated data to zarr with history tracking."""
    
    print("\n" + "="*60)
    print("SAVING INTERPOLATED DATA...")
    print("="*60)
    
    # Create preprocessing group if needed
    if 'preprocessing' not in root:
        prep_group = root.create_group('preprocessing')
        prep_group.attrs['created_at'] = datetime.now().isoformat()
        prep_group.attrs['description'] = 'Preprocessed detection data'
    else:
        prep_group = root['preprocessing']
    
    # Generate version name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    existing_versions = list(prep_group.keys())
    version_num = len([k for k in existing_versions if k.startswith('v')]) + 1
    version_name = f'v{version_num}_interpolated_{timestamp}'
    
    # Create version group
    version_group = prep_group.create_group(version_name)
    version_group.attrs['created_at'] = datetime.now().isoformat()
    
    # Add this step to history
    history.append({
        'step': 'interpolate_gaps',
        'timestamp': datetime.now().isoformat(),
        'params': {
            'max_gap': max_gap,
            'method': method,
            'confidence_decay': confidence_decay,
            'min_confidence': min_confidence
        },
        'frames_modified': filled_count,
        'gaps_filled': gaps_filled
    })
    
    version_group.attrs['history'] = json.dumps(history)
    
    # Save data
    version_group.create_dataset('bboxes', data=bboxes, dtype='float32')
    version_group.create_dataset('scores', data=scores, dtype='float32')
    version_group.create_dataset('class_ids', data=class_ids, dtype='int32')
    version_group.create_dataset('n_detections', data=n_detections, dtype='int32')
    version_group.create_dataset('interpolation_mask', data=interpolation_mask, dtype=bool)
    
    # Update latest pointer
    prep_group.attrs['latest'] = version_name
    
    print(f"✓ Saved as: {version_name}")
    print(f"✓ Set as latest preprocessed version")
    print(f"✓ History: {len(history)} steps")
    
    # Print history
    print("\nProcessing History:")
    for i, step in enumerate(history, 1):
        print(f"  {i}. {step['step']}: {step.get('frames_modified', 0)} frames modified")


def main():
    parser = argparse.ArgumentParser(
        description='Fill gaps in detection data using interpolation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview gap filling with default settings
  %(prog)s detections.zarr --visualize
  
  # Fill small gaps only
  %(prog)s detections.zarr --max-gap 5 --visualize
  
  # Use cubic interpolation for smoother paths
  %(prog)s detections.zarr --method cubic --visualize
  
  # Save after reviewing
  %(prog)s detections.zarr --max-gap 10 --visualize --save
  
  # Auto-save without preview
  %(prog)s detections.zarr --max-gap 10 --save
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    
    parser.add_argument(
        '--max-gap',
        type=int,
        default=10,
        help='Maximum gap size to interpolate (frames, default: 10)'
    )
    
    parser.add_argument(
        '--method',
        choices=['linear', 'nearest'],
        default='linear',
        help='Interpolation method (default: linear)'
    )
    
    parser.add_argument(
        '--confidence-decay',
        type=float,
        default=0.95,
        help='Confidence decay per frame from real detection (default: 0.95)'
    )
    
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.1,
        help='Minimum confidence for interpolated detections (default: 0.1)'
    )
    
    parser.add_argument(
        '--source',
        default='latest',
        help='Data source: "latest", "original", or specific version name'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show before/after visualization'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save interpolated data'
    )
    
    args = parser.parse_args()
    
    # Run interpolation
    results = fill_gaps(
        zarr_path=args.zarr_path,
        max_gap=args.max_gap,
        method=args.method,
        confidence_decay=args.confidence_decay,
        min_confidence=args.min_confidence,
        source=args.source,
        visualize=args.visualize,
        save=args.save
    )
    
    if results:
        print(f"\n{'='*60}")
        print("Interpolation complete!")
        if results['filled_gaps'] > 0:
            print(f"Coverage improved by {(results['coverage_after'] - results['coverage_before'])*100:.1f}%")
    
    return 0


if __name__ == '__main__':
    exit(main())