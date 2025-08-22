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
    
    # Normalize path to avoid double slashes in output
    zarr_path = str(zarr_path).rstrip('/')
    
    print(f"Zarr file: {zarr_path}")
    print(f"Max gap size: {max_gap} frames")
    print(f"Method: {method}")
    print(f"Source: {source}")
    print()
    
    # Open in read-write mode if we might save
    open_mode = 'r+' if (save or visualize) else 'r'
    root = zarr.open(zarr_path, mode=open_mode)
    
    # Get image dimensions for visualization
    img_width = root.attrs.get('width', 4512)
    img_height = root.attrs.get('height', 4512)
    
    # Determine which data to use
    if source == 'latest':
        # Priority order:
        # 1. Latest preprocessed data (from previous interpolation runs)
        # 2. Latest filtered data (from frame_distance_analyzer.py)
        # 3. Original data
        
        if 'preprocessing' in root and 'latest' in root['preprocessing'].attrs:
            source_path = root['preprocessing'].attrs['latest']
            source_group = root['preprocessing'][source_path]
            print(f"Using latest preprocessed data: {source_path}")
            
        elif 'filtered_runs' in root and 'latest' in root['filtered_runs'].attrs:
            source_name = root['filtered_runs'].attrs['latest']
            source_group = root['filtered_runs'][source_name]
            print(f"Using latest filtered data: {source_name}")
            
            # Check if gap information is available from the filtering step
            if 'output_statistics' in source_group.attrs:
                stats = source_group.attrs['output_statistics']
                if 'gaps' in stats:
                    print(f"  Found {stats['number_of_gaps']} gaps from filtering step")
                    print(f"  Largest gap: {stats['largest_gap']} frames")
            
        else:
            source_group = root
            print("Using original data (no preprocessing found)")
            
    elif source == 'original':
        source_group = root
        print("Using original data")
        
    else:
        # Try to find specific run in all possible locations
        found = False
        
        # Check preprocessing group
        if 'preprocessing' in root and source in root['preprocessing']:
            source_group = root['preprocessing'][source]
            print(f"Using preprocessing run: {source}")
            found = True
            
        # Check filtered_runs group (from frame_distance_analyzer.py)
        elif 'filtered_runs' in root and source in root['filtered_runs']:
            source_group = root['filtered_runs'][source]
            print(f"Using filtered run: {source}")
            found = True
            
        if not found:
            print(f"Error: Could not find source '{source}'")
            print("\nAvailable sources:")
            
            # List available preprocessing runs
            if 'preprocessing' in root:
                print("  In preprocessing/:")
                for key in root['preprocessing'].keys():
                    if key != 'metadata':  # Skip metadata entries
                        print(f"    - {key}")
                        
            # List available filtered runs
            if 'filtered_runs' in root:
                print("  In filtered_runs/:")
                for key in root['filtered_runs'].keys():
                    if key != 'metadata':  # Skip metadata entries
                        print(f"    - {key}")
                        
            print("\nUse --source with one of the above names, or 'latest' for the most recent, or 'original' for unprocessed data")
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
        print("  No gaps to fill with current max_gap setting!")
    
    # Show details of ALL gaps
    if all_gaps:
        print(f"\nDetailed gap information:")
        for i, gap in enumerate(all_gaps, 1):
            start_time = gap['start'] / fps
            end_time = (gap['end'] + 1) / fps
            print(f"  Gap {i}: Frames {gap['start']}-{gap['end']} "
                  f"({gap['size']} frames, {gap['size']/fps:.2f} sec, "
                  f"time {start_time:.2f}s-{end_time:.2f}s)")
            if gap['size'] > max_gap:
                print(f"         ^ Too large for interpolation (>{max_gap} frames)")
    
    # Don't return early if visualize is requested
    if not fillable_gaps and not visualize:
        return None
    
    # Create interpolated data
    interpolated_bboxes = bboxes.copy()
    interpolated_scores = scores.copy()
    interpolated_n_detections = n_detections.copy()
    interpolation_mask = np.zeros(total_frames, dtype=bool)
    
    # Track processing history
    history = []
    if hasattr(source_group, 'attrs'):
        # Check for history from frame_distance_analyzer
        if 'parameters' in source_group.attrs:
            history.append({
                'step': 'filter_jumps_and_segments',
                'timestamp': source_group.attrs.get('created_at', 'unknown'),
                'params': source_group.attrs['parameters'],
                'frames_modified': source_group.attrs.get('frames_dropped', 0)
            })
        # Check for existing interpolation history
        if 'history' in source_group.attrs:
            existing_history = json.loads(source_group.attrs['history'])
            if isinstance(existing_history, list):
                history.extend(existing_history)
    
    # Fill each gap
    if fillable_gaps:
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
    else:
        filled_count = 0
        print(f"\nNo gaps to fill with max_gap={max_gap}")
    
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
        before_valid_frames = []
        for frame_idx in range(total_frames):
            if n_detections[frame_idx] > 0:
                centroid = calculate_centroid(bboxes[frame_idx, 0])
                before_centroids.append(centroid)
                before_valid_frames.append(frame_idx)
        before_centroids = np.array(before_centroids) if before_centroids else np.empty((0, 2))
        
        if len(before_centroids) > 0:
            ax1.scatter(before_centroids[:, 0], before_centroids[:, 1],
                       c=before_valid_frames, cmap='viridis', s=2, alpha=0.6)
            
            # Plot segments with gaps shown as red dotted lines
            for i in range(1, len(before_valid_frames)):
                prev_frame = before_valid_frames[i-1]
                curr_frame = before_valid_frames[i]
                prev_centroid = before_centroids[i-1]
                curr_centroid = before_centroids[i]
                
                if curr_frame - prev_frame == 1:
                    # Continuous segment
                    ax1.plot([prev_centroid[0], curr_centroid[0]], 
                            [prev_centroid[1], curr_centroid[1]],
                            'b-', alpha=0.3, linewidth=0.5)
                else:
                    # Gap - show as red dotted line
                    ax1.plot([prev_centroid[0], curr_centroid[0]], 
                            [prev_centroid[1], curr_centroid[1]],
                            'r--', alpha=0.5, linewidth=1)
        
        ax1.set_title(f'BEFORE - {frames_with_detections} detections')
        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Y Position (pixels)')
        ax1.set_xlim(0, img_width)
        ax1.set_ylim(0, img_height)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # After trajectory
        ax2 = axes[0, 1]
        after_centroids = []
        after_valid_frames = []
        for frame_idx in range(total_frames):
            if interpolated_n_detections[frame_idx] > 0:
                centroid = calculate_centroid(interpolated_bboxes[frame_idx, 0])
                after_centroids.append(centroid)
                after_valid_frames.append(frame_idx)
        after_centroids = np.array(after_centroids) if after_centroids else np.empty((0, 2))
        
        if len(after_centroids) > 0:
            # Plot all points
            ax2.scatter(after_centroids[:, 0], after_centroids[:, 1],
                       c=after_valid_frames, cmap='viridis', s=2, alpha=0.6)
            
            # Mark interpolated points in red
            interp_centroids = []
            interp_frames = []
            for frame_idx in range(total_frames):
                if interpolation_mask[frame_idx]:
                    centroid = calculate_centroid(interpolated_bboxes[frame_idx, 0])
                    interp_centroids.append(centroid)
                    interp_frames.append(frame_idx)
            
            if interp_centroids:
                interp_centroids = np.array(interp_centroids)
                ax2.scatter(interp_centroids[:, 0], interp_centroids[:, 1],
                          c='red', s=4, alpha=0.5, label='Interpolated', zorder=5)
            
            # Plot continuous trajectory
            ax2.plot(after_centroids[:, 0], after_centroids[:, 1],
                    'g-', alpha=0.3, linewidth=0.5)
        
        ax2.set_title(f'AFTER - {new_frames_with_detections} detections')
        ax2.set_xlabel('X Position (pixels)')
        ax2.set_ylabel('Y Position (pixels)')
        ax2.set_xlim(0, img_width)
        ax2.set_ylim(0, img_height)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Statistics (updated to show all gaps even when none are filled)
        ax3 = axes[0, 2]
        ax3.axis('off')
        
        # Create more detailed gap breakdown
        gap_breakdown = ""
        if all_gaps:
            gap_breakdown = "\n\nAll gaps found:"
            for i, gap in enumerate(all_gaps[:10], 1):  # Show first 10 gaps
                gap_breakdown += f"\n• Gap {i}: {gap['size']} frames @ {gap['start']/fps:.1f}s"
                if gap['size'] > max_gap:
                    gap_breakdown += " (too large)"
            if len(all_gaps) > 10:
                gap_breakdown += f"\n• ... and {len(all_gaps)-10} more gaps"
        
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
• ≤5 frames: {sum(1 for g in all_gaps if g['size'] <= 5)}
• 6-10 frames: {sum(1 for g in all_gaps if 5 < g['size'] <= 10)}
• 11-30 frames: {sum(1 for g in all_gaps if 10 < g['size'] <= 30)}
• 31-100 frames: {sum(1 for g in all_gaps if 30 < g['size'] <= 100)}
• >100 frames: {sum(1 for g in all_gaps if g['size'] > 100)}
{gap_breakdown}"""
        
        ax3.text(0.05, 0.5, stats_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Gap visualization
        ax4 = axes[1, 0]
        gap_sizes_all = [g['size'] for g in all_gaps]
        gap_sizes_filled = [g['size'] for g in fillable_gaps]
        
        if gap_sizes_all:
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
        interpolated_only_scores = interpolated_scores[interpolation_mask, 0] if np.any(interpolation_mask) else []
        
        if len(original_scores) > 0:
            ax6.hist(original_scores, bins=30, label='Original', color='blue', alpha=0.6)
        if len(interpolated_only_scores) > 0:
            ax6.hist(interpolated_only_scores, bins=30, label='Interpolated', color='red', alpha=0.6)
        ax6.set_xlabel('Confidence Score')
        ax6.set_ylabel('Count')
        ax6.set_title('Confidence Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add save instruction
        fig.text(0.5, 0.01,
                "Press 'W' to save | Press 'Q' to quit without saving",
                ha='center', fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        def on_key(event):
            if event.key.lower() == 'w':
                save_interpolated_data(
                    root, interpolated_bboxes, interpolated_scores,
                    class_ids, interpolated_n_detections, interpolation_mask,
                    history, max_gap, method, confidence_decay,
                    min_confidence, filled_count, len(fillable_gaps),
                    source_group
                )
                plt.close('all')
            elif event.key.lower() == 'q':
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
            min_confidence, filled_count, len(fillable_gaps),
            source_group
        )
    
    return {
        'filled_gaps': len(fillable_gaps),
        'frames_added': filled_count,
        'coverage_before': frames_with_detections / total_frames,
        'coverage_after': new_frames_with_detections / total_frames
    }


def save_interpolated_data(root, bboxes, scores, class_ids, n_detections,
                           interpolation_mask, history, max_gap, method,
                           confidence_decay, min_confidence, filled_count, 
                           gaps_filled, source_group):
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
    
    # Copy over source metadata if available
    if hasattr(source_group, 'attrs'):
        if 'source_info' in source_group.attrs:
            version_group.attrs['source_info'] = source_group.attrs['source_info']
        if 'output_statistics' in source_group.attrs:
            version_group.attrs['filtering_statistics'] = source_group.attrs['output_statistics']
    
    # Calculate new gap statistics
    remaining_gaps = find_gaps(n_detections, max_gap=None)
    version_group.attrs['interpolation_statistics'] = {
        'gaps_filled': gaps_filled,
        'frames_added': filled_count,
        'remaining_gaps': len(remaining_gaps),
        'largest_remaining_gap': max(g['size'] for g in remaining_gaps) if remaining_gaps else 0,
        'coverage_percentage': float(np.sum(n_detections > 0) / len(n_detections) * 100)
    }
    
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
  
  # Use nearest neighbor interpolation
  %(prog)s detections.zarr --method nearest --visualize
  
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
        print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    exit(main())