#!/usr/bin/env python3
"""
Gap Interpolator for Multi-Fish Tracker Zarr

Fills detection gaps in multi-fish tracking data using linear interpolation.
Works with the multi-fish tracker zarr structure and can process either
original detect_runs or filtered_runs data.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import uniform_filter1d
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional


def load_multifish_data_for_interpolation(zarr_path: str, 
                                         source: str = 'latest',
                                         fish_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load detection data from multi-fish zarr for interpolation.
    
    Args:
        zarr_path: Path to multi-fish zarr file
        source: 'latest', 'filtered', or specific run name
        fish_id: Optional fish ID to process
    
    Returns:
        bbox_coords: Array of normalized bbox coordinates
        n_detections: Array of detections per frame
        metadata: Dictionary with zarr metadata and source info
    """
    root = zarr.open(str(zarr_path), mode='r')
    
    # Determine which data to load
    if source == 'filtered' and 'filtered_runs' in root:
        group_name = 'filtered_runs'
        run_name = root['filtered_runs'].attrs['latest']
    elif source == 'latest':
        if 'filtered_runs' in root:
            group_name = 'filtered_runs'
            run_name = root['filtered_runs'].attrs['latest']
            print(f"Using filtered data: {run_name}")
        else:
            group_name = 'detect_runs'
            run_name = root['detect_runs'].attrs['latest']
            print(f"Using original detections: {run_name}")
    else:
        # Try to find specific run
        if '/' in source:
            group_name, run_name = source.split('/')
        else:
            group_name = 'detect_runs'
            run_name = source
    
    data_group = root[f'{group_name}/{run_name}']
    
    # Load data
    n_detections = data_group['n_detections'][:]
    bbox_coords = data_group['bbox_norm_coords'][:]
    
    # Get metadata
    if 'raw_video' in root:
        width = root['raw_video/images_ds'].shape[2]
        height = root['raw_video/images_ds'].shape[1]
        fps = root.attrs.get('fps', 60.0) if 'fps' in root.attrs else 60.0
    else:
        width = 640
        height = 640
        fps = 60.0
    
    total_frames = len(n_detections)
    
    metadata = {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'source_group': group_name,
        'source_run': run_name,
        'fish_id': fish_id
    }
    
    return bbox_coords, n_detections, metadata


def identify_gaps(n_detections: np.ndarray, max_gap: int = 20) -> List[Tuple[int, int]]:
    """
    Identify gaps in detection coverage.
    
    Returns list of (start_frame, end_frame) tuples for gaps.
    """
    # Find frames with detections
    detected_frames = np.where(n_detections > 0)[0]
    
    if len(detected_frames) < 2:
        return []
    
    gaps = []
    for i in range(1, len(detected_frames)):
        gap_start = detected_frames[i-1] + 1
        gap_end = detected_frames[i] - 1
        gap_length = gap_end - gap_start + 1
        
        if gap_length > 0 and gap_length <= max_gap:
            gaps.append((gap_start, gap_end))
    
    return gaps


def interpolate_gaps(bbox_coords: np.ndarray, n_detections: np.ndarray, 
                     max_gap: int = 20, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill detection gaps using linear interpolation.
    
    Returns:
        interpolated_coords: New bbox coordinates including interpolated points
        new_n_detections: Updated detection counts
        interpolation_mask: Boolean mask indicating which detections are interpolated
    """
    # Identify gaps to fill
    gaps = identify_gaps(n_detections, max_gap)
    
    if verbose:
        print(f"\nFound {len(gaps)} gaps to interpolate (max_gap={max_gap} frames)")
        if len(gaps) > 0:
            gap_lengths = [end - start + 1 for start, end in gaps]
            print(f"  Gap lengths: min={min(gap_lengths)}, max={max(gap_lengths)}, mean={np.mean(gap_lengths):.1f}")
    
    if len(gaps) == 0:
        # No gaps to fill, return original data
        return bbox_coords.copy(), n_detections.copy(), np.zeros(len(bbox_coords), dtype=bool)
    
    # Build frame-to-detection mapping
    detected_frames = np.where(n_detections > 0)[0]
    frame_to_bbox_idx = {}
    cumulative_detections = np.cumsum(np.insert(n_detections, 0, 0))
    
    for frame_idx in detected_frames:
        start_idx = cumulative_detections[frame_idx]
        end_idx = cumulative_detections[frame_idx + 1]
        if end_idx > start_idx:
            frame_to_bbox_idx[frame_idx] = start_idx  # Take first detection in frame
    
    # Prepare for interpolation
    new_coords_list = []
    interpolation_mask_list = []
    new_n_detections = n_detections.copy()
    
    # Copy existing detections
    for i in range(len(bbox_coords)):
        new_coords_list.append(bbox_coords[i])
        interpolation_mask_list.append(False)
    
    # Fill each gap
    filled_gaps = 0
    for gap_start, gap_end in gaps:
        # Get bounding boxes before and after gap
        frame_before = gap_start - 1
        frame_after = gap_end + 1
        
        if frame_before not in frame_to_bbox_idx or frame_after not in frame_to_bbox_idx:
            continue
        
        bbox_before = bbox_coords[frame_to_bbox_idx[frame_before]]
        bbox_after = bbox_coords[frame_to_bbox_idx[frame_after]]
        
        # Linear interpolation for each coordinate
        gap_length = gap_end - gap_start + 1
        for i, frame in enumerate(range(gap_start, gap_end + 1)):
            alpha = (i + 1) / (gap_length + 1)  # Interpolation weight
            interpolated_bbox = bbox_before * (1 - alpha) + bbox_after * alpha
            
            new_coords_list.append(interpolated_bbox)
            interpolation_mask_list.append(True)
            new_n_detections[frame] = 1
        
        filled_gaps += 1
    
    if verbose:
        print(f"  Successfully filled {filled_gaps}/{len(gaps)} gaps")
        new_coverage = np.sum(new_n_detections > 0) / len(new_n_detections) * 100
        old_coverage = np.sum(n_detections > 0) / len(n_detections) * 100
        print(f"  Coverage: {old_coverage:.1f}% → {new_coverage:.1f}% (+{new_coverage - old_coverage:.1f}%)")
    
    interpolated_coords = np.array(new_coords_list)
    interpolation_mask = np.array(interpolation_mask_list)
    
    return interpolated_coords, new_n_detections, interpolation_mask


def save_interpolated_data(zarr_path: str, interpolated_coords: np.ndarray,
                          new_n_detections: np.ndarray, interpolation_mask: np.ndarray,
                          metadata: Dict, max_gap: int) -> str:
    """
    Save interpolated data to zarr in preprocessing group.
    """
    root = zarr.open(str(zarr_path), mode='r+')
    
    # Create preprocessing group if it doesn't exist
    if 'preprocessing' not in root:
        root.create_group('preprocessing')
    
    # Create timestamped run
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f'interpolated_{timestamp}'
    run_group = root['preprocessing'].create_group(run_name)
    root['preprocessing'].attrs['latest'] = run_name
    
    # Store parameters and metadata
    run_group.attrs['interpolation_timestamp_utc'] = datetime.now().isoformat()
    run_group.attrs['max_gap'] = max_gap
    run_group.attrs['source_group'] = metadata['source_group']
    run_group.attrs['source_run'] = metadata['source_run']
    if metadata['fish_id'] is not None:
        run_group.attrs['fish_id'] = metadata['fish_id']
    
    # Reorganize data to match original structure
    # We need to map interpolated coords back to the frame structure
    final_bbox_list = []
    final_n_detections = np.zeros(metadata['total_frames'], dtype='i4')
    
    coord_idx = 0
    for frame_idx in range(metadata['total_frames']):
        if new_n_detections[frame_idx] > 0:
            final_bbox_list.append(interpolated_coords[coord_idx])
            final_n_detections[frame_idx] = 1
            coord_idx += 1
    
    # Store data
    run_group.create_dataset('n_detections', data=final_n_detections, chunks=(1000,))
    run_group.create_dataset('bbox_norm_coords', data=np.array(final_bbox_list), chunks=(1000, 4))
    run_group.create_dataset('interpolation_mask', 
                            data=interpolation_mask[:len(final_bbox_list)], 
                            chunks=(1000,))
    
    # Calculate statistics
    n_interpolated = np.sum(interpolation_mask[:len(final_bbox_list)])
    n_original = len(final_bbox_list) - n_interpolated
    coverage = np.sum(final_n_detections > 0) / len(final_n_detections) * 100
    
    run_group.attrs['summary_statistics'] = {
        'total_frames': metadata['total_frames'],
        'frames_with_detections': int(np.sum(final_n_detections > 0)),
        'coverage_percent': float(coverage),
        'n_original_detections': int(n_original),
        'n_interpolated_detections': int(n_interpolated),
        'interpolation_percent': float(n_interpolated / len(final_bbox_list) * 100) if len(final_bbox_list) > 0 else 0
    }
    
    print(f"\nSaved interpolated data to: {run_group.path}")
    print(f"  Total detections: {len(final_bbox_list)}")
    print(f"  Original: {n_original}, Interpolated: {n_interpolated}")
    print(f"  Final coverage: {coverage:.1f}%")
    
    return run_group.path


def plot_interpolation_results(bbox_coords: np.ndarray, n_detections_orig: np.ndarray, 
                              n_detections_interp: np.ndarray, interpolation_mask: np.ndarray,
                              metadata: Dict, gaps_filled: List[Tuple[int, int]],
                              save_path: Optional[str] = None, interactive: bool = True):
    """
    Create comprehensive gap interpolation visualization.
    """
    # Calculate trajectories
    width, height = metadata['width'], metadata['height']
    
    # Original trajectory
    orig_positions = []
    cumulative_orig = np.cumsum(np.insert(n_detections_orig, 0, 0))
    for frame_idx in range(len(n_detections_orig)):
        if n_detections_orig[frame_idx] > 0:
            bbox_idx = cumulative_orig[frame_idx]
            if bbox_idx < len(bbox_coords):
                center_x = bbox_coords[bbox_idx][0] * width
                center_y = bbox_coords[bbox_idx][1] * height
                orig_positions.append([center_x, center_y])
    
    # Interpolated trajectory
    interp_positions = []
    interp_flags = []
    cumulative_interp = np.cumsum(np.insert(n_detections_interp, 0, 0))
    for frame_idx in range(len(n_detections_interp)):
        if n_detections_interp[frame_idx] > 0:
            bbox_idx = cumulative_interp[frame_idx]
            if bbox_idx < len(bbox_coords):
                center_x = bbox_coords[bbox_idx][0] * width
                center_y = bbox_coords[bbox_idx][1] * height
                interp_positions.append([center_x, center_y])
                is_interp = interpolation_mask[bbox_idx] if bbox_idx < len(interpolation_mask) else False
                interp_flags.append(is_interp)
    
    orig_positions = np.array(orig_positions) if orig_positions else np.empty((0, 2))
    interp_positions = np.array(interp_positions) if interp_positions else np.empty((0, 2))
    
    # Create figure with 2x3 layout
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Gap Interpolation Results', fontsize=16, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Original trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    if len(orig_positions) > 0:
        frames_with_det = np.where(n_detections_orig > 0)[0]
        scatter = ax1.scatter(orig_positions[:, 0], orig_positions[:, 1],
                            c=frames_with_det[:len(orig_positions)], 
                            cmap='viridis', s=2, alpha=0.7)
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title(f'BEFORE - {len(orig_positions)} detections')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Interpolated trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    if len(interp_positions) > 0:
        frames_with_det = np.where(n_detections_interp > 0)[0]
        # Plot original points
        orig_mask = np.array(interp_flags) == False
        if np.any(orig_mask):
            ax2.scatter(interp_positions[orig_mask, 0], interp_positions[orig_mask, 1],
                       c=frames_with_det[:len(interp_positions)][orig_mask],
                       cmap='viridis', s=2, alpha=0.7)
        # Mark interpolated points
        interp_mask = np.array(interp_flags) == True
        if np.any(interp_mask):
            ax2.scatter(interp_positions[interp_mask, 0], interp_positions[interp_mask, 1],
                       color='red', s=4, alpha=0.5, marker='x', label='Interpolated')
            ax2.legend()
    ax2.set_xlabel('X Position (pixels)')
    ax2.set_ylabel('Y Position (pixels)')
    ax2.set_title(f'AFTER - {len(interp_positions)} detections')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Summary panel
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    # Calculate statistics
    n_gaps = len(gaps_filled)
    gap_sizes = [end - start + 1 for start, end in gaps_filled]
    
    summary_text = f"""INTERPOLATION SUMMARY

Gaps filled: {n_gaps}/{len(identify_gaps(n_detections_orig, 1000))}
Frames added: {np.sum(interp_flags) if interp_flags else 0}
Coverage improvement: +{(np.sum(n_detections_interp > 0) - np.sum(n_detections_orig > 0))/len(n_detections_orig)*100:.1f}%

Parameters:
• Max gap: {metadata.get('max_gap', 'N/A')} frames
• Method: linear
• Confidence decay: 0.95
• Min confidence: 0.1

Gap size distribution:
• 1-10 frames: {sum(1 for s in gap_sizes if s <= 10)}
• 11-50 frames: {sum(1 for s in gap_sizes if 10 < s <= 50)}
• 51-100 frames: {sum(1 for s in gap_sizes if 50 < s <= 100)}
• >100 frames: {sum(1 for s in gap_sizes if s > 100)}

All gaps found:
• Gap 1: {gap_sizes[0] if gap_sizes else 0} frames @ {gaps_filled[0][0] if gaps_filled else 0:.0f}s
• Gap 2: {gap_sizes[1] if len(gap_sizes) > 1 else 0} frames @ {gaps_filled[1][0]/metadata['fps'] if len(gaps_filled) > 1 else 0:.0f}s
"""
    
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 4: Gap size histogram
    ax4 = fig.add_subplot(gs[1, 0])
    all_gaps = identify_gaps(n_detections_orig, 1000)
    if all_gaps:
        gap_lengths = [end - start + 1 for start, end in all_gaps]
        filled_lengths = [end - start + 1 for start, end in gaps_filled]
        
        bins = np.logspace(0, np.log10(max(gap_lengths) + 1), 20)
        ax4.hist(gap_lengths, bins=bins, alpha=0.5, label='All gaps', color='red')
        ax4.hist(filled_lengths, bins=bins, alpha=0.7, label='Filled gaps', color='green')
        ax4.axvline(x=metadata.get('max_gap', 20), color='black', linestyle='--', 
                   label=f"Max gap = {metadata.get('max_gap', 20)}")
        ax4.set_xscale('log')
        ax4.set_xlabel('Gap Size (frames)')
        ax4.set_ylabel('Count')
        ax4.set_title('Gap Size Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Rolling coverage
    ax5 = fig.add_subplot(gs[1, 1])
    window = 100  # frames
    
    # Calculate rolling coverage
    coverage_orig = np.zeros(len(n_detections_orig))
    coverage_orig[n_detections_orig > 0] = 100
    coverage_interp = np.zeros(len(n_detections_interp))
    coverage_interp[n_detections_interp > 0] = 100
    
    # Simple rolling mean
    from scipy.ndimage import uniform_filter1d
    rolling_orig = uniform_filter1d(coverage_orig, size=window, mode='constant')
    rolling_interp = uniform_filter1d(coverage_interp, size=window, mode='constant')
    
    frames = np.arange(len(n_detections_orig))
    ax5.plot(frames, rolling_orig, 'b-', alpha=0.7, label='Before')
    ax5.plot(frames, rolling_interp, 'g-', alpha=0.7, label='After')
    ax5.fill_between(frames, rolling_orig, rolling_interp, 
                     where=(rolling_interp >= rolling_orig), 
                     color='green', alpha=0.3)
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Detection Coverage (%)')
    ax5.set_title(f'Rolling Coverage (window={window} frames)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 105)
    
    # Plot 6: Confidence distribution (if we had confidence scores)
    ax6 = fig.add_subplot(gs[1, 2])
    # Since we're using linear interpolation, create synthetic confidence scores
    confidence_scores = np.ones(len(bbox_coords))
    if len(interpolation_mask) > 0:
        # Lower confidence for interpolated points based on gap size
        for gap_start, gap_end in gaps_filled:
            gap_size = gap_end - gap_start + 1
            # Confidence decays with gap size
            gap_confidence = max(0.1, 1.0 - (gap_size / 100))
            # Find indices that correspond to this gap
            for frame in range(gap_start, gap_end + 1):
                if frame < len(n_detections_interp) and n_detections_interp[frame] > 0:
                    idx = cumulative_interp[frame]
                    if idx < len(confidence_scores):
                        confidence_scores[idx] = gap_confidence
    
    # Separate original and interpolated confidence
    orig_conf = confidence_scores[~interpolation_mask[:len(confidence_scores)]]
    interp_conf = confidence_scores[interpolation_mask[:len(confidence_scores)]]
    
    bins = np.linspace(0, 1, 20)
    ax6.hist(orig_conf, bins=bins, alpha=0.7, label='Original', color='blue')
    if len(interp_conf) > 0:
        ax6.hist(interp_conf, bins=bins, alpha=0.7, label='Interpolated', color='red')
    ax6.set_xlabel('Confidence Score')
    ax6.set_ylabel('Count')
    ax6.set_title('Confidence Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add interactive elements if requested
    if interactive:
        fig.text(0.5, 0.01, "Press 'S' to save | Press 'Q' to quit without saving",
                ha='center', fontsize=10, color='blue',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Fill detection gaps in multi-fish tracker data'
    )
    parser.add_argument('zarr_path', help='Path to multi-fish zarr file')
    parser.add_argument('--max-gap', type=int, default=20,
                       help='Maximum gap size to interpolate (frames)')
    parser.add_argument('--source', type=str, default='latest',
                       help='Data source: "latest", "filtered", or specific run name')
    parser.add_argument('--fish-id', type=int, default=None,
                       help='Process specific fish ID')
    parser.add_argument('--save', action='store_true',
                       help='Save interpolated data to zarr')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--output-plot', type=str, default=None,
                       help='Path to save plot')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MULTI-FISH GAP INTERPOLATOR")
    print("=" * 60)
    print(f"Zarr file: {args.zarr_path}")
    print(f"Max gap size: {args.max_gap} frames")
    print(f"Data source: {args.source}")
    
    try:
        # Load data
        bbox_coords, n_detections, metadata = load_multifish_data_for_interpolation(
            args.zarr_path,
            source=args.source,
            fish_id=args.fish_id
        )
        
        print(f"Loaded {len(bbox_coords)} detections from {metadata['total_frames']} frames")
        
        # Perform interpolation
        interpolated_coords, new_n_detections, interpolation_mask = interpolate_gaps(
            bbox_coords,
            n_detections,
            max_gap=args.max_gap,
            verbose=True
        )
        
        # Save if requested
        if args.save:
            save_interpolated_data(
                args.zarr_path,
                interpolated_coords,
                new_n_detections,
                interpolation_mask,
                metadata,
                args.max_gap
            )
        
        # Plot if requested
        if args.plot or args.output_plot:
            # Get the gaps that were filled
            gaps_filled = identify_gaps(n_detections, args.max_gap)
            
            plot_interpolation_results(
                interpolated_coords,
                n_detections,
                new_n_detections,
                interpolation_mask,
                metadata,
                gaps_filled,
                args.output_plot,
                interactive=True
            )
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())