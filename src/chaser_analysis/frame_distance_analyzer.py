#!/usr/bin/env python3
"""
Frame-to-Frame Distance Analyzer for Multi-Fish Tracker Zarr

Analyzes frame-to-frame distances in multi-fish tracking data to identify
and optionally remove tracking jumps, outliers, and artifacts.

Compatible with the multi-fish tracker zarr structure that organizes data
into timestamped run groups.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings


def load_multifish_detections(zarr_path: str, fish_id: Optional[int] = None, 
                              run_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load detection data from multi-fish tracker zarr.
    
    Args:
        zarr_path: Path to multi-fish zarr file
        fish_id: Optional fish ID to extract (if id_assignments available)
        run_name: Specific detect run to use (default: latest)
    
    Returns:
        bboxes: Array of bounding boxes in pixel coordinates
        n_detections: Array of detections per frame
        metadata: Dictionary with zarr metadata
    """
    root = zarr.open(str(zarr_path), mode='r')
    
    # Get detect run
    if 'detect_runs' not in root:
        raise ValueError("No detect_runs found in zarr file")
    
    if run_name is None:
        run_name = root['detect_runs'].attrs['latest']
    
    detect_group = root[f'detect_runs/{run_name}']
    
    # Load detection data
    n_detections_all = detect_group['n_detections'][:]
    bbox_coords_norm = detect_group['bbox_norm_coords'][:]
    
    # Get image dimensions
    if 'raw_video' in root:
        width = root['raw_video/images_ds'].shape[2]
        height = root['raw_video/images_ds'].shape[1]
        fps = root.attrs.get('fps', 60.0) if 'fps' in root.attrs else 60.0
    else:
        # Fallback dimensions
        width = 640
        height = 640
        fps = 60.0
    
    total_frames = len(n_detections_all)
    
    # Handle fish ID filtering if specified
    detection_mask = None
    if fish_id is not None and 'id_assignments_runs' in root:
        latest_assign = root['id_assignments_runs'].attrs['latest']
        assign_group = root[f'id_assignments_runs/{latest_assign}']
        detection_ids = assign_group['detection_ids'][:]
        detection_mask = detection_ids == fish_id
        print(f"Filtering for fish ID {fish_id}")
    
    # Process detections frame by frame
    bboxes_list = []
    n_detections = np.zeros(total_frames, dtype='i4')
    cumulative_detections = np.cumsum(np.insert(n_detections_all, 0, 0))
    
    for frame_idx in range(total_frames):
        start_idx = cumulative_detections[frame_idx]
        end_idx = cumulative_detections[frame_idx + 1]
        
        if end_idx > start_idx:
            frame_bboxes = bbox_coords_norm[start_idx:end_idx]
            
            # Apply fish ID filter if specified
            if detection_mask is not None:
                frame_mask = detection_mask[start_idx:end_idx]
                frame_bboxes = frame_bboxes[frame_mask]
            
            if len(frame_bboxes) > 0:
                # Take first detection in frame (or could take closest to previous)
                bbox = frame_bboxes[0]
                
                # Convert normalized coordinates to pixels
                center_x = bbox[0] * width
                center_y = bbox[1] * height
                box_width = bbox[2] * width
                box_height = bbox[3] * height
                
                # Store as (x1, y1, x2, y2) format
                x1 = center_x - box_width / 2
                y1 = center_y - box_height / 2
                x2 = center_x + box_width / 2
                y2 = center_y + box_height / 2
                
                bboxes_list.append([x1, y1, x2, y2])
                n_detections[frame_idx] = 1
    
    bboxes = np.array(bboxes_list) if bboxes_list else np.empty((0, 4))
    
    metadata = {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'detect_run': run_name,
        'fish_id': fish_id
    }
    
    return bboxes, n_detections, metadata


def analyze_frame_distances(zarr_path: str, threshold: float = 100.0,
                           fish_id: Optional[int] = None,
                           verbose: bool = True) -> Dict:
    """
    Analyze frame-to-frame distances in multi-fish tracking data.
    """
    # Load detection data
    bboxes, n_detections, metadata = load_multifish_detections(zarr_path, fish_id)
    
    if len(bboxes) == 0:
        raise ValueError("No detections found in zarr file")
    
    # Get centroids
    centroids = np.column_stack([
        (bboxes[:, 0] + bboxes[:, 2]) / 2,
        (bboxes[:, 1] + bboxes[:, 3]) / 2
    ])
    
    # Map detections to frames
    frame_indices = np.where(n_detections > 0)[0]
    
    # Calculate frame-to-frame distances
    distances = []
    frame_gaps = []
    
    for i in range(1, len(centroids)):
        dist = np.linalg.norm(centroids[i] - centroids[i-1])
        distances.append(dist)
        
        # Calculate frame gap
        gap = frame_indices[i] - frame_indices[i-1]
        frame_gaps.append(gap)
    
    distances = np.array(distances)
    frame_gaps = np.array(frame_gaps)
    
    # Find outliers
    outlier_indices = np.where(distances > threshold)[0]
    
    # Identify different types of issues
    jumps = []  # Large movements
    islands = []  # Single isolated detections
    blips = []  # Brief appearances
    
    for idx in outlier_indices:
        actual_idx = idx + 1  # Adjust for distance array offset
        
        # Check if it's an island (single detection between gaps)
        if idx > 0 and idx < len(distances) - 1:
            if distances[idx-1] > threshold and distances[idx] > threshold:
                islands.append(actual_idx)
                continue
        
        # Check if it's a blip (appears and disappears quickly)
        if frame_gaps[idx] > 5:  # Large gap suggests a blip
            blips.append(actual_idx)
        else:
            jumps.append(actual_idx)
    
    results = {
        'centroids': centroids,
        'distances': distances,
        'frame_gaps': frame_gaps,
        'frame_indices': frame_indices,
        'threshold': threshold,
        'outlier_indices': outlier_indices + 1,  # Adjust for 1-based indexing
        'jumps': np.array(jumps),
        'islands': np.array(islands),
        'blips': np.array(blips),
        'n_detections': n_detections,
        'metadata': metadata
    }
    
    if verbose:
        print(f"\nAnalysis Results:")
        print(f"  Total frames: {metadata['total_frames']}")
        print(f"  Frames with detections: {len(frame_indices)} ({len(frame_indices)/metadata['total_frames']*100:.1f}%)")
        print(f"  Distance threshold: {threshold:.1f} pixels")
        print(f"  Outliers found: {len(outlier_indices)}")
        if len(outlier_indices) > 0:
            print(f"    - Jumps: {len(jumps)}")
            print(f"    - Islands: {len(islands)}")
            print(f"    - Blips: {len(blips)}")
    
    return results


def save_filtered_data(zarr_path: str, results: Dict, drop_outliers: bool = False,
                       fish_id: Optional[int] = None) -> str:
    """
    Save filtered data back to multi-fish zarr structure.
    Creates a new filtered_runs group with cleaned detections.
    """
    root = zarr.open(str(zarr_path), mode='r+')
    
    # Create filtered_runs group if it doesn't exist
    if 'filtered_runs' not in root:
        root.create_group('filtered_runs')
    
    # Create timestamped run
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f'filtered_{timestamp}'
    run_group = root['filtered_runs'].create_group(run_name)
    root['filtered_runs'].attrs['latest'] = run_name
    
    # Store parameters
    run_group.attrs['filter_timestamp_utc'] = datetime.now().isoformat()
    run_group.attrs['threshold'] = results['threshold']
    run_group.attrs['drop_outliers'] = drop_outliers
    run_group.attrs['source_detect_run'] = results['metadata']['detect_run']
    if fish_id is not None:
        run_group.attrs['fish_id'] = fish_id
    
    # Determine which detections to keep
    if drop_outliers and len(results['outlier_indices']) > 0:
        # Remove outlier detections
        outlier_mask = np.ones(len(results['centroids']), dtype=bool)
        outlier_mask[results['outlier_indices'] - 1] = False  # Adjust for 0-based indexing
        
        filtered_centroids = results['centroids'][outlier_mask]
        filtered_frame_indices = results['frame_indices'][outlier_mask]
    else:
        filtered_centroids = results['centroids']
        filtered_frame_indices = results['frame_indices']
    
    # Create n_detections array for all frames
    n_detections_filtered = np.zeros(results['metadata']['total_frames'], dtype='i4')
    n_detections_filtered[filtered_frame_indices] = 1
    
    # Convert centroids back to normalized bbox format
    width = results['metadata']['width']
    height = results['metadata']['height']
    
    bbox_norm_list = []
    for centroid in filtered_centroids:
        # Assume a default box size (can be refined)
        box_width = 50 / width  # 50 pixels normalized
        box_height = 50 / height
        
        center_x_norm = centroid[0] / width
        center_y_norm = centroid[1] / height
        
        bbox_norm_list.append([center_x_norm, center_y_norm, box_width, box_height])
    
    # Store filtered data
    run_group.create_dataset('n_detections', data=n_detections_filtered, chunks=(1000,))
    run_group.create_dataset('bbox_norm_coords', data=np.array(bbox_norm_list), chunks=(1000, 4))
    
    # Store summary statistics
    coverage = len(filtered_frame_indices) / results['metadata']['total_frames'] * 100
    run_group.attrs['summary_statistics'] = {
        'total_frames': results['metadata']['total_frames'],
        'frames_with_detections': len(filtered_frame_indices),
        'coverage_percent': coverage,
        'outliers_removed': len(results['outlier_indices']) if drop_outliers else 0
    }
    
    print(f"\nSaved filtered data to: {run_group.path}")
    print(f"  Coverage: {coverage:.1f}%")
    if drop_outliers:
        print(f"  Removed {len(results['outlier_indices'])} outlier detections")
    
    return run_group.path


def plot_analysis(results: Dict, save_path: Optional[str] = None):
    """
    Create visualization of frame distance analysis.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Distance over time
    ax = axes[0]
    ax.plot(results['distances'], alpha=0.6, label='Frame-to-frame distance')
    ax.axhline(y=results['threshold'], color='r', linestyle='--', 
               label=f'Threshold ({results["threshold"]:.0f} px)')
    
    # Mark outliers
    if len(results['outlier_indices']) > 0:
        outlier_distances = results['distances'][results['outlier_indices'] - 1]
        ax.scatter(results['outlier_indices'] - 1, outlier_distances, 
                  color='red', s=50, zorder=5, label='Outliers')
    
    ax.set_xlabel('Detection pair index')
    ax.set_ylabel('Distance (pixels)')
    ax.set_title('Frame-to-Frame Movement Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Trajectory
    ax = axes[1]
    centroids = results['centroids']
    ax.plot(centroids[:, 0], centroids[:, 1], 'b-', alpha=0.5, linewidth=0.5)
    ax.scatter(centroids[:, 0], centroids[:, 1], c=range(len(centroids)), 
              cmap='viridis', s=5)
    
    # Mark outliers on trajectory
    if len(results['jumps']) > 0:
        jump_points = centroids[results['jumps'] - 1]
        ax.scatter(jump_points[:, 0], jump_points[:, 1], 
                  color='red', s=100, marker='x', label='Jumps')
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Fish Trajectory')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    if len(results['jumps']) > 0:
        ax.legend()
    
    # Plot 3: Coverage timeline
    ax = axes[2]
    frame_coverage = results['n_detections'].astype(bool)
    ax.imshow([frame_coverage], aspect='auto', cmap='RdYlGn', 
              interpolation='nearest')
    ax.set_xlabel('Frame')
    ax.set_title('Detection Coverage (Green = detected, Red = missing)')
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze frame-to-frame distances in multi-fish tracker data'
    )
    parser.add_argument('zarr_path', help='Path to multi-fish zarr file')
    parser.add_argument('--threshold', type=float, default=100.0,
                       help='Distance threshold for outlier detection (pixels)')
    parser.add_argument('--fish-id', type=int, default=None,
                       help='Analyze specific fish ID (if id_assignments available)')
    parser.add_argument('--drop', action='store_true',
                       help='Drop outlier detections')
    parser.add_argument('--save', action='store_true',
                       help='Save filtered data back to zarr')
    parser.add_argument('--plot', action='store_true',
                       help='Generate analysis plots')
    parser.add_argument('--output-plot', type=str, default=None,
                       help='Path to save plot')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MULTI-FISH FRAME-TO-FRAME DISTANCE ANALYZER")
    print("=" * 60)
    print(f"Zarr file: {args.zarr_path}")
    print(f"Distance threshold: {args.threshold} pixels")
    if args.fish_id is not None:
        print(f"Fish ID filter: {args.fish_id}")
    
    try:
        # Run analysis
        results = analyze_frame_distances(
            args.zarr_path,
            threshold=args.threshold,
            fish_id=args.fish_id,
            verbose=True
        )
        
        # Save filtered data if requested
        if args.save:
            save_filtered_data(
                args.zarr_path,
                results,
                drop_outliers=args.drop,
                fish_id=args.fish_id
            )
        
        # Generate plots if requested
        if args.plot or args.output_plot:
            plot_analysis(results, args.output_plot)
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())