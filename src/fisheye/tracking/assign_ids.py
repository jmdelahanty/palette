# src/fisheye/tracking/assign_ids.py
"""
Fish ID assignment for multi-fish tracking.

Currently implements spatial assignment (one fish per ROI).
TODO: Add trajectory-based tracking for free-swimming fish.
"""

import numpy as np
import zarr
import time
from typing import Dict, Optional, Any
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from ..shared.zarr.schema import get_run_group
from ..utils.system import get_environment_info


def assign_ids_spatial(
    zarr_path: str,
    config: Dict[str, Any],
    console: Optional[Console] = None
) -> Dict[str, Any]:
    """
    Assign IDs to detections based on spatial location (sub-dish ROIs).
    
    This method assigns a unique ID to each detection based on which
    predefined ROI it falls into. Suitable for experiments where fish
    are confined to specific regions and don't move between ROIs.
    
    Args:
        zarr_path: Path to zarr archive
        config: Pipeline configuration dictionary
        console: Rich console for output
        
    Returns:
        Dictionary with summary statistics
        
    Note:
        Requires 'subdish_mask_tuning' in zarr analysis_metadata OR
        'sub_dish_rois' in config['assign_ids'] with format:
        [
            {'id': 0, 'roi_pixels': [x, y, w, h]},
            {'id': 1, 'roi_pixels': [x, y, w, h]},
            ...
        ]
    """
    if console is None:
        console = Console()
    
    console.rule("[bold]Stage: Spatial ID Assignment[/bold]")
    start_time = time.perf_counter()
    
    root = zarr.open(zarr_path, mode='a')
    
    # Check prerequisites
    if 'detect_runs' not in root:
        raise ValueError("Detection stage not run. Run detect before assign_ids.")
    
    # Get sub-dish masks - Priority: zarr tuning > config
    subdish_masks = None
    param_source = 'none'
    
    # Priority 1: Check for tuned sub-dish masks in zarr
    if 'analysis_metadata' in root:
        analysis_meta = root['analysis_metadata']
        if 'subdish_mask_tuning' in analysis_meta.attrs:
            mask_data = analysis_meta.attrs['subdish_mask_tuning']
            subdish_masks = mask_data['masks']
            param_source = 'zarr_tuned'
            console.print(f"[green]✓ Using tuned sub-dish masks from zarr[/green]")
            console.print(f"  Tuned on: {mask_data.get('tuned_timestamp', 'unknown')}")
    
    # Priority 2: Fall back to config
    if subdish_masks is None:
        assign_params = config.get('assign_ids', {})
        if assign_params and 'sub_dish_rois' in assign_params:
            subdish_masks = assign_params['sub_dish_rois']
            param_source = 'config'
            console.print(f"[yellow]Using sub-dish ROIs from config[/yellow]")
    
    # No masks found
    if subdish_masks is None:
        console.print("[yellow]Warning: No sub-dish masks defined.[/yellow]")
        console.print("[yellow]Run the tuner first: python -m fisheye data.zarr --tune subdish[/yellow]")
        console.print("[yellow]Or add 'sub_dish_rois' to config under 'assign_ids'[/yellow]")
        return {'total_detections': 0, 'assigned': 0, 'unassigned': 0}
    
    console.print(f"Assigning IDs based on [cyan]{len(subdish_masks)}[/cyan] sub-dish masks ({param_source})")
    
    # Create run group
    assign_group, run_group_name = get_run_group(root, 'id_assignment', console)
    
    # Get latest detection run
    latest_detect_run = root['detect_runs'].attrs['latest']
    detect_group = root[f'detect_runs/{latest_detect_run}']
    
    # Store metadata
    metadata_dict = {
        'assignment_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'parameters': {'num_masks': len(subdish_masks), 'masks': subdish_masks},
        'parameter_source': param_source,
        'source_detect_run': latest_detect_run,
        'assignment_method': 'spatial',
        'num_masks': len(subdish_masks)
    }
    
    assign_group.attrs.update(metadata_dict)
    
    # Load detection data
    bbox_coords = detect_group['bbox_norm_coords'][:]
    n_detections = detect_group['n_detections'][:]
    
    # Get image dimensions for coordinate conversion
    ds_img_shape = root['raw_video/images_ds'].shape[1:]  # (H, W)
    
    console.print(f"Processing [green]{len(bbox_coords)}[/green] detections...")
    
    # Convert normalized coordinates to pixel coordinates
    bbox_coords_px = bbox_coords.copy()
    bbox_coords_px[:, 0] *= ds_img_shape[1]  # center_x
    bbox_coords_px[:, 1] *= ds_img_shape[0]  # center_y
    
    # Initialize detection IDs array (-1 = unassigned)
    detection_ids = np.full(len(bbox_coords), -1, dtype='i4')
    
    # Assign IDs based on which sub-dish mask each detection falls into
    for mask in subdish_masks:
        mask_id = mask['id']
        x, y, w, h = mask['roi_pixels']
        
        # Check which detections fall within this sub-dish mask
        in_mask = (
            (bbox_coords_px[:, 0] >= x) & 
            (bbox_coords_px[:, 0] < x + w) &
            (bbox_coords_px[:, 1] >= y) & 
            (bbox_coords_px[:, 1] < y + h)
        )
        
        detection_ids[in_mask] = mask_id
    
    # Save detection IDs
    assign_group.create_array(
        'detection_ids',
        data=detection_ids,
        chunks=(min(1000, len(detection_ids)),),
        dtype='i4',
        overwrite=True
    )
    
    # Calculate per-mask detection counts per frame
    n_frames = len(n_detections)
    n_masks = len(subdish_masks)
    per_mask_counts = np.zeros((n_frames, n_masks), dtype='i4')
    
    # Use cumulative detection indices to map detections to frames
    cumulative_detections = np.cumsum(np.insert(n_detections, 0, 0))
    
    for frame_idx in range(n_frames):
        start_idx = cumulative_detections[frame_idx]
        end_idx = cumulative_detections[frame_idx + 1]
        
        if end_idx > start_idx:
            frame_ids = detection_ids[start_idx:end_idx]
            
            for mask_id in range(n_masks):
                per_mask_counts[frame_idx, mask_id] = np.sum(frame_ids == mask_id)
    
    # Save per-mask counts
    assign_group.create_array(
        'n_detections_per_mask',
        data=per_mask_counts,
        chunks=(min(100, n_frames), None),
        dtype='i4',
        overwrite=True
    )
    
    # Store sub-dish mask definitions for reference
    assign_group.attrs['subdish_masks'] = subdish_masks
    
    # Calculate statistics
    n_assigned = np.sum(detection_ids != -1)
    n_unassigned = np.sum(detection_ids == -1)
    assignment_rate = (n_assigned / len(detection_ids) * 100) if len(detection_ids) > 0 else 0
    
    # Per-mask statistics
    mask_stats = []
    for mask in subdish_masks:
        mask_id = mask['id']
        n_in_mask = np.sum(detection_ids == mask_id)
        frames_with_detections = np.sum(per_mask_counts[:, mask_id] > 0)
        mask_stats.append({
            'mask_id': mask_id,
            'total_detections': int(n_in_mask),
            'frames_with_detections': int(frames_with_detections),
            'coverage_percent': round((frames_with_detections / n_frames * 100), 2)
        })
    
    duration = time.perf_counter() - start_time
    
    summary_stats = {
        'total_detections': int(len(detection_ids)),
        'assigned_detections': int(n_assigned),
        'unassigned_detections': int(n_unassigned),
        'assignment_rate_percent': round(assignment_rate, 2),
        'num_masks': n_masks,
        'per_mask_statistics': mask_stats
    }
    
    assign_group.attrs['summary_statistics'] = summary_stats
    assign_group.attrs['duration_seconds'] = duration
    
    # Environment info
    env_info = get_environment_info()
    assign_group.attrs.update({
        'git_commit': env_info['git'].get('commit_hash', 'unknown'),
        'git_branch': env_info['git'].get('branch', 'unknown'),
        'hostname': env_info['platform']['hostname']
    })
    
    # Mark latest
    parent_group = root['id_assignment_runs']
    parent_group.attrs['latest'] = run_group_name
    
    # Completion panel
    mask_summary = "\n".join([
        f"    Mask {s['mask_id']}: {s['total_detections']} detections ({s['coverage_percent']:.1f}% frames)"
        for s in mask_stats
    ])
    
    completion_text = f"""[green]✓[/green] ID assignment completed

[bold]Performance:[/bold]
  Time: {duration:.1f}s

[bold]Results:[/bold]
  Assigned: {n_assigned}/{len(detection_ids)} ({assignment_rate:.1f}%)
  Unassigned: {n_unassigned}
  
[bold]Per-Mask Summary:[/bold]
{mask_summary}

[bold]Output:[/bold]
  Path: {zarr_path}
  Group: id_assignment_runs/{run_group_name}
  Arrays: detection_ids, n_detections_per_mask"""
    
    panel = Panel(
        completion_text,
        title="[bold]ID Assignment Complete[/bold]",
        border_style="green",
        padding=(1, 2)
    )
    
    console.print("\n")
    console.print(panel)
    
    return summary_stats


# TODO: Implement trajectory-based tracking for free-swimming fish OR INTEGRATE SLEAP
# 
# def assign_ids_trajectory(
#     zarr_path: str,
#     config: Dict[str, Any],
#     console: Optional[Console] = None
# ) -> Dict[str, Any]:
#     """
#     Assign IDs to detections using trajectory tracking.
#     
#     This method tracks individual fish across frames by matching
#     keypoints and building trajectories. Suitable for experiments
#     where fish swim freely and may cross paths.
#     
#     Approach:
#     1. Use keypoint features (position, heading, size) for matching
#     2. Implement Hungarian algorithm for frame-to-frame assignment
#     3. Handle track initiation, continuation, and termination
#     4. Resolve ID switches and occlusions
#     
#     Args:
#         zarr_path: Path to zarr archive
#         config: Pipeline configuration dictionary
#         console: Rich console for output
#         
#     Returns:
#         Dictionary with summary statistics
#     """
#     raise NotImplementedError(
#         "Trajectory-based tracking not yet implemented. "
#         "Use spatial assignment for now, or contribute an implementation!"
#     )


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Assign IDs to fish detections")
    parser.add_argument("zarr_path", help="Path to zarr archive")
    parser.add_argument("--config", default="configs/fisheye/default.yaml",
                       help="Configuration file")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    console = Console()
    
    # Run spatial ID assignment
    results = assign_ids_spatial(
        args.zarr_path,
        config,
        console=console
    )
    
    console.print(f"\n[green]Assigned IDs to {results['assigned_detections']} detections[/green]")