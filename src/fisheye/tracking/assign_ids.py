# src/fisheye/tracking/assign_ids.py
"""
Fish ID assignment for multi-fish tracking.

Currently implements spatial assignment (one fish per ROI).
TODO: Add trajectory-based tracking for free-swimming fish.
"""

import numpy as np
import zarr
import time
import sys
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from ..shared.zarr.schema import get_run_group
from ..utils.system import get_environment_info


def get_single_dish_roi_from_mask(root: zarr.Group, console: Console) -> Optional[List[Dict]]:
    """
    Create a single ROI from the dish mask for single-fish experiments.
    
    Args:
        root: Zarr root group
        console: Rich console for output
        
    Returns:
        List with single ROI dictionary, or None if no dish mask found
    """
    # Try to get dish mask from analysis_metadata
    if 'analysis_metadata' not in root:
        return None
    
    analysis_meta = root['analysis_metadata']
    
    # Check for tuned dish mask
    if 'dish_mask' in analysis_meta.attrs:
        mask_data = analysis_meta.attrs['dish_mask']
        
        if 'detected_circle' in mask_data:
            circle = mask_data['detected_circle']
            center = circle.get('center', [0, 0])
            radius = circle.get('radius', 0)
            
            # Convert circle to bounding box
            x = int(center[0] - radius)
            y = int(center[1] - radius)
            w = int(radius * 2)
            h = int(radius * 2)
            
            console.print(f"[green]✓ Using dish mask as single ROI[/green]")
            console.print(f"  Circle center: {center}, radius: {radius}")
            console.print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")
            
            return [{
                'id': 0,
                'roi_pixels': [x, y, w, h],
                'source': 'dish_mask_circle'
            }]
    
    # Try detection config as fallback
    # Look for detect_runs to get dish_mask parameters
    if 'detect_runs' in root:
        latest_detect = root['detect_runs'].attrs.get('latest')
        if latest_detect:
            detect_group = root[f'detect_runs/{latest_detect}']
            if 'parameters' in detect_group.attrs:
                params = detect_group.attrs['parameters']
                dish_mask = params.get('dish_mask', {})
                
                if dish_mask.get('shape') == 'rectangle' and 'roi' in dish_mask:
                    x, y, w, h = dish_mask['roi']
                    console.print(f"[green]✓ Using rectangular dish mask as single ROI[/green]")
                    console.print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")
                    
                    return [{
                        'id': 0,
                        'roi_pixels': [x, y, w, h],
                        'source': 'dish_mask_rectangle'
                    }]
                
                elif dish_mask.get('shape') == 'circle':
                    center = dish_mask.get('center', [0, 0])
                    radius = dish_mask.get('radius', 0)
                    
                    x = int(center[0] - radius)
                    y = int(center[1] - radius)
                    w = int(radius * 2)
                    h = int(radius * 2)
                    
                    console.print(f"[green]✓ Using circular dish mask as single ROI[/green]")
                    console.print(f"  Circle center: {center}, radius: {radius}")
                    console.print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")
                    
                    return [{
                        'id': 0,
                        'roi_pixels': [x, y, w, h],
                        'source': 'dish_mask_circle'
                    }]
    
    return None


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
    
    For single-dish experiments (num_dishes=1), automatically uses the
    dish mask as a single ROI without requiring sub-dish tuning.
    
    For multi-dish experiments (num_dishes>1), requires sub-dish ROI
    definitions from tuning or config.
    
    Args:
        zarr_path: Path to zarr archive
        config: Pipeline configuration dictionary
        console: Rich console for output
        
    Returns:
        Dictionary with summary statistics
        
    Note:
        Checks experiment_setup metadata to determine single vs multi-dish mode.
        
        Single-dish mode uses dish mask automatically.
        Multi-dish mode requires 'subdish_mask_tuning' in zarr analysis_metadata OR
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
    
    # Check experiment setup metadata
    experiment_setup = root.attrs.get('experiment_setup', {})
    setup_type = experiment_setup.get('setup_type', 'unknown')
    num_dishes = experiment_setup.get('num_dishes', 0)
    
    if experiment_setup:
        console.print(f"[cyan]Experiment setup detected:[/cyan]")
        console.print(f"  Setup type: {setup_type}")
        console.print(f"  Dishes: {num_dishes}")
        console.print(f"  Fish per dish: {experiment_setup.get('fish_per_dish', '?')}")
        console.print(f"  Expected total: {experiment_setup.get('total_expected_fish', '?')}")
        console.print()
    else:
        console.print("[yellow]No experiment setup metadata found.[/yellow]")
        console.print("[yellow]Run: python setup_experiment_metadata.py data.zarr --interactive[/yellow]")
        console.print("[yellow]Defaulting to multi-dish mode...[/yellow]\n")
        setup_type = 'multi_dish'
    
    # Get sub-dish masks based on setup type
    subdish_masks = None
    param_source = 'none'
    
    # SINGLE-DISH MODE: Use dish mask as single ROI
    if setup_type == 'single_dish':
        console.print("[bold cyan]Single-dish mode:[/bold cyan] Using dish mask as single ROI")
        subdish_masks = get_single_dish_roi_from_mask(root, console)
        
        if subdish_masks:
            param_source = 'dish_mask_auto'
        else:
            console.print("[yellow]Warning: Could not extract dish mask for single-dish mode.[/yellow]")
            console.print("[yellow]Make sure dish mask is tuned: python -m fisheye data.zarr --tune mask[/yellow]")
            return {'total_detections': 0, 'assigned': 0, 'unassigned': 0}
    
    # MULTI-DISH MODE: Require sub-dish ROI definitions
    else:
        console.print("[bold cyan]Multi-dish mode:[/bold cyan] Loading sub-dish ROI definitions")
        
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
    
    # Validate expected vs actual number of ROIs
    if experiment_setup and num_dishes > 0:
        expected_rois = num_dishes
        actual_rois = len(subdish_masks)
        
        if expected_rois != actual_rois:
            console.print(f"[yellow]  Warning: ROI count mismatch![/yellow]")
            console.print(f"  Expected {expected_rois} dishes, found {actual_rois} ROIs")
            console.print(f"  Proceeding with {actual_rois} ROIs...")
    
    console.print(f"\nAssigning IDs based on [cyan]{len(subdish_masks)}[/cyan] ROI(s) ({param_source})")
    
    # Create run group
    assign_group, run_group_name = get_run_group(root, 'id_assignment', console)
    
    # Get latest detection run
    latest_detect_run = root['detect_runs'].attrs['latest']
    detect_group = root[f'detect_runs/{latest_detect_run}']
    
    # Store metadata
    metadata_dict = {
        'assign_ids_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'parameters': {
            'num_masks': len(subdish_masks), 
            'masks': subdish_masks,
            'setup_type': setup_type,
            'experiment_setup': experiment_setup
        },
        'parameter_source': param_source,
        'source_detect_run': latest_detect_run,
        'assignment_method': 'spatial',
        'num_masks': len(subdish_masks)
    }
    
    assign_group.attrs.update(metadata_dict)
    
    # Load detection data
    frame_indices = detect_group['frame_indices'][:].astype(np.int64, copy=False)
    bbox_coords = detect_group['bbox_norm_coords'][:]
    if 'frame_counts' in detect_group:
        frame_counts = detect_group['frame_counts'][:]
        num_frames = len(frame_counts)
    else:
        num_frames = root['raw_video/images_ds'].shape[0]
        frame_counts = np.bincount(frame_indices, minlength=num_frames)
    if frame_indices.size > 0:
        max_frame = int(frame_indices.max()) + 1
        if max_frame > num_frames:
            frame_counts = np.pad(frame_counts, (0, max_frame - num_frames), mode='constant')
            num_frames = len(frame_counts)
    
    # Get image dimensions for coordinate conversion
    ds_img_shape = root['raw_video/images_ds'].shape[1:]  # (H, W)
    
    console.print(f"Processing [green]{len(bbox_coords)}[/green] detections...")
    
    # Convert normalized coordinates to pixel coordinates
    bbox_coords_px = bbox_coords.copy()
    bbox_coords_px[:, 0] *= ds_img_shape[1]  # center_x
    bbox_coords_px[:, 1] *= ds_img_shape[0]  # center_y
    
    # Initialize detection IDs array (-1 = unassigned)
    detection_ids = np.full(len(bbox_coords), -1, dtype=np.int32)
    
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
        overwrite=True
    )
    
    # Calculate per-mask detection counts per frame
    n_masks = len(subdish_masks)
    per_mask_counts = np.zeros((num_frames, n_masks), dtype=np.int32)
    mask_id_to_idx = {mask['id']: idx for idx, mask in enumerate(subdish_masks)}
    
    if detection_ids.size > 0:
        for mask_id, column_idx in mask_id_to_idx.items():
            mask_hits = detection_ids == mask_id
            if np.any(mask_hits):
                counts = np.bincount(frame_indices[mask_hits], minlength=num_frames)
                per_mask_counts[:, column_idx] = counts[:num_frames]
    
    # Save per-mask counts
    assign_group.create_array(
        'n_detections_per_mask',
        data=per_mask_counts,
        chunks=(min(100, num_frames) if num_frames > 0 else 1, n_masks),
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
        column_idx = mask_id_to_idx[mask_id]
        n_in_mask = np.sum(detection_ids == mask_id)
        frames_with_detections = int(np.sum(per_mask_counts[:, column_idx] > 0))
        mask_stats.append({
            'mask_id': mask_id,
            'total_detections': int(n_in_mask),
            'frames_with_detections': frames_with_detections,
            'coverage_percent': round((frames_with_detections / num_frames * 100), 2) if num_frames > 0 else 0.0
        })
    
    duration = time.perf_counter() - start_time
    
    summary_stats = {
        'total_detections': int(len(detection_ids)),
        'assigned_detections': int(n_assigned),
        'unassigned_detections': int(n_unassigned),
        'assignment_rate_percent': round(assignment_rate, 2),
        'num_masks': n_masks,
        'per_mask_statistics': mask_stats,
        'setup_type': setup_type,
        'total_frames': num_frames
    }
    
    assign_group.attrs['summary_statistics'] = summary_stats
    assign_group.attrs['duration_seconds'] = duration
    
    # Environment info and provenance
    env_info = get_environment_info()
    assign_group.attrs['provenance'] = {
        'command': ' '.join(sys.argv),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_detect_run': latest_detect_run
    }
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
        f"    {'Dish' if setup_type == 'single_dish' else 'Mask'} {s['mask_id']}: {s['total_detections']} detections ({s['coverage_percent']:.1f}% frames)"
        for s in mask_stats
    ])
    
    # Add validation info if experiment setup exists
    validation_text = ""
    if experiment_setup and num_dishes > 0:
        expected_fish = experiment_setup.get('total_expected_fish', 0)
        validation_text = f"\n[bold]Validation:[/bold]\n  Expected: {expected_fish} fish\n  Found: {n_masks} ROI(s)"
    
    completion_text = f"""[green]✓[/green] ID assignment completed

[bold]Setup:[/bold]
  Mode: {setup_type}
  Source: {param_source}

[bold]Performance:[/bold]
  Time: {duration:.1f}s

[bold]Results:[/bold]
  Assigned: {n_assigned}/{len(detection_ids)} ({assignment_rate:.1f}%)
  Unassigned: {n_unassigned}
{validation_text}
  
[bold]Per-ROI Summary:[/bold]
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
