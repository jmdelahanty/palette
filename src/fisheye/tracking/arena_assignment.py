# src/fisheye/tracking/arena_assignment.py
"""
Arena assignment for multi-fish tracking.

Currently implements spatial arena assignment (one subject per ROI).
TODO: Add trajectory-based tracking for free-swimming fish within arenas.
"""

import numpy as np
import zarr
import time
import sys
from typing import Dict, Optional, Any, List, Tuple, Sequence
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from ..shared.experiment_setup import infer_experiment_setup
from ..registry.stage_complete import emit_stage_completion
from ..shared.run_provenance import build_run_provenance_from_stage_record
from ..shared.frame_domains import FrameDomain, FrameDomainError, FrameDomains
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.type_conversions import normalize_attr
from ..shared.zarr.schema import get_run_group
from ..shared.zarr_run_completion import mark_run_complete, mark_run_started, note_pending_latest
from ..shared.zarr_io import open_zarr_root
from .single_subject_per_arena import (
    TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA,
    write_single_subject_per_arena_tracking_run,
)
from ..shared.system_metadata import get_environment_info

_ARENA_ASSIGN_STATUS_SOURCE = "runtime_arena_assignment"


def _count_from_domains(
    root: zarr.Group,
    domain: FrameDomain,
    *,
    run_group: Optional[zarr.Group] = None,
) -> Optional[int]:
    try:
        return int(FrameDomains(root=root, run_group=run_group).count(domain))
    except FrameDomainError:
        return None


def _emit_tracking_step_statuses(
    *,
    root: zarr.Group,
    zarr_path: str,
    id_status: str,
    id_reason: str,
    id_run_name: Optional[str],
    id_method: Optional[str],
    id_coverage_pct: Optional[float],
    id_details: Dict[str, object],
    tracks_status: str,
    tracks_reason: str,
    tracks_run_name: Optional[str],
    tracks_method: Optional[str],
    tracks_details: Dict[str, object],
    console: Optional[Console],
) -> None:
    zarr_file = Path(zarr_path).expanduser().resolve()
    emit_stage_completion(
        root,
        zarr_file,
        step_name="arena_assignment",
        status=id_status,
        source=_ARENA_ASSIGN_STATUS_SOURCE,
        run_name=id_run_name,
        method=id_method,
        coverage_pct=id_coverage_pct,
        details_json={**id_details, "reason": id_reason},
        console=console,
        warning_label="arena_assignment/tracks",
        auto_registry_from_env=True,
        require_env_registry_exists=False,
        invalidate_on_ok=False,
    )
    emit_stage_completion(
        root,
        zarr_file,
        step_name="tracks",
        status=tracks_status,
        source=_ARENA_ASSIGN_STATUS_SOURCE,
        run_name=tracks_run_name,
        method=tracks_method,
        coverage_pct=None,
        details_json={**tracks_details, "reason": tracks_reason},
        console=console,
        warning_label="arena_assignment/tracks",
        auto_registry_from_env=True,
        require_env_registry_exists=False,
        invalidate_on_ok=False,
    )


def _infer_num_frames(
    root: zarr.Group, detection_group: zarr.Group, frame_indices: np.ndarray
) -> int:
    """Infer total frame count without requiring raw video arrays."""
    candidates: List[int] = []

    for key in ("n_frames", "total_frames", "source_total_frames"):
        value = detection_group.attrs.get(key)
        if isinstance(value, (int, np.integer)) and value > 0:
            candidates.append(int(value))

    params = detection_group.attrs.get("parameters")
    if isinstance(params, dict):
        for key in ("n_frames", "total_frames"):
            value = params.get(key)
            if isinstance(value, (int, np.integer)) and value > 0:
                candidates.append(int(value))

    for key in ("palette_total_frames", "total_frames", "n_frames"):
        value = root.attrs.get(key)
        if isinstance(value, (int, np.integer)) and value > 0:
            candidates.append(int(value))

    if "raw_video" in root and "images_ds" in root["raw_video"]:
        stored_count = _count_from_domains(root, FrameDomain.STORED_ZARR)
        candidates.append(stored_count if stored_count is not None else int(root["raw_video/images_ds"].shape[0]))

    if candidates:
        return max(candidates)

    if frame_indices.size:
        return int(frame_indices.max()) + 1

    raise ValueError(
        "Unable to infer total frame count; detection metadata missing 'frame_counts' and video attributes."
    )


def _get_group_by_path(root: zarr.Group, path: str) -> zarr.Group:
    """Resolve a slash-delimited group path for zarr groups and test fakes."""

    current = root
    for part in path.split("/"):
        if not part:
            continue
        if part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _resolve_frame_shape(root: zarr.Group, detection_group: zarr.Group) -> Tuple[int, int]:
    """Return (height, width) using detection metadata or fallbacks."""

    def _from_resolution(value: Optional[Any]) -> Tuple[Optional[int], Optional[int]]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                h = int(value[0])
                w = int(value[1])
                if h > 0 and w > 0:
                    return h, w
            except (TypeError, ValueError):
                return None, None
        return None, None

    attrs = detection_group.attrs
    height = (
        attrs.get("inference_height")
        or attrs.get("source_video_height")
        or attrs.get("source_full_height")
        or attrs.get("height")
    )
    width = (
        attrs.get("inference_width")
        or attrs.get("source_video_width")
        or attrs.get("source_full_width")
        or attrs.get("width")
    )

    if not height or not width:
        for key in (
            "palette_downsampled_resolution",
            "palette_original_resolution",
            "inference_resolution",
            "source_full_resolution",
        ):
            res_h, res_w = _from_resolution(attrs.get(key))
            height = height or res_h
            width = width or res_w
            if height and width:
                break

    if not height or not width:
        root_attrs = root.attrs
        height = height or root_attrs.get("inference_height") or root_attrs.get("palette_video_height") or root_attrs.get("video_height")
        width = width or root_attrs.get("inference_width") or root_attrs.get("palette_video_width") or root_attrs.get("video_width")

        if not height or not width:
            for key in (
                "palette_downsampled_resolution",
                "palette_original_resolution",
                "inference_resolution",
                "source_video_resolution",
                "source_full_resolution",
            ):
                res_h, res_w = _from_resolution(root_attrs.get(key))
                height = height or res_h
                width = width or res_w
                if height and width:
                    break

    if not height or not width:
        if "raw_video" in root and "images_ds" in root["raw_video"]:
            _, h, w = root["raw_video/images_ds"].shape
            return int(h), int(w)
        raise ValueError(
            "Unable to determine frame dimensions; detection run missing inference/source resolution metadata."
        )

    return int(height), int(width)


def _resolve_assignment_shape(
    root: zarr.Group,
    detection_group: zarr.Group,
    arena_masks: Sequence[Dict[str, Any]],
) -> Tuple[int, int]:
    """Return the pixel coordinate shape used by arena mask definitions."""

    for mask in arena_masks:
        image_shape = mask.get("image_shape")
        if isinstance(image_shape, (list, tuple)) and len(image_shape) >= 2:
            try:
                height = int(image_shape[0])
                width = int(image_shape[1])
            except (TypeError, ValueError):
                continue
            if height > 0 and width > 0:
                return height, width
    return _resolve_frame_shape(root, detection_group)


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
        mask_attr = analysis_meta.attrs['dish_mask']
        mask_data = dict(mask_attr)
        shape = mask_data.get('shape')
        metrics = mask_data.get("metrics") if isinstance(mask_data.get("metrics"), dict) else {}
        image_shape = metrics.get("image_shape")
        if not shape:
            if 'detected_circle' in mask_data:
                shape = 'circle'
            elif 'rectangle' in mask_data:
                shape = 'rectangle'
        
        if shape == 'rectangle' and 'rectangle' in mask_data:
            roi = mask_data['rectangle'].get('roi')
            if roi:
                x, y, w, h = [int(v) for v in roi]
                console.print(f"[green]✓ Using dish mask rectangle as single ROI[/green]")
                console.print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")
                return [{
                    'id': 0,
                    'roi_pixels': [x, y, w, h],
                    'source': 'dish_mask_rectangle',
                    'image_shape': image_shape,
                }]
        
        if shape == 'circle' and 'detected_circle' in mask_data:
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
                'source': 'dish_mask_circle',
                'image_shape': image_shape,
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


def _missing_assignment_result(*, setup_type: str, reason: str) -> Dict[str, Any]:
    """Return the standard summary shape for prerequisite-missing exits."""

    return {
        "status": "missing",
        "reason": reason,
        "setup_type": setup_type,
        "total_detections": 0,
        "assigned_detections": 0,
        "unassigned_detections": 0,
        "assignment_rate_percent": 0.0,
        "num_arenas": 0,
        # Legacy aliases for callers that predate the canonical names.
        "assigned": 0,
        "unassigned": 0,
    }


def assign_arenas_spatial(
    zarr_path: str,
    config: Dict[str, Any],
    console: Optional[Console] = None,
    source_rowset_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assign detections to arenas based on spatial location (sub-dish ROIs).
    
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
        source_rowset_path: Optional exact rowset to assign. Use this when a
            downstream crop/keypoint rowset, not refined-detect instances, is
            the row identity that consumers must track.
        
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
    
    console.rule("[bold]Stage: Spatial Arena Assignment[/bold]")
    start_time = time.perf_counter()
    
    root = open_zarr_root(zarr_path, mode='a')

    def _resolve_tracks_status(id_status: str) -> Tuple[str, str, Optional[str], Optional[str]]:
        tracks_parent = root.get("tracking_runs")
        if tracks_parent is not None:
            latest_track = normalize_attr(tracks_parent.attrs.get("latest")) if hasattr(tracks_parent, "attrs") else None
            if latest_track and latest_track in tracks_parent:
                track_group = tracks_parent[latest_track]
                track_method = normalize_attr(track_group.attrs.get("method")) if hasattr(track_group, "attrs") else None
                return "ok", "present", latest_track, track_method
        if id_status == "ok":
            return "missing", "run_missing", None, None
        return "absent", "upstream_missing", None, None
    
    # Check prerequisites
    if 'detect_runs' not in root:
        raise ValueError("Detection stage not run. Run detect before arena assignment.")
    
    # Check experiment setup metadata
    experiment_setup = root.attrs.get('experiment_setup', {})
    setup_info = infer_experiment_setup(root.attrs)
    setup_type = setup_info.setup_type
    num_dishes = setup_info.num_dishes or 0

    if setup_info.source == "experiment_setup" and experiment_setup:
        console.print(f"[cyan]Experiment setup detected:[/cyan]")
        console.print(f"  Setup type: {setup_type}")
        console.print(f"  Dishes: {num_dishes}")
        console.print(f"  Fish per dish: {experiment_setup.get('fish_per_dish', '?')}")
        console.print(f"  Expected total: {experiment_setup.get('total_expected_fish', '?')}")
        console.print()
    elif setup_info.source == "experimental_chamber":
        console.print("[yellow]No experiment_setup metadata found; using experimental_chamber to infer single-dish mode.[/yellow]")
        console.print("[yellow]Consider backfilling experiment_setup for future runs.[/yellow]")
        console.print()
    else:
        console.print("[yellow]No experiment setup metadata found.[/yellow]")
        console.print("[yellow]Run: python setup_experiment_metadata.py data.zarr --interactive[/yellow]")
        console.print("[yellow]Defaulting to multi-dish mode...[/yellow]\n")
        setup_type = 'multi_dish'

    if setup_type not in {'single_dish', 'multi_dish'}:
        console.print("[yellow]Unknown setup type; defaulting to multi-dish mode.[/yellow]\n")
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
            tracks_status, tracks_reason, tracks_run_name, tracks_method = _resolve_tracks_status("missing")
            _emit_tracking_step_statuses(
                root=root,
                zarr_path=zarr_path,
                id_status="missing",
                id_reason="dish_mask_missing",
                id_run_name=None,
                id_method="spatial",
                id_coverage_pct=0.0,
                id_details={"setup_type": setup_type},
                tracks_status=tracks_status,
                tracks_reason=tracks_reason,
                tracks_run_name=tracks_run_name,
                tracks_method=tracks_method,
                tracks_details={"upstream": {"arena_assignment": "missing"}},
                console=console,
            )
            return _missing_assignment_result(
                setup_type=setup_type,
                reason="dish_mask_missing",
            )
    
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
            tracks_status, tracks_reason, tracks_run_name, tracks_method = _resolve_tracks_status("missing")
            _emit_tracking_step_statuses(
                root=root,
                zarr_path=zarr_path,
                id_status="missing",
                id_reason="subdish_masks_missing",
                id_run_name=None,
                id_method="spatial",
                id_coverage_pct=0.0,
                id_details={"setup_type": setup_type},
                tracks_status=tracks_status,
                tracks_reason=tracks_reason,
                tracks_run_name=tracks_run_name,
                tracks_method=tracks_method,
                tracks_details={"upstream": {"arena_assignment": "missing"}},
                console=console,
            )
            return _missing_assignment_result(
                setup_type=setup_type,
                reason="subdish_masks_missing",
            )
    
    # Validate expected vs actual number of ROIs
    if experiment_setup and num_dishes > 0:
        expected_rois = num_dishes
        actual_rois = len(subdish_masks)
        
        if expected_rois != actual_rois:
            console.print(f"[yellow]  Warning: ROI count mismatch![/yellow]")
            console.print(f"  Expected {expected_rois} dishes, found {actual_rois} ROIs")
            console.print(f"  Proceeding with {actual_rois} ROIs...")
    
    console.print(f"\nAssigning arenas based on [cyan]{len(subdish_masks)}[/cyan] ROI(s) ({param_source})")
    
    # Create run group
    assign_group, run_group_name = get_run_group(root, 'arena_assignment', console)
    arena_parent = root.get("arena_assignment_runs")
    if arena_parent is not None:
        mark_run_started(assign_group, run_name=run_group_name, stage="arena_assignment")
        note_pending_latest(arena_parent, run_group_name)
    
    # Select detection data, preferring the canonical sparse refined surface when available.
    refined_parent = root.get('refined_detect_runs')
    refined_run_name = None
    detection_group = None
    source_detect_run = None
    assignment_source = 'detect_raw'
    selected_source_rowset_path = source_rowset_path
    
    if source_rowset_path:
        detection_group = _get_group_by_path(root, source_rowset_path)
        parts = source_rowset_path.split("/")
        head = parts[0] if parts else ""
        if head == "crop_runs":
            refined_run_name = detection_group.attrs.get("source_refined_run")
            source_detect_run = detection_group.attrs.get("source_detect_run")
            assignment_source = "explicit_crop_rows"
        elif head == "refined_detect_runs":
            if len(parts) < 2:
                raise ValueError(f"Malformed refined source rowset path: {source_rowset_path}")
            refined_run_name = parts[1]
            refined_parent_group = _get_group_by_path(root, "/".join(parts[:2]))
            source_detect_run = refined_parent_group.attrs.get("source_detect_run")
            assignment_source = "explicit_refined_rows"
        elif head == "detect_runs":
            if len(parts) < 2:
                raise ValueError(f"Malformed detect source rowset path: {source_rowset_path}")
            source_detect_run = parts[1]
            assignment_source = "explicit_detect_rows"
        else:
            raise ValueError(
                "Unsupported --source-rowset. Expected crop_runs/, detect_runs/, "
                f"or refined_detect_runs/ path (got {source_rowset_path})."
            )
        if not source_detect_run:
            raise ValueError(
                f"Unable to determine source_detect_run for source rowset {source_rowset_path}."
            )
        console.print(f"[cyan]Using explicit arena-assignment rowset:[/cyan] {source_rowset_path}")
        if refined_run_name:
            console.print(f"  Source refined run: {refined_run_name}")
        console.print(f"  Source detect run: {source_detect_run}")

    if detection_group is None and refined_parent is not None:
        candidate_run = refined_parent.attrs.get('latest')
        if candidate_run and candidate_run in refined_parent:
            candidate_group = refined_parent[candidate_run]
            if 'instances' in candidate_group:
                refined_run_name = candidate_run
                detection_group = candidate_group['instances']
                source_detect_run = candidate_group.attrs.get('source_detect_run')
                assignment_source = 'refined_instances'
                selected_source_rowset_path = f"refined_detect_runs/{refined_run_name}/instances"
                console.print(f"[cyan]Using refined detections:[/cyan] refined_detect_runs/{refined_run_name} (instances)")
                if source_detect_run:
                    console.print(f"  Source detect run: {source_detect_run}")
            else:
                console.print(f"[yellow]Refined run '{candidate_run}' missing 'instances' subgroup; falling back to raw detections.[/yellow]")
    
    if detection_group is None:
        if 'detect_runs' not in root:
            raise ValueError("Detection stage not run. Run detect before arena assignment.")
        detect_parent = root['detect_runs']
        latest_detect_run = detect_parent.attrs.get('latest')
        if not latest_detect_run or latest_detect_run not in detect_parent:
            raise ValueError("No valid detection runs found for arena assignment.")
        detection_group = detect_parent[latest_detect_run]
        source_detect_run = latest_detect_run
        assignment_source = 'detect_raw'
        selected_source_rowset_path = f"detect_runs/{source_detect_run}"
        console.print(f"[cyan]Using raw detections:[/cyan] detect_runs/{source_detect_run}")
    else:
        if not source_detect_run:
            if 'detect_runs' in root and root['detect_runs'].attrs.get('latest'):
                source_detect_run = root['detect_runs'].attrs['latest']
            else:
                raise ValueError("Unable to determine source detect run for refined detections.")
    
    # Store metadata
    metadata_dict = {
        'arena_assignment_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'spatial',
        'parameters': {
            'num_arenas': len(subdish_masks),
            'arenas': subdish_masks,
            'setup_type': setup_type,
            'experiment_setup': experiment_setup
        },
        'parameter_source': param_source,
        'source_detect_run': source_detect_run,
        'source_rowset_path': selected_source_rowset_path,
        'assignment_method': 'spatial',
        'assignment_source': assignment_source,
        'num_arenas': len(subdish_masks)
    }
    if refined_run_name:
        metadata_dict['source_refined_run'] = refined_run_name
    
    assign_group.attrs.update(metadata_dict)
    
    # Load detection data
    frame_indices = detection_group['frame_indices'][:].astype(np.int64, copy=False)
    bbox_coords = detection_group['bbox_norm_coords'][:]
    if 'frame_counts' in detection_group:
        frame_counts = detection_group['frame_counts'][:]
        run_frame_count = _count_from_domains(root, FrameDomain.RUN_FRAME, run_group=detection_group)
        num_frames = run_frame_count if run_frame_count is not None else len(frame_counts)
    else:
        num_frames = _infer_num_frames(root, detection_group, frame_indices)
        frame_counts = np.bincount(frame_indices, minlength=num_frames)
    if frame_indices.size > 0:
        max_frame = int(frame_indices.max()) + 1
        if max_frame > num_frames:
            frame_counts = np.pad(frame_counts, (0, max_frame - num_frames), mode='constant')
            num_frames = len(frame_counts)
    
    # Get image dimensions for coordinate conversion. Dish/subdish masks are
    # stored in the coordinate system they were tuned in, which can differ from
    # the source rowset metadata for crop-backed rowsets.
    height_px, width_px = _resolve_assignment_shape(root, detection_group, subdish_masks)
    ds_img_shape = (height_px, width_px)
    
    console.print(f"Processing [green]{len(bbox_coords)}[/green] detections...")
    
    # Convert normalized coordinates to pixel coordinates
    bbox_coords_px = bbox_coords.copy()
    bbox_coords_px[:, 0] *= ds_img_shape[1]  # center_x
    bbox_coords_px[:, 1] *= ds_img_shape[0]  # center_y
    
    # Initialize arena IDs array (-1 = unassigned)
    arena_ids = np.full(len(bbox_coords), -1, dtype=np.int32)
    
    # Assign arena IDs based on which sub-dish mask each detection falls into
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
        
        arena_ids[in_mask] = mask_id
    
    # Save arena IDs
    assign_group.create_array(
        'arena_ids',
        data=arena_ids,
        chunks=(min(1000, len(arena_ids)),),
        overwrite=True
    )
    
    # Calculate per-arena detection counts per frame
    n_masks = len(subdish_masks)
    per_mask_counts = np.zeros((num_frames, n_masks), dtype=np.int32)
    mask_id_to_idx = {mask['id']: idx for idx, mask in enumerate(subdish_masks)}
    
    if arena_ids.size > 0:
        for mask_id, column_idx in mask_id_to_idx.items():
            mask_hits = arena_ids == mask_id
            if np.any(mask_hits):
                counts = np.bincount(frame_indices[mask_hits], minlength=num_frames)
                per_mask_counts[:, column_idx] = counts[:num_frames]
    
    # Save per-arena counts
    assign_group.create_array(
        'n_detections_per_arena',
        data=per_mask_counts,
        chunks=(min(100, num_frames) if num_frames > 0 else 1, n_masks),
        overwrite=True
    )
    
    # Store arena definitions for reference
    assign_group.attrs['arena_definitions'] = subdish_masks
    
    # Calculate statistics
    n_assigned = np.sum(arena_ids != -1)
    n_unassigned = np.sum(arena_ids == -1)
    assignment_rate = (n_assigned / len(arena_ids) * 100) if len(arena_ids) > 0 else 0
    
    # Per-mask statistics
    mask_stats = []
    for mask in subdish_masks:
        mask_id = mask['id']
        column_idx = mask_id_to_idx[mask_id]
        n_in_mask = np.sum(arena_ids == mask_id)
        frames_with_detections = int(np.sum(per_mask_counts[:, column_idx] > 0))
        mask_stats.append({
            'arena_id': mask_id,
            'total_detections': int(n_in_mask),
            'frames_with_detections': frames_with_detections,
            'coverage_percent': round((frames_with_detections / num_frames * 100), 2) if num_frames > 0 else 0.0
        })
    
    duration = time.perf_counter() - start_time
    
    summary_stats = {
        'total_detections': int(len(arena_ids)),
        'assigned_detections': int(n_assigned),
        'unassigned_detections': int(n_unassigned),
        'assignment_rate_percent': round(assignment_rate, 2),
        'num_arenas': n_masks,
        'per_arena_statistics': mask_stats,
        'setup_type': setup_type,
        'total_frames': num_frames
    }
    
    assign_group.attrs['summary_statistics'] = summary_stats
    assign_group.attrs['duration_seconds'] = duration
    
    # Environment info and canonical stage provenance
    env_info = get_environment_info()
    provenance_inputs = {
        'source_detect_run': source_detect_run,
        'source_rowset_path': selected_source_rowset_path,
        'assignment_source': assignment_source,
    }
    if refined_run_name:
        provenance_inputs['source_refined_run'] = refined_run_name

    provenance_record = build_stage_provenance(
        stage='arena_assignment',
        command=' '.join(sys.argv),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        version=env_info['git'].get('short_hash') or env_info['git'].get('commit_hash'),
        git={
            'commit': env_info['git'].get('commit_hash'),
            'short': env_info['git'].get('short_hash'),
            'branch': env_info['git'].get('branch'),
            'is_dirty': env_info['git'].get('is_dirty'),
            'remote': env_info['git'].get('remote_url'),
        },
        environment=env_info.get('environment'),
        platform={
            'hostname': env_info['platform'].get('hostname'),
            'system': env_info['platform'].get('system'),
            'release': env_info['platform'].get('release'),
            'python_version': env_info['platform'].get('python_version'),
            'machine': env_info['platform'].get('machine'),
        },
        parameters={
            'assignment_method': 'spatial',
            'parameter_source': param_source,
            'num_arenas': len(subdish_masks),
            'setup_type': setup_type,
            'experiment_setup': experiment_setup,
        },
        inputs=provenance_inputs,
    )
    # Preserve legacy top-level provenance keys while adopting contract payload.
    provenance_record['source_detect_run'] = source_detect_run
    provenance_record['assignment_source'] = assignment_source
    if refined_run_name:
        provenance_record['source_refined_run'] = refined_run_name
    write_stage_provenance(assign_group, provenance_record)
    assign_group.attrs.update({
        'git_commit': env_info['git'].get('commit_hash', 'unknown'),
        'git_branch': env_info['git'].get('branch', 'unknown'),
        'hostname': env_info['platform']['hostname']
    })
    
    if arena_parent is not None:
        mark_run_complete(
            assign_group,
            parent_group=arena_parent,
            run_name=run_group_name,
            run_provenance=build_run_provenance_from_stage_record(provenance_record),
        )

    try:
        track_run_name, _track_group, tracking_summary = (
            write_single_subject_per_arena_tracking_run(
                root=root,
                arena_ids=arena_ids,
                frame_indices=frame_indices,
                source_detect_run=str(source_detect_run),
                source_refined_run=str(refined_run_name) if refined_run_name else None,
                source_arena_assignment_run=run_group_name,
                source_rowset_path=str(selected_source_rowset_path),
                conflict_policy="fail",
                console=console,
            )
        )
        tracks_status = "ok"
        tracks_reason = "present"
        tracks_method = TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA
    except Exception as exc:
        track_run_name = None
        tracking_summary = {}
        tracks_status = "error"
        tracks_reason = "tracking_generation_failed"
        tracks_method = TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA
        _emit_tracking_step_statuses(
            root=root,
            zarr_path=zarr_path,
            id_status="ok",
            id_reason="present",
            id_run_name=run_group_name,
            id_method="spatial",
            id_coverage_pct=float(summary_stats.get("assignment_rate_percent", 0.0)),
            id_details={
                "setup_type": setup_type,
                "assignment_source": assignment_source,
                "source_detect_run": source_detect_run,
                "source_refined_run": refined_run_name,
                "source_rowset_path": selected_source_rowset_path,
                "num_arenas": len(subdish_masks),
                "assigned_detections": int(n_assigned),
                "unassigned_detections": int(n_unassigned),
            },
            tracks_status=tracks_status,
            tracks_reason=tracks_reason,
            tracks_run_name=track_run_name,
            tracks_method=tracks_method,
            tracks_details={
                "upstream": {"arena_assignment": "ok"},
                "source_arena_assignment_run": run_group_name,
                "error": str(exc),
            },
            console=console,
        )
        raise

    # Completion panel
    mask_summary = "\n".join([
        f"    {'Dish' if setup_type == 'single_dish' else 'Arena'} {s['arena_id']}: {s['total_detections']} detections ({s['coverage_percent']:.1f}% frames)"
        for s in mask_stats
    ])
    
    # Add validation info if experiment setup exists
    validation_text = ""
    if experiment_setup and num_dishes > 0:
        expected_fish = experiment_setup.get('total_expected_fish', 0)
        validation_text = f"\n[bold]Validation:[/bold]\n  Expected: {expected_fish} fish\n  Found: {n_masks} ROI(s)"
    
    assignment_label = "refined/instances" if assignment_source == 'refined_instances' else 'raw detections'
    source_lines = [
        f"  Arena assignment source: {assignment_label}",
        f"  Detect run: detect_runs/{source_detect_run}"
    ]
    if refined_run_name:
        source_lines.append(f"  Refined run: refined_detect_runs/{refined_run_name}")
    source_block = "\n".join(source_lines)
    
    completion_text = f"""[green]✓[/green] Arena assignment completed
    
[bold]Setup:[/bold]
  Mode: {setup_type}
  ROI source: {param_source}

[bold]Data Source:[/bold]
{source_block}

[bold]Performance:[/bold]
  Time: {duration:.1f}s

[bold]Results:[/bold]
  Assigned: {n_assigned}/{len(arena_ids)} ({assignment_rate:.1f}%)
  Unassigned: {n_unassigned}
{validation_text}
  
[bold]Per-Arena Summary:[/bold]
{mask_summary}

[bold]Output:[/bold]
  Path: {zarr_path}
  Group: arena_assignment_runs/{run_group_name}
  Arrays: arena_ids, n_detections_per_arena
  Tracking: tracking_runs/{track_run_name} ({tracks_method})"""
    
    panel = Panel(
        completion_text,
        title="[bold]Arena Assignment Complete[/bold]",
        border_style="green",
        padding=(1, 2)
    )
    
    console.print("\n")
    console.print(panel)

    _emit_tracking_step_statuses(
        root=root,
        zarr_path=zarr_path,
        id_status="ok",
        id_reason="present",
        id_run_name=run_group_name,
        id_method="spatial",
        id_coverage_pct=float(summary_stats.get("assignment_rate_percent", 0.0)),
        id_details={
            "setup_type": setup_type,
            "assignment_source": assignment_source,
            "source_detect_run": source_detect_run,
            "source_refined_run": refined_run_name,
            "source_rowset_path": selected_source_rowset_path,
            "num_arenas": len(subdish_masks),
            "assigned_detections": int(n_assigned),
            "unassigned_detections": int(n_unassigned),
        },
        tracks_status=tracks_status,
        tracks_reason=tracks_reason,
        tracks_run_name=track_run_name,
        tracks_method=tracks_method,
        tracks_details={
            "upstream": {"arena_assignment": "ok"},
            "source_run": track_run_name,
            "source_arena_assignment_run": run_group_name,
            "num_tracks": int(tracking_summary.get("n_tracks", 0)),
            "n_assigned_rows": int(tracking_summary.get("n_assigned_rows", 0)),
            "n_unassigned_rows": int(tracking_summary.get("n_unassigned_rows", 0)),
            "unassigned_row_rate_percent": float(tracking_summary.get("unassigned_row_rate_percent", 0.0)),
            "tracking_qc_state": tracking_summary.get("tracking_qc_state"),
            "tracking_warn_threshold_rows": int(tracking_summary.get("tracking_warn_threshold_rows", 1)),
            "tracking_warn_threshold_percent": float(tracking_summary.get("tracking_warn_threshold_percent", 0.0)),
            "tracking_block_threshold_rows": int(tracking_summary.get("tracking_block_threshold_rows", 10)),
            "tracking_block_threshold_percent": float(tracking_summary.get("tracking_block_threshold_percent", 1.0)),
        },
        console=console,
    )
    
    return summary_stats


# TODO: Implement trajectory-based tracking for free-swimming fish OR INTEGRATE SLEAP
# 
# def assign_tracks_trajectory(
#     zarr_path: str,
#     config: Dict[str, Any],
#     console: Optional[Console] = None
# ) -> Dict[str, Any]:
#     """
#     Assign track IDs to detections using trajectory tracking.
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
    
    parser = argparse.ArgumentParser(description="Assign arena IDs to fish detections")
    parser.add_argument("zarr_path", help="Path to zarr archive")
    parser.add_argument("--config", default="configs/fisheye/default.yaml",
                       help="Configuration file")
    parser.add_argument(
        "--source-rowset",
        help=(
            "Exact rowset to assign, e.g. crop_runs/<run>. Use this when "
            "downstream keypoints/track kinematics are aligned to crop rows."
        ),
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    console = Console()
    
    # Run spatial arena assignment
    results = assign_arenas_spatial(
        args.zarr_path,
        config,
        console=console,
        source_rowset_path=args.source_rowset,
    )
    
    if results.get("status") == "missing":
        console.print(
            f"\n[yellow]Arena assignment not completed:[/yellow] "
            f"{results.get('reason', 'missing prerequisite')}"
        )
    else:
        console.print(
            f"\n[green]Assigned arenas to {results.get('assigned_detections', 0)} detections[/green]"
        )
