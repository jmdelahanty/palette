# src/fisheye/refinement/refine_detect.py
"""
Detection Refinement Pipeline

Filters and interpolates detection data based on quality labels.

Workflow:
1. Load detection data and quality labels
2. Filter: Remove jumps/artifacts (creates filtered/)
3. Interpolate: Fill gaps between clean detections (creates interpolated/)
4. Save both stages with full metadata and traceability
"""

import numpy as np
import zarr
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from rich.console import Console

from ..utils.metadata import get_total_frames, get_detection_method
from ..utils.system import get_environment_info, get_git_info

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"


def filter_detections(
    bbox_coords: np.ndarray,
    scores: np.ndarray,
    frame_indices: np.ndarray,
    detection_quality_labels: np.ndarray,
    num_frames: int,
    filters: List[str] = ['remove_jumps']
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Filter detections based on quality labels.
    
    Args:
        bbox_coords: Original bounding boxes (N, 4)
        scores: Original confidence scores (N,)
        frame_indices: Frame index for each detection (N,)
        detection_quality_labels: Quality label per detection (N,)
        num_frames: Total frame count for the source video
        filters: List of filters to apply
        
    Returns:
        Tuple of (filtered_bboxes, filtered_scores, frame_counts, 
                  frame_indices, drop_stats)
    """
    # Determine which detections to keep
    keep_mask = np.ones(len(detection_quality_labels), dtype=bool)
    drop_reasons = {}
    
    if 'remove_jumps' in filters:
        jump_mask = detection_quality_labels == 3
        keep_mask &= ~jump_mask
        drop_reasons['jumps'] = int(np.sum(jump_mask))
    
    if 'remove_blips' in filters:
        blip_mask = detection_quality_labels == 2
        keep_mask &= ~blip_mask
        drop_reasons['blips'] = int(np.sum(blip_mask))
    
    # Only keep clean detections (label=0)
    # This also excludes multi-detections (label=4) if present
    keep_mask = detection_quality_labels == 0
    
    # Apply filter
    filtered_bboxes = bbox_coords[keep_mask]
    filtered_scores = scores[keep_mask]
    filtered_frame_indices = frame_indices[keep_mask].astype('i4', copy=False)
    
    # Update per-frame detection counts
    filtered_counts = np.bincount(filtered_frame_indices, minlength=num_frames).astype('i4', copy=False)
    
    # Stats
    drop_stats = {
        'total_dropped': int(np.sum(~keep_mask)),
        'reasons': drop_reasons,
        'kept': int(np.sum(keep_mask)),
        'original': len(detection_quality_labels)
    }
    
    return filtered_bboxes, filtered_scores, filtered_counts, filtered_frame_indices, drop_stats


def find_gaps_to_interpolate(
    frame_indices: np.ndarray,
    max_gap: int = 20
) -> List[Dict]:
    """
    Find gaps between detections suitable for interpolation.
    
    Args:
        frame_indices: Frame index for each detection
        max_gap: Maximum gap size to interpolate
        
    Returns:
        List of gap dictionaries with start, end, size
    """
    if frame_indices.size == 0:
        return []
    
    detected_frames = np.unique(frame_indices.astype('i4', copy=False))
    
    if detected_frames.size < 2:
        return []
    
    gaps = []
    for i in range(detected_frames.size - 1):
        gap_start = int(detected_frames[i])
        gap_end = int(detected_frames[i + 1])
        gap_size = gap_end - gap_start - 1
        
        if 0 < gap_size <= max_gap:
            gaps.append({
                'start_frame': gap_start,
                'end_frame': gap_end,
                'size': gap_size,
                'fill_frames': list(range(gap_start + 1, gap_end))
            })
    
    return gaps


def interpolate_gap(
    bbox_start: np.ndarray,
    bbox_end: np.ndarray,
    score_start: float,
    score_end: float,
    gap_frames: List[int],
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate bounding boxes and scores across a gap.
    
    Args:
        bbox_start: Starting bbox [cx, cy, w, h]
        bbox_end: Ending bbox [cx, cy, w, h]
        score_start: Starting confidence score
        score_end: Ending confidence score
        gap_frames: Frame indices to fill
        method: Interpolation method ('linear')
        
    Returns:
        Tuple of (interpolated_bboxes, interpolated_scores)
    """
    n_interp = len(gap_frames)
    
    if method == 'linear':
        # Linear interpolation for each bbox component
        t = np.linspace(0, 1, n_interp + 2)[1:-1]  # Exclude endpoints
        
        interp_bboxes = np.zeros((n_interp, 4), dtype='f8')
        for i in range(4):
            interp_bboxes[:, i] = bbox_start[i] + t * (bbox_end[i] - bbox_start[i])
        
        # Score interpolation with decay toward gap center
        # Score is lowest at gap center, higher near endpoints
        gap_fraction = np.abs(t - 0.5) * 2  # 0 at center, 1 at edges
        min_score = min(score_start, score_end) * 0.5  # Minimum score
        interp_scores = min_score + gap_fraction * (min(score_start, score_end) - min_score)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return interp_bboxes, interp_scores


def interpolate_detections(
    filtered_bboxes: np.ndarray,
    filtered_scores: np.ndarray,
    filtered_frame_indices: np.ndarray,
    num_frames: int,
    max_gap: int = 20,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Interpolate gaps in filtered detections.
    
    Args:
        filtered_bboxes: Filtered bounding boxes
        filtered_scores: Filtered scores
        filtered_frame_indices: Frame index for each filtered detection
        num_frames: Total frame count for the source video
        max_gap: Maximum gap size to interpolate
        method: Interpolation method
        
    Returns:
        Tuple of (interp_bboxes, interp_scores, frame_counts,
                  frame_indices, detection_source, interp_stats)
    """
    # Find gaps
    gaps = find_gaps_to_interpolate(filtered_frame_indices, max_gap)
    
    if len(gaps) == 0:
        # No gaps to fill - return filtered data with source labels
        detection_source = np.zeros(len(filtered_bboxes), dtype='i1')
        stats = {
            'gaps_filled': 0,
            'interpolated_detections': 0,
            'mean_gap_size': 0.0,
            'max_gap_size': 0
        }
        filtered_counts = np.bincount(filtered_frame_indices, minlength=num_frames).astype('i4', copy=False)
        return (filtered_bboxes, filtered_scores, filtered_counts,
                filtered_frame_indices.astype('i4', copy=False), detection_source, stats)
    
    # Build index mapping for filtered data
    frame_to_idx = {}
    for idx, frame in enumerate(filtered_frame_indices):
        frame = int(frame)
        if frame not in frame_to_idx:
            frame_to_idx[frame] = idx
    
    # Collect all interpolated detections
    all_bboxes = [filtered_bboxes]
    all_scores = [filtered_scores]
    all_frame_indices = [filtered_frame_indices.astype('i4', copy=False)]
    all_sources = [np.zeros(len(filtered_bboxes), dtype='i1')]  # 0 = original
    
    total_interpolated = 0
    gap_sizes = []
    
    for gap in gaps:
        start_frame = gap['start_frame']
        end_frame = gap['end_frame']
        fill_frames = gap['fill_frames']
        
        # Get start and end detections
        start_idx = frame_to_idx[start_frame]
        end_idx = frame_to_idx[end_frame]
        
        bbox_start = filtered_bboxes[start_idx]
        bbox_end = filtered_bboxes[end_idx]
        score_start = filtered_scores[start_idx]
        score_end = filtered_scores[end_idx]
        
        # Interpolate
        interp_bboxes, interp_scores = interpolate_gap(
            bbox_start, bbox_end,
            score_start, score_end,
            fill_frames,
            method=method
        )
        
        all_bboxes.append(interp_bboxes)
        all_scores.append(interp_scores)
        all_frame_indices.append(np.array(fill_frames, dtype='i4'))
        all_sources.append(np.ones(len(fill_frames), dtype='i1'))  # 1 = interpolated
        
        total_interpolated += len(fill_frames)
        gap_sizes.append(gap['size'])
    
    # Concatenate all detections
    interp_bboxes = np.concatenate(all_bboxes, axis=0)
    interp_scores = np.concatenate(all_scores)
    interp_frame_indices = np.concatenate(all_frame_indices)
    detection_source = np.concatenate(all_sources)
    
    # Sort by frame to maintain temporal order
    sort_idx = np.argsort(interp_frame_indices)
    interp_bboxes = interp_bboxes[sort_idx]
    interp_scores = interp_scores[sort_idx]
    interp_frame_indices = interp_frame_indices[sort_idx]
    detection_source = detection_source[sort_idx]
    
    # Update n_detections
    interp_counts = np.bincount(interp_frame_indices, minlength=num_frames).astype('i4', copy=False)
    
    # Stats
    stats = {
        'gaps_filled': len(gaps),
        'interpolated_detections': int(total_interpolated),
        'mean_gap_size': float(np.mean(gap_sizes)) if gap_sizes else 0.0,
        'max_gap_size': int(max(gap_sizes)) if gap_sizes else 0,
        'min_gap_size': int(min(gap_sizes)) if gap_sizes else 0
    }
    
    return (interp_bboxes, interp_scores, interp_counts,
            interp_frame_indices.astype('i4', copy=False), detection_source, stats)


def get_refinement_parameters(
    config: Dict[str, Any],
    cli_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Get refinement parameters from config (no tuning step needed).
    
    Args:
        config: Loaded config dictionary
        cli_overrides: Optional CLI parameter overrides
        
    Returns:
        Tuple of (parameters dict, source string)
    """
    # Start with config defaults
    refine_params = config.get('refine_detect', {}).copy()
    refine_params.setdefault('filters', {'remove_jumps': True, 'remove_blips': False})
    refine_params.setdefault('max_gap', 20)
    refine_params.setdefault('interpolation_method', 'linear')
    
    # Apply CLI overrides if provided
    if cli_overrides:
        refine_params.update(cli_overrides)
        return refine_params, 'cli_override'
    
    return refine_params, 'config'


def create_refined_run(
    zarr_path: str,
    detect_run: Optional[str] = None,
    quality_run: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    max_gap: Optional[int] = None,
    interpolation_method: Optional[str] = None,
    remove_jumps: Optional[bool] = None,
    remove_blips: Optional[bool] = None,
    console: Optional[Console] = None,
    *,
    command: Optional[str] = None,
    created_at_utc: Optional[str] = None,
    save_visuals: bool = False,
    show_visuals: bool = False,
    visuals_dpi: int = 150,
) -> str:
    """
    Create a refined detection run with filtered and interpolated data.
    
    Args:
        zarr_path: Path to zarr file
        detect_run: Source detect run (default: latest)
        quality_run: Source quality run (default: latest)
        config: Config dictionary (optional, will load if not provided)
        max_gap: Maximum gap size for interpolation (overrides config)
        interpolation_method: Method for interpolation (overrides config)
        remove_jumps: Remove jump artifacts (overrides config)
        remove_blips: Remove blip artifacts (overrides config)
        console: Rich console for output
        
    Returns:
        Name of created refined run
    """
    if console is None:
        console = Console()
    
    console.rule("[bold]Detection Refinement[/bold]")
    
    import time
    start_time = time.perf_counter()
    
    # Load config if not provided
    if config is None:
        import yaml
        from pathlib import Path
        config_path = Path("pipeline_config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            config = {}
    
    # Build CLI overrides
    cli_overrides = {}
    if max_gap is not None:
        cli_overrides['max_gap'] = max_gap
    if interpolation_method is not None:
        cli_overrides['interpolation_method'] = interpolation_method
    
    # Get parameters
    params, param_source = get_refinement_parameters(config, cli_overrides if cli_overrides else None)
    
    # Handle filter overrides
    filters_config = params.get('filters', {'remove_jumps': True, 'remove_blips': False})
    if remove_jumps is not None:
        filters_config['remove_jumps'] = remove_jumps
        param_source = 'cli_override'
    if remove_blips is not None:
        filters_config['remove_blips'] = remove_blips
        param_source = 'cli_override'
    
    # Build filter list
    filters = []
    if filters_config.get('remove_jumps', True):
        filters.append('remove_jumps')
    if filters_config.get('remove_blips', False):
        filters.append('remove_blips')
    
    max_gap_val = params['max_gap']
    interp_method = params['interpolation_method']
    
    console.print(f"Parameters source: [cyan]{param_source}[/cyan]")
    console.print(f"  Max gap: {max_gap_val}")
    console.print(f"  Interpolation: {interp_method}")
    console.print(f"  Filters: {filters}")
    
    # Open zarr
    root = zarr.open(zarr_path, mode='a')
    
    # Get detect run
    if detect_run is None:
        detect_run = root['detect_runs'].attrs['latest']
    
    detect_group = root[f'detect_runs/{detect_run}']
    console.print(f"Source detect run: [cyan]{detect_run}[/cyan]")
    
    # Load detection data
    console.print("\nLoading detection data...")
    bbox_coords = detect_group['bbox_norm_coords'][:]
    frame_indices = detect_group['frame_indices'][:]

    # Load quality labels if available, otherwise assume all detections are clean
    detection_quality_labels: np.ndarray
    resolved_quality_run: Optional[str] = None
    quality_group = None
    if 'quality_reports' in detect_group and detect_group['quality_reports'].attrs.get('latest'):
        if quality_run is None:
            quality_run = detect_group['quality_reports'].attrs['latest']
        resolved_quality_run = quality_run
        quality_group = detect_group[f'quality_reports/{quality_run}']
        console.print(f"Source quality run: [cyan]{quality_run}[/cyan]")
        detection_quality_labels = quality_group['detection_quality_labels'][:]
    else:
        console.print("[yellow]⚠ No detection quality reports found; assuming all detections are valid.[/yellow]")
        detection_quality_labels = np.zeros(len(bbox_coords), dtype='i1')

    # Get total frames using unified metadata helper
    num_frames = get_total_frames(root, detect_group)
    
    if num_frames is None:
        # Infer from detections as last resort
        num_frames = int(frame_indices.max() + 1)
        console.print(f"[yellow]⚠ No 'total_frames' in metadata, inferred {num_frames} from detections[/yellow]")
        
        # Log detection method for context
        detect_method = get_detection_method(detect_group)
        console.print(f"[yellow]  Detection method: {detect_method}[/yellow]")
    else:
        console.print(f"Using {num_frames} frames from metadata")
    
    # Get frame counts
    if 'frame_counts' in detect_group:
        frame_counts = detect_group['frame_counts'][:]
    else:
        frame_counts = np.bincount(frame_indices, minlength=num_frames).astype('i4')
    
    # Scores may not exist for blob detection - create placeholder if missing
    if 'scores' in detect_group:
        scores = detect_group['scores'][:]
    else:
        # Create placeholder scores (all 1.0 for blob detections)
        scores = np.ones(len(bbox_coords), dtype='f4')
        console.print("  [yellow]Note: No scores array found, using placeholder values[/yellow]")
    
    console.print(f"  Total detections: {len(bbox_coords)}")
    console.print(f"  Total frames: {num_frames}")
    
    # Step 1: Filter
    console.print(f"\n[bold]Step 1: Filtering[/bold]")
    console.print(f"  Filters: {filters}")
    
    (filtered_bboxes, filtered_scores, filtered_counts,
     filtered_frame_indices, drop_stats) = filter_detections(
        bbox_coords, scores, frame_indices, detection_quality_labels, num_frames, filters
    )
    
    console.print(f"  Kept: {drop_stats['kept']} detections")
    console.print(f"  Dropped: {drop_stats['total_dropped']} detections")
    for reason, count in drop_stats['reasons'].items():
        console.print(f"    - {reason}: {count}")
    
    # Step 2: Interpolate
    console.print(f"\n[bold]Step 2: Interpolation[/bold]")
    console.print(f"  Max gap: {max_gap_val} frames")
    console.print(f"  Method: {interp_method}")
    
    (interp_bboxes, interp_scores, interp_counts,
     interp_frame_indices, detection_source, interp_stats) = interpolate_detections(
        filtered_bboxes, filtered_scores, filtered_frame_indices, num_frames, max_gap_val, interp_method
    )
    
    console.print(f"  Gaps filled: {interp_stats['gaps_filled']}")
    console.print(f"  Interpolated detections: {interp_stats['interpolated_detections']}")
    if interp_stats['gaps_filled'] > 0:
        console.print(f"  Gap sizes: {interp_stats['min_gap_size']}-{interp_stats['max_gap_size']} "
                     f"(mean: {interp_stats['mean_gap_size']:.1f}) frames")
    
    # Calculate coverage comparison
    original_coverage = (np.sum(frame_counts > 0) / num_frames) * 100
    filtered_coverage = (np.sum(filtered_counts > 0) / num_frames) * 100
    interpolated_coverage = (np.sum(interp_counts > 0) / num_frames) * 100
    
    comparison_stats = {
        'original': {
            'total_detections': int(len(bbox_coords)),
            'frames_with_detections': int(np.sum(frame_counts > 0)),
            'coverage_percent': float(original_coverage)
        },
        'filtered': {
            'total_detections': int(len(filtered_bboxes)),
            'frames_with_detections': int(np.sum(filtered_counts > 0)),
            'coverage_percent': float(filtered_coverage),
            'detections_removed': int(drop_stats['total_dropped']),
            'coverage_loss': float(filtered_coverage - original_coverage)
        },
        'interpolated': {
            'total_detections': int(len(interp_bboxes)),
            'frames_with_detections': int(np.sum(interp_counts > 0)),
            'coverage_percent': float(interpolated_coverage),
            'detections_added': int(interp_stats['interpolated_detections']),
            'coverage_gain': float(interpolated_coverage - filtered_coverage)
        }
    }
    
    # Step 3: Save
    console.print(f"\n[bold]Step 3: Saving Refined Run[/bold]")
    
    # Calculate processing time
    duration = time.perf_counter() - start_time
    
    # Create refined detect group
    if REFINED_DETECT_GROUP not in root:
        root.create_group(REFINED_DETECT_GROUP)
    refined_runs = root[REFINED_DETECT_GROUP]
    
    # Create timestamped run
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"refined_detect_{timestamp}"
    refined_group = refined_runs.create_group(run_name)
    refined_runs.attrs['latest'] = run_name
    
    # Store root metadata
    created_timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()

    parameters_payload = {
        'max_gap': max_gap_val,
        'interpolation_method': interp_method,
        'filters_applied': filters,
        'parameter_source': param_source
    }

    refined_group.attrs['source_detect_run'] = detect_run
    refined_group.attrs['source_quality_run'] = resolved_quality_run or 'N/A'
    refined_group.attrs['refinement_timestamp'] = created_timestamp
    refined_group.attrs['processing_time_seconds'] = float(duration)
    refined_group.attrs['operations'] = ['filter', 'interpolate']
    refined_group.attrs['parameters'] = parameters_payload
    refined_group.attrs['coverage_comparison'] = comparison_stats
    refined_group.attrs['inputs'] = {
        'detect_run': detect_run,
        'quality_run': quality_run or 'N/A'
    }

    git_info = get_git_info()
    env_info = get_environment_info()
    environment_info = {
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "python_version": env_info["platform"].get("python_version", "unknown"),
        "system": env_info["platform"].get("system", "unknown"),
        "release": env_info["platform"].get("release", "unknown"),
    }

    scheduler_info = None

    artifact_keys = [
        'model_path',
        'model_name',
        'model_version',
        'detection_method',
        'pipeline_type',
        'training_run',
        'checkpoint_path',
        'quality_source',
    ]
    artifact_info = {key: detect_group.attrs[key] for key in artifact_keys if key in detect_group.attrs}
    if quality_group is not None and 'artifact_detection_params' in quality_group.attrs:
        artifact_info['quality_detection_params'] = quality_group.attrs['artifact_detection_params']

    provenance_record = {
        'stage': 'refine_detect',
        'command': command or ' '.join(sys.argv),
        'created_at_utc': created_timestamp,
        'version': git_info.get('short_hash') or git_info.get('commit_hash'),
        'git': {
            'commit': git_info.get('commit_hash'),
            'short': git_info.get('short_hash'),
            'branch': git_info.get('branch'),
            'is_dirty': git_info.get('is_dirty'),
            'remote': git_info.get('remote_url'),
        },
        'environment': environment_info,
        'scheduler': scheduler_info,
        'parameters': parameters_payload,
        'inputs': {
            'detect_run': detect_run,
            'quality_run': quality_run or 'N/A',
        },
        'artifacts': artifact_info,
    }
    provenance_record = {k: v for k, v in provenance_record.items() if v is not None}
    refined_group.attrs['provenance'] = provenance_record
    
    # Save filtered data
    filtered_grp = refined_group.create_group('filtered')
    filtered_grp.create_array('bbox_norm_coords', data=filtered_bboxes, chunks=(1000, 4))
    filtered_grp.create_array('scores', data=filtered_scores, chunks=(1000,))
    filtered_grp.create_array('frame_indices', data=filtered_frame_indices, chunks=(1000,))
    filtered_grp.create_array('frame_counts', data=filtered_counts, chunks=(10000,))
    filtered_grp.create_array('n_detections', data=filtered_counts, chunks=(10000,))
    filtered_grp.create_array('frame_mapping', data=filtered_frame_indices, chunks=(1000,))
    
    filtered_grp.attrs['total_detections'] = int(len(filtered_bboxes))
    filtered_grp.attrs['dropped_detections'] = drop_stats['total_dropped']
    filtered_grp.attrs['drop_reasons'] = drop_stats['reasons']
    
    # Save interpolated data
    interp_grp = refined_group.create_group('interpolated')
    interp_grp.create_array('bbox_norm_coords', data=interp_bboxes, chunks=(1000, 4))
    interp_grp.create_array('scores', data=interp_scores, chunks=(1000,))
    interp_grp.create_array('frame_indices', data=interp_frame_indices, chunks=(1000,))
    interp_grp.create_array('frame_counts', data=interp_counts, chunks=(10000,))
    interp_grp.create_array('n_detections', data=interp_counts, chunks=(10000,))
    interp_grp.create_array('frame_mapping', data=interp_frame_indices, chunks=(1000,))
    interp_grp.create_array('detection_source', data=detection_source, chunks=(1000,))
    
    interp_grp.attrs['total_detections'] = int(len(interp_bboxes))
    interp_grp.attrs['original_detections'] = int(len(filtered_bboxes))
    interp_grp.attrs['interpolated_detections'] = interp_stats['interpolated_detections']
    interp_grp.attrs['gaps_filled'] = interp_stats['gaps_filled']
    interp_grp.attrs['interpolation_stats'] = interp_stats
    
    console.print(f"[green]✓[/green] Refined run saved: {refined_group.path}")
    console.print(f"[green]✓[/green] Processing completed in {duration:.2f} seconds")
    
    console.print(f"\n[bold green]Coverage Comparison:[/bold green]")
    console.print(f"  Original:     {comparison_stats['original']['frames_with_detections']:5d} frames ({comparison_stats['original']['coverage_percent']:.2f}%)")
    console.print(f"  Filtered:     {comparison_stats['filtered']['frames_with_detections']:5d} frames ({comparison_stats['filtered']['coverage_percent']:.2f}%) "
                 f"[red]{comparison_stats['filtered']['coverage_loss']:+.2f}%[/red]")
    console.print(f"  Interpolated: {comparison_stats['interpolated']['frames_with_detections']:5d} frames ({comparison_stats['interpolated']['coverage_percent']:.2f}%) "
                 f"[green]{comparison_stats['interpolated']['coverage_gain']:+.2f}%[/green]")
    
    console.print(f"\n[bold green]Detection Summary:[/bold green]")
    console.print(f"  Filtered: {len(filtered_bboxes)} detections")
    console.print(f"  Interpolated: {len(interp_bboxes)} detections")
    console.print(f"    - Real: {len(filtered_bboxes)}")
    console.print(f"    - Synthetic: {interp_stats['interpolated_detections']}")

    if save_visuals or show_visuals:
        if quality_group is None:
            console.print("[yellow]Visualizations requested, but no quality report is available; skipping.[/yellow]")
        else:
            try:
                from ..visualization.visualize_detect_quality import render_quality_png

                png_bytes, quality_meta = render_quality_png(
                    zarr_path,
                    detect_run=detect_run,
                    quality_run=resolved_quality_run,
                    dpi=visuals_dpi,
                    show=show_visuals,
                )
                if save_visuals:
                    vis_group = refined_group.require_group('visualizations')
                    array_name = 'detect_quality_overview_png'
                    if array_name in vis_group:
                        del vis_group[array_name]
                    data = np.frombuffer(png_bytes, dtype=np.uint8)
                    chunk = max(1, min(len(data), 1_048_576))
                    ds = vis_group.create_array(
                        array_name,
                        data=data,
                        chunks=(chunk,),
                        overwrite=True,
                    )
                    ds.attrs.update({
                        'mime': 'image/png',
                        'description': 'Detection quality overview visualization',
                        'source_detect_run': detect_run,
                        'source_quality_run': resolved_quality_run,
                        'quality_grade': quality_meta['quality_score'].get('grade'),
                    })
                    manifest = dict(refined_group.attrs.get('visualizations', {}))
                    manifest['detect_quality_overview_png'] = {
                        'path': 'visualizations/detect_quality_overview_png',
                        'description': 'Detection quality overview PNG',
                    }
                    refined_group.attrs['visualizations'] = manifest
                    console.print("[green]✓[/green] Detection quality visualization stored in refined run.")
            except Exception as exc:
                console.print(f"[yellow]Warning:[/yellow] Failed to render detection visualization: {exc}")
    
    return run_name


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Refine detection data by filtering and interpolating",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement (uses config defaults)
  python -m fisheye.refinement.refine_detect data.zarr

  # Override max gap
  python -m fisheye.refinement.refine_detect data.zarr --max-gap 15

  # Remove both jumps and blips
  python -m fisheye.refinement.refine_detect data.zarr --remove-jumps --remove-blips

  # Keep jumps (don't filter them)
  python -m fisheye.refinement.refine_detect data.zarr --no-remove-jumps

  # Specify source runs
  python -m fisheye.refinement.refine_detect data.zarr --detect-run detect_2025-10-03_20-28-11
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--detect-run', help='Source detect run (default: latest)')
    parser.add_argument('--quality-run', help='Source quality run (default: latest)')
    parser.add_argument('--max-gap', type=int, default=None,
                       help='Maximum gap size for interpolation (overrides config)')
    parser.add_argument('--method', default=None,
                       choices=['linear'],
                       help='Interpolation method (overrides config)')
    parser.add_argument('--remove-jumps', action='store_true', default=None,
                       help='Remove jump artifacts (overrides config)')
    parser.add_argument('--no-remove-jumps', action='store_false', dest='remove_jumps',
                       help='Keep jump artifacts (overrides config)')
    parser.add_argument('--remove-blips', action='store_true', default=None,
                       help='Remove blip artifacts (overrides config)')
    parser.add_argument('--no-remove-blips', action='store_false', dest='remove_blips',
                       help='Keep blip artifacts (overrides config)')
    parser.add_argument('--config', default='pipeline_config.yaml',
                       help='Path to config file (default: pipeline_config.yaml)')
    parser.add_argument('--save-visuals', action='store_true',
                       help='Store detection quality visualization inside the refined run.')
    parser.add_argument('--show-visuals', action='store_true',
                       help='Display detection quality visualization interactively after refinement.')
    parser.add_argument('--visuals-dpi', type=int, default=150,
                       help='DPI to use when rendering saved visualizations (default: 150).')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    from pathlib import Path
    
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    try:
        run_name = create_refined_run(
            zarr_path=args.zarr_path,
            detect_run=args.detect_run,
            quality_run=args.quality_run,
            config=config,
            max_gap=args.max_gap,
            interpolation_method=args.method,
            remove_jumps=args.remove_jumps,
            remove_blips=args.remove_blips,
            command=' '.join(sys.argv),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            save_visuals=args.save_visuals,
            show_visuals=args.show_visuals,
            visuals_dpi=args.visuals_dpi,
        )
        
        print(f"\n✓ Created refined run: {run_name}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
