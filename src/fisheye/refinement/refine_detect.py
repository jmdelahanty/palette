# src/fisheye/refinement/refine_detect.py
"""
Detection Refinement Pipeline

Builds canonical refined-detect surfaces from raw detections.

Workflow:
1. Load detection data and quality labels
2. Filter: Remove jumps/artifacts from the curated surface
3. Write sparse curated `instances/` and `source_detections/`
4. Refresh the dense compatibility root and save metadata
"""

import numpy as np
import zarr
import sys
from json import JSONDecodeError, loads as json_loads
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from rich.console import Console

from ..utils.metadata import get_total_frames, get_detection_method
from ..utils.system import get_environment_info, get_git_info
from ..shared.refined_detect_curation import write_curated_refined_detect_surfaces
from ..shared.registry_stage_complete import emit_stage_completion
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.type_conversions import normalize_attr

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"
_REFINED_DETECT_STATUS_SOURCE = "runtime_refine_detect"
_DEPRECATED_INTERPOLATION_OVERRIDE_MESSAGE = (
    "Interpolation overrides are deprecated and unsupported for refine_detect. "
    "The current sparse-first refine_detect workflow always runs with interpolation disabled."
)


def _reject_deprecated_interpolation_overrides(
    *,
    max_gap: Optional[int] = None,
    interpolation_method: Optional[str] = None,
) -> None:
    if max_gap is None and interpolation_method is None:
        return
    flags: list[str] = []
    if max_gap is not None:
        flags.append("--max-gap")
    if interpolation_method is not None:
        flags.append("--method")
    raise ValueError(f"{_DEPRECATED_INTERPOLATION_OVERRIDE_MESSAGE} Remove {' and '.join(flags)}.")


def _emit_refined_detect_status(
    *,
    root: zarr.Group,
    zarr_path: Path,
    status: str,
    run_name: Optional[str],
    method: Optional[str],
    coverage_pct: Optional[float],
    review_status: Optional[Dict[str, object]],
    details: Dict[str, object],
    console: Optional[Console],
) -> None:
    emit_stage_completion(
        root,
        zarr_path,
        step_name="refined_detect",
        status=status,
        source=_REFINED_DETECT_STATUS_SOURCE,
        run_name=run_name,
        method=method,
        coverage_pct=coverage_pct,
        review_status_json=review_status,
        details_json=details,
        console=console,
        warning_label="refined_detect",
        auto_registry_from_env=True,
        require_env_registry_exists=False,
        invalidate_on_ok=True,
        trigger_run_name=run_name,
    )

def _parse_mapping(value: object) -> Optional[Dict[str, object]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        payload = json_loads(text)
    except JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _read_sampled_import_meta(root: zarr.Group) -> Tuple[bool, Dict[str, Any]]:
    raw = root.get("raw_video")
    if raw is None:
        return False, {}
    attrs = raw.attrs
    import_mode = normalize_attr(attrs.get("import_mode"))
    import_purpose = normalize_attr(attrs.get("import_purpose"))
    frame_step = attrs.get("frame_step")
    has_mapping = "original_frame_indices" in raw

    sampled = False
    if import_mode == "sampled":
        sampled = True
    if import_purpose == "training_data":
        sampled = True
    try:
        if frame_step is not None and int(frame_step) > 1:
            sampled = True
    except Exception:
        pass
    if has_mapping:
        sampled = True

    meta = {
        "import_mode": import_mode,
        "import_purpose": import_purpose,
        "frame_step": int(frame_step) if isinstance(frame_step, (int, np.integer)) else frame_step,
        "has_original_frame_indices": bool(has_mapping),
    }
    return sampled, meta


def _get_sampled_frame_count(root: zarr.Group, detect_group: Optional[zarr.Group]) -> Optional[int]:
    raw = root.get("raw_video")
    if raw is not None:
        if "original_frame_indices" in raw:
            return int(raw["original_frame_indices"].shape[0])
        if "images_ds" in raw:
            return int(raw["images_ds"].shape[0])
        if "images_full" in raw:
            return int(raw["images_full"].shape[0])
    if detect_group is not None and "frame_counts" in detect_group:
        return int(detect_group["frame_counts"].shape[0])
    return None


def _quality_guardrail_error(detect_run: str, reason: str, quality_run: Optional[str] = None) -> ValueError:
    target = f"quality run '{quality_run}'" if quality_run else "latest quality run"
    return ValueError(
        f"Missing usable detect_quality context for detect run '{detect_run}' ({target}): {reason}. "
        "Run `python -m fisheye.refinement.detect_quality <zarr_path>` for this detect run, "
        "or pass --allow-missing-quality to opt out."
    )


def _resolve_detection_quality_labels(
    detect_group: zarr.Group,
    *,
    detect_run: str,
    quality_run: Optional[str],
    total_detections: int,
    require_quality: bool,
    allow_missing_reason: str,
    console: Optional[Console],
) -> Tuple[np.ndarray, Optional[str], Optional[zarr.Group]]:
    """Resolve per-detection quality labels and fail closed when required."""
    total = int(total_detections)
    quality_reports = detect_group.get("quality_reports")
    requested_quality_run = normalize_attr(quality_run)
    resolved_quality_run = requested_quality_run
    quality_group: Optional[zarr.Group] = None

    if quality_reports is not None and resolved_quality_run is None:
        resolved_quality_run = normalize_attr(quality_reports.attrs.get("latest"))

    missing_reason: Optional[str] = None
    if quality_reports is None:
        missing_reason = "quality_reports group is missing"
    elif not resolved_quality_run:
        missing_reason = "quality_reports/latest is missing and no --quality-run was provided"
    elif resolved_quality_run not in quality_reports:
        missing_reason = f"quality run '{resolved_quality_run}' was not found"
    else:
        quality_group = quality_reports[resolved_quality_run]
        if "detection_quality_labels" not in quality_group:
            missing_reason = "missing detection_quality_labels array"
        else:
            labels = np.asarray(quality_group["detection_quality_labels"][:], dtype="i1")
            if labels.shape[0] != total:
                missing_reason = (
                    "detection_quality_labels length "
                    f"{int(labels.shape[0])} does not match detections {total}"
                )
            else:
                if console is not None:
                    console.print(f"Source quality run: [cyan]{resolved_quality_run}[/cyan]")
                return labels, resolved_quality_run, quality_group

    if require_quality:
        raise _quality_guardrail_error(
            detect_run=detect_run,
            reason=missing_reason or "quality report is missing",
            quality_run=resolved_quality_run,
        )

    if console is not None:
        console.print(
            "[yellow]⚠ No usable detection quality report found; proceeding with all "
            f"detections marked clean ({allow_missing_reason}).[/yellow]"
        )
    return np.zeros(total, dtype="i1"), None, None

def filter_detections(
    bbox_coords: np.ndarray,
    scores: np.ndarray,
    frame_indices: np.ndarray,
    class_ids: np.ndarray,
    detection_quality_labels: np.ndarray,
    num_frames: int,
    filters: List[str] = ['remove_jumps']
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
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
                  frame_indices, class_ids, drop_stats)
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
    filtered_class_ids = class_ids[keep_mask].astype('i4', copy=False)
    
    # Update per-frame detection counts
    filtered_counts = np.bincount(filtered_frame_indices, minlength=num_frames).astype('i4', copy=False)
    
    # Stats
    drop_stats = {
        'total_dropped': int(np.sum(~keep_mask)),
        'reasons': drop_reasons,
        'kept': int(np.sum(keep_mask)),
        'original': len(detection_quality_labels)
    }
    
    return (
        filtered_bboxes,
        filtered_scores,
        filtered_counts,
        filtered_frame_indices,
        filtered_class_ids,
        drop_stats,
    )


def find_gaps_to_interpolate(
    frame_indices: np.ndarray,
    max_gap: int = 50
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
    filtered_class_ids: np.ndarray,
    num_frames: int,
    max_gap: int = 50,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
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
                  frame_indices, class_ids, detection_source, interp_stats)
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
        return (
            filtered_bboxes,
            filtered_scores,
            filtered_counts,
            filtered_frame_indices.astype('i4', copy=False),
            filtered_class_ids.astype('i4', copy=False),
            detection_source,
            stats,
        )
    
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
    all_class_ids = [filtered_class_ids.astype('i4', copy=False)]
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

        start_class = int(filtered_class_ids[start_idx])
        end_class = int(filtered_class_ids[end_idx])
        if start_class == end_class:
            gap_class_ids = np.full(len(fill_frames), start_class, dtype='i4')
        else:
            gap_class_ids = np.full(len(fill_frames), start_class, dtype='i4')
        
        all_bboxes.append(interp_bboxes)
        all_scores.append(interp_scores)
        all_frame_indices.append(np.array(fill_frames, dtype='i4'))
        all_class_ids.append(gap_class_ids)
        all_sources.append(np.ones(len(fill_frames), dtype='i1'))  # 1 = interpolated
        
        total_interpolated += len(fill_frames)
        gap_sizes.append(gap['size'])
    
    # Concatenate all detections
    interp_bboxes = np.concatenate(all_bboxes, axis=0)
    interp_scores = np.concatenate(all_scores)
    interp_frame_indices = np.concatenate(all_frame_indices)
    interp_class_ids = np.concatenate(all_class_ids)
    detection_source = np.concatenate(all_sources)
    
    # Sort by frame to maintain temporal order
    sort_idx = np.argsort(interp_frame_indices)
    interp_bboxes = interp_bboxes[sort_idx]
    interp_scores = interp_scores[sort_idx]
    interp_frame_indices = interp_frame_indices[sort_idx]
    interp_class_ids = interp_class_ids[sort_idx]
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
    
    return (
        interp_bboxes,
        interp_scores,
        interp_counts,
        interp_frame_indices.astype('i4', copy=False),
        interp_class_ids.astype('i4', copy=False),
        detection_source,
        stats,
    )


def _filtered_reason_from_quality_label(label: int) -> str:
    if label == 4:
        return "multi_detection"
    if label == 2:
        return "filtered_blip"
    if label == 3:
        return "filtered_jump"
    return f"filtered_quality_{int(label)}"


def _build_sparse_refined_inputs_from_filtered(
    *,
    raw_bboxes: np.ndarray,
    raw_scores: np.ndarray,
    raw_frame_indices: np.ndarray,
    raw_class_ids: np.ndarray,
    detection_quality_labels: np.ndarray,
    interp_bboxes: np.ndarray,
    interp_scores: np.ndarray,
    interp_frame_indices: np.ndarray,
    interp_class_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    raw_bboxes_arr = np.asarray(raw_bboxes, dtype=np.float64).reshape(-1, 4)
    raw_scores_arr = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
    raw_frame_indices_arr = np.asarray(raw_frame_indices, dtype=np.int32).reshape(-1)
    raw_class_ids_arr = np.asarray(raw_class_ids, dtype=np.int32).reshape(-1)
    quality_labels_arr = np.asarray(detection_quality_labels, dtype=np.int8).reshape(-1)
    filtered_bboxes_arr = np.asarray(interp_bboxes, dtype=np.float64).reshape(-1, 4)
    filtered_scores_arr = np.asarray(interp_scores, dtype=np.float32).reshape(-1)
    filtered_frame_indices_arr = np.asarray(interp_frame_indices, dtype=np.int32).reshape(-1)
    filtered_class_ids_arr = np.asarray(interp_class_ids, dtype=np.int32).reshape(-1)

    if not (
        raw_bboxes_arr.shape[0]
        == raw_scores_arr.shape[0]
        == raw_frame_indices_arr.shape[0]
        == raw_class_ids_arr.shape[0]
        == quality_labels_arr.shape[0]
    ):
        raise ValueError("Raw detect arrays and quality labels must agree on row count.")
    if not (
        filtered_bboxes_arr.shape[0]
        == filtered_scores_arr.shape[0]
        == filtered_frame_indices_arr.shape[0]
        == filtered_class_ids_arr.shape[0]
    ):
        raise ValueError("Filtered detect arrays must agree on row count.")

    kept_raw_indices = np.flatnonzero(quality_labels_arr == 0).astype(np.int32, copy=False)
    if kept_raw_indices.shape[0] != filtered_frame_indices_arr.shape[0]:
        raise ValueError(
            "Filtered detection rows do not match the raw keep-mask derived from detection_quality_labels."
        )

    source_decision_labels = np.full(raw_frame_indices_arr.shape[0], "filtered", dtype=object)
    source_reason_labels = np.asarray(
        [_filtered_reason_from_quality_label(int(label)) for label in quality_labels_arr.tolist()],
        dtype=object,
    )
    if kept_raw_indices.size:
        source_decision_labels[kept_raw_indices] = "accepted"
        source_reason_labels[kept_raw_indices] = "clean"

    return {
        "instance_frame_indices": filtered_frame_indices_arr,
        "instance_bbox_norm_coords": filtered_bboxes_arr,
        "instance_source_kind_labels": np.full(filtered_frame_indices_arr.shape[0], "raw_detect", dtype=object),
        "instance_reason_labels": np.full(filtered_frame_indices_arr.shape[0], "clean", dtype=object),
        "instance_source_detect_row_index": kept_raw_indices,
        "instance_manual_edit_flags": np.zeros(filtered_frame_indices_arr.shape[0], dtype=bool),
        "instance_confidence_scores": filtered_scores_arr,
        "instance_class_ids": filtered_class_ids_arr,
        "source_detection_source_detect_row_index": np.arange(raw_frame_indices_arr.shape[0], dtype=np.int32),
        "source_detection_frame_indices": raw_frame_indices_arr,
        "source_detection_bbox_norm_coords": raw_bboxes_arr,
        "source_detection_decision_labels": source_decision_labels,
        "source_detection_reason_labels": source_reason_labels,
        "source_detection_confidence_scores": raw_scores_arr,
        "source_detection_class_ids": raw_class_ids_arr,
    }


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
    refine_params.pop('max_gap', None)
    refine_params.pop('interpolation_method', None)
    refine_params.setdefault('filters', {'remove_jumps': True, 'remove_blips': False})
    refine_params.setdefault('max_gap', 0)
    refine_params.setdefault('interpolation_method', 'disabled')
    
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
    require_detect_quality: bool = True,
) -> str:
    """
    Create a refined detection run with sparse-first curated detect surfaces.
    
    Args:
        zarr_path: Path to zarr file
        detect_run: Source detect run (default: latest)
        quality_run: Source quality run (default: latest)
        config: Config dictionary (optional, will load if not provided)
        max_gap: Deprecated compatibility argument; ignored because interpolation is disabled
        interpolation_method: Deprecated compatibility argument; ignored because interpolation is disabled
        remove_jumps: Remove jump artifacts (overrides config)
        remove_blips: Remove blip artifacts (overrides config)
        console: Rich console for output
        require_detect_quality: Require usable detect_quality labels for refinement
        
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
        config_path = Path("pipeline_config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            config = {}

    _reject_deprecated_interpolation_overrides(
        max_gap=max_gap,
        interpolation_method=interpolation_method,
    )

    refine_config = config.get("refine_detect", {}) if isinstance(config, dict) else {}
    configured_max_gap = refine_config.get("max_gap")
    configured_method = refine_config.get("interpolation_method")
    if configured_max_gap not in (None, 0):
        console.print(
            "[yellow]Warning:[/yellow] refine_detect.max_gap is deprecated and ignored; "
            "the sparse-first refine_detect workflow no longer interpolates gaps."
        )
    if configured_method not in (None, "disabled"):
        console.print(
            "[yellow]Warning:[/yellow] refine_detect.interpolation_method is deprecated and ignored; "
            "the sparse-first refine_detect workflow no longer interpolates gaps."
        )
    
    # Get parameters
    params, param_source = get_refinement_parameters(config, None)
    
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
    
    max_gap_val = 0
    interp_method = 'disabled'
    refine_mode = "standard"
    
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

    sampled_import, sampled_meta = _read_sampled_import_meta(root)
    require_quality_for_run = bool(require_detect_quality and not sampled_import)
    if not require_detect_quality:
        allow_missing_reason = "explicit opt-out (--allow-missing-quality)"
    elif sampled_import:
        allow_missing_reason = "sampled import passthrough mode"
    else:
        allow_missing_reason = "guardrail disabled"
    (detection_quality_labels, resolved_quality_run, quality_group) = _resolve_detection_quality_labels(
        detect_group,
        detect_run=detect_run,
        quality_run=quality_run,
        total_detections=len(bbox_coords),
        require_quality=require_quality_for_run,
        allow_missing_reason=allow_missing_reason,
        console=console,
    )
    if sampled_import:
        console.print("[yellow]⚠ Sampled training import detected; disabling refine filters.[/yellow]")
        detection_quality_labels = np.zeros_like(detection_quality_labels)
        filters = []
        refine_mode = "passthrough"
        param_source = "sampled_import_guard"

    console.print(f"Parameters source: [cyan]{param_source}[/cyan]")
    console.print("  Interpolation: disabled")
    console.print(f"  Filters: {filters}")
    if sampled_import:
        console.print("  Refine mode: passthrough (sampled import)")

    # Get total frames using unified metadata helper
    num_frames = get_total_frames(root, detect_group)
    full_frame_count = num_frames
    coverage_frame_source = "full"
    if sampled_import:
        sampled_frames = _get_sampled_frame_count(root, detect_group)
        if sampled_frames is not None:
            if full_frame_count is not None and sampled_frames != full_frame_count:
                console.print(
                    f"[yellow]⚠ Using sampled frame count ({sampled_frames}) "
                    f"for coverage stats (full={full_frame_count}).[/yellow]"
                )
            num_frames = sampled_frames
            coverage_frame_source = "sampled"

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

    if 'class_ids' in detect_group:
        class_ids = detect_group['class_ids'][:].astype('i4', copy=False)
    else:
        class_ids = np.zeros(len(bbox_coords), dtype='i4')
        console.print("  [yellow]Note: No class_ids array found, defaulting to 0[/yellow]")
    
    console.print(f"  Total detections: {len(bbox_coords)}")
    console.print(f"  Total frames: {num_frames}")
    
    # Step 1: Filter
    console.print(f"\n[bold]Step 1: Filtering[/bold]")
    console.print(f"  Filters: {filters}")
    
    (filtered_bboxes, filtered_scores, filtered_counts,
     filtered_frame_indices, filtered_class_ids, drop_stats) = filter_detections(
        bbox_coords,
        scores,
        frame_indices,
        class_ids,
        detection_quality_labels,
        num_frames,
        filters,
    )
    
    console.print(f"  Kept: {drop_stats['kept']} detections")
    console.print(f"  Dropped: {drop_stats['total_dropped']} detections")
    for reason, count in drop_stats['reasons'].items():
        console.print(f"    - {reason}: {count}")
    
    # Step 2: Build sparse curated surfaces
    console.print(f"\n[bold]Step 2: Sparse Curated Surfaces[/bold]")
    console.print("  Interpolation: disabled")

    sparse_refined = _build_sparse_refined_inputs_from_filtered(
        raw_bboxes=bbox_coords,
        raw_scores=scores,
        raw_frame_indices=frame_indices,
        raw_class_ids=class_ids,
        detection_quality_labels=detection_quality_labels,
        interp_bboxes=filtered_bboxes,
        interp_scores=filtered_scores,
        interp_frame_indices=filtered_frame_indices,
        interp_class_ids=filtered_class_ids,
    )

    # Calculate coverage comparison
    original_coverage = (np.sum(frame_counts > 0) / num_frames) * 100
    refined_coverage = (np.sum(filtered_counts > 0) / num_frames) * 100
    
    comparison_stats = {
        'original': {
            'total_detections': int(len(bbox_coords)),
            'frames_with_detections': int(np.sum(frame_counts > 0)),
            'coverage_percent': float(original_coverage)
        },
        'refined': {
            'total_detections': int(len(filtered_bboxes)),
            'frames_with_detections': int(np.sum(filtered_counts > 0)),
            'coverage_percent': float(refined_coverage),
            'detections_removed': int(drop_stats['total_dropped']),
            'coverage_delta': float(refined_coverage - original_coverage),
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
        'filters_applied': filters,
        'parameter_source': param_source,
        'refine_mode': refine_mode,
        'interpolation_enabled': False,
        'interpolation_method': 'disabled',
        'max_gap': 0,
        'sampled_import': sampled_import,
        'sampled_import_meta': sampled_meta,
        'detect_quality_guardrail_requested': bool(require_detect_quality),
        'detect_quality_guardrail_enforced': bool(require_quality_for_run),
    }

    refined_group.attrs['source_detect_run'] = detect_run
    refined_group.attrs['source_quality_run'] = resolved_quality_run or 'N/A'
    refined_group.attrs['refinement_timestamp'] = created_timestamp
    refined_group.attrs['processing_time_seconds'] = float(duration)
    refined_group.attrs['operations'] = ['filter'] if refine_mode == "standard" else ['passthrough']
    refined_group.attrs['parameters'] = parameters_payload
    refined_group.attrs['coverage_comparison'] = comparison_stats
    refined_group.attrs['coverage_frames_total'] = int(num_frames)
    refined_group.attrs['coverage_frame_source'] = coverage_frame_source
    if full_frame_count is not None:
        refined_group.attrs['coverage_frames_full'] = int(full_frame_count)
    refined_group.attrs['inputs'] = {
        'detect_run': detect_run,
        'quality_run': resolved_quality_run or 'N/A'
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

    provenance_record = build_stage_provenance(
        stage='refine_detect',
        command=command or ' '.join(sys.argv),
        created_at_utc=created_timestamp,
        version=git_info.get('short_hash') or git_info.get('commit_hash'),
        git={
            'commit': git_info.get('commit_hash'),
            'short': git_info.get('short_hash'),
            'branch': git_info.get('branch'),
            'is_dirty': git_info.get('is_dirty'),
            'remote': git_info.get('remote_url'),
        },
        environment=environment_info,
        scheduler=scheduler_info,
        parameters=parameters_payload,
        inputs={
            'detect_run': detect_run,
            'quality_run': resolved_quality_run or 'N/A',
            'frame_source': 'zarr' if root.attrs.get('has_raw_video', True) else 'external',
            'source_video_path': root.attrs.get('source_video_path'),
        },
        artifacts=artifact_info,
    )
    write_stage_provenance(refined_group, provenance_record)
    
    write_curated_refined_detect_surfaces(
        root,
        zarr_path=Path(zarr_path).expanduser().resolve(),
        refined_run_name=run_name,
        instance_frame_indices=np.asarray(sparse_refined["instance_frame_indices"], dtype=np.int32),
        instance_bbox_norm_coords=np.asarray(sparse_refined["instance_bbox_norm_coords"], dtype=np.float64),
        instance_source_kind_labels=np.asarray(sparse_refined["instance_source_kind_labels"], dtype=object),
        instance_reason_labels=np.asarray(sparse_refined["instance_reason_labels"], dtype=object),
        instance_source_detect_row_index=np.asarray(sparse_refined["instance_source_detect_row_index"], dtype=np.int32),
        instance_manual_edit_flags=np.asarray(sparse_refined["instance_manual_edit_flags"], dtype=bool),
        instance_confidence_scores=np.asarray(sparse_refined["instance_confidence_scores"], dtype=np.float32),
        instance_class_ids=np.asarray(sparse_refined["instance_class_ids"], dtype=np.int32),
        source_detection_source_detect_row_index=np.asarray(
            sparse_refined["source_detection_source_detect_row_index"], dtype=np.int32
        ),
        source_detection_frame_indices=np.asarray(sparse_refined["source_detection_frame_indices"], dtype=np.int32),
        source_detection_bbox_norm_coords=np.asarray(sparse_refined["source_detection_bbox_norm_coords"], dtype=np.float64),
        source_detection_decision_labels=np.asarray(sparse_refined["source_detection_decision_labels"], dtype=object),
        source_detection_reason_labels=np.asarray(sparse_refined["source_detection_reason_labels"], dtype=object),
        source_detection_confidence_scores=np.asarray(
            sparse_refined["source_detection_confidence_scores"], dtype=np.float32
        ),
        source_detection_class_ids=np.asarray(sparse_refined["source_detection_class_ids"], dtype=np.int32),
        command=command or ' '.join(sys.argv),
        env_info=env_info,
        source_context={
            "source_detect_run": detect_run,
            "quality_run": resolved_quality_run or "N/A",
            "selection_policy": "quality_filtered_sparse_instances_no_interpolation",
        },
    )

    console.print(f"[green]✓[/green] Refined run saved: {refined_group.path}")
    console.print(f"[green]✓[/green] Processing completed in {duration:.2f} seconds")
    
    console.print(f"\n[bold green]Coverage Comparison:[/bold green]")
    console.print(f"  Original:     {comparison_stats['original']['frames_with_detections']:5d} frames ({comparison_stats['original']['coverage_percent']:.2f}%)")
    console.print(f"  Refined:      {comparison_stats['refined']['frames_with_detections']:5d} frames ({comparison_stats['refined']['coverage_percent']:.2f}%) "
                 f"[red]{comparison_stats['refined']['coverage_delta']:+.2f}%[/red]")
    
    console.print(f"\n[bold green]Detection Summary:[/bold green]")
    console.print(f"  Refined present detections: {len(filtered_bboxes)}")
    console.print(f"  Filtered out detections: {drop_stats['total_dropped']}")

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

    detect_review_status = _parse_mapping(refined_group.attrs.get("detect_review_status"))
    _emit_refined_detect_status(
        root=root,
        zarr_path=Path(zarr_path).expanduser().resolve(),
        status="ok",
        run_name=run_name,
        method=(
            (normalize_attr(refined_group.attrs.get("method")) if hasattr(refined_group, "attrs") else None)
            or refine_mode
        ),
        coverage_pct=comparison_stats.get("refined", {}).get("coverage_percent"),
        review_status=detect_review_status,
        details={
            "reason": "present",
            "source_detect_run": detect_run,
            "source_quality_run": resolved_quality_run,
            "refine_mode": refine_mode,
            "parameter_source": param_source,
            "filters_applied": filters,
            "coverage_frame_source": coverage_frame_source,
            "coverage_frames_total": num_frames,
            "coverage_frames_full": full_frame_count,
            "refined_present_detections": int(len(filtered_bboxes)),
            "filtered_out_detections": int(drop_stats["total_dropped"]),
            "interpolation_enabled": False,
        },
        console=console,
    )
    
    return run_name


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Refine detection data into a sparse curated detect surface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement (uses config defaults)
  scripts/py -m fisheye.refinement.refine_detect data.zarr

  # Remove both jumps and blips
  scripts/py -m fisheye.refinement.refine_detect data.zarr --remove-jumps --remove-blips

  # Keep jumps (don't filter them)
  scripts/py -m fisheye.refinement.refine_detect data.zarr --no-remove-jumps

  # Specify source runs
  scripts/py -m fisheye.refinement.refine_detect data.zarr --detect-run detect_2025-10-03_20-28-11
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--detect-run', help='Source detect run (default: latest)')
    parser.add_argument('--quality-run', help='Source quality run (default: latest)')
    parser.add_argument('--max-gap', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--method', default=None, choices=['linear'], help=argparse.SUPPRESS)
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
    parser.add_argument(
        '--allow-missing-quality',
        action='store_true',
        help='Allow refinement without detect_quality labels (legacy fallback).',
    )
    
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
            require_detect_quality=not args.allow_missing_quality,
        )
        
        print(f"\n✓ Created refined run: {run_name}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
