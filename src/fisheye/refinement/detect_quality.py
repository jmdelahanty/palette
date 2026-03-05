# src/fisheye/refinement/detect_quality.py
"""
Detection quality assessment and artifact identification.

Evaluates the quality of detection data from detect_runs by:
1. Analyzing coverage and gaps
2. Identifying temporal artifacts (jumps and blips)
3. Validating bounding boxes
4. Computing overall quality score
"""

import numpy as np
import zarr
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from .utils import identify_gaps, categorize_gaps, calculate_coverage_stats, Gap

from ..shared.registry_stage_complete import emit_stage_completion
from ..shared.type_conversions import normalize_attr

_DETECT_QUALITY_STATUS_SOURCE = "runtime_detect_quality"


def _emit_detect_quality_status(
    *,
    zarr_path: str,
    quality_run_name: str,
    source_detect_run: str,
    quality_score: Dict[str, object],
    console: Optional[object] = None,
) -> None:
    """Write a detect_quality step status row to the registry (non-fatal)."""
    try:
        zp = Path(zarr_path).expanduser().resolve()
        root = zarr.open(str(zp), mode="r")
        emit_stage_completion(
            root,
            zp,
            step_name="detect_quality",
            status="ok",
            source=_DETECT_QUALITY_STATUS_SOURCE,
            run_name=quality_run_name,
            method=None,
            coverage_pct=None,
            review_status_json=None,
            details_json={
                "quality_grade": quality_score.get("grade"),
                "quality_score": quality_score.get("overall_score"),
                "clean_percentage": quality_score.get("coverage_score"),
                "source_detect_run": source_detect_run,
            },
            console=None,
            auto_registry_from_env=True,
            require_env_registry_exists=False,
            invalidate_on_ok=False,
        )
    except Exception as exc:
        if console is not None:
            from rich.console import Console as _Console
            if isinstance(console, _Console):
                console.print(
                    f"[yellow]Warning:[/yellow] failed to write recording step status "
                    f"for detect_quality: {exc}"
                )
            else:
                print(f"Warning: failed to write recording step status for detect_quality: {exc}")
        else:
            print(f"Warning: failed to write recording step status for detect_quality: {exc}")


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


def _get_expected_subject_count(root: zarr.Group) -> Optional[int]:
    setup = root.attrs.get("experiment_setup", {})
    if setup:
        total_expected = setup.get("total_expected_fish")
        if total_expected is None:
            total_expected = setup.get("subject_count")
        try:
            if total_expected is not None:
                return int(total_expected)
        except Exception:
            pass
    return None


def _as_positive_int(value: object) -> Optional[int]:
    try:
        ivalue = int(value)  # type: ignore[arg-type]
    except Exception:
        return None
    return ivalue if ivalue > 0 else None


def _resolve_detect_geometry(
    root: zarr.Group,
    detect_group: zarr.Group,
    frame_indices: np.ndarray,
) -> Tuple[int, int, int]:
    """Resolve (width, height, num_frames) without requiring imported raw frames."""
    raw = root.get("raw_video")

    # Preferred source: imported frame datasets, when present.
    if raw is not None and "images_ds" in raw:
        ds_images = raw["images_ds"]
        return int(ds_images.shape[2]), int(ds_images.shape[1]), int(ds_images.shape[0])
    if raw is not None and "images_full" in raw:
        full_images = raw["images_full"]
        return int(full_images.shape[2]), int(full_images.shape[1]), int(full_images.shape[0])

    # Fallback image geometry from attrs.
    width = _as_positive_int(root.attrs.get("width")) or 640
    height = _as_positive_int(root.attrs.get("height")) or 640
    if raw is not None:
        ds_res = raw.attrs.get("downsampled_resolution")
        if isinstance(ds_res, (list, tuple)) and len(ds_res) == 2:
            ds_h = _as_positive_int(ds_res[0])
            ds_w = _as_positive_int(ds_res[1])
            if ds_h and ds_w:
                height = ds_h
                width = ds_w

    # Fallback frame universe in order of reliability.
    num_frames = None
    if "frame_counts" in detect_group:
        num_frames = _as_positive_int(detect_group["frame_counts"].shape[0])
    if num_frames is None:
        num_frames = _as_positive_int(detect_group.attrs.get("total_frames"))
    if num_frames is None:
        num_frames = _as_positive_int(root.attrs.get("total_frames"))
    if num_frames is None and frame_indices.size:
        num_frames = int(frame_indices.max()) + 1
    if num_frames is None:
        num_frames = 0

    return width, height, int(num_frames)


def _effective_jump_threshold_pixels(
    jump_threshold: float,
    threshold_mode: str,
    width: float,
    height: float,
    threshold_reference_width: float = 640.0,
) -> float:
    """Convert threshold modes to an effective pixel displacement threshold."""
    if threshold_mode == "pixels":
        return float(jump_threshold)

    if threshold_mode == "normalized":
        return float(jump_threshold) * float(min(width, height))

    if threshold_mode == "scaled":
        ref = float(threshold_reference_width)
        if ref <= 0:
            raise ValueError("threshold_reference_width must be > 0 for scaled mode")
        scale = float(min(width, height)) / ref
        return float(jump_threshold) * scale

    raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")


def identify_temporal_artifacts(
    bbox_coords: np.ndarray,
    frame_indices: np.ndarray,
    width: float,
    height: float,
    jump_threshold_pixels: float = 50.0,
    blip_gap_threshold: int = 10,
) -> Dict:
    """
    Identify temporal artifacts in detection data.
    
    Flags detections that:
    1. Jump too far from the last known valid position (jumps)
    2. Appear after long gaps in detections (blips)
    """
    if bbox_coords.size == 0:
        return {
            "blips": [],
            "jumps": [],
            "total_artifacts": 0,
            "distances": np.array([]),
            "frame_gaps": np.array([]),
            "jump_threshold": jump_threshold_pixels,
        }

    centroids_px = np.column_stack(
        (bbox_coords[:, 0] * width, bbox_coords[:, 1] * height)
    ).astype(np.float32, copy=False)
    frame_indices = frame_indices.astype(np.int32, copy=False)
    
    if len(centroids_px) < 2:
        return {
            "blips": [],
            "jumps": [],
            "total_artifacts": 0,
            "distances": np.array([]),
            "frame_gaps": np.array([]),
            "jump_threshold": jump_threshold_pixels,
        }

    # Calculate distances and gaps between consecutive detections
    distances = np.linalg.norm(np.diff(centroids_px, axis=0), axis=1)
    frame_gaps = np.diff(frame_indices)

    jumps = []
    blips = []
    
    # Track the last known valid position
    last_valid_idx = 0
    last_valid_pos = centroids_px[0].copy()

    # Check each detection
    for det_idx in range(1, len(centroids_px)):
        current_pos = centroids_px[det_idx]
        current_frame = int(frame_indices[det_idx])
        frame_gap = int(frame_gaps[det_idx - 1])
        
        # Calculate distance from last known valid position
        dist_from_valid = float(np.linalg.norm(current_pos - last_valid_pos))
        
        # Check if this is a jump
        if dist_from_valid > jump_threshold_pixels:
            # This is a jump - flag this frame
            jumps.append(current_frame)
            # Don't update last_valid - keep it at the previous valid position
        else:
            # Valid detection - update last known valid position
            last_valid_idx = det_idx
            last_valid_pos = current_pos.copy()
        
        # Also flag as blip if there was a long gap (regardless of jump status)
        if frame_gap >= blip_gap_threshold:
            if current_frame not in jumps:
                blips.append(current_frame)

    # Remove duplicates and sort
    jumps = sorted(set(jumps))
    blips = sorted(set(blips))

    return {
        "blips": blips,
        "jumps": jumps,
        "total_artifacts": len(jumps) + len(blips),
        "distances": distances,
        "frame_gaps": frame_gaps,
        "jump_threshold": jump_threshold_pixels,
        "blip_gap_threshold": blip_gap_threshold,
    }


def validate_bboxes(bbox_coords: np.ndarray, frame_indices: np.ndarray) -> Dict:
    """
    Validate bounding box quality.

    Checks:
    - Coordinate range (should be [0, 1] for normalized)
    - Size consistency
    - Aspect ratio
    - Malformed boxes
    """
    # Get valid bboxes
    if bbox_coords.size == 0:
        return {
            "total_bboxes": 0,
            "out_of_range": 0,
            "size_outliers": 0,
            "malformed": 0,
        }

    frame_indices = frame_indices.astype(np.int64, copy=False)
    _, first_indices = np.unique(frame_indices, return_index=True)
    valid_bboxes = bbox_coords[first_indices]

    # Check coordinate range
    out_of_range = np.sum(
        (valid_bboxes[:, :2] < 0)
        | (valid_bboxes[:, :2] > 1)
        | (valid_bboxes[:, 2:] <= 0)
        | (valid_bboxes[:, 2:] > 1)
    )

    # Calculate sizes
    sizes = np.sqrt(valid_bboxes[:, 2] ** 2 + valid_bboxes[:, 3] ** 2)
    mean_size = float(np.mean(sizes))
    std_size = float(np.std(sizes))

    # Size outliers (>3 std from mean)
    size_outliers = int(np.sum(np.abs(sizes - mean_size) > (3 * std_size)))

    # Malformed (zero or negative dimensions)
    malformed = int(np.sum((valid_bboxes[:, 2] <= 0) | (valid_bboxes[:, 3] <= 0)))

    return {
        "total_bboxes": int(len(valid_bboxes)),
        "out_of_range": int(out_of_range),
        "size_outliers": int(size_outliers),
        "malformed": int(malformed),
        "mean_size": mean_size,
        "std_size": std_size,
        "size_cv": float(std_size / mean_size) if mean_size > 0 else 0.0,
    }


def calculate_quality_score(
    coverage_stats: Dict, artifacts: Dict, bbox_validation: Dict
) -> Dict:
    """
    Calculate overall detection quality score.

    Components:
    - Coverage score (0-100): Percentage of frames with detections
    - Artifact score (0-100): Penalizes temporal artifacts
    - Bbox score (0-100): Penalizes invalid bounding boxes
    - Overall score: Weighted combination
    """
    # Coverage score (direct percentage)
    coverage_score = float(coverage_stats["coverage_percent"])

    # Artifact score (penalize artifacts)
    if coverage_stats["present_frames"] > 0:
        artifact_ratio = artifacts["total_artifacts"] / coverage_stats["present_frames"]
        artifact_score = max(0.0, 100.0 - (artifact_ratio * 100.0))
    else:
        artifact_score = 0.0

    # Bbox score (penalize invalid boxes)
    if bbox_validation["total_bboxes"] > 0:
        invalid_ratio = (
            bbox_validation["out_of_range"] + bbox_validation["malformed"]
        ) / bbox_validation["total_bboxes"]
        bbox_score = max(0.0, 100.0 - (invalid_ratio * 100.0))
    else:
        bbox_score = 0.0

    # Overall score (weighted average)
    overall_score = (
        coverage_score * 0.5 +
        artifact_score * 0.3 +
        bbox_score * 0.2
    )

    # Assign letter grade
    if overall_score >= 90.0:
        grade = "A"
    elif overall_score >= 80.0:
        grade = "B"
    elif overall_score >= 70.0:
        grade = "C"
    elif overall_score >= 60.0:
        grade = "D"
    else:
        grade = "F"

    return {
        "coverage_score": float(coverage_score),
        "artifact_score": float(artifact_score),
        "bbox_score": float(bbox_score),
        "overall_score": float(overall_score),
        "grade": grade,
    }


def calculate_sampled_quality_score(
    coverage_stats: Dict,
    frame_counts: np.ndarray,
    expected_count: Optional[int],
    bbox_validation: Dict,
) -> Dict:
    """
    Calculate quality score for sampled imports (training data).

    Focuses on per-frame detection counts vs expected subjects and overall detection coverage.
    """
    coverage_score = float(coverage_stats["coverage_percent"])
    expected_score = None
    if expected_count and expected_count > 0:
        expected_score = float(np.mean(frame_counts == expected_count) * 100.0)
    else:
        expected_score = coverage_score

    if bbox_validation["total_bboxes"] > 0:
        invalid_ratio = (
            bbox_validation["out_of_range"] + bbox_validation["malformed"]
        ) / bbox_validation["total_bboxes"]
        bbox_score = max(0.0, 100.0 - (invalid_ratio * 100.0))
    else:
        bbox_score = 0.0

    overall_score = (
        expected_score * 0.6 +
        coverage_score * 0.3 +
        bbox_score * 0.1
    )

    if overall_score >= 90.0:
        grade = "A"
    elif overall_score >= 80.0:
        grade = "B"
    elif overall_score >= 70.0:
        grade = "C"
    elif overall_score >= 60.0:
        grade = "D"
    else:
        grade = "F"

    return {
        "coverage_score": float(coverage_score),
        "expected_count_score": float(expected_score),
        "bbox_score": float(bbox_score),
        "overall_score": float(overall_score),
        "grade": grade,
        "mode": "sampled",
        "expected_count": expected_count,
    }


def save_quality_report(
    zarr_path: str,
    quality_report: Dict,
    console: Optional[Any] = None
) -> str:
    """
    Save quality report to zarr file within the source detect run.

    Creates a quality_reports subgroup within the detect run containing:
    - Per-frame quality flags (-1=no detection, 0=clean, 2=blip, 3=jump, 4=multi)
    - Per-detection quality labels (0=clean, 2=blip, 3=jump, 4=multi)
    - Full quality metrics and parameters

    Args:
        zarr_path: Path to zarr file
        quality_report: Report from analyze_detect_quality()
        console: Optional Rich console for output

    Returns:
        Path to created quality report group
    """
    from datetime import datetime

    root = zarr.open(zarr_path, mode="a")

    # Navigate to source detect run
    source_run = quality_report["source_run"]
    detect_group = root[f"detect_runs/{source_run}"]

    # Create quality_reports subgroup if needed
    if "quality_reports" not in detect_group:
        detect_group.create_group("quality_reports")

    quality_reports_group = detect_group["quality_reports"]

    # Create timestamped run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"detect_quality_{timestamp}"
    quality_group = quality_reports_group.create_group(run_name)
    quality_reports_group.attrs["latest"] = run_name

    # Store metadata
    quality_group.attrs["analysis_timestamp"] = datetime.now().isoformat()
    quality_group.attrs["quality_score"] = quality_report["quality_score"]
    quality_group.attrs["coverage_stats"] = quality_report["coverage"]
    quality_group.attrs["bbox_validation"] = quality_report["bbox_validation"]
    quality_group.attrs["artifact_detection_params"] = {
        "jump_threshold": quality_report["artifacts"]["jump_threshold"],
        "blip_gap_threshold": quality_report["artifacts"].get("blip_gap_threshold", 10),
        "jump_threshold_mode": quality_report["artifacts"].get("jump_threshold_mode", "pixels"),
        "jump_threshold_pixels_effective": quality_report["artifacts"].get("jump_threshold_pixels_effective"),
        "jump_threshold_reference_width": quality_report["artifacts"].get("jump_threshold_reference_width"),
    }
    if "sampling" in quality_report:
        quality_group.attrs["sampling"] = quality_report["sampling"]

    # Load detection data
    frame_indices = detect_group["frame_indices"][:].astype(np.int64, copy=False)
    n_frames = int(quality_report["coverage"]["total_frames"])
    if "frame_counts" in detect_group:
        frame_counts = detect_group["frame_counts"][:]
        if len(frame_counts) < n_frames:
            frame_counts = np.pad(
                frame_counts,
                (0, n_frames - len(frame_counts)),
                mode="constant"
            )
    else:
        frame_counts = np.bincount(frame_indices, minlength=n_frames)
    
    # ============================================================================
    # Create per-frame quality flags
    # ============================================================================
    # -1 = no detection (empty frame)
    #  0 = clean detection
    #  2 = blip
    #  3 = jump
    #  4 = multi-detection
    quality_flags = np.zeros(n_frames, dtype="i1")

    # First, mark all empty frames as -1
    no_detection_frames = np.where(frame_counts == 0)[0]
    quality_flags[no_detection_frames] = -1

    # Then mark artifacts in frames that have detections
    for frame_idx in quality_report["artifacts"]["blips"]:
        if 0 <= frame_idx < n_frames:
            quality_flags[frame_idx] = 2
    for frame_idx in quality_report["artifacts"]["jumps"]:
        if 0 <= frame_idx < n_frames:
            quality_flags[frame_idx] = 3

    # Mark multi-detection frames (will be 0 if max_fish=1)
    multi_det_frames = np.where(frame_counts > 1)[0]
    quality_flags[multi_det_frames] = 4

    # ============================================================================
    # Create per-detection quality labels
    # ============================================================================
    # This array has one label per detection, matching the indexing of bbox_coords
    # Note: Only includes detections, no entries for empty frames
    total_detections = int(frame_indices.size)
    if total_detections > 0:
        detection_quality_labels = quality_flags[frame_indices].astype("i1", copy=False)
    else:
        detection_quality_labels = np.zeros(0, dtype="i1")
    
    # ============================================================================
    # Calculate summary statistics
    # ============================================================================
    n_empty_frames = int(np.sum(quality_flags == -1))
    n_clean_frames = int(np.sum(quality_flags == 0))
    
    n_clean_detections = int(np.sum(detection_quality_labels == 0))
    n_blip_detections = int(np.sum(detection_quality_labels == 2))
    n_jump_detections = int(np.sum(detection_quality_labels == 3))
    n_multi_detections = int(np.sum(detection_quality_labels == 4))
    
    quality_group.attrs["detection_quality_summary"] = {
        "total_frames": n_frames,
        "empty_frames": n_empty_frames,
        "frames_with_detections": n_frames - n_empty_frames,
        "clean_frames": n_clean_frames,
        "total_detections": int(total_detections),
        "clean_detections": n_clean_detections,
        "blip_detections": n_blip_detections,
        "jump_detections": n_jump_detections,
        "multi_detections": n_multi_detections,
        "clean_percentage": float(n_clean_detections / total_detections * 100) if total_detections > 0 else 0.0,
    }

    quality_group.attrs['provenance'] = {
        'command': ' '.join(sys.argv),
        'created_at_utc': datetime.now().isoformat(),
        'source_detect_run': source_run,
        'jump_threshold': quality_report['artifacts']['jump_threshold'],
        'blip_gap_threshold': quality_report['artifacts'].get('blip_gap_threshold'),
        'jump_threshold_mode': quality_report['artifacts'].get('jump_threshold_mode', 'pixels'),
        'jump_threshold_pixels_effective': quality_report['artifacts'].get('jump_threshold_pixels_effective'),
        'jump_threshold_reference_width': quality_report['artifacts'].get('jump_threshold_reference_width'),
    }
    
    # ============================================================================
    # Save arrays
    # ============================================================================
    quality_group.create_array("quality_flags", data=quality_flags, chunks=(10000,))
    quality_group.create_array("detection_quality_labels", data=detection_quality_labels, chunks=(10000,))

    # Save gap information if any
    if quality_report["coverage"]["gaps"]["total_count"] > 0:
        quality_group.attrs["gap_stats"] = {
            "total_count": quality_report["coverage"]["gaps"]["total_count"],
            "longest_gap": quality_report["coverage"]["gaps"]["longest_gap"],
            "mean_gap_size": quality_report["coverage"]["gaps"]["mean_gap_size"],
            "categories": quality_report["coverage"]["gaps"]["categories"],
        }

    # ============================================================================
    # Print summary
    # ============================================================================
    if console:
        console.print(f"[green]✓[/green] Quality report saved: {quality_group.path}")
        console.print(f"[cyan]Frame Quality Summary:[/cyan]")
        console.print(f"  Total frames: {n_frames}")
        console.print(f"  Empty frames: {n_empty_frames}")
        console.print(f"  Frames with detections: {n_frames - n_empty_frames}")
        console.print(f"  Clean frames: {n_clean_frames}")
        console.print(f"[cyan]Detection Quality Summary:[/cyan]")
        console.print(f"  Total detections: {total_detections}")
        console.print(f"  Clean: {n_clean_detections} ({n_clean_detections/total_detections*100:.1f}%)" if total_detections > 0 else "  Clean: 0 (0.0%)")
        console.print(f"  Blips: {n_blip_detections}")
        console.print(f"  Jumps: {n_jump_detections}")
        if n_multi_detections > 0:
            console.print(f"  Multi-detection: {n_multi_detections}")
    else:
        print(f"Quality report saved: {quality_group.path}")
        print(f"Frame Quality Summary:")
        print(f"  Total frames: {n_frames}")
        print(f"  Empty frames: {n_empty_frames}")
        print(f"  Frames with detections: {n_frames - n_empty_frames}")
        print(f"  Clean frames: {n_clean_frames}")
        print(f"Detection Quality Summary:")
        print(f"  Total detections: {total_detections}")
        print(f"  Clean: {n_clean_detections} ({n_clean_detections/total_detections*100:.1f}%)" if total_detections > 0 else "  Clean: 0 (0.0%)")
        print(f"  Blips: {n_blip_detections}")
        print(f"  Jumps: {n_jump_detections}")
        if n_multi_detections > 0:
            print(f"  Multi-detection: {n_multi_detections}")

    _emit_detect_quality_status(
        zarr_path=zarr_path,
        quality_run_name=run_name,
        source_detect_run=source_run,
        quality_score=quality_report["quality_score"],
        console=console,
    )

    return quality_group.path


def analyze_detect_quality(
    zarr_path: str,
    run_name: Optional[str] = None,
    jump_threshold: float = 100.0,
    blip_gap_threshold: int = 10,
    threshold_mode: str = "pixels",
    threshold_reference_width: float = 640.0,
) -> Dict:
    """
    Comprehensive detection quality analysis.

    Args:
        zarr_path: Path to zarr file
        run_name: Specific detect run to analyze (default: latest)
        jump_threshold: Distance threshold for jump detection (pixels)
        threshold_mode: Threshold interpretation mode (`pixels`, `normalized`, `scaled`)
        threshold_reference_width: Reference width for `scaled` threshold mode

    Returns:
        Complete quality analysis report
    """
    root = zarr.open(zarr_path, mode="r")
    sampled, sampled_meta = _read_sampled_import_meta(root)
    expected_count = _get_expected_subject_count(root)

    # Get detect run
    if run_name is None:
        run_name = root["detect_runs"].attrs["latest"]

    detect_group = root[f"detect_runs/{run_name}"]
    frame_indices = detect_group["frame_indices"][:]
    bbox_coords = detect_group["bbox_norm_coords"][:]

    # Resolve image geometry and frame universe (works with and without imported frames).
    width, height, num_frames = _resolve_detect_geometry(root, detect_group, frame_indices)
    effective_jump_threshold = _effective_jump_threshold_pixels(
        jump_threshold=jump_threshold,
        threshold_mode=threshold_mode,
        width=width,
        height=height,
        threshold_reference_width=threshold_reference_width,
    )

    if "frame_counts" in detect_group:
        frame_counts = detect_group["frame_counts"][:]
        if len(frame_counts) < num_frames:
            pad_width = num_frames - len(frame_counts)
            frame_counts = np.pad(frame_counts, (0, pad_width), mode="constant")
    else:
        frame_counts = np.bincount(
            frame_indices.astype(np.int64, copy=False),
            minlength=num_frames,
        )

    # Coverage analysis
    presence_mask = frame_counts > 0
    coverage_stats = calculate_coverage_stats(presence_mask)
    all_gaps = identify_gaps(presence_mask)
    gap_categories = categorize_gaps(all_gaps)

    # Multi-detection frames (should be 0 for single-fish)
    multi_detection_frames = int(np.sum(frame_counts > 1))

    # Temporal artifact detection
    if sampled:
        artifacts = {
            "blips": [],
            "jumps": [],
            "total_artifacts": 0,
            "distances": np.array([]),
            "frame_gaps": np.array([]),
            "jump_threshold": jump_threshold,
            "blip_gap_threshold": blip_gap_threshold,
            "jump_threshold_mode": threshold_mode,
            "jump_threshold_pixels_effective": effective_jump_threshold,
            "jump_threshold_reference_width": (
                float(threshold_reference_width) if threshold_mode == "scaled" else None
            ),
        }
    else:
        artifacts = identify_temporal_artifacts(
            bbox_coords,
            frame_indices,
            width,
            height,
            jump_threshold_pixels=effective_jump_threshold,
            blip_gap_threshold=blip_gap_threshold
        )
        artifacts["jump_threshold"] = jump_threshold
        artifacts["jump_threshold_mode"] = threshold_mode
        artifacts["jump_threshold_pixels_effective"] = effective_jump_threshold
        artifacts["jump_threshold_reference_width"] = (
            float(threshold_reference_width) if threshold_mode == "scaled" else None
        )

    # Bounding box validation
    bbox_validation = validate_bboxes(bbox_coords, frame_indices)

    # Quality score
    if sampled:
        quality_score = calculate_sampled_quality_score(
            coverage_stats,
            frame_counts,
            expected_count,
            bbox_validation,
        )
    else:
        quality_score = calculate_quality_score(
            coverage_stats, artifacts, bbox_validation
        )

    # Gap statistics
    gap_sizes = [gap.size for gap in all_gaps]

    coverage_extras: Dict[str, Any] = {}
    if expected_count and expected_count > 0:
        frames_expected = int(np.sum(frame_counts == expected_count))
        frames_under = int(np.sum((frame_counts > 0) & (frame_counts < expected_count)))
        frames_over = int(np.sum(frame_counts > expected_count))
        coverage_extras = {
            "expected_count": int(expected_count),
            "frames_with_expected_count": frames_expected,
            "frames_under_expected": frames_under,
            "frames_over_expected": frames_over,
            "expected_count_percent": float(frames_expected / max(num_frames, 1) * 100.0),
        }

    return {
        "source_run": run_name,
        "coverage": {
            **coverage_stats,
            "multi_detection_frames": multi_detection_frames,
            **coverage_extras,
            "gaps": {
                "total_count": len(all_gaps),
                "categories": gap_categories,
                "longest_gap": int(max(gap_sizes)) if gap_sizes else 0,
                "mean_gap_size": float(np.mean(gap_sizes)) if gap_sizes else 0.0,
                "median_gap_size": float(np.median(gap_sizes)) if gap_sizes else 0.0,
            },
        },
        "artifacts": artifacts,
        "bbox_validation": bbox_validation,
        "quality_score": quality_score,
        "sampling": {
            "sampled": bool(sampled),
            **sampled_meta,
        },
    }


if __name__ == "__main__":
    import argparse
    import sys
    from rich.console import Console

    parser = argparse.ArgumentParser(
        description="Analyze detection quality and identify artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest detect run
  python -m fisheye.refinement.detect_quality data.zarr

  # Analyze specific detect run
  python -m fisheye.refinement.detect_quality data.zarr --run detect_2025-01-15_12-00-00

  # Use custom jump threshold and save report
  python -m fisheye.refinement.detect_quality data.zarr --threshold 150 --save

  # Quick check without saving
  python -m fisheye.refinement.detect_quality data.zarr --no-save
        """,
    )

    parser.add_argument("zarr_path", help="Path to zarr file")
    parser.add_argument("--run", help="Specific detect run to analyze (default: latest)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help=(
            "Jump threshold value. In scaled mode (default), interpreted as pixels "
            "at reference width."
        ),
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["scaled", "pixels", "normalized"],
        default="scaled",
        help="How to interpret --threshold (default: scaled).",
    )
    parser.add_argument(
        "--threshold-reference-width",
        type=float,
        default=640.0,
        help="Reference width for scaled threshold mode (default: 640).",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save quality report to zarr (default: True)"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving report to zarr"
    )

    args = parser.parse_args()

    console = Console()

    console.rule("[bold]Detection Quality Analysis (Simplified)[/bold]")
    console.print(f"Zarr: {args.zarr_path}")
    if args.run:
        console.print(f"Run: {args.run}")
    if args.threshold_mode == "scaled":
        console.print(
            f"Jump threshold: {args.threshold} px @ {args.threshold_reference_width:.0f}px reference "
            f"(mode: {args.threshold_mode})\n"
        )
    else:
        console.print(f"Jump threshold: {args.threshold} (mode: {args.threshold_mode})\n")

    try:
        # Run analysis
        report = analyze_detect_quality(
            args.zarr_path,
            run_name=args.run,
            jump_threshold=args.threshold,
            threshold_mode=args.threshold_mode,
            threshold_reference_width=args.threshold_reference_width,
        )

        # Print detailed summary
        console.print("[bold cyan]COVERAGE[/bold cyan]")
        cov = report["coverage"]
        console.print(f"  Total frames: {cov['total_frames']}")
        console.print(f"  Detected: {cov['present_frames']} ({cov['coverage_percent']:.1f}%)")
        console.print(f"  Multi-detection: {cov['multi_detection_frames']}")

        console.print("\n[bold yellow]GAPS[/bold yellow]")
        gaps = cov["gaps"]
        console.print(f"  Total: {gaps['total_count']}")
        console.print(f"  Longest: {gaps['longest_gap']} frames")
        console.print(f"  Mean: {gaps['mean_gap_size']:.1f} frames")

        console.print("\n[bold red]ARTIFACTS[/bold red]")
        art = report["artifacts"]
        console.print(f"  Blips: {len(art['blips'])}")
        console.print(f"  Jumps: {len(art['jumps'])}")
        console.print(f"  Total: {art['total_artifacts']}")

        console.print("\n[bold magenta]BBOX VALIDATION[/bold magenta]")
        bbox = report["bbox_validation"]
        console.print(f"  Total: {bbox['total_bboxes']}")
        console.print(f"  Out of range: {bbox['out_of_range']}")
        console.print(f"  Size outliers: {bbox['size_outliers']}")
        console.print(f"  Malformed: {bbox['malformed']}")

        console.print("\n[bold green]QUALITY SCORE[/bold green]")
        score = report["quality_score"]
        grade_color = {
            "A": "green",
            "B": "cyan",
            "C": "yellow",
            "D": "red",
            "F": "bold red",
        }
        console.print(
            f"  Grade: [{grade_color[score['grade']]}]{score['grade']}[/{grade_color[score['grade']]}] "
            f"({score['overall_score']:.1f}/100)"
        )
        console.print(f"  Coverage: {score['coverage_score']:.1f}/100")
        console.print(f"  Artifacts: {score['artifact_score']:.1f}/100")
        console.print(f"  Bbox: {score['bbox_score']:.1f}/100")

        # Save if requested
        if args.save and not args.no_save:
            console.print()
            save_quality_report(args.zarr_path, report, console=console)

        sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
