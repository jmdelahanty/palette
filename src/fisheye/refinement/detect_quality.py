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
from typing import Dict, List, Optional, Tuple, Any
from .utils import identify_gaps, categorize_gaps, calculate_coverage_stats, Gap


def identify_temporal_artifacts(
    bbox_coords: np.ndarray,
    n_detections: np.ndarray,
    width: float,
    height: float,
    jump_threshold_pixels: float = 100.0,
    blip_gap_threshold: int = 10,
) -> Dict:
    """
    Identify temporal artifacts in detection data.
    
    Flags detections that:
    1. Jump too far from the last known valid position (jumps)
    2. Appear after long gaps in detections (blips)
    """
    # Convert to pixel coordinates
    centroids_px = []
    frame_indices = []

    cumulative = np.cumsum(np.insert(n_detections, 0, 0))
    for frame_idx in range(len(n_detections)):
        if n_detections[frame_idx] > 0:
            idx = cumulative[frame_idx]
            center_x = bbox_coords[idx, 0] * width
            center_y = bbox_coords[idx, 1] * height
            centroids_px.append([center_x, center_y])
            frame_indices.append(frame_idx)

    if len(centroids_px) < 2:
        return {
            "blips": [],
            "jumps": [],
            "total_artifacts": 0,
            "distances": np.array([]),
            "frame_gaps": np.array([]),
            "jump_threshold": jump_threshold_pixels,
        }

    centroids_px = np.array(centroids_px, dtype=np.float32)
    frame_indices = np.array(frame_indices, dtype=np.int32)

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
    }


def validate_bboxes(bbox_coords: np.ndarray, n_detections: np.ndarray) -> Dict:
    """
    Validate bounding box quality.

    Checks:
    - Coordinate range (should be [0, 1] for normalized)
    - Size consistency
    - Aspect ratio
    - Malformed boxes
    """
    # Get valid bboxes
    valid_bboxes = []
    cumulative = np.cumsum(np.insert(n_detections, 0, 0))
    for frame_idx in range(len(n_detections)):
        if n_detections[frame_idx] > 0:
            idx = cumulative[frame_idx]
            valid_bboxes.append(bbox_coords[idx])

    if len(valid_bboxes) == 0:
        return {
            "total_bboxes": 0,
            "out_of_range": 0,
            "size_outliers": 0,
            "malformed": 0,
        }

    valid_bboxes = np.array(valid_bboxes)

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
    }

    # Load detection data
    n_detections = detect_group["n_detections"][:]
    n_frames = int(quality_report["coverage"]["total_frames"])
    
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
    no_detection_frames = np.where(n_detections == 0)[0]
    quality_flags[no_detection_frames] = -1

    # Then mark artifacts in frames that have detections
    for frame_idx in quality_report["artifacts"]["blips"]:
        if 0 <= frame_idx < n_frames:
            quality_flags[frame_idx] = 2
    for frame_idx in quality_report["artifacts"]["jumps"]:
        if 0 <= frame_idx < n_frames:
            quality_flags[frame_idx] = 3

    # Mark multi-detection frames (will be 0 if max_fish=1)
    multi_det_frames = np.where(n_detections > 1)[0]
    quality_flags[multi_det_frames] = 4

    # ============================================================================
    # Create per-detection quality labels
    # ============================================================================
    # This array has one label per detection, matching the indexing of bbox_coords
    # Note: Only includes detections, no entries for empty frames
    total_detections = int(np.sum(n_detections))
    detection_quality_labels = np.zeros(total_detections, dtype="i1")
    
    # Build cumulative index for mapping frames to detection indices
    cumulative_detections = np.cumsum(np.insert(n_detections, 0, 0))
    
    # Assign quality labels to each detection based on its frame
    for frame_idx in range(n_frames):
        if n_detections[frame_idx] > 0:
            start_idx = int(cumulative_detections[frame_idx])
            end_idx = int(cumulative_detections[frame_idx + 1])
            
            # All detections in this frame get the frame's quality label
            # Note: quality_flags[frame_idx] will be 0, 2, 3, or 4 (never -1 since n_detections > 0)
            detection_quality_labels[start_idx:end_idx] = quality_flags[frame_idx]
    
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

    return quality_group.path


def analyze_detect_quality(
    zarr_path: str,
    run_name: Optional[str] = None,
    jump_threshold: float = 100.0,
    blip_gap_threshold: int = 10,
) -> Dict:
    """
    Comprehensive detection quality analysis.

    Args:
        zarr_path: Path to zarr file
        run_name: Specific detect run to analyze (default: latest)
        jump_threshold: Distance threshold for jump detection (pixels)

    Returns:
        Complete quality analysis report
    """
    root = zarr.open(zarr_path, mode="r")

    # Get detect run
    if run_name is None:
        run_name = root["detect_runs"].attrs["latest"]

    detect_group = root[f"detect_runs/{run_name}"]
    n_detections = detect_group["n_detections"][:]
    bbox_coords = detect_group["bbox_norm_coords"][:]

    # Get image dimensions
    if "raw_video" in root:
        width = root["raw_video/images_ds"].shape[2]
        height = root["raw_video/images_ds"].shape[1]
    else:
        width = 640
        height = 640

    # Coverage analysis
    presence_mask = n_detections > 0
    coverage_stats = calculate_coverage_stats(presence_mask)
    all_gaps = identify_gaps(presence_mask)
    gap_categories = categorize_gaps(all_gaps)

    # Multi-detection frames (should be 0 for single-fish)
    multi_detection_frames = int(np.sum(n_detections > 1))

    # Temporal artifact detection
    artifacts = identify_temporal_artifacts(
        bbox_coords,
        n_detections,
        width,
        height,
        jump_threshold_pixels=jump_threshold,
        blip_gap_threshold=blip_gap_threshold
    )

    # Bounding box validation
    bbox_validation = validate_bboxes(bbox_coords, n_detections)

    # Quality score
    quality_score = calculate_quality_score(
        coverage_stats, artifacts, bbox_validation
    )

    # Gap statistics
    gap_sizes = [gap.size for gap in all_gaps]

    return {
        "source_run": run_name,
        "coverage": {
            **coverage_stats,
            "multi_detection_frames": multi_detection_frames,
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
        "--threshold", type=float, default=100.0, help="Jump threshold in pixels (default: 100)"
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
    console.print(f"Jump threshold: {args.threshold} pixels\n")

    try:
        # Run analysis
        report = analyze_detect_quality(
            args.zarr_path,
            run_name=args.run,
            jump_threshold=args.threshold,
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