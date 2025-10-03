# src/fisheye/refinement/detect_quality.py
"""
Detection quality assessment and artifact identification.

Evaluates the quality of detection data from detect_runs by:
1. Analyzing coverage and gaps
2. Identifying temporal artifacts (islands, blips, jumps)
3. Validating bounding boxes
4. Computing overall quality score
"""

import numpy as np
import zarr
from typing import Dict, List, Optional, Tuple, Any
from .utils import identify_gaps, categorize_gaps, calculate_coverage_stats, Gap


def _is_mostly_consecutive(frames: np.ndarray, tolerance: int = 2) -> bool:
    """True if there are at most `tolerance` non-1 gaps across consecutive frames."""
    if len(frames) <= 1:
        return True
    gaps = np.diff(frames)
    return int(np.sum(gaps != 1)) <= int(tolerance)


def _build_prev_anchor_indices(
    frame_indices: np.ndarray,
    flagged_frames_set: set,
    max_gap_for_valid: int,
    minimum_valid_streak: int,
) -> np.ndarray:
    """
    For each detection index i (NOT frame number), prev_anchor[i] is the index
    of the most recent detection that ended a valid streak (length >= minimum_valid_streak)
    consisting of UNFLAGGED detections with internal gaps <= max_gap_for_valid.
    If none exists yet, prev_anchor[i] = -1.
    """
    n = len(frame_indices)
    prev_anchor = np.full(n, -1, dtype=np.int32)

    streak_len = 0
    last_det_idx_in_streak = -1
    current_anchor = -1

    for i in range(n):
        f = int(frame_indices[i])
        if f in flagged_frames_set:
            # break the streak on flagged frames
            streak_len = 0
            last_det_idx_in_streak = -1
        else:
            if last_det_idx_in_streak >= 0 and (
                f - int(frame_indices[last_det_idx_in_streak])
            ) <= max_gap_for_valid:
                streak_len += 1
            else:
                streak_len = 1
            last_det_idx_in_streak = i

            if streak_len >= minimum_valid_streak:
                current_anchor = i  # end of a valid streak becomes the anchor

        prev_anchor[i] = current_anchor

    return prev_anchor


def identify_temporal_artifacts(
    bbox_coords: np.ndarray,
    n_detections: np.ndarray,
    width: float,
    height: float,
    jump_threshold_pixels: float = 20.0,
    max_gap_for_valid: int = 100,
    minimum_valid_streak: int = 3,
    *,
    blip_gap_threshold: int = 10,
    max_island_len: int = 20,
    require_consecutive_for_island: bool = True,
) -> Dict:
    """
    Identify temporal artifacts in detection data.
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
            "islands": [],
            "blips": [],
            "jumps": [],
            "total_artifacts": 0,
            "distances": np.array([]),
            "frame_gaps": np.array([]),
            "jump_threshold": jump_threshold_pixels,
            "max_gap_for_valid": max_gap_for_valid,
            "minimum_valid_streak": minimum_valid_streak,
        }

    centroids_px = np.array(centroids_px, dtype=np.float32)
    frame_indices = np.array(frame_indices, dtype=np.int32)

    distances = np.linalg.norm(np.diff(centroids_px, axis=0), axis=1)
    frame_gaps = np.diff(frame_indices)
    outlier_indices = np.where(distances > jump_threshold_pixels)[0]

    # ---------- PASS 1: Build initial anchors (motion-based only) ----------
    # Build anchors assuming no islands yet - just skip outlier detections
    initial_flagged = set()
    for idx in outlier_indices:
        initial_flagged.add(int(frame_indices[idx + 1]))  # Flag detection after big jump
    
    prev_anchor = _build_prev_anchor_indices(
        frame_indices, initial_flagged, max_gap_for_valid, minimum_valid_streak
    )

    # ---------- PASS 2: Identify island CANDIDATES (motion pattern) ----------
    island_candidates = []  # List of (start_det, end_det, entry_idx, exit_idx)
    jumps = []
    processed_indices = set()

    for idx in outlier_indices:
        if idx in processed_indices:
            continue

        entry_idx = idx
        start_det = idx + 1
        j = start_det

        # Walk forward through calm region
        while j < len(distances) and distances[j] <= jump_threshold_pixels:
            j += 1

        exit_exists = (j < len(distances) and distances[j] > jump_threshold_pixels)
        end_det = j

        candidate_det_indices = np.arange(start_det, end_det + 1, dtype=np.int32)
        candidate_frames = frame_indices[candidate_det_indices] if len(candidate_det_indices) else np.array([], dtype=np.int32)
        island_len = int(len(candidate_frames))

        if exit_exists:
            ok_len = (island_len > 0 and island_len <= max_island_len)
            ok_consec = True if not require_consecutive_for_island else _is_mostly_consecutive(candidate_frames, tolerance=2)

            if ok_len and ok_consec:
                # Store as candidate for spatial validation
                island_candidates.append((start_det, end_det, entry_idx, j))
                processed_indices.add(idx)
                for k in range(idx + 1, end_det + 1):
                    processed_indices.add(k)
                continue
            else:
                # Too long or not consecutive
                jumps.append(int(frame_indices[start_det]))
                processed_indices.add(idx)
                continue
        else:
            # No exit jump
            jumps.append(int(frame_indices[start_det]))
            processed_indices.add(idx)

    # ---------- PASS 3: Validate island candidates spatially ----------
    islands = []
    
    for start_det, end_det, entry_idx, exit_idx in island_candidates:
        candidate_det_indices = np.arange(start_det, end_det + 1, dtype=np.int32)
        
        # Find anchor before the island
        anchor_before_idx = int(prev_anchor[start_det - 1]) if start_det > 0 else -1
        
        if anchor_before_idx >= 0:
            # Check if ANY detection in the candidate island is far from anchor
            is_actually_far = False
            for cand_idx in candidate_det_indices:
                dist_to_anchor = float(np.linalg.norm(centroids_px[cand_idx] - centroids_px[anchor_before_idx]))
                if dist_to_anchor > jump_threshold_pixels:
                    is_actually_far = True
                    break
            
            if is_actually_far:
                # Confirmed as island - add all frames
                islands.extend([int(frame_indices[i]) for i in candidate_det_indices])
            else:
                # Motion pattern looked like island, but spatially on trajectory
                # Don't flag as island
                pass
        else:
            # No anchor available - accept based on motion pattern alone (early in video)
            islands.extend([int(frame_indices[i]) for i in candidate_det_indices])

    # ---------- PASS 4: Rebuild anchors with confirmed islands ----------
    all_flagged_frames = set(islands + jumps)
    prev_anchor = _build_prev_anchor_indices(
        frame_indices, all_flagged_frames, max_gap_for_valid, minimum_valid_streak
    )

    # ---------- PASS 5: Check returns after islands ----------
    unique_islands = sorted(set(islands))
    island_idx_map = {f: int(np.where(frame_indices == f)[0][0]) for f in unique_islands if np.any(frame_indices == f)}

    for island_frame in unique_islands:
        island_det_idx = island_idx_map.get(island_frame, None)
        if island_det_idx is None:
            continue

        # Find island cluster start
        island_cluster_start = island_det_idx
        while island_cluster_start > 0 and frame_indices[island_cluster_start - 1] in all_flagged_frames:
            island_cluster_start -= 1
        
        # Get anchor from before cluster
        anchor_idx = int(prev_anchor[island_cluster_start - 1]) if island_cluster_start > 0 else -1
        if anchor_idx == -1:
            continue

        # Check frames after this island
        for check_idx in range(island_det_idx + 1, min(island_det_idx + 4, len(frame_indices))):
            check_frame = int(frame_indices[check_idx])
            if check_frame in all_flagged_frames:
                continue
            
            dist_to_anchor = float(np.linalg.norm(centroids_px[check_idx] - centroids_px[anchor_idx]))
            if dist_to_anchor > jump_threshold_pixels:
                islands.append(check_frame)
                all_flagged_frames.add(check_frame)
            else:
                break

    # ---------- PASS 6: Blips after long gaps ----------
    blips = []
    for det_idx in range(1, len(frame_indices)):
        frame_gap = int(frame_indices[det_idx] - frame_indices[det_idx - 1])
        current_frame = int(frame_indices[det_idx])

        if current_frame in all_flagged_frames or frame_gap < blip_gap_threshold:
            continue

        anchor_idx = int(prev_anchor[det_idx - 1])
        if anchor_idx == -1:
            continue

        dist_to_anchor = float(np.linalg.norm(centroids_px[det_idx] - centroids_px[anchor_idx]))
        if dist_to_anchor > jump_threshold_pixels:
            blips.append(current_frame)
            all_flagged_frames.add(current_frame)

    return {
        "islands": islands,
        "blips": blips,
        "jumps": jumps,
        "total_artifacts": len(islands) + len(blips) + len(jumps),
        "distances": distances,
        "frame_gaps": frame_gaps,
        "jump_threshold": jump_threshold_pixels,
        "max_gap_for_valid": max_gap_for_valid,
        "minimum_valid_streak": minimum_valid_streak,
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


def check_data_integrity(detect_group: zarr.Group, n_detections: np.ndarray) -> Dict:
    """
    Check basic data integrity issues.

    Args:
        detect_group: Zarr group containing detection data
        n_detections: Detection counts array

    Returns:
        Integrity check results
    """
    issues = []

    # Check array consistency
    bbox_coords = detect_group["bbox_norm_coords"][:]
    expected_bbox_count = int(np.sum(n_detections))
    actual_bbox_count = int(len(bbox_coords))

    if expected_bbox_count != actual_bbox_count:
        issues.append({
            "type": "array_mismatch",
            "message": f"n_detections sum ({expected_bbox_count}) != bbox array length ({actual_bbox_count})",
        })

    # Check for NaN coordinates
    nan_count = int(np.sum(np.isnan(bbox_coords)))
    if nan_count > 0:
        issues.append({
            "type": "nan_coordinates",
            "message": f"Found {nan_count} NaN values in bbox coordinates",
        })

    return {
        "has_issues": len(issues) > 0,
        "issue_count": int(len(issues)),
        "issues": issues,
    }


def save_quality_report(
    zarr_path: str,
    quality_report: Dict,
    console: Optional[Any] = None
) -> str:
    """
    Save quality report to zarr file within the source detect run.

    Creates a quality_reports subgroup within the detect run containing:
    - Full quality metrics
    - Frame indices of artifacts
    - Quality flags array (per-frame quality indicators)

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
        "max_gap_for_valid": quality_report["artifacts"]["max_gap_for_valid"],
        "minimum_valid_streak": quality_report["artifacts"]["minimum_valid_streak"],
    }

    # Create per-frame quality flags array
    # 0 = good, 1 = island, 2 = blip, 3 = jump, 4 = multi-detection
    n_frames = int(quality_report["coverage"]["total_frames"])
    quality_flags = np.zeros(n_frames, dtype="i1")

    # Mark artifacts
    for frame_idx in quality_report["artifacts"]["islands"]:
        if 0 <= frame_idx < n_frames:
            quality_flags[frame_idx] = 1
    for frame_idx in quality_report["artifacts"]["blips"]:
        if 0 <= frame_idx < n_frames:
            quality_flags[frame_idx] = 2
    for frame_idx in quality_report["artifacts"]["jumps"]:
        if 0 <= frame_idx < n_frames:
            quality_flags[frame_idx] = 3

    # Mark multi-detection frames if any
    n_detections = detect_group["n_detections"][:]
    multi_det_frames = np.where(n_detections > 1)[0]
    quality_flags[multi_det_frames] = 4

    # Save quality flags array
    quality_group.create_array("quality_flags", data=quality_flags, chunks=(10000,))

    # Save artifact frame indices as separate arrays for easy access
    quality_group.create_array(
        "island_frames",
        data=np.array(quality_report["artifacts"]["islands"], dtype="i4"),
    )
    quality_group.create_array(
        "blip_frames",
        data=np.array(quality_report["artifacts"]["blips"], dtype="i4"),
    )
    quality_group.create_array(
        "jump_frames",
        data=np.array(quality_report["artifacts"]["jumps"], dtype="i4"),
    )

    # Save gap information if any
    if quality_report["coverage"]["gaps"]["total_count"] > 0:
        quality_group.attrs["gap_stats"] = {
            "total_count": quality_report["coverage"]["gaps"]["total_count"],
            "longest_gap": quality_report["coverage"]["gaps"]["longest_gap"],
            "mean_gap_size": quality_report["coverage"]["gaps"]["mean_gap_size"],
            "categories": quality_report["coverage"]["gaps"]["categories"],
        }

    if console:
        console.print(f"[green]✓[/green] Quality report saved: {quality_group.path}")
    else:
        print(f"Quality report saved: {quality_group.path}")

    return quality_group.path


def analyze_detect_quality(
    zarr_path: str,
    run_name: Optional[str] = None,
    jump_threshold: float = 100.0,
    max_gap_for_valid: int = 3,
    minimum_valid_streak: int = 3,
) -> Dict:
    """
    Comprehensive detection quality analysis.

    Args:
        zarr_path: Path to zarr file
        run_name: Specific detect run to analyze (default: latest)
        jump_threshold: Distance threshold for jump detection (pixels)
        max_gap_for_valid: Max gap size to still consider previous detection as "known valid"

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
        max_gap_for_valid=max_gap_for_valid,
        minimum_valid_streak=minimum_valid_streak,
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
        "--max-gap", type=int, default=3, help="Max gap for valid streak continuity (default: 3)"
    )
    parser.add_argument(
        "--min-streak", type=int, default=3, help="Minimum valid streak length (default: 3)"
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save quality report to zarr (default: True)"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving report to zarr"
    )

    args = parser.parse_args()

    console = Console()

    console.rule("[bold]Detection Quality Analysis[/bold]")
    console.print(f"Zarr: {args.zarr_path}")
    if args.run:
        console.print(f"Run: {args.run}")
    console.print(f"Jump threshold: {args.threshold} pixels")
    console.print(f"Max gap for valid streak: {args.max_gap}")
    console.print(f"Minimum valid streak: {args.min_streak}\n")

    try:
        # Run analysis
        report = analyze_detect_quality(
            args.zarr_path,
            run_name=args.run,
            jump_threshold=args.threshold,
            max_gap_for_valid=args.max_gap,
            minimum_valid_streak=args.min_streak,
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
        if gaps["total_count"] > 0:
            console.print("  Categories:")
            for cat, count in gaps["categories"].items():
                if count > 0:
                    console.print(f"    {cat}: {count}")

        console.print("\n[bold red]ARTIFACTS[/bold red]")
        art = report["artifacts"]
        console.print(f"  Islands: {len(art['islands'])}")
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
