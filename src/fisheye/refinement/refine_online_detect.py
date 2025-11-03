#!/usr/bin/env python3
"""
Online Detection Refinement Pipeline

Refines online (H5-imported) target position data from stimulus runs by applying
smoothing, outlier removal, and gap interpolation to reduce tracking artifacts.

Workflow:
1. Load online target positions from stimulus run (chaser_states)
2. Apply coordinate transformation (texture -> camera space)
3. Smooth positions using Savitzky-Golay filter
4. Detect and remove outliers (large jumps, teleportation artifacts)
5. Interpolate small gaps
6. Save refined data with full provenance and metadata

This creates a new refined_online_runs group similar to refined_detect_runs.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
from scipy.signal import savgol_filter

from ..analysis.chaser_metrics_loader import load_chaser_metrics
from ..utils.calibration import load_run_calibration
from ..utils.system import get_environment_info, get_git_info

REFINED_ONLINE_GROUP = "refined_online_runs"


def load_online_positions(
    zarr_path: str,
    stimulus_run: Optional[str] = None,
    chaser_index: int = 0,
    console: Optional[Console] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, Dict[str, Any]]:
    """Load online target positions from stimulus run in texture space.

    Args:
        zarr_path: Path to zarr archive
        stimulus_run: Stimulus run name (defaults to latest)
        chaser_index: Chaser index to load
        console: Rich console for output

    Returns:
        Tuple of (camera_frames, positions_texture_px, valid_mask, texture_to_camera_scale,
                  pixels_per_mm_projector, metadata)

    Note:
        Positions are returned in TEXTURE SPACE (not camera space) for accurate distance calculations.
        Use pixels_per_mm_projector to convert texture-space distances to millimeters.
    """
    if console is None:
        console = Console()

    # Load chaser metrics bundle
    bundle = load_chaser_metrics(
        zarr_path,
        stimulus_run=stimulus_run,
        chaser_index=chaser_index,
    )

    # Extract online target positions (in texture space)
    target_pos_x_raw = bundle.online.get("target_pos_x")
    target_pos_y_raw = bundle.online.get("target_pos_y")

    if target_pos_x_raw is None or target_pos_y_raw is None:
        raise ValueError("No target position data in online metrics")

    # Load coordinate transformation and projector calibration
    root = zarr.open(str(zarr_path), mode="r")
    stimulus_run_name = bundle.provenance.get("stimulus_run")

    if stimulus_run_name:
        try:
            calibration = load_run_calibration(root, stimulus_run_name)
            texture_to_camera_scale = float(calibration.texture_to_camera_scale)
            pixels_per_mm_projector = calibration.pixels_per_mm_projector
            console.print(
                f"[cyan]Loaded calibration ({calibration.source}): texture_to_camera_scale = {texture_to_camera_scale:.6f}"
            )
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Failed to load calibration: {exc}")
            texture_to_camera_scale = 1.0
            pixels_per_mm_projector = None
    else:
        texture_to_camera_scale = 1.0
        pixels_per_mm_projector = None

    # Keep positions in TEXTURE SPACE (do not transform to camera space)
    # This ensures accurate distance calculations using the projector calibration
    target_pos_x = np.asarray(target_pos_x_raw, dtype=np.float64)
    target_pos_y = np.asarray(target_pos_y_raw, dtype=np.float64)
    camera_frames = bundle.camera_frame_ids

    # Create position array (texture space)
    positions = np.column_stack([target_pos_x, target_pos_y])

    # Valid mask (non-NaN positions)
    valid_mask = np.isfinite(positions[:, 0]) & np.isfinite(positions[:, 1])

    # Metadata
    metadata = {
        "stimulus_run": stimulus_run_name,
        "chaser_index": chaser_index,
        "total_frames": len(camera_frames),
        "valid_frames": int(valid_mask.sum()),
        "coverage_percent": float(valid_mask.sum() / len(camera_frames) * 100),
        "texture_to_camera_scale": texture_to_camera_scale,
        "pixels_per_mm_projector": pixels_per_mm_projector,
        "coordinate_space": "texture",
    }

    return camera_frames, positions, valid_mask, texture_to_camera_scale, pixels_per_mm_projector, metadata


def smooth_positions(
    positions: np.ndarray,
    valid_mask: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth positions using Savitzky-Golay filter.

    Args:
        positions: Position array (N, 2)
        valid_mask: Boolean mask of valid positions
        window_length: Filter window length (must be odd)
        polyorder: Polynomial order for fitting

    Returns:
        Tuple of (smoothed_positions, smoothed_mask)
    """
    smoothed = np.full_like(positions, np.nan)
    smoothed_mask = np.zeros(len(positions), dtype=bool)

    # Need at least window_length consecutive valid points to smooth
    if valid_mask.sum() < window_length:
        return positions.copy(), valid_mask.copy()

    # Find consecutive valid segments
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return positions.copy(), valid_mask.copy()

    # Group consecutive indices
    segments = []
    start_idx = valid_indices[0]
    for i in range(1, len(valid_indices)):
        if valid_indices[i] != valid_indices[i - 1] + 1:
            # Gap found, save segment
            segments.append((start_idx, valid_indices[i - 1]))
            start_idx = valid_indices[i]
    segments.append((start_idx, valid_indices[-1]))

    # Smooth each segment
    for start, end in segments:
        segment_length = end - start + 1

        if segment_length < window_length:
            # Too short to smooth, keep original
            smoothed[start : end + 1] = positions[start : end + 1]
            smoothed_mask[start : end + 1] = True
            continue

        # Apply Savitzky-Golay filter
        for axis in [0, 1]:
            smoothed[start : end + 1, axis] = savgol_filter(
                positions[start : end + 1, axis],
                window_length=window_length,
                polyorder=polyorder,
                mode="interp",
            )

        smoothed_mask[start : end + 1] = True

    return smoothed, smoothed_mask


def detect_outliers(
    positions: np.ndarray,
    frames: np.ndarray,
    valid_mask: np.ndarray,
    displacement_threshold: float = 100.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect outliers based on displacement threshold.

    Args:
        positions: Position array (N, 2)
        frames: Frame indices
        valid_mask: Boolean mask of valid positions
        displacement_threshold: Maximum reasonable displacement (pixels)

    Returns:
        Tuple of (outlier_mask, outlier_stats)
    """
    outlier_mask = np.zeros(len(positions), dtype=bool)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return outlier_mask, {"outliers_detected": 0, "threshold": displacement_threshold}

    # Calculate frame-to-frame displacement
    for i in range(len(valid_indices) - 1):
        idx1 = valid_indices[i]
        idx2 = valid_indices[i + 1]

        # Only check consecutive frames
        if frames[idx2] - frames[idx1] != 1:
            continue

        displacement = np.linalg.norm(positions[idx2] - positions[idx1])

        if displacement > displacement_threshold:
            # Mark the second point as outlier (assumes first is correct)
            outlier_mask[idx2] = True

    outlier_stats = {
        "outliers_detected": int(outlier_mask.sum()),
        "threshold": float(displacement_threshold),
        "outlier_rate": float(outlier_mask.sum() / valid_mask.sum() * 100) if valid_mask.sum() > 0 else 0.0,
    }

    return outlier_mask, outlier_stats


def interpolate_gaps(
    positions: np.ndarray,
    frames: np.ndarray,
    valid_mask: np.ndarray,
    max_gap: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Interpolate small gaps in position data.

    Args:
        positions: Position array (N, 2)
        frames: Frame indices
        valid_mask: Boolean mask of valid positions
        max_gap: Maximum gap size to interpolate (frames)

    Returns:
        Tuple of (interpolated_positions, interpolation_mask, interp_stats)
    """
    interpolated = positions.copy()
    interpolation_mask = np.zeros(len(positions), dtype=bool)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return interpolated, interpolation_mask, {"gaps_filled": 0, "interpolated_frames": 0}

    # Find gaps
    gaps_filled = 0
    interpolated_frames = 0

    for i in range(len(valid_indices) - 1):
        idx1 = valid_indices[i]
        idx2 = valid_indices[i + 1]

        frame1 = frames[idx1]
        frame2 = frames[idx2]
        gap_size = frame2 - frame1 - 1

        if gap_size <= 0 or gap_size > max_gap:
            continue

        # Find indices to interpolate
        gap_indices = []
        for j in range(idx1 + 1, idx2):
            if frames[j] > frame1 and frames[j] < frame2:
                gap_indices.append(j)

        if len(gap_indices) == 0:
            continue

        # Linear interpolation
        t = (frames[gap_indices] - frame1) / (frame2 - frame1)
        for axis in [0, 1]:
            interpolated[gap_indices, axis] = positions[idx1, axis] + t * (
                positions[idx2, axis] - positions[idx1, axis]
            )

        interpolation_mask[gap_indices] = True
        gaps_filled += 1
        interpolated_frames += len(gap_indices)

    interp_stats = {
        "gaps_filled": gaps_filled,
        "interpolated_frames": interpolated_frames,
        "max_gap": max_gap,
    }

    return interpolated, interpolation_mask, interp_stats


def refine_online_positions(
    zarr_path: str,
    stimulus_run: Optional[str] = None,
    chaser_index: int = 0,
    window_length: int = 11,
    polyorder: int = 3,
    displacement_threshold: float = 100.0,
    max_gap: int = 20,
    console: Optional[Console] = None,
    created_at_utc: Optional[str] = None,
) -> str:
    """Refine online target positions with smoothing, outlier removal, and interpolation.

    Args:
        zarr_path: Path to zarr archive
        stimulus_run: Stimulus run name (defaults to latest)
        chaser_index: Chaser index to load
        window_length: Savitzky-Golay filter window length (must be odd)
        polyorder: Polynomial order for Savitzky-Golay filter
        displacement_threshold: Maximum reasonable displacement (pixels)
        max_gap: Maximum gap size to interpolate (frames)
        console: Rich console for output
        created_at_utc: Optional creation timestamp

    Returns:
        Name of created refined run
    """
    if console is None:
        console = Console()

    console.rule("[bold]Online Detection Refinement[/bold]")
    start_time = time.perf_counter()

    # Step 1: Load data
    console.print("[bold]Step 1: Loading Online Data[/bold]")
    frames, positions, valid_mask, scale, pixels_per_mm_projector, metadata = load_online_positions(
        zarr_path, stimulus_run, chaser_index, console
    )

    console.print(f"  Stimulus run: [cyan]{metadata['stimulus_run']}[/cyan]")
    console.print(f"  Total frames: {metadata['total_frames']}")
    console.print(f"  Valid positions: {metadata['valid_frames']} ({metadata['coverage_percent']:.1f}%)")
    console.print(f"  Coordinate space: {metadata['coordinate_space']}")
    if pixels_per_mm_projector:
        console.print(f"  Projector calibration: {pixels_per_mm_projector:.6f} pixels/mm")

    # Step 2: Smooth positions
    console.print("\n[bold]Step 2: Smoothing Positions[/bold]")
    console.print(f"  Window length: {window_length}")
    console.print(f"  Polynomial order: {polyorder}")

    smoothed_positions, smoothed_mask = smooth_positions(
        positions, valid_mask, window_length, polyorder
    )

    console.print(f"  Smoothed frames: {smoothed_mask.sum()}")

    # Step 3: Detect outliers
    console.print("\n[bold]Step 3: Detecting Outliers[/bold]")
    console.print(f"  Displacement threshold: {displacement_threshold} pixels")

    outlier_mask, outlier_stats = detect_outliers(
        smoothed_positions, frames, smoothed_mask, displacement_threshold
    )

    console.print(f"  Outliers detected: {outlier_stats['outliers_detected']} ({outlier_stats['outlier_rate']:.2f}%)")

    # Remove outliers
    clean_mask = smoothed_mask & ~outlier_mask
    clean_positions = smoothed_positions.copy()
    clean_positions[~clean_mask] = np.nan

    # Step 4: Interpolate gaps
    console.print("\n[bold]Step 4: Interpolating Gaps[/bold]")
    console.print(f"  Max gap: {max_gap} frames")

    interpolated_positions, interpolation_mask, interp_stats = interpolate_gaps(
        clean_positions, frames, clean_mask, max_gap
    )

    console.print(f"  Gaps filled: {interp_stats['gaps_filled']}")
    console.print(f"  Interpolated frames: {interp_stats['interpolated_frames']}")

    # Final statistics
    final_valid = np.isfinite(interpolated_positions[:, 0]) & np.isfinite(interpolated_positions[:, 1])
    final_coverage = final_valid.sum() / len(frames) * 100

    console.print("\n[bold]Coverage Comparison:[/bold]")
    console.print(f"  Original: {metadata['valid_frames']} frames ({metadata['coverage_percent']:.1f}%)")
    console.print(f"  After smoothing: {smoothed_mask.sum()} frames ({smoothed_mask.sum()/len(frames)*100:.1f}%)")
    console.print(f"  After outlier removal: {clean_mask.sum()} frames ({clean_mask.sum()/len(frames)*100:.1f}%)")
    console.print(f"  After interpolation: {final_valid.sum()} frames ({final_coverage:.1f}%)")

    # Step 5: Save
    console.print("\n[bold]Step 5: Saving Refined Run[/bold]")

    root = zarr.open(zarr_path, mode="a")

    if REFINED_ONLINE_GROUP not in root:
        root.create_group(REFINED_ONLINE_GROUP)
    refined_runs = root[REFINED_ONLINE_GROUP]

    # Create timestamped run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"refined_online_{timestamp}"
    refined_group = refined_runs.create_group(run_name)
    refined_runs.attrs["latest"] = run_name

    # Save filtered data (after smoothing and outlier removal)
    filtered_grp = refined_group.create_group("filtered")
    filtered_grp.create_array("camera_frame_ids", data=frames, chunks=(10000,))
    filtered_grp.create_array("positions_px", data=clean_positions, chunks=(10000, 2))
    filtered_grp.create_array("valid_mask", data=clean_mask, chunks=(10000,))

    filtered_grp.attrs["total_frames"] = len(frames)
    filtered_grp.attrs["valid_frames"] = int(clean_mask.sum())
    filtered_grp.attrs["coverage_percent"] = float(clean_mask.sum() / len(frames) * 100)
    filtered_grp.attrs["smoothing_applied"] = True
    filtered_grp.attrs["outliers_removed"] = outlier_stats["outliers_detected"]

    # Save interpolated data (final refined positions)
    interp_grp = refined_group.create_group("interpolated")
    interp_grp.create_array("camera_frame_ids", data=frames, chunks=(10000,))
    interp_grp.create_array("positions_px", data=interpolated_positions, chunks=(10000, 2))
    interp_grp.create_array("valid_mask", data=final_valid, chunks=(10000,))
    interp_grp.create_array("interpolation_mask", data=interpolation_mask, chunks=(10000,))

    interp_grp.attrs["total_frames"] = len(frames)
    interp_grp.attrs["valid_frames"] = int(final_valid.sum())
    interp_grp.attrs["coverage_percent"] = float(final_coverage)
    interp_grp.attrs["gaps_filled"] = interp_stats["gaps_filled"]
    interp_grp.attrs["interpolated_frames"] = interp_stats["interpolated_frames"]

    # Store metadata arrays for tracking pipeline stages
    refined_group.create_array("camera_frame_ids", data=frames, chunks=(10000,))
    refined_group.create_array("original_valid_mask", data=valid_mask, chunks=(10000,))
    refined_group.create_array("smoothed_mask", data=smoothed_mask, chunks=(10000,))
    refined_group.create_array("outlier_mask", data=outlier_mask, chunks=(10000,))

    # Metadata
    duration = time.perf_counter() - start_time
    created_timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()

    parameters = {
        "window_length": window_length,
        "polyorder": polyorder,
        "displacement_threshold": displacement_threshold,
        "max_gap": max_gap,
    }

    coverage_stats = {
        "original": {
            "valid_frames": metadata["valid_frames"],
            "coverage_percent": metadata["coverage_percent"],
        },
        "smoothed": {
            "valid_frames": int(smoothed_mask.sum()),
            "coverage_percent": float(smoothed_mask.sum() / len(frames) * 100),
        },
        "clean": {
            "valid_frames": int(clean_mask.sum()),
            "coverage_percent": float(clean_mask.sum() / len(frames) * 100),
            "outliers_removed": outlier_stats["outliers_detected"],
        },
        "final": {
            "valid_frames": int(final_valid.sum()),
            "coverage_percent": float(final_coverage),
            "interpolated_frames": interp_stats["interpolated_frames"],
        },
    }

    git_info = get_git_info()
    env_info = get_environment_info()
    environment_info = {
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "python_version": env_info["platform"].get("python_version", "unknown"),
        "system": env_info["platform"].get("system", "unknown"),
        "release": env_info["platform"].get("release", "unknown"),
    }

    provenance_record = {
        "stage": "refine_online_detect",
        "command": " ".join(sys.argv),
        "created_at_utc": created_timestamp,
        "version": git_info.get("short_hash") or git_info.get("commit_hash"),
        "git": {
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        "environment": environment_info,
        "parameters": parameters,
        "inputs": {
            "stimulus_run": metadata["stimulus_run"],
            "chaser_index": chaser_index,
        },
    }
    provenance_record = {k: v for k, v in provenance_record.items() if v is not None}

    refined_group.attrs["source_stimulus_run"] = metadata["stimulus_run"]
    refined_group.attrs["chaser_index"] = chaser_index
    refined_group.attrs["texture_to_camera_scale"] = scale
    refined_group.attrs["coordinate_space"] = "texture"
    refined_group.attrs["pixels_per_mm_projector"] = pixels_per_mm_projector
    refined_group.attrs["refinement_timestamp"] = created_timestamp
    refined_group.attrs["processing_time_seconds"] = float(duration)
    refined_group.attrs["operations"] = ["smooth", "outlier_removal", "interpolate"]
    refined_group.attrs["parameters"] = parameters
    refined_group.attrs["coverage_stats"] = coverage_stats
    refined_group.attrs["outlier_stats"] = outlier_stats
    refined_group.attrs["interpolation_stats"] = interp_stats
    refined_group.attrs["provenance"] = provenance_record

    console.print(f"[green]✓[/green] Refined run saved: {refined_group.path}")
    console.print(f"[green]✓[/green] Processing completed in {duration:.2f} seconds")

    return run_name


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Refine online target positions from stimulus runs"
    )
    parser.add_argument("zarr_path", help="Path to Palette zarr archive")
    parser.add_argument("--stimulus-run", help="Stimulus run name (defaults to latest)")
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index to process (default: 0)",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=11,
        help="Savitzky-Golay filter window length (must be odd, default: 11)",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=3,
        help="Polynomial order for Savitzky-Golay filter (default: 3)",
    )
    parser.add_argument(
        "--displacement-threshold",
        type=float,
        default=100.0,
        help="Maximum reasonable displacement in pixels (default: 100)",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=20,
        help="Maximum gap size to interpolate in frames (default: 20)",
    )

    args = parser.parse_args(argv)

    console = Console()

    try:
        refine_online_positions(
            zarr_path=args.zarr_path,
            stimulus_run=args.stimulus_run,
            chaser_index=args.chaser_index,
            window_length=args.window_length,
            polyorder=args.polyorder,
            displacement_threshold=args.displacement_threshold,
            max_gap=args.max_gap,
            console=console,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
