#!/usr/bin/env python3
"""
Inspect chaser position and event frame alignment in Zarr archives.

This script analyzes the relationship between stimulus frames and camera frames
to detect potential duplication issues where multiple stimulus frames map to
the same camera frame. It validates both chaser states and events for alignment
problems.

Usage:
    python -m fisheye.diagnostics.inspect_chaser_frame_alignment <zarr_path> \
        [--stimulus-run NAME] [--frame-range START END] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import zarr
import matplotlib.pyplot as plt
import pandas as pd

from fisheye.analysis.chaser_state_interpolator import load_structured_dataset


@dataclass
class FrameMappingStats:
    """Statistics about frame mapping relationships."""
    total_camera_frames: int
    total_stimulus_frames: int
    camera_frames_with_multiple_stim: int
    max_stim_per_camera: int
    mean_stim_per_camera: float
    camera_to_stim_mapping: Dict[int, List[int]]
    stim_to_camera_mapping: Dict[int, int]


@dataclass
class ChaserStateIssues:
    """Issues detected in chaser state alignment."""
    camera_frames_with_multiple_chasers: int
    camera_frames_with_different_positions: int
    duplicate_records: List[Dict]
    position_variance_within_duplicates: float
    position_variance_overall: float


@dataclass
class EventIssues:
    """Issues detected in event alignment."""
    total_events: int
    camera_frames_with_multiple_events: int
    events_with_invalid_camera_frames: int
    events_with_misaligned_frames: int
    out_of_order_events: int
    protocol_start_issues: int
    duplicate_events: List[Dict]


def load_zarr_data(zarr_path: Path, stimulus_run: Optional[str] = None) -> Tuple[zarr.Group, str]:
    """Load Zarr archive and determine stimulus run."""
    root = zarr.open(str(zarr_path), mode='r')

    if stimulus_run is None:
        # Find the most recent stimulus run
        runs = list(root["analysis/stimulus_runs"].group_keys())
        if not runs:
            raise ValueError("No stimulus runs found in Zarr archive")
        stimulus_run = runs[-1]
        print(f"Using most recent stimulus run: {stimulus_run}")

    return root, stimulus_run


def analyze_frame_mapping(
    frame_metadata: np.ndarray,
    frame_range: Optional[Tuple[int, int]] = None
) -> FrameMappingStats:
    """
    Analyze the mapping between stimulus frames and camera frames.

    Args:
        frame_metadata: Structured numpy array containing frame metadata
        frame_range: Optional (start, end) camera frame range to analyze

    Returns:
        FrameMappingStats with detailed mapping information
    """
    # Use metadata directly (already loaded as numpy array)
    metadata = frame_metadata

    # Filter by frame range if specified
    if frame_range is not None:
        start, end = frame_range
        camera_field = 'triggering_camera_frame_id' if 'triggering_camera_frame_id' in metadata.dtype.names else 'camera_frame_id'
        mask = (metadata[camera_field] >= start) & (metadata[camera_field] <= end)
        metadata = metadata[mask]

    # Determine field names
    camera_field = 'triggering_camera_frame_id' if 'triggering_camera_frame_id' in metadata.dtype.names else 'camera_frame_id'
    stim_field = 'stimulus_frame_num' if 'stimulus_frame_num' in metadata.dtype.names else 'frame_number'

    # Build mappings
    camera_to_stim: Dict[int, List[int]] = defaultdict(list)
    stim_to_camera: Dict[int, int] = {}

    for record in metadata:
        camera_id = int(record[camera_field])
        stim_id = int(record[stim_field])

        camera_to_stim[camera_id].append(stim_id)
        stim_to_camera[stim_id] = camera_id  # Note: overwrites if duplicate!

    # Calculate statistics
    stim_counts_per_camera = [len(stims) for stims in camera_to_stim.values()]

    return FrameMappingStats(
        total_camera_frames=len(camera_to_stim),
        total_stimulus_frames=len(stim_to_camera),
        camera_frames_with_multiple_stim=sum(1 for count in stim_counts_per_camera if count > 1),
        max_stim_per_camera=max(stim_counts_per_camera) if stim_counts_per_camera else 0,
        mean_stim_per_camera=np.mean(stim_counts_per_camera) if stim_counts_per_camera else 0.0,
        camera_to_stim_mapping=dict(camera_to_stim),
        stim_to_camera_mapping=stim_to_camera,
    )


def validate_chaser_states(
    chaser_states: np.ndarray,
    frame_mapping: FrameMappingStats,
    frame_range: Optional[Tuple[int, int]] = None
) -> ChaserStateIssues:
    """
    Validate chaser state alignment with camera frames.

    Args:
        chaser_states: Structured numpy array containing chaser state data
        frame_mapping: Frame mapping statistics
        frame_range: Optional (start, end) camera frame range to analyze

    Returns:
        ChaserStateIssues with detailed problem information
    """
    dtype_names = chaser_states.dtype.names or ()

    # Determine which field to use for camera frame reference
    if 'camera_frame_id' in dtype_names:
        camera_frames = np.asarray(chaser_states['camera_frame_id'], dtype=np.int64)
    elif 'triggering_camera_frame_id' in dtype_names:
        camera_frames = np.asarray(chaser_states['triggering_camera_frame_id'], dtype=np.int64)
    else:
        # Need to convert from stimulus_frame_num
        stim_field = 'stimulus_frame_num' if 'stimulus_frame_num' in dtype_names else 'frame_number'
        stim_frames = np.asarray(chaser_states[stim_field], dtype=np.int64)
        camera_frames = np.array(
            [frame_mapping.stim_to_camera_mapping.get(int(s), -1) for s in stim_frames],
            dtype=np.int64
        )

    # Load positions - try different field names
    if 'chaser_pos_x' in dtype_names:
        x_positions = np.asarray(chaser_states['chaser_pos_x'], dtype=np.float64)
        y_positions = np.asarray(chaser_states['chaser_pos_y'], dtype=np.float64)
    elif 'x' in dtype_names:
        x_positions = np.asarray(chaser_states['x'], dtype=np.float64)
        y_positions = np.asarray(chaser_states['y'], dtype=np.float64)
    else:
        raise ValueError(f"Chaser states missing position fields. Available fields: {dtype_names}")

    # Filter by frame range if specified
    if frame_range is not None:
        start, end = frame_range
        mask = (camera_frames >= start) & (camera_frames <= end)
        camera_frames = camera_frames[mask]
        x_positions = x_positions[mask]
        y_positions = y_positions[mask]

    # Build camera frame to chaser records mapping
    camera_to_chasers: Dict[int, List[int]] = defaultdict(list)
    for idx, camera_frame in enumerate(camera_frames):
        if camera_frame >= 0:  # Skip invalid mappings
            camera_to_chasers[camera_frame].append(idx)

    # Analyze duplicates
    duplicate_records = []
    position_variances_within_dups = []
    camera_frames_with_different_positions = 0

    for camera_frame, indices in camera_to_chasers.items():
        if len(indices) > 1:
            # Multiple chaser records for same camera frame
            xs = x_positions[indices]
            ys = y_positions[indices]

            # Check if positions differ
            x_var = np.var(xs)
            y_var = np.var(ys)

            if x_var > 1e-6 or y_var > 1e-6:  # Positions are different
                camera_frames_with_different_positions += 1

            position_variances_within_dups.append(x_var + y_var)

            duplicate_records.append({
                'camera_frame': camera_frame,
                'num_records': len(indices),
                'x_positions': xs.tolist(),
                'y_positions': ys.tolist(),
                'x_variance': x_var,
                'y_variance': y_var,
            })

    # Overall position variance
    position_variance_overall = float(np.var(x_positions) + np.var(y_positions))
    position_variance_within_dups = float(np.mean(position_variances_within_dups)) if position_variances_within_dups else 0.0

    return ChaserStateIssues(
        camera_frames_with_multiple_chasers=len(duplicate_records),
        camera_frames_with_different_positions=camera_frames_with_different_positions,
        duplicate_records=duplicate_records,
        position_variance_within_duplicates=position_variance_within_dups,
        position_variance_overall=position_variance_overall,
    )


def validate_events(
    events: np.ndarray,
    frame_mapping: FrameMappingStats,
    valid_camera_frames: Set[int],
    frame_range: Optional[Tuple[int, int]] = None
) -> EventIssues:
    """
    Validate event alignment with camera frames.

    Args:
        events: Structured numpy array containing event data
        frame_mapping: Frame mapping statistics
        valid_camera_frames: Set of valid camera frame IDs from metadata
        frame_range: Optional (start, end) camera frame range to analyze

    Returns:
        EventIssues with detailed problem information
    """
    # Load event data
    camera_frames = np.asarray(events['camera_frame_id'], dtype=np.int64)
    stimulus_frames = np.asarray(events['stimulus_frame_num'], dtype=np.int64)
    event_types = np.asarray(events['event_type_id'], dtype=np.int64)

    # Filter by frame range if specified
    if frame_range is not None:
        start, end = frame_range
        mask = (camera_frames >= start) & (camera_frames <= end)
        camera_frames = camera_frames[mask]
        stimulus_frames = stimulus_frames[mask]
        event_types = event_types[mask]

    total_events = len(camera_frames)

    # Check for duplicate camera frames in events
    camera_to_events: Dict[int, List[int]] = defaultdict(list)
    for idx, camera_frame in enumerate(camera_frames):
        camera_to_events[camera_frame].append(idx)

    duplicate_events = []
    for camera_frame, indices in camera_to_events.items():
        if len(indices) > 1:
            duplicate_events.append({
                'camera_frame': camera_frame,
                'num_events': len(indices),
                'event_types': event_types[indices].tolist(),
                'stimulus_frames': stimulus_frames[indices].tolist(),
            })

    # Check for invalid camera frames
    events_with_invalid_camera_frames = np.sum(~np.isin(camera_frames, list(valid_camera_frames)))

    # Check for misaligned frames (stimulus→camera mapping doesn't match event's camera_frame)
    events_with_misaligned_frames = 0
    for idx in range(len(camera_frames)):
        expected_camera = frame_mapping.stim_to_camera_mapping.get(int(stimulus_frames[idx]), -1)
        if expected_camera != -1 and expected_camera != camera_frames[idx]:
            events_with_misaligned_frames += 1

    # Check for out-of-order events
    out_of_order_events = 0
    for i in range(1, len(camera_frames)):
        if camera_frames[i] < camera_frames[i-1]:
            out_of_order_events += 1

    # Check PROTOCOL_START events (event_type_id == 0)
    protocol_start_indices = np.where(event_types == 0)[0]
    protocol_start_issues = 0
    for idx in protocol_start_indices:
        # PROTOCOL_START should have stimulus_frame_num=0
        if stimulus_frames[idx] != 0:
            protocol_start_issues += 1

    return EventIssues(
        total_events=total_events,
        camera_frames_with_multiple_events=len(duplicate_events),
        events_with_invalid_camera_frames=int(events_with_invalid_camera_frames),
        events_with_misaligned_frames=events_with_misaligned_frames,
        out_of_order_events=out_of_order_events,
        protocol_start_issues=protocol_start_issues,
        duplicate_events=duplicate_events,
    )


def print_summary(
    frame_mapping: FrameMappingStats,
    chaser_issues: ChaserStateIssues,
    event_issues: EventIssues
) -> None:
    """Print a comprehensive summary of findings."""
    print("\n" + "="*80)
    print("FRAME ALIGNMENT INSPECTION SUMMARY")
    print("="*80)

    print("\n--- Frame Mapping Statistics ---")
    print(f"Total camera frames: {frame_mapping.total_camera_frames}")
    print(f"Total stimulus frames: {frame_mapping.total_stimulus_frames}")
    print(f"Stimulus:Camera ratio: {frame_mapping.mean_stim_per_camera:.2f}")
    print(f"Camera frames with multiple stimulus frames: {frame_mapping.camera_frames_with_multiple_stim}")
    print(f"  ({100*frame_mapping.camera_frames_with_multiple_stim/frame_mapping.total_camera_frames:.1f}% of camera frames)")
    print(f"Max stimulus frames per camera: {frame_mapping.max_stim_per_camera}")

    print("\n--- Chaser State Issues ---")
    print(f"Camera frames with multiple chaser records: {chaser_issues.camera_frames_with_multiple_chasers}")
    print(f"Camera frames with DIFFERENT positions: {chaser_issues.camera_frames_with_different_positions}")
    if chaser_issues.camera_frames_with_multiple_chasers > 0:
        print(f"  ({100*chaser_issues.camera_frames_with_different_positions/chaser_issues.camera_frames_with_multiple_chasers:.1f}% of duplicates have different positions)")
    print(f"Position variance (overall): {chaser_issues.position_variance_overall:.4f}")
    print(f"Position variance (within duplicates): {chaser_issues.position_variance_within_duplicates:.4f}")

    print("\n--- Event Issues ---")
    print(f"Total events: {event_issues.total_events}")
    print(f"Camera frames with multiple events: {event_issues.camera_frames_with_multiple_events}")
    print(f"Events with invalid camera frames: {event_issues.events_with_invalid_camera_frames}")
    print(f"Events with misaligned frames: {event_issues.events_with_misaligned_frames}")
    print(f"Out-of-order events: {event_issues.out_of_order_events}")
    print(f"PROTOCOL_START events with issues: {event_issues.protocol_start_issues}")

    print("\n--- Recommendations ---")
    if chaser_issues.camera_frames_with_different_positions > 0:
        print("WARNING: Found chaser records with DIFFERENT positions for the same camera frame!")
        print("This indicates the 1:1 duplication problem is real and needs fixing.")
        print("\nSuggested fix strategies:")
        print("  1. Keep FIRST stimulus frame per camera (earliest chaser position)")
        print("  2. Keep LAST stimulus frame per camera (latest chaser position)")
        print("  3. Keep stimulus frame CLOSEST to camera timestamp")
        print("  4. INTERPOLATE/AVERAGE chaser positions when multiple exist")
    else:
        print("No significant chaser position duplication issues detected.")

    if event_issues.camera_frames_with_multiple_events > 0:
        print(f"\nWARNING: Found {event_issues.camera_frames_with_multiple_events} camera frames with duplicate events!")
        print("Review duplicate_events.csv for details.")

    if event_issues.protocol_start_issues > 0:
        print(f"\nWARNING: Found {event_issues.protocol_start_issues} PROTOCOL_START events with incorrect stimulus_frame_num!")
        print("These should be corrected using fix_protocol_start_frames.py")

    print("\n" + "="*80)


def export_diagnostics(
    output_dir: Path,
    frame_mapping: FrameMappingStats,
    chaser_issues: ChaserStateIssues,
    event_issues: EventIssues
) -> None:
    """Export diagnostic data to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export camera frames with multiple stimulus frames
    if frame_mapping.camera_frames_with_multiple_stim > 0:
        rows = []
        for camera_frame, stim_frames in frame_mapping.camera_to_stim_mapping.items():
            if len(stim_frames) > 1:
                rows.append({
                    'camera_frame': camera_frame,
                    'num_stimulus_frames': len(stim_frames),
                    'stimulus_frames': stim_frames,
                })
        df = pd.DataFrame(rows)
        csv_path = output_dir / "camera_frames_with_multiple_stimulus.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nExported: {csv_path}")

    # Export chaser state duplicates
    if chaser_issues.duplicate_records:
        df = pd.DataFrame(chaser_issues.duplicate_records)
        csv_path = output_dir / "duplicate_chaser_records.csv"
        df.to_csv(csv_path, index=False)
        print(f"Exported: {csv_path}")

    # Export event duplicates
    if event_issues.duplicate_events:
        df = pd.DataFrame(event_issues.duplicate_events)
        csv_path = output_dir / "duplicate_events.csv"
        df.to_csv(csv_path, index=False)
        print(f"Exported: {csv_path}")


def plot_diagnostics(
    output_dir: Path,
    chaser_states: np.ndarray,
    events: np.ndarray,
    chaser_issues: ChaserStateIssues,
    frame_range: Optional[Tuple[int, int]] = None
) -> None:
    """Generate diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_names = chaser_states.dtype.names or ()

    # Load chaser data
    if 'camera_frame_id' in dtype_names:
        camera_frames = np.asarray(chaser_states['camera_frame_id'], dtype=np.int64)
    elif 'triggering_camera_frame_id' in dtype_names:
        camera_frames = np.asarray(chaser_states['triggering_camera_frame_id'], dtype=np.int64)
    else:
        print("Warning: Cannot plot chaser positions without camera_frame_id")
        return

    # Load positions
    if 'chaser_pos_x' in dtype_names:
        x_positions = np.asarray(chaser_states['chaser_pos_x'], dtype=np.float64)
        y_positions = np.asarray(chaser_states['chaser_pos_y'], dtype=np.float64)
    elif 'x' in dtype_names:
        x_positions = np.asarray(chaser_states['x'], dtype=np.float64)
        y_positions = np.asarray(chaser_states['y'], dtype=np.float64)
    else:
        print("Warning: Cannot plot chaser positions without position fields")
        return

    # Filter by frame range
    if frame_range is not None:
        start, end = frame_range
        mask = (camera_frames >= start) & (camera_frames <= end)
        camera_frames = camera_frames[mask]
        x_positions = x_positions[mask]
        y_positions = y_positions[mask]

    # Identify duplicate camera frames
    duplicate_camera_frames = set()
    for record in chaser_issues.duplicate_records:
        duplicate_camera_frames.add(record['camera_frame'])

    is_duplicate = np.isin(camera_frames, list(duplicate_camera_frames))

    # Plot chaser positions over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot non-duplicates
    ax1.scatter(camera_frames[~is_duplicate], x_positions[~is_duplicate],
                c='blue', s=1, alpha=0.5, label='Normal')
    ax1.scatter(camera_frames[is_duplicate], x_positions[is_duplicate],
                c='red', s=3, alpha=0.8, label='Duplicate camera frame')
    ax1.set_ylabel('Chaser X Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Chaser Position Over Time (Duplicates Highlighted)')

    ax2.scatter(camera_frames[~is_duplicate], y_positions[~is_duplicate],
                c='blue', s=1, alpha=0.5, label='Normal')
    ax2.scatter(camera_frames[is_duplicate], y_positions[is_duplicate],
                c='red', s=3, alpha=0.8, label='Duplicate camera frame')
    ax2.set_ylabel('Chaser Y Position')
    ax2.set_xlabel('Camera Frame ID')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "chaser_positions_with_duplicates.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved: {plot_path}")
    plt.close()

    # Plot event timeline if events exist
    if len(events) > 0:
        event_camera_frames = np.asarray(events['camera_frame_id'], dtype=np.int64)
        event_types = np.asarray(events['event_type_id'], dtype=np.int64)

        if frame_range is not None:
            start, end = frame_range
            mask = (event_camera_frames >= start) & (event_camera_frames <= end)
            event_camera_frames = event_camera_frames[mask]
            event_types = event_types[mask]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.scatter(event_camera_frames, event_types, s=10, alpha=0.6)
        ax.set_xlabel('Camera Frame ID')
        ax.set_ylabel('Event Type ID')
        ax.set_title('Event Timeline')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "event_timeline.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect chaser position and event frame alignment in Zarr archives"
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Zarr archive")
    parser.add_argument("--stimulus-run", type=str, help="Stimulus run name (default: most recent)")
    parser.add_argument("--frame-range", type=int, nargs=2, metavar=("START", "END"),
                        help="Camera frame range to analyze")
    parser.add_argument("--output-dir", type=Path, default=Path("./alignment_diagnostics"),
                        help="Output directory for diagnostics (default: ./alignment_diagnostics)")

    args = parser.parse_args()

    # Validate inputs
    if not args.zarr_path.exists():
        print(f"Error: Zarr path does not exist: {args.zarr_path}", file=sys.stderr)
        return 1

    frame_range = tuple(args.frame_range) if args.frame_range else None

    # Load data
    print(f"Loading Zarr archive: {args.zarr_path}")
    root, stimulus_run = load_zarr_data(args.zarr_path, args.stimulus_run)

    run_group = root[f"analysis/stimulus_runs/{stimulus_run}"]

    # Load required arrays - metadata is under video_metadata subgroup
    meta_group = run_group["video_metadata"]

    # Try camera_aligned_frame_metadata first, fall back to frame_metadata
    if "camera_aligned_frame_metadata" in meta_group:
        frame_metadata, _ = load_structured_dataset(meta_group, "camera_aligned_frame_metadata")
    elif "frame_metadata" in meta_group:
        frame_metadata, _ = load_structured_dataset(meta_group, "frame_metadata")
    else:
        raise ValueError("No frame_metadata or camera_aligned_frame_metadata found in video_metadata group")

    # Load chaser states from tracking_data subgroup
    tracking_group = run_group["tracking_data"]
    chaser_states, _ = load_structured_dataset(tracking_group, "chaser_states")

    # Load events - they're stored in columnar format
    # Events may have complex string fields, so we load only the numeric fields we need
    try:
        events_group = run_group["events"]
        # Manually load only the fields we need to avoid string field issues
        event_camera_frames = np.asarray(events_group['camera_frame_id'][:], dtype=np.int64)
        event_stimulus_frames = np.asarray(events_group['stimulus_frame_num'][:], dtype=np.int64)
        event_type_ids = np.asarray(events_group['event_type_id'][:], dtype=np.int64)

        # Create a minimal structured array with just the fields we need
        events = np.empty(
            len(event_camera_frames),
            dtype=[
                ('camera_frame_id', np.int64),
                ('stimulus_frame_num', np.int64),
                ('event_type_id', np.int64),
            ]
        )
        events['camera_frame_id'] = event_camera_frames
        events['stimulus_frame_num'] = event_stimulus_frames
        events['event_type_id'] = event_type_ids
    except Exception as e:
        print(f"Warning: Could not load events: {e}")
        print("Skipping event validation...")
        events = np.array([], dtype=[
            ('camera_frame_id', np.int64),
            ('stimulus_frame_num', np.int64),
            ('event_type_id', np.int64),
        ])

    print(f"Analyzing stimulus run: {stimulus_run}")
    if frame_range:
        print(f"Frame range: {frame_range[0]} to {frame_range[1]}")

    # Analyze frame mapping
    print("\n[1/3] Analyzing frame mappings...")
    frame_mapping = analyze_frame_mapping(frame_metadata, frame_range)

    # Validate chaser states
    print("[2/3] Validating chaser states...")
    chaser_issues = validate_chaser_states(chaser_states, frame_mapping, frame_range)

    # Validate events
    print("[3/3] Validating events...")
    valid_camera_frames = set(frame_mapping.camera_to_stim_mapping.keys())
    event_issues = validate_events(events, frame_mapping, valid_camera_frames, frame_range)

    # Print summary
    print_summary(frame_mapping, chaser_issues, event_issues)

    # Export diagnostics
    print(f"\nExporting diagnostics to: {args.output_dir}")
    export_diagnostics(args.output_dir, frame_mapping, chaser_issues, event_issues)

    # Generate plots
    print("\nGenerating plots...")
    plot_diagnostics(args.output_dir, chaser_states, events, chaser_issues, frame_range)

    print("\nInspection complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
