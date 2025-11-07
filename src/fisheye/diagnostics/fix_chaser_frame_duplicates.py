#!/usr/bin/env python3
"""
Fix chaser position and event frame duplicates by keeping the last stimulus frame per camera.

This script deduplicates chaser states and events that have multiple records for the
same camera frame by keeping only the LAST stimulus frame for each camera frame.

Usage:
    python -m fisheye.diagnostics.fix_chaser_frame_duplicates <zarr_path> \
        [--stimulus-run NAME] [--dry-run] [--backup]
"""

import argparse
import sys
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import zarr

from fisheye.analysis.chaser_state_interpolator import load_structured_dataset


def deduplicate_by_last_stimulus(
    data: np.ndarray,
    camera_field: str = 'camera_frame_id',
    stimulus_field: str = 'stimulus_frame_num',
) -> Tuple[np.ndarray, Dict]:
    """
    Deduplicate structured array by keeping the last stimulus frame per camera frame.

    Args:
        data: Structured numpy array with camera_frame_id and stimulus_frame_num fields
        camera_field: Name of the camera frame ID field
        stimulus_field: Name of the stimulus frame number field

    Returns:
        Tuple of (deduplicated_array, stats_dict)
    """
    if len(data) == 0:
        return data, {'original_count': 0, 'deduplicated_count': 0, 'removed_count': 0}

    camera_frames = data[camera_field]
    stimulus_frames = data[stimulus_field]

    # Build mapping: camera_frame -> list of (index, stimulus_frame)
    camera_to_records: Dict[int, list] = defaultdict(list)
    for idx in range(len(data)):
        camera_id = int(camera_frames[idx])
        stim_id = int(stimulus_frames[idx])
        camera_to_records[camera_id].append((idx, stim_id))

    # For each camera frame, keep only the record with the HIGHEST stimulus frame number
    indices_to_keep = []
    duplicates_removed = 0

    for camera_id, records in camera_to_records.items():
        if len(records) == 1:
            # No duplicates, keep the only record
            indices_to_keep.append(records[0][0])
        else:
            # Multiple records - keep the one with highest stimulus frame number (last)
            records_sorted = sorted(records, key=lambda x: x[1])  # Sort by stimulus frame
            indices_to_keep.append(records_sorted[-1][0])  # Keep last (highest)
            duplicates_removed += len(records) - 1

    # Sort indices to maintain temporal order
    indices_to_keep = sorted(indices_to_keep)

    deduplicated = data[indices_to_keep]

    stats = {
        'original_count': len(data),
        'deduplicated_count': len(deduplicated),
        'removed_count': len(data) - len(deduplicated),
        'camera_frames_with_duplicates': sum(1 for records in camera_to_records.values() if len(records) > 1),
    }

    return deduplicated, stats


def write_columnar_dataset(
    group: zarr.Group,
    name: str,
    data: np.ndarray,
    attrs: Optional[Dict] = None,
) -> None:
    """
    Write a structured numpy array as a columnar Zarr group.

    Args:
        group: Parent Zarr group
        name: Name for the dataset
        data: Structured numpy array to write
        attrs: Optional attributes to set on the group
    """
    if name in group:
        del group[name]

    dataset_group = group.create_group(name)

    # Store field names
    field_names = list(data.dtype.names or [])
    dataset_group.attrs['field_names'] = field_names
    dataset_group.attrs['storage_layout'] = 'columnar'

    # Store field dtypes
    field_dtypes = {field: str(data.dtype[field]) for field in field_names}
    dataset_group.attrs['field_dtypes'] = field_dtypes

    # Write each field as a separate array
    # Zarr v3 uses codec objects instead of compressor dicts
    serializer = zarr.codecs.BytesCodec()
    compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")

    for field in field_names:
        field_data = data[field]
        # Zarr v3 API - pass data and let shape/dtype be inferred
        dataset_group.create_array(
            field,
            data=field_data,
            chunks=(min(10000, len(field_data)),),
            serializer=serializer,
            compressors=[compressor],
        )

    # Set additional attributes
    if attrs:
        for key, value in attrs.items():
            if key not in ['field_names', 'storage_layout', 'field_dtypes']:
                dataset_group.attrs[key] = value


def backup_dataset(zarr_path: Path, stimulus_run: str, dataset_path: str) -> Path:
    """Create a backup of a dataset before modifying it."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_name = f"{dataset_path.replace('/', '_')}_backup_{timestamp}"

    root = zarr.open(str(zarr_path), mode='r')
    full_path = f"analysis/stimulus_runs/{stimulus_run}/{dataset_path}"

    # Check if it's a group or array
    node = root[full_path]

    backup_path = zarr_path / "backups" / stimulus_run / backup_name
    backup_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(node, zarr.Group):
        # Copy entire group
        source_path = zarr_path / full_path
        shutil.copytree(source_path, backup_path)
    else:
        # Copy array
        source_path = zarr_path / full_path
        shutil.copytree(source_path, backup_path)

    return backup_path


def fix_chaser_states(
    zarr_path: Path,
    stimulus_run: str,
    dry_run: bool = False,
    create_backup: bool = True,
) -> Dict:
    """
    Fix chaser state duplicates by keeping the last stimulus frame per camera.

    Args:
        zarr_path: Path to Zarr archive
        stimulus_run: Stimulus run name
        dry_run: If True, don't write changes
        create_backup: If True, create backup before modifying

    Returns:
        Dictionary with statistics about the fix
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Fixing chaser states...")

    root = zarr.open(str(zarr_path), mode='r' if dry_run else 'r+')
    run_group = root[f"analysis/stimulus_runs/{stimulus_run}"]
    tracking_group = run_group["tracking_data"]

    # Load chaser states
    try:
        chaser_states, chaser_attrs = load_structured_dataset(tracking_group, "chaser_states")
        print(f"Loaded {len(chaser_states)} chaser state records")
    except KeyError as e:
        # If loading fails, the dataset may be corrupted from a previous failed run
        # Check if there's a backup we can restore from
        print(f"Error loading chaser_states: {e}")
        print("The dataset may be corrupted from a previous interrupted run.")
        print("Looking for recent backups...")

        backup_dir = zarr_path / "backups" / stimulus_run
        if backup_dir.exists():
            backups = sorted(backup_dir.glob("tracking_data_chaser_states_backup_*"))
            if backups:
                latest_backup = backups[-1]
                print(f"Found backup: {latest_backup}")
                print("Please restore from backup manually, or delete the corrupted chaser_states group.")
                print(f"\nTo restore: Delete the corrupted group and copy from backup:")
                print(f"  rm -rf {zarr_path}/analysis/stimulus_runs/{stimulus_run}/tracking_data/chaser_states")
                print(f"  cp -r {latest_backup} {zarr_path}/analysis/stimulus_runs/{stimulus_run}/tracking_data/chaser_states")
                return {'original_count': 0, 'deduplicated_count': 0, 'removed_count': 0}

        raise

    # Determine field names
    dtype_names = chaser_states.dtype.names or ()
    print(f"Available fields: {dtype_names}")

    # Check if camera_frame_id exists
    if 'camera_frame_id' in dtype_names:
        camera_field = 'camera_frame_id'
    elif 'triggering_camera_frame_id' in dtype_names:
        camera_field = 'triggering_camera_frame_id'
    else:
        # Camera frame ID doesn't exist - chaser states only have stimulus_frame_num
        # We need to add camera_frame_id by building the mapping from frame_metadata
        print("Chaser states don't have camera_frame_id field.")
        print("Building camera_frame_id from frame_metadata mapping...")

        # Load frame metadata to build stimulus->camera mapping
        meta_group = run_group["video_metadata"]
        if "camera_aligned_frame_metadata" in meta_group:
            frame_metadata, _ = load_structured_dataset(meta_group, "camera_aligned_frame_metadata")
        elif "frame_metadata" in meta_group:
            frame_metadata, _ = load_structured_dataset(meta_group, "frame_metadata")
        else:
            raise ValueError("No frame_metadata found to build stimulus->camera mapping")

        # Build stim_to_camera mapping (keeps last camera per stimulus)
        meta_dtype_names = frame_metadata.dtype.names or ()
        meta_camera_field = 'triggering_camera_frame_id' if 'triggering_camera_frame_id' in meta_dtype_names else 'camera_frame_id'
        meta_stim_field = 'stimulus_frame_num' if 'stimulus_frame_num' in meta_dtype_names else 'frame_number'

        stim_to_camera = {}
        for record in frame_metadata:
            stim_id = int(record[meta_stim_field])
            camera_id = int(record[meta_camera_field])
            stim_to_camera[stim_id] = camera_id

        # Add camera_frame_id field to chaser states
        stim_field = 'stimulus_frame_num' if 'stimulus_frame_num' in dtype_names else 'frame_number'
        stimulus_frames = chaser_states[stim_field]

        # Create new structured array with camera_frame_id added
        new_dtype = list(chaser_states.dtype.descr)
        new_dtype.insert(0, ('camera_frame_id', '<i8'))  # Insert at beginning

        chaser_states_with_camera = np.empty(len(chaser_states), dtype=new_dtype)

        # Copy all existing fields
        for field in dtype_names:
            chaser_states_with_camera[field] = chaser_states[field]

        # Add camera_frame_id
        chaser_states_with_camera['camera_frame_id'] = np.array(
            [stim_to_camera.get(int(s), -1) for s in stimulus_frames],
            dtype=np.int64
        )

        chaser_states = chaser_states_with_camera
        camera_field = 'camera_frame_id'

        print(f"Added camera_frame_id field using stimulus->camera mapping")

    stimulus_field = 'stimulus_frame_num' if 'stimulus_frame_num' in chaser_states.dtype.names else 'frame_number'

    # Deduplicate
    deduplicated, stats = deduplicate_by_last_stimulus(chaser_states, camera_field, stimulus_field)

    print(f"Original records: {stats['original_count']}")
    print(f"After deduplication: {stats['deduplicated_count']}")
    print(f"Records removed: {stats['removed_count']}")
    print(f"Camera frames that had duplicates: {stats['camera_frames_with_duplicates']}")

    if not dry_run:
        # Create backup if requested
        if create_backup:
            backup_path = backup_dataset(zarr_path, stimulus_run, "tracking_data/chaser_states")
            print(f"Backup created: {backup_path}")

        # Write deduplicated data
        write_columnar_dataset(tracking_group, "chaser_states", deduplicated, chaser_attrs)

        # Add metadata about the fix
        chaser_group = tracking_group["chaser_states"]
        chaser_group.attrs['deduplication_timestamp'] = datetime.now(timezone.utc).isoformat()
        chaser_group.attrs['deduplication_strategy'] = 'keep_last_stimulus_per_camera'
        chaser_group.attrs['original_record_count'] = stats['original_count']
        chaser_group.attrs['deduplicated_record_count'] = stats['deduplicated_count']

        print("Chaser states updated successfully!")

    return stats


def fix_events(
    zarr_path: Path,
    stimulus_run: str,
    dry_run: bool = False,
    create_backup: bool = True,
) -> Dict:
    """
    Fix event duplicates by keeping the last stimulus frame per camera.

    Args:
        zarr_path: Path to Zarr archive
        stimulus_run: Stimulus run name
        dry_run: If True, don't write changes
        create_backup: If True, create backup before modifying

    Returns:
        Dictionary with statistics about the fix
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Fixing events...")

    root = zarr.open(str(zarr_path), mode='r' if dry_run else 'r+')
    run_group = root[f"analysis/stimulus_runs/{stimulus_run}"]

    # Load events
    try:
        events, event_attrs = load_structured_dataset(run_group, "events")
    except Exception as e:
        print(f"Could not load events using load_structured_dataset: {e}")
        print("Events likely don't need fixing or don't exist.")
        return {'original_count': 0, 'deduplicated_count': 0, 'removed_count': 0}

    print(f"Loaded {len(events)} event records")

    # Deduplicate
    deduplicated, stats = deduplicate_by_last_stimulus(events, 'camera_frame_id', 'stimulus_frame_num')

    print(f"Original records: {stats['original_count']}")
    print(f"After deduplication: {stats['deduplicated_count']}")
    print(f"Records removed: {stats['removed_count']}")
    print(f"Camera frames that had duplicates: {stats['camera_frames_with_duplicates']}")

    if not dry_run and stats['removed_count'] > 0:
        # Create backup if requested
        if create_backup:
            backup_path = backup_dataset(zarr_path, stimulus_run, "events")
            print(f"Backup created: {backup_path}")

        # Write deduplicated data
        write_columnar_dataset(run_group, "events", deduplicated, event_attrs)

        # Add metadata about the fix
        events_group = run_group["events"]
        events_group.attrs['deduplication_timestamp'] = datetime.now(timezone.utc).isoformat()
        events_group.attrs['deduplication_strategy'] = 'keep_last_stimulus_per_camera'
        events_group.attrs['original_record_count'] = stats['original_count']
        events_group.attrs['deduplicated_record_count'] = stats['deduplicated_count']

        print("Events updated successfully!")
    elif stats['removed_count'] == 0:
        print("No duplicate events found, skipping update.")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fix chaser and event frame duplicates by keeping last stimulus frame per camera"
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Zarr archive")
    parser.add_argument("--stimulus-run", type=str, help="Stimulus run name (default: most recent)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups before modifying")
    parser.add_argument("--chaser-only", action="store_true", help="Only fix chaser states, skip events")
    parser.add_argument("--events-only", action="store_true", help="Only fix events, skip chaser states")

    args = parser.parse_args()

    # Validate inputs
    if not args.zarr_path.exists():
        print(f"Error: Zarr path does not exist: {args.zarr_path}", file=sys.stderr)
        return 1

    # Load Zarr and determine stimulus run
    print(f"Loading Zarr archive: {args.zarr_path}")
    root = zarr.open(str(args.zarr_path), mode='r')

    if args.stimulus_run is None:
        runs = list(root["analysis/stimulus_runs"].group_keys())
        if not runs:
            print("Error: No stimulus runs found in Zarr archive", file=sys.stderr)
            return 1
        stimulus_run = runs[-1]
        print(f"Using most recent stimulus run: {stimulus_run}")
    else:
        stimulus_run = args.stimulus_run

    print(f"\n{'='*80}")
    print(f"CHASER/EVENT DEDUPLICATION {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'='*80}")
    print(f"Zarr archive: {args.zarr_path}")
    print(f"Stimulus run: {stimulus_run}")
    print(f"Strategy: Keep LAST stimulus frame per camera frame")
    print(f"Backup: {'No' if args.no_backup else 'Yes'}")
    print(f"{'='*80}")

    create_backup = not args.no_backup

    # Fix chaser states
    if not args.events_only:
        chaser_stats = fix_chaser_states(args.zarr_path, stimulus_run, args.dry_run, create_backup)
    else:
        chaser_stats = None

    # Fix events
    if not args.chaser_only:
        event_stats = fix_events(args.zarr_path, stimulus_run, args.dry_run, create_backup)
    else:
        event_stats = None

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if chaser_stats:
        print(f"Chaser states: Removed {chaser_stats['removed_count']} duplicate records")

    if event_stats:
        print(f"Events: Removed {event_stats['removed_count']} duplicate records")

    if args.dry_run:
        print("\n[DRY RUN] No changes were made. Run without --dry-run to apply fixes.")
    else:
        print("\nDeduplication complete!")
        print("\nNext steps:")
        print("1. Run the inspection script again to verify the fix")
        print("2. Check that downstream analysis still works correctly")
        if create_backup:
            print(f"3. If issues occur, backups are in: {args.zarr_path}/backups/{stimulus_run}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
