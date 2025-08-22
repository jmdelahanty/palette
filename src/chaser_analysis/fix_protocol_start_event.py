#!/usr/bin/env python3
"""
Fix PROTOCOL_START event frame ID in analysis.h5 files.

This script corrects the issue where PROTOCOL_START events have stale frame IDs
from previous recordings that weren't cleared from shared memory queues.
It updates the PROTOCOL_START event to use the first valid frame ID from the
actual recording.
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import shutil

# Event type mappings
EXPERIMENT_EVENT_TYPE = {
    0: "PROTOCOL_START",
    4: "PROTOCOL_FINISH",
    24: "CHASER_PRE_PERIOD_START",
    25: "CHASER_TRAINING_START",
    26: "CHASER_POST_PERIOD_START",
    27: "CHASER_CHASE_SEQUENCE_START",
    28: "CHASER_CHASE_SEQUENCE_END",
}

def fix_protocol_start_event(h5_path: str, backup: bool = True, verbose: bool = True):
    """
    Fix PROTOCOL_START event to use the correct frame ID.
    
    Args:
        h5_path: Path to the analysis.h5 file
        backup: Whether to create a backup before modifying
        verbose: Print detailed information
    """
    h5_path = Path(h5_path)
    
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")
    
    # Create backup if requested
    if backup:
        backup_path = h5_path.with_suffix('.h5.bak')
        if verbose:
            print(f"📁 Creating backup: {backup_path}")
        shutil.copy2(h5_path, backup_path)
    
    print(f"\n🔧 Fixing PROTOCOL_START event in: {h5_path.name}")
    print("=" * 70)
    
    with h5py.File(h5_path, 'r+') as f:
        # Load events
        if '/events' not in f:
            print("❌ No /events dataset found!")
            return
        
        events = f['/events']
        events_data = events[:]
        
        # Find event type field
        event_type_field = None
        for field in ['event_type_id', 'event_type', 'type']:
            if field in events.dtype.names:
                event_type_field = field
                break
        
        if not event_type_field:
            print("❌ Could not find event type field!")
            return
        
        # Check if camera_frame_id field exists
        if 'camera_frame_id' not in events.dtype.names:
            print("⚠️  No camera_frame_id field in events - nothing to fix")
            return
        
        # Find PROTOCOL_START event
        protocol_start_idx = None
        for i, event in enumerate(events_data):
            if event[event_type_field] == 0:  # PROTOCOL_START
                protocol_start_idx = i
                break
        
        if protocol_start_idx is None:
            print("⚠️  No PROTOCOL_START event found")
            return
        
        # Get current PROTOCOL_START frame ID
        old_frame_id = events_data[protocol_start_idx]['camera_frame_id']
        
        if verbose:
            print(f"\n📊 Current PROTOCOL_START state:")
            print(f"  Event index: {protocol_start_idx}")
            print(f"  Current camera_frame_id: {old_frame_id}")
        
        # Find the first valid frame ID from the recording
        # Option 1: Use the minimum frame ID from frame_metadata
        min_metadata_frame = None
        if '/video_metadata/frame_metadata' in f:
            metadata = f['/video_metadata/frame_metadata'][:]
            if 'triggering_camera_frame_id' in metadata.dtype.names:
                valid_frames = metadata['triggering_camera_frame_id'][metadata['triggering_camera_frame_id'] > 0]
                if len(valid_frames) > 0:
                    min_metadata_frame = np.min(valid_frames)
                    if verbose:
                        print(f"\n  Minimum frame from metadata: {min_metadata_frame}")
        
        # Option 2: Use the next event's frame ID (usually CHASER_PRE_PERIOD_START)
        next_event_frame = None
        if protocol_start_idx + 1 < len(events_data):
            next_frame = events_data[protocol_start_idx + 1]['camera_frame_id']
            if next_frame > 0:
                next_event_frame = next_frame
                next_event_type = events_data[protocol_start_idx + 1][event_type_field]
                next_event_name = EXPERIMENT_EVENT_TYPE.get(next_event_type, f"ID_{next_event_type}")
                if verbose:
                    print(f"  Next event ({next_event_name}) frame: {next_event_frame}")
        
        # Option 3: Use minimum from all other events
        other_event_frames = []
        for i, event in enumerate(events_data):
            if i != protocol_start_idx and event['camera_frame_id'] > 0:
                other_event_frames.append(event['camera_frame_id'])
        
        min_other_event_frame = min(other_event_frames) if other_event_frames else None
        if verbose and min_other_event_frame:
            print(f"  Minimum frame from other events: {min_other_event_frame}")
        
        # Determine the correct frame ID to use
        # Priority: metadata minimum > next event > minimum of all others
        new_frame_id = None
        source = ""
        
        if min_metadata_frame is not None:
            new_frame_id = min_metadata_frame
            source = "frame_metadata minimum"
        elif next_event_frame is not None:
            new_frame_id = next_event_frame
            source = "next event frame"
        elif min_other_event_frame is not None:
            new_frame_id = min_other_event_frame
            source = "minimum of other events"
        else:
            print("❌ Could not determine a valid frame ID for PROTOCOL_START")
            return
        
        # Check if we need to update
        if old_frame_id == new_frame_id:
            print(f"\n✅ PROTOCOL_START already has correct frame ID: {old_frame_id}")
            return
        
        # Update the PROTOCOL_START event
        print(f"\n🔄 Updating PROTOCOL_START:")
        print(f"  Old camera_frame_id: {old_frame_id}")
        print(f"  New camera_frame_id: {new_frame_id} (from {source})")
        print(f"  Difference: {old_frame_id - new_frame_id} frames")
        
        # Create a modified copy of the events data
        modified_events = events_data.copy()
        modified_events[protocol_start_idx]['camera_frame_id'] = new_frame_id
        
        # Write back to the file
        del f['/events']
        f.create_dataset('/events', data=modified_events)
        
        # Add metadata about the fix
        f['/events'].attrs['protocol_start_fixed'] = True
        f['/events'].attrs['protocol_start_fix_timestamp'] = datetime.now().isoformat()
        f['/events'].attrs['protocol_start_old_frame_id'] = int(old_frame_id)
        f['/events'].attrs['protocol_start_new_frame_id'] = int(new_frame_id)
        f['/events'].attrs['protocol_start_fix_source'] = source
        
        print(f"\n✅ Successfully fixed PROTOCOL_START event!")
        
        # Verify the fix by checking event ordering
        print(f"\n📋 Verifying event ordering:")
        updated_events = f['/events'][:]
        
        # Check first few events
        for i in range(min(5, len(updated_events))):
            event = updated_events[i]
            event_type = event[event_type_field]
            event_name = EXPERIMENT_EVENT_TYPE.get(event_type, f"ID_{event_type}")
            frame_id = event['camera_frame_id']
            print(f"  {i}: {event_name:30s} - Frame {frame_id}")
        
        # Check if events are now properly ordered
        frame_ids = [e['camera_frame_id'] for e in updated_events if e['camera_frame_id'] > 0]
        if len(frame_ids) > 1:
            first_jump = frame_ids[1] - frame_ids[0]
            if first_jump < 0:
                print(f"\n⚠️  Warning: Events still not monotonic (first jump: {first_jump})")
            else:
                print(f"\n✅ Events now start with monotonic frame IDs")


def main():
    parser = argparse.ArgumentParser(
        description="Fix PROTOCOL_START event frame ID in analysis.h5 files"
    )
    parser.add_argument(
        'h5_file',
        type=str,
        help='Path to the analysis.h5 file to fix'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help="Don't create a backup file before modifying"
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        fix_protocol_start_event(
            args.h5_file,
            backup=not args.no_backup,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())