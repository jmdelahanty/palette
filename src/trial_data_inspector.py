#!/usr/bin/env python3
"""
Trial Data Inspector

Diagnostic tool to examine the alignment between chase events and available data
in H5 files. Helps identify why trials might appear to have missing data.

Updated to use the new camera_frame_id field in events for better alignment.
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# Event type mappings
EXPERIMENT_EVENT_TYPE = {
    0: "PROTOCOL_START", 1: "PROTOCOL_STOP", 2: "PROTOCOL_PAUSE", 3: "PROTOCOL_RESUME", 4: "PROTOCOL_FINISH",
    5: "PROTOCOL_CLEAR", 6: "PROTOCOL_LOAD", 7: "STEP_ADD", 8: "STEP_REMOVE", 9: "STEP_MOVE_UP",
    10: "STEP_MOVE_DOWN", 11: "STEP_START", 12: "STEP_END", 13: "ITI_START", 14: "ITI_END",
    15: "PARAMS_APPLIED", 16: "MANAGER_REINIT", 17: "MANAGER_REINIT_FAIL", 18: "LOOM_AUTO_REPEAT_TRIGGER",
    19: "LOOM_MANUAL_START", 20: "USER_INTERVENTION", 21: "ERROR_RUNTIME", 22: "LOG_MESSAGE",
    23: "IPC_BOUNDING_BOX_RECEIVED", 24: "CHASER_PRE_PERIOD_START", 25: "CHASER_TRAINING_START",
    26: "CHASER_POST_PERIOD_START", 27: "CHASER_CHASE_SEQUENCE_START", 28: "CHASER_CHASE_SEQUENCE_END",
    29: "CHASER_RANDOM_TARGET_SET"
}

class TrialDataInspector:
    """Inspect trial data coverage and alignment in H5 files."""
    
    def __init__(self, h5_path: str, verbose: bool = True):
        self.h5_path = Path(h5_path)
        self.verbose = verbose
        
    def inspect_trials(self):
        """Inspect all chase trials and their data coverage."""
        with h5py.File(self.h5_path, 'r') as f:
            # Get events
            if '/events' not in f:
                print("❌ No events dataset found")
                return
            
            events = f['/events'][:]
            
            # Find chase sequences
            chase_starts = []
            chase_ends = []
            
            for event in events:
                if event['event_type_id'] == 27:  # CHASE_START
                    chase_starts.append(event)
                elif event['event_type_id'] == 28:  # CHASE_END
                    chase_ends.append(event)
            
            print(f"\n📊 Found {len(chase_starts)} chase sequences")
            print("=" * 70)
            
            # Get all timestamps for reference
            chaser_states = f['/tracking_data/chaser_states'][:]
            frame_metadata = f['/video_metadata/frame_metadata'][:]
            bounding_boxes = f['/tracking_data/bounding_boxes'][:]
            
            # Analyze each trial
            for i, (start_event, end_event) in enumerate(zip(chase_starts, chase_ends)):
                print(f"\n🎯 Trial {i+1}:")
                print("-" * 40)
                
                # Event timestamps
                start_time_ns = start_event['timestamp_ns_session']
                end_time_ns = end_event['timestamp_ns_session']
                duration_s = (end_time_ns - start_time_ns) / 1e9
                
                print(f"  Duration: {duration_s:.2f} seconds")
                print(f"  Start time: {start_time_ns / 1e9:.3f}s")
                print(f"  End time: {end_time_ns / 1e9:.3f}s")
                
                # NEW: Use camera_frame_id from events if available
                if 'camera_frame_id' in start_event.dtype.names:
                    start_camera_frame = start_event['camera_frame_id']
                    end_camera_frame = end_event['camera_frame_id']
                    
                    print(f"\n  📷 Camera frames from events:")
                    print(f"     Start camera frame: {start_camera_frame}")
                    print(f"     End camera frame: {end_camera_frame}")
                    
                    # Check bounding boxes for these camera frames directly
                    if start_camera_frame > 0 and end_camera_frame > 0:
                        # Find bboxes within the camera frame range
                        bbox_mask = (bounding_boxes['payload_frame_id'] >= start_camera_frame) & \
                                   (bounding_boxes['payload_frame_id'] <= end_camera_frame)
                        trial_bboxes_direct = bounding_boxes[bbox_mask]
                        
                        if len(trial_bboxes_direct) > 0:
                            print(f"     Bounding boxes in camera frame range: {len(trial_bboxes_direct)}")
                            unique_bbox_frames = np.unique(trial_bboxes_direct['payload_frame_id'])
                            expected_frames = end_camera_frame - start_camera_frame + 1
                            coverage = len(unique_bbox_frames) / expected_frames * 100
                            print(f"     Detection coverage: {coverage:.1f}% ({len(unique_bbox_frames)}/{expected_frames} frames)")
                        else:
                            print(f"     ⚠️  No bounding boxes found for camera frames {start_camera_frame}-{end_camera_frame}")
                
                # Original analysis using chaser states and timestamps
                time_mask = (chaser_states['timestamp_ns_session'] >= start_time_ns) & \
                           (chaser_states['timestamp_ns_session'] <= end_time_ns)
                
                trial_chaser_states = chaser_states[time_mask]
                
                if len(trial_chaser_states) > 0:
                    stim_frames = trial_chaser_states['stimulus_frame_num']
                    min_stim = stim_frames.min()
                    max_stim = stim_frames.max()
                    
                    print(f"\n  📍 Stimulus frames: {min_stim} to {max_stim}")
                    print(f"     Chaser states in trial: {len(trial_chaser_states)}")
                    
                    # Check frame metadata using stimulus frames
                    meta_mask = (frame_metadata['stimulus_frame_num'] >= min_stim) & \
                               (frame_metadata['stimulus_frame_num'] <= max_stim)
                    trial_metadata = frame_metadata[meta_mask]
                    
                    if len(trial_metadata) > 0:
                        print(f"     Frame metadata records: {len(trial_metadata)}")
                        
                        # Get camera frames from metadata
                        camera_frames = trial_metadata['triggering_camera_frame_id']
                        unique_camera_frames = np.unique(camera_frames)
                        print(f"     Unique camera frames (from metadata): {len(unique_camera_frames)}")
                        print(f"     Camera frame range (from metadata): {camera_frames.min()} to {camera_frames.max()}")
                        
                        # Compare with event camera frames if available
                        if 'camera_frame_id' in start_event.dtype.names and start_camera_frame > 0:
                            if camera_frames.min() != start_camera_frame or camera_frames.max() != end_camera_frame:
                                print(f"     ⚠️  Camera frame mismatch:")
                                print(f"        Events: {start_camera_frame} to {end_camera_frame}")
                                print(f"        Metadata: {camera_frames.min()} to {camera_frames.max()}")
                        
                        # Check bounding boxes for these camera frames
                        bbox_mask = np.isin(bounding_boxes['payload_frame_id'], unique_camera_frames)
                        trial_bboxes = bounding_boxes[bbox_mask]
                        
                        if len(trial_bboxes) > 0:
                            print(f"     Bounding boxes (via metadata): {len(trial_bboxes)}")
                            unique_bbox_frames = np.unique(trial_bboxes['payload_frame_id'])
                            coverage = len(unique_bbox_frames) / len(unique_camera_frames) * 100
                            print(f"     Detection coverage (via metadata): {coverage:.1f}%")
                        else:
                            print(f"     ⚠️  No bounding boxes found via metadata")
                    else:
                        print(f"     ⚠️  No frame metadata found for stimulus frames {min_stim}-{max_stim}")
                else:
                    print(f"  ⚠️  No chaser states found in time window")
                
                # Check what's happening with timestamps
                if 'trial_metadata' in locals() and len(trial_metadata) > 0:
                    meta_times = trial_metadata['timestamp_ns']
                    print(f"\n  ⏱️  Metadata timestamps:")
                    print(f"     First: {meta_times.min() / 1e9:.3f}s")
                    print(f"     Last: {meta_times.max() / 1e9:.3f}s")
                    print(f"     Count: {len(meta_times)}")
                
                if self.verbose and i < 3:  # Show detailed info for first 3 trials
                    print(f"\n  📝 Details:")
                    if 'details_json' in start_event.dtype.names:
                        try:
                            start_details = json.loads(start_event['details_json'].decode('utf-8', errors='ignore'))
                            if start_details:
                                print(f"     Start event details: {start_details}")
                        except:
                            pass
                
                # Summary for this trial
                has_chaser = len(trial_chaser_states) > 0
                has_metadata = len(trial_metadata) > 0 if 'trial_metadata' in locals() else False
                has_bboxes = (len(trial_bboxes_direct) > 0 if 'trial_bboxes_direct' in locals() else 
                             len(trial_bboxes) > 0 if 'trial_bboxes' in locals() else False)
                
                status = "✅ Complete" if all([has_chaser, has_metadata, has_bboxes]) else "⚠️  Incomplete"
                print(f"\n  Status: {status}")
                
                # Clear local variables for next iteration
                if 'trial_metadata' in locals():
                    del trial_metadata
                if 'trial_bboxes' in locals():
                    del trial_bboxes
                if 'trial_bboxes_direct' in locals():
                    del trial_bboxes_direct
                
                if i >= 4 and not self.verbose:  # Limit output unless verbose
                    remaining = len(chase_starts) - i - 1
                    if remaining > 0:
                        print(f"\n... and {remaining} more trials")
                    break
    
    def analyze_time_alignment(self):
        """Analyze the time alignment between different data sources."""
        print("\n" + "=" * 70)
        print("⏰ TIME ALIGNMENT ANALYSIS")
        print("=" * 70)
        
        with h5py.File(self.h5_path, 'r') as f:
            # Get all timestamp ranges
            ranges = {}
            
            # Events
            if '/events' in f:
                events = f['/events'][:]
                ranges['events'] = {
                    'min': events['timestamp_ns_session'].min() / 1e9,
                    'max': events['timestamp_ns_session'].max() / 1e9,
                    'span': (events['timestamp_ns_session'].max() - events['timestamp_ns_session'].min()) / 1e9
                }
                
                # NEW: Check camera_frame_id range in events
                if 'camera_frame_id' in events.dtype.names:
                    valid_frames = events['camera_frame_id'][events['camera_frame_id'] > 0]
                    if len(valid_frames) > 0:
                        ranges['events']['camera_frames'] = {
                            'min': valid_frames.min(),
                            'max': valid_frames.max(),
                            'count': len(np.unique(valid_frames))
                        }
            
            # Chaser states
            if '/tracking_data/chaser_states' in f:
                chaser = f['/tracking_data/chaser_states'][:]
                ranges['chaser_states'] = {
                    'min': chaser['timestamp_ns_session'].min() / 1e9,
                    'max': chaser['timestamp_ns_session'].max() / 1e9,
                    'span': (chaser['timestamp_ns_session'].max() - chaser['timestamp_ns_session'].min()) / 1e9
                }
            
            # Frame metadata
            if '/video_metadata/frame_metadata' in f:
                metadata = f['/video_metadata/frame_metadata'][:]
                ranges['frame_metadata'] = {
                    'min': metadata['timestamp_ns'].min() / 1e9,
                    'max': metadata['timestamp_ns'].max() / 1e9,
                    'span': (metadata['timestamp_ns'].max() - metadata['timestamp_ns'].min()) / 1e9,
                    'camera_frames': {
                        'min': metadata['triggering_camera_frame_id'].min(),
                        'max': metadata['triggering_camera_frame_id'].max(),
                        'count': len(np.unique(metadata['triggering_camera_frame_id']))
                    }
                }
            
            # Bounding boxes
            if '/tracking_data/bounding_boxes' in f:
                bboxes = f['/tracking_data/bounding_boxes'][:]
                ranges['bounding_boxes'] = {
                    'min': bboxes['payload_timestamp_ns_epoch'].min() / 1e9,
                    'max': bboxes['payload_timestamp_ns_epoch'].max() / 1e9,
                    'span': (bboxes['payload_timestamp_ns_epoch'].max() - bboxes['payload_timestamp_ns_epoch'].min()) / 1e9,
                    'camera_frames': {
                        'min': bboxes['payload_frame_id'].min(),
                        'max': bboxes['payload_frame_id'].max(),
                        'count': len(np.unique(bboxes['payload_frame_id']))
                    }
                }
            
            # Find the reference time (earliest timestamp)
            ref_time = min(r['min'] for r in ranges.values())
            
            print("\n📊 Timestamp Ranges (seconds from start):")
            for name, r in ranges.items():
                offset_start = r['min'] - ref_time
                offset_end = r['max'] - ref_time
                print(f"\n  {name}:")
                print(f"    Start: {offset_start:.3f}s")
                print(f"    End:   {offset_end:.3f}s")
                print(f"    Span:  {r['span']:.3f}s")
                
                # NEW: Show camera frame info if available
                if 'camera_frames' in r:
                    cf = r['camera_frames']
                    print(f"    Camera frames: {cf['min']} to {cf['max']} ({cf['count']} unique)")
            
            # Check for epoch vs session timestamps
            if 'bounding_boxes' in ranges:
                print("\n⚠️  Note: Bounding boxes use epoch timestamps (absolute time)")
                print("    Other datasets use session timestamps (relative to session start)")
                
                # Calculate session start time
                session_start_epoch = ranges['bounding_boxes']['min'] - ranges.get('chaser_states', {}).get('min', 0)
                print(f"    Estimated session start (epoch): {session_start_epoch:.3f}s")
                print(f"    Session date/time: {datetime.fromtimestamp(session_start_epoch)}")
    
    def check_frame_id_alignment(self):
        """Check alignment between frame IDs across datasets."""
        print("\n" + "=" * 70)
        print("🔢 FRAME ID ALIGNMENT")
        print("=" * 70)
        
        with h5py.File(self.h5_path, 'r') as f:
            # Get frame ID ranges
            metadata = f['/video_metadata/frame_metadata'][:]
            bboxes = f['/tracking_data/bounding_boxes'][:]
            events = f['/events'][:]
            
            meta_camera_frames = np.unique(metadata['triggering_camera_frame_id'])
            bbox_camera_frames = np.unique(bboxes['payload_frame_id'])
            
            # NEW: Get camera frames from events
            event_camera_frames = None
            if 'camera_frame_id' in events.dtype.names:
                valid_event_frames = events['camera_frame_id'][events['camera_frame_id'] > 0]
                if len(valid_event_frames) > 0:
                    event_camera_frames = np.unique(valid_event_frames)
                    print(f"\n  Event camera frames: {len(event_camera_frames)}")
                    print(f"    Range: {event_camera_frames.min()} to {event_camera_frames.max()}")
            
            # Find intersection and differences
            common_frames = np.intersect1d(meta_camera_frames, bbox_camera_frames)
            meta_only = np.setdiff1d(meta_camera_frames, bbox_camera_frames)
            bbox_only = np.setdiff1d(bbox_camera_frames, meta_camera_frames)
            
            print(f"\n  Frame metadata camera frames: {len(meta_camera_frames)}")
            print(f"    Range: {meta_camera_frames.min()} to {meta_camera_frames.max()}")
            print(f"  Bounding box camera frames: {len(bbox_camera_frames)}")
            print(f"    Range: {bbox_camera_frames.min()} to {bbox_camera_frames.max()}")
            print(f"\n  Common frames: {len(common_frames)}")
            print(f"  Frames in metadata only: {len(meta_only)}")
            print(f"  Frames in bboxes only: {len(bbox_only)}")
            
            # Check alignment with event frames
            if event_camera_frames is not None:
                print(f"\n  Alignment with events:")
                events_in_meta = np.intersect1d(event_camera_frames, meta_camera_frames)
                events_in_bbox = np.intersect1d(event_camera_frames, bbox_camera_frames)
                print(f"    Event frames in metadata: {len(events_in_meta)}/{len(event_camera_frames)}")
                print(f"    Event frames in bboxes: {len(events_in_bbox)}/{len(event_camera_frames)}")
                
                # Check specific chase event frames
                chase_start_frames = events['camera_frame_id'][events['event_type_id'] == 27]
                chase_end_frames = events['camera_frame_id'][events['event_type_id'] == 28]
                
                chase_frames_valid = chase_start_frames[chase_start_frames > 0]
                if len(chase_frames_valid) > 0:
                    print(f"\n  Chase event camera frames:")
                    for j, (start_frame, end_frame) in enumerate(zip(chase_start_frames[chase_start_frames > 0], 
                                                                     chase_end_frames[chase_end_frames > 0])):
                        if j >= 3 and not self.verbose:
                            print(f"    ... and {len(chase_frames_valid) - j} more trials")
                            break
                        
                        # Check if these frames have bounding boxes
                        bbox_in_range = bboxes[(bboxes['payload_frame_id'] >= start_frame) & 
                                              (bboxes['payload_frame_id'] <= end_frame)]
                        coverage = len(np.unique(bbox_in_range['payload_frame_id'])) / (end_frame - start_frame + 1) * 100
                        print(f"    Trial {j+1}: frames {start_frame}-{end_frame}, bbox coverage: {coverage:.1f}%")
            
            if len(meta_only) > 0 and self.verbose:
                print(f"\n  📍 Sample frames in metadata but not bboxes:")
                for frame in meta_only[:5]:
                    print(f"     Frame {frame}")
            
            if len(bbox_only) > 0 and self.verbose:
                print(f"\n  📍 Sample frames in bboxes but not metadata:")
                for frame in bbox_only[:5]:
                    print(f"     Frame {frame}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Inspect trial data coverage in H5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This inspector helps diagnose trial coverage issues by:
- Examining each chase trial individually
- Using camera_frame_id from events for direct alignment
- Checking time alignment between datasets
- Verifying frame ID consistency

Examples:
  %(prog)s out_analysis.h5
  %(prog)s out_analysis.h5 --verbose
  %(prog)s out_analysis.h5 --time-only
        """
    )
    
    parser.add_argument('h5_path', help='Path to H5 analysis file')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed information for all trials')
    parser.add_argument('--time-only', action='store_true',
                       help='Only show time alignment analysis')
    parser.add_argument('--frames-only', action='store_true',
                       help='Only show frame alignment analysis')
    
    args = parser.parse_args()
    
    inspector = TrialDataInspector(
        h5_path=args.h5_path,
        verbose=args.verbose
    )
    
    if args.time_only:
        inspector.analyze_time_alignment()
    elif args.frames_only:
        inspector.check_frame_id_alignment()
    else:
        inspector.inspect_trials()
        inspector.analyze_time_alignment()
        inspector.check_frame_id_alignment()
    
    return 0


if __name__ == '__main__':
    exit(main())