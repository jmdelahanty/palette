#!/usr/bin/env python3
"""
H5 Analysis File Validator (v2.0)

Comprehensive validation of analysis.h5 files to ensure they contain all necessary
data for trial-by-trial analysis of fish responses to chase events.

Updated to validate new H5 structure including:
- camera_frame_id field in events
- stimulus_coordinates group with texture dimensions
- stimulus_output_width/height in root attributes
- Enhanced coordinate system validation

This validator checks:
1. Dataset structure and completeness
2. Frame alignment between different data sources
3. Event sequence integrity with camera frame sync
4. Trial segmentation readiness
5. Data quality metrics
6. Coordinate system consistency
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
import json

# Event type mappings (from your stimulus program)
EXPERIMENT_EVENT_TYPE = {
    0: "PROTOCOL_START", 1: "PROTOCOL_STOP", 2: "PROTOCOL_PAUSE", 3: "PROTOCOL_RESUME", 
    4: "PROTOCOL_FINISH", 5: "PROTOCOL_CLEAR", 6: "PROTOCOL_LOAD", 
    11: "STEP_START", 12: "STEP_END", 13: "ITI_START", 14: "ITI_END",
    24: "CHASER_PRE_PERIOD_START", 25: "CHASER_TRAINING_START",
    26: "CHASER_POST_PERIOD_START", 27: "CHASER_CHASE_SEQUENCE_START", 
    28: "CHASER_CHASE_SEQUENCE_END", 29: "CHASER_RANDOM_TARGET_SET"
}

class H5AnalysisValidator:
    """Validates H5 analysis files for completeness and integrity."""
    
    def __init__(self, h5_path: str, verbose: bool = True):
        self.h5_path = Path(h5_path)
        self.verbose = verbose
        self.validation_results = {
            'structure': {},
            'alignment': {},
            'events': {},
            'trials': {},
            'quality': {},
            'coordinates': {},
            'errors': [],
            'warnings': [],
            'info': []
        }
        
    def log(self, message: str, level: str = 'info'):
        """Log a message with appropriate formatting."""
        if not self.verbose and level == 'info':
            return
            
        prefix = {
            'info': '  ',
            'success': '  ✅',
            'warning': '  ⚠️ ',
            'error': '  ❌',
            'header': '\n📋'
        }.get(level, '  ')
        
        print(f"{prefix} {message}")
    
    def validate_all(self) -> Dict:
        """Run all validation checks."""
        self.log(f"Validating H5 Analysis File: {self.h5_path}", 'header')
        self.log(f"Validator Version: 2.0 (supports new H5 structure)", 'info')
        
        if not self.h5_path.exists():
            self.validation_results['errors'].append(f"File not found: {self.h5_path}")
            return self.validation_results
        
        with h5py.File(self.h5_path, 'r') as f:
            # Run all validation checks
            self.validate_structure(f)
            self.validate_coordinates(f)  # NEW: Validate coordinate systems
            self.validate_frame_alignment(f)
            self.validate_events_enhanced(f)  # ENHANCED: Check camera_frame_id
            self.validate_trial_segmentation(f)
            self.validate_data_quality(f)
            
        # Generate summary
        self.generate_summary()
        return self.validation_results
    
    def validate_structure(self, f: h5py.File):
        """Validate the basic structure of the H5 file."""
        self.log("Dataset Structure Validation", 'header')
        
        required_groups = {
            '/tracking_data': ['bounding_boxes', 'chaser_states'],
            '/video_metadata': ['frame_metadata'],
            '/calibration_snapshot': [],
            '/protocol_snapshot': []
        }
        
        optional_groups = {
            '/events': [],
            '/analysis': ['gap_info', 'interpolation_mask'],
            '/stimulus_coordinates': [],  # NEW: Check for stimulus_coordinates
            '/subject_metadata': []
        }
        
        # Check required groups
        for group_path, datasets in required_groups.items():
            if group_path in f:
                self.log(f"Found required group: {group_path}", 'success')
                self.validation_results['structure'][group_path] = True
                
                # Check datasets within group
                for dataset in datasets:
                    dataset_path = f"{group_path}/{dataset}"
                    if dataset_path in f:
                        shape = f[dataset_path].shape
                        dtype = f[dataset_path].dtype
                        self.log(f"  - {dataset}: shape={shape}, dtype={dtype}", 'info')
                        self.validation_results['structure'][dataset_path] = True
                    else:
                        self.log(f"  - Missing dataset: {dataset}", 'error')
                        self.validation_results['structure'][dataset_path] = False
                        self.validation_results['errors'].append(f"Missing dataset: {dataset_path}")
            else:
                self.log(f"Missing required group: {group_path}", 'error')
                self.validation_results['structure'][group_path] = False
                self.validation_results['errors'].append(f"Missing group: {group_path}")
        
        # Check optional groups
        for group_path, datasets in optional_groups.items():
            if group_path in f:
                self.log(f"Found optional group: {group_path}", 'success')
                self.validation_results['structure'][group_path] = True
                
                # Special handling for stimulus_coordinates
                if group_path == '/stimulus_coordinates':
                    self.log("  NEW: stimulus_coordinates group present", 'success')
                    self.validation_results['info'].append("Has stimulus_coordinates (new format)")
            else:
                if group_path == '/stimulus_coordinates':
                    self.log(f"stimulus_coordinates not present (older format)", 'info')
                else:
                    self.log(f"Optional group not present: {group_path}", 'warning')
                    self.validation_results['warnings'].append(f"Optional group missing: {group_path}")
        
        # Check root attributes for new fields
        if 'stimulus_output_width' in f.attrs and 'stimulus_output_height' in f.attrs:
            width = f.attrs['stimulus_output_width']
            height = f.attrs['stimulus_output_height']
            self.log(f"NEW: Stimulus output dimensions in root: {width}×{height}", 'success')
            self.validation_results['info'].append(f"Has stimulus dimensions: {width}×{height}")
    
    def validate_coordinates(self, f: h5py.File):
        """Validate coordinate system information (NEW)."""
        self.log("Coordinate System Validation", 'header')
        
        coord_info = {}
        
        # Check for stimulus_coordinates group (new format)
        if '/stimulus_coordinates' in f:
            stim_coords = f['/stimulus_coordinates']
            arena_names = list(stim_coords.keys())
            
            if arena_names:
                arena_name = arena_names[0]
                arena_group = stim_coords[arena_name]
                
                # Extract texture dimensions
                if 'texture_width_px' in arena_group and 'texture_height_px' in arena_group:
                    width = int(arena_group['texture_width_px'][()])
                    height = int(arena_group['texture_height_px'][()])
                    coord_info['texture_dimensions'] = [width, height]
                    self.log(f"Texture dimensions: {width}×{height}", 'success')
                
                # Check for texture origin
                if 'texture_origin' in arena_group:
                    origin = arena_group['texture_origin'][()].decode('utf-8')
                    coord_info['texture_origin'] = origin
                    self.log(f"Texture origin: {origin}", 'info')
                
                # Check for active stimulus mode
                if 'active_stimulus_mode' in arena_group:
                    mode = arena_group['active_stimulus_mode'][()].decode('utf-8')
                    coord_info['active_stimulus_mode'] = mode
                    self.log(f"Active stimulus mode: {mode}", 'info')
                
                # Check custom coordinates
                if 'custom_coordinates' in arena_group:
                    self.log("Custom coordinates present", 'info')
                    coord_info['has_custom_coordinates'] = True
        else:
            # Try to get from root attributes (fallback)
            if 'stimulus_output_width' in f.attrs and 'stimulus_output_height' in f.attrs:
                width = int(f.attrs['stimulus_output_width'])
                height = int(f.attrs['stimulus_output_height'])
                coord_info['stimulus_output_dimensions'] = [width, height]
                self.log(f"Stimulus output from root: {width}×{height}", 'info')
        
        # Check analysis group for coordinate transform info
        if '/analysis' in f and 'coordinate_transform' in f['/analysis'].attrs:
            transform_str = f['/analysis'].attrs['coordinate_transform']
            transform_info = json.loads(transform_str)
            coord_info['coordinate_transform'] = transform_info
            
            if 'texture_to_camera_scale' in transform_info:
                scale = transform_info['texture_to_camera_scale']
                self.log(f"Texture→Camera scale factor: {scale:.3f}", 'success')
            
            if 'coordinate_note' in transform_info:
                self.log(f"Coordinate note: {transform_info['coordinate_note']}", 'info')
        
        # Validate coordinate consistency
        if 'texture_dimensions' in coord_info:
            tex_w, tex_h = coord_info['texture_dimensions']
            
            # Check if chaser positions are within texture bounds
            if '/tracking_data/chaser_states' in f:
                chaser = f['/tracking_data/chaser_states']
                if 'chaser_pos_x' in chaser.dtype.names and 'chaser_pos_y' in chaser.dtype.names:
                    chaser_x = chaser['chaser_pos_x'][:]
                    chaser_y = chaser['chaser_pos_y'][:]
                    
                    # Filter out -1 values (not visible)
                    valid_x = chaser_x[chaser_x >= 0]
                    valid_y = chaser_y[chaser_y >= 0]
                    
                    if len(valid_x) > 0:
                        max_x = np.max(valid_x)
                        max_y = np.max(valid_y)
                        
                        if max_x <= tex_w and max_y <= tex_h:
                            self.log(f"Chaser positions within texture bounds ({tex_w}×{tex_h})", 'success')
                        else:
                            self.log(f"Chaser positions exceed texture bounds: max ({max_x:.0f}, {max_y:.0f})", 'warning')
                            self.validation_results['warnings'].append("Chaser positions may exceed texture bounds")
        
        self.validation_results['coordinates'] = coord_info
    
    def validate_frame_alignment(self, f: h5py.File):
        """Validate frame alignment between different data sources."""
        self.log("Frame Alignment Validation", 'header')
        
        alignment = {}
        
        # Get frame ranges from different sources
        if '/video_metadata/frame_metadata' in f:
            metadata = f['/video_metadata/frame_metadata']
            camera_frames = metadata['triggering_camera_frame_id'][:]
            stimulus_frames = metadata['stimulus_frame_num'][:]
            
            alignment['camera_frame_range'] = (int(camera_frames.min()), int(camera_frames.max()))
            alignment['stimulus_frame_range'] = (int(stimulus_frames.min()), int(stimulus_frames.max()))
            alignment['metadata_records'] = len(metadata)
            
            self.log(f"Camera frames: {alignment['camera_frame_range'][0]} - {alignment['camera_frame_range'][1]}", 'info')
            self.log(f"Stimulus frames: {alignment['stimulus_frame_range'][0]} - {alignment['stimulus_frame_range'][1]}", 'info')
            
            # Check for gaps in camera frames
            unique_camera = np.unique(camera_frames)
            expected_frames = alignment['camera_frame_range'][1] - alignment['camera_frame_range'][0] + 1
            missing_camera_frames = expected_frames - len(unique_camera)
            
            if missing_camera_frames > 0:
                self.log(f"Missing camera frames: {missing_camera_frames}", 'warning')
                alignment['missing_camera_frames'] = missing_camera_frames
                self.validation_results['warnings'].append(f"Missing {missing_camera_frames} camera frames")
            else:
                self.log("All camera frames present", 'success')
                alignment['missing_camera_frames'] = 0
            
            # Check for frame reuse (120Hz stimulus vs 60Hz camera)
            camera_frame_counts = Counter(camera_frames)
            max_reuse = max(camera_frame_counts.values())
            frames_reused = sum(1 for count in camera_frame_counts.values() if count > 1)
            
            alignment['max_frame_reuse'] = max_reuse
            alignment['frames_reused'] = frames_reused
            
            if max_reuse > 2:
                self.log(f"High frame reuse detected: max {max_reuse} times", 'warning')
                self.validation_results['warnings'].append(f"Camera frames reused up to {max_reuse} times")
            else:
                self.log(f"Frame reuse within expected range (max: {max_reuse})", 'success')
        
        # Check bounding boxes alignment
        if '/tracking_data/bounding_boxes' in f:
            bboxes = f['/tracking_data/bounding_boxes']
            bbox_frames = bboxes['payload_frame_id'][:]
            unique_bbox_frames = np.unique(bbox_frames)
            
            alignment['bbox_frame_range'] = (int(bbox_frames.min()), int(bbox_frames.max()))
            alignment['unique_bbox_frames'] = len(unique_bbox_frames)
            alignment['total_bboxes'] = len(bboxes)
            
            self.log(f"Bounding box frames: {alignment['bbox_frame_range'][0]} - {alignment['bbox_frame_range'][1]}", 'info')
            self.log(f"Unique frames with detections: {alignment['unique_bbox_frames']}", 'info')
        
        # Check chaser states alignment
        if '/tracking_data/chaser_states' in f:
            chaser = f['/tracking_data/chaser_states']
            chaser_frames = chaser['stimulus_frame_num'][:]
            
            alignment['chaser_frame_range'] = (int(chaser_frames.min()), int(chaser_frames.max()))
            alignment['chaser_records'] = len(chaser)
            
            # Check for gaps in chaser states
            unique_chaser = np.unique(chaser_frames)
            expected_chaser = alignment['chaser_frame_range'][1] - alignment['chaser_frame_range'][0] + 1
            missing_chaser = expected_chaser - len(unique_chaser)
            
            if missing_chaser > 0:
                self.log(f"Missing chaser states: {missing_chaser} frames", 'warning')
                alignment['missing_chaser_frames'] = missing_chaser
            else:
                self.log("All chaser states present", 'success')
                alignment['missing_chaser_frames'] = 0
        
        self.validation_results['alignment'] = alignment
    
    def validate_events_enhanced(self, f: h5py.File):
        """Validate event data with enhanced camera_frame_id checking."""
        self.log("Event Data Validation (Enhanced)", 'header')
        
        if '/events' not in f:
            self.log("No events dataset found", 'warning')
            self.validation_results['warnings'].append("Events dataset missing - cannot perform trial segmentation")
            return
        
        events_ds = f['/events']
        
        # First, check the structure of the events dataset
        self.log(f"Events dataset shape: {events_ds.shape}, dtype: {events_ds.dtype}", 'info')
        
        # Check what fields are available
        if events_ds.dtype.names:
            self.log(f"Event fields: {events_ds.dtype.names}", 'info')
        
        events = events_ds[:]
        event_analysis = {
            'total_events': len(events),
            'event_types': {},
            'chase_sequences': [],
            'training_periods': {},
            'has_camera_frame_sync': False  # NEW
        }
        
        # Check for camera_frame_id field (NEW)
        if 'camera_frame_id' in events.dtype.names:
            self.log("NEW: camera_frame_id field present in events", 'success')
            event_analysis['has_camera_frame_sync'] = True
            self.validation_results['info'].append("Events have camera_frame_id (new format)")
            
            # Check camera frame coverage in events
            camera_ids = events['camera_frame_id'][:]
            valid_camera_ids = camera_ids[camera_ids > 0]  # Filter out invalid IDs
            
            if len(valid_camera_ids) > 0:
                event_analysis['camera_frame_range'] = (int(valid_camera_ids.min()), int(valid_camera_ids.max()))
                event_analysis['events_with_camera_sync'] = len(valid_camera_ids)
                self.log(f"  Camera frame range in events: {event_analysis['camera_frame_range'][0]} - {event_analysis['camera_frame_range'][1]}", 'info')
                self.log(f"  Events with camera sync: {event_analysis['events_with_camera_sync']}/{len(events)}", 'info')
            else:
                self.log("  Warning: camera_frame_id field exists but no valid IDs", 'warning')
        else:
            self.log("camera_frame_id field not present (older format)", 'info')
        
        # Determine the correct field name for event type
        event_type_field = None
        for field_name in ['event_type', 'event_type_id', 'type', 'event_id']:
            if field_name in events.dtype.names:
                event_type_field = field_name
                break
        
        if event_type_field is None:
            self.log("Could not find event type field in events dataset", 'warning')
            self.validation_results['warnings'].append("Event type field not found - cannot analyze event sequences")
            self.validation_results['events'] = event_analysis
            return
        
        # Analyze event types
        event_type_counts = Counter(events[event_type_field])
        for event_type, count in event_type_counts.items():
            event_name = EXPERIMENT_EVENT_TYPE.get(event_type, f"UNKNOWN_{event_type}")
            event_analysis['event_types'][event_name] = count
            self.log(f"{event_name}: {count} events", 'info')
        
        # Find chase sequences
        chase_starts = []
        chase_ends = []
        
        for i, event in enumerate(events):
            event_type = event[event_type_field]
            if event_type == 27:  # CHASER_CHASE_SEQUENCE_START
                chase_starts.append(i)
            elif event_type == 28:  # CHASER_CHASE_SEQUENCE_END
                chase_ends.append(i)
        
        # Match chase sequences
        if len(chase_starts) == len(chase_ends):
            self.log(f"Found {len(chase_starts)} complete chase sequences", 'success')
            
            for start_idx, end_idx in zip(chase_starts, chase_ends):
                start_event = events[start_idx]
                end_event = events[end_idx]
                
                # Find timestamp and frame fields
                timestamp_field = None
                for field_name in ['timestamp_ns_session', 'timestamp_ns', 'timestamp']:
                    if field_name in events.dtype.names:
                        timestamp_field = field_name
                        break
                
                frame_field = None
                for field_name in ['stimulus_frame_num', 'frame_num', 'frame']:
                    if field_name in events.dtype.names:
                        frame_field = field_name
                        break
                
                chase_info = {}
                
                if timestamp_field:
                    duration_s = (end_event[timestamp_field] - start_event[timestamp_field]) / 1e9
                    chase_info['start_timestamp_ns'] = int(start_event[timestamp_field])
                    chase_info['end_timestamp_ns'] = int(end_event[timestamp_field])
                    chase_info['duration_s'] = float(duration_s)
                
                if frame_field:
                    chase_info['start_stimulus_frame'] = int(start_event[frame_field])
                    chase_info['end_stimulus_frame'] = int(end_event[frame_field])
                
                # NEW: Add camera frame info if available
                if 'camera_frame_id' in events.dtype.names:
                    start_cam = start_event['camera_frame_id']
                    end_cam = end_event['camera_frame_id']
                    if start_cam > 0 and end_cam > 0:
                        chase_info['start_camera_frame'] = int(start_cam)
                        chase_info['end_camera_frame'] = int(end_cam)
                        chase_info['camera_frame_duration'] = int(end_cam - start_cam)
                
                event_analysis['chase_sequences'].append(chase_info)
                
                if self.verbose and len(chase_starts) <= 5:
                    duration_str = f"{chase_info.get('duration_s', 0):.2f}s"
                    if 'camera_frame_duration' in chase_info:
                        duration_str += f" ({chase_info['camera_frame_duration']} camera frames)"
                    self.log(f"  Chase {len(event_analysis['chase_sequences'])}: {duration_str}", 'info')
        else:
            self.log(f"Mismatched chase sequences: {len(chase_starts)} starts, {len(chase_ends)} ends", 'error')
            self.validation_results['errors'].append("Incomplete chase sequences detected")
        
        # Identify training periods
        for event in events:
            event_type = event[event_type_field]
            
            # Check for timestamp field
            timestamp_field = None
            for field_name in ['timestamp_ns_session', 'timestamp_ns', 'timestamp']:
                if field_name in events.dtype.names:
                    timestamp_field = field_name
                    break
            
            if timestamp_field:
                timestamp = event[timestamp_field]
                
                if event_type == 24:  # CHASER_PRE_PERIOD_START
                    event_analysis['training_periods']['pre_start'] = int(timestamp)
                elif event_type == 25:  # CHASER_TRAINING_START
                    event_analysis['training_periods']['training_start'] = int(timestamp)
                elif event_type == 26:  # CHASER_POST_PERIOD_START
                    event_analysis['training_periods']['post_start'] = int(timestamp)
        
        if 'training_start' in event_analysis['training_periods']:
            self.log("Training period markers found", 'success')
        else:
            self.log("No training period markers found", 'warning')
        
        self.validation_results['events'] = event_analysis
    
    def validate_trial_segmentation(self, f: h5py.File):
        """Validate readiness for trial-by-trial analysis."""
        self.log("Trial Segmentation Readiness", 'header')
        
        trial_info = {
            'ready_for_trials': False,
            'data_coverage': {},
            'issues': []
        }
        
        # Check if we have chase sequences from events
        if 'chase_sequences' in self.validation_results['events']:
            chase_sequences = self.validation_results['events']['chase_sequences']
            
            if chase_sequences:
                self.log(f"Analyzing data coverage for {len(chase_sequences)} trials", 'info')
                
                # Check if we have camera frame sync (NEW)
                has_camera_sync = self.validation_results['events'].get('has_camera_frame_sync', False)
                if has_camera_sync:
                    self.log("  Using camera_frame_id for precise synchronization", 'success')
                
                # For each chase sequence, check data availability
                for i, chase in enumerate(chase_sequences[:5]):  # Check first 5
                    trial_coverage = {
                        'trial_num': i + 1,
                        'has_chaser': False,
                        'has_metadata': False,
                        'has_bboxes': False,
                        'has_camera_sync': False,  # NEW
                        'duration_s': chase.get('duration_s', 0)
                    }
                    
                    # NEW: If we have camera frame IDs, use them for precise matching
                    if 'start_camera_frame' in chase and 'end_camera_frame' in chase:
                        start_cam = chase['start_camera_frame']
                        end_cam = chase['end_camera_frame']
                        trial_coverage['has_camera_sync'] = True
                        
                        # Check bounding boxes by camera frame
                        if '/tracking_data/bounding_boxes' in f:
                            bboxes = f['/tracking_data/bounding_boxes']
                            bbox_frames = bboxes['payload_frame_id'][:]
                            has_data = np.any((bbox_frames >= start_cam) & (bbox_frames <= end_cam))
                            trial_coverage['has_bboxes'] = has_data
                    
                    # If we have stimulus frame numbers, use them for matching
                    if 'start_stimulus_frame' in chase and 'end_stimulus_frame' in chase:
                        start_frame = chase['start_stimulus_frame']
                        end_frame = chase['end_stimulus_frame']
                        
                        # Check chaser data
                        if '/tracking_data/chaser_states' in f:
                            chaser = f['/tracking_data/chaser_states']
                            chaser_frames = chaser['stimulus_frame_num'][:]
                            has_data = np.any((chaser_frames >= start_frame) & (chaser_frames <= end_frame))
                            trial_coverage['has_chaser'] = has_data
                        
                        # Check metadata
                        if '/video_metadata/frame_metadata' in f:
                            metadata = f['/video_metadata/frame_metadata']
                            stim_frames = metadata['stimulus_frame_num'][:]
                            has_data = np.any((stim_frames >= start_frame) & (stim_frames <= end_frame))
                            trial_coverage['has_metadata'] = has_data
                    
                    # Store coverage info
                    trial_info['data_coverage'][f'trial_{i+1}'] = trial_coverage
                    
                    # Check if trial is complete
                    required_data = [trial_coverage['has_chaser'], trial_coverage['has_metadata']]
                    if trial_coverage['has_camera_sync']:
                        required_data.append(trial_coverage['has_bboxes'])
                    
                    if all(required_data):
                        status = "Complete with camera sync" if trial_coverage['has_camera_sync'] else "Complete"
                        self.log(f"  Trial {i+1}: {status}", 'success')
                    else:
                        missing = [k for k, v in trial_coverage.items() 
                                 if not v and k not in ['trial_num', 'duration_s']]
                        self.log(f"  Trial {i+1}: Missing {', '.join(missing)}", 'warning')
                        trial_info['issues'].append(f"Trial {i+1} missing: {', '.join(missing)}")
                
                # Overall readiness
                if trial_info['issues']:
                    trial_info['ready_for_trials'] = False
                    self.log("Some trials have incomplete data", 'warning')
                else:
                    trial_info['ready_for_trials'] = True
                    self.log("All checked trials have complete data", 'success')
        else:
            self.log("No chase sequences found - cannot validate trial segmentation", 'warning')
            trial_info['issues'].append("No chase sequences found in events")
        
        self.validation_results['trials'] = trial_info
    
    def validate_data_quality(self, f: h5py.File):
        """Validate data quality metrics."""
        self.log("Data Quality Validation", 'header')
        
        quality = {
            'interpolation_ratio': 0,
            'detection_coverage': 0,
            'chaser_continuity': 0,
            'suspicious_values': []
        }
        
        # Check interpolation mask if present
        if '/analysis/interpolation_mask' in f:
            mask = f['/analysis/interpolation_mask'][:]
            interpolated = np.sum(~mask)  # False = interpolated
            total = len(mask)
            quality['interpolation_ratio'] = interpolated / total if total > 0 else 0
            
            if quality['interpolation_ratio'] > 0.1:
                self.log(f"High interpolation ratio: {quality['interpolation_ratio']:.1%}", 'warning')
                self.validation_results['warnings'].append(f"High interpolation: {quality['interpolation_ratio']:.1%}")
            else:
                self.log(f"Interpolation ratio: {quality['interpolation_ratio']:.1%}", 'success')
        
        # Check bounding box coverage
        if '/tracking_data/bounding_boxes' in f and '/video_metadata/frame_metadata' in f:
            bboxes = f['/tracking_data/bounding_boxes']
            metadata = f['/video_metadata/frame_metadata']
            
            unique_bbox_frames = len(np.unique(bboxes['payload_frame_id'][:]))
            unique_camera_frames = len(np.unique(metadata['triggering_camera_frame_id'][:]))
            
            if unique_camera_frames > 0:
                quality['detection_coverage'] = unique_bbox_frames / unique_camera_frames
                self.log(f"Detection coverage: {quality['detection_coverage']:.1%}", 'info')
                
                if quality['detection_coverage'] < 0.8:
                    self.log("Low detection coverage", 'warning')
                    self.validation_results['warnings'].append(f"Low detection coverage: {quality['detection_coverage']:.1%}")
        
        # Check for suspicious bounding box values
        if '/tracking_data/bounding_boxes' in f:
            bboxes = f['/tracking_data/bounding_boxes']
            
            # Check for negative coordinates
            if np.any(bboxes['x_min'][:] < 0) or np.any(bboxes['y_min'][:] < 0):
                quality['suspicious_values'].append("Negative bbox coordinates found")
                self.log("Found negative bounding box coordinates", 'error')
            
            # Check for zero-size boxes
            if np.any(bboxes['width'][:] <= 0) or np.any(bboxes['height'][:] <= 0):
                quality['suspicious_values'].append("Zero-size bboxes found")
                self.log("Found zero-size bounding boxes", 'error')
            
            # Check for unreasonably large boxes
            # Get camera dimensions from coordinate info if available
            max_dim = 4512  # Default
            if 'coordinates' in self.validation_results:
                coord_info = self.validation_results['coordinates']
                if 'coordinate_transform' in coord_info:
                    transform = coord_info['coordinate_transform']
                    if 'camera_dimensions' in transform:
                        max_dim = max(transform['camera_dimensions'])
            
            if np.any(bboxes['x_min'][:] + bboxes['width'][:] > max_dim) or \
               np.any(bboxes['y_min'][:] + bboxes['height'][:] > max_dim):
                quality['suspicious_values'].append("Bboxes exceed frame boundaries")
                self.log(f"Found bounding boxes exceeding frame boundaries ({max_dim}×{max_dim})", 'warning')
        
        # Check chaser state continuity
        if '/tracking_data/chaser_states' in f:
            chaser = f['/tracking_data/chaser_states']
            frames = chaser['stimulus_frame_num'][:]
            
            if len(frames) > 1:
                frame_diffs = np.diff(np.sort(frames))
                # Expect mostly 1-frame differences for continuous tracking
                continuous_frames = np.sum(frame_diffs == 1)
                quality['chaser_continuity'] = continuous_frames / (len(frames) - 1) if len(frames) > 1 else 0
                
                if quality['chaser_continuity'] < 0.9:
                    self.log(f"Chaser tracking gaps detected: {quality['chaser_continuity']:.1%} continuity", 'warning')
                else:
                    self.log(f"Chaser tracking continuity: {quality['chaser_continuity']:.1%}", 'success')
        
        self.validation_results['quality'] = quality
    
    def generate_summary(self):
        """Generate a summary of validation results."""
        self.log("VALIDATION SUMMARY", 'header')
        
        # Count issues
        n_errors = len(self.validation_results['errors'])
        n_warnings = len(self.validation_results['warnings'])
        n_info = len(self.validation_results['info'])
        
        # Overall status
        if n_errors > 0:
            status = "❌ FAILED"
            self.log(f"Status: {status} - {n_errors} errors found", 'error')
        elif n_warnings > 5:
            status = "⚠️  NEEDS ATTENTION"
            self.log(f"Status: {status} - {n_warnings} warnings", 'warning')
        elif n_warnings > 0:
            status = "✓ PASSED WITH WARNINGS"
            self.log(f"Status: {status} - {n_warnings} minor issues", 'warning')
        else:
            status = "✅ PASSED"
            self.log(f"Status: {status} - No issues found", 'success')
        
        self.validation_results['summary'] = {
            'status': status,
            'errors': n_errors,
            'warnings': n_warnings,
            'info_notes': n_info,
            'timestamp': datetime.now().isoformat()
        }
        
        # Report on new features detected
        if n_info > 0:
            self.log("\n🆕 New Format Features Detected:", 'info')
            for info in self.validation_results['info']:
                self.log(f"  • {info}", 'info')
        
        # Key metrics
        self.log("\nKey Metrics:", 'info')
        
        if 'alignment' in self.validation_results:
            align = self.validation_results['alignment']
            if 'metadata_records' in align:
                self.log(f"  • Metadata records: {align['metadata_records']}", 'info')
            if 'unique_bbox_frames' in align:
                self.log(f"  • Frames with detections: {align['unique_bbox_frames']}", 'info')
            if 'missing_camera_frames' in align:
                self.log(f"  • Missing camera frames: {align['missing_camera_frames']}", 'info')
        
        if 'events' in self.validation_results:
            events = self.validation_results['events']
            if 'chase_sequences' in events:
                self.log(f"  • Chase sequences: {len(events['chase_sequences'])}", 'info')
            if events.get('has_camera_frame_sync'):
                self.log(f"  • Camera frame sync: AVAILABLE ✅", 'success')
        
        if 'coordinates' in self.validation_results:
            coords = self.validation_results['coordinates']
            if 'texture_dimensions' in coords:
                w, h = coords['texture_dimensions']
                self.log(f"  • Texture dimensions: {w}×{h}", 'info')
            if 'coordinate_transform' in coords:
                if 'texture_to_camera_scale' in coords['coordinate_transform']:
                    scale = coords['coordinate_transform']['texture_to_camera_scale']
                    self.log(f"  • Texture→Camera scale: {scale:.3f}", 'info')
        
        if 'quality' in self.validation_results:
            quality = self.validation_results['quality']
            if quality['interpolation_ratio'] > 0:
                self.log(f"  • Interpolation ratio: {quality['interpolation_ratio']:.1%}", 'info')
            if quality['detection_coverage'] > 0:
                self.log(f"  • Detection coverage: {quality['detection_coverage']:.1%}", 'info')
        
        # Trial readiness
        if 'trials' in self.validation_results:
            trials = self.validation_results['trials']
            if trials['ready_for_trials']:
                self.log("\n✅ File is ready for trial-by-trial analysis", 'success')
            else:
                self.log("\n⚠️  File needs attention before trial analysis", 'warning')
                for issue in trials['issues'][:3]:
                    self.log(f"  - {issue}", 'info')
        
        # List critical errors
        if n_errors > 0:
            self.log("\nCritical Errors:", 'error')
            for error in self.validation_results['errors'][:5]:
                self.log(f"  • {error}", 'info')
        
        # List warnings
        if n_warnings > 0 and self.verbose:
            self.log("\nWarnings:", 'warning')
            for warning in self.validation_results['warnings'][:5]:
                self.log(f"  • {warning}", 'info')
    
    def save_report(self, output_path: Optional[str] = None):
        """Save validation report to JSON file."""
        if output_path is None:
            output_path = self.h5_path.with_suffix('.validation.json')
        
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        self.log(f"\nReport saved to: {output_path}", 'success')
        return output_path


def main():
    """Command-line interface for the validator."""
    parser = argparse.ArgumentParser(
        description='Validate H5 analysis files for trial-by-trial analysis (v2.0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This validator checks:
- Dataset structure and completeness
- stimulus_coordinates group and texture dimensions
- camera_frame_id in events for direct synchronization
- Frame alignment between video and stimulus
- Event sequences and chase trials  
- Data quality metrics
- Coordinate system consistency
- Readiness for trial-by-trial analysis
Examples:
  %(prog)s out_analysis.h5
  %(prog)s out_analysis.h5 --save-report
  %(prog)s out_analysis.h5 --quiet --save-report validation.json
        """
    )
    
    parser.add_argument('h5_path', help='Path to H5 analysis file')
    parser.add_argument('--save-report', nargs='?', const=True, 
                       help='Save validation report to JSON (optional path)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create validator
    validator = H5AnalysisValidator(
        h5_path=args.h5_path,
        verbose=not args.quiet
    )
    
    # Run validation
    results = validator.validate_all()
    
    # Save report if requested
    if args.save_report:
        if isinstance(args.save_report, str):
            validator.save_report(args.save_report)
        else:
            validator.save_report()
    
    # Return exit code based on validation status
    if results['summary']['errors'] > 0:
        return 1
    elif results['summary']['warnings'] > 5:
        return 2
    else:
        return 0


if __name__ == '__main__':
    exit(main())