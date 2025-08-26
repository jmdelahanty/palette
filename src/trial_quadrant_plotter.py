#!/usr/bin/env python3
"""
Grating Trial Quadrant Proportion Plotter

Reads fish position data from zarr/H5 files and plots the proportion of time
each fish spends in each quadrant for each grating trial.

Works with the moving gratings experimental setup, not chaser experiments.
"""

import zarr
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set nice plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Event type mappings from your H5 inspector
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

STIMULUS_MODE_TYPE = {
    -1: "UNDEFINED", 2: "COHERENT_DOTS", 3: "MOVING_GRATING", 4: "SOLID_BLACK", 5: "SOLID_WHITE",
    6: "CONCENTRIC_GRATING", 7: "LOOMING_DOT", 8: "STATIC_IMAGE", 9: "CALIBRATION_GRID",
    10: "ARENA_DEFINITION_SQUARE", 11: "SPOTLIGHT", 12: "CHASER", 99: "NONE"
}

def load_grating_trials_from_h5(h5_path: str) -> pd.DataFrame:
    """Extract grating trial timing and types from H5 events data."""
    trials = []
    
    with h5py.File(h5_path, 'r') as f:
        print("Available groups in H5 file:")
        for key in f.keys():
            print(f"  - {key}")
        
        # Load events data
        if 'events' not in f:
            print("❌ No events data found in H5 file")
            return pd.DataFrame()
        
        events_data = f['events'][:]
        print(f"Found {len(events_data)} events in H5 file")
        
        # Convert to more manageable format
        events_df = pd.DataFrame({
            'timestamp_ns_session': events_data['timestamp_ns_session'],
            'event_type_id': events_data['event_type_id'],
            'current_step_index': events_data['current_step_index'],
            'stimulus_mode_id': events_data['stimulus_mode_id'],
            'details_json': [s.decode('utf-8').strip('\x00') for s in events_data['details_json']]
        })
        
        # Add human-readable names
        events_df['event_type'] = events_df['event_type_id'].map(EXPERIMENT_EVENT_TYPE)
        events_df['stimulus_mode'] = events_df['stimulus_mode_id'].map(STIMULUS_MODE_TYPE)
        
        print("\nEvent summary:")
        print(events_df['event_type'].value_counts())
        
        print("\nStimulus mode summary:")
        print(events_df['stimulus_mode'].value_counts())
        
        # Get video metadata for frame alignment
        video_fps = 60.0  # Default assumption
        video_start_time = None
        
        if 'video_metadata' in f and 'frame_metadata' in f['video_metadata']:
            frame_meta = f['video_metadata']['frame_metadata'][:]
            if len(frame_meta) > 0:
                video_start_time = frame_meta[0]['timestamp_ns']
                # Calculate FPS from frame timestamps
                if len(frame_meta) > 10:
                    frame_times = frame_meta['timestamp_ns'][:100]  # Use first 100 frames
                    time_diffs = np.diff(frame_times)
                    avg_frame_interval = np.median(time_diffs) / 1e9  # Convert to seconds
                    video_fps = 1.0 / avg_frame_interval
                    print(f"Detected video FPS: {video_fps:.1f}")
        
        # Find step boundaries to identify trials
        step_starts = events_df[events_df['event_type'] == 'STEP_START'].copy()
        step_ends = events_df[events_df['event_type'] == 'STEP_END'].copy()
        
        print(f"\nFound {len(step_starts)} step starts and {len(step_ends)} step ends")
        
        if len(step_starts) == 0:
            print("❌ No STEP_START events found - cannot identify trials")
            return pd.DataFrame()
        
        # Match step starts with step ends
        for i, (_, start_event) in enumerate(step_starts.iterrows()):
            step_index = start_event['current_step_index']
            
            # Find corresponding step end
            matching_ends = step_ends[step_ends['current_step_index'] == step_index]
            
            if len(matching_ends) == 0:
                print(f"Warning: No step end found for step {step_index}")
                continue
            
            end_event = matching_ends.iloc[0]
            
            # Convert timestamps to frame numbers (approximate)
            if video_start_time is not None:
                start_frame = int((start_event['timestamp_ns_session'] - video_start_time) / 1e9 * video_fps)
                end_frame = int((end_event['timestamp_ns_session'] - video_start_time) / 1e9 * video_fps)
            else:
                # Use session timestamp directly with estimated start
                session_start_estimate = events_df['timestamp_ns_session'].min()
                start_frame = int((start_event['timestamp_ns_session'] - session_start_estimate) / 1e9 * video_fps)
                end_frame = int((end_event['timestamp_ns_session'] - session_start_estimate) / 1e9 * video_fps)
            
            # Get stimulus events during this step
            step_events = events_df[
                (events_df['timestamp_ns_session'] >= start_event['timestamp_ns_session']) &
                (events_df['timestamp_ns_session'] <= end_event['timestamp_ns_session'])
            ]
            
            # Determine predominant stimulus type during this step
            stim_modes_during_step = step_events['stimulus_mode'].dropna()
            if len(stim_modes_during_step) > 0:
                # Get most common stimulus mode
                mode_counts = stim_modes_during_step.value_counts()
                dominant_mode = mode_counts.index[0]
                
                # Convert to trial type
                if dominant_mode == 'MOVING_GRATING':
                    trial_type = 'grating'
                elif dominant_mode == 'SOLID_WHITE':
                    trial_type = 'white'
                elif dominant_mode == 'SOLID_BLACK':
                    trial_type = 'black'
                else:
                    trial_type = dominant_mode.lower() if dominant_mode != 'UNKNOWN' else 'unknown'
            else:
                trial_type = 'unknown'
            
            # Extract orientation if available from details_json
            orientation = None
            grating_events = step_events[step_events['stimulus_mode'] == 'MOVING_GRATING']
            if len(grating_events) > 0 and not grating_events.iloc[0]['details_json'] == '':
                try:
                    import json
                    details = json.loads(grating_events.iloc[0]['details_json'])
                    orientation = details.get('orientation', None)
                except:
                    pass
            
            trials.append({
                'trial_number': i + 1,
                'step_index': step_index,
                'start_frame': max(0, start_frame),  # Ensure non-negative
                'end_frame': max(start_frame + 1, end_frame),  # Ensure end > start
                'start_timestamp': start_event['timestamp_ns_session'],
                'end_timestamp': end_event['timestamp_ns_session'],
                'duration_s': (end_event['timestamp_ns_session'] - start_event['timestamp_ns_session']) / 1e9,
                'trial_type': trial_type,
                'orientation': orientation,
                'dominant_stimulus': dominant_mode
            })
        
        trials_df = pd.DataFrame(trials)
        
        if len(trials_df) > 0:
            print(f"\n✓ Extracted {len(trials_df)} trials:")
            trial_summary = trials_df.groupby('trial_type').size()
            for trial_type, count in trial_summary.items():
                print(f"  {trial_type}: {count} trials")
        
        return trials_df


def load_fish_positions_from_zarr(zarr_path: str, use_interpolated: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load fish position data from zarr file."""
    zarr_root = zarr.open(zarr_path, mode='r')
    
    # Print available keys for debugging
    print(f"Available zarr groups/datasets: {list(zarr_root.keys())}")
    
    # Try to use merged detections first if available
    if use_interpolated and 'merged_detections' in zarr_root:
        merged_group = zarr_root['merged_detections']
        if 'latest' in merged_group.attrs:
            latest = merged_group.attrs['latest']
            data_group = merged_group[latest]
            print(f"Using merged detections: {latest}")
            
            # Load from merged format
            bbox_coords = data_group['bbox_coords'][:]
            detection_ids = data_group['detection_ids'][:]
            n_detections = data_group['n_detections'][:]
            
            # Convert to standard format [frames, max_fish, 4]
            max_fish = max(detection_ids) + 1 if len(detection_ids) > 0 else 1
            centroids = np.full((len(n_detections), max_fish, 2), np.nan)
            
            # Fill in the centroids
            cumulative_idx = 0
            for frame_idx in range(len(n_detections)):
                frame_det_count = n_detections[frame_idx]
                if frame_det_count > 0:
                    frame_bboxes = bbox_coords[cumulative_idx:cumulative_idx + frame_det_count]
                    frame_ids = detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
                    
                    for i, roi_id in enumerate(frame_ids):
                        bbox = frame_bboxes[i]
                        # Convert normalized coordinates to pixels if needed
                        if np.all(bbox <= 1.0):  # Normalized coordinates
                            # Assume 4512x4512 based on project knowledge
                            bbox = bbox * 4512
                        
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        centroids[frame_idx, roi_id] = [cx, cy]
                    
                    cumulative_idx += frame_det_count
            
            return centroids, n_detections
    
    # Try interpolated detections
    elif use_interpolated and 'interpolated_detections' in zarr_root:
        interp_group = zarr_root['interpolated_detections']
        print(f"Available interpolated runs: {list(interp_group.keys())}")
        
        if 'latest' in interp_group.attrs:
            latest = interp_group.attrs['latest']
            data_group = interp_group[latest]
            print(f"Using interpolated detections: {latest}")
            
            # Load interpolated detection data
            frame_indices = data_group['frame_indices'][:]
            roi_ids = data_group['roi_ids'][:]
            bboxes = data_group['bboxes'][:]
            
            # Also need original detections
            detect_group = zarr_root['detect_runs']
            latest_detect = detect_group.attrs['latest']
            orig_n_detections = detect_group[latest_detect]['n_detections'][:]
            orig_bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
            
            # Load ID assignments
            id_key = 'id_assignments_runs' if 'id_assignments_runs' in zarr_root else 'id_assignments'
            id_group = zarr_root[id_key]
            latest_id = id_group.attrs['latest']
            orig_detection_ids = id_group[latest_id]['detection_ids'][:]
            
            # Combine original and interpolated data
            max_fish = max(max(roi_ids) if len(roi_ids) > 0 else 0,
                          max(orig_detection_ids) if len(orig_detection_ids) > 0 else 0) + 1
            
            centroids = np.full((len(orig_n_detections), max_fish, 2), np.nan)
            
            # Fill in original detections
            cumulative_idx = 0
            for frame_idx in range(len(orig_n_detections)):
                frame_det_count = int(orig_n_detections[frame_idx])
                if frame_det_count > 0:
                    frame_bboxes = orig_bbox_coords[cumulative_idx:cumulative_idx + frame_det_count]
                    frame_ids = orig_detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
                    
                    for i, roi_id in enumerate(frame_ids):
                        bbox = frame_bboxes[i]
                        # Convert normalized coordinates to pixels
                        if np.all(bbox <= 1.0):
                            bbox = bbox * 4512
                        
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        centroids[frame_idx, roi_id] = [cx, cy]
                    
                    cumulative_idx += frame_det_count
            
            # Add interpolated detections
            for i, frame_idx in enumerate(frame_indices):
                roi_id = roi_ids[i]
                bbox = bboxes[i]
                
                # Convert normalized coordinates to pixels if needed
                if np.all(bbox <= 1.0):
                    bbox = bbox * 4512
                
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                centroids[frame_idx, roi_id] = [cx, cy]
            
            # Calculate effective n_detections
            n_detections = np.sum(~np.isnan(centroids).any(axis=2), axis=1)
            
            return centroids, n_detections
    
    # Fallback to original detect_runs
    else:
        detect_group = zarr_root['detect_runs']
        latest_detect = detect_group.attrs['latest']
        print(f"Using original detections: {latest_detect}")
        
        orig_n_detections = detect_group[latest_detect]['n_detections'][:]
        orig_bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
        
        # Load ID assignments
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in zarr_root else 'id_assignments'
        id_group = zarr_root[id_key]
        latest_id = id_group.attrs['latest']
        orig_detection_ids = id_group[latest_id]['detection_ids'][:]
        
        # Convert to standard format
        max_fish = max(orig_detection_ids) + 1 if len(orig_detection_ids) > 0 else 1
        centroids = np.full((len(orig_n_detections), max_fish, 2), np.nan)
        
        cumulative_idx = 0
        for frame_idx in range(len(orig_n_detections)):
            frame_det_count = int(orig_n_detections[frame_idx])
            if frame_det_count > 0:
                frame_bboxes = orig_bbox_coords[cumulative_idx:cumulative_idx + frame_det_count]
                frame_ids = orig_detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
                
                for i, roi_id in enumerate(frame_ids):
                    bbox = frame_bboxes[i]
                    # Convert normalized coordinates to pixels
                    if np.all(bbox <= 1.0):
                        bbox = bbox * 4512
                    
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    centroids[frame_idx, roi_id] = [cx, cy]
                
                cumulative_idx += frame_det_count
        
        return centroids, orig_n_detections


def get_arena_bounds(h5_path: str, zarr_path: str) -> Dict[str, float]:
    """Get arena boundaries for quadrant calculation."""
    # Try to get from calibration in zarr first (based on project knowledge)
    try:
        zarr_root = zarr.open(zarr_path, mode='r')
        if 'calibration' in zarr_root:
            calib = zarr_root['calibration']
            print(f"Found calibration group with keys: {list(calib.keys()) if hasattr(calib, 'keys') else 'no keys'}")
            print(f"Calibration attrs: {dict(calib.attrs) if hasattr(calib, 'attrs') else 'no attrs'}")
            
            # Check for arena subgroup
            if 'arena' in calib:
                arena = calib['arena']
                print(f"Arena attrs: {dict(arena.attrs)}")
                
                # Try the structure from project knowledge
                center_x = arena.attrs.get('center_x_px', None)
                center_y = arena.attrs.get('center_y_px', None)  
                radius = arena.attrs.get('radius_px', None)
                
                if center_x is not None and center_y is not None and radius is not None:
                    print(f"✓ Found arena calibration: center=({center_x}, {center_y}), radius={radius}")
                    return {
                        'center_x': center_x,
                        'center_y': center_y,
                        'radius': radius,
                        'left': center_x - radius,
                        'right': center_x + radius,
                        'top': center_y - radius,
                        'bottom': center_y + radius
                    }
            
            # Check for swimmable area info in calibration attrs
            swimmable_center_x = calib.attrs.get('swimmable_area_center_x_px', None)
            swimmable_center_y = calib.attrs.get('swimmable_area_center_y_px', None)
            swimmable_radius = calib.attrs.get('swimmable_area_radius_px', None)
            
            if swimmable_center_x is not None and swimmable_center_y is not None and swimmable_radius is not None:
                print(f"✓ Found swimmable area: center=({swimmable_center_x}, {swimmable_center_y}), radius={swimmable_radius}")
                return {
                    'center_x': swimmable_center_x,
                    'center_y': swimmable_center_y,
                    'radius': swimmable_radius,
                    'left': swimmable_center_x - swimmable_radius,
                    'right': swimmable_center_x + swimmable_radius,
                    'top': swimmable_center_y - swimmable_radius,
                    'bottom': swimmable_center_y + swimmable_radius
                }
                
    except Exception as e:
        print(f"Error reading zarr calibration: {e}")
        pass
    
    # Try H5 calibration_snapshot
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'calibration_snapshot' in f:
                calib = f['calibration_snapshot']
                # Look for arena config JSON
                if 'arena_config_json' in calib:
                    arena_json = calib['arena_config_json'][()].decode('utf-8')
                    import json
                    arena_config = json.loads(arena_json)
                    
                    # Try to find swimmable area parameters
                    swimmable_center_x = arena_config.get('swimmable_area_center_x_px', None)
                    swimmable_center_y = arena_config.get('swimmable_area_center_y_px', None)
                    swimmable_radius = arena_config.get('swimmable_area_radius_px', None)
                    
                    if swimmable_center_x is not None and swimmable_center_y is not None and swimmable_radius is not None:
                        print(f"✓ Found H5 swimmable area: center=({swimmable_center_x}, {swimmable_center_y}), radius={swimmable_radius}")
                        return {
                            'center_x': swimmable_center_x,
                            'center_y': swimmable_center_y,
                            'radius': swimmable_radius,
                            'left': swimmable_center_x - swimmable_radius,
                            'right': swimmable_center_x + swimmable_radius,
                            'top': swimmable_center_y - swimmable_radius,
                            'bottom': swimmable_center_y + swimmable_radius
                        }
    except Exception as e:
        print(f"Warning: Error reading H5 calibration: {e}")
        pass
    
    # Ultimate fallback: estimate from data
    print("Warning: Could not find arena calibration, estimating from data...")
    zarr_root = zarr.open(zarr_path, mode='r')
    
    # Get position data from merged detections or detect runs
    all_x, all_y = [], []
    
    try:
        if 'merged_detections' in zarr_root:
            merged_group = zarr_root['merged_detections']
            if 'latest' in merged_group.attrs:
                latest = merged_group.attrs['latest']
                data_group = merged_group[latest]
                bbox_coords = data_group['bbox_coords'][:]
                
                for bbox in bbox_coords:
                    if not np.isnan(bbox).any():
                        # Convert normalized to pixels if needed
                        if np.all(bbox <= 1.0):
                            bbox = bbox * 4512
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        all_x.append(cx)
                        all_y.append(cy)
        
        elif 'detect_runs' in zarr_root:
            detect_group = zarr_root['detect_runs']
            latest_detect = detect_group.attrs['latest']
            bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
            
            for bbox in bbox_coords:
                if not np.isnan(bbox).any():
                    # Convert normalized to pixels
                    if np.all(bbox <= 1.0):
                        bbox = bbox * 4512
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    all_x.append(cx)
                    all_y.append(cy)
        
        if len(all_x) > 0:
            center_x = np.mean(all_x)
            center_y = np.mean(all_y)
            span_x = np.max(all_x) - np.min(all_x)
            span_y = np.max(all_y) - np.min(all_y)
            radius = max(span_x, span_y) / 2 * 1.1  # Add 10% margin
            
            print(f"Estimated arena: center=({center_x:.1f}, {center_y:.1f}), radius={radius:.1f}")
            
            return {
                'center_x': center_x,
                'center_y': center_y,
                'radius': radius,
                'left': center_x - radius,
                'right': center_x + radius,
                'top': center_y - radius,
                'bottom': center_y + radius
            }
    
    except Exception as e:
        print(f"Error estimating arena bounds: {e}")
    
    # Absolute last resort
    print("Using default arena bounds")
    return {
        'center_x': 2256, 'center_y': 2256, 'radius': 400,
        'left': 1856, 'right': 2656, 'top': 1856, 'bottom': 2656
    }


def calculate_quadrant_proportions(centroids: np.ndarray, arena_bounds: Dict, 
                                 trials_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate proportion of time each fish spends in each quadrant per trial."""
    results = []
    
    center_x = arena_bounds['center_x']
    center_y = arena_bounds['center_y']
    
    n_fish = centroids.shape[1]
    
    for _, trial in trials_df.iterrows():
        start_frame = trial['start_frame']
        end_frame = trial['end_frame']
        
        # Extract trial data
        trial_centroids = centroids[start_frame:end_frame+1]
        
        for fish_id in range(n_fish):
            fish_positions = trial_centroids[:, fish_id, :]
            
            # Remove NaN positions
            valid_mask = ~np.isnan(fish_positions).any(axis=1)
            if not valid_mask.any():
                continue
                
            valid_positions = fish_positions[valid_mask]
            total_valid_frames = len(valid_positions)
            
            # Calculate quadrant for each position
            x_coords = valid_positions[:, 0]
            y_coords = valid_positions[:, 1]
            
            # Define quadrants relative to arena center
            q1_mask = (x_coords >= center_x) & (y_coords <= center_y)  # Top-right
            q2_mask = (x_coords < center_x) & (y_coords <= center_y)   # Top-left  
            q3_mask = (x_coords < center_x) & (y_coords > center_y)    # Bottom-left
            q4_mask = (x_coords >= center_x) & (y_coords > center_y)   # Bottom-right
            
            # Calculate proportions
            q1_prop = np.sum(q1_mask) / total_valid_frames
            q2_prop = np.sum(q2_mask) / total_valid_frames
            q3_prop = np.sum(q3_mask) / total_valid_frames
            q4_prop = np.sum(q4_mask) / total_valid_frames
            
            # Top proportion (Q1 + Q2)
            top_prop = q1_prop + q2_prop
            
            results.append({
                'roi_id': fish_id,
                'trial_number': trial['trial_number'],
                'trial_type': trial['trial_type'],
                'orientation': trial.get('orientation', None),
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration_frames': end_frame - start_frame + 1,
                'valid_frames': total_valid_frames,
                'detection_rate': total_valid_frames / (end_frame - start_frame + 1),
                'q1_proportion': q1_prop,  # Top-right
                'q2_proportion': q2_prop,  # Top-left
                'q3_proportion': q3_prop,  # Bottom-left
                'q4_proportion': q4_prop,  # Bottom-right
                'top_proportion': top_prop,
                'bottom_proportion': q3_prop + q4_prop,
                'left_proportion': q2_prop + q3_prop,
                'right_proportion': q1_prop + q4_prop
            })
    
    return pd.DataFrame(results)


def plot_trial_proportions(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot trial-by-trial quadrant proportions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color scheme
    stimulus_colors = {
        'grating': '#FF6B6B',
        'white': '#4ECDC4', 
        'black': '#45B7D1'
    }
    
    # 1. Top proportion by trial for each fish
    ax = axes[0, 0]
    for fish_id in sorted(df['roi_id'].unique()):
        fish_data = df[df['roi_id'] == fish_id].sort_values('trial_number')
        ax.plot(fish_data['trial_number'], fish_data['top_proportion'], 
               marker='o', alpha=0.7, label=f'Fish {fish_id}', linewidth=2)
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='No preference')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Top Quadrant Proportion')
    ax.set_title('Top Quadrant Preference by Trial', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 2. Heatmap: Fish vs Trial
    ax = axes[0, 1]
    pivot_data = df.pivot_table(values='top_proportion', 
                               index='roi_id', 
                               columns='trial_number', 
                               aggfunc='mean')
    
    im = ax.imshow(pivot_data, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Fish ID')
    ax.set_title('Top Proportion Heatmap', fontweight='bold')
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels([f'Fish {int(i)}' for i in pivot_data.index])
    plt.colorbar(im, ax=ax, label='Top Proportion', shrink=0.6)
    
    # 3. Distribution by stimulus type
    ax = axes[1, 0]
    for stim_type in df['trial_type'].unique():
        if stim_type in stimulus_colors:
            stim_data = df[df['trial_type'] == stim_type]['top_proportion']
            ax.hist(stim_data, bins=20, alpha=0.6, 
                   color=stimulus_colors[stim_type], label=stim_type, density=True)
    
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Top Quadrant Proportion')
    ax.set_ylabel('Density')
    ax.set_title('Distribution by Stimulus Type', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Mean by stimulus with error bars
    ax = axes[1, 1]
    stim_summary = df.groupby(['trial_type', 'roi_id'])['top_proportion'].mean().reset_index()
    stim_means = stim_summary.groupby('trial_type')['top_proportion'].agg(['mean', 'std', 'count'])
    
    x_pos = range(len(stim_means.index))
    colors = [stimulus_colors.get(stim, 'gray') for stim in stim_means.index]
    
    bars = ax.bar(x_pos, stim_means['mean'], yerr=stim_means['std'], 
                 capsize=5, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stim_means.index)
    ax.set_ylabel('Mean Top Proportion')
    ax.set_xlabel('Stimulus Type')
    ax.set_title('Mean Preference by Stimulus', fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, stim_means['mean']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Grating Trial Quadrant Analysis', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def save_proportions_csv(df: pd.DataFrame, csv_path: str):
    """Save proportion data to CSV for use with other tools."""
    df.to_csv(csv_path, index=False)
    print(f"✓ Proportions data saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot trial-by-trial quadrant proportions for grating experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('zarr_path', help='Path to zarr detection file')
    parser.add_argument('h5_path', help='Path to H5 experiment file')
    parser.add_argument('--save-plot', type=str, help='Save plot to file')
    parser.add_argument('--save-csv', type=str, help='Save data to CSV file')
    parser.add_argument('--no-interpolated', action='store_true', 
                       help='Use original detections instead of interpolated')
    parser.add_argument('--print-summary', action='store_true',
                       help='Print summary statistics')
    
    args = parser.parse_args()
    
    print("Loading grating trial quadrant analysis...")
    print(f"Zarr: {args.zarr_path}")
    print(f"H5: {args.h5_path}")
    
    # Load trial structure
    trials_df = load_grating_trials_from_h5(args.h5_path)
    if trials_df.empty:
        print("❌ Could not load trial structure from H5 file")
        return
    
    print(f"Found {len(trials_df)} trials:")
    for trial_type in trials_df['trial_type'].unique():
        count = len(trials_df[trials_df['trial_type'] == trial_type])
        print(f"  {trial_type}: {count} trials")
    
    # Load fish positions
    centroids, n_detections = load_fish_positions_from_zarr(
        args.zarr_path, use_interpolated=not args.no_interpolated
    )
    print(f"Loaded positions for {centroids.shape[1]} fish across {centroids.shape[0]} frames")
    
    # Get arena bounds
    arena_bounds = get_arena_bounds(args.h5_path, args.zarr_path)
    print(f"Arena center: ({arena_bounds['center_x']:.1f}, {arena_bounds['center_y']:.1f})")
    print(f"Arena radius: {arena_bounds['radius']:.1f} px")
    
    # Calculate quadrant proportions
    proportions_df = calculate_quadrant_proportions(centroids, arena_bounds, trials_df)
    
    if proportions_df.empty:
        print("❌ No valid position data found")
        return
    
    print(f"✓ Calculated proportions for {len(proportions_df)} fish-trial combinations")
    
    # Print summary if requested
    if args.print_summary:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        overall_mean = proportions_df['top_proportion'].mean()
        print(f"Overall mean top proportion: {overall_mean:.3f}")
        
        print("\nBy stimulus type:")
        for stim_type in proportions_df['trial_type'].unique():
            stim_data = proportions_df[proportions_df['trial_type'] == stim_type]
            mean_prop = stim_data['top_proportion'].mean()
            std_prop = stim_data['top_proportion'].std()
            n_obs = len(stim_data)
            print(f"  {stim_type:8s}: {mean_prop:.3f} ± {std_prop:.3f} (n={n_obs})")
        
        print(f"\nBy fish:")
        fish_stats = proportions_df.groupby('roi_id')['top_proportion'].agg(['mean', 'std', 'count'])
        for fish_id, stats in fish_stats.iterrows():
            print(f"  Fish {int(fish_id):2d}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={int(stats['count'])})")
    
    # Save CSV if requested
    if args.save_csv:
        save_proportions_csv(proportions_df, args.save_csv)
    
    # Create plot
    plot_trial_proportions(proportions_df, save_path=args.save_plot)


if __name__ == '__main__':
    main()