#!/usr/bin/env python3
"""
Trial-by-Trial Chase Response Analyzer (v2.0)

Analyzes fish behavioral responses to individual chase trials, integrating:
- YOLO detections from zarr files
- Chase events and chaser positions from H5 files
- Proper frame alignment between 60Hz camera and 120Hz stimulus
- Correct coordinate transformations (texture→camera)

This analyzer extracts metrics for each chase trial including:
- Fish-chaser distance over time
- Escape responses and velocities
- Spatial distribution changes
- Response latencies
"""

import zarr
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import logging

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Event type mappings
EVENT_TYPES = {
    0: "PROTOCOL_START",
    4: "PROTOCOL_FINISH",
    24: "CHASER_PRE_PERIOD_START",
    25: "CHASER_TRAINING_START", 
    26: "CHASER_POST_PERIOD_START",
    27: "CHASER_CHASE_SEQUENCE_START",
    28: "CHASER_CHASE_SEQUENCE_END"
}

@dataclass
class TrialMetrics:
    """Metrics for a single chase trial."""
    trial_number: int
    start_time_s: float
    duration_s: float
    phase: str  # 'pre', 'training', 'post'
    
    # Distance metrics
    mean_distance_px: float
    min_distance_px: float
    max_distance_px: float
    initial_distance_px: float
    final_distance_px: float
    distance_change_px: float  # final - initial
    
    # Movement metrics
    fish_total_distance_px: float
    fish_mean_speed_px_per_s: float
    fish_max_speed_px_per_s: float
    chaser_total_distance_px: float
    chaser_mean_speed_px_per_s: float
    
    # Response metrics
    escape_detected: bool
    escape_latency_s: Optional[float]
    escape_speed_px_per_s: Optional[float]
    escape_distance_px: Optional[float]  # Distance when escape triggered
    approach_events: int
    min_approach_distance_px: Optional[float]
    
    # Coverage
    frames_with_detection: int
    total_frames: int
    detection_rate: float
    
    # Additional behavioral metrics
    time_below_threshold_s: float  # Time spent within approach threshold
    mean_relative_velocity_px_per_s: float  # Negative = approaching


class TrialByTrialAnalyzer:
    """Analyzes fish responses to individual chase trials with proper alignment."""
    
    def __init__(self, 
                 zarr_path: str,
                 h5_path: str,
                 escape_threshold_px_per_s: float = 500,
                 approach_threshold_px: float = 200,
                 verbose: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            zarr_path: Path to zarr file with YOLO detections
            h5_path: Path to H5 file with chase events and chaser positions
            escape_threshold_px_per_s: Speed threshold for escape detection
            approach_threshold_px: Distance threshold for approach detection
            verbose: Print progress messages
        """
        self.zarr_path = Path(zarr_path)
        self.h5_path = Path(h5_path)
        self.escape_threshold = escape_threshold_px_per_s
        self.approach_threshold = approach_threshold_px
        self.verbose = verbose
        
        # Setup logging
        self._setup_logging()
        
        # Camera to texture scaling (standard for your system)
        self.camera_width = 4512
        self.camera_height = 4512
        self.texture_width = 358
        self.texture_height = 358
        self.texture_to_camera_scale = self.camera_width / self.texture_width  # 12.604
        
        # Frame rate info
        self.camera_fps = 60.0
        self.stimulus_hz = 120.0
        self.fps_ratio = self.stimulus_hz / self.camera_fps  # ~2
        
        # Data containers
        self.zarr_root = None
        self.events = None
        self.chaser_states = None
        self.frame_metadata = None
        self.trials = []
        self.camera_to_stimulus = {}
        self.stimulus_to_camera = {}
        self.zarr_to_camera_offset = 0
        
        # Load data
        self.load_data()
        
        # Create frame alignment
        self.create_frame_alignment()
        
        # Extract trials
        self.extract_trials()
    
    def _setup_logging(self):
        """Configure logging."""
        level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """Load data from zarr and H5 files."""
        self.logger.info("Loading data...")
        
        # Load zarr data (YOLO detections)
        self.zarr_root = zarr.open(self.zarr_path, mode='r')
        
        # Try to get best available data
        if 'interpolation_runs' in self.zarr_root and self.zarr_root['interpolation_runs'].attrs.get('latest'):
            latest = self.zarr_root['interpolation_runs'].attrs['latest']
            data = self.zarr_root['interpolation_runs'][latest]
            self.bboxes = data['bboxes'][:]
            self.n_detections = data['n_detections'][:]
            self.interpolation_mask = data['interpolation_mask'][:]
            self.logger.info(f"  Using interpolated data: {latest}")
        else:
            self.bboxes = self.zarr_root['bboxes'][:]
            self.n_detections = self.zarr_root['n_detections'][:]
            self.interpolation_mask = np.zeros(len(self.n_detections), dtype=bool)
            self.logger.info("  Using raw detection data")
        
        # Load H5 data
        with h5py.File(self.h5_path, 'r') as f:
            # Events
            self.events = f['/events'][:]
            
            # Check if PROTOCOL_START was fixed
            if 'protocol_start_fixed' in f['/events'].attrs:
                self.logger.info("  ✅ Using fixed PROTOCOL_START")
            
            # Chaser states
            self.chaser_states = f['/tracking_data/chaser_states'][:]
            
            # Frame metadata for alignment
            self.frame_metadata = f['/video_metadata/frame_metadata'][:]
        
        self.logger.info(f"  Loaded {len(self.bboxes)} frames of detections")
        self.logger.info(f"  Loaded {len(self.events)} events")
        self.logger.info(f"  Loaded {len(self.chaser_states)} chaser states")
        self.logger.info(f"  Loaded {len(self.frame_metadata)} frame metadata records")
    
    def create_frame_alignment(self):
        """Create proper frame alignment between zarr and H5 data."""
        self.logger.info("Creating frame alignment...")
        
        # Build camera→stimulus mapping from frame metadata
        for record in self.frame_metadata:
            cam_frame = int(record['triggering_camera_frame_id'])
            stim_frame = int(record['stimulus_frame_num'])
            
            # Handle multiple stimulus frames per camera frame (120Hz vs 60Hz)
            if cam_frame not in self.camera_to_stimulus:
                self.camera_to_stimulus[cam_frame] = []
            self.camera_to_stimulus[cam_frame].append(stim_frame)
            
            # Keep reverse mapping (many-to-one)
            self.stimulus_to_camera[stim_frame] = cam_frame
        
        # Determine zarr offset (zarr frames start at 0, H5 camera frames may have offset)
        min_camera_frame = min(self.camera_to_stimulus.keys()) if self.camera_to_stimulus else 0
        self.zarr_to_camera_offset = min_camera_frame
        
        self.logger.info(f"  Camera frame range: {min_camera_frame} to {max(self.camera_to_stimulus.keys())}")
        self.logger.info(f"  Zarr to camera offset: {self.zarr_to_camera_offset}")
        self.logger.info(f"  Average stimulus frames per camera: {np.mean([len(s) for s in self.camera_to_stimulus.values()]):.2f}")
    
    def extract_trials(self):
        """Extract individual chase trials from events."""
        self.trials = []
        
        # Find chase sequences
        chase_starts = []
        chase_ends = []
        
        for event in self.events:
            if event['event_type_id'] == 27:  # CHASE_START
                chase_starts.append(event)
            elif event['event_type_id'] == 28:  # CHASE_END
                chase_ends.append(event)
        
        # Determine training phases from events
        training_start_frame = None
        post_start_frame = None
        
        for event in self.events:
            if event['event_type_id'] == 25:  # TRAINING_START
                training_start_frame = event['camera_frame_id'] if 'camera_frame_id' in event.dtype.names else None
            elif event['event_type_id'] == 26:  # POST_START  
                post_start_frame = event['camera_frame_id'] if 'camera_frame_id' in event.dtype.names else None
        
        # Process each chase
        for i, (start_event, end_event) in enumerate(zip(chase_starts, chase_ends)):
            # Get camera frames for phase determination
            start_cam_frame = start_event['camera_frame_id'] if 'camera_frame_id' in start_event.dtype.names else 0
            
            # Determine phase based on camera frame
            if training_start_frame and start_cam_frame < training_start_frame:
                phase = 'pre'
            elif post_start_frame and start_cam_frame >= post_start_frame:
                phase = 'post'
            else:
                phase = 'training'
            
            trial = {
                'number': i + 1,
                'phase': phase,
                'start_event': start_event,
                'end_event': end_event,
                'start_camera_frame': int(start_cam_frame),
                'end_camera_frame': int(end_event['camera_frame_id']) if 'camera_frame_id' in end_event.dtype.names else 0,
                'start_stimulus_frame': int(start_event['stimulus_frame_num']) if 'stimulus_frame_num' in start_event.dtype.names else 0,
                'end_stimulus_frame': int(end_event['stimulus_frame_num']) if 'stimulus_frame_num' in end_event.dtype.names else 0,
                'start_time_ns': int(start_event['timestamp_ns_session']),
                'end_time_ns': int(end_event['timestamp_ns_session']),
                'duration_s': (end_event['timestamp_ns_session'] - start_event['timestamp_ns_session']) / 1e9
            }
            
            self.trials.append(trial)
        
        if self.verbose:
            self.logger.info(f"\nExtracted {len(self.trials)} trials:")
            phase_counts = {'pre': 0, 'training': 0, 'post': 0}
            for trial in self.trials:
                phase_counts[trial['phase']] += 1
            for phase, count in phase_counts.items():
                self.logger.info(f"  {phase}: {count} trials")
    
    def get_trial_data(self, trial: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get fish positions, chaser positions, and timestamps for a trial.
        
        Returns:
            fish_positions: Array of (x, y) positions in camera space
            chaser_positions: Array of (x, y) positions in camera space  
            timestamps: Array of timestamps in seconds
        """
        # Get stimulus frame range for this trial
        stim_start = trial['start_stimulus_frame']
        stim_end = trial['end_stimulus_frame']
        
        # Get chaser states for this stimulus range
        chaser_mask = (self.chaser_states['stimulus_frame_num'] >= stim_start) & \
                     (self.chaser_states['stimulus_frame_num'] <= stim_end)
        trial_chaser = self.chaser_states[chaser_mask]
        
        if len(trial_chaser) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Get unique camera frames for this trial
        camera_frames = []
        for stim_frame in trial_chaser['stimulus_frame_num']:
            if stim_frame in self.stimulus_to_camera:
                cam_frame = self.stimulus_to_camera[stim_frame]
                if cam_frame not in camera_frames:
                    camera_frames.append(cam_frame)
        
        camera_frames = sorted(camera_frames)
        
        # Initialize arrays
        fish_positions = []
        chaser_positions = []
        timestamps = []
        
        for cam_frame in camera_frames:
            # Convert camera frame to zarr index
            zarr_idx = cam_frame - self.zarr_to_camera_offset
            
            # Get fish position from zarr
            if 0 <= zarr_idx < len(self.bboxes) and self.n_detections[zarr_idx] > 0:
                bbox = self.bboxes[zarr_idx, 0]  # First detection
                fish_x = (bbox[0] + bbox[2]) / 2  # Center x
                fish_y = (bbox[1] + bbox[3]) / 2  # Center y
                fish_positions.append([fish_x, fish_y])
            else:
                fish_positions.append([np.nan, np.nan])
            
            # Get chaser position for this camera frame
            # Use middle stimulus frame if multiple map to this camera frame
            if cam_frame in self.camera_to_stimulus:
                stim_frames = self.camera_to_stimulus[cam_frame]
                middle_stim = stim_frames[len(stim_frames)//2]
                
                # Find chaser state for this stimulus frame
                chaser_idx = np.where(trial_chaser['stimulus_frame_num'] == middle_stim)[0]
                if len(chaser_idx) > 0:
                    chaser_state = trial_chaser[chaser_idx[0]]
                    # Convert texture space to camera space
                    chaser_x = chaser_state['chaser_pos_x'] * self.texture_to_camera_scale
                    chaser_y = chaser_state['chaser_pos_y'] * self.texture_to_camera_scale
                    chaser_positions.append([chaser_x, chaser_y])
                else:
                    chaser_positions.append([np.nan, np.nan])
            else:
                chaser_positions.append([np.nan, np.nan])
            
            # Calculate timestamp (use camera frame timing)
            timestamps.append((cam_frame - camera_frames[0]) / self.camera_fps)
        
        return np.array(fish_positions), np.array(chaser_positions), np.array(timestamps)
    
    def calculate_trial_metrics(self, trial: Dict) -> TrialMetrics:
        """Calculate comprehensive metrics for a single trial."""
        # Get trial data
        fish_pos, chaser_pos, timestamps = self.get_trial_data(trial)
        
        if len(fish_pos) == 0:
            # Return empty metrics if no data
            return TrialMetrics(
                trial_number=trial['number'],
                start_time_s=trial['start_time_ns'] / 1e9,
                duration_s=trial['duration_s'],
                phase=trial['phase'],
                mean_distance_px=np.nan,
                min_distance_px=np.nan,
                max_distance_px=np.nan,
                initial_distance_px=np.nan,
                final_distance_px=np.nan,
                distance_change_px=np.nan,
                fish_total_distance_px=0,
                fish_mean_speed_px_per_s=0,
                fish_max_speed_px_per_s=0,
                chaser_total_distance_px=0,
                chaser_mean_speed_px_per_s=0,
                escape_detected=False,
                escape_latency_s=None,
                escape_speed_px_per_s=None,
                escape_distance_px=None,
                approach_events=0,
                min_approach_distance_px=None,
                frames_with_detection=0,
                total_frames=len(fish_pos) if len(fish_pos) > 0 else 1,
                detection_rate=0,
                time_below_threshold_s=0,
                mean_relative_velocity_px_per_s=0
            )
        
        # Calculate distances
        valid_mask = ~np.isnan(fish_pos[:, 0]) & ~np.isnan(chaser_pos[:, 0])
        distances = np.sqrt(np.sum((fish_pos - chaser_pos)**2, axis=1))
        valid_distances = distances[valid_mask]
        
        # Distance metrics
        if len(valid_distances) > 0:
            mean_distance = np.mean(valid_distances)
            min_distance = np.min(valid_distances)
            max_distance = np.max(valid_distances)
            initial_distance = valid_distances[0] if len(valid_distances) > 0 else np.nan
            final_distance = valid_distances[-1] if len(valid_distances) > 0 else np.nan
            distance_change = final_distance - initial_distance
            
            # Time below threshold
            time_below = np.sum(valid_distances < self.approach_threshold)
            if time_below > 0 and len(timestamps) > 1:
                time_below_threshold_s = time_below * np.mean(np.diff(timestamps))
            else:
                time_below_threshold_s = 0
            
            # Min approach distance
            approaches = valid_distances[valid_distances < self.approach_threshold]
            min_approach_distance = np.min(approaches) if len(approaches) > 0 else None
        else:
            mean_distance = min_distance = max_distance = np.nan
            initial_distance = final_distance = distance_change = np.nan
            time_below_threshold_s = 0
            min_approach_distance = None
        
        # Fish movement metrics with noise filtering
        valid_fish = fish_pos[valid_mask]
        if len(valid_fish) > 1:
            fish_displacements = np.sqrt(np.sum(np.diff(valid_fish, axis=0)**2, axis=1))
            fish_total_distance = np.sum(fish_displacements)
            
            # Calculate speeds with smoothing
            time_diffs = np.diff(timestamps[valid_mask])
            time_diffs[time_diffs == 0] = 1/self.camera_fps  # Avoid division by zero
            
            # Raw speeds
            raw_speeds = fish_displacements / time_diffs
            
            # Apply smoothing to reduce noise
            # Method 1: Moving average (window of 3 frames)
            if len(raw_speeds) > 3:
                fish_speeds = np.convolve(raw_speeds, np.ones(3)/3, mode='same')
            else:
                fish_speeds = raw_speeds
            
            # Method 2: Filter out unrealistic single-frame spikes
            # If a speed is >3x the median and lasts only 1 frame, it's likely noise
            median_speed = np.median(fish_speeds)
            for i in range(1, len(fish_speeds) - 1):
                if fish_speeds[i] > 3 * median_speed:
                    # Check if it's an isolated spike
                    if fish_speeds[i-1] < median_speed * 1.5 and fish_speeds[i+1] < median_speed * 1.5:
                        # Replace with average of neighbors
                        fish_speeds[i] = (fish_speeds[i-1] + fish_speeds[i+1]) / 2
            
            fish_mean_speed = np.mean(fish_speeds)
            fish_max_speed = np.max(fish_speeds)
            
            # Detect escape response with minimum displacement check
            # Only consider it an escape if there's substantial movement
            MIN_ESCAPE_DISPLACEMENT = 20  # pixels - fish must move at least this much
            
            escape_detected = False
            escape_latency = None
            escape_speed = None
            escape_distance = None
            
            # Check for escape: high speed AND significant displacement
            for i in range(len(fish_speeds)):
                if fish_speeds[i] > self.escape_threshold:
                    # Check if there's real movement (not just noise)
                    if fish_displacements[i] > MIN_ESCAPE_DISPLACEMENT:
                        escape_detected = True
                        escape_idx = i
                        escape_latency = timestamps[valid_mask][escape_idx + 1] - timestamps[0]
                        escape_speed = fish_speeds[escape_idx]
                        escape_distance = valid_distances[escape_idx] if escape_idx < len(valid_distances) else None
                        break
        else:
            fish_total_distance = 0
            fish_mean_speed = 0
            fish_max_speed = 0
            escape_detected = False
            escape_latency = None
            escape_speed = None
            escape_distance = None
        
        # Chaser movement
        valid_chaser = chaser_pos[valid_mask]
        if len(valid_chaser) > 1:
            chaser_displacements = np.sqrt(np.sum(np.diff(valid_chaser, axis=0)**2, axis=1))
            chaser_total_distance = np.sum(chaser_displacements)
            chaser_mean_speed = np.mean(chaser_displacements / time_diffs)
        else:
            chaser_total_distance = 0
            chaser_mean_speed = 0
        
        # Relative velocity (rate of change of distance)
        if len(valid_distances) > 1:
            distance_changes = np.diff(valid_distances)
            relative_velocities = distance_changes / time_diffs
            mean_relative_velocity = np.mean(relative_velocities)
        else:
            mean_relative_velocity = 0
        
        # Approach events
        approach_events = 0
        if len(valid_distances) > 1:
            below_threshold = valid_distances < self.approach_threshold
            # Count transitions from above to below threshold
            transitions = np.diff(below_threshold.astype(int))
            approach_events = np.sum(transitions > 0)
        
        return TrialMetrics(
            trial_number=trial['number'],
            start_time_s=trial['start_time_ns'] / 1e9,
            duration_s=trial['duration_s'],
            phase=trial['phase'],
            mean_distance_px=mean_distance,
            min_distance_px=min_distance,
            max_distance_px=max_distance,
            initial_distance_px=initial_distance,
            final_distance_px=final_distance,
            distance_change_px=distance_change,
            fish_total_distance_px=fish_total_distance,
            fish_mean_speed_px_per_s=fish_mean_speed,
            fish_max_speed_px_per_s=fish_max_speed,
            chaser_total_distance_px=chaser_total_distance,
            chaser_mean_speed_px_per_s=chaser_mean_speed,
            escape_detected=escape_detected,
            escape_latency_s=escape_latency,
            escape_speed_px_per_s=escape_speed,
            escape_distance_px=escape_distance,
            approach_events=approach_events,
            min_approach_distance_px=min_approach_distance,
            frames_with_detection=np.sum(valid_mask),
            total_frames=len(fish_pos),
            detection_rate=np.sum(valid_mask) / len(fish_pos) if len(fish_pos) > 0 else 0,
            time_below_threshold_s=time_below_threshold_s,
            mean_relative_velocity_px_per_s=mean_relative_velocity
        )
    
    def analyze_all_trials(self) -> pd.DataFrame:
        """Analyze all trials and return results as DataFrame."""
        self.logger.info("\nAnalyzing trials...")
        
        metrics_list = []
        for trial in self.trials:
            metrics = self.calculate_trial_metrics(trial)
            metrics_list.append(asdict(metrics))
            
            if self.verbose and trial['number'] % 5 == 0:
                self.logger.info(f"  Processed trial {trial['number']}/{len(self.trials)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_list)
        
        # Add pixel to mm conversion
        pixel_to_mm = 88.5 / self.camera_width  # Arena is 88.5mm wide
        for col in df.columns:
            if '_px' in col and 'px_per_s' not in col:  # Don't convert velocities
                mm_col = col.replace('_px', '_mm')
                df[mm_col] = df[col] * pixel_to_mm
        
        self.logger.info(f"\nAnalysis complete! Processed {len(df)} trials")
        
        return df
    
    def plot_trial_summary(self, trial_num: int):
        """Plot detailed summary for a single trial with improved visualization."""
        trial = self.trials[trial_num - 1]
        fish_pos, chaser_pos, timestamps = self.get_trial_data(trial)
        
        if len(fish_pos) == 0:
            print(f"No data available for trial {trial_num}")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Color scheme
        fish_color = '#2E86AB'  # Blue
        chaser_color = '#A23B72'  # Purple
        
        # 1. Trajectory plot
        ax1 = fig.add_subplot(gs[:2, :2])
        valid = ~np.isnan(fish_pos[:, 0])
        
        # Plot trajectories with gradient color for time
        if np.sum(valid) > 0:
            time_colors = timestamps - timestamps[0]
            
            # Fish trajectory
            for i in range(len(fish_pos[valid]) - 1):
                if valid[i] and valid[i+1]:
                    ax1.plot(fish_pos[i:i+2, 0], fish_pos[i:i+2, 1], 
                           color=fish_color, alpha=0.3 + 0.7*i/len(fish_pos), linewidth=2)
            
            # Chaser trajectory
            for i in range(len(chaser_pos[valid]) - 1):
                if valid[i] and valid[i+1]:
                    ax1.plot(chaser_pos[i:i+2, 0], chaser_pos[i:i+2, 1],
                           color=chaser_color, alpha=0.3 + 0.7*i/len(chaser_pos), linewidth=2)
            
            # Mark start and end points
            ax1.scatter(fish_pos[valid][0, 0], fish_pos[valid][0, 1], 
                      c=fish_color, s=150, marker='o', edgecolors='white', linewidth=2,
                      label='Fish start', zorder=5)
            ax1.scatter(fish_pos[valid][-1, 0], fish_pos[valid][-1, 1], 
                      c=fish_color, s=150, marker='s', edgecolors='white', linewidth=2,
                      label='Fish end', zorder=5)
            ax1.scatter(chaser_pos[valid][0, 0], chaser_pos[valid][0, 1],
                      c=chaser_color, s=150, marker='o', edgecolors='white', linewidth=2,
                      label='Chaser start', zorder=5)
            
            # Add arena boundary
            arena = Circle((self.camera_width/2, self.camera_height/2), 
                         self.camera_width/2, fill=False, 
                         edgecolor='gray', linewidth=1, linestyle='--', alpha=0.5)
            ax1.add_patch(arena)
        
        ax1.set_xlim(0, self.camera_width)
        ax1.set_ylim(0, self.camera_height)
        ax1.set_xlabel('X (pixels)', fontsize=11)
        ax1.set_ylabel('Y (pixels)', fontsize=11)
        ax1.set_title(f'Trial {trial_num} ({trial["phase"].capitalize()}) - Spatial Trajectories', 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Match image coordinates
        
        # 2. Distance over time
        ax2 = fig.add_subplot(gs[2, :])
        distances = np.sqrt(np.sum((fish_pos - chaser_pos)**2, axis=1))
        valid_times = timestamps - timestamps[0]
        ax2.plot(valid_times[valid], distances[valid], color='#2F4858', linewidth=2.5)
        ax2.fill_between(valid_times[valid], 0, distances[valid], alpha=0.3, color='#2F4858')
        ax2.axhline(y=self.approach_threshold, color='red', linestyle='--', 
                   alpha=0.7, linewidth=1.5, label=f'Approach threshold ({self.approach_threshold:.0f} px)')
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Distance (pixels)', fontsize=11)
        ax2.set_title('Fish-Chaser Distance Over Time', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Speed profile with raw and smoothed
        ax3 = fig.add_subplot(gs[0, 2])
        if np.sum(valid) > 1:
            time_diffs = np.diff(timestamps[valid])
            time_diffs[time_diffs == 0] = 1/self.camera_fps
            fish_displacements = np.sqrt(np.sum(np.diff(fish_pos[valid], axis=0)**2, axis=1))
            
            # Calculate both raw and smoothed speeds
            raw_speeds = fish_displacements / time_diffs
            
            # Smooth speeds
            if len(raw_speeds) > 3:
                smooth_speeds = np.convolve(raw_speeds, np.ones(3)/3, mode='same')
            else:
                smooth_speeds = raw_speeds
            
            speed_times = valid_times[valid][1:]
            
            # Plot both raw (faint) and smoothed (bold)
            ax3.plot(speed_times, raw_speeds, color=fish_color, linewidth=0.5, alpha=0.3, label='Raw')
            ax3.plot(speed_times, smooth_speeds, color=fish_color, linewidth=2, alpha=0.8, label='Smoothed')
            ax3.fill_between(speed_times, 0, smooth_speeds, alpha=0.3, color=fish_color)
            
            ax3.axhline(y=self.escape_threshold, color='orange', linestyle='--', 
                       alpha=0.7, linewidth=1.5, label=f'Escape threshold ({self.escape_threshold:.0f} px/s)')
            
            # Mark escape events (only if displacement is significant)
            MIN_ESCAPE_DISPLACEMENT = 20
            real_escapes = (smooth_speeds > self.escape_threshold) & (fish_displacements > MIN_ESCAPE_DISPLACEMENT)
            if np.any(real_escapes):
                ax3.scatter(speed_times[real_escapes], smooth_speeds[real_escapes], 
                          color='red', s=50, zorder=5, label='Escape events')
            
            # Add displacement subplot (inset)
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ax3_inset = inset_axes(ax3, width="40%", height="40%", loc='upper right')
            ax3_inset.plot(speed_times, fish_displacements, color='gray', linewidth=1)
            ax3_inset.axhline(y=MIN_ESCAPE_DISPLACEMENT, color='red', linestyle=':', alpha=0.5)
            ax3_inset.set_xlabel('Time (s)', fontsize=8)
            ax3_inset.set_ylabel('Displacement (px)', fontsize=8)
            ax3_inset.tick_params(labelsize=7)
            ax3_inset.grid(True, alpha=0.3)
            
            ax3.set_xlabel('Time (s)', fontsize=11)
            ax3.set_ylabel('Speed (px/s)', fontsize=11)
            ax3.set_title('Fish Speed Profile', fontsize=13, fontweight='bold')
            ax3.legend(loc='upper left', fontsize=9)
            ax3.grid(True, alpha=0.3)
        
        # 4. Metrics summary
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        metrics = self.calculate_trial_metrics(trial)
        
        # Create formatted text with better layout
        text_lines = [
            f"{'Trial Summary':^25s}",
            f"{'─' * 25}",
            f"Phase: {trial['phase'].capitalize()}",
            f"Duration: {metrics.duration_s:.2f} s",
            f"Detection: {metrics.detection_rate:.1%}",
            "",
            "Distance Metrics:",
            f"  Mean: {metrics.mean_distance_px:.0f} px",
            f"  Min: {metrics.min_distance_px:.0f} px",
            f"  Max: {metrics.max_distance_px:.0f} px",
            f"  Change: {metrics.distance_change_px:+.0f} px",
            "",
            "Fish Behavior:",
            f"  Mean speed: {metrics.fish_mean_speed_px_per_s:.0f} px/s",
            f"  Max speed: {metrics.fish_max_speed_px_per_s:.0f} px/s",
            f"  Escape: {'✓' if metrics.escape_detected else '✗'}",
        ]
        
        if metrics.escape_detected:
            text_lines.append(f"  → Latency: {metrics.escape_latency_s:.3f} s")
            text_lines.append(f"  → Speed: {metrics.escape_speed_px_per_s:.0f} px/s")
        
        text_lines.extend([
            "",
            "Approach Events:",
            f"  Count: {metrics.approach_events}",
        ])
        
        if metrics.min_approach_distance_px:
            text_lines.append(f"  Min dist: {metrics.min_approach_distance_px:.0f} px")
        
        text_lines.append(f"  Time close: {metrics.time_below_threshold_s:.1f} s")
        
        text = '\n'.join(text_lines)
        
        # Add background box for text
        bbox = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                             boxstyle="round,pad=0.05",
                             transform=ax4.transAxes,
                             facecolor='lightgray', alpha=0.3)
        ax4.add_patch(bbox)
        
        ax4.text(0.5, 0.5, text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='center',
                horizontalalignment='center', fontfamily='monospace')
        
        plt.suptitle(f'Trial {trial_num} Detailed Analysis', fontsize=15, fontweight='bold', y=0.98)
        plt.show()
    
    def plot_phase_comparison(self, df: pd.DataFrame):
        """Plot comparison of metrics across phases with improved visualization."""
        # Set style
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        
        phase_colors = {'pre': '#8B9DC3', 'training': '#F4B942', 'post': '#C73E1D'}
        phase_order = ['pre', 'training', 'post']
        
        # 1. Mean distance by phase
        ax = axes[0, 0]
        sns.boxplot(data=df, x='phase', y='mean_distance_px', ax=ax, 
                   order=phase_order, palette=phase_colors)
        ax.set_title('Mean Fish-Chaser Distance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance (pixels)')
        ax.set_xlabel('')
        
        # 2. Fish speed by phase
        ax = axes[0, 1]
        sns.violinplot(data=df, x='phase', y='fish_mean_speed_px_per_s', ax=ax,
                      order=phase_order, palette=phase_colors)
        ax.set_title('Fish Swimming Speed', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speed (px/s)')
        ax.set_xlabel('')
        
        # 3. Escape probability by phase
        ax = axes[0, 2]
        escape_rates = df.groupby('phase')['escape_detected'].agg(['mean', 'sem'])
        escape_rates = escape_rates.reindex(phase_order)
        bars = ax.bar(escape_rates.index, escape_rates['mean'], 
                     yerr=escape_rates['sem'], capsize=5,
                     color=[phase_colors[p] for p in escape_rates.index])
        ax.set_title('Escape Response Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('Proportion of Trials')
        ax.set_ylim([0, 1])
        ax.set_xlabel('')
        
        # Add percentage labels on bars
        for bar, val in zip(bars, escape_rates['mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.1%}', ha='center', va='bottom')
        
        # 4. Escape latency by phase
        ax = axes[1, 0]
        escape_df = df[df['escape_detected']].copy()
        if len(escape_df) > 0:
            sns.boxplot(data=escape_df, x='phase', y='escape_latency_s', ax=ax,
                       order=phase_order, palette=phase_colors)
        ax.set_title('Escape Response Latency', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latency (s)')
        ax.set_xlabel('')
        
        # 5. Min distance by phase
        ax = axes[1, 1]
        sns.boxplot(data=df, x='phase', y='min_distance_px', ax=ax,
                   order=phase_order, palette=phase_colors)
        ax.set_title('Minimum Approach Distance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance (pixels)')
        ax.set_xlabel('')
        
        # 6. Distance change by phase
        ax = axes[1, 2]
        sns.boxplot(data=df, x='phase', y='distance_change_px', ax=ax,
                   order=phase_order, palette=phase_colors)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_title('Distance Change (Final - Initial)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Change (pixels)')
        ax.set_xlabel('')
        
        # 7. Time below threshold
        ax = axes[2, 0]
        sns.boxplot(data=df, x='phase', y='time_below_threshold_s', ax=ax,
                   order=phase_order, palette=phase_colors)
        ax.set_title('Time Within Approach Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (s)')
        ax.set_xlabel('Phase')
        
        # 8. Relative velocity
        ax = axes[2, 1]
        sns.boxplot(data=df, x='phase', y='mean_relative_velocity_px_per_s', ax=ax,
                   order=phase_order, palette=phase_colors)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_title('Relative Velocity (Negative = Approaching)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Velocity (px/s)')
        ax.set_xlabel('Phase')
        
        # 9. Detection rate by phase
        ax = axes[2, 2]
        sns.boxplot(data=df, x='phase', y='detection_rate', ax=ax,
                   order=phase_order, palette=phase_colors)
        ax.set_title('Detection Quality', fontsize=12, fontweight='bold')
        ax.set_ylabel('Detection Rate')
        ax.set_xlabel('Phase')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.suptitle('Behavioral Metrics Across Training Phases', 
                    fontsize=15, fontweight='bold', y=1.02)
        plt.show()
        
        # Print statistical summary
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON")
        print("="*60)
        
        for phase in phase_order:
            phase_df = df[df['phase'] == phase]
            if len(phase_df) > 0:
                print(f"\n{phase.upper()} Phase (n={len(phase_df)}):")
                print(f"  Distance: {phase_df['mean_distance_px'].mean():.0f} ± {phase_df['mean_distance_px'].std():.0f} px")
                print(f"  Speed: {phase_df['fish_mean_speed_px_per_s'].mean():.0f} ± {phase_df['fish_mean_speed_px_per_s'].std():.0f} px/s")
                print(f"  Escapes: {phase_df['escape_detected'].sum()}/{len(phase_df)} ({phase_df['escape_detected'].mean():.1%})")
                
                escape_trials = phase_df[phase_df['escape_detected']]
                if len(escape_trials) > 0:
                    print(f"  Escape latency: {escape_trials['escape_latency_s'].mean():.3f} ± {escape_trials['escape_latency_s'].std():.3f} s")
    
    def save_results(self, df: pd.DataFrame, output_path: Optional[str] = None):
        """Save analysis results to CSV and JSON summary."""
        if output_path is None:
            output_path = self.h5_path.with_suffix('.trial_metrics.csv')
        
        df.to_csv(output_path, index=False)
        self.logger.info(f"\nResults saved to: {output_path}")
        
        # Also save summary statistics
        summary_path = Path(output_path).with_suffix('.summary.json')
        
        summary = {
            'n_trials': len(df),
            'phases': {},
            'analysis_date': datetime.now().isoformat(),
            'parameters': {
                'escape_threshold_px_per_s': self.escape_threshold,
                'approach_threshold_px': self.approach_threshold,
                'texture_to_camera_scale': self.texture_to_camera_scale,
                'camera_fps': self.camera_fps,
                'stimulus_hz': self.stimulus_hz
            },
            'files': {
                'zarr': str(self.zarr_path),
                'h5': str(self.h5_path)
            }
        }
        
        for phase in ['pre', 'training', 'post']:
            phase_df = df[df['phase'] == phase]
            if len(phase_df) > 0:
                summary['phases'][phase] = {
                    'n_trials': int(len(phase_df)),
                    'mean_distance_px': float(phase_df['mean_distance_px'].mean()),
                    'std_distance_px': float(phase_df['mean_distance_px'].std()),
                    'mean_speed_px_per_s': float(phase_df['fish_mean_speed_px_per_s'].mean()),
                    'std_speed_px_per_s': float(phase_df['fish_mean_speed_px_per_s'].std()),
                    'escape_rate': float(phase_df['escape_detected'].mean()),
                    'n_escapes': int(phase_df['escape_detected'].sum()),
                    'detection_rate': float(phase_df['detection_rate'].mean())
                }
                
                # Add escape latency stats if there were escapes
                escape_trials = phase_df[phase_df['escape_detected']]
                if len(escape_trials) > 0:
                    summary['phases'][phase]['mean_escape_latency_s'] = float(escape_trials['escape_latency_s'].mean())
                    summary['phases'][phase]['std_escape_latency_s'] = float(escape_trials['escape_latency_s'].std())
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved to: {summary_path}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Analyze fish responses to individual chase trials (v2.0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This analyzer extracts behavioral metrics for each chase trial including:
- Fish-chaser distances with proper coordinate transformation
- Movement speeds and escape responses  
- Comparison across training phases (pre/training/post)
- Proper frame alignment for 120Hz/60FPS systems

Examples:
  %(prog)s detections.zarr analysis.h5
  %(prog)s detections.zarr analysis.h5 --plot-trial 5
  %(prog)s detections.zarr analysis.h5 --plot-comparison --save
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file with YOLO detections')
    parser.add_argument('h5_path', help='Path to H5 analysis file')
    parser.add_argument('--plot-trial', type=int, help='Plot detailed view of specific trial')
    parser.add_argument('--plot-comparison', action='store_true', 
                       help='Plot comparison across phases')
    parser.add_argument('--save', action='store_true', 
                       help='Save results to CSV')
    parser.add_argument('--output', help='Output path for CSV (default: auto-generated)')
    parser.add_argument('--escape-threshold', type=float, default=500,
                       help='Speed threshold for escape detection (px/s)')
    parser.add_argument('--approach-threshold', type=float, default=200,
                       help='Distance threshold for approach detection (px)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TrialByTrialAnalyzer(
        zarr_path=args.zarr_path,
        h5_path=args.h5_path,
        escape_threshold_px_per_s=args.escape_threshold,
        approach_threshold_px=args.approach_threshold,
        verbose=not args.quiet
    )
    
    # Run analysis
    df = analyzer.analyze_all_trials()
    
    # Print summary
    if not args.quiet:
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        for phase in ['pre', 'training', 'post']:
            phase_df = df[df['phase'] == phase]
            if len(phase_df) > 0:
                print(f"\n{phase.upper()} Phase ({len(phase_df)} trials):")
                print(f"  Mean distance: {phase_df['mean_distance_px'].mean():.0f} px")
                print(f"  Mean speed: {phase_df['fish_mean_speed_px_per_s'].mean():.0f} px/s")
                print(f"  Escape rate: {phase_df['escape_detected'].mean():.1%}")
                print(f"  Detection rate: {phase_df['detection_rate'].mean():.1%}")
    
    # Plot if requested
    if args.plot_trial:
        analyzer.plot_trial_summary(args.plot_trial)
    
    if args.plot_comparison:
        analyzer.plot_phase_comparison(df)
    
    # Save if requested
    if args.save:
        analyzer.save_results(df, args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())