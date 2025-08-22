#!/usr/bin/env python3
"""
Chaser-Fish Distance Analyzer (v2.1)

Analyzes distances between fish (from YOLO detections in zarr) and chaser 
(from H5 files) with proper coordinate transformation and comprehensive plotting.

IMPORTANT: 
- Chaser positions are in TEXTURE space (358×358)
- Fish positions are in CAMERA space (4512×4512)
- Uses simple scaling (×12.604) to transform texture→camera
- Handles 120Hz stimulus / 60FPS camera frame rate mismatch
- Works with fixed PROTOCOL_START events

This module handles:
- Frame alignment between 60 FPS video and 120 Hz stimulus
- Coordinate transformation from texture to camera space
- Distance calculations in camera pixel coordinates
- Integration with interpolated detection data
- Comprehensive metric calculation, visualization, and storage
"""

import zarr
import h5py
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle
import seaborn as sns
import warnings

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Event type mappings for reference
EXPERIMENT_EVENT_TYPE = {
    0: "PROTOCOL_START",
    4: "PROTOCOL_FINISH", 
    24: "CHASER_PRE_PERIOD_START",
    25: "CHASER_TRAINING_START",
    26: "CHASER_POST_PERIOD_START",
    27: "CHASER_CHASE_SEQUENCE_START",
    28: "CHASER_CHASE_SEQUENCE_END",
}

# Try to import coordinate transform module
try:
    from coordinate_transform_module import CoordinateSystem
    HAS_COORD_MODULE = True
except ImportError:
    HAS_COORD_MODULE = False
    print("Warning: coordinate_transform_module not found. Using default scaling.")


@dataclass
class AnalysisMetadata:
    """Metadata for chaser-fish analysis."""
    created_at: str
    h5_source: str
    h5_checksum: str
    zarr_source: str
    interpolation_run: str
    camera_id: str
    coordinate_system: str
    video_dimensions: List[int]
    texture_dimensions: List[int]
    texture_to_camera_scale: List[float]
    frame_rate_video: float
    frame_rate_stimulus: float
    frame_rate_ratio: float
    total_frames_analyzed: int
    valid_frames: int
    version: str = "2.1.0"  # Updated for plotting features


class ChaserFishDistanceAnalyzer:
    """
    Analyzes fish-chaser distances with proper coordinate transformation.
    
    This class handles:
    - Loading and aligning data from zarr (YOLO detections) and H5 (chaser states)
    - Coordinate transformation from texture to camera space
    - Distance and behavioral metric calculations
    - Comprehensive visualization and reporting
    - Saving results back to the zarr file
    """
    
    def __init__(self, 
                 zarr_path: str, 
                 h5_path: str,
                 interpolation_run: Optional[str] = None,
                 use_texture_scaling: bool = True,
                 verbose: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            zarr_path: Path to YOLO detection zarr file
            h5_path: Path to H5 file with chaser states
            interpolation_run: Specific interpolation run to use (default: latest)
            use_texture_scaling: If True, use texture→camera scaling (recommended)
            verbose: Enable verbose logging
        """
        self.zarr_path = Path(zarr_path)
        self.h5_path = Path(h5_path)
        self.use_texture_scaling = use_texture_scaling
        self.verbose = verbose
        
        # Setup logging
        self._setup_logging()
        
        # Load data sources
        self.logger.info(f"Initializing analyzer v2.1 (with plotting)")
        self.logger.info(f"  Zarr: {zarr_path}")
        self.logger.info(f"  H5: {h5_path}")
        
        self.zarr_root = zarr.open(str(zarr_path), mode='r+')
        
        # Determine interpolation run
        if interpolation_run is None:
            if 'interpolation_runs' in self.zarr_root:
                interpolation_run = self.zarr_root['interpolation_runs'].attrs.get('latest')
                self.logger.info(f"  Using latest interpolation run: {interpolation_run}")
        self.interpolation_run = interpolation_run
        
        # Data containers
        self.camera_id = None
        self.video_width = None
        self.video_height = None
        self.texture_width = 358  # Standard texture dimensions
        self.texture_height = 358
        self.texture_to_camera_scale_x = None
        self.texture_to_camera_scale_y = None
        self.camera_to_stimulus = {}
        self.stimulus_to_camera = {}
        self.chaser_states = None
        self.chaser_by_stimulus = {}
        self.zarr_to_stimulus = None
        self.events = None
        self.metadata = None
        self.coord_sys = None
        
        # Initialize coordinate system
        self._initialize_coordinate_system()
        
        # Load all necessary data
        self._load_video_info()
        self._load_h5_metadata()
        self._load_events()
        self._create_frame_alignment()
        # Load smoothed fish speed if available
        self.smoothed_speed_px_s, self.speed_window = self._load_smoothed_speed()

    
    def _setup_logging(self):
        """Configure logging."""
        level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_coordinate_system(self):
        """Initialize coordinate transformation system."""
        self.logger.info("Initializing coordinate system...")
        
        if HAS_COORD_MODULE:
            try:
                self.coord_sys = CoordinateSystem(self.h5_path, verbose=self.verbose)
                self.logger.info("  ✅ Coordinate system module loaded")
                
                # Get texture dimensions from module
                self.texture_width = self.coord_sys.texture_dims[0]
                self.texture_height = self.coord_sys.texture_dims[1]
                
                # Get transformation parameters
                if 'texture_to_camera' in self.coord_sys.transforms:
                    transform = self.coord_sys.transforms['texture_to_camera']
                    self.texture_to_camera_scale_x = transform.scale_x
                    self.texture_to_camera_scale_y = transform.scale_y
                    self.logger.info(f"  Texture dims: {self.texture_width}×{self.texture_height}")
                    self.logger.info(f"  Scale factors: x={self.texture_to_camera_scale_x:.3f}, y={self.texture_to_camera_scale_y:.3f}")
                else:
                    self._set_default_scaling()
            except Exception as e:
                self.logger.warning(f"  Could not initialize coordinate system: {e}")
                self._set_default_scaling()
        else:
            self._set_default_scaling()
    
    def _set_default_scaling(self):
        """Set default texture to camera scaling factors."""
        # Default: assume standard dimensions
        self.texture_width = 358
        self.texture_height = 358
        camera_width = 4512  # Standard camera resolution
        camera_height = 4512
        
        self.texture_to_camera_scale_x = camera_width / self.texture_width  # 12.604
        self.texture_to_camera_scale_y = camera_height / self.texture_height  # 12.604
        
        self.logger.info("  Using default texture→camera scaling")
        self.logger.info(f"  Texture: {self.texture_width}×{self.texture_height}")
        self.logger.info(f"  Camera: {camera_width}×{camera_height}")
        self.logger.info(f"  Scale: {self.texture_to_camera_scale_x:.3f}")
    
    def _load_video_info(self):
        """Load video dimensions from zarr attributes."""
        self.video_width = self.zarr_root.attrs.get('width', 4512)
        self.video_height = self.zarr_root.attrs.get('height', 4512)
        self.fps_video = self.zarr_root.attrs.get('fps', 60.0)
        
        # Update scaling if we have actual video dimensions
        if not self.coord_sys:  # Only update if using defaults
            self.texture_to_camera_scale_x = self.video_width / self.texture_width
            self.texture_to_camera_scale_y = self.video_height / self.texture_height
        
        self.logger.info(f"  Video: {self.video_width}×{self.video_height} @ {self.fps_video} FPS")
    
    def _load_h5_metadata(self):
        """Load frame metadata and chaser states from H5."""
        self.logger.info("Loading H5 metadata and chaser states...")
        
        with h5py.File(self.h5_path, 'r') as h5f:
            # Try to get camera ID from calibration
            if '/calibration_snapshot' in h5f:
                calib = h5f['/calibration_snapshot']
                camera_ids = [k for k in calib.keys() if isinstance(calib[k], h5py.Group)]
                if camera_ids:
                    self.camera_id = camera_ids[0]
                    self.logger.info(f"  Camera ID: {self.camera_id}")
            
            # Load frame metadata
            if '/video_metadata/frame_metadata' not in h5f:
                raise ValueError("No frame_metadata found in H5 file")
            
            frame_meta = h5f['/video_metadata/frame_metadata'][:]
            self.logger.info(f"  Loaded {len(frame_meta)} frame metadata records")
            
            # Build bidirectional mapping (handling 120Hz/60FPS mismatch)
            self.camera_to_stimulus = {}
            self.stimulus_to_camera = {}
            
            for record in frame_meta:
                cam_frame = int(record['triggering_camera_frame_id'])
                stim_frame = int(record['stimulus_frame_num'])
                
                # Handle multiple stimulus frames per camera frame (120Hz vs 60FPS)
                if cam_frame not in self.camera_to_stimulus:
                    self.camera_to_stimulus[cam_frame] = []
                self.camera_to_stimulus[cam_frame].append(stim_frame)
                
                # Keep track of reverse mapping
                self.stimulus_to_camera[stim_frame] = cam_frame
            
            # Estimate frame rate ratio
            avg_stim_per_cam = np.mean([len(stims) for stims in self.camera_to_stimulus.values()])
            self.fps_stimulus = self.fps_video * avg_stim_per_cam
            self.logger.info(f"  Stimulus rate: {self.fps_stimulus:.1f} Hz")
            self.logger.info(f"  Ratio: {avg_stim_per_cam:.2f} stimulus frames per camera frame")
            
            # Load chaser states
            if '/tracking_data/chaser_states' not in h5f:
                raise ValueError("No chaser_states found in H5 file")
            
            self.chaser_states = h5f['/tracking_data/chaser_states'][:]
            self.logger.info(f"  Loaded {len(self.chaser_states)} chaser state records")
            
            # Analyze chaser positions to confirm texture space
            chaser_x = self.chaser_states['chaser_pos_x']
            chaser_y = self.chaser_states['chaser_pos_y']
            self.logger.info(f"  Chaser X range: {chaser_x.min():.1f} to {chaser_x.max():.1f}")
            self.logger.info(f"  Chaser Y range: {chaser_y.min():.1f} to {chaser_y.max():.1f}")
            self.logger.info(f"  Chaser mean position: ({chaser_x.mean():.1f}, {chaser_y.mean():.1f})")
            
            # Confirm texture space (chaser at ~179,179 means 358×358 texture)
            if abs(chaser_x.mean() - 179) < 10 and abs(chaser_y.mean() - 179) < 10:
                self.logger.info("  ✅ Confirmed: Chaser in 358×358 texture space")
            
            # Create stimulus frame index for fast lookup
            self.chaser_by_stimulus = {}
            for state in self.chaser_states:
                stim_frame = int(state['stimulus_frame_num'])
                self.chaser_by_stimulus[stim_frame] = state
    
    def _load_events(self):
        """Load experimental events from H5."""
        self.logger.info("Loading experimental events...")
        
        with h5py.File(self.h5_path, 'r') as h5f:
            if '/events' in h5f:
                self.events = h5f['/events'][:]
                self.logger.info(f"  Loaded {len(self.events)} events")
                
                # Check for fixed PROTOCOL_START
                if 'protocol_start_fixed' in h5f['/events'].attrs:
                    self.logger.info("  ✅ PROTOCOL_START has been fixed")
                
                # Parse important events
                self.chase_events = []
                for event in self.events:
                    if 'event_type_id' in event.dtype.names:
                        event_type = event['event_type_id']
                        if event_type == 27:  # CHASER_CHASE_SEQUENCE_START
                            camera_frame = event['camera_frame_id'] if 'camera_frame_id' in event.dtype.names else -1
                            stimulus_frame = event['stimulus_frame_num'] if 'stimulus_frame_num' in event.dtype.names else -1
                            self.chase_events.append({
                                'type': 'start',
                                'camera_frame': int(camera_frame),
                                'stimulus_frame': int(stimulus_frame)
                            })
                        elif event_type == 28:  # CHASER_CHASE_SEQUENCE_END
                            camera_frame = event['camera_frame_id'] if 'camera_frame_id' in event.dtype.names else -1
                            stimulus_frame = event['stimulus_frame_num'] if 'stimulus_frame_num' in event.dtype.names else -1
                            self.chase_events.append({
                                'type': 'end',
                                'camera_frame': int(camera_frame),
                                'stimulus_frame': int(stimulus_frame)
                            })
                
                self.logger.info(f"  Found {len([e for e in self.chase_events if e['type'] == 'start'])} chase sequences")
    
    def _create_frame_alignment(self):
        """Create alignment between zarr indices and stimulus frames."""
        self.logger.info("Creating frame alignment...")
        
        # Get total frames from zarr
        n_frames = len(self.zarr_root['n_detections'])
        self.logger.info(f"  Total zarr frames: {n_frames}")
        
        # Determine camera frame offset
        min_camera_frame = min(self.camera_to_stimulus.keys()) if self.camera_to_stimulus else 0
        max_camera_frame = max(self.camera_to_stimulus.keys()) if self.camera_to_stimulus else 0
        
        self.logger.info(f"  Camera frame range in H5: {min_camera_frame} to {max_camera_frame}")
        
        # Build alignment array
        self.zarr_to_stimulus = np.full(n_frames, -1, dtype=np.int32)
        aligned_count = 0
        
        for zarr_idx in range(n_frames):
            # Map zarr index to camera frame ID
            camera_frame_id = zarr_idx + min_camera_frame
            
            if camera_frame_id in self.camera_to_stimulus:
                stimulus_frames = self.camera_to_stimulus[camera_frame_id]
                # Choose middle stimulus frame for best temporal alignment
                middle_idx = len(stimulus_frames) // 2
                self.zarr_to_stimulus[zarr_idx] = stimulus_frames[middle_idx]
                aligned_count += 1
        
        self.logger.info(f"  ✅ Aligned {aligned_count}/{n_frames} frames")
        
        if aligned_count == 0:
            self.logger.warning("  ⚠️ No frames could be aligned - check frame numbering!")
    
    def _load_smoothed_speed(self, preferred_window=(10, 20, 30)):
        """Return (smoothed_speed_px_s, chosen_window) or (None, None) if absent."""
        if 'speed_metrics' not in self.zarr_root:
            self.logger.info("  No speed_metrics group in zarr; skipping smoothed speed.")
            return None, None

        sm = self.zarr_root['speed_metrics']
        if 'batch_processing' not in sm or 'latest' not in sm['batch_processing'].attrs:
            self.logger.info("  No latest speed batch found; skipping smoothed speed.")
            return None, None

        batch_name = sm['batch_processing'].attrs['latest']
        batch = sm['batch_processing'][batch_name]

        # try preferred windows first
        for w in preferred_window:
            key = f'window_{w}'
            if key in batch and 'smoothed_speed_px_s' in batch[key]:
                arr = batch[key]['smoothed_speed_px_s'][:].astype(np.float32)
                self.logger.info(f"  Loaded smoothed speed (window={w} frames) from {batch_name}")
                return arr, w

        # fallback: smallest available window
        windows = sorted(int(k.split('_')[1]) for k in batch.group_keys() if k.startswith('window_'))
        if windows:
            w = windows[0]
            arr = batch[f'window_{w}']['smoothed_speed_px_s'][:].astype(np.float32)
            self.logger.info(f"  Loaded smoothed speed (window={w} frames) from {batch_name}")
            return arr, w

        self.logger.info("  No smoothed speed arrays present.")
        return None, None
    
    def _detect_escapes(self, dist_px, rel_vel_px_s, speed_px_s, fps):
        """
        Returns a dict with:
        escape_onset [bool], escape_mask [bool], triggered_by_approach [bool],
        near_and_approach [bool], near_and_approach_lag [bool],
        thresholds {D_thresh, S_escape, V_close, tau_close, tau_escape, lag_frames},
        latency_ms (array for triggered escapes)
        """
        import numpy as np

        n = len(dist_px)
        valid_speed = speed_px_s[~np.isnan(speed_px_s)] if speed_px_s is not None else np.array([])
        valid_dist  = dist_px[~np.isnan(dist_px)]
        valid_vel   = rel_vel_px_s[~np.isnan(rel_vel_px_s)]

        if speed_px_s is None or valid_speed.size < 50 or valid_dist.size < 50:
            # Not enough info
            return {
                "escape_onset": np.zeros(n, dtype=bool),
                "escape_mask": np.zeros(n, dtype=bool),
                "triggered_by_approach": np.zeros(n, dtype=bool),
                "near_and_approach": np.zeros(n, dtype=bool),
                "near_and_approach_lag": np.zeros(n, dtype=bool),
                "thresholds": {"D_thresh": np.nan, "S_escape": np.nan,
                            "V_close": np.nan, "tau_close": 0, "tau_escape": 0,
                            "lag_frames": 0},
                "latency_ms": np.array([], dtype=float),
            }

        # --- thresholds (robust) ---
        baseline = np.nanmedian(valid_speed)
        mad = np.nanmedian(np.abs(valid_speed - baseline))
        robust_sd = 1.4826 * mad if mad > 0 else np.nanstd(valid_speed)
        S_escape = baseline + 3.0 * robust_sd

        D_thresh = np.nanpercentile(valid_dist, 20)  # “near” = closest 20%

        # chaser “approach” = distance decreasing fast enough
        V_close = -100.0  # px/s default; tune as needed or expose as arg
        tau_close  = max(2, int(round(0.05 * fps)))  # ~50 ms persistence
        tau_escape = max(3, int(round(0.08 * fps)))  # ~80 ms persistence
        lag_frames = max(1, int(round(0.10 * fps)))  # allow 100 ms from approach to escape

        # --- primitive masks ---
        near        = dist_px < D_thresh
        approaching = rel_vel_px_s < V_close
        fast_fish   = speed_px_s > S_escape

        near = np.nan_to_num(near)
        approaching = np.nan_to_num(approaching)
        fast_fish = np.nan_to_num(fast_fish)

        # --- persistence helper ---
        def persist(mask, k):
            if k <= 1: return mask.astype(bool)
            out = np.zeros_like(mask, dtype=bool)
            run = 0
            for i, m in enumerate(mask.astype(bool)):
                run = (run + 1) if m else 0
                if run >= k:
                    out[i] = True
            return out

        near_persist        = persist(near, 1)               # distance already fairly stable
        approaching_persist = persist(approaching, tau_close)
        escape_persist      = persist(fast_fish,   tau_escape)

        # escape onset frames (first frame of a sustained run)
        onset = escape_persist & (~np.roll(escape_persist, 1))
        onset[0] = escape_persist[0]

        # near & approach (with short look‑back lag)
        near_and_approach = near_persist & approaching_persist
        near_and_approach_lag = near_and_approach.copy()
        for i in range(1, lag_frames + 1):
            near_and_approach_lag[i:] |= near_and_approach[:-i]

        # classify which onsets were “triggered” by near & approach in the last lag
        triggered = onset & near_and_approach_lag

        # latency from last near&approach to escape onset (ms)
        latencies = []
        idx_on = np.where(onset)[0]
        naa = np.where(near_and_approach)[0]
        for t in idx_on:
            prev = naa[naa <= t]
            if prev.size:
                dt = (t - prev[-1]) / fps * 1000.0
                if 0 <= dt <= (lag_frames / fps * 1000.0) + 500.0:  # bound outliers
                    latencies.append(dt)
        latency_ms = np.array(latencies, dtype=float)

        return {
            "escape_onset": onset,
            "escape_mask": escape_persist,
            "triggered_by_approach": triggered,
            "near_and_approach": near_and_approach,
            "near_and_approach_lag": near_and_approach_lag,
            "thresholds": {
                "D_thresh": float(D_thresh), "S_escape": float(S_escape),
                "V_close": float(V_close), "tau_close": int(tau_close),
                "tau_escape": int(tau_escape), "lag_frames": int(lag_frames),
            },
            "latency_ms": latency_ms,
        }
            
    def transform_texture_to_camera(self, texture_x: float, texture_y: float) -> Tuple[float, float]:
        """
        Transform from texture space (358×358) to camera space (4512×4512).
        
        Args:
            texture_x, texture_y: Position in texture space
            
        Returns:
            camera_x, camera_y: Position in camera space
        """
        if self.coord_sys and HAS_COORD_MODULE:
            # Use coordinate system module if available
            cam_x, cam_y = self.coord_sys.transform_coordinates(
                np.array([texture_x]), 
                np.array([texture_y]),
                from_space='texture',
                to_space='camera'
            )
            return cam_x[0], cam_y[0]
        else:
            # Use simple scaling
            camera_x = texture_x * self.texture_to_camera_scale_x
            camera_y = texture_y * self.texture_to_camera_scale_y
            return camera_x, camera_y
    
    def calculate_distances(self) -> Dict[str, np.ndarray]:
        """
        Calculate fish-chaser distances with coordinate transformation.
        
        Returns:
            Dictionary with distance metrics and position data
        """
        self.logger.info("Calculating fish-chaser distances...")
        
        # Get fish detection data
        if self.interpolation_run:
            self.logger.info(f"  Using interpolation run: {self.interpolation_run}")
            fish_data = self.zarr_root[f'interpolation_runs/{self.interpolation_run}']
            bboxes = fish_data['bboxes'][:]
            n_detections = fish_data['n_detections'][:]
            mask = fish_data['interpolation_mask'][:]
        else:
            self.logger.info("  Using original detections")
            bboxes = self.zarr_root['bboxes'][:]
            n_detections = self.zarr_root['n_detections'][:]
            mask = np.zeros(len(n_detections), dtype=bool)
        
        n_frames = len(n_detections)
        
        # Initialize output arrays
        distances_pixels = np.full(n_frames, np.nan)
        relative_velocities = np.full(n_frames, np.nan)
        pursuit_angles = np.full(n_frames, np.nan)
        
        # Position arrays
        fish_positions_camera = np.full((n_frames, 2), np.nan)
        chaser_positions_camera = np.full((n_frames, 2), np.nan)
        chaser_positions_texture = np.full((n_frames, 2), np.nan)
        
        # Process each frame
        valid_count = 0
        
        for frame_idx in range(n_frames):
            if n_detections[frame_idx] == 0:
                continue  # No fish detected
            
            # Get fish position in camera coordinates (center of bbox)
            fish_bbox = bboxes[frame_idx, 0]  # First detection
            fish_x_camera = (fish_bbox[0] + fish_bbox[2]) / 2
            fish_y_camera = (fish_bbox[1] + fish_bbox[3]) / 2
            
            # Store fish camera position
            fish_positions_camera[frame_idx] = [fish_x_camera, fish_y_camera]
            
            # Get corresponding stimulus frame
            stim_frame = self.zarr_to_stimulus[frame_idx]
            if stim_frame < 0 or stim_frame not in self.chaser_by_stimulus:
                continue  # No chaser data for this frame
            
            # Get chaser state (in TEXTURE coordinates!)
            chaser = self.chaser_by_stimulus[stim_frame]
            
            # Extract chaser position in TEXTURE space
            chaser_x_texture = float(chaser['chaser_pos_x'])
            chaser_y_texture = float(chaser['chaser_pos_y'])
            
            # Store chaser texture position
            chaser_positions_texture[frame_idx] = [chaser_x_texture, chaser_y_texture]
            
            # Transform chaser from TEXTURE to CAMERA space
            chaser_x_camera, chaser_y_camera = self.transform_texture_to_camera(
                chaser_x_texture, chaser_y_texture
            )
            chaser_positions_camera[frame_idx] = [chaser_x_camera, chaser_y_camera]
            
            # Calculate distance in CAMERA PIXELS
            dist_pixels = np.sqrt((fish_x_camera - chaser_x_camera)**2 + 
                                 (fish_y_camera - chaser_y_camera)**2)
            
            distances_pixels[frame_idx] = dist_pixels
            valid_count += 1
        
        self.logger.info(f"  ✅ Calculated distances for {valid_count}/{n_frames} frames")
        
        # Calculate velocities (rate of change of distance)
        valid_distances = ~np.isnan(distances_pixels)
        if np.sum(valid_distances) > 1:
            # Use gradient for smoother velocity calculation
            relative_velocities[valid_distances] = np.gradient(distances_pixels[valid_distances]) * self.fps_video
        
        # Calculate summary statistics
        valid_mask = ~np.isnan(distances_pixels)
        if np.sum(valid_mask) > 0:
            mean_dist = np.nanmean(distances_pixels)
            min_dist = np.nanmin(distances_pixels)
            max_dist = np.nanmax(distances_pixels)
            
            self.logger.info(f"  Distance statistics (pixels):")
            self.logger.info(f"    Mean: {mean_dist:.1f}")
            self.logger.info(f"    Min: {min_dist:.1f}")
            self.logger.info(f"    Max: {max_dist:.1f}")
        
        results = {
            'fish_chaser_distance_pixels': distances_pixels,
            'relative_velocity': relative_velocities,
            'pursuit_angle': pursuit_angles,
            'fish_position_camera': fish_positions_camera,
            'chaser_position_camera': chaser_positions_camera,
            'chaser_position_texture': chaser_positions_texture,
            'fish_interpolated': mask,
            'valid_frames': valid_mask
        }
        if self.smoothed_speed_px_s is not None:
            results['smoothed_speed_px_s'] = self.smoothed_speed_px_s

        return results
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Generate comprehensive visualization of analysis results.
        
        Args:
            save_path: Optional path to save the figure
        """
        self.logger.info("Generating analysis plots...")
        
        # Calculate metrics if not already done
        metrics = self.calculate_distances()
        if metrics is None:
            raise RuntimeError("calculate_distances() returned None")

        # Try to detect escapes if we have smoothed speed
        esc = None
        if 'smoothed_speed_px_s' in metrics:
            esc = self._detect_escapes(
                metrics['fish_chaser_distance_pixels'],
                metrics['relative_velocity'],
                metrics['smoothed_speed_px_s'],
                self.fps_video
            )

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Get data
        distances = metrics['fish_chaser_distance_pixels']
        velocities = metrics['relative_velocity']
        fish_pos = metrics['fish_position_camera']
        chaser_pos_cam = metrics['chaser_position_camera']
        chaser_pos_tex = metrics['chaser_position_texture']
        valid_frames = metrics['valid_frames']
        interpolated = metrics['fish_interpolated']
        
        # Time axis (convert frames to seconds)
        time_seconds = np.arange(len(distances)) / self.fps_video
        
        # 1. Distance over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time_seconds[valid_frames], distances[valid_frames], 
                'b-', linewidth=1, alpha=0.7, label='Measured')
        ax1.plot(time_seconds[interpolated], distances[interpolated], 
                'r.', markersize=2, alpha=0.5, label='Interpolated')
        
        # Mark chase events if available
        if hasattr(self, 'chase_events'):
            for event in self.chase_events:
                if event['type'] == 'start':
                    ax1.axvline(x=event['camera_frame']/self.fps_video, 
                              color='g', linestyle='--', alpha=0.5)
                elif event['type'] == 'end':
                    ax1.axvline(x=event['camera_frame']/self.fps_video, 
                              color='r', linestyle='--', alpha=0.5)
        
        # Mark escape onsets and approach windows
        if esc is not None:
            t = time_seconds

            # Mark escape onsets with vertical crimson lines
            onset_idx = np.where(esc['escape_onset'])[0]
            if onset_idx.size:
                ax1.vlines(
                    t[onset_idx],
                    ymin=np.nanmin(distances),
                    ymax=np.nanmax(distances),
                    colors='crimson',
                    alpha=0.4,
                    linewidth=1.0,
                    label='Escape onset'
                )

            # Shade time spans where fish was near & chaser approaching (with lag)
            naa = esc['near_and_approach_lag']
            in_block = False
            start = 0
            for i, flag in enumerate(naa):
                if flag and not in_block:
                    in_block = True
                    start = i
                if in_block and (not flag or i == len(naa)-1):
                    end = i if not flag else i + 1
                    ax1.axvspan(t[start], t[end - 1], color='lime', alpha=0.08)
                    in_block = False

        
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Distance (pixels)', fontsize=12)
        ax1.set_title('Fish-Chaser Distance Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # 2. Velocity over time
        ax2 = fig.add_subplot(gs[1, :])
        valid_vel = ~np.isnan(velocities)
        ax2.plot(time_seconds[valid_vel], velocities[valid_vel], 
                'g-', linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.fill_between(time_seconds[valid_vel], 0, velocities[valid_vel],
                         where=(velocities[valid_vel] < 0), color='red', alpha=0.3, 
                         label='Approaching')
        ax2.fill_between(time_seconds[valid_vel], 0, velocities[valid_vel],
                         where=(velocities[valid_vel] > 0), color='blue', alpha=0.3,
                         label='Escaping')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Velocity (pixels/second)', fontsize=12)
        ax2.set_title('Relative Velocity (Negative = Approaching)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        # 3. Spatial trajectory (Camera space)
        ax3 = fig.add_subplot(gs[2, 0])
        valid_pos = valid_frames
        ax3.scatter(fish_pos[valid_pos, 0], fish_pos[valid_pos, 1], 
                   c=time_seconds[valid_pos], cmap='viridis', s=1, alpha=0.5)
        ax3.scatter(chaser_pos_cam[valid_pos, 0], chaser_pos_cam[valid_pos, 1],
                   c=time_seconds[valid_pos], cmap='plasma', s=1, alpha=0.5)
        ax3.set_xlim(0, self.video_width)
        ax3.set_ylim(0, self.video_height)
        ax3.set_xlabel('X (pixels)', fontsize=12)
        ax3.set_ylabel('Y (pixels)', fontsize=12)
        ax3.set_title('Trajectories in Camera Space', fontsize=14, fontweight='bold')
        ax3.set_aspect('equal')
        ax3.invert_yaxis()  # Match image coordinates
        
        # Add legend
        ax3.plot([], [], 'o', color='green', label='Fish')
        ax3.plot([], [], 'o', color='purple', label='Chaser')
        ax3.legend(loc='upper right')
        
        # 4. Distance histogram
        ax4 = fig.add_subplot(gs[2, 1])
        valid_distances = distances[valid_frames]
        ax4.hist(valid_distances, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax4.axvline(x=np.mean(valid_distances), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(valid_distances):.1f}')
        ax4.axvline(x=np.median(valid_distances), color='green', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(valid_distances):.1f}')
        ax4.set_xlabel('Distance (pixels)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Distance Distribution', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Chaser trajectory in texture space
        ax5 = fig.add_subplot(gs[2, 2])
        valid_tex = ~np.isnan(chaser_pos_tex[:, 0])
        ax5.scatter(chaser_pos_tex[valid_tex, 0], chaser_pos_tex[valid_tex, 1],
                   c=time_seconds[valid_tex], cmap='coolwarm', s=1, alpha=0.5)
        ax5.set_xlim(0, self.texture_width)
        ax5.set_ylim(0, self.texture_height)
        ax5.set_xlabel('X (texture pixels)', fontsize=12)
        ax5.set_ylabel('Y (texture pixels)', fontsize=12)
        ax5.set_title('Chaser in Texture Space (358×358)', fontsize=14, fontweight='bold')
        ax5.set_aspect('equal')
        ax5.invert_yaxis()
        
        # Add texture center marker
        ax5.plot(179, 179, 'r+', markersize=10, markeredgewidth=2, label='Center')
        ax5.legend()
        
        # Add colorbar for time
        cbar = plt.colorbar(ax5.collections[0], ax=ax5, label='Time (s)')
        
        # Overall title
        fig.suptitle(f'Chaser-Fish Distance Analysis\n{self.h5_path.name}', 
                    fontsize=16, fontweight='bold')
        
        # Add summary statistics as text
        stats_text = self._generate_stats_text(metrics)
        if esc is not None:
            on = int(esc['escape_onset'].sum())
            tr = int(esc['triggered_by_approach'].sum())
            frac = (tr / on * 100.0) if on else 0.0
            stats_text += (
                f"\nEscapes: {on}  Triggered: {tr} ({frac:.1f}%)"
                f"\nD_thresh: {esc['thresholds']['D_thresh']:.1f}px"
                f"  S_escape: {esc['thresholds']['S_escape']:.1f}px/s"
                f"  V_close: {esc['thresholds']['V_close']:.0f}px/s"
            )
        fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"  ✅ Saved plot to: {save_path}")
        else:
            plt.show()
    
    def _generate_stats_text(self, metrics):
        """Generate summary statistics text for plots."""
        distances = metrics['fish_chaser_distance_pixels']
        velocities = metrics['relative_velocity']
        valid_frames = metrics['valid_frames']
        interpolated = metrics['fish_interpolated']
        
        valid_distances = distances[valid_frames]
        valid_velocities = velocities[~np.isnan(velocities)]
        
        stats = []
        stats.append("Summary Statistics:")
        stats.append(f"Total frames: {len(distances):,}")
        stats.append(f"Valid frames: {np.sum(valid_frames):,} ({np.sum(valid_frames)/len(distances)*100:.1f}%)")
        stats.append(f"Interpolated: {np.sum(interpolated):,} ({np.sum(interpolated)/len(distances)*100:.1f}%)")
        
        if len(valid_distances) > 0:
            stats.append(f"\nDistance (pixels):")
            stats.append(f"  Mean: {np.mean(valid_distances):.1f}")
            stats.append(f"  Std: {np.std(valid_distances):.1f}")
            stats.append(f"  Min: {np.min(valid_distances):.1f}")
            stats.append(f"  Max: {np.max(valid_distances):.1f}")
        
        if len(valid_velocities) > 0:
            stats.append(f"\nVelocity (px/s):")
            stats.append(f"  Mean: {np.mean(valid_velocities):.1f}")
            stats.append(f"  Max approach: {np.min(valid_velocities):.1f}")
            stats.append(f"  Max escape: {np.max(valid_velocities):.1f}")
        
        stats.append(f"\nCoordinate System:")
        stats.append(f"  Texture: {self.texture_width}×{self.texture_height}")
        stats.append(f"  Camera: {self.video_width}×{self.video_height}")
        stats.append(f"  Scale: {self.texture_to_camera_scale_x:.2f}×")
        
        return '\n'.join(stats)
    
    def generate_report(self) -> str:
        """
        Generate a detailed text report of the analysis.
        
        Returns:
            Formatted report string
        """
        metrics = self.calculate_distances()
        
        report = []
        report.append("=" * 80)
        report.append("CHASER-FISH DISTANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().isoformat()}")
        report.append(f"Version: 2.1.0")
        
        report.append("\n" + "-" * 40)
        report.append("DATA SOURCES")
        report.append("-" * 40)
        report.append(f"H5 File: {self.h5_path.name}")
        report.append(f"Zarr File: {self.zarr_path.name}")
        report.append(f"Interpolation Run: {self.interpolation_run or 'original'}")
        
        report.append("\n" + "-" * 40)
        report.append("SYSTEM CONFIGURATION")
        report.append("-" * 40)
        report.append(f"Camera: {self.camera_id or 'unknown'}")
        report.append(f"Video: {self.video_width}×{self.video_height} @ {self.fps_video} FPS")
        report.append(f"Stimulus: {self.fps_stimulus:.1f} Hz")
        report.append(f"Frame Rate Ratio: {self.fps_stimulus/self.fps_video:.2f}:1")
        
        report.append("\n" + "-" * 40)
        report.append("COORDINATE TRANSFORMATION")
        report.append("-" * 40)
        report.append(f"Texture Space: {self.texture_width}×{self.texture_height} pixels")
        report.append(f"Camera Space: {self.video_width}×{self.video_height} pixels")
        report.append(f"Scale Factor X: {self.texture_to_camera_scale_x:.3f}")
        report.append(f"Scale Factor Y: {self.texture_to_camera_scale_y:.3f}")
        
        # Analysis results
        distances = metrics['fish_chaser_distance_pixels']
        velocities = metrics['relative_velocity']
        valid_frames = metrics['valid_frames']
        interpolated = metrics['fish_interpolated']
        
        report.append("\n" + "-" * 40)
        report.append("FRAME ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total Frames: {len(distances):,}")
        report.append(f"Valid Frames: {np.sum(valid_frames):,} ({np.sum(valid_frames)/len(distances)*100:.1f}%)")
        report.append(f"Interpolated: {np.sum(interpolated):,} ({np.sum(interpolated)/len(distances)*100:.1f}%)")
        report.append(f"Missing Data: {len(distances) - np.sum(valid_frames):,}")
        
        if np.sum(valid_frames) > 0:
            valid_distances = distances[valid_frames]
            
            report.append("\n" + "-" * 40)
            report.append("DISTANCE STATISTICS (pixels)")
            report.append("-" * 40)
            report.append(f"Mean:     {np.mean(valid_distances):10.1f}")
            report.append(f"Median:   {np.median(valid_distances):10.1f}")
            report.append(f"Std Dev:  {np.std(valid_distances):10.1f}")
            report.append(f"Minimum:  {np.min(valid_distances):10.1f}")
            report.append(f"Maximum:  {np.max(valid_distances):10.1f}")
            report.append(f"Q1 (25%): {np.percentile(valid_distances, 25):10.1f}")
            report.append(f"Q3 (75%): {np.percentile(valid_distances, 75):10.1f}")
        
        if np.sum(~np.isnan(velocities)) > 0:
            valid_velocities = velocities[~np.isnan(velocities)]
            
            report.append("\n" + "-" * 40)
            report.append("VELOCITY STATISTICS (pixels/second)")
            report.append("-" * 40)
            report.append(f"Mean:          {np.mean(valid_velocities):10.1f}")
            report.append(f"Max Approach:  {np.min(valid_velocities):10.1f} (most negative)")
            report.append(f"Max Escape:    {np.max(valid_velocities):10.1f} (most positive)")
            
            # Behavior classification
            approaching = np.sum(valid_velocities < -10)
            escaping = np.sum(valid_velocities > 10)
            stationary = len(valid_velocities) - approaching - escaping
            
            report.append(f"\nBehavior Classification (|v| > 10 px/s):")
            report.append(f"  Approaching: {approaching:6d} frames ({approaching/len(valid_velocities)*100:5.1f}%)")
            report.append(f"  Escaping:    {escaping:6d} frames ({escaping/len(valid_velocities)*100:5.1f}%)")
            report.append(f"  Stationary:  {stationary:6d} frames ({stationary/len(valid_velocities)*100:5.1f}%)")
        
        if hasattr(self, 'chase_events') and self.chase_events:
            chase_starts = len([e for e in self.chase_events if e['type'] == 'start'])
            report.append("\n" + "-" * 40)
            report.append("EXPERIMENTAL EVENTS")
            report.append("-" * 40)
            report.append(f"Chase Sequences: {chase_starts}")
        
        # Escape-trigger summary if smoothed speed available
        if 'smoothed_speed_px_s' in metrics:
            esc = self._detect_escapes(
                distances, velocities, metrics['smoothed_speed_px_s'], self.fps_video
            )
            on = int(esc['escape_onset'].sum())
            tr = int(esc['triggered_by_approach'].sum())
            frac = (tr / on * 100.0) if on else 0.0

            report.append("\n" + "-" * 40)
            report.append("ESCAPE TRIGGER ANALYSIS")
            report.append("-" * 40)
            report.append(f"Escapes detected: {on}")
            report.append(f"Triggered by near & approaching: {tr} ({frac:.1f}%)")
            report.append(f"D_thresh (near): {esc['thresholds']['D_thresh']:.1f} px")
            report.append(f"S_escape (speed): {esc['thresholds']['S_escape']:.1f} px/s (window={self.speed_window or 'n/a'})")
            report.append(f"V_close (approach): {esc['thresholds']['V_close']:.0f} px/s")
            if esc['latency_ms'].size:
                report.append(
                    f"Latency (approach→escape): median {np.median(esc['latency_ms']):.0f} ms "
                    f"[IQR {np.percentile(esc['latency_ms'],25):.0f}–{np.percentile(esc['latency_ms'],75):.0f} ms]"
                )

        report.append("\n" + "=" * 80)
        
        return '\n'.join(report)
    
    def save_analysis(self):
        """Save analysis results to the zarr file."""
        # [Previous save_analysis implementation remains the same]
        self.logger.info("Saving analysis to zarr...")
        
        # Create chaser_comparison group if needed
        if 'chaser_comparison' not in self.zarr_root:
            comp_group = self.zarr_root.create_group('chaser_comparison')
            comp_group.attrs['created_at'] = datetime.now().isoformat()
        else:
            comp_group = self.zarr_root['chaser_comparison']
        
        # Store metadata
        meta_group = comp_group.require_group('metadata')
        meta_group.attrs['h5_source'] = str(self.h5_path)
        meta_group.attrs['h5_checksum'] = self._calculate_file_checksum(self.h5_path)
        meta_group.attrs['camera_id'] = self.camera_id or 'unknown'
        meta_group.attrs['coordinate_system'] = 'texture_to_camera_v2.1'
        meta_group.attrs['video_dimensions'] = [self.video_width, self.video_height]
        meta_group.attrs['texture_dimensions'] = [self.texture_width, self.texture_height]
        meta_group.attrs['texture_to_camera_scale'] = [self.texture_to_camera_scale_x, self.texture_to_camera_scale_y]
        meta_group.attrs['fps_video'] = self.fps_video
        meta_group.attrs['fps_stimulus'] = self.fps_stimulus
        meta_group.attrs['updated_at'] = datetime.now().isoformat()
        meta_group.attrs['version'] = '2.1.0'
        
        # Store frame alignment
        align_group = comp_group.require_group('frame_alignment')
        if 'zarr_to_stimulus' in align_group:
            del align_group['zarr_to_stimulus']
        align_group.create_dataset('zarr_to_stimulus', data=self.zarr_to_stimulus)

        # Calculate metrics
        metrics = self.calculate_distances()

        # Create run-specific group
        run_name = self.interpolation_run or 'original'
        if run_name in comp_group:
            self.logger.info(f"  Overwriting existing analysis for run: {run_name}")
            del comp_group[run_name]

        run_group = comp_group.create_group(run_name)
        run_group.attrs['created_at'] = datetime.now().isoformat()
        run_group.attrs['interpolation_source'] = self.interpolation_run or 'original_detections'
        run_group.attrs['coordinate_transform'] = 'texture_to_camera_scaling'

        # Save all metrics
        for metric_name, metric_data in metrics.items():
            # Handle different data types
            if metric_data.dtype == np.float64:
                dtype = 'float32'
            elif metric_data.dtype == bool:
                dtype = bool
            else:
                dtype = metric_data.dtype

            # Create dataset with appropriate chunking
            if metric_data.ndim == 1:
                chunks = (min(10000, len(metric_data)),)
            else:
                chunks = (min(10000, metric_data.shape[0]), metric_data.shape[1])

            dataset = run_group.create_dataset(
                metric_name,
                data=metric_data,
                chunks=chunks,
                dtype=dtype
            )

            # Add descriptive attributes
            if 'distance_pixels' in metric_name:
                dataset.attrs['units'] = 'pixels'
                dataset.attrs['description'] = 'Euclidean distance between fish and chaser in camera coordinates'
            elif 'velocity' in metric_name:
                dataset.attrs['units'] = 'pixels/second'
                dataset.attrs['description'] = 'Rate of change of distance (negative = approaching)'
            elif 'position_camera' in metric_name:
                dataset.attrs['units'] = 'pixels'
                dataset.attrs['description'] = 'Position in camera/pixel coordinates'
            elif 'position_texture' in metric_name:
                dataset.attrs['units'] = 'texture_pixels'
                dataset.attrs['description'] = 'Original position in texture space (358×358)'

        # Calculate and save summary statistics
        self._save_summary_statistics(run_group, metrics)

        # ---------- INSERTED BLOCK: escape detection & saving ----------
        # Use smoothed speed if present (from speed_metrics batch)
        speed_px_s = metrics.get('smoothed_speed_px_s', None)

        esc = self._detect_escapes(
            metrics['fish_chaser_distance_pixels'],
            metrics['relative_velocity'],
            speed_px_s,
            self.fps_video
        )

        # Save boolean masks as compact uint8 arrays with attrs describing meaning
        for name in ['escape_onset', 'escape_mask', 'triggered_by_approach',
                    'near_and_approach', 'near_and_approach_lag']:
            data = esc[name].astype(np.uint8)
            if name in run_group:
                del run_group[name]
            run_group.create_dataset(name, data=data, chunks=(min(10000, len(data)),))
            run_group[name].attrs['dtype'] = 'bool'
            run_group[name].attrs['description'] = (
                '1=TRUE; ' +
                ('escape onset' if name == 'escape_onset' else
                'sustained escape' if name == 'escape_mask' else
                'approach-triggered onset' if name == 'triggered_by_approach' else
                'near & approaching (strict)' if name == 'near_and_approach' else
                'near & approaching with look-back lag')
            )

        # Merge thresholds & counts into the existing summary
        summary = run_group.attrs.get('summary', {})
        summary.update({
            'escape_onsets': int(esc['escape_onset'].sum()),
            'escape_onsets_triggered': int(esc['triggered_by_approach'].sum()),
            'trigger_fraction': float(
                esc['triggered_by_approach'].sum() / max(1, esc['escape_onset'].sum())
            ),
        })
        summary.update(esc['thresholds'])
        run_group.attrs['summary'] = summary

        # Latency distribution (ms) for triggered escapes
        if esc['latency_ms'].size:
            if 'latency_ms' in run_group:
                del run_group['latency_ms']
            run_group.create_dataset(
                'latency_ms',
                data=esc['latency_ms'].astype('float32'),
                chunks=(min(10000, len(esc['latency_ms'])),)
            )
        # ---------- END INSERTED BLOCK ----------

        # Update latest run pointer
        comp_group.attrs['latest'] = run_name

        self.logger.info(f"  ✅ Saved analysis for run: {run_name}")
        self.logger.info(f"  Zarr file: {self.zarr_path}")

    
    def _save_summary_statistics(self, group, metrics):
        """Calculate and save summary statistics."""
        distances_pixels = metrics['fish_chaser_distance_pixels']
        velocities = metrics['relative_velocity']
        
        valid_distances = distances_pixels[~np.isnan(distances_pixels)]
        
        stats = {}
        
        if len(valid_distances) > 0:
            # Distance statistics
            stats['mean_distance_pixels'] = float(np.mean(valid_distances))
            stats['median_distance_pixels'] = float(np.median(valid_distances))
            stats['min_distance_pixels'] = float(np.min(valid_distances))
            stats['max_distance_pixels'] = float(np.max(valid_distances))
            stats['std_distance_pixels'] = float(np.std(valid_distances))
            
            # Frame coverage
            stats['frames_analyzed'] = len(valid_distances)
            stats['total_frames'] = len(distances_pixels)
            stats['coverage'] = len(valid_distances) / len(distances_pixels)
            
            # Velocity statistics
            valid_velocities = velocities[~np.isnan(velocities)]
            if len(valid_velocities) > 0:
                stats['mean_velocity'] = float(np.mean(valid_velocities))
                stats['max_approach_velocity'] = float(np.min(valid_velocities))
                stats['max_escape_velocity'] = float(np.max(valid_velocities))
        
        # Coordinate system info
        stats['texture_width'] = self.texture_width
        stats['texture_height'] = self.texture_height
        stats['camera_width'] = self.video_width
        stats['camera_height'] = self.video_height
        stats['texture_to_camera_scale_x'] = self.texture_to_camera_scale_x
        stats['texture_to_camera_scale_y'] = self.texture_to_camera_scale_y
        
        group.attrs['summary'] = stats
        
        # Log summary
        self.logger.info("  Summary statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                self.logger.info(f"    {key}: {value:.2f}")
            else:
                self.logger.info(f"    {key}: {value}")
    
    def _calculate_file_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def main():
    """Command-line interface for the analyzer."""
    parser = argparse.ArgumentParser(
        description='Analyze fish-chaser distances with coordinate transformation and plotting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This version (v2.1) includes:
- Proper handling of 120Hz/60FPS frame rate mismatch
- Correct texture→camera coordinate transformation
- Comprehensive plotting and reporting
- Support for fixed PROTOCOL_START events

Examples:
  %(prog)s detections.zarr analysis.h5 --plot
  %(prog)s detections.zarr analysis.h5 --interpolation-run interp_linear --report
  %(prog)s detections.zarr analysis.h5 --save-plot results.png
        """
    )
    
    parser.add_argument('zarr_path', help='Path to YOLO detection zarr file')
    parser.add_argument('h5_path', help='Path to H5 analysis file')
    parser.add_argument('--interpolation-run', help='Specific interpolation run to use')
    parser.add_argument('--plot', action='store_true', help='Generate and show plots')
    parser.add_argument('--save-plot', help='Path to save plot figure')
    parser.add_argument('--report', action='store_true', help='Print detailed report')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ChaserFishDistanceAnalyzer(
        zarr_path=args.zarr_path,
        h5_path=args.h5_path,
        interpolation_run=args.interpolation_run,
        use_texture_scaling=True,  # Always use correct scaling
        verbose=not args.quiet
    )
    
    # Run analysis
    analyzer.save_analysis()
    
    # Generate report if requested
    if args.report:
        print("\n" + analyzer.generate_report())
    
    # Generate plots if requested  
    if args.plot or args.save_plot:
        analyzer.plot_results(save_path=args.save_plot)
    
    return 0


if __name__ == '__main__':
    exit(main())