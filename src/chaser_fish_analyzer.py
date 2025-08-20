#!/usr/bin/env python3
"""
Chaser-Fish Distance Analyzer

Analyzes distances between fish (from YOLO detections in zarr) and chaser 
(from H5 files) with proper coordinate transformation.

IMPORTANT: Chaser positions are in TEXTURE space (358×358), not world/projector space.
Fish positions are in CAMERA space (4512×4512).
We use simple scaling (×12.604) to transform texture→camera, NOT homography.

This module handles:
- Frame alignment between 60 FPS video and 120 FPS stimulus
- Coordinate transformation from texture to camera space
- Distance calculations in camera pixel coordinates
- Integration with interpolated detection data
- Comprehensive metric calculation and storage
"""

import zarr
import h5py
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import json
import hashlib
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings

# Import coordinate transform module if available
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
    version: str = "2.0.0"  # Updated version for new coordinate system


class ChaserFishDistanceAnalyzer:
    """
    Analyzes fish-chaser distances with proper coordinate transformation.
    
    This class handles:
    - Loading and aligning data from zarr (YOLO detections) and H5 (chaser states)
    - Coordinate transformation from texture to camera space (NOT using homography)
    - Distance and behavioral metric calculations
    - Saving results back to the zarr file
    """
    
    def __init__(self, 
                 zarr_path: str, 
                 h5_path: str,
                 interpolation_run: Optional[str] = None,
                 use_texture_scaling: bool = True,  # Changed default
                 verbose: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            zarr_path: Path to YOLO detection zarr file
            h5_path: Path to interpolated H5 file (from h5_frame_interpolator)
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
        self.logger.info(f"Initializing analyzer v2.0 (texture-aware)")
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
        self.metadata = None
        self.coord_sys = None
        
        # Initialize coordinate system
        self._initialize_coordinate_system()
        
        # Load all necessary data
        self._load_video_info()
        self._load_h5_metadata()
        self._create_frame_alignment()
    
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
        """Load frame metadata and chaser states from interpolated H5."""
        self.logger.info("Loading H5 metadata and chaser states...")
        
        with h5py.File(self.h5_path, 'r') as h5f:
            # Try to get camera ID from calibration
            if '/calibration_snapshot' in h5f:
                calib = h5f['/calibration_snapshot']
                camera_ids = [k for k in calib.keys() if isinstance(calib[k], h5py.Group)]
                if camera_ids:
                    self.camera_id = camera_ids[0]
                    self.logger.info(f"  Camera ID: {self.camera_id}")
            
            # Load frame metadata (should be interpolated if using h5_frame_interpolator output)
            if '/video_metadata/frame_metadata' not in h5f:
                raise ValueError("No frame_metadata found in H5 file")
            
            frame_meta = h5f['/video_metadata/frame_metadata'][:]
            self.logger.info(f"  Loaded {len(frame_meta)} frame metadata records")
            
            # Build bidirectional mapping
            self.camera_to_stimulus = {}
            self.stimulus_to_camera = {}
            
            for record in frame_meta:
                cam_frame = int(record['triggering_camera_frame_id'])
                stim_frame = int(record['stimulus_frame_num'])
                
                # Handle multiple stimulus frames per camera frame (120Hz vs 60Hz)
                if cam_frame not in self.camera_to_stimulus:
                    self.camera_to_stimulus[cam_frame] = []
                self.camera_to_stimulus[cam_frame].append(stim_frame)
                
                # Keep track of reverse mapping
                self.stimulus_to_camera[stim_frame] = cam_frame
            
            # Estimate frame rate ratio
            avg_stim_per_cam = np.mean([len(stims) for stims in self.camera_to_stimulus.values()])
            self.fps_stimulus = self.fps_video * avg_stim_per_cam
            self.logger.info(f"  Stimulus rate: {self.fps_stimulus:.1f} FPS")
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
            
            # Log chaser state fields for reference
            if len(self.chaser_states) > 0:
                self.logger.info(f"  Chaser state fields: {list(self.chaser_states.dtype.names)}")
    
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
    
    def transform_texture_to_camera(self, texture_x: float, texture_y: float) -> Tuple[float, float]:
        """
        Transform from texture space (358×358) to camera space (4512×4512).
        
        This is the CORRECT transformation for chaser positions.
        Does NOT use homography - uses simple linear scaling.
        
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
        Calculate fish-chaser distances with CORRECT coordinate transformation.
        
        Key point: Chaser positions are in TEXTURE space (358×358), not world/projector!
        
        Returns:
            Dictionary with distance metrics and position data
        """
        self.logger.info("Calculating fish-chaser distances...")
        self.logger.info(f"  Coordinate mode: {'texture→camera scaling' if self.use_texture_scaling else 'legacy'}")
        
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
        chaser_positions_texture = np.full((n_frames, 2), np.nan)  # Original texture coords
        
        # Process each frame
        valid_count = 0
        transformation_logged = False
        
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
            chaser_x_texture = None
            chaser_y_texture = None
            
            for x_field in ['chaser_pos_x', 'x_position', 'x_pos', 'x', 'pos_x']:
                if x_field in chaser.dtype.names:
                    chaser_x_texture = float(chaser[x_field])
                    break
            
            for y_field in ['chaser_pos_y', 'y_position', 'y_pos', 'y', 'pos_y']:
                if y_field in chaser.dtype.names:
                    chaser_y_texture = float(chaser[y_field])
                    break
            
            if chaser_x_texture is None or chaser_y_texture is None:
                self.logger.warning(f"Could not find position fields in chaser state. Available: {chaser.dtype.names}")
                continue
            
            # Store chaser texture position
            chaser_positions_texture[frame_idx] = [chaser_x_texture, chaser_y_texture]
            
            # ===== CRITICAL: Transform chaser from TEXTURE to CAMERA space =====
            chaser_x_camera, chaser_y_camera = self.transform_texture_to_camera(
                chaser_x_texture, chaser_y_texture
            )
            chaser_positions_camera[frame_idx] = [chaser_x_camera, chaser_y_camera]
            
            # Log transformation details for first valid frame
            if not transformation_logged:
                self.logger.info(f"  Transformation example:")
                self.logger.info(f"    Chaser texture: ({chaser_x_texture:.1f}, {chaser_y_texture:.1f})")
                self.logger.info(f"    Chaser camera: ({chaser_x_camera:.1f}, {chaser_y_camera:.1f})")
                self.logger.info(f"    Scale factors: x={self.texture_to_camera_scale_x:.3f}, y={self.texture_to_camera_scale_y:.3f}")
                
                # Check if chaser is at texture center
                if abs(chaser_x_texture - 179) < 1 and abs(chaser_y_texture - 179) < 1:
                    expected_camera_x = self.video_width / 2
                    expected_camera_y = self.video_height / 2
                    error_x = abs(chaser_x_camera - expected_camera_x)
                    error_y = abs(chaser_y_camera - expected_camera_y)
                    if error_x < 10 and error_y < 10:
                        self.logger.info(f"    ✅ Chaser at texture center correctly maps to camera center!")
                    else:
                        self.logger.warning(f"    ⚠️ Chaser center mapping error: ({error_x:.1f}, {error_y:.1f}) pixels")
                
                transformation_logged = True
            
            # Calculate distance in CAMERA PIXELS
            dist_pixels = np.sqrt((fish_x_camera - chaser_x_camera)**2 + 
                                 (fish_y_camera - chaser_y_camera)**2)
            
            distances_pixels[frame_idx] = dist_pixels
            valid_count += 1
            
            # Calculate pursuit angle if chaser has heading
            for heading_field in ['heading', 'heading_angle', 'angle', 'orientation']:
                if heading_field in chaser.dtype.names:
                    chaser_heading = float(chaser[heading_field])
                    # Calculate angle from chaser to fish IN CAMERA SPACE
                    angle_to_fish = np.arctan2(fish_y_camera - chaser_y_camera,
                                              fish_x_camera - chaser_x_camera)
                    # Normalize angle difference to [-pi, pi]
                    angle_diff = angle_to_fish - chaser_heading
                    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                    pursuit_angles[frame_idx] = angle_diff
                    break
        
        self.logger.info(f"  ✅ Calculated distances for {valid_count}/{n_frames} frames")
        
        # Calculate velocities
        valid_distances = ~np.isnan(distances_pixels)
        if np.sum(valid_distances) > 1:
            temp_velocities = np.full_like(distances_pixels, np.nan)
            valid_indices = np.where(valid_distances)[0]
            
            if len(valid_indices) > 1:
                valid_dists = distances_pixels[valid_indices]
                valid_vels = np.gradient(valid_dists)
                
                for i in range(len(valid_indices) - 1):
                    frame_gap = valid_indices[i+1] - valid_indices[i]
                    valid_vels[i] *= self.fps_video / frame_gap
                
                temp_velocities[valid_indices] = valid_vels
                relative_velocities = temp_velocities
        
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
        
        return {
            'fish_chaser_distance_pixels': distances_pixels,
            'relative_velocity': relative_velocities,
            'pursuit_angle': pursuit_angles,
            'fish_position_camera': fish_positions_camera,
            'chaser_position_camera': chaser_positions_camera,
            'chaser_position_texture': chaser_positions_texture,
            'fish_interpolated': mask,
            'valid_frames': valid_mask
        }
    
    def save_analysis(self):
        """Save analysis results to the zarr file."""
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
        meta_group.attrs['coordinate_system'] = 'texture_to_camera_v2'
        meta_group.attrs['video_dimensions'] = [self.video_width, self.video_height]
        meta_group.attrs['texture_dimensions'] = [self.texture_width, self.texture_height]
        meta_group.attrs['texture_to_camera_scale'] = [self.texture_to_camera_scale_x, self.texture_to_camera_scale_y]
        meta_group.attrs['fps_video'] = self.fps_video
        meta_group.attrs['fps_stimulus'] = self.fps_stimulus
        meta_group.attrs['updated_at'] = datetime.now().isoformat()
        meta_group.attrs['version'] = '2.0.0'
        
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
            elif 'pursuit_angle' in metric_name:
                dataset.attrs['units'] = 'radians'
                dataset.attrs['description'] = 'Angle difference between chaser heading and direction to fish'
            elif 'position_camera' in metric_name:
                dataset.attrs['units'] = 'pixels'
                dataset.attrs['description'] = 'Position in camera/pixel coordinates'
            elif 'position_texture' in metric_name:
                dataset.attrs['units'] = 'texture_pixels'
                dataset.attrs['description'] = 'Original position in texture space (358×358)'
        
        # Calculate and save summary statistics
        self._save_summary_statistics(run_group, metrics)
        
        # Update latest run
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
                stats['max_approach_velocity'] = float(np.min(valid_velocities))  # Most negative
                stats['max_escape_velocity'] = float(np.max(valid_velocities))   # Most positive
                
                # Detect potential escape events
                if len(valid_velocities) > 10:
                    threshold = np.percentile(valid_velocities, 95)
                    escape_frames = np.where(velocities > threshold)[0]
                    stats['potential_escapes'] = len(escape_frames)
                    stats['escape_velocity_threshold'] = float(threshold)
        
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
    
    # ... (plot_results and generate_report methods remain the same) ...


def main():
    """Command-line interface for the analyzer."""
    parser = argparse.ArgumentParser(
        description='Analyze fish-chaser distances with CORRECT coordinate transformation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This version (v2.0) correctly handles coordinate transformations:
- Chaser positions are in TEXTURE space (358x358)
- Fish positions are in CAMERA space (4512x4512)
- Uses simple scaling (x12.604) to transform texture→camera
- Does NOT use homography for chaser transformation

Examples:
  %(prog)s detections.zarr out_analysis.h5
  %(prog)s detections.zarr out_analysis.h5 --interpolation-run interp_linear_20240120
  %(prog)s detections.zarr out_analysis.h5 --plot
        """
    )
    
    parser.add_argument('zarr_path', help='Path to YOLO detection zarr file')
    parser.add_argument('h5_path', help='Path to interpolated H5 file')
    parser.add_argument('--interpolation-run', help='Specific interpolation run to use')
    parser.add_argument('--plot', action='store_true', help='Generate and show plots')
    parser.add_argument('--save-plot', help='Path to save plot figure')
    parser.add_argument('--report', action='store_true', help='Print detailed report')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create analyzer with correct coordinate transformation
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