#!/usr/bin/env python3
"""
Chaser-Fish Distance Analyzer

Analyzes distances between fish (from YOLO detections in zarr) and chaser 
(from H5 files) with proper coordinate transformation using homography.

This module handles:
- Frame alignment between 60 FPS video and 120 FPS stimulus
- Coordinate transformation using homography matrices
- Distance calculations in both pixel and world coordinates
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
    frame_rate_video: float
    frame_rate_stimulus: float
    frame_rate_ratio: float
    homography_determinant: float
    total_frames_analyzed: int
    valid_frames: int
    version: str = "1.0.0"


class ChaserFishDistanceAnalyzer:
    """
    Analyzes fish-chaser distances with proper coordinate transformation.
    
    This class handles:
    - Loading and aligning data from zarr (YOLO detections) and H5 (chaser states)
    - Coordinate transformation using homography matrices
    - Distance and behavioral metric calculations
    - Saving results back to the zarr file
    """
    
    def __init__(self, 
                 zarr_path: str, 
                 h5_path: str,
                 interpolation_run: Optional[str] = None,
                 use_world_coords: bool = False,
                 verbose: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            zarr_path: Path to YOLO detection zarr file
            h5_path: Path to interpolated H5 file (from h5_frame_interpolator)
            interpolation_run: Specific interpolation run to use (default: latest)
            use_world_coords: If True, transform fish to world coords instead of
                            transforming chaser to camera coords
            verbose: Enable verbose logging
        """
        self.zarr_path = Path(zarr_path)
        self.h5_path = Path(h5_path)
        self.use_world_coords = use_world_coords
        self.verbose = verbose
        
        # Setup logging
        self._setup_logging()
        
        # Load data sources
        self.logger.info(f"Initializing analyzer")
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
        self.homography = None
        self.homography_inv = None
        self.camera_id = None
        self.video_width = None
        self.video_height = None
        self.camera_to_stimulus = {}
        self.stimulus_to_camera = {}
        self.chaser_states = None
        self.chaser_by_stimulus = {}
        self.zarr_to_stimulus = None
        self.metadata = None
        
        # Load all necessary data
        self._load_homography()
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
    
    def _load_homography(self):
        """Load and prepare homography matrix from H5 file."""
        self.logger.info("Loading homography matrix...")
        
        with h5py.File(self.h5_path, 'r') as h5f:
            # Check for calibration snapshot
            if '/calibration_snapshot' not in h5f:
                # Try direct homography dataset as fallback
                if '/homography' in h5f:
                    self.homography = h5f['/homography'][:]
                    self.camera_id = 'default'
                    self.logger.warning("Using direct homography dataset (no calibration_snapshot)")
                else:
                    raise ValueError("No calibration_snapshot or homography found in H5 file")
            else:
                calib = h5f['/calibration_snapshot']
                
                # Find camera calibration
                camera_ids = [k for k in calib.keys() if isinstance(calib[k], h5py.Group)]
                if not camera_ids:
                    raise ValueError("No camera calibration found in calibration_snapshot")
                
                # Use first camera (or could make this configurable)
                cam_id = camera_ids[0]
                self.camera_id = cam_id
                self.logger.info(f"  Using camera: {cam_id}")
                
                if 'homography_matrix_yml' not in calib[cam_id]:
                    raise ValueError(f"No homography matrix found for camera {cam_id}")
                
                # Parse YAML to get homography matrix
                yaml_data = calib[cam_id]['homography_matrix_yml'][()].decode('utf-8')
                self.homography = self._parse_homography_from_yaml(yaml_data)
        
        # Calculate inverse homography
        try:
            self.homography_inv = np.linalg.inv(self.homography)
        except np.linalg.LinAlgError:
            raise ValueError("Homography matrix is not invertible")
        
        # Verify homography validity
        det = np.linalg.det(self.homography)
        if abs(det) < 1e-10:
            raise ValueError("Homography matrix is singular")
        
        self.logger.info(f"  ✅ Loaded 3x3 homography matrix")
        self.logger.info(f"  Determinant: {det:.6f}")
        
        # Get video dimensions from zarr
        self.video_width = self.zarr_root.attrs.get('width', 1920)
        self.video_height = self.zarr_root.attrs.get('height', 1080)
        self.fps_video = self.zarr_root.attrs.get('fps', 60.0)
        
        self.logger.info(f"  Video dimensions: {self.video_width}x{self.video_height} @ {self.fps_video} FPS")
    
    def _parse_homography_from_yaml(self, yaml_data: str) -> np.ndarray:
        """Parse homography matrix from YAML string."""
        # Extract matrix values from YAML
        matrix_values = []
        
        # Look for lines with matrix data
        for line in yaml_data.split('\n'):
            # Skip metadata lines
            if 'rows:' in line or 'cols:' in line or 'dt:' in line:
                continue
            
            # Extract numbers from data lines
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            if numbers and 'data:' not in line:
                matrix_values.extend([float(n) for n in numbers])
        
        if len(matrix_values) != 9:
            # Try alternative parsing for flat data format
            if 'data:' in yaml_data:
                data_section = yaml_data.split('data:')[1]
                # Remove brackets and split
                data_section = data_section.replace('[', '').replace(']', '')
                numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', data_section)
                matrix_values = [float(n) for n in numbers]
        
        if len(matrix_values) != 9:
            raise ValueError(f"Expected 9 values for 3x3 homography, got {len(matrix_values)}")
        
        return np.array(matrix_values).reshape(3, 3)
    
    def _load_h5_metadata(self):
        """Load frame metadata and chaser states from interpolated H5."""
        self.logger.info("Loading H5 metadata and chaser states...")
        
        with h5py.File(self.h5_path, 'r') as h5f:
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
                
                # Keep track of reverse mapping (may be overwritten for multiple mappings)
                self.stimulus_to_camera[stim_frame] = cam_frame
            
            # Estimate frame rate ratio
            avg_stim_per_cam = np.mean([len(stims) for stims in self.camera_to_stimulus.values()])
            self.fps_stimulus = self.fps_video * avg_stim_per_cam
            self.logger.info(f"  Estimated stimulus rate: {self.fps_stimulus:.1f} FPS")
            self.logger.info(f"  Ratio: {avg_stim_per_cam:.2f} stimulus frames per camera frame")
            
            # Load chaser states
            if '/tracking_data/chaser_states' not in h5f:
                raise ValueError("No chaser_states found in H5 file")
            
            self.chaser_states = h5f['/tracking_data/chaser_states'][:]
            self.logger.info(f"  Loaded {len(self.chaser_states)} chaser state records")
            
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
        # Zarr uses 0-based indices, camera_frame_ids may start at any value
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
    
    def transform_point_to_camera(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """
        Transform a point from world coordinates to camera coordinates.
        
        Args:
            world_x, world_y: Position in world/arena coordinates
            
        Returns:
            camera_x, camera_y: Position in camera/pixel coordinates
        """
        # Create homogeneous coordinates
        world_point = np.array([world_x, world_y, 1.0])
        
        # Apply homography (world -> camera)
        camera_point = self.homography @ world_point
        
        # Normalize by w component
        if abs(camera_point[2]) > 1e-10:
            camera_point = camera_point / camera_point[2]
        
        return camera_point[0], camera_point[1]
    
    def transform_point_to_world(self, camera_x: float, camera_y: float) -> Tuple[float, float]:
        """
        Transform a point from camera coordinates to world coordinates.
        
        Args:
            camera_x, camera_y: Position in camera/pixel coordinates
            
        Returns:
            world_x, world_y: Position in world/arena coordinates
        """
        # Create homogeneous coordinates
        camera_point = np.array([camera_x, camera_y, 1.0])
        
        # Apply inverse homography (camera -> world)
        world_point = self.homography_inv @ camera_point
        
        # Normalize by w component
        if abs(world_point[2]) > 1e-10:
            world_point = world_point / world_point[2]
        
        return world_point[0], world_point[1]
    
    def calculate_distances(self) -> Dict[str, np.ndarray]:
        """
        Calculate fish-chaser distances with proper coordinate transformation.
        
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
        distances_world = np.full(n_frames, np.nan)
        relative_velocities = np.full(n_frames, np.nan)
        pursuit_angles = np.full(n_frames, np.nan)
        
        # Position arrays for debugging and visualization
        fish_positions_camera = np.full((n_frames, 2), np.nan)
        fish_positions_world = np.full((n_frames, 2), np.nan)
        chaser_positions_camera = np.full((n_frames, 2), np.nan)
        chaser_positions_world = np.full((n_frames, 2), np.nan)
        
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
            
            # Transform fish to world coordinates
            fish_x_world, fish_y_world = self.transform_point_to_world(
                fish_x_camera, fish_y_camera
            )
            fish_positions_world[frame_idx] = [fish_x_world, fish_y_world]
            
            # Get corresponding stimulus frame
            stim_frame = self.zarr_to_stimulus[frame_idx]
            if stim_frame < 0 or stim_frame not in self.chaser_by_stimulus:
                continue  # No chaser data for this frame
            
            # Get chaser state (assumed to be in world coordinates)
            chaser = self.chaser_by_stimulus[stim_frame]
            
            # Extract chaser position (field names may vary)
            # Try common field names
            chaser_x_world = None
            chaser_y_world = None
            
            for x_field in ['chaser_pos_x', 'x_position', 'x_pos', 'x', 'pos_x']:
                if x_field in chaser.dtype.names:
                    chaser_x_world = float(chaser[x_field])
                    break
            
            for y_field in ['chaser_pos_y', 'y_position', 'y_pos', 'y', 'pos_y']:
                if y_field in chaser.dtype.names:
                    chaser_y_world = float(chaser[y_field])
                    break
            
            if chaser_x_world is None or chaser_y_world is None:
                self.logger.warning(f"Could not find position fields in chaser state. Available: {chaser.dtype.names}")
                continue
            
            # Store chaser world position
            chaser_positions_world[frame_idx] = [chaser_x_world, chaser_y_world]
            
            # Transform chaser to camera coordinates
            chaser_x_camera, chaser_y_camera = self.transform_point_to_camera(
                chaser_x_world, chaser_y_world
            )
            chaser_positions_camera[frame_idx] = [chaser_x_camera, chaser_y_camera]
            
            # Calculate distances in both coordinate systems
            dist_pixels = np.sqrt((fish_x_camera - chaser_x_camera)**2 + 
                                 (fish_y_camera - chaser_y_camera)**2)
            dist_world = np.sqrt((fish_x_world - chaser_x_world)**2 + 
                               (fish_y_world - chaser_y_world)**2)
            
            distances_pixels[frame_idx] = dist_pixels
            distances_world[frame_idx] = dist_world
            valid_count += 1
            
            # Calculate pursuit angle if chaser has heading
            for heading_field in ['heading', 'heading_angle', 'angle', 'orientation']:
                if heading_field in chaser.dtype.names:
                    chaser_heading = float(chaser[heading_field])
                    # Calculate angle from chaser to fish
                    angle_to_fish = np.arctan2(fish_y_world - chaser_y_world,
                                              fish_x_world - chaser_x_world)
                    # Normalize angle difference to [-pi, pi]
                    angle_diff = angle_to_fish - chaser_heading
                    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                    pursuit_angles[frame_idx] = angle_diff
                    break
        
        self.logger.info(f"  ✅ Calculated distances for {valid_count}/{n_frames} frames")
        
        # Calculate velocities (rate of change of distance)
        valid_distances = ~np.isnan(distances_pixels)
        if np.sum(valid_distances) > 1:
            # Use gradient for better handling of gaps
            temp_velocities = np.full_like(distances_pixels, np.nan)
            valid_indices = np.where(valid_distances)[0]
            
            if len(valid_indices) > 1:
                # Calculate velocities only for valid frames
                valid_dists = distances_pixels[valid_indices]
                valid_vels = np.gradient(valid_dists)
                
                # Scale by actual time difference between valid frames
                for i in range(len(valid_indices) - 1):
                    frame_gap = valid_indices[i+1] - valid_indices[i]
                    valid_vels[i] *= self.fps_video / frame_gap
                
                temp_velocities[valid_indices] = valid_vels
                relative_velocities = temp_velocities
        
        # Calculate summary statistics
        valid_mask = ~np.isnan(distances_pixels)
        if np.sum(valid_mask) > 0:
            mean_dist_pixels = np.nanmean(distances_pixels)
            mean_dist_world = np.nanmean(distances_world)
            min_dist_pixels = np.nanmin(distances_pixels)
            min_dist_world = np.nanmin(distances_world)
            
            self.logger.info(f"  Mean distance: {mean_dist_pixels:.1f} pixels, {mean_dist_world:.1f} world units")
            self.logger.info(f"  Min distance: {min_dist_pixels:.1f} pixels, {min_dist_world:.1f} world units")
        
        return {
            'fish_chaser_distance_pixels': distances_pixels,
            'fish_chaser_distance_world': distances_world,
            'relative_velocity': relative_velocities,
            'pursuit_angle': pursuit_angles,
            'fish_position_camera': fish_positions_camera,
            'fish_position_world': fish_positions_world,
            'chaser_position_camera': chaser_positions_camera,
            'chaser_position_world': chaser_positions_world,
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
        meta_group.attrs['camera_id'] = self.camera_id
        meta_group.attrs['coordinate_system'] = 'world' if self.use_world_coords else 'camera'
        meta_group.attrs['video_dimensions'] = [self.video_width, self.video_height]
        meta_group.attrs['fps_video'] = self.fps_video
        meta_group.attrs['fps_stimulus'] = self.fps_stimulus
        meta_group.attrs['updated_at'] = datetime.now().isoformat()
        
        # Store homography matrices
        if 'homography' not in meta_group:
            meta_group.create_dataset('homography', data=self.homography)
            meta_group.create_dataset('homography_inv', data=self.homography_inv)
        
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
            elif 'distance_world' in metric_name:
                dataset.attrs['units'] = 'world_units'
                dataset.attrs['description'] = 'Euclidean distance between fish and chaser in world coordinates'
            elif 'velocity' in metric_name:
                dataset.attrs['units'] = 'pixels/second'
                dataset.attrs['description'] = 'Rate of change of distance (negative = approaching)'
            elif 'pursuit_angle' in metric_name:
                dataset.attrs['units'] = 'radians'
                dataset.attrs['description'] = 'Angle difference between chaser heading and direction to fish'
            elif 'position_camera' in metric_name:
                dataset.attrs['units'] = 'pixels'
                dataset.attrs['description'] = 'Position in camera/pixel coordinates'
            elif 'position_world' in metric_name:
                dataset.attrs['units'] = 'world_units'
                dataset.attrs['description'] = 'Position in world/arena coordinates'
        
        # Calculate and save summary statistics
        self._save_summary_statistics(run_group, metrics)
        
        # Update latest run
        comp_group.attrs['latest'] = run_name
        
        self.logger.info(f"  ✅ Saved analysis for run: {run_name}")
        self.logger.info(f"  Zarr file: {self.zarr_path}")
    
    def _save_summary_statistics(self, group, metrics):
        """Calculate and save summary statistics."""
        distances_pixels = metrics['fish_chaser_distance_pixels']
        distances_world = metrics['fish_chaser_distance_world']
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
            
            # World coordinate statistics
            valid_world = distances_world[~np.isnan(distances_world)]
            if len(valid_world) > 0:
                stats['mean_distance_world'] = float(np.mean(valid_world))
                stats['min_distance_world'] = float(np.min(valid_world))
                stats['max_distance_world'] = float(np.max(valid_world))
            
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
                
                # Detect potential escape events (sudden increases in distance)
                if len(valid_velocities) > 10:
                    threshold = np.percentile(valid_velocities, 95)
                    escape_frames = np.where(velocities > threshold)[0]
                    stats['potential_escapes'] = len(escape_frames)
                    stats['escape_velocity_threshold'] = float(threshold)
        
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
    
    def plot_results(self, save_path: Optional[str] = None, frame_range: Optional[Tuple[int, int]] = None):
        """
        Plot analysis results for visualization.
        
        Args:
            save_path: Optional path to save the figure
            frame_range: Optional (start, end) frame indices to plot
        """
        # Load results
        run_name = self.interpolation_run or 'original'
        if 'chaser_comparison' not in self.zarr_root or run_name not in self.zarr_root['chaser_comparison']:
            self.logger.error("No analysis results found. Run save_analysis() first.")
            return
        
        results = self.zarr_root[f'chaser_comparison/{run_name}']
        
        # Load data
        distances_pixels = results['fish_chaser_distance_pixels'][:]
        distances_world = results['fish_chaser_distance_world'][:]
        velocities = results['relative_velocity'][:]
        fish_pos_cam = results['fish_position_camera'][:]
        chaser_pos_cam = results['chaser_position_camera'][:]
        
        # Apply frame range if specified
        if frame_range:
            start, end = frame_range
            distances_pixels = distances_pixels[start:end]
            distances_world = distances_world[start:end]
            velocities = velocities[start:end]
            fish_pos_cam = fish_pos_cam[start:end]
            chaser_pos_cam = chaser_pos_cam[start:end]
            frame_offset = start
        else:
            frame_offset = 0
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Distance over time
        ax1 = axes[0, 0]
        frames = np.arange(len(distances_pixels)) + frame_offset
        ax1.plot(frames, distances_pixels, 'b-', alpha=0.7, label='Distance (pixels)')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Distance (pixels)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        # Add world distance on secondary axis
        ax1_twin = ax1.twinx()
        ax1_twin.plot(frames, distances_world, 'r-', alpha=0.7, label='Distance (world)')
        ax1_twin.set_ylabel('Distance (world units)', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        ax1.set_title('Fish-Chaser Distance Over Time')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Plot 2: Velocity over time
        ax2 = axes[0, 1]
        ax2.plot(frames, velocities, 'g-', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Relative Velocity (pixels/second)')
        ax2.set_title('Relative Velocity (negative = approaching)')
        ax2.grid(True, alpha=0.3)
        
        # Highlight escape events
        if np.any(~np.isnan(velocities)):
            threshold = np.nanpercentile(velocities, 95)
            escape_frames = frames[velocities > threshold]
            escape_velocities = velocities[velocities > threshold]
            ax2.scatter(escape_frames, escape_velocities, c='r', s=50, alpha=0.5, label='Potential escapes')
            ax2.legend()
        
        # Plot 3: Spatial trajectories
        ax3 = axes[1, 0]
        valid_fish = ~np.isnan(fish_pos_cam[:, 0])
        valid_chaser = ~np.isnan(chaser_pos_cam[:, 0])
        
        # Plot trajectories
        ax3.plot(fish_pos_cam[valid_fish, 0], fish_pos_cam[valid_fish, 1], 
                'b-', alpha=0.5, linewidth=0.5, label='Fish')
        ax3.plot(chaser_pos_cam[valid_chaser, 0], chaser_pos_cam[valid_chaser, 1], 
                'r-', alpha=0.5, linewidth=0.5, label='Chaser')
        
        # Mark start and end positions
        if np.any(valid_fish):
            first_fish = np.where(valid_fish)[0][0]
            last_fish = np.where(valid_fish)[0][-1]
            ax3.scatter(fish_pos_cam[first_fish, 0], fish_pos_cam[first_fish, 1], 
                       c='b', s=100, marker='o', label='Fish start')
            ax3.scatter(fish_pos_cam[last_fish, 0], fish_pos_cam[last_fish, 1], 
                       c='b', s=100, marker='s', label='Fish end')
        
        if np.any(valid_chaser):
            first_chaser = np.where(valid_chaser)[0][0]
            last_chaser = np.where(valid_chaser)[0][-1]
            ax3.scatter(chaser_pos_cam[first_chaser, 0], chaser_pos_cam[first_chaser, 1], 
                       c='r', s=100, marker='o', label='Chaser start')
            ax3.scatter(chaser_pos_cam[last_chaser, 0], chaser_pos_cam[last_chaser, 1], 
                       c='r', s=100, marker='s', label='Chaser end')
        
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        ax3.set_title('Spatial Trajectories (Camera Coordinates)')
        ax3.set_aspect('equal')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Set video dimensions as limits
        ax3.set_xlim(0, self.video_width)
        ax3.set_ylim(0, self.video_height)
        ax3.invert_yaxis()  # Invert y-axis for image coordinates
        
        # Plot 4: Distance histogram
        ax4 = axes[1, 1]
        valid_distances = distances_pixels[~np.isnan(distances_pixels)]
        if len(valid_distances) > 0:
            ax4.hist(valid_distances, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax4.axvline(np.mean(valid_distances), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(valid_distances):.1f}')
            ax4.axvline(np.median(valid_distances), color='g', linestyle='--', 
                       label=f'Median: {np.median(valid_distances):.1f}')
            ax4.set_xlabel('Distance (pixels)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distance Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Chaser-Fish Analysis: {run_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"  Saved plot to: {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a text report of the analysis.
        
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("CHASER-FISH DISTANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # File information
        report.append("DATA SOURCES")
        report.append("-" * 40)
        report.append(f"Zarr file: {self.zarr_path}")
        report.append(f"H5 file: {self.h5_path}")
        report.append(f"Interpolation run: {self.interpolation_run or 'original'}")
        report.append("")
        
        # Coordinate system info
        report.append("COORDINATE SYSTEM")
        report.append("-" * 40)
        report.append(f"Camera ID: {self.camera_id}")
        report.append(f"Video dimensions: {self.video_width}x{self.video_height}")
        report.append(f"Video FPS: {self.fps_video}")
        report.append(f"Stimulus FPS: {self.fps_stimulus:.1f}")
        report.append(f"Homography determinant: {np.linalg.det(self.homography):.6f}")
        report.append(f"Coordinate system used: {'world' if self.use_world_coords else 'camera'}")
        report.append("")
        
        # Frame alignment
        report.append("FRAME ALIGNMENT")
        report.append("-" * 40)
        valid_alignments = np.sum(self.zarr_to_stimulus >= 0)
        total_frames = len(self.zarr_to_stimulus)
        report.append(f"Total frames: {total_frames}")
        report.append(f"Successfully aligned: {valid_alignments}")
        report.append(f"Alignment rate: {valid_alignments/total_frames*100:.1f}%")
        report.append("")
        
        # Analysis results
        run_name = self.interpolation_run or 'original'
        if 'chaser_comparison' in self.zarr_root and run_name in self.zarr_root['chaser_comparison']:
            results = self.zarr_root[f'chaser_comparison/{run_name}']
            
            if 'summary' in results.attrs:
                report.append("ANALYSIS RESULTS")
                report.append("-" * 40)
                summary = results.attrs['summary']
                
                for key, value in summary.items():
                    if isinstance(value, float):
                        report.append(f"{key}: {value:.2f}")
                    else:
                        report.append(f"{key}: {value}")
        else:
            report.append("No analysis results found")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Command-line interface for the analyzer."""
    parser = argparse.ArgumentParser(
        description='Analyze fish-chaser distances with coordinate transformation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s detections.zarr out_analysis.h5
  %(prog)s detections.zarr out_analysis.h5 --interpolation-run interp_linear_20240120
  %(prog)s detections.zarr out_analysis.h5 --plot
  %(prog)s detections.zarr out_analysis.h5 --use-world-coords
        """
    )
    
    parser.add_argument(
        'zarr_path',
        help='Path to YOLO detection zarr file'
    )
    parser.add_argument(
        'h5_path',
        help='Path to interpolated H5 file (from h5_frame_interpolator)'
    )
    parser.add_argument(
        '--interpolation-run',
        help='Specific interpolation run to use (default: latest)'
    )
    parser.add_argument(
        '--use-world-coords',
        action='store_true',
        help='Transform fish to world coordinates instead of chaser to camera'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate and show plots of the results'
    )
    parser.add_argument(
        '--save-plot',
        help='Path to save plot figure'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Print detailed report'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ChaserFishDistanceAnalyzer(
        zarr_path=args.zarr_path,
        h5_path=args.h5_path,
        interpolation_run=args.interpolation_run,
        use_world_coords=args.use_world_coords,
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