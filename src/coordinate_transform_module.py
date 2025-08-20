#!/usr/bin/env python3
"""
Coordinate Transformation Module

Handles transformations between different coordinate systems:
- Texture/Stimulus space: Where chaser/target positions are defined
- Camera space: Raw camera pixel coordinates (e.g., 4512x4512)
- Projector space: Sub-arena projection coordinates
- World space: Physical arena coordinates (mm)

This module auto-detects the coordinate system from arena configuration
and provides the appropriate transformations.
"""

import numpy as np
import json
import h5py
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging


@dataclass
class CoordinateTransform:
    """Stores transformation parameters between coordinate systems."""
    source_space: str
    target_space: str
    scale_x: float
    scale_y: float
    offset_x: float = 0.0
    offset_y: float = 0.0
    source_dimensions: Tuple[float, float] = None
    target_dimensions: Tuple[float, float] = None
    
    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply transformation to coordinates."""
        tx = x * self.scale_x + self.offset_x
        ty = y * self.scale_y + self.offset_y
        return tx, ty
    
    def inverse_transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply inverse transformation."""
        ix = (x - self.offset_x) / self.scale_x
        iy = (y - self.offset_y) / self.scale_y
        return ix, iy


class CoordinateSystem:
    """
    Manages coordinate transformations for the experiment.
    
    Auto-detects coordinate systems from H5 arena configuration and
    provides transformations between texture, camera, and projector spaces.
    """
    
    def __init__(self, h5_path: str, verbose: bool = True):
        """
        Initialize coordinate system from H5 file.
        
        Args:
            h5_path: Path to H5 file with arena configuration
            verbose: Enable verbose logging
        """
        self.h5_path = h5_path
        self.verbose = verbose
        self.logger = self._setup_logger()
        
        # Load configuration
        self.arena_config = None
        self.texture_dims = None
        self.camera_dims = None
        self.projector_dims = None
        self.transforms = {}
        
        self._load_configuration()
        self._setup_transformations()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(__name__)
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        return logger
    
    def _load_configuration(self):
        """Load arena configuration from H5 file."""
        with h5py.File(self.h5_path, 'r') as h5f:
            # Load arena configuration JSON
            if '/calibration_snapshot/arena_config_json' in h5f:
                config_str = h5f['/calibration_snapshot/arena_config_json'][()].decode('utf-8')
                self.arena_config = json.loads(config_str)
                self.logger.info(f"Loaded arena config: {self.arena_config.get('config_name', 'unnamed')}")
            else:
                raise ValueError("No arena configuration found in H5 file")
            
            # Extract dimensions
            self._extract_dimensions(h5f)
    
    def _extract_dimensions(self, h5f):
        """Extract dimension information from configuration."""
        # Projector/sub-arena dimensions
        self.projector_dims = (
            self.arena_config['sub_arena_width_px'],
            self.arena_config['sub_arena_height_px']
        )
        self.projector_offset = (
            self.arena_config['sub_arena_x_px'],
            self.arena_config['sub_arena_y_px']
        )
        
        # Camera dimensions
        if 'camera_calibrations' in self.arena_config:
            cam_calib = self.arena_config['camera_calibrations'][0]
            self.camera_dims = (
                cam_calib['native_width_px'],
                cam_calib['native_height_px']
            )
            self.camera_id = cam_calib['camera_id']
        else:
            # Default to common camera resolution
            self.camera_dims = (4512, 4512)
            self.camera_id = 'unknown'
        
        # Texture dimensions - inferred from chaser positions or sub-arena
        self.texture_dims = self._infer_texture_dimensions(h5f)
        
        self.logger.info(f"Coordinate spaces detected:")
        self.logger.info(f"  Texture: {self.texture_dims[0]}x{self.texture_dims[1]}")
        self.logger.info(f"  Camera: {self.camera_dims[0]}x{self.camera_dims[1]}")
        self.logger.info(f"  Projector: {self.projector_dims[0]}x{self.projector_dims[1]}")
    
    def _infer_texture_dimensions(self, h5f) -> Tuple[float, float]:
        """Infer texture dimensions from chaser positions or configuration."""
        # Method 1: Check if texture dimensions match sub-arena
        # This is the most common case
        if self.projector_dims[0] == self.projector_dims[1]:  # Square sub-arena
            # Check chaser positions to confirm
            if '/tracking_data/chaser_states' in h5f:
                chaser_states = h5f['/tracking_data/chaser_states'][:100]  # Sample
                if len(chaser_states) > 0:
                    chaser_x = chaser_states['chaser_pos_x']
                    chaser_y = chaser_states['chaser_pos_y']
                    
                    # If chaser is at constant position near half of sub-arena dimension
                    mean_x = np.mean(chaser_x)
                    mean_y = np.mean(chaser_y)
                    
                    # Check if mean is approximately half of sub-arena
                    expected_center = self.projector_dims[0] / 2
                    if abs(mean_x - expected_center) < 10 and abs(mean_y - expected_center) < 10:
                        # Texture dimensions match sub-arena
                        return self.projector_dims
                    
                    # Check if chaser at 179 implies 358x358 texture
                    if abs(mean_x - 179) < 1 and abs(mean_y - 179) < 1:
                        return (358, 358)
        
        # Default: assume texture matches sub-arena
        return self.projector_dims
    
    def _setup_transformations(self):
        """Setup transformation matrices between coordinate spaces."""
        # Texture to Camera transformation
        # Texture center maps to camera center
        texture_center_x = self.texture_dims[0] / 2
        texture_center_y = self.texture_dims[1] / 2
        camera_center_x = self.camera_dims[0] / 2
        camera_center_y = self.camera_dims[1] / 2
        
        # Calculate scale for texture->camera
        # Assuming texture fills the camera view
        scale_x = self.camera_dims[0] / self.texture_dims[0]
        scale_y = self.camera_dims[1] / self.texture_dims[1]
        
        self.transforms['texture_to_camera'] = CoordinateTransform(
            source_space='texture',
            target_space='camera',
            scale_x=scale_x,
            scale_y=scale_y,
            offset_x=0.0,
            offset_y=0.0,
            source_dimensions=self.texture_dims,
            target_dimensions=self.camera_dims
        )
        
        self.logger.info(f"Transformation: texture -> camera")
        self.logger.info(f"  Scale: ({scale_x:.4f}, {scale_y:.4f})")
        
        # Test the transformation
        test_x, test_y = self.transforms['texture_to_camera'].transform(
            texture_center_x, texture_center_y
        )
        self.logger.info(f"  Test: ({texture_center_x}, {texture_center_y}) -> ({test_x:.1f}, {test_y:.1f})")
        self.logger.info(f"  Expected: ({camera_center_x:.1f}, {camera_center_y:.1f})")
    
    def transform_coordinates(self, 
                             x: np.ndarray, 
                             y: np.ndarray,
                             from_space: str,
                             to_space: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform coordinates between spaces.
        
        Args:
            x, y: Input coordinates
            from_space: Source coordinate space ('texture', 'camera', 'projector')
            to_space: Target coordinate space
            
        Returns:
            Transformed (x, y) coordinates
        """
        transform_key = f"{from_space}_to_{to_space}"
        
        if transform_key in self.transforms:
            return self.transforms[transform_key].transform(x, y)
        
        # Try inverse transformation
        inverse_key = f"{to_space}_to_{from_space}"
        if inverse_key in self.transforms:
            return self.transforms[inverse_key].inverse_transform(x, y)
        
        raise ValueError(f"No transformation available from {from_space} to {to_space}")
    
    def get_transform_params(self) -> Dict[str, Any]:
        """Get transformation parameters for saving."""
        params = {
            'texture_dimensions': self.texture_dims,
            'camera_dimensions': self.camera_dims,
            'projector_dimensions': self.projector_dims,
            'camera_id': self.camera_id,
            'transforms': {}
        }
        
        for key, transform in self.transforms.items():
            params['transforms'][key] = {
                'scale_x': transform.scale_x,
                'scale_y': transform.scale_y,
                'offset_x': transform.offset_x,
                'offset_y': transform.offset_y
            }
        
        return params


def apply_coordinate_transform_to_analyzer(analyzer_instance, h5_path: str):
    """
    Apply coordinate transformation to an existing analyzer instance.
    
    This function can be called from chaser_fish_analyzer.py to set up
    proper coordinate transformations.
    
    Args:
        analyzer_instance: Instance of ChaserFishDistanceAnalyzer
        h5_path: Path to H5 file with configuration
    """
    # Create coordinate system
    coord_sys = CoordinateSystem(h5_path)
    
    # Store in analyzer
    analyzer_instance.coord_system = coord_sys
    
    # Override the transformation methods
    def transform_chaser_to_camera(chaser_x, chaser_y):
        """Transform chaser from texture to camera coordinates."""
        return coord_sys.transform_coordinates(
            chaser_x, chaser_y, 
            from_space='texture', 
            to_space='camera'
        )
    
    analyzer_instance.transform_chaser_to_camera = transform_chaser_to_camera
    
    # Log the setup
    analyzer_instance.logger.info("Coordinate transformation configured:")
    analyzer_instance.logger.info(f"  Texture space: {coord_sys.texture_dims}")
    analyzer_instance.logger.info(f"  Camera space: {coord_sys.camera_dims}")
    
    return coord_sys


# Standalone utility functions
def verify_coordinate_transformation(h5_path: str, 
                                    test_points: Optional[list] = None) -> bool:
    """
    Verify coordinate transformation is working correctly.
    
    Args:
        h5_path: Path to H5 file
        test_points: Optional list of (x, y, label) tuples to test
        
    Returns:
        True if transformation appears correct
    """
    coord_sys = CoordinateSystem(h5_path)
    
    if test_points is None:
        # Default test points
        test_points = [
            (179, 179, "texture_center"),
            (0, 0, "texture_origin"),
            (coord_sys.texture_dims[0], coord_sys.texture_dims[1], "texture_max")
        ]
    
    print("\n" + "=" * 60)
    print("COORDINATE TRANSFORMATION VERIFICATION")
    print("=" * 60)
    
    all_valid = True
    for tx, ty, label in test_points:
        cx, cy = coord_sys.transform_coordinates(
            np.array([tx]), np.array([ty]),
            from_space='texture',
            to_space='camera'
        )
        
        print(f"\n{label}:")
        print(f"  Texture: ({tx:.1f}, {ty:.1f})")
        print(f"  Camera: ({cx[0]:.1f}, {cy[0]:.1f})")
        
        # Validate bounds
        if cx[0] < 0 or cx[0] > coord_sys.camera_dims[0]:
            print(f"  ⚠️ WARNING: X coordinate out of camera bounds!")
            all_valid = False
        if cy[0] < 0 or cy[0] > coord_sys.camera_dims[1]:
            print(f"  ⚠️ WARNING: Y coordinate out of camera bounds!")
            all_valid = False
    
    if all_valid:
        print("\n✅ All transformations within valid bounds")
    else:
        print("\n⚠️ Some transformations may be incorrect")
    
    return all_valid


def main():
    """Test the coordinate transformation system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test coordinate transformation system'
    )
    parser.add_argument('h5_path', help='Path to H5 file with arena configuration')
    parser.add_argument('--verify', action='store_true', 
                       help='Run verification tests')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_coordinate_transformation(args.h5_path)
    else:
        # Just load and display info
        coord_sys = CoordinateSystem(args.h5_path, verbose=True)
        print(f"\nCoordinate system loaded successfully")
        print(f"Available transformations: {list(coord_sys.transforms.keys())}")


if __name__ == '__main__':
    main()