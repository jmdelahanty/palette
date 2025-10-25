#!/usr/bin/env python3
"""
Calibration Data Manager

Extracts calibration data from experiment H5 files and adds it to detection zarr files.
This enables conversion from pixels to physical units (mm).
"""

import zarr
import h5py
import numpy as np
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any


# Custom YAML constructor for OpenCV matrices
def opencv_matrix_constructor(loader, node):
    """Custom constructor for OpenCV matrix YAML format."""
    mapping = loader.construct_mapping(node, deep=True)
    return mapping

# Register the custom constructor for the opencv-matrix tag
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:opencv-matrix', opencv_matrix_constructor)


class CalibrationManager:
    """Manage calibration data for fish tracking experiments."""
    
    def __init__(self, zarr_path: str, verbose: bool = True):
        """
        Initialize calibration manager.
        
        Args:
            zarr_path: Path to zarr file
            verbose: Print progress messages
        """
        self.zarr_path = Path(zarr_path)
        self.root = zarr.open(str(self.zarr_path), mode='r+')
        self.verbose = verbose
    
    def find_default_h5(self) -> Optional[Path]:
        """
        Locate an experiment H5 file in the parent directory of the zarr.
        
        Preference order:
            1. Exact stem match to the zarr folder name (sans .zarr).
            2. Single file whose name contains the zarr stem.
            3. Most recently modified *.h5 file.
        """
        parent_dir = self.zarr_path.parent
        if not parent_dir.exists():
            return None
        
        zarr_stem = self.zarr_path.name
        if zarr_stem.endswith(".zarr"):
            zarr_stem = zarr_stem[:-5]
        
        candidates = sorted(parent_dir.glob("*.h5"))
        if not candidates:
            return None
        
        exact = [p for p in candidates if p.stem == zarr_stem]
        if exact:
            return exact[0]
        
        contains = [p for p in candidates if zarr_stem in p.stem]
        if len(contains) == 1:
            return contains[0]
        
        # Fallback to most recently modified file
        try:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        except Exception:
            return candidates[0]
        
    def extract_from_h5(self, h5_path: str) -> Dict[str, Any]:
        """
        Extract calibration data from experiment H5 file.
        
        Args:
            h5_path: Path to experiment H5 file
            
        Returns:
            Dictionary with calibration data
        """
        if self.verbose:
            print(f"\nExtracting calibration from: {Path(h5_path).name}")
        
        calibration_data = {}
        
        with h5py.File(h5_path, 'r') as hf:
            # Extract arena configuration from calibration_snapshot
            if '/calibration_snapshot' in hf:
                calib_group = hf['/calibration_snapshot']
                
                # Get arena config JSON
                if 'arena_config_json' in calib_group:
                    arena_json = calib_group['arena_config_json'][()].decode('utf-8')
                    arena_config = json.loads(arena_json)
                    
                    if self.verbose:
                        print(f"  Found arena config")
                    
                    # Extract camera calibration data
                    if 'camera_calibrations' in arena_config:
                        for cam_cal in arena_config['camera_calibrations']:
                            # Extract pixel to mm conversion - THIS IS THE KEY VALUE!
                            if 'pixels_per_mm_camera' in cam_cal:
                                pixels_per_mm = cam_cal['pixels_per_mm_camera']
                                calibration_data['pixel_to_mm'] = 1.0 / pixels_per_mm
                                calibration_data['pixels_per_mm'] = pixels_per_mm
                                
                                if self.verbose:
                                    print(f"  Found pixels_per_mm_camera: {pixels_per_mm:.2f}")
                            
                            # Extract other camera info
                            calibration_data['camera_info'] = {
                                'camera_id': cam_cal.get('camera_id', 'unknown'),
                                'native_width_px': cam_cal.get('native_width_px'),
                                'native_height_px': cam_cal.get('native_height_px'),
                                'pixels_per_mm_projector': cam_cal.get('pixels_per_mm_projector'),
                                'real_world_ref_mm': cam_cal.get('real_world_ref_mm')
                            }
                            
                            # Sub-arena info (the tracked region)
                            calibration_data['sub_arena'] = {
                                'width_px': cam_cal.get('sub_arena_width_px'),
                                'height_px': cam_cal.get('sub_arena_height_px'),
                                'x_px': cam_cal.get('sub_arena_x_px'),
                                'y_px': cam_cal.get('sub_arena_y_px')
                            }
                            break  # Use first camera
                    
                    # Extract swimmable area info
                    calibration_data['swimmable_area'] = {
                        'shape': arena_config.get('swimmable_area_shape', 'CIRCLE'),
                        'center_x_px': arena_config.get('swimmable_area_center_x_px'),
                        'center_y_px': arena_config.get('swimmable_area_center_y_px'),
                        'radius_px': arena_config.get('swimmable_area_radius_px')
                    }
                    
                    # Calculate arena diameter in mm if we have the data
                    if 'swimmable_area_radius_px' in arena_config and 'pixel_to_mm' in calibration_data:
                        radius_px = arena_config['swimmable_area_radius_px']
                        radius_mm = radius_px * calibration_data['pixel_to_mm']
                        calibration_data['arena_diameter_mm'] = radius_mm * 2
                        
                        if self.verbose:
                            print(f"  Calculated arena diameter: {radius_mm * 2:.1f} mm")
                
                # Get camera-specific data (homography, etc.)
                for item_name in calib_group.keys():
                    # Check if this is a camera ID (just numbers)
                    if item_name.isdigit():
                        cam_group = calib_group[item_name]

                        # Extract homography matrix if available
                        if 'homography_matrix_yml' in cam_group:
                            yml_str = cam_group['homography_matrix_yml'][()].decode('utf-8')
                            try:
                                cleaned_lines = []
                                for line in yml_str.splitlines():
                                    stripped = line.lstrip()
                                    if stripped.startswith('%YAML') or (stripped == '---' and not cleaned_lines):
                                        continue
                                    cleaned_lines.append(line)

                                cleaned_text = '\n'.join(cleaned_lines)
                                yaml_data = yaml.safe_load(cleaned_text) or {}

                                matrix_meta = yaml_data.get('homography_matrix')
                                if isinstance(matrix_meta, dict):
                                    rows = int(matrix_meta.get('rows', 0))
                                    cols = int(matrix_meta.get('cols', 0))
                                    data = matrix_meta.get('data')
                                    if rows > 0 and cols > 0 and isinstance(data, (list, tuple)):
                                        matrix = np.array(data, dtype=float).reshape(rows, cols)
                                        calibration_data['homography_matrix'] = matrix.tolist()
                                        calibration_data['homography_metadata'] = {
                                            'rows': rows,
                                            'cols': cols,
                                            'dt': matrix_meta.get('dt'),
                                            'calibration_timestamp_utc': yaml_data.get('calibration_timestamp_utc')
                                        }
                                        if self.verbose:
                                            print("  ✓ Parsed homography matrix")
                                    else:
                                        if self.verbose:
                                            print("  Homography YAML missing matrix data")
                                else:
                                    if self.verbose:
                                        print("  Unexpected homography YAML structure")
                            except Exception as e:
                                if self.verbose:
                                    print(f"  Could not parse homography matrix: {e}")

                        break  # Use first camera found
            
            # Extract session metadata from root attributes
            root_attrs = dict(hf.attrs)
            for key, value in root_attrs.items():
                if isinstance(value, bytes):
                    root_attrs[key] = value.decode('utf-8')
            
            # Extract rig information
            calibration_data['rig_info'] = {
                'rig_id': root_attrs.get('rig_id', 'unknown'),
                'arena_id': root_attrs.get('arena_id', 'unknown'),
                'session_id': root_attrs.get('session_uuid', 'unknown'),
                'hostname': root_attrs.get('hostname', 'unknown'),
                'software_version': root_attrs.get('software_version', 'unknown'),
                'session_start': root_attrs.get('session_start_iso8601_utc', 'unknown')
            }
            
            if self.verbose:
                print(f"  Found rig_id: {calibration_data['rig_info']['rig_id']}")
            
            # Extract protocol information
            if '/protocol_snapshot' in hf:
                protocol_group = hf['/protocol_snapshot']
                if 'protocol_definition_json' in protocol_group:
                    protocol_json = protocol_group['protocol_definition_json'][()].decode('utf-8')
                    protocol_data = json.loads(protocol_json)
                    
                    calibration_data['protocol_info'] = {
                        'name': protocol_data.get('protocol_name', 'unknown'),
                        'steps': len(protocol_data.get('steps', []))
                    }
            
            # Extract subject metadata
            if '/subject_metadata' in hf:
                subject_group = hf['/subject_metadata']
                calibration_data['subject_metadata'] = dict(subject_group.attrs)
                for key, value in calibration_data['subject_metadata'].items():
                    if isinstance(value, bytes):
                        calibration_data['subject_metadata'][key] = value.decode('utf-8')
            
            # Estimate FPS from frame metadata if available
            if '/video_metadata/frame_metadata' in hf:
                frame_ds = hf['/video_metadata/frame_metadata']
                if len(frame_ds) > 1:
                    timestamps = frame_ds['timestamp_ns']
                    # Calculate average FPS from timestamp differences
                    time_diffs = np.diff(timestamps)
                    avg_time_diff_ns = np.mean(time_diffs)
                    avg_fps = 1e9 / avg_time_diff_ns
                    calibration_data['measured_fps'] = float(avg_fps)
                    
                    if self.verbose:
                        print(f"  Calculated FPS from timestamps: {avg_fps:.1f}")
        
        return calibration_data
    
    def add_manual_calibration(self, 
                              pixel_to_mm: Optional[float] = None,
                              arena_diameter_mm: Optional[float] = None,
                              water_depth_mm: Optional[float] = None,
                              camera_model: Optional[str] = None,
                              rig_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add manual calibration values.
        
        Args:
            pixel_to_mm: Conversion factor from pixels to mm
            arena_diameter_mm: Arena diameter in mm
            water_depth_mm: Water depth in mm
            camera_model: Camera model string
            rig_id: Rig identifier
            
        Returns:
            Dictionary with manual calibration values
        """
        manual_data = {}
        
        if pixel_to_mm is not None:
            manual_data['pixel_to_mm'] = pixel_to_mm
            manual_data['pixels_per_mm'] = 1.0 / pixel_to_mm
            
        if arena_diameter_mm is not None:
            if 'arena' not in manual_data:
                manual_data['arena'] = {}
            manual_data['arena']['diameter_mm'] = arena_diameter_mm
            
        if water_depth_mm is not None:
            manual_data['water_depth_mm'] = water_depth_mm
            
        if camera_model is not None:
            manual_data['camera_model'] = camera_model
            
        if rig_id is not None:
            if 'rig_info' not in manual_data:
                manual_data['rig_info'] = {}
            manual_data['rig_info']['rig_id'] = rig_id
        
        return manual_data
    
    def save_calibration(self, calibration_data: Dict[str, Any], overwrite: bool = False) -> bool:
        """
        Save calibration data to zarr.
        
        Args:
            calibration_data: Dictionary with calibration data
            overwrite: Whether to overwrite existing calibration
            
        Returns:
            True if successful
        """
        if 'calibration' in self.root:
            if not overwrite:
                print("Calibration already exists. Use --overwrite to replace.")
                return False
            del self.root['calibration']
        
        if self.verbose:
            print("\nSaving calibration to zarr...")
        
        calib_group = self.root.create_group('calibration')
        
        # Save the most critical value
        if 'pixel_to_mm' in calibration_data:
            calib_group.attrs['pixel_to_mm'] = calibration_data['pixel_to_mm']
            if self.verbose:
                print(f"  ✓ pixel_to_mm: {calibration_data['pixel_to_mm']:.6f}")
        
        # Save pixels_per_mm for convenience
        if 'pixels_per_mm' in calibration_data:
            calib_group.attrs['pixels_per_mm'] = calibration_data['pixels_per_mm']
        
        # Save arena info
        if 'swimmable_area' in calibration_data or 'arena_diameter_mm' in calibration_data:
            arena_group = calib_group.create_group('arena')
            
            if 'swimmable_area' in calibration_data:
                for key, value in calibration_data['swimmable_area'].items():
                    if value is not None:
                        arena_group.attrs[key] = value
            
            if 'arena_diameter_mm' in calibration_data:
                arena_group.attrs['diameter_mm'] = calibration_data['arena_diameter_mm']
        
        # Save rig info
        if 'rig_info' in calibration_data:
            rig_group = calib_group.create_group('rig_info')
            for key, value in calibration_data['rig_info'].items():
                if value is not None:
                    rig_group.attrs[key] = value
            if self.verbose:
                print(f"  ✓ Rig info saved")
        
        # Save homography matrix if available
        if 'homography_matrix' in calibration_data:
            calib_group.create_array('homography_matrix', 
                                      data=np.array(calibration_data['homography_matrix']))
            if self.verbose:
                print(f"  ✓ Homography matrix saved")
        
        # Save other calibration data
        for key in ['water_depth_mm', 'camera_model', 'measured_fps', 'camera_id']:
            if key in calibration_data:
                calib_group.attrs[key] = calibration_data[key]
                if self.verbose:
                    print(f"  ✓ {key}: {calibration_data[key]}")
        
        # Save subject metadata if available
        if 'subject_metadata' in calibration_data:
            subject_group = calib_group.create_group('subject_metadata')
            for key, value in calibration_data['subject_metadata'].items():
                subject_group.attrs[key] = value
        
        # Save protocol info if available
        if 'protocol_info' in calibration_data:
            protocol_group = calib_group.create_group('protocol_info')
            for key, value in calibration_data['protocol_info'].items():
                protocol_group.attrs[key] = value
        
        if self.verbose:
            print(f"\n✓ Calibration saved to {self.zarr_path}/calibration")
        
        return True
    
    def get_calibration(self) -> Optional[Dict]:
        """
        Get existing calibration data from zarr.
        
        Returns:
            Dictionary with calibration data or None if not found
        """
        if 'calibration' not in self.root:
            return None
        
        calib_group = self.root['calibration']
        calibration_data = dict(calib_group.attrs)
        
        # Load subgroups
        for group_name in ['arena', 'rig_info', 'subject_metadata', 'protocol_info']:
            if group_name in calib_group:
                calibration_data[group_name] = dict(calib_group[group_name].attrs)
        
        # Load homography matrix if present
        if 'homography_matrix' in calib_group:
            calibration_data['homography_matrix'] = calib_group['homography_matrix'][:]
        
        return calibration_data
    
    def print_calibration(self):
        """Print existing calibration data."""
        calibration = self.get_calibration()
        
        if calibration is None:
            print("No calibration data found.")
            return
        
        print("\n" + "="*60)
        print("CALIBRATION DATA")
        print("="*60)
        
        # Most important metric
        if 'pixel_to_mm' in calibration:
            print(f"\n📏 Pixel to mm conversion: {calibration['pixel_to_mm']:.6f}")
            print(f"   (1 pixel = {calibration['pixel_to_mm']:.4f} mm)")
            print(f"   (1 mm = {1/calibration['pixel_to_mm']:.2f} pixels)")
        
        # Arena info
        if 'arena' in calibration:
            print(f"\n🎯 Arena:")
            for key, value in calibration['arena'].items():
                print(f"   {key}: {value}")
        
        # Rig info
        if 'rig_info' in calibration:
            print(f"\n🔬 Rig:")
            for key, value in calibration['rig_info'].items():
                print(f"   {key}: {value}")
        
        # Other measurements
        for key in ['water_depth_mm', 'camera_model', 'measured_fps', 'camera_id']:
            if key in calibration:
                print(f"\n{key}: {calibration[key]}")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Manage calibration data for fish tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract calibration from H5 file
  %(prog)s detections.zarr --from-h5 experiment.h5
  
  # Add manual calibration
  %(prog)s detections.zarr --pixel-to-mm 0.0265 --arena-diameter 50
  
  # View existing calibration
  %(prog)s detections.zarr --show
  
  # Overwrite existing calibration
  %(prog)s detections.zarr --from-h5 experiment.h5 --overwrite
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    
    parser.add_argument('--from-h5', dest='h5_path',
                       help='Extract calibration from experiment H5 file')
    
    parser.add_argument('--pixel-to-mm', type=float,
                       help='Manual pixel to mm conversion factor')
    
    parser.add_argument('--arena-diameter', type=float, dest='arena_diameter_mm',
                       help='Arena diameter in mm')
    
    parser.add_argument('--water-depth', type=float, dest='water_depth_mm',
                       help='Water depth in mm')
    
    parser.add_argument('--camera-model',
                       help='Camera model string')
    
    parser.add_argument('--rig-id',
                       help='Rig identifier')
    
    parser.add_argument('--show', action='store_true',
                       help='Show existing calibration')
    
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing calibration')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = CalibrationManager(args.zarr_path)
    
    if args.show:
        manager.print_calibration()
        return 0
    
    calibration_data = {}
    
    # Resolve H5 source (CLI overrides auto-discovery)
    h5_path = args.h5_path
    if not h5_path:
        auto_h5 = manager.find_default_h5()
        if auto_h5:
            h5_path = str(auto_h5)
            if manager.verbose:
                print(f"Auto-detected calibration H5: {auto_h5}")
        else:
            if manager.verbose:
                print("No H5 file found alongside the zarr; use --from-h5 to specify one.")
    
    if h5_path:
        calibration_data = manager.extract_from_h5(h5_path)
    
    # Add/override with manual values
    manual_data = manager.add_manual_calibration(
        pixel_to_mm=args.pixel_to_mm,
        arena_diameter_mm=args.arena_diameter_mm,
        water_depth_mm=args.water_depth_mm,
        camera_model=args.camera_model,
        rig_id=args.rig_id
    )
    calibration_data.update(manual_data)
    
    # Save if we have data
    if calibration_data:
        success = manager.save_calibration(calibration_data, overwrite=args.overwrite)
        if success:
            manager.print_calibration()
    else:
        print("No calibration data to save. Use --from-h5 or provide manual values.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
