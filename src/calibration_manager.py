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
                                # Parse YAML to get homography matrix
                                # The YAML format might vary, so we'll be flexible
                                lines = yml_str.strip().split('\n')
                                matrix_data = []
                                reading_data = False
                                
                                for line in lines:
                                    if 'data:' in line:
                                        # Start reading data after this line
                                        data_str = line.split('data:')[1].strip()
                                        if data_str.startswith('['):
                                            # Data is inline
                                            data_str = data_str.strip('[]')
                                            matrix_data = [float(x.strip()) for x in data_str.split(',')]
                                        else:
                                            reading_data = True
                                    elif reading_data and line.strip():
                                        # Parse data lines
                                        values = line.strip().strip('[]').split(',')
                                        matrix_data.extend([float(x.strip()) for x in values if x.strip()])
                                
                                if len(matrix_data) == 9:
                                    calibration_data['homography_matrix'] = np.array(matrix_data).reshape(3, 3).tolist()
                                    if self.verbose:
                                        print(f"  Found homography matrix")
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
                    
                    # Extract chaser parameters if this is a chaser protocol
                    if protocol_data.get('steps'):
                        for step in protocol_data['steps']:
                            if step.get('stimulus_mode_str') == 'CHASER':
                                params = step.get('parameters', {})
                                calibration_data['chaser_params'] = {
                                    'chaser_radius_px': params.get('chasers', [{}])[0].get('radius_px'),
                                    'chaser_speed_pps': params.get('chasers', [{}])[0].get('speed_pps'),
                                    'chase_duration_s': params.get('chase_duration_s')
                                }
                                break
            
            # Extract video metadata for actual FPS calculation
            if '/video_metadata' in hf:
                video_meta = hf['/video_metadata']
                if 'frame_metadata' in video_meta:
                    frame_data = video_meta['frame_metadata']
                    if len(frame_data) > 100:
                        # Calculate actual frame rate from timestamps
                        timestamps = frame_data['timestamp_ns']
                        # Use middle section for stable estimate
                        mid_start = len(timestamps) // 4
                        mid_end = 3 * len(timestamps) // 4
                        time_diff = timestamps[mid_end] - timestamps[mid_start]
                        frame_diff = mid_end - mid_start
                        actual_fps = frame_diff / (time_diff / 1e9)
                        calibration_data['measured_fps'] = actual_fps
                        
                        if self.verbose:
                            print(f"  Calculated FPS from timestamps: {actual_fps:.1f}")
        
        return calibration_data
    
    def add_manual_calibration(self, 
                               pixel_to_mm: Optional[float] = None,
                               arena_diameter_mm: Optional[float] = None,
                               water_depth_mm: Optional[float] = None,
                               camera_model: Optional[str] = None,
                               rig_id: Optional[str] = None) -> Dict:
        """
        Add manual calibration values.
        
        Args:
            pixel_to_mm: Conversion factor from pixels to millimeters
            arena_diameter_mm: Physical arena diameter
            water_depth_mm: Water depth in arena
            camera_model: Camera model string
            rig_id: Rig identifier
            
        Returns:
            Dictionary of added calibration data
        """
        calibration_data = {}
        
        if pixel_to_mm is not None:
            calibration_data['pixel_to_mm'] = pixel_to_mm
            
        if arena_diameter_mm is not None:
            calibration_data['arena'] = calibration_data.get('arena', {})
            calibration_data['arena']['diameter_mm'] = arena_diameter_mm
            
        if water_depth_mm is not None:
            calibration_data['water_depth_mm'] = water_depth_mm
            
        if camera_model is not None:
            calibration_data['camera_model'] = camera_model
            
        if rig_id is not None:
            calibration_data['rig_info'] = calibration_data.get('rig_info', {})
            calibration_data['rig_info']['rig_id'] = rig_id
        
        return calibration_data
    
    def save_calibration(self, calibration_data: Dict, overwrite: bool = False) -> bool:
        """
        Save calibration data to zarr file.
        
        Args:
            calibration_data: Dictionary with calibration information
            overwrite: Whether to overwrite existing calibration
            
        Returns:
            Success status
        """
        if self.verbose:
            print("\nSaving calibration to zarr...")
        
        # Check if calibration exists
        if 'calibration' in self.root and not overwrite:
            print("Error: Calibration already exists. Use --overwrite to replace.")
            return False
        
        # Create or overwrite calibration group
        if 'calibration' in self.root:
            del self.root['calibration']
        
        calib_group = self.root.create_group('calibration')
        
        # Add timestamp
        calib_group.attrs['created_at'] = datetime.now().isoformat()
        
        # Save pixel to mm conversion (most important!)
        if 'pixel_to_mm' in calibration_data:
            calib_group.attrs['pixel_to_mm'] = float(calibration_data['pixel_to_mm'])
            if self.verbose:
                print(f"  ✓ pixel_to_mm: {calibration_data['pixel_to_mm']:.6f}")
        
        # Save arena information
        if 'arena' in calibration_data:
            arena_group = calib_group.create_group('arena')
            for key, value in calibration_data['arena'].items():
                if value is not None:
                    arena_group.attrs[key] = value
            if self.verbose:
                print(f"  ✓ Arena data saved")
        
        # Save rig information
        if 'rig_info' in calibration_data:
            rig_group = calib_group.create_group('rig_info')
            for key, value in calibration_data['rig_info'].items():
                if value is not None:
                    rig_group.attrs[key] = value
            if self.verbose:
                print(f"  ✓ Rig info saved")
        
        # Save homography matrix if available
        if 'homography_matrix' in calibration_data:
            calib_group.create_dataset('homography_matrix', 
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
    
    # Extract from H5 if provided
    if args.h5_path:
        calibration_data = manager.extract_from_h5(args.h5_path)
    
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