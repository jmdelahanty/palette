#!/usr/bin/env python3
"""
Test script to verify homography matrix is loadable and usable in the analysis H5 file.
Handles both direct matrix format and YAML format in calibration_snapshot.
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
import yaml
import re


def parse_homography_from_yaml(yaml_string):
    """Parse homography matrix from YAML string."""
    # Try to parse as YAML
    try:
        data = yaml.safe_load(yaml_string)
        
        # Check different possible YAML structures
        if isinstance(data, dict):
            # Look for common keys
            for key in ['homography', 'homography_matrix', 'matrix', 'H']:
                if key in data:
                    matrix_data = data[key]
                    if isinstance(matrix_data, list):
                        return np.array(matrix_data).reshape(3, 3)
                    elif isinstance(matrix_data, np.ndarray):
                        return matrix_data.reshape(3, 3)
        
        # If it's a direct list (unlikely but possible)
        if isinstance(data, list) and len(data) == 9:
            return np.array(data).reshape(3, 3)
            
    except yaml.YAMLError:
        pass
    
    # Try regex pattern matching for OpenCV YAML format
    # Looking for patterns like: data: [ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]
    data_match = re.search(r'data:\s*\[([\d\s,.\-e]+)\]', yaml_string)
    if data_match:
        values_str = data_match.group(1)
        values = [float(x.strip()) for x in values_str.split(',')]
        if len(values) == 9:
            return np.array(values).reshape(3, 3)
    
    # Try line-by-line parsing for matrix format
    lines = yaml_string.strip().split('\n')
    matrix_values = []
    for line in lines:
        # Look for lines with numbers
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
        if numbers:
            matrix_values.extend([float(n) for n in numbers])
    
    if len(matrix_values) == 9:
        return np.array(matrix_values).reshape(3, 3)
    
    raise ValueError(f"Could not parse homography matrix from YAML. Found {len(matrix_values)} values instead of 9")


def test_homography(h5_path):
    """Test that homography matrix exists and is valid in the H5 file."""
    
    print(f"\n🔍 Testing homography in: {h5_path}")
    print("=" * 60)
    
    try:
        with h5py.File(h5_path, 'r') as f:
            homography = None
            source_path = None
            
            # First check for direct homography dataset
            if '/homography' in f:
                homography = f['/homography'][:]
                source_path = '/homography'
                print("✅ Found direct homography dataset")
            
            # Check for homography in calibration_snapshot
            elif '/calibration_snapshot' in f:
                calib = f['/calibration_snapshot']
                
                # Find camera groups
                camera_ids = [k for k in calib.keys() if isinstance(calib[k], h5py.Group)]
                
                if camera_ids:
                    # Use first camera found
                    cam_id = camera_ids[0]
                    print(f"📷 Found camera calibration: {cam_id}")
                    
                    if f'homography_matrix_yml' in calib[cam_id]:
                        yaml_data = calib[cam_id]['homography_matrix_yml'][()].decode('utf-8')
                        print(f"   Found homography YAML (length: {len(yaml_data)} chars)")
                        
                        # Parse the YAML to get the matrix
                        try:
                            homography = parse_homography_from_yaml(yaml_data)
                            source_path = f'/calibration_snapshot/{cam_id}/homography_matrix_yml'
                            print("✅ Successfully parsed homography from YAML")
                        except Exception as e:
                            print(f"❌ Failed to parse YAML: {e}")
                            print("\nYAML content (first 500 chars):")
                            print(yaml_data[:500])
                            return False
                    else:
                        print(f"❌ No homography_matrix_yml found for camera {cam_id}")
                        print(f"   Available datasets: {list(calib[cam_id].keys())}")
                else:
                    print("❌ No camera groups found in calibration_snapshot")
                    print(f"   Available items: {list(calib.keys())}")
            
            if homography is None:
                print("❌ FAIL: Homography not found in any expected location!")
                print("   Available root groups:", list(f.keys()))
                return False
            
            print(f"📍 Homography source: {source_path}")
            
            # Check shape
            print(f"\n📐 Matrix Properties:")
            print(f"   Shape: {homography.shape}")
            print(f"   Dtype: {homography.dtype}")
            
            # Validate it's a 3x3 matrix
            if homography.shape != (3, 3):
                print(f"❌ FAIL: Expected 3x3 matrix, got {homography.shape}")
                return False
            print("✅ Correct 3x3 shape")
            
            # Check for NaN or Inf values
            if np.any(np.isnan(homography)):
                print("❌ FAIL: Matrix contains NaN values")
                return False
            if np.any(np.isinf(homography)):
                print("❌ FAIL: Matrix contains Inf values")
                return False
            print("✅ No NaN or Inf values")
            
            # Check matrix is invertible (determinant != 0)
            det = np.linalg.det(homography)
            print(f"\n🔢 Matrix Analysis:")
            print(f"   Determinant: {det:.6f}")
            
            if abs(det) < 1e-10:
                print("❌ FAIL: Matrix is singular (not invertible)")
                return False
            print("✅ Matrix is invertible")
            
            # Display the matrix
            print(f"\n📊 Homography Matrix:")
            for i, row in enumerate(homography):
                print(f"   [{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f}]")
            
            # Test transformation of sample points
            print(f"\n🎯 Testing Coordinate Transformation:")
            
            # Test corners of a typical image (assuming ~1920x1080 or similar)
            test_points = [
                [0, 0, 1],           # Top-left
                [1920, 0, 1],        # Top-right
                [0, 1080, 1],        # Bottom-left
                [1920, 1080, 1],     # Bottom-right
                [960, 540, 1],       # Center
            ]
            
            print("   Camera coords -> World coords:")
            for i, point in enumerate(test_points):
                # Apply homography transformation
                world_point = homography @ point
                # Normalize by w component
                if world_point[2] != 0:
                    world_point = world_point / world_point[2]
                
                labels = ['Top-left', 'Top-right', 'Bottom-left', 'Bottom-right', 'Center']
                print(f"   {labels[i]:12s}: ({point[0]:6.1f}, {point[1]:6.1f}) -> "
                      f"({world_point[0]:8.2f}, {world_point[1]:8.2f})")
            
            # Test inverse transformation
            print(f"\n   Inverse transformation check:")
            try:
                homography_inv = np.linalg.inv(homography)
                
                # Transform a point and back
                test_pt = np.array([500, 500, 1])
                transformed = homography @ test_pt
                transformed = transformed / transformed[2]
                back_transformed = homography_inv @ transformed
                back_transformed = back_transformed / back_transformed[2]
                
                error = np.linalg.norm(test_pt[:2] - back_transformed[:2])
                print(f"   Round-trip error: {error:.6f} pixels")
                
                if error < 0.001:
                    print("   ✅ Inverse transformation works correctly")
                else:
                    print(f"   ⚠️  Higher than expected round-trip error")
                    
            except np.linalg.LinAlgError:
                print("   ❌ Could not compute inverse")
                return False
            
            # Check if it looks like an identity or near-identity matrix
            identity = np.eye(3)
            if np.allclose(homography, identity, atol=0.01):
                print("\n⚠️  WARNING: Homography is very close to identity matrix!")
                print("   This might indicate uncalibrated or default values")
            
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED - Homography is valid and usable!")
            print("=" * 60)
            return True
            
    except Exception as e:
        print(f"\n❌ ERROR: Failed to test homography: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_homographies(file1, file2):
    """Compare homography matrices from two H5 files."""
    print(f"\n📊 Comparing homographies:")
    print(f"   File 1: {file1}")
    print(f"   File 2: {file2}")
    
    try:
        # Load homographies from both files
        h1 = None
        h2 = None
        
        with h5py.File(file1, 'r') as f1:
            if '/homography' in f1:
                h1 = f1['/homography'][:]
            elif '/calibration_snapshot' in f1:
                calib = f1['/calibration_snapshot']
                camera_ids = [k for k in calib.keys() if isinstance(calib[k], h5py.Group)]
                if camera_ids:
                    cam_id = camera_ids[0]
                    if 'homography_matrix_yml' in calib[cam_id]:
                        yaml_data = calib[cam_id]['homography_matrix_yml'][()].decode('utf-8')
                        h1 = parse_homography_from_yaml(yaml_data)
        
        with h5py.File(file2, 'r') as f2:
            if '/homography' in f2:
                h2 = f2['/homography'][:]
            elif '/calibration_snapshot' in f2:
                calib = f2['/calibration_snapshot']
                camera_ids = [k for k in calib.keys() if isinstance(calib[k], h5py.Group)]
                if camera_ids:
                    cam_id = camera_ids[0]
                    if 'homography_matrix_yml' in calib[cam_id]:
                        yaml_data = calib[cam_id]['homography_matrix_yml'][()].decode('utf-8')
                        h2 = parse_homography_from_yaml(yaml_data)
        
        if h1 is None or h2 is None:
            print("❌ Could not load homography from one or both files")
            return False
        
        if np.allclose(h1, h2):
            print("✅ Homographies are identical")
            return True
        else:
            print("⚠️  Homographies differ:")
            diff = h1 - h2
            print(f"   Max difference: {np.max(np.abs(diff)):.6e}")
            print(f"   Mean difference: {np.mean(np.abs(diff)):.6e}")
            
            print("\nMatrix 1:")
            for row in h1:
                print(f"   [{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f}]")
            
            print("\nMatrix 2:")
            for row in h2:
                print(f"   [{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f}]")
            
            return False
            
    except Exception as e:
        print(f"❌ Error comparing files: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test homography matrix in H5 file (supports both direct and YAML formats)'
    )
    parser.add_argument('h5_file', help='Path to H5 file to test')
    parser.add_argument('--compare', help='Compare with another H5 file')
    
    args = parser.parse_args()
    
    # Run test
    success = test_homography(args.h5_file)
    
    # Additional comparison if requested
    if args.compare:
        compare_success = compare_homographies(args.h5_file, args.compare)
        success = success and compare_success
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())