import numpy as np
import h5py
import zarr

# Load the zarr file that has the homography
zarr_root = zarr.open('/home/delahantyj@hhmi.org/Desktop/escape_2/2025-08-12T20-25-51Z_arena_4_chaser_detections.zarr', 'r')

# Get the homography matrix we stored
if 'chaser_comparison/metadata/homography' in zarr_root:
    homography = zarr_root['chaser_comparison/metadata/homography'][:]
    print("Homography matrix:")
    print(homography)
    
    # Compute inverse homography (projector -> camera)
    homography_inv = np.linalg.inv(homography)
    print("\nInverse homography matrix:")
    print(homography_inv)
    
    # Test transformation of chaser position (179, 179)
    # Using homogeneous coordinates
    chaser_proj = np.array([179.0, 179.0, 1.0])
    chaser_cam = homography_inv @ chaser_proj
    chaser_cam = chaser_cam / chaser_cam[2]  # Normalize
    
    print(f"\nChaser transformation using inverse homography:")
    print(f"  Projector space: (179.0, 179.0)")
    print(f"  Camera space: ({chaser_cam[0]:.1f}, {chaser_cam[1]:.1f})")
    print(f"  Expected center: (2256, 2256)")
    
    # Check if this looks more reasonable
    distance_from_center = np.sqrt((chaser_cam[0] - 2256)**2 + (chaser_cam[1] - 2256)**2)
    print(f"  Distance from center: {distance_from_center:.1f} pixels")
