import numpy as np
import cv2
import h5py

# Load your H5 file
h5f = h5py.File('/home/delahantyj@hhmi.org/Desktop/escape_2/out_analysis.h5', 'r')

# Find and load homography matrix
# (adjust path based on your H5 structure)
homography = None
for path in ['/calibration/homography_matrix', '/rig_info/homography_matrix']:
    if path in h5f:
        homography = h5f[path][:]
        print(f"Found homography at {path}")
        break

if homography is not None:
    # Compute inverse
    homography_inv = np.linalg.inv(homography)
    
    # Test with known chaser position
    chaser_proj = np.array([[[179.0, 179.0]]], dtype=np.float32)
    chaser_cam = cv2.perspectiveTransform(chaser_proj, homography_inv)
    
    print(f"\nTest transformation:")
    print(f"  Projector space: {chaser_proj[0, 0]}")
    print(f"  Camera space: {chaser_cam[0, 0]}")
    print(f"  Expected ~(2256, 2256) if chaser is at arena center")
    
    # Load actual chaser states and transform a few
    if '/tracking_data/chaser_states' in h5f:
        chaser_states = h5f['/tracking_data/chaser_states'][:10]
        print(f"\nFirst 10 chaser transformations:")
        for i, state in enumerate(chaser_states):
            proj_pos = np.array([[[state['chaser_pos_x'], state['chaser_pos_y']]]], dtype=np.float32)
            cam_pos = cv2.perspectiveTransform(proj_pos, homography_inv)
            print(f"  Frame {i}: ({proj_pos[0,0,0]:.1f}, {proj_pos[0,0,1]:.1f}) -> "
                  f"({cam_pos[0,0,0]:.1f}, {cam_pos[0,0,1]:.1f})")

h5f.close()