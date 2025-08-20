#!/usr/bin/env python3
"""
Fix existing chaser-fish analyses with proper coordinate transformation.
"""

import zarr
import numpy as np
from coordinate_transform_module import CoordinateSystem

# Paths
h5_path = '/home/delahantyj@hhmi.org/Desktop/escape_2/out_analysis.h5'
zarr_path = '/home/delahantyj@hhmi.org/Desktop/escape_2/2025-08-12T20-25-51Z_arena_4_chaser_detections.zarr'

print("Fixing coordinate transformations in existing analysis...")

# Create coordinate system
coord_sys = CoordinateSystem(h5_path)

# Open zarr file
root = zarr.open(zarr_path, 'r+')

# Fix each interpolation run
if 'chaser_comparison' in root:
    for run_name in root['chaser_comparison'].keys():
        if run_name in ['metadata', 'frame_alignment']:
            continue
            
        print(f"\nProcessing run: {run_name}")
        analysis = root[f'chaser_comparison/{run_name}']
        
        # Get chaser world positions (in texture space)
        if 'chaser_position_world' in analysis:
            chaser_world = analysis['chaser_position_world'][:]
            
            # Transform to camera coordinates
            chaser_camera_fixed = np.full_like(chaser_world, np.nan)
            valid_mask = ~np.isnan(chaser_world[:, 0])
            
            if np.any(valid_mask):
                valid_x = chaser_world[valid_mask, 0]
                valid_y = chaser_world[valid_mask, 1]
                
                cam_x, cam_y = coord_sys.transform_coordinates(
                    valid_x, valid_y,
                    from_space='texture',
                    to_space='camera'
                )
                
                chaser_camera_fixed[valid_mask, 0] = cam_x
                chaser_camera_fixed[valid_mask, 1] = cam_y
                
                # Save fixed positions
                if 'chaser_position_camera_fixed' in analysis:
                    del analysis['chaser_position_camera_fixed']
                analysis.create_dataset('chaser_position_camera_fixed', data=chaser_camera_fixed)
                
                # Update the main chaser_position_camera dataset
                if 'chaser_position_camera' in analysis:
                    del analysis['chaser_position_camera']
                analysis.create_dataset('chaser_position_camera', data=chaser_camera_fixed)
                
                print(f"  Fixed {np.sum(valid_mask)} chaser positions")
                
                # Recalculate distances with fixed positions
                fish_pos = analysis['fish_position_camera'][:]
                distances = np.full(len(fish_pos), np.nan)
                
                for i in range(len(fish_pos)):
                    if not np.isnan(fish_pos[i, 0]) and not np.isnan(chaser_camera_fixed[i, 0]):
                        dist = np.sqrt((fish_pos[i, 0] - chaser_camera_fixed[i, 0])**2 + 
                                     (fish_pos[i, 1] - chaser_camera_fixed[i, 1])**2)
                        distances[i] = dist
                
                # Update distance dataset
                if 'fish_chaser_distance_pixels' in analysis:
                    del analysis['fish_chaser_distance_pixels']
                analysis.create_dataset('fish_chaser_distance_pixels', data=distances)
                
                print(f"  Recalculated distances")
    
    # Update metadata with transformation info
    metadata = root['chaser_comparison/metadata']
    transform_params = coord_sys.get_transform_params()
    for key, value in transform_params.items():
        metadata.attrs[f'coord_transform_{key}'] = str(value)
    
    print("\n✅ Fixed all coordinate transformations!")
    print("\nYou can now re-run the heatmap analyzer and the chaser should appear at the center.")

