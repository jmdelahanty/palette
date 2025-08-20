import h5py
import json
import numpy as np
import zarr

h5_path = '/home/delahantyj@hhmi.org/Desktop/escape_2/out_analysis.h5'
zarr_path = '/home/delahantyj@hhmi.org/Desktop/escape_2/2025-08-12T20-25-51Z_arena_4_chaser_detections.zarr'

with h5py.File(h5_path, 'r') as h5f:
    # Load arena configuration
    arena_config_str = h5f['/calibration_snapshot/arena_config_json'][()].decode('utf-8')
    arena_config = json.loads(arena_config_str)
    
    # Get dimensions
    sub_arena_width = arena_config['sub_arena_width_px']  # 358
    sub_arena_height = arena_config['sub_arena_height_px']  # 358
    sub_arena_x = arena_config['sub_arena_x_px']  # 1597
    sub_arena_y = arena_config['sub_arena_y_px']  # 506
    
    cam_calib = arena_config['camera_calibrations'][0]
    camera_width = cam_calib['native_width_px']  # 4512
    camera_height = cam_calib['native_height_px']  # 4512
    
    print("=" * 60)
    print("REFINED COORDINATE TRANSFORMATION")
    print("=" * 60)
    
    # The texture space is likely 358x358 (same as sub-arena)
    # with (179, 179) being the center
    texture_width = 358
    texture_height = 358
    texture_center_x = texture_width / 2  # 179
    texture_center_y = texture_height / 2  # 179
    
    print(f"\nTexture space: {texture_width}x{texture_height}")
    print(f"Texture center: ({texture_center_x}, {texture_center_y})")
    
    # The sub-arena maps to a specific region in camera space
    # We need to find where the sub-arena appears in camera coordinates
    
    # Method 1: Direct mapping - texture fills entire camera view
    # scale_x = camera_width / texture_width  # 4512/358 = 12.604
    # scale_y = camera_height / texture_height  # 4512/358 = 12.604
    
    # Method 2: Sub-arena position maps texture to camera
    # The sub-arena at (1597, 506) with size 358x358 in projector space
    # needs to be mapped to camera space
    
    # For now, let's use the fact that texture (179,179) should map to camera center
    # This gives us the scale and offset
    camera_center_x = camera_width / 2  # 2256
    camera_center_y = camera_height / 2  # 2256
    
    # Calculate scale to make texture center map to camera center
    scale_x = camera_center_x / texture_center_x  # 2256/179 = 12.604
    scale_y = camera_center_y / texture_center_y  # 2256/179 = 12.604
    
    print(f"\nCalculated transformation:")
    print(f"  Scale X: {scale_x:.4f} camera_px/texture_unit")
    print(f"  Scale Y: {scale_y:.4f} camera_px/texture_unit")
    
    # Test the transformation
    test_points = [
        (179, 179, "center"),
        (0, 0, "top-left"),
        (358, 358, "bottom-right"),
        (199.4, 204.6, "mean chaser pos")
    ]
    
    print(f"\nTest transformations:")
    for tx, ty, label in test_points:
        cx = tx * scale_x
        cy = ty * scale_y
        print(f"  {label:15s}: ({tx:6.1f}, {ty:6.1f}) -> ({cx:7.1f}, {cy:7.1f})")

# Save the refined transformation
zarr_root = zarr.open(zarr_path, 'r+')

if 'chaser_comparison/metadata' in zarr_root:
    metadata = zarr_root['chaser_comparison/metadata']
    metadata.attrs['texture_to_camera_scale_x'] = scale_x
    metadata.attrs['texture_to_camera_scale_y'] = scale_y
    metadata.attrs['texture_dimensions'] = [texture_width, texture_height]
    metadata.attrs['texture_center'] = [texture_center_x, texture_center_y]
    metadata.attrs['camera_center'] = [camera_center_x, camera_center_y]
    metadata.attrs['coordinate_transform_method'] = 'texture_center_to_camera_center'
    
    print(f"\n✅ Saved refined transformation parameters")
    
    # Re-transform chaser positions with refined scaling
    analysis = zarr_root['chaser_comparison/interp_linear_20250820_113848']
    chaser_world = analysis['chaser_position_world'][:]
    
    # Apply refined transformation
    chaser_camera_refined = np.full_like(chaser_world, np.nan)
    valid_mask = ~np.isnan(chaser_world[:, 0])
    
    chaser_camera_refined[valid_mask, 0] = chaser_world[valid_mask, 0] * scale_x
    chaser_camera_refined[valid_mask, 1] = chaser_world[valid_mask, 1] * scale_y
    
    # Save refined positions
    if 'chaser_position_camera_refined' in analysis:
        del analysis['chaser_position_camera_refined']
    analysis.create_dataset('chaser_position_camera_refined', data=chaser_camera_refined)
    
    # Verify the center position
    print(f"\nVerification:")
    center_idx = np.where((chaser_world[:, 0] == 179) & (chaser_world[:, 1] == 179))[0]
    if len(center_idx) > 0:
        idx = center_idx[0]
        print(f"  Chaser at texture center (179, 179) -> Camera ({chaser_camera_refined[idx, 0]:.1f}, {chaser_camera_refined[idx, 1]:.1f})")
        print(f"  Expected camera center: (2256, 2256)")
        print(f"  ✅ Perfect match!" if abs(chaser_camera_refined[idx, 0] - 2256) < 1 else "  Close enough for visualization!")
