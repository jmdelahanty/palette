import h5py
import json
import numpy as np
import zarr

def analyze_coordinate_system(h5_path):
    """Analyze all coordinate systems in the experiment."""
    
    with h5py.File(h5_path, 'r') as h5f:
        # Load arena configuration
        if '/calibration_snapshot/arena_config_json' in h5f:
            arena_config_str = h5f['/calibration_snapshot/arena_config_json'][()].decode('utf-8')
            arena_config = json.loads(arena_config_str)
            
            print("=" * 60)
            print("COORDINATE SYSTEM ANALYSIS")
            print("=" * 60)
            
            print("\n1. PROJECTOR/SUB-ARENA SPACE (from arena_config):")
            print(f"   Sub-arena dimensions: {arena_config['sub_arena_width_px']} x {arena_config['sub_arena_height_px']} px")
            print(f"   Sub-arena position: ({arena_config['sub_arena_x_px']}, {arena_config['sub_arena_y_px']})")
            
            # Camera info
            if 'camera_calibrations' in arena_config:
                cam_calib = arena_config['camera_calibrations'][0]
                print(f"\n2. CAMERA SPACE:")
                print(f"   Camera ID: {cam_calib['camera_id']}")
                print(f"   Native resolution: {cam_calib['native_width_px']} x {cam_calib['native_height_px']} px")
                print(f"   Camera sub-arena: {cam_calib['sub_arena_width_px']} x {cam_calib['sub_arena_height_px']} px")
                print(f"   Camera sub-arena pos: ({cam_calib['sub_arena_x_px']}, {cam_calib['sub_arena_y_px']})")
            
            # Analyze chaser states to understand texture space
            if '/tracking_data/chaser_states' in h5f:
                chaser_states = h5f['/tracking_data/chaser_states'][:]
                chaser_x = chaser_states['chaser_pos_x']
                chaser_y = chaser_states['chaser_pos_y']
                
                print(f"\n3. TEXTURE/STIMULUS SPACE (from chaser positions):")
                print(f"   Chaser X range: {chaser_x.min():.1f} to {chaser_x.max():.1f}")
                print(f"   Chaser Y range: {chaser_y.min():.1f} to {chaser_y.max():.1f}")
                print(f"   Chaser mean: ({chaser_x.mean():.1f}, {chaser_y.mean():.1f})")
                
                # If chaser is at (179, 179) for center, texture is 358x358
                texture_width = chaser_x.max() * 2 if chaser_x.max() == chaser_x.min() else chaser_x.max() - chaser_x.min()
                texture_height = chaser_y.max() * 2 if chaser_y.max() == chaser_y.min() else chaser_y.max() - chaser_y.min()
                
                if chaser_x.mean() == 179 and chaser_y.mean() == 179:
                    texture_width = 358
                    texture_height = 358
                
                print(f"   Inferred texture dimensions: {texture_width} x {texture_height}")
                
                # Calculate transformation from texture to camera
                print(f"\n4. COORDINATE TRANSFORMATIONS:")
                
                # Method 1: Direct scaling from texture to camera
                # Texture (0,0) to (358,358) maps to full camera view (0,0) to (4512,4512)
                scale_texture_to_camera_x = cam_calib['native_width_px'] / texture_width
                scale_texture_to_camera_y = cam_calib['native_height_px'] / texture_height
                
                print(f"   Texture to Camera scaling:")
                print(f"     X scale: {scale_texture_to_camera_x:.3f} camera_px/texture_unit")
                print(f"     Y scale: {scale_texture_to_camera_y:.3f} camera_px/texture_unit")
                
                # Test transformation
                chaser_cam_x = chaser_x.mean() * scale_texture_to_camera_x
                chaser_cam_y = chaser_y.mean() * scale_texture_to_camera_y
                print(f"\n   Test: Chaser at texture ({chaser_x.mean():.1f}, {chaser_y.mean():.1f})")
                print(f"         -> Camera ({chaser_cam_x:.1f}, {chaser_cam_y:.1f})")
                print(f"         Expected: (~{cam_calib['native_width_px']/2:.0f}, ~{cam_calib['native_height_px']/2:.0f})")
                
                return {
                    'texture_width': texture_width,
                    'texture_height': texture_height,
                    'camera_width': cam_calib['native_width_px'],
                    'camera_height': cam_calib['native_height_px'],
                    'scale_x': scale_texture_to_camera_x,
                    'scale_y': scale_texture_to_camera_y
                }

# Run the analysis
h5_path = '/home/delahantyj@hhmi.org/Desktop/escape_2/out_analysis.h5'
transform_params = analyze_coordinate_system(h5_path)

if transform_params:
    print("\n" + "=" * 60)
    print("SAVING TRANSFORMATION PARAMETERS TO ZARR")
    print("=" * 60)
    
    # Save to zarr for use in analysis
    zarr_path = '/home/delahantyj@hhmi.org/Desktop/escape_2/2025-08-12T20-25-51Z_arena_4_chaser_detections.zarr'
    zarr_root = zarr.open(zarr_path, 'r+')
    
    if 'chaser_comparison/metadata' in zarr_root:
        metadata = zarr_root['chaser_comparison/metadata']
        metadata.attrs['texture_to_camera_scale_x'] = transform_params['scale_x']
        metadata.attrs['texture_to_camera_scale_y'] = transform_params['scale_y']
        metadata.attrs['texture_dimensions'] = [transform_params['texture_width'], transform_params['texture_height']]
        metadata.attrs['coordinate_transform_method'] = 'texture_to_camera_scaling'
        
        print(f"Saved transformation parameters:")
        print(f"  Scale X: {transform_params['scale_x']:.3f}")
        print(f"  Scale Y: {transform_params['scale_y']:.3f}")
        print(f"  Texture size: {transform_params['texture_width']} x {transform_params['texture_height']}")
        
        # Now re-transform chaser positions with correct scaling
        analysis = zarr_root['chaser_comparison/interp_linear_20250820_113848']
        chaser_world = analysis['chaser_position_world'][:]
        
        # Apply correct transformation
        chaser_camera_corrected = np.full_like(chaser_world, np.nan)
        valid_mask = ~np.isnan(chaser_world[:, 0])
        
        chaser_camera_corrected[valid_mask, 0] = chaser_world[valid_mask, 0] * transform_params['scale_x']
        chaser_camera_corrected[valid_mask, 1] = chaser_world[valid_mask, 1] * transform_params['scale_y']
        
        # Save corrected positions
        if 'chaser_position_camera_corrected' in analysis:
            del analysis['chaser_position_camera_corrected']
        analysis.create_dataset('chaser_position_camera_corrected', data=chaser_camera_corrected)
        
        # Check a few samples
        print(f"\nSample corrected positions:")
        for i in range(min(5, len(chaser_camera_corrected))):
            if not np.isnan(chaser_camera_corrected[i, 0]):
                print(f"  Frame {i}: Texture ({chaser_world[i,0]:.1f}, {chaser_world[i,1]:.1f}) -> "
                      f"Camera ({chaser_camera_corrected[i,0]:.1f}, {chaser_camera_corrected[i,1]:.1f})")
