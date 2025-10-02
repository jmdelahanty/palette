import cv2
import numpy as np
import zarr
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Global variables to store trackbar values
hough_param1 = 50
hough_param2 = 30
radius_adjustment = 0
frame_index = 0
max_frames = 0

# Global variables to store detected circle
detected_circle = None

def update_param1(val):
    global hough_param1
    hough_param1 = max(1, val)

def update_param2(val):
    global hough_param2
    hough_param2 = max(1, val)

def update_radius_adj(val):
    global radius_adjustment
    radius_adjustment = val - 20

def update_frame(val):
    global frame_index
    frame_index = val

def save_mask_to_zarr(zarr_path, detected_circle, params, array_name, frame_index):
    """Save mask parameters to Zarr metadata ONLY."""
    try:
        zarr_root = zarr.open(zarr_path, mode='r+')
        
        if 'analysis_metadata' not in zarr_root:
            meta = zarr_root.create_group('analysis_metadata')
        else:
            meta = zarr_root['analysis_metadata']
        
        # Get existing metadata
        metadata = dict(meta.attrs) if meta.attrs else {}
        
        # Store mask data
        metadata['dish_mask'] = {
            'method': 'hough_circle',
            'version': '1.0',
            'tuned_timestamp': datetime.now().isoformat(),
            'detected_circle': {
                'center': [int(detected_circle[0]), int(detected_circle[1])],
                'radius': int(detected_circle[2])
            },
            'hough_params': {
                'param1': int(params['param1']),
                'param2': int(params['param2']),
                'radius_adjustment': int(params['radius_adjustment'])
            },
            'source': {
                'array': array_name,
                'frame': int(frame_index)
            }
        }
        
        meta.attrs.update(metadata)
        
        print(f"\n✅ Mask saved to zarr: {zarr_path}")
        print(f"   Location: analysis_metadata/attrs['dish_mask']")
        print(f"   Center: {metadata['dish_mask']['detected_circle']['center']}")
        print(f"   Radius: {metadata['dish_mask']['detected_circle']['radius']}")
        print(f"   This mask will be used automatically in detect and crop stages")
        
        return True
    except Exception as e:
        print(f"❌ Error saving to Zarr: {e}")
        return False

def load_from_zarr(zarr_path):
    """Load previously tuned mask parameters from Zarr if they exist"""
    try:
        zarr_root = zarr.open(zarr_path, mode='r')
        
        # Check new location first
        if 'analysis_metadata' in zarr_root:
            analysis_meta = zarr_root['analysis_metadata']
            if 'dish_mask' in analysis_meta.attrs:
                mask_data = analysis_meta.attrs['dish_mask']
                print(f" Found mask tuning in analysis_metadata")
                return mask_data.get('detected_circle')
        
        # Fall back to old location for backward compatibility
        if 'raw_video' in zarr_root:
            raw_video = zarr_root['raw_video']
            if 'dish_mask_tuning' in raw_video.attrs:
                print(f" Found mask tuning in raw_video (legacy location)")
                mask_data = raw_video.attrs['dish_mask_tuning']
                return mask_data.get('detected_circle')
        
        return None
    except Exception as e:
        print(f"Could not load existing mask data: {e}")
        return None

def main(zarr_path, use_full_res=False, frame_idx=None, config_path=None):
    global hough_param1, hough_param2, radius_adjustment, detected_circle, frame_index, max_frames
    
    if config_path is None:
        config_path = Path("src/pipeline_config.yaml")
    else:
        config_path = Path(config_path)
    
    # Try to load existing mask parameters
    existing_mask = load_from_zarr(zarr_path)
    if existing_mask:
        hough_param1 = existing_mask.get('hough_param1', 50)
        hough_param2 = existing_mask.get('hough_param2', 30)
        radius_adjustment = existing_mask.get('radius_adjustment', 0)
        print(f"   Loaded parameters: param1={hough_param1}, param2={hough_param2}, radius_adj={radius_adjustment}")
    
    try:
        zarr_root = zarr.open(zarr_path, mode='r')
        raw_video = zarr_root['raw_video']
        
        # Check what's available
        has_full = 'images_full' in raw_video
        has_ds = 'images_ds' in raw_video
        
        if not has_full and not has_ds:
            print(" No video arrays found in raw_video group")
            return
        
        # Select array based on preference and availability
        if use_full_res and has_full:
            video_array = raw_video['images_full']
            array_name = "images_full"
        elif has_ds:
            video_array = raw_video['images_ds']
            array_name = "images_ds"
        else:
            video_array = raw_video['images_full']
            array_name = "images_full"
        
        print(f"\n Using array: {array_name}")
        print(f"   Shape: {video_array.shape}")
        
        max_frames = video_array.shape[0]
        
        # Use specified frame or find a good representative frame
        if frame_idx is not None:
            frame_index = min(frame_idx, max_frames - 1)
        elif existing_mask and 'tuned_on_frame' in existing_mask:
            frame_index = existing_mask['tuned_on_frame']
        else:
            # Use middle frame as default
            frame_index = max_frames // 2
            
    except Exception as e:
        print(f" Error opening Zarr file: {e}")
        return

    window_name = "Dish Mask Tuner"
    cv2.namedWindow(window_name)
    
    # Create trackbars
    cv2.createTrackbar("Frame", window_name, frame_index, max_frames - 1, update_frame)
    cv2.createTrackbar("param1", window_name, hough_param1, 200, update_param1)
    cv2.createTrackbar("param2", window_name, hough_param2, 200, update_param2)
    cv2.createTrackbar("Radius Adjust", window_name, radius_adjustment + 20, 40, update_radius_adj)
    
    print(f"\n Starting Dish Mask Tuner...")
    print("Controls:")
    print("  - Use 'Frame' slider to select different frames")
    print("  - Adjust Hough sliders to find the best circle fit")
    print(f"  - Press 's' to SAVE the circle parameters to Zarr")
    print("  - Press 'a' to auto-detect best parameters")
    print("  - Press 'q' or Esc to quit without saving")
    print("\nGoal: Find parameters that draw a single, stable green circle around the dish.")

    auto_detected = False

    while True:
        # Load current frame
        try:
            current_frame = video_array[frame_index]
            if current_frame.dtype != np.uint8:
                current_frame = (current_frame * 255).astype(np.uint8)
        except Exception as e:
            print(f"Error loading frame {frame_index}: {e}")
            continue
        
        # Convert to BGR for display
        if len(current_frame.shape) == 2:
            display_image = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
            gray_frame = current_frame
        else:
            display_image = current_frame.copy()
            gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for better circle detection
        blurred = cv2.GaussianBlur(gray_frame, (9, 9), 2)
        
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=hough_param1,
                                   param2=hough_param2,
                                   minRadius=0, maxRadius=0)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Take only the first (best) circle
            i = circles[0, 0]
            
            # Apply radius adjustment
            adjusted_radius = int(i[2]) + radius_adjustment
            
            # Store the detected circle
            detected_circle = (int(i[0]), int(i[1]), adjusted_radius)
            
            # Draw the outer circle with the adjusted radius
            cv2.circle(display_image, (i[0], i[1]), adjusted_radius, (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(display_image, (i[0], i[1]), 2, (0, 0, 255), 3)
        
        # Update status text
        status_text = f"Frame {frame_index}/{max_frames-1} | param1={hough_param1}, param2={hough_param2}, radius_adj={radius_adjustment}"
        cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if detected_circle:
            info_text = f"Circle: center=({detected_circle[0]}, {detected_circle[1]}), radius={detected_circle[2]}"
            cv2.putText(display_image, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display_image)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            if detected_circle is not None:
                # define params before using it
                params = {
                    'param1': hough_param1,
                    'param2': hough_param2,
                    'radius_adjustment': radius_adjustment
                }
                
                success = save_mask_to_zarr(
                    zarr_path,
                    detected_circle,
                    params,
                    array_name,
                    frame_index
                )
                
                if success:
                    print("Press 'q' to quit or continue tuning")
            else:
                print(" No circle detected - adjust parameters and try again")
        elif key == ord('a'):
            # Auto-detect mode
            print("\n🔍 Auto-detecting best parameters...")
            best_params = None
            best_score = 0
            
            for p1 in range(20, 150, 20):
                for p2 in range(10, 50, 10):
                    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100,
                                              param1=p1, param2=p2,
                                              minRadius=0, maxRadius=0)
                    if circles is not None and len(circles[0]) == 1:
                        score = 100 - abs(p1 - 50) - abs(p2 - 30)
                        if score > best_score:
                            best_score = score
                            best_params = (p1, p2)
            
            if best_params:
                hough_param1, hough_param2 = best_params
                cv2.setTrackbarPos("param1", window_name, hough_param1)
                cv2.setTrackbarPos("param2", window_name, hough_param2)
                auto_detected = True
                print(f"   Found: param1={hough_param1}, param2={hough_param2}")
            else:
                print("   Could not find good parameters automatically")
            
    cv2.destroyAllWindows()
    print("\nTuner closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively tune Hough Circle parameters for dish detection.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file containing imported video.")
    parser.add_argument("--full", action="store_true", help="Use full resolution array instead of downsampled")
    parser.add_argument("--frame", type=int, help="Specific frame index to use")
    parser.add_argument("--config", type=str, help="Path to YAML config file (default: src/pipeline_config.yaml)")
    args = parser.parse_args()

    main(args.zarr_path, use_full_res=args.full, frame_idx=args.frame,
         config_path=args.config)