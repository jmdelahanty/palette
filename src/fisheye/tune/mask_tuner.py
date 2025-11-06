import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Global variables to store trackbar values
hough_param1 = 50
hough_param2 = 30
radius_adjustment = 0
frame_index = 0
max_frames = 0

# Global variables to store detected circle or rectangle
detected_circle = None
rectangle_roi = None
mask_mode = "circle"

from fisheye.utils.zarr_io import open_zarr_root


def _normalize_mask_data(mask_data):
    """Return a normalized copy of mask data with explicit shape field."""
    if mask_data is None:
        return None
    normalized = dict(mask_data)
    shape = normalized.get('shape')
    if not shape:
        if 'detected_circle' in normalized:
            shape = 'circle'
        elif 'rectangle' in normalized:
            shape = 'rectangle'
    normalized['shape'] = shape
    if 'detected_circle' in normalized:
        normalized['detected_circle'] = dict(normalized['detected_circle'])
    if 'hough_params' in normalized:
        normalized['hough_params'] = dict(normalized['hough_params'])
    if 'rectangle' in normalized:
        normalized['rectangle'] = dict(normalized['rectangle'])
    if 'source' in normalized:
        normalized['source'] = dict(normalized['source'])
    return normalized

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

def save_mask_to_zarr(zarr_path, mask_definition, array_name, frame_index, params=None):
    """Save mask parameters to Zarr metadata ONLY."""
    try:
        zarr_root = open_zarr_root(zarr_path, mode='r+')
        
        if 'analysis_metadata' not in zarr_root:
            meta = zarr_root.create_group('analysis_metadata')
        else:
            meta = zarr_root['analysis_metadata']
        
        # Get existing metadata
        metadata = dict(meta.attrs) if meta.attrs else {}
        
        shape = mask_definition.get('shape')
        payload = {
            'shape': shape,
            'version': '2.0',
            'tuned_timestamp': datetime.now().isoformat(),
            'source': {
                'array': array_name,
                'frame': int(frame_index)
            }
        }
        payload['tuned_on_array'] = array_name
        payload['tuned_on_frame'] = int(frame_index)

        if shape == 'circle':
            circle = mask_definition['detected_circle']
            payload['method'] = mask_definition.get('method', 'hough_circle')
            payload['detected_circle'] = {
                'center': [int(circle['center'][0]), int(circle['center'][1])],
                'radius': int(circle['radius'])
            }
            if params:
                payload['hough_params'] = {
                    'param1': int(params.get('param1', 0)),
                    'param2': int(params.get('param2', 0)),
                    'radius_adjustment': int(params.get('radius_adjustment', 0))
                }
        elif shape == 'rectangle':
            payload['method'] = mask_definition.get('method', 'manual_rectangle')
            roi = mask_definition['rectangle']['roi']
            payload['rectangle'] = {
                'roi': [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
            }
        else:
            raise ValueError(f"Unsupported mask shape '{shape}'")
        
        metadata['dish_mask'] = payload
        
        meta.attrs.update(metadata)
        
        print(f"\n✅ Mask saved to zarr: {zarr_path}")
        print(f"   Location: analysis_metadata/attrs['dish_mask']")
        if shape == 'circle':
            print(f"   Shape: circle | Center: {payload['detected_circle']['center']} | Radius: {payload['detected_circle']['radius']}")
        elif shape == 'rectangle':
            print(f"   Shape: rectangle | ROI: {payload['rectangle']['roi']}")
        print(f"   This mask will be used automatically in detect and crop stages")
        
        return True
    except Exception as e:
        print(f"❌ Error saving to Zarr: {e}")
        return False

def load_from_zarr(zarr_path):
    """Load previously tuned mask parameters from Zarr if they exist"""
    try:
        zarr_root = open_zarr_root(zarr_path, mode='r')
        
        # Check new location first
        if 'analysis_metadata' in zarr_root:
            analysis_meta = zarr_root['analysis_metadata']
            if 'dish_mask' in analysis_meta.attrs:
                mask_data = analysis_meta.attrs['dish_mask']
                print(f" Found mask tuning in analysis_metadata")
                return _normalize_mask_data(mask_data)
        
        # Fall back to old location for backward compatibility
        if 'raw_video' in zarr_root:
            raw_video = zarr_root['raw_video']
            if 'dish_mask_tuning' in raw_video.attrs:
                print(f" Found mask tuning in raw_video (legacy location)")
                mask_data = raw_video.attrs['dish_mask_tuning']
                normalized = _normalize_mask_data(mask_data)
                if normalized is not None and 'shape' not in normalized:
                    normalized['shape'] = 'circle'
                return normalized
        
        return None
    except Exception as e:
        print(f"Could not load existing mask data: {e}")
        return None

def main(zarr_path, use_full_res=False, frame_idx=None, config_path=None, mode="auto"):
    global hough_param1, hough_param2, radius_adjustment, detected_circle, frame_index, max_frames, rectangle_roi, mask_mode
    
    detected_circle = None
    rectangle_roi = None
    
    if config_path is None:
        config_path = Path("src/pipeline_config.yaml")
    else:
        config_path = Path(config_path)
    
    # Try to load existing mask parameters
    existing_mask = load_from_zarr(zarr_path)
    mask_shape = existing_mask.get('shape') if existing_mask else None
    
    if mode == "auto":
        resolved_mode = mask_shape if mask_shape in {"circle", "rectangle"} else "circle"
    else:
        resolved_mode = mode
        if mask_shape and mask_shape != resolved_mode:
            print(f"⚠️  Existing mask shape '{mask_shape}' differs from requested mode '{resolved_mode}'.")
    mask_mode = resolved_mode
    
    if existing_mask:
        if mask_mode == 'circle' and 'detected_circle' in existing_mask:
            hough_params = existing_mask.get('hough_params', {})
            hough_param1 = int(hough_params.get('param1', hough_param1))
            hough_param2 = int(hough_params.get('param2', hough_param2))
            radius_adjustment = int(hough_params.get('radius_adjustment', radius_adjustment))
            circle = existing_mask['detected_circle']
            detected_circle = (
                int(circle['center'][0]),
                int(circle['center'][1]),
                int(circle['radius'])
            )
            print(f"   Loaded circle parameters: param1={hough_param1}, param2={hough_param2}, radius_adj={radius_adjustment}")
        elif mask_mode == 'rectangle' and 'rectangle' in existing_mask:
            roi = existing_mask['rectangle'].get('roi')
            if roi:
                rectangle_roi = [int(v) for v in roi]
                print(f"   Loaded rectangular ROI: {rectangle_roi}")
    
    try:
        zarr_root = open_zarr_root(zarr_path, mode='r')
        if 'raw_video' not in zarr_root:
            available = ", ".join(sorted(zarr_root.keys()))
            raise KeyError(f"'raw_video' group not found. Children: {available or 'none'}")
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
        elif existing_mask:
            source = existing_mask.get('source', {})
            tuned_frame = source.get('frame') or existing_mask.get('tuned_on_frame')
            if tuned_frame is not None:
                frame_index = int(np.clip(tuned_frame, 0, max_frames - 1))
            else:
                frame_index = max_frames // 2
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
    if mask_mode == 'circle':
        cv2.createTrackbar("param1", window_name, hough_param1, 200, update_param1)
        cv2.createTrackbar("param2", window_name, hough_param2, 200, update_param2)
        cv2.createTrackbar("Radius Adjust", window_name, radius_adjustment + 20, 40, update_radius_adj)
    
    print(f"\n Starting Dish Mask Tuner...")
    print("Controls:")
    print("  - Use 'Frame' slider to select different frames")
    if mask_mode == 'circle':
        print("  - Adjust Hough sliders to refine the circle fit")
        print("  - Press 's' to SAVE the circle parameters to Zarr")
        print("  - Press 'a' to auto-detect circle parameters")
        print("  - Press 'q' or Esc to quit without saving")
        print("\nGoal: Find parameters that draw a single, stable green circle around the dish.")
    else:
        print("  - Press 'r' (or space) to draw/select a rectangle ROI")
        print("  - Press 'x' to clear the current rectangle")
        print("  - Press 's' to SAVE the rectangle ROI to Zarr")
        print("  - Press 'q' or Esc to quit without saving")
        print("\nGoal: Draw a rectangle that covers the usable dish area.")

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
        
        if mask_mode == 'circle':
            # Apply Gaussian blur for better circle detection
            blurred = cv2.GaussianBlur(gray_frame, (9, 9), 2)
            
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                1,
                100,
                param1=hough_param1,
                param2=hough_param2,
                minRadius=0,
                maxRadius=0,
            )
            
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
            
            status_text = (
                f"Frame {frame_index}/{max_frames-1} | "
                f"param1={hough_param1}, param2={hough_param2}, radius_adj={radius_adjustment}"
            )
            cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if detected_circle:
                info_text = f"Circle: center=({detected_circle[0]}, {detected_circle[1]}), radius={detected_circle[2]}"
                cv2.putText(display_image, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            status_text = f"Frame {frame_index}/{max_frames-1} | press 'r' to select rectangle"
            cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if rectangle_roi:
                x, y, w, h = rectangle_roi
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                info_text = f"Rectangle: x={x}, y={y}, w={w}, h={h}"
                cv2.putText(display_image, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_image, "No rectangle selected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        cv2.imshow(window_name, display_image)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            if mask_mode == 'circle':
                if detected_circle is not None:
                    params = {
                        'param1': hough_param1,
                        'param2': hough_param2,
                        'radius_adjustment': radius_adjustment
                    }
                    mask_payload = {
                        'shape': 'circle',
                        'detected_circle': {
                            'center': [detected_circle[0], detected_circle[1]],
                            'radius': detected_circle[2]
                        }
                    }
                    success = save_mask_to_zarr(
                        zarr_path,
                        mask_payload,
                        array_name,
                        frame_index,
                        params=params
                    )
                    if success:
                        print("Press 'q' to quit or continue tuning")
                else:
                    print(" No circle detected - adjust parameters and try again")
            else:
                if rectangle_roi:
                    mask_payload = {
                        'shape': 'rectangle',
                        'rectangle': {
                            'roi': rectangle_roi
                        }
                    }
                    success = save_mask_to_zarr(
                        zarr_path,
                        mask_payload,
                        array_name,
                        frame_index
                    )
                    if success:
                        print("Press 'q' to quit or continue tuning")
                else:
                    print(" Draw a rectangle first (press 'r')")
        elif mask_mode == 'circle' and key == ord('a'):
            # Auto-detect mode
            print("\n🔍 Auto-detecting best parameters...")
            best_params = None
            best_score = 0
            
            blurred = cv2.GaussianBlur(gray_frame, (9, 9), 2)
            
            for p1 in range(20, 150, 20):
                for p2 in range(10, 50, 10):
                    circles = cv2.HoughCircles(
                        blurred,
                        cv2.HOUGH_GRADIENT,
                        1,
                        100,
                        param1=p1,
                        param2=p2,
                        minRadius=0,
                        maxRadius=0
                    )
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
        elif mask_mode == 'rectangle' and key in (ord('r'), ord(' ')):
            # Launch ROI selector
            roi = cv2.selectROI(window_name, display_image, fromCenter=False, showCrosshair=True)
            if roi and roi[2] > 0 and roi[3] > 0:
                rectangle_roi = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
                print(f"  Selected rectangle ROI: {rectangle_roi}")
            else:
                print("  Rectangle selection cancelled")
        elif mask_mode == 'rectangle' and key == ord('x'):
            rectangle_roi = None
            print("  Cleared rectangle ROI")
            
    cv2.destroyAllWindows()
    print("\nTuner closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively tune dish masks (circle or rectangle).")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file containing imported video.")
    parser.add_argument("--full", action="store_true", help="Use full resolution array instead of downsampled")
    parser.add_argument("--frame", type=int, help="Specific frame index to use")
    parser.add_argument("--config", type=str, help="Path to YAML config file (default: src/pipeline_config.yaml)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "circle", "rectangle"],
        default="auto",
        help="Mask tuning mode (default: auto – reuse stored shape or circle).",
    )
    args = parser.parse_args()

    main(args.zarr_path, use_full_res=args.full, frame_idx=args.frame,
         config_path=args.config, mode=args.mode)
