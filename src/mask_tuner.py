import cv2
import numpy as np
import zarr
import argparse
import yaml
from pathlib import Path

# Global variables to store trackbar values
hough_param1 = 50
hough_param2 = 30
radius_adjustment = 0

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

def main(zarr_path):
    global hough_param1, hough_param2, radius_adjustment, detected_circle
    
    config_path = Path("src/pipeline_config.yaml")
    
    try:
        zarr_root = zarr.open(zarr_path, mode='r')
        latest_bg_run = zarr_root['background_runs'].attrs['latest']
        background_ds = zarr_root[f'background_runs/{latest_bg_run}/background_ds'][:]
    except Exception as e:
        print(f"Error opening Zarr file or finding background image: {e}")
        return

    window_name = "Dish Mask Tuner"
    cv2.namedWindow(window_name)
    
    # Create trackbars
    cv2.createTrackbar("param1", window_name, hough_param1, 200, update_param1)
    cv2.createTrackbar("param2", window_name, hough_param2, 200, update_param2)
    cv2.createTrackbar("Radius Adjust", window_name, radius_adjustment + 20, 40, update_radius_adj)
    
    print("🚀 Starting Dish Mask Tuner...")
    print("Controls:")
    print("  - Adjust sliders to find the best circle fit")
    print("  - Press 's' to SAVE the circle parameters to pipeline_config.yaml")
    print("  - Press 'q' or Esc to quit without saving")
    print("Goal: Find parameters that draw a single, stable green circle perfectly around the dish.")

    while True:
        display_image = cv2.cvtColor(background_ds, cv2.COLOR_GRAY2BGR)
        
        circles = cv2.HoughCircles(background_ds, cv2.HOUGH_GRADIENT, 1, 100,
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
        status_text = f"param1={hough_param1}, param2={hough_param2}, radius_adj={radius_adjustment}"
        cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if detected_circle:
            info_text = f"Circle: center=({detected_circle[0]}, {detected_circle[1]}), radius={detected_circle[2]}"
            cv2.putText(display_image, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display_image)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            if detected_circle:
                try:
                    # Load existing config
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    else:
                        config = {}
                    
                    # Prepare the circle mask parameters
                    circle_mask = {
                        'shape': 'circle',
                        'center': [detected_circle[0], detected_circle[1]],
                        'radius': detected_circle[2],
                        'hough_params': {
                            'param1': hough_param1,
                            'param2': hough_param2,
                            'radius_adjustment': radius_adjustment
                        }
                    }
                    
                    # Update both detect and crop sections
                    if 'detect' not in config:
                        config['detect'] = {}
                    config['detect']['dish_mask'] = circle_mask
                    
                    if 'crop' not in config:
                        config['crop'] = {}
                    config['crop']['dish_mask'] = circle_mask
                    
                    # Save updated config
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    
                    print(f"✅ Circle mask parameters saved to {config_path}")
                    print(f"   Center: ({detected_circle[0]}, {detected_circle[1]})")
                    print(f"   Radius: {detected_circle[2]}")
                    print(f"   Hough params: param1={hough_param1}, param2={hough_param2}, radius_adj={radius_adjustment}")
                except Exception as e:
                    print(f"❌ Error saving parameters: {e}")
            else:
                print("⚠️ No circle detected. Please adjust parameters until a circle is detected.")
            
    cv2.destroyAllWindows()
    print("\nTuner closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively tune Hough Circle parameters for dish detection.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file containing the background model.")
    args = parser.parse_args()
    
    main(args.zarr_path)