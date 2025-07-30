import cv2
import numpy as np
import zarr
import argparse

# Global variables to store trackbar values
hough_param1 = 50
hough_param2 = 30
# --- NEW: Global for radius adjustment ---
radius_adjustment = 0

def update_param1(val):
    global hough_param1
    hough_param1 = max(1, val) # Ensure value is at least 1

def update_param2(val):
    global hough_param2
    hough_param2 = max(1, val) # Ensure value is at least 1

# --- NEW: Callback for radius adjustment slider ---
def update_radius_adj(val):
    """Callback to handle the radius adjustment slider."""
    global radius_adjustment
    # The slider goes from 0 to 40, so we subtract 20 to get a range of -20 to +20
    radius_adjustment = val - 20

def main(zarr_path):
    global hough_param1, hough_param2, radius_adjustment
    
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
    # --- NEW: Radius adjustment trackbar ---
    # We create a trackbar from 0-40 and will offset it by -20 to get a range of [-20, 20]
    cv2.createTrackbar("Radius Adjust", window_name, radius_adjustment + 20, 40, update_radius_adj)
    
    print("🚀 Starting Dish Mask Tuner...")
    print("Controls: Adjust sliders to find the best circle fit. Press 'q' or Esc to quit.")
    print("Goal: Find parameters that draw a single, stable green circle perfectly around the dish.")

    while True:
        display_image = cv2.cvtColor(background_ds, cv2.COLOR_GRAY2BGR)
        
        circles = cv2.HoughCircles(background_ds, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=hough_param1,
                                   param2=hough_param2,
                                   minRadius=0, maxRadius=0)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # --- MODIFICATION: Apply radius adjustment ---
                adjusted_radius = int(i[2]) + radius_adjustment
                
                # Draw the outer circle with the adjusted radius
                cv2.circle(display_image, (i[0], i[1]), adjusted_radius, (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(display_image, (i[0], i[1]), 2, (0, 0, 255), 3)
        
        # --- MODIFICATION: Update status text ---
        status_text = f"param1={hough_param1}, param2={hough_param2}, radius_adj={radius_adjustment}"
        cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(window_name, display_image)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    cv2.destroyAllWindows()
    print("\nTuner closed. Update your pipeline_config.yaml with the best parameters you found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively tune Hough Circle parameters for dish detection.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file containing the background model.")
    args = parser.parse_args()
    
    main(args.zarr_path)