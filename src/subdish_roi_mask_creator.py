# src/roi_mask_creator.py

import cv2
import numpy as np
import zarr
import argparse
from pathlib import Path
import yaml

# Global variables
rectangles = []
drawing = False
start_point = (-1, -1)
display_image = None
background_image = None

def draw_rectangle_callback(event, x, y, flags, param):
    """Mouse callback function to draw multiple rectangles."""
    global start_point, drawing, display_image, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        # Store the rectangle with normalized coordinates
        rectangles.append((start_point, end_point))
        # Draw the final rectangle on the display image permanently
        cv2.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)
        # Add an ID label to the rectangle
        id_text = f"ID: {len(rectangles) - 1}"
        cv2.putText(display_image, id_text, (start_point[0] + 5, start_point[1] + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def main(zarr_path, config_path):
    global background_image, display_image, rectangles
    
    try:
        zarr_root = zarr.open(zarr_path, mode='r')
        latest_bg_run = zarr_root['background_runs'].attrs['latest']
        background_image = zarr_root[f'background_runs/{latest_bg_run}/background_ds'][:]
    except Exception as e:
        print(f"❌ Error opening Zarr file: {e}")
        return

    window_name = "ROI Mask Creator"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle_callback)
    
    print("🚀 Starting ROI Mask Creator...")
    print("Instructions:")
    print("1. Click and drag to draw a rectangle for each sub-dish.")
    print("2. The ID of the dish will be assigned in the order you draw them (0, 1, 2...).")
    print("3. Press 'r' to reset and clear all rectangles.")
    print("4. Press 's' to SAVE the ROI coordinates to your config file.")
    print("5. Press 'q' or Esc to quit without saving.")

    # Create the initial display image
    display_image = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)

    while True:
        # Create a temporary image for drawing the current rectangle in real-time
        temp_image = display_image.copy()
        
        cv2.imshow(window_name, temp_image)
        
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            rectangles = []
            display_image = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR) # Reset the display
            print("🔄 ROIs cleared. You can start drawing again.")
        elif key == ord('s'):
            if rectangles:
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Prepare ROI data for saving
                    saved_rois = []
                    for i, (start, end) in enumerate(rectangles):
                        x1, y1 = start
                        x2, y2 = end
                        roi_entry = {
                            'id': i,
                            'roi_pixels': [min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)]
                        }
                        saved_rois.append(roi_entry)

                    # Add or update the 'sub_dish_rois' section in the config
                    config['assign_ids'] = {'sub_dish_rois': saved_rois}
                    
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    
                    print(f"✅ {len(rectangles)} ROIs saved to: {config_path}")
                    break
                except Exception as e:
                    print(f"❌ Error saving to config file: {e}")
            else:
                print("⚠️ No ROIs drawn. Please draw at least one rectangle before saving.")
            
    cv2.destroyAllWindows()
    print("👋 ROI Mask Creator closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively draw multiple rectangular ROIs for ID assignment.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file containing the background model.")
    parser.add_argument("--config", type=str, default="src/pipeline_config.yaml", 
                        help="Path to the pipeline configuration YAML file to update.")
    args = parser.parse_args()
    main(args.zarr_path, args.config)