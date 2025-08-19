import cv2
import numpy as np
import zarr
import argparse
from pathlib import Path
import yaml

# Global variables
drawing = False
start_point = (-1, -1)
end_point = (-1, -1)
background_image = None
display_image = None

def draw_rectangle(event, x, y, flags, param):
    """Mouse callback function to draw a rectangle."""
    global start_point, end_point, drawing, display_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        cv2.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)

def main(zarr_path, output_config_path):
    global background_image, display_image, start_point, end_point
    
    try:
        zarr_root = zarr.open(zarr_path, mode='r')
        latest_bg_run = zarr_root['background_runs'].attrs['latest']
        background_image = zarr_root[f'background_runs/{latest_bg_run}/background_ds'][:]
    except Exception as e:
        print(f"Error opening Zarr file or finding background image: {e}")
        return

    window_name = "ROI Mask Creator"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle)
    
    print("🚀 Starting ROI Mask Creator...")
    print("Instructions:")
    print("1. Click and drag to draw the desired rectangular mask.")
    print("2. Press 'r' to reset and draw again.")
    print("3. Press 's' to save the mask coordinates to your config file.")
    print("4. Press 'q' or Esc to quit without saving.")

    while True:
        display_image = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)
        
        if drawing:
            cv2.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)
        elif start_point != (-1, -1):
            cv2.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)
            
        cv2.imshow(window_name, display_image)
        
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            start_point = (-1, -1)
            end_point = (-1, -1)
            print("Resetting ROI. Draw a new rectangle.")
        elif key == ord('s'):
            if start_point != (-1, -1) and end_point != (-1, -1):
                x1, y1 = start_point
                x2, y2 = end_point
                
                # Ensure x1 < x2 and y1 < y2
                roi_x = min(x1, x2)
                roi_y = min(y1, y2)
                roi_w = abs(x1 - x2)
                roi_h = abs(y1 - y2)
                
                # Load existing config
                with open(output_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Update dish_mask section
                if 'crop' not in config: config['crop'] = {}
                config['crop']['dish_mask'] = {
                    'shape': 'rectangle',
                    'roi': [roi_x, roi_y, roi_w, roi_h]
                }
                
                # Save updated config
                with open(output_config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                print(f"✅ Mask coordinates saved to: {output_config_path}")
                print(f"   ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
                break
            else:
                print("⚠️  No ROI drawn. Please draw a rectangle before saving.")
            
    cv2.destroyAllWindows()
    print("👋 ROI Mask Creator closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively draw a rectangular ROI for the dish mask.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file containing the background model.")
    parser.add_argument("--config", type=str, default="src/pipeline_config.yaml", 
                        help="Path to the pipeline configuration YAML file to update.")
    args = parser.parse_args()
    main(args.zarr_path, args.config)