import cv2
import numpy as np
import zarr
import argparse
from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops
from pathlib import Path
import yaml

# Global variables to store trackbar values (used as a fallback)
ds_thresh = 55
current_frame = 1
ds_se1_radius = 1
ds_se4_radius = 4
min_area = 50
max_area = 500

def update_ds_thresh(val):
    global ds_thresh
    ds_thresh = val

def update_frame(val):
    global current_frame
    current_frame = val

def update_ds_se1(val):
    global ds_se1_radius
    ds_se1_radius = val

def update_ds_se4(val):
    global ds_se4_radius
    ds_se4_radius = val

def update_min_area(val):
    global min_area
    min_area = val

def update_max_area(val):
    global max_area
    max_area = val

def create_tuner_dashboard(frame_idx, zarr_root, dish_mask):
    """
    Creates the visual dashboard for a given frame, applying current threshold values
    and displaying all detected fish.
    """
    global ds_thresh, ds_se1_radius, ds_se4_radius, min_area, max_area
    
    try:
        latest_bg_run = zarr_root['background_runs'].attrs['latest']
        images_ds_array = zarr_root['raw_video/images_ds']
        background_ds_array = zarr_root[f'background_runs/{latest_bg_run}/background_ds']
    except KeyError as e:
        print(f"Error: Missing necessary data in Zarr file: {e}")
        print("Please ensure you have run the 'import' and 'background' stages.")
        return None

    image_ds = images_ds_array[frame_idx]
    background_ds = background_ds_array[:]
    
    panel1 = cv2.cvtColor(image_ds, cv2.COLOR_GRAY2BGR)
    
    diff_ds = np.clip(background_ds.astype(np.int16) - image_ds.astype(np.int16), 0, 255).astype(np.uint8)
    if dish_mask is not None:
        diff_ds[dish_mask == 0] = 0
    panel2 = cv2.cvtColor(diff_ds, cv2.COLOR_GRAY2BGR)

    mask_ds = (diff_ds >= ds_thresh).astype(np.uint8) * 255
    se1_ds = disk(max(1, ds_se1_radius))
    se4_ds = disk(max(1, ds_se4_radius))
    processed_mask_ds = erosion(dilation(erosion(mask_ds, se1_ds), se4_ds), se1_ds)
    panel3 = cv2.cvtColor(processed_mask_ds, cv2.COLOR_GRAY2BGR)

    panel4 = cv2.cvtColor(image_ds, cv2.COLOR_GRAY2BGR)
    
    all_blobs = regionprops(label(processed_mask_ds))
    valid_blobs = [r for r in all_blobs if min_area <= r.area <= max_area]
    
    for blob in valid_blobs:
        min_r, min_c, max_r, max_c = blob.bbox
        cv2.rectangle(panel4, (min_c, min_r), (max_c, max_r), (0, 255, 0), 2)
        cv2.putText(panel4, f"A:{blob.area}", (min_c, min_r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    display_size = (480, 480)
    
    cv2.putText(panel1, "Original DS Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(panel2, "Background Difference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(panel3, f"Mask (thresh={ds_thresh}, se1={ds_se1_radius}, se4={ds_se4_radius})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(panel4, f"Detections: {len(valid_blobs)} (Area: {min_area}-{max_area})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    top_row = np.hstack((cv2.resize(panel1, display_size), cv2.resize(panel2, display_size)))
    bottom_row = np.hstack((cv2.resize(panel3, display_size), cv2.resize(panel4, display_size)))
    dashboard = np.vstack((top_row, bottom_row))
    
    return dashboard

def main(zarr_path, start_frame):
    global current_frame, ds_thresh, ds_se1_radius, ds_se4_radius, min_area, max_area
    current_frame = start_frame
    
    config_path = Path("src/pipeline_config.yaml")
    config = {}
    
    # --- NEW: Load parameters from config file first ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            detect_params = config.get('detect', {})
            ds_thresh = detect_params.get('ds_thresh', ds_thresh)
            ds_se1_radius = detect_params.get('se1_radius', ds_se1_radius)
            ds_se4_radius = detect_params.get('se4_radius', ds_se4_radius)
            min_area = detect_params.get('min_area', min_area)
            max_area = detect_params.get('max_area', max_area)
            print(f"✅ Loaded initial parameters from {config_path}")
    except FileNotFoundError:
        print(f"⚠️ Config file not found at {config_path}. Using default parameters.")
    except Exception as e:
        print(f"❌ Error reading config file: {e}. Using default parameters.")

    try:
        zarr_root = zarr.open(zarr_path, mode='r')
        num_frames = zarr_root['raw_video/images_ds'].shape[0]
        ds_img_shape = zarr_root['raw_video/images_ds'].shape[1:]
        
        # --- NEW: Robustly load dish mask ---
        mask_params = {}
        # Priority 1: Load from the latest detect run
        if 'detect_runs' in zarr_root:
            latest_run = zarr_root['detect_runs'].attrs['latest']
            mask_params = zarr_root[f'detect_runs/{latest_run}'].attrs.get('parameters', {}).get('dish_mask', {})
            print("Found 'detect_runs'. Will use mask parameters from the latest run.")
        # Priority 2: Load from the config file if detect has not been run
        elif config:
             mask_params = config.get('detect', {}).get('dish_mask', {})
             print("No 'detect_runs' found. Will use mask parameters from config file.")
        
        dish_mask = None # By default, no mask
        if mask_params.get('shape') == 'rectangle' and 'roi' in mask_params:
            x, y, w, h = mask_params['roi']
            dish_mask = np.zeros(ds_img_shape, dtype=np.uint8)
            cv2.rectangle(dish_mask, (x, y), (x + w, y + h), 255, -1)
            print("✅ Loaded rectangular dish mask.")
        else:
            print("ℹ️ No dish mask defined or found. Processing the full frame.")

    except Exception as e:
        print(f"❌ Error opening Zarr file or loading data: {e}")
        return

    window_name = "Detection Parameter Tuner"
    cv2.namedWindow(window_name)
    
    cv2.createTrackbar("Frame", window_name, current_frame, num_frames - 1, update_frame)
    cv2.createTrackbar("ds_thresh", window_name, ds_thresh, 255, update_ds_thresh)
    cv2.createTrackbar("ds_erode(se1)", window_name, ds_se1_radius, 10, update_ds_se1)
    cv2.createTrackbar("ds_dilate(se4)", window_name, ds_se4_radius, 10, update_ds_se4)
    cv2.createTrackbar("min_area", window_name, min_area, 1000, update_min_area)
    cv2.createTrackbar("max_area", window_name, max_area, 1000, update_max_area)

    print("\n🚀 Starting Detection Parameter Tuner...")
    print("Controls:")
    print("  - Adjust sliders to see results.")
    print("  - Use Left/Right arrow keys to step through frames.")
    print("  - Press 's' to SAVE the current parameters to pipeline_config.yaml.")
    print("  - Press 'q' or Esc to quit without saving.")

    while True:
        dashboard = create_tuner_dashboard(current_frame, zarr_root, dish_mask)
        if dashboard is None:
            break # Exit if there was a critical error creating the dashboard
        
        cv2.imshow(window_name, dashboard)
        cv2.setTrackbarPos("Frame", window_name, current_frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == 83: # Right arrow
            current_frame = min(num_frames - 1, current_frame + 1)
        elif key == 81: # Left arrow
            current_frame = max(0, current_frame - 1)
        elif key == ord('s'):
            try:
                # Re-read the config in case it changed
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    config = {} # Create a new config dict if file doesn't exist
                
                if 'detect' not in config:
                    config['detect'] = {}
                
                # Update parameters
                config['detect']['ds_thresh'] = ds_thresh
                config['detect']['se1_radius'] = ds_se1_radius
                config['detect']['se4_radius'] = ds_se4_radius
                config['detect']['min_area'] = min_area
                config['detect']['max_area'] = max_area
                
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                print(f"✅ Parameters saved to {config_path}")
            except Exception as e:
                print(f"❌ Error saving parameters: {e}")

    cv2.destroyAllWindows()
    print("\nTuner closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively tune detection thresholds for the tracking pipeline.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file.")
    parser.add_argument("start_frame", type=int, nargs='?', default=1, help="Frame to start on.")
    args = parser.parse_args()
    
    main(args.zarr_path, args.start_frame)