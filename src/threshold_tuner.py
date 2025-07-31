import cv2
import numpy as np
import zarr
import argparse
from skimage.morphology import disk, erosion, dilation

# Global variables to store trackbar values
ds_thresh = 55
roi_thresh = 115
current_frame = 1
# --- NEW: Globals for morphology parameters ---
ds_se1_radius = 1
ds_se4_radius = 4
roi_se1_radius = 1
roi_se2_radius = 2

def update_ds_thresh(val):
    global ds_thresh
    ds_thresh = val

def update_roi_thresh(val):
    global roi_thresh
    roi_thresh = val

def update_frame(val):
    global current_frame
    current_frame = val

# --- Callbacks for morphology sliders ---
def update_ds_se1(val):
    global ds_se1_radius
    ds_se1_radius = val

def update_ds_se4(val):
    global ds_se4_radius
    ds_se4_radius = val

def update_roi_se1(val):
    global roi_se1_radius
    roi_se1_radius = val

def update_roi_se2(val):
    global roi_se2_radius
    roi_se2_radius = val


def create_tuner_dashboard(frame_idx, zarr_root):
    """
    Creates the visual dashboard for a given frame, applying current threshold values.
    """
    global ds_thresh, roi_thresh, ds_se1_radius, ds_se4_radius, roi_se1_radius, roi_se2_radius
    
    # --- Dynamically get the latest runs ---
    try:
        latest_bg_run = zarr_root['background_runs'].attrs['latest']
        latest_crop_run = zarr_root['crop_runs'].attrs['latest']
        
        images_ds_array = zarr_root['raw_video/images_ds']
        background_ds_array = zarr_root[f'background_runs/{latest_bg_run}/background_ds']
        
        roi_images_array = zarr_root[f'crop_runs/{latest_crop_run}/roi_images']
        roi_coords_full_array = zarr_root[f'crop_runs/{latest_crop_run}/roi_coordinates_full']
        background_full_array = zarr_root[f'background_runs/{latest_bg_run}/background_full']
    except KeyError as e:
        print(f"Error: Missing necessary data in Zarr file: {e}")
        return None

    # Load data for the current frame
    image_ds = images_ds_array[frame_idx]
    background_ds = background_ds_array[:]
    roi_image = roi_images_array[frame_idx]
    
    # --- Panel 1 & 2: Downsampled View and Mask ---
    diff_ds = np.clip(background_ds.astype(np.int16) - image_ds.astype(np.int16), 0, 255).astype(np.uint8)
    mask_ds = (diff_ds >= ds_thresh).astype(np.uint8) * 255
    
    # --- MODIFICATION: Use slider values for morphology ---
    # Ensure radii are at least 1 to avoid errors with disk(0)
    se1_ds = disk(max(1, ds_se1_radius))
    se4_ds = disk(max(1, ds_se4_radius))
    processed_mask_ds = erosion(dilation(erosion(mask_ds, se1_ds), se4_ds), se1_ds)

    # --- Panel 3 & 4: ROI View and Mask ---
    roi_coords = roi_coords_full_array[frame_idx]
    
    if roi_coords[0] != -1: # Check if ROI is valid
        x1, y1 = roi_coords
        h, w = roi_image.shape
        background_roi = background_full_array[y1:y1+h, x1:x1+w]
        
        if background_roi.shape == roi_image.shape:
            diff_roi = np.clip(background_roi.astype(np.int16) - roi_image.astype(np.int16), 0, 255).astype(np.uint8)
            mask_roi = (diff_roi >= roi_thresh).astype(np.uint8) * 255
            
            # --- MODIFICATION: Use slider values for morphology ---
            se1_roi = disk(max(1, roi_se1_radius))
            se2_roi = disk(max(1, roi_se2_radius))
            processed_mask_roi = erosion(dilation(erosion(mask_roi, se1_roi), se2_roi), se1_roi)
        else:
            diff_roi = np.zeros_like(roi_image)
            processed_mask_roi = np.zeros_like(roi_image)
    else:
        diff_roi = np.zeros_like(roi_image)
        processed_mask_roi = np.zeros_like(roi_image)

    # --- Assemble Dashboard ---
    display_size = (480, 480)
    
    diff_ds_bgr = cv2.cvtColor(diff_ds, cv2.COLOR_GRAY2BGR)
    mask_ds_bgr = cv2.cvtColor(processed_mask_ds, cv2.COLOR_GRAY2BGR)
    diff_roi_bgr = cv2.cvtColor(diff_roi, cv2.COLOR_GRAY2BGR)
    mask_roi_bgr = cv2.cvtColor(processed_mask_roi, cv2.COLOR_GRAY2BGR)
    
    panel1 = cv2.resize(diff_ds_bgr, display_size)
    panel2 = cv2.resize(mask_ds_bgr, display_size)
    panel3 = cv2.resize(diff_roi_bgr, display_size)
    panel4 = cv2.resize(mask_roi_bgr, display_size)

    # --- MODIFICATION: Update titles to show morphology values ---
    cv2.putText(panel1, f"Downsampled Diff", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(panel2, f"Result (thresh={ds_thresh}, se1={ds_se1_radius}, se4={ds_se4_radius})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(panel3, f"ROI Diff", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(panel4, f"Result (thresh={roi_thresh}, se1={roi_se1_radius}, se2={roi_se2_radius})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    top_row = np.hstack((panel1, panel2))
    bottom_row = np.hstack((panel3, panel4))
    dashboard = np.vstack((top_row, bottom_row))
    
    return dashboard

def main(zarr_path, start_frame):
    global current_frame, ds_thresh, roi_thresh, ds_se1_radius, ds_se4_radius, roi_se1_radius, roi_se2_radius
    current_frame = start_frame
    
    try:
        zarr_root = zarr.open(zarr_path, mode='r')
        num_frames = zarr_root['raw_video/images_ds'].shape[0]
    except Exception as e:
        print(f"Error opening Zarr file: {e}")
        return

    window_name = "Threshold Tuner"
    cv2.namedWindow(window_name)
    
    # Create trackbars
    cv2.createTrackbar("Frame", window_name, current_frame, num_frames - 1, update_frame)
    cv2.createTrackbar("ds_thresh", window_name, ds_thresh, 255, update_ds_thresh)
    cv2.createTrackbar("roi_thresh", window_name, roi_thresh, 255, update_roi_thresh)
    # --- NEW: Morphology trackbars ---
    cv2.createTrackbar("ds_erode(se1)", window_name, ds_se1_radius, 10, update_ds_se1)
    cv2.createTrackbar("ds_dilate(se4)", window_name, ds_se4_radius, 10, update_ds_se4)
    cv2.createTrackbar("roi_erode(se1)", window_name, roi_se1_radius, 10, update_roi_se1)
    cv2.createTrackbar("roi_dilate(se2)", window_name, roi_se2_radius, 10, update_roi_se2)
    
    print("🚀 Starting Threshold Tuner...")
    print("Controls: Adjust sliders to see results. Press 'q' or Esc to quit.")

    while True:
        dashboard = create_tuner_dashboard(current_frame, zarr_root)
        
        if dashboard is not None:
            cv2.imshow(window_name, dashboard)
        
        cv2.setTrackbarPos("Frame", window_name, current_frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == 83: # Right arrow
            current_frame = min(num_frames - 1, current_frame + 1)
        elif key == 81: # Left arrow
            current_frame = max(0, current_frame - 1)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively tune detection thresholds for the tracking pipeline.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file.")
    parser.add_argument("start_frame", type=int, nargs='?', default=1, help="Frame to start on.")
    args = parser.parse_args()
    
    main(args.zarr_path, args.start_frame)