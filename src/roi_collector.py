import cv2
import numpy as np
import zarr
import argparse
from pathlib import Path

# Global variables
clicked_point = None
current_frame_info = None

def find_correction_candidates(zarr_root):
    """Finds frames where cropping likely succeeded but tracking failed."""
    print("Scanning for frames that need correction...")
    
    latest_tracking_run = zarr_root['tracking_runs'].attrs['latest']
    latest_crop_run = zarr_root['crop_runs'].attrs['latest']
    
    tracking_results = zarr_root[f'tracking_runs/{latest_tracking_run}/tracking_results']
    crop_coords = zarr_root[f'crop_runs/{latest_crop_run}/bbox_norm_coords']

    # Condition 1: Tracking failed (heading is NaN)
    tracking_failed_mask = np.isnan(tracking_results[:, 0])
    
    # Condition 2: Cropping succeeded (coordinates are not NaN)
    cropping_succeeded_mask = ~np.isnan(crop_coords[:, 0])
    
    # Candidates are where both conditions are true
    candidate_mask = tracking_failed_mask & cropping_succeeded_mask
    candidate_indices = np.where(candidate_mask)[0]
    
    print(f"Found {len(candidate_indices)} frames to review.")
    return candidate_indices.tolist()

def mouse_callback(event, x, y, flags, param):
    """Handles mouse click events to record the new center point."""
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"New center selected at: {clicked_point}")

def save_correction(zarr_root, frame_idx, new_center_px_ds, roi_sz):
    """Recalculates and overwrites the ROI data in the Zarr file."""
    print(f"Saving correction for frame {frame_idx}...")
    
    # Get necessary arrays and metadata
    latest_crop_run = zarr_root['crop_runs'].attrs['latest']
    images_full_array = zarr_root['raw_video/images_full']
    images_ds_array = zarr_root['raw_video/images_ds']
    roi_images_array = zarr_root[f'crop_runs/{latest_crop_run}/roi_images']
    roi_coords_full_array = zarr_root[f'crop_runs/{latest_crop_run}/roi_coordinates_full']
    roi_coords_ds_array = zarr_root[f'crop_runs/{latest_crop_run}/roi_coordinates_ds']
    crop_bbox_array = zarr_root[f'crop_runs/{latest_crop_run}/bbox_norm_coords']
    
    full_img_shape = images_full_array.shape[1:]
    ds_img_shape = images_ds_array.shape[1:]
    
    # --- Recalculate everything based on the new center point ---
    
    # 1. New normalized coordinates
    new_center_norm = (new_center_px_ds[0] / ds_img_shape[1], new_center_px_ds[1] / ds_img_shape[0])
    
    # 2. New full-resolution pixel coordinates
    full_centroid_px = (int(new_center_norm[0] * full_img_shape[1]), int(new_center_norm[1] * full_img_shape[0]))
    
    # 3. New full-resolution ROI coordinates (top-left corner)
    roi_x1_full = full_centroid_px[0] - roi_sz[1] // 2
    roi_y1_full = full_centroid_px[1] - roi_sz[0] // 2
    
    # 4. New downsampled ROI coordinates
    roi_size_ds = (int(roi_sz[0] * (ds_img_shape[0] / full_img_shape[0])), int(roi_sz[1] * (ds_img_shape[1] / full_img_shape[1])))
    roi_x1_ds = new_center_px_ds[0] - roi_size_ds[1] // 2
    roi_y1_ds = new_center_px_ds[1] - roi_size_ds[0] // 2
    
    # 5. Extract the new ROI image
    new_roi_image = images_full_array[frame_idx, roi_y1_full:roi_y1_full + roi_sz[0], roi_x1_full:roi_x1_full + roi_sz[1]]

    if new_roi_image.shape != tuple(roi_sz):
        print("Error: New ROI is out of bounds. Correction not saved.")
        return

    # --- Overwrite the data in the Zarr file ---
    # This is a destructive action!
    roi_images_array[frame_idx] = new_roi_image
    roi_coords_full_array[frame_idx] = (roi_x1_full, roi_y1_full)
    roi_coords_ds_array[frame_idx] = (roi_x1_ds, roi_y1_ds)
    crop_bbox_array[frame_idx] = new_center_norm
    
    print(f"Successfully saved new data for frame {frame_idx}.")

def main(zarr_path):
    global clicked_point, current_frame_info
    
    try:
        # Open in append mode to allow writing
        zarr_root = zarr.open(zarr_path, mode='a')
    except Exception as e:
        print(f"Error opening Zarr file in 'a' mode: {e}")
        return

    candidate_indices = find_correction_candidates(zarr_root)
    if not candidate_indices:
        print("No frames need correction. Exiting.")
        return
        
    # Get ROI size from config
    try:
        roi_sz = zarr_root['pipeline_params'].attrs['crop']['roi_sz']
    except KeyError:
        print("Warning: Could not read ROI size from metadata. Using default [256, 256].")
        roi_sz = [256, 256]

    window_name = "ROI Corrector"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    idx_pointer = 0
    
    print("\n--- Starting ROI Corrector ---")
    print("Instructions:")
    print("  - Click on the fish to set the new center.")
    print("  - Press 's' to SAVE the correction.")
    print("  - Press 'n' to SKIP to the next frame.")
    print("  - Press 'q' to QUIT.")

    while idx_pointer < len(candidate_indices):
        frame_idx = candidate_indices[idx_pointer]
        
        image_ds = zarr_root['raw_video/images_ds'][frame_idx]
        display_image = cv2.cvtColor(image_ds, cv2.COLOR_GRAY2BGR)
        
        # Draw a marker for the newly clicked point
        if clicked_point:
            cv2.drawMarker(display_image, clicked_point, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        status_text = f"Reviewing Frame: {frame_idx} ({idx_pointer + 1}/{len(candidate_indices)})"
        cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(window_name, display_image)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            if clicked_point:
                save_correction(zarr_root, frame_idx, clicked_point, roi_sz)
                clicked_point = None
                idx_pointer += 1
            else:
                print("No point selected. Click on the fish first, then press 's'.")
        elif key == ord('n'):
            print(f"Skipping frame {frame_idx}.")
            clicked_point = None
            idx_pointer += 1
            
    cv2.destroyAllWindows()
    print("\nROI correction session finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively correct misplaced ROIs in a Zarr file.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file to correct.")
    args = parser.parse_args()
    main(args.zarr_path)