import cv2
import numpy as np
import os
import argparse
import zarr

def create_dashboard_view(frame_number, zarr_group, column_map):
    """
    Reads all image data for a frame from the Zarr group and assembles a 
    2x2 dashboard for visualization.
    """
    # --- CHANGED: Point to the faster, downsampled datasets ---
    images_array = zarr_group['raw_video/images_ds'] 
    background_array = zarr_group['background_models/background_ds']
    
    # --- UNCHANGED: These paths are still correct ---
    roi_images_array = zarr_group['crop_data/roi_images']
    results_array = zarr_group['tracking/tracking_results']
    
    num_frames = images_array.shape[0]
    frame_index = frame_number - 1

    if not (0 <= frame_index < num_frames):
        return None

    # Read all necessary data for the frame
    main_image = images_array[frame_index]
    roi_image = roi_images_array[frame_index]
    background_image = background_array[:] 
    results = results_array[frame_index]

    # --- Panel 1: Main View with ROI Box ---
    main_view = cv2.cvtColor(main_image, cv2.COLOR_GRAY2BGR)
    full_h, full_w = main_view.shape[:2] # This is now 640x640
    roi_sz = roi_image.shape
    
    bbox_x_norm = results[column_map['bbox_x_norm']]
    bbox_y_norm = results[column_map['bbox_y_norm']]
    if not np.isnan(bbox_x_norm):
        # The normalized coordinates work perfectly regardless of image size
        full_centroid_px = (int(bbox_x_norm * full_w), int(bbox_y_norm * full_h))
        # We need to scale the ROI size for drawing the box on the downsampled image
        roi_display_w = int((roi_sz[1] / 4512) * full_w) # Scale ROI width relative to original size
        roi_display_h = int((roi_sz[0] / 4512) * full_h) # Scale ROI height relative to original size
        x1 = full_centroid_px[0] - (roi_display_w // 2)
        y1 = full_centroid_px[1] - (roi_display_h // 2)
        cv2.rectangle(main_view, (x1, y1), (x1 + roi_display_w, y1 + roi_display_h), (0, 255, 255), 1)
    
    # --- Panel 2: Annotated ROI View (Unchanged) ---
    roi_view = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    colors = {'bladder': (0, 0, 255), 'eye_l': (0, 255, 0), 'eye_r': (255, 100, 0)}
    keypoints = {
        'bladder': (results[column_map['bladder_x_roi_norm']], results[column_map['bladder_y_roi_norm']]),
        'eye_l': (results[column_map['eye_l_x_roi_norm']], results[column_map['eye_l_y_roi_norm']]),
        'eye_r': (results[column_map['eye_r_x_roi_norm']], results[column_map['eye_r_y_roi_norm']])
    }
    for name, (x_norm, y_norm) in keypoints.items():
        if not np.isnan(x_norm):
            x_center = int(x_norm * roi_sz[1])
            y_center = int(y_norm * roi_sz[0])
            cv2.circle(roi_view, (x_center, y_center), 4, colors.get(name), -1)
            cv2.circle(roi_view, (x_center, y_center), 5, (0,0,0), 1)

    # --- Panel 3: Background Model ---
    background_view = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)

    # --- Panel 4: Difference Image ---
    diff_image = cv2.absdiff(main_image, background_image)
    diff_view = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2BGR)

    # --- Assemble the Dashboard ---
    display_size = (480, 480)
    # The main_view is now 640x640, so resizing is still appropriate
    main_resized = cv2.resize(main_view, display_size, interpolation=cv2.INTER_AREA)
    # The roi_view is 320x320, so we resize it up
    roi_resized = cv2.resize(roi_view, display_size, interpolation=cv2.INTER_NEAREST)
    bg_resized = cv2.resize(background_view, display_size, interpolation=cv2.INTER_AREA)
    diff_resized = cv2.resize(diff_view, display_size, interpolation=cv2.INTER_AREA)

    # Add titles to each panel
    cv2.putText(main_resized, "Full View (Downsampled)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(roi_resized, "ROI Detail", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(bg_resized, "Background Model (DS)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(diff_resized, "Difference Image (DS)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    top_row = np.hstack((main_resized, roi_resized))
    bottom_row = np.hstack((bg_resized, diff_resized))
    dashboard = np.vstack((top_row, bottom_row))
    
    cv2.putText(dashboard, f"Frame: {frame_number}", (10, dashboard.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return dashboard

def main(zarr_path, start_frame):
    """Main display loop for interactive dashboard visualization."""
    try:
        zarr_group = zarr.open_group(zarr_path, mode='r')
    except Exception as e:
        print(f"Error opening Zarr store at '{zarr_path}': {e}")
        return

    # CHANGED: Point to correct dataset paths
    num_frames = zarr_group['raw_video/images_full'].shape[0]
    results_array = zarr_group['tracking/tracking_results']
    
    column_names = results_array.attrs['column_names']
    column_map = {name: i for i, name in enumerate(column_names)}
    
    current_frame = start_frame
    
    print("Starting interactive dashboard...")
    print("Controls: → (Next Frame), ← (Previous Frame), 'q' or Esc (Quit)")

    while True:
        dashboard = create_dashboard_view(current_frame, zarr_group, column_map)
        
        if dashboard is None:
            display_image = np.zeros((960, 960, 3), dtype=np.uint8)
            cv2.putText(display_image, "Frame Not Found", (300, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        else:
            display_image = dashboard
            
        cv2.imshow("Interactive Dashboard", display_image)
        
        key = cv2.waitKey(0)

        if key == ord('q') or key == 27:
            break
        elif key == 83: # Right arrow
            current_frame = min(num_frames, current_frame + 1)
        elif key == 81: # Left arrow
            current_frame = max(1, current_frame - 1)

    cv2.destroyAllWindows()
    print("Visualizer closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively visualize fish tracking results.")
    # CHANGED: zarr_path is now a required argument
    parser.add_argument("zarr_path", type=str, help="Path to the unified Zarr file (e.g., video.zarr).")
    parser.add_argument("start_frame", type=int, nargs='?', default=1, 
                        help="The frame number to start visualizing from. Defaults to 1.")
    args = parser.parse_args()

    main(args.zarr_path, args.start_frame)