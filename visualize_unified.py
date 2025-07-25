import cv2
import numpy as np
import os
import argparse
import zarr

def rotate_roi_to_heading(roi_image, heading_degrees):
    """
    Rotate the ROI image so the fish is oriented according to the heading angle.
    """
    if np.isnan(heading_degrees):
        return roi_image
    
    # Get ROI center
    h, w = roi_image.shape
    center = (w // 2, h // 2)
    
    # Create rotation matrix (negative because CV2 rotates clockwise, but we want counterclockwise)
    rotation_matrix = cv2.getRotationMatrix2D(center, -heading_degrees, 1.0)
    
    # Rotate the image
    rotated_roi = cv2.warpAffine(roi_image, rotation_matrix, (w, h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=0)
    
    return rotated_roi

def create_fish_reference_frame_view(main_image, roi_image, background_image, bbox_x_norm, bbox_y_norm, heading_degrees):
    """
    Transform the current frame into the fish's reference frame - fish stays centered and 
    oriented upward while everything else rotates and translates around it.
    """
    if np.isnan(bbox_x_norm) or np.isnan(heading_degrees):
        # If no fish detected, just return the current frame
        return cv2.cvtColor(main_image, cv2.COLOR_GRAY2BGR)
    
    # Get fish position in image coordinates
    full_h, full_w = main_image.shape
    fish_x = bbox_x_norm * full_w
    fish_y = bbox_y_norm * full_h
    
    # Center of the output image
    center_x = full_w / 2
    center_y = full_h / 2
    
    # Translation to center the fish
    translate_x = center_x - fish_x
    translate_y = center_y - fish_y
    
    # Combined transformation matrix: translate then rotate around new center
    # First, translate to put fish at center
    T1 = np.array([[1, 0, translate_x],
                   [0, 1, translate_y],
                   [0, 0, 1]], dtype=np.float32)
    
    # Then rotate around the center to make fish point upward
    # We want fish to point upward (90 degrees), so rotate by (90 - heading)
    target_angle = 90 - heading_degrees
    R = cv2.getRotationMatrix2D((center_x, center_y), target_angle, 1.0)
    
    # Convert to 3x3 matrix for multiplication
    R_3x3 = np.vstack([R, [0, 0, 1]])
    
    # Combine transformations: final = R * T1
    combined = R_3x3 @ T1
    final_transform = combined[:2, :]  # Back to 2x3 for cv2.warpAffine
    
    # Apply transformation to the current frame (not just background!)
    transformed_frame = cv2.warpAffine(main_image, final_transform, (full_w, full_h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
    
    # Convert to color
    fish_ref_view = cv2.cvtColor(transformed_frame, cv2.COLOR_GRAY2BGR)
    
    # Draw a small cross at the center to show where the fish is
    cv2.drawMarker(fish_ref_view, (int(center_x), int(center_y)), (0, 255, 255), 
                   cv2.MARKER_CROSS, 20, 2)
    
    # Draw an arrow pointing up to show fish orientation
    arrow_start = (int(center_x), int(center_y + 15))
    arrow_end = (int(center_x), int(center_y - 15))
    cv2.arrowedLine(fish_ref_view, arrow_start, arrow_end, (255, 0, 255), 2, tipLength=0.3)
    
    return fish_ref_view

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
    heading_degrees = results[column_map['heading_degrees']]
    
    if not np.isnan(bbox_x_norm):
        # The normalized coordinates work perfectly regardless of image size
        full_centroid_px = (int(bbox_x_norm * full_w), int(bbox_y_norm * full_h))
        # We need to scale the ROI size for drawing the box on the downsampled image
        roi_display_w = int((roi_sz[1] / 4512) * full_w) # Scale ROI width relative to original size
        roi_display_h = int((roi_sz[0] / 4512) * full_h) # Scale ROI height relative to original size
        x1 = full_centroid_px[0] - (roi_display_w // 2)
        y1 = full_centroid_px[1] - (roi_display_h // 2)
        cv2.rectangle(main_view, (x1, y1), (x1 + roi_display_w, y1 + roi_display_h), (0, 255, 255), 1)
        
        # Draw heading arrow
        if not np.isnan(heading_degrees):
            arrow_length = 30
            arrow_end_x = int(full_centroid_px[0] + arrow_length * np.cos(np.deg2rad(heading_degrees)))
            arrow_end_y = int(full_centroid_px[1] - arrow_length * np.sin(np.deg2rad(heading_degrees)))
            cv2.arrowedLine(main_view, full_centroid_px, (arrow_end_x, arrow_end_y), (255, 0, 255), 2, tipLength=0.3)
    
    # --- Panel 2: Original ROI View ---
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

    # Add heading text to original ROI view
    if not np.isnan(heading_degrees):
        cv2.putText(roi_view, f"Heading: {heading_degrees:.1f}°", (10, roi_sz[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Panel 3: Stabilized ROI View ---
    if not np.isnan(heading_degrees):
        # Rotate the ROI so fish always points upward (90 degrees)
        # We need to rotate by (90 - heading_degrees) to make the fish point up
        rotation_needed = 90 - heading_degrees
        stabilized_roi = rotate_roi_to_heading(roi_image, rotation_needed)
        stabilized_view = cv2.cvtColor(stabilized_roi, cv2.COLOR_GRAY2BGR)
        
        # For stabilized view, we need to rotate the keypoint coordinates too
        roi_center = np.array([roi_sz[1]/2, roi_sz[0]/2])
        rotation_angle_rad = np.deg2rad(rotation_needed)  # Same rotation as image
        cos_angle, sin_angle = np.cos(rotation_angle_rad), np.sin(rotation_angle_rad)
        
        for name, (x_norm, y_norm) in keypoints.items():
            if not np.isnan(x_norm):
                # Convert normalized coords to pixel coords
                x_pixel = x_norm * roi_sz[1]
                y_pixel = y_norm * roi_sz[0]
                
                # Rotate around ROI center
                rel_x = x_pixel - roi_center[0]
                rel_y = y_pixel - roi_center[1]
                rotated_x = rel_x * cos_angle - rel_y * sin_angle + roi_center[0]
                rotated_y = rel_x * sin_angle + rel_y * cos_angle + roi_center[1]
                
                cv2.circle(stabilized_view, (int(rotated_x), int(rotated_y)), 4, colors.get(name), -1)
                cv2.circle(stabilized_view, (int(rotated_x), int(rotated_y)), 5, (0,0,0), 1)
        
        # Add text showing this is stabilized
        cv2.putText(stabilized_view, "Stabilized (Fish Up)", (10, roi_sz[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        # No heading data, show normal ROI
        stabilized_view = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        for name, (x_norm, y_norm) in keypoints.items():
            if not np.isnan(x_norm):
                x_center = int(x_norm * roi_sz[1])
                y_center = int(y_norm * roi_sz[0])
                cv2.circle(stabilized_view, (x_center, y_center), 4, colors.get(name), -1)
                cv2.circle(stabilized_view, (x_center, y_center), 5, (0,0,0), 1)

    # --- Panel 4: Difference Image ---
    diff_image = cv2.absdiff(main_image, background_image)
    diff_view = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2BGR)

    # --- Assemble the Dashboard ---
    display_size = (480, 480)
    # The main_view is now 640x640, so resizing is still appropriate
    main_resized = cv2.resize(main_view, display_size, interpolation=cv2.INTER_AREA)
    # The roi_view is 320x320, so we resize it up
    roi_resized = cv2.resize(roi_view, display_size, interpolation=cv2.INTER_NEAREST)
    stabilized_resized = cv2.resize(stabilized_view, display_size, interpolation=cv2.INTER_NEAREST)
    diff_resized = cv2.resize(diff_view, display_size, interpolation=cv2.INTER_AREA)

    # Add titles to each panel
    cv2.putText(main_resized, "Full View + Heading", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(roi_resized, "Original ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(stabilized_resized, "Stabilized ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(diff_resized, "Difference Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    top_row = np.hstack((main_resized, roi_resized))
    bottom_row = np.hstack((stabilized_resized, diff_resized))
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