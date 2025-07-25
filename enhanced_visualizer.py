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

def detect_data_format(column_names):
    """
    Detect whether this is original or enhanced data format.
    
    Returns:
        dict: Column mappings for the detected format
    """
    if 'bbox_x_norm_ds' in column_names:
        # Enhanced format (20 columns)
        print("✅ Detected enhanced data format with multi-scale coordinates")
        return {
            'format': 'enhanced',
            'heading_degrees': 'heading_degrees',
            'bbox_x_norm': 'bbox_x_norm_ds',  # Use downsampled coords for visualization
            'bbox_y_norm': 'bbox_y_norm_ds',
            'bladder_x_roi_norm': 'bladder_x_roi_norm',
            'bladder_y_roi_norm': 'bladder_y_roi_norm',
            'eye_l_x_roi_norm': 'eye_l_x_roi_norm',
            'eye_l_y_roi_norm': 'eye_l_y_roi_norm',
            'eye_r_x_roi_norm': 'eye_r_x_roi_norm',
            'eye_r_y_roi_norm': 'eye_r_y_roi_norm',
            # Additional enhanced columns available
            'bbox_width_norm_ds': 'bbox_width_norm_ds',
            'bbox_height_norm_ds': 'bbox_height_norm_ds',
            'bbox_x_norm_full': 'bbox_x_norm_full',
            'bbox_y_norm_full': 'bbox_y_norm_full',
            'confidence_score': 'confidence_score'
        }
    else:
        # Original format (9 columns)
        print("✅ Detected original data format")
        return {
            'format': 'original',
            'heading_degrees': 'heading_degrees',
            'bbox_x_norm': 'bbox_x_norm',
            'bbox_y_norm': 'bbox_y_norm',
            'bladder_x_roi_norm': 'bladder_x_roi_norm',
            'bladder_y_roi_norm': 'bladder_y_roi_norm',
            'eye_l_x_roi_norm': 'eye_l_x_roi_norm',
            'eye_l_y_roi_norm': 'eye_l_y_roi_norm',
            'eye_r_x_roi_norm': 'eye_r_x_roi_norm',
            'eye_r_y_roi_norm': 'eye_r_y_roi_norm'
        }

def create_dashboard_view(frame_number, zarr_group, column_map, format_info):
    """
    Enhanced dashboard that works with both original and enhanced data formats.
    """
    # Use downsampled images for visualization
    images_array = zarr_group['raw_video/images_ds'] 
    background_array = zarr_group['background_models/background_ds']
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
    full_h, full_w = main_view.shape[:2] # This is 640x640
    roi_sz = roi_image.shape
    
    bbox_x_norm = results[column_map[format_info['bbox_x_norm']]]
    bbox_y_norm = results[column_map[format_info['bbox_y_norm']]]
    heading_degrees = results[column_map[format_info['heading_degrees']]]
    
    # Enhanced format has bounding box dimensions
    if format_info['format'] == 'enhanced':
        bbox_width_norm = results[column_map[format_info['bbox_width_norm_ds']]]
        bbox_height_norm = results[column_map[format_info['bbox_height_norm_ds']]]
        confidence = results[column_map[format_info['confidence_score']]]
        
        # Draw actual bounding box with dimensions
        if not any(np.isnan([bbox_x_norm, bbox_y_norm, bbox_width_norm, bbox_height_norm])):
            center_x = int(bbox_x_norm * full_w)
            center_y = int(bbox_y_norm * full_h)
            box_w = int(bbox_width_norm * full_w)
            box_h = int(bbox_height_norm * full_h)
            
            x1 = center_x - box_w // 2
            y1 = center_y - box_h // 2
            x2 = x1 + box_w
            y2 = y1 + box_h
            
            # Draw YOLO-style bounding box
            cv2.rectangle(main_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(main_view, (center_x, center_y), 3, (0, 255, 0), -1)
            
            # Add confidence score
            label = f"Fish {confidence:.3f}"
            cv2.putText(main_view, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        # Original format - draw ROI box like before
        if not np.isnan(bbox_x_norm):
            full_centroid_px = (int(bbox_x_norm * full_w), int(bbox_y_norm * full_h))
            roi_display_w = int((roi_sz[1] / 4512) * full_w)
            roi_display_h = int((roi_sz[0] / 4512) * full_h) 
            x1 = full_centroid_px[0] - (roi_display_w // 2)
            y1 = full_centroid_px[1] - (roi_display_h // 2)
            cv2.rectangle(main_view, (x1, y1), (x1 + roi_display_w, y1 + roi_display_h), (0, 255, 255), 1)
    
    # Draw heading arrow (same for both formats)
    if not any(np.isnan([bbox_x_norm, bbox_y_norm, heading_degrees])):
        center_px = (int(bbox_x_norm * full_w), int(bbox_y_norm * full_h))
        arrow_length = 30
        arrow_end_x = int(center_px[0] + arrow_length * np.cos(np.deg2rad(heading_degrees)))
        arrow_end_y = int(center_px[1] - arrow_length * np.sin(np.deg2rad(heading_degrees)))
        cv2.arrowedLine(main_view, center_px, (arrow_end_x, arrow_end_y), (255, 0, 255), 2, tipLength=0.3)
    
    # --- Panel 2: Original ROI View ---
    roi_view = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    colors = {'bladder': (0, 0, 255), 'eye_l': (0, 255, 0), 'eye_r': (255, 100, 0)}
    keypoints = {
        'bladder': (results[column_map[format_info['bladder_x_roi_norm']]], results[column_map[format_info['bladder_y_roi_norm']]]),
        'eye_l': (results[column_map[format_info['eye_l_x_roi_norm']]], results[column_map[format_info['eye_l_y_roi_norm']]]),
        'eye_r': (results[column_map[format_info['eye_r_x_roi_norm']]], results[column_map[format_info['eye_r_y_roi_norm']]])
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
        rotation_needed = 90 - heading_degrees
        stabilized_roi = rotate_roi_to_heading(roi_image, rotation_needed)
        stabilized_view = cv2.cvtColor(stabilized_roi, cv2.COLOR_GRAY2BGR)
        
        # For stabilized view, we need to rotate the keypoint coordinates too
        roi_center = np.array([roi_sz[1]/2, roi_sz[0]/2])
        rotation_angle_rad = np.deg2rad(rotation_needed)
        cos_angle, sin_angle = np.cos(rotation_angle_rad), np.sin(rotation_angle_rad)
        
        for name, (x_norm, y_norm) in keypoints.items():
            if not np.isnan(x_norm):
                x_pixel = x_norm * roi_sz[1]
                y_pixel = y_norm * roi_sz[0]
                
                rel_x = x_pixel - roi_center[0]
                rel_y = y_pixel - roi_center[1]
                rotated_x = rel_x * cos_angle - rel_y * sin_angle + roi_center[0]
                rotated_y = rel_x * sin_angle + rel_y * cos_angle + roi_center[1]
                
                cv2.circle(stabilized_view, (int(rotated_x), int(rotated_y)), 4, colors.get(name), -1)
                cv2.circle(stabilized_view, (int(rotated_x), int(rotated_y)), 5, (0,0,0), 1)
        
        cv2.putText(stabilized_view, "Stabilized (Fish Up)", (10, roi_sz[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
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
    main_resized = cv2.resize(main_view, display_size, interpolation=cv2.INTER_AREA)
    roi_resized = cv2.resize(roi_view, display_size, interpolation=cv2.INTER_NEAREST)
    stabilized_resized = cv2.resize(stabilized_view, display_size, interpolation=cv2.INTER_NEAREST)
    diff_resized = cv2.resize(diff_view, display_size, interpolation=cv2.INTER_AREA)

    # Add titles to each panel
    if format_info['format'] == 'enhanced':
        title1 = "Full View + YOLO Bbox"
        if not np.isnan(confidence):
            title1 += f" (conf: {confidence:.3f})"
    else:
        title1 = "Full View + Heading"
        
    cv2.putText(main_resized, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(roi_resized, "Original ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(stabilized_resized, "Stabilized ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(diff_resized, "Difference Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    top_row = np.hstack((main_resized, roi_resized))
    bottom_row = np.hstack((stabilized_resized, diff_resized))
    dashboard = np.vstack((top_row, bottom_row))
    
    # Enhanced status info
    status_text = f"Frame: {frame_number}"
    if format_info['format'] == 'enhanced':
        status_text += f" | Format: Enhanced ({len(column_map)} cols)"
        if not np.isnan(confidence):
            status_text += f" | Confidence: {confidence:.3f}"
    else:
        status_text += f" | Format: Original ({len(column_map)} cols)"
    
    cv2.putText(dashboard, status_text, (10, dashboard.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return dashboard

def main(zarr_path, start_frame):
    """Main display loop for interactive dashboard visualization."""
    try:
        zarr_group = zarr.open_group(zarr_path, mode='r')
    except Exception as e:
        print(f"Error opening Zarr store at '{zarr_path}': {e}")
        return

    num_frames = zarr_group['raw_video/images_ds'].shape[0]
    results_array = zarr_group['tracking/tracking_results']
    
    column_names = results_array.attrs['column_names']
    column_map = {name: i for i, name in enumerate(column_names)}
    
    # Detect data format and create appropriate mappings
    format_info = detect_data_format(column_names)
    
    print(f"📊 Data info: {num_frames} frames, {len(column_names)} tracking columns")
    if format_info['format'] == 'enhanced':
        print("🎯 Enhanced features: Multi-scale coordinates, YOLO bounding boxes, confidence scores")
    
    current_frame = start_frame
    
    print("Starting enhanced interactive dashboard...")
    print("Controls: → (Next Frame), ← (Previous Frame), 'q' or Esc (Quit)")

    while True:
        dashboard = create_dashboard_view(current_frame, zarr_group, column_map, format_info)
        
        if dashboard is None:
            display_image = np.zeros((960, 960, 3), dtype=np.uint8)
            cv2.putText(display_image, "Frame Not Found", (300, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        else:
            display_image = dashboard
            
        cv2.imshow("Enhanced Interactive Dashboard", display_image)
        
        key = cv2.waitKey(0)

        if key == ord('q') or key == 27:
            break
        elif key == 83: # Right arrow
            current_frame = min(num_frames, current_frame + 1)
        elif key == 81: # Left arrow
            current_frame = max(1, current_frame - 1)

    cv2.destroyAllWindows()
    print("Enhanced visualizer closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced interactive visualizer for fish tracking results.")
    parser.add_argument("zarr_path", type=str, help="Path to the unified Zarr file (e.g., video.zarr).")
    parser.add_argument("start_frame", type=int, nargs='?', default=1, 
                        help="The frame number to start visualizing from. Defaults to 1.")
    args = parser.parse_args()

    main(args.zarr_path, args.start_frame)