import cv2
import numpy as np
import os
import argparse
import zarr
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

current_frame = 1

def update_frame(val):
    """Callback function to update the global frame index when the slider is moved."""
    global current_frame
    current_frame = val

def create_summary_panel(crop_success_mask, track_success_mask, headings, positions, image_shape, output_path):
    """Creates a single summary panel with all static plots using matplotlib subplots."""
    num_frames = len(crop_success_mask)
    valid_track_indices = np.where(track_success_mask)[0]
    valid_headings = headings[track_success_mask].copy() # Use copy to avoid changing original data
    
    # Normalize headings to be in the [0, 360] range for consistent plotting
    valid_headings[valid_headings < 0] += 360
    
    # Define the layout: 4 rows, with the heatmap being much larger
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), dpi=150, 
                             gridspec_kw={'height_ratios': [0.5, 0.5, 2, 6]})
    fig.patch.set_facecolor('#2c2c2c') # Dark background for the whole figure

    # 1. Crop Success Timeline
    ax1 = axes[0]
    crop_timeline = np.zeros((1, num_frames))
    crop_timeline[0, crop_success_mask] = 1
    ax1.imshow(crop_timeline, cmap='Greens', vmin=0, vmax=1, aspect='auto', interpolation='none')
    ax1.set_title('Crop Success', color='white', fontsize=10)
    ax1.axis('off')

    # 2. Track Success Timeline
    ax2 = axes[1]
    track_timeline_data = np.zeros((1, num_frames))
    track_timeline_data[0, track_success_mask] = 1
    ax2.imshow(track_timeline_data, cmap='Greens', vmin=0, vmax=1, aspect='auto', interpolation='none')
    ax2.set_title('Track (Pose) Success', color='white', fontsize=10)
    ax2.axis('off')
    
    # 3. Heading Plot
    ax3 = axes[2]
    ax3.plot(valid_track_indices, valid_headings, color='cyan', linewidth=0.8)
    ax3.set_title('Fish Heading Over Time', color='white', fontsize=10)
    ax3.set_xlim(0, num_frames)
    ax3.set_ylim(-10, 370) # Y-axis is now padded
    ax3.tick_params(axis='x', colors='white', labelsize=8)
    ax3.tick_params(axis='y', colors='white', labelsize=8)
    ax3.set_facecolor('#1e1e1e')
    ax3.grid(True, alpha=0.2)

    # 4. Position Heatmap
    ax4 = axes[3]
    valid_positions = positions[~np.isnan(positions).any(axis=1)]
    if len(valid_positions) > 0:
        heatmap, _, _ = np.histogram2d(
            valid_positions[:, 1] * image_shape[0], # y
            valid_positions[:, 0] * image_shape[1], # x
            bins=100, range=[[0, image_shape[0]], [0, image_shape[1]]]
        )
        heatmap = np.log(heatmap + 1)
        ax4.imshow(heatmap.T, cmap='jet', aspect='equal', origin='lower', extent=[0, image_shape[1], 0, image_shape[0]])
    ax4.set_title('Position Heatmap', color='white', fontsize=10)
    ax4.set_xlim(0, image_shape[1])
    ax4.set_ylim(0, image_shape[0])
    ax4.axis('off')

    plt.tight_layout(pad=1.5)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return cv2.imread(str(output_path))


def apply_circular_mask(image):
    """Applies a circular mask to the image to hide corners."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(center[0], center[1])
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply the mask
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def rotate_roi_to_heading(roi_image, heading_degrees):
    """
    Rotate the ROI image so the fish is oriented according to the heading angle.
    A positive heading should result in a counter-clockwise rotation to stabilize.
    """
    if np.isnan(heading_degrees):
        return roi_image
    
    h, w = roi_image.shape
    center = (w // 2, h // 2)
    
    # Rotate by the negative of the heading to counteract the fish's orientation
    rotation_matrix = cv2.getRotationMatrix2D(center, -heading_degrees, 1.0)
    
    rotated_roi = cv2.warpAffine(roi_image, rotation_matrix, (w, h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=0)
    
    return rotated_roi

def create_dashboard_view(frame_number, zarr_group, column_map):
    """
    Creates a dashboard that visualizes the fish tracking data,
    showing crop results even if tracking fails.
    """
    # --- Dynamically get the latest runs ---
    latest_tracking_run = zarr_group['tracking_runs'].attrs['latest']
    latest_crop_run = zarr_group['crop_runs'].attrs['latest']
    
    # --- Load all necessary arrays ---
    images_array = zarr_group['raw_video/images_ds'] 
    roi_images_array = zarr_group[f'crop_runs/{latest_crop_run}/roi_images']
    results_array = zarr_group[f'tracking_runs/{latest_tracking_run}/tracking_results']
    
    if 'refine_runs' in zarr_group and 'latest' in zarr_group['refine_runs'].attrs:
        latest_refine_run = zarr_group['refine_runs'].attrs['latest']
        crop_bbox_array = zarr_group[f'refine_runs/{latest_refine_run}/refined_bbox_norm_coords']
    else:
        crop_bbox_array = zarr_group[f'crop_runs/{latest_crop_run}/bbox_norm_coords']

    num_frames = images_array.shape[0]
    frame_index = frame_number - 1

    if not (0 <= frame_index < num_frames):
        return None

    main_image = images_array[frame_index]
    roi_image = roi_images_array[frame_index]
    results = results_array[frame_index]
    crop_bbox = crop_bbox_array[frame_index]

    main_view = cv2.cvtColor(main_image, cv2.COLOR_GRAY2BGR)
    full_h, full_w = main_view.shape[:2]
    
    heading_degrees = results[column_map['heading_degrees']]
    confidence = results[column_map['confidence_score']]

    # Initialize a blank image for the crop view
    bbox_crop_view = np.zeros_like(main_view)
    crop_title = "BBox Crop (Not Tracked)"

    if not np.isnan(results[column_map['bbox_x_norm_ds']]):
        bbox_x_norm, bbox_y_norm = results[column_map['bbox_x_norm_ds']], results[column_map['bbox_y_norm_ds']]
        bbox_width_norm, bbox_height_norm = results[column_map['bbox_width_norm_ds']], results[column_map['bbox_height_norm_ds']]

        center_x, center_y = int(bbox_x_norm * full_w), int(bbox_y_norm * full_h)
        box_w, box_h = int(bbox_width_norm * full_w), int(bbox_height_norm * full_h)
        
        x1, y1 = center_x - box_w // 2, center_y - box_h // 2
        x2, y2 = x1 + box_w, y1 + box_h
        
        cv2.rectangle(main_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(main_view, (center_x, center_y), 3, (0, 255, 0), -1)
        label = f"Tracked {confidence:.3f}"
        cv2.putText(main_view, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Create the bounding box crop view
        crop_x1, crop_y1 = max(0, x1), max(0, y1)
        crop_x2, crop_y2 = min(full_w, x2), min(full_h, y2)
        if crop_x1 < crop_x2 and crop_y1 < crop_y2:
            bbox_crop = main_image[crop_y1:crop_y2, crop_x1:crop_x2]
            bbox_crop_view = cv2.cvtColor(bbox_crop, cv2.COLOR_GRAY2BGR)
            crop_title = f"BBox Crop ({box_w}x{box_h}px)"


    elif not np.isnan(crop_bbox[0]):
        bbox_x_norm, bbox_y_norm = crop_bbox
        box_w, box_h = int(0.08 * full_w), int(0.08 * full_h)
        center_x, center_y = int(bbox_x_norm * full_w), int(bbox_y_norm * full_h)
        x1, y1 = center_x - box_w // 2, center_y - box_h // 2
        x2, y2 = x1 + box_w, y1 + box_h
        cv2.rectangle(main_view, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = "Cropped (not tracked)"
        cv2.putText(main_view, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        bbox_crop_view = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        crop_title = "BBox Crop (ROI Fallback)"


    if not np.isnan(heading_degrees):
        center_px = (int(results[column_map['bbox_x_norm_ds']] * full_w), int(results[column_map['bbox_y_norm_ds']] * full_h))
        arrow_length = 30
        arrow_end_x = int(center_px[0] + arrow_length * np.cos(np.deg2rad(heading_degrees)))
        arrow_end_y = int(center_px[1] - arrow_length * np.sin(np.deg2rad(heading_degrees)))
        cv2.arrowedLine(main_view, center_px, (arrow_end_x, arrow_end_y), (255, 0, 255), 2, tipLength=0.3)
    
    roi_view = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    roi_sz = roi_image.shape
    colors = {'bladder': (0, 0, 255), 'eye_l': (0, 255, 0), 'eye_r': (255, 100, 0)}
    keypoints = {
        'bladder': (results[column_map['bladder_x_roi_norm']], results[column_map['bladder_y_roi_norm']]),
        'eye_l': (results[column_map['eye_l_x_roi_norm']], results[column_map['eye_l_y_roi_norm']]),
        'eye_r': (results[column_map['eye_r_x_roi_norm']], results[column_map['eye_r_y_roi_norm']])
    }
    for name, (x_norm, y_norm) in keypoints.items():
        if not np.isnan(x_norm):
            x_center, y_center = int(x_norm * roi_sz[1]), int(y_norm * roi_sz[0])
            cv2.circle(roi_view, (x_center, y_center), 4, colors.get(name), -1)
            cv2.circle(roi_view, (x_center, y_center), 5, (0,0,0), 1)

    if not np.isnan(heading_degrees):
        # Normalize heading for display
        heading_display = heading_degrees if heading_degrees >= 0 else heading_degrees + 360
        cv2.putText(roi_view, f"Heading: {heading_display:.1f}°", (10, roi_sz[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    stabilized_roi = rotate_roi_to_heading(roi_image, heading_degrees)
    stabilized_view = cv2.cvtColor(stabilized_roi, cv2.COLOR_GRAY2BGR)
    
    stabilized_view = apply_circular_mask(stabilized_view)
    
    if not np.isnan(heading_degrees):
        roi_center = np.array([roi_sz[1]/2, roi_sz[0]/2])
        rotation_angle_rad = np.deg2rad(-heading_degrees)
        cos_angle, sin_angle = np.cos(rotation_angle_rad), np.sin(rotation_angle_rad)
        
        for name, (x_norm, y_norm) in keypoints.items():
            if not np.isnan(x_norm):
                x_pixel, y_pixel = x_norm * roi_sz[1], y_norm * roi_sz[0]
                rel_x, rel_y = x_pixel - roi_center[0], y_pixel - roi_center[1]
                rotated_x = rel_x * cos_angle + rel_y * sin_angle + roi_center[0]
                rotated_y = -rel_x * sin_angle + rel_y * cos_angle + roi_center[1]
                cv2.circle(stabilized_view, (int(rotated_x), int(rotated_y)), 4, colors.get(name), -1)
                cv2.circle(stabilized_view, (int(rotated_x), int(rotated_y)), 5, (0,0,0), 1)
        
    cv2.putText(stabilized_view, "Stabilized (Facing Right)", (10, roi_sz[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    display_size = (480, 480)
    main_resized = cv2.resize(main_view, display_size, interpolation=cv2.INTER_AREA)
    roi_resized = cv2.resize(roi_view, display_size, interpolation=cv2.INTER_NEAREST)
    stabilized_resized = cv2.resize(stabilized_view, display_size, interpolation=cv2.INTER_NEAREST)
    crop_resized = cv2.resize(bbox_crop_view, display_size, interpolation=cv2.INTER_NEAREST)

    title1 = f"Full View + Bbox (conf: {confidence:.3f})" if not np.isnan(confidence) else "Full View"
    cv2.putText(main_resized, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(roi_resized, "Original ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(stabilized_resized, "Stabilized ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(crop_resized, crop_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    top_row = np.hstack((main_resized, roi_resized))
    bottom_row = np.hstack((stabilized_resized, crop_resized))
    dashboard = np.vstack((top_row, bottom_row))
    
    status_text = f"Frame: {frame_number}"
    if not np.isnan(confidence):
        status_text += f" | Confidence: {confidence:.3f}"
    
    cv2.putText(dashboard, status_text, (10, dashboard.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return dashboard

def main(zarr_path, start_frame):
    global current_frame
    current_frame = start_frame
    
    try:
        zarr_group = zarr.open_group(zarr_path, mode='r')
    except Exception as e:
        print(f"Error opening Zarr store at '{zarr_path}': {e}")
        return

    try:
        latest_tracking_run = zarr_group['tracking_runs'].attrs['latest']
        results_array = zarr_group[f'tracking_runs/{latest_tracking_run}/tracking_results']
        print(f"Using latest tracking run: {latest_tracking_run}")
    except KeyError:
        print("Error: Could not find 'tracking_runs' or the 'latest' attribute in the Zarr file.")
        return

    column_names = results_array.attrs['column_names']
    
    if 'bbox_x_norm_ds' not in column_names:
        print("Error: This visualizer requires the new multi-scale data format from tracker.py.")
        return

    column_map = {name: i for i, name in enumerate(column_names)}
    num_frames = zarr_group['raw_video/images_ds'].shape[0]

    print("Generating summary plots...")
    temp_dir = Path("./temp_plots")
    temp_dir.mkdir(exist_ok=True)
    
    tracking_data_full = results_array[:]
    crop_bbox_array = zarr_group[f"crop_runs/{zarr_group['crop_runs'].attrs['latest']}/bbox_norm_coords"][:]

    crop_success_mask = ~np.isnan(crop_bbox_array[:, 0])
    track_success_mask = ~np.isnan(tracking_data_full[:, column_map['heading_degrees']])
    headings = tracking_data_full[:, column_map['heading_degrees']]
    positions = tracking_data_full[:, [column_map['bbox_x_norm_ds'], column_map['bbox_y_norm_ds']]]
    
    summary_panel = create_summary_panel(
        crop_success_mask, track_success_mask, headings, positions,
        zarr_group['raw_video/images_ds'].shape[1:], temp_dir / "summary_panel.png"
    )
    
    window_name = "Interactive Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Frame", window_name, current_frame, num_frames - 1, update_frame)
    
    print("Starting interactive dashboard...")
    print("Controls: Use slider or arrow keys for navigation. 'q' or Esc to quit.")

    while True:
        dashboard = create_dashboard_view(current_frame, zarr_group, column_map)
        
        display_image = dashboard if dashboard is not None else np.zeros((960, 960, 3), dtype=np.uint8)
        if dashboard is None:
            cv2.putText(display_image, "Frame Not Found", (300, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        dash_h, dash_w = display_image.shape[:2]
        summary_h, summary_w = summary_panel.shape[:2]
        target_summary_w = int(summary_w * (dash_h / summary_h))
        summary_resized = cv2.resize(summary_panel, (target_summary_w, dash_h))

        final_view = np.hstack([display_image, summary_resized])

        cv2.imshow(window_name, final_view)
        cv2.setTrackbarPos("Frame", window_name, current_frame)
        
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == 83: # Right arrow
            current_frame = min(num_frames - 1, current_frame + 1)
        elif key == 81: # Left arrow
            current_frame = max(0, current_frame - 1)

    cv2.destroyAllWindows()
    for f in temp_dir.glob("*.png"): f.unlink()
    temp_dir.rmdir()
    print("Visualizer closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive visualizer for fish tracking results.")
    parser.add_argument("zarr_path", type=str, help="Path to the unified Zarr file (e.g., video.zarr).")
    parser.add_argument("start_frame", type=int, nargs='?', default=1, 
                        help="The frame number to start visualizing from. Defaults to 1.")
    args = parser.parse_args()
    main(args.zarr_path, args.start_frame)