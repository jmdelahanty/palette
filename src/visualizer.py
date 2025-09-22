import cv2
import numpy as np
import os
import argparse
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

current_frame = 1
current_detection = 0  # For multi-fish tracking

def update_frame(val):
    """Callback function to update the global frame index when the slider is moved."""
    global current_frame
    current_frame = val

def update_detection(val):
    """Callback function to update which detection to visualize in multi-fish scenarios."""
    global current_detection
    current_detection = val

def create_summary_panel(n_detections, n_tracked, headings, positions, image_shape, output_path):
    """Creates a single summary panel with detection and tracking statistics."""
    num_frames = len(n_detections)
    
    # Create masks for frames with detections and successful tracks
    detect_success_mask = n_detections > 0
    track_success_mask = n_tracked > 0
    
    # Define the layout
    fig, axes = plt.subplots(5, 1, figsize=(8, 12), dpi=150, 
                             gridspec_kw={'height_ratios': [0.5, 0.5, 0.5, 2, 6]})
    fig.patch.set_facecolor('#2c2c2c')

    # 1. Detection Count Timeline
    ax1 = axes[0]
    detection_timeline = n_detections.reshape(1, -1)
    im1 = ax1.imshow(detection_timeline, cmap='YlOrRd', vmin=0, vmax=n_detections.max(), 
                     aspect='auto', interpolation='none')
    ax1.set_title('Detections per Frame', color='white', fontsize=10)
    ax1.axis('off')
    
    # 2. Detection Success Timeline
    ax2 = axes[1]
    detect_timeline = np.zeros((1, num_frames))
    detect_timeline[0, detect_success_mask] = 1
    ax2.imshow(detect_timeline, cmap='Greens', vmin=0, vmax=1, aspect='auto', interpolation='none')
    ax2.set_title('Detection Success', color='white', fontsize=10)
    ax2.axis('off')

    # 3. Track Success Timeline
    ax3 = axes[2]
    track_timeline_data = np.zeros((1, num_frames))
    track_timeline_data[0, track_success_mask] = 1
    ax3.imshow(track_timeline_data, cmap='Blues', vmin=0, vmax=1, aspect='auto', interpolation='none')
    ax3.set_title('Tracking Success', color='white', fontsize=10)
    ax3.axis('off')
    
    # 4. Heading Plot (if available)
    ax4 = axes[3]
    valid_indices = np.where(~np.isnan(headings))[0]
    if len(valid_indices) > 0:
        valid_headings = headings[valid_indices].copy()
        valid_headings[valid_headings < 0] += 360
        ax4.scatter(valid_indices, valid_headings, c='cyan', s=1, alpha=0.5)
        ax4.set_title('Fish Headings Over Time', color='white', fontsize=10)
        ax4.set_xlim(0, num_frames)
        ax4.set_ylim(-10, 370)
    else:
        ax4.text(0.5, 0.5, 'No tracking data', ha='center', va='center', 
                transform=ax4.transAxes, color='white')
        ax4.set_title('Fish Headings (No Data)', color='white', fontsize=10)
    ax4.tick_params(axis='x', colors='white', labelsize=8)
    ax4.tick_params(axis='y', colors='white', labelsize=8)
    ax4.set_facecolor('#1e1e1e')
    ax4.grid(True, alpha=0.2)

    # 5. Position Heatmap
    ax5 = axes[4]
    valid_positions = positions[~np.isnan(positions).any(axis=1)]
    if len(valid_positions) > 0:
        heatmap, _, _ = np.histogram2d(
            valid_positions[:, 1] * image_shape[0],
            valid_positions[:, 0] * image_shape[1],
            bins=50, range=[[0, image_shape[0]], [0, image_shape[1]]]
        )
        heatmap = np.log(heatmap + 1)
        ax5.imshow(heatmap.T, cmap='jet', aspect='equal', origin='lower', 
                  extent=[0, image_shape[1], 0, image_shape[0]])
    ax5.set_title('Position Heatmap (All Detections)', color='white', fontsize=10)
    ax5.set_xlim(0, image_shape[1])
    ax5.set_ylim(0, image_shape[0])
    ax5.axis('off')

    plt.tight_layout(pad=1.5)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return cv2.imread(str(output_path))

def create_dashboard_view(frame_number, detection_idx, zarr_group, column_map):
    """Creates a dashboard for multi-fish tracking visualization."""
    # Get latest runs
    latest_tracking_run = zarr_group['tracking_runs'].attrs['latest']
    latest_crop_run = zarr_group['crop_runs'].attrs['latest']
    latest_detect_run = zarr_group[f'crop_runs/{latest_crop_run}'].attrs.get('source_detect_run')
    if not latest_detect_run:
        latest_detect_run = zarr_group['detect_runs'].attrs['latest']
    
    # Load arrays
    images_array = zarr_group['raw_video/images_ds']
    n_detections = zarr_group[f'detect_runs/{latest_detect_run}/n_detections']
    
    frame_index = frame_number - 1
    if not (0 <= frame_index < images_array.shape[0]):
        return None
    
    # Get the number of detections in this frame
    n_dets_frame = n_detections[frame_index]
    if n_dets_frame == 0:
        # No detections in this frame
        main_image = images_array[frame_index]
        main_view = cv2.cvtColor(main_image, cv2.COLOR_GRAY2BGR)
        cv2.putText(main_view, "No detections in frame", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Create empty panels
        display_size = (480, 480)
        main_resized = cv2.resize(main_view, display_size)
        empty_panel = np.zeros((display_size[0], display_size[1], 3), dtype=np.uint8)
        
        top_row = np.hstack((main_resized, empty_panel))
        bottom_row = np.hstack((empty_panel, empty_panel))
        dashboard = np.vstack((top_row, bottom_row))
        
        cv2.putText(dashboard, f"Frame: {frame_number} | No Detections", 
                   (10, dashboard.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return dashboard
    
    # Calculate detection index range for this frame
    cumulative_dets = np.cumsum(np.insert(n_detections[:frame_index+1], 0, 0))
    start_idx = cumulative_dets[frame_index]
    end_idx = cumulative_dets[frame_index + 1]
    
    # Ensure detection index is valid for this frame
    det_idx = min(detection_idx, n_dets_frame - 1)
    global_det_idx = start_idx + det_idx
    
    # Load data for this detection
    main_image = images_array[frame_index]
    roi_images_array = zarr_group[f'crop_runs/{latest_crop_run}/roi_images']
    roi_image = roi_images_array[global_det_idx]
    
    tracking_results = zarr_group[f'tracking_runs/{latest_tracking_run}/tracking_results']
    results = tracking_results[global_det_idx]
    
    bbox_coords = zarr_group[f'detect_runs/{latest_detect_run}/bbox_norm_coords'][global_det_idx]
    
    # Create main view with bounding box
    main_view = cv2.cvtColor(main_image, cv2.COLOR_GRAY2BGR)
    full_h, full_w = main_view.shape[:2]
    
    # Draw all detections in this frame (faint)
    for i in range(n_dets_frame):
        other_bbox = zarr_group[f'detect_runs/{latest_detect_run}/bbox_norm_coords'][start_idx + i]
        cx_norm, cy_norm, w_norm, h_norm = other_bbox
        cx, cy = int(cx_norm * full_w), int(cy_norm * full_h)
        w, h = int(w_norm * full_w), int(h_norm * full_h)
        x1, y1 = cx - w//2, cy - h//2
        color = (0, 128, 0) if i == det_idx else (64, 64, 64)
        thickness = 2 if i == det_idx else 1
        cv2.rectangle(main_view, (x1, y1), (x1+w, y1+h), color, thickness)
        cv2.putText(main_view, str(i+1), (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # ROI view with keypoints
    roi_view = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    
    # Check if tracking was successful
    heading_degrees = results[column_map['heading_degrees']]
    confidence = results[column_map['confidence_score']] if 'confidence_score' in column_map else np.nan
    
    if not np.isnan(heading_degrees):
        # Draw keypoints
        roi_h, roi_w = roi_image.shape
        colors = {'bladder': (0, 0, 255), 'eye_l': (0, 255, 0), 'eye_r': (255, 100, 0)}
        
        for keypoint, color in colors.items():
            if f'{keypoint}_x_roi_norm' in column_map:
                x_norm = results[column_map[f'{keypoint}_x_roi_norm']]
                y_norm = results[column_map[f'{keypoint}_y_roi_norm']]
                if not np.isnan(x_norm):
                    x = int(x_norm * roi_w)
                    y = int(y_norm * roi_h)
                    cv2.circle(roi_view, (x, y), 4, color, -1)
                    cv2.circle(roi_view, (x, y), 5, (0, 0, 0), 1)
        
        # Draw heading arrow
        center_px = (roi_w // 2, roi_h // 2)
        arrow_length = 30
        arrow_end_x = int(center_px[0] + arrow_length * np.cos(np.deg2rad(heading_degrees)))
        arrow_end_y = int(center_px[1] - arrow_length * np.sin(np.deg2rad(heading_degrees)))
        cv2.arrowedLine(roi_view, center_px, (arrow_end_x, arrow_end_y), (255, 0, 255), 2, tipLength=0.3)
        
        cv2.putText(roi_view, f"Heading: {heading_degrees:.1f}°", (10, roi_h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(roi_view, "Tracking Failed", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Create summary stats panel
    stats_panel = np.zeros((480, 480, 3), dtype=np.uint8)
    y_offset = 30
    line_height = 25
    
    stats_text = [
        f"Frame: {frame_number}",
        f"Detection: {det_idx + 1} of {n_dets_frame}",
        f"Global Index: {global_det_idx}",
        f"",
        f"BBox Center: ({bbox_coords[0]:.3f}, {bbox_coords[1]:.3f})",
        f"BBox Size: ({bbox_coords[2]:.3f}, {bbox_coords[3]:.3f})",
        f"",
        f"Tracking: {'Success' if not np.isnan(heading_degrees) else 'Failed'}",
        f"Confidence: {confidence:.3f}" if not np.isnan(confidence) else "Confidence: N/A",
        f"Heading: {heading_degrees:.1f}°" if not np.isnan(heading_degrees) else "Heading: N/A"
    ]
    
    for text in stats_text:
        cv2.putText(stats_panel, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height
    
    # Resize and arrange panels
    display_size = (480, 480)
    main_resized = cv2.resize(main_view, display_size)
    roi_resized = cv2.resize(roi_view, display_size)
    
    # Add titles
    cv2.putText(main_resized, f"Full View (Det {det_idx+1}/{n_dets_frame})", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(roi_resized, "ROI View", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(stats_panel, "Statistics", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Create empty panel for future use
    empty_panel = np.zeros(display_size + (3,), dtype=np.uint8)
    cv2.putText(empty_panel, "Reserved", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 64, 64), 1)
    
    top_row = np.hstack((main_resized, roi_resized))
    bottom_row = np.hstack((stats_panel, empty_panel))
    dashboard = np.vstack((top_row, bottom_row))
    
    return dashboard

def main(zarr_path, start_frame):
    global current_frame, current_detection
    current_frame = start_frame
    current_detection = 0
    
    try:
        zarr_group = zarr.open_group(zarr_path, mode='r')
    except Exception as e:
        print(f"Error opening Zarr store at '{zarr_path}': {e}")
        return

    # Check for required data
    try:
        latest_tracking_run = zarr_group['tracking_runs'].attrs['latest']
        latest_crop_run = zarr_group['crop_runs'].attrs['latest']
        latest_detect_run = zarr_group[f'crop_runs/{latest_crop_run}'].attrs.get('source_detect_run')
        if not latest_detect_run:
            latest_detect_run = zarr_group['detect_runs'].attrs['latest']
        
        print(f"Using runs:")
        print(f"  Detection: {latest_detect_run}")
        print(f"  Crop: {latest_crop_run}")
        print(f"  Tracking: {latest_tracking_run}")
    except KeyError as e:
        print(f"Error: Could not find required runs in Zarr file: {e}")
        return

    # Get metadata
    tracking_results = zarr_group[f'tracking_runs/{latest_tracking_run}/tracking_results']
    column_names = tracking_results.attrs['column_names']
    column_map = {name: i for i, name in enumerate(column_names)}
    
    num_frames = zarr_group['raw_video/images_ds'].shape[0]
    n_detections = zarr_group[f'detect_runs/{latest_detect_run}/n_detections'][:]
    max_detections = int(n_detections.max())
    
    # Calculate tracking success
    n_tracked = zarr_group[f'tracking_runs/{latest_tracking_run}/n_detections'][:]
    
    print(f"\nDetection Statistics:")
    print(f"  Total frames: {num_frames}")
    print(f"  Frames with detections: {(n_detections > 0).sum()}")
    print(f"  Total detections: {n_detections.sum()}")
    print(f"  Max detections per frame: {max_detections}")
    print(f"  Frames with successful tracking: {(n_tracked > 0).sum()}")
    print(f"  Total successful tracks: {n_tracked.sum()}")
    
    # Generate summary panel
    print("\nGenerating summary plots...")
    temp_dir = Path("./temp_plots")
    temp_dir.mkdir(exist_ok=True)
    
    # Get all tracking data for summary
    all_headings = tracking_results[:, column_map['heading_degrees']]
    all_positions = tracking_results[:, [column_map['bbox_x_norm_ds'], column_map['bbox_y_norm_ds']]]
    
    summary_panel = create_summary_panel(
        n_detections, n_tracked, all_headings, all_positions,
        zarr_group['raw_video/images_ds'].shape[1:], 
        temp_dir / "summary_panel.png"
    )
    
    # Create window and controls
    window_name = "Multi-Fish Tracking Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Frame", window_name, current_frame, num_frames - 1, update_frame)
    if max_detections > 1:
        cv2.createTrackbar("Detection", window_name, 0, max_detections - 1, update_detection)
    
    print("\nStarting interactive visualizer...")
    print("Controls:")
    print("  - Frame slider: Navigate through frames")
    print("  - Detection slider: Select which detection to view (for multi-fish frames)")
    print("  - Arrow keys: Navigate frames")
    print("  - q or Esc: Quit")
    
    while True:
        dashboard = create_dashboard_view(current_frame, current_detection, zarr_group, column_map)
        
        if dashboard is None:
            dashboard = np.zeros((960, 960, 3), dtype=np.uint8)
            cv2.putText(dashboard, "Error loading frame", (300, 480), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # Combine dashboard with summary panel
        dash_h, dash_w = dashboard.shape[:2]
        summary_h, summary_w = summary_panel.shape[:2]
        target_summary_w = int(summary_w * (dash_h / summary_h))
        summary_resized = cv2.resize(summary_panel, (target_summary_w, dash_h))
        
        final_view = np.hstack([dashboard, summary_resized])
        
        cv2.imshow(window_name, final_view)
        cv2.setTrackbarPos("Frame", window_name, current_frame)
        
        # Update detection trackbar max based on current frame
        n_dets_current = n_detections[current_frame - 1]
        if n_dets_current > 1 and max_detections > 1:
            cv2.setTrackbarMax("Detection", window_name, n_dets_current - 1)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == 83:  # Right arrow
            current_frame = min(num_frames - 1, current_frame + 1)
        elif key == 81:  # Left arrow
            current_frame = max(1, current_frame - 1)
        elif key == 82:  # Up arrow
            if n_detections[current_frame - 1] > 1:
                current_detection = min(current_detection + 1, n_detections[current_frame - 1] - 1)
        elif key == 84:  # Down arrow
            current_detection = max(0, current_detection - 1)
    
    cv2.destroyAllWindows()
    for f in temp_dir.glob("*.png"): 
        f.unlink()
    temp_dir.rmdir()
    print("\nVisualizer closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-fish tracking visualizer")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file")
    parser.add_argument("start_frame", type=int, nargs='?', default=1, 
                        help="Starting frame number (default: 1)")
    args = parser.parse_args()
    main(args.zarr_path, args.start_frame)