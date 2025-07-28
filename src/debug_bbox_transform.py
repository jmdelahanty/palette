#!/usr/bin/env python3
"""
Debug script to examine the coordinate transformation issues
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def debug_coordinate_transformation(zarr_path, sample_indices=[0, 1, 2]):
    """Debug the coordinate transformation for specific samples."""
    
    root = zarr.open(zarr_path, mode='r')
    
    # Get data arrays
    tracking_results = root['tracking/tracking_results']
    roi_coordinates = root['crop_data/roi_coordinates']
    roi_images = root['crop_data/roi_images']
    full_images = root['raw_video/images_ds']
    
    print("=== COORDINATE TRANSFORMATION DEBUG ===")
    print()
    
    for sample_idx in sample_indices:
        print(f"--- SAMPLE {sample_idx} ---")
        
        # Get the data
        tracking_data = tracking_results[sample_idx]
        roi_coords = roi_coordinates[sample_idx]
        roi_image = roi_images[sample_idx]
        full_image = full_images[sample_idx]
        
        print(f"Tracking data: {tracking_data}")
        print(f"ROI coordinates (x1, y1): {roi_coords}")
        print()
        
        # Extract keypoint data (in ROI normalized coordinates)
        bladder_x_roi = tracking_data[3]
        bladder_y_roi = tracking_data[4]
        eye_l_x_roi = tracking_data[5]
        eye_l_y_roi = tracking_data[6]
        eye_r_x_roi = tracking_data[7]
        eye_r_y_roi = tracking_data[8]
        
        print("Keypoints in ROI normalized coordinates:")
        print(f"  Bladder: ({bladder_x_roi:.3f}, {bladder_y_roi:.3f})")
        print(f"  Eye L:   ({eye_l_x_roi:.3f}, {eye_l_y_roi:.3f})")
        print(f"  Eye R:   ({eye_r_x_roi:.3f}, {eye_r_y_roi:.3f})")
        print()
        
        # Convert to ROI pixel coordinates
        bladder_x_roi_px = bladder_x_roi * 320
        bladder_y_roi_px = bladder_y_roi * 320
        eye_l_x_roi_px = eye_l_x_roi * 320
        eye_l_y_roi_px = eye_l_y_roi * 320
        eye_r_x_roi_px = eye_r_x_roi * 320
        eye_r_y_roi_px = eye_r_y_roi * 320
        
        print("Keypoints in ROI pixel coordinates:")
        print(f"  Bladder: ({bladder_x_roi_px:.1f}, {bladder_y_roi_px:.1f})")
        print(f"  Eye L:   ({eye_l_x_roi_px:.1f}, {eye_l_y_roi_px:.1f})")
        print(f"  Eye R:   ({eye_r_x_roi_px:.1f}, {eye_r_y_roi_px:.1f})")
        print()
        
        # Calculate fish center and bbox in ROI space
        keypoint_x_coords = [bladder_x_roi, eye_l_x_roi, eye_r_x_roi]
        keypoint_y_coords = [bladder_y_roi, eye_l_y_roi, eye_r_y_roi]
        
        min_x_roi = min(keypoint_x_coords)
        max_x_roi = max(keypoint_x_coords)
        min_y_roi = min(keypoint_y_coords)
        max_y_roi = max(keypoint_y_coords)
        
        center_x_roi = (min_x_roi + max_x_roi) / 2.0
        center_y_roi = (min_y_roi + max_y_roi) / 2.0
        
        print(f"Fish center in ROI normalized: ({center_x_roi:.3f}, {center_y_roi:.3f})")
        print(f"Fish center in ROI pixels: ({center_x_roi * 320:.1f}, {center_y_roi * 320:.1f})")
        print()
        
        # Transform to full image coordinates
        roi_x1, roi_y1 = roi_coords
        center_x_full_px = roi_x1 + center_x_roi * 320
        center_y_full_px = roi_y1 + center_y_roi * 320
        
        print(f"Fish center in full image pixels: ({center_x_full_px:.1f}, {center_y_full_px:.1f})")
        print(f"Fish center in full image normalized: ({center_x_full_px/640:.3f}, {center_y_full_px/640:.3f})")
        print()
        
        # Also check what the original bbox_x_norm, bbox_y_norm values are
        original_bbox_x = tracking_data[1]
        original_bbox_y = tracking_data[2]
        print(f"Original bbox center (from tracking): ({original_bbox_x:.3f}, {original_bbox_y:.3f})")
        print(f"Original bbox center in pixels: ({original_bbox_x * 640:.1f}, {original_bbox_y * 640:.1f})")
        print()
        
        # Calculate bbox dimensions
        bbox_width_roi = (max_x_roi - min_x_roi) * 1.5
        bbox_height_roi = (max_y_roi - min_y_roi) * 1.5
        bbox_width_full = bbox_width_roi * 320 / 640  # Convert to full image normalized
        bbox_height_full = bbox_height_roi * 320 / 640
        
        print(f"Bbox dimensions in ROI normalized: ({bbox_width_roi:.3f}, {bbox_height_roi:.3f})")
        print(f"Bbox dimensions in full image normalized: ({bbox_width_full:.3f}, {bbox_height_full:.3f})")
        print(f"Bbox dimensions in full image pixels: ({bbox_width_full * 640:.1f}, {bbox_height_full * 640:.1f})")
        print()
        print("="*50)
        print()


def visualize_debug_sample(zarr_path, sample_idx=0):
    """Create a detailed debug visualization for one sample."""
    
    root = zarr.open(zarr_path, mode='r')
    
    # Get the data
    tracking_data = root['tracking/tracking_results'][sample_idx]
    roi_coords = root['crop_data/roi_coordinates'][sample_idx]
    roi_image = root['crop_data/roi_images'][sample_idx]
    full_image = root['raw_video/images_ds'][sample_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: ROI with keypoints
    ax1 = axes[0]
    ax1.imshow(roi_image, cmap='gray')
    ax1.set_title(f'ROI Image (320×320)\nExtracted from: {roi_coords}')
    
    # Draw keypoints
    bladder_x = tracking_data[3] * 320
    bladder_y = tracking_data[4] * 320
    eye_l_x = tracking_data[5] * 320
    eye_l_y = tracking_data[6] * 320
    eye_r_x = tracking_data[7] * 320
    eye_r_y = tracking_data[8] * 320
    
    ax1.plot(bladder_x, bladder_y, 'yo', markersize=10, label='Swim Bladder')
    ax1.plot(eye_l_x, eye_l_y, 'bo', markersize=8, label='Left Eye')
    ax1.plot(eye_r_x, eye_r_y, 'ro', markersize=8, label='Right Eye')
    
    # Calculate and draw fish center and bbox in ROI
    keypoint_x_coords = [tracking_data[3], tracking_data[5], tracking_data[7]]
    keypoint_y_coords = [tracking_data[4], tracking_data[6], tracking_data[8]]
    
    min_x_roi = min(keypoint_x_coords)
    max_x_roi = max(keypoint_x_coords)
    min_y_roi = min(keypoint_y_coords)
    max_y_roi = max(keypoint_y_coords)
    
    center_x_roi = (min_x_roi + max_x_roi) / 2.0
    center_y_roi = (min_y_roi + max_y_roi) / 2.0
    bbox_width_roi = (max_x_roi - min_x_roi) * 1.5
    bbox_height_roi = (max_y_roi - min_y_roi) * 1.5
    
    # Draw calculated center
    ax1.plot(center_x_roi * 320, center_y_roi * 320, 'g+', markersize=15, markeredgewidth=3, label='Calculated Center')
    
    # Draw calculated bbox
    bbox_x1 = (center_x_roi - bbox_width_roi/2) * 320
    bbox_y1 = (center_y_roi - bbox_height_roi/2) * 320
    bbox_w = bbox_width_roi * 320
    bbox_h = bbox_height_roi * 320
    
    rect = patches.Rectangle((bbox_x1, bbox_y1), bbox_w, bbox_h, 
                           linewidth=2, edgecolor='lime', facecolor='none', label='Calculated Bbox')
    ax1.add_patch(rect)
    
    ax1.legend()
    ax1.set_xlim(0, 320)
    ax1.set_ylim(320, 0)  # Flip y-axis for image coordinates
    
    # Panel 2: Full image with ROI outline
    ax2 = axes[1]
    ax2.imshow(full_image, cmap='gray')
    ax2.set_title(f'Full Image (640×640)\nROI position shown')
    
    # Draw ROI outline
    roi_x1, roi_y1 = roi_coords
    roi_rect = patches.Rectangle((roi_x1, roi_y1), 320, 320, 
                               linewidth=2, edgecolor='yellow', facecolor='none', label='ROI Region')
    ax2.add_patch(roi_rect)
    
    # Draw where we think the fish center should be
    center_x_full_px = roi_x1 + center_x_roi * 320
    center_y_full_px = roi_y1 + center_y_roi * 320
    ax2.plot(center_x_full_px, center_y_full_px, 'g+', markersize=15, markeredgewidth=3, label='Transformed Center')
    
    # Draw original tracking center for comparison
    original_center_x = tracking_data[1] * 640
    original_center_y = tracking_data[2] * 640
    ax2.plot(original_center_x, original_center_y, 'r+', markersize=15, markeredgewidth=3, label='Original bbox_x/y_norm')
    
    ax2.legend()
    ax2.set_xlim(0, 640)
    ax2.set_ylim(640, 0)
    
    # Panel 3: Close-up of the region where fish should be
    ax3 = axes[2]
    
    # Crop around the ROI region
    crop_margin = 50
    crop_x1 = max(0, roi_x1 - crop_margin)
    crop_x2 = min(640, roi_x1 + 320 + crop_margin)
    crop_y1 = max(0, roi_y1 - crop_margin)
    crop_y2 = min(640, roi_y1 + 320 + crop_margin)
    
    cropped_image = full_image[crop_y1:crop_y2, crop_x1:crop_x2]
    ax3.imshow(cropped_image, cmap='gray', extent=[crop_x1, crop_x2, crop_y2, crop_y1])
    ax3.set_title('Close-up of ROI Region')
    
    # Draw ROI outline in cropped view
    roi_rect_crop = patches.Rectangle((roi_x1, roi_y1), 320, 320, 
                                    linewidth=2, edgecolor='yellow', facecolor='none')
    ax3.add_patch(roi_rect_crop)
    
    # Draw centers
    ax3.plot(center_x_full_px, center_y_full_px, 'g+', markersize=15, markeredgewidth=3)
    ax3.plot(original_center_x, original_center_y, 'r+', markersize=15, markeredgewidth=3)
    
    ax3.set_xlim(crop_x1, crop_x2)
    ax3.set_ylim(crop_y2, crop_y1)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    zarr_path = '/home/delahantyj@hhmi.org/Desktop/concentricOMR3/video.zarr'
    
    print("Running coordinate transformation debug...")
    debug_coordinate_transformation(zarr_path, [0, 1, 2])
    
    print("Creating debug visualization...")
    visualize_debug_sample(zarr_path, 0)