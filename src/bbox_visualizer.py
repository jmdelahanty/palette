#!/usr/bin/env python3
"""
YOLO Bounding Box Visualizer
Visualizes the transformed bounding boxes on full 640x640 images to verify correctness.
"""

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

# Import our dataset
from zarr_yolo_dataset_bbox import ZarrYOLODataset


def draw_yolo_bbox(image, bbox, class_id=0, confidence=1.0, color=(0, 255, 0), thickness=2):
    """
    Draw YOLO format bounding box on image.
    
    Args:
        image: OpenCV image (H, W, C)
        bbox: [center_x, center_y, width, height] in normalized coordinates
        class_id: Class ID for labeling
        confidence: Confidence score for labeling
        color: BGR color tuple
        thickness: Line thickness
    
    Returns:
        Modified image with bounding box drawn
    """
    h, w = image.shape[:2]
    
    # Convert normalized YOLO coordinates to pixel coordinates
    center_x_norm, center_y_norm, width_norm, height_norm = bbox
    
    center_x = int(center_x_norm * w)
    center_y = int(center_y_norm * h)
    box_width = int(width_norm * w)
    box_height = int(height_norm * h)
    
    # Calculate top-left corner
    x1 = int(center_x - box_width // 2)
    y1 = int(center_y - box_height // 2)
    x2 = x1 + box_width
    y2 = y1 + box_height
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw center point
    cv2.circle(image, (center_x, center_y), 3, color, -1)
    
    # Add label
    label = f"Fish {confidence:.2f}"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def visualize_dataset_samples(dataset, num_samples=9, save_path=None):
    """
    Create a grid visualization of dataset samples with bounding boxes.
    
    Args:
        dataset: ZarrYOLODataset instance
        num_samples: Number of samples to visualize
        save_path: Optional path to save the visualization
    """
    # Calculate grid dimensions
    grid_cols = int(np.ceil(np.sqrt(num_samples)))
    grid_rows = int(np.ceil(num_samples / grid_cols))
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 15))
    if grid_rows == 1 and grid_cols == 1:
        axes = [axes]
    elif grid_rows == 1 or grid_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(num_samples):
        if i >= len(dataset):
            break
            
        # Get sample from dataset
        sample = dataset[i]
        image = sample['img']  # Shape: (3, 640, 640)
        bbox = sample['bboxes'][0]  # [center_x, center_y, width, height]
        cls = sample['cls'][0]
        
        # Convert from CHW to HWC and to uint8
        image_np = (image.transpose(1, 2, 0)).astype(np.uint8)
        
        # Create colored version for visualization
        # Check if already 3-channel or needs conversion
        if image_np.shape[-1] == 3:
            # Already 3-channel, assume it's grayscale repeated 3 times
            image_bgr = image_np.copy()
        else:
            # Single channel, convert to BGR
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding box
        image_with_bbox = draw_yolo_bbox(image_bgr.copy(), bbox, int(cls))
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB)
        
        # Plot
        ax = axes[i] if num_samples > 1 else axes[0]
        ax.imshow(image_rgb)
        ax.set_title(f'Sample {i}\nBbox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]', 
                    fontsize=10)
        ax.axis('off')
        
        # Add pixel coordinate info
        h, w = image_np.shape[:2]
        center_x_px = int(bbox[0] * w)
        center_y_px = int(bbox[1] * h)
        width_px = int(bbox[2] * w)
        height_px = int(bbox[3] * h)
        
        ax.text(0.02, 0.98, f'Center: ({center_x_px}, {center_y_px})\nSize: {width_px}×{height_px}px', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('YOLO Dataset Samples with Bounding Boxes\n(Full 640×640 Images)', 
                 fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    
    plt.show()


def create_side_by_side_comparison(dataset, sample_idx=0, save_path=None):
    """
    Create a side-by-side comparison showing the ROI crop vs full image with bbox.
    """
    import zarr
    
    # Get the sample
    sample = dataset[sample_idx]
    zarr_index = dataset.indices[sample_idx]
    
    # Get full image and bbox
    full_image = sample['img'].transpose(1, 2, 0).astype(np.uint8)  # (640, 640, 3)
    bbox = sample['bboxes'][0]
    
    # Get ROI image for comparison
    roi_image = dataset.root['crop_data/roi_images'][zarr_index]  # (320, 320)
    roi_coords = dataset.root['crop_data/roi_coordinates'][zarr_index]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Left: ROI crop
    ax1.imshow(roi_image, cmap='gray')
    ax1.set_title(f'ROI Crop (320×320)\nExtracted from position: {roi_coords}', fontsize=12)
    ax1.axis('off')
    
    # Add ROI keypoints if available
    tracking_data = dataset.root['tracking/tracking_results'][zarr_index]
    if not np.isnan(tracking_data[3]):  # Check if keypoints are valid
        # Draw keypoints on ROI
        bladder_x = int(tracking_data[3] * 320)
        bladder_y = int(tracking_data[4] * 320)
        eye_l_x = int(tracking_data[5] * 320)
        eye_l_y = int(tracking_data[6] * 320)
        eye_r_x = int(tracking_data[7] * 320)
        eye_r_y = int(tracking_data[8] * 320)
        
        ax1.plot(bladder_x, bladder_y, 'yo', markersize=8, label='Swim Bladder')
        ax1.plot(eye_l_x, eye_l_y, 'bo', markersize=6, label='Left Eye')
        ax1.plot(eye_r_x, eye_r_y, 'ro', markersize=6, label='Right Eye')
        ax1.legend(loc='upper right')
    
    # Right: Full image with transformed bbox
    # Check if image is already 3-channel or needs conversion
    if full_image.shape[-1] == 3:
        # Already 3-channel, check if it's RGB or needs BGR conversion
        full_image_bgr = full_image.copy()
    else:
        # Single channel, convert to BGR
        full_image_bgr = cv2.cvtColor(full_image, cv2.COLOR_GRAY2BGR)
    
    full_image_with_bbox = draw_yolo_bbox(full_image_bgr.copy(), bbox)
    full_image_rgb = cv2.cvtColor(full_image_with_bbox, cv2.COLOR_BGR2RGB)
    
    ax2.imshow(full_image_rgb)
    ax2.set_title(f'Full Image (640×640) with Transformed Bbox\nBbox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]', fontsize=12)
    ax2.axis('off')
    
    # Draw ROI outline on full image
    roi_x1, roi_y1 = roi_coords
    roi_rect = patches.Rectangle((roi_x1, roi_y1), 320, 320, 
                                linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
    ax2.add_patch(roi_rect)
    ax2.text(roi_x1, roi_y1-10, 'ROI Region', color='yellow', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Comparison saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO bounding boxes on full images")
    parser.add_argument("zarr_path", type=str, help="Path to the video.zarr file")
    parser.add_argument("--mode", choices=['train', 'val'], default='train', 
                       help="Dataset mode to visualize")
    parser.add_argument("--num-samples", type=int, default=9, 
                       help="Number of samples to visualize in grid")
    parser.add_argument("--sample-idx", type=int, default=0,
                       help="Specific sample index for side-by-side comparison")
    parser.add_argument("--save-dir", type=str, help="Directory to save visualizations")
    parser.add_argument("--comparison", action='store_true',
                       help="Show side-by-side ROI vs full image comparison")
    
    args = parser.parse_args()
    
    # Validate paths
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"❌ Error: Zarr file not found: {zarr_path}")
        return
    
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    print(f"🔍 Loading dataset from: {zarr_path}")
    print(f"📊 Mode: {args.mode}")
    
    try:
        # Load dataset
        dataset = ZarrYOLODataset(zarr_path=str(zarr_path), mode=args.mode)
        print(f"✅ Loaded dataset with {len(dataset)} samples")
        
        if args.comparison:
            # Create side-by-side comparison
            print(f"🔄 Creating ROI vs Full Image comparison for sample {args.sample_idx}")
            save_path = save_dir / f"roi_vs_full_comparison_{args.mode}_sample{args.sample_idx}.png" if save_dir else None
            create_side_by_side_comparison(dataset, args.sample_idx, save_path)
        else:
            # Create grid visualization
            print(f"🎨 Creating grid visualization with {args.num_samples} samples")
            save_path = save_dir / f"yolo_bbox_grid_{args.mode}_{args.num_samples}samples.png" if save_dir else None
            visualize_dataset_samples(dataset, args.num_samples, save_path)
        
        print("✅ Visualization complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()