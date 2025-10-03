# src/fisheye/tune/subdish_mask_tuner.py
"""
Interactive tuner for defining multiple sub-dish masks for spatial ID assignment.

Allows users to draw rectangular regions where individual fish are expected,
which are then used by the assign_ids stage to assign consistent IDs.
"""

import cv2
import numpy as np
import zarr
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone

# Global variables for interactive drawing
drawing = False
start_point = (-1, -1)
current_point = (-1, -1)
rois = []
current_roi_id = 0
background_image = None
display_image = None
window_name = "Sub-Dish Mask Tuner"


def draw_callback(event, x, y, flags, param):
    """Mouse callback for drawing rectangles."""
    global drawing, start_point, current_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_point = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_point = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_point = (x, y)
        
        # Add completed ROI
        x1, y1 = start_point
        x2, y2 = current_point
        
        # Ensure x1 < x2 and y1 < y2
        roi_x = min(x1, x2)
        roi_y = min(y1, y2)
        roi_w = abs(x1 - x2)
        roi_h = abs(y1 - y2)
        
        # Only add if rectangle has area
        if roi_w > 5 and roi_h > 5:
            global current_roi_id, rois
            rois.append({
                'id': current_roi_id,
                'roi_pixels': [roi_x, roi_y, roi_w, roi_h]
            })
            print(f"  Added ROI {current_roi_id}: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
            current_roi_id += 1
        
        # Reset for next rectangle
        start_point = (-1, -1)
        current_point = (-1, -1)


def load_existing_subdish_masks(zarr_path: str) -> Optional[List[Dict]]:
    """Load existing sub-dish mask definitions from zarr if they exist."""
    try:
        root = zarr.open(zarr_path, mode='r')
        
        # Check analysis_metadata for existing sub-dish mask tuning
        if 'analysis_metadata' in root:
            analysis_meta = root['analysis_metadata']
            
            if 'subdish_mask_tuning' in analysis_meta.attrs:
                subdish_data = analysis_meta.attrs['subdish_mask_tuning']
                print(f"  Found {len(subdish_data['masks'])} existing sub-dish masks in analysis_metadata")
                return subdish_data['masks']
        
        return None
    except Exception as e:
        print(f"Could not load existing sub-dish mask data: {e}")
        return None


def save_subdish_masks_to_zarr(zarr_path: str, masks: List[Dict], array_name: str) -> bool:
    """Save sub-dish mask definitions to zarr analysis_metadata."""
    if not masks:
        print("  No sub-dish masks to save")
        return False
    
    try:
        root = zarr.open(zarr_path, mode='a')
        
        # Create analysis_metadata group if it doesn't exist
        if 'analysis_metadata' not in root:
            root.create_group('analysis_metadata')
        
        analysis_meta = root['analysis_metadata']
        
        # Save sub-dish mask data
        subdish_data = {
            'masks': masks,
            'tuned_timestamp': datetime.now(timezone.utc).isoformat(),
            'tuned_on_array': array_name,
            'num_masks': len(masks)
        }
        
        analysis_meta.attrs['subdish_mask_tuning'] = subdish_data
        
        print(f"\n  Saved {len(masks)} sub-dish masks to zarr analysis_metadata")
        print(f"  Timestamp: {subdish_data['tuned_timestamp']}")
        print(f"  Array: {array_name}")
        
        return True
        
    except Exception as e:
        print(f"  Error saving sub-dish masks to zarr: {e}")
        return False


def update_display():
    """Update the display with current sub-dish masks and drawing state."""
    global display_image, background_image, rois, drawing, start_point, current_point
    
    # Reset display to background
    display_image = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR).copy()
    
    # Draw all completed sub-dish masks
    for mask in rois:
        mask_id = mask['id']
        x, y, w, h = mask['roi_pixels']
        
        # Different colors for different masks
        color = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ][mask_id % 6]
        
        cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)
        
        # Label with ID
        label = f"ID {mask_id}"
        cv2.putText(display_image, label, (x + 5, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw current rectangle being drawn
    if drawing and start_point != (-1, -1):
        x1, y1 = start_point
        x2, y2 = current_point
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Add instructions overlay
    instructions = [
        f"Masks defined: {len(rois)}",
        "Draw: Click & drag",
        "Undo: 'u' | Clear: 'c'",
        "Save: 's' | Quit: 'q'"
    ]
    
    y_offset = 30
    for text in instructions:
        cv2.putText(display_image, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 25
    
    cv2.imshow(window_name, display_image)


def main(zarr_path: str, use_full_res: bool = False, frame_idx: Optional[int] = None):
    """Main function for sub-dish mask tuner."""
    global background_image, display_image, rois, current_roi_id
    
    print("\n" + "="*60)
    print("  Sub-Dish Mask Tuner for Spatial ID Assignment")
    print("="*60)
    
    # Try to load existing sub-dish masks
    existing_masks = load_existing_subdish_masks(zarr_path)
    if existing_masks:
        print(f"\n  Loaded {len(existing_masks)} existing sub-dish masks")
        print("  You can edit them or start fresh ('c' to clear all)")
        rois = existing_masks
        current_roi_id = max(mask['id'] for mask in rois) + 1 if rois else 0
    
    # Load background image
    try:
        root = zarr.open(zarr_path, mode='r')
        
        # Get background image
        if 'background_runs' not in root:
            print("  Error: No background computed. Run background stage first.")
            return 1
        
        latest_bg_run = root['background_runs'].attrs['latest']
        
        # Choose resolution
        if use_full_res:
            background_image = root[f'background_runs/{latest_bg_run}/background_full'][:]
            array_name = "background_full"
        else:
            background_image = root[f'background_runs/{latest_bg_run}/background_ds'][:]
            array_name = "background_ds"
        
        print(f"\n  Using: {array_name}")
        print(f"  Shape: {background_image.shape}")
        
    except Exception as e:
        print(f"  Error loading background: {e}")
        return 1
    
    # Setup window
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_callback)
    
    print("\n" + "-"*60)
    print("  Instructions:")
    print("-"*60)
    print("  1. Click and drag to draw rectangular ROIs")
    print("  2. Each ROI will be assigned a unique ID (color-coded)")
    print("  3. Press 'u' to undo last ROI")
    print("  4. Press 'c' to clear all ROIs and start over")
    print("  5. Press 's' to SAVE ROIs to zarr")
    print("  6. Press 'q' or Esc to quit")
    print()
    print("  Tip: Draw one ROI per fish/region you want to track")
    print("-"*60 + "\n")
    
    # Main loop
    while True:
        update_display()
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:  # Quit
            print("\n  Quitting without saving...")
            break
        
        elif key == ord('u'):  # Undo last mask
            if rois:
                removed = rois.pop()
                print(f"  Removed sub-dish mask {removed['id']}")
                current_roi_id -= 1
            else:
                print("  No sub-dish masks to remove")
        
        elif key == ord('c'):  # Clear all
            if rois:
                rois = []
                current_roi_id = 0
                print("  Cleared all sub-dish masks")
            else:
                print("  No sub-dish masks to clear")
        
        elif key == ord('s'):  # Save
            if not rois:
                print("  No sub-dish masks to save. Draw at least one mask first.")
                continue
            
            print(f"\n  Saving {len(rois)} sub-dish masks...")
            success = save_subdish_masks_to_zarr(zarr_path, rois, array_name)
            
            if success:
                print("\n  " + "="*56)
                print("    Sub-dish mask definitions saved successfully!")
                print("  " + "="*56)
                print("\n  Next steps:")
                print("    1. Run: python -m fisheye data.zarr --stages assign_ids")
                print("    2. Or run full pipeline including assign_ids")
                print()
                break
            else:
                print("  Failed to save sub-dish masks")
    
    cv2.destroyAllWindows()
    
    # Print summary
    if rois:
        print("\n" + "="*60)
        print("  Final Sub-Dish Mask Summary")
        print("="*60)
        for mask in rois:
            x, y, w, h = mask['roi_pixels']
            print(f"  Mask ID {mask['id']}: x={x}, y={y}, w={w}, h={h}")
        print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive tuner for defining sub-dish masks for spatial ID assignment"
    )
    parser.add_argument("zarr_path", type=str, 
                       help="Path to zarr archive")
    parser.add_argument("--full", action="store_true",
                       help="Use full resolution background instead of downsampled")
    
    args = parser.parse_args()
    
    exit_code = main(args.zarr_path, use_full_res=args.full)
    exit(exit_code)