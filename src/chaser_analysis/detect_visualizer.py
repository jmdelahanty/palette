# src/detection_visualizer.py

import zarr
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import os

# Global variables
fig, ax = plt.subplots(figsize=(10, 10))
zarr_root = None
images_ds = None
n_detections = None
bbox_coords = None
detection_ids = None # NEW: Global variable for IDs
cumulative_detections = None
output_dir = None

def update_frame(frame_idx):
    """
    This function is called whenever the slider is moved.
    It clears the current axes and draws the new frame, detections, and IDs.
    """
    frame_idx = int(frame_idx)
    ax.clear()

    image = images_ds[frame_idx]
    ax.imshow(image, cmap='gray')

    num_dets_in_frame = n_detections[frame_idx]
    ax.set_title(f"Frame: {frame_idx} | Detections: {num_dets_in_frame}", fontsize=12)

    if num_dets_in_frame > 0:
        start_idx = cumulative_detections[frame_idx]
        end_idx = cumulative_detections[frame_idx + 1]
        
        frame_bboxes = bbox_coords[start_idx:end_idx]
        # NEW: Get the corresponding IDs for this frame's detections
        frame_ids = detection_ids[start_idx:end_idx] if detection_ids is not None else [-1] * num_dets_in_frame

        # MODIFIED: Loop through detections with their IDs
        for i, bbox in enumerate(frame_bboxes):
            assigned_id = frame_ids[i]
            
            # Set color based on whether the fish was assigned an ID
            box_color = 'lime' if assigned_id != -1 else 'red'
            
            center_x_norm, center_y_norm, width_norm, height_norm = bbox
            img_height, img_width = image.shape
            center_x = center_x_norm * img_width
            center_y = center_y_norm * img_height
            box_w = width_norm * img_width
            box_h = height_norm * img_height
            
            x1 = center_x - (box_w / 2)
            y1 = center_y - (box_h / 2)
            
            rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)
            
            # NEW: Add text for the ID
            id_text = f"ID: {assigned_id}" if assigned_id != -1 else "Unassigned"
            ax.text(x1, y1 - 5, id_text, color=box_color, fontsize=10, fontweight='bold')

    ax.axis('off')
    plt.draw()

def on_key_press(event):
    # ... (this function remains the same) ...
    global frame_slider
    if event.key == 's':
        save_current_frame()
    elif event.key == 'right':
        new_val = min(frame_slider.val + 1, frame_slider.valmax)
        frame_slider.set_val(new_val)
    elif event.key == 'left':
        new_val = max(frame_slider.val - 1, frame_slider.valmin)
        frame_slider.set_val(new_val)


def save_current_frame():
    # ... (this function remains the same) ...
    global output_dir
    if output_dir is None:
        print("⚠️  Cannot save: Please specify an output directory using --output-dir.")
        return
        
    current_frame_idx = int(frame_slider.val)
    save_path = Path(output_dir) / f"detection_frame_{current_frame_idx:06d}.png"
    
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    print(f"✅ Frame {current_frame_idx} saved to: {save_path}")


def main(args):
    global zarr_root, images_ds, n_detections, bbox_coords, cumulative_detections, output_dir, frame_slider, detection_ids

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"🖼️  Saved frames will be stored in: {output_dir}")

    try:
        zarr_root = zarr.open(args.zarr_path, mode='r')
        latest_detect_run = zarr_root['detect_runs'].attrs['latest']
        detect_group = zarr_root[f'detect_runs/{latest_detect_run}']
        
        images_ds = zarr_root['raw_video/images_ds']
        n_detections = detect_group['n_detections'][:]
        bbox_coords = detect_group['bbox_norm_coords'][:]
        cumulative_detections = np.cumsum(np.insert(n_detections, 0, 0))

        # NEW: Load the detection IDs if they exist
        if 'id_assignments_runs' in zarr_root and 'latest' in zarr_root['id_assignments_runs'].attrs:
            latest_id_run = zarr_root['id_assignments_runs'].attrs['latest']
            id_group = zarr_root[f'id_assignments_runs/{latest_id_run}']
            detection_ids = id_group['detection_ids'][:]
            print("✅ Loaded detection IDs.")
        else:
            print("⚠️  No ID assignment data found. Will only display bounding boxes.")

    except Exception as e:
        print(f"❌ Error opening Zarr file or finding data: {e}")
        return

    num_frames = images_ds.shape[0]
    plt.subplots_adjust(bottom=0.2)
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=num_frames - 1,
        valinit=args.start_frame,
        valstep=1
    )

    frame_slider.on_changed(update_frame)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    update_frame(args.start_frame)
    
    print("\n🚀 Starting Detection Visualizer...")
    print("Controls:")
    print("  - Use the slider or LEFT/RIGHT arrow keys to navigate.")
    print("  - Press 's' to save the current view as a PNG.")
    print("  - Close the window to quit.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize fish detections and assigned IDs.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame to start on.")
    parser.add_argument("--output-dir", type=str, default="detection_snapshots", help="Directory to save snapshot images.")
    args = parser.parse_args()
    
    main(args)