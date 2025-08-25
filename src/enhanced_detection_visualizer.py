#!/usr/bin/env python3
"""
Enhanced Detection Visualizer with Interpolation Support

Visualizes original detections, assigned IDs, and interpolated detections
on the raw video frames. Shows which detections are real vs interpolated.
"""

import argparse
import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from pathlib import Path
from rich.console import Console

console = Console()

# Global variables
fig, ax = None, None
zarr_root = None
images_ds = None
n_detections = None
bbox_coords = None
detection_ids = None
n_detections_per_roi = None
interpolation_masks = None
interpolated_detections = {}  # Frame -> ROI -> bbox mapping
cumulative_detections = None
output_dir = None
frame_slider = None
current_roi_focus = None  # If set, only show this ROI


def load_interpolation_data(zarr_root):
    """Load interpolation data if available."""
    global interpolation_masks, interpolated_detections
    
    interpolation_masks = None
    interpolated_detections = {}
    
    # Load interpolation masks
    if 'interpolation_runs' in zarr_root:
        interp_group = zarr_root['interpolation_runs']
        if 'latest' in interp_group.attrs:
            latest_run = interp_group.attrs['latest']
            run_group = interp_group[latest_run]
            if 'interpolation_masks' in run_group:
                interpolation_masks = run_group['interpolation_masks'][:]
                console.print(f"[green]✓ Loaded interpolation masks from:[/green] {latest_run}")
    
    # Load actual interpolated bounding boxes
    if 'interpolated_detections' in zarr_root:
        interp_det_group = zarr_root['interpolated_detections']
        if 'latest' in interp_det_group.attrs:
            latest_det = interp_det_group.attrs['latest']
            det_group = interp_det_group[latest_det]
            
            # Load the sparse representation
            frame_indices = det_group['frame_indices'][:]
            roi_ids = det_group['roi_ids'][:]
            bboxes = det_group['bboxes'][:]
            
            # Build frame -> roi -> bbox mapping
            for i in range(len(frame_indices)):
                frame_idx = int(frame_indices[i])
                roi_id = int(roi_ids[i])
                bbox = bboxes[i]
                
                if frame_idx not in interpolated_detections:
                    interpolated_detections[frame_idx] = {}
                interpolated_detections[frame_idx][roi_id] = bbox
            
            console.print(f"[green]✓ Loaded interpolated bboxes from:[/green] {latest_det}")
            console.print(f"  Total interpolated detections: {len(frame_indices)}")
            return True
    
    if interpolation_masks is None and not interpolated_detections:
        console.print("[yellow]No interpolation data found[/yellow]")
    
    return False


def get_roi_detection_in_frame(roi_id, frame_idx, n_detections_per_roi, detection_ids, bbox_coords, cumulative_detections):
    """Get the detection for a specific ROI in a specific frame."""
    if n_detections_per_roi[frame_idx, roi_id] == 0:
        return None, None
    
    # Find the detection with this ROI ID in this frame
    start_idx = int(cumulative_detections[frame_idx])
    end_idx = int(cumulative_detections[frame_idx + 1])
    
    frame_ids = detection_ids[start_idx:end_idx]
    frame_bboxes = bbox_coords[start_idx:end_idx]
    
    # Find which detection belongs to this ROI
    roi_mask = frame_ids == roi_id
    if np.any(roi_mask):
        roi_idx = np.where(roi_mask)[0][0]
        return frame_bboxes[roi_idx], frame_ids[roi_idx]
    
    return None, None


def update_frame(frame_idx):
    """Update the displayed frame with detections and interpolation status."""
    global ax, images_ds, n_detections, bbox_coords, detection_ids, n_detections_per_roi
    global interpolation_masks, interpolated_detections, cumulative_detections, current_roi_focus
    
    frame_idx = int(frame_idx)
    ax.clear()
    
    # Display the image
    image = images_ds[frame_idx]
    ax.imshow(image, cmap='gray')
    
    num_dets_in_frame = int(n_detections[frame_idx])
    img_height, img_width = image.shape[:2]
    
    # Build title
    title_parts = [f"Frame: {frame_idx}"]
    if current_roi_focus is not None:
        title_parts.append(f"ROI: {current_roi_focus}")
    title_parts.append(f"Detections: {num_dets_in_frame}")
    
    ax.set_title(" | ".join(title_parts), fontsize=12)
    
    # Track which ROIs have been drawn
    drawn_rois = set()
    
    # First, draw real detections
    if num_dets_in_frame > 0 and detection_ids is not None and n_detections_per_roi is not None:
        # Get detections for this frame
        start_idx = int(cumulative_detections[frame_idx])
        end_idx = int(cumulative_detections[frame_idx + 1])
        
        frame_bboxes = bbox_coords[start_idx:end_idx]
        frame_ids = detection_ids[start_idx:end_idx]
        
        # Draw each real detection
        for i, (bbox, det_id) in enumerate(zip(frame_bboxes, frame_ids)):
            det_id = int(det_id)
            drawn_rois.add(det_id)
            
            # Skip if we're focusing on a specific ROI and this isn't it
            if current_roi_focus is not None and det_id != current_roi_focus:
                continue
            
            # Real detections are always green (or red if unassigned)
            box_color = 'lime' if det_id != -1 else 'red'
            label = f"ID: {det_id}" if det_id != -1 else "Unassigned"
            
            # Draw bounding box
            center_x_norm, center_y_norm, width_norm, height_norm = bbox
            center_x = float(center_x_norm) * img_width
            center_y = float(center_y_norm) * img_height
            box_w = float(width_norm) * img_width
            box_h = float(height_norm) * img_height
            
            x1 = center_x - (box_w / 2)
            y1 = center_y - (box_h / 2)
            
            rect = patches.Rectangle((x1, y1), box_w, box_h, 
                                    linewidth=2, 
                                    edgecolor=box_color, 
                                    facecolor='none')
            ax.add_patch(rect)
            
            # Draw center point
            ax.plot(center_x, center_y, 'o', color=box_color, markersize=4)
            
            # Add label
            ax.text(x1, y1 - 5, label, color=box_color, 
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='black', alpha=0.7))
    
    # Now draw interpolated detections (if they don't already have real detections)
    interpolated_rois = []
    if frame_idx in interpolated_detections:
        for roi_id, bbox in interpolated_detections[frame_idx].items():
            # Skip if we already drew a real detection for this ROI
            if roi_id in drawn_rois:
                continue
                
            # Skip if we're focusing on a different ROI
            if current_roi_focus is not None and roi_id != current_roi_focus:
                continue
            
            interpolated_rois.append(roi_id)
            
            # Draw interpolated detection in yellow
            center_x_norm, center_y_norm, width_norm, height_norm = bbox
            center_x = float(center_x_norm) * img_width
            center_y = float(center_y_norm) * img_height
            box_w = float(width_norm) * img_width
            box_h = float(height_norm) * img_height
            
            x1 = center_x - (box_w / 2)
            y1 = center_y - (box_h / 2)
            
            # Draw with thicker yellow line
            rect = patches.Rectangle((x1, y1), box_w, box_h, 
                                    linewidth=3, 
                                    edgecolor='yellow', 
                                    facecolor='none',
                                    linestyle='--')  # Dashed for interpolated
            ax.add_patch(rect)
            
            # Draw center point
            ax.plot(center_x, center_y, 'o', color='yellow', markersize=6)
            
            # Add label
            ax.text(x1, y1 - 5, f"ID: {roi_id} (interp)", 
                   color='yellow', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='black', alpha=0.7))
    
    # Add legend
    legend_elements = [
        patches.Patch(color='lime', label='Real Detection'),
        patches.Patch(color='yellow', label='Interpolated'),
        patches.Patch(color='red', label='Unassigned')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Add interpolation indicator
    if interpolated_rois:
        info_text = f"Interpolated ROIs: {', '.join(map(str, interpolated_rois))}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, color='yellow', fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='black', alpha=0.7))
    
    ax.axis('off')
    fig.canvas.draw_idle()


def on_key_press(event):
    """Handle keyboard shortcuts."""
    global frame_slider, current_roi_focus
    
    if event.key == 's':
        save_current_frame()
    elif event.key == 'right' and frame_slider is not None:
        new_val = min(frame_slider.val + 1, frame_slider.valmax)
        frame_slider.set_val(new_val)
    elif event.key == 'left' and frame_slider is not None:
        new_val = max(frame_slider.val - 1, frame_slider.valmin)
        frame_slider.set_val(new_val)
    elif event.key == 'r':
        # Reset ROI focus
        current_roi_focus = None
        update_frame(frame_slider.val)
        print("Reset: Showing all ROIs")
    elif event.key.isdigit():
        # Focus on specific ROI
        roi_id = int(event.key)
        if n_detections_per_roi is not None and roi_id < n_detections_per_roi.shape[1]:
            current_roi_focus = roi_id
            update_frame(frame_slider.val)
            print(f"Focusing on ROI {roi_id}")
    elif event.key in ('q', 'escape'):
        print("Closing figure...")
        plt.close(fig)


def save_current_frame():
    """Save the current frame visualization."""
    global output_dir, frame_slider
    
    if output_dir is None:
        print("Cannot save: Please specify an output directory using --output-dir.")
        return
    
    current_frame_idx = int(frame_slider.val) if frame_slider is not None else 0
    suffix = f"_roi{current_roi_focus}" if current_roi_focus is not None else "_all"
    save_path = Path(output_dir) / f"detection_frame_{current_frame_idx:06d}{suffix}.png"
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    print(f"Frame {current_frame_idx} saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize detections with interpolation status',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  • Left/Right arrows: Navigate frames
  • 0-9: Focus on specific ROI
  • R: Reset to show all ROIs
  • S: Save current frame
  • Q/Esc: Quit

Color Coding:
  • Green boxes: Real detections
  • Yellow boxes: Interpolated detections
  • Red boxes: Unassigned detections
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame (default: 0)')
    parser.add_argument('--roi', type=int,
                       help='Focus on specific ROI')
    parser.add_argument('--output-dir', type=str, default='detection_snapshots',
                       help='Directory for saved frames')
    
    args = parser.parse_args()
    
    global fig, ax, zarr_root, images_ds, n_detections, bbox_coords, detection_ids
    global n_detections_per_roi, interpolation_masks, cumulative_detections
    global output_dir, frame_slider, current_roi_focus
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    current_roi_focus = args.roi
    
    console.print(f"\n[bold cyan]Enhanced Detection Visualizer[/bold cyan]")
    console.print(f"Loading: {args.zarr_path}")
    
    try:
        # Load zarr data
        zarr_root = zarr.open(args.zarr_path, mode='r')
        
        # Load detection data
        latest_detect_run = zarr_root['detect_runs'].attrs['latest']
        detect_group = zarr_root[f'detect_runs/{latest_detect_run}']
        
        images_ds = zarr_root['raw_video/images_ds']
        n_detections = detect_group['n_detections'][:]
        bbox_coords = detect_group['bbox_norm_coords'][:]
        cumulative_detections = np.cumsum(np.insert(n_detections, 0, 0))
        
        # Load ID assignments
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in zarr_root else 'id_assignments'
        if id_key in zarr_root and 'latest' in zarr_root[id_key].attrs:
            latest_id_run = zarr_root[id_key].attrs['latest']
            id_group = zarr_root[f'{id_key}/{latest_id_run}']
            detection_ids = id_group['detection_ids'][:]
            n_detections_per_roi = id_group['n_detections_per_roi'][:]
            console.print(f"[green]✓ Loaded detection IDs[/green] ({n_detections_per_roi.shape[1]} ROIs)")
        else:
            console.print("[yellow]No ID assignments found[/yellow]")
            detection_ids = None
            n_detections_per_roi = None
        
        # Load interpolation data
        load_interpolation_data(zarr_root)
        
    except Exception as e:
        console.print(f"[red]Error loading data:[/red] {e}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.subplots_adjust(bottom=0.15)
    
    # Create slider
    num_frames = images_ds.shape[0]
    ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
    frame_slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=num_frames - 1,
        valinit=max(0, min(args.start_frame, num_frames - 1)),
        valstep=1
    )
    
    frame_slider.on_changed(update_frame)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Initial display
    update_frame(args.start_frame)
    
    console.print("\n[bold]Controls:[/bold]")
    console.print("  • [cyan]←/→[/cyan]: Navigate frames")
    console.print("  • [cyan]0-9[/cyan]: Focus on specific ROI")
    console.print("  • [cyan]R[/cyan]: Reset to show all ROIs")
    console.print("  • [cyan]S[/cyan]: Save current frame")
    console.print("  • [cyan]Q/Esc[/cyan]: Quit")
    console.print("\n[bold]Color Legend:[/bold]")
    console.print("  • [green]Green[/green]: Real detections")
    console.print("  • [yellow]Yellow[/yellow]: Interpolated detections")
    console.print("  • [red]Red[/red]: Unassigned detections")
    
    plt.show()


if __name__ == "__main__":
    main()