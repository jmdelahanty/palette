"""
Utility for inspecting traditional keypoint detection results.

Loads the most recent crop and keypoint runs from a Palette Zarr store,
then overlays detected keypoints (swim bladder + eyes) on both the ROI
crop and the original frame.

Enhanced with slider to scroll through all frames.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.widgets import Slider, Button


@dataclass
class KeypointRecord:
    """Container for a single ROI's keypoint results and context."""

    roi_index: int
    frame_index: int
    roi_image: Optional[np.ndarray]
    full_image: np.ndarray
    roi_origin_xy: Optional[np.ndarray]  # top-left corner in full image (x, y)
    keypoints_px_roi: Optional[dict]  # bladder/eye_left/eye_right -> (x, y)
    keypoints_px_full: Optional[dict]
    heading_deg: Optional[float]
    confidence: Optional[float]
    effective_thresh: Optional[float]
    success: bool


def open_zarr(zarr_path: Path) -> zarr.Group:
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")
    return zarr.open_group(str(zarr_path), mode="r")


def get_latest_run(root: zarr.Group, group_name: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    runs_group_name = f"{group_name}_runs"
    if runs_group_name not in root:
        raise RuntimeError(f"No '{runs_group_name}' group found in Zarr store.")
    runs_group = root[runs_group_name]
    latest = runs_group.attrs.get("latest")
    if not latest:
        raise RuntimeError(f"No runs recorded under '{runs_group_name}'.")
    return latest


def get_record_for_frame(
    root: zarr.Group,
    frame_idx: int,
    keypoint_run: str,
    crop_run: str,
    labels: Sequence[str],
) -> KeypointRecord:
    """Get keypoint record for a specific frame, handling missing data gracefully."""
    
    crop_group = root[f"crop_runs/{crop_run}"]
    frame_indices = crop_group["frame_indices"][:]
    
    # Find ROIs for this frame
    roi_indices = np.where(frame_indices == frame_idx)[0]
    
    # Get full frame
    full_frames = root["raw_video/images_full"]
    full_img = full_frames[frame_idx]
    full_h, full_w = full_frames.shape[1], full_frames.shape[2]
    
    # If no crops for this frame, return minimal record
    if len(roi_indices) == 0:
        return KeypointRecord(
            roi_index=-1,
            frame_index=frame_idx,
            roi_image=None,
            full_image=full_img,
            roi_origin_xy=None,
            keypoints_px_roi=None,
            keypoints_px_full=None,
            heading_deg=None,
            confidence=None,
            effective_thresh=None,
            success=False,
        )
    
    # Use first ROI for this frame
    roi_idx = roi_indices[0]
    
    # Load crop data
    roi_images = crop_group["roi_images"]
    roi_coords_full = crop_group["roi_coordinates_full"]
    roi_origin = roi_coords_full[roi_idx]
    roi_img = roi_images[roi_idx]
    
    # Try to load keypoint data
    try:
        keypoint_group = root[f"keypoints_runs/{keypoint_run}"]
        keypoints_roi = keypoint_group["keypoints_roi"]
        keypoints_img = keypoint_group["keypoints_img"]
        heading_arr = keypoint_group["heading"]
        confidence_arr = keypoint_group["confidence"]
        effective_thresh_arr = keypoint_group["effective_threshold"]
        success_arr = keypoint_group["detection_success"]
        
        success = bool(success_arr[roi_idx])
        
        if success:
            kp_roi = {label: keypoints_roi[roi_idx][i] for i, label in enumerate(labels)}
            kp_full = {label: keypoints_img[roi_idx][i] for i, label in enumerate(labels)}
            
            # Clamp keypoints to frame bounds
            for kpt in kp_full.values():
                kpt[0] = np.clip(kpt[0], 0, full_w - 1)
                kpt[1] = np.clip(kpt[1], 0, full_h - 1)
            
            return KeypointRecord(
                roi_index=int(roi_idx),
                frame_index=frame_idx,
                roi_image=roi_img,
                full_image=full_img,
                roi_origin_xy=roi_origin,
                keypoints_px_roi={k: np.array(v) for k, v in kp_roi.items()},
                keypoints_px_full={k: np.array(v) for k, v in kp_full.items()},
                heading_deg=float(heading_arr[roi_idx]),
                confidence=float(confidence_arr[roi_idx]),
                effective_thresh=float(effective_thresh_arr[roi_idx]),
                success=True,
            )
        else:
            # Keypoint detection failed, but we have the crop
            return KeypointRecord(
                roi_index=int(roi_idx),
                frame_index=frame_idx,
                roi_image=roi_img,
                full_image=full_img,
                roi_origin_xy=roi_origin,
                keypoints_px_roi=None,
                keypoints_px_full=None,
                heading_deg=None,
                confidence=None,
                effective_thresh=None,
                success=False,
            )
    except (KeyError, IndexError) as e:
        # Keypoint run doesn't exist or is incomplete, just show the crop
        return KeypointRecord(
            roi_index=int(roi_idx),
            frame_index=frame_idx,
            roi_image=roi_img,
            full_image=full_img,
            roi_origin_xy=roi_origin,
            keypoints_px_roi=None,
            keypoints_px_full=None,
            heading_deg=None,
            confidence=None,
            effective_thresh=None,
            success=False,
        )


def plot_record_interactive(
    root: zarr.Group,
    keypoint_run: str,
    crop_run: str,
    labels: Sequence[str],
    start_frame: int = 0,
) -> None:
    """Interactive viewer with slider to scroll through frames."""
    
    # Get total number of frames
    full_frames = root["raw_video/images_full"]
    n_frames = full_frames.shape[0]
    
    # Pre-build frame-to-ROI mapping for fast lookups
    print("Building frame-to-ROI index...")
    crop_group = root[f"crop_runs/{crop_run}"]
    frame_indices = crop_group["frame_indices"][:]
    
    frame_to_roi_map = {}
    for roi_idx, frame_idx in enumerate(frame_indices):
        if frame_idx not in frame_to_roi_map:
            frame_to_roi_map[frame_idx] = []
        frame_to_roi_map[frame_idx].append(roi_idx)
    
    # Pre-load keypoint data if available
    keypoint_data = None
    if keypoint_run:
        try:
            keypoint_group = root[f"keypoints_runs/{keypoint_run}"]
            keypoint_data = {
                'keypoints_roi': keypoint_group["keypoints_roi"],
                'keypoints_img': keypoint_group["keypoints_img"],
                'heading': keypoint_group["heading"],
                'confidence': keypoint_group["confidence"],
                'effective_threshold': keypoint_group["effective_threshold"],
                'detection_success': keypoint_group["detection_success"],
            }
            print(f"Loaded keypoint data: {keypoint_run}")
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not load keypoint data: {e}")
            keypoint_data = None
    
    # Pre-load crop data references (but don't load all images)
    roi_images = crop_group["roi_images"]
    roi_coords_full = crop_group["roi_coordinates_full"]
    full_h, full_w = full_frames.shape[1], full_frames.shape[2]
    
    print(f"Ready! {len(frame_to_roi_map)} frames with detections out of {n_frames} total.")
    
    colors = {
        "bladder": "#ffcc00",
        "eye_left": "#00ff88",
        "eye_right": "#ff557f",
    }
    cmap = "gray"
    
    # Create figure with two subplots
    fig, (ax_roi, ax_full) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.2)
    
    # Current frame state
    current_frame = [start_frame]  # Use list to allow mutation in closure
    
    def update_display(frame_idx: int):
        """Update both plots for the given frame."""
        frame_idx = int(frame_idx)
        current_frame[0] = frame_idx
        
        # Fast lookup: check if this frame has any ROIs
        roi_indices = frame_to_roi_map.get(frame_idx, [])
        
        # Load full frame (single zarr access)
        full_img = full_frames[frame_idx]
        
        # Clear axes
        ax_roi.clear()
        ax_full.clear()
        
        # --- LEFT PANEL: ROI with keypoints ---
        if len(roi_indices) > 0:
            # Use first ROI for this frame
            roi_idx = roi_indices[0]
            
            # Load crop data (single zarr accesses)
            roi_img = roi_images[roi_idx]
            roi_origin = roi_coords_full[roi_idx]
            
            ax_roi.imshow(roi_img, cmap=cmap)
            
            # Check if we have keypoint data for this ROI
            has_keypoints = False
            if keypoint_data is not None:
                try:
                    success = bool(keypoint_data['detection_success'][roi_idx])
                    if success:
                        # Load keypoint positions
                        kp_roi_arr = keypoint_data['keypoints_roi'][roi_idx]
                        kp_full_arr = keypoint_data['keypoints_img'][roi_idx]
                        
                        kp_roi = {label: kp_roi_arr[i] for i, label in enumerate(labels)}
                        kp_full = {label: kp_full_arr[i] for i, label in enumerate(labels)}
                        
                        # Draw keypoints on ROI
                        for label, pt in kp_roi.items():
                            ax_roi.scatter(pt[0], pt[1], s=60, c=colors[label], 
                                         edgecolors="black", linewidths=1.0, label=label)
                        ax_roi.legend(loc="lower right", fontsize=8)
                        
                        # Get metadata
                        heading = float(keypoint_data['heading'][roi_idx])
                        confidence = float(keypoint_data['confidence'][roi_idx])
                        thresh = float(keypoint_data['effective_threshold'][roi_idx])
                        
                        status = "✓ Keypoints detected"
                        title_color = "green"
                        title = (
                            f"ROI {roi_idx} — Frame {frame_idx} [{status}]\n"
                            f"heading={heading:.1f}° | conf={confidence:.2f} | thresh={thresh:.0f}"
                        )
                        has_keypoints = True
                        
                        # Store for full frame overlay
                        kp_full_clamped = {}
                        for label, pt in kp_full.items():
                            kp_full_clamped[label] = np.array([
                                np.clip(pt[0], 0, full_w - 1),
                                np.clip(pt[1], 0, full_h - 1)
                            ])
                except (KeyError, IndexError):
                    pass
            
            if not has_keypoints:
                # Have crop but no keypoints
                status = "✗ No keypoints"
                title_color = "orange"
                title = f"ROI {roi_idx} — Frame {frame_idx} [{status}]"
                kp_full_clamped = None
        else:
            # No crop available - show black square
            dummy_img = np.zeros((100, 100), dtype=np.uint8)
            ax_roi.imshow(dummy_img, cmap=cmap)
            status = "✗ No detection"
            title_color = "red"
            title = f"Frame {frame_idx} [{status}]"
            roi_img = None
            roi_origin = None
            kp_full_clamped = None
        
        ax_roi.set_title(title, color=title_color, fontweight='bold')
        ax_roi.set_axis_off()
        
        # --- RIGHT PANEL: Full frame with overlay ---
        ax_full.imshow(full_img, cmap=cmap)
        ax_full.set_title(f"Full Frame {frame_idx}")
        ax_full.set_axis_off()
        
        # Draw ROI rectangle if available
        if len(roi_indices) > 0 and roi_origin is not None and roi_img is not None:
            origin_x, origin_y = roi_origin
            roi_h, roi_w = roi_img.shape
            rect_x = [origin_x, origin_x + roi_w, origin_x + roi_w, origin_x, origin_x]
            rect_y = [origin_y, origin_y, origin_y + roi_h, origin_y + roi_h, origin_y]
            ax_full.plot(rect_x, rect_y, color="cyan", linewidth=1.5, linestyle="--")
        
        # Draw keypoints on full frame if available
        if has_keypoints and kp_full_clamped is not None:
            for label, pt in kp_full_clamped.items():
                ax_full.scatter(pt[0], pt[1], s=60, c=colors[label], 
                              edgecolors="black", linewidths=1.0)
                ax_full.text(pt[0] + 3, pt[1] - 3, label, color=colors[label], 
                           fontsize=8, weight="bold",
                           bbox=dict(facecolor='black', alpha=0.5, pad=2))
        
        fig.canvas.draw_idle()
    
    # Initial display
    update_display(start_frame)
    
    # Add slider
    ax_slider = plt.axes([0.15, 0.08, 0.65, 0.03])
    slider = Slider(
        ax_slider,
        'Frame',
        0,
        n_frames - 1,
        valinit=start_frame,
        valstep=1,
        color='steelblue'
    )
    slider.on_changed(update_display)
    
    # Add navigation buttons
    ax_prev = plt.axes([0.15, 0.02, 0.1, 0.04])
    ax_next = plt.axes([0.26, 0.02, 0.1, 0.04])
    ax_prev10 = plt.axes([0.37, 0.02, 0.1, 0.04])
    ax_next10 = plt.axes([0.48, 0.02, 0.1, 0.04])
    
    btn_prev = Button(ax_prev, 'Prev')
    btn_next = Button(ax_next, 'Next')
    btn_prev10 = Button(ax_prev10, 'Prev 10')
    btn_next10 = Button(ax_next10, 'Next 10')
    
    def prev_frame(event):
        new_frame = max(0, current_frame[0] - 1)
        slider.set_val(new_frame)
    
    def next_frame(event):
        new_frame = min(n_frames - 1, current_frame[0] + 1)
        slider.set_val(new_frame)
    
    def prev_10_frames(event):
        new_frame = max(0, current_frame[0] - 10)
        slider.set_val(new_frame)
    
    def next_10_frames(event):
        new_frame = min(n_frames - 1, current_frame[0] + 10)
        slider.set_val(new_frame)
    
    btn_prev.on_clicked(prev_frame)
    btn_next.on_clicked(next_frame)
    btn_prev10.on_clicked(prev_10_frames)
    btn_next10.on_clicked(next_10_frames)
    
    # Keyboard shortcuts
    def on_key(event):
        if event.key == 'left':
            prev_frame(None)
        elif event.key == 'right':
            next_frame(None)
        elif event.key == 'pagedown':
            prev_10_frames(None)
        elif event.key == 'pageup':
            next_10_frames(None)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.suptitle(f'Keypoint Viewer: {keypoint_run}', fontsize=12, fontweight='bold')
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize traditional keypoint detections with interactive slider."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr directory.")
    parser.add_argument("--keypoint-run", help="Specific keypoint run to inspect (defaults to latest).")
    parser.add_argument("--crop-run", help="Specific crop run to use alongside keypoints.")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame to start viewing from (default: 0).",
    )

    args = parser.parse_args()

    root = open_zarr(args.zarr_path)
    
    # Try to get keypoint run
    try:
        keypoint_run = get_latest_run(root, "keypoints", args.keypoint_run)
    except RuntimeError as exc:
        # Fall back to legacy naming without trailing 's'
        try:
            keypoint_run = get_latest_run(root, "keypoint", args.keypoint_run)
        except RuntimeError:
            print(f"Warning: {exc}")
            print("Will show crops without keypoints.")
            keypoint_run = None
    
    crop_run = get_latest_run(root, "crop", args.crop_run)
    
    # Get keypoint labels if available
    if keypoint_run:
        keypoint_group = root[f"keypoints_runs/{keypoint_run}"]
        labels = keypoint_group.attrs.get("keypoint_labels", ["bladder", "eye_left", "eye_right"])
    else:
        labels = ["bladder", "eye_left", "eye_right"]
    
    print(f"\nKeypoint Visualizer")
    print(f"  Zarr: {args.zarr_path}")
    print(f"  Keypoint run: {keypoint_run or 'None (will show crops only)'}")
    print(f"  Crop run: {crop_run}")
    print(f"\nControls:")
    print(f"  - Slider: Navigate to specific frame")
    print(f"  - Buttons: Prev/Next (±1), Prev 10/Next 10 (±10)")
    print(f"  - Arrow keys: ← → (±1 frame)")
    print(f"  - Page Up/Down: (±10 frames)")
    print()
    
    plot_record_interactive(
        root=root,
        keypoint_run=keypoint_run or "",  # Empty string will trigger error handling
        crop_run=crop_run,
        labels=labels,
        start_frame=args.start_frame,
    )


if __name__ == "__main__":
    main()