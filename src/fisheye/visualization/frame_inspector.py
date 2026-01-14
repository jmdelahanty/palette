#!/usr/bin/env python3
"""
Frame Inspector - View raw video frames from zarr at any resolution.

Helps debug issues by letting you view:
- Full resolution frames (images_full)
- Downsampled frames (images_ds)
- Compare specific frames between resolutions

Usage:
    python -m fisheye.visualization.frame_inspector path/to/data.zarr
    python -m fisheye.visualization.frame_inspector path/to/data.zarr --frame 352
    python -m fisheye.visualization.frame_inspector path/to/data.zarr --resolution full
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser(
        description="Inspect raw video frames from zarr at any resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("zarr_path", help="Path to zarr file")
    parser.add_argument("--frame", "-f", type=int, default=0, help="Starting frame index")
    parser.add_argument(
        "--resolution", "-r",
        choices=["full", "ds", "both"],
        default="both",
        help="Which resolution to display (default: both)"
    )

    args = parser.parse_args()

    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"Error: {zarr_path} does not exist")
        sys.exit(1)

    # Open zarr
    root = zarr.open(zarr_path, mode='r')

    if 'raw_video' not in root:
        print("Error: No raw_video group found in zarr")
        sys.exit(1)

    raw = root['raw_video']

    # Check available arrays
    has_full = 'images_full' in raw
    has_ds = 'images_ds' in raw

    print(f"\nZarr: {zarr_path.name}")
    print(f"Available arrays:")

    images_full = None
    images_ds = None

    if has_full:
        images_full = raw['images_full']
        print(f"  • images_full: {images_full.shape} ({images_full.dtype})")
    else:
        print(f"  • images_full: NOT FOUND")

    if has_ds:
        images_ds = raw['images_ds']
        print(f"  • images_ds: {images_ds.shape} ({images_ds.dtype})")
    else:
        print(f"  • images_ds: NOT FOUND")

    # Get metadata
    attrs = dict(raw.attrs)
    orig_res = attrs.get('original_resolution', [0, 0])
    ds_res = attrs.get('downsampled_resolution', [0, 0])
    total_frames = attrs.get('total_frames', 0)
    fps = attrs.get('fps', 30)

    print(f"\nMetadata:")
    print(f"  • Original resolution: {orig_res[1]}x{orig_res[0]}")
    print(f"  • Downsampled resolution: {ds_res[1]}x{ds_res[0]}")
    print(f"  • Total frames: {total_frames}")
    print(f"  • FPS: {fps}")

    # Determine what to show
    if args.resolution == "full" and not has_full:
        print("\nError: Full resolution not available, switching to ds")
        args.resolution = "ds"
    if args.resolution == "ds" and not has_ds:
        print("\nError: Downsampled not available, switching to full")
        args.resolution = "full"
    if args.resolution == "both" and not (has_full and has_ds):
        if has_full:
            args.resolution = "full"
        elif has_ds:
            args.resolution = "ds"
        else:
            print("\nError: No image arrays found!")
            sys.exit(1)

    # Determine number of frames
    if images_full is not None:
        n_frames = images_full.shape[0]
    elif images_ds is not None:
        n_frames = images_ds.shape[0]
    else:
        print("Error: No frames available")
        sys.exit(1)

    # Validate starting frame
    start_frame = min(args.frame, n_frames - 1)
    start_frame = max(0, start_frame)

    # Create figure
    if args.resolution == "both":
        fig, (ax_full, ax_ds) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Frame Inspector: {zarr_path.name}", fontsize=12)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle(f"Frame Inspector: {zarr_path.name}", fontsize=12)

    plt.subplots_adjust(bottom=0.2)

    # Create slider
    ax_slider = plt.axes([0.15, 0.08, 0.7, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=start_frame, valstep=1)

    # State
    current_frame = [start_frame]

    def update_display(frame_idx):
        frame_idx = int(frame_idx)
        current_frame[0] = frame_idx

        if args.resolution == "both":
            # Full resolution
            ax_full.clear()
            if images_full is not None:
                img_full = images_full[frame_idx]
                ax_full.imshow(img_full, cmap='gray')
                h, w = img_full.shape[:2]
                # Check if frame is all black
                is_black = np.max(img_full) == 0
                status = "[BLACK]" if is_black else f"[max={np.max(img_full)}]"
                ax_full.set_title(f"Full Resolution ({w}x{h}) {status}")
            else:
                ax_full.set_title("Full Resolution: N/A")
            ax_full.axis('off')

            # Downsampled
            ax_ds.clear()
            if images_ds is not None:
                img_ds = images_ds[frame_idx]
                ax_ds.imshow(img_ds, cmap='gray')
                h, w = img_ds.shape[:2]
                is_black = np.max(img_ds) == 0
                status = "[BLACK]" if is_black else f"[max={np.max(img_ds)}]"
                ax_ds.set_title(f"Downsampled ({w}x{h}) {status}")
            else:
                ax_ds.set_title("Downsampled: N/A")
            ax_ds.axis('off')

            fig.suptitle(f"Frame {frame_idx} / {n_frames - 1}", fontsize=12)
        else:
            ax.clear()
            if args.resolution == "full":
                img = images_full[frame_idx]
                res_label = "Full"
            else:
                img = images_ds[frame_idx]
                res_label = "Downsampled"

            ax.imshow(img, cmap='gray')
            h, w = img.shape[:2]
            is_black = np.max(img) == 0
            status = "[BLACK]" if is_black else f"[max={np.max(img)}]"
            ax.set_title(f"Frame {frame_idx} - {res_label} ({w}x{h}) {status}")
            ax.axis('off')

        fig.canvas.draw_idle()

    def on_slider_change(val):
        update_display(val)

    slider.on_changed(on_slider_change)

    # Add keyboard navigation
    def on_key(event):
        if event.key == 'right':
            new_val = min(current_frame[0] + 1, n_frames - 1)
            slider.set_val(new_val)
        elif event.key == 'left':
            new_val = max(current_frame[0] - 1, 0)
            slider.set_val(new_val)
        elif event.key == 'up':
            new_val = min(current_frame[0] + 10, n_frames - 1)
            slider.set_val(new_val)
        elif event.key == 'down':
            new_val = max(current_frame[0] - 10, 0)
            slider.set_val(new_val)
        elif event.key == 'pageup':
            new_val = min(current_frame[0] + 100, n_frames - 1)
            slider.set_val(new_val)
        elif event.key == 'pagedown':
            new_val = max(current_frame[0] - 100, 0)
            slider.set_val(new_val)
        elif event.key == 'home':
            slider.set_val(0)
        elif event.key == 'end':
            slider.set_val(n_frames - 1)
        elif event.key == 'p':
            # Print frame stats
            frame_idx = current_frame[0]
            print(f"\nFrame {frame_idx} stats:")
            if images_full is not None:
                img = images_full[frame_idx]
                print(f"  Full: shape={img.shape}, min={np.min(img)}, max={np.max(img)}, mean={np.mean(img):.1f}")
            if images_ds is not None:
                img = images_ds[frame_idx]
                print(f"  DS: shape={img.shape}, min={np.min(img)}, max={np.max(img)}, mean={np.mean(img):.1f}")

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial display
    update_display(start_frame)

    print(f"\nControls:")
    print(f"  Left/Right arrows: Previous/next frame")
    print(f"  Up/Down arrows: Skip 10 frames")
    print(f"  PageUp/PageDown: Skip 100 frames")
    print(f"  Home/End: First/last frame")
    print(f"  P: Print frame statistics")

    plt.show()


if __name__ == "__main__":
    main()
