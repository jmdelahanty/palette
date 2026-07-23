"""
Utility for inspecting traditional keypoint detection results.

Loads the most recent crop and keypoint runs from a Palette Zarr store,
then overlays detected keypoints (swim bladder + eyes) on the ROI
crop.

Enhanced with slider to scroll through all frames.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import zarr
from matplotlib.widgets import Slider, Button

from ..pose.schema import PoseSchema, schema_from_metadata, schema_from_package
from ..shared.keypoint_coordinate_publication import (
    BoundKeypointCoordinateSurfaces,
    load_persisted_keypoint_coordinate_surfaces,
    require_bound_keypoint_coordinate_surfaces,
)
from ..shared.zarr_io import open_zarr_root


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
    return open_zarr_root(zarr_path, mode="r")


def _text_attr(value: object) -> Optional[str]:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "strict")
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _resolve_canonical_keypoint_selection(
    root: zarr.Group,
    explicit: Optional[str],
) -> str:
    if "keypoints_runs" not in root:
        raise RuntimeError("Canonical keypoint viewer requires 'keypoints_runs'.")
    parent = root["keypoints_runs"]
    requested = _text_attr(explicit)
    if requested is not None and requested.lower() not in {"latest"}:
        if requested.lower() in {
            "latest_traditional",
            "traditional",
            "latest_yolo",
            "yolo",
        }:
            raise RuntimeError(
                "Method-based keypoint shortcuts are legacy inference. Pass one "
                "exact canonical run name or use the active canonical run."
            )
        if requested not in parent:
            raise RuntimeError(
                f"Canonical keypoint run {requested!r} is absent from keypoints_runs."
            )
        return requested

    latest_complete = _text_attr(parent.attrs.get("latest_complete"))
    latest = _text_attr(parent.attrs.get("latest"))
    root_pointer = _text_attr(root.attrs.get("current_keypoint_group_path"))
    if latest_complete is None or latest != latest_complete:
        raise RuntimeError(
            "Canonical keypoint selection requires matching latest and "
            "latest_complete selectors."
        )
    expected_path = f"keypoints_runs/{latest_complete}"
    if root_pointer != expected_path:
        raise RuntimeError(
            "Canonical keypoint selection requires the exact matching root pointer."
        )
    if latest_complete not in parent:
        raise RuntimeError("Canonical keypoint selector names a missing run.")
    return latest_complete


def _load_canonical_keypoint_view(
    root: zarr.Group,
    explicit: Optional[str],
) -> tuple[str, BoundKeypointCoordinateSurfaces]:
    run_name = _resolve_canonical_keypoint_selection(root, explicit)
    surfaces = load_persisted_keypoint_coordinate_surfaces(
        root,
        f"keypoints_runs/{run_name}",
    )
    return run_name, surfaces


def _resolve_latest_by_method(group: zarr.Group, method: str) -> Optional[str]:
    candidates: List[tuple[datetime, str]] = []
    for run_name in group.keys():
        run_group = group[run_name]
        if run_group.attrs.get("method") != method:
            continue
        ts_raw = (
            run_group.attrs.get("keypoints_timestamp_utc")
            or run_group.attrs.get("eye_masks_timestamp_utc")
            or run_group.attrs.get("timestamp_utc")
        )
        try:
            ts = datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else datetime.min
        except Exception:
            ts = datetime.min
        candidates.append((ts, run_name))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def get_latest_run(root: zarr.Group, group_name: str, explicit: Optional[str]) -> str:
    runs_group_name = f"{group_name}_runs"
    if runs_group_name not in root:
        raise RuntimeError(f"No '{runs_group_name}' group found in Zarr store.")
    runs_group = root[runs_group_name]
    if not runs_group:
        raise RuntimeError(f"No runs recorded under '{runs_group_name}'.")

    special_map = {
        "keypoints": {
            "latest_traditional": "traditional_pose",
            "traditional": "traditional_pose",
            "latest_yolo": "yolo_pose",
            "yolo": "yolo_pose",
        },
    }

    if explicit:
        requested = explicit.strip()
        if requested and requested.lower() not in {"latest"}:
            norm = requested.lower()
            method_map = special_map.get(group_name, {})
            method = method_map.get(norm)
            if method:
                resolved = _resolve_latest_by_method(runs_group, method)
                if resolved:
                    return resolved
                raise RuntimeError(
                    f"No run with method '{method}' found under '{runs_group_name}'."
                )
            if requested in runs_group:
                return requested
            raise RuntimeError(
                f"Run '{requested}' not found under '{runs_group_name}'. "
                f"Available: {', '.join(list(runs_group.keys()))}"
            )

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
    *,
    legacy_unverified: bool = False,
) -> KeypointRecord:
    """Read one historical raw record only through explicit diagnostic mode."""
    if legacy_unverified is not True:
        raise RuntimeError(
            "Raw per-frame keypoint access is legacy-unverified. Use sealed canonical "
            "surfaces through the interactive viewer or opt in explicitly for diagnosis."
        )
    
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
    except (KeyError, IndexError):
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


def _preflight_interactive_keypoint_view(
    *,
    keypoint_run: str,
    crop_run: str,
    canonical_surfaces: Optional[BoundKeypointCoordinateSurfaces],
    legacy_unverified: bool,
) -> Optional[BoundKeypointCoordinateSurfaces]:
    if canonical_surfaces is None:
        if legacy_unverified is not True:
            raise RuntimeError(
                "Interactive keypoint viewing requires sealed canonical surfaces. "
                "Raw arrays are available only through explicit legacy-unverified mode."
            )
        return None
    if legacy_unverified is True:
        raise RuntimeError(
            "Canonical keypoint evidence and legacy-unverified mode are mutually exclusive."
        )
    bound = require_bound_keypoint_coordinate_surfaces(canonical_surfaces)
    expected_run_path = f"keypoints_runs/{keypoint_run}"
    expected_crop_path = f"crop_runs/{crop_run}"
    if bound.context.run_path != expected_run_path:
        raise RuntimeError(
            "Canonical keypoint evidence does not match the requested keypoint run."
        )
    if bound.context.source.crop_path != expected_crop_path:
        raise RuntimeError(
            "Canonical keypoint evidence does not match the requested source crop."
        )
    return bound


def plot_record_interactive(
    root: zarr.Group,
    keypoint_run: str,
    crop_run: str,
    labels: Sequence[str],
    pose_schema: Optional[PoseSchema],
    start_frame: int = 0,
    keypoint_method: str = "unknown",
    canonical_surfaces: Optional[BoundKeypointCoordinateSurfaces] = None,
    legacy_unverified: bool = False,
) -> None:
    """Interactive viewer with slider to scroll through frames."""
    canonical_surfaces = _preflight_interactive_keypoint_view(
        keypoint_run=keypoint_run,
        crop_run=crop_run,
        canonical_surfaces=canonical_surfaces,
        legacy_unverified=legacy_unverified,
    )
    
    # Get total number of frames
    n_frames = root["raw_video/images_full"].shape[0]
    
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
            keypoints_roi = (
                canonical_surfaces.keypoints_roi.coordinate_node
                if canonical_surfaces is not None
                else keypoint_group["keypoints_roi"]
            )
            keypoint_data = {
                'keypoints_roi': keypoints_roi,
                'heading': keypoint_group["heading"],
                'confidence': keypoint_group["confidence"],
                'effective_threshold': keypoint_group["effective_threshold"],
                'detection_success': keypoint_group["detection_success"],
            }
            print(f"Loaded keypoint data: {keypoint_run}")
        except (KeyError, IndexError) as e:
            if canonical_surfaces is not None:
                raise RuntimeError(
                    "Canonical keypoint view is missing a required scalar or "
                    "coordinate surface."
                ) from e
            print(f"Warning: Could not load keypoint data: {e}")
            keypoint_data = None
    
    # Pre-load crop data references (but don't load all images)
    roi_images = crop_group["roi_images"]
    
    print(
        f"Ready! {len(frame_to_roi_map)} frames with detections out of {n_frames} total."
    )
    
    cmap_palette = plt.colormaps.get_cmap("tab10").resampled(max(len(labels), 3))
    colors = {label: mcolors.to_hex(cmap_palette(i)) for i, label in enumerate(labels)}
    cmap = "gray"
    
    # Create figure with one subplot (ROI)
    fig, ax_roi = plt.subplots(1, 1, figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)
    
    # Current frame state
    current_frame = [start_frame]  # Use list to allow mutation in closure
    
    def update_display(frame_idx: int):
        """Update both plots for the given frame."""
        frame_idx = int(frame_idx)
        current_frame[0] = frame_idx
        
        # Fast lookup: check if this frame has any ROIs
        roi_indices = frame_to_roi_map.get(frame_idx, [])
        roi_idx = roi_indices[0] if roi_indices else -1
        
        # Clear axes
        ax_roi.clear()
        
        # --- LEFT PANEL: ROI with keypoints ---
        kp_roi = None
        if len(roi_indices) > 0:
            # Load crop data (single zarr accesses)
            roi_img = roi_images[roi_idx]
            
            ax_roi.imshow(roi_img, cmap=cmap)
            
            # Check if we have keypoint data for this ROI
            has_keypoints = False
            if keypoint_data is not None:
                try:
                    success = bool(keypoint_data['detection_success'][roi_idx])
                    if success:
                        # Load keypoint positions
                        kp_roi_arr = keypoint_data['keypoints_roi'][roi_idx]
                        
                        kp_roi = {label: kp_roi_arr[i] for i, label in enumerate(labels)}
                        
                        # Draw keypoints on ROI
                        for label, pt in kp_roi.items():
                            ax_roi.scatter(pt[0], pt[1], s=60, c=colors[label], 
                                         edgecolors="black", linewidths=1.0, label=label)
                        ax_roi.legend(loc="lower right", fontsize=8)
                        
                        # Get metadata
                        heading_raw = float(keypoint_data['heading'][roi_idx])
                        confidence = float(keypoint_data['confidence'][roi_idx])
                        thresh = float(keypoint_data['effective_threshold'][roi_idx])
                        
                        status = "✓ Keypoints detected"
                        title_color = "green"
                        title = (
                            f"ROI {roi_idx} — Frame {frame_idx} [{status}]\n"
                            f"heading={heading_raw:.1f}° | conf={confidence:.2f} | thresh={thresh:.0f}"
                        )
                        has_keypoints = True
                except (KeyError, IndexError):
                    pass
            
            if not has_keypoints:
                # Have crop but no keypoints
                status = "✗ No keypoints"
                title_color = "orange"
                title = f"ROI {roi_idx} — Frame {frame_idx} [{status}]"
        else:
            # No crop available - show black square
            dummy_img = np.zeros((100, 100), dtype=np.uint8)
            ax_roi.imshow(dummy_img, cmap=cmap)
            status = "✗ No detection"
            title_color = "red"
            title = f"Frame {frame_idx} [{status}]"
            roi_img = None
        
        ax_roi.set_title(title, color=title_color, fontweight='bold')
        ax_roi.set_axis_off()

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
    
    if keypoint_run:
        title_suffix = f" ({keypoint_method})" if keypoint_method else ""
        title_text = f"Keypoint Viewer: {keypoint_run}{title_suffix}"
    else:
        title_text = "Keypoint Viewer"
    if legacy_unverified:
        title_text = f"LEGACY UNVERIFIED — {title_text}"
    plt.suptitle(title_text, fontsize=12, fontweight='bold')
    plt.show()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualize traditional keypoint detections with interactive slider."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr directory.")
    parser.add_argument(
        "--keypoint-run",
        help=(
            "Exact canonical keypoint run (default: active canonical run). "
            "Method shortcuts are available only in explicit legacy mode."
        ),
    )
    parser.add_argument(
        "--crop-run",
        help=(
            "Exact crop run; canonical mode requires it to equal the keypoint "
            "run's persisted source crop."
        ),
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame to start viewing from (default: 0).",
    )
    parser.add_argument(
        "--allow-legacy-unverified-coordinate-input",
        action="store_true",
        help=(
            "Allow historical raw/latest keypoint and crop selection for visual "
            "diagnosis only. The viewer is visibly labelled unverified."
        ),
    )

    args = parser.parse_args(argv)

    root = open_zarr(args.zarr_path)
    canonical_surfaces: Optional[BoundKeypointCoordinateSurfaces] = None
    legacy_unverified = bool(args.allow_legacy_unverified_coordinate_input)

    if legacy_unverified:
        print(
            "WARNING: legacy unverified coordinate input enabled; overlays are "
            "diagnostic and are not coordinate authority."
        )
        try:
            keypoint_run = get_latest_run(root, "keypoints", args.keypoint_run)
        except RuntimeError as exc:
            try:
                keypoint_run = get_latest_run(root, "keypoint", args.keypoint_run)
            except RuntimeError:
                print(f"Warning: {exc}")
                print("Will show crops without keypoints.")
                keypoint_run = None

        keypoint_group = None
        if keypoint_run:
            if "keypoints_runs" in root and keypoint_run in root["keypoints_runs"]:
                keypoint_group = root[f"keypoints_runs/{keypoint_run}"]
            elif "keypoint_runs" in root and keypoint_run in root["keypoint_runs"]:
                keypoint_group = root[f"keypoint_runs/{keypoint_run}"]

        crop_run = None
        if args.crop_run:
            crop_run = get_latest_run(root, "crop", args.crop_run)
        elif keypoint_group is not None:
            source_crop_run = keypoint_group.attrs.get("source_crop_run")
            if (
                source_crop_run
                and "crop_runs" in root
                and source_crop_run in root["crop_runs"]
            ):
                crop_run = source_crop_run
        if crop_run is None:
            crop_run = get_latest_run(root, "crop", None)
    else:
        keypoint_run, canonical_surfaces = _load_canonical_keypoint_view(
            root,
            args.keypoint_run,
        )
        keypoint_group = root[f"keypoints_runs/{keypoint_run}"]
        crop_path = canonical_surfaces.context.source.crop_path
        crop_run = crop_path.split("/", 1)[1]
        if args.crop_run is not None and args.crop_run != crop_run:
            raise RuntimeError(
                "Canonical keypoint viewing requires the exact source crop "
                f"{crop_run!r}; received {args.crop_run!r}."
            )
    
    # Get keypoint labels if available
    keypoint_method = "unknown"
    pose_schema = None
    labels = ["swim_bladder", "eye_left", "eye_right"]
    if keypoint_run:
        if keypoint_group is None:
            keypoint_group = root[f"keypoints_runs/{keypoint_run}"]
        if keypoint_group is not None:
            schema_meta = keypoint_group.attrs.get("pose_schema")
            if isinstance(schema_meta, dict):
                try:
                    pose_schema = schema_from_metadata(schema_meta)
                except Exception:
                    schema_name = schema_meta.get("name")
                    if schema_name:
                        try:
                            pose_schema = schema_from_package(schema_name)
                        except FileNotFoundError:
                            pose_schema = None
            default_labels = ["swim_bladder", "eye_left", "eye_right"]
            if pose_schema:
                labels = pose_schema.node_names
            else:
                labels = keypoint_group.attrs.get("keypoint_labels", default_labels)
            keypoint_method = keypoint_group.attrs.get("method", "unknown")

    print("\nKeypoint Visualizer")
    print(f"  Zarr: {args.zarr_path}")
    print(f"  Keypoint run: {keypoint_run or 'None (will show crops only)'}")
    print(f"  Coordinate verification: {'legacy unverified' if legacy_unverified else 'canonical_v2'}")
    if keypoint_run:
        print(f"  Keypoint method: {keypoint_method}")
    print(f"  Crop run: {crop_run}")
    if pose_schema:
        print(f"  Pose schema: {pose_schema.name} ({pose_schema.num_keypoints} keypoints)")
    print("\nControls:")
    print("  - Slider: Navigate to specific frame")
    print("  - Buttons: Prev/Next (±1), Prev 10/Next 10 (±10)")
    print("  - Arrow keys: ← → (±1 frame)")
    print("  - Page Up/Down: (±10 frames)")
    print()
    
    plot_record_interactive(
        root=root,
        keypoint_run=keypoint_run or "",  # Empty string will trigger error handling
        crop_run=crop_run,
        labels=labels,
        pose_schema=pose_schema,
        start_frame=args.start_frame,
        keypoint_method=keypoint_method,
        canonical_surfaces=canonical_surfaces,
        legacy_unverified=legacy_unverified,
    )


if __name__ == "__main__":
    main()
