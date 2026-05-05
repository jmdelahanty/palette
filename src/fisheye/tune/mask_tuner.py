import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Global variables to store trackbar values
hough_param1 = 50
hough_param2 = 30
radius_adjustment = 0
frame_index = 0
max_frames = 0

# Global variables to store detected circle or rectangle
detected_circle = None
rectangle_roi = None
mask_mode = "circle"

from fisheye.utils.zarr_io import open_zarr_root


def _as_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolution_from_attr(value: Any) -> Optional[tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    height = _as_positive_int(value[0])
    width = _as_positive_int(value[1])
    if height is None or width is None:
        return None
    return height, width


def _resolve_source_video_path(zarr_root) -> Optional[Path]:
    """Return the source video path recorded on a metadata-only production Zarr."""

    candidates = [
        zarr_root.attrs.get("source_video_path"),
        zarr_root.attrs.get("source_path"),
    ]
    raw_video = zarr_root.get("raw_video") if hasattr(zarr_root, "get") else None
    if raw_video is not None:
        candidates.extend(
            [
                raw_video.attrs.get("source_path"),
                raw_video.attrs.get("source_video_path"),
            ]
        )

    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(str(candidate)).expanduser()
        if path.exists():
            return path
    return None


def _resolve_source_video_shape(zarr_root) -> Optional[tuple[int, int]]:
    """Return the native source-video frame shape as (height, width)."""

    for height_key, width_key in (
        ("source_video_height", "source_video_width"),
        ("source_full_height", "source_full_width"),
        ("height", "width"),
        ("video_height", "video_width"),
    ):
        height = _as_positive_int(zarr_root.attrs.get(height_key))
        width = _as_positive_int(zarr_root.attrs.get(width_key))
        if height is not None and width is not None:
            return height, width

    for key in ("source_video_resolution", "source_full_resolution"):
        shape = _resolution_from_attr(zarr_root.attrs.get(key))
        if shape is not None:
            return shape

    raw_video = zarr_root.get("raw_video") if hasattr(zarr_root, "get") else None
    if raw_video is not None:
        shape = _resolution_from_attr(raw_video.attrs.get("original_resolution"))
        if shape is not None:
            return shape
    return None


def _resolve_metadata_video_preview_shape(
    zarr_root,
    *,
    use_full_res: bool,
    source_shape: tuple[int, int],
) -> tuple[str, tuple[int, int]]:
    """Choose the coordinate space used when tuning from source-video fallback."""

    if use_full_res:
        return "images_full", source_shape

    height = _as_positive_int(zarr_root.attrs.get("inference_height"))
    width = _as_positive_int(zarr_root.attrs.get("inference_width"))
    if height is not None and width is not None:
        return "images_ds", (height, width)

    for key in ("inference_resolution", "palette_downsampled_resolution"):
        shape = _resolution_from_attr(zarr_root.attrs.get(key))
        if shape is not None:
            return "images_ds", shape

    raw_video = zarr_root.get("raw_video") if hasattr(zarr_root, "get") else None
    if raw_video is not None:
        shape = _resolution_from_attr(raw_video.attrs.get("downsampled_resolution"))
        if shape is not None:
            return "images_ds", shape

    source_h, source_w = source_shape
    max_dim = max(source_h, source_w)
    if max_dim > 640:
        scale = 640.0 / float(max_dim)
        return "images_ds", (max(1, round(source_h * scale)), max(1, round(source_w * scale)))
    return "images_full", source_shape


class _SourceVideoPreviewArray:
    """Small array-like adapter that decodes source-video frames on demand."""

    dtype = np.dtype("uint8")

    def __init__(
        self,
        video_path: Path,
        *,
        frame_count: int,
        source_shape: tuple[int, int],
        preview_shape: tuple[int, int],
    ) -> None:
        self.video_path = video_path
        self.shape = (int(frame_count), int(preview_shape[0]), int(preview_shape[1]))
        self._source_shape = source_shape
        self._preview_shape = preview_shape
        self._capture = cv2.VideoCapture(str(video_path))
        if not self._capture.isOpened():
            raise ValueError(f"Could not open source video: {video_path}")

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        index = int(frame_idx)
        if index < 0 or index >= self.shape[0]:
            raise IndexError(f"Frame index {index} out of bounds for source video with {self.shape[0]} frames.")
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not decode frame {index} from {self.video_path}.")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if tuple(frame.shape[:2]) != tuple(self._preview_shape):
            frame = cv2.resize(
                frame,
                (int(self._preview_shape[1]), int(self._preview_shape[0])),
                interpolation=cv2.INTER_AREA,
            )
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        self._capture.release()


def _resolve_video_array(zarr_root, *, use_full_res: bool):
    """Return `(array_like, array_name, description)` for tuning frames."""

    if "raw_video" not in zarr_root:
        available = ", ".join(sorted(zarr_root.keys()))
        raise KeyError(f"'raw_video' group not found. Children: {available or 'none'}")
    raw_video = zarr_root["raw_video"]

    has_full = "images_full" in raw_video
    has_ds = "images_ds" in raw_video
    if use_full_res and has_full:
        return raw_video["images_full"], "images_full", "raw_video/images_full"
    if has_ds:
        return raw_video["images_ds"], "images_ds", "raw_video/images_ds"
    if has_full:
        return raw_video["images_full"], "images_full", "raw_video/images_full"

    source_path = _resolve_source_video_path(zarr_root)
    source_shape = _resolve_source_video_shape(zarr_root)
    frame_count = _as_positive_int(
        raw_video.attrs.get("total_frames")
        or raw_video.attrs.get("source_video_total_frames")
        or zarr_root.attrs.get("total_frames")
        or zarr_root.attrs.get("n_frames")
    )
    if source_path is None or source_shape is None or frame_count is None:
        raise ValueError(
            "No video arrays found in raw_video, and source-video fallback metadata is incomplete."
        )

    array_name, preview_shape = _resolve_metadata_video_preview_shape(
        zarr_root,
        use_full_res=use_full_res,
        source_shape=source_shape,
    )
    return (
        _SourceVideoPreviewArray(
            source_path,
            frame_count=frame_count,
            source_shape=source_shape,
            preview_shape=preview_shape,
        ),
        array_name,
        f"source_video:{source_path}",
    )


def _normalize_mask_data(mask_data):
    """Return a normalized copy of mask data with explicit shape field."""
    if mask_data is None:
        return None
    normalized = dict(mask_data)
    shape = normalized.get('shape')
    if not shape:
        if 'detected_circle' in normalized:
            shape = 'circle'
        elif 'rectangle' in normalized:
            shape = 'rectangle'
    normalized['shape'] = shape
    if 'detected_circle' in normalized:
        normalized['detected_circle'] = dict(normalized['detected_circle'])
    if 'hough_params' in normalized:
        normalized['hough_params'] = dict(normalized['hough_params'])
    if 'rectangle' in normalized:
        normalized['rectangle'] = dict(normalized['rectangle'])
    if 'source' in normalized:
        normalized['source'] = dict(normalized['source'])
    return normalized


def _existing_dish_mask(zarr_root):
    """Return existing dish-mask metadata from the canonical attrs location."""
    if 'analysis_metadata' not in zarr_root:
        return None
    analysis_meta = zarr_root['analysis_metadata']
    if 'dish_mask' not in analysis_meta.attrs:
        return None
    return _normalize_mask_data(analysis_meta.attrs['dish_mask'])


def _circle_mask_payload(
    *,
    center: tuple[int, int] | list[int],
    radius: int,
    method: str = "headless_circle",
):
    center_x, center_y = int(center[0]), int(center[1])
    radius = int(radius)
    if radius <= 0:
        raise ValueError(f"Circle radius must be positive, got {radius}.")
    return {
        'shape': 'circle',
        'method': method,
        'detected_circle': {
            'center': [center_x, center_y],
            'radius': radius,
        },
    }


def save_headless_circle_mask(
    zarr_path,
    *,
    center: tuple[int, int] | list[int],
    radius: int,
    frame_idx: Optional[int] = None,
    use_full_res: bool = False,
    overwrite: bool = False,
    method: str = "headless_circle",
):
    """Save a circle dish mask without launching the OpenCV UI.

    The headless path is intended for deterministic backfills where the circle
    geometry has already been reviewed elsewhere. It refuses to overwrite an
    existing mask unless ``overwrite`` is explicitly enabled.
    """
    zarr_root = open_zarr_root(zarr_path, mode='r')
    existing_mask = _existing_dish_mask(zarr_root)
    if existing_mask is not None and not overwrite:
        return {
            "saved": False,
            "status": "exists",
            "zarr_path": str(zarr_path),
            "existing_mask": existing_mask,
        }

    video_array, array_name, array_source = _resolve_video_array(
        zarr_root,
        use_full_res=use_full_res,
    )
    try:
        max_frame_count = int(video_array.shape[0])
        if max_frame_count <= 0:
            raise ValueError("Video array has no frames.")
        if frame_idx is None:
            resolved_frame_idx = max_frame_count // 2
        else:
            resolved_frame_idx = int(np.clip(int(frame_idx), 0, max_frame_count - 1))
        image_shape = tuple(int(v) for v in video_array.shape[1:3])

        mask_payload = _circle_mask_payload(
            center=center,
            radius=radius,
            method=method,
        )
        saved = save_mask_to_zarr(
            zarr_path,
            mask_payload,
            array_name,
            resolved_frame_idx,
            image_shape=image_shape,
        )
    finally:
        if hasattr(video_array, "close"):
            video_array.close()

    return {
        "saved": bool(saved),
        "status": "saved" if saved else "error",
        "zarr_path": str(zarr_path),
        "array_name": array_name,
        "array_source": array_source,
        "frame_index": int(resolved_frame_idx),
        "center": [int(center[0]), int(center[1])],
        "radius": int(radius),
        "overwrite": bool(overwrite),
    }


def update_param1(val):
    global hough_param1
    hough_param1 = max(1, val)

def update_param2(val):
    global hough_param2
    hough_param2 = max(1, val)

def update_radius_adj(val):
    global radius_adjustment
    radius_adjustment = val - 20

def update_frame(val):
    global frame_index
    frame_index = val

def save_mask_to_zarr(zarr_path, mask_definition, array_name, frame_index, params=None, image_shape=None):
    """Save mask parameters to Zarr metadata ONLY."""
    try:
        zarr_root = open_zarr_root(zarr_path, mode='r+')
        
        if 'analysis_metadata' not in zarr_root:
            meta = zarr_root.create_group('analysis_metadata')
        else:
            meta = zarr_root['analysis_metadata']
        
        # Get existing metadata
        metadata = dict(meta.attrs) if meta.attrs else {}
        
        shape = mask_definition.get('shape')
        payload = {
            'shape': shape,
            'version': '2.0',
            'tuned_timestamp': datetime.now().isoformat(),
            'source': {
                'array': array_name,
                'frame': int(frame_index)
            }
        }
        payload['tuned_on_array'] = array_name
        payload['tuned_on_frame'] = int(frame_index)

        metrics = None
        if image_shape is not None:
            height, width = int(image_shape[0]), int(image_shape[1])
            metrics = {
                'image_shape': [height, width],
            }

        if shape == 'circle':
            circle = mask_definition['detected_circle']
            center_x, center_y, radius = int(circle['center'][0]), int(circle['center'][1]), int(circle['radius'])
            payload['method'] = mask_definition.get('method', 'hough_circle')
            payload['detected_circle'] = {
                'center': [center_x, center_y],
                'radius': radius
            }
            if metrics is not None:
                area_px = 3.141592653589793 * (radius ** 2)
                metrics.update({
                    'center_px': [center_x, center_y],
                    'center_norm': [center_x / width, center_y / height],
                    'radius_px': radius,
                    'radius_norm': radius / min(width, height),
                    'area_px': area_px,
                    'area_fraction': area_px / float(width * height),
                })
            if params:
                payload['hough_params'] = {
                    'param1': int(params.get('param1', 0)),
                    'param2': int(params.get('param2', 0)),
                    'radius_adjustment': int(params.get('radius_adjustment', 0))
                }
        elif shape == 'rectangle':
            payload['method'] = mask_definition.get('method', 'manual_rectangle')
            roi = mask_definition['rectangle']['roi']
            roi = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
            payload['rectangle'] = {
                'roi': roi
            }
            if metrics is not None:
                x, y, w, h = roi
                center_x = x + (w / 2.0)
                center_y = y + (h / 2.0)
                area_px = float(w * h)
                metrics.update({
                    'center_px': [center_x, center_y],
                    'center_norm': [center_x / width, center_y / height],
                    'area_px': area_px,
                    'area_fraction': area_px / float(width * height),
                })
        else:
            raise ValueError(f"Unsupported mask shape '{shape}'")

        if metrics is not None:
            payload['metrics'] = metrics
        
        metadata['dish_mask'] = payload
        
        meta.attrs.update(metadata)
        
        print(f"\n✅ Mask saved to zarr: {zarr_path}")
        print(f"   Location: analysis_metadata/attrs['dish_mask']")
        if shape == 'circle':
            print(f"   Shape: circle | Center: {payload['detected_circle']['center']} | Radius: {payload['detected_circle']['radius']}")
        elif shape == 'rectangle':
            print(f"   Shape: rectangle | ROI: {payload['rectangle']['roi']}")
        print(f"   This mask will be used automatically in detect and crop stages")
        
        return True
    except Exception as e:
        print(f"❌ Error saving to Zarr: {e}")
        return False

def load_from_zarr(zarr_path):
    """Load previously tuned mask parameters from Zarr if they exist"""
    try:
        zarr_root = open_zarr_root(zarr_path, mode='r')
        
        # Check new location first
        if 'analysis_metadata' in zarr_root:
            analysis_meta = zarr_root['analysis_metadata']
            if 'dish_mask' in analysis_meta.attrs:
                mask_data = analysis_meta.attrs['dish_mask']
                print(f" Found mask tuning in analysis_metadata")
                return _normalize_mask_data(mask_data)
        
        # Fall back to old location for backward compatibility
        if 'raw_video' in zarr_root:
            raw_video = zarr_root['raw_video']
            if 'dish_mask_tuning' in raw_video.attrs:
                print(f" Found mask tuning in raw_video (legacy location)")
                mask_data = raw_video.attrs['dish_mask_tuning']
                normalized = _normalize_mask_data(mask_data)
                if normalized is not None and 'shape' not in normalized:
                    normalized['shape'] = 'circle'
                return normalized
        
        return None
    except Exception as e:
        print(f"Could not load existing mask data: {e}")
        return None

def main(zarr_path, use_full_res=False, frame_idx=None, config_path=None, mode="auto"):
    global hough_param1, hough_param2, radius_adjustment, detected_circle, frame_index, max_frames, rectangle_roi, mask_mode
    
    detected_circle = None
    rectangle_roi = None
    
    if config_path is None:
        config_path = Path("src/pipeline_config.yaml")
    else:
        config_path = Path(config_path)
    
    # Try to load existing mask parameters
    existing_mask = load_from_zarr(zarr_path)
    mask_shape = existing_mask.get('shape') if existing_mask else None
    
    if mode == "auto":
        resolved_mode = mask_shape if mask_shape in {"circle", "rectangle"} else "circle"
    else:
        resolved_mode = mode
        if mask_shape and mask_shape != resolved_mode:
            print(f"⚠️  Existing mask shape '{mask_shape}' differs from requested mode '{resolved_mode}'.")
    mask_mode = resolved_mode
    
    if existing_mask:
        if mask_mode == 'circle' and 'detected_circle' in existing_mask:
            hough_params = existing_mask.get('hough_params', {})
            hough_param1 = int(hough_params.get('param1', hough_param1))
            hough_param2 = int(hough_params.get('param2', hough_param2))
            radius_adjustment = int(hough_params.get('radius_adjustment', radius_adjustment))
            circle = existing_mask['detected_circle']
            detected_circle = (
                int(circle['center'][0]),
                int(circle['center'][1]),
                int(circle['radius'])
            )
            print(f"   Loaded circle parameters: param1={hough_param1}, param2={hough_param2}, radius_adj={radius_adjustment}")
        elif mask_mode == 'rectangle' and 'rectangle' in existing_mask:
            roi = existing_mask['rectangle'].get('roi')
            if roi:
                rectangle_roi = [int(v) for v in roi]
                print(f"   Loaded rectangular ROI: {rectangle_roi}")
    
    try:
        zarr_root = open_zarr_root(zarr_path, mode='r')
        video_array, array_name, array_source = _resolve_video_array(
            zarr_root,
            use_full_res=use_full_res,
        )
        
        print(f"\n Using frame source: {array_source}")
        print(f"   Coordinate space: {array_name}")
        print(f"   Shape: {video_array.shape}")
        
        max_frames = video_array.shape[0]
        
        # Use specified frame or find a good representative frame
        if frame_idx is not None:
            frame_index = min(frame_idx, max_frames - 1)
        elif existing_mask:
            source = existing_mask.get('source', {})
            tuned_frame = source.get('frame') or existing_mask.get('tuned_on_frame')
            if tuned_frame is not None:
                frame_index = int(np.clip(tuned_frame, 0, max_frames - 1))
            else:
                frame_index = max_frames // 2
        else:
            # Use middle frame as default
            frame_index = max_frames // 2
            
    except Exception as e:
        print(f" Error opening Zarr file: {e}")
        return

    window_name = "Dish Mask Tuner"
    cv2.namedWindow(window_name)
    
    # Create trackbars
    cv2.createTrackbar("Frame", window_name, frame_index, max_frames - 1, update_frame)
    if mask_mode == 'circle':
        cv2.createTrackbar("param1", window_name, hough_param1, 200, update_param1)
        cv2.createTrackbar("param2", window_name, hough_param2, 200, update_param2)
        cv2.createTrackbar("Radius Adjust", window_name, radius_adjustment + 20, 40, update_radius_adj)
    
    print(f"\n Starting Dish Mask Tuner...")
    print("Controls:")
    print("  - Use 'Frame' slider to select different frames")
    if mask_mode == 'circle':
        print("  - Adjust Hough sliders to refine the circle fit")
        print("  - Press 's' to SAVE the circle parameters to Zarr")
        print("  - Press 'a' to auto-detect circle parameters")
        print("  - Press 'q' or Esc to quit without saving")
        print("\nGoal: Find parameters that draw a single, stable green circle around the dish.")
    else:
        print("  - Press 'r' (or space) to draw/select a rectangle ROI")
        print("  - Press 'x' to clear the current rectangle")
        print("  - Press 's' to SAVE the rectangle ROI to Zarr")
        print("  - Press 'q' or Esc to quit without saving")
        print("\nGoal: Draw a rectangle that covers the usable dish area.")

    while True:
        # Load current frame
        try:
            current_frame = video_array[frame_index]
            if current_frame.dtype != np.uint8:
                current_frame = (current_frame * 255).astype(np.uint8)
        except Exception as e:
            print(f"Error loading frame {frame_index}: {e}")
            continue
        
        # Convert to BGR for display
        if len(current_frame.shape) == 2:
            display_image = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
            gray_frame = current_frame
        else:
            display_image = current_frame.copy()
            gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if mask_mode == 'circle':
            # Apply Gaussian blur for better circle detection
            blurred = cv2.GaussianBlur(gray_frame, (9, 9), 2)
            
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                1,
                100,
                param1=hough_param1,
                param2=hough_param2,
                minRadius=0,
                maxRadius=0,
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Take only the first (best) circle
                i = circles[0, 0]
                
                # Apply radius adjustment
                adjusted_radius = int(i[2]) + radius_adjustment
                
                # Store the detected circle
                detected_circle = (int(i[0]), int(i[1]), adjusted_radius)
                
                # Draw the outer circle with the adjusted radius
                cv2.circle(display_image, (i[0], i[1]), adjusted_radius, (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(display_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            
            status_text = (
                f"Frame {frame_index}/{max_frames-1} | "
                f"param1={hough_param1}, param2={hough_param2}, radius_adj={radius_adjustment}"
            )
            cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if detected_circle:
                info_text = f"Circle: center=({detected_circle[0]}, {detected_circle[1]}), radius={detected_circle[2]}"
                cv2.putText(display_image, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            status_text = f"Frame {frame_index}/{max_frames-1} | press 'r' to select rectangle"
            cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if rectangle_roi:
                x, y, w, h = rectangle_roi
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                info_text = f"Rectangle: x={x}, y={y}, w={w}, h={h}"
                cv2.putText(display_image, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_image, "No rectangle selected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        cv2.imshow(window_name, display_image)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            if mask_mode == 'circle':
                if detected_circle is not None:
                    params = {
                        'param1': hough_param1,
                        'param2': hough_param2,
                        'radius_adjustment': radius_adjustment
                    }
                    mask_payload = {
                        'shape': 'circle',
                        'detected_circle': {
                            'center': [detected_circle[0], detected_circle[1]],
                            'radius': detected_circle[2]
                        }
                    }
                    success = save_mask_to_zarr(
                        zarr_path,
                        mask_payload,
                        array_name,
                        frame_index,
                        params=params,
                        image_shape=current_frame.shape[:2],
                    )
                    if success:
                        print("Press 'q' to quit or continue tuning")
                else:
                    print(" No circle detected - adjust parameters and try again")
            else:
                if rectangle_roi:
                    mask_payload = {
                        'shape': 'rectangle',
                        'rectangle': {
                            'roi': rectangle_roi
                        }
                    }
                    success = save_mask_to_zarr(
                        zarr_path,
                        mask_payload,
                        array_name,
                        frame_index,
                        image_shape=current_frame.shape[:2],
                    )
                    if success:
                        print("Press 'q' to quit or continue tuning")
                else:
                    print(" Draw a rectangle first (press 'r')")
        elif mask_mode == 'circle' and key == ord('a'):
            # Auto-detect mode
            print("\n🔍 Auto-detecting best parameters...")
            best_params = None
            best_score = 0
            
            blurred = cv2.GaussianBlur(gray_frame, (9, 9), 2)
            
            for p1 in range(20, 150, 20):
                for p2 in range(10, 50, 10):
                    circles = cv2.HoughCircles(
                        blurred,
                        cv2.HOUGH_GRADIENT,
                        1,
                        100,
                        param1=p1,
                        param2=p2,
                        minRadius=0,
                        maxRadius=0
                    )
                    if circles is not None and len(circles[0]) == 1:
                        score = 100 - abs(p1 - 50) - abs(p2 - 30)
                        if score > best_score:
                            best_score = score
                            best_params = (p1, p2)
            
            if best_params:
                hough_param1, hough_param2 = best_params
                cv2.setTrackbarPos("param1", window_name, hough_param1)
                cv2.setTrackbarPos("param2", window_name, hough_param2)
                auto_detected = True
                print(f"   Found: param1={hough_param1}, param2={hough_param2}")
            else:
                print("   Could not find good parameters automatically")
        elif mask_mode == 'rectangle' and key in (ord('r'), ord(' ')):
            # Launch ROI selector
            roi = cv2.selectROI(window_name, display_image, fromCenter=False, showCrosshair=True)
            if roi and roi[2] > 0 and roi[3] > 0:
                rectangle_roi = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
                print(f"  Selected rectangle ROI: {rectangle_roi}")
            else:
                print("  Rectangle selection cancelled")
        elif mask_mode == 'rectangle' and key == ord('x'):
            rectangle_roi = None
            print("  Cleared rectangle ROI")
            
    cv2.destroyAllWindows()
    if hasattr(video_array, "close"):
        video_array.close()
    print("\nTuner closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively tune dish masks (circle or rectangle).")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file containing imported video.")
    parser.add_argument("--full", action="store_true", help="Use full resolution array instead of downsampled")
    parser.add_argument("--frame", type=int, help="Specific frame index to use")
    parser.add_argument("--config", type=str, help="Path to YAML config file (default: src/pipeline_config.yaml)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "circle", "rectangle"],
        default="auto",
        help="Mask tuning mode (default: auto – reuse stored shape or circle).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Write a reviewed circle mask without launching the OpenCV UI.",
    )
    parser.add_argument(
        "--circle-center",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="Circle center in the selected mask coordinate space for --headless.",
    )
    parser.add_argument(
        "--circle-radius",
        type=int,
        help="Circle radius in pixels for --headless.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --headless to replace an existing dish mask.",
    )
    args = parser.parse_args()

    if args.headless:
        if args.mode not in {"auto", "circle"}:
            parser.error("--headless currently supports circle masks only.")
        if args.circle_center is None or args.circle_radius is None:
            parser.error("--headless requires --circle-center X Y and --circle-radius R.")
        result = save_headless_circle_mask(
            args.zarr_path,
            center=args.circle_center,
            radius=args.circle_radius,
            frame_idx=args.frame,
            use_full_res=args.full,
            overwrite=bool(args.overwrite),
        )
        print(result)
        raise SystemExit(0 if result["status"] in {"saved", "exists"} else 1)

    main(args.zarr_path, use_full_res=args.full, frame_idx=args.frame,
         config_path=args.config, mode=args.mode)
