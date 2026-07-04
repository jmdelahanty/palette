"""
Helpers for inspecting Palette Zarr metadata (frame shapes, formats, etc.).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import zarr


def _require_raw_video(root: zarr.Group) -> zarr.Group:
    if "raw_video" not in root:
        raise KeyError("Zarr archive missing 'raw_video' group.")
    return root["raw_video"]


def get_full_frame_shape(root: zarr.Group) -> Optional[Tuple[int, int]]:
    """Return (height, width) of raw full-resolution frames, if available."""
    raw = _require_raw_video(root)
    if "images_full" in raw:
        _, h, w = raw["images_full"].shape
        return int(h), int(w)
    shape = raw.attrs.get("original_resolution")
    if shape and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    return None


def get_downsample_formats(root: zarr.Group) -> List[str]:
    """Return list of downsampled formats recorded for this archive."""
    raw = _require_raw_video(root)
    from_attr = raw.attrs.get("downsample_formats")
    if isinstance(from_attr, (list, tuple)):
        normalized = []
        for item in from_attr:
            val = str(item).strip().lower()
            if val in {"gray", "grey", "grayscale"}:
                val = "gray"
            elif val in {"rgb", "color", "colour"}:
                val = "rgb"
            else:
                continue
            if val not in normalized:
                normalized.append(val)
        if normalized:
            return normalized

    # Fallback: inspect arrays
    formats: List[str] = []
    if "images_ds" in raw:
        formats.append("gray")
    if "images_ds_rgb" in raw:
        formats.append("rgb")
    return formats


def get_downsample_shape(
    root: zarr.Group,
    *,
    format_hint: str = "gray",
) -> Optional[Tuple[int, ...]]:
    """Return the spatial shape for the requested downsampled format."""
    name = get_downsample_array_name(root, format_hint)
    if name is None:
        return None
    raw = _require_raw_video(root)
    arr = raw[name]
    return tuple(int(dim) for dim in arr.shape[1:])


def get_downsample_array_name(
    root: zarr.Group,
    format_hint: str = "gray",
) -> Optional[str]:
    """Return array name inside raw_video for the requested format."""
    raw = _require_raw_video(root)
    preferred = format_hint.strip().lower() if isinstance(format_hint, str) else "gray"
    if preferred == "rgb" and "images_ds_rgb" in raw:
        return "images_ds_rgb"
    if "images_ds" in raw:
        return "images_ds"
    if "images_ds_rgb" in raw:
        return "images_ds_rgb"
    return None


def describe_downsample_arrays(root: zarr.Group) -> Dict[str, Tuple[int, ...]]:
    """Return mapping of array names -> spatial shapes for downsampled data."""
    raw = _require_raw_video(root)
    shapes = raw.attrs.get("downsampled_shapes")
    if isinstance(shapes, dict) and shapes:
        try:
            return {str(k): tuple(int(x) for x in v) for k, v in shapes.items()}
        except Exception:
            pass
    result: Dict[str, Tuple[int, ...]] = {}
    for name in ("images_ds", "images_ds_rgb"):
        if name in raw:
            shape = tuple(int(dim) for dim in raw[name].shape[1:])
            result[name] = shape
    return result


def get_downsample_array_path(
    root: zarr.Group,
    *,
    format_hint: str = "gray",
) -> Optional[str]:
    """Return fully-qualified array path (raw_video/...) for requested format."""
    array_name = get_downsample_array_name(root, format_hint)
    if array_name is None:
        return None
    return f"raw_video/{array_name}"


def _get_refined_keypoints_group(root: zarr.Group) -> Optional[zarr.Group]:
    if "refined_keypoints_runs" in root:
        return root["refined_keypoints_runs"]
    if "keypoints_refined_runs" in root:
        return root["keypoints_refined_runs"]
    return None


def get_latest_refined_keypoints_run(root: zarr.Group) -> Optional[str]:
    """Return latest refined keypoints run name if available."""
    group = _get_refined_keypoints_group(root)
    if group is None:
        return None
    latest = group.attrs.get("latest")
    if latest and latest in group:
        return str(latest)
    return None


def get_latest_refined_keypoints_summary(root: zarr.Group) -> Optional[Dict[str, object]]:
    """Return summary_statistics for the latest refined keypoints run."""
    group = _get_refined_keypoints_group(root)
    if group is None:
        return None
    run_name = get_latest_refined_keypoints_run(root)
    if run_name is None:
        return None
    return group[run_name].attrs.get("summary_statistics")
