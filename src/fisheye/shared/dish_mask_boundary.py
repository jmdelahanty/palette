"""Versioned physical tolerance for the refined-detection dish-mask gate."""

from __future__ import annotations

import math
from typing import Any, Mapping


DISH_MASK_BOUNDARY_TOLERANCE_CONTRACT = "palette.dish_mask_boundary_tolerance.v1"
DEFAULT_DISH_MASK_BOUNDARY_TOLERANCE_MM = 0.5


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _group_at(root: Any, path: str) -> Any | None:
    current = root
    for part in str(path).strip("/").split("/"):
        if not part:
            continue
        try:
            current = current[part]
        except Exception:
            return None
    return current


def _pixels_per_mm_from_attrs(attrs: Mapping[str, Any]) -> tuple[float | None, str | None]:
    direct = _positive_float(attrs.get("pixels_per_mm_camera"))
    if direct is not None:
        return direct, "pixels_per_mm_camera"
    mm_per_pixel = _positive_float(attrs.get("pixel_to_mm"))
    if mm_per_pixel is not None:
        return 1.0 / mm_per_pixel, "pixel_to_mm_inverse"
    return None, None


def _resolve_pixels_per_mm_camera(
    root: Any,
    *,
    explicit_pixels_per_mm_camera: float | None,
) -> tuple[float, str]:
    if explicit_pixels_per_mm_camera is not None:
        explicit = _positive_float(explicit_pixels_per_mm_camera)
        if explicit is None:
            raise ValueError("pixels_per_mm_camera override must be finite and positive")
        return explicit, "explicit_argument"

    candidates: list[tuple[float, str]] = []
    for path in ("analysis/calibration", "calibration"):
        group = _group_at(root, path)
        if group is None:
            continue
        value, attr_name = _pixels_per_mm_from_attrs(group.attrs)
        if value is not None and attr_name is not None:
            candidates.append((value, f"{path}.attrs.{attr_name}"))
    value, attr_name = _pixels_per_mm_from_attrs(root.attrs)
    if value is not None and attr_name is not None:
        candidates.append((value, f"root.attrs.{attr_name}"))

    if not candidates:
        raise ValueError(
            "A positive camera-space calibration is required for the physical dish-mask "
            "boundary tolerance. Store analysis/calibration.attrs.pixels_per_mm_camera "
            "or pass an explicit pixels_per_mm_camera override."
        )
    reference = candidates[0][0]
    conflicts = [
        (value, source)
        for value, source in candidates[1:]
        if not math.isclose(value, reference, rel_tol=1e-9, abs_tol=1e-12)
    ]
    if conflicts:
        raise ValueError(f"Conflicting pixels_per_mm_camera calibration values: {candidates}")
    return candidates[0]


def _shape_from_attrs(attrs: Mapping[str, Any]) -> tuple[float, float] | None:
    for height_name, width_name in (
        ("source_video_height", "source_video_width"),
        ("video_height", "video_width"),
        ("height", "width"),
    ):
        height = _positive_float(attrs.get(height_name))
        width = _positive_float(attrs.get(width_name))
        if height is not None and width is not None:
            return height, width
    for name in ("original_resolution", "frame_source_shape"):
        raw = attrs.get(name)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            height = _positive_float(raw[-2])
            width = _positive_float(raw[-1])
            if height is not None and width is not None:
                return height, width
    return None


def _resolve_source_frame_shape(root: Any, source_group: Any | None) -> tuple[float, float, str]:
    raw_video = _group_at(root, "raw_video")
    if raw_video is not None:
        shape = _shape_from_attrs(raw_video.attrs)
        if shape is not None:
            return shape[0], shape[1], "raw_video.attrs"
    shape = _shape_from_attrs(root.attrs)
    if shape is not None:
        return shape[0], shape[1], "root.attrs"
    if source_group is not None:
        shape = _shape_from_attrs(source_group.attrs)
        if shape is not None:
            return shape[0], shape[1], "source_group.attrs"
    if raw_video is not None:
        for name in ("images_full", "images_ds", "images_ds_rgb"):
            if name not in raw_video:
                continue
            shape = getattr(raw_video[name], "shape", ())
            if len(shape) >= 2:
                height = _positive_float(shape[-2])
                width = _positive_float(shape[-1])
                if height is not None and width is not None:
                    return height, width, f"raw_video/{name}.shape"
    raise ValueError(
        "Full camera-frame width and height are required to convert the physical "
        "dish-mask tolerance to normalized coordinates."
    )


def resolve_dish_mask_boundary_tolerance(
    root: Any,
    *,
    source_group: Any | None,
    tolerance_mm: float = DEFAULT_DISH_MASK_BOUNDARY_TOLERANCE_MM,
    pixels_per_mm_camera: float | None = None,
) -> dict[str, Any]:
    """Resolve a physical tolerance into camera-pixel and normalized units."""

    try:
        requested_mm = float(tolerance_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError("dish-mask boundary tolerance must be numeric") from exc
    if not math.isfinite(requested_mm) or requested_mm < 0:
        raise ValueError("dish-mask boundary tolerance must be finite and non-negative")
    if requested_mm == 0:
        return {
            "contract": DISH_MASK_BOUNDARY_TOLERANCE_CONTRACT,
            "enabled": False,
            "requested_mm": 0.0,
            "pixels_per_mm_camera": None,
            "calibration_source": None,
            "source_frame_shape_hw": None,
            "source_frame_shape_source": None,
            "tolerance_source_px": 0.0,
            "tolerance_norm_x": 0.0,
            "tolerance_norm_y": 0.0,
        }

    ppm, calibration_source = _resolve_pixels_per_mm_camera(
        root,
        explicit_pixels_per_mm_camera=pixels_per_mm_camera,
    )
    frame_h, frame_w, frame_shape_source = _resolve_source_frame_shape(root, source_group)
    tolerance_source_px = requested_mm * ppm
    return {
        "contract": DISH_MASK_BOUNDARY_TOLERANCE_CONTRACT,
        "enabled": True,
        "requested_mm": requested_mm,
        "pixels_per_mm_camera": ppm,
        "calibration_source": calibration_source,
        "source_frame_shape_hw": [frame_h, frame_w],
        "source_frame_shape_source": frame_shape_source,
        "tolerance_source_px": tolerance_source_px,
        "tolerance_norm_x": tolerance_source_px / frame_w,
        "tolerance_norm_y": tolerance_source_px / frame_h,
    }


def apply_dish_mask_boundary_tolerance(
    mask_spec: Mapping[str, Any],
    tolerance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an effective mask spec while preserving its fitted base geometry."""

    effective = dict(mask_spec)
    tol_x = float(tolerance.get("tolerance_norm_x") or 0.0)
    tol_y = float(tolerance.get("tolerance_norm_y") or 0.0)
    shape = str(effective.get("shape") or "")
    if shape == "circle":
        base_x = float(effective.get("radius_norm_x") or 0.0)
        base_y = float(effective.get("radius_norm_y") or 0.0)
        effective["base_radius_norm_x"] = base_x
        effective["base_radius_norm_y"] = base_y
        effective["radius_norm_x"] = base_x + tol_x
        effective["radius_norm_y"] = base_y + tol_y
    elif shape == "rectangle":
        base = {
            "x_min_norm": float(effective.get("x_min_norm") or 0.0),
            "y_min_norm": float(effective.get("y_min_norm") or 0.0),
            "x_max_norm": float(effective.get("x_max_norm") or 0.0),
            "y_max_norm": float(effective.get("y_max_norm") or 0.0),
        }
        effective["base_rectangle_norm"] = base
        effective["x_min_norm"] = base["x_min_norm"] - tol_x
        effective["y_min_norm"] = base["y_min_norm"] - tol_y
        effective["x_max_norm"] = base["x_max_norm"] + tol_x
        effective["y_max_norm"] = base["y_max_norm"] + tol_y
    else:
        raise ValueError(f"Unsupported dish-mask shape for boundary tolerance: {shape!r}")

    mask_shape = effective.get("mask_image_shape_hw")
    tolerance_payload = dict(tolerance)
    if isinstance(mask_shape, (list, tuple)) and len(mask_shape) >= 2:
        mask_h = _positive_float(mask_shape[-2])
        mask_w = _positive_float(mask_shape[-1])
        if mask_h is not None and mask_w is not None:
            tolerance_payload["effective_tolerance_mask_px_xy"] = [
                tol_x * mask_w,
                tol_y * mask_h,
            ]
    effective["boundary_tolerance"] = tolerance_payload
    return effective


__all__ = [
    "DEFAULT_DISH_MASK_BOUNDARY_TOLERANCE_MM",
    "DISH_MASK_BOUNDARY_TOLERANCE_CONTRACT",
    "apply_dish_mask_boundary_tolerance",
    "resolve_dish_mask_boundary_tolerance",
]
