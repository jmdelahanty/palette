"""Canonical arena geometry: prefer the fitted dish mask over the nominal circle.

There are two circles in a Citrus archive and they are not the same circle.

``calibration/arena_geometry.experimental_area_*`` is the **projector's nominal model** of the
dish: the configured design radius, placed at the arena-region centre. It is what the stimulus
system draws against.

``analysis_metadata.attrs["dish_mask"]`` is a **Hough circle fitted to the actual camera
image** of the actual dish. ``refine_detect`` already gates detections with it. It is where the
dish really is.

On the one GoodCopBadCop recording we have, mapped into the same arena-canvas frame:

    nominal experimental_area : centre (40.94, 40.94) mm, radius 40.00 mm
    fitted dish mask          : centre (38.06, 41.75) mm, radius 42.44 mm
    the fish's own envelope   : centre (38.05, 41.78) mm, radius 41.90 mm

The mask agrees with the fish to 0.03 mm on centre; the 0.5 mm radius gap is the fish's body
half-width (a tracked centroid cannot reach the wall). The nominal circle is 3.0 mm off-centre
and 2.4 mm small -- plausibly the refraction stack, which a flat design circle does not model.

Reading the nominal circle put 56% of the post-period frames "outside the arena", which
``cra_near_field._thigmotaxis_for_phase`` then silently dropped from its numerator but not its
denominator -- turning a 0.37 -> 0.87 thigmotaxis increase into 0.354 -> 0.353. See
``docs/diagnostics/arena_calibration_and_thigmotaxis_2026-07-14.md``.

So: resolve from the dish mask. Fall back to the nominal circle only when there is no mask, and
say so in QC.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.coordinate_transform import projector_to_camera_px


GEOMETRY_STATUS_DISH_MASK = "dish_mask"
GEOMETRY_STATUS_NOMINAL_CIRCLE = "circle"
GEOMETRY_STATUS_RECTANGLE = "rectangular_approximation"

# The homography is projective, so a circle in camera pixels maps to a conic, not exactly a
# circle. In practice the residual is ~0.1 mm; above this the circle model is not trustworthy.
DEFAULT_MAX_FIT_RESIDUAL_MM = 1.0
# Above this fraction of fish frames outside the resolved arena, the geometry is wrong, not the
# fish. Say so loudly instead of quietly dropping the frames.
DEFAULT_OUT_OF_BOUNDS_WARN_FRACTION = 0.02


@dataclass(frozen=True)
class ArenaGeometry:
    status: str
    source: str | None
    width_px: float
    height_px: float
    shape: str
    center_x_px: float | None
    center_y_px: float | None
    radius_px: float | None
    fit_residual_px: float | None = None


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _optional_float(value: Any) -> float | None:
    out = _safe_float(value)
    return out if math.isfinite(out) else None


def _coerce_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _get_group(root: zarr.Group, path: str | None) -> zarr.Group | None:
    normalized = "/".join(part for part in str(path or "").strip("/").split("/") if part)
    if not normalized:
        return root
    current: Any = root
    for part in normalized.split("/"):
        try:
            current = current[part]
        except Exception:
            return None
        if not isinstance(current, zarr.Group):
            return None
    return current


def read_dish_mask_circle(root: zarr.Group) -> dict[str, float] | None:
    """The dish mask as normalized coordinates, exactly as ``refine_detect`` reads it.

    Returns ``{center_norm_x, center_norm_y, radius_norm}`` or None. Normalizing here means the
    result is independent of whatever resolution the mask was tuned at (it is tuned on the
    downsampled ``images_ds``, typically 640x640, while detections live at full source
    resolution).
    """

    meta = root.get("analysis_metadata")
    if meta is None or "dish_mask" not in meta.attrs:
        return None
    data = _coerce_mapping(meta.attrs.get("dish_mask"))
    if not data:
        return None
    if str(data.get("shape") or "").strip().lower() != "circle":
        return None

    metrics = _coerce_mapping(data.get("metrics")) or {}
    image_shape = metrics.get("image_shape")
    mask_h = mask_w = None
    if isinstance(image_shape, Sequence) and len(image_shape) >= 2:
        mask_h = _optional_float(image_shape[0])
        mask_w = _optional_float(image_shape[1])

    circle = _coerce_mapping(data.get("detected_circle"))
    if circle is not None and mask_w and mask_h and mask_w > 0 and mask_h > 0:
        center = circle.get("center")
        radius = _optional_float(circle.get("radius"))
        if isinstance(center, Sequence) and len(center) >= 2 and radius and radius > 0:
            cx = _optional_float(center[0])
            cy = _optional_float(center[1])
            if cx is not None and cy is not None:
                return {
                    "center_norm_x": cx / mask_w,
                    "center_norm_y": cy / mask_h,
                    "radius_norm": radius / mask_w,
                }

    center_norm = metrics.get("center_norm")
    radius_norm = _optional_float(metrics.get("radius_norm"))
    if isinstance(center_norm, Sequence) and len(center_norm) >= 2 and radius_norm and radius_norm > 0:
        cx = _optional_float(center_norm[0])
        cy = _optional_float(center_norm[1])
        if cx is not None and cy is not None:
            return {"center_norm_x": cx, "center_norm_y": cy, "radius_norm": radius_norm}
    return None


def detection_frame_size_px(root: zarr.Group) -> tuple[float, float] | None:
    """(width, height) of the frame the detections were produced in."""

    for family in ("detect_runs", "refined_detect_runs"):
        parent = root.get(family)
        if parent is None:
            continue
        for key in parent.keys():
            attrs = parent[key].attrs
            width = _optional_float(attrs.get("source_video_width")) or _optional_float(attrs.get("source_full_width"))
            height = _optional_float(attrs.get("source_video_height")) or _optional_float(attrs.get("source_full_height"))
            if width and height and width > 0 and height > 0:
                return float(width), float(height)
    calibration = _get_group(root, "analysis/calibration")
    if calibration is not None:
        width = _optional_float(calibration.attrs.get("native_width_px"))
        height = _optional_float(calibration.attrs.get("native_height_px"))
        if width and height and width > 0 and height > 0:
            return float(width), float(height)
    return None


def _arena_origin_px(root: zarr.Group, run_group: zarr.Group) -> tuple[float, float] | None:
    source_path = str(run_group.attrs.get("source_stimulus_path") or "").strip()
    group = _get_group(root, f"{source_path}/calibration/arena_geometry") if source_path else None
    if group is None:
        return None
    x = _optional_float(group.attrs.get("arena_origin_in_canvas_x_px"))
    y = _optional_float(group.attrs.get("arena_origin_in_canvas_y_px"))
    if x is None or y is None:
        return None
    return float(x), float(y)


def _arena_region_size_px(root: zarr.Group, run_group: zarr.Group) -> tuple[float, float]:
    source_path = str(run_group.attrs.get("source_stimulus_path") or "").strip()
    group = _get_group(root, f"{source_path}/calibration/arena_geometry") if source_path else None
    if group is not None:
        width = _optional_float(group.attrs.get("arena_region_width_px"))
        height = _optional_float(group.attrs.get("arena_region_height_px"))
        if width and height and width > 0 and height > 0:
            return float(width), float(height)
    return math.nan, math.nan


def _fit_circle(points: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic (Kasa) circle fit. Returns (cx, cy, r, rms_residual)."""

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    design = np.stack([2.0 * pts[:, 0], 2.0 * pts[:, 1], np.ones(pts.shape[0])], axis=1)
    target = pts[:, 0] ** 2 + pts[:, 1] ** 2
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    cx, cy = float(solution[0]), float(solution[1])
    radius = float(math.sqrt(max(solution[2] + cx * cx + cy * cy, 0.0)))
    residual = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - radius
    return cx, cy, radius, float(np.sqrt(np.mean(residual**2)))


def resolve_dish_mask_geometry(
    root: zarr.Group,
    run_group: zarr.Group,
    *,
    pixels_per_mm: float,
    n_samples: int = 720,
    max_fit_residual_mm: float = DEFAULT_MAX_FIT_RESIDUAL_MM,
) -> tuple[ArenaGeometry | None, list[str]]:
    """The fitted dish mask, projected into the arena-canvas frame the fish positions live in.

    The mask is fitted in camera-image pixels. Fish positions in a chaser-distance run are
    ``arena_relative_canvas_px`` -- camera pixels pushed through the homography and shifted by
    the arena origin. To compare them, the mask has to make the same trip.
    """

    notes: list[str] = []
    circle = read_dish_mask_circle(root)
    if circle is None:
        return None, notes

    frame = detection_frame_size_px(root)
    if frame is None:
        notes.append("dish_mask_unusable:no_detection_frame_size")
        return None, notes
    width_px, height_px = frame

    calibration = _get_group(root, "analysis/calibration")
    if calibration is None or "homography_matrix" not in calibration:
        notes.append("dish_mask_unusable:no_homography_matrix")
        return None, notes
    homography = np.asarray(calibration["homography_matrix"][:], dtype=np.float64)
    if homography.shape != (3, 3):
        notes.append("dish_mask_unusable:bad_homography_shape")
        return None, notes

    origin = _arena_origin_px(root, run_group)
    if origin is None:
        notes.append("dish_mask_unusable:no_arena_origin")
        return None, notes

    center_x = float(circle["center_norm_x"]) * width_px
    center_y = float(circle["center_norm_y"]) * height_px
    radius = float(circle["radius_norm"]) * width_px
    if not (radius > 0):
        notes.append("dish_mask_unusable:non_positive_radius")
        return None, notes

    theta = np.linspace(0.0, 2.0 * math.pi, int(max(16, n_samples)), endpoint=False)
    ring = np.stack([center_x + radius * np.cos(theta), center_y + radius * np.sin(theta)], axis=1)
    # Same transform the fish takes: camera px -> canvas px -> minus arena origin.
    canvas = projector_to_camera_px(ring, homography)
    arena = canvas - np.asarray(origin, dtype=np.float64).reshape(1, 2)
    if not np.isfinite(arena).all():
        notes.append("dish_mask_unusable:non_finite_projection")
        return None, notes

    cx, cy, r, residual_px = _fit_circle(arena)
    residual_mm = residual_px / float(pixels_per_mm) if pixels_per_mm > 0 else math.inf
    if residual_mm > float(max_fit_residual_mm):
        # The homography is projective, so the mask maps to a conic. A large residual means the
        # circle model no longer describes it and we should not pretend otherwise.
        notes.append(f"dish_mask_projection_not_circular:residual_mm={residual_mm:.3f}")
        return None, notes

    region_w, region_h = _arena_region_size_px(root, run_group)
    if not math.isfinite(region_w) or not math.isfinite(region_h):
        region_w = region_h = 2.0 * r

    return (
        ArenaGeometry(
            status=GEOMETRY_STATUS_DISH_MASK,
            source="analysis_metadata.dish_mask",
            width_px=float(region_w),
            height_px=float(region_h),
            shape="circle",
            center_x_px=float(cx),
            center_y_px=float(cy),
            radius_px=float(r),
            fit_residual_px=float(residual_px),
        ),
        notes,
    )


def resolve_nominal_geometry(
    root: zarr.Group,
    run_group: zarr.Group,
    *,
    pixels_per_mm: float,
) -> ArenaGeometry | None:
    """The projector's nominal ``experimental_area`` circle -- the fallback, not the truth."""

    candidates: list[tuple[str, Mapping[str, Any]]] = []
    source_path = str(run_group.attrs.get("source_stimulus_path") or "").strip()
    if source_path:
        group = _get_group(root, f"{source_path}/calibration/arena_geometry")
        if group is not None:
            candidates.append((f"{source_path}/calibration/arena_geometry", group.attrs))
    calibration = _get_group(root, "analysis/calibration")
    if calibration is not None:
        candidates.append(("analysis/calibration", calibration.attrs))

    for source, attrs in candidates:
        width = _optional_float(attrs.get("arena_region_width_px")) or _optional_float(
            attrs.get("experimental_area_width_px")
        )
        height = _optional_float(attrs.get("arena_region_height_px")) or _optional_float(
            attrs.get("experimental_area_height_px")
        )
        shape = str(attrs.get("experimental_area_shape") or attrs.get("arena_shape") or "").strip().lower()
        cx = _optional_float(attrs.get("experimental_area_center_x_px"))
        cy = _optional_float(attrs.get("experimental_area_center_y_px"))
        radius = _optional_float(attrs.get("experimental_area_radius_px"))
        if radius is None:
            radius_mm = _optional_float(attrs.get("experimental_area_radius_mm"))
            if radius_mm is not None and pixels_per_mm > 0:
                radius = float(radius_mm) * float(pixels_per_mm)
        if shape == "circle" and cx is not None and cy is not None and radius and radius > 0:
            box = 2.0 * float(radius)
            return ArenaGeometry(
                status=GEOMETRY_STATUS_NOMINAL_CIRCLE,
                source=source,
                width_px=float(width or box),
                height_px=float(height or box),
                shape="circle",
                center_x_px=float(cx),
                center_y_px=float(cy),
                radius_px=float(radius),
            )
        if width and height and width > 0 and height > 0:
            return ArenaGeometry(
                status=GEOMETRY_STATUS_RECTANGLE,
                source=source,
                width_px=float(width),
                height_px=float(height),
                shape="rectangle",
                center_x_px=None,
                center_y_px=None,
                radius_px=None,
            )
    return None


def resolve_arena_geometry(
    root: zarr.Group,
    run_group: zarr.Group,
    *,
    pixels_per_mm: float,
    prefer_dish_mask: bool = True,
) -> tuple[ArenaGeometry, list[str]]:
    """The dish mask if there is one, the nominal circle otherwise.

    Returns the geometry and a list of QC notes. A caller that silently ignores the notes will
    not know it fell back to a circle that is 3 mm off.
    """

    notes: list[str] = []
    if prefer_dish_mask:
        geometry, mask_notes = resolve_dish_mask_geometry(root, run_group, pixels_per_mm=pixels_per_mm)
        notes.extend(mask_notes)
        if geometry is not None:
            return geometry, notes

    fallback = resolve_nominal_geometry(root, run_group, pixels_per_mm=pixels_per_mm)
    if fallback is None:
        raise ValueError(
            "Unable to resolve arena geometry: no usable analysis_metadata.dish_mask and no "
            "calibration/arena_geometry with a circle (centre + radius) or rectangle definition."
        )
    notes.append(
        f"arena_geometry_fallback_to_nominal:{fallback.status}"
        if prefer_dish_mask
        else f"arena_geometry_nominal_requested:{fallback.status}"
    )
    return fallback, notes


def out_of_bounds_fraction(
    points_px: np.ndarray,
    valid: np.ndarray | None,
    geometry: ArenaGeometry,
) -> float:
    """Fraction of valid points lying outside the arena. Should be ~0. If it is not, the
    geometry is wrong -- not the fish."""

    xy = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    ok = np.isfinite(xy).all(axis=1)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool).reshape(-1)
    if not np.any(ok):
        return math.nan
    pts = xy[ok]
    if geometry.shape == "circle" and geometry.center_x_px is not None and geometry.radius_px:
        radial = np.hypot(pts[:, 0] - float(geometry.center_x_px), pts[:, 1] - float(geometry.center_y_px))
        return float(np.mean(radial > float(geometry.radius_px)))
    outside = (
        (pts[:, 0] < 0.0)
        | (pts[:, 1] < 0.0)
        | (pts[:, 0] > float(geometry.width_px))
        | (pts[:, 1] > float(geometry.height_px))
    )
    return float(np.mean(outside))


def out_of_bounds_notes(
    points_px: np.ndarray,
    valid: np.ndarray | None,
    geometry: ArenaGeometry,
    *,
    label: str = "fish",
    warn_fraction: float = DEFAULT_OUT_OF_BOUNDS_WARN_FRACTION,
) -> list[str]:
    fraction = out_of_bounds_fraction(points_px, valid, geometry)
    if math.isfinite(fraction) and fraction > float(warn_fraction):
        return [
            f"arena_geometry_out_of_bounds:{label}:{fraction:.3f}_of_valid_frames_outside_"
            f"{geometry.status}_arena"
        ]
    return []
