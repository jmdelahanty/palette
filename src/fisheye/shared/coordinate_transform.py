"""
Projector-to-camera coordinate transforms.

Uses the homography matrix stored in ``analysis/calibration/`` to transform
points from projector/texture space (358x358) to camera space (4512x4512),
then converts to millimetres using the camera-space ``pixel_to_mm``
calibration.

The homography maps projector -> camera.  To go the other direction
(camera -> projector), invert the matrix.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def load_calibration_transform(
    root: Any,
) -> Dict[str, Any]:
    """Load calibration from the zarr calibration group.

    Returns a dict with keys:
    - ``homography``: 3x3 float64 array (projector -> camera), or None
    - ``pixel_to_mm``: float (camera pixels to mm), or None
    - ``arena_center_px``: (x, y) in camera pixels, or None
    - ``pixels_per_mm_projector``: float (projector/texture pixels to mm), or None
    - ``z_eff_mm``: float (effective viewing distance through media), or None
    """
    result: Dict[str, Any] = {
        "homography": None,
        "pixel_to_mm": None,
        "arena_center_px": None,
        "pixels_per_mm_projector": None,
        "z_eff_mm": None,
    }

    calib = root.get("analysis", {})
    if hasattr(calib, "__getitem__"):
        calib = calib.get("calibration")
    if calib is None:
        return result

    # Homography matrix.
    if "homography_matrix" in calib:
        h = calib["homography_matrix"][:]
        if h.shape == (3, 3):
            result["homography"] = h.astype(np.float64)

    # pixel_to_mm (camera space).
    attrs = calib.attrs if hasattr(calib, "attrs") else {}
    ptm = attrs.get("pixel_to_mm") or attrs.get("pixels_per_mm_camera")
    if ptm is not None:
        result["pixel_to_mm"] = float(ptm)

    # pixels_per_mm_projector (projector/texture space).
    ppm_proj = attrs.get("pixels_per_mm_projector")
    if ppm_proj is not None:
        result["pixels_per_mm_projector"] = float(ppm_proj)

    # z_eff_mm (effective viewing distance for visual angle computation).
    z_eff = attrs.get("z_eff_mm")
    if z_eff is not None:
        result["z_eff_mm"] = float(z_eff)

    # Arena center in camera pixels.
    cx = attrs.get("arena_center_x_px")
    cy = attrs.get("arena_center_y_px")
    if cx is not None and cy is not None:
        result["arena_center_px"] = (float(cx), float(cy))

    return result


def projector_to_camera_px(
    points_proj: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    """Transform points from projector space to camera pixel space.

    Parameters
    ----------
    points_proj : (N, 2) or (2,) float
        XY coordinates in projector/texture pixels.
    homography : (3, 3) float
        Projector-to-camera homography matrix.

    Returns
    -------
    points_cam : same shape as input, float64
        XY coordinates in camera pixels.
    """
    pts = np.asarray(points_proj, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 2)

    # Homogeneous coordinates.
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([pts, ones])  # (N, 3)
    transformed = (homography @ homogeneous.T).T  # (N, 3)

    # Normalize by w.
    w = transformed[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1.0, w)
    result = transformed[:, :2] / w

    if single:
        return result.ravel()
    return result


def projector_to_camera_mm(
    points_proj: np.ndarray,
    homography: np.ndarray,
    pixel_to_mm: float,
) -> np.ndarray:
    """Transform projector points to camera-space millimetres.

    Parameters
    ----------
    points_proj : (N, 2) or (2,) float
        XY in projector/texture pixels.
    homography : (3, 3) float
        Projector-to-camera homography.
    pixel_to_mm : float
        Camera-space pixels-to-mm conversion factor.

    Returns
    -------
    points_mm : same shape, float64
        XY in camera-space millimetres.
    """
    cam_px = projector_to_camera_px(points_proj, homography)
    return cam_px * pixel_to_mm


def projector_px_to_mm(
    value_px: np.ndarray,
    pixels_per_mm_projector: float,
) -> np.ndarray:
    """Convert projector/texture pixel values to millimetres.

    Parameters
    ----------
    value_px : array-like
        Values in projector/texture pixels (358x358 space).
    pixels_per_mm_projector : float
        Projector-space calibration (pixels per mm).

    Returns
    -------
    value_mm : same shape, float64
    """
    return np.asarray(value_px, dtype=np.float64) / pixels_per_mm_projector


def visual_angle_deg(
    radius_mm: np.ndarray,
    z_eff_mm: float,
) -> np.ndarray:
    """Compute visual angle subtended by a stimulus of given radius.

    Uses the standard formula: θ = 2 * arctan(radius / z_eff).

    Parameters
    ----------
    radius_mm : array-like
        Stimulus radius in millimetres.
    z_eff_mm : float
        Effective viewing distance through media (accounts for refraction).

    Returns
    -------
    angle_deg : same shape, float64
    """
    r = np.asarray(radius_mm, dtype=np.float64)
    return np.degrees(2.0 * np.arctan(r / z_eff_mm))


def resolve_concentric_center_mm(
    root: Any,
    step_params: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    """Resolve the concentric grating center in camera-space mm.

    Tries (in order):
    1. ``center_x_mm`` / ``center_y_mm`` directly in step_params (pre-transformed).
    2. Projector-space center from step_params + homography transform.
    3. Arena center from calibration (fallback: assumes grating is centered on dish).

    Returns (x_mm, y_mm) or None if no center can be resolved.
    """
    # 1. Already in mm.
    cx_mm = step_params.get("center_x_mm")
    cy_mm = step_params.get("center_y_mm")
    if cx_mm is not None and cy_mm is not None:
        return (float(cx_mm), float(cy_mm))

    cal = load_calibration_transform(root)

    # 2. Projector-space center + homography.
    cx_proj = step_params.get("center_x_px") or step_params.get("center_x_texture_px")
    cy_proj = step_params.get("center_y_px") or step_params.get("center_y_texture_px")
    if cx_proj is not None and cy_proj is not None and cal["homography"] is not None and cal["pixel_to_mm"] is not None:
        pt_mm = projector_to_camera_mm(
            np.array([float(cx_proj), float(cy_proj)]),
            cal["homography"],
            cal["pixel_to_mm"],
        )
        return (float(pt_mm[0]), float(pt_mm[1]))

    # 3. Arena center fallback.
    if cal["arena_center_px"] is not None and cal["pixel_to_mm"] is not None:
        cx_cam, cy_cam = cal["arena_center_px"]
        return (cx_cam * cal["pixel_to_mm"], cy_cam * cal["pixel_to_mm"])

    return None


__all__ = [
    "load_calibration_transform",
    "projector_to_camera_px",
    "projector_to_camera_mm",
    "projector_px_to_mm",
    "visual_angle_deg",
    "resolve_concentric_center_mm",
]
