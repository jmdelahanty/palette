"""Historical calibration lookup plus direction-neutral homography math.

The lookup helpers in this module are legacy compatibility surfaces.  They do
not prove a transform direction, selected camera, or reference extent and must
not authorize a future-normal coordinate publication.  Canonical callers load
a typed directed-transform record and pass its already validated numerical
matrix to :func:`apply_homography_xy`.

The historical ``projector_to_camera_*`` names remain for explicit legacy
inspection callers.  Their names describe the caller's assertion; they are not
coordinate authority.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


def _safe_get(group: Any, key: str) -> Any:
    getter = getattr(group, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            return None
    try:
        return group[key]
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_group_keys(group: Any) -> Iterable[str]:
    keys_fn = getattr(group, "group_keys", None)
    if callable(keys_fn):
        try:
            yield from (str(key) for key in keys_fn())
            return
        except Exception:
            pass
    keys_fn = getattr(group, "keys", None)
    if callable(keys_fn):
        try:
            yield from (str(key) for key in keys_fn())
        except Exception:
            return


def _read_homography(group: Any) -> Optional[np.ndarray]:
    if group is None or "homography_matrix" not in group:
        return None
    try:
        matrix = np.asarray(group["homography_matrix"][:], dtype=np.float64)
    except Exception:
        return None
    if matrix.shape != (3, 3):
        return None
    return matrix


def _merge_calibration_group(result: Dict[str, Any], group: Any) -> None:
    if group is None:
        return

    homography = _read_homography(group)
    if result["homography"] is None and homography is not None:
        result["homography"] = homography

    attrs = group.attrs if hasattr(group, "attrs") else {}
    if result["pixel_to_mm"] is None:
        pixel_to_mm = _as_float(attrs.get("pixel_to_mm"))
        if pixel_to_mm is not None:
            result["pixel_to_mm"] = pixel_to_mm
        else:
            pixels_per_mm = (
                _as_float(attrs.get("pixels_per_mm_camera"))
                or _as_float(attrs.get("pixels_per_mm"))
            )
            if pixels_per_mm is not None and pixels_per_mm > 0:
                result["pixel_to_mm"] = 1.0 / pixels_per_mm

    if result["pixels_per_mm_projector"] is None:
        result["pixels_per_mm_projector"] = _as_float(attrs.get("pixels_per_mm_projector"))

    if result["z_eff_mm"] is None:
        z_eff = _as_float(attrs.get("z_eff_mm"))
        if z_eff is not None and z_eff > 0:
            result["z_eff_mm"] = z_eff

    if result["arena_center_px"] is None:
        cx = _as_float(attrs.get("arena_center_x_px"))
        cy = _as_float(attrs.get("arena_center_y_px"))
        if cx is not None and cy is not None:
            result["arena_center_px"] = (cx, cy)


def _merge_stimulus_run_calibration(
    result: Dict[str, Any],
    root: Any,
    stimulus_run: Optional[str],
) -> None:
    analysis = _safe_get(root, "analysis")
    stim_parent = _safe_get(analysis, "stimulus_runs") if analysis is not None else None
    if stim_parent is None:
        return

    run_name = stimulus_run
    if not run_name:
        attrs = stim_parent.attrs if hasattr(stim_parent, "attrs") else {}
        latest = attrs.get("latest")
        if latest is not None:
            run_name = str(latest)
    if not run_name:
        return

    stim_group = _safe_get(stim_parent, run_name)
    calib_group = _safe_get(stim_group, "calibration") if stim_group is not None else None
    if calib_group is None:
        return

    for camera_id in _iter_group_keys(calib_group):
        cam_group = _safe_get(calib_group, camera_id)
        if cam_group is not None:
            _merge_calibration_group(result, cam_group)


def load_calibration_transform(
    root: Any,
    stimulus_run: Optional[str] = None,
) -> Dict[str, Any]:
    """Load calibration from canonical and stimulus-run Zarr calibration groups.

    Import-time normalization writes machine-readable fields to
    ``analysis/calibration``. Older archives may only have root ``calibration`` or
    per-stimulus-run ``analysis/stimulus_runs/<run>/calibration/<camera_id>``;
    those are treated as conservative fallbacks.

    Returns a dict with keys:
    - ``homography``: 3x3 float64 array (projector -> camera), or None
    - ``pixel_to_mm``: float (camera millimetres per pixel), or None
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

    analysis = _safe_get(root, "analysis")
    _merge_calibration_group(result, _safe_get(analysis, "calibration") if analysis is not None else None)
    _merge_calibration_group(result, _safe_get(root, "calibration"))
    _merge_stimulus_run_calibration(result, root, stimulus_run)

    return result


def apply_homography_xy(
    points_xy: np.ndarray,
    matrix: np.ndarray,
    *,
    denominator_epsilon: float = 1e-12,
) -> np.ndarray:
    """Apply one already direction-validated 3x3 homography to XY points.

    This function deliberately has no source/destination space names.  Those
    semantics belong to the persisted directed-transform authority used by the
    caller.  Malformed inputs and points whose homogeneous denominator is zero
    fail closed instead of being coerced to a plausible finite coordinate.

    Parameters
    ----------
    points_xy : (N, 2) or (2,) float
        XY coordinates in the transform record's declared source space.
    matrix : (3, 3) float
        Homography payload from that exact directed-transform record.
    denominator_epsilon : float
        Strict positive absolute threshold below which homogeneous ``w`` is
        treated as undefined.

    Returns
    -------
    transformed_xy : same shape as input, float64
        XY coordinates in the transform record's declared destination space.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        if pts.shape != (2,):
            raise ValueError("A single homography point must have shape (2,).")
        pts = pts.reshape(1, 2)
    elif pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Homography points must have shape (N, 2) or (2,).")

    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError("Homography matrix must be one finite 3x3 array.")
    if (
        isinstance(denominator_epsilon, (bool, np.bool_))
        or not np.isscalar(denominator_epsilon)
        or not np.isfinite(float(denominator_epsilon))
        or float(denominator_epsilon) <= 0.0
    ):
        raise ValueError("denominator_epsilon must be one finite positive scalar.")

    if pts.size == 0:
        return np.empty_like(pts, dtype=np.float64)
    if not np.isfinite(pts).all():
        raise ValueError("Homography points must be finite; select valid rows first.")

    # Homogeneous coordinates.
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([pts, ones])  # (N, 3)
    transformed = (transform @ homogeneous.T).T  # (N, 3)

    # Normalize by w.
    w = transformed[:, 2:3]
    if not np.isfinite(transformed).all() or np.any(
        np.abs(w) <= float(denominator_epsilon)
    ):
        raise ValueError(
            "Homography maps at least one point to an undefined point at infinity."
        )
    result = transformed[:, :2] / w

    if single:
        return result.ravel()
    return result


def projector_to_camera_px(
    points_proj: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    """Apply a caller-asserted projector-to-camera homography.

    This is a historical compatibility wrapper.  The function name does not
    validate direction or calibration lineage; future-normal callers must load
    those semantics from a typed directed-transform record first.
    """

    return apply_homography_xy(points_proj, homography)


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
    stimulus_run: Optional[str] = None,
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

    cal = load_calibration_transform(root, stimulus_run=stimulus_run)

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
    "apply_homography_xy",
    "load_calibration_transform",
    "projector_to_camera_px",
    "projector_to_camera_mm",
    "projector_px_to_mm",
    "visual_angle_deg",
    "resolve_concentric_center_mm",
]
