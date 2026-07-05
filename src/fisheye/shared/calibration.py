from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:  # Optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class RunCalibration:
    texture_to_camera_scale: float = 1.0
    pixels_per_mm_projector: Optional[float] = None
    pixels_per_mm_camera: Optional[float] = None
    texture_width_px: Optional[float] = None
    camera_width_px: Optional[float] = None
    homography_matrix: Optional[np.ndarray] = None
    source: str = "default"


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (np.generic,)):
        try:
            return float(value.item())
        except Exception:  # pragma: no cover
            return None
    return None


def _cam_keys(group) -> Sequence[str]:
    keys_fn = getattr(group, "group_keys", None)
    try:
        keys = list(keys_fn()) if callable(keys_fn) else []
    except Exception:  # pragma: no cover
        keys = []
    if not keys:
        try:
            keys = list(group.keys())
        except Exception:  # pragma: no cover
            keys = []
    return [key for key in keys if isinstance(key, str)]


def load_run_calibration(root, stimulus_run: Optional[str]) -> RunCalibration:
    """Load run-level calibration preferring analysis/stimulus_runs/<run>/calibration."""
    if not stimulus_run:
        return RunCalibration()

    try:
        analysis = root["analysis"]
        stim_parent = analysis["stimulus_runs"]
        stim_group = stim_parent[stimulus_run]
    except Exception:
        return RunCalibration()

    calibration = _load_from_calibration_group(stim_group)
    if calibration:
        return calibration

    attr_calibration = _load_from_coordinate_attr(stim_group)
    if attr_calibration:
        return attr_calibration

    return RunCalibration()


def _load_from_calibration_group(stim_group) -> Optional[RunCalibration]:
    calib_group = stim_group.get("calibration")
    if calib_group is None:
        return None

    camera_ids = _cam_keys(calib_group)
    for cam_id in camera_ids:
        cam_group = calib_group.get(cam_id)
        if cam_group is None:
            continue

        ppm_proj = _to_float(cam_group.attrs.get("pixels_per_mm_projector"))
        ppm_cam = _to_float(cam_group.attrs.get("pixels_per_mm_camera"))
        scale = None
        if ppm_proj and ppm_cam and ppm_proj != 0:
            scale = ppm_cam / ppm_proj

        texture_width = _to_float(cam_group.attrs.get("texture_width_px"))
        camera_width = _to_float(cam_group.attrs.get("camera_width_px"))

        homography = None
        if "homography_matrix" in cam_group:
            try:
                homography = np.asarray(cam_group["homography_matrix"][:], dtype=np.float64)
            except Exception:  # pragma: no cover
                homography = None
        elif "homography_matrix_yml" in cam_group.attrs and yaml is not None:
            try:
                yml_str = cam_group.attrs["homography_matrix_yml"]
                if isinstance(yml_str, bytes):
                    yml_str = yml_str.decode("utf-8", "ignore")
                data = yaml.safe_load(yml_str)
                if isinstance(data, dict):
                    matrix = data.get("homography_matrix") or data.get("data") or data
                    if isinstance(matrix, dict) and "data" in matrix:
                        matrix = matrix["data"]
                    if isinstance(matrix, (list, tuple)):
                        arr = np.asarray(matrix, dtype=np.float64)
                        homography = arr.reshape(3, 3)
                    else:
                        homography = None
            except Exception:  # pragma: no cover
                homography = None

        if scale is None:
            scale = _derive_scale_from_attr(stim_group)

        return RunCalibration(
            texture_to_camera_scale=scale if scale is not None else 1.0,
            pixels_per_mm_projector=ppm_proj,
            pixels_per_mm_camera=ppm_cam,
            texture_width_px=texture_width,
            camera_width_px=camera_width,
            homography_matrix=homography,
            source=f"calibration/{cam_id}",
        )

    return None


def _load_from_coordinate_attr(stim_group) -> Optional[RunCalibration]:
    attr = stim_group.attrs.get("coordinate_transform")
    parsed: Optional[dict] = None
    if isinstance(attr, (bytes, bytearray)):
        try:
            attr = attr.decode("utf-8")
        except Exception:
            attr = None
    if isinstance(attr, str):
        try:
            parsed = json.loads(attr)
        except json.JSONDecodeError:
            parsed = None
    elif isinstance(attr, dict):
        parsed = attr

    if not isinstance(parsed, dict):
        return None

    scale = _to_float(parsed.get("texture_to_camera_scale"))
    tex_dims = parsed.get("texture_dimensions")
    cam_dims = parsed.get("camera_dimensions")
    texture_width = None
    camera_width = None
    if isinstance(tex_dims, (list, tuple)) and tex_dims:
        texture_width = _to_float(tex_dims[0])
    if isinstance(cam_dims, (list, tuple)) and cam_dims:
        camera_width = _to_float(cam_dims[0])

    if scale is None:
        return None

    return RunCalibration(
        texture_to_camera_scale=scale,
        texture_width_px=texture_width,
        camera_width_px=camera_width,
        source="coordinate_transform",
    )


def _derive_scale_from_attr(stim_group) -> Optional[float]:
    attr = stim_group.attrs.get("coordinate_transform")
    if isinstance(attr, (bytes, bytearray)):
        try:
            attr = attr.decode("utf-8")
        except Exception:
            attr = None
    if isinstance(attr, str):
        try:
            attr = json.loads(attr)
        except json.JSONDecodeError:
            attr = None
    if isinstance(attr, dict):
        value = attr.get("texture_to_camera_scale")
        return _to_float(value)
    return None
