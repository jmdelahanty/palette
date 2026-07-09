#!/usr/bin/env python3
"""
Import stimulus H5 data into a Palette detection Zarr archive.

This script mirrors the functionality of ``create_analysis_h5.py`` but writes the
output directly inside the Zarr hierarchy under ``analysis/stimulus_runs``.

It copies frame metadata (with interpolation), chaser states, events, protocol
snapshots, and calibration information so downstream tooling no longer relies on
a separate analysis H5.

Data Architecture & Design Philosophy
--------------------------------------

**H5 File = Immutable Source of Truth**
    The source H5 file is NEVER modified. It is opened read-only and serves as the
    permanent record of raw experimental data, including:
    - Raw stimulus frame metadata (may contain gaps/duplicates)
    - Raw chaser tracking states (may have missing frames)
    - Events, protocol snapshots, calibration data

**Zarr Archive = Clean Working Copy**
    Each import creates a new timestamped run under ``analysis/stimulus_runs/``:

    .. code-block::

        analysis/stimulus_runs/
            └── stimulus_20251107_160756/
                ├── video_metadata/
                │   └── frame_metadata/          # Interpolated, contiguous
                ├── tracking_data/
                │   ├── chaser_states/           # Interpolated, contiguous
                │   └── chaser_interpolation_mask/  # Boolean: True=original, False=interpolated
                ├── frame_alignment/
                └── ...

**Interpolation Strategy**
    Gaps in stimulus metadata and chaser states are automatically filled via
    linear/nearest-neighbor interpolation to create **clean, contiguous datasets**
    for downstream analysis. This means:

    - Downstream tools can iterate frame-by-frame without gap handling
    - Camera frames align 1:1 with metadata indices
    - Chaser positions exist for every stimulus frame

    **Original data is preserved** via boolean masks:
    - ``interpolation_mask`` (metadata level)
    - ``chaser_interpolation_mask`` (chaser_states level)

    These masks are ``True`` for original H5 rows and ``False`` for synthesized rows.

**The "Happy Path"**
    Most analysis tools should use the interpolated data directly. Filtering back to
    original-only data is the exception, not the rule. The masks exist for:
    - Debugging alignment issues
    - Quality assessment (how much data was interpolated?)
    - Specialized analyses requiring ground-truth measurements only

**Re-import Safety**
    Each ``import_stimulus_to_zarr`` run creates an **independent, timestamped group**.
    There is no risk of re-interpolating already-interpolated data because each run
    starts fresh from the immutable H5 source. Previous runs remain untouched.

    The ``latest`` attribute on ``analysis/stimulus_runs`` points to the most recent
    import for convenience.

**Workflow**
    Typical usage pattern:

    1. Create Zarr archive with detections/tracking (``palette detect``, etc.)
    2. Import stimulus data: ``python -m fisheye.analysis.import_stimulus_to_zarr <zarr_path>``
    3. Analysis tools read from ``analysis/stimulus_runs/latest/`` or specific run

    All existing Zarr groups (``detect_runs``, ``keypoints_runs``, etc.) are preserved.
    The import is purely additive.
"""

from __future__ import annotations

import argparse
import json
import re
from hashlib import sha256
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from numpy.lib import recfunctions as rfn
import yaml
import zarr
from rich.console import Console

from .calibration_manager import CalibrationManager
from .chaser_state_interpolator import (
    analyze_frame_gaps,
    interpolate_metadata,
    interpolate_run,
    pick_chunks,
    store_array,
    write_columnar_dataset,
)
from fisheye.shared.citrus_enums import (
    EVENT_STEP_END,
    EVENT_STEP_START,
    EXPERIMENT_EVENT_TYPE,
    STIMULUS_MODE,
    STIMULUS_MODE_NAME_TO_ID,
)
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started, require_runs_parent


def _log(console: Optional[Console], message: str) -> None:
    if console is not None:
        console.log(message)


def _normalize_attr_value(value):
    """Convert HDF5 attribute values into JSON-serializable Python types."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


_json_safe_value = json_attr_safe


def _json_dumps_safe(value: Any) -> str:
    return strict_json_dumps(value, separators=(",", ": "))


def _update_attrs_json_safe(group: zarr.Group, attrs: Dict[str, Any]) -> None:
    """Update attrs, omitting unknown ``None`` values to keep metadata compact."""

    for key, value in attrs.items():
        safe = _json_safe_value(value)
        if safe is None:
            continue
        group.attrs[str(key)] = safe


def _collect_attrs(h5_obj: h5py.Group | h5py.Dataset) -> Dict[str, object]:
    """Collect and normalize attributes from an HDF5 object."""
    return {name: _normalize_attr_value(val) for name, val in h5_obj.attrs.items()}


def _parse_json_payload(raw: object) -> Optional[Dict[str, object]]:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    if isinstance(raw, dict):
        return raw
    return None


def _decode_h5_text(raw: object) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", "ignore")
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, str):
        return raw
    return None


def _parse_homography_matrix_yml(raw: object) -> Optional[np.ndarray]:
    """Parse Citrus/OpenCV homography YAML into a numeric 3x3 matrix."""
    text = _decode_h5_text(raw)
    if not text:
        return None

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%YAML") or (stripped == "---" and not cleaned_lines):
            continue
        cleaned_lines.append(line)

    try:
        parsed = yaml.safe_load("\n".join(cleaned_lines)) or {}
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None

    matrix_meta = parsed.get("homography_matrix")
    if isinstance(matrix_meta, dict):
        rows = int(matrix_meta.get("rows", 0) or 0)
        cols = int(matrix_meta.get("cols", 0) or 0)
        data = matrix_meta.get("data")
        if rows == 3 and cols == 3 and isinstance(data, (list, tuple)):
            return np.asarray(data, dtype=np.float64).reshape(3, 3)

    data = parsed.get("data")
    if isinstance(data, (list, tuple)) and len(data) == 9:
        return np.asarray(data, dtype=np.float64).reshape(3, 3)
    return None


def _find_default_h5_for_zarr_path(zarr_path: Path) -> Optional[Path]:
    """Locate a likely stimulus H5 alongside a zarr path without opening zarr."""

    parent_dir = zarr_path.parent
    if not parent_dir.exists():
        return None

    zarr_stem = zarr_path.name
    if zarr_stem.endswith(".zarr"):
        zarr_stem = zarr_stem[:-5]

    candidates = sorted(parent_dir.glob("*.h5"))
    if not candidates:
        return None

    exact = [path for path in candidates if path.stem == zarr_stem]
    if exact:
        return exact[0]

    contains = [path for path in candidates if zarr_stem in path.stem]
    if len(contains) == 1:
        return contains[0]

    try:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    except Exception:
        return candidates[0]


def _first_camera_config(arena_config: Dict[str, object]) -> Dict[str, object]:
    camera_configs = arena_config.get("camera_calibrations")
    if not isinstance(camera_configs, list) or not camera_configs:
        return {}
    active_camera_id = arena_config.get("active_camera_id")
    for item in camera_configs:
        if isinstance(item, dict) and str(item.get("camera_id")) == str(active_camera_id):
            return item
    first = camera_configs[0]
    return first if isinstance(first, dict) else {}


def _set_optional_float_attr(group: zarr.Group, name: str, value: object) -> None:
    if value is None:
        return
    try:
        group.attrs[name] = float(value)
    except (TypeError, ValueError):
        return


def _set_optional_int_attr(group: zarr.Group, name: str, value: object) -> None:
    if value is None:
        return
    try:
        group.attrs[name] = int(value)
    except (TypeError, ValueError):
        return


def _write_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.create_array(name, data=data, chunks=data.shape, overwrite=True)


def _copy_h5_attrs_to_zarr_attrs(src: h5py.Group | h5py.Dataset, dst: zarr.Group) -> None:
    for attr_name, attr_value in src.attrs.items():
        dst.attrs[attr_name] = _normalize_attr_value(attr_value)


def _copy_h5_dataset_to_zarr_mirror(src: h5py.Dataset, dst: zarr.Group, name: str) -> None:
    data = src[()]
    if isinstance(data, bytes):
        dst.attrs[name] = data.decode("utf-8", "ignore")
    elif np.isscalar(data) or (isinstance(data, np.ndarray) and data.ndim == 0):
        dst.attrs[name] = data.item() if hasattr(data, "item") else data
    else:
        arr = np.asarray(data)
        if name in dst:
            del dst[name]
        dst.create_array(
            name,
            data=arr,
            chunks=arr.shape,
            overwrite=True,
        )


def _copy_h5_group_to_zarr_mirror(src: h5py.Group, dst: zarr.Group) -> None:
    """Mirror an H5 group into a Zarr group for archival stimulus metadata.

    Existing stimulus imports stored scalar H5 datasets as Zarr attributes and
    non-scalar datasets as arrays. Preserve that contract recursively so newer
    nested calibration snapshots do not fail when a child is a group.
    """

    _copy_h5_attrs_to_zarr_attrs(src, dst)
    source_children = set(src.keys())
    existing_children = set(getattr(dst, "group_keys", lambda: [])())
    existing_children.update(getattr(dst, "array_keys", lambda: [])())
    for leftover in existing_children - source_children:
        del dst[leftover]

    for child_name, child in src.items():
        if isinstance(child, h5py.Group):
            if child_name in dst and not isinstance(dst[child_name], zarr.Group):
                del dst[child_name]
            child_dst = dst.require_group(child_name)
            _copy_h5_group_to_zarr_mirror(child, child_dst)
        elif isinstance(child, h5py.Dataset):
            _copy_h5_dataset_to_zarr_mirror(child, dst, child_name)


def _materialize_analysis_calibration(
    root: zarr.Group,
    h5: h5py.File,
    *,
    source_h5: Path,
    run_name: str,
    console: Optional[Console],
) -> None:
    """Normalize H5 calibration snapshot into analysis/calibration.

    The stimulus run still keeps the raw/mirrored H5 snapshot. This canonical
    group is the machine-readable analysis surface used by downstream metrics.
    """
    if "/calibration_snapshot" not in h5:
        return

    calib_src = h5["/calibration_snapshot"]
    arena_config = _read_h5_arena_config(h5) or {}
    camera_config = _first_camera_config(arena_config)

    analysis = root.require_group("analysis")
    calib_dst = analysis.require_group("calibration")
    calib_dst.attrs["schema_version"] = 1
    calib_dst.attrs["source"] = "h5_calibration_snapshot"
    calib_dst.attrs["source_h5"] = str(source_h5.resolve())
    calib_dst.attrs["source_stimulus_run"] = run_name

    active_camera_id = arena_config.get("active_camera_id") or camera_config.get("camera_id")
    if active_camera_id is not None:
        calib_dst.attrs["active_camera_id"] = str(active_camera_id)
        calib_dst.attrs["primary_camera_id"] = str(active_camera_id)

    ppm_camera = camera_config.get("pixels_per_mm_camera")
    if ppm_camera is None and isinstance(active_camera_id, (str, int)) and str(active_camera_id) in calib_src:
        ppm_camera = calib_src[str(active_camera_id)].attrs.get("pixels_per_mm_camera")
    _set_optional_float_attr(calib_dst, "pixels_per_mm_camera", ppm_camera)
    if ppm_camera is not None:
        try:
            ppm_camera_f = float(ppm_camera)
            if ppm_camera_f > 0:
                calib_dst.attrs["pixel_to_mm"] = 1.0 / ppm_camera_f
        except (TypeError, ValueError):
            pass

    ppm_projector = camera_config.get("pixels_per_mm_projector")
    if ppm_projector is None and isinstance(active_camera_id, (str, int)) and str(active_camera_id) in calib_src:
        ppm_projector = calib_src[str(active_camera_id)].attrs.get("pixels_per_mm_projector")
    _set_optional_float_attr(calib_dst, "pixels_per_mm_projector", ppm_projector)

    _set_optional_float_attr(calib_dst, "real_world_ref_mm", camera_config.get("real_world_ref_mm"))
    _set_optional_int_attr(calib_dst, "native_width_px", camera_config.get("native_width_px"))
    _set_optional_int_attr(calib_dst, "native_height_px", camera_config.get("native_height_px"))

    # Preserve Citrus arena config values with explicit coordinate-space names.
    for key in (
        "experimental_area_center_x_px",
        "experimental_area_center_y_px",
        "experimental_area_radius_px",
        "experimental_area_radius_mm",
        "experimental_area_width_px",
        "experimental_area_height_px",
        "sub_arena_x_px",
        "sub_arena_y_px",
        "sub_arena_width_px",
        "sub_arena_height_px",
        "sub_arena_width_mm",
        "sub_arena_height_mm",
        "calculated_z_eff_mm",
    ):
        _set_optional_float_attr(calib_dst, key, arena_config.get(key))
    if "experimental_area_shape" in arena_config:
        calib_dst.attrs["experimental_area_shape"] = str(arena_config["experimental_area_shape"])

    z_eff = arena_config.get("calculated_z_eff_mm")
    try:
        z_eff_f = float(z_eff) if z_eff is not None else None
    except (TypeError, ValueError):
        z_eff_f = None
    if z_eff_f is not None and z_eff_f > 0:
        calib_dst.attrs["z_eff_mm"] = z_eff_f
        calib_dst.attrs["z_eff_source"] = "calibration_snapshot.arena_config_json.calculated_z_eff_mm"
    elif z_eff is not None:
        calib_dst.attrs["z_eff_status"] = "unusable_nonpositive"

    camera_ids = [
        key for key in calib_src.keys()
        if key != "arena_config_json" and isinstance(calib_src.get(key), h5py.Group)
    ]
    homography_written = False
    for camera_id in camera_ids:
        cam_src = calib_src[camera_id]
        if "homography_matrix_yml" not in cam_src:
            continue
        matrix = _parse_homography_matrix_yml(cam_src["homography_matrix_yml"][()])
        if matrix is None:
            continue
        _write_array(calib_dst, "homography_matrix", matrix)
        calib_dst.attrs["homography_source"] = f"calibration_snapshot/{camera_id}/homography_matrix_yml"
        homography_written = True
        break

    if not homography_written:
        calib_dst.attrs["homography_status"] = "missing_numeric_matrix"

    _log(console, "[green]✓ Normalized H5 calibration to analysis/calibration[/green]")


def _filter_camera_metadata(payload: Dict[str, object]) -> Dict[str, object]:
    filtered = dict(payload)
    for key in ("device_snmp_comm_read", "device_snmp_comm_write", "yolo"):
        filtered.pop(key, None)
    return filtered


def _read_h5_arena_config(h5: h5py.File) -> Optional[Dict[str, object]]:
    if "/calibration_snapshot/arena_config_json" not in h5:
        return None
    node = h5["/calibration_snapshot/arena_config_json"]
    raw = node[()] if isinstance(node, h5py.Dataset) else node.attrs.get("arena_config_json")
    payload = _parse_json_payload(raw)
    return payload if isinstance(payload, dict) else None


def _read_h5_recording_snapshot(h5: h5py.File) -> Optional[Dict[str, object]]:
    if "/recording_snapshot" not in h5:
        return None
    node = h5["/recording_snapshot"]
    if isinstance(node, h5py.Dataset):
        payload = _parse_json_payload(node[()])
        return payload if isinstance(payload, dict) else None
    for key in ("recording_snapshot_json", "recording_snapshot"):
        if key in node and isinstance(node[key], h5py.Dataset):
            payload = _parse_json_payload(node[key][()])
            return payload if isinstance(payload, dict) else None
        raw = node.attrs.get(key)
        payload = _parse_json_payload(raw)
        if isinstance(payload, dict):
            return payload
    return None


def _read_h5_group_attrs(h5: h5py.File, path: str) -> Optional[Dict[str, object]]:
    if path not in h5:
        return None
    group = h5[path]
    if not hasattr(group, "attrs"):
        return None
    attrs = _collect_attrs(group)
    return attrs or None


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _derive_experiment_setup(subject_meta: Dict[str, object]) -> Optional[Dict[str, object]]:
    subject_count = _parse_int(subject_meta.get("subject_count"))
    if subject_count is None or subject_count < 1:
        return None
    num_dishes = 1
    fish_per_dish = subject_count
    total_expected = num_dishes * fish_per_dish
    return {
        "num_dishes": num_dishes,
        "fish_per_dish": fish_per_dish,
        "total_expected_fish": total_expected,
        "setup_type": "single_dish" if num_dishes == 1 else "multi_dish",
        "source": "subject_metadata",
        "configured_at": datetime.now(timezone.utc).isoformat(),
        "subject_count": subject_count,
        "subject_type": subject_meta.get("subject_type"),
    }


def _read_h5_camera_metadata(h5: h5py.File) -> Optional[Dict[str, object]]:
    for path in ("/camera_metadata", "/device_metadata"):
        if path not in h5:
            continue
        node = h5[path]
        if isinstance(node, h5py.Dataset):
            payload = _parse_json_payload(node[()])
            if payload:
                return _filter_camera_metadata(payload)
            return None
        raw = None
        if "config_json" in node:
            raw = node["config_json"][()]
        if raw is None:
            raw = node.attrs.get("config_json") or node.attrs.get("camera_config_json")
        payload = _parse_json_payload(raw)
        if payload:
            return _filter_camera_metadata(payload)
        attrs = _read_h5_group_attrs(h5, path)
        if attrs:
            return _filter_camera_metadata(attrs)
    snapshot = _read_h5_recording_snapshot(h5)
    if not isinstance(snapshot, dict):
        return None
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        return None

    camera_id = None
    arena_config = _read_h5_arena_config(h5)
    if isinstance(arena_config, dict):
        camera_id = arena_config.get("active_camera_id")
    if camera_id is None:
        camera_id = _derive_camera_id(h5.attrs.get("ipc_source_name"))
    if camera_id is not None:
        camera_id = str(camera_id)
        if camera_id in cameras and isinstance(cameras[camera_id], dict):
            return _filter_camera_metadata(cameras[camera_id])
    if len(cameras) == 1:
        only_payload = next(iter(cameras.values()))
        if isinstance(only_payload, dict):
            return _filter_camera_metadata(only_payload)
    return None


def _camera_metadata_hash(payload: Dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _read_experimental_chamber_from_h5(h5: h5py.File) -> Optional[str]:
    if "experimental_chamber" in h5.attrs:
        value = _normalize_attr_value(h5.attrs.get("experimental_chamber"))
        if isinstance(value, str) and value:
            return value

    if "/calibration_snapshot/arena_config_json" in h5:
        node = h5["/calibration_snapshot/arena_config_json"]
        raw = node[()] if isinstance(node, h5py.Dataset) else node.attrs.get("arena_config_json")
        payload = _parse_json_payload(raw)
        if payload:
            value = payload.get("experimental_chamber")
            value = _normalize_attr_value(value)
            if isinstance(value, str) and value:
                return value
            value = payload.get("selected_dish_type_name")
            value = _normalize_attr_value(value)
            if isinstance(value, str) and value:
                return value
    return None


def _derive_camera_id(ipc_source_name: object) -> Optional[str]:
    if ipc_source_name is None:
        return None
    value = _normalize_attr_value(ipc_source_name)
    if value is None:
        return None
    text = str(value)
    match = re.search(r"cam_(\d+)", text)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", text)
    return digits[-1] if digits else None


def _resolve_stimulus_video_path(stimulus_h5: Path) -> Optional[Path]:
    """Return the rendered stimulus video path next to H5 when present."""
    rendered = stimulus_h5.with_suffix(".mp4")
    if rendered.exists() and rendered.is_file():
        return rendered.resolve()
    return None


def _read_h5_session_context(h5: h5py.File) -> Optional[Dict[str, object]]:
    root_attrs = h5.attrs
    keys = [
        "session_uuid",
        "session_start_iso8601_utc",
        "rig_id",
        "arena_id",
        "camera_id",
        "canvas_name",
        "protocol_name_from_definition",
        "loaded_protocol_filepath",
        "stimulus_output_width",
        "stimulus_output_height",
        "ipc_source_name",
        "active_ipc_source",
        "hostname",
        "software_version",
    ]
    context: Dict[str, object] = {}
    for key in keys:
        if key in root_attrs:
            context[key] = _normalize_attr_value(root_attrs.get(key))

    camera_id = context.get("camera_id")
    if not camera_id:
        derived = _derive_camera_id(context.get("ipc_source_name"))
        if derived:
            context["camera_id"] = derived
            context["camera_id_source"] = "ipc_source_name"

    return context or None


def _resolve_struct_field(array: np.ndarray, *candidates: str) -> str:
    names = array.dtype.names or ()
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(f"Structured array missing expected field (tried {', '.join(candidates)})")


def _build_camera_aligned_metadata(
    metadata: np.ndarray,
    *,
    camera_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse metadata so each camera frame keeps only its latest stimulus entry.

    Returns a tuple of (camera_aligned_metadata, source_indices).
    """
    camera_ids = np.asarray(metadata[camera_field], dtype=np.int64)
    last_index: Dict[int, int] = {}
    for idx, cam in enumerate(camera_ids):
        last_index[int(cam)] = idx  # overwrite so we keep the most recent occurrence

    if not last_index:
        return metadata[:0]

    sorted_cameras = sorted(last_index.keys())
    indices = np.fromiter((last_index[cam] for cam in sorted_cameras), dtype=np.int64)
    return metadata[indices], indices


def _compute_camera_alignment(
    metadata: np.ndarray,
    metadata_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Derive helper arrays to map between camera frame IDs and metadata indices.
    """
    camera_ids = metadata["triggering_camera_frame_id"].astype(np.int64)
    min_camera = int(camera_ids.min())
    max_camera = int(camera_ids.max())
    size = max_camera + 1

    camera_to_index = np.full(size, -1, dtype=np.int64)
    camera_mask = np.zeros(size, dtype=bool)

    for idx, cam in enumerate(camera_ids):
        if camera_to_index[cam] == -1:
            camera_to_index[cam] = idx
            camera_mask[cam] = bool(metadata_mask[idx])
        else:
            camera_mask[cam] &= bool(metadata_mask[idx])

    return {
        "camera_frame_offset": int(min_camera),
        "camera_to_metadata_index": camera_to_index,
        "camera_interpolation_mask": camera_mask,
    }


def _ensure_utf8_column(values: np.ndarray) -> np.ndarray:
    """
    Decode bytes/object string arrays to Unicode for stable UTF-8 storage.
    """
    dtype_kind = values.dtype.kind
    if dtype_kind == "S":
        decoded = np.char.decode(values, "utf-8", errors="ignore")
        return np.char.rstrip(decoded, "\x00")
    if dtype_kind == "U":
        return np.char.rstrip(values, "\x00")
    if dtype_kind == "O":
        cleaned = []
        for item in values:
            if isinstance(item, bytes):
                cleaned.append(item.decode("utf-8", errors="ignore").rstrip("\x00"))
            elif item is None:
                cleaned.append("")
            else:
                cleaned.append(str(item).rstrip("\x00"))
        return np.asarray(cleaned, dtype="U")
    return values


COLUMNAR_DATASETS = {"chaser_states", "bounding_boxes"}
POSITION_COORDINATE_GROUPS = ("tracking_data/chaser_states", "tracking_data/bounding_boxes")
POSITION_COORDINATE_ATTRS = (
    "coordinate_frame",
    "coordinate_origin",
    "position_fields",
    "x_axis_direction",
    "y_axis_direction",
)


def _build_legacy_texture_to_camera_transform(arena_config: Dict[str, Any]) -> Dict[str, Any]:
    coord_info: Dict[str, Any] = {
        "texture_dimensions": [358, 358],
        "camera_dimensions": [4512, 4512],
        "texture_to_camera_scale": 4512 / 358,
        "coordinate_note": "Legacy texture-to-camera scale for older texture-space stimulus coordinates.",
    }
    try:
        cam_calib = arena_config.get("camera_calibrations", [{}])[0]
        width = cam_calib.get("native_width_px")
        height = cam_calib.get("native_height_px")
        if width and height:
            coord_info["camera_dimensions"] = [int(width), int(height)]
            coord_info["texture_to_camera_scale"] = float(width) / 358.0
    except Exception:
        pass
    return coord_info


def _position_coordinate_metadata(run_group: zarr.Group) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []
    for relative_path in POSITION_COORDINATE_GROUPS:
        try:
            group = run_group[relative_path]
        except Exception:
            continue
        attrs = dict(group.attrs)
        coordinate_frame = attrs.get("coordinate_frame")
        if not coordinate_frame:
            continue
        entry: Dict[str, Any] = {"path": relative_path}
        for attr_name in POSITION_COORDINATE_ATTRS:
            value = attrs.get(attr_name)
            if value is not None:
                entry[attr_name] = value
        metadata.append(entry)
    return metadata


def _write_run_coordinate_metadata(run_group: zarr.Group, arena_config: Dict[str, Any]) -> None:
    """Write run-level coordinate metadata without overriding child group contracts."""

    coord_info = _build_legacy_texture_to_camera_transform(arena_config)
    position_groups = _position_coordinate_metadata(run_group)
    if position_groups:
        coord_info["scope"] = "legacy_texture_space_fallback"
        coord_info[
            "coordinate_note"
        ] = (
            "Legacy texture-to-camera scale retained for older texture-space consumers. "
            "Position-bearing child groups with coordinate_frame attrs define their own "
            "coordinate contracts and take precedence."
        )
        run_group.attrs["legacy_texture_to_camera_transform"] = json.dumps(coord_info, sort_keys=True)
        run_group.attrs[
            "coordinate_transform_status"
        ] = "suppressed_child_group_coordinate_metadata_authoritative"
        run_group.attrs["position_coordinate_groups"] = json.dumps(position_groups, sort_keys=True)
        if "coordinate_transform" in run_group.attrs:
            del run_group.attrs["coordinate_transform"]
        return

    coord_info["scope"] = "run_level_legacy_texture_space"
    run_group.attrs["coordinate_transform"] = json.dumps(coord_info, sort_keys=True)
    run_group.attrs["coordinate_transform_status"] = "legacy_run_level_texture_to_camera"


def _copy_h5_dataset(
    h5_group: h5py.Group,
    zarr_group: zarr.Group,
    name: str,
) -> None:
    """Copy a dataset from H5 into Zarr if present."""
    if name not in h5_group:
        return
    data = h5_group[name][:]
    # Skip empty datasets (common for experiments without tracking output).
    if data.size == 0 or (data.shape and data.shape[0] == 0):
        return
    if (
        name == "bounding_boxes"
        and data.dtype.names is not None
        and "centroid_x" not in data.dtype.names
    ):
        required_fields = {"x_min", "y_min", "width", "height"}
        if required_fields.issubset(data.dtype.names):
            centroid_x = np.asarray(data["x_min"] + (data["width"] * 0.5), dtype=np.float32)
            centroid_y = np.asarray(data["y_min"] + (data["height"] * 0.5), dtype=np.float32)
            data = rfn.append_fields(
                data,
                ("centroid_x", "centroid_y"),
                (centroid_x, centroid_y),
                usemask=False,
            )
    attrs = _collect_attrs(h5_group[name])
    if name in COLUMNAR_DATASETS and data.dtype.names:
        write_columnar_dataset(zarr_group, name, data, attrs)
    else:
        store_array(zarr_group, name, data, attrs)


def _copy_enums(h5: h5py.File, analysis_group: zarr.Group, console: Optional[Console]) -> None:
    """Copy /enums datasets from H5 into analysis/enums, converting to columnar format.

    Converts structured arrays [('id', 'i4'), ('name', 'S128')] into separate columnar arrays:
    - enums/{name}/id: int32 array
    - enums/{name}/name: variable-length UTF-8 string array

    This matches the storage pattern used for events and provides better TensorStore compatibility.
    """
    if "/enums" not in h5:
        _log(console, "[dim]/enums group not found in H5; skipping enum import.[/dim]")
        return

    enums_src = h5["/enums"]
    enums_dst = analysis_group.require_group("enums")

    copied = 0
    existing = set(enums_dst.group_keys())

    # Clean up stale enum groups
    for leftover in existing - set(enums_src.keys()):
        del enums_dst[leftover]

    for name in enums_src.keys():
        data = enums_src[name][:]  # Load structured array from H5

        # Validate structure
        if not data.dtype.names or 'id' not in data.dtype.names or 'name' not in data.dtype.names:
            _log(console, f"[yellow]⚠ Skipping malformed enum table '{name}': missing 'id' or 'name' fields[/yellow]")
            continue

        # Extract fields from structured array
        ids = np.asarray(data['id'], dtype=np.int32)  # Ensure int32

        # Convert byte strings to UTF-8, handling various encodings
        name_values = []
        for raw_name in data['name']:
            if isinstance(raw_name, bytes):
                decoded = raw_name.decode('utf-8', errors='ignore').rstrip('\x00')
            elif raw_name is None:
                decoded = ""
            else:
                decoded = str(raw_name).rstrip('\x00')
            name_values.append(decoded)
        names = np.asarray(name_values, dtype=str)  # Variable-length UTF-8

        # Create group for this enum table (columnar format)
        enum_group = enums_dst.require_group(name)

        # Store original H5 attributes on the group
        src_attrs = _collect_attrs(enums_src[name])
        if src_attrs:
            enum_group.attrs.update(src_attrs)

        # Mark as columnar format for compatibility
        enum_group.attrs['storage_layout'] = 'columnar'
        enum_group.attrs['field_names'] = ['id', 'name']

        # Store as separate arrays (columnar)
        store_array(enum_group, 'id', ids, {})
        store_array(enum_group, 'name', names, {})

        copied += 1

    if copied:
        _log(console, f"[green]✓ Imported {copied} enum tables into analysis/enums (columnar format)[/green]")


def _read_protocol_snapshot(h5: h5py.File) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Read Citrus protocol snapshot JSON from modern or legacy H5 paths."""

    proto_key = None
    if "/protocol_snapshot/protocol_definition_json" in h5:
        proto_key = "/protocol_snapshot/protocol_definition_json"
    elif "/protocol_snapshot/protocol_json" in h5:
        proto_key = "/protocol_snapshot/protocol_json"
    if proto_key is None:
        return None, None

    text = _decode_h5_text(h5[proto_key][()])
    if not text:
        return None, None
    payload = _parse_json_payload(text)
    return text, payload if isinstance(payload, dict) else None


def _protocol_steps_list(protocol: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(protocol, dict):
        return []
    steps = protocol.get("steps") or protocol.get("protocol_steps") or []
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]


def _flatten_protocol_params(step_params: Dict[str, Any]) -> Dict[str, Any]:
    nested = step_params.get("parameters")
    if isinstance(nested, dict):
        return dict(nested)
    return dict(step_params)


def _first_float(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(out):
            return out
    return None


def _first_bool(*values: Any) -> Optional[bool]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
    return None


def _structured_column(data: Optional[np.ndarray], *names: str) -> Optional[np.ndarray]:
    if data is None or data.dtype.names is None:
        return None
    for name in names:
        if name in data.dtype.names:
            return np.asarray(data[name])
    return None


def _decode_text_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").rstrip("\x00")
    if isinstance(value, np.generic):
        return _decode_text_scalar(value.item())
    return str(value).rstrip("\x00")


def _normalize_event_names(events_data: Optional[np.ndarray]) -> np.ndarray:
    names = _structured_column(events_data, "event_name", "name")
    if names is not None:
        return np.asarray([_decode_text_scalar(value) for value in names], dtype=object)

    event_type = _structured_column(events_data, "event_type_id", "event_type")
    if event_type is None:
        return np.asarray([], dtype=object)
    return np.asarray(
        [EXPERIMENT_EVENT_TYPE.get(int(value), f"UNKNOWN_{int(value)}") for value in event_type],
        dtype=object,
    )


def _normalize_stimulus_mode_id(value: Any) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = _decode_text_scalar(value)
    if text in STIMULUS_MODE_NAME_TO_ID:
        return int(STIMULUS_MODE_NAME_TO_ID[text])
    try:
        return int(text)
    except ValueError:
        return 0


def _infer_camera_fps(metadata: np.ndarray) -> Optional[float]:
    """Infer camera FPS from metadata timestamps and camera frame IDs when possible."""

    if metadata.dtype.names is None:
        return None
    try:
        camera_field = _resolve_struct_field(metadata, "triggering_camera_frame_id", "camera_frame_id")
    except ValueError:
        return None
    timestamp_field = None
    for candidate in ("timestamp_ns", "timestamp_relative_ns", "time_ns"):
        if candidate in metadata.dtype.names:
            timestamp_field = candidate
            break
    if timestamp_field is None or metadata.shape[0] < 2:
        return None

    camera = np.asarray(metadata[camera_field], dtype=np.float64)
    time_ns = np.asarray(metadata[timestamp_field], dtype=np.float64)
    camera_delta = camera[-1] - camera[0]
    time_delta_s = (time_ns[-1] - time_ns[0]) / 1e9
    if camera_delta <= 0 or time_delta_s <= 0:
        return None
    fps = camera_delta / time_delta_s
    return float(fps) if np.isfinite(fps) and fps > 0 else None


def _copy_h5_tree(src: h5py.Group, dst: zarr.Group) -> None:
    """Copy an H5 group tree into Zarr, preserving attrs and simple datasets."""

    for attr_name, attr_value in src.attrs.items():
        dst.attrs[attr_name] = _json_safe_value(_normalize_attr_value(attr_value))

    for name, node in src.items():
        if isinstance(node, h5py.Group):
            child = dst.require_group(name)
            _copy_h5_tree(node, child)
            continue

        value = node[()]
        if isinstance(value, bytes):
            dst.attrs[name] = value.decode("utf-8", errors="ignore")
            continue
        if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
            dst.attrs[name] = _json_safe_value(value.item() if hasattr(value, "item") else value)
            continue

        arr = np.asarray(value)
        if name in dst:
            del dst[name]
        dst.create_array(name, data=arr, chunks=arr.shape, overwrite=True)


def _delete_zarr_child_if_exists(group: zarr.Group, name: str) -> None:
    """Delete a child without trusting possibly stale consolidated metadata."""

    try:
        del group[name]
    except (KeyError, FileNotFoundError):
        return


def _copy_stimulus_coordinates(h5: h5py.File, run_group: zarr.Group, console: Optional[Console]) -> None:
    if "/stimulus_coordinates" not in h5:
        return
    _delete_zarr_child_if_exists(run_group, "stimulus_coordinates")
    dst = run_group.create_group("stimulus_coordinates")
    _copy_h5_tree(h5["/stimulus_coordinates"], dst)
    _log(console, "[green]✓ Copied H5 stimulus_coordinates into stimulus run[/green]")


def _projector_pixels_per_mm(h5: h5py.File, arena_config: Dict[str, Any]) -> Optional[float]:
    camera_config = _first_camera_config(arena_config)
    ppm = _first_float(camera_config.get("pixels_per_mm_projector"))
    if ppm is not None:
        return ppm
    active_camera_id = arena_config.get("active_camera_id") or camera_config.get("camera_id")
    if active_camera_id is not None and f"/calibration_snapshot/{active_camera_id}" in h5:
        return _first_float(h5[f"/calibration_snapshot/{active_camera_id}"].attrs.get("pixels_per_mm_projector"))
    return None


def _stimulus_coordinates_center(h5: h5py.File) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if "/stimulus_coordinates" not in h5:
        return None, None, None
    parent = h5["/stimulus_coordinates"]
    for arena_name in parent.keys():
        arena = parent[arena_name]
        if not isinstance(arena, h5py.Group):
            continue
        custom = arena.get("custom_coordinates")
        if custom is None or not hasattr(custom, "attrs"):
            continue
        cx = _first_float(custom.attrs.get("texture_center_x"), custom.attrs.get("texture_center_x_px"))
        cy = _first_float(custom.attrs.get("texture_center_y"), custom.attrs.get("texture_center_y_px"))
        if cx is not None and cy is not None:
            return cx, cy, f"stimulus_coordinates/{arena_name}/custom_coordinates.texture_center"
    return None, None, None


def _resolve_concentric_center_attrs(
    h5: h5py.File,
    flat_params: Dict[str, Any],
    arena_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve source-derived concentric grating center metadata for step attrs."""

    ppm_projector = _projector_pixels_per_mm(h5, arena_config)
    cx_mm = _first_float(flat_params.get("center_x_mm"))
    cy_mm = _first_float(flat_params.get("center_y_mm"))
    if cx_mm is not None and cy_mm is not None:
        return {
            "center_x_mm": cx_mm,
            "center_y_mm": cy_mm,
            "center_source": "protocol_parameters.center_mm",
            "center_coordinate_frame": "stimulus_mm",
        }

    cx_px = _first_float(flat_params.get("center_x_px"), flat_params.get("center_x_texture_px"))
    cy_px = _first_float(flat_params.get("center_y_px"), flat_params.get("center_y_texture_px"))
    center_source = "protocol_parameters.center_px" if cx_px is not None and cy_px is not None else ""
    center_frame = "arena_texture_px"

    if cx_px is None or cy_px is None:
        cx_px, cy_px, center_source = _stimulus_coordinates_center(h5)
        center_frame = "arena_texture_px"

    if cx_px is None or cy_px is None:
        cx_px = _first_float(
            arena_config.get("arena_region_center_in_canvas_x_px"),
            arena_config.get("experimental_area_center_x_px"),
        )
        cy_px = _first_float(
            arena_config.get("arena_region_center_in_canvas_y_px"),
            arena_config.get("experimental_area_center_y_px"),
        )
        if cx_px is not None and cy_px is not None:
            center_source = "calibration_snapshot.arena_config_json.experimental_area_center_px"
            center_frame = "projector_canvas_px"

    if cx_px is None or cy_px is None:
        width_px = _first_float(
            arena_config.get("arena_region_width_px"),
            arena_config.get("sub_arena_width_px"),
            arena_config.get("experimental_area_width_px"),
        )
        height_px = _first_float(
            arena_config.get("arena_region_height_px"),
            arena_config.get("sub_arena_height_px"),
            arena_config.get("experimental_area_height_px"),
        )
        if width_px is not None and height_px is not None:
            cx_px = width_px * 0.5
            cy_px = height_px * 0.5
            center_source = "calibration_snapshot.arena_config_json.arena_size_half"
            center_frame = "arena_texture_px"

    attrs: Dict[str, Any] = {}
    if cx_px is not None and cy_px is not None:
        attrs.update({
            "center_x_px": cx_px,
            "center_y_px": cy_px,
            "center_source": center_source or "unknown",
            "center_coordinate_frame": center_frame,
        })
        if ppm_projector is not None and ppm_projector > 0:
            attrs["center_x_mm"] = cx_px / ppm_projector
            attrs["center_y_mm"] = cy_px / ppm_projector
            attrs["center_mm_source"] = f"{attrs['center_source']} / pixels_per_mm_projector"
            attrs["center_mm_coordinate_frame"] = "projector_relative_mm"
    if ppm_projector is not None:
        attrs["pixels_per_mm_projector"] = ppm_projector
    return attrs


def _write_moving_grating_step_metadata(
    step_group: zarr.Group,
    step_params: Dict[str, Any],
    *,
    camera_to_projector_offset_deg: float = 0.0,
) -> None:
    flat = _flatten_protocol_params(step_params)
    orientation = _first_float(
        flat.get("orientation_degrees"),
        flat.get("angle_degrees"),
        flat.get("grating_orientation"),
    )
    speed_mm_s = _first_float(flat.get("grating_speed_mm_s"), flat.get("speed_mm_per_sec"), flat.get("speed_mm_s"))
    speed_pps = _first_float(flat.get("speed_pps"))
    spatial_mm = _first_float(flat.get("spatial_freq_cycles_per_mm"), flat.get("spatial_freq_cpmm"))
    spatial_px = _first_float(flat.get("spatial_freq_rpp"), flat.get("spatial_freq_cpp"))
    offset = float(camera_to_projector_offset_deg)
    direction_camera = ((orientation + offset) % 360.0) if orientation is not None else None
    has_configured_offset = abs(offset) > 1e-9
    attrs = {
        "metadata_schema_version": 1,
        "source": "protocol_snapshot",
        "orientation_degrees_authored": orientation,
        "grating_direction_camera_deg": direction_camera,
        "camera_to_projector_offset_deg": offset,
        "direction_mapping_source": (
            "protocol_orientation_degrees_plus_configured_offset"
            if has_configured_offset else "protocol_orientation_degrees_no_offset"
        ),
        "direction_mapping_status": (
            "configured_camera_offset" if has_configured_offset else "unvalidated_default_zero_offset"
        ),
        "direction_mapping_validated": False,
        "speed_mm_s": speed_mm_s,
        "speed_pps": speed_pps,
        "spatial_freq_cycles_per_mm": spatial_mm,
        "spatial_freq_rpp": spatial_px,
        "temporal_frequency_hz": abs(speed_mm_s * spatial_mm) if speed_mm_s is not None and spatial_mm is not None else None,
        "actual_rendered_temporal_frequency_hz": abs(speed_pps * spatial_px) if speed_pps is not None and spatial_px is not None else None,
        "duty_cycle": _first_float(flat.get("duty_cycle")),
    }
    group = step_group.require_group("moving_grating")
    _update_attrs_json_safe(group, attrs)


def _write_concentric_grating_step_metadata(
    h5: h5py.File,
    step_group: zarr.Group,
    step_params: Dict[str, Any],
    arena_config: Dict[str, Any],
) -> None:
    flat = _flatten_protocol_params(step_params)
    is_expanding = _first_bool(flat.get("is_expanding"))
    speed_mm_s = _first_float(flat.get("speed_mm_per_sec"), flat.get("grating_speed_mm_s"), flat.get("speed_mm_s"))
    speed_pps = _first_float(flat.get("speed_pps"))
    spatial_mm = _first_float(flat.get("spatial_freq_cycles_per_mm"), flat.get("spatial_freq_cpmm"))
    spatial_px = _first_float(flat.get("spatial_freq_rpp"), flat.get("spatial_freq_cpp"))
    attrs = {
        "metadata_schema_version": 1,
        "source": "protocol_snapshot",
        "stimulus_role": flat.get("stimulus_role", "unknown"),
        "radial_polarity_authored": (
            "expanding" if is_expanding is True else "contracting" if is_expanding is False else None
        ),
        "radial_sign_authored": (1 if is_expanding is True else -1 if is_expanding is False else None),
        "radial_polarity_source": "protocol_parameters.is_expanding" if is_expanding is not None else None,
        "radial_polarity_validated": False,
        "speed_mm_s": speed_mm_s,
        "speed_pps": speed_pps,
        "spatial_freq_cycles_per_mm": spatial_mm,
        "spatial_freq_rpp": spatial_px,
        "temporal_frequency_hz": abs(speed_mm_s * spatial_mm) if speed_mm_s is not None and spatial_mm is not None else None,
        "actual_rendered_temporal_frequency_hz": abs(speed_pps * spatial_px) if speed_pps is not None and spatial_px is not None else None,
        "duty_cycle": _first_float(flat.get("duty_cycle")),
        "target_radius_min_mm": _first_float(flat.get("target_radius_min_mm")),
        "target_radius_max_mm": _first_float(flat.get("target_radius_max_mm")),
        "target_radius_source": flat.get("target_radius_source"),
        "centering_success_fraction_threshold": _first_float(flat.get("centering_success_fraction_threshold")),
    }
    attrs.update(_resolve_concentric_center_attrs(h5, flat, arena_config))
    group = step_group.require_group("concentric_grating")
    _update_attrs_json_safe(group, attrs)


def _materialize_stimulus_steps(
    run_group: zarr.Group,
    *,
    h5: h5py.File,
    events_data: Optional[np.ndarray],
    protocol: Optional[Dict[str, Any]],
    arena_config: Dict[str, Any],
    metadata: np.ndarray,
    moving_grating_camera_offset_deg: float = 0.0,
    console: Optional[Console],
) -> None:
    """Write canonical source-derived step metadata under a stimulus run."""

    if events_data is None or events_data.dtype.names is None:
        return

    event_names = _normalize_event_names(events_data)
    step_indices_raw = _structured_column(events_data, "step_index", "current_step_index")
    camera_frames_raw = _structured_column(events_data, "camera_frame_id", "camera_frame_num", "triggering_camera_frame_id")
    stimulus_modes_raw = _structured_column(events_data, "stimulus_mode_id", "stimulus_mode")
    if event_names.size == 0 or step_indices_raw is None or camera_frames_raw is None:
        return

    step_indices = np.asarray(step_indices_raw, dtype=np.int32)
    camera_frames = np.asarray(camera_frames_raw, dtype=np.int64)
    stimulus_modes = (
        np.asarray([_normalize_stimulus_mode_id(value) for value in stimulus_modes_raw], dtype=np.int32)
        if stimulus_modes_raw is not None
        else np.zeros(step_indices.shape[0], dtype=np.int32)
    )

    starts: Dict[int, int] = {}
    ends: Dict[int, int] = {}
    modes: Dict[int, int] = {}
    for i, name in enumerate(event_names):
        step_index = int(step_indices[i])
        if str(name).strip() == EVENT_STEP_START:
            starts[step_index] = int(camera_frames[i])
            modes[step_index] = int(stimulus_modes[i])
        elif str(name).strip() == EVENT_STEP_END:
            ends[step_index] = int(camera_frames[i])

    if not starts:
        return

    _delete_zarr_child_if_exists(run_group, "steps")
    steps_group = run_group.create_group("steps")
    steps_group.attrs["metadata_schema_version"] = 1
    steps_group.attrs["source"] = "h5_events_and_protocol_snapshot"

    protocol_steps = _protocol_steps_list(protocol)
    inferred_fps = _infer_camera_fps(metadata)
    for step_index in sorted(starts):
        protocol_step = protocol_steps[step_index] if 0 <= step_index < len(protocol_steps) else {}
        step_params = {
            key: value for key, value in protocol_step.items()
            if key not in ("name", "step_name", "step_index")
        }
        step_name = str(protocol_step.get("name", protocol_step.get("step_name", f"step_{step_index}")))
        start_frame = int(starts[step_index])
        end_frame = int(ends.get(step_index, start_frame + 1))
        mode_id = int(modes.get(step_index, protocol_step.get("stimulus_mode_id", 0) or 0))
        mode_name = str(protocol_step.get("stimulus_mode_str") or STIMULUS_MODE.get(mode_id, f"UNKNOWN_{mode_id}"))
        duration_s = _first_float(protocol_step.get("duration_seconds"), protocol_step.get("duration_s"))
        if duration_s is None and inferred_fps is not None and inferred_fps > 0:
            duration_s = max(0.0, (end_frame - start_frame) / inferred_fps)

        step_group = steps_group.create_group(f"step_{step_index}")
        _update_attrs_json_safe(step_group, {
            "metadata_schema_version": 1,
            "step_index": step_index,
            "step_name": step_name,
            "stimulus_mode_id": mode_id,
            "stimulus_mode": mode_name,
            "start_camera_frame": start_frame,
            "end_camera_frame": end_frame,
            "duration_s": duration_s,
            "raw_protocol_params_json": _json_dumps_safe(step_params),
        })

        if mode_name == "MOVING_GRATING":
            _write_moving_grating_step_metadata(
                step_group,
                step_params,
                camera_to_projector_offset_deg=moving_grating_camera_offset_deg,
            )
        elif mode_name == "CONCENTRIC_GRATING":
            _write_concentric_grating_step_metadata(h5, step_group, step_params, arena_config)

    _log(console, f"[green]✓ Materialized {len(starts)} canonical stimulus step metadata groups[/green]")


def _open_zarr_group_unconsolidated(path: Path, *, mode: str) -> zarr.Group:
    """Open a mutable Zarr group while bypassing stale consolidated metadata."""

    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode, consolidated=False)


def _zarr_group_keys(group: zarr.Group) -> List[str]:
    keys_fn = getattr(group, "group_keys", None)
    if callable(keys_fn):
        try:
            return sorted(str(key) for key in keys_fn())
        except Exception:
            pass
    return sorted(str(key) for key in group.keys() if isinstance(group.get(key), zarr.Group))


def _local_zarr_child_group_names(parent_path: Path) -> List[str]:
    if not parent_path.exists():
        return []
    return sorted(
        child.name
        for child in parent_path.iterdir()
        if child.is_dir() and (child / "zarr.json").exists()
    )


def _resolve_existing_path(raw_path: object, *, bases: Iterable[Path]) -> Optional[Path]:
    if raw_path in (None, ""):
        return None
    path = Path(str(raw_path)).expanduser()
    candidates = [path] if path.is_absolute() else [base / path for base in bases] + [path]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            try:
                return candidate.resolve()
            except OSError:
                return candidate
    return None


def _recording_dir_for_zarr(zarr_path: Path) -> Optional[Path]:
    try:
        parent = zarr_path.resolve().parent
    except OSError:
        parent = zarr_path.parent
    if parent.name == "zarr":
        return parent.parent
    return None


def _source_h5_from_recording_dir(zarr_path: Path) -> Tuple[Optional[Path], str]:
    recording_dir = _recording_dir_for_zarr(zarr_path)
    if recording_dir is None:
        return None, "zarr is not under a recording/zarr directory"

    candidates: List[Path] = []
    raw_dir = recording_dir / "raw"
    for root in (raw_dir, recording_dir):
        if not root.exists():
            continue
        for suffix in ("*.h5", "*.hdf5"):
            candidates.extend(sorted(path for path in root.glob(suffix) if path.is_file()))

    unique: Dict[str, Path] = {}
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        unique[key] = candidate
    candidates = list(unique.values())
    if not candidates:
        return None, f"no H5 files found under {recording_dir}/raw or {recording_dir}"

    zarr_name = zarr_path.name
    recording_stem = None
    if zarr_name.endswith("_analysis.zarr"):
        recording_stem = zarr_name[: -len("_analysis.zarr")]
    elif zarr_name.endswith("_training.zarr"):
        recording_stem = zarr_name[: -len("_training.zarr")]
    if recording_stem:
        preferred = [path for path in candidates if path.stem == recording_stem]
        if len(preferred) == 1:
            return preferred[0], ""

    if len(candidates) == 1:
        return candidates[0], ""

    joined = ", ".join(str(path) for path in candidates[:5])
    suffix = "" if len(candidates) <= 5 else f", ... ({len(candidates)} total)"
    return None, f"ambiguous H5 candidates: {joined}{suffix}"


def _resolve_stimulus_backfill_h5(
    zarr_path: Path,
    run_group: zarr.Group,
    *,
    source_h5: Optional[Path],
) -> Tuple[Optional[Path], str]:
    bases: List[Path] = [zarr_path.parent]
    recording_dir = _recording_dir_for_zarr(zarr_path)
    if recording_dir is not None:
        bases.extend([recording_dir, recording_dir / "raw"])

    explicit = _resolve_existing_path(source_h5, bases=bases)
    if source_h5 is not None:
        return (explicit, "" if explicit is not None else f"explicit H5 path does not exist: {source_h5}")

    attr_path = _resolve_existing_path(run_group.attrs.get("source_h5"), bases=bases)
    if attr_path is not None:
        return attr_path, ""

    return _source_h5_from_recording_dir(zarr_path)


def _count_step_start_events(events_data: Optional[np.ndarray]) -> int:
    if events_data is None or events_data.dtype.names is None:
        return 0
    event_names = _normalize_event_names(events_data)
    return int(sum(1 for name in event_names if str(name).strip() == EVENT_STEP_START))


def _zarr_child_exists(run_group: zarr.Group, run_path: Path, name: str) -> bool:
    """Check child existence using both Zarr metadata and local file layout."""

    if name in run_group:
        return True
    return (run_path / name / "zarr.json").exists()


def _empty_frame_metadata() -> np.ndarray:
    return np.asarray([], dtype=np.dtype([]))


def backfill_stimulus_step_metadata(
    zarr_path: Path,
    *,
    stimulus_run: Optional[str] = None,
    source_h5: Optional[Path] = None,
    moving_grating_camera_offset_deg: float = 0.0,
    overwrite: bool = False,
    apply: bool = False,
    consolidate_metadata: bool = False,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
    """Backfill canonical stimulus step metadata into existing stimulus runs.

    This is intentionally narrower than a full stimulus re-import. It reads the
    immutable Citrus H5 snapshot associated with each existing stimulus run and
    materializes only:

    - ``analysis/stimulus_runs/<run>/steps/step_<i>``
    - ``analysis/stimulus_runs/<run>/stimulus_coordinates``
    - missing ``protocol_json`` attrs, when available

    Default mode is a dry-run. Pass ``apply=True`` to write.
    """

    zarr_path = Path(zarr_path).expanduser()
    mode = "a" if apply else "r"
    root = _open_zarr_group_unconsolidated(zarr_path, mode=mode)
    summary: Dict[str, Any] = {
        "zarr_path": str(zarr_path),
        "apply": bool(apply),
        "overwrite": bool(overwrite),
        "moving_grating_camera_offset_deg": float(moving_grating_camera_offset_deg),
        "runs_scanned": 0,
        "details": [],
    }

    def add_detail(status: str, **detail: Any) -> None:
        summary[status] = int(summary.get(status, 0)) + 1
        detail_out = {"status": status, **{key: _json_safe_value(value) for key, value in detail.items()}}
        summary["details"].append(detail_out)

    analysis = root.get("analysis")
    if analysis is None:
        add_detail("skipped_missing_analysis")
        return summary
    runs_parent = analysis.get("stimulus_runs")
    if runs_parent is None:
        add_detail("skipped_missing_stimulus_runs")
        return summary

    if stimulus_run is not None:
        run_names = [stimulus_run]
    else:
        runs_path = zarr_path / "analysis" / "stimulus_runs"
        run_names = sorted(set(_zarr_group_keys(runs_parent)) | set(_local_zarr_child_group_names(runs_path)))
    if not run_names:
        add_detail("skipped_no_stimulus_runs")
        return summary

    for run_name in run_names:
        summary["runs_scanned"] += 1
        run_path = zarr_path / "analysis" / "stimulus_runs" / str(run_name)
        try:
            run_group = runs_parent[run_name]
        except (KeyError, FileNotFoundError):
            if (run_path / "zarr.json").exists():
                run_group = _open_zarr_group_unconsolidated(run_path, mode=mode)
            else:
                add_detail("skipped_missing_run", run_name=run_name)
                continue
        if not isinstance(run_group, zarr.Group):
            add_detail("skipped_non_group_run", run_name=run_name)
            continue

        need_steps = bool(overwrite or not _zarr_child_exists(run_group, run_path, "steps"))
        need_coordinates = bool(overwrite or not _zarr_child_exists(run_group, run_path, "stimulus_coordinates"))
        need_protocol = bool(overwrite or "protocol_json" not in run_group.attrs)
        if not need_steps and not need_coordinates and not need_protocol:
            add_detail("skipped_existing", run_name=run_name)
            continue

        h5_path, h5_reason = _resolve_stimulus_backfill_h5(zarr_path, run_group, source_h5=source_h5)
        if h5_path is None:
            status = "skipped_ambiguous_h5" if h5_reason.startswith("ambiguous") else "skipped_missing_h5"
            add_detail(status, run_name=run_name, reason=h5_reason)
            continue

        try:
            with h5py.File(h5_path, "r") as h5:
                events_data = h5["/events"][:] if "/events" in h5 else None
                step_count = _count_step_start_events(events_data)
                if need_steps and step_count == 0:
                    add_detail(
                        "skipped_no_step_events",
                        run_name=run_name,
                        source_h5=str(h5_path),
                    )
                    continue

                protocol_text, protocol_payload = _read_protocol_snapshot(h5)
                frame_metadata = (
                    h5["/video_metadata/frame_metadata"][:]
                    if "/video_metadata/frame_metadata" in h5
                    else _empty_frame_metadata()
                )
                arena_config = _read_h5_arena_config(h5) or {}

                planned_status = "would_overwrite" if overwrite else "would_backfill"
                written_status = "overwritten" if overwrite else "backfilled"
                if not apply:
                    add_detail(
                        planned_status,
                        run_name=run_name,
                        source_h5=str(h5_path),
                        step_count=step_count,
                        write_steps=need_steps,
                        write_stimulus_coordinates=need_coordinates,
                        write_protocol_json=bool(need_protocol and protocol_text is not None),
                        moving_grating_camera_offset_deg=float(moving_grating_camera_offset_deg),
                    )
                    continue

                if need_protocol and protocol_text is not None:
                    run_group.attrs["protocol_json"] = protocol_text
                if need_coordinates:
                    _copy_stimulus_coordinates(h5, run_group, console)
                if need_steps:
                    _materialize_stimulus_steps(
                        run_group,
                        h5=h5,
                        events_data=events_data,
                        protocol=protocol_payload,
                        arena_config=arena_config,
                        metadata=frame_metadata,
                        moving_grating_camera_offset_deg=float(moving_grating_camera_offset_deg),
                        console=console,
                    )
                add_detail(
                    written_status,
                    run_name=run_name,
                    source_h5=str(h5_path),
                    step_count=step_count,
                    wrote_steps=need_steps,
                    wrote_stimulus_coordinates=need_coordinates,
                    wrote_protocol_json=bool(need_protocol and protocol_text is not None),
                    moving_grating_camera_offset_deg=float(moving_grating_camera_offset_deg),
                )
        except Exception as exc:
            add_detail("error", run_name=run_name, source_h5=str(h5_path), reason=str(exc))

    if apply and consolidate_metadata:
        consolidate_report = consolidate_metadata_capture_expected_warnings(zarr_path)
        summary["consolidated_metadata"] = True
        summary["metadata_consolidation"] = consolidate_report
    else:
        summary["consolidated_metadata"] = False

    return summary


def import_stimulus_to_zarr(
    stimulus_h5: Optional[Path],
    zarr_path: Path,
    *,
    run_name: Optional[str],
    overwrite: bool,
    verbose: bool,
    repair_chaser_gaps: bool = True,
) -> str:
    """Main import routine."""
    console = Console() if verbose else None

    resolved_h5: Optional[Path] = stimulus_h5
    if resolved_h5 is None:
        auto_h5 = _find_default_h5_for_zarr_path(zarr_path)
        if auto_h5:
            resolved_h5 = auto_h5
            _log(console, f"[dim]Auto-detected stimulus H5: {auto_h5}[/dim]")
    if resolved_h5 is None or not resolved_h5.exists():
        raise FileNotFoundError(
            "Stimulus H5 not specified and no .h5 file found alongside the zarr. "
            "Provide one explicitly."
        )

    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    zarr.open_group(str(zarr_path), mode="a")
    calib_manager = CalibrationManager(str(zarr_path), verbose=verbose, console=console)

    # Ensure calibration metadata is populated before ingesting stimulus data
    try:
        existing_calib = calib_manager.get_calibration()
        if existing_calib:
            _log(console, "[dim]Calibration already present in zarr; skipping auto-import.[/dim]")
        else:
            calib_data = calib_manager.extract_from_h5(str(resolved_h5))
            if calib_data:
                saved = calib_manager.save_calibration(calib_data, overwrite=False)
                if saved:
                    _log(console, "[green]✓ Imported calibration metadata from stimulus H5[/green]")
    except Exception as exc:
        _log(console, f"[yellow]Warning:[/yellow] Unable to import calibration automatically ({exc})")

    root = zarr.open(zarr_path, mode="a")
    analysis = root.require_group("analysis")
    runs_parent = require_runs_parent(analysis, "stimulus_runs")

    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("stimulus_%Y%m%d_%H%M%S")

    if run_name in runs_parent:
        if not overwrite:
            raise ValueError(
                f"analysis/stimulus_runs/{run_name} already exists. "
                "Use --overwrite to replace the existing run."
            )
        del runs_parent[run_name]

    run_group = runs_parent.create_group(run_name)
    mark_run_started(run_group, run_name=run_name, stage="stimulus")
    with h5py.File(resolved_h5, "r") as h5:
        _materialize_analysis_calibration(
            root,
            h5,
            source_h5=resolved_h5,
            run_name=run_name,
            console=console,
        )

        analysis_meta = root.require_group("analysis_metadata")
        subject_meta = _read_h5_group_attrs(h5, "/subject_metadata")
        if subject_meta:
            analysis_meta.attrs["subject_metadata"] = json.dumps(subject_meta, sort_keys=True)
            if "experiment_setup" not in root.attrs:
                experiment_setup = _derive_experiment_setup(subject_meta)
                if experiment_setup:
                    root.attrs["experiment_setup"] = experiment_setup
                    _log(
                        console,
                        "[green]✓ Auto-set experiment setup from subject_metadata "
                        f"(subject_count={experiment_setup.get('subject_count')})[/green]",
                    )
                else:
                    _log(
                        console,
                        "[yellow]No valid subject_count in subject_metadata; "
                        "experiment_setup not set.[/yellow]",
                    )
            else:
                _log(console, "[dim]experiment_setup already present; leaving as-is.[/dim]")
        camera_meta = _read_h5_camera_metadata(h5)
        if camera_meta:
            analysis_meta.attrs["camera_metadata"] = json.dumps(camera_meta, sort_keys=True)
            analysis_meta.attrs["camera_config_hash"] = _camera_metadata_hash(camera_meta)
        session_context = _read_h5_session_context(h5)
        if session_context:
            analysis_meta.attrs["session_context"] = json.dumps(session_context, sort_keys=True)
        session_uuid = h5.attrs.get("session_uuid")
        if session_uuid:
            analysis_meta.attrs["session_uuid"] = _normalize_attr_value(session_uuid)
        if "experimental_chamber" not in root.attrs:
            chamber = _read_experimental_chamber_from_h5(h5)
            if chamber:
                root.attrs["experimental_chamber"] = chamber
                _log(console, f"[green]✓ Set experimental_chamber: {chamber}[/green]")
            else:
                _log(console, "[dim]experimental_chamber not found in H5.[/dim]")
        if "dish_design" not in root.attrs:
            arena_config = _read_h5_arena_config(h5)
            if isinstance(arena_config, dict):
                dish_name = _normalize_attr_value(arena_config.get("selected_dish_type_name"))
                if isinstance(dish_name, str) and dish_name:
                    root.attrs["dish_design"] = dish_name
                    _log(console, f"[green]✓ Set dish_design: {dish_name}[/green]")

        _copy_enums(h5, analysis, console)
        if "/video_metadata/frame_metadata" not in h5:
            raise ValueError("Stimulus H5 missing /video_metadata/frame_metadata dataset.")

        frame_metadata = h5["/video_metadata/frame_metadata"][:]
        stats = analyze_frame_gaps(frame_metadata, console)
        combined_metadata, interpolation_mask = interpolate_metadata(frame_metadata, stats, console)
        alignment = _compute_camera_alignment(combined_metadata, interpolation_mask)

        meta_group = run_group.create_group("video_metadata")
        meta_attrs = {}
        if "/video_metadata/frame_metadata" in h5:
            meta_attrs.update(_collect_attrs(h5["/video_metadata/frame_metadata"]))
        meta_attrs["interpolated"] = bool(stats.missing_frames > 0 if stats else False)
        meta_attrs["original_records"] = int(frame_metadata.shape[0])
        meta_attrs["total_records"] = int(combined_metadata.shape[0])
        write_columnar_dataset(
            meta_group,
            "frame_metadata",
            combined_metadata,
            meta_attrs,
        )

        # Build a camera-aligned view that keeps the latest stimulus entry per camera frame.
        camera_field = _resolve_struct_field(
            combined_metadata,
            "triggering_camera_frame_id",
            "camera_frame_id",
        )
        camera_aligned_metadata, camera_aligned_indices = _build_camera_aligned_metadata(
            combined_metadata,
            camera_field=camera_field,
        )
        camera_aligned_mask = None
        if interpolation_mask is not None and interpolation_mask.shape[0] == combined_metadata.shape[0]:
            camera_aligned_mask = interpolation_mask[camera_aligned_indices]
        camera_aligned_attrs = {
            "alignment": "latest_per_camera_frame",
            "source_dataset": "frame_metadata",
            "unique_camera_frames": int(camera_aligned_metadata.shape[0]),
            "notes": "Derived from frame_metadata; keeps the final stimulus entry for each camera frame.",
        }
        write_columnar_dataset(
            meta_group,
            "camera_aligned_frame_metadata",
            camera_aligned_metadata,
            camera_aligned_attrs,
        )

        mask_chunks = pick_chunks(interpolation_mask.shape)
        run_group.create_array(
            "interpolation_mask",
            data=interpolation_mask,
            chunks=mask_chunks,
            overwrite=True,
        )
        if camera_aligned_mask is not None:
            aligned_mask_chunks = pick_chunks(camera_aligned_mask.shape)
            run_group.create_array(
                "camera_aligned_interpolation_mask",
                data=camera_aligned_mask,
                chunks=aligned_mask_chunks,
                overwrite=True,
            )

        # Alignment helpers
        align_group = run_group.create_group("frame_alignment")
        camera_offset = alignment.pop("camera_frame_offset", None)
        if camera_offset is not None:
            align_group.attrs["camera_frame_offset"] = int(camera_offset)

        for key, value in alignment.items():
            chunks = pick_chunks(value.shape)
            align_group.create_array(
                key,
                data=value,
                chunks=chunks,
                overwrite=True,
            )

        # Copy tracking data
        if "/tracking_data" in h5:
            track_group = run_group.create_group("tracking_data")
            _copy_h5_dataset(h5["/tracking_data"], track_group, "chaser_states")
            _copy_h5_dataset(h5["/tracking_data"], track_group, "bounding_boxes")

        # Events
        events_data: Optional[np.ndarray] = None
        if "/events" in h5:
            events_data = h5["/events"][:]
            events_attrs = _collect_attrs(h5["/events"])

            if "events" in run_group:
                del run_group["events"]
            events_group = run_group.create_group("events")
            if events_attrs:
                events_group.attrs.update(events_attrs)

            if events_data.dtype.names:
                field_names = list(events_data.dtype.names)
                events_group.attrs["field_names"] = field_names
                events_group.attrs["storage_layout"] = "columnar"
                for field_name in field_names:
                    field_values = np.asarray(events_data[field_name])
                    if field_values.dtype.kind in ("S", "U", "O"):
                        field_values = np.asarray(_ensure_utf8_column(field_values))
                    store_array(events_group, field_name, field_values, {})
            else:
                values = np.asarray(events_data)
                if values.dtype.kind in ("S", "U", "O"):
                    values = np.asarray(_ensure_utf8_column(values))
                store_array(events_group, "values", values, {})

        # Protocol snapshot
        protocol_text, protocol_payload = _read_protocol_snapshot(h5)
        if protocol_text is not None:
            run_group.attrs["protocol_json"] = protocol_text

        # Calibration snapshot
        if "/calibration_snapshot/arena_config_json" in h5:
            calib_bytes = h5["/calibration_snapshot/arena_config_json"][()]
            try:
                calib_json = calib_bytes.decode("utf-8")
            except AttributeError:
                calib_json = str(calib_bytes)
            run_group.attrs["arena_config_json"] = calib_json

            try:
                arena_config = json.loads(calib_json)
            except json.JSONDecodeError:
                arena_config = {}

            _write_run_coordinate_metadata(run_group, arena_config)

        if "/calibration_snapshot" in h5:
            calib_src = h5["/calibration_snapshot"]
            calib_dst = run_group.require_group("calibration")

            existing = set(getattr(calib_dst, "group_keys", lambda: [])())
            source_cameras = {
                key for key in calib_src.keys() if key != "arena_config_json"
            }
            for leftover in existing - source_cameras:
                del calib_dst[leftover]

            for cam_id in source_cameras:
                cam_src = calib_src[cam_id]
                if not isinstance(cam_src, h5py.Group):
                    continue
                cam_dst = calib_dst.require_group(cam_id)
                _copy_h5_group_to_zarr_mirror(cam_src, cam_dst)

                if "homography_matrix_yml" in cam_src and "homography_matrix" not in cam_dst:
                    matrix = _parse_homography_matrix_yml(cam_src["homography_matrix_yml"][()])
                    if matrix is not None:
                        _write_array(cam_dst, "homography_matrix", matrix)
                        cam_dst.attrs["homography_matrix_source"] = "homography_matrix_yml"

        _copy_stimulus_coordinates(h5, run_group, console)
        _materialize_stimulus_steps(
            run_group,
            h5=h5,
            events_data=events_data,
            protocol=protocol_payload,
            arena_config=_read_h5_arena_config(h5) or {},
            metadata=combined_metadata,
            moving_grating_camera_offset_deg=0.0,
            console=console,
        )

        if stats:
            run_group.attrs.update(asdict(stats))

        if repair_chaser_gaps and stats and stats.missing_frames:
            _log(console, "[cyan]Detected chaser stimulus gaps; interpolating chaser states...[/cyan]")
            try:
                interpolate_run(
                    zarr_path=zarr_path,
                    run_name=run_name,
                    update_metadata=False,
                    update_chaser=True,
                    verbose=False,
                    console=console,
                )
            except KeyError as exc:
                # Some recordings legitimately lack tracking_data/chaser_states.
                # In that case we keep imported metadata and skip chaser-gap repair.
                msg = str(exc)
                if "tracking_data/chaser_states" in msg or "lacks tracking_data group" in msg:
                    reason = "missing tracking_data/chaser_states"
                    run_group.attrs["chaser_interpolation_skipped"] = True
                    run_group.attrs["chaser_interpolation_skipped_reason"] = reason
                    _log(
                        console,
                        "[yellow]Skipping chaser interpolation repair:[/yellow] "
                        "run lacks tracking_data/chaser_states",
                    )
                else:
                    raise

    run_attrs = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_h5": str(resolved_h5),
        "import_version": "1.0.0",
    }
    rendered_video = _resolve_stimulus_video_path(resolved_h5)
    if rendered_video is not None:
        run_attrs["source_stimulus_video_path"] = str(rendered_video)
    run_group.attrs.update(run_attrs)

    mark_run_complete(
        run_group,
        parent_group=runs_parent,
        run_name=run_name,
        run_provenance=build_writer_run_provenance(
            command="fisheye.analysis.import_stimulus_to_zarr",
            params={
                "import_version": run_attrs["import_version"],
                "repair_chaser_gaps": bool(repair_chaser_gaps),
            },
            input_run_ids={
                "source_h5": str(resolved_h5),
                "source_stimulus_video_path": (
                    str(rendered_video) if rendered_video is not None else None
                ),
            },
        ),
    )
    _log(console, f"\n[bold green] Imported stimulus data to analysis/stimulus_runs/{run_name}[/bold green]")
    return run_name


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import stimulus H5 contents into a Palette detection Zarr archive.",
    )
    parser.add_argument(
        "stimulus_h5",
        type=Path,
        nargs="?",
        help="Path to the raw stimulus H5 file. If omitted, the tool will search alongside the zarr.",
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive to update.")
    parser.add_argument("--run-name", help="Optional run name inside analysis/stimulus_runs/.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing stimulus run with the same name.",
    )
    parser.add_argument(
        "--skip-chaser-repair",
        action="store_true",
        help="Skip the post-import chaser state interpolation step.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    import_stimulus_to_zarr(
        stimulus_h5=args.stimulus_h5,
        zarr_path=args.zarr_path,
        run_name=args.run_name,
        overwrite=args.overwrite,
        verbose=not args.quiet,
        repair_chaser_gaps=not args.skip_chaser_repair,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
