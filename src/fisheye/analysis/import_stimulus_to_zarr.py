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
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
from numpy.lib import recfunctions as rfn
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

    calib_manager = CalibrationManager(str(zarr_path), verbose=verbose, console=console)

    resolved_h5: Optional[Path] = stimulus_h5
    if resolved_h5 is None:
        auto_h5 = calib_manager.find_default_h5()
        if auto_h5:
            resolved_h5 = auto_h5
            _log(console, f"[dim]Auto-detected stimulus H5: {auto_h5}[/dim]")
    if resolved_h5 is None or not resolved_h5.exists():
        raise FileNotFoundError(
            "Stimulus H5 not specified and no .h5 file found alongside the zarr. "
            "Provide one explicitly."
        )

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
    runs_parent = analysis.require_group("stimulus_runs")

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
    with h5py.File(resolved_h5, "r") as h5:
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
        if "/protocol_snapshot/protocol_json" in h5:
            proto_bytes = h5["/protocol_snapshot/protocol_json"][()]
            if isinstance(proto_bytes, bytes):
                run_group.attrs["protocol_json"] = proto_bytes.decode("utf-8")
            else:
                run_group.attrs["protocol_json"] = proto_bytes

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

            coord_info = {
                "texture_dimensions": [358, 358],
                "camera_dimensions": [4512, 4512],
                "texture_to_camera_scale": 4512 / 358,
                "coordinate_note": "Chaser positions are in texture space (358x358); fish in camera space (4512x4512).",
            }
            try:
                cam_calib = arena_config.get("camera_calibrations", [{}])[0]
                width = cam_calib.get("native_width_px")
                height = cam_calib.get("native_height_px")
                if width and height:
                    coord_info["camera_dimensions"] = [int(width), int(height)]
                    coord_info["texture_to_camera_scale"] = width / 358
            except Exception:
                pass
            run_group.attrs["coordinate_transform"] = json.dumps(coord_info)

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

                for attr_name, attr_value in cam_src.attrs.items():
                    if isinstance(attr_value, bytes):
                        attr_value = attr_value.decode("utf-8", "ignore")
                    elif isinstance(attr_value, np.generic):
                        attr_value = attr_value.item()
                    cam_dst.attrs[attr_name] = attr_value

                for dset_name, dset in cam_src.items():
                    data = dset[()]
                    if isinstance(data, bytes):
                        cam_dst.attrs[dset_name] = data.decode("utf-8", "ignore")
                    elif np.isscalar(data) or (isinstance(data, np.ndarray) and data.ndim == 0):
                        cam_dst.attrs[dset_name] = data.item() if hasattr(data, "item") else data
                    else:
                        arr = np.asarray(data)
                        if dset_name in cam_dst:
                            del cam_dst[dset_name]
                        cam_dst.create_array(
                            dset_name,
                            data=arr,
                            chunks=arr.shape,
                            overwrite=True,
                        )

        if stats:
            run_group.attrs.update(asdict(stats))

        if repair_chaser_gaps and stats and stats.missing_frames:
            _log(console, "[cyan]Detected chaser stimulus gaps; interpolating chaser states...[/cyan]")
            interpolate_run(
                zarr_path=zarr_path,
                run_name=run_name,
                update_metadata=False,
                update_chaser=True,
                verbose=False,
                console=console,
            )

    run_group.attrs.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_h5": str(resolved_h5),
            "import_version": "1.0.0",
        }
    )

    runs_parent.attrs["latest"] = run_name
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
