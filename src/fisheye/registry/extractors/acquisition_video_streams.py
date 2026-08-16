"""Acquisition video stream row extractors for registry scans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import zarr

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.type_conversions import normalize_attr as _decode_attr


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_bool_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    text = _decode_attr(value)
    if text is None:
        return None
    norm = str(text).strip().lower()
    if norm in {"1", "true", "yes", "y", "ok", "available"}:
        return 1
    if norm in {"0", "false", "no", "n", "missing", "absent"}:
        return 0
    return None


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    elif isinstance(value, str):
        text = value.strip()
    else:
        return {}
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _canonical_json_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError:
        return json.dumps(str(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _group_keys(group: zarr.Group) -> List[str]:
    try:
        return sorted(str(name) for name in group.group_keys())
    except Exception:
        return sorted(str(name) for name in group.keys() if isinstance(name, str))


def _file_entry(files: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = files.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _file_path(files: Mapping[str, Any], key: str) -> Optional[str]:
    value = _decode_attr(_file_entry(files, key).get("path"))
    return str(value) if value else None


def _file_exists(files: Mapping[str, Any], key: str) -> Optional[int]:
    return _as_bool_int(_file_entry(files, key).get("exists"))


def _file_row_count(files: Mapping[str, Any], key: str) -> Optional[int]:
    return _as_int(_file_entry(files, key).get("data_row_count"))


def _extract_stream_attrs(streams_group: zarr.Group, stream_key: str) -> Dict[str, Any]:
    if stream_key not in streams_group:
        return {}
    return dict(streams_group[stream_key].attrs)


def _extract_acquisition_video_stream_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: Optional[str],
    zarr_use: Optional[str],
) -> List[Dict[str, Any]]:
    analysis = root.get("analysis")
    if analysis is None:
        return []
    parent = analysis.get("acquisition_video_streams")
    if parent is None:
        return []
    streams_group = parent.get("streams")
    if streams_group is None:
        return []

    parent_attrs = dict(parent.attrs)
    inventory_status = _decode_attr(parent_attrs.get("inventory_status"))
    updated_utc = utc_now()
    rows: List[Dict[str, Any]] = []
    for stream_key in _group_keys(streams_group):
        stream_attrs = _extract_stream_attrs(streams_group, stream_key)
        if not stream_attrs:
            continue
        contract = _coerce_mapping(stream_attrs.get("contract"))
        files = _coerce_mapping(stream_attrs.get("files"))
        summary = _coerce_mapping(stream_attrs.get("summary"))

        rows.append(
            {
                "stream_key": stream_key,
                "recording_id": recording_id,
                "zarr_use": zarr_use,
                "stream_id": _decode_attr(contract.get("stream_id")),
                "role": _decode_attr(contract.get("role")),
                "output_kind": _decode_attr(contract.get("output_kind")),
                "source": _decode_attr(contract.get("source")),
                "camera_id": _decode_attr(contract.get("camera_id")),
                "frame_clock": _decode_attr(contract.get("frame_clock") or parent_attrs.get("source_frame_clock")),
                "video_path": _file_path(files, "video"),
                "metadata_path": _file_path(files, "metadata"),
                "frame_clock_metadata_path": _file_path(files, "frame_clock_metadata"),
                "keyframes_path": _file_path(files, "keyframes"),
                "summary_path": _file_path(files, "summary"),
                "status_path": _file_path(files, "status"),
                "width": _as_int(contract.get("width")),
                "height": _as_int(contract.get("height")),
                "frame_count": _as_int(contract.get("frame_count")),
                "frame_rate": _as_float(contract.get("frame_rate")),
                "codec": _decode_attr(contract.get("codec")),
                "container": _decode_attr(contract.get("container")),
                "encoded_format": _decode_attr(contract.get("encoded_format")),
                "pixel_source_format": _decode_attr(contract.get("pixel_source_format")),
                "color_range": _decode_attr(contract.get("color_range")),
                "color_space": _decode_attr(contract.get("color_space")),
                "color_transfer": _decode_attr(contract.get("color_transfer")),
                "color_primaries": _decode_attr(contract.get("color_primaries")),
                "video_pixel_coordinate_space": _decode_attr(contract.get("video_pixel_coordinate_space")),
                "source_geometry_coordinate_space": _decode_attr(contract.get("source_geometry_coordinate_space")),
                "blank_frame_policy": _decode_attr(contract.get("blank_frame_policy")),
                "selection_policy": _decode_attr(contract.get("selection_policy")),
                "availability_status": _decode_attr(stream_attrs.get("availability_status")),
                "inventory_status": inventory_status,
                "video_exists": _file_exists(files, "video"),
                "metadata_exists": _file_exists(files, "metadata"),
                "frame_clock_metadata_exists": _file_exists(files, "frame_clock_metadata"),
                "keyframes_exists": _file_exists(files, "keyframes"),
                "summary_exists": _file_exists(files, "summary"),
                "status_exists": _file_exists(files, "status"),
                "metadata_row_count": _file_row_count(files, "metadata"),
                "frame_clock_metadata_row_count": _file_row_count(files, "frame_clock_metadata"),
                "frames_encoded": _as_int(summary.get("frames_encoded")),
                "frames_dropped": _as_int(summary.get("frames_dropped")),
                "canonical_ledger_status": _decode_attr(
                    stream_attrs.get("canonical_ledger_status")
                ),
                "canonical_ledger_run": _decode_attr(stream_attrs.get("canonical_ledger_run")),
                "canonical_ledger_path": _decode_attr(stream_attrs.get("canonical_ledger_path")),
                "canonical_ledger_record_sha256": _decode_attr(
                    stream_attrs.get("canonical_ledger_record_sha256")
                ),
                "canonical_ledger_source_metadata_sha256": _decode_attr(
                    stream_attrs.get("canonical_ledger_source_metadata_sha256")
                ),
                "canonical_ledger_source_video_fingerprint": _decode_attr(
                    stream_attrs.get("canonical_ledger_source_video_fingerprint")
                ),
                "canonical_ledger_row_count": _as_int(
                    stream_attrs.get("canonical_ledger_row_count")
                ),
                "canonical_ledger_detected_row_count": _as_int(
                    stream_attrs.get("canonical_ledger_detected_row_count")
                ),
                "canonical_ledger_blank_row_count": _as_int(
                    stream_attrs.get("canonical_ledger_blank_row_count")
                ),
                "canonical_ledger_imported_at_utc": _decode_attr(
                    stream_attrs.get("canonical_ledger_imported_at_utc")
                ),
                "contract_json": _canonical_json_text(contract),
                "files_json": _canonical_json_text(files),
                "summary_json": _canonical_json_text(summary),
                "updated_utc": updated_utc,
            }
        )
    return rows


__all__ = ["_extract_acquisition_video_stream_rows"]
