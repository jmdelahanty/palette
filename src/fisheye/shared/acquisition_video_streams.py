"""Recording-level acquisition video stream inventory helpers.

These helpers mirror ``recording_manifest.json`` ``video_streams`` metadata into
an analysis zarr without treating acquisition crop videos as Palette-generated
``crop_runs`` outputs.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACQUISITION_VIDEO_STREAMS_SCHEMA_ID = "palette.acquisition_video_streams.v1"
ACQUISITION_VIDEO_STREAMS_GROUP = "analysis/acquisition_video_streams"

_PATH_FIELDS = (
    "video",
    "metadata",
    "frame_clock_metadata",
    "keyframes",
    "summary",
    "status",
)

_SUMMARY_KEYS = (
    "schema_id",
    "status",
    "output_kind",
    "stream_id",
    "frames_received",
    "frames_encoded",
    "frames_dropped",
    "frame_count",
    "width",
    "height",
    "frame_rate",
    "codec",
    "container",
    "encoded_format",
    "pixel_source_format",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(nested) for nested in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _resolve_relative(recording_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return recording_dir / path


def _infer_status_path(summary_value: Any) -> str | None:
    if not isinstance(summary_value, str) or not summary_value.endswith("_summary.json"):
        return None
    return f"{summary_value[: -len('_summary.json')]}_status.json"


def _count_csv_data_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for _row in reader)
    except OSError:
        return None


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _summary_subset(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {key: _json_safe(payload[key]) for key in _SUMMARY_KEYS if key in payload}


def _file_availability(
    recording_dir: Path,
    stream: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    files: dict[str, dict[str, Any]] = {}
    derived: dict[str, Any] = {}
    warnings: list[str] = []

    stream_paths: dict[str, Any] = {field: stream.get(field) for field in _PATH_FIELDS}
    if "status" not in stream_paths or stream_paths.get("status") is None:
        inferred = _infer_status_path(stream.get("summary"))
        if inferred:
            stream_paths["status"] = inferred
            derived["inferred_status"] = inferred

    for field, value in stream_paths.items():
        path = _resolve_relative(recording_dir, value)
        if path is None:
            continue
        exists = path.exists()
        entry: dict[str, Any] = {
            "path": str(value),
            "exists": bool(exists),
        }
        if exists:
            try:
                entry["size_bytes"] = int(path.stat().st_size)
            except OSError:
                warnings.append(f"{field}_stat_failed")
            if field in {"metadata", "frame_clock_metadata"} and path.suffix.lower() == ".csv":
                row_count = _count_csv_data_rows(path)
                if row_count is None:
                    warnings.append(f"{field}_row_count_failed")
                else:
                    entry["data_row_count"] = int(row_count)
            if field == "summary":
                summary = _load_json_object(path)
                if summary is None:
                    warnings.append("summary_json_unreadable")
                else:
                    derived["summary"] = _summary_subset(summary)
            if field == "status":
                status = _load_json_object(path)
                if status is None:
                    warnings.append("status_json_unreadable")
                else:
                    derived["status"] = _summary_subset(status)
        else:
            entry["size_bytes"] = None
        files[field] = entry

    return files, derived, warnings


def _expected_frame_count(stream: Mapping[str, Any]) -> int | None:
    value = stream.get("frame_count")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stream_inventory(
    recording_dir: Path,
    stream_key: str,
    stream: Mapping[str, Any],
) -> dict[str, Any]:
    files, derived, warnings = _file_availability(recording_dir, stream)
    required_missing: list[str] = []
    if not files.get("video", {}).get("exists"):
        required_missing.append("video")
    if stream.get("output_kind") == "crop" and not files.get("metadata", {}).get("exists"):
        required_missing.append("metadata")

    expected_frames = _expected_frame_count(stream)
    if expected_frames is not None:
        for field in ("metadata", "frame_clock_metadata"):
            row_count = files.get(field, {}).get("data_row_count")
            if row_count is not None and int(row_count) != expected_frames:
                warnings.append(f"{field}_row_count_mismatch")

    availability_status = "ok"
    if required_missing:
        availability_status = "missing_required_file"
    elif warnings:
        availability_status = "warn"

    payload: dict[str, Any] = {
        "stream_key": stream_key,
        "availability_status": availability_status,
        "required_missing": required_missing,
        "warnings": sorted(set(warnings)),
        "files": files,
        "contract": _json_safe(dict(stream)),
    }
    payload.update(derived)
    return payload


def build_acquisition_video_stream_inventory(
    recording_dir: Path,
    manifest: Mapping[str, Any],
    *,
    imported_at_utc: str | None = None,
) -> dict[str, Any] | None:
    """Build a zarr-serializable acquisition video stream inventory.

    Returns ``None`` when the manifest does not declare ``video_streams``.
    """

    video_streams = manifest.get("video_streams")
    if not isinstance(video_streams, Mapping):
        return None
    streams = video_streams.get("streams")
    if not isinstance(streams, Mapping):
        return None

    stream_payloads: dict[str, Any] = {}
    for stream_key, stream in sorted(streams.items()):
        if not isinstance(stream_key, str) or not isinstance(stream, Mapping):
            continue
        stream_payloads[stream_key] = _stream_inventory(recording_dir, stream_key, stream)

    if not stream_payloads:
        return None

    crop_stream = stream_payloads.get("crop")
    statuses = {
        payload.get("availability_status")
        for payload in stream_payloads.values()
        if isinstance(payload, Mapping)
    }
    inventory_status = "ok" if statuses == {"ok"} else "warn"
    if any(status == "missing_required_file" for status in statuses):
        inventory_status = "missing_required_file"

    return {
        "schema_id": ACQUISITION_VIDEO_STREAMS_SCHEMA_ID,
        "schema_version": 1,
        "source_schema_id": video_streams.get("schema_id"),
        "source_frame_clock": video_streams.get("frame_clock"),
        "recording_manifest_path": str(recording_dir / "recording_manifest.json"),
        "recording_dir": str(recording_dir),
        "imported_at_utc": imported_at_utc or _utc_now_iso(),
        "inventory_status": inventory_status,
        "stream_count": len(stream_payloads),
        "stream_keys": sorted(stream_payloads),
        "crop_stream_available": bool(
            crop_stream and crop_stream.get("files", {}).get("video", {}).get("exists")
        ),
        "streams": stream_payloads,
    }


def _put_attrs(group: Any, updates: Mapping[str, Any]) -> None:
    attrs = dict(group.attrs)
    attrs.update(_json_safe(dict(updates)))
    group.attrs.put(attrs)


def write_acquisition_video_stream_inventory(
    root: Any,
    recording_dir: Path,
    manifest: Mapping[str, Any],
    *,
    imported_at_utc: str | None = None,
) -> dict[str, Any] | None:
    """Write manifest-declared acquisition video streams into an analysis zarr."""

    inventory = build_acquisition_video_stream_inventory(
        recording_dir,
        manifest,
        imported_at_utc=imported_at_utc,
    )
    if inventory is None:
        return None

    analysis = root.require_group("analysis")
    parent = analysis.require_group("acquisition_video_streams")
    streams_group = parent.require_group("streams")

    for stream_key, stream_payload in inventory["streams"].items():
        stream_group = streams_group.require_group(stream_key)
        _put_attrs(stream_group, stream_payload)

    _put_attrs(parent, inventory)
    _put_attrs(
        root,
        {
            "acquisition_video_streams_available": True,
            "acquisition_video_streams_path": ACQUISITION_VIDEO_STREAMS_GROUP,
            "acquisition_video_stream_count": inventory["stream_count"],
            "acquisition_crop_video_available": inventory["crop_stream_available"],
            "acquisition_video_stream_inventory_status": inventory["inventory_status"],
        },
    )
    return inventory
