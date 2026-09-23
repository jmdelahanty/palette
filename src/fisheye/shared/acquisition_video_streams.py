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

import zarr

from fisheye.shared.acquisition_crop_stream_ledger import (
    ACQUISITION_CROP_SOURCE_PROFILE_COLLECTION,
    publish_acquisition_crop_stream_collection_ledger,
    publish_acquisition_crop_stream_ledger,
)
from fisheye.shared.import_video_metadata import probe_video_colorimetry_attrs
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
    load_strict_json_object,
)


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

_COLORIMETRY_FIELDS = (
    "color_range",
    "color_space",
    "color_transfer",
    "color_primaries",
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
    if not isinstance(summary_value, str) or not summary_value.endswith(
        "_summary.json"
    ):
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
            if (
                field in {"metadata", "frame_clock_metadata"}
                and path.suffix.lower() == ".csv"
            ):
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
    contract = _json_safe(dict(stream))
    video_path = _resolve_relative(recording_dir, stream.get("video"))
    colorimetry_observation: dict[str, Any] | None = None
    if video_path is not None and video_path.is_file():
        probed = probe_video_colorimetry_attrs(video_path)
        observed = {
            field: str(probed[f"video_{field}"])
            for field in _COLORIMETRY_FIELDS
            if probed.get(f"video_{field}") not in (None, "")
        }
        if observed:
            declared = {
                field: str(contract[field])
                for field in _COLORIMETRY_FIELDS
                if contract.get(field) not in (None, "")
            }
            mismatches = {
                field: {"declared": declared[field], "observed": value}
                for field, value in observed.items()
                if field in declared and declared[field] != value
            }
            warnings.extend(
                f"{field}_manifest_ffprobe_mismatch" for field in mismatches
            )
            contract.update(observed)
            colorimetry_observation = {
                "schema_id": "palette.acquisition_video_colorimetry_observation.v1",
                "authority": "ffprobe_stream",
                "video_path": str(video_path),
                "observed": observed,
                "manifest_declared": declared,
                "mismatches": mismatches,
            }
    required_missing: list[str] = []
    if not files.get("video", {}).get("exists"):
        required_missing.append("video")
    if stream.get("output_kind") == "crop" and not files.get("metadata", {}).get(
        "exists"
    ):
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
        "contract": contract,
    }
    if colorimetry_observation is not None:
        payload["colorimetry_observation"] = colorimetry_observation
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
        rolling = manifest.get("rolling_clip_streams")
        if not isinstance(rolling, Mapping):
            return None
        output_kinds = _rolling_output_kinds(manifest, rolling)
        index_value = rolling.get("recording_clip_index") or manifest.get(
            "recording_clip_index"
        )
        index_path = _resolve_relative(recording_dir, index_value)
        index_payload = (
            _load_json_object(index_path)
            if index_path is not None and index_path.is_file()
            else None
        )
        camera_id = str(manifest.get("camera_id") or "").strip()
        camera_range = (
            index_payload.get("camera_ranges", {}).get(camera_id, {})
            if isinstance(index_payload, Mapping)
            and isinstance(index_payload.get("camera_ranges"), Mapping)
            else {}
        )
        member_count = int(camera_range.get("clip_count") or 0)
        frame_count = int(camera_range.get("total_frame_count") or 0)
        index_file = {
            "path": str(index_value or ""),
            "exists": bool(index_path is not None and index_path.is_file()),
            "size_bytes": (
                int(index_path.stat().st_size)
                if index_path is not None and index_path.is_file()
                else None
            ),
        }
        availability = (
            "ok"
            if index_file["exists"] and member_count > 0
            else "missing_required_file"
        )
        collection_contract = {
            "source_profile": ACQUISITION_CROP_SOURCE_PROFILE_COLLECTION,
            "output_kind": "crop",
            "frame_clock": "recording_frame_id",
            "camera_id": camera_id,
            "member_count": member_count,
            "frame_count": frame_count,
            "recording_clip_index": str(index_value or ""),
        }
        stream_payloads = {
            "crop": {
                "stream_key": "crop",
                "availability_status": availability,
                "required_missing": []
                if availability == "ok"
                else ["recording_clip_index"],
                "warnings": [],
                "files": {"recording_clip_index": index_file},
                "contract": collection_contract,
            },
            "full": {
                "stream_key": "full",
                "availability_status": availability,
                "required_missing": []
                if availability == "ok"
                else ["recording_clip_index"],
                "warnings": [],
                "files": {"recording_clip_index": index_file},
                "contract": {
                    **collection_contract,
                    "output_kind": "full",
                },
            },
        }
        stream_payloads = {key: stream_payloads[key] for key in output_kinds}
        return {
            "schema_id": ACQUISITION_VIDEO_STREAMS_SCHEMA_ID,
            "schema_version": 1,
            "source_schema_id": rolling.get("schema_id"),
            "source_frame_clock": rolling.get("frame_clock"),
            "source_profile": ACQUISITION_CROP_SOURCE_PROFILE_COLLECTION,
            "recording_manifest_path": str(recording_dir / "recording_manifest.json"),
            "recording_dir": str(recording_dir),
            "imported_at_utc": imported_at_utc or _utc_now_iso(),
            "inventory_status": availability,
            "stream_count": len(stream_payloads),
            "stream_keys": sorted(stream_payloads),
            "crop_stream_available": "crop" in stream_payloads and availability == "ok",
            "streams": stream_payloads,
        }
    streams = video_streams.get("streams")
    if not isinstance(streams, Mapping):
        return None

    stream_payloads: dict[str, Any] = {}
    for stream_key, stream in sorted(streams.items()):
        if not isinstance(stream_key, str) or not isinstance(stream, Mapping):
            continue
        stream_payloads[stream_key] = _stream_inventory(
            recording_dir, stream_key, stream
        )

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


def _rolling_output_kinds(
    manifest: Mapping[str, Any], rolling: Mapping[str, Any]
) -> tuple[str, ...]:
    """Preserve legacy inventory while requiring truthful current stream roles."""

    kinds = rolling.get("output_kinds")
    current = (
        manifest.get(SOURCE_RECORDING_IDENTITY_PROFILE_ATTR)
        == SOURCE_RECORDING_IDENTITY_PROFILE
    )
    if "output_kinds" not in rolling and not current:
        return ("crop", "full")
    if (
        not isinstance(kinds, list)
        or not kinds
        or any(type(kind) is not str or kind not in {"crop", "full"} for kind in kinds)
        or len(kinds) != len(set(kinds))
        or "full" not in kinds
    ):
        raise ValueError(
            "rolling_clip_streams.output_kinds must declare full and optionally crop exactly once"
        )
    return tuple(sorted(kinds))


def resolve_acquisition_manifest_file(
    recording_dir: Path, value: Any, *, label: str, allow_absolute: bool = False
) -> Path:
    """Resolve a confined file, permitting absolute index-row paths only by opt-in."""
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or (Path(value).is_absolute() and not allow_absolute)
    ):
        raise ValueError(f"{label} must be an exact recording-relative file")
    path = (recording_dir / value).resolve()
    if not path.is_relative_to(recording_dir.resolve()) or not path.is_file():
        raise ValueError(f"{label} must name an existing file inside the recording")
    return path


def _validate_current_rolling_streams(
    recording_dir: Path, manifest: Mapping[str, Any]
) -> None:
    """Validate current stream roles, retaining geometry/clock/ledger owners.

    In particular an index may not relabel a crop member as the full source.
    Canonical collection geometry and recording-wide rows are validated by the
    collection/clock owners; crop payload semantics stay with the ledger owner.
    """

    rolling = manifest.get("rolling_clip_streams")
    if not isinstance(rolling, Mapping):
        raise ValueError("rolling_clip_streams must be an object")
    if manifest.get("video_streams") is not None:
        raise ValueError(
            "current acquisition cannot declare both single-video and rolling streams"
        )
    if (
        manifest.get("source_layout") != "rolling_clips"
        or rolling.get("schema_id") != "palette.orange_rolling_clip_streams.v1"
        or rolling.get("frame_clock") != "recording_frame_id"
    ):
        raise ValueError(
            "current rolling streams require the existing rolling-clips schema and frame clock"
        )
    if (
        "source_profile" in rolling
        and rolling["source_profile"] != ACQUISITION_CROP_SOURCE_PROFILE_COLLECTION
    ):
        raise ValueError(
            "current rolling streams declare an unsupported source_profile"
        )
    kinds = _rolling_output_kinds(manifest, rolling)
    index_path = resolve_acquisition_manifest_file(
        recording_dir, rolling.get("recording_clip_index"), label="recording_clip_index"
    )
    index = load_strict_json_object(index_path)
    if (
        index.get("schema_id") != "palette.orange_external_ipc_recording_clip_index.v1"
        or index.get("mode") != "rolling_clips"
    ):
        raise ValueError(
            "current recording clip index must declare the rolling-clips schema and mode"
        )
    rows = index.get("rows")
    camera_id = manifest.get("camera_id")
    if (
        not isinstance(rows, list)
        or not rows
        or any(
            not isinstance(row, Mapping) or row.get("camera_serial") != camera_id
            for row in rows
        )
    ):
        raise ValueError(
            "current recording clip index must contain only the exact recording camera"
        )
    frame_count = 0
    for row in rows:
        count = row.get("frame_count")
        if type(count) is not int or count <= 0:
            raise ValueError("clip frame_count must be a positive exact integer")
        frame_count += count
        clip_path = resolve_acquisition_manifest_file(
            recording_dir,
            row.get("clip_manifest_path"),
            label="clip_manifest_path",
            allow_absolute=True,
        )
        clip = load_strict_json_object(clip_path)
        if (
            type(row.get("clip_index")) is not int
            or type(clip.get("clip_index")) is not int
            or clip["clip_index"] != row["clip_index"]
        ):
            raise ValueError(
                "clip manifest index differs from its recording clip index"
            )
        outputs = clip.get("recording_outputs")
        camera_outputs = (
            outputs.get(camera_id) if isinstance(outputs, Mapping) else None
        )
        if not isinstance(camera_outputs, Mapping) or set(camera_outputs) != set(kinds):
            raise ValueError(
                "clip outputs differ from the declared rolling output_kinds"
            )
        for role, output in camera_outputs.items():
            if not isinstance(output, Mapping) or output.get("output_kind") != role:
                raise ValueError("clip output does not bind its declared stream role")
            for field in (
                "first_recording_frame_id",
                "last_recording_frame_id",
                "frame_count",
            ):
                if (
                    type(output.get(field)) is not int
                    or type(row.get(field)) is not int
                    or output[field] != row[field]
                ):
                    raise ValueError(
                        f"clip {role} {field} differs from its recording clip index"
                    )
            required = {"video", "metadata"} if role == "crop" else {"video"}
            for field in sorted(required | (set(_PATH_FIELDS) & set(output))):
                path = resolve_acquisition_manifest_file(
                    recording_dir, output.get(field), label=f"clip {role}.{field}"
                )
                if field in {"summary", "status"}:
                    payload = load_strict_json_object(path)
                    if str(payload.get("status", "")).strip().lower() in {
                        "fail",
                        "failed",
                        "error",
                    }:
                        raise ValueError(f"clip {role}.{field} records a failure")
            if role == "full":
                for output_field, row_field in (
                    ("video", "video_path"),
                    ("metadata", "metadata_path"),
                    ("keyframes", "keyframe_path"),
                ):
                    if output_field in output and resolve_acquisition_manifest_file(
                        recording_dir,
                        output[output_field],
                        label=f"full {output_field}",
                    ) != resolve_acquisition_manifest_file(
                        recording_dir,
                        row.get(row_field),
                        label=row_field,
                        allow_absolute=True,
                    ):
                        raise ValueError(
                            f"clip index {row_field} does not select the declared full stream"
                        )
    ranges = index.get("camera_ranges")
    camera_range = ranges.get(camera_id) if isinstance(ranges, Mapping) else None
    if not isinstance(camera_range, Mapping) or (
        camera_range.get("clip_count") != len(rows)
        or camera_range.get("total_frame_count") != frame_count
    ):
        raise ValueError(
            "clip camera_ranges counts differ from the declared member rows"
        )


def validate_acquisition_video_stream_inventory(
    recording_dir: Path,
    manifest: Mapping[str, Any],
    *,
    imported_at_utc: str | None = None,
) -> dict[str, Any] | None:
    """Admit declared streams for ingestion; diagnostic warnings are not approval.

    Undeclared optional files stay optional. A declared file or stream may not
    disappear, fail parsing, or conflict with observed media during ingestion.
    The read-only inventory builder remains available for diagnostic reports.
    """

    if (
        manifest.get(SOURCE_RECORDING_IDENTITY_PROFILE_ATTR)
        == SOURCE_RECORDING_IDENTITY_PROFILE
        and "rolling_clip_streams" in manifest
    ):
        _validate_current_rolling_streams(recording_dir, manifest)
    if manifest.get("video_streams") is not None:
        declared = manifest["video_streams"]
        if not isinstance(declared, Mapping):
            raise ValueError("acquisition video streams must be an object")
        streams = declared.get("streams")
        if not isinstance(streams, Mapping) or not streams:
            raise ValueError("acquisition video stream declarations must be a nonempty object")
        for key, stream in streams.items():
            if not isinstance(key, str) or not key or not isinstance(stream, Mapping):
                raise ValueError("acquisition video stream declaration is malformed")
            video_path = _resolve_relative(recording_dir, stream.get("video"))
            if key == "full" and "frame_clock_metadata" not in stream and video_path is not None:
                conventional = video_path.with_name(f"{video_path.stem}_meta.csv")
                if conventional.exists():
                    raise ValueError("acquisition video stream full must explicitly bind its clock; conventional or crop-clock fallback is forbidden")
            for field in _PATH_FIELDS:
                if field not in stream:
                    continue
                path = _resolve_relative(recording_dir, stream[field])
                if path is None or not path.is_file():
                    raise ValueError(f"acquisition video stream {key}.{field} must name an existing file")
                if field in {"summary", "status"}:
                    try:
                        payload = load_strict_json_object(path)
                    except ValueError as exc:
                        raise ValueError(f"acquisition video stream {key}.{field} is invalid: {exc}") from exc
                    if str(payload.get("status", "")).strip().lower() in {"fail", "failed", "error"}:
                        raise ValueError(f"acquisition video stream {key}.{field} records a failure")
    inventory = build_acquisition_video_stream_inventory(
        recording_dir, manifest, imported_at_utc=imported_at_utc,
    )
    if inventory is not None and inventory["inventory_status"] != "ok":
        failures = {
            key: {"missing": item["required_missing"], "warnings": item["warnings"]}
            for key, item in inventory["streams"].items()
            if item["availability_status"] != "ok"
        }
        raise ValueError(f"acquisition video stream inventory failed: {failures}")
    return inventory


def _reopen_group_direct(group: Any) -> Any:
    try:
        return zarr.open_group(
            store=group.store_path.store,
            path=str(group.path),
            mode="r+",
            use_consolidated=False,
        )
    except (AttributeError, TypeError):
        return group


def _require_group_direct(parent: Any, name: str) -> Any:
    return _reopen_group_direct(parent.require_group(name))


def write_acquisition_video_stream_inventory(
    root: Any,
    recording_dir: Path,
    manifest: Mapping[str, Any],
    *,
    imported_at_utc: str | None = None,
) -> dict[str, Any] | None:
    """Write manifest-declared acquisition video streams into an analysis zarr."""

    inventory = validate_acquisition_video_stream_inventory(
        recording_dir,
        manifest,
        imported_at_utc=imported_at_utc,
    )
    if inventory is None:
        return None

    analysis = _require_group_direct(root, "analysis")
    parent = _require_group_direct(analysis, "acquisition_video_streams")
    streams_group = _require_group_direct(parent, "streams")

    for stream_key, stream_payload in inventory["streams"].items():
        stream_group = _require_group_direct(streams_group, stream_key)
        _put_attrs(stream_group, stream_payload)
        if stream_key == "crop":
            try:
                if (
                    inventory.get("source_profile")
                    == ACQUISITION_CROP_SOURCE_PROFILE_COLLECTION
                ):
                    publication = publish_acquisition_crop_stream_collection_ledger(
                        stream_group,
                        recording_dir,
                        manifest,
                        imported_at_utc=str(inventory["imported_at_utc"]),
                    )
                else:
                    publication = publish_acquisition_crop_stream_ledger(
                        stream_group,
                        recording_dir,
                        manifest,
                        imported_at_utc=str(inventory["imported_at_utc"]),
                    )
            except Exception as exc:
                _put_attrs(
                    stream_group,
                    {
                        "canonical_ledger_status": "failed",
                        "canonical_ledger_failure": str(exc),
                    },
                )
                raise
            ledger_attrs = publication.attrs()
            stream_payload["canonical_ledger"] = ledger_attrs
            _put_attrs(stream_group, ledger_attrs)

    _put_attrs(parent, inventory)
    _put_attrs(
        root,
        {
            "acquisition_video_streams_available": True,
            "acquisition_video_streams_path": ACQUISITION_VIDEO_STREAMS_GROUP,
            "acquisition_video_stream_count": inventory["stream_count"],
            "acquisition_crop_video_available": inventory["crop_stream_available"],
            "acquisition_crop_ledger_available": bool(
                inventory.get("streams", {})
                .get("crop", {})
                .get("canonical_ledger", {})
                .get("canonical_ledger_status")
                == "complete"
            ),
            "acquisition_video_stream_inventory_status": inventory["inventory_status"],
        },
    )
    return inventory
