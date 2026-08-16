"""Canonical, immutable acquisition crop-stream ledger publication.

The Orange CSV remains the producer artifact.  This module mirrors its complete
row coverage into a producer-neutral Zarr contract so consumers do not need to
parse Orange-specific CSV files.  Publication is pointer-last: a partially
written ledger run is never made current.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.import_source_fingerprint import source_stat_fingerprint_attrs
from fisheye.shared.zarr.chunk_profiles import (
    create_geometry_preload_array,
    stamp_geometry_preload_attrs,
)


ACQUISITION_CROP_LEDGER_SCHEMA_ID = "palette.acquisition_crop_stream_ledger.v1"
ACQUISITION_CROP_LEDGER_SCHEMA_VERSION = 1
ACQUISITION_CROP_LEDGER_RUNS_GROUP = "ledger_runs"

_REQUIRED_COLUMNS = frozenset(
    {
        "recording_frame_id",
        "has_detection",
        "blank_frame",
        "crop_x",
        "crop_y",
        "crop_w",
        "crop_h",
    }
)
_CATEGORICAL_COLUMNS = (
    "crop_state",
    "crop_rect_coordinate_space",
    "crop_rect_layout",
    "crop_rect_semantics",
    "detection_rect_coordinate_space",
    "detection_rect_layout",
    "detection_rect_semantics",
    "detection_source",
    "selection_policy",
)
_ARRAY_NAMES = (
    "source_crop_meta_row_indices",
    "source_recording_frame_ids",
    "source_recording_frame_indices",
    "source_crop_video_frame_indices",
    "source_session_crop_video_frame_indices",
    "source_crop_local_frame_ids",
    "source_camera_frame_ids",
    "source_timestamp_ns",
    "source_timestamp_sys_ns",
    "has_detection",
    "blank_frame",
    "source_crop_xywh",
    "source_detection_xywh",
    "source_detection_confidence",
    "crop_rect_valid",
    "detection_rect_valid",
    *(f"{name}_codes" for name in _CATEGORICAL_COLUMNS),
)


@dataclass(frozen=True)
class AcquisitionCropLedgerPublication:
    status: str
    run_name: str
    group_path: str
    record_sha256: str
    source_metadata_sha256: str
    source_video_fingerprint: str
    row_count: int
    detected_row_count: int
    blank_row_count: int
    imported_at_utc: str
    idempotent: bool = False

    def attrs(self) -> dict[str, Any]:
        return {
            "canonical_ledger_status": self.status,
            "canonical_ledger_run": self.run_name,
            "canonical_ledger_path": self.group_path,
            "canonical_ledger_record_sha256": self.record_sha256,
            "canonical_ledger_source_metadata_sha256": self.source_metadata_sha256,
            "canonical_ledger_source_video_fingerprint": self.source_video_fingerprint,
            "canonical_ledger_row_count": self.row_count,
            "canonical_ledger_detected_row_count": self.detected_row_count,
            "canonical_ledger_blank_row_count": self.blank_row_count,
            "canonical_ledger_imported_at_utc": self.imported_at_utc,
            "canonical_ledger_idempotent": self.idempotent,
        }


@dataclass(frozen=True)
class _ParsedLedger:
    fieldnames: tuple[str, ...]
    arrays: dict[str, np.ndarray]
    code_maps: dict[str, dict[str, int]]
    row_count: int
    detected_row_count: int
    blank_row_count: int


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_utc_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"imported_at_utc must be an ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError("imported_at_utc must include an explicit timezone.")
    return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_relative(recording_dir: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Crop stream contract is missing {label}.")
    path = Path(value)
    return path if path.is_absolute() else recording_dir / path


def _source_sidecar_identities(
    recording_dir: Path,
    crop: Mapping[str, Any],
    *,
    metadata_sha256: str,
) -> dict[str, dict[str, Any]]:
    declared: dict[str, str] = {}
    for name in ("metadata", "keyframes", "summary", "status"):
        value = crop.get(name)
        if isinstance(value, str) and value.strip():
            declared[name] = value
    if "status" not in declared and declared.get("summary", "").endswith("_summary.json"):
        declared["status"] = declared["summary"][: -len("_summary.json")] + "_status.json"

    identities: dict[str, dict[str, Any]] = {}
    for name, declared_path in sorted(declared.items()):
        path = _resolve_relative(recording_dir, declared_path, label=name)
        if not path.is_file():
            if name == "status" and crop.get("status") is None:
                continue
            raise ValueError(f"Declared crop {name} sidecar does not exist: {path}")
        identities[name] = {
            "path": declared_path,
            "size_bytes": int(path.stat().st_size),
            "sha256": metadata_sha256 if name == "metadata" else _sha256_file(path),
        }
    return identities


def _strict_int(value: Any, *, column: str, row_number: int) -> int:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"Crop metadata row {row_number} has an empty {column}.")
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(
            f"Crop metadata row {row_number} has non-integer {column}={text!r}."
        ) from exc
    return parsed


def _optional_int(value: Any, *, column: str, row_number: int, default: int) -> int:
    text = "" if value is None else str(value).strip()
    if not text:
        return int(default)
    return _strict_int(text, column=column, row_number=row_number)


def _strict_float(value: Any, *, column: str, row_number: int) -> float:
    text = "" if value is None else str(value).strip()
    if not text:
        return math.nan
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Crop metadata row {row_number} has non-numeric {column}={text!r}."
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Crop metadata row {row_number} has non-finite {column}.")
    return parsed


def _strict_bool(value: Any, *, column: str, row_number: int) -> bool:
    text = "" if value is None else str(value).strip().lower()
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    raise ValueError(
        f"Crop metadata row {row_number} has invalid boolean {column}={value!r}."
    )


def _categorical_codes(values: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    vocabulary = sorted(set(values))
    mapping = {value: index for index, value in enumerate(vocabulary)}
    return np.asarray([mapping[value] for value in values], dtype=np.int16), mapping


def _parse_crop_metadata(path: Path, *, width: int, height: int) -> _ParsedLedger:
    columns: dict[str, list[Any]] = {
        "source_crop_meta_row_indices": [],
        "source_recording_frame_ids": [],
        "source_recording_frame_indices": [],
        "source_crop_video_frame_indices": [],
        "source_session_crop_video_frame_indices": [],
        "source_crop_local_frame_ids": [],
        "source_camera_frame_ids": [],
        "source_timestamp_ns": [],
        "source_timestamp_sys_ns": [],
        "has_detection": [],
        "blank_frame": [],
        "source_crop_xywh": [],
        "source_detection_xywh": [],
        "source_detection_confidence": [],
        "crop_rect_valid": [],
        "detection_rect_valid": [],
    }
    categorical: dict[str, list[str]] = {name: [] for name in _CATEGORICAL_COLUMNS}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - set(fieldnames))
        if missing:
            raise ValueError(f"Crop metadata is missing required columns: {missing}")
        previous_frame_id = 0
        for row_index, row in enumerate(reader):
            row_number = row_index + 2
            frame_id = _strict_int(
                row.get("recording_frame_id"), column="recording_frame_id", row_number=row_number
            )
            if frame_id != previous_frame_id + 1:
                raise ValueError(
                    "Crop metadata recording_frame_id must be contiguous and one-based; "
                    f"row {row_number} expected {previous_frame_id + 1}, found {frame_id}."
                )
            previous_frame_id = frame_id
            has_detection = _strict_bool(
                row.get("has_detection"), column="has_detection", row_number=row_number
            )
            blank_frame = _strict_bool(
                row.get("blank_frame"), column="blank_frame", row_number=row_number
            )
            if blank_frame == has_detection:
                raise ValueError(
                    f"Crop metadata row {row_number} must encode exactly one of detection or blank."
                )
            crop_xywh = tuple(
                _strict_float(row.get(name), column=name, row_number=row_number)
                for name in ("crop_x", "crop_y", "crop_w", "crop_h")
            )
            detection_xywh = tuple(
                _strict_float(row.get(name), column=name, row_number=row_number)
                for name in ("detection_x", "detection_y", "detection_w", "detection_h")
            )
            crop_valid_default = has_detection and not blank_frame
            crop_rect_valid = (
                _strict_bool(row.get("crop_rect_valid"), column="crop_rect_valid", row_number=row_number)
                if str(row.get("crop_rect_valid") or "").strip()
                else crop_valid_default
            )
            detection_rect_valid = (
                _strict_bool(
                    row.get("detection_rect_valid"),
                    column="detection_rect_valid",
                    row_number=row_number,
                )
                if str(row.get("detection_rect_valid") or "").strip()
                else crop_valid_default
            )
            if crop_rect_valid != crop_valid_default or detection_rect_valid != crop_valid_default:
                raise ValueError(
                    f"Crop metadata row {row_number} validity flags disagree with detection/blank state."
                )
            if crop_rect_valid:
                if not all(math.isfinite(value) for value in crop_xywh):
                    raise ValueError(f"Crop metadata row {row_number} has non-finite crop geometry.")
                if crop_xywh[2:] != (float(width), float(height)):
                    raise ValueError(
                        f"Crop metadata row {row_number} crop size {crop_xywh[2:]} does not match "
                        f"declared crop video size {(width, height)}."
                    )
                if crop_xywh[0] < 0 or crop_xywh[1] < 0:
                    raise ValueError(f"Crop metadata row {row_number} has a negative crop origin.")
                if not all(math.isfinite(value) for value in detection_xywh):
                    raise ValueError(
                        f"Crop metadata row {row_number} has non-finite detection geometry."
                    )
                if detection_xywh[2] <= 0 or detection_xywh[3] <= 0:
                    raise ValueError(
                        f"Crop metadata row {row_number} has a non-positive detection size."
                    )

            columns["source_crop_meta_row_indices"].append(row_index)
            columns["source_recording_frame_ids"].append(frame_id)
            columns["source_recording_frame_indices"].append(frame_id - 1)
            crop_video_frame_index = _optional_int(
                row.get("crop_video_frame_index"),
                column="crop_video_frame_index",
                row_number=row_number,
                default=row_index,
            )
            if crop_video_frame_index != row_index:
                raise ValueError(
                    f"Crop metadata row {row_number} has crop_video_frame_index "
                    f"{crop_video_frame_index}; expected ordered frame {row_index}."
                )
            columns["source_crop_video_frame_indices"].append(crop_video_frame_index)
            columns["source_session_crop_video_frame_indices"].append(
                _optional_int(
                    row.get("session_crop_video_frame_index"),
                    column="session_crop_video_frame_index",
                    row_number=row_number,
                    default=row_index,
                )
            )
            columns["source_crop_local_frame_ids"].append(
                _optional_int(
                    row.get("local_frame_id"), column="local_frame_id", row_number=row_number, default=-1
                )
            )
            columns["source_camera_frame_ids"].append(
                _optional_int(
                    row.get("camera_frame_id"),
                    column="camera_frame_id",
                    row_number=row_number,
                    default=-1,
                )
            )
            columns["source_timestamp_ns"].append(
                _optional_int(row.get("timestamp"), column="timestamp", row_number=row_number, default=-1)
            )
            columns["source_timestamp_sys_ns"].append(
                _optional_int(
                    row.get("timestamp_sys"), column="timestamp_sys", row_number=row_number, default=-1
                )
            )
            columns["has_detection"].append(has_detection)
            columns["blank_frame"].append(blank_frame)
            columns["source_crop_xywh"].append(crop_xywh)
            columns["source_detection_xywh"].append(detection_xywh)
            detection_confidence = _strict_float(
                row.get("detection_confidence"),
                column="detection_confidence",
                row_number=row_number,
            )
            columns["source_detection_confidence"].append(detection_confidence)
            columns["crop_rect_valid"].append(crop_rect_valid)
            columns["detection_rect_valid"].append(detection_rect_valid)
            for name in _CATEGORICAL_COLUMNS:
                categorical[name].append(str(row.get(name) or ""))

    row_count = len(columns["source_recording_frame_ids"])
    arrays = {
        "source_crop_meta_row_indices": np.asarray(columns["source_crop_meta_row_indices"], dtype=np.int64),
        "source_recording_frame_ids": np.asarray(columns["source_recording_frame_ids"], dtype=np.int64),
        "source_recording_frame_indices": np.asarray(columns["source_recording_frame_indices"], dtype=np.int64),
        "source_crop_video_frame_indices": np.asarray(columns["source_crop_video_frame_indices"], dtype=np.int64),
        "source_session_crop_video_frame_indices": np.asarray(
            columns["source_session_crop_video_frame_indices"], dtype=np.int64
        ),
        "source_crop_local_frame_ids": np.asarray(columns["source_crop_local_frame_ids"], dtype=np.int64),
        "source_camera_frame_ids": np.asarray(columns["source_camera_frame_ids"], dtype=np.int64),
        "source_timestamp_ns": np.asarray(columns["source_timestamp_ns"], dtype=np.int64),
        "source_timestamp_sys_ns": np.asarray(columns["source_timestamp_sys_ns"], dtype=np.int64),
        "has_detection": np.asarray(columns["has_detection"], dtype=bool),
        "blank_frame": np.asarray(columns["blank_frame"], dtype=bool),
        "source_crop_xywh": np.asarray(columns["source_crop_xywh"], dtype=np.float64).reshape(-1, 4),
        "source_detection_xywh": np.asarray(columns["source_detection_xywh"], dtype=np.float64).reshape(-1, 4),
        "source_detection_confidence": np.asarray(
            columns["source_detection_confidence"], dtype=np.float64
        ),
        "crop_rect_valid": np.asarray(columns["crop_rect_valid"], dtype=bool),
        "detection_rect_valid": np.asarray(columns["detection_rect_valid"], dtype=bool),
    }
    code_maps: dict[str, dict[str, int]] = {}
    for name, values in categorical.items():
        arrays[f"{name}_codes"], code_maps[name] = _categorical_codes(values)
    return _ParsedLedger(
        fieldnames=fieldnames,
        arrays=arrays,
        code_maps=code_maps,
        row_count=row_count,
        detected_row_count=int(np.count_nonzero(arrays["has_detection"])),
        blank_row_count=int(np.count_nonzero(arrays["blank_frame"])),
    )


def _validated_contract(
    recording_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Path, Path, int, int]:
    video_streams = manifest.get("video_streams")
    if not isinstance(video_streams, Mapping):
        raise ValueError("Recording manifest has no video_streams object.")
    if video_streams.get("schema_id") != "orange_runtime_video_streams_v1":
        raise ValueError("Crop ledger import requires orange_runtime_video_streams_v1.")
    if video_streams.get("frame_clock") != "recording_frame_id":
        raise ValueError("Crop ledger import requires recording_frame_id as the frame clock.")
    streams = video_streams.get("streams")
    crop = streams.get("crop") if isinstance(streams, Mapping) else None
    if not isinstance(crop, Mapping) or crop.get("output_kind") != "crop":
        raise ValueError("Recording manifest has no valid crop output stream.")
    if crop.get("frame_clock") != "recording_frame_id":
        raise ValueError("Crop stream frame clock must be recording_frame_id.")
    if crop.get("video_pixel_coordinate_space") != "crop_frame_pixels":
        raise ValueError("Crop stream video pixels must use crop_frame_pixels.")
    if crop.get("source_geometry_coordinate_space") != "full_frame_pixels":
        raise ValueError("Crop stream source geometry must use full_frame_pixels.")
    manifest_camera = str(manifest.get("camera_id") or "").strip()
    crop_camera = str(crop.get("camera_id") or "").strip()
    if not manifest_camera or not crop_camera or manifest_camera != crop_camera:
        raise ValueError("Crop stream camera_id must exactly match recording_manifest camera_id.")
    try:
        width = int(crop.get("width"))
        height = int(crop.get("height"))
        frame_count = int(crop.get("frame_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Crop stream width, height, and frame_count must be integers.") from exc
    if width <= 0 or height <= 0 or frame_count < 0:
        raise ValueError("Crop stream width/height must be positive and frame_count non-negative.")
    metadata_path = _resolve_relative(recording_dir, crop.get("metadata"), label="metadata")
    video_path = _resolve_relative(recording_dir, crop.get("video"), label="video")
    if not metadata_path.is_file():
        raise ValueError(f"Crop metadata does not exist: {metadata_path}")
    if not video_path.is_file():
        raise ValueError(f"Crop video does not exist: {video_path}")
    return crop, metadata_path, video_path, width, height


def _validate_current_run(stream_group: Any, publication: AcquisitionCropLedgerPublication) -> None:
    runs = stream_group.get(ACQUISITION_CROP_LEDGER_RUNS_GROUP)
    if runs is None or publication.run_name not in runs:
        raise ValueError("Current crop ledger pointer names a missing immutable run.")
    run = runs[publication.run_name]
    attrs = dict(run.attrs)
    if attrs.get("publication_status") != "complete":
        raise ValueError("Current crop ledger run is not complete.")
    if attrs.get("record_sha256") != publication.record_sha256:
        raise ValueError("Current crop ledger digest does not match its immutable run.")
    for name in _ARRAY_NAMES:
        if name not in run or int(run[name].shape[0]) != publication.row_count:
            raise ValueError(f"Current crop ledger array {name!r} is missing or incomplete.")


def publish_acquisition_crop_stream_ledger(
    stream_group: Any,
    recording_dir: Path,
    manifest: Mapping[str, Any],
    *,
    imported_at_utc: str | None = None,
) -> AcquisitionCropLedgerPublication:
    """Publish and pointer-link the complete crop ledger for one recording."""

    try:
        stream_group = zarr.open_group(
            store=stream_group.store_path.store,
            path=str(stream_group.path),
            mode="r+",
            use_consolidated=False,
        )
    except (AttributeError, TypeError):
        # In-memory/fake groups used by deterministic unit tests have no Zarr
        # store path. Their mappings are already the direct authority.
        pass

    crop, metadata_path, video_path, width, height = _validated_contract(
        Path(recording_dir), manifest
    )
    parsed = _parse_crop_metadata(metadata_path, width=width, height=height)
    expected_rows = int(crop["frame_count"])
    if parsed.row_count != expected_rows:
        raise ValueError(
            f"Crop metadata row count {parsed.row_count} does not match frame_count {expected_rows}."
        )
    video_fingerprint_attrs = source_stat_fingerprint_attrs(
        video_path,
        attr_prefix="source_video",
        extra={
            "stream_id": crop.get("stream_id"),
            "frame_count": expected_rows,
            "width": width,
            "height": height,
        },
    )
    metadata_sha256 = _sha256_file(metadata_path)
    sidecar_identities = _source_sidecar_identities(
        Path(recording_dir), crop, metadata_sha256=metadata_sha256
    )
    record = {
        "schema_id": ACQUISITION_CROP_LEDGER_SCHEMA_ID,
        "schema_version": ACQUISITION_CROP_LEDGER_SCHEMA_VERSION,
        "source_metadata_sha256": metadata_sha256,
        "source_sidecar_identities": sidecar_identities,
        "source_video_fingerprint": video_fingerprint_attrs["source_video_fingerprint"],
        "source_stream_contract": dict(crop),
        "source_columns": list(parsed.fieldnames),
        "row_count": parsed.row_count,
    }
    record_sha256 = hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()
    now = imported_at_utc or datetime.now(timezone.utc).isoformat()

    current_digest = str(stream_group.attrs.get("canonical_ledger_record_sha256") or "")
    current_run = str(stream_group.attrs.get("canonical_ledger_run") or "")
    if current_digest or current_run:
        if current_digest != record_sha256 or not current_run:
            raise ValueError(
                "Crop stream already has a different immutable canonical ledger; "
                "refusing to replace recording authority."
            )
        existing = AcquisitionCropLedgerPublication(
            status="complete",
            run_name=current_run,
            group_path=f"{ACQUISITION_CROP_LEDGER_RUNS_GROUP}/{current_run}",
            record_sha256=record_sha256,
            source_metadata_sha256=metadata_sha256,
            source_video_fingerprint=str(video_fingerprint_attrs["source_video_fingerprint"]),
            row_count=parsed.row_count,
            detected_row_count=parsed.detected_row_count,
            blank_row_count=parsed.blank_row_count,
            imported_at_utc=str(stream_group.attrs.get("canonical_ledger_imported_at_utc") or now),
            idempotent=True,
        )
        _validate_current_run(stream_group, existing)
        return existing

    compact_time = _compact_utc_timestamp(now)
    run_name = f"crop_ledger_{compact_time}_{record_sha256[:12]}"
    runs = stream_group.require_group(ACQUISITION_CROP_LEDGER_RUNS_GROUP)
    if run_name in runs:
        raise ValueError(f"Crop ledger run collision: {run_name}")
    run = runs.create_group(run_name)
    run.attrs.update(
        {
            "schema_id": ACQUISITION_CROP_LEDGER_SCHEMA_ID,
            "schema_version": ACQUISITION_CROP_LEDGER_SCHEMA_VERSION,
            "publication_status": "publishing",
            "selector_eligible": False,
            "record_sha256": record_sha256,
            "source_metadata_sha256": metadata_sha256,
            "source_metadata_path": str(metadata_path),
            "source_video_path": str(video_path),
            "source_columns": list(parsed.fieldnames),
            "source_sidecar_identities": sidecar_identities,
            "source_stream_contract": dict(crop),
            "row_count": parsed.row_count,
            "created_at_utc": now,
            **video_fingerprint_attrs,
        }
    )
    stamp_geometry_preload_attrs(run)
    try:
        for name, values in parsed.arrays.items():
            create_geometry_preload_array(
                run,
                name,
                data=values,
                overwrite=False,
                attrs={"row_axis_semantics": "ordered_acquisition_crop_video_frame"},
            )
        run.attrs.update(
            {
                "categorical_code_maps": parsed.code_maps,
                "record": record,
                "detected_row_count": parsed.detected_row_count,
                "blank_row_count": parsed.blank_row_count,
                "publication_status": "complete",
                "completed_at_utc": now,
            }
        )
    except Exception as exc:
        run.attrs.update(
            {
                "publication_status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure": str(exc),
            }
        )
        raise

    publication = AcquisitionCropLedgerPublication(
        status="complete",
        run_name=run_name,
        group_path=f"{ACQUISITION_CROP_LEDGER_RUNS_GROUP}/{run_name}",
        record_sha256=record_sha256,
        source_metadata_sha256=metadata_sha256,
        source_video_fingerprint=str(video_fingerprint_attrs["source_video_fingerprint"]),
        row_count=parsed.row_count,
        detected_row_count=parsed.detected_row_count,
        blank_row_count=parsed.blank_row_count,
        imported_at_utc=now,
    )
    # Pointer-last publication: readers ignore every run not named here.
    stream_group.attrs.update(publication.attrs())
    return publication


def validate_current_acquisition_crop_stream_ledger(
    root: Any,
    *,
    expected_record_sha256: str | None = None,
) -> AcquisitionCropLedgerPublication:
    """Validate the pointer-selected ledger through the supplied metadata view."""

    analysis = root.get("analysis")
    parent = analysis.get("acquisition_video_streams") if analysis is not None else None
    streams = parent.get("streams") if parent is not None else None
    stream = streams.get("crop") if streams is not None else None
    if stream is None:
        raise ValueError("Analysis Zarr has no acquisition crop stream.")
    attrs = dict(stream.attrs)
    if attrs.get("canonical_ledger_status") != "complete":
        raise ValueError("Acquisition crop stream has no complete canonical ledger.")
    record_sha256 = str(attrs.get("canonical_ledger_record_sha256") or "")
    if expected_record_sha256 is not None and record_sha256 != expected_record_sha256:
        raise ValueError("Canonical crop ledger digest differs from the expected publication.")
    publication = AcquisitionCropLedgerPublication(
        status="complete",
        run_name=str(attrs.get("canonical_ledger_run") or ""),
        group_path=str(attrs.get("canonical_ledger_path") or ""),
        record_sha256=record_sha256,
        source_metadata_sha256=str(
            attrs.get("canonical_ledger_source_metadata_sha256") or ""
        ),
        source_video_fingerprint=str(
            attrs.get("canonical_ledger_source_video_fingerprint") or ""
        ),
        row_count=int(attrs.get("canonical_ledger_row_count")),
        detected_row_count=int(attrs.get("canonical_ledger_detected_row_count")),
        blank_row_count=int(attrs.get("canonical_ledger_blank_row_count")),
        imported_at_utc=str(attrs.get("canonical_ledger_imported_at_utc") or ""),
    )
    if not publication.run_name or not publication.record_sha256:
        raise ValueError("Canonical crop ledger pointer identity is incomplete.")
    _validate_current_run(stream, publication)
    return publication


__all__ = [
    "ACQUISITION_CROP_LEDGER_RUNS_GROUP",
    "ACQUISITION_CROP_LEDGER_SCHEMA_ID",
    "ACQUISITION_CROP_LEDGER_SCHEMA_VERSION",
    "AcquisitionCropLedgerPublication",
    "publish_acquisition_crop_stream_ledger",
    "validate_current_acquisition_crop_stream_ledger",
]
