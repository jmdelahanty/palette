"""Immutable recording-frame clock authority for imported Palette Zarrs.

The clock authority is deliberately separate from the acquisition pixel-frame
authority.  ``recording_frame_id`` and ``parent_frame_index`` identify the
temporal row; Orange ``timestamp`` and ``timestamp_sys`` retain the camera/PTP
and host/system clocks associated with that row.  Sampled image rows join this
authority through ``raw_video/original_frame_indices`` rather than carrying a
second, potentially ambiguous timestamp array.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from fisheye.shared.import_source_fingerprint import optional_source_stat_fingerprint_attrs
from fisheye.shared.json_safety import json_attr_safe_mapping, strict_json_dumps
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.columnar import store_array
from fisheye.shared.zarr_run_completion import (
    is_run_complete_in_parent,
    is_run_selector_eligible,
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
    resolve_latest_complete_run_group,
)


ACQUISITION_FRAME_CLOCK_SCHEMA_ID = "palette.acquisition_frame_clock.v1"
ACQUISITION_FRAME_CLOCK_SCHEMA_VERSION = 1
ACQUISITION_FRAME_CLOCK_RUNS_PATH = "analysis/acquisition_frame_clock_runs"
ACQUISITION_FRAME_CLOCK_RECORD_ATTR = "acquisition_frame_clock_record"
ACQUISITION_FRAME_CLOCK_SHA256_ATTR = "acquisition_frame_clock_sha256"
ACQUISITION_FRAME_CLOCK_SHARD_ROWS = 131_072
MISSING_TIMESTAMP_SENTINEL = np.iinfo(np.int64).min

_ARRAY_NAMES = (
    "recording_frame_id",
    "parent_frame_index",
    "camera_timestamp_ns",
    "system_timestamp_ns",
    "camera_timestamp_valid",
    "system_timestamp_valid",
)


class AcquisitionFrameClockError(ValueError):
    """Raised when an acquisition clock source or publication is inconsistent."""


@dataclass(frozen=True)
class AcquisitionFrameClockSource:
    source_path: Path
    source_kind: str
    source_locator: str
    camera_id: str
    recording_frame_id: np.ndarray
    parent_frame_index: np.ndarray
    camera_timestamp_ns: np.ndarray
    system_timestamp_ns: np.ndarray
    camera_timestamp_valid: np.ndarray
    system_timestamp_valid: np.ndarray
    clock_surfaces: Mapping[str, Mapping[str, Any]]
    clock_semantic_evidence: Mapping[str, Any]

    @property
    def row_count(self) -> int:
        return int(self.recording_frame_id.shape[0])


@dataclass(frozen=True)
class ResolvedAcquisitionFrameClock:
    run_name: str
    group_path: str
    record: Mapping[str, Any]
    record_sha256: str
    camera_id: str
    row_count: int


def _safe_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _recording_relative_locator(recording_dir: Path, source_path: Path) -> str:
    root = recording_dir.expanduser().resolve()
    source = source_path.expanduser().resolve()
    try:
        return source.relative_to(root).as_posix()
    except ValueError:
        return str(source)


def _resolve_recording_path(recording_dir: Path, value: object) -> Path:
    text = str(value or "").strip()
    if not text:
        raise AcquisitionFrameClockError("Frame-clock source path is empty.")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root = recording_dir.expanduser().resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise AcquisitionFrameClockError(
            f"Recording-relative frame-clock path escapes the recording root: {value!r}"
        )
    return resolved


def _manifest_clock_csv(
    recording_dir: Path,
    *,
    video_path: Path,
) -> tuple[Path | None, bool]:
    """Return a manifest-declared full-stream clock CSV and whether it was declared."""

    manifest = _safe_json_object(recording_dir / "recording_manifest.json")
    video_streams = manifest.get("video_streams")
    if not isinstance(video_streams, Mapping):
        return None, False
    streams = video_streams.get("streams")
    if not isinstance(streams, Mapping):
        return None, False

    wanted = video_path.expanduser().resolve()
    selected: Mapping[str, Any] | None = None
    for value in streams.values():
        if not isinstance(value, Mapping):
            continue
        raw_video = value.get("video")
        if raw_video in (None, ""):
            continue
        candidate = _resolve_recording_path(recording_dir, raw_video)
        if candidate == wanted:
            selected = value
            break
    if selected is None:
        fallback = streams.get("full")
        selected = fallback if isinstance(fallback, Mapping) else None
    if selected is None:
        return None, False

    raw_clock = selected.get("frame_clock_metadata")
    if raw_clock in (None, ""):
        return None, False
    return _resolve_recording_path(recording_dir, raw_clock), True


def _parse_optional_int(value: object, *, label: str, row_number: int) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise AcquisitionFrameClockError(
            f"Invalid {label} at row {row_number}: {text!r}"
        ) from exc


def _timestamp_array(values: Sequence[int | None]) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray([value is not None for value in values], dtype=np.bool_)
    data = np.asarray(
        [MISSING_TIMESTAMP_SENTINEL if value is None else int(value) for value in values],
        dtype=np.int64,
    )
    return data, valid


def _clock_summary_path(recording_dir: Path) -> Path | None:
    for candidate in (
        recording_dir / "raw" / "ptp_sync_summary.json",
        recording_dir / "ptp_sync_summary.json",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _matching_camera_summary(
    payload: Mapping[str, Any], camera_id: str
) -> Mapping[str, Any]:
    cameras = payload.get("cameras")
    if not isinstance(cameras, Mapping):
        return {}
    wanted = str(camera_id).removeprefix("Cam")
    for key, value in cameras.items():
        if not isinstance(value, Mapping):
            continue
        candidates = {
            str(key).removeprefix("Cam"),
            str(value.get("camera_serial") or "").removeprefix("Cam"),
        }
        if wanted in candidates:
            return value
    return {}


def _valid_clock_delta_summary(
    camera: np.ndarray,
    camera_valid: np.ndarray,
    system: np.ndarray,
    system_valid: np.ndarray,
) -> dict[str, int] | None:
    shared = np.asarray(camera_valid, dtype=np.bool_) & np.asarray(
        system_valid, dtype=np.bool_
    )
    if not np.any(shared):
        return None
    deltas = np.asarray(camera[shared], dtype=np.int64) - np.asarray(
        system[shared], dtype=np.int64
    )
    return {
        "sample_count": int(deltas.size),
        "median_ns": int(np.median(deltas)),
        "minimum_ns": int(np.min(deltas)),
        "maximum_ns": int(np.max(deltas)),
    }


def _clock_semantics(
    recording_dir: Path,
    *,
    camera_id: str,
    camera: np.ndarray,
    camera_valid: np.ndarray,
    system: np.ndarray,
    system_valid: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Describe clock domains without upgrading unsupported epoch assumptions."""

    summary_path = _clock_summary_path(recording_dir)
    summary = _safe_json_object(summary_path) if summary_path is not None else {}
    camera_summary = _matching_camera_summary(summary, camera_id)
    sync = summary.get("sync") if isinstance(summary.get("sync"), Mapping) else {}
    delta = _valid_clock_delta_summary(camera, camera_valid, system, system_valid)

    sync_mode = str(sync.get("mode") or "").strip().lower()
    sync_enabled = bool(
        camera_summary.get("sync_camera_enabled") is True
        and sync.get("camera_sync_enabled") is True
        and sync_mode.startswith("ptp")
    )
    register_reads = int(camera_summary.get("ptp_register_reads") or 0)
    ptp_offset = (
        camera_summary.get("ptp_offset_ns")
        if isinstance(camera_summary.get("ptp_offset_ns"), Mapping)
        else {}
    )
    ptp_offset_samples = int(ptp_offset.get("samples") or 0)
    ptp_offset_extrema = [
        abs(int(ptp_offset[field]))
        for field in ("minimum", "min", "maximum", "max")
        if ptp_offset.get(field) is not None
    ]
    ptp_offsets_indicate_sync = bool(
        ptp_offset_samples > 0
        and ptp_offset_extrema
        and max(ptp_offset_extrema) <= 1_000_000
    )
    latch_agreement = (
        camera_summary.get("latch_minus_frame_ns")
        if isinstance(camera_summary.get("latch_minus_frame_ns"), Mapping)
        else {}
    )
    latch_samples = (
        int(latch_agreement.get("samples") or 0)
    )
    latch_extrema = [
        abs(int(latch_agreement[field]))
        for field in ("minimum", "min", "maximum", "max")
        if latch_agreement.get(field) is not None
    ]
    latch_agrees_with_frames = bool(
        latch_samples > 0
        and latch_extrema
        and max(latch_extrema) < 1_000_000_000
    )
    explicit_ptp_status = str(
        camera_summary.get("ptp_status")
        or camera_summary.get("ptp_state")
        or camera_summary.get("gev_ieee1588_status")
        or ""
    ).strip()
    normalized_ptp_status = explicit_ptp_status.lower()
    explicit_status_does_not_contradict_sync = bool(
        not explicit_ptp_status
        or any(
            marker in normalized_ptp_status
            for marker in ("locked", "slave", "master", "synchronized")
        )
    )
    expected_tai_minus_utc_ns = 37_000_000_000
    expected_tai_utc_offset = bool(
        delta is not None
        and abs(int(delta["median_ns"]) - expected_tai_minus_utc_ns)
        < 1_000_000_000
        and int(delta["maximum_ns"]) - int(delta["minimum_ns"]) < 1_000_000_000
    )
    ptp_operationally_inferred = bool(
        sync_enabled
        and register_reads > 0
        and ptp_offsets_indicate_sync
        and latch_agrees_with_frames
        and expected_tai_utc_offset
        and explicit_status_does_not_contradict_sync
    )

    evidence: dict[str, Any] = {
        "camera_ptp_semantics_inferred": ptp_operationally_inferred,
        "camera_system_delta_ns": delta,
        "classifier_expected_tai_minus_utc_ns": expected_tai_minus_utc_ns,
        "classifier_tai_utc_tolerance_ns": 1_000_000_000,
        "ptp_sync_enabled": sync_enabled,
        "ptp_sync_mode": sync_mode or "unspecified",
        "ptp_register_reads": register_reads,
        "ptp_offset_samples": ptp_offset_samples,
        "ptp_offset_max_abs_ns": (
            max(ptp_offset_extrema) if ptp_offset_extrema else None
        ),
        "ptp_offsets_indicate_synchronization": ptp_offsets_indicate_sync,
        "ptp_latch_agreement_samples": latch_samples,
        "ptp_latch_max_abs_ns": max(latch_extrema) if latch_extrema else None,
        "ptp_latch_agrees_with_embedded_frame_time": latch_agrees_with_frames,
        "explicit_ptp_status": explicit_ptp_status or "not_recorded",
        "explicit_ptp_status_does_not_contradict_synchronization": (
            explicit_status_does_not_contradict_sync
        ),
    }
    if summary_path is not None:
        stat = summary_path.stat()
        evidence["ptp_sync_summary"] = {
            "locator": _recording_relative_locator(recording_dir, summary_path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    if ptp_operationally_inferred:
        camera_semantics: dict[str, Any] = {
            "source_field": "timestamp",
            "source_api": "Emergent::CEmergentFrame.timestamp",
            "producer": "camera_firmware_via_emergent_esdk",
            "orange_transformation": "none",
            "clock_role": "camera_embedded_frame_time",
            "clock_domain": "camera_hardware_ptp_clock",
            "unit": "nanosecond",
            "time_reference_kind": "absolute_epoch",
            "origin": "1970-01-01T00:00:00_TAI",
            "timescale": "IEEE-1588_PTP_TAI",
            "semantic_status": "inferred_from_recording_evidence_not_sdk_declared",
            "validity_array": "camera_timestamp_valid",
        }
    else:
        camera_semantics = {
            "source_field": "timestamp",
            "source_api": "Emergent::CEmergentFrame.timestamp",
            "producer": "camera_firmware_via_emergent_esdk",
            "orange_transformation": "none",
            "clock_role": "camera_embedded_frame_time",
            "clock_domain": "camera_hardware_clock",
            "unit": "nanosecond",
            "time_reference_kind": "device_defined_unknown_epoch",
            "origin": "unspecified",
            "timescale": "unspecified",
            "semantic_status": "producer_passthrough_epoch_not_established",
            "validity_array": "camera_timestamp_valid",
        }

    system_semantics = {
        "source_field": "timestamp_sys",
        "source_api": "clock_gettime(CLOCK_REALTIME)",
        "producer": "orange_acquisition_thread",
        "capture_point": "immediately_after_EVT_CameraGetFrame_returns",
        "clock_role": "host_wall_clock",
        "clock_domain": "host_CLOCK_REALTIME",
        "unit": "nanosecond",
        "time_reference_kind": "absolute_epoch",
        "origin": "1970-01-01T00:00:00_UTC",
        "timescale": "POSIX_UTC_excluding_leap_seconds",
        "semantic_status": "producer_declared_by_orange_code",
        "validity_array": "system_timestamp_valid",
    }
    return (
        {
            "camera_timestamp_ns": camera_semantics,
            "system_timestamp_ns": system_semantics,
        },
        evidence,
    )


def _frame_id_offset(first_frame_id: int) -> int:
    if int(first_frame_id) == 0:
        return 0
    if int(first_frame_id) == 1:
        return 1
    raise AcquisitionFrameClockError(
        "A complete recording clock must begin at recording_frame_id 0 or 1; "
        f"observed {first_frame_id}."
    )


def _load_csv_source(
    path: Path,
    *,
    recording_dir: Path,
    camera_id: str,
) -> AcquisitionFrameClockSource:
    frame_ids: list[int] = []
    camera_values: list[int | None] = []
    system_values: list[int | None] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(str(value).strip() for value in (reader.fieldnames or ()))
        frame_field = (
            "recording_frame_id"
            if "recording_frame_id" in fields
            else "frame_id"
            if "frame_id" in fields
            else "local_frame_id"
            if "local_frame_id" in fields
            else None
        )
        if frame_field is None or "timestamp" not in fields or "timestamp_sys" not in fields:
            raise AcquisitionFrameClockError(
                f"Frame-clock CSV {path} must contain recording_frame_id/frame_id, "
                "timestamp, and timestamp_sys."
            )
        for row_number, row in enumerate(reader, start=2):
            frame_id = _parse_optional_int(
                row.get(frame_field), label=frame_field, row_number=row_number
            )
            if frame_id is None:
                raise AcquisitionFrameClockError(
                    f"Missing {frame_field} at row {row_number}."
                )
            frame_ids.append(frame_id)
            camera_values.append(
                _parse_optional_int(
                    row.get("timestamp"), label="timestamp", row_number=row_number
                )
            )
            system_values.append(
                _parse_optional_int(
                    row.get("timestamp_sys"),
                    label="timestamp_sys",
                    row_number=row_number,
                )
            )

    if not frame_ids:
        raise AcquisitionFrameClockError(f"Frame-clock CSV has no data rows: {path}")
    recording_ids = np.asarray(frame_ids, dtype=np.int64)
    offset = _frame_id_offset(int(recording_ids[0]))
    parent = recording_ids - np.int64(offset)
    camera, camera_valid = _timestamp_array(camera_values)
    system, system_valid = _timestamp_array(system_values)
    clock_surfaces, semantic_evidence = _clock_semantics(
        recording_dir,
        camera_id=str(camera_id),
        camera=camera,
        camera_valid=camera_valid,
        system=system,
        system_valid=system_valid,
    )
    return AcquisitionFrameClockSource(
        source_path=path.resolve(),
        source_kind="orange_camera_metadata_csv",
        source_locator=_recording_relative_locator(recording_dir, path),
        camera_id=str(camera_id),
        recording_frame_id=recording_ids,
        parent_frame_index=parent,
        camera_timestamp_ns=camera,
        system_timestamp_ns=system,
        camera_timestamp_valid=camera_valid,
        system_timestamp_valid=system_valid,
        clock_surfaces=clock_surfaces,
        clock_semantic_evidence=semantic_evidence,
    )


def _nullable_arrow_ints(values: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    normalized: list[int | None] = []
    for value in values:
        normalized.append(None if value is None else int(value))
    return _timestamp_array(normalized)


def _load_parquet_source(
    path: Path,
    *,
    recording_dir: Path,
    camera_id: str,
) -> AcquisitionFrameClockSource:
    required = (
        "camera_serial",
        "recording_frame_id",
        "parent_frame_index",
        "timestamp",
        "timestamp_sys",
    )
    table = pq.read_table(path, columns=list(required))
    columns = table.to_pydict()
    indices = [
        index
        for index, value in enumerate(columns["camera_serial"])
        if str(value) == str(camera_id)
    ]
    if not indices:
        raise AcquisitionFrameClockError(
            f"Recording frame index {path} has no rows for camera {camera_id}."
        )
    recording_ids = np.asarray(
        [columns["recording_frame_id"][index] for index in indices], dtype=np.int64
    )
    parent = np.asarray(
        [columns["parent_frame_index"][index] for index in indices], dtype=np.int64
    )
    camera, camera_valid = _nullable_arrow_ints(
        [columns["timestamp"][index] for index in indices]
    )
    system, system_valid = _nullable_arrow_ints(
        [columns["timestamp_sys"][index] for index in indices]
    )
    clock_surfaces, semantic_evidence = _clock_semantics(
        recording_dir,
        camera_id=str(camera_id),
        camera=camera,
        camera_valid=camera_valid,
        system=system,
        system_valid=system_valid,
    )
    return AcquisitionFrameClockSource(
        source_path=path.resolve(),
        source_kind="recording_frame_index_parquet",
        source_locator=_recording_relative_locator(recording_dir, path),
        camera_id=str(camera_id),
        recording_frame_id=recording_ids,
        parent_frame_index=parent,
        camera_timestamp_ns=camera,
        system_timestamp_ns=system,
        camera_timestamp_valid=camera_valid,
        system_timestamp_valid=system_valid,
        clock_surfaces=clock_surfaces,
        clock_semantic_evidence=semantic_evidence,
    )


def _validate_source(
    source: AcquisitionFrameClockSource,
    *,
    expected_frame_count: int | None,
) -> AcquisitionFrameClockSource:
    arrays = {
        name: np.asarray(getattr(source, name))
        for name in _ARRAY_NAMES
    }
    row_count = source.row_count
    if row_count <= 0:
        raise AcquisitionFrameClockError("Acquisition frame clock is empty.")
    if any(array.ndim != 1 or int(array.shape[0]) != row_count for array in arrays.values()):
        raise AcquisitionFrameClockError(
            "Acquisition frame-clock arrays must be aligned one-dimensional vectors."
        )
    if expected_frame_count is not None and row_count != int(expected_frame_count):
        raise AcquisitionFrameClockError(
            "Acquisition frame-clock row count does not match the source video: "
            f"clock={row_count}, video={int(expected_frame_count)}."
        )
    if not np.array_equal(source.parent_frame_index, np.arange(row_count, dtype=np.int64)):
        raise AcquisitionFrameClockError(
            "Acquisition frame-clock parent_frame_index must be the complete ordered "
            "zero-based video-frame domain."
        )
    if row_count > 1 and not np.all(np.diff(source.recording_frame_id) == 1):
        raise AcquisitionFrameClockError(
            "Acquisition frame-clock recording_frame_id values must be contiguous."
        )
    if not bool(np.any(source.camera_timestamp_valid) or np.any(source.system_timestamp_valid)):
        raise AcquisitionFrameClockError(
            "Acquisition frame-clock source contains no usable camera or system timestamps."
        )
    for label, values, valid in (
        ("camera_timestamp_ns", source.camera_timestamp_ns, source.camera_timestamp_valid),
        ("system_timestamp_ns", source.system_timestamp_ns, source.system_timestamp_valid),
    ):
        selected = np.asarray(values[valid], dtype=np.int64)
        if selected.size > 1 and np.any(np.diff(selected) < 0):
            raise AcquisitionFrameClockError(f"{label} must be monotonic.")
    return source


def load_acquisition_frame_clock_source(
    recording_dir: str | Path,
    *,
    camera_id: str,
    video_path: str | Path,
    expected_frame_count: int | None = None,
) -> AcquisitionFrameClockSource | None:
    """Load the authoritative Orange clock source available for one camera.

    A manifest-declared path is fail-closed.  Conventional sibling CSVs are the
    single-video fallback; a recording-level Parquet index is used for clipped
    recordings.  H5 stimulus timestamps are intentionally not relabeled as
    camera acquisition timestamps.
    """

    root = Path(recording_dir).expanduser().resolve()
    video = Path(video_path).expanduser().resolve()
    declared_path, declared = _manifest_clock_csv(root, video_path=video)
    if declared:
        assert declared_path is not None
        if not declared_path.is_file():
            raise FileNotFoundError(
                f"Manifest-declared frame-clock metadata is missing: {declared_path}"
            )
        return _validate_source(
            _load_csv_source(declared_path, recording_dir=root, camera_id=str(camera_id)),
            expected_frame_count=expected_frame_count,
        )

    conventional = video.with_name(f"{video.stem}_meta.csv")
    if conventional.is_file():
        return _validate_source(
            _load_csv_source(conventional, recording_dir=root, camera_id=str(camera_id)),
            expected_frame_count=expected_frame_count,
        )

    frame_index = root / "recording_frame_index.parquet"
    if frame_index.is_file():
        return _validate_source(
            _load_parquet_source(frame_index, recording_dir=root, camera_id=str(camera_id)),
            expected_frame_count=expected_frame_count,
        )
    return None


def _array_digests(source: AcquisitionFrameClockSource) -> dict[str, str]:
    return {
        name: _array_values_sha256(np.asarray(getattr(source, name)))
        for name in _ARRAY_NAMES
    }


def _array_values_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    if array.dtype.hasobject:
        raise AcquisitionFrameClockError(
            "Object arrays do not have deterministic acquisition-clock bytes."
        )
    header = {
        "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        "dtype": np.lib.format.dtype_to_descr(array.dtype),
        "shape": [int(value) for value in array.shape],
    }
    digest = sha256()
    digest.update(strict_json_dumps(header).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _source_evidence(source: AcquisitionFrameClockSource) -> dict[str, Any]:
    stat = source.source_path.stat()
    return {
        "kind": source.source_kind,
        "locator": source.source_locator,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_record(source: AcquisitionFrameClockSource) -> dict[str, Any]:
    return {
        "schema_id": ACQUISITION_FRAME_CLOCK_SCHEMA_ID,
        "schema_version": ACQUISITION_FRAME_CLOCK_SCHEMA_VERSION,
        "camera_id": source.camera_id,
        "row_count": source.row_count,
        "frame_domain": {
            "recording_frame_id": "session_continuous_source_identifier",
            "parent_frame_index": "zero_based_complete_video_frame_index",
            "sampled_row_join": "raw_video/original_frame_indices -> parent_frame_index",
        },
        "clock_surfaces": json_attr_safe_mapping(source.clock_surfaces),
        "clock_semantic_evidence": json_attr_safe_mapping(
            source.clock_semantic_evidence
        ),
        "missing_timestamp_sentinel": int(MISSING_TIMESTAMP_SENTINEL),
        "source": _source_evidence(source),
        "array_sha256": _array_digests(source),
        "array_canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
    }


def acquisition_frame_clock_sha256(record: Mapping[str, Any]) -> str:
    return sha256(strict_json_dumps(record).encode("utf-8")).hexdigest()


def _validate_record(record: Mapping[str, Any], digest: str | None = None) -> dict[str, Any]:
    canonical = json_attr_safe_mapping(record)
    if canonical.get("schema_id") != ACQUISITION_FRAME_CLOCK_SCHEMA_ID:
        raise AcquisitionFrameClockError("Unsupported acquisition frame-clock schema_id.")
    if canonical.get("schema_version") != ACQUISITION_FRAME_CLOCK_SCHEMA_VERSION:
        raise AcquisitionFrameClockError("Unsupported acquisition frame-clock schema_version.")
    if not str(canonical.get("camera_id") or "").strip():
        raise AcquisitionFrameClockError("Acquisition frame clock has no camera_id.")
    try:
        row_count = int(canonical.get("row_count"))
    except (TypeError, ValueError) as exc:
        raise AcquisitionFrameClockError("Acquisition frame clock has invalid row_count.") from exc
    if row_count <= 0:
        raise AcquisitionFrameClockError("Acquisition frame clock row_count must be positive.")
    clock_surfaces = canonical.get("clock_surfaces")
    if not isinstance(clock_surfaces, Mapping):
        raise AcquisitionFrameClockError("Acquisition frame clock has no clock semantics.")
    expected_clock_fields = {
        "camera_timestamp_ns": "camera_timestamp_valid",
        "system_timestamp_ns": "system_timestamp_valid",
    }
    for name, validity_array in expected_clock_fields.items():
        surface = clock_surfaces.get(name)
        if not isinstance(surface, Mapping):
            raise AcquisitionFrameClockError(
                f"Acquisition frame clock has no semantics for {name}."
            )
        for field in (
            "clock_domain",
            "unit",
            "time_reference_kind",
            "origin",
            "timescale",
            "semantic_status",
        ):
            if not str(surface.get(field) or "").strip():
                raise AcquisitionFrameClockError(
                    f"Acquisition frame-clock semantics for {name} omit {field}."
                )
        if surface.get("unit") != "nanosecond":
            raise AcquisitionFrameClockError(
                f"Acquisition frame-clock semantics for {name} have unsupported units."
            )
        if surface.get("validity_array") != validity_array:
            raise AcquisitionFrameClockError(
                f"Acquisition frame-clock semantics for {name} bind the wrong validity array."
            )
    array_digests = canonical.get("array_sha256")
    if not isinstance(array_digests, Mapping) or set(array_digests) != set(_ARRAY_NAMES):
        raise AcquisitionFrameClockError("Acquisition frame clock has incomplete array digests.")
    actual = acquisition_frame_clock_sha256(canonical)
    if digest is not None and str(digest) != actual:
        raise AcquisitionFrameClockError(
            f"Acquisition frame-clock digest mismatch: stored={digest!r}, computed={actual!r}."
        )
    return canonical


def _validate_run(parent: Any, run_name: str, run: Any) -> ResolvedAcquisitionFrameClock:
    if not is_run_complete_in_parent(parent, run, legacy_default=False):
        raise AcquisitionFrameClockError(f"Acquisition frame-clock run {run_name!r} is incomplete.")
    if not is_run_selector_eligible(run):
        raise AcquisitionFrameClockError(
            f"Acquisition frame-clock run {run_name!r} is not selector eligible."
        )
    raw_record = run.attrs.get(ACQUISITION_FRAME_CLOCK_RECORD_ATTR)
    if not isinstance(raw_record, Mapping):
        raise AcquisitionFrameClockError(f"Acquisition frame-clock run {run_name!r} has no record.")
    digest = str(run.attrs.get(ACQUISITION_FRAME_CLOCK_SHA256_ATTR) or "")
    record = _validate_record(raw_record, digest)
    row_count = int(record["row_count"])
    expected_digests = record["array_sha256"]
    for name in _ARRAY_NAMES:
        if name not in run:
            raise AcquisitionFrameClockError(
                f"Acquisition frame-clock run {run_name!r} is missing {name}."
            )
        values = np.asarray(run[name][:])
        if values.ndim != 1 or int(values.shape[0]) != row_count:
            raise AcquisitionFrameClockError(
                f"Acquisition frame-clock array {name} has an invalid shape."
            )
        if _array_values_sha256(values) != str(expected_digests[name]):
            raise AcquisitionFrameClockError(
                f"Acquisition frame-clock array {name} differs from its bound digest."
            )
    return ResolvedAcquisitionFrameClock(
        run_name=run_name,
        group_path=f"{ACQUISITION_FRAME_CLOCK_RUNS_PATH}/{run_name}",
        record=record,
        record_sha256=digest,
        camera_id=str(record["camera_id"]),
        row_count=row_count,
    )


def _stamp_root_binding(root: Any, resolved: ResolvedAcquisitionFrameClock) -> None:
    root.attrs.update(
        {
            "acquisition_frame_clock_available": True,
            "acquisition_frame_clock_status": "complete",
            "acquisition_frame_clock_ref": resolved.group_path,
            "acquisition_frame_clock_sha256": resolved.record_sha256,
            "acquisition_frame_clock_camera_id": resolved.camera_id,
            "acquisition_frame_clock_row_count": resolved.row_count,
        }
    )


def resolve_acquisition_frame_clock(
    root: Any,
    *,
    required: bool = True,
) -> ResolvedAcquisitionFrameClock | None:
    analysis = root.get("analysis")
    parent = analysis.get("acquisition_frame_clock_runs") if analysis is not None else None
    if parent is None:
        if required:
            raise AcquisitionFrameClockError(f"Missing {ACQUISITION_FRAME_CLOCK_RUNS_PATH}.")
        return None
    run_name, run = resolve_latest_complete_run_group(parent, legacy_default=False)
    if run_name is None or run is None:
        raise AcquisitionFrameClockError(
            f"{ACQUISITION_FRAME_CLOCK_RUNS_PATH} has no selected complete run."
        )
    resolved = _validate_run(parent, run_name, run)
    bindings = {
        "acquisition_frame_clock_ref": resolved.group_path,
        "acquisition_frame_clock_sha256": resolved.record_sha256,
        "acquisition_frame_clock_camera_id": resolved.camera_id,
        "acquisition_frame_clock_row_count": resolved.row_count,
    }
    for name, expected in bindings.items():
        if root.attrs.get(name) != expected:
            raise AcquisitionFrameClockError(
                f"Root {name} does not bind the selected acquisition frame clock."
            )
    if root.attrs.get("acquisition_frame_clock_available") is not True:
        raise AcquisitionFrameClockError(
            "Root does not mark the selected acquisition frame clock available."
        )
    return resolved


def publish_acquisition_frame_clock(root: Any, source: AcquisitionFrameClockSource) -> ResolvedAcquisitionFrameClock:
    """Idempotently publish and bind one immutable acquisition clock."""

    source = _validate_source(source, expected_frame_count=None)
    record = _validate_record(_build_record(source))
    digest = acquisition_frame_clock_sha256(record)
    run_name = f"acquisition_frame_clock_{digest[:16]}"
    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, "acquisition_frame_clock_runs")

    if run_name in parent:
        existing = parent[run_name]
        resolved = _validate_run(parent, run_name, existing)
        if resolved.record_sha256 != digest:
            raise AcquisitionFrameClockError(
                f"Existing acquisition frame-clock run {run_name!r} conflicts."
            )
        parent.attrs["latest_complete"] = run_name
        parent.attrs["latest"] = run_name
        _stamp_root_binding(root, resolved)
        return resolve_acquisition_frame_clock(root, required=True)  # type: ignore[return-value]

    run = parent.create_group(run_name)
    mark_run_started(run, run_name=run_name, stage="acquisition_frame_clock")
    note_pending_latest(parent, run_name)
    run.attrs["stage_selector_eligible"] = False
    run.attrs["schema_id"] = ACQUISITION_FRAME_CLOCK_SCHEMA_ID
    run.attrs["schema_version"] = ACQUISITION_FRAME_CLOCK_SCHEMA_VERSION
    run.attrs[ACQUISITION_FRAME_CLOCK_RECORD_ATTR] = record
    run.attrs[ACQUISITION_FRAME_CLOCK_SHA256_ATTR] = digest
    run.attrs["immutable"] = True
    for name in _ARRAY_NAMES:
        store_array(
            run,
            name,
            np.asarray(getattr(source, name)),
            shard_rows=ACQUISITION_FRAME_CLOCK_SHARD_ROWS,
        )
    run.attrs["stage_selector_eligible"] = True
    source_fingerprint = optional_source_stat_fingerprint_attrs(
        source.source_path,
        attr_prefix="source_frame_clock",
    ).get("source_frame_clock_fingerprint")
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=run_name,
        run_provenance=build_writer_run_provenance(
            command="import:publish_acquisition_frame_clock",
            params={
                "schema_id": ACQUISITION_FRAME_CLOCK_SCHEMA_ID,
                "record_sha256": digest,
                "camera_id": source.camera_id,
                "row_count": source.row_count,
                "source_kind": source.source_kind,
            },
            input_run_ids={},
            input_artifacts=[
                {
                    "role": "acquisition_frame_clock_source",
                    "kind": source.source_kind,
                    "path": str(source.source_path),
                    "stat_fingerprint": source_fingerprint,
                }
            ],
        ),
    )
    resolved = _validate_run(parent, run_name, run)
    _stamp_root_binding(root, resolved)
    return resolve_acquisition_frame_clock(root, required=True)  # type: ignore[return-value]


def import_acquisition_frame_clock(
    root: Any,
    *,
    recording_dir: str | Path,
    camera_id: str,
    video_path: str | Path,
    expected_frame_count: int | None = None,
) -> ResolvedAcquisitionFrameClock | None:
    """Import a recording clock when a source exists, otherwise mark absence."""

    source = load_acquisition_frame_clock_source(
        recording_dir,
        camera_id=str(camera_id),
        video_path=video_path,
        expected_frame_count=expected_frame_count,
    )
    if source is None:
        existing = resolve_acquisition_frame_clock(root, required=False)
        if existing is not None:
            return existing
        root.attrs.update(
            {
                "acquisition_frame_clock_available": False,
                "acquisition_frame_clock_status": "unavailable_no_camera_clock_source",
            }
        )
        return None
    return publish_acquisition_frame_clock(root, source)


__all__ = [
    "ACQUISITION_FRAME_CLOCK_RECORD_ATTR",
    "ACQUISITION_FRAME_CLOCK_RUNS_PATH",
    "ACQUISITION_FRAME_CLOCK_SCHEMA_ID",
    "ACQUISITION_FRAME_CLOCK_SHA256_ATTR",
    "AcquisitionFrameClockError",
    "AcquisitionFrameClockSource",
    "ResolvedAcquisitionFrameClock",
    "acquisition_frame_clock_sha256",
    "import_acquisition_frame_clock",
    "load_acquisition_frame_clock_source",
    "publish_acquisition_frame_clock",
    "resolve_acquisition_frame_clock",
]
