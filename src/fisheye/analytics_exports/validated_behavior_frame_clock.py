"""Strict source binding for session-aware acquisition frame-clock exports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from fisheye.analysis_workflows.validated_behavior_cohort_adapters import sha256_file
from fisheye.shared.acquisition_frame_clock import (
    AcquisitionFrameClockSource,
    acquisition_frame_clock_source_sha256,
    load_acquisition_frame_clock_source,
)
from fisheye.shared.pixel_frame_authority import BoundAcquisitionCameraFrame
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


class ValidatedBehaviorFrameClockError(ValueError):
    """A raw clock, session declaration, or acquisition binding is inexact."""


def _fail(message: str) -> None:
    raise ValidatedBehaviorFrameClockError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field_name} must be one object.")
    return value


def _required_text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field_name} must be non-empty normalized text.")
    return value


def _utc_timestamp(value: object, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidatedBehaviorFrameClockError(
            f"{field_name} must be an ISO-8601 timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail(f"{field_name} must declare UTC.")
    return text


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _plain(body)
    return {**normalized, "payload_sha256": canonical_json_sha256(normalized)}


def _read_json(path: Path, *, field_name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorFrameClockError(
            f"{field_name} is not readable JSON: {path}"
        ) from exc
    return _mapping(value, field_name=field_name)


def _recording_relative_path(recording_dir: Path, relative_path: object) -> Path:
    text = _required_text(relative_path, field_name="recording-relative path")
    candidate = (recording_dir / text).resolve()
    if candidate != recording_dir and recording_dir not in candidate.parents:
        _fail(f"Recording-relative path escapes its recording root: {text!r}.")
    return candidate


def _declared_frame_index(
    recording_dir: Path, source_video_metadata: Mapping[str, Any]
) -> tuple[Path | None, str | None]:
    collection = source_video_metadata.get("collection")
    if not isinstance(collection, Mapping):
        return None, None
    evidence = collection.get("recording_frame_index")
    if not isinstance(evidence, Mapping):
        _fail("Clipped source metadata lacks recording_frame_index evidence.")
    path = _recording_relative_path(recording_dir, evidence.get("relative_path"))
    digest = _required_text(
        evidence.get("sha256"),
        field_name="source_video_metadata.collection.recording_frame_index.sha256",
    )
    return path, digest


def _validate_parquet_session_identity(
    source: AcquisitionFrameClockSource,
    *,
    recording_id: str,
    session_id: str,
    camera_id: str,
) -> None:
    """Stream the raw identity columns and prove the selected camera's session."""

    if source.source_kind != "recording_frame_index_parquet":
        return
    parquet = pq.ParquetFile(source.source_path)
    required = {"recording_id", "session_id", "camera_serial"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        _fail(
            "Recording frame index lacks session identity columns: "
            f"{sorted(missing)!r}."
        )
    selected_count = 0
    for batch in parquet.iter_batches(
        batch_size=65_536,
        columns=["recording_id", "session_id", "camera_serial"],
    ):
        columns = batch.to_pydict()
        for observed_recording, observed_session, observed_camera in zip(
            columns["recording_id"],
            columns["session_id"],
            columns["camera_serial"],
            strict=True,
        ):
            if str(observed_camera) != camera_id:
                continue
            selected_count += 1
            if observed_recording != recording_id or observed_session != session_id:
                _fail(
                    "Recording frame-index row identity disagrees with its "
                    "recording manifest."
                )
    if selected_count != source.row_count:
        _fail(
            "Recording frame-index session identity count differs from the bound "
            "clock row count."
        )


def acquisition_frame_clock_projection_contract() -> dict[str, Any]:
    """Return the exact normalized clock projection and permitted join rules."""

    return _sealed(
        {
            "schema_id": "palette.acquisition_frame_clock.export_projection",
            "schema_version": 1,
            "source_schema_id": "palette.acquisition_frame_clock.v1",
            "source_grain": "one_raw_recording_camera_frame",
            "output_metadata_grain": "one_recording_clock_declaration",
            "output_sample_grain": "one_recording_camera_frame_clock_observation",
            "row_selection": "complete_ordered_parent_frame_index_domain",
            "within_recording_join": {
                "keys": ["recording_id", "source_acquisition_frame_index"]
            },
            "within_session_cross_camera_join": {
                "requirements": [
                    "equal_session_id",
                    "camera_timestamp_valid",
                    "compatible_camera_clock_domain_and_timescale",
                    "declared_timestamp_tolerance",
                ],
                "coordinate": "camera_timestamp_ns",
            },
            "cross_session_join": {
                "status": "not_implied_by_this_projection",
                "requires": "validated_traceable_absolute_clock_or_external_anchor_mapping",
            },
            "equal_frame_rate_alignment_valid": False,
            "system_timestamp_role": "host_wall_clock_diagnostic_not_primary_camera_alignment",
            "missing_timestamp_policy": (
                "interpret_timestamp_value_only_when_its_validity_flag_is_true"
            ),
        }
    )


@dataclass(frozen=True)
class BoundValidatedBehaviorFrameClock:
    """One acquisition-bound raw clock plus its sealed export declaration."""

    source: AcquisitionFrameClockSource
    source_binding: Mapping[str, Any]
    projection_contract: Mapping[str, Any]


def bind_validated_behavior_frame_clock(
    root: Any,
    *,
    analysis_zarr: str | Path,
    expected_recording_id: str,
    acquisition: BoundAcquisitionCameraFrame,
) -> BoundValidatedBehaviorFrameClock:
    """Bind raw timestamps to the exact admitted acquisition camera authority."""

    attrs = getattr(root, "attrs", None)
    if not isinstance(attrs, Mapping):
        _fail("Analysis Zarr root does not expose mapping attributes.")
    record = acquisition.record
    recording_id = _required_text(
        expected_recording_id, field_name="expected_recording_id"
    )
    if record.recording_id != recording_id:
        _fail("Acquisition clock request names another recording authority.")
    camera_id = _required_text(record.camera_id, field_name="acquisition camera_id")

    recording_dir = (
        Path(
            _required_text(
                attrs.get("recording_path"), field_name="root recording_path"
            )
        )
        .expanduser()
        .resolve()
    )
    if not recording_dir.is_dir():
        raise FileNotFoundError(
            f"Raw recording directory does not exist: {recording_dir}"
        )
    root_recording_id = _required_text(
        attrs.get("recording_id"), field_name="root recording_id"
    )
    if root_recording_id != recording_id:
        _fail("Analysis root recording_id disagrees with acquisition authority.")

    raw_manifest_path = (recording_dir / "recording_manifest.json").resolve()
    raw_manifest = _read_json(raw_manifest_path, field_name="recording manifest")
    if raw_manifest.get("recording_id") != recording_id:
        _fail("Recording manifest recording_id disagrees with acquisition authority.")
    if str(raw_manifest.get("camera_id")) != camera_id:
        _fail("Recording manifest camera_id disagrees with acquisition authority.")
    session_id = _required_text(
        raw_manifest.get("orange_session_id"),
        field_name="recording_manifest.orange_session_id",
    )
    session_start = _utc_timestamp(
        raw_manifest.get("session_start_iso8601_utc"),
        field_name="recording_manifest.session_start_iso8601_utc",
    )
    root_session = attrs.get("session_id")
    if root_session not in (None, "") and str(root_session) != session_id:
        _fail("Analysis root session_id disagrees with the Orange session ID.")

    source_metadata = _mapping(
        record.source_video_metadata, field_name="source_video_metadata"
    )
    declared_path, declared_sha256 = _declared_frame_index(
        recording_dir, source_metadata
    )
    source = load_acquisition_frame_clock_source(
        recording_dir,
        camera_id=camera_id,
        video_path=Path(analysis_zarr).expanduser().resolve(),
        expected_frame_count=int(record.source_total_frames),
    )
    if source is None:
        _fail("No acquisition frame-clock source is available for this recording.")
    if source.camera_id != camera_id:
        _fail("Frame-clock source camera differs from acquisition authority.")
    if declared_path is not None and source.source_path != declared_path:
        _fail("Loaded frame-clock source differs from acquisition metadata evidence.")
    source_file_sha256 = sha256_file(source.source_path)
    if declared_sha256 is not None and source_file_sha256 != declared_sha256:
        _fail("Frame-clock source digest differs from acquisition metadata evidence.")
    _validate_parquet_session_identity(
        source,
        recording_id=recording_id,
        session_id=session_id,
        camera_id=camera_id,
    )

    semantics = {
        "clock_surfaces": _plain(source.clock_surfaces),
        "clock_semantic_evidence": _plain(source.clock_semantic_evidence),
    }
    camera_semantics = _mapping(
        source.clock_surfaces.get("camera_timestamp_ns"),
        field_name="camera_timestamp_ns semantics",
    )
    system_semantics = _mapping(
        source.clock_surfaces.get("system_timestamp_ns"),
        field_name="system_timestamp_ns semantics",
    )
    ptp_supported = bool(
        source.clock_semantic_evidence.get("camera_ptp_semantics_inferred") is True
        and camera_semantics.get("clock_domain") == "camera_hardware_ptp_clock"
    )
    within_session_status = (
        "supported_by_inferred_ptp_evidence"
        if ptp_supported
        else "recording_local_only_no_validated_shared_clock"
    )
    cross_session_status = (
        "not_validated_requires_traceable_absolute_clock_or_external_anchor"
    )

    ptp_path: Path | None = None
    ptp_evidence = source.clock_semantic_evidence.get("ptp_sync_summary")
    if isinstance(ptp_evidence, Mapping):
        ptp_path = _recording_relative_path(recording_dir, ptp_evidence.get("locator"))
        if not ptp_path.is_file():
            raise FileNotFoundError(
                f"PTP synchronization summary is missing: {ptp_path}"
            )

    binding = _sealed(
        {
            "schema_id": "palette.validated_behavior.acquisition_frame_clock_source",
            "schema_version": 1,
            "recording_id": recording_id,
            "session_id": session_id,
            "session_start_iso8601_utc": session_start,
            "camera_id": camera_id,
            "row_count": int(source.row_count),
            "analysis_zarr": str(Path(analysis_zarr).expanduser().resolve()),
            "raw_recording_path": str(recording_dir),
            "frame_clock_source_path": str(source.source_path),
            "frame_clock_source_kind": source.source_kind,
            "frame_clock_source_locator": source.source_locator,
            "frame_clock_source_file_sha256": source_file_sha256,
            "recording_manifest_path": str(raw_manifest_path),
            "recording_manifest_file_sha256": sha256_file(raw_manifest_path),
            "ptp_sync_summary_path": str(ptp_path) if ptp_path is not None else None,
            "ptp_sync_summary_file_sha256": (
                sha256_file(ptp_path) if ptp_path is not None else None
            ),
            "acquisition_camera_frame_ref": acquisition.record_ref,
            "acquisition_camera_frame_sha256": acquisition.record_sha256,
            "source_video_metadata_sha256": record.source_video_metadata_sha256,
            "acquisition_frame_clock_source_sha256": (
                acquisition_frame_clock_source_sha256(source)
            ),
            "clock_semantics": semantics,
            "clock_semantics_sha256": canonical_json_sha256(semantics),
            "within_session_alignment_status": within_session_status,
            "cross_session_alignment_status": cross_session_status,
            "equal_frame_rate_alignment_valid": False,
        }
    )
    return BoundValidatedBehaviorFrameClock(
        source=source,
        source_binding=binding,
        projection_contract=acquisition_frame_clock_projection_contract(),
    )


__all__ = [
    "BoundValidatedBehaviorFrameClock",
    "ValidatedBehaviorFrameClockError",
    "acquisition_frame_clock_projection_contract",
    "bind_validated_behavior_frame_clock",
]
