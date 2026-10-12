"""Immutable recording-frame clock authority for imported Palette Zarrs.

The clock authority is deliberately separate from the acquisition pixel-frame
authority.  ``recording_frame_id`` and ``parent_frame_index`` identify the
temporal row; Orange ``timestamp`` and ``timestamp_sys`` retain the camera/PTP
and host/system clocks associated with that row.  Sampled image rows join this
authority through ``raw_video/original_frame_indices`` rather than carrying a
second, potentially ambiguous timestamp array.
"""

from __future__ import annotations

import copy
import csv
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pyarrow.parquet as pq

from fisheye.shared.archive_identity import archive_identity
from fisheye.shared.import_source_fingerprint import optional_source_stat_fingerprint_attrs
from fisheye.shared.json_safety import json_attr_safe_mapping, strict_json_dumps
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.selector_activation import (
    DeferredSelectorActivation,
    activate_selector_eligible_run,
    commit_deferred_selector_activation,
    rollback_deferred_selector_activation,
)
from fisheye.shared.zarr.columnar import store_array
from fisheye.shared.zarr.whole_array_read import read_whole_array
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
    zarr_store_metadata_publication_lock,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    is_run_complete_in_parent,
    is_run_selector_eligible,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    require_runs_parent,
    resolve_latest_complete_run_group,
)


# v1: Palette infers the camera timescale. v2 (2026-10-12, agent-contracts
# orange-palette/camera_exposure_host_clock_v1): Palette reads Orange's
# declared TAI classification and TAI-UTC offset when Orange read them directly
# from the PTP grandmaster, and records who decided. New publications are v2;
# v1 records stay valid and are rebuilt with v1 rules (their digests are
# recomputed from source on every live re-verification).
ACQUISITION_FRAME_CLOCK_SCHEMA_IDS = {
    1: "palette.acquisition_frame_clock.v1",
    2: "palette.acquisition_frame_clock.v2",
}
ACQUISITION_FRAME_CLOCK_SCHEMA_VERSION = 2
ACQUISITION_FRAME_CLOCK_SCHEMA_ID = ACQUISITION_FRAME_CLOCK_SCHEMA_IDS[
    ACQUISITION_FRAME_CLOCK_SCHEMA_VERSION
]
ORANGE_DIRECT_PTP_RESULT = "direct_ptp_time_properties_pass"
_CAMERA_SEMANTIC_AUTHORITIES = ("orange_declared", "palette_inferred")


def is_supported_acquisition_frame_clock_schema(schema_id: Any, schema_version: Any) -> bool:
    """True for a (schema_id, schema_version) pair this code reads."""

    return (
        type(schema_version) is int
        and ACQUISITION_FRAME_CLOCK_SCHEMA_IDS.get(schema_version) == schema_id
    )
ACQUISITION_FRAME_CLOCK_RUNS_PATH = "analysis/acquisition_frame_clock_runs"
ACQUISITION_FRAME_CLOCK_RECORD_ATTR = "acquisition_frame_clock_record"
ACQUISITION_FRAME_CLOCK_SHA256_ATTR = "acquisition_frame_clock_sha256"
ACQUISITION_FRAME_CLOCK_SHARD_ROWS = 131_072
_PUBLICATION_OWNER_ATTR = "acquisition_frame_clock_publication_owner_uuid"
_PUBLICATION_POLICY_ATTR = "acquisition_frame_clock_publication_policy"
_PUBLICATION_GENERATION_ATTR = "acquisition_frame_clock_publication_generation"
_PUBLICATION_LEASE_ATTR = "acquisition_frame_clock_publication_lease"
_PUBLICATION_TOMBSTONE_ATTR = "acquisition_frame_clock_publication_tombstone"
_PUBLICATION_POLICY = "owner_generation_guarded_selectors_then_eligibility_v1"
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
    # Orange's camera clock declaration as read from recording_session.json
    # (see _orange_clock_declaration); None when the source was rebuilt from a
    # published record. v1 records never read it.
    orange_clock_declaration: Mapping[str, Any] | None = None

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
        result = int(text)
        if not -(1 << 63) <= result < (1 << 63):
            raise ValueError("outside int64 range")
        return result
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


def _ptp_tai_camera_semantics(semantic_status: str) -> dict[str, Any]:
    return {
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
        "semantic_status": semantic_status,
        "validity_array": "camera_timestamp_valid",
    }


def _device_defined_camera_semantics(semantic_status: str) -> dict[str, Any]:
    return {
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
        "semantic_status": semantic_status,
        "validity_array": "camera_timestamp_valid",
    }


def _ptp_status_readback(recording_dir: Path, camera_id: str) -> dict[str, Any] | None:
    """``cameras.<serial>.ptp_status_readback`` from ptp_sync_summary.json, verbatim.

    This is Orange's sampled PTP port-state readback ({first, last, changes,
    samples}). v1 reads status keys Orange never wrote and so never sees it;
    v2 records and checks it. Never raises: absent is None.
    """

    summary_path = _clock_summary_path(recording_dir)
    summary = _safe_json_object(summary_path) if summary_path is not None else {}
    readback = _matching_camera_summary(summary, camera_id).get("ptp_status_readback")
    return copy.deepcopy(dict(readback)) if isinstance(readback, Mapping) else None


def _readback_is_stable_slave(readback: Mapping[str, Any]) -> bool:
    changes = readback.get("changes")
    return (
        readback.get("first") == "Slave"
        and readback.get("last") == "Slave"
        and type(changes) is int
        and changes == 0
    )


def _orange_session_manifest_path(recording_dir: Path) -> Path | None:
    for candidate in (
        recording_dir / "raw" / "acquisition" / "recording_session.json",
        recording_dir / "recording_session.json",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _orange_clock_declaration(recording_dir: Path, camera_id: str) -> dict[str, Any]:
    """v2-only inputs: Orange's declared camera clock block (verbatim, plus
    where it came from) and the camera's PTP status readback.

    Never raises: a missing or unreadable manifest is ``present: False`` and
    the v2 record then falls back to Palette's inference.
    """

    declaration = _orange_clock_declaration_block(recording_dir, camera_id)
    declaration["ptp_status_readback"] = _ptp_status_readback(recording_dir, camera_id)
    return declaration


def _orange_clock_declaration_block(recording_dir: Path, camera_id: str) -> dict[str, Any]:
    path = _orange_session_manifest_path(recording_dir)
    if path is None:
        return {"present": False, "reason": "no_recording_session_json"}
    data = path.read_bytes()
    manifest_evidence = {
        "locator": _recording_relative_locator(recording_dir, path),
        "size_bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }
    try:
        manifest = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {"present": False, "reason": "unreadable_recording_session_json",
                "manifest": manifest_evidence}
    contract = manifest.get("timestamp_clock_contract") if isinstance(manifest, Mapping) else None
    clocks = contract.get("clocks") if isinstance(contract, Mapping) else None
    key = f"camera_{str(camera_id).removeprefix('Cam')}"
    block = clocks.get(key) if isinstance(clocks, Mapping) else None
    if not isinstance(block, Mapping):
        return {"present": False, "reason": f"no_clock_declaration:{key}",
                "manifest": manifest_evidence}
    recording = manifest.get("recording") if isinstance(manifest.get("recording"), Mapping) else {}
    started = recording.get("started_at_utc")
    return {
        "present": True,
        "clock_key": key,
        "block": copy.deepcopy(dict(block)),
        "recording_started_at_utc": started if isinstance(started, str) else None,
        "manifest": manifest_evidence,
    }


def _declaration_refusal(block: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """(reason Orange's declaration cannot be used or None, the fields read)."""

    inference = block.get("inference") if isinstance(block.get("inference"), Mapping) else {}
    snapshot = (
        block.get("evidence_snapshot") if isinstance(block.get("evidence_snapshot"), Mapping) else {}
    )
    host = (
        snapshot.get("host_ptp_management_evidence")
        if isinstance(snapshot.get("host_ptp_management_evidence"), Mapping)
        else {}
    )
    properties = (
        host.get("time_properties") if isinstance(host.get("time_properties"), Mapping) else {}
    )
    checks = inference.get("checks")
    offset = inference.get("expected_tai_minus_utc_seconds")
    fields = {
        "classification": block.get("classification"),
        "timescale": block.get("timescale"),
        "origin": block.get("origin"),
        "semantic_authority": block.get("semantic_authority"),
        "rule_id": inference.get("rule_id"),
        "result": inference.get("result"),
        "expected_tai_minus_utc_seconds": offset,
        "checks": copy.deepcopy(dict(checks)) if isinstance(checks, Mapping) else None,
        "time_properties": {
            name: properties.get(name)
            for name in ("currentUtcOffset", "currentUtcOffsetValid", "ptpTimescale")
        },
        "host_evidence_available": host.get("available"),
        "host_evidence_captured_at_utc": host.get("captured_at_utc"),
        "pmc_exit_status": host.get("pmc_exit_status"),
    }
    if fields["classification"] != "ieee1588_tai":
        return f"classification:{fields['classification']!r}", fields
    if fields["timescale"] != "TAI":
        return f"timescale:{fields['timescale']!r}", fields
    if fields["result"] != ORANGE_DIRECT_PTP_RESULT:
        return f"result:{fields['result']!r}", fields
    if not isinstance(checks, Mapping) or not checks:
        return "checks_missing", fields
    for name, value in checks.items():
        if value is not True:
            return f"check_false:{name}", fields
    if type(offset) is not int:
        return "offset_not_integer", fields
    if fields["host_evidence_available"] is not True:
        return "host_evidence_unavailable", fields
    if type(fields["pmc_exit_status"]) is not int or fields["pmc_exit_status"] != 0:
        return f"pmc_exit_status:{fields['pmc_exit_status']!r}", fields
    current = properties.get("currentUtcOffset")
    if type(current) is not int or current != offset:
        return f"current_utc_offset:{current!r}", fields
    if properties.get("currentUtcOffsetValid") is not True:
        return "current_utc_offset_not_valid", fields
    if properties.get("ptpTimescale") is not True:
        return "ptp_timescale_not_true", fields
    return None, fields


def _v2_clock_semantics(
    source: "AcquisitionFrameClockSource",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """v2 surfaces: Orange's direct declaration when usable, else v1 inference.

    A usable declaration that Palette's own camera-system delta contradicts
    refuses the publication (fail closed) instead of being downgraded.
    """

    v1_surfaces = json_attr_safe_mapping(source.clock_surfaces)
    v1_evidence = json_attr_safe_mapping(source.clock_semantic_evidence)
    v1_camera = dict(v1_surfaces["camera_timestamp_ns"])
    declaration = source.orange_clock_declaration
    if declaration is None:
        raise AcquisitionFrameClockError(
            "A v2 frame-clock record needs the source's Orange clock declaration."
        )
    readback = declaration.get("ptp_status_readback")
    stable_slave = None if readback is None else _readback_is_stable_slave(readback)
    readback_evidence = {
        "present": readback is not None,
        "value": readback,
        "stable_slave": stable_slave,
        "rule": "first == last == 'Slave' and changes == 0",
    }
    record: dict[str, Any] = {
        "present": bool(declaration.get("present")),
        "used": False,
        "reason": declaration.get("reason"),
        "manifest": declaration.get("manifest"),
        "clock_key": declaration.get("clock_key"),
        "recording_started_at_utc": declaration.get("recording_started_at_utc"),
    }
    cross_check = None
    if declaration.get("present"):
        reason, fields = _declaration_refusal(declaration["block"])
        record.update(fields)
        record["reason"] = reason
        if reason is None:
            offset_ns = int(fields["expected_tai_minus_utc_seconds"]) * 1_000_000_000
            delta = _valid_clock_delta_summary(
                source.camera_timestamp_ns,
                source.camera_timestamp_valid,
                source.system_timestamp_ns,
                source.system_timestamp_valid,
            )
            agrees = bool(
                delta is not None
                and abs(int(delta["median_ns"]) - offset_ns) < 1_000_000_000
                and int(delta["maximum_ns"]) - int(delta["minimum_ns"]) < 1_000_000_000
            )
            cross_check = {
                "measured_camera_system_delta_ns": delta,
                "declared_tai_minus_utc_ns": offset_ns,
                "tolerance_ns": 1_000_000_000,
                "agrees": agrees,
            }
            if not agrees:
                raise AcquisitionFrameClockError(
                    "Orange declares camera timestamps as TAI with TAI-UTC "
                    f"{fields['expected_tai_minus_utc_seconds']} s, but the measured "
                    f"camera-system delta {delta} contradicts it."
                )
            if stable_slave is False:
                raise AcquisitionFrameClockError(
                    "Orange declares camera timestamps as TAI, but the camera's PTP "
                    f"status readback {readback} is not a stable Slave."
                )
            record["used"] = True
    if record["used"]:
        camera = _ptp_tai_camera_semantics("producer_declared_by_orange_ptp_time_properties")
        camera["semantic_authority"] = "orange_declared"
        camera["tai_minus_utc_seconds"] = int(record["expected_tai_minus_utc_seconds"])
    elif stable_slave is False and v1_camera.get("timescale") == "IEEE-1588_PTP_TAI":
        # v1 inferred TAI without seeing the port-state readback; a readback
        # that is not a stable Slave contradicts that inference.
        camera = {
            **_device_defined_camera_semantics("ptp_inference_contradicted_by_status_readback"),
            "semantic_authority": "palette_inferred",
            "tai_minus_utc_seconds": None,
        }
    else:
        camera = {
            **v1_camera,
            "semantic_authority": "palette_inferred",
            "tai_minus_utc_seconds": (
                37 if v1_camera.get("timescale") == "IEEE-1588_PTP_TAI" else None
            ),
        }
    surfaces = {**v1_surfaces, "camera_timestamp_ns": camera}
    evidence = {
        **v1_evidence,
        "orange_declaration": record,
        "palette_cross_check": cross_check,
        "ptp_status_readback": readback_evidence,
    }
    return surfaces, evidence


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
        # Match complete states: "unlocked" and "unsynchronized" are negative.
        or normalized_ptp_status in {"locked", "slave", "master", "synchronized"}
    )
    expected_tai_minus_utc_ns = 37_000_000_000
    expected_tai_utc_offset = bool(
        delta is not None
        and abs(int(delta["median_ns"]) - expected_tai_minus_utc_ns) < 1_000_000_000
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
        camera_semantics: dict[str, Any] = _ptp_tai_camera_semantics(
            "inferred_from_recording_evidence_not_sdk_declared"
        )
    else:
        camera_semantics = _device_defined_camera_semantics(
            "producer_passthrough_epoch_not_established"
        )

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


CLOCK_CSV_FRAME_ID_FIELDS = ("recording_frame_id", "frame_id", "local_frame_id")
CLOCK_CSV_TIMESTAMP_FIELDS = ("timestamp", "timestamp_sys")


@dataclass(frozen=True)
class ParsedFrameClockCsv:
    """Unvalidated rows of one Orange camera-metadata CSV."""

    frame_field: str
    recording_frame_id: np.ndarray
    camera_timestamp_ns: np.ndarray
    camera_timestamp_valid: np.ndarray
    system_timestamp_ns: np.ndarray
    system_timestamp_valid: np.ndarray


def clock_csv_frame_field(fieldnames: Sequence[str]) -> str | None:
    fields = {str(value).strip() for value in fieldnames}
    return next((name for name in CLOCK_CSV_FRAME_ID_FIELDS if name in fields), None)


def parse_acquisition_frame_clock_csv(path: str | Path) -> ParsedFrameClockCsv:
    """Parse clock rows with the authority's rules, without validating the clock.

    This is the only Orange camera-metadata CSV parser.  Diagnostics that report
    on a raw CSV use it so they read rows exactly as clock publication does.
    """

    path = Path(path)
    frame_ids: list[int] = []
    camera_values: list[int | None] = []
    system_values: list[int | None] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(str(value).strip() for value in (reader.fieldnames or ()))
        frame_field = clock_csv_frame_field(fields)
        if frame_field is None or not set(CLOCK_CSV_TIMESTAMP_FIELDS) <= set(fields):
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
    camera, camera_valid = _timestamp_array(camera_values)
    system, system_valid = _timestamp_array(system_values)
    return ParsedFrameClockCsv(
        frame_field=frame_field,
        recording_frame_id=np.asarray(frame_ids, dtype=np.int64),
        camera_timestamp_ns=camera,
        camera_timestamp_valid=camera_valid,
        system_timestamp_ns=system,
        system_timestamp_valid=system_valid,
    )


def _load_csv_source(
    path: Path,
    *,
    recording_dir: Path,
    camera_id: str,
) -> AcquisitionFrameClockSource:
    parsed = parse_acquisition_frame_clock_csv(path)
    if not parsed.recording_frame_id.size:
        raise AcquisitionFrameClockError(f"Frame-clock CSV has no data rows: {path}")
    recording_ids = parsed.recording_frame_id
    offset = _frame_id_offset(int(recording_ids[0]))
    parent = recording_ids - np.int64(offset)
    camera, camera_valid = parsed.camera_timestamp_ns, parsed.camera_timestamp_valid
    system, system_valid = parsed.system_timestamp_ns, parsed.system_timestamp_valid
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
        orange_clock_declaration=_orange_clock_declaration(recording_dir, str(camera_id)),
    )


def _nullable_arrow_ints(
    values: Sequence[object], *, label: str, allow_null: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    normalized: list[int | None] = []
    for row_number, value in enumerate(values, start=1):
        if value is None and allow_null:
            normalized.append(None)
            continue
        # Arrow's Python integers preserve exact values, unlike int(float).
        if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
            raise AcquisitionFrameClockError(
                f"{label} at camera row {row_number} must be an exact int64 integer; "
                f"got {value!r}."
            )
        normalized.append(value)
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
    recording_ids, _ = _nullable_arrow_ints(
        [columns["recording_frame_id"][index] for index in indices],
        label="recording_frame_id",
        allow_null=False,
    )
    parent, _ = _nullable_arrow_ints(
        [columns["parent_frame_index"][index] for index in indices],
        label="parent_frame_index",
        allow_null=False,
    )
    camera, camera_valid = _nullable_arrow_ints(
        [columns["timestamp"][index] for index in indices],
        label="timestamp",
    )
    system, system_valid = _nullable_arrow_ints(
        [columns["timestamp_sys"][index] for index in indices],
        label="timestamp_sys",
    )
    # Rows are keyed by parent_frame_index, not by their order on disk.
    # Already-ordered files are unchanged, so published digests are stable.
    order = np.argsort(parent, kind="stable")
    recording_ids, parent = recording_ids[order], parent[order]
    camera, camera_valid = camera[order], camera_valid[order]
    system, system_valid = system[order], system_valid[order]
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
        orange_clock_declaration=_orange_clock_declaration(recording_dir, str(camera_id)),
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
    for name, array in arrays.items():
        if name.endswith("_valid"):
            if array.dtype != np.dtype(np.bool_):
                raise AcquisitionFrameClockError(f"{name} must be a boolean vector.")
        elif array.dtype.kind not in "iu" or int(array.max()) >= (1 << 63):
            raise AcquisitionFrameClockError(
                f"{name} must contain exact int64 integers."
            )
    if expected_frame_count is not None and row_count != int(expected_frame_count):
        raise AcquisitionFrameClockError(
            "Acquisition frame-clock row count does not match the source video: "
            f"clock={row_count}, video={int(expected_frame_count)}."
        )
    if not np.array_equal(
        source.parent_frame_index, np.arange(row_count, dtype=np.int64)
    ):
        raise AcquisitionFrameClockError(
            "Acquisition frame-clock parent_frame_index must be the complete ordered "
            "zero-based video-frame domain."
        )
    ids = source.recording_frame_id
    if row_count > 1 and (np.any(ids[1:] <= ids[:-1]) or not np.all(np.diff(ids) == 1)):
        raise AcquisitionFrameClockError(
            "Acquisition frame-clock recording_frame_id values must be contiguous."
        )
    if not bool(
        np.any(source.camera_timestamp_valid) or np.any(source.system_timestamp_valid)
    ):
        raise AcquisitionFrameClockError(
            "Acquisition frame-clock source contains no usable camera or system timestamps."
        )
    for label, values, valid in (
        (
            "camera_timestamp_ns",
            source.camera_timestamp_ns,
            source.camera_timestamp_valid,
        ),
        (
            "system_timestamp_ns",
            source.system_timestamp_ns,
            source.system_timestamp_valid,
        ),
    ):
        selected = np.asarray(values[valid], dtype=np.int64)
        if selected.size > 1 and np.any(selected[1:] < selected[:-1]):
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


def load_clipped_acquisition_frame_clock_source(
    recording_dir: str | Path,
    *,
    camera_id: str,
    frame_index_path: str | Path,
    expected_frame_count: int,
) -> AcquisitionFrameClockSource:
    """Load only the explicitly bound, complete clipped-recording frame index.

    A clip-local CSV or a conventional sibling index cannot stand in for this
    recording-wide source. The existing Parquet, validity, clock semantics and
    persisted digest contracts remain unchanged.
    """

    root = Path(recording_dir).expanduser().resolve()
    if not root.is_dir():
        raise AcquisitionFrameClockError("Clipped recording directory is missing.")
    if type(camera_id) is not str or not camera_id or camera_id != camera_id.strip():
        raise AcquisitionFrameClockError("Clipped clock requires an exact camera_id.")
    if type(expected_frame_count) is not int or expected_frame_count <= 0:
        raise AcquisitionFrameClockError(
            "Clipped clock requires a positive exact frame count."
        )
    path = Path(frame_index_path).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise AcquisitionFrameClockError(
            "Clipped frame index must be an existing regular file inside the recording."
        )
    return _validate_source(
        _load_parquet_source(path, recording_dir=root, camera_id=camera_id),
        expected_frame_count=expected_frame_count,
    )


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


def _build_record(
    source: AcquisitionFrameClockSource,
    schema_version: int = ACQUISITION_FRAME_CLOCK_SCHEMA_VERSION,
) -> dict[str, Any]:
    if schema_version == 1:
        surfaces, evidence = source.clock_surfaces, source.clock_semantic_evidence
    elif schema_version == 2:
        surfaces, evidence = _v2_clock_semantics(source)
    else:
        raise AcquisitionFrameClockError(
            f"Unsupported acquisition frame-clock schema_version {schema_version!r}."
        )
    return {
        "schema_id": ACQUISITION_FRAME_CLOCK_SCHEMA_IDS[schema_version],
        "schema_version": schema_version,
        "camera_id": source.camera_id,
        "row_count": source.row_count,
        "frame_domain": {
            "recording_frame_id": "session_continuous_source_identifier",
            "parent_frame_index": "zero_based_complete_video_frame_index",
            "sampled_row_join": "raw_video/original_frame_indices -> parent_frame_index",
        },
        "clock_surfaces": json_attr_safe_mapping(surfaces),
        "clock_semantic_evidence": json_attr_safe_mapping(evidence),
        "missing_timestamp_sentinel": int(MISSING_TIMESTAMP_SENTINEL),
        "source": _source_evidence(source),
        "array_sha256": _array_digests(source),
        "array_canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
    }


def acquisition_frame_clock_sha256(record: Mapping[str, Any]) -> str:
    return sha256(strict_json_dumps(record).encode("utf-8")).hexdigest()


def acquisition_frame_clock_source_sha256(
    source: AcquisitionFrameClockSource,
    *,
    schema_version: int = ACQUISITION_FRAME_CLOCK_SCHEMA_VERSION,
) -> str:
    """Validate a live source and hash the publisher's record grammar.

    Re-verification passes the published record's ``schema_version`` so a v1
    publication is rebuilt with v1 rules, byte for byte.
    """

    _validate_source(source, expected_frame_count=None)
    return acquisition_frame_clock_sha256(
        _validate_record(_build_record(source, schema_version))
    )


def _validate_record(record: Mapping[str, Any], digest: str | None = None) -> dict[str, Any]:
    canonical = json_attr_safe_mapping(record)
    version = canonical.get("schema_version")
    if canonical.get("schema_id") not in ACQUISITION_FRAME_CLOCK_SCHEMA_IDS.values():
        raise AcquisitionFrameClockError("Unsupported acquisition frame-clock schema_id.")
    if not is_supported_acquisition_frame_clock_schema(canonical.get("schema_id"), version):
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
    if version == 2:
        camera = clock_surfaces["camera_timestamp_ns"]
        authority = camera.get("semantic_authority")
        if authority not in _CAMERA_SEMANTIC_AUTHORITIES:
            raise AcquisitionFrameClockError(
                "A v2 frame clock must say who decided the camera timescale."
            )
        evidence = canonical.get("clock_semantic_evidence")
        declaration = evidence.get("orange_declaration") if isinstance(evidence, Mapping) else None
        if not isinstance(declaration, Mapping) or declaration.get("used") is not (
            authority == "orange_declared"
        ):
            raise AcquisitionFrameClockError(
                "A v2 frame clock's declaration evidence differs from its camera authority."
            )
        readback = evidence.get("ptp_status_readback") if isinstance(evidence, Mapping) else None
        if not isinstance(readback, Mapping) or readback.get("stable_slave") not in (True, False, None):
            raise AcquisitionFrameClockError(
                "A v2 frame clock must record the camera's PTP status readback."
            )
        if camera.get("timescale") == "IEEE-1588_PTP_TAI" and readback.get("stable_slave") is False:
            raise AcquisitionFrameClockError(
                "A v2 frame clock labels TAI against an unstable PTP status readback."
            )
        if authority == "orange_declared" and (
            camera.get("timescale") != "IEEE-1588_PTP_TAI"
            or type(camera.get("tai_minus_utc_seconds")) is not int
        ):
            raise AcquisitionFrameClockError(
                "An Orange-declared camera clock must be TAI with an integer offset."
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
    return _validate_run_payload(run_name, run)


def _validate_run_payload(run_name: str, run: Any) -> ResolvedAcquisitionFrameClock:
    """Validate the same clock bytes before or after lifecycle completion."""

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
        values = read_whole_array(run[name])
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


def _root_binding(resolved: ResolvedAcquisitionFrameClock) -> dict[str, Any]:
    # The unique immutable attempt path is written first and restored last. It
    # proves ownership of a partially persisted root binding during rollback.
    return {
        "acquisition_frame_clock_ref": resolved.group_path,
        "acquisition_frame_clock_sha256": resolved.record_sha256,
        "acquisition_frame_clock_camera_id": resolved.camera_id,
        "acquisition_frame_clock_row_count": resolved.row_count,
        "acquisition_frame_clock_available": True,
        "acquisition_frame_clock_status": "complete",
    }


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


def load_published_acquisition_frame_clock(
    root: Any,
    *,
    recording_dir: str | Path,
) -> tuple[ResolvedAcquisitionFrameClock, AcquisitionFrameClockSource]:
    """Return the selected published clock and its arrays; never re-reads raw files.

    Consumers that need timestamp values read them here, so the import-time
    publication stays the single clock authority for every reader.
    """

    resolved = resolve_acquisition_frame_clock(root, required=True)
    assert resolved is not None
    run = root[resolved.group_path]
    arrays = {name: read_whole_array(run[name]) for name in _ARRAY_NAMES}
    for name, values in arrays.items():
        if _array_values_sha256(values) != str(resolved.record["array_sha256"][name]):
            raise AcquisitionFrameClockError(
                f"Acquisition frame-clock array {name} differs from its bound digest."
            )
    evidence = resolved.record["source"]
    source = AcquisitionFrameClockSource(
        source_path=_resolve_recording_path(
            Path(recording_dir).expanduser().resolve(), evidence["locator"]
        ),
        source_kind=str(evidence["kind"]),
        source_locator=str(evidence["locator"]),
        camera_id=resolved.camera_id,
        clock_surfaces=resolved.record["clock_surfaces"],
        clock_semantic_evidence=resolved.record["clock_semantic_evidence"],
        **arrays,
    )
    return resolved, source


NOMINAL_FRAME_TIME_TOLERANCE_FRAMES = 0.5


def camera_tai_minus_utc(resolved: ResolvedAcquisitionFrameClock) -> dict[str, Any]:
    """The published clock's TAI-UTC offset for camera timestamps, and who decided it.

    v2 records carry it on the camera surface: ``orange_declared`` when Orange
    declared a TAI camera clock (and Palette's cross-check agreed),
    ``palette_inferred`` when Palette's own PTP evidence labelled it TAI. A v1
    record that labelled the camera ``IEEE-1588_PTP_TAI`` counts as
    ``palette_inferred`` with 37 s. Otherwise ``seconds`` is None: the camera
    timestamps have no known relation to UTC.
    """

    camera = resolved.record["clock_surfaces"]["camera_timestamp_ns"]
    if resolved.record.get("schema_version") == 2:
        seconds = camera.get("tai_minus_utc_seconds")
        authority = camera.get("semantic_authority")
    elif camera.get("timescale") == "IEEE-1588_PTP_TAI":
        seconds, authority = 37, "palette_inferred"
    else:
        seconds, authority = None, None
    if type(seconds) is not int:
        return {"seconds": None, "authority": None}
    return {"seconds": seconds, "authority": authority}


def nominal_frame_time_deviation_s(
    camera_timestamp_ns: np.ndarray,
    camera_timestamp_valid: np.ndarray,
    parent_frame_index: np.ndarray,
    *,
    nominal_fps: float,
) -> float | None:
    """Largest gap between camera time and ``frame / nominal_fps``, in seconds."""

    valid = np.asarray(camera_timestamp_valid, dtype=np.bool_)
    if int(np.count_nonzero(valid)) < 2:
        return None
    stamps = np.asarray(camera_timestamp_ns, dtype=np.int64)[valid]
    frames = np.asarray(parent_frame_index, dtype=np.int64)[valid]
    measured = (stamps - stamps[0]).astype(np.float64) / 1e9
    nominal = (frames - frames[0]).astype(np.float64) / float(nominal_fps)
    return float(np.max(np.abs(measured - nominal)))


def require_nominal_frame_timing(
    root: Any,
    *,
    nominal_fps: float,
    tolerance_frames: float = NOMINAL_FRAME_TIME_TOLERANCE_FRAMES,
) -> dict[str, Any] | None:
    """Fail when ``frame / nominal_fps`` disagrees with the published camera clock.

    Analysis time is frame index divided by nominal fps.  That is only correct
    while the camera clock ticks at the nominal rate, so producers of fps-based
    time call this first.  Returns ``None`` for a recording without a published
    clock; the caller keeps its existing nominal-fps behavior there.
    """

    if not np.isfinite(nominal_fps) or float(nominal_fps) <= 0:
        raise AcquisitionFrameClockError(f"Nominal fps must be positive: {nominal_fps!r}.")
    resolved = resolve_acquisition_frame_clock(root, required=False)
    if resolved is None:
        return None
    run = root[resolved.group_path]
    deviation = nominal_frame_time_deviation_s(
        read_whole_array(run["camera_timestamp_ns"]),
        read_whole_array(run["camera_timestamp_valid"]),
        read_whole_array(run["parent_frame_index"]),
        nominal_fps=float(nominal_fps),
    )
    tolerance_s = float(tolerance_frames) / float(nominal_fps)
    evidence = {
        "acquisition_frame_clock_sha256": resolved.record_sha256,
        "nominal_fps": float(nominal_fps),
        "max_abs_deviation_s": deviation,
        "tolerance_s": tolerance_s,
    }
    if deviation is not None and deviation > tolerance_s:
        raise AcquisitionFrameClockError(
            "Frame-index time disagrees with the published camera clock: "
            f"max deviation {deviation:.4f} s exceeds {tolerance_s:.4f} s "
            f"({tolerance_frames} frames at {float(nominal_fps)} fps)."
        )
    return evidence


def _owned_clock_run(root: Any, run_name: str, owner: str) -> Any:
    path = f"{ACQUISITION_FRAME_CLOCK_RUNS_PATH}/{run_name}"
    run = root[path]
    if (
        archive_identity(run) != archive_identity(root)
        or run.attrs.get(_PUBLICATION_OWNER_ATTR) != owner
        or run.attrs.get("stage_selector_eligible") is not False
    ):
        raise AcquisitionFrameClockError(
            "Acquisition clock publication lost its owner or ineligible state."
        )
    return run


class _OwnedClockLifecycle(MutableMapping):
    """Fresh owner checks for every shared lifecycle-helper attr operation."""

    def __init__(self, root: Any, run_name: str, owner: str):
        self.root, self.run_name, self.owner = root, run_name, owner

    @property
    def attrs(self):
        return self

    def _current(self):
        return _owned_clock_run(self.root, self.run_name, self.owner).attrs

    def __getitem__(self, key):
        return self._current()[key]

    def __setitem__(self, key, value):
        self._current()[key] = value

    def __delitem__(self, key):
        del self._current()[key]

    def __iter__(self):
        return iter(self._current())

    def __len__(self):
        return len(self._current())


def _fail_owned_clock(
    root: Any, run_name: str, owner: str, error: BaseException
) -> None:
    """Retain a failed public name, never mutate a same-path replacement."""

    path = f"{ACQUISITION_FRAME_CLOCK_RUNS_PATH}/{run_name}"
    run = root.get(path)
    if run is None or run.attrs.get(_PUBLICATION_OWNER_ATTR) != owner:
        return
    failed_at = datetime.now(timezone.utc).isoformat()
    lifecycle = _OwnedClockLifecycle(root, run_name, owner)
    mark_run_failed(
        lifecycle, run_name=run_name, failed_at_utc=failed_at, error=str(error)
    )
    lifecycle.attrs[_PUBLICATION_TOMBSTONE_ATTR] = {
        "schema_id": "palette.acquisition_frame_clock_publication_tombstone",
        "schema_version": 1,
        "owner_uuid": owner,
        "failed_at_utc": failed_at,
        "error": str(error),
        "public_path_retained": True,
        "retry_requires_new_run_name": True,
    }


def _restore_clock_root_binding(
    root: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
    attempted: Mapping[str, Any],
    activation: DeferredSelectorActivation,
) -> None:
    """Undo only this unique attempt's root values, not another publication."""

    for key in reversed(tuple(attempted)):
        current = root[""]
        ref_key = "acquisition_frame_clock_ref"
        current_ref = current.attrs.get(ref_key)
        current_parent = root[ACQUISITION_FRAME_CLOCK_RUNS_PATH]
        # A failed/lost first ref write can leave later binding fields written.
        # Only the exact still-owned lease permits undoing that partial state.
        unchanged_ref_under_lease = (
            (ref_key in current.attrs, current_ref) == snapshot[ref_key]
            and current_parent.attrs.get(_PUBLICATION_LEASE_ATTR)
            == dict(activation.lease)
            and current_parent.attrs.get(_PUBLICATION_GENERATION_ATTR)
            == activation.next_generation
        )
        if current_ref != attempted[ref_key] and not unchanged_ref_under_lease:
            return
        if current.attrs.get(key) != attempted[key]:
            continue
        present, previous = snapshot[key]
        if present:
            current.attrs[key] = copy.deepcopy(previous)
        elif key in current.attrs:
            del current.attrs[key]
    # Do not "refresh" a caller's cached handle by writing its stale attrs back.
    # On failure callers must reopen the mutable root before attempting a retry.


def publish_acquisition_frame_clock(
    root: Any, source: AcquisitionFrameClockSource
) -> ResolvedAcquisitionFrameClock:
    """Publish a fresh owned attempt; replay a selected identical clock read-only.

    This mutates an active ingestion archive. The owning importer performs the
    archive's final consolidation, after its remaining payload/manifest writes.
    """

    if (
        getattr(getattr(root, "metadata", None), "consolidated_metadata", None)
        is not None
    ):
        raise AcquisitionFrameClockError(
            "Clock publication requires a mutable use_consolidated=False root."
        )
    with zarr_store_metadata_publication_lock(getattr(root, "store", None)):
        return _publish_acquisition_frame_clock_locked(root, source)


def _publish_acquisition_frame_clock_locked(
    root: Any, source: AcquisitionFrameClockSource
) -> ResolvedAcquisitionFrameClock:

    source = _validate_source(source, expected_frame_count=None)
    record = _validate_record(_build_record(source))
    digest = acquisition_frame_clock_sha256(record)
    if dict(root.attrs) != dict(root[""].attrs):
        raise AcquisitionFrameClockError(
            "Clock publication requires a freshly opened root; metadata changed."
        )
    # Mutable Zarr-v3 attr writes discard the root's inline consolidated view.
    # If this local archive already had one, a rolled-back root binding must
    # restore published-reader visibility as well as its direct selector state.
    identity = archive_identity(root)
    published_archive = None
    if identity.kind == "local_store_root":
        archive_path = Path(identity.key[0])
        root_metadata = json.loads(
            (archive_path / "zarr.json").read_text(encoding="utf-8")
        )
        if root_metadata.get("consolidated_metadata") is not None:
            published_archive = archive_path
    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, "acquisition_frame_clock_runs")

    # Never repair a conflicting/in-progress selector pair as incidental reuse.
    if "latest" in parent.attrs or "latest_complete" in parent.attrs:
        selected = resolve_acquisition_frame_clock(root)
        if selected is not None and selected.record_sha256 == digest:
            return selected
        # A clock published by an earlier record version stays selected when
        # this source still rebuilds it exactly under that version's rules:
        # replaying an import never republishes an unchanged clock.
        if selected is not None and selected.record.get("schema_version") != record[
            "schema_version"
        ] and selected.record_sha256 == acquisition_frame_clock_source_sha256(
            source, schema_version=int(selected.record["schema_version"])
        ):
            return selected

    owner = str(uuid.uuid4())
    run_name = f"acquisition_frame_clock_{digest[:16]}"
    if run_name in parent:
        existing = parent[run_name]
        if existing.attrs.get(ACQUISITION_FRAME_CLOCK_SHA256_ATTR) not in (
            None,
            digest,
        ):
            raise AcquisitionFrameClockError(
                f"Existing acquisition frame-clock run {run_name!r} conflicts."
            )
        # Public names are permanently occupied, including failed attempts and
        # previously selected clocks. Their arrays/attrs are never rewritten.
        run_name = f"{run_name}_{uuid.UUID(owner).hex}"

    activation = None
    root_snapshot = None
    bindings = None
    binding_staged = False
    source_fingerprint = optional_source_stat_fingerprint_attrs(
        source.source_path,
        attr_prefix="source_frame_clock",
    ).get("source_frame_clock_fingerprint")
    provenance = build_writer_run_provenance(
        command="import:publish_acquisition_frame_clock",
        params={
            "schema_id": record["schema_id"],
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
    )
    try:
        parent.create_group(
            run_name,
            attributes={
                _PUBLICATION_OWNER_ATTR: owner,
                "stage_selector_eligible": False,
                "palette_run_completion_status": "running",
            },
        )
        lifecycle = _OwnedClockLifecycle(root, run_name, owner)
        mark_run_started(lifecycle, run_name=run_name, stage="acquisition_frame_clock")
        for key, value in {
            "schema_id": record["schema_id"],
            "schema_version": record["schema_version"],
            ACQUISITION_FRAME_CLOCK_RECORD_ATTR: record,
            ACQUISITION_FRAME_CLOCK_SHA256_ATTR: digest,
            "immutable": True,
        }.items():
            _owned_clock_run(root, run_name, owner).attrs[key] = value
        for name in _ARRAY_NAMES:
            store_array(
                _owned_clock_run(root, run_name, owner),
                name,
                np.asarray(getattr(source, name)),
                shard_rows=ACQUISITION_FRAME_CLOCK_SHARD_ROWS,
            )
        resolved = _validate_run_payload(
            run_name, _owned_clock_run(root, run_name, owner)
        )
        if resolved.record_sha256 != digest:
            raise AcquisitionFrameClockError(
                "Persisted acquisition clock differs from the requested digest."
            )
        mark_run_complete(
            lifecycle,
            parent_group=root[ACQUISITION_FRAME_CLOCK_RUNS_PATH],
            run_name=run_name,
            run_provenance=provenance,
        )

        def proof():
            candidate = _owned_clock_run(root, run_name, owner)
            current_parent = root[ACQUISITION_FRAME_CLOCK_RUNS_PATH]
            if not is_run_complete_in_parent(
                current_parent, candidate, legacy_default=False
            ):
                raise AcquisitionFrameClockError(
                    "Acquisition clock candidate is incomplete."
                )
            current = _validate_run_payload(run_name, candidate)
            if current != resolved:
                raise AcquisitionFrameClockError(
                    "Acquisition clock candidate changed after validation."
                )
            if binding_staged:
                current_root = root[""]
                if any(
                    current_root.attrs.get(key) != value
                    for key, value in bindings.items()
                ):
                    raise AcquisitionFrameClockError(
                        "Acquisition clock root binding changed before commit."
                    )
            return current

        activation = activate_selector_eligible_run(
            root,
            root[ACQUISITION_FRAME_CLOCK_RUNS_PATH],
            _owned_clock_run(root, run_name, owner),
            parent_path=ACQUISITION_FRAME_CLOCK_RUNS_PATH,
            run_path=resolved.group_path,
            run_name=run_name,
            owner_attr=_PUBLICATION_OWNER_ATTR,
            expected_owner_uuid=owner,
            policy_attr=_PUBLICATION_POLICY_ATTR,
            generation_attr=_PUBLICATION_GENERATION_ATTR,
            lease_attr=_PUBLICATION_LEASE_ATTR,
            policy=_PUBLICATION_POLICY,
            lease_schema_id="palette.acquisition_frame_clock_publication_lease",
            proof_loader=proof,
            defer_eligibility=True,
        )
        bindings = _root_binding(resolved)
        current_root = root[""]
        if dict(root.attrs) != dict(current_root.attrs):
            raise AcquisitionFrameClockError(
                "Concurrent root metadata mutation during clock publication."
            )
        root_snapshot = {
            key: (key in current_root.attrs, copy.deepcopy(current_root.attrs.get(key)))
            for key in bindings
        }
        root.attrs.update(bindings)
        binding_staged = True
        current_root = root[""]
        if any(current_root.attrs.get(key) != value for key, value in bindings.items()):
            raise AcquisitionFrameClockError(
                "Acquisition clock root binding did not persist."
            )
        commit_deferred_selector_activation(
            activation,
            root=root,
            parent_group=root[ACQUISITION_FRAME_CLOCK_RUNS_PATH],
            run_group=_owned_clock_run(root, run_name, owner),
            proof_loader=proof,
        )
        # Eligibility is the literal final metadata operation. Return only the
        # prevalidated value; do not call the public resolver after committing.
        return resolved
    except BaseException as exc:
        failures = []
        if root_snapshot is not None and bindings is not None:
            try:
                _restore_clock_root_binding(root, root_snapshot, bindings, activation)
            except BaseException as rollback_exc:
                failures.append(f"root binding: {rollback_exc}")
        if activation is not None:
            try:
                rollback_deferred_selector_activation(activation)
            except BaseException as rollback_exc:
                failures.append(f"selector activation: {rollback_exc}")
        try:
            _fail_owned_clock(root, run_name, owner, exc)
        except BaseException as rollback_exc:
            failures.append(f"failed public tombstone: {rollback_exc}")
        if not failures and published_archive is not None and root_snapshot is not None:
            try:
                current_root = root[""]
                current_parent = root[ACQUISITION_FRAME_CLOCK_RUNS_PATH]
                root_restored = all(
                    (key in current_root.attrs) is present
                    and (not present or current_root.attrs.get(key) == previous)
                    for key, (present, previous) in root_snapshot.items()
                )
                selectors_restored = activation is not None and all(
                    (key in current_parent.attrs) is present
                    and (not present or current_parent.attrs.get(key) == previous)
                    for key, (present, previous), _written in activation.mutations
                    if key in activation.selectors
                )
                if root_restored and selectors_restored:
                    consolidate_metadata_capture_expected_warnings(published_archive)
                    # The helper builds the replacement view from direct
                    # metadata; published-reader validation is independent.
                    published = open_zarr_root(
                        published_archive, mode="r", use_consolidated=True
                    )
                    if "latest" in current_parent.attrs:
                        if resolve_acquisition_frame_clock(
                            published
                        ) != resolve_acquisition_frame_clock(current_root):
                            raise AcquisitionFrameClockError(
                                "Rollback consolidated clock selection differs from direct metadata."
                            )
                    elif any(
                        key in published[ACQUISITION_FRAME_CLOCK_RUNS_PATH].attrs
                        for key in ("latest", "latest_complete")
                    ):
                        raise AcquisitionFrameClockError(
                            "Rollback consolidation exposed an uncommitted clock."
                        )
            except BaseException as rollback_exc:
                failures.append(f"published metadata visibility: {rollback_exc}")
        if failures:
            raise AcquisitionFrameClockError(
                f"Clock publication rollback was incomplete: {failures!r}."
            ) from exc
        raise


def publish_clipped_acquisition_frame_clock(
    root: Any,
    *,
    recording_dir: str | Path,
    camera_id: str,
    acquisition_record: Any,
) -> ResolvedAcquisitionFrameClock:
    """Publish the clock for a clipped recording from its bound frame index.

    ``acquisition_record`` is the persisted acquisition camera record; its
    locator names the only recording-wide frame index that may back the clock.
    """

    source = load_clipped_acquisition_frame_clock_source(
        recording_dir,
        camera_id=camera_id,
        frame_index_path=acquisition_record.source_video_metadata["locator"][
            "relative_path"
        ],
        expected_frame_count=acquisition_record.source_total_frames,
    )
    return publish_acquisition_frame_clock(root, source)


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
    "camera_tai_minus_utc",
    "ACQUISITION_FRAME_CLOCK_SCHEMA_IDS",
    "ORANGE_DIRECT_PTP_RESULT",
    "is_supported_acquisition_frame_clock_schema",
    "ACQUISITION_FRAME_CLOCK_RECORD_ATTR",
    "ACQUISITION_FRAME_CLOCK_RUNS_PATH",
    "ACQUISITION_FRAME_CLOCK_SCHEMA_ID",
    "ACQUISITION_FRAME_CLOCK_SHA256_ATTR",
    "AcquisitionFrameClockError",
    "AcquisitionFrameClockSource",
    "ResolvedAcquisitionFrameClock",
    "acquisition_frame_clock_sha256",
    "acquisition_frame_clock_source_sha256",
    "import_acquisition_frame_clock",
    "load_acquisition_frame_clock_source",
    "load_clipped_acquisition_frame_clock_source",
    "load_published_acquisition_frame_clock",
    "publish_acquisition_frame_clock",
    "publish_clipped_acquisition_frame_clock",
    "resolve_acquisition_frame_clock",
]
