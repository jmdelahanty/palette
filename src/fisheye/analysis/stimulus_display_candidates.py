"""Which stimulus states could have been on screen during each camera exposure.

Unified (v2) recordings only. For every camera frame of the published
acquisition frame clock this computes the range of Citrus stimulus states that
could have been visible while the frame was exposed:

* Exposure time on the pancake0 host clock is ``camera_timestamp_ns - offset``,
  with the TAI-UTC offset taken only from the published acquisition frame clock
  (``camera_tai_minus_utc``), which records whether Orange declared it or
  Palette inferred it. The run records that authority; a clock with no offset
  is refused. Orange's ``timestamp_clock_contract`` and PTP summary are read
  for evidence only (clock-state intervals, lock state). The camera
  timestamp's exposure point is undeclared, so ``exposure_us / 2`` is recorded
  as uncertainty.
* Citrus records when each state was handed to the display
  (``swap_call_end_ns_epoch``, same host clock). With vsync off a frame is fully
  visible within two projector refreshes of the swap, plus projector latency.
* Candidates for exposure ``t`` with projector latency ``L`` are the submitted
  states from the latest one swapped at or before ``t - L - 2 refreshes``
  through the latest one swapped at or before ``t - L``. ``L`` is unmeasured
  until the photodiode is commissioned, so the run stores the ``L = 0`` range
  and the per-state swap table reference; consumers that assume a latency
  shift both bounds earlier by it.

This is the display ("what was on screen") alignment. The run also carries
the causal ("driving frame") alignment: the states Citrus computed from each
camera frame, read from Citrus's ``/correspondence/frames/sources``. That table
names Orange's 1-based ``recording_frame_id`` for every state. The
``triggering_camera_frame_id`` column is not used: in unified recordings it is
Citrus's IPC frame counter, which runs ahead of ``recording_frame_id`` by a
per-session offset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.acquisition_frame_clock import (
    MISSING_TIMESTAMP_SENTINEL,
    camera_tai_minus_utc,
    load_published_acquisition_frame_clock,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.metadata import get_fps
from fisheye.shared.run_lineage_fingerprint import (
    build_run_lineage_payload,
    write_run_lineage_attrs,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.unified_h5 import PROFILE as UNIFIED_PROFILE
from fisheye.shared.unified_h5.correspondence import FRAME_SOURCES
from fisheye.shared.unified_h5.display_submissions import load_stimulus_display_submissions
from fisheye.shared.unified_h5.reference import UnifiedSource, open_unified_source
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    require_runs_parent,
)

PARENT_NAME = "stimulus_display_candidate_runs"
SCHEMA_ID = "palette.stimulus_display_candidates"
SCHEMA_VERSION = 3
METHOD = "latest_submitted_within_full_visibility_window_v1"
DRIVING_METHOD = "citrus_correspondence_source_recording_frame_id_v1"
NO_STATE = -1

# agent-contracts orange-palette/camera_exposure_host_clock_v1 (PR #78).
CADENCE_FLAG_NS = 5_000
PTP_OFFSET_FLAG_NS = 3_000
PTP_OFFSET_REFUSE_NS = 10_000
FULL_VISIBILITY_REFRESHES = 2


class StimulusDisplayCandidateError(ValueError):
    """The inputs do not support a display alignment for this recording."""


def _fail(message: str) -> None:
    raise StimulusDisplayCandidateError(message)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Orange clock evidence
# --------------------------------------------------------------------------


def orange_exposure_clock(
    recording_session: Mapping[str, Any] | None,
    ptp_summary: Mapping[str, Any] | None,
    *,
    camera_serial: str,
) -> dict[str, Any]:
    """Orange's clock evidence for one camera: classification and state intervals.

    Read from ``recording_session.json`` ``timestamp_clock_contract``; older
    recordings without it fall back to the host PTP evidence in
    ``ptp_sync_summary.json``. Evidence only: the offset used for exposure time
    comes from the published frame clock, never from here.
    """

    contract = (recording_session or {}).get("timestamp_clock_contract")
    if isinstance(contract, Mapping):
        clock = (contract.get("clocks") or {}).get(f"camera_{camera_serial}")
        if not isinstance(clock, Mapping):
            _fail(f"timestamp_clock_contract has no clock for camera {camera_serial}.")
        properties = (
            ((clock.get("evidence_snapshot") or {}).get("host_ptp_management_evidence") or {})
            .get("time_properties")
            or {}
        )
        intervals = (contract.get("clock_state_intervals") or {}).get(str(camera_serial)) or []
        evidence = {
            "source": "recording_session.timestamp_clock_contract",
            "contract_schema_id": contract.get("schema_id"),
            "contract_status": contract.get("status"),
            "timescale": clock.get("timescale"),
            "classification": clock.get("classification"),
            "inference_result": (clock.get("inference") or {}).get("result"),
            "clock_state_intervals": [
                {
                    "first_recording_frame_id": int(item["first_recording_frame_id"]),
                    "last_recording_frame_id": int(item["last_recording_frame_id"]),
                    "classification": item.get("classification"),
                }
                for item in intervals
            ],
        }
    else:
        properties = ((ptp_summary or {}).get("host_ptp_management_evidence") or {}).get(
            "time_properties"
        ) or {}
        evidence = {"source": "ptp_sync_summary.host_ptp_management_evidence", "clock_state_intervals": []}
    offset = properties.get("currentUtcOffset")
    evidence["orange_reported_current_utc_offset_s"] = offset if type(offset) is int else None
    evidence["current_utc_offset_valid"] = properties.get("currentUtcOffsetValid") is True
    evidence["time_traceable"] = properties.get("timeTraceable") is True
    return evidence


def ptp_camera_evidence(ptp_summary: Mapping[str, Any] | None, *, camera_serial: str) -> dict[str, Any]:
    """Offset bounds and sampled lock-state runs; refuse beyond the contract limit."""

    camera = ((ptp_summary or {}).get("cameras") or {}).get(str(camera_serial))
    if not isinstance(camera, Mapping):
        _fail(f"ptp_sync_summary has no camera {camera_serial}.")
    offset = camera.get("ptp_offset_ns") or {}
    extrema = [abs(int(offset[key])) for key in ("min", "max") if offset.get(key) is not None]
    max_abs = max(extrema) if extrema else None
    if max_abs is not None and max_abs > PTP_OFFSET_REFUSE_NS:
        _fail(f"PTP offset {max_abs} ns exceeds the {PTP_OFFSET_REFUSE_NS} ns contract limit.")
    runs = [
        {
            "first_recording_frame_id": int(item["first_recording_frame_id"]),
            "last_recording_frame_id": int(item["last_recording_frame_id"]),
            "locked": item.get("ptp_status") == "Slave" and item.get("ptp_mode") == "TwoStep",
            "ptp_status": item.get("ptp_status"),
            "ptp_mode": item.get("ptp_mode"),
        }
        for item in camera.get("ptp_readback_observations") or []
    ]
    return {
        "ptp_offset_max_abs_ns": max_abs,
        "ptp_offset_samples": offset.get("samples"),
        "ptp_offset_flagged": max_abs is not None and max_abs > PTP_OFFSET_FLAG_NS,
        "readback_runs": runs,
    }


def clock_trusted_mask(
    recording_frame_id: np.ndarray,
    *,
    clock_state_intervals: Sequence[Mapping[str, Any]],
    readback_runs: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Frames inside Orange's declared clock interval(s) and outside unlocked PTP samples."""

    ids = np.asarray(recording_frame_id, dtype=np.int64)
    if clock_state_intervals:
        trusted = np.zeros(ids.size, dtype=bool)
        for item in clock_state_intervals:
            trusted |= (ids >= item["first_recording_frame_id"]) & (ids <= item["last_recording_frame_id"])
    else:
        trusted = np.ones(ids.size, dtype=bool)
    for run in readback_runs:
        if not run["locked"]:
            trusted &= ~((ids >= run["first_recording_frame_id"]) & (ids <= run["last_recording_frame_id"]))
    return trusted


def cadence_ok_mask(camera_ns: np.ndarray, valid: np.ndarray, *, period_ns: float) -> np.ndarray:
    """False where a frame's delta from the previous valid frame is off-period by > 5 µs."""

    stamps = np.asarray(camera_ns, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    ok = valid.copy()
    index = np.flatnonzero(valid)
    if index.size > 1:
        steps = np.diff(index)
        deviation = np.abs(np.diff(stamps[index]).astype(np.float64) - steps * period_ns)
        ok[index[1:][deviation > CADENCE_FLAG_NS]] = False
    return ok


# --------------------------------------------------------------------------
# Candidate range
# --------------------------------------------------------------------------


def display_candidates(
    exposure_host_ns: np.ndarray,
    exposure_valid: np.ndarray,
    *,
    swap_end_ns: np.ndarray,
    stimulus_frame_num: np.ndarray,
    full_visibility_ns: int,
    display_end_ns: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First/last candidate stimulus state per exposure (latency 0), and last-swap age.

    ``swap_end_ns`` and ``stimulus_frame_num`` list submitted states only, in
    order. ``NO_STATE`` means no state had been submitted by that bound, or the
    exposure is later than ``display_end_ns + full_visibility_ns`` (Citrus had
    stopped submitting, so what the screen showed is unknown).
    """

    swaps = np.asarray(swap_end_ns, dtype=np.int64)
    states = np.asarray(stimulus_frame_num, dtype=np.int64)
    if swaps.size and np.any(np.diff(swaps) < 0):
        _fail("Display swap times must not decrease.")
    exposure = np.asarray(exposure_host_ns, dtype=np.int64)
    valid = np.asarray(exposure_valid, dtype=bool)
    last = np.searchsorted(swaps, exposure, side="right") - 1
    first = np.searchsorted(swaps, exposure - int(full_visibility_ns), side="right") - 1
    last_state = np.where(last >= 0, states[np.maximum(last, 0)], NO_STATE)
    first_state = np.where(first >= 0, states[np.maximum(first, 0)], NO_STATE)
    age = np.where(last >= 0, exposure - swaps[np.maximum(last, 0)], NO_STATE)
    if display_end_ns is not None:
        valid = valid & (exposure <= int(display_end_ns) + int(full_visibility_ns))
    for array in (first_state, last_state, age):
        array[~valid] = NO_STATE
    return first_state.astype(np.int64), last_state.astype(np.int64), age.astype(np.int64)


def driven_states(
    recording_frame_id: np.ndarray,
    *,
    source_recording_frame_id: np.ndarray,
    source_recording_frame_valid: np.ndarray,
    stimulus_frame_num: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """First/last stimulus state computed from each camera frame.

    Citrus repeats the last received camera frame id until a new one arrives,
    so the states sharing an id form one contiguous run. ``NO_STATE`` means no
    state was computed from that frame (Citrus skipped it, or it fell outside
    the session).
    """

    frames = np.asarray(recording_frame_id, dtype=np.int64)
    states = np.asarray(stimulus_frame_num, dtype=np.int64)
    valid = np.asarray(source_recording_frame_valid, dtype=bool)
    sources = np.asarray(source_recording_frame_id, dtype=np.int64)
    if states.size and np.any(np.diff(states) <= 0):
        _fail("Stimulus frame numbers must increase.")
    sources, states = sources[valid], states[valid]
    if sources.size and np.any(np.diff(sources) < 0):
        _fail("Citrus source recording frame ids must not decrease.")
    if frames.size and np.any(np.diff(frames) <= 0):
        _fail("Clock recording frame ids must increase.")
    position = np.searchsorted(frames, sources)
    inside = position < frames.size
    if not np.all(inside) or not np.array_equal(frames[position], sources):
        _fail("Citrus names recording frame ids that are not in the frame clock.")
    if not sources.size:
        empty = np.full(frames.size, NO_STATE, dtype=np.int64)
        return empty, empty.copy()
    left = np.searchsorted(sources, frames, side="left")
    right = np.searchsorted(sources, frames, side="right")
    drove = right > left
    first = np.where(drove, states[np.minimum(left, states.size - 1)], NO_STATE)
    last = np.where(drove, states[np.maximum(right - 1, 0)], NO_STATE)
    return first.astype(np.int64), last.astype(np.int64)


# --------------------------------------------------------------------------
# Archive-level build and publication
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StimulusDisplayCandidates:
    recording_frame_id: np.ndarray
    exposure_host_ns: np.ndarray
    exposure_valid: np.ndarray
    clock_trusted: np.ndarray
    cadence_ok: np.ndarray
    first_candidate_stimulus_frame_num: np.ndarray
    last_candidate_stimulus_frame_num: np.ndarray
    last_candidate_swap_age_ns: np.ndarray
    first_driven_stimulus_frame_num: np.ndarray
    last_driven_stimulus_frame_num: np.ndarray
    record: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _unified_stimulus_run(root: Any, stimulus_run: str | None) -> str:
    runs = root["analysis/stimulus_runs"]
    if stimulus_run is not None:
        return stimulus_run
    names = [
        name
        for name in runs.group_keys()
        if runs[name].attrs.get("source_profile") == UNIFIED_PROFILE
    ]
    if len(names) != 1:
        _fail(f"Expected exactly one unified stimulus run; found {sorted(names)}.")
    return names[0]


def _session_number(source: UnifiedSource, name: str) -> float:
    descriptor = source.typed_attributes("/metadata/session").get(name)
    if descriptor is None:
        _fail(f"/metadata/session lacks {name}.")
    raw = bytes.fromhex(descriptor["payload_hex"])
    return float(np.frombuffer(raw, dtype="<f8")[0])


def build_stimulus_display_candidates(
    archive: str | Path, *, stimulus_run: str | None = None
) -> StimulusDisplayCandidates:
    archive = Path(archive).resolve()
    recording_dir = archive.parent.parent
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    resolved, clock = load_published_acquisition_frame_clock(root, recording_dir=recording_dir)
    serial = resolved.camera_id
    fps = get_fps(root)
    if not fps:
        _fail("Recording has no fps.")

    session_path = recording_dir / "raw/acquisition/recording_session.json"
    ptp_path = recording_dir / "raw/ptp_sync_summary.json"
    snapshot_path = recording_dir / "raw/acquisition/recording_snapshot_start.json"
    session, ptp, snapshot = _read_json(session_path), _read_json(ptp_path), _read_json(snapshot_path)
    if ptp is None:
        _fail(f"Missing {ptp_path}.")
    clock_evidence = orange_exposure_clock(session, ptp, camera_serial=serial)
    ptp_evidence = ptp_camera_evidence(ptp, camera_serial=serial)
    tai_minus_utc = camera_tai_minus_utc(resolved)
    if tai_minus_utc["seconds"] is None:
        _fail("The published frame clock gives no TAI-UTC offset for the camera; exposure time is unknown.")
    exposure_us = (((snapshot or {}).get("camera_runtime") or {}).get(serial) or {}).get("runtime", {}).get("exposure")

    offset_ns = int(tai_minus_utc["seconds"]) * 1_000_000_000
    camera_valid = np.asarray(clock.camera_timestamp_valid, dtype=bool)
    exposure_host = np.where(
        camera_valid, np.asarray(clock.camera_timestamp_ns, dtype=np.int64) - offset_ns, MISSING_TIMESTAMP_SENTINEL
    )
    # Orange's receive stamp follows exposure; a negative delay means the offset is wrong.
    both = camera_valid & np.asarray(clock.system_timestamp_valid, dtype=bool)
    receive_delay = np.asarray(clock.system_timestamp_ns, dtype=np.int64)[both] - exposure_host[both]
    if receive_delay.size and int(receive_delay.min()) < 0:
        _fail("Host receipt precedes camera exposure; the TAI-UTC offset does not hold.")

    trusted = clock_trusted_mask(
        clock.recording_frame_id,
        clock_state_intervals=clock_evidence["clock_state_intervals"],
        readback_runs=ptp_evidence["readback_runs"],
    )
    cadence = cadence_ok_mask(clock.camera_timestamp_ns, camera_valid, period_ns=1e9 / float(fps))

    run_name = _unified_stimulus_run(root, stimulus_run)
    source = open_unified_source(root, run_name=run_name)
    submissions = load_stimulus_display_submissions(source)
    refresh_hz = _session_number(source, "display_refresh_rate_reported_hz")
    full_visibility_ns = int(round(FULL_VISIBILITY_REFRESHES * 1e9 / refresh_hz))
    submitted = submissions.submitted
    first, last, age = display_candidates(
        exposure_host,
        camera_valid & trusted,
        swap_end_ns=submissions.first_swap_end_ns_epoch[submitted],
        stimulus_frame_num=submissions.stimulus_frame_num[submitted],
        full_visibility_ns=full_visibility_ns,
        display_end_ns=int(submissions.last_swap_end_ns_epoch[submitted].max()),
    )

    frame_sources = source.read_table(FRAME_SOURCES)
    driven_first, driven_last = driven_states(
        clock.recording_frame_id,
        source_recording_frame_id=frame_sources["source_recording_frame_id"],
        source_recording_frame_valid=frame_sources["source_recording_frame_valid"],
        stimulus_frame_num=frame_sources["stimulus_frame_num"],
    )
    source_valid = np.asarray(frame_sources["source_recording_frame_valid"], dtype=bool)

    record = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "row_axis": "acquisition_parent_frame_index",
        "camera_id": serial,
        "acquisition_frame_clock_sha256": resolved.record_sha256,
        "stimulus_run": run_name,
        "unified_reference_sha256": source.reference_sha256,
        "exposure_clock": {
            **clock_evidence,
            "tai_minus_utc_seconds": tai_minus_utc["seconds"],
            "tai_minus_utc_authority": tai_minus_utc["authority"],
            "tai_minus_utc_source": "published_acquisition_frame_clock",
            "exposure_host_rule": "camera_timestamp_ns - tai_minus_utc_seconds * 1e9",
            "exposure_point": "undeclared_by_vendor",
            "exposure_us": exposure_us,
            "exposure_uncertainty_ns": None if exposure_us is None else int(exposure_us) * 500,
            "host_receive_delay_ns": {
                "min": int(receive_delay.min()) if receive_delay.size else None,
                "median": int(np.median(receive_delay)) if receive_delay.size else None,
                "p999": int(np.percentile(receive_delay, 99.9)) if receive_delay.size else None,
            },
        },
        "ptp": ptp_evidence,
        "driving": {
            "method": DRIVING_METHOD,
            "source_table": FRAME_SOURCES,
            "states": int(source_valid.size),
            "states_without_source_frame": int(np.count_nonzero(~source_valid)),
        },
        "display": {
            "timing_semantics": submissions.timing_semantics,
            "display_refresh_rate_reported_hz": refresh_hz,
            "display_submission_rate_target_hz": submissions.display_rate_target_hz,
            "glfw_swap_interval": submissions.glfw_swap_interval,
            "full_visibility_refreshes": FULL_VISIBILITY_REFRESHES,
            "full_visibility_ns": full_visibility_ns,
            "projector_latency_ns_assumed": 0,
            "projector_latency_status": "unmeasured_awaiting_photodiode",
            "states": int(submissions.stimulus_frame_num.size),
            "never_submitted_states": submissions.never_submitted_count,
            "held_states": submissions.held_state_count,
            "invalid_prefix_rows": submissions.invalid_prefix_rows,
        },
        "thresholds_ns": {
            "cadence_flag": CADENCE_FLAG_NS,
            "ptp_offset_flag": PTP_OFFSET_FLAG_NS,
            "ptp_offset_refuse": PTP_OFFSET_REFUSE_NS,
        },
        "counts": {
            "frames": int(camera_valid.size),
            "clock_untrusted_frames": int(np.count_nonzero(~trusted)),
            "cadence_flagged_frames": int(np.count_nonzero(camera_valid & ~cadence)),
            "frames_with_candidates": int(np.count_nonzero(last != NO_STATE)),
            "frames_driving_states": int(np.count_nonzero(driven_last != NO_STATE)),
        },
    }
    return StimulusDisplayCandidates(
        recording_frame_id=np.asarray(clock.recording_frame_id, dtype=np.int64),
        exposure_host_ns=exposure_host,
        exposure_valid=camera_valid,
        clock_trusted=trusted,
        cadence_ok=cadence,
        first_candidate_stimulus_frame_num=first,
        last_candidate_stimulus_frame_num=last,
        last_candidate_swap_age_ns=age,
        first_driven_stimulus_frame_num=driven_first,
        last_driven_stimulus_frame_num=driven_last,
        record=record,
    )


ARRAYS = (
    "recording_frame_id",
    "exposure_host_ns",
    "exposure_valid",
    "clock_trusted",
    "cadence_ok",
    "first_candidate_stimulus_frame_num",
    "last_candidate_stimulus_frame_num",
    "last_candidate_swap_age_ns",
    "first_driven_stimulus_frame_num",
    "last_driven_stimulus_frame_num",
)


def publish_stimulus_display_candidate_run(
    archive: str | Path, result: StimulusDisplayCandidates, *, run_name: str
) -> str:
    """Write a selector-ineligible run; consumers bind to it by explicit path."""

    archive = Path(archive).resolve()
    with archive_metadata_publication_lock(archive):
        return _publish_locked(archive, result, run_name=run_name)


def _publish_locked(archive: Path, result: StimulusDisplayCandidates, *, run_name: str) -> str:
    root = zarr.open_group(str(archive), mode="r+", use_consolidated=False)
    parent = require_runs_parent(root.require_group("analysis"), PARENT_NAME)
    if run_name in parent:
        _fail(f"Run already exists: analysis/{PARENT_NAME}/{run_name}")
    run = parent.create_group(run_name)
    mark_run_started(run, run_name=run_name, stage="stimulus_display_candidates")
    run.attrs["stage_selector_eligible"] = False
    try:
        for name in ARRAYS:
            values = np.asarray(getattr(result, name))
            run.create_array(name, data=values, chunks=(min(131_072, max(1, values.size)),))
        record = dict(result.record)
        record["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        run.attrs.update(json_attr_safe({"record": record, "arrays": list(ARRAYS)}))
        source_refs = {
            "acquisition_frame_clock_sha256": record["acquisition_frame_clock_sha256"],
            "stimulus_run": record["stimulus_run"],
            "unified_reference_sha256": record["unified_reference_sha256"],
        }
        parameters = {"method": METHOD, "driving_method": DRIVING_METHOD, "thresholds_ns": record["thresholds_ns"], "display": record["display"]}
        write_run_lineage_attrs(
            run,
            build_run_lineage_payload(
                run_family=f"analysis/{PARENT_NAME}",
                analysis_schema={"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION},
                method=METHOD,
                method_version=str(SCHEMA_VERSION),
                source_refs=source_refs,
                parameters=parameters,
            ),
            fingerprint_status="best_effort",
            overwrite=True,
        )
        mark_run_complete(
            run,
            parent_group=None,
            run_name=run_name,
            run_provenance=build_writer_run_provenance(
                command="fisheye.analysis.stimulus_display_candidates",
                params=parameters,
                input_run_ids=source_refs,
            ),
        )
    except Exception as exc:
        mark_run_failed(run, parent_group=None, run_name=run_name, error=str(exc))
        raise
    consolidate_metadata_capture_expected_warnings(archive)
    return f"analysis/{PARENT_NAME}/{run_name}"
