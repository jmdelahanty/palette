"""Frame clock v2: Orange's declared camera timescale (camera_exposure_host_clock_v1).

docs/design/2026-10-12-frame-clock-declared-tai/README.md. The declaration
block mirrors Orange's 2026-10-11 gate recording (12 boolean checks, integer
offset, boolean time properties, origin spelled with a space).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import zarr

from fisheye.shared import acquisition_frame_clock as clock
from fisheye.shared.acquisition_frame_clock import (
    AcquisitionFrameClockError,
    acquisition_frame_clock_source_sha256,
    import_acquisition_frame_clock,
    load_acquisition_frame_clock_source,
)
from tests.unit.fisheye.test_acquisition_frame_clock import (
    _write_clock_csv,
    _write_ptp_summary,
)

CAMERA = "2010093"
CHECKS = (
    "camera_evidence_available", "camera_minus_realtime_matches", "camera_ptp_enabled",
    "host_current_utc_offset_valid", "host_management_evidence_available",
    "host_ptp_timescale", "latch_agreement", "ptp_offset_bounded", "ptp_readbacks_stable",
    "sampled_readbacks_complete", "session_ptp_enabled", "source_summary_finalized",
)


def _declaration(**changes) -> dict:
    block = {
        "classification": "ieee1588_tai",
        "clock_domain": "camera_hardware_ptp",
        "clock_id": f"camera_{CAMERA}",
        "origin": "1970-01-01T00:00:00 TAI",
        "timescale": "TAI",
        "unit": "nanosecond",
        "semantic_authority": "host_ptp_management_dataset_plus_camera_lock_evidence",
        "inference": {
            "rule_id": "orange_camera_clock_classification_v3",
            "result": "direct_ptp_time_properties_pass",
            "expected_tai_minus_utc_seconds": 37,
            "checks": {name: True for name in CHECKS},
        },
        "evidence_snapshot": {
            "host_ptp_management_evidence": {
                "available": True,
                "schema_id": "orange.host_ptp.management_evidence",
                "schema_version": 1,
                "captured_at_utc": "2026-10-12T01:51:35Z",
                "pmc_exit_status": 0,
                "time_properties": {
                    "currentUtcOffset": 37,
                    "currentUtcOffsetValid": True,
                    "ptpTimescale": True,
                },
            }
        },
    }
    for path, value in changes.items():
        node = block
        *parents, leaf = path.split(".")
        for name in parents:
            node = node[name]
        node[leaf] = value
    return block


def _recording(
    tmp_path: Path,
    *,
    block: dict | None = None,
    camera_minus_system_ns: int = 36_990_853_682,
    ptp_summary: bool = True,
) -> tuple[Path, Path]:
    recording = tmp_path / "recording"
    video = recording / "cams" / f"Cam{CAMERA}_recording.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    _write_clock_csv(
        video.with_name(f"{video.stem}_meta.csv"),
        frame_count=4,
        system_start_ns=1_784_662_711_195_121_161,
        camera_minus_system_ns=camera_minus_system_ns,
    )
    if ptp_summary:
        _write_ptp_summary(recording, camera_id=CAMERA)
    if block is not None:
        manifest = recording / "raw" / "acquisition" / "recording_session.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "recording": {"started_at_utc": "2026-10-12T01:51:45Z",
                          "created_at_utc": "2026-10-12T01:51:42Z"},
            "timestamp_clock_contract": {"clocks": {f"camera_{CAMERA}": block}},
        }))
    return recording, video


def _publish(recording: Path, video: Path):
    root = zarr.open_group(str(recording / "zarr" / "a.zarr"), mode="a", zarr_format=3)
    resolved = import_acquisition_frame_clock(
        root, recording_dir=recording, camera_id=CAMERA, video_path=video, expected_frame_count=4
    )
    return root, resolved


def test_a_direct_orange_declaration_is_the_camera_authority(tmp_path):
    recording, video = _recording(tmp_path, block=_declaration())
    _root, resolved = _publish(recording, video)
    record = resolved.record
    assert (record["schema_id"], record["schema_version"]) == ("palette.acquisition_frame_clock.v2", 2)
    camera = record["clock_surfaces"]["camera_timestamp_ns"]
    assert camera["semantic_authority"] == "orange_declared"
    assert camera["tai_minus_utc_seconds"] == 37
    assert (camera["timescale"], camera["origin"], camera["time_reference_kind"]) == (
        "IEEE-1588_PTP_TAI", "1970-01-01T00:00:00_TAI", "absolute_epoch",
    )
    assert camera["clock_domain"] == "camera_hardware_ptp_clock"
    assert camera["semantic_status"] == "producer_declared_by_orange_ptp_time_properties"
    evidence = record["clock_semantic_evidence"]
    declaration = evidence["orange_declaration"]
    assert declaration["used"] is True and declaration["reason"] is None
    assert declaration["origin"] == "1970-01-01T00:00:00 TAI"  # verbatim
    assert declaration["recording_started_at_utc"] == "2026-10-12T01:51:45Z"
    assert declaration["host_evidence_captured_at_utc"] == "2026-10-12T01:51:35Z"
    assert declaration["manifest"]["locator"] == "raw/acquisition/recording_session.json"
    assert len(declaration["checks"]) == 12
    assert evidence["palette_cross_check"]["agrees"] is True
    # v1 evidence keys stay, so v1 consumers keep working on v2 records.
    assert evidence["camera_ptp_semantics_inferred"] is True
    assert record["clock_surfaces"]["system_timestamp_ns"]["origin"] == "1970-01-01T00:00:00_UTC"


@pytest.mark.parametrize(
    ("block", "reason"),
    [
        (None, "no_recording_session_json"),
        (_declaration(**{"inference.result": "fallback_inference_pass"}), "result:'fallback_inference_pass'"),
        (_declaration(classification="device_defined"), "classification:'device_defined'"),
        (_declaration(**{"inference.checks.latch_agreement": False}), "check_false:latch_agreement"),
        (_declaration(**{"inference.checks.ptp_offset_bounded": 1}), "check_false:ptp_offset_bounded"),
        # Q2: a failed pmc query is not a declaration.
        (_declaration(**{"evidence_snapshot.host_ptp_management_evidence.pmc_exit_status": 1}),
         "pmc_exit_status:1"),
        (_declaration(**{"evidence_snapshot.host_ptp_management_evidence.available": False}),
         "host_evidence_unavailable"),
        (_declaration(**{"evidence_snapshot.host_ptp_management_evidence.time_properties.currentUtcOffset": 36}),
         "current_utc_offset:36"),
        (_declaration(**{"evidence_snapshot.host_ptp_management_evidence.time_properties.currentUtcOffsetValid": 1}),
         "current_utc_offset_not_valid"),
        (_declaration(**{"inference.expected_tai_minus_utc_seconds": 37.0}), "offset_not_integer"),
    ],
)
def test_anything_but_a_direct_declaration_falls_back_to_inference(tmp_path, block, reason):
    recording, video = _recording(tmp_path, block=block)
    _root, resolved = _publish(recording, video)
    camera = resolved.record["clock_surfaces"]["camera_timestamp_ns"]
    declaration = resolved.record["clock_semantic_evidence"]["orange_declaration"]
    assert camera["semantic_authority"] == "palette_inferred"
    assert declaration["used"] is False and declaration["reason"] == reason
    # v1 inference still recognizes this PTP recording as TAI.
    assert camera["semantic_status"] == "inferred_from_recording_evidence_not_sdk_declared"
    assert camera["tai_minus_utc_seconds"] == 37


def test_an_inferred_device_clock_has_no_offset(tmp_path):
    recording, video = _recording(tmp_path, block=None, ptp_summary=False)
    _root, resolved = _publish(recording, video)
    camera = resolved.record["clock_surfaces"]["camera_timestamp_ns"]
    assert camera["semantic_authority"] == "palette_inferred"
    assert camera["time_reference_kind"] == "device_defined_unknown_epoch"
    assert camera["tai_minus_utc_seconds"] is None


def test_a_declaration_the_measured_delta_contradicts_is_refused(tmp_path):
    # Q1: Orange declares TAI-UTC 37 s, but camera - system is -1 s here.
    recording, video = _recording(tmp_path, block=_declaration(), camera_minus_system_ns=-1_000_000_000)
    with pytest.raises(AcquisitionFrameClockError, match="contradicts"):
        _publish(recording, video)


def test_old_host_evidence_is_recorded_not_gated(tmp_path):
    block = _declaration(**{
        "evidence_snapshot.host_ptp_management_evidence.captured_at_utc": "2026-10-01T00:00:00Z"
    })
    recording, video = _recording(tmp_path, block=block)
    _root, resolved = _publish(recording, video)
    declaration = resolved.record["clock_semantic_evidence"]["orange_declaration"]
    assert declaration["used"] is True
    assert declaration["host_evidence_captured_at_utc"] == "2026-10-01T00:00:00Z"


def test_a_v1_publication_replays_without_a_new_run_and_reverifies(tmp_path, monkeypatch):
    recording, video = _recording(tmp_path, block=_declaration())
    original = clock._build_record
    monkeypatch.setattr(clock, "_build_record", lambda source, schema_version=1: original(source, 1))
    root, v1 = _publish(recording, video)
    assert v1.record["schema_version"] == 1
    monkeypatch.setattr(clock, "_build_record", original)

    root, replayed = _publish(recording, video)
    assert replayed == v1  # the unchanged v1 clock stays selected
    assert list(root["analysis/acquisition_frame_clock_runs"].group_keys()) == [v1.run_name]
    source = load_acquisition_frame_clock_source(
        recording, camera_id=CAMERA, video_path=video, expected_frame_count=4
    )
    assert acquisition_frame_clock_source_sha256(source, schema_version=1) == v1.record_sha256
    assert acquisition_frame_clock_source_sha256(source) != v1.record_sha256  # v2 differs


def test_a_v2_record_must_agree_with_its_declaration_evidence(tmp_path):
    recording, video = _recording(tmp_path, block=_declaration())
    _root, resolved = _publish(recording, video)
    tampered = copy.deepcopy(dict(resolved.record))
    tampered["clock_semantic_evidence"]["orange_declaration"]["used"] = False
    with pytest.raises(AcquisitionFrameClockError, match="declaration evidence differs"):
        clock._validate_record(tampered)
    unknown = copy.deepcopy(dict(resolved.record))
    unknown["schema_version"] = 3
    with pytest.raises(AcquisitionFrameClockError, match="schema_version"):
        clock._validate_record(unknown)
    mismatched = copy.deepcopy(dict(resolved.record))
    mismatched["schema_id"] = "palette.acquisition_frame_clock.v1"  # v1 id with version 2
    with pytest.raises(AcquisitionFrameClockError, match="schema_version"):
        clock._validate_record(mismatched)
