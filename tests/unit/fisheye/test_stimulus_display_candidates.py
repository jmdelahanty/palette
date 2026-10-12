from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis.stimulus_display_candidates import (
    NO_STATE,
    StimulusDisplayCandidateError,
    cadence_ok_mask,
    clock_trusted_mask,
    display_candidates,
    driven_states,
    orange_exposure_clock,
    ptp_camera_evidence,
)

MS = 1_000_000


def _session(valid: bool) -> dict:
    return {
        "timestamp_clock_contract": {
            "schema_id": "orange.recording.timestamp_clocks",
            "status": "finalized",
            "clocks": {
                "camera_2010093": {
                    "timescale": "TAI" if valid else "device_defined",
                    "classification": "ieee1588_tai" if valid else "device_defined",
                    "inference": {"result": "direct_ptp_time_properties_pass"},
                    "evidence_snapshot": {
                        "host_ptp_management_evidence": {
                            "time_properties": {"currentUtcOffset": 37, "currentUtcOffsetValid": valid}
                        }
                    },
                }
            },
            "clock_state_intervals": {
                "2010093": [{"first_recording_frame_id": 1, "last_recording_frame_id": 10, "classification": "x"}]
            },
        }
    }


def test_offset_comes_from_the_recordings_clock_contract() -> None:
    evidence = orange_exposure_clock(_session(True), None, camera_serial="2010093")
    assert evidence["current_utc_offset_s"] == 37
    assert evidence["current_utc_offset_valid"] is True
    assert evidence["clock_state_intervals"][0]["last_recording_frame_id"] == 10
    assert orange_exposure_clock(_session(False), None, camera_serial="2010093")["current_utc_offset_valid"] is False


def test_offset_falls_back_to_ptp_summary_and_refuses_when_absent() -> None:
    ptp = {"host_ptp_management_evidence": {"time_properties": {"currentUtcOffset": 37}}}
    assert orange_exposure_clock(None, ptp, camera_serial="1")["source"].startswith("ptp_sync_summary")
    with pytest.raises(StimulusDisplayCandidateError, match="currentUtcOffset"):
        orange_exposure_clock(None, {}, camera_serial="1")


def _ptp(lo: int, hi: int, status: str = "Slave") -> dict:
    return {
        "cameras": {
            "1": {
                "ptp_offset_ns": {"min": lo, "max": hi, "samples": 5},
                "ptp_readback_observations": [
                    {"first_recording_frame_id": 0, "last_recording_frame_id": 9, "ptp_status": status, "ptp_mode": "TwoStep"}
                ],
            }
        }
    }


def test_ptp_offset_flag_and_refusal_follow_the_contract() -> None:
    assert ptp_camera_evidence(_ptp(-1015, -739), camera_serial="1")["ptp_offset_flagged"] is False
    assert ptp_camera_evidence(_ptp(-3500, 100), camera_serial="1")["ptp_offset_flagged"] is True
    with pytest.raises(StimulusDisplayCandidateError, match="exceeds"):
        ptp_camera_evidence(_ptp(-12_000, 0), camera_serial="1")


def test_unlocked_ptp_runs_and_undeclared_frames_are_untrusted() -> None:
    ids = np.arange(1, 13)
    locked = ptp_camera_evidence(_ptp(0, 0), camera_serial="1")["readback_runs"]
    intervals = [{"first_recording_frame_id": 1, "last_recording_frame_id": 10}]
    trusted = clock_trusted_mask(ids, clock_state_intervals=intervals, readback_runs=locked)
    np.testing.assert_array_equal(trusted, [True] * 10 + [False] * 2)
    unlocked = ptp_camera_evidence(_ptp(0, 0, status="Listening"), camera_serial="1")["readback_runs"]
    assert not clock_trusted_mask(ids, clock_state_intervals=[], readback_runs=unlocked)[:9].any()


def test_cadence_flags_off_period_frames() -> None:
    stamps = np.array([0, 10, 20, 30.006, 40.006, 50.006]) * MS
    ok = cadence_ok_mask(stamps.astype(np.int64), np.ones(6, bool), period_ns=10 * MS)
    np.testing.assert_array_equal(ok, [True, True, True, False, True, True])


def test_candidate_range_spans_the_full_visibility_window() -> None:
    swaps = np.array([0, 8, 16, 24, 32]) * MS + 1_000 * MS
    states = np.array([10, 11, 13, 14, 15])  # state 12 was never submitted
    exposure = np.array([999, 1_000, 1_017, 1_020, 1_060, 1_100]) * MS
    first, last, age = display_candidates(
        exposure,
        np.array([True, True, True, True, True, False]),
        swap_end_ns=swaps,
        stimulus_frame_num=states,
        full_visibility_ns=16_700_000,
        display_end_ns=int(swaps.max()),
    )
    np.testing.assert_array_equal(last, [NO_STATE, 10, 13, 13, NO_STATE, NO_STATE])
    # 1_020 ms - 16.7 ms = 1_003.3 ms: the latest swap by then is state 10 (1_000 ms).
    np.testing.assert_array_equal(first, [NO_STATE, NO_STATE, 10, 10, NO_STATE, NO_STATE])
    assert age[2] == 1 * MS


def test_driven_states_follow_citrus_correspondence_ids() -> None:
    # Clock frames 1..6. Citrus held frame 2 for two states, skipped frame 4,
    # and had no source frame for its first state.
    first, last = driven_states(
        np.arange(1, 7),
        source_recording_frame_id=np.array([0, 2, 2, 3, 5, 6]),
        source_recording_frame_valid=np.array([0, 1, 1, 1, 1, 1]),
        stimulus_frame_num=np.arange(10, 16),
    )
    assert first.tolist() == [NO_STATE, 11, 13, NO_STATE, 14, 15]
    assert last.tolist() == [NO_STATE, 12, 13, NO_STATE, 14, 15]


def test_driven_states_refuse_ids_outside_the_clock_or_out_of_order() -> None:
    kwargs = dict(source_recording_frame_valid=np.ones(2, dtype=bool), stimulus_frame_num=np.array([0, 1]))
    with pytest.raises(StimulusDisplayCandidateError, match="not in the frame clock"):
        driven_states(np.arange(1, 4), source_recording_frame_id=np.array([3, 4]), **kwargs)
    with pytest.raises(StimulusDisplayCandidateError, match="must not decrease"):
        driven_states(np.arange(1, 4), source_recording_frame_id=np.array([3, 2]), **kwargs)
    first, last = driven_states(
        np.arange(1, 4),
        source_recording_frame_id=np.array([0, 0]),
        source_recording_frame_valid=np.zeros(2, dtype=bool),
        stimulus_frame_num=np.array([0, 1]),
    )
    assert first.tolist() == last.tolist() == [NO_STATE] * 3


def test_publication_holds_the_archive_lock(tmp_path, monkeypatch) -> None:
    import zarr

    from fisheye.analysis import stimulus_display_candidates as module
    from fisheye.shared.zarr_helpers import archive_publication_lock_held_by_current_thread

    archive = tmp_path / "rec_analysis.zarr"
    zarr.open_group(str(archive), mode="w", zarr_format=3)
    seen = []
    monkeypatch.setattr(
        module, "_publish_locked",
        lambda path, result, run_name: seen.append(archive_publication_lock_held_by_current_thread(path)) or "ok",
    )
    assert module.publish_stimulus_display_candidate_run(archive, None, run_name="r") == "ok"
    assert seen == [True]
