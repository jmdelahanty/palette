from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fisheye.analysis_workflows.recording_distribution_timebase_adapter as subject
from fisheye.group_statistics.recording_distribution_scopes import (
    session_time_bracket_scope,
    whole_session_scope,
)


def _timebase() -> subject.RecordingSessionTimebase:
    return subject.RecordingSessionTimebase(
        acquisition_frame_id=np.asarray([10, 11, 13], dtype=np.int64),
        timestamp_ns_session=np.asarray([100, 110, 130], dtype=np.int64),
        timestamp_valid=np.asarray([True, False, True]),
        source_binding={"clock_run_path": "analysis/clock_runs/exact"},
    )


def test_exact_timebase_mapping_leaves_missing_and_invalid_frames_uncovered() -> None:
    timestamps, valid = _timebase().map_frames(
        np.asarray([9, 10, 11, 12, 13], dtype=np.int64)
    )

    assert timestamps.tolist() == [0, 100, 110, 0, 130]
    assert valid.tolist() == [False, True, False, False, True]
    assert len(_timebase().binding["timebase_sha256"]) == 64


def test_time_scope_must_bind_the_exact_requested_clock() -> None:
    timebase = _timebase()
    scopes = (
        whole_session_scope(),
        session_time_bracket_scope(
            scope_id="selected",
            scope_label="Selected",
            order=1,
            start_timestamp_ns_session=100,
            end_timestamp_ns_session_exclusive=130,
            timebase_binding=timebase.binding,
        ),
    )
    assert subject.require_scope_timebase_binding(scopes, timebase) is True

    other = subject.RecordingSessionTimebase(
        acquisition_frame_id=timebase.acquisition_frame_id,
        timestamp_ns_session=timebase.timestamp_ns_session,
        timestamp_valid=timebase.timestamp_valid,
        source_binding={"clock_run_path": "analysis/clock_runs/other"},
    )
    with pytest.raises(subject.RecordingDistributionTimebaseError, match="another"):
        subject.require_scope_timebase_binding(scopes, other)


def test_bundle_timebase_loader_uses_paired_relative_consensus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seal = {
        "run_path": "analysis/chaser_relative_frame_runs/exact",
        "manifest_sha256": "1" * 64,
        "payload_digest": "2" * 64,
        "receipt_path": "/relative.receipt.json",
        "receipt_sha256": "3" * 64,
    }

    class _Source:
        recording_id = "recording-1"
        bundle_sha256 = "4" * 64
        analysis_zarr = "/archive"
        bundle = {
            "source_bindings": {
                "row_axis_timing_and_scale": {
                    "binding_type": "paired_relative_frame_consensus_v1",
                    "authority": {
                        "shared_timing_semantics": {
                            "timestamp_field": "timestamp_ns_session",
                            "policy_id": "exact.v1",
                        }
                    },
                    "sealed_by": {"keypoint": seal, "detection": seal},
                }
            }
        }

        def scientific_child(self, capability: str):
            assert capability == "chaser_relative_keypoint"
            return SimpleNamespace(binding=seal)

    class _Handle:
        run_path = seal["run_path"]
        manifest_sha256 = seal["manifest_sha256"]
        payload_digest = seal["payload_digest"]
        receipt_digest = seal["receipt_sha256"]

        def frame_array(self, name: str):
            return {
                "acquisition_frame_id": np.asarray([10, 11], dtype=np.int64),
                "timestamp_ns": np.asarray([100, 110], dtype=np.int64),
                "timestamp_valid": np.asarray([True, True]),
            }[name]

    monkeypatch.setattr(subject, "ValidatedRecordingBehaviorSource", _Source)
    monkeypatch.setattr(
        subject,
        "load_chaser_relative_frame_targeted_source_handle",
        lambda *_args, **_kwargs: _Handle(),
    )

    result = subject.load_bundle_recording_session_timebase(_Source())

    assert result.acquisition_frame_id.tolist() == [10, 11]
    assert result.binding["relative_frame_receipt_sha256"] == "3" * 64
