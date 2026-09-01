from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fisheye.analysis_workflows.controller_trial_successor import (
    ControllerTrialInput,
    prepare_controller_trial_successor,
)
from fisheye.analysis_workflows.escape_freeze_successor import (
    RESPONSE_CLASS_ESCAPE,
    RESPONSE_CLASS_FREEZE,
    TRACE_REASON_NO_POST_EVENT_DISTANCE,
    EscapeFreezeInput,
    EscapeFreezeSuccessorError,
    prepare_escape_freeze_successor,
)
from fisheye.analysis_workflows.generalized_bout_response_successor import (
    prepare_generalized_bout_response_successor,
)
from tests.unit.fisheye.test_generalized_bout_response_successor import (
    _motion_projection,
    _source as _bout_source,
)


def _semantic_binding() -> dict[str, object]:
    return {
        "run_name": "semantic-v1",
        "run_path": "analysis/protocol_semantic_chaser_selection_runs/semantic-v1",
        "manifest_sha256": "a" * 64,
        "selection_identity_sha256": "b" * 64,
        "protocol_semantic_hash": "sha256:" + "c" * 64,
        "trial_index_integrity_status": "producer_authored_exact_bytes_sha256",
        "roles": ["chaser_pre", "chaser_training", "chaser_post"],
        "semantic_role_bindings": [{"analysis_role": "chaser_training"}],
        "selector_eligible": False,
        "production_authority": False,
    }


def _dependencies():
    frames = np.arange(100, 105, dtype=np.int64)
    controller = prepare_controller_trial_successor(
        ControllerTrialInput(
            recording_id="recording-1",
            source_run_path="analysis/chaser_relative_frame_runs/r1",
            source_manifest_sha256="a" * 64,
            n_frames=5,
            n_chasers=1,
            acquisition_frame_id=frames,
            timestamp_ns=frames * 1_000_000_000,
            timestamp_valid=np.ones(5, dtype=bool),
            chaser_identity_code=np.ones(5, dtype=np.uint16),
            selection_member=np.ones(5, dtype=bool),
            chaser_occurrence_member=np.ones(5, dtype=bool),
            trial_id=np.ones(5, dtype=np.int64),
            trial_valid=np.ones(5, dtype=bool),
            active_state_code=np.ones(5, dtype=np.uint8),
            active_state_valid=np.ones(5, dtype=bool),
            semantic_selection_binding=_semantic_binding(),
        )
    )
    bout_input = replace(
        _bout_source(),
        source_controller_trial_payload_sha256=controller.payload_digest,
        controller_trial_row_id=controller.array("trial_row_id_by_source_row"),
        controller_trial_envelope_row_id=controller.array(
            "trial_envelope_row_id_by_source_row"
        ),
        controller_trial_gap_reason_code=controller.array(
            "trial_gap_reason_code_by_source_row"
        ),
        bout_peak_speed_mm_s=np.asarray([25.0, 5.0], dtype=np.float32),
        body_heading_deg_by_frame=np.asarray(
            [0.0, 0.0, 60.0, 60.0, 50.0], dtype=np.float32
        ),
    )
    bout = prepare_generalized_bout_response_successor(bout_input)
    return controller, bout


def _source() -> EscapeFreezeInput:
    controller, bout = _dependencies()
    return EscapeFreezeInput(
        recording_id="recording-1",
        source_motion_run_path="analysis/track_kinematics_runs/provider/m1",
        source_motion_manifest_sha256="b" * 64,
        source_speed_level="filtered",
        source_motion_frame_projection=_motion_projection(),
        controller_trials=controller,
        bout_response=bout,
        n_frames=5,
        n_chasers=1,
        acquisition_frame_id_by_frame=np.arange(100, 105, dtype=np.int64),
        timestamp_ns_by_frame=np.arange(5, dtype=np.int64) * 1_000_000_000,
        timestamp_valid_by_frame=np.ones(5, dtype=bool),
        speed_mm_s_by_frame=np.ones(5, dtype=np.float32),
        speed_valid_by_frame=np.ones(5, dtype=bool),
        chaser_identity_code=np.ones(5, dtype=np.uint16),
        distance_mm=np.asarray([10.0, 8.0, 12.0, 7.0, 6.0], dtype=np.float32),
        distance_valid=np.ones(5, dtype=bool),
        escape_speed_threshold_mm_s=20.0,
        high_turn_threshold_deg=45.0,
        freeze_window_s=2.0,
        threshold_sweep_mm_s=(10.0, 20.0, 30.0),
    )


def test_speed_escape_high_turn_and_recapture_are_separate_exact_facts() -> None:
    result = prepare_escape_freeze_successor(_source())

    assert result.n_trials == 1
    assert result.n_events == 1
    assert result.array("event_bout_id").tolist() == [101]
    assert result.array("event_high_turn").tolist() == [True]
    assert result.array("event_latency_from_trigger_s").tolist() == [1.0]
    assert result.array("event_trigger_distance_mm").tolist() == [10.0]
    assert result.array("event_recaptured").tolist() == [True]
    assert result.array("event_recapture_latency_s").tolist() == [2.0]
    assert result.array("trial_escape_event_count").tolist() == [1]
    assert result.array("trial_high_turn_escape_count").tolist() == [1]
    assert result.array("trial_envelope_frame_count").tolist() == [5]
    assert result.array("trial_gap_frame_count").tolist() == [0]
    assert result.array("trial_logged_active_id_unavailable_count").tolist() == [0]
    assert result.array("trial_valid_time_s").tolist() == [4.0]
    assert result.array("trial_escape_event_rate_per_min").tolist() == [15.0]
    assert result.array("trial_response_class_code").tolist() == [
        RESPONSE_CLASS_ESCAPE
    ]
    assert result.manifest["policy"]["high_turn_tier"].startswith("optional")
    assert result.manifest["selector_eligible"] is False


def test_freeze_candidate_requires_no_speed_escape_and_coverage() -> None:
    result = prepare_escape_freeze_successor(
        replace(_source(), escape_speed_threshold_mm_s=100.0)
    )

    assert result.n_events == 0
    assert result.array("trial_freeze_valid_fraction").tolist() == [1.0]
    assert result.array("trial_freeze_low_speed_fraction").tolist() == [1.0]
    assert result.array("trial_freeze_candidate").tolist() == [True]
    assert result.array("trial_response_class_code").tolist() == [
        RESPONSE_CLASS_FREEZE
    ]


def test_event_count_survives_unusable_recapture_trace() -> None:
    source = _source()
    valid = source.distance_valid.copy()
    valid[2:] = False
    result = prepare_escape_freeze_successor(replace(source, distance_valid=valid))

    assert result.array("recording_escape_event_count").tolist() == [1]
    assert result.array("event_trace_valid").tolist() == [False]
    assert result.array("event_trace_exclusion_reason_code").tolist() == [
        TRACE_REASON_NO_POST_EVENT_DISTANCE
    ]


def test_stale_controller_binding_is_rejected() -> None:
    source = _source()
    other_controller = replace(
        source.controller_trials,
        manifest={**dict(source.controller_trials.manifest), "payload_digest": "f" * 64},
    )
    with pytest.raises(EscapeFreezeSuccessorError, match="binding is stale"):
        prepare_escape_freeze_successor(
            replace(source, controller_trials=other_controller)
        )


def test_manifest_records_exact_speed_signal_provenance() -> None:
    result = prepare_escape_freeze_successor(_source())

    motion = result.manifest["sources"]["motion"]
    parameters = result.manifest["parameters"]
    assert motion["speed_level"] == "filtered"
    assert motion["raw_speed_level_reason"] is None
    assert parameters["escape_speed_threshold_mm_s"] == 20.0
    assert parameters["high_turn_threshold_deg"] == 45.0
    assert parameters["freeze_speed_threshold_mm_s"] == 2.0
    assert parameters["freeze_window_s"] == 2.0


def test_raw_speed_level_requires_one_recorded_reason() -> None:
    with pytest.raises(
        EscapeFreezeSuccessorError, match="requires one non-empty exact"
    ):
        prepare_escape_freeze_successor(
            replace(_source(), source_speed_level="raw")
        )
    with pytest.raises(
        EscapeFreezeSuccessorError, match="requires one non-empty exact"
    ):
        prepare_escape_freeze_successor(
            replace(
                _source(),
                source_speed_level="raw",
                source_raw_speed_level_reason="  padded  ",
            )
        )


def test_raw_speed_level_with_reason_is_recorded_in_the_manifest() -> None:
    result = prepare_escape_freeze_successor(
        replace(
            _source(),
            source_speed_level="raw",
            source_raw_speed_level_reason="noise-floor sensitivity probe",
        )
    )

    motion = result.manifest["sources"]["motion"]
    assert motion["speed_level"] == "raw"
    assert motion["raw_speed_level_reason"] == "noise-floor sensitivity probe"


def test_raw_reason_is_rejected_for_non_raw_speed_levels() -> None:
    with pytest.raises(
        EscapeFreezeSuccessorError, match="only recordable when"
    ):
        prepare_escape_freeze_successor(
            replace(
                _source(),
                source_raw_speed_level_reason="not a raw run",
            )
        )
