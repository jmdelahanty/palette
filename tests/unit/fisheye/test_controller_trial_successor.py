from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fisheye.analysis_workflows.controller_trial_successor import (
    TRIAL_GAP_REASON_EXPLICITLY_INACTIVE,
    ControllerTrialInput,
    ControllerTrialSuccessorError,
    prepare_controller_trial_successor,
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


def _source() -> ControllerTrialInput:
    # frame-major/chaser-minor. Chaser 1 trial 10 has a preserved gap at frame
    # 2; chaser 2 has two adjacent exact trials whose ordinals are independent.
    n_frames, n_chasers = 6, 2
    acquisition = np.repeat(np.arange(100, 106, dtype=np.int64), n_chasers)
    chaser = np.tile(np.asarray([1, 2], dtype=np.uint16), n_frames)
    trial = np.asarray(
        [10, 20, 10, 20, -1, 21, 10, 21, -1, -1, -1, -1],
        dtype=np.int64,
    )
    valid = trial >= 0
    active = valid.astype(np.uint8)
    return ControllerTrialInput(
        recording_id="recording-1",
        source_run_path="analysis/chaser_relative_frame_runs/relative-v1",
        source_manifest_sha256="d" * 64,
        n_frames=n_frames,
        n_chasers=n_chasers,
        acquisition_frame_id=acquisition,
        timestamp_ns=acquisition * 1_000,
        timestamp_valid=np.ones(n_frames * n_chasers, dtype=bool),
        chaser_identity_code=chaser,
        selection_member=np.ones(n_frames * n_chasers, dtype=bool),
        chaser_occurrence_member=np.ones(n_frames * n_chasers, dtype=bool),
        trial_id=trial,
        trial_valid=valid,
        active_state_code=active,
        active_state_valid=np.ones(n_frames * n_chasers, dtype=bool),
        semantic_selection_binding=_semantic_binding(),
    )


def test_exact_logged_trials_preserve_gaps_and_per_chaser_ordinals() -> None:
    result = prepare_controller_trial_successor(_source())

    assert result.n_trials == 3
    assert result.array("chaser_identity_code").tolist() == [1, 2, 2]
    assert result.array("logged_trial_id").tolist() == [10, 20, 21]
    assert result.array("trial_ordinal").tolist() == [1, 1, 2]
    assert result.array("active_member_count").tolist() == [3, 2, 2]
    assert result.array("envelope_frame_count").tolist() == [4, 2, 2]
    assert result.array("gap_frame_count").tolist() == [1, 0, 0]
    assert np.flatnonzero(result.array("trial_gap_member")).tolist() == [4]
    assert result.array("trial_row_id_by_source_row").tolist() == [
        0,
        1,
        0,
        1,
        -1,
        2,
        0,
        2,
        -1,
        -1,
        -1,
        -1,
    ]
    assert result.array("trial_envelope_row_id_by_source_row").tolist() == [
        0,
        1,
        0,
        1,
        0,
        2,
        0,
        2,
        -1,
        -1,
        -1,
        -1,
    ]
    assert result.array("trial_gap_reason_code_by_source_row")[4] == (
        TRIAL_GAP_REASON_EXPLICITLY_INACTIVE
    )
    assert not np.any(result.array("fallback_used"))
    assert result.manifest["policy"]["fallback"] == "prohibited_fail_closed"
    assert result.manifest["policy"]["legacy_contiguous_interval_reconstruction"] == (
        "rejected"
    )
    assert result.manifest["selector_eligible"] is False
    assert result.manifest["semantic_selection"]["run_name"] == "semantic-v1"
    assert all(not value.flags.writeable for value in result.arrays.values())


def test_active_row_without_logged_trial_fails_closed() -> None:
    source = _source()
    active = source.active_state_code.copy()
    active[4] = 1

    with pytest.raises(
        ControllerTrialSuccessorError,
        match="legacy contiguous-interval trial reconstruction is prohibited",
    ):
        prepare_controller_trial_successor(replace(source, active_state_code=active))


def test_valid_negative_trial_id_is_rejected() -> None:
    source = _source()
    valid = source.trial_valid.copy()
    valid[8] = True

    with pytest.raises(ControllerTrialSuccessorError, match="strictly positive"):
        prepare_controller_trial_successor(replace(source, trial_valid=valid))


def test_zero_logged_trial_id_marked_valid_is_rejected() -> None:
    source = _source()
    valid = source.trial_valid.copy()
    valid[8] = True
    trial = source.trial_id.copy()
    trial[8] = 0

    with pytest.raises(ControllerTrialSuccessorError, match="strictly positive"):
        prepare_controller_trial_successor(
            replace(source, trial_id=trial, trial_valid=valid)
        )


def test_missing_explicit_active_state_on_eligible_row_is_rejected() -> None:
    source = _source()
    active_valid = source.active_state_valid.copy()
    active_valid[0] = False

    with pytest.raises(
        ControllerTrialSuccessorError,
        match="legacy active-interval reconstruction is prohibited",
    ):
        prepare_controller_trial_successor(
            replace(source, active_state_valid=active_valid)
        )


def test_changed_chaser_identity_axis_is_rejected() -> None:
    source = _source()
    identity = source.chaser_identity_code.copy()
    identity[-1] = 3

    with pytest.raises(ControllerTrialSuccessorError, match="identity codes changed"):
        prepare_controller_trial_successor(
            replace(source, chaser_identity_code=identity)
        )


def test_interleaved_logged_trials_with_overlapping_envelopes_are_rejected() -> None:
    source = _source()
    trial = source.trial_id.copy()
    trial[[0, 2, 4]] = np.asarray([10, 11, 10], dtype=np.int64)
    valid = source.trial_valid.copy()
    valid[4] = True
    active = source.active_state_code.copy()
    active[4] = 1

    with pytest.raises(ControllerTrialSuccessorError, match="envelopes overlap"):
        prepare_controller_trial_successor(
            replace(
                source,
                trial_id=trial,
                trial_valid=valid,
                active_state_code=active,
            )
        )


def test_semantic_source_must_remain_selector_ineligible() -> None:
    source = _source()
    binding = _semantic_binding()
    binding["selector_eligible"] = True

    with pytest.raises(ControllerTrialSuccessorError, match="selector-ineligible"):
        prepare_controller_trial_successor(
            replace(source, semantic_selection_binding=binding)
        )
