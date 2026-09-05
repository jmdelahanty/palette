from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fisheye.analysis_workflows.controller_trial_successor import (
    TRIAL_GAP_REASON_TRIAL_ID_UNAVAILABLE,
)
from fisheye.analysis_workflows.generalized_bout_response_successor import (
    ATTACHMENT_REASON_FRAME_UNAVAILABLE,
    ATTACHMENT_REASON_TRIAL_UNAVAILABLE,
    GeneralizedBoutResponseInput,
    GeneralizedBoutResponseSuccessorError,
    ROLE_CODES,
    exact_core_motion_frame_projection,
    exact_provider_frame_projection,
    prepare_generalized_bout_response_successor,
)
from fisheye.analysis_workflows.core_motion_source_handle import (
    bind_core_motion_track_source_handle,
    core_motion_dependency_record,
)


def _motion_projection(n_frames: int = 5) -> dict[str, object]:
    return {
        "schema_id": "palette.provider_motion.relative_frame_projection",
        "schema_version": 1,
        "join_policy": "left_join_missing_provider_rows_invalid_no_interpolation",
        "relative_frame_count": n_frames,
        "fallback": "prohibited",
    }


def _source(*, body: bool = True) -> GeneralizedBoutResponseInput:
    n_frames, n_chasers = 5, 1
    kwargs = {}
    if body:
        kwargs = {
            "body_heading_deg_by_frame": np.asarray(
                [0.0, 0.0, 20.0, 20.0, 10.0], dtype=np.float32
            ),
            "body_heading_valid_by_frame": np.ones(n_frames, dtype=bool),
            "chaser_bearing_deg": np.asarray(
                [30.0, 30.0, 10.0, -40.0, -30.0], dtype=np.float32
            ),
            "chaser_bearing_valid": np.ones(n_frames, dtype=bool),
        }
    return GeneralizedBoutResponseInput(
        recording_id="recording-1",
        source_relative_frame_run_path="analysis/chaser_relative_frame_runs/r1",
        source_relative_frame_manifest_sha256="a" * 64,
        source_motion_run_path="analysis/track_kinematics_runs/provider/m1",
        source_motion_manifest_sha256="b" * 64,
        source_swim_bout_run_path="analysis/swim_bout_runs/b1",
        source_swim_bout_lineage_sha256="c" * 64,
        source_signal_id=4,
        source_signal_level="speed_filtered",
        source_semantic_selection_manifest_sha256="d" * 64,
        source_controller_trial_payload_sha256="e" * 64,
        source_motion_frame_projection=_motion_projection(n_frames),
        n_frames=n_frames,
        n_chasers=n_chasers,
        acquisition_frame_id_by_frame=np.arange(100, 105, dtype=np.int64),
        timestamp_ns_by_frame=np.arange(n_frames, dtype=np.int64) * 1_000_000_000,
        timestamp_valid_by_frame=np.ones(n_frames, dtype=bool),
        transition_valid_by_frame=np.asarray([False, True, True, True, True]),
        semantic_role_code_by_frame=np.full(
            n_frames, ROLE_CODES["chaser_training"], dtype=np.uint8
        ),
        chaser_identity_code=np.ones(n_frames, dtype=np.uint16),
        distance_mm=np.asarray([10.0, 8.0, 12.0, 20.0, 30.0], dtype=np.float32),
        distance_valid=np.ones(n_frames, dtype=bool),
        controller_trial_row_id=np.asarray([0, 0, 0, -1, -1], dtype=np.int64),
        controller_trial_envelope_row_id=np.asarray([0, 0, 0, -1, -1], dtype=np.int64),
        controller_trial_gap_reason_code=np.zeros(n_frames, dtype=np.uint8),
        bout_id=np.asarray([101, 102], dtype=np.int64),
        bout_start_acquisition_frame_id=np.asarray([101, 103], dtype=np.int64),
        bout_end_acquisition_frame_id=np.asarray([102, 104], dtype=np.int64),
        bout_peak_speed_mm_s=np.asarray([20.0, 30.0], dtype=np.float32),
        bout_mean_speed_mm_s=np.asarray([10.0, 15.0], dtype=np.float32),
        bout_duration_s=np.asarray([0.2, 0.3], dtype=np.float32),
        bout_path_length_mm=np.asarray([3.0, 5.0], dtype=np.float32),
        bout_net_displacement_mm=np.asarray([2.0, 4.0], dtype=np.float32),
        distance_bin_edges_mm=(0.0, 15.0, float("inf")),
        **kwargs,
    )


def test_exact_provider_projection_retains_missing_rows_as_invalid() -> None:
    rows, present, record = exact_provider_frame_projection(
        np.asarray([99, 100, 102, 104, 106], dtype=np.int64),
        np.asarray([100, 101, 102, 103, 104], dtype=np.int64),
    )

    assert rows.tolist() == [1, -1, 2, -1, 3]
    assert present.tolist() == [True, False, True, False, True]
    assert record["missing_relative_frame_count"] == 2
    assert record["provider_only_frame_count"] == 2
    assert record["fallback"] == "prohibited"


def test_exact_provider_projection_rejects_nonunique_axes() -> None:
    with pytest.raises(
        GeneralizedBoutResponseSuccessorError,
        match="unique and strictly increasing",
    ):
        exact_provider_frame_projection(
            np.asarray([100, 100], dtype=np.int64),
            np.asarray([100], dtype=np.int64),
        )


def test_core_motion_projection_requires_and_persists_one_sealed_dependency(
    tmp_path,
) -> None:
    from tests.unit.fisheye.test_core_authority_roster import _bound_core_motion

    bound = _bound_core_motion(tmp_path)
    handle = bind_core_motion_track_source_handle(
        bound,
        consumer_id="goodbatbadbat.composable_chaser_successors_v1",
        required_capabilities=(
            "cross_grain_join_authority",
            "kinematics_samples",
            "canonical_swim_bouts",
        ),
        track_id=7,
    )
    _rows, _present, projection = exact_core_motion_frame_projection(
        np.arange(100, 105, dtype=np.int64),
        np.arange(100, 105, dtype=np.int64),
        core_authority_roster_sha256=handle.core_authority_roster_sha256,
    )
    dependency = core_motion_dependency_record(handle)
    source = replace(
        _source(),
        recording_id=handle.recording_id,
        source_motion_run_path=handle.run_path,
        source_motion_manifest_sha256=handle.source_manifest_sha256,
        source_swim_bout_run_path=dependency["swim_bout_run_path"],
        source_swim_bout_lineage_sha256=dependency["swim_bout_source_binding_sha256"],
        source_motion_frame_projection=projection,
        source_core_authority=dependency,
    )

    result = prepare_generalized_bout_response_successor(source)

    assert (
        result.manifest["sources"]["core_authority"]["record_sha256"]
        == dependency["record_sha256"]
    )
    assert (
        result.manifest["sources"]["core_authority"]["core_authority_roster_sha256"]
        == handle.core_authority_roster_sha256
    )
    with pytest.raises(
        GeneralizedBoutResponseSuccessorError,
        match="lacks its sealed authority dependency",
    ):
        prepare_generalized_bout_response_successor(
            replace(source, source_core_authority=None)
        )


def test_builds_exact_bout_chaser_rows_and_distance_band_rates() -> None:
    result = prepare_generalized_bout_response_successor(_source())

    assert result.n_bouts == 2
    assert result.n_bout_chaser_rows == 2
    assert result.array("bout_id").tolist() == [101, 102]
    np.testing.assert_allclose(result.array("distance_at_onset_mm"), [8.0, 20.0])
    np.testing.assert_allclose(result.array("delta_distance_mm"), [4.0, 10.0])
    assert result.array("controller_trial_row_id").tolist() == [0, -1]
    assert result.array("controller_trial_envelope_row_id").tolist() == [0, -1]
    assert result.array("attachment_reason_code").tolist() == [
        0,
        ATTACHMENT_REASON_TRIAL_UNAVAILABLE,
    ]
    assert result.array("directed_valid").tolist() == [True, True]
    assert result.array("turn_toward_chaser").tolist() == [True, True]

    training = result.array("summary_role_code") == ROLE_CODES["chaser_training"]
    training_rows = np.flatnonzero(training)
    assert training_rows.tolist() == [2, 3]
    assert result.array("summary_valid_time_s")[training_rows].tolist() == [3.0, 1.0]
    assert result.array("summary_bout_count")[training_rows].tolist() == [1, 1]
    np.testing.assert_allclose(
        result.array("summary_bout_rate_per_min")[training_rows],
        [20.0, 60.0],
    )
    assert result.manifest["sources"]["swim_bouts"]["signal_id"] == 4
    assert result.manifest["policy"]["bout_signal"] == (
        "one_explicit_default_signal_only"
    )
    assert result.manifest["selector_eligible"] is False


def test_body_frame_extension_is_optional_without_direction_fallback() -> None:
    result = prepare_generalized_bout_response_successor(_source(body=False))

    assert not np.any(result.array("directed_valid"))
    assert not np.any(result.array("turn_toward_chaser"))
    assert result.manifest["scientific_schema"]["body_extension_present"] is False


def test_unmatched_bout_frame_is_retained_with_reason() -> None:
    source = _source()
    starts = source.bout_start_acquisition_frame_id.copy()
    ends = source.bout_end_acquisition_frame_id.copy()
    starts[1] = 999
    ends[1] = 1_000
    result = prepare_generalized_bout_response_successor(
        replace(
            source,
            bout_start_acquisition_frame_id=starts,
            bout_end_acquisition_frame_id=ends,
        )
    )

    assert result.array("base_valid").tolist() == [True, False]
    assert result.array("attachment_reason_code").tolist() == [
        0,
        ATTACHMENT_REASON_FRAME_UNAVAILABLE,
    ]


def test_bout_on_trial_gap_retains_envelope_and_gap_reason_without_attachment() -> None:
    source = _source()
    exact = source.controller_trial_row_id.copy()
    exact[1] = -1
    reasons = source.controller_trial_gap_reason_code.copy()
    reasons[1] = TRIAL_GAP_REASON_TRIAL_ID_UNAVAILABLE

    result = prepare_generalized_bout_response_successor(
        replace(
            source,
            controller_trial_row_id=exact,
            controller_trial_gap_reason_code=reasons,
        )
    )

    assert result.array("controller_trial_row_id")[0] == -1
    assert result.array("controller_trial_envelope_row_id")[0] == 0
    assert result.array("controller_trial_gap_reason_code")[0] == (
        TRIAL_GAP_REASON_TRIAL_ID_UNAVAILABLE
    )
    assert result.array("attachment_reason_code")[0] == (
        ATTACHMENT_REASON_TRIAL_UNAVAILABLE
    )


def test_duplicate_bout_ids_are_rejected_as_signal_duplication() -> None:
    source = _source()
    with pytest.raises(
        GeneralizedBoutResponseSuccessorError, match="duplicate bout IDs"
    ):
        prepare_generalized_bout_response_successor(
            replace(source, bout_id=np.asarray([101, 101], dtype=np.int64))
        )


def test_partial_body_extension_is_rejected() -> None:
    source = _source()
    with pytest.raises(
        GeneralizedBoutResponseSuccessorError, match="all present or all absent"
    ):
        prepare_generalized_bout_response_successor(
            replace(source, chaser_bearing_valid=None)
        )
