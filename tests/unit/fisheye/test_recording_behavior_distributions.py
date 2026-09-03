from __future__ import annotations

import numpy as np
import pytest

from fisheye.group_statistics.recording_behavior_distributions import (
    RecordingBehaviorDistributionConfig,
    RecordingBehaviorDistributionError,
    RecordingDistributionMetricInput,
    canonical_grid_indices,
    compute_recording_behavior_distributions,
)
from fisheye.group_statistics.recording_distribution_scopes import (
    exact_source_membership_masks,
    frame_interval_scope,
    sample_scope_masks,
    transition_scope_masks,
    whole_session_scope,
)
from fisheye.group_statistics.validated_behavior_distribution_specs import (
    DistributionMetricSpec,
)


def _scopes():
    return (
        whole_session_scope(),
        frame_interval_scope(
            scope_id="phase_a",
            scope_label="Phase A",
            scope_family="fixture_phase",
            scope_provider_id="fixture_intervals.v1",
            order=1,
            start_frame=10,
            end_frame_exclusive=20,
            source_binding={"interval_digest": "a" * 64},
        ),
        frame_interval_scope(
            scope_id="phase_b",
            scope_label="Phase B",
            scope_family="fixture_phase",
            scope_provider_id="fixture_intervals.v1",
            order=2,
            start_frame=20,
            end_frame_exclusive=30,
            source_binding={"interval_digest": "b" * 64},
        ),
    )


def _config():
    return RecordingBehaviorDistributionConfig(
        distribution_run_id="fixture-recording-distributions-v1",
        recording_id="recording-1",
        scopes=_scopes(),
        source_record={"bundle_sha256": "c" * 64},
    )


def _event_spec(*, metric_id: str = "bout.fixture") -> DistributionMetricSpec:
    return DistributionMetricSpec(
        metric_id=metric_id,
        metric_family="bout_kinematics",
        source_surface="bout_observations",
        value_column="fixture_value",
        unit="mm",
        bin_width=0.25,
        lower_bound=0.0,
        upper_bound=None,
        coverage_policy="zero_anchored_cover_valid_max",
        weighting_ids=("event",),
        group_columns=(),
        validity_policy_id="fixture_finite_nonnegative_v1",
        scope_binding_id="explicit_scope_membership_v1",
        interpretation="Fixture event value",
    )


def _source_identity(count: int):
    return {
        "source_run_path": np.asarray(["analysis/example/run"] * count, dtype=object),
        "source_manifest_sha256": np.asarray(["d" * 64] * count, dtype=object),
    }


def _source_fallback():
    return {
        "source_run_path": "analysis/example/run",
        "source_manifest_sha256": "d" * 64,
    }


def test_event_distributions_keep_whole_session_and_exact_scope_membership() -> None:
    values = np.asarray([0.1, 0.2, 0.4, np.nan], dtype=np.float64)
    result = compute_recording_behavior_distributions(
        _config(),
        (
            RecordingDistributionMetricInput(
                spec=_event_spec(),
                values=values,
                valid=np.asarray([True, True, True, True]),
                scope_projection=exact_source_membership_masks(
                    _scopes(), source_scope_id=["phase_a", None, "phase_b", "phase_a"]
                ),
                source_identity_arrays=_source_identity(values.size),
                source_identity_fallback=_source_fallback(),
                valid_duration_s_by_scope={
                    "whole_session": 120.0,
                    "phase_a": 30.0,
                    "phase_b": 40.0,
                },
            ),
        ),
    )

    support = {row["scope_id"]: row for row in result.support}
    assert support["whole_session"]["candidate_count"] == 4
    assert support["whole_session"]["valid_count"] == 3
    assert support["whole_session"]["event_rate_per_valid_min"] == 1.5
    assert support["phase_a"]["candidate_count"] == 2
    assert support["phase_a"]["valid_count"] == 1
    assert support["phase_b"]["valid_count"] == 1
    assert {
        (row["scope_id"], row["grid_index"], row["bin_count"])
        for row in result.sparse_bins
    } == {
        ("whole_session", 0, 2),
        ("whole_session", 1, 1),
        ("phase_a", 0, 1),
        ("phase_b", 1, 1),
    }
    assert len(result.record["record_sha256"]) == 64


def test_empty_event_metric_retains_zero_support_and_bound_source_identity() -> None:
    result = compute_recording_behavior_distributions(
        _config(),
        (
            RecordingDistributionMetricInput(
                spec=_event_spec(),
                values=np.asarray([], dtype=np.float64),
                valid=np.asarray([], dtype=bool),
                scope_projection=exact_source_membership_masks(
                    _scopes(), source_scope_id=[]
                ),
                source_identity_arrays=_source_identity(0),
                source_identity_fallback=_source_fallback(),
                valid_duration_s_by_scope={
                    "whole_session": 120.0,
                    "phase_a": 30.0,
                    "phase_b": 40.0,
                },
            ),
        ),
    )

    assert len(result.support) == 3
    assert all(row["support_state"] == "zero_denominator" for row in result.support)
    assert result.sparse_bins == ()
    assert result.axis_audits[0]["minimum_grid_index"] is None


def test_frame_and_time_weighting_have_distinct_denominators() -> None:
    spec = DistributionMetricSpec(
        metric_id="motion.fixture_speed",
        metric_family="motion_speed",
        source_surface="provider_motion_samples",
        value_column="fixture_speed",
        unit="mm/s",
        bin_width=1.0,
        lower_bound=0.0,
        upper_bound=None,
        coverage_policy="zero_anchored_cover_valid_max",
        weighting_ids=("frame", "time"),
        group_columns=("provider_role",),
        validity_policy_id="fixture_motion_valid_v1",
        scope_binding_id="sample_or_transition_scope_v1",
        interpretation="Fixture speed",
    )
    frames = np.asarray([10, 11, 20, 21], dtype=np.int64)
    sample = sample_scope_masks(_scopes(), acquisition_frame_id=frames)
    transition = transition_scope_masks(
        _scopes(),
        acquisition_frame_id=frames,
        acquisition_frame_delta=np.ones(4, dtype=np.int64),
    )
    result = compute_recording_behavior_distributions(
        _config(),
        (
            RecordingDistributionMetricInput(
                spec=spec,
                values=np.asarray([0.2, 1.2, 0.2, 1.2]),
                valid=np.ones(4, dtype=bool),
                scope_projection=sample,
                group_arrays={
                    "provider_role": np.asarray(["keypoint"] * 4, dtype=object)
                },
                source_identity_arrays=_source_identity(4),
                source_identity_fallback=_source_fallback(),
                time_weights_s=np.asarray([1.0, 3.0, 5.0, 7.0]),
                time_scope_projection=transition,
            ),
        ),
    )

    whole = {
        row["weighting_id"]: row
        for row in result.support
        if row["scope_id"] == "whole_session"
    }
    assert whole["frame"]["denominator_weight"] == 4.0
    assert whole["time"]["denominator_weight"] == 16.0
    phase_a_time = next(
        row
        for row in result.support
        if row["scope_id"] == "phase_a" and row["weighting_id"] == "time"
    )
    assert phase_a_time["candidate_count"] == 1
    assert phase_a_time["denominator_weight"] == 3.0


def test_fixed_signed_grid_keeps_closed_upper_endpoint() -> None:
    spec = DistributionMetricSpec(
        metric_id="bout.signed_fixture",
        metric_family="bout_heading",
        source_surface="bout_observations",
        value_column="angle",
        unit="deg",
        bin_width=10.0,
        lower_bound=-180.0,
        upper_bound=180.0,
        coverage_policy="fixed_closed_terminal",
        weighting_ids=("event",),
        group_columns=(),
        validity_policy_id="fixture_angle_v1",
        scope_binding_id="fixture_scope_v1",
        interpretation="Fixture signed angle",
    )

    assert canonical_grid_indices(np.asarray([-180.0, -170.0, 180.0]), spec).tolist() == [
        0,
        1,
        35,
    ]
    with pytest.raises(RecordingBehaviorDistributionError, match="fixed axis"):
        canonical_grid_indices(np.asarray([181.0]), spec)


def test_log_grid_uses_shared_exponent_indices() -> None:
    spec = DistributionMetricSpec(
        metric_id="bout.log_fixture",
        metric_family="bout_kinematics",
        source_surface="bout_observations",
        value_column="ratio",
        unit="dimensionless",
        bin_width=0.5,
        lower_bound=0.0,
        upper_bound=None,
        coverage_policy="log10_cover_valid_positive_range",
        weighting_ids=("event",),
        group_columns=(),
        validity_policy_id="fixture_positive_v1",
        scope_binding_id="fixture_scope_v1",
        interpretation="Fixture positive ratio",
    )

    assert canonical_grid_indices(np.asarray([0.1, 1.0, 10.0]), spec).tolist() == [
        -2,
        0,
        2,
    ]


def test_reducer_rejects_scope_roster_mismatch() -> None:
    values = np.asarray([1.0])
    only_whole = (whole_session_scope(),)
    with pytest.raises(RecordingBehaviorDistributionError, match="scope roster"):
        compute_recording_behavior_distributions(
            _config(),
            (
                RecordingDistributionMetricInput(
                    spec=_event_spec(),
                    values=values,
                    valid=np.asarray([True]),
                    scope_projection=exact_source_membership_masks(
                        only_whole, source_scope_id=[None]
                    ),
                    source_identity_arrays=_source_identity(1),
                    source_identity_fallback=_source_fallback(),
                    valid_duration_s_by_scope={"whole_session": 1.0},
                ),
            ),
        )
