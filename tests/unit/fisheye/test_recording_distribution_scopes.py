from __future__ import annotations

import numpy as np
import pytest

from fisheye.group_statistics.recording_distribution_scopes import (
    RecordingDistributionScopeError,
    exact_source_membership_masks,
    frame_interval_scope,
    fully_contained_frame_event_masks,
    sample_scope_masks,
    scope_registry_record,
    session_time_bracket_scope,
    transition_scope_masks,
    validate_scope_registry,
    whole_session_scope,
)


def _chaser_scopes():
    return (
        whole_session_scope(),
        frame_interval_scope(
            scope_id="chaser_pre",
            scope_label="Pre",
            scope_family="protocol_semantic_epoch",
            scope_provider_id="protocol_semantic_intervals.v1",
            order=1,
            start_frame=10,
            end_frame_exclusive=20,
            source_binding={"source_interval_sha256": "a" * 64},
        ),
        frame_interval_scope(
            scope_id="chaser_training",
            scope_label="Training",
            scope_family="protocol_semantic_epoch",
            scope_provider_id="protocol_semantic_intervals.v1",
            order=2,
            start_frame=20,
            end_frame_exclusive=30,
            source_binding={"source_interval_sha256": "b" * 64},
        ),
    )


def test_scope_registry_is_ordered_and_self_digested() -> None:
    registry = scope_registry_record(tuple(reversed(_chaser_scopes())))

    assert registry["scope_order"] == [
        "whole_session",
        "chaser_pre",
        "chaser_training",
    ]
    assert len(registry["scope_registry_sha256"]) == 64
    assert all(len(row["scope_sha256"]) == 64 for row in registry["scopes"])


def test_mutually_exclusive_scope_provider_rejects_overlap() -> None:
    scopes = list(_chaser_scopes())
    scopes[2] = frame_interval_scope(
        scope_id="chaser_training",
        scope_label="Training",
        scope_family="protocol_semantic_epoch",
        scope_provider_id="protocol_semantic_intervals.v1",
        order=2,
        start_frame=19,
        end_frame_exclusive=30,
        source_binding={"source_interval_sha256": "b" * 64},
    )

    with pytest.raises(RecordingDistributionScopeError, match="overlap"):
        validate_scope_registry(scopes)


def test_named_session_time_brackets_may_overlap() -> None:
    binding = {"timebase_id": "timestamp_ns_session.v1", "sha256": "c" * 64}
    scopes = (
        whole_session_scope(),
        session_time_bracket_scope(
            scope_id="minute_1",
            scope_label="Minute 1",
            order=1,
            start_timestamp_ns_session=0,
            end_timestamp_ns_session_exclusive=60_000_000_000,
            timebase_binding=binding,
        ),
        session_time_bracket_scope(
            scope_id="overlap",
            scope_label="Overlapping bracket",
            order=2,
            start_timestamp_ns_session=30_000_000_000,
            end_timestamp_ns_session_exclusive=90_000_000_000,
            timebase_binding=binding,
        ),
    )

    assert len(validate_scope_registry(scopes)) == 3


def test_sample_scope_masks_use_half_open_frames() -> None:
    frames = np.asarray([9, 10, 19, 20, 29, 30], dtype=np.int64)
    projection = sample_scope_masks(
        _chaser_scopes(), acquisition_frame_id=frames
    )

    assert projection.masks["whole_session"].tolist() == [True] * 6
    assert projection.masks["chaser_pre"].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    assert projection.masks["chaser_training"].tolist() == [
        False,
        False,
        False,
        True,
        True,
        False,
    ]


def test_session_time_scope_keeps_invalid_timestamps_as_uncovered() -> None:
    scopes = (
        whole_session_scope(),
        session_time_bracket_scope(
            scope_id="selected_seconds",
            scope_label="Selected seconds",
            order=1,
            start_timestamp_ns_session=10,
            end_timestamp_ns_session_exclusive=20,
            timebase_binding={"timebase_id": "fixture"},
        ),
    )
    projection = sample_scope_masks(
        scopes,
        timestamp_ns_session=np.asarray([9, 10, 19, 20], dtype=np.int64),
        timestamp_valid=np.asarray([True, True, False, True]),
    )

    assert projection.masks["selected_seconds"].tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert projection.uncovered["selected_seconds"].tolist() == [
        False,
        False,
        True,
        False,
    ]


def test_transition_scope_requires_both_frame_endpoints() -> None:
    projection = transition_scope_masks(
        _chaser_scopes(),
        acquisition_frame_id=np.asarray([10, 11, 20, 21], dtype=np.int64),
        acquisition_frame_delta=np.asarray([1, 1, 1, 1], dtype=np.int64),
    )

    assert projection.masks["chaser_pre"].tolist() == [False, True, False, False]
    assert projection.masks["chaser_training"].tolist() == [
        False,
        False,
        False,
        True,
    ]


def test_exact_source_membership_keeps_unassigned_events_whole_only() -> None:
    projection = exact_source_membership_masks(
        _chaser_scopes(),
        source_scope_id=["chaser_pre", None, "chaser_training"],
    )

    assert projection.masks["whole_session"].tolist() == [True, True, True]
    assert projection.masks["chaser_pre"].tolist() == [True, False, False]
    assert projection.masks["chaser_training"].tolist() == [False, False, True]

    with pytest.raises(RecordingDistributionScopeError, match="unknown scopes"):
        exact_source_membership_masks(
            _chaser_scopes(), source_scope_id=["made_up_epoch"]
        )


def test_event_scope_requires_full_containment() -> None:
    projection = fully_contained_frame_event_masks(
        _chaser_scopes(),
        start_acquisition_frame_id=np.asarray([9, 10, 19, 20], dtype=np.int64),
        end_acquisition_frame_id=np.asarray([10, 19, 20, 29], dtype=np.int64),
    )

    assert projection.masks["whole_session"].tolist() == [True] * 4
    assert projection.masks["chaser_pre"].tolist() == [False, True, False, False]
    assert projection.masks["chaser_training"].tolist() == [
        False,
        False,
        False,
        True,
    ]
