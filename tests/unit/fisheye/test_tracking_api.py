from __future__ import annotations

import numpy as np
import pytest

from fisheye.tracking import (
    TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA,
    available_tracking_methods,
    build_tracking,
)
from fisheye.tracking.contracts import TrackingObservations


def test_method_neutral_api_runs_current_tracker() -> None:
    observations = TrackingObservations.from_arrays(
        arena_ids=np.array([9, 5, 9], dtype=np.int32),
        frame_indices=np.array([0, 0, 1], dtype=np.int64),
        instance_key=np.array([100, 200, 300], dtype=np.uint64),
        source_refined_row_ids=np.array([10, 20, 30], dtype=np.int64),
    )

    result = build_tracking(
        observations,
        method=TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA,
        parameters={"conflict_policy": "fail"},
    )

    assert available_tracking_methods() == (TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA,)
    assert result.method == TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA
    assert result.track_ids.tolist() == [1, 0, 1]


def test_method_neutral_api_rejects_unknown_methods_and_parameters() -> None:
    observations = TrackingObservations.from_arrays(
        arena_ids=np.array([0], dtype=np.int32),
        frame_indices=np.array([0], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="Unknown tracking method"):
        build_tracking(observations, method="not_registered")
    with pytest.raises(ValueError, match="Unsupported single_subject_per_arena parameters"):
        build_tracking(
            observations,
            method=TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA,
            parameters={"mystery": True},
        )


def test_tracking_observations_reject_duplicate_instance_keys() -> None:
    with pytest.raises(ValueError, match="instance_key values must be unique"):
        TrackingObservations.from_arrays(
            arena_ids=np.array([0, 1], dtype=np.int32),
            frame_indices=np.array([0, 0], dtype=np.int64),
            instance_key=np.array([4, 4], dtype=np.uint64),
        )
