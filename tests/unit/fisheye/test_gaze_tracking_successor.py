from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fisheye.analysis_workflows.gaze_tracking_successor import (
    GazeTrackingInput,
    GazeTrackingSuccessorError,
    prepare_gaze_tracking_successor,
)
from fisheye.analysis_workflows.generalized_bout_response_successor import (
    ROLE_CODES,
)


def _source() -> GazeTrackingInput:
    n_frames = 6
    bearing = np.asarray([-20.0, -10.0, 0.0, 10.0, 20.0, 30.0], dtype=np.float32)
    gaze = np.stack([bearing, bearing + 5.0], axis=1).astype(np.float32)
    return GazeTrackingInput(
        recording_id="recording-1",
        source_relative_frame_run_path="analysis/chaser_relative_frame_runs/r1",
        source_relative_frame_manifest_sha256="a" * 64,
        source_eye_run_path="analysis/eye_angle_runs/e1",
        source_eye_manifest_sha256="b" * 64,
        source_eye_convention_receipt_sha256="c" * 64,
        source_eye_channel_policy=(
            "smoothed:left_gaze_signed_deg_smoothed,"
            "right_gaze_signed_deg_smoothed:vergence_eye_angle_deg_smoothed"
        ),
        source_semantic_selection_manifest_sha256="d" * 64,
        n_frames=n_frames,
        n_chasers=1,
        acquisition_frame_id_by_frame=np.arange(100, 106, dtype=np.int64),
        timestamp_ns_by_frame=np.arange(n_frames, dtype=np.int64) * 100_000_000,
        timestamp_valid_by_frame=np.ones(n_frames, dtype=bool),
        semantic_role_code_by_frame=np.full(
            n_frames, ROLE_CODES["chaser_training"], dtype=np.uint8
        ),
        chaser_identity_code=np.ones(n_frames, dtype=np.uint16),
        distance_mm=np.full(n_frames, 10.0, dtype=np.float32),
        distance_valid=np.ones(n_frames, dtype=bool),
        chaser_bearing_deg=bearing,
        chaser_bearing_valid=np.ones(n_frames, dtype=bool),
        gaze_signed_deg=gaze,
        gaze_valid=np.ones((n_frames, 2), dtype=bool),
        vergence_deg=np.full(n_frames, 12.0, dtype=np.float32),
        vergence_valid=np.ones(n_frames, dtype=bool),
        lock_threshold_deg=10.0,
        minimum_lock_duration_s=0.1,
    )


def test_body_frame_gaze_rows_summaries_and_lock_events() -> None:
    result = prepare_gaze_tracking_successor(_source())

    assert result.n_gaze_rows == 12
    assert result.n_summary_rows == 6
    assert result.n_lock_events == 2
    assert set(result.array("eye_code").tolist()) == {1, 2}
    assert np.all(result.array("gaze_error_deg")[result.array("eye_code") == 1] == 0)
    assert np.all(result.array("gaze_error_deg")[result.array("eye_code") == 2] == 5)
    training = result.array("summary_role_code") == ROLE_CODES["chaser_training"]
    training_rows = np.flatnonzero(training)
    assert training_rows.tolist() == [2, 3]
    np.testing.assert_allclose(
        result.array("summary_tracking_gain")[training_rows], [1.0, 1.0]
    )
    np.testing.assert_allclose(
        result.array("summary_tracking_correlation")[training_rows], [1.0, 1.0]
    )
    assert result.array("summary_lock_fraction")[training_rows].tolist() == [1.0, 1.0]
    assert result.array("lock_event_sample_count").tolist() == [4, 5]
    assert result.manifest["policy"]["world_frame_gaze"] == "prohibited"
    assert result.manifest["policy"]["orientation_fallback"] == "prohibited"
    assert result.manifest["selector_eligible"] is False


def test_invalid_bearing_row_is_retained_but_excluded() -> None:
    source = _source()
    valid = source.chaser_bearing_valid.copy()
    valid[2] = False
    result = prepare_gaze_tracking_successor(
        replace(source, chaser_bearing_valid=valid)
    )

    frame_two = result.array("acquisition_frame_id") == 102
    assert not np.any(result.array("valid")[frame_two])
    assert np.count_nonzero(result.array("valid")) < 12


def test_valid_nonfinite_eye_orientation_is_rejected() -> None:
    source = _source()
    gaze = source.gaze_signed_deg.copy()
    gaze[0, 0] = np.nan

    with pytest.raises(GazeTrackingSuccessorError, match="valid gaze value"):
        prepare_gaze_tracking_successor(replace(source, gaze_signed_deg=gaze))


def test_eye_convention_receipt_is_mandatory() -> None:
    with pytest.raises(GazeTrackingSuccessorError, match="convention"):
        prepare_gaze_tracking_successor(
            replace(_source(), source_eye_convention_receipt_sha256="")
        )
