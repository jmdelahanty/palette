from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

import fisheye.analysis_workflows.recording_distribution_motion_adapter as subject
from fisheye.analysis_workflows.validated_recording_behavior_source import (
    ProviderMotionTrackProjection,
)
from fisheye.group_statistics.recording_distribution_scopes import (
    frame_interval_scope,
    session_time_bracket_scope,
    whole_session_scope,
)
from fisheye.group_statistics.validated_behavior_distribution_specs import (
    DEFAULT_DISTRIBUTION_METRICS,
)
from fisheye.analysis_workflows.recording_distribution_timebase_adapter import (
    RecordingSessionTimebase,
)


class _Source:
    bundle = {
        "source_bindings": {
            "fish_position_keypoint": {
                "authority": {
                    "provider_id": "anatomical_keypoint_mean.v1",
                    "provider_digest": "a" * 64,
                }
            }
        }
    }

    def provider_motion_track_projection(self, requested):
        frame = np.asarray([10, 11, 20, 21], dtype=np.int64)
        arrays = {
            "track_sample_key": np.column_stack(
                [np.full(frame.size, 7, dtype=np.int64), frame]
            ),
            "source_acquisition_frame_index": frame,
            "time_seconds": np.asarray([1.0, 1.1, 2.0, 2.1], dtype=np.float32),
            "linear_sample_valid": np.ones(frame.size, dtype=bool),
            "angular_sample_valid": np.ones(frame.size, dtype=bool),
            "smoothed_heading_degrees": np.asarray([0, 5, 10, 20], dtype=np.float32),
            "transition_valid": np.asarray([False, True, False, True]),
            "delta_frames": np.asarray([0, 1, 9, 1], dtype=np.int64),
            "delta_seconds": np.asarray([0.0, 0.1, 0.9, 0.1], dtype=np.float32),
            "speed_filtered_mm": np.asarray([0, 1, 2, 3], dtype=np.float32),
        }
        assert set(requested).issubset(arrays)
        selected = MappingProxyType({name: arrays[name] for name in requested})
        return ProviderMotionTrackProjection(
            analysis_zarr="/archive",
            bundle_path="/bundle.json",
            bundle_sha256="b" * 64,
            run_path="analysis/track_kinematics_runs/provider/exact",
            manifest_sha256="c" * 64,
            verification_digest="d" * 64,
            track_id=7,
            track_row_start=0,
            track_row_stop=frame.size,
            arrays=selected,
            array_sha256=MappingProxyType({name: "e" * 64 for name in requested}),
            source_paths=MappingProxyType({name: name for name in requested}),
        )


def _scopes():
    return (
        whole_session_scope(),
        frame_interval_scope(
            scope_id="phase_a",
            scope_label="Phase A",
            scope_family="fixture",
            scope_provider_id="fixture.v1",
            order=1,
            start_frame=10,
            end_frame_exclusive=20,
            source_binding={"sha256": "f" * 64},
        ),
    )


def test_motion_adapter_keeps_frame_and_transition_scope_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "ValidatedRecordingBehaviorSource", _Source)
    spec = next(
        item
        for item in DEFAULT_DISTRIBUTION_METRICS
        if item.metric_id == "motion.filtered_speed_mm_s"
    )

    context, (metric,) = subject.provider_motion_distribution_inputs(
        _Source(), _scopes(), (spec,)
    )

    assert context.fps == pytest.approx(10.0)
    assert metric.scope_projection.masks["phase_a"].tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert metric.time_scope_projection is not None
    assert metric.time_scope_projection.masks["phase_a"].tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert metric.valid.tolist() == [False, True, False, True]
    assert metric.group_arrays["provider_role"].tolist() == ["keypoint"] * 4


def test_motion_adapter_uses_exact_bound_session_timestamps_for_time_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "ValidatedRecordingBehaviorSource", _Source)
    timebase = RecordingSessionTimebase(
        acquisition_frame_id=np.asarray([10, 11, 20, 21], dtype=np.int64),
        timestamp_ns_session=np.asarray([100, 110, 200, 210], dtype=np.int64),
        timestamp_valid=np.ones(4, dtype=bool),
        source_binding={"clock": "exact"},
    )
    scopes = (
        whole_session_scope(),
        session_time_bracket_scope(
            scope_id="selected",
            scope_label="Selected",
            order=1,
            start_timestamp_ns_session=105,
            end_timestamp_ns_session_exclusive=205,
            timebase_binding=timebase.binding,
        ),
    )
    spec = next(
        item
        for item in DEFAULT_DISTRIBUTION_METRICS
        if item.metric_id == "motion.filtered_speed_mm_s"
    )

    _context, (metric,) = subject.provider_motion_distribution_inputs(
        _Source(), scopes, (spec,), session_timebase=timebase
    )

    assert metric.scope_projection.masks["selected"].tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert metric.time_scope_projection is not None
    assert metric.time_scope_projection.masks["selected"].tolist() == [
        False,
        False,
        True,
        False,
    ]
