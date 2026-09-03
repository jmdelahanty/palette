from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import fisheye.analysis_workflows.recording_distribution_bout_adapter as subject
from fisheye.analysis_workflows.recording_distribution_motion_adapter import (
    ProviderMotionDistributionContext,
)
from fisheye.group_statistics.recording_distribution_scopes import (
    frame_interval_scope,
    sample_scope_masks,
    transition_scope_masks,
    whole_session_scope,
)
from fisheye.group_statistics.validated_behavior_distribution_specs import (
    DEFAULT_DISTRIBUTION_METRICS,
)


_DIGEST = "a" * 64


def _scopes():
    return (
        whole_session_scope(),
        frame_interval_scope(
            scope_id="phase_pre",
            scope_label="Pre",
            scope_family="protocol_semantic_epoch",
            scope_provider_id=subject.PROTOCOL_SEMANTIC_SCOPE_PROVIDER_ID,
            order=1,
            start_frame=10,
            end_frame_exclusive=20,
            source_binding={"source_interval_sha256": "1" * 64},
        ),
        frame_interval_scope(
            scope_id="phase_training",
            scope_label="Training",
            scope_family="protocol_semantic_epoch",
            scope_provider_id=subject.PROTOCOL_SEMANTIC_SCOPE_PROVIDER_ID,
            order=2,
            start_frame=20,
            end_frame_exclusive=30,
            source_binding={"source_interval_sha256": "2" * 64},
        ),
    )


def _bouts() -> np.ndarray:
    dtype = np.dtype(
        [
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("track_id", "i4"),
            ("bout_id", "i4"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("duration_s", "f8"),
            ("start_time_s", "f8"),
            ("end_time_s", "f8"),
            ("path_length_mm", "f8"),
            ("net_displacement_mm", "f8"),
            ("mean_speed_mm_s", "f8"),
            ("peak_physical_speed_mm_s", "f8"),
        ]
    )
    values = np.zeros(3, dtype=dtype)
    values["candidate_id"] = 3
    values["signal_id"] = 4
    values["track_id"] = 7
    values["bout_id"] = [100, 101, 102]
    values["start_frame"] = [10, 14, 20]
    values["end_frame"] = [11, 15, 21]
    values["duration_s"] = 0.1
    values["start_time_s"] = [1.0, 1.4, 2.0]
    values["end_time_s"] = [1.1, 1.5, 2.1]
    values["path_length_mm"] = [1.0, 2.0, 3.0]
    values["net_displacement_mm"] = [1.0, 1.0, 1.5]
    values["mean_speed_mm_s"] = [10.0, 20.0, 30.0]
    values["peak_physical_speed_mm_s"] = [12.0, 22.0, 32.0]
    return values


def _intervals() -> np.ndarray:
    dtype = np.dtype(
        [
            ("valid", "?"),
            ("prev_end_frame", "i8"),
            ("next_start_frame", "i8"),
            ("prev_end_time_s", "f8"),
            ("next_start_time_s", "f8"),
            ("interval_s", "f8"),
        ]
    )
    values = np.zeros(2, dtype=dtype)
    values["valid"] = True
    values["prev_end_frame"] = [11, 15]
    values["next_start_frame"] = [14, 20]
    values["prev_end_time_s"] = [1.1, 1.5]
    values["next_start_time_s"] = [1.4, 2.0]
    values["interval_s"] = [0.3, 0.5]
    return values


class _Source:
    recording_id = "recording-1"
    analysis_zarr = "/archive"
    bundle = {
        "source_bindings": {
            "canonical_swim_bouts": {
                "source": {
                    "run_path": "analysis/swim_bout_runs/exact",
                    "lineage_hash": "3" * 64,
                    "frame_axis_sha256": "4" * 64,
                    "source_track_motion_manifest_sha256": "5" * 64,
                    "source_track_motion_verification_digest": "6" * 64,
                    "track_id": 7,
                    "default_candidate_id": 3,
                    "default_signal_id": 4,
                    "default_signal_level": "filtered",
                }
            }
        }
    }

    def canonical_swim_bout_tables(self):
        return SimpleNamespace(
            bouts=_bouts(),
            inter_bout_intervals=_intervals(),
            run_attrs={"fps": 10.0},
        )

    def scientific_child(self, capability: str):
        assert capability == "epoch_behavior"
        return SimpleNamespace(
            binding={
                "run_path": "analysis/stimulus_epoch_behavior_summary_runs/exact",
                "manifest_sha256": "7" * 64,
                "payload_digest": "8" * 64,
                "receipt_path": "/receipt.json",
                "receipt_sha256": "9" * 64,
            }
        )


def _motion_context() -> ProviderMotionDistributionContext:
    frames = np.asarray([10, 11, 14, 15, 20, 21], dtype=np.int64)
    arrays = MappingProxyType(
        {
            "source_acquisition_frame_index": frames,
            "smoothed_heading_degrees": np.asarray(
                [0, 10, 20, 30, 40, 50], dtype=np.float32
            ),
            "angular_sample_valid": np.ones(frames.size, dtype=bool),
        }
    )
    scopes = _scopes()
    return ProviderMotionDistributionContext(
        projection=SimpleNamespace(),
        arrays=arrays,
        fps=10.0,
        sample_scopes=sample_scope_masks(scopes, acquisition_frame_id=frames),
        transition_scopes=transition_scope_masks(
            scopes,
            acquisition_frame_id=frames,
            acquisition_frame_delta=np.asarray([0, 1, 3, 1, 5, 1]),
        ),
        valid_duration_s_by_scope=MappingProxyType(
            {"whole_session": 0.6, "phase_pre": 0.4, "phase_training": 0.2}
        ),
        provider_role="keypoint",
        provider_id="provider.v1",
        provider_digest=_DIGEST,
    )


def _epoch_handle(*, tamper_interval: bool = False):
    interval = ["1" * 64, "1" * 64, "2" * 64]
    if tamper_interval:
        interval[1] = "f" * 64
    arrays = {
        "bout_source_row": np.asarray([0, 1, 2], dtype=np.int64),
        "bout_id": np.asarray([100, 101, 102], dtype=np.int64),
        "bout_start_frame": np.asarray([10, 14, 20], dtype=np.int64),
        "bout_end_frame": np.asarray([11, 15, 21], dtype=np.int64),
        "bout_net_heading_change_deg": np.asarray([10.0, 10.0, 10.0]),
        "abs_bout_net_heading_change_deg": np.asarray([10.0, 10.0, 10.0]),
        "bout_heading_path_deg": np.asarray([10.0, 10.0, 10.0]),
        "analysis_role": np.asarray(
            ["phase_pre", "phase_pre", "phase_training"], dtype=object
        ),
        "source_interval_sha256": np.asarray(interval, dtype=object),
    }

    class _Handle:
        run_path = "analysis/stimulus_epoch_behavior_summary_runs/exact"
        manifest_sha256 = "7" * 64
        payload_digest = "8" * 64
        receipt_digest = "9" * 64
        verified_array_paths = tuple(
            f"per_epoch_bouts/{name}" for name in subject._EPOCH_BOUT_ARRAYS
        )

        def require_verified_arrays(self, paths):
            assert tuple(paths) == self.verified_array_paths

        def array(self, path: str):
            return arrays[path.rsplit("/", 1)[-1]]

    return _Handle()


def _specs():
    selected = {
        "bout.duration_s",
        "bout.net_heading_change_deg",
        "bout.inter_bout_interval_s",
    }
    return tuple(
        spec for spec in DEFAULT_DISTRIBUTION_METRICS if spec.metric_id in selected
    )


def test_bout_adapter_uses_exact_bout_membership_and_ibi_containment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "ValidatedRecordingBehaviorSource", _Source)
    monkeypatch.setattr(
        subject,
        "load_provider_epoch_behavior_summary_source_handle",
        lambda *_args, **_kwargs: _epoch_handle(),
    )

    _context, metrics, receipt = subject.canonical_bout_distribution_inputs(
        _Source(), _scopes(), _specs(), motion_context=_motion_context()
    )
    by_id = {metric.spec.metric_id: metric for metric in metrics}

    duration = by_id["bout.duration_s"]
    assert duration.scope_projection.masks["phase_pre"].tolist() == [
        True,
        True,
        False,
    ]
    interval = by_id["bout.inter_bout_interval_s"]
    assert interval.scope_projection.masks["phase_pre"].tolist() == [True, False]
    assert interval.scope_projection.masks["phase_training"].tolist() == [
        False,
        False,
    ]
    assert receipt["receipt_sha256"] == "9" * 64


def test_bout_adapter_rejects_epoch_membership_from_another_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "ValidatedRecordingBehaviorSource", _Source)
    monkeypatch.setattr(
        subject,
        "load_provider_epoch_behavior_summary_source_handle",
        lambda *_args, **_kwargs: _epoch_handle(tamper_interval=True),
    )

    with pytest.raises(
        subject.RecordingDistributionBoutAdapterError,
        match="different semantic interval",
    ):
        subject.canonical_bout_distribution_inputs(
            _Source(), _scopes(), _specs(), motion_context=_motion_context()
        )
