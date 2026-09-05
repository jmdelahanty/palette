from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import fisheye.analysis_workflows.recording_distribution_distance_adapter as subject
from fisheye.group_statistics.recording_distribution_scopes import (
    frame_interval_scope,
    whole_session_scope,
)
from fisheye.group_statistics.validated_behavior_distribution_specs import (
    DEFAULT_DISTRIBUTION_METRICS,
)


def _scopes():
    return (
        whole_session_scope(),
        frame_interval_scope(
            scope_id="training",
            scope_label="Training",
            scope_family="fixture",
            scope_provider_id="fixture.v1",
            order=1,
            start_frame=10,
            end_frame_exclusive=12,
            source_binding={"sha256": "1" * 64},
        ),
    )


def _binding(role: str):
    return {
        "run_path": f"analysis/chaser_relative_frame_runs/{role}",
        "manifest_sha256": "2" * 64,
        "payload_digest": "3" * 64,
        "receipt_path": f"/{role}.receipt.json",
        "receipt_sha256": "4" * 64,
    }


class _Source:
    recording_id = "recording-1"
    analysis_zarr = "/archive"

    def capability_record(self, capability: str):
        return {
            "state": "complete" if capability == "chaser_relative_keypoint" else "absent"
        }

    def scientific_child(self, capability: str):
        assert capability == "chaser_relative_keypoint"
        return SimpleNamespace(binding=_binding("keypoint"))


def _handle():
    count = 4
    arrays = {
        "acquisition_frame_delta": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "acquisition_frame_id": np.asarray([10, 10, 11, 11], dtype=np.int64),
        "chaser_behavior_role_code": np.asarray([1, 2, 1, 2], dtype=np.uint8),
        "chaser_behavior_role_valid": np.ones(count, dtype=bool),
        "chaser_identity_code": np.asarray([10, 11, 10, 11], dtype=np.uint16),
        "chaser_occurrence_member": np.ones(count, dtype=bool),
        "relative_distance_physical": np.asarray([1.0, 2.0, 1.5, 2.5]),
        "relative_physical_valid": np.ones(count, dtype=bool),
        "relative_transition_valid": np.asarray([False, False, True, True]),
        "row_valid": np.ones(count, dtype=bool),
        "selection_member": np.ones(count, dtype=bool),
        "timestamp_delta_ns": np.asarray([0, 0, 100_000_000, 100_000_000]),
        "timestamp_ns": np.asarray(
            [1_000_000_000, 1_000_000_000, 1_100_000_000, 1_100_000_000]
        ),
        "timestamp_valid": np.ones(count, dtype=bool),
    }
    binding = _binding("keypoint")
    return SimpleNamespace(
        run_path=binding["run_path"],
        manifest_sha256=binding["manifest_sha256"],
        payload_digest=binding["payload_digest"],
        receipt_digest=binding["receipt_sha256"],
        n_rows=count,
        base_arrays=MappingProxyType(arrays),
        manifest={
            "identity_registries": {
                "behavior_role": {"1": "aggressive", "2": "inert"},
                "chaser": {"10": "blue", "11": "red"},
            }
        },
        source_authorities={
            "fish_position": {
                "provider_id": "anatomical_keypoint_mean.v1",
                "provider_digest": "5" * 64,
            }
        },
    )


def test_optional_distance_adapter_preserves_role_identity_and_time_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "ValidatedRecordingBehaviorSource", _Source)
    monkeypatch.setattr(
        subject,
        "load_chaser_relative_frame_targeted_source_handle",
        lambda *_args, **_kwargs: _handle(),
    )
    spec = next(
        item
        for item in DEFAULT_DISTRIBUTION_METRICS
        if item.metric_id == "chaser.relative_distance_mm"
    )

    (metric,), bindings = subject.chaser_distance_distribution_inputs(
        _Source(), _scopes(), (spec,)
    )

    assert metric.group_arrays["provider_role"].tolist() == ["keypoint"] * 4
    assert metric.group_arrays["behavior_role"].tolist() == [
        "aggressive",
        "inert",
        "aggressive",
        "inert",
    ]
    assert metric.time_weights_s is not None
    assert np.isnan(metric.time_weights_s[:2]).all()
    assert metric.time_weights_s[2:].tolist() == pytest.approx([0.1, 0.1])
    assert metric.source_identity_arrays["chaser_identity"].tolist() == [
        "blue",
        "red",
        "blue",
        "red",
    ]
    assert bindings[0]["receipt_sha256"] == "4" * 64
