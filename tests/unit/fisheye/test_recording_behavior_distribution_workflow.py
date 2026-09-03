from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import fisheye.analysis_workflows.recording_behavior_distribution_workflow as subject
from fisheye.group_statistics.recording_behavior_distribution_specs import (
    DEFAULT_RECORDING_DISTRIBUTION_METRICS,
)
from fisheye.group_statistics.recording_distribution_scopes import (
    whole_session_scope,
)


def _spec(metric_id: str):
    return next(
        item
        for item in DEFAULT_RECORDING_DISTRIBUTION_METRICS
        if item.metric_id == metric_id
    )


class _Source:
    recording_id = "recording-1"
    analysis_zarr = Path("/archive/recording-1.zarr")
    bundle_path = Path("/evidence/recording-1.json")
    bundle_sha256 = "1" * 64
    bundle = {"source_bindings": {}}

    def __init__(self, *, distance_state: str = "missing") -> None:
        self.distance_state = distance_state
        self.capability_calls: list[str] = []

    def capability_record(self, capability: str):
        self.capability_calls.append(capability)
        return {"state": self.distance_state}


def _motion_context():
    return SimpleNamespace(
        projection=SimpleNamespace(
            run_path="analysis/track_kinematics_runs/exact",
            manifest_sha256="2" * 64,
            verification_digest="3" * 64,
            track_id=0,
            array_sha256={"speed_filtered_mm_s": "4" * 64},
        )
    )


def _install_common(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "ValidatedRecordingBehaviorSource", _Source)
    monkeypatch.setattr(subject, "sha256_file", lambda _path: "5" * 64)


def test_workflow_records_optional_distance_omission_without_null_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_common(monkeypatch)
    motion_spec = _spec("motion.filtered_speed_mm_s")
    distance_spec = _spec("chaser.relative_distance_mm")
    motion_input = SimpleNamespace(spec=motion_spec)
    monkeypatch.setattr(
        subject,
        "provider_motion_distribution_inputs",
        lambda *_args, **_kwargs: (_motion_context(), (motion_input,)),
    )
    monkeypatch.setattr(
        subject,
        "chaser_distance_distribution_inputs",
        lambda *_args, **_kwargs: pytest.fail("unavailable distance was invoked"),
    )
    captured = {}
    sentinel = object()

    def _compute(config, inputs):
        captured["config"] = config
        captured["inputs"] = inputs
        return sentinel

    monkeypatch.setattr(subject, "compute_recording_behavior_distributions", _compute)
    source = _Source(distance_state="missing")

    prepared = subject.prepare_recording_behavior_distribution(
        source,
        distribution_run_id="distribution-v1",
        scopes=(whole_session_scope(),),
        metric_specs=(motion_spec, distance_spec),
    )

    assert prepared.result is sentinel
    assert [item.spec.metric_id for item in captured["inputs"]] == [
        "motion.filtered_speed_mm_s"
    ]
    assert [row["metric_id"] for row in prepared.omitted_metrics] == [
        "chaser.relative_distance_mm"
    ]
    source_record = captured["config"].source_record
    assert source_record["published_metric_ids"] == [
        "motion.filtered_speed_mm_s"
    ]
    assert source_record["bundle_file_sha256"] == "5" * 64
    assert source.capability_calls == [
        "chaser_relative_keypoint",
        "chaser_relative_detection",
    ]


def test_workflow_can_require_optional_distance_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_common(monkeypatch)
    distance_spec = _spec("chaser.relative_distance_mm")

    with pytest.raises(
        subject.RecordingBehaviorDistributionWorkflowError,
        match="no complete provider",
    ):
        subject.prepare_recording_behavior_distribution(
            _Source(distance_state="missing"),
            distribution_run_id="distribution-v1",
            scopes=(whole_session_scope(),),
            metric_specs=(distance_spec,),
            require_all_metrics=True,
        )


@pytest.mark.parametrize(
    "roles",
    ["keypoint", (), ("keypoint", "keypoint"), ("unknown",)],
)
def test_workflow_rejects_inexact_distance_provider_roster(
    monkeypatch: pytest.MonkeyPatch,
    roles,
) -> None:
    _install_common(monkeypatch)
    distance_spec = _spec("chaser.relative_distance_mm")
    source = _Source(distance_state="complete")

    with pytest.raises(
        subject.RecordingBehaviorDistributionWorkflowError,
        match="unique keypoint/detection",
    ):
        subject.prepare_recording_behavior_distribution(
            source,
            distribution_run_id="distribution-v1",
            scopes=(whole_session_scope(),),
            metric_specs=(distance_spec,),
            chaser_provider_roles=roles,
        )
    assert source.capability_calls == []


def test_workflow_admits_only_complete_requested_distance_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_common(monkeypatch)
    distance_spec = _spec("chaser.relative_distance_mm")
    distance_input = SimpleNamespace(spec=distance_spec)
    observed = {}

    def _distance(_source, _scopes, _specs, **kwargs):
        observed.update(kwargs)
        return (distance_input,), ({"run_path": "analysis/chaser/exact"},)

    monkeypatch.setattr(subject, "chaser_distance_distribution_inputs", _distance)
    sentinel = object()
    monkeypatch.setattr(
        subject,
        "compute_recording_behavior_distributions",
        lambda _config, _inputs: sentinel,
    )

    prepared = subject.prepare_recording_behavior_distribution(
        _Source(distance_state="complete"),
        distribution_run_id="distribution-v1",
        scopes=(whole_session_scope(),),
        metric_specs=(distance_spec,),
        chaser_provider_roles=("detection",),
    )

    assert prepared.result is sentinel
    assert prepared.omitted_metrics == ()
    assert observed["provider_roles"] == ("detection",)
    assert prepared.adapter_evidence["chaser_relative_distance"][
        "provider_roles"
    ] == ["detection"]
