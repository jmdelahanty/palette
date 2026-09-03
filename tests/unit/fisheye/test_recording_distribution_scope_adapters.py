from __future__ import annotations

from types import SimpleNamespace

import pytest

import fisheye.analysis_workflows.recording_distribution_scope_adapters as subject
from fisheye.analysis_workflows.validated_recording_behavior_source import (
    ValidatedSemanticEpoch,
)


class _Source:
    recording_id = "recording-1"
    bundle_sha256 = "a" * 64

    def scientific_child(self, capability: str):
        assert capability == "semantic_epochs"
        return SimpleNamespace(
            binding={
                "run_path": "analysis/protocol_semantic_chaser_selection_runs/exact",
                "manifest_sha256": "b" * 64,
                "payload_digest": "c" * 64,
                "receipt_sha256": "d" * 64,
            }
        )

    def semantic_epoch_records(self):
        return (
            ValidatedSemanticEpoch(
                window_id=2,
                analysis_role="stimulus_b",
                source_label="Stimulus B",
                start_frame=20,
                end_frame_exclusive=30,
                source_interval_sha256="f" * 64,
                protocol_semantic_hash="protocol-hash",
                protocol_semantic_step_index=2,
                protocol_semantic_step_ref="steps/2",
                terminal_frame_excluded_pending_step_end_contract=False,
            ),
            ValidatedSemanticEpoch(
                window_id=1,
                analysis_role="stimulus_a",
                source_label="Stimulus A",
                start_frame=10,
                end_frame_exclusive=20,
                source_interval_sha256="e" * 64,
                protocol_semantic_hash="protocol-hash",
                protocol_semantic_step_index=1,
                protocol_semantic_step_ref="steps/1",
                terminal_frame_excluded_pending_step_end_contract=False,
            ),
        )


def test_protocol_scope_adapter_preserves_dynamic_roles_and_interval_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "ValidatedRecordingBehaviorSource", _Source)

    scopes = subject.protocol_semantic_distribution_scopes(_Source())

    assert [scope.scope_id for scope in scopes] == [
        "whole_session",
        "stimulus_a",
        "stimulus_b",
    ]
    assert scopes[1].source_binding["source_interval_sha256"] == "e" * 64
    assert scopes[2].start_inclusive == 20
    assert scopes[2].end_exclusive == 30


def test_named_time_scope_adapter_persists_requested_clock_binding() -> None:
    scopes = subject.named_session_time_distribution_scopes(
        (
            subject.NamedSessionTimeBracket(
                scope_id="minute_one",
                scope_label="Minute one",
                start_timestamp_ns_session=0,
                end_timestamp_ns_session_exclusive=60_000_000_000,
            ),
        ),
        timebase_binding={"timebase_id": "analysis_frame_over_fps.v1", "sha256": "a" * 64},
    )

    assert [scope.scope_id for scope in scopes] == ["whole_session", "minute_one"]
    assert scopes[1].axis_kind == "session_time_ns"
    assert scopes[1].source_binding["interpolation"] == "prohibited"
