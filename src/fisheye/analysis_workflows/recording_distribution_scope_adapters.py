"""Validated source adapters for recording-distribution scope registries."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from fisheye.group_statistics.recording_distribution_scopes import (
    RecordingDistributionScope,
    frame_interval_scope,
    session_time_bracket_scope,
    validate_scope_registry,
    whole_session_scope,
)

from .validated_recording_behavior_source import ValidatedRecordingBehaviorSource


PROTOCOL_SEMANTIC_SCOPE_PROVIDER_ID = "protocol_semantic_intervals.v1"
NAMED_TIME_SCOPE_PROVIDER_ID = "named_session_time_brackets.v1"


class RecordingDistributionScopeAdapterError(ValueError):
    """A validated source cannot supply an exact requested scope registry."""


@dataclass(frozen=True, slots=True)
class NamedSessionTimeBracket:
    """One caller-named half-open interval on an already bound session clock."""

    scope_id: str
    scope_label: str
    start_timestamp_ns_session: int
    end_timestamp_ns_session_exclusive: int


def protocol_semantic_distribution_scopes(
    source: ValidatedRecordingBehaviorSource,
) -> tuple[RecordingDistributionScope, ...]:
    """Adapt exact bundle semantic intervals without inferring role names."""

    if type(source) is not ValidatedRecordingBehaviorSource:
        raise RecordingDistributionScopeAdapterError(
            "Protocol-semantic scopes require one validated recording source."
        )
    child = source.scientific_child("semantic_epochs")
    epochs = tuple(
        sorted(source.semantic_epoch_records(), key=lambda row: row.start_frame)
    )
    if not epochs:
        raise RecordingDistributionScopeAdapterError(
            "Validated recording source has no semantic intervals."
        )
    scopes: list[RecordingDistributionScope] = [whole_session_scope()]
    for order, epoch in enumerate(epochs, start=1):
        scopes.append(
            frame_interval_scope(
                scope_id=epoch.analysis_role,
                scope_label=epoch.source_label,
                scope_family="protocol_semantic_epoch",
                scope_provider_id=PROTOCOL_SEMANTIC_SCOPE_PROVIDER_ID,
                order=order,
                start_frame=epoch.start_frame,
                end_frame_exclusive=epoch.end_frame_exclusive,
                source_binding={
                    "recording_id": source.recording_id,
                    "bundle_sha256": source.bundle_sha256,
                    "semantic_child_run_path": child.binding["run_path"],
                    "semantic_child_manifest_sha256": child.binding[
                        "manifest_sha256"
                    ],
                    "semantic_child_payload_digest": child.binding[
                        "payload_digest"
                    ],
                    "semantic_child_receipt_sha256": child.binding[
                        "receipt_sha256"
                    ],
                    "source_window_id": epoch.window_id,
                    "source_interval_sha256": epoch.source_interval_sha256,
                    "protocol_semantic_hash": epoch.protocol_semantic_hash,
                    "protocol_semantic_step_index": (
                        epoch.protocol_semantic_step_index
                    ),
                    "protocol_semantic_step_ref": epoch.protocol_semantic_step_ref,
                    "terminal_frame_excluded_pending_step_end_contract": (
                        epoch.terminal_frame_excluded_pending_step_end_contract
                    ),
                },
            )
        )
    return validate_scope_registry(scopes)


def named_session_time_distribution_scopes(
    brackets: Sequence[NamedSessionTimeBracket],
    *,
    timebase_binding: Mapping[str, Any],
) -> tuple[RecordingDistributionScope, ...]:
    """Build reproducible custom brackets without assigning protocol meaning."""

    requested = tuple(brackets)
    if not requested:
        raise RecordingDistributionScopeAdapterError(
            "At least one named session-time bracket is required."
        )
    binding = MappingProxyType(dict(timebase_binding))
    scopes: list[RecordingDistributionScope] = [whole_session_scope()]
    for order, bracket in enumerate(requested, start=1):
        if type(bracket) is not NamedSessionTimeBracket:
            raise RecordingDistributionScopeAdapterError(
                "Named time scopes require NamedSessionTimeBracket records."
            )
        scopes.append(
            session_time_bracket_scope(
                scope_id=bracket.scope_id,
                scope_label=bracket.scope_label,
                order=order,
                start_timestamp_ns_session=bracket.start_timestamp_ns_session,
                end_timestamp_ns_session_exclusive=(
                    bracket.end_timestamp_ns_session_exclusive
                ),
                timebase_binding=binding,
            )
        )
    return validate_scope_registry(scopes)


__all__ = [
    "NAMED_TIME_SCOPE_PROVIDER_ID",
    "NamedSessionTimeBracket",
    "PROTOCOL_SEMANTIC_SCOPE_PROVIDER_ID",
    "RecordingDistributionScopeAdapterError",
    "named_session_time_distribution_scopes",
    "protocol_semantic_distribution_scopes",
]
