"""Closed execution profiles for the shared analysis-workflow DAG.

The profile controls lifecycle authority, not scientific algorithms.  A
selector-ineligible canary must use the same producers and strict readers as
production while naming every candidate explicitly and leaving every parent
selector unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


PRODUCTION_EXECUTION_PROFILE_ID = "selector_activated_production_v1"
SELECTOR_INELIGIBLE_CANARY_EXECUTION_PROFILE_ID = (
    "selector_ineligible_canary_v1"
)


@dataclass(frozen=True)
class WorkflowExecutionProfile:
    profile_id: str
    expected_selector_eligible: bool
    selector_policy: str
    registry_policy: str
    unsupported_producer_stage_ids: frozenset[str] = frozenset()

    def supports_stage_producer(self, stage_id: str | None) -> bool:
        """Return whether this lifecycle profile may create ``stage_id``."""

        return stage_id is None or stage_id not in self.unsupported_producer_stage_ids


_PROFILES: Mapping[str, WorkflowExecutionProfile] = MappingProxyType(
    {
        PRODUCTION_EXECUTION_PROFILE_ID: WorkflowExecutionProfile(
            profile_id=PRODUCTION_EXECUTION_PROFILE_ID,
            expected_selector_eligible=True,
            selector_policy="activate_after_complete_strict_validation",
            registry_policy="deferred_to_serial_finalizer",
        ),
        SELECTOR_INELIGIBLE_CANARY_EXECUTION_PROFILE_ID: WorkflowExecutionProfile(
            profile_id=SELECTOR_INELIGIBLE_CANARY_EXECUTION_PROFILE_ID,
            expected_selector_eligible=False,
            selector_policy="exact_named_candidates_parent_selectors_unchanged",
            registry_policy="disabled_selector_ineligible_canary",
            # The maintained arena-assignment/tracking producer publishes a
            # selector-eligible run.  Until that producer has its own complete
            # candidate lifecycle, canaries must reuse one exact admitted track
            # run for both full-acquisition and clipped recording rowsets.
            unsupported_producer_stage_ids=frozenset({"tracks"}),
        ),
    }
)


def resolve_workflow_execution_profile(
    profile_id: str,
) -> WorkflowExecutionProfile:
    """Return one installed profile; unknown profiles fail closed."""

    value = str(profile_id).strip()
    try:
        return _PROFILES[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported workflow execution profile {value!r}.") from exc


def workflow_execution_profile_ids() -> tuple[str, ...]:
    return tuple(_PROFILES)


__all__ = [
    "PRODUCTION_EXECUTION_PROFILE_ID",
    "SELECTOR_INELIGIBLE_CANARY_EXECUTION_PROFILE_ID",
    "WorkflowExecutionProfile",
    "resolve_workflow_execution_profile",
    "workflow_execution_profile_ids",
]
