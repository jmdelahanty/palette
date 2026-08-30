"""Metadata-only discovery for exact protocol-semantic epoch behavior."""

from __future__ import annotations

from typing import Any, Mapping

from fisheye.analysis_workflows.materializers.provider_epoch_behavior_summary import (
    PARENT_PATH,
)
from fisheye.analysis_workflows.provider_epoch_behavior_summary_source_handle import (
    ProviderEpochBehaviorSummarySourceError,
    validate_provider_epoch_behavior_summary_metadata,
)


_FORBIDDEN_SELECTORS = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "current",
        "selected",
        "authoritative",
        "authoritative_run",
        "default",
    }
)


def _group_names(group: Any) -> tuple[str, ...]:
    try:
        return tuple(str(value) for value in group.group_keys())
    except Exception:
        return tuple(
            str(key)
            for key, value in getattr(group, "items", lambda: ())()
            if hasattr(value, "attrs")
        )


def compatible_epoch_behavior_binding(
    root: Any,
    *,
    recording_id: str,
    spatial_sources: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Resolve exactly one semantic-v2 child by immutable source identity."""

    semantic = spatial_sources.get("protocol_semantic_selection")
    if not isinstance(semantic, Mapping):
        return None
    try:
        parent = root[PARENT_PATH]
    except Exception:
        return None
    if _FORBIDDEN_SELECTORS.intersection(getattr(parent, "attrs", {})):
        return None
    matches: list[Mapping[str, Any]] = []
    for run_name in _group_names(parent):
        if run_name in {".", ".."} or run_name.casefold() in _FORBIDDEN_SELECTORS:
            continue
        run_path = f"{PARENT_PATH}/{run_name}"
        try:
            binding = validate_provider_epoch_behavior_summary_metadata(
                dict(getattr(parent[run_name], "attrs", {})),
                run_path=run_path,
                run_name=run_name,
                expected_recording_id=recording_id,
                expected_semantic_selection=semantic,
            )
        except (ProviderEpochBehaviorSummarySourceError, KeyError, TypeError):
            continue
        matches.append(binding)
    return matches[0] if len(matches) == 1 else None


__all__ = ["compatible_epoch_behavior_binding"]
