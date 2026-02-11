"""Helpers for source lineage attribute compatibility."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR = "source_keypoints_run"
LEGACY_SOURCE_KEYPOINT_RUN_ATTR = "source_keypoint_run"


def resolve_source_keypoints_run(attrs: Optional[Mapping[str, Any]]) -> Any:
    """Return canonical keypoint-run lineage value with legacy fallback."""
    if attrs is None:
        return None
    value = attrs.get(CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR)
    if value is None:
        value = attrs.get(LEGACY_SOURCE_KEYPOINT_RUN_ATTR)
    return value


def build_source_keypoints_attrs(
    source_keypoints_run: Any,
    *,
    include_legacy_alias: bool = True,
) -> Dict[str, Any]:
    """Build attrs payload with canonical key and optional legacy alias."""
    payload: Dict[str, Any] = {
        CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR: source_keypoints_run,
    }
    if include_legacy_alias:
        payload[LEGACY_SOURCE_KEYPOINT_RUN_ATTR] = source_keypoints_run
    return payload

