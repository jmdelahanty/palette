"""Registry integrity checks for one recording row's scientific context.

Operator/legacy rows (``context_source`` NULL) keep the historical rules: the
subtype comes from the vocabulary and, for behavior, equals ``behavior_mode``.
Rows with a producer-declared context (``citrus.parent_recording_context``)
carry an optional free-label subtype independent of ``behavior_mode``; their
declared context version, intent and data origin are checked instead.
"""

from __future__ import annotations

from typing import Any, Mapping

from fisheye.shared.recording_manifest_context import (
    DEFAULT_ALLOWED_BEHAVIOR_MODES,
    PRODUCER_CONTEXT_SOURCE,
    producer_context_row_issues,
)


def _text(row: Mapping[str, Any], field: str) -> str:
    return str(row[field]).strip() if row[field] is not None else ""


def recording_context_row_issues(
    row: Mapping[str, Any],
    *,
    allowed_recording_types: set[str],
    allowed_subtypes_by_type: Mapping[str, set[str]],
) -> list[tuple[str, str]]:
    """Return ``(code, detail)`` pairs for one ``recordings`` row."""

    issues: list[tuple[str, str]] = []
    recording_id = str(row["recording_id"])
    recording_type = _text(row, "recording_type")
    recording_subtype = _text(row, "recording_subtype")
    behavior_mode = _text(row, "behavior_mode")
    producer_context = row["context_source"] == PRODUCER_CONTEXT_SOURCE
    if not recording_type:
        issues.append((
            "recording_missing_type",
            f"recording_id={recording_id} has NULL/empty recording_type",
        ))
    elif recording_type not in allowed_recording_types:
        issues.append((
            "recording_invalid_type",
            f"recording_id={recording_id} recording_type={recording_type} "
            f"not in allowed={','.join(sorted(allowed_recording_types))}",
        ))
    if producer_context:
        issues.extend(
            (f"recording_{code}", f"recording_id={recording_id} {detail}")
            for code, detail in producer_context_row_issues(row)
        )
    allowed_subtypes = allowed_subtypes_by_type.get(recording_type)
    if allowed_subtypes is not None and not producer_context:
        if not recording_subtype:
            issues.append((
                "recording_missing_subtype",
                f"recording_id={recording_id} recording_type={recording_type} "
                "has NULL/empty recording_subtype",
            ))
        elif recording_subtype not in allowed_subtypes:
            issues.append((
                "recording_invalid_subtype",
                f"recording_id={recording_id} recording_type={recording_type} "
                f"recording_subtype={recording_subtype} "
                f"not in allowed={','.join(sorted(allowed_subtypes))}",
            ))
    if not behavior_mode:
        issues.append((
            "recording_missing_behavior_mode",
            f"recording_id={recording_id} has NULL/empty behavior_mode",
        ))
    elif behavior_mode not in DEFAULT_ALLOWED_BEHAVIOR_MODES:
        issues.append((
            "recording_invalid_behavior_mode",
            f"recording_id={recording_id} behavior_mode={behavior_mode} "
            f"not in allowed={','.join(sorted(DEFAULT_ALLOWED_BEHAVIOR_MODES))}",
        ))
    if (
        not producer_context
        and recording_type == "behavior"
        and recording_subtype
        and behavior_mode
        and recording_subtype != behavior_mode
    ):
        issues.append((
            "recording_behavior_mode_mismatch",
            f"recording_id={recording_id} recording_type=behavior "
            f"requires recording_subtype==behavior_mode, got "
            f"{recording_subtype}!={behavior_mode}",
        ))
    return issues
