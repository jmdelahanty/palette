"""Recording-context validation shared by diagnostics, intake, and replay.

This owns the existing manifest utility's vocabulary and required-field rules;
it does not infer context, validate source identity, or classify output products.
"""

from __future__ import annotations

from typing import Any, Mapping


DEFAULT_ALLOWED_TYPES = {"behavior", "microscopy", "histology"}
DEFAULT_ALLOWED_SUBTYPES = {
    "behavior": {"free", "embedded"},
    "microscopy": {"lightsheet", "confocal", "2p"},
    "histology": {"section", "wholemount"},
}
DEFAULT_ALLOWED_BEHAVIOR_MODES = {"free", "embedded", "none"}
REQUIRED_FIELDS = (
    "recording_type", "recording_subtype", "behavior_mode", "artifact_schema_id",
)


def recording_manifest_context_issues(
    payload: Mapping[str, Any],
    *,
    allowed_types: set[str] | None = None,
    allowed_subtypes: Mapping[str, set[str]] | None = None,
) -> list[tuple[str, str]]:
    """Return the existing diagnostic codes without repairing the payload."""

    types = DEFAULT_ALLOWED_TYPES if allowed_types is None else allowed_types
    subtypes = DEFAULT_ALLOWED_SUBTYPES if allowed_subtypes is None else allowed_subtypes
    issues: list[tuple[str, str]] = []
    values: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        raw = payload.get(field)
        value = raw.strip() if isinstance(raw, str) else ""
        values[field] = value
        if not value:
            issues.append(("missing_required_field", field))

    recording_type = values["recording_type"]
    recording_subtype = values["recording_subtype"]
    behavior_mode = values["behavior_mode"]
    if recording_type and recording_type not in types:
        issues.append((
            "invalid_recording_type",
            f"{recording_type} (allowed={','.join(sorted(types))})",
        ))
    if recording_type and recording_subtype:
        allowed_for_type = subtypes.get(recording_type)
        if allowed_for_type and recording_subtype not in allowed_for_type:
            issues.append((
                "invalid_recording_subtype",
                f"type={recording_type} subtype={recording_subtype} "
                f"(allowed={','.join(sorted(allowed_for_type))})",
            ))
    if behavior_mode and behavior_mode not in DEFAULT_ALLOWED_BEHAVIOR_MODES:
        issues.append((
            "invalid_behavior_mode",
            f"{behavior_mode} (allowed={','.join(sorted(DEFAULT_ALLOWED_BEHAVIOR_MODES))})",
        ))
    if recording_type == "behavior" and recording_subtype and behavior_mode:
        if recording_subtype != behavior_mode:
            issues.append((
                "behavior_mode_mismatch",
                f"recording_subtype={recording_subtype} behavior_mode={behavior_mode}",
            ))
    return issues


def validate_recording_manifest_context(payload: Mapping[str, Any]) -> None:
    """Require producer-declared context before a new import or sealed replay."""

    issues = recording_manifest_context_issues(payload)
    if issues:
        raise ValueError("invalid recording manifest context: " + "; ".join(
            f"{code}: {detail}" for code, detail in issues
        ))
