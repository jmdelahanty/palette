"""Implementation-status artifact helpers for web labeling launch bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .web_report_renderers import (
    _IMPLEMENTATION_STATUS_GATE_CONTRACT_SENTENCE,
    _IMPLEMENTATION_STATUS_INSPECT_FIELDS_TEXT,
    _IMPLEMENTATION_STATUS_NOT_LAUNCH_APPROVAL_SENTENCE,
    _IMPLEMENTATION_STATUS_SAFE_SHARE_REQUIREMENT_SENTENCE,
    _IMPLEMENTATION_STATUS_STALE_PACKAGE_FAIL_CLOSED_SENTENCE,
)

_IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS = [
    "schema",
    "role",
    "checklist_declared_path",
    "file",
    "is_launch_approval",
    "operator_evidence_required_before_share",
    "safe_share_gate",
    "safe_share_required_inspection_field",
    "safe_share_required_inspection_value",
    "require_shareable_inspection_before_share",
]
_IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT = len(
    _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS
)

_IMPLEMENTATION_STATUS_FLAT_FIELDS = [
    "implementation_status",
    "implementation_status_file",
    "implementation_status_required_path",
    "implementation_status_declared_present",
    "implementation_status_declared_present_count",
    "implementation_status_present",
    "implementation_status_is_launch_approval",
    "implementation_status_operator_evidence_required_before_share",
    "implementation_status_safe_share_gate",
    "implementation_status_safe_share_required_inspection_field",
    "implementation_status_safe_share_required_inspection_value",
    "implementation_status_require_shareable_inspection_before_share",
    "implementation_status_matched_paths",
]
_IMPLEMENTATION_STATUS_FLAT_FIELD_COUNT = len(_IMPLEMENTATION_STATUS_FLAT_FIELDS)

_IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELDS = [
    "implementation_status",
    "implementation_status_artifact_schema",
    "implementation_status_file",
    "implementation_status_role",
    "implementation_status_is_launch_approval",
    "implementation_status_operator_evidence_required_before_share",
    "implementation_status_safe_share_gate",
    "implementation_status_safe_share_required_inspection_field",
    "implementation_status_safe_share_required_inspection_value",
    "implementation_status_require_shareable_inspection_before_share",
]
_IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELD_COUNT = len(
    _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELDS
)

_IMPLEMENTATION_STATUS_INSPECT_FIELDS_ARE_SENTENCE = (
    f"Implementation status inspect fields are {_IMPLEMENTATION_STATUS_INSPECT_FIELDS_TEXT}."
)
_IMPLEMENTATION_STATUS_MACHINE_READABLE_FIELDS_SENTENCE = (
    f"Machine-readable inspection fields: {_IMPLEMENTATION_STATUS_INSPECT_FIELDS_TEXT}."
)
_IMPLEMENTATION_STATUS_FILE_ADVISORY_SENTENCE = (
    "This implementation-status file is advisory status metadata, not launch approval."
)
_IMPLEMENTATION_STATUS_DOES_NOT_REPLACE_SENTENCE = (
    "It does not replace inspect-handoff --require-shareable or labeler_links_safe_to_share=true."
)


def _implementation_status_advisory_fields(
    *, prefix: str = "implementation_status"
) -> dict[str, object]:
    return {
        f"{prefix}_artifact_schema": "palette.web_labeling_implementation_status_artifact.v1",
        f"{prefix}_file": "implementation-status.txt",
        f"{prefix}_role": "bundle_local_implementation_evidence_status_summary",
        f"{prefix}_is_launch_approval": False,
        f"{prefix}_operator_evidence_required_before_share": True,
        f"{prefix}_safe_share_gate": "labeler_links_safe_to_share",
        f"{prefix}_safe_share_required_inspection_field": "labeler_links_safe_to_share",
        f"{prefix}_safe_share_required_inspection_value": True,
        f"{prefix}_require_shareable_inspection_before_share": True,
    }


def _implementation_status_artifact(
    *,
    checklist_declared_path: str = "",
    file: str = "implementation-status.txt",
) -> dict[str, object]:
    fields = _implementation_status_advisory_fields()
    return {
        "schema": fields["implementation_status_artifact_schema"],
        "role": fields["implementation_status_role"],
        "checklist_declared_path": checklist_declared_path,
        "file": file or str(fields["implementation_status_file"]),
        "is_launch_approval": fields["implementation_status_is_launch_approval"],
        "operator_evidence_required_before_share": fields[
            "implementation_status_operator_evidence_required_before_share"
        ],
        "safe_share_gate": fields["implementation_status_safe_share_gate"],
        "safe_share_required_inspection_field": fields[
            "implementation_status_safe_share_required_inspection_field"
        ],
        "safe_share_required_inspection_value": fields[
            "implementation_status_safe_share_required_inspection_value"
        ],
        "require_shareable_inspection_before_share": fields[
            "implementation_status_require_shareable_inspection_before_share"
        ],
    }


def _implementation_status_flat_fields_from_artifact(
    artifact: Mapping[str, object],
    *,
    prefix: str = "implementation_status",
) -> dict[str, object]:
    return {
        f"{prefix}_artifact_schema": str(artifact.get("schema") or ""),
        f"{prefix}_file": str(artifact.get("file") or ""),
        f"{prefix}_role": str(artifact.get("role") or ""),
        f"{prefix}_is_launch_approval": bool(artifact.get("is_launch_approval")),
        f"{prefix}_operator_evidence_required_before_share": bool(
            artifact.get("operator_evidence_required_before_share")
        ),
        f"{prefix}_safe_share_gate": str(artifact.get("safe_share_gate") or ""),
        f"{prefix}_safe_share_required_inspection_field": str(
            artifact.get("safe_share_required_inspection_field") or ""
        ),
        f"{prefix}_safe_share_required_inspection_value": bool(
            artifact.get("safe_share_required_inspection_value")
        ),
        f"{prefix}_require_shareable_inspection_before_share": bool(
            artifact.get("require_shareable_inspection_before_share")
        ),
    }


def _implementation_status_metadata_fields(
    *,
    checklist_declared_path: str = "",
    file: str = "implementation-status.txt",
) -> dict[str, object]:
    artifact = _implementation_status_artifact(
        checklist_declared_path=checklist_declared_path,
        file=file,
    )
    return {
        "implementation_status_artifact": artifact,
        "implementation_status_artifact_required_fields": list(
            _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS
        ),
        "implementation_status_artifact_required_field_count": (
            _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT
        ),
        "implementation_status_flat_fields": list(
            _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELDS
        ),
        "implementation_status_flat_field_count": (
            _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELD_COUNT
        ),
        **_implementation_status_flat_fields_from_artifact(artifact),
    }


def _implementation_status_inspection_target_fields(
    *, prefix: str = "shareability_implementation_status"
) -> dict[str, object]:
    return {
        **_implementation_status_advisory_fields(prefix=prefix),
        f"{prefix}_included_in_checksums": True,
        f"{prefix}_artifact_field": "shareability.implementation_status_artifact",
        f"{prefix}_top_level_field": "implementation_status_artifact",
        f"{prefix}_payload_path_field": "implementation_status",
        f"{prefix}_payload_artifact_field": "implementation_status_artifact",
        f"{prefix}_payload_artifact_schema_field": (
            "implementation_status_artifact_schema"
        ),
        f"{prefix}_artifact_contract": _implementation_status_artifact(),
        f"{prefix}_artifact_required_fields": list(
            _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS
        ),
        f"{prefix}_artifact_required_field_count": (
            _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT
        ),
        f"{prefix}_payload_artifact_required_fields_field": (
            "implementation_status_artifact_required_fields"
        ),
        f"{prefix}_payload_artifact_required_field_count_field": (
            "implementation_status_artifact_required_field_count"
        ),
        f"{prefix}_payload_flat_fields_field": "implementation_status_flat_fields",
        f"{prefix}_payload_flat_field_count_field": (
            "implementation_status_flat_field_count"
        ),
        f"{prefix}_payload_flat_fields": list(
            _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELDS
        ),
        f"{prefix}_payload_flat_field_count": (
            _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELD_COUNT
        ),
        f"{prefix}_inspect_flat_fields_field": "implementation_status_flat_fields",
        f"{prefix}_inspect_flat_field_count_field": (
            "implementation_status_flat_field_count"
        ),
        f"{prefix}_flat_fields": list(_IMPLEMENTATION_STATUS_FLAT_FIELDS),
        f"{prefix}_flat_field_count": _IMPLEMENTATION_STATUS_FLAT_FIELD_COUNT,
        f"{prefix}_checklist_artifact_present_field": (
            "implementation_status_checklist_artifact_present"
        ),
        f"{prefix}_checklist_artifact_gate_field": (
            "implementation_status_checklist_artifact_gate"
        ),
        f"{prefix}_checklist_artifact_gate_schema": (
            "palette.web_labeling_implementation_status_checklist_artifact_gate.v1"
        ),
        f"{prefix}_checklist_artifact_gate_contract": {
            "schema": "palette.web_labeling_implementation_status_checklist_artifact_gate.v1",
            "gate_field": "implementation_status_checklist_artifact_gate",
            "observed_value_field": "implementation_status_checklist_artifact_complete",
            "required_value": True,
            "matches_required_value_field": (
                "implementation_status_checklist_artifact_complete_matches_required_value"
            ),
            "missing_fields_field": "implementation_status_checklist_artifact_missing_fields",
            "missing_field_count_field": (
                "implementation_status_checklist_artifact_missing_field_count"
            ),
            "fail_closed_reason": "implementation_status_artifact_incomplete",
            "required_value_mismatch_blocking_reason": (
                "implementation_status_checklist_artifact_complete_required_value_mismatch"
            ),
            "repair_command_id": "regenerate_package_with_implementation_status_artifact",
        },
        f"{prefix}_checklist_artifact_complete_field": (
            "implementation_status_checklist_artifact_complete"
        ),
        f"{prefix}_checklist_artifact_complete_required_value": True,
        f"{prefix}_checklist_artifact_complete_matches_required_value_field": (
            "implementation_status_checklist_artifact_complete_matches_required_value"
        ),
        f"{prefix}_checklist_artifact_missing_fields_field": (
            "implementation_status_checklist_artifact_missing_fields"
        ),
        f"{prefix}_checklist_artifact_missing_field_count_field": (
            "implementation_status_checklist_artifact_missing_field_count"
        ),
        f"{prefix}_stale_package_fail_closed_reason": (
            "implementation_status_artifact_incomplete"
        ),
        f"{prefix}_required_value_mismatch_blocking_reason": (
            "implementation_status_checklist_artifact_complete_required_value_mismatch"
        ),
        f"{prefix}_stale_package_repair_command_id": (
            "regenerate_package_with_implementation_status_artifact"
        ),
    }


def _implementation_status_artifact_name(
    payload: Mapping[str, object] | None,
) -> str:
    configured_path = (
        str(payload.get("implementation_status") or "").strip() if payload else ""
    )
    artifact_name = (
        Path(configured_path).name if configured_path else "implementation-status.txt"
    )
    return artifact_name or "implementation-status.txt"


def _implementation_status_artifact_summary(
    payload: Mapping[str, object] | None,
    *,
    required_path: str,
    matched_paths: list[str],
    related_paths: list[str],
) -> dict[str, object]:
    configured_path = (
        str(payload.get("implementation_status") or "").strip() if payload else ""
    )
    counts = (
        payload.get("counts")
        if payload and isinstance(payload.get("counts"), Mapping)
        else {}
    )
    try:
        declared_present_count = int(counts.get("implementation_status_present") or 0)
    except (TypeError, ValueError):
        declared_present_count = 0
    artifact = (
        payload.get("implementation_status_artifact")
        if payload and isinstance(payload.get("implementation_status_artifact"), Mapping)
        else {}
    )
    artifact_missing_fields = [
        field
        for field in _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS
        if field not in artifact
    ]
    return {
        "implementation_status": configured_path,
        "implementation_status_file": _implementation_status_artifact_name(payload),
        "implementation_status_required_path": required_path,
        "implementation_status_artifact_present": bool(artifact),
        "implementation_status_artifact_complete": bool(artifact)
        and not artifact_missing_fields,
        "implementation_status_artifact_missing_fields": artifact_missing_fields,
        "implementation_status_artifact_missing_field_count": len(
            artifact_missing_fields
        ),
        "implementation_status_artifact_required_fields": list(
            _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS
        ),
        "implementation_status_artifact_required_field_count": (
            _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT
        ),
        "implementation_status_declared_present": bool(declared_present_count),
        "implementation_status_declared_present_count": declared_present_count,
        "implementation_status_present": bool(matched_paths),
        "implementation_status_matched_paths": list(matched_paths),
        "implementation_status_related_paths": list(related_paths),
    }
