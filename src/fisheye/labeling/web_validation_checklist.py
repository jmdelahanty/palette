"""Validation checklist helpers for web labeling launch artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .admin_dashboard import (
    OPERATOR_EVIDENCE_VALIDATION_GATE_IDS,
    _VALIDATION_GATE_BLOCKS_INVITATION_LEGACY_SEMANTICS,
    _VALIDATION_GATE_BLOCKS_INVITATION_SAFE_SHARE_FIELD,
    _operator_validation_command_templates,
    _safe_share_checklist_gate_status_fields,
    _safe_share_gate_flat_fields,
    _safe_share_gate_policy,
    _validation_gate_classification,
    _validation_gate_kind,
)

VALIDATION_GATE_STATUSES = {
    "passed",
    "pending_operator_evidence",
    "needs_review",
    "not_applicable",
}


def _validation_gate_blocks_invitation_semantics_fields() -> dict[str, object]:
    return {
        "blocks_invitation_legacy_semantics": (
            _VALIDATION_GATE_BLOCKS_INVITATION_LEGACY_SEMANTICS
        ),
        "blocks_invitation_is_safe_share_approval": False,
        "blocks_invitation_safe_share_field": (
            _VALIDATION_GATE_BLOCKS_INVITATION_SAFE_SHARE_FIELD
        ),
    }


def _validation_gate(
    gate_id: str,
    title: str,
    status: str,
    *,
    required: bool = True,
    blocks_invitation: bool = True,
    evidence_files: Sequence[object] = (),
    operator_evidence: Sequence[str] = (),
    details: str = "",
) -> dict[str, object]:
    return {
        "id": gate_id,
        "title": title,
        "status": status,
        "required": bool(required),
        "blocks_invitation": bool(blocks_invitation),
        **_validation_gate_blocks_invitation_semantics_fields(),
        "evidence_files": [
            str(value) for value in evidence_files if str(value or "").strip()
        ],
        "operator_evidence": [
            str(value) for value in operator_evidence if str(value or "").strip()
        ],
        "details": details,
    }


def _validation_checklist_gate_choices(
    payload: Mapping[str, object],
) -> list[dict[str, object]]:
    gates = payload.get("gates") if isinstance(payload.get("gates"), list) else []
    choices: list[dict[str, object]] = []
    for gate in gates:
        if not isinstance(gate, Mapping):
            continue
        gate_id = str(gate.get("id") or "").strip()
        if not gate_id:
            continue
        choices.append(
            {
                "id": gate_id,
                "title": str(gate.get("title") or ""),
                "status": str(gate.get("status") or ""),
                "required": bool(gate.get("required", True)),
                "blocks_invitation": bool(gate.get("blocks_invitation", True)),
                "blocks_invitation_legacy_semantics": str(
                    gate.get("blocks_invitation_legacy_semantics")
                    or _VALIDATION_GATE_BLOCKS_INVITATION_LEGACY_SEMANTICS
                ),
                "blocks_invitation_is_safe_share_approval": bool(
                    gate.get("blocks_invitation_is_safe_share_approval", False)
                ),
                "blocks_invitation_safe_share_field": str(
                    gate.get("blocks_invitation_safe_share_field")
                    or _VALIDATION_GATE_BLOCKS_INVITATION_SAFE_SHARE_FIELD
                ),
                "gate_kind": _validation_gate_kind(gate_id),
            }
        )
    return choices


def _recompute_validation_checklist_summary(
    payload: dict[str, object],
) -> dict[str, object]:
    gates = payload.get("gates") if isinstance(payload.get("gates"), list) else []
    existing_counts = (
        payload.get("counts") if isinstance(payload.get("counts"), Mapping) else {}
    )
    preserved_counts = {
        str(key): value
        for key, value in existing_counts.items()
        if str(key) != "gates" and str(key) not in VALIDATION_GATE_STATUSES
    }
    gate_counts: dict[str, int] = {}
    valid_gates: list[Mapping[str, object]] = []
    for gate in gates:
        if not isinstance(gate, Mapping):
            continue
        status = str(gate.get("status") or "unknown").strip() or "unknown"
        gate_counts[status] = gate_counts.get(status, 0) + 1
        valid_gates.append(gate)
    gate_classification = _validation_gate_classification(valid_gates)
    payload["all_validation_complete"] = not any(
        str(gate.get("status") or "")
        in {"needs_review", "pending_operator_evidence"}
        for gate in valid_gates
        if bool(gate.get("required", True))
    )
    payload["ready_for_operator_validation"] = not any(
        str(gate.get("status") or "") == "needs_review" for gate in valid_gates
    )
    payload["validation_gate_classification"] = gate_classification
    payload["operator_evidence_gate_ids"] = gate_classification[
        "operator_evidence_gate_ids"
    ]
    payload["generated_contract_gate_ids"] = gate_classification[
        "generated_contract_gate_ids"
    ]
    payload["operator_evidence_pending_gate_ids"] = gate_classification[
        "operator_evidence_pending_gate_ids"
    ]
    payload["operator_evidence_needs_review_gate_ids"] = gate_classification[
        "operator_evidence_needs_review_gate_ids"
    ]
    payload["operator_validation_command_templates"] = (
        _operator_validation_command_templates(
            [
                str(gate_id)
                for gate_id in [
                    *gate_classification["operator_evidence_pending_gate_ids"],
                    *gate_classification["operator_evidence_needs_review_gate_ids"],
                ]
                if str(gate_id).strip()
            ]
        )
    )
    safe_share_gate = (
        payload.get("safe_share_gate")
        if isinstance(payload.get("safe_share_gate"), Mapping)
        else _safe_share_gate_policy()
    )
    payload["safe_share_gate"] = safe_share_gate
    payload.update(_safe_share_gate_flat_fields(safe_share_gate))
    payload.update(
        _safe_share_checklist_gate_status_fields(
            gates=valid_gates,
            safe_share_gate=safe_share_gate,
        )
    )
    payload["generated_contract_failed_gate_ids"] = gate_classification[
        "generated_contract_failed_gate_ids"
    ]
    payload["counts"] = {
        **preserved_counts,
        "gates": len(valid_gates),
        "operator_evidence_gates": int(
            gate_classification["operator_evidence_gate_count"]
        ),
        "generated_contract_gates": int(
            gate_classification["generated_contract_gate_count"]
        ),
        "operator_evidence_pending_gates": int(
            gate_classification["operator_evidence_pending_gate_count"]
        ),
        "operator_evidence_needs_review_gates": int(
            gate_classification["operator_evidence_needs_review_gate_count"]
        ),
        "generated_contract_failed_gates": int(
            gate_classification["generated_contract_failed_gate_count"]
        ),
        **dict(sorted(gate_counts.items())),
    }
    return payload


def _update_validation_checklist_payload(
    payload: dict[str, object],
    *,
    gate_id: str,
    status: str,
    evidence: Sequence[str],
    evidence_files: Sequence[str],
    operator: str | None,
) -> dict[str, object]:
    if str(payload.get("schema") or "") != "palette.web_labeling_validation_checklist.v1":
        raise ValueError(
            "Input is not a palette.web_labeling_validation_checklist.v1 payload."
        )
    normalized_gate_id = str(gate_id or "").strip()
    normalized_status = str(status or "").strip()
    if normalized_status not in VALIDATION_GATE_STATUSES:
        raise ValueError(f"Unsupported validation gate status: {normalized_status}")
    gates = payload.get("gates") if isinstance(payload.get("gates"), list) else []
    target_gate: dict[str, object] | None = None
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        if str(gate.get("id") or "").strip() == normalized_gate_id:
            target_gate = gate
            break
    if target_gate is None:
        available = _validation_checklist_gate_choices(payload)
        available_text = ", ".join(
            f"{gate['id']}({gate['status'] or 'unknown'})" for gate in available
        ) or "none"
        raise ValueError(
            f"Validation gate not found: {normalized_gate_id}. "
            f"Available gates: {available_text}"
        )
    updated_at_utc = datetime.now(timezone.utc).isoformat()
    previous_status = str(target_gate.get("status") or "")
    for key, value in _validation_gate_blocks_invitation_semantics_fields().items():
        target_gate.setdefault(key, value)
    target_gate["status"] = normalized_status
    target_gate["evidence_recorded_at_utc"] = updated_at_utc
    if operator:
        target_gate["evidence_recorded_by"] = str(operator)
    clean_evidence = [str(item) for item in evidence if str(item or "").strip()]
    clean_evidence_files = list(
        dict.fromkeys(str(item) for item in evidence_files if str(item or "").strip())
    )
    if clean_evidence:
        target_gate["evidence_notes"] = clean_evidence
    if clean_evidence_files:
        existing_evidence_files = (
            target_gate.get("evidence_files")
            if isinstance(target_gate.get("evidence_files"), list)
            else []
        )
        target_gate["evidence_files"] = list(
            dict.fromkeys(
                [
                    *(
                        str(item)
                        for item in existing_evidence_files
                        if str(item or "").strip()
                    ),
                    *clean_evidence_files,
                ]
            )
        )
    evidence_entries = target_gate.get("evidence")
    if not isinstance(evidence_entries, list):
        evidence_entries = []
    evidence_entries.append(
        {
            "recorded_at_utc": updated_at_utc,
            "recorded_by": str(operator or ""),
            "previous_status": previous_status,
            "status": normalized_status,
            "notes": clean_evidence,
            "evidence_files": clean_evidence_files,
        }
    )
    target_gate["evidence"] = evidence_entries
    payload["updated_at_utc"] = updated_at_utc
    payload["updated_by"] = str(operator or "")
    _recompute_validation_checklist_summary(payload)
    return payload


def _append_validation_log_evidence(
    *,
    path: Path,
    checklist_path: Path,
    gate_id: str,
    status: str,
    evidence: Sequence[str],
    evidence_files: Sequence[str],
    operator: str | None,
    updated_payload: Mapping[str, object],
) -> None:
    gates = (
        updated_payload.get("gates")
        if isinstance(updated_payload.get("gates"), list)
        else []
    )
    target_gate = next(
        (
            gate
            for gate in gates
            if isinstance(gate, Mapping)
            and str(gate.get("id") or "").strip() == str(gate_id).strip()
        ),
        {},
    )
    recorded_at = str(
        target_gate.get("evidence_recorded_at_utc")
        or updated_payload.get("updated_at_utc")
        or ""
    )
    title = str(target_gate.get("title") or gate_id)
    clean_evidence = [str(item) for item in evidence if str(item or "").strip()]
    clean_evidence_files = [
        str(item) for item in evidence_files if str(item or "").strip()
    ]
    entry_lines = [
        "",
        f"## Validation Evidence: {title}",
        "",
        f"- Gate ID: {gate_id}",
        f"- Status: {status}",
        f"- Recorded at UTC: {recorded_at}",
        f"- Operator: {operator or ''}",
        f"- Checklist: {checklist_path}",
        "- Evidence notes:",
        *[f"  - {item}" for item in clean_evidence],
        "- Evidence files:",
        *[f"  - {item}" for item in clean_evidence_files],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        path.read_text(encoding="utf-8")
        if path.exists()
        else "# Web Labeling Validation Log\n"
    )
    separator = "" if existing.endswith("\n") else "\n"
    path.write_text(existing + separator + "\n".join(entry_lines), encoding="utf-8")


def _update_validation_checklist_file(
    *,
    path: Path,
    gate_id: str,
    status: str,
    evidence: Sequence[str],
    evidence_files: Sequence[str],
    operator: str | None,
    append_log: Path | None,
    output: Path | None,
    overwrite: bool,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Validation checklist must be a JSON object: {path}")
    available_gates_before = _validation_checklist_gate_choices(payload)
    previous_gate = next(
        (
            gate
            for gate in available_gates_before
            if str(gate.get("id") or "") == str(gate_id).strip()
        ),
        None,
    )
    previous_status = str((previous_gate or {}).get("status") or "")
    recorded_evidence_files = list(
        dict.fromkeys(str(item) for item in evidence_files if str(item or "").strip())
    )
    if append_log is not None:
        append_log_text = str(append_log)
        if append_log_text not in recorded_evidence_files:
            recorded_evidence_files.append(append_log_text)
    updated = _update_validation_checklist_payload(
        payload,
        gate_id=gate_id,
        status=status,
        evidence=evidence,
        evidence_files=recorded_evidence_files,
        operator=operator,
    )
    output_path = output or path
    if output is not None and output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing validation checklist: {output_path}"
        )
    validation_log_appended = False
    if append_log is not None:
        _append_validation_log_evidence(
            path=append_log,
            checklist_path=output_path,
            gate_id=gate_id,
            status=status,
            evidence=evidence,
            evidence_files=recorded_evidence_files,
            operator=operator,
            updated_payload=updated,
        )
        validation_log_appended = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    safe_share_gate = (
        updated.get("safe_share_gate")
        if isinstance(updated.get("safe_share_gate"), Mapping)
        else _safe_share_gate_policy()
    )
    safe_share_fields = _safe_share_gate_flat_fields(safe_share_gate)
    safe_share_checklist_fields = _safe_share_checklist_gate_status_fields(
        gates=updated.get("gates") if isinstance(updated.get("gates"), list) else [],
        safe_share_gate=safe_share_gate,
    )
    return {
        "ok": True,
        "path": str(path),
        "output": str(output_path),
        "gate_id": str(gate_id),
        "previous_status": previous_status,
        "status": str(status),
        "all_validation_complete": bool(updated.get("all_validation_complete")),
        "ready_for_operator_validation": bool(
            updated.get("ready_for_operator_validation")
        ),
        "validation_gate_classification": updated.get(
            "validation_gate_classification", {}
        ),
        "operator_evidence_gate_ids": updated.get("operator_evidence_gate_ids", []),
        "generated_contract_gate_ids": updated.get("generated_contract_gate_ids", []),
        "operator_evidence_pending_gate_ids": updated.get(
            "operator_evidence_pending_gate_ids", []
        ),
        "operator_evidence_needs_review_gate_ids": updated.get(
            "operator_evidence_needs_review_gate_ids", []
        ),
        "generated_contract_failed_gate_ids": updated.get(
            "generated_contract_failed_gate_ids", []
        ),
        "counts": updated.get("counts", {}),
        "safe_share_gate": safe_share_gate,
        **safe_share_fields,
        **safe_share_checklist_fields,
        "available_gates": _validation_checklist_gate_choices(updated),
        "validation_log_appended": validation_log_appended,
        "validation_log": str(append_log) if append_log is not None else "",
        "validation_checklist": updated,
    }
