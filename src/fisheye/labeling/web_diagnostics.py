"""Shared compact diagnostic payload helpers for labeling web surfaces."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from .admin_dashboard import (
    _assignment_ownership_contract_fields,
    _assignment_ownership_contract_policy,
    _assignment_ownership_policy,
    _browser_mutation_target_contract_summary,
    _dataset_queue_direct_start_policy_fields,
    _direct_browser_start_contract_summary,
    _labeler_route_authorization_runtime_checklist_compact_fields,
    _safe_share_external_launch_evidence_gap_todo_fields,
)

__all__ = [
    "_add_payload_contract_compact_fields",
    "_browser_mutation_target_contract_compact_fields",
    "_browser_mutation_target_contract_source_from_checklist",
    "_direct_browser_start_contract_compact_fields",
    "_personalized_launch_readiness_field_names",
    "_personalized_launch_readiness_summary",
    "_queue_first_entry_contract_flat_fields",
]

def _queue_first_entry_contract_flat_fields(
    source: Mapping[str, object] | None = None,
) -> dict[str, object]:
    contract_obj = source.get("queue_first_entry_contract") if isinstance(source, Mapping) else None
    contract: Mapping[str, object] = contract_obj if isinstance(contract_obj, Mapping) else {}
    return {
        "queue_first_entry_contract_schema": str(contract.get("schema") or ""),
        "queue_first_entry_contract_ready": bool(contract.get("ready")),
        "queue_first_entry_contract_preferred_labeler_entrypoint": str(
            contract.get("preferred_labeler_entrypoint") or ""
        ),
        "queue_first_entry_contract_preferred_labeler_entry_url": str(
            contract.get("preferred_labeler_entry_url") or ""
        ),
        "queue_first_entry_contract_personalized_labeler_entrypoint": str(
            contract.get("personalized_labeler_entrypoint") or ""
        ),
        "queue_first_entry_contract_personalized_labeler_entry_url": str(
            contract.get("personalized_labeler_entry_url") or ""
        ),
        "queue_first_entry_contract_personalized_entry_required": bool(
            contract.get("personalized_entry_required")
        ),
        "queue_first_entry_contract_personalized_labeler_entry_url_matches_personal_dataset_queue": bool(
            contract.get("personalized_labeler_entry_url_matches_personal_dataset_queue")
        ),
        "queue_first_entry_contract_preferred_labeler_entry_url_matches_personal_dataset_queue": bool(
            contract.get("preferred_labeler_entry_url_matches_personal_dataset_queue")
        ),
        "queue_first_entry_contract_preferred_labeler_entry_url_is_expected_user_guarded": bool(
            contract.get("preferred_labeler_entry_url_is_expected_user_guarded")
        ),
        "queue_first_entry_contract_personalized_labeler_entry_url_is_expected_user_guarded": bool(
            contract.get("personalized_labeler_entry_url_is_expected_user_guarded")
        ),
        "queue_first_entry_contract_landing_ready": bool(contract.get("landing_ready")),
        "queue_first_entry_contract_labeling_home_ready": bool(
            contract.get("labeling_home_ready")
        ),
        "queue_first_entry_contract_dataset_queue_ready": bool(
            contract.get("dataset_queue_ready")
        ),
        "queue_first_entry_contract_personal_dataset_queue_ready": bool(
            contract.get("personal_dataset_queue_ready")
        ),
        "queue_first_entry_contract_personal_work_ready": bool(
            contract.get("personal_work_ready")
        ),
        "queue_first_entry_contract_queue_first_paths_ready": bool(
            contract.get("queue_first_paths_ready")
        ),
        "queue_first_entry_contract_datasets_waiting_aliases_ready": bool(
            contract.get("datasets_waiting_aliases_ready")
        ),
        "queue_first_entry_contract_expected_user_landing_guard": bool(
            contract.get("expected_user_landing_guard")
        ),
        "queue_first_entry_contract_expected_user_queue_guard": bool(
            contract.get("expected_user_queue_guard")
        ),
        "queue_first_entry_contract_expected_user_dashboard_guard": bool(
            contract.get("expected_user_dashboard_guard")
        ),
    }


def _browser_mutation_target_contract_source_from_checklist(
    checklist: Mapping[str, object] | None,
    *,
    user: str = "",
) -> dict[str, object]:
    source = checklist if isinstance(checklist, Mapping) else {}
    return {
        "user": str(user),
        "browser_mutation_write_ready": bool(source.get("ready")),
        "browser_mutation_label_mutation_target_kind": str(
            source.get("label_mutation_target_kind") or ""
        ),
        "browser_mutation_browser_label_write_target": str(
            source.get("browser_label_write_target") or ""
        ),
        "browser_mutation_csv_handoff_artifact_role": str(
            source.get("csv_handoff_artifact_role") or ""
        ),
        "browser_mutation_csv_handoff_artifacts_are_label_write_targets": bool(
            source.get("csv_handoff_artifacts_are_label_write_targets")
        ),
        "browser_mutation_handoff_csv_artifacts_are_label_write_targets": bool(
            source.get("handoff_csv_artifacts_are_label_write_targets")
        ),
        "browser_mutation_intermediate_csv_artifacts_are_label_write_targets": bool(
            source.get("intermediate_csv_artifacts_are_label_write_targets")
        ),
        "browser_mutation_browser_writes_csv_or_handoff_files": bool(
            source.get("browser_writes_csv_or_handoff_files")
        ),
        "browser_mutation_browser_writes_handoff_csv": bool(
            source.get("browser_writes_handoff_csv")
        ),
        "browser_mutation_browser_writes_intermediate_csv": bool(
            source.get("browser_writes_intermediate_csv")
        ),
        "browser_mutation_browser_has_direct_zarr_write_authority": bool(
            source.get("browser_has_direct_zarr_write_authority")
        ),
    }


def _browser_mutation_target_contract_compact_fields(
    checklist: Mapping[str, object] | None,
    *,
    user: str = "",
) -> dict[str, object]:
    summary = _browser_mutation_target_contract_summary(
        [_browser_mutation_target_contract_source_from_checklist(checklist, user=user)]
    )
    return {
        "browser_mutation_target_contract_met": bool(summary.get("met")),
        "browser_mutation_target_mismatch_count": int(summary.get("mismatch_count") or 0),
        "browser_mutation_target_mismatch_users": list(summary.get("mismatch_users", [])),
    }


def _direct_browser_start_contract_compact_fields(
    policy: Mapping[str, object] | None,
    *,
    user: str = "",
) -> dict[str, object]:
    source = {
        "user": str(user),
        "dataset_queue_direct_start_policy_present": isinstance(policy, Mapping),
        **_dataset_queue_direct_start_policy_fields(
            policy if isinstance(policy, Mapping) else None
        ),
    }
    summary = _direct_browser_start_contract_summary([source])
    return {
        "direct_browser_start_contract_met": bool(summary.get("met")),
        "direct_browser_start_mismatch_count": int(summary.get("mismatch_count") or 0),
        "direct_browser_start_mismatch_users": list(summary.get("mismatch_users", [])),
    }






def _personalized_launch_readiness_field_names() -> list[str]:
    return [
        "schema",
        "fields",
        "field_count",
        "expected_user",
        "personalized_labeler_entry_url",
        "preferred_labeler_entry_url",
        "queue_first_entry_contract_ready",
        "personalized_labeler_entry_url_matches_personal_dataset_queue",
        "labeler_start_ready",
        "labeler_start_status",
        "labeler_action",
        "labeler_work_completion_status",
        "safe_share_gate_id",
        "safe_share_required_inspection_field",
        "safe_share_required_inspection_value",
        "safe_share_checklist_gate_evidence_complete",
        "external_launch_evidence_gap_action_required",
        "external_launch_evidence_gap_count",
        "external_launch_evidence_gap_gate_ids",
        "external_launch_evidence_gap_statuses",
        "external_launch_evidence_gap_summary",
        "external_launch_evidence_gap_todos",
        "external_launch_evidence_gap_todo_count",
        "external_launch_evidence_gap_todo_fields",
        "external_launch_evidence_gap_template_paths_by_gate_id",
        "external_launch_evidence_gap_record_command_ids_by_gate_id",
        "browser_label_write_target",
        "browser_writes_csv_or_handoff_files",
        "browser_has_direct_zarr_write_authority",
    ]


def _personalized_launch_readiness_summary(
    source: Mapping[str, object],
    *,
    fallback: Mapping[str, object] | None = None,
) -> dict[str, object]:
    fallback_source = fallback if isinstance(fallback, Mapping) else {}
    fields = _personalized_launch_readiness_field_names()

    def _value(name: str, default: object = "") -> object:
        value = source.get(name)
        if value is None:
            value = fallback_source.get(name)
        return default if value is None else value

    def _mapping_value(name: str, field: str, default: object = None) -> object:
        mapping = _value(name, None)
        if isinstance(mapping, Mapping):
            value = mapping.get(field)
            if value is not None:
                return value
        return default

    def _first_contract_value(
        field_names: Sequence[str],
        nested_fields: Sequence[tuple[str, str]],
        *,
        default: object = "",
    ) -> object:
        for name in field_names:
            value = _value(name, None)
            if value is not None and value != "":
                return value
        for mapping_name, field_name in nested_fields:
            value = _mapping_value(mapping_name, field_name, None)
            if value is not None and value != "":
                return value
        return default

    def _coerce_bool(value: object, *, default: bool = False) -> bool:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        return bool(value)

    def _contract_bool(
        field_names: Sequence[str],
        nested_fields: Sequence[tuple[str, str]],
        *,
        default: bool,
    ) -> bool:
        value = _first_contract_value(field_names, nested_fields, default=default)
        return _coerce_bool(value, default=default)

    browser_label_write_target = str(
        _first_contract_value(
            [
                "browser_label_write_target",
                "browser_mutation_browser_label_write_target",
                "dataset_queue_direct_start_browser_label_write_target",
                "direct_browser_start_contract_summary_browser_label_write_target",
            ],
            [
                ("browser_mutation_write_checklist", "browser_label_write_target"),
                ("browser_mutation_write_policy", "browser_label_write_target"),
                ("dataset_queue_direct_start_policy", "browser_label_write_target"),
                ("direct_browser_start_contract_summary", "browser_label_write_target"),
            ],
            default="",
        )
        or ""
    )
    browser_writes_csv_or_handoff_files = _contract_bool(
        [
            "browser_writes_csv_or_handoff_files",
            "browser_mutation_browser_writes_csv_or_handoff_files",
            "dataset_queue_direct_start_browser_writes_csv_or_handoff_files",
            "direct_browser_start_contract_summary_browser_writes_csv_or_handoff_files",
        ],
        [
            ("browser_mutation_write_checklist", "browser_writes_csv_or_handoff_files"),
            ("browser_mutation_write_policy", "browser_writes_csv_or_handoff_files"),
            ("dataset_queue_direct_start_policy", "browser_writes_csv_or_handoff_files"),
            ("direct_browser_start_contract_summary", "browser_writes_csv_or_handoff_files"),
        ],
        default=True,
    )
    browser_has_direct_zarr_write_authority = _contract_bool(
        [
            "browser_has_direct_zarr_write_authority",
            "browser_mutation_browser_has_direct_zarr_write_authority",
            "dataset_queue_direct_start_browser_has_direct_zarr_write_authority",
            "direct_browser_start_contract_summary_browser_has_direct_zarr_write_authority",
        ],
        [
            ("browser_mutation_write_checklist", "browser_has_direct_zarr_write_authority"),
            ("browser_mutation_write_policy", "browser_has_direct_zarr_write_authority"),
            ("dataset_queue_direct_start_policy", "browser_has_direct_zarr_write_authority"),
            ("direct_browser_start_contract_summary", "browser_has_direct_zarr_write_authority"),
        ],
        default=True,
    )

    def _first_named_value(names: Sequence[str], default: object = None) -> object:
        for name in names:
            value = _value(name, None)
            if value is not None and value != "":
                return value
        return default

    def _list_contract_value(names: Sequence[str]) -> list[object]:
        value = _first_named_value(names, [])
        if isinstance(value, list):
            return list(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return parsed
        return []

    def _mapping_contract_value(names: Sequence[str]) -> dict[str, object]:
        value = _first_named_value(names, {})
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, Mapping):
                return dict(parsed)
        return {}

    def _int_contract_value(names: Sequence[str]) -> int | None:
        value = _first_named_value(names, None)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _bool_contract_value(names: Sequence[str], *, default: bool) -> bool:
        value = _first_named_value(names, None)
        return _coerce_bool(value, default=default)

    external_gap_todos = _list_contract_value(
        [
            "safe_share_external_launch_evidence_gap_todos",
            "external_launch_evidence_gap_todos",
        ]
    )
    derived_gap_gate_ids: list[str] = []
    derived_gap_statuses: dict[str, str] = {}
    derived_gap_template_paths: dict[str, str] = {}
    derived_gap_record_commands: dict[str, list[str]] = {}
    for todo in external_gap_todos:
        if not isinstance(todo, Mapping):
            continue
        gate_id = str(todo.get("gate_id") or "").strip()
        if not gate_id:
            continue
        if gate_id not in derived_gap_gate_ids:
            derived_gap_gate_ids.append(gate_id)
        status = str(todo.get("status") or "").strip()
        if status:
            derived_gap_statuses[gate_id] = status
        template_path = str(
            todo.get("operator_validation_evidence_template_path") or ""
        ).strip()
        if template_path:
            derived_gap_template_paths[gate_id] = template_path
        record_command_ids = todo.get("operator_validation_record_command_ids")
        if isinstance(record_command_ids, list):
            derived_gap_record_commands[gate_id] = [
                str(command_id)
                for command_id in record_command_ids
                if str(command_id)
            ]

    external_gap_gate_ids = _list_contract_value(
        [
            "safe_share_external_launch_evidence_gap_gate_ids",
            "external_launch_evidence_gap_gate_ids",
        ]
    ) or derived_gap_gate_ids
    if (
        not external_gap_gate_ids
        and not external_gap_todos
        and not _coerce_bool(_value("safe_share_checklist_gate_evidence_complete"))
    ):
        external_gap_gate_ids = ["browser_smoke"]
    external_gap_statuses = _mapping_contract_value(
        [
            "safe_share_external_launch_evidence_gap_statuses",
            "external_launch_evidence_gap_statuses",
        ]
    ) or derived_gap_statuses
    external_gap_todo_fields = _list_contract_value(
        [
            "safe_share_external_launch_evidence_gap_todo_fields",
            "external_launch_evidence_gap_todo_fields",
        ]
    )
    if external_gap_todos and not external_gap_todo_fields:
        external_gap_todo_fields = _safe_share_external_launch_evidence_gap_todo_fields()
    external_gap_template_paths = _mapping_contract_value(
        [
            "safe_share_external_launch_evidence_gap_template_paths_by_gate_id",
            "external_launch_evidence_gap_template_paths_by_gate_id",
        ]
    ) or derived_gap_template_paths
    external_gap_record_command_ids = _mapping_contract_value(
        [
            "safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id",
            "external_launch_evidence_gap_record_command_ids_by_gate_id",
        ]
    ) or derived_gap_record_commands
    external_gap_count = _int_contract_value(
        [
            "safe_share_external_launch_evidence_gap_count",
            "external_launch_evidence_gap_count",
        ]
    )
    if external_gap_count is None:
        external_gap_count = len(external_gap_gate_ids)
    external_gap_todo_count = _int_contract_value(
        [
            "safe_share_external_launch_evidence_gap_todo_count",
            "external_launch_evidence_gap_todo_count",
        ]
    )
    if external_gap_todo_count is None:
        external_gap_todo_count = len(external_gap_todos)
    external_gap_action_required = _bool_contract_value(
        [
            "safe_share_external_launch_evidence_gap_action_required",
            "external_launch_evidence_gap_action_required",
        ],
        default=bool(external_gap_gate_ids or external_gap_todos),
    )
    if external_gap_gate_ids or external_gap_todos:
        external_gap_action_required = True

    queue_first_entry_contract = (
        _value("queue_first_entry_contract", {})
        if isinstance(_value("queue_first_entry_contract", {}), Mapping)
        else {}
    )
    labeler_work_completion = (
        _value("labeler_work_completion", {})
        if isinstance(_value("labeler_work_completion", {}), Mapping)
        else {}
    )
    return {
        "schema": "palette.web_labeling_personalized_launch_readiness.v1",
        "fields": fields,
        "field_count": len(fields),
        "expected_user": str(_value("expected_user") or _value("user") or ""),
        "personalized_labeler_entry_url": str(
            _value("personalized_labeler_entry_url") or ""
        ),
        "preferred_labeler_entry_url": str(_value("preferred_labeler_entry_url") or ""),
        "queue_first_entry_contract_ready": _coerce_bool(
            queue_first_entry_contract.get("ready")
        ),
        "personalized_labeler_entry_url_matches_personal_dataset_queue": _coerce_bool(
            _first_named_value(
                ["personalized_labeler_entry_url_matches_personal_dataset_queue"],
                queue_first_entry_contract.get(
                    "personalized_labeler_entry_url_matches_personal_dataset_queue"
                ),
            )
        ),
        "labeler_start_ready": _coerce_bool(_value("labeler_start_ready")),
        "labeler_start_status": str(_value("labeler_start_status") or ""),
        "labeler_action": str(_value("labeler_action") or ""),
        "labeler_work_completion_status": str(
            _value("labeler_work_completion_status")
            or labeler_work_completion.get("status")
            or ""
        ),
        "safe_share_gate_id": str(_value("safe_share_gate_id") or ""),
        "safe_share_required_inspection_field": str(
            _value("safe_share_required_inspection_field") or ""
        ),
        "safe_share_required_inspection_value": _coerce_bool(
            _value("safe_share_required_inspection_value", True),
            default=True,
        ),
        "safe_share_checklist_gate_evidence_complete": _coerce_bool(
            _value("safe_share_checklist_gate_evidence_complete")
        ),
        "external_launch_evidence_gap_action_required": external_gap_action_required,
        "external_launch_evidence_gap_count": external_gap_count,
        "external_launch_evidence_gap_gate_ids": external_gap_gate_ids,
        "external_launch_evidence_gap_statuses": external_gap_statuses,
        "external_launch_evidence_gap_summary": str(
            _value("safe_share_external_launch_evidence_gap_summary") or ""
        ),
        "external_launch_evidence_gap_todos": external_gap_todos,
        "external_launch_evidence_gap_todo_count": external_gap_todo_count,
        "external_launch_evidence_gap_todo_fields": external_gap_todo_fields,
        "external_launch_evidence_gap_template_paths_by_gate_id": external_gap_template_paths,
        "external_launch_evidence_gap_record_command_ids_by_gate_id": (
            external_gap_record_command_ids
        ),
        "browser_label_write_target": str(browser_label_write_target),
        "browser_writes_csv_or_handoff_files": browser_writes_csv_or_handoff_files,
        "browser_has_direct_zarr_write_authority": browser_has_direct_zarr_write_authority,
    }


def _add_payload_contract_compact_fields(payload: dict[str, object]) -> dict[str, object]:
    user = str(payload.get("expected_user") or payload.get("user") or "")
    payload.update(
        _browser_mutation_target_contract_compact_fields(
            payload.get("browser_mutation_write_checklist")
            if isinstance(payload.get("browser_mutation_write_checklist"), Mapping)
            else None,
            user=user,
        )
    )
    payload.update(
        _direct_browser_start_contract_compact_fields(
            payload.get("dataset_queue_direct_start_policy")
            if isinstance(payload.get("dataset_queue_direct_start_policy"), Mapping)
            else None,
            user=user,
        )
    )
    payload.update(
        _labeler_route_authorization_runtime_checklist_compact_fields(
            payload.get("labeler_route_authorization_checklist")
            if isinstance(payload.get("labeler_route_authorization_checklist"), Mapping)
            else None,
            user=user,
        )
    )
    policy = (
        payload.get("single_owner_policy")
        if isinstance(payload.get("single_owner_policy"), Mapping)
        else _assignment_ownership_policy()
    )
    integrity = (
        payload.get("assignment_ownership_integrity")
        if isinstance(payload.get("assignment_ownership_integrity"), Mapping)
        else {}
    )
    store_single_owner_contract = (
        payload.get("single_owner_assignment_contract")
        if isinstance(payload.get("single_owner_assignment_contract"), Mapping)
        else None
    )
    if store_single_owner_contract is None:
        existing_contract = (
            payload.get("assignment_ownership_contract")
            if isinstance(payload.get("assignment_ownership_contract"), Mapping)
            else {}
        )
        nested_store_contract = existing_contract.get(
            "store_single_owner_assignment_contract"
        )
        if isinstance(nested_store_contract, Mapping) and nested_store_contract:
            store_single_owner_contract = nested_store_contract
    contract = _assignment_ownership_contract_policy(
        policy,
        integrity,
        store_single_owner_contract=store_single_owner_contract,
    )
    payload["assignment_ownership_contract"] = contract
    payload.update(_assignment_ownership_contract_fields(contract))
    payload["single_owner_policy_contract_met"] = bool(contract.get("ready")) and (
        int(integrity.get("duplicate_active_owner_count") or 0) == 0
    )
    payload["personalized_launch_readiness"] = _personalized_launch_readiness_summary(
        payload
    )
    nested_work = payload.get("work")
    if isinstance(nested_work, dict):
        nested_work["personalized_launch_readiness"] = (
            _personalized_launch_readiness_summary(nested_work, fallback=payload)
        )
    return payload

