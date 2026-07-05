"""Read-only Zarr backup planning helpers for web labeling."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from .assignment_store import LabelingStore


def configure_zarr_backup_plan_dependencies(dependencies: Mapping[str, object]) -> None:
    globals().update(dependencies)

ZARR_BACKUP_PATH_KEYS = (
    "zarr_path",
    "training_zarr",
    "training_zarr_path",
    "promote_training_zarr",
    "promote_training_zarr_path",
    "analysis_zarr",
    "analysis_zarr_path",
)


def _zarr_backup_contract_policy(
    policy: Mapping[str, object],
    counts: Mapping[str, object],
    files: Mapping[str, object],
) -> dict[str, object]:
    read_only_plan = bool(policy.get("read_only_plan"))
    operator_only = bool(policy.get("operator_only"))
    copy_before_labeling = bool(policy.get("copy_before_labeling"))
    mutable_zarr_backup_required_before_invite = bool(policy.get("mutable_zarr_backup_required_before_invite"))
    labelers_do_not_edit_zarrs_directly = bool(policy.get("labelers_do_not_edit_zarrs_directly"))
    labelers_do_not_receive_backup_paths = bool(policy.get("labelers_do_not_receive_backup_paths"))
    pause_or_unassign_recording_before_restore = bool(policy.get("pause_or_unassign_recording_before_restore"))
    rollback_owner = str(policy.get("rollback_owner") or "")
    zarr_targets_by_role = (
        counts.get("zarr_backup_targets_by_role")
        if isinstance(counts.get("zarr_backup_targets_by_role"), Mapping)
        else {}
    )
    backup_required_targets_by_role = (
        counts.get("zarr_backup_required_targets_by_role")
        if isinstance(counts.get("zarr_backup_required_targets_by_role"), Mapping)
        else {}
    )
    ready = (
        read_only_plan
        and operator_only
        and copy_before_labeling
        and mutable_zarr_backup_required_before_invite
        and labelers_do_not_edit_zarrs_directly
        and labelers_do_not_receive_backup_paths
        and pause_or_unassign_recording_before_restore
        and rollback_owner == "operator"
    )
    return {
        "schema": "palette.web_labeling_zarr_backup_contract.v1",
        "ready": ready,
        "read_only_plan": read_only_plan,
        "operator_only": operator_only,
        "copy_before_labeling": copy_before_labeling,
        "mutable_zarr_backup_required_before_invite": mutable_zarr_backup_required_before_invite,
        "labelers_do_not_edit_zarrs_directly": labelers_do_not_edit_zarrs_directly,
        "labelers_do_not_receive_backup_paths": labelers_do_not_receive_backup_paths,
        "pause_or_unassign_recording_before_restore": pause_or_unassign_recording_before_restore,
        "rollback_owner": rollback_owner,
        "validation_gate": str(policy.get("validation_gate") or ""),
        "zarr_backup_plan": str(files.get("zarr_backup_plan") or ""),
        "zarr_targets": int(counts.get("zarr_backup_targets") or 0),
        "backup_required_targets": int(counts.get("zarr_backup_required_targets") or 0),
        "tasks_missing_zarr_path": int(counts.get("zarr_backup_missing_path_tasks") or 0),
        "zarr_targets_by_role": dict(zarr_targets_by_role),
        "backup_required_targets_by_role": dict(backup_required_targets_by_role),
    }


def _iter_zarr_path_values(value: object) -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in ZARR_BACKUP_PATH_KEYS and isinstance(child, str) and child.strip():
                yield key_text, child.strip()
            yield from _iter_zarr_path_values(child)
        return
    if isinstance(value, list):
        for child in value:
            yield from _iter_zarr_path_values(child)


def _zarr_backup_role_for_key(key: str, task: Mapping[str, object]) -> str:
    workflow_kind = str(task.get("workflow_kind") or "")
    zarr_use = str(task.get("zarr_use") or "")
    if "training" in key:
        return "training"
    if "analysis" in key:
        return "analysis"
    if zarr_use:
        return zarr_use
    if workflow_kind == "detect_analysis":
        return "analysis"
    return "unspecified"


def _zarr_backup_plan_impl(
    *,
    store: LabelingStore,
    store_path: Path,
    recording_id: str | None = None,
    user: str | None = None,
    include_completed: bool = True,
    include_inactive: bool = False,
) -> dict[str, object]:
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    tasks = store.list_tasks(recording_id=recording_id, assignee_user=user, include_completed=include_completed)
    if not include_inactive:
        tasks = [task for task in tasks if str(task.get("assignment_status") or "") == "active"]
    grouped: dict[str, dict[str, object]] = {}
    missing_path_tasks: list[dict[str, object]] = []
    for task in tasks:
        scope = task.get("scope") if isinstance(task.get("scope"), Mapping) else {}
        paths = list(_iter_zarr_path_values(scope))
        if not paths:
            missing_path_tasks.append(
                {
                    "task_id": task.get("task_id"),
                    "recording_id": task.get("recording_id"),
                    "assignee_user": task.get("assignee_user"),
                    "workflow_kind": task.get("workflow_kind"),
                    "zarr_use": task.get("zarr_use"),
                    "details": "Task scope does not expose a zarr path for backup planning.",
                }
            )
            continue
        registry_paths = sorted(
            {
                str(value).strip()
                for key, value in scope.items()
                if str(key) == "registry_path" and isinstance(value, str) and value.strip()
            }
        )
        for key, zarr_path in paths:
            role = _zarr_backup_role_for_key(key, task)
            group_key = json.dumps([zarr_path, role], separators=(",", ":"), sort_keys=True)
            target = grouped.setdefault(
                group_key,
                {
                    "zarr_path": zarr_path,
                    "zarr_role": role,
                    "backup_required": role in {"training", "analysis", "unspecified"},
                    "copy_before_labeling": True,
                    "restore_requires_paused_assignment": True,
                    "recording_ids": set(),
                    "assignee_users": set(),
                    "task_ids": set(),
                    "workflow_kinds": set(),
                    "dataset_ids": set(),
                    "zarr_uses": set(),
                    "stage_groups": set(),
                    "run_names": set(),
                    "component_names": set(),
                    "registry_paths": set(),
                    "source_scope_keys": set(),
                },
            )
            for target_key, task_key in (
                ("recording_ids", "recording_id"),
                ("assignee_users", "assignee_user"),
                ("task_ids", "task_id"),
                ("workflow_kinds", "workflow_kind"),
                ("dataset_ids", "dataset_id"),
                ("zarr_uses", "zarr_use"),
                ("stage_groups", "stage_group"),
                ("run_names", "run_name"),
                ("component_names", "component_name"),
            ):
                value = str(task.get(task_key) or "").strip()
                if value:
                    target[target_key].add(value)  # type: ignore[union-attr]
            for registry_path in registry_paths:
                target["registry_paths"].add(registry_path)  # type: ignore[union-attr]
            target["source_scope_keys"].add(key)  # type: ignore[union-attr]
    zarr_targets: list[dict[str, object]] = []
    for target in grouped.values():
        normalized = dict(target)
        for key in (
            "recording_ids",
            "assignee_users",
            "task_ids",
            "workflow_kinds",
            "dataset_ids",
            "zarr_uses",
            "stage_groups",
            "run_names",
            "component_names",
            "registry_paths",
            "source_scope_keys",
        ):
            values = normalized.get(key, set())
            normalized[key] = sorted(str(value) for value in values) if isinstance(values, set) else []
        normalized["backup_manifest_template"] = {
            "backup_schema": "palette.web_labeling_zarr_backup.v1",
            "created_utc": generated_at_utc,
            "reason": "before web labeling batch",
            "zarr_path": normalized["zarr_path"],
            "zarr_role": normalized["zarr_role"],
            "recording_ids": normalized["recording_ids"],
            "dataset_ids": normalized["dataset_ids"],
            "task_ids": normalized["task_ids"],
            "registry_paths": normalized["registry_paths"],
        }
        zarr_targets.append(normalized)
    zarr_targets.sort(key=lambda row: (str(row.get("zarr_role") or ""), str(row.get("zarr_path") or "")))
    zarr_targets_by_role: dict[str, int] = {}
    backup_required_targets_by_role: dict[str, int] = {}
    for target in zarr_targets:
        role = str(target.get("zarr_role") or "unspecified")
        zarr_targets_by_role[role] = zarr_targets_by_role.get(role, 0) + 1
        if bool(target.get("backup_required")):
            backup_required_targets_by_role[role] = backup_required_targets_by_role.get(role, 0) + 1
    warnings = [{"code": "task_missing_zarr_path_for_backup_plan", **task} for task in missing_path_tasks]
    return {
        "ok": True,
        "schema": "palette.web_labeling_zarr_backup_plan.v1",
        "generated_at_utc": generated_at_utc,
        "store_path": str(store_path),
        "filters": {
            "recording_id": recording_id,
            "user": user,
            "include_completed": bool(include_completed),
            "include_inactive": bool(include_inactive),
        },
        "policy": _zarr_backup_policy(),
        "counts": {
            "tasks_examined": len(tasks),
            "tasks_missing_zarr_path": len(missing_path_tasks),
            "zarr_targets": len(zarr_targets),
            "backup_required_targets": sum(1 for target in zarr_targets if bool(target.get("backup_required"))),
            "unique_zarr_paths": len({str(target.get("zarr_path") or "") for target in zarr_targets}),
            "zarr_targets_by_role": dict(sorted(zarr_targets_by_role.items())),
            "backup_required_targets_by_role": dict(sorted(backup_required_targets_by_role.items())),
        },
        "warning_count": len(warnings),
        "warning_codes": sorted({str(warning.get("code") or "") for warning in warnings}),
        "warnings": warnings,
        "zarr_targets": zarr_targets,
    }


# Preserve original helper name inside this module for local recursive/cross calls.
_zarr_backup_plan = _zarr_backup_plan_impl
