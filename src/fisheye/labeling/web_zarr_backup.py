"""Read-only Zarr backup planning helpers for web labeling."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
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


def _safe_backup_slug_impl(value: object, *, fallback: str) -> str:
    text = str(value or "").strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@._-"
    slug = "".join(char if char in allowed else "_" for char in text).strip("._-")
    return slug or fallback


def _backup_target_destination_impl(backup_dir: Path, target: Mapping[str, object], index: int) -> Path:
    role = _safe_backup_slug(target.get("zarr_role"), fallback="zarr")
    basename = _safe_backup_slug(Path(str(target.get("zarr_path") or "")).name, fallback="target")
    digest = hashlib.sha256(str(target.get("zarr_path") or "").encode("utf-8")).hexdigest()[:12]
    return backup_dir / f"target-{index:04d}-{role}-{digest}-{basename}"


def _copy_backup_source_impl(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing backup destination: {destination}")
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination)


def _execute_zarr_backup_plan_impl(
    *,
    plan_path: Path,
    backup_dir: Path,
    operator: str,
    output: Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_missing: bool = False,
) -> dict[str, object]:
    loaded = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Zarr backup plan must be a JSON object.")
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    targets = loaded.get("zarr_targets") if isinstance(loaded.get("zarr_targets"), list) else []
    manifest_path = output or (backup_dir / "zarr-backup-execution-manifest.json")
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Zarr backup execution manifest: {manifest_path}")
    manifest_dir = backup_dir / "manifests"
    results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping) or not bool(target.get("backup_required")):
            continue
        source = Path(str(target.get("zarr_path") or ""))
        destination = _backup_target_destination(backup_dir, target, index)
        target_manifest_path = manifest_dir / f"target-{index:04d}-backup-manifest.json"
        source_exists = source.exists()
        status = "dry_run" if dry_run else "backed_up"
        error = ""
        if not source_exists:
            status = "missing_source"
            error = f"Backup source does not exist: {source}"
            errors.append({"target_index": index, "zarr_path": str(source), "error": error})
        elif not dry_run:
            _copy_backup_source(source, destination, overwrite=overwrite)
            target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            target_manifest = {
                **(
                    target.get("backup_manifest_template")
                    if isinstance(target.get("backup_manifest_template"), Mapping)
                    else {}
                ),
                "backup_schema": "palette.web_labeling_zarr_backup.v1",
                "backup_created_at_utc": generated_at_utc,
                "backup_created_by": operator,
                "source_zarr_path": str(source),
                "backup_destination": str(destination),
                "backup_execution_manifest": str(manifest_path),
                "target_index": index,
                "copy_method": "shutil.copytree" if source.is_dir() else "shutil.copy2",
                "restore_requires_paused_assignment": True,
                "labelers_do_not_receive": True,
            }
            target_manifest_path.write_text(
                json.dumps(target_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        restore_command = (
            "scripts/py -m fisheye.utils.labeling_work "
            f"--store {loaded.get('store_path') or '/path/to/labeling_work.sqlite'} "
            "restore-zarr-backup "
            f"--manifest {manifest_path} --target-index {index} "
            f"--operator {operator} --replace-current"
        )
        results.append(
            {
                "target_index": index,
                "status": status,
                "error": error,
                "source_exists": source_exists,
                "source_zarr_path": str(source),
                "zarr_role": str(target.get("zarr_role") or ""),
                "backup_required": bool(target.get("backup_required")),
                "backup_destination": str(destination),
                "backup_manifest_path": str(target_manifest_path),
                "recording_ids": list(target.get("recording_ids") or [])
                if isinstance(target.get("recording_ids"), list)
                else [],
                "task_ids": list(target.get("task_ids") or []) if isinstance(target.get("task_ids"), list) else [],
                "registry_paths": list(target.get("registry_paths") or [])
                if isinstance(target.get("registry_paths"), list)
                else [],
                "restore_requires_paused_assignment": True,
                "restore_command": restore_command,
            }
        )
    ok = not errors or bool(allow_missing)
    payload = {
        "ok": ok,
        "schema": "palette.web_labeling_zarr_backup_execution_manifest.v1",
        "generated_at_utc": generated_at_utc,
        "operator": operator,
        "dry_run": bool(dry_run),
        "source_plan_path": str(plan_path),
        "source_plan_schema": str(loaded.get("schema") or ""),
        "source_plan_generated_at_utc": str(loaded.get("generated_at_utc") or ""),
        "store_path": str(loaded.get("store_path") or ""),
        "backup_dir": str(backup_dir),
        "policy": _zarr_backup_policy(),
        "restore_policy": {
            "operator_only": True,
            "requires_assignment_store_check": True,
            "default_blocks_active_recording_assignments": True,
            "replace_current_moves_existing_path_aside": True,
            "labelers_do_not_receive_backup_paths": True,
        },
        "counts": {
            "targets_examined": len([target for target in targets if isinstance(target, Mapping)]),
            "backup_required_targets": len(results),
            "backed_up_targets": sum(1 for result in results if result["status"] == "backed_up"),
            "dry_run_targets": sum(1 for result in results if result["status"] == "dry_run"),
            "missing_source_targets": sum(1 for result in results if result["status"] == "missing_source"),
        },
        "errors": errors,
        "targets": results,
    }
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _restore_assignment_conflicts_impl(
    store: LabelingStore,
    recording_ids: Sequence[object],
) -> list[dict[str, object]]:
    conflicts: list[dict[str, object]] = []
    for recording_id in sorted({str(item).strip() for item in recording_ids if str(item).strip()}):
        assignment = store.get_assignment(recording_id)
        if assignment is None:
            continue
        if str(assignment.get("status") or "") == "active":
            conflicts.append(
                {
                    "recording_id": recording_id,
                    "assignee_user": assignment.get("assignee_user"),
                    "status": assignment.get("status"),
                    "details": "Recording has an active assignment; pause or unassign before restoring a Zarr backup.",
                }
            )
    return conflicts


def _restore_backup_target_impl(
    *,
    source_backup: Path,
    restore_path: Path,
    replace_current: bool,
    generated_at_utc: str,
) -> dict[str, object]:
    moved_existing_to = ""
    if restore_path.exists():
        if not replace_current:
            raise FileExistsError(f"Refusing to replace current Zarr path without --replace-current: {restore_path}")
        stamp = re.sub(r"[^0-9A-Za-z]+", "", generated_at_utc)[:20] or "restore"
        moved_path = restore_path.with_name(f"{restore_path.name}.pre-restore-{stamp}")
        if moved_path.exists():
            suffix = uuid.uuid4().hex[:8]
            moved_path = restore_path.with_name(f"{restore_path.name}.pre-restore-{stamp}-{suffix}")
        shutil.move(str(restore_path), str(moved_path))
        moved_existing_to = str(moved_path)
    restore_path.parent.mkdir(parents=True, exist_ok=True)
    if source_backup.is_dir():
        shutil.copytree(source_backup, restore_path, symlinks=True)
        copy_method = "shutil.copytree"
    else:
        shutil.copy2(source_backup, restore_path)
        copy_method = "shutil.copy2"
    return {
        "restored": True,
        "restore_path": str(restore_path),
        "moved_existing_to": moved_existing_to,
        "copy_method": copy_method,
    }


def _restore_zarr_backup_manifest_impl(
    *,
    store: LabelingStore,
    manifest_path: Path,
    operator: str,
    target_indexes: Sequence[int],
    restore_all: bool = False,
    replace_current: bool = False,
    allow_active_assignment: bool = False,
) -> dict[str, object]:
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Zarr backup execution manifest must be a JSON object.")
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    targets = loaded.get("targets") if isinstance(loaded.get("targets"), list) else []
    requested = set(int(index) for index in target_indexes)
    results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for row in targets:
        if not isinstance(row, Mapping):
            continue
        index = int(row.get("target_index") or 0)
        if not restore_all and index not in requested:
            continue
        backup_destination = Path(str(row.get("backup_destination") or ""))
        restore_path = Path(str(row.get("source_zarr_path") or ""))
        conflicts = _restore_assignment_conflicts(
            store,
            row.get("recording_ids") if isinstance(row.get("recording_ids"), list) else [],
        )
        if conflicts and not allow_active_assignment:
            errors.append(
                {
                    "target_index": index,
                    "zarr_path": str(restore_path),
                    "error": "active_assignment_blocks_restore",
                    "assignment_conflicts": conflicts,
                }
            )
            results.append(
                {
                    "target_index": index,
                    "status": "blocked_active_assignment",
                    "source_backup": str(backup_destination),
                    "restore_path": str(restore_path),
                    "assignment_conflicts": conflicts,
                }
            )
            continue
        if not backup_destination.exists():
            error = f"Backup destination does not exist: {backup_destination}"
            errors.append({"target_index": index, "zarr_path": str(restore_path), "error": error})
            results.append(
                {
                    "target_index": index,
                    "status": "missing_backup",
                    "source_backup": str(backup_destination),
                    "restore_path": str(restore_path),
                    "error": error,
                }
            )
            continue
        restore_result = _restore_backup_target(
            source_backup=backup_destination,
            restore_path=restore_path,
            replace_current=replace_current,
            generated_at_utc=generated_at_utc,
        )
        results.append(
            {
                "target_index": index,
                "status": "restored",
                "source_backup": str(backup_destination),
                "restore_path": str(restore_path),
                "assignment_conflicts": conflicts,
                **restore_result,
            }
        )
    ok = bool(results) and not errors
    return {
        "ok": ok,
        "schema": "palette.web_labeling_zarr_backup_restore_report.v1",
        "generated_at_utc": generated_at_utc,
        "operator": operator,
        "source_manifest_path": str(manifest_path),
        "source_manifest_schema": str(loaded.get("schema") or ""),
        "replace_current": bool(replace_current),
        "allow_active_assignment": bool(allow_active_assignment),
        "counts": {
            "targets_requested": len(targets) if restore_all else len(requested),
            "targets_restored": sum(1 for result in results if result.get("status") == "restored"),
            "blocked_active_assignment_targets": sum(
                1 for result in results if result.get("status") == "blocked_active_assignment"
            ),
            "missing_backup_targets": sum(1 for result in results if result.get("status") == "missing_backup"),
        },
        "errors": errors,
        "targets": results,
    }


def _record_zarr_backup_evidence_impl(
    *,
    evidence_path: Path,
    execution_manifest_path: Path,
    operator: str,
    restore_test_result: str,
    target_indexes: Sequence[int],
    record_all: bool = False,
    output: Path | None = None,
    overwrite: bool = False,
    notes: str | None = None,
) -> dict[str, object]:
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence_payload, dict):
        raise ValueError("Zarr backup evidence template must be a JSON object.")
    execution_manifest = json.loads(execution_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(execution_manifest, dict):
        raise ValueError("Zarr backup execution manifest must be a JSON object.")
    destination_path = output or evidence_path
    if output is not None and destination_path.exists() and destination_path != evidence_path and not overwrite:
        raise FileExistsError(f"Output file already exists: {destination_path}")
    targets = evidence_payload.get("targets") if isinstance(evidence_payload.get("targets"), list) else []
    execution_targets = (
        execution_manifest.get("targets")
        if isinstance(execution_manifest.get("targets"), list)
        else []
    )
    evidence_by_index = {
        int(row.get("target_index") or 0): row
        for row in targets
        if isinstance(row, dict)
    }
    requested = set(int(index) for index in target_indexes)
    updated_indexes: list[int] = []
    errors: list[dict[str, object]] = []
    now = datetime.now(timezone.utc).isoformat()
    for execution_target in execution_targets:
        if not isinstance(execution_target, Mapping):
            continue
        target_index = int(execution_target.get("target_index") or 0)
        if not record_all and target_index not in requested:
            continue
        evidence_row = evidence_by_index.get(target_index)
        if evidence_row is None:
            errors.append(
                {
                    "target_index": target_index,
                    "error": "evidence_target_missing",
                    "details": "Execution manifest target has no matching row in the backup evidence template.",
                }
            )
            continue
        if str(execution_target.get("status") or "") != "backed_up":
            errors.append(
                {
                    "target_index": target_index,
                    "error": "target_not_backed_up",
                    "status": str(execution_target.get("status") or ""),
                }
            )
            continue
        evidence_row.update(
            {
                "status": "operator_approved",
                "backup_execution_manifest_path": str(execution_manifest_path),
                "backup_manifest_path": str(execution_target.get("backup_manifest_path") or ""),
                "backup_destination": str(execution_target.get("backup_destination") or ""),
                "backup_created_at_utc": str(execution_manifest.get("generated_at_utc") or now),
                "backup_verified_at_utc": now,
                "restore_test_result": str(restore_test_result or "").strip(),
                "operator": operator,
                "operator_approved_at_utc": now,
            }
        )
        if notes is not None:
            evidence_row["notes"] = str(notes)
        updated_indexes.append(target_index)
    approved_count = sum(
        1
        for row in targets
        if isinstance(row, Mapping) and str(row.get("status") or "") == "operator_approved"
    )
    pending_count = sum(
        1
        for row in targets
        if isinstance(row, Mapping) and str(row.get("status") or "") != "operator_approved"
    )
    evidence_payload["updated_at_utc"] = now
    evidence_payload["updated_by"] = operator
    evidence_payload["counts"] = {
        **(
            evidence_payload.get("counts")
            if isinstance(evidence_payload.get("counts"), Mapping)
            else {}
        ),
        "backup_required_targets": len([row for row in targets if isinstance(row, Mapping)]),
        "pending_operator_confirmation": pending_count,
        "operator_approved": approved_count,
    }
    ok = bool(updated_indexes) and not errors
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": ok,
        "schema": "palette.web_labeling_zarr_backup_evidence_update_report.v1",
        "updated_at_utc": now,
        "operator": operator,
        "evidence_path": str(evidence_path),
        "execution_manifest_path": str(execution_manifest_path),
        "output_path": str(destination_path),
        "target_indexes": updated_indexes,
        "error_count": len(errors),
        "errors": errors,
        "counts": evidence_payload["counts"],
    }


# Preserve original helper names inside this module so moved helpers can call each other.
_safe_backup_slug = _safe_backup_slug_impl
_backup_target_destination = _backup_target_destination_impl
_copy_backup_source = _copy_backup_source_impl
_execute_zarr_backup_plan = _execute_zarr_backup_plan_impl
_restore_assignment_conflicts = _restore_assignment_conflicts_impl
_restore_backup_target = _restore_backup_target_impl
_restore_zarr_backup_manifest = _restore_zarr_backup_manifest_impl
_record_zarr_backup_evidence = _record_zarr_backup_evidence_impl
