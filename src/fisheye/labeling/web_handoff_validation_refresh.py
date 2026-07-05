"""Validation-checklist refresh helpers for web-labeling handoffs."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def configure_handoff_validation_refresh_dependencies(dependencies: Mapping[str, object]) -> None:
    globals().update(dependencies)

def _refresh_user_handoff_visible_files_impl(
    *,
    manifest_path: Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
    refreshed: list[str] = []
    skipped: list[dict[str, str]] = []
    user = str(manifest.get("user") or "").strip()
    if not user:
        return {
            "manifest": str(manifest_path),
            "refreshed_files": refreshed,
            "skipped_files": [{"file": str(manifest_path), "reason": "missing_user"}],
        }

    quickstart_path = _manifest_relative_path(
        manifest_path,
        files.get("quickstart") if isinstance(files, Mapping) else "",
        "labeler-quickstart.txt",
    )
    message_path = _manifest_relative_path(
        manifest_path,
        files.get("message") if isinstance(files, Mapping) else "",
        "message.txt",
    )
    html_path = _manifest_relative_path(
        manifest_path,
        files.get("html_index") if isinstance(files, Mapping) else "",
        "index.html",
    )
    work_path = _manifest_relative_path(
        manifest_path,
        files.get("work_summary") if isinstance(files, Mapping) else "",
        "work-summary.json",
    )
    links_path = _manifest_relative_path(
        manifest_path,
        files.get("signed_links") if isinstance(files, Mapping) else "",
        "signed-links.jsonl",
    )
    dataset_queue_path = _manifest_relative_path(
        manifest_path,
        files.get("dataset_queue") if isinstance(files, Mapping) else "",
        "dataset-queue.json",
    )

    try:
        _write_user_handoff_quickstart(user=user, manifest=manifest, output_path=quickstart_path)
        refreshed.append(str(quickstart_path))
    except OSError as exc:
        skipped.append({"file": str(quickstart_path), "reason": str(exc)})
    try:
        _write_user_handoff_message(user=user, manifest=manifest, output_path=message_path)
        refreshed.append(str(message_path))
    except OSError as exc:
        skipped.append({"file": str(message_path), "reason": str(exc)})

    if work_path.is_file() and links_path.is_file():
        try:
            work_payload = json.loads(work_path.read_text(encoding="utf-8"))
            links = _read_jsonl_objects(links_path)
            if isinstance(work_payload, dict):
                work = (
                    work_payload.get("work")
                    if isinstance(work_payload.get("work"), dict)
                    else work_payload
                )
                route_policy = (
                    manifest.get("labeler_route_authorization_policy")
                    if isinstance(manifest.get("labeler_route_authorization_policy"), Mapping)
                    else _labeler_route_authorization_policy()
                )
                route_checklist = (
                    manifest.get("labeler_route_authorization_checklist")
                    if isinstance(manifest.get("labeler_route_authorization_checklist"), Mapping)
                    else _labeler_route_authorization_runtime_checklist(
                        policy=route_policy,
                        user=str(manifest.get("user") or ""),
                        expected_user=str(manifest.get("expected_user") or manifest.get("user") or ""),
                        known_user_status=(
                            manifest.get("known_user_status")
                            if isinstance(manifest.get("known_user_status"), Mapping)
                            else {}
                        ),
                        assignment_ownership_contract=(
                            manifest.get("assignment_ownership_contract")
                            if isinstance(
                                manifest.get("assignment_ownership_contract"),
                                Mapping,
                            )
                            else None
                        ),
                    )
                )
                browser_mutation_write_policy = (
                    manifest.get("browser_mutation_write_policy")
                    if isinstance(manifest.get("browser_mutation_write_policy"), Mapping)
                    else _browser_mutation_write_policy()
                )
                browser_mutation_write_checklist = (
                    manifest.get("browser_mutation_write_checklist")
                    if isinstance(manifest.get("browser_mutation_write_checklist"), Mapping)
                    else _browser_mutation_write_runtime_checklist(browser_mutation_write_policy)
                )
                reassignment_session_safety_source = (
                    manifest.get("reassignment_session_safety")
                    if isinstance(manifest.get("reassignment_session_safety"), Mapping)
                    else (
                        manifest.get("store_consistency", {}).get("reassignment_session_safety")
                        if isinstance(manifest.get("store_consistency"), Mapping)
                        and isinstance(
                            manifest.get("store_consistency", {}).get("reassignment_session_safety"),
                            Mapping,
                        )
                        else {}
                    )
                )
                public_reassignment_session_safety = _public_reassignment_session_safety_fields(
                    reassignment_session_safety_source,
                    recording_ids=_work_recording_ids(work) if isinstance(work, Mapping) else None,
                )
                dataset_queue_direct_start_policy = (
                    manifest.get("dataset_queue_direct_start_policy")
                    if isinstance(manifest.get("dataset_queue_direct_start_policy"), Mapping)
                    else _dataset_queue_direct_start_policy()
                )
                runtime_gate_cli_policy = (
                    manifest.get("runtime_operator_validation_gate_cli_policy")
                    if isinstance(
                        manifest.get("runtime_operator_validation_gate_cli_policy"),
                        Mapping,
                    )
                    else _runtime_operator_validation_gate_cli_policy()
                )
                work_payload["labeler_route_authorization_policy"] = dict(route_policy)
                work_payload["labeler_route_authorization_checklist"] = dict(route_checklist)
                work_payload["browser_mutation_write_policy"] = dict(browser_mutation_write_policy)
                work_payload["browser_mutation_write_checklist"] = dict(browser_mutation_write_checklist)
                work_payload["dataset_queue_direct_start_policy"] = dict(dataset_queue_direct_start_policy)
                work_payload["runtime_operator_validation_gate_cli_policy"] = dict(
                    runtime_gate_cli_policy
                )
                if isinstance(manifest.get("single_owner_policy"), Mapping):
                    work_payload["single_owner_policy"] = dict(manifest["single_owner_policy"])
                if isinstance(manifest.get("single_owner_assignment_contract"), Mapping):
                    work_payload["single_owner_assignment_contract"] = dict(
                        manifest["single_owner_assignment_contract"]
                    )
                if isinstance(manifest.get("assignment_ownership_integrity"), Mapping):
                    work_payload["assignment_ownership_integrity"] = dict(
                        manifest["assignment_ownership_integrity"]
                    )
                if isinstance(manifest.get("assignment_ownership_contract"), Mapping):
                    work_payload["assignment_ownership_contract"] = dict(
                        manifest["assignment_ownership_contract"]
                    )
                work_payload["reassignment_session_safety"] = dict(public_reassignment_session_safety)
                work_payload.update(
                    _reassignment_session_safety_flat_fields(public_reassignment_session_safety)
                )
                _add_payload_contract_compact_fields(work_payload)
                work_payload.update(_handoff_dataset_queue_start_fields(work_payload))
                work_payload["ready_to_send"] = bool(manifest.get("ready_to_send"))
                work_payload["sendability_reasons"] = manifest.get("sendability_reasons", [])
                work_payload["sendability_actions"] = manifest.get("sendability_actions", [])
                work_payload.update(_operator_validation_public_fields(manifest))
                work_payload.update(_operator_validation_gate_flat_fields(manifest))
                work_payload["operator_validation_command_templates"] = manifest.get(
                    "operator_validation_command_templates",
                    _operator_validation_command_templates(
                        manifest.get("operator_validation_required_missing_evidence_gate_ids")
                        if isinstance(
                            manifest.get("operator_validation_required_missing_evidence_gate_ids"),
                            list,
                        )
                        else None
                    ),
                )
                work_payload["operator_validation_visibility_policy"] = (
                    _operator_validation_visibility_policy()
                )
                _add_payload_contract_compact_fields(work_payload)
                work_payload.update(_handoff_dataset_queue_start_fields(work_payload))
                work_payload["links_expire_at_utc"] = manifest.get("links_expire_at_utc")
                if isinstance(work, dict):
                    work["labeler_route_authorization_policy"] = dict(route_policy)
                    work["labeler_route_authorization_checklist"] = dict(route_checklist)
                    work["browser_mutation_write_policy"] = dict(browser_mutation_write_policy)
                    work["browser_mutation_write_checklist"] = dict(browser_mutation_write_checklist)
                    work["dataset_queue_direct_start_policy"] = dict(dataset_queue_direct_start_policy)
                    work["runtime_operator_validation_gate_cli_policy"] = dict(
                        runtime_gate_cli_policy
                    )
                    if isinstance(manifest.get("single_owner_policy"), Mapping):
                        work["single_owner_policy"] = dict(manifest["single_owner_policy"])
                    if isinstance(manifest.get("single_owner_assignment_contract"), Mapping):
                        work["single_owner_assignment_contract"] = dict(
                            manifest["single_owner_assignment_contract"]
                        )
                    if isinstance(manifest.get("assignment_ownership_integrity"), Mapping):
                        work["assignment_ownership_integrity"] = dict(
                            manifest["assignment_ownership_integrity"]
                        )
                    if isinstance(manifest.get("assignment_ownership_contract"), Mapping):
                        work["assignment_ownership_contract"] = dict(
                            manifest["assignment_ownership_contract"]
                        )
                    _add_payload_contract_compact_fields(work)
                    work.update(_operator_validation_public_fields(manifest))
                    work.update(_operator_validation_gate_flat_fields(manifest))
                    work["operator_validation_command_templates"] = manifest.get(
                        "operator_validation_command_templates",
                        _operator_validation_command_templates(
                            manifest.get("operator_validation_required_missing_evidence_gate_ids")
                            if isinstance(
                                manifest.get("operator_validation_required_missing_evidence_gate_ids"),
                                list,
                            )
                            else None
                        ),
                    )
                    work["operator_validation_visibility_policy"] = (
                        _operator_validation_visibility_policy()
                    )
                    _add_payload_contract_compact_fields(work)
                    _add_work_summary_fields(
                        work,
                        reassignment_session_safety=public_reassignment_session_safety,
                    )
                    work.update(_handoff_dataset_queue_start_fields(work))
                work_path.write_text(
                    json.dumps(work_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                refreshed.append(str(work_path))
                if isinstance(work, dict):
                    dataset_queue_payload = _handoff_status_from_manifest(
                        manifest,
                        datetime.now(timezone.utc),
                    )
                    dataset_queue_payload.update(
                        {
                            "schema": "palette.web_labeling_dataset_queue.v1",
                            "store_path": str(manifest.get("store_path") or work_payload.get("store_path") or ""),
                            "include_completed": bool(
                                manifest.get(
                                    "include_completed",
                                    work_payload.get("include_completed", False),
                                )
                            ),
                            "base_url": manifest.get("base_url"),
                            "expected_user": str(manifest.get("expected_user") or manifest.get("user") or ""),
                            "labeler_landing_page_path": str(
                                manifest.get("labeler_landing_page_path") or "/"
                            ),
                            "dashboard_path": str(manifest.get("dashboard_path") or DASHBOARD_PATH),
                            "dataset_queue_page_path": str(
                                manifest.get("dataset_queue_page_path") or DATASET_QUEUE_PATH
                            ),
                            "dataset_queue_url": str(manifest.get("dataset_queue_url") or ""),
                            "empty_state": work.get("empty_state", {}),
                            "progress_summary": work.get("progress_summary", {}),
                            "dataset_queue_summary": work.get("dataset_queue_summary", {}),
                            "direct_browser_start_contract_summary": work.get(
                                "direct_browser_start_contract_summary", {}
                            ),
                            "dataset_queue_state": work.get("dataset_queue_state", {}),
                            "reassignment_session_safety": dict(public_reassignment_session_safety),
                            **_reassignment_session_safety_flat_fields(
                                public_reassignment_session_safety
                            ),
                            "labeler_start_ready": bool(work.get("labeler_start_ready")),
                            "labeler_start_status": str(work.get("labeler_start_status") or ""),
                            "labeler_action": str(work.get("labeler_action") or ""),
                            "labeler_start_message": str(work.get("labeler_start_message") or ""),
                            "labeler_start_operator_action": str(
                                work.get("labeler_start_operator_action") or ""
                            ),
                            "dataset_queue": work.get("dataset_queue", []),
                            "datasets": work.get("dataset_queue", []),
                            "assignment_snapshot": manifest.get("assignment_snapshot", {}),
                            "known_user_status": manifest.get("known_user_status", {}),
                            "operator_authorization_policy": manifest.get(
                                "operator_authorization_policy",
                                _operator_authorization_policy(),
                            ),
                            "operator_validation_visibility_policy": (
                                _operator_validation_visibility_policy()
                            ),
                            "operator_validation_command_templates": manifest.get(
                                "operator_validation_command_templates",
                                _operator_validation_command_templates(
                                    manifest.get("operator_validation_required_missing_evidence_gate_ids")
                                    if isinstance(
                                        manifest.get("operator_validation_required_missing_evidence_gate_ids"),
                                        list,
                                    )
                                    else None
                                ),
                            ),
                            **_operator_validation_public_fields(manifest),
                            **_operator_validation_gate_flat_fields(manifest),
                        }
                    )
                    if isinstance(manifest.get("single_owner_policy"), Mapping):
                        dataset_queue_payload["single_owner_policy"] = dict(
                            manifest["single_owner_policy"]
                        )
                    if isinstance(manifest.get("single_owner_assignment_contract"), Mapping):
                        dataset_queue_payload["single_owner_assignment_contract"] = dict(
                            manifest["single_owner_assignment_contract"]
                        )
                    if isinstance(manifest.get("assignment_ownership_integrity"), Mapping):
                        dataset_queue_payload["assignment_ownership_integrity"] = dict(
                            manifest["assignment_ownership_integrity"]
                        )
                    if isinstance(manifest.get("assignment_ownership_contract"), Mapping):
                        dataset_queue_payload["assignment_ownership_contract"] = dict(
                            manifest["assignment_ownership_contract"]
                        )
                    dataset_queue_payload["labeler_route_authorization_policy"] = dict(
                        route_policy
                    )
                    dataset_queue_payload["labeler_route_authorization_checklist"] = dict(
                        route_checklist
                    )
                    dataset_queue_payload["browser_mutation_write_policy"] = dict(
                        browser_mutation_write_policy
                    )
                    dataset_queue_payload["browser_mutation_write_checklist"] = dict(
                        browser_mutation_write_checklist
                    )
                    dataset_queue_payload["dataset_queue_direct_start_policy"] = dict(
                        dataset_queue_direct_start_policy
                    )
                    dataset_queue_payload["runtime_operator_validation_gate_cli_policy"] = dict(
                        runtime_gate_cli_policy
                    )
                    _add_payload_contract_compact_fields(dataset_queue_payload)
                    dataset_queue_payload.update(
                        _handoff_dataset_queue_start_fields(dataset_queue_payload)
                    )
                    dataset_queue_path.write_text(
                        json.dumps(dataset_queue_payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    refreshed.append(str(dataset_queue_path))
                _write_user_handoff_html_index(
                    user=user,
                    work=work,
                    links=links,
                    manifest=manifest,
                    output_path=html_path,
                )
                refreshed.append(str(html_path))
            else:
                skipped.append({"file": str(html_path), "reason": "work_summary_not_object"})
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append({"file": str(html_path), "reason": str(exc)})
    else:
        missing = [
            str(path)
            for path in (work_path, links_path)
            if not path.is_file()
        ]
        skipped.append(
            {
                "file": str(html_path),
                "reason": "missing_html_inputs",
                "details": ", ".join(missing),
            }
        )

    return {
        "manifest": str(manifest_path),
        "refreshed_files": refreshed,
        "skipped_files": skipped,
    }


def _handoff_manifest_paths_for_validation_checklist_impl(path: Path) -> list[Path]:
    root = path.parent
    candidates: list[Path] = []
    for candidate in (root / "manifest.json",):
        if candidate.is_file():
            candidates.append(candidate)
    for parent in (root, root / "handoffs"):
        if not parent.is_dir():
            continue
        for candidate in sorted(parent.glob("*/manifest.json")):
            if candidate.is_file():
                candidates.append(candidate)
    unique: dict[str, Path] = {}
    for candidate in candidates:
        unique[str(candidate.resolve())] = candidate
    return list(unique.values())


def _refresh_handoff_manifests_from_validation_checklist_impl(
    *,
    checklist_path: Path,
    checklist_payload: Mapping[str, object],
    enabled: bool,
    store: LabelingStore | None = None,
) -> dict[str, object]:
    if not enabled:
        return {
            "enabled": False,
            "reason": "disabled",
            "manifest_count": 0,
            "refreshed_manifest_count": 0,
            "refreshed_manifests": [],
            "refreshed_file_count": 0,
            "refreshed_files": [],
            "refreshed_visible_json_file_count": 0,
            "refreshed_visible_json_files": [],
            "skipped_count": 0,
            "skipped": [],
            "checksum_refresh_required": False,
            "checksum_refresh_command": "",
        }
    operator_validation_fields = _operator_validation_invitation_fields(checklist_payload)
    operator_validation_command_templates = _operator_validation_command_templates(
        operator_validation_fields.get("operator_validation_required_missing_evidence_gate_ids")
        if isinstance(
            operator_validation_fields.get("operator_validation_required_missing_evidence_gate_ids"),
            list,
        )
        else None
    )
    manifest_paths = _handoff_manifest_paths_for_validation_checklist(checklist_path)
    refreshed_manifests: list[dict[str, object]] = []
    refreshed_manifest_payloads: list[dict[str, object]] = []
    refreshed_files: list[str] = []
    skipped: list[dict[str, object]] = []
    updated_by_user: dict[str, dict[str, object]] = {}
    live_assignment_contract_fields: dict[str, object] = {}
    if store is not None:
        assignment_ownership_integrity = _assignment_ownership_integrity(
            store.list_assignments(status=None),
            schema_integrity=store.assignment_schema_integrity(),
        )
        live_assignment_contract_fields = _single_owner_assignment_live_contract_fields(
            store,
            integrity=assignment_ownership_integrity,
        )
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append({"path": str(manifest_path), "reason": str(exc)})
            continue
        if not isinstance(manifest, dict) or not str(manifest.get("user") or "").strip():
            skipped.append({"path": str(manifest_path), "reason": "not_user_handoff_manifest"})
            continue
        user = str(manifest.get("user") or "")
        files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
        work_summary_path = Path(
            str(files.get("work_summary") or manifest_path.parent / "work-summary.json")
        )
        if not work_summary_path.is_absolute():
            work_summary_path = manifest_path.parent / work_summary_path
        if work_summary_path.is_file():
            try:
                work_summary_payload = json.loads(work_summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                work_summary_payload = {}
            if isinstance(work_summary_payload, Mapping):
                work_summary_source = (
                    work_summary_payload.get("work")
                    if isinstance(work_summary_payload.get("work"), Mapping)
                    else work_summary_payload
                )
                if isinstance(work_summary_source, dict):
                    _add_work_summary_fields(work_summary_source)
                for key in (
                    "empty_state",
                    "progress_summary",
                    "dataset_queue_summary",
                    "direct_browser_start_contract_summary",
                    "dataset_queue_state",
                    "labeler_work_completion",
                    "labeler_start_ready",
                    "labeler_start_status",
                    "labeler_action",
                    "labeler_start_message",
                    "labeler_start_operator_action",
                    "dataset_queue",
                ):
                    if isinstance(work_summary_source, Mapping) and key in work_summary_source:
                        manifest[key] = work_summary_source[key]
        if live_assignment_contract_fields:
            manifest.update(live_assignment_contract_fields)
        elif isinstance(manifest.get("assignment_ownership_contract"), Mapping):
            _add_payload_contract_compact_fields(manifest)
        if store is not None and user:
            manifest["known_user_status"] = _known_labeler_status(store, user)
        manifest.update(_handoff_dataset_queue_start_fields(manifest))
        manifest.update(operator_validation_fields)
        manifest.update(_operator_validation_gate_flat_fields(manifest))
        manifest["operator_validation_command_templates"] = operator_validation_command_templates
        manifest["operator_validation_visibility_policy"] = (
            _operator_validation_visibility_policy()
        )
        if not isinstance(manifest.get("browser_mutation_write_policy"), Mapping):
            manifest["browser_mutation_write_policy"] = _browser_mutation_write_policy()
        if not isinstance(manifest.get("browser_mutation_write_checklist"), Mapping):
            manifest["browser_mutation_write_checklist"] = (
                _browser_mutation_write_runtime_checklist(
                    manifest.get("browser_mutation_write_policy")
                    if isinstance(manifest.get("browser_mutation_write_policy"), Mapping)
                    else None
                )
            )
        if not isinstance(manifest.get("dataset_queue_direct_start_policy"), Mapping):
            manifest["dataset_queue_direct_start_policy"] = (
                _dataset_queue_direct_start_policy()
            )
        if not isinstance(
            manifest.get("runtime_operator_validation_gate_cli_policy"),
            Mapping,
        ):
            manifest["runtime_operator_validation_gate_cli_policy"] = (
                _runtime_operator_validation_gate_cli_policy()
            )
        manifest.update(
            _runtime_operator_validation_gate_cli_policy_fields(
                manifest.get("runtime_operator_validation_gate_cli_policy")
                if isinstance(
                    manifest.get("runtime_operator_validation_gate_cli_policy"),
                    Mapping,
                )
                else None
            )
        )
        labeler_route_authorization_policy = (
            manifest.get("labeler_route_authorization_policy")
            if isinstance(manifest.get("labeler_route_authorization_policy"), Mapping)
            else _labeler_route_authorization_policy()
        )
        known_user_status = (
            manifest.get("known_user_status")
            if isinstance(manifest.get("known_user_status"), Mapping)
            else {}
        )
        manifest["labeler_route_authorization_policy"] = labeler_route_authorization_policy
        manifest["labeler_route_authorization_checklist"] = _labeler_route_authorization_runtime_checklist(
            policy=labeler_route_authorization_policy,
            user=user,
            expected_user=str(manifest.get("expected_user") or user),
            known_user_status=known_user_status,
            assignment_ownership_contract=(
                manifest.get("assignment_ownership_contract")
                if isinstance(manifest.get("assignment_ownership_contract"), Mapping)
                else None
            ),
        )
        manifest.update(_handoff_labeler_route_authorization_fields(manifest))
        manifest["ready_to_send"] = _handoff_ready_to_send(manifest)
        manifest["sendability_reasons"] = _handoff_sendability_reasons(manifest)
        manifest["sendability_actions"] = _handoff_sendability_actions(manifest["sendability_reasons"])
        manifest["sendability_warnings"] = _handoff_sendability_summary([manifest])["warnings"]
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        refreshed_files.append(str(manifest_path))
        visible_report = _refresh_user_handoff_visible_files(
            manifest_path=manifest_path,
            manifest=manifest,
        )
        refreshed_files.extend(
            str(path)
            for path in visible_report.get("refreshed_files", [])
            if str(path).strip()
        )
        for row in visible_report.get("skipped_files", []):
            if isinstance(row, Mapping):
                skipped.append({"path": str(row.get("file") or ""), "reason": str(row.get("reason") or ""), "details": str(row.get("details") or "")})
        status = _handoff_status_from_manifest(manifest, datetime.now(timezone.utc))
        refreshed_manifests.append(
            {
                "path": str(manifest_path),
                "user": str(manifest.get("user") or ""),
                "ready_to_send": bool(status.get("ready_to_send")),
                "sendability_reasons": status.get("sendability_reasons", []),
            }
        )
        refreshed_manifest_payloads.append(manifest)
        updated_by_user[str(manifest.get("user") or "")] = {
            "ready_to_send": bool(manifest.get("ready_to_send")),
            "sendability_reasons": manifest.get("sendability_reasons", []),
            "sendability_actions": manifest.get("sendability_actions", []),
            "sendability_warnings": manifest.get("sendability_warnings", []),
            "operator_validation_command_templates": operator_validation_command_templates,
            **_operator_validation_public_fields(manifest),
            **_operator_validation_gate_flat_fields(manifest),
            **_operator_validation_command_template_fields(manifest),
            "status": status.get("status", ""),
            "ok": bool(status.get("ok", manifest.get("ok"))),
        }

    index_paths = [
        path
        for path in (checklist_path.parent / "index.json", checklist_path.parent / "handoffs" / "index.json")
        if path.is_file()
    ]
    for index_path in index_paths:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append({"path": str(index_path), "reason": str(exc)})
            continue
        if not isinstance(index, dict) or not isinstance(index.get("handoffs"), list):
            skipped.append({"path": str(index_path), "reason": "not_handoff_index"})
            continue
        refreshed_rows: list[dict[str, object]] = []
        for row in index["handoffs"]:
            if not isinstance(row, Mapping):
                continue
            user = str(row.get("user") or "")
            refreshed = updated_by_user.get(user)
            refreshed_rows.append({**dict(row), **(refreshed or {})})
        index["handoffs"] = refreshed_rows
        index["operator_validation_command_templates"] = operator_validation_command_templates
        sendability = _handoff_sendability_summary(
            refreshed_manifest_payloads if refreshed_manifest_payloads else refreshed_rows
        )
        counts = index.get("counts") if isinstance(index.get("counts"), dict) else {}
        index["counts"] = {
            **counts,
            "ready_to_send": sendability["ready_to_send_count"],
            "not_ready_to_send": sendability["not_ready_to_send_count"],
            "sendability_reasons": _count_handoff_sendability_reasons(sendability["warnings"]),
        }
        index["sendability_warnings"] = sendability["warnings"]
        index["sendability_actions"] = _handoff_sendability_actions(
            _count_handoff_sendability_reasons(sendability["warnings"]).keys()
        )
        index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        refreshed_files.append(str(index_path))
        files = index.get("files") if isinstance(index.get("files"), Mapping) else {}
        html_path = Path(str(files.get("html_index") or index_path.with_suffix(".html")))
        if not html_path.is_absolute():
            html_path = index_path.parent / html_path
        roster_path = Path(str(files.get("handoffs_roster") or files.get("labeler_roster") or index_path.parent / "labeler-roster.csv"))
        if not roster_path.is_absolute():
            roster_path = index_path.parent / roster_path
        try:
            _write_user_handoffs_html_index(index, html_path)
            refreshed_files.append(str(html_path))
        except OSError as exc:
            skipped.append({"path": str(html_path), "reason": str(exc)})
        try:
            _write_user_handoffs_roster_csv(index, roster_path)
            refreshed_files.append(str(roster_path))
        except OSError as exc:
            skipped.append({"path": str(roster_path), "reason": str(exc)})

    refreshed_file_list = sorted(set(refreshed_files))
    refreshed_visible_json_files = [
        path
        for path in refreshed_file_list
        if Path(path).name in {"manifest.json", "work-summary.json", "dataset-queue.json"}
    ]
    return {
        "enabled": True,
        "manifest_count": len(manifest_paths),
        "refreshed_manifest_count": len(refreshed_manifests),
        "refreshed_manifests": refreshed_manifests,
        "refreshed_file_count": len(refreshed_file_list),
        "refreshed_files": refreshed_file_list,
        "refreshed_visible_json_file_count": len(refreshed_visible_json_files),
        "refreshed_visible_json_files": refreshed_visible_json_files,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "checksum_refresh_required": bool(refreshed_files),
        "checksum_refresh_command": (
            f"refresh-handoff-checksums --path {checklist_path.parent} --operator OPERATOR --reason 'operator evidence update'"
            if refreshed_files
            else ""
        ),
    }


def _apply_operator_evidence_templates_to_validation_checklist_file_impl(
    *,
    path: Path,
    operator: str | None,
    append_log: Path | None,
    output: Path | None,
    overwrite: bool,
    refresh_handoffs: bool = True,
    store: LabelingStore | None = None,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Validation checklist must be a JSON object: {path}")
    if str(payload.get("schema") or "") != "palette.web_labeling_validation_checklist.v1":
        raise ValueError("Input is not a palette.web_labeling_validation_checklist.v1 payload.")
    output_path = output or path
    if output is not None and output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing validation checklist: {output_path}")
    summary = _operator_evidence_template_summary(
        payload,
        load_template=lambda template_path: _load_operator_evidence_template_from_directory(
            path.parent,
            template_path,
        ),
    )
    statuses = (
        summary.get("operator_evidence_template_statuses")
        if isinstance(summary.get("operator_evidence_template_statuses"), Mapping)
        else {}
    )
    gates = payload.get("gates") if isinstance(payload.get("gates"), list) else []
    gate_status_by_id = {
        str(gate.get("id") or ""): str(gate.get("status") or "")
        for gate in gates
        if isinstance(gate, Mapping)
    }
    applied: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for gate_id in OPERATOR_EVIDENCE_TEMPLATE_FIELDS:
        status = statuses.get(gate_id) if isinstance(statuses.get(gate_id), Mapping) else {}
        gate_status = gate_status_by_id.get(gate_id, "")
        if gate_id not in gate_status_by_id:
            skipped.append({"gate_id": gate_id, "reason": "gate_not_present"})
            continue
        if gate_status == "passed":
            skipped.append({"gate_id": gate_id, "reason": "already_passed"})
            continue
        if not bool(status.get("ready")):
            skipped.append(
                {
                    "gate_id": gate_id,
                    "reason": str(status.get("approval_status") or "pending_or_missing"),
                    "template_path": str(status.get("template_path") or ""),
                }
            )
            continue
        template_path = str(status.get("template_path") or "")
        evidence_note = (
            "Approved operator evidence template applied: "
            f"{gate_id} ({status.get('approval_status')}, "
            f"{status.get('approved_count')}/{status.get('required_count')})."
        )
        previous_status = gate_status
        _update_validation_checklist_payload(
            payload,
            gate_id=gate_id,
            status="passed",
            evidence=[evidence_note],
            evidence_files=[template_path],
            operator=operator,
        )
        if append_log is not None:
            _append_validation_log_evidence(
                path=append_log,
                checklist_path=output_path,
                gate_id=gate_id,
                status="passed",
                evidence=[evidence_note],
                evidence_files=[template_path],
                operator=operator,
                updated_payload=payload,
            )
        applied.append(
            {
                "gate_id": gate_id,
                "previous_status": previous_status,
                "status": "passed",
                "template_path": template_path,
            }
    )
    _recompute_validation_checklist_summary(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    refreshed_summary = _validation_checklist_gate_summary(payload)
    safe_share_gate = (
        payload.get("safe_share_gate")
        if isinstance(payload.get("safe_share_gate"), Mapping)
        else _safe_share_gate_policy()
    )
    safe_share_fields = _safe_share_gate_flat_fields(safe_share_gate)
    safe_share_checklist_fields = _safe_share_checklist_gate_status_fields(
        gates=payload.get("gates") if isinstance(payload.get("gates"), list) else [],
        safe_share_gate=safe_share_gate,
    )
    handoff_refresh = _refresh_handoff_manifests_from_validation_checklist(
        checklist_path=output_path,
        checklist_payload=payload,
        enabled=bool(refresh_handoffs and output is None),
        store=store,
    )
    return {
        "ok": True,
        "schema": "palette.web_labeling_operator_evidence_template_apply_report.v1",
        "path": str(path),
        "output": str(output_path),
        "operator": str(operator or ""),
        "applied_count": len(applied),
        "skipped_count": len(skipped),
        "applied": applied,
        "skipped": skipped,
        "all_validation_complete": bool(payload.get("all_validation_complete")),
        "ready_for_operator_validation": bool(payload.get("ready_for_operator_validation")),
        "validation_gate_classification": payload.get("validation_gate_classification", {}),
        "operator_evidence_gate_ids": payload.get("operator_evidence_gate_ids", []),
        "generated_contract_gate_ids": payload.get("generated_contract_gate_ids", []),
        "operator_evidence_pending_gate_ids": payload.get("operator_evidence_pending_gate_ids", []),
        "operator_evidence_needs_review_gate_ids": payload.get("operator_evidence_needs_review_gate_ids", []),
        "generated_contract_failed_gate_ids": payload.get("generated_contract_failed_gate_ids", []),
        "counts": payload.get("counts", {}),
        "safe_share_gate": safe_share_gate,
        **safe_share_fields,
        **safe_share_checklist_fields,
        "validation_summary": refreshed_summary,
        "validation_log_appended": append_log is not None and bool(applied),
        "validation_log": str(append_log) if append_log is not None else "",
        "handoff_refresh": handoff_refresh,
        "handoff_refresh_enabled": bool(handoff_refresh.get("enabled")),
        "handoff_refresh_refreshed_manifest_count": int(
            handoff_refresh.get("refreshed_manifest_count") or 0
        ),
        "handoff_refresh_refreshed_file_count": int(
            handoff_refresh.get("refreshed_file_count") or 0
        ),
        "handoff_refresh_refreshed_files": handoff_refresh.get("refreshed_files", []),
        "handoff_refresh_refreshed_visible_json_file_count": int(
            handoff_refresh.get("refreshed_visible_json_file_count") or 0
        ),
        "handoff_refresh_refreshed_visible_json_files": handoff_refresh.get(
            "refreshed_visible_json_files", []
        ),
        "handoff_refresh_skipped_count": int(handoff_refresh.get("skipped_count") or 0),
        "handoff_refresh_skipped": handoff_refresh.get("skipped", []),
        "checksum_refresh_required": bool(handoff_refresh.get("checksum_refresh_required")),
        "checksum_refresh_command": str(handoff_refresh.get("checksum_refresh_command") or ""),
        "validation_checklist": payload,
    }


# Preserve original helper names inside this module so moved helpers can
# continue to call each other exactly as they did in web.py.
_refresh_user_handoff_visible_files = _refresh_user_handoff_visible_files_impl
_handoff_manifest_paths_for_validation_checklist = _handoff_manifest_paths_for_validation_checklist_impl
_refresh_handoff_manifests_from_validation_checklist = _refresh_handoff_manifests_from_validation_checklist_impl
_apply_operator_evidence_templates_to_validation_checklist_file = _apply_operator_evidence_templates_to_validation_checklist_file_impl
