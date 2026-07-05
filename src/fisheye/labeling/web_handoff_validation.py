"""User-handoff validation/evidence file helpers for web labeling."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _require_validation_dependencies(
    dependencies: Mapping[str, object],
    names: tuple[str, ...],
) -> dict[str, Any]:
    missing = [name for name in names if name not in dependencies]
    if missing:
        raise KeyError(f"missing user handoff validation dependencies: {', '.join(missing)}")
    return {name: dependencies[name] for name in names}


def _write_user_handoff_validation_log_impl(
    manifest: dict[str, object],
    output_path: Path,
    *,
    dependencies: Mapping[str, object],
) -> None:
    validation_dependencies = _require_validation_dependencies(
        dependencies, ('_write_web_labeling_validation_log',)
    )
    _write_web_labeling_validation_log = validation_dependencies['_write_web_labeling_validation_log']
    _write_web_labeling_validation_log(manifest, output_path, bundle_label="single-user handoff bundle")



def _write_user_handoff_validation_checklist_impl(
    manifest: dict[str, object],
    output_path: Path,
    *,
    dependencies: Mapping[str, object],
) -> None:
    validation_dependencies = _require_validation_dependencies(
        dependencies, ('_write_web_labeling_validation_checklist',)
    )
    _write_web_labeling_validation_checklist = validation_dependencies['_write_web_labeling_validation_checklist']
    _write_web_labeling_validation_checklist(manifest, output_path, bundle_label="single-user handoff bundle")



def _inspect_handoff_validation_log(path: Path) -> dict[str, object]:
    required_name = "validation-log-template.md"
    if path.is_dir():
        required_path = path / required_name
        related_paths = sorted(str(candidate) for candidate in path.rglob(required_name) if candidate.is_file())
        return {
            "required": True,
            "present": required_path.exists(),
            "required_path": str(required_path),
            "matched_paths": [str(required_path)] if required_path.exists() else [],
            "related_paths": related_paths,
        }
    if path.is_file() and path.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name == required_name or name.endswith(f"/{required_name}")
            )
        top_level_names = [
            name
            for name in names
            if name == required_name or len([part for part in name.split("/") if part]) == 2
        ]
        return {
            "required": True,
            "present": bool(top_level_names),
            "required_path": f"*/{required_name}",
            "matched_paths": top_level_names,
            "related_paths": names,
        }
    return {
        "required": True,
        "present": False,
        "required_path": required_name,
        "matched_paths": [],
        "related_paths": [],
    }



def _operator_evidence_command_sheet_boundary_status(text: str) -> dict[str, object]:
    required_phrases = [
        "Boundary: operator-only",
        "not labeler instructions",
        "Do not send this command sheet",
        "Labelers should use only their guarded browser links",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in text]
    return {
        "operator_only_boundary_present": not missing_phrases,
        "operator_only_boundary_required_phrases": required_phrases,
        "operator_only_boundary_missing_phrases": missing_phrases,
    }



def _inspect_handoff_operator_evidence_commands(path: Path, *, required: bool = True) -> dict[str, object]:
    required_name = "operator-evidence-commands.txt"
    if path.is_dir():
        required_path = path / required_name
        related_paths = sorted(str(candidate) for candidate in path.rglob(required_name) if candidate.is_file())
        present = required_path.is_file()
        boundary_status = _operator_evidence_command_sheet_boundary_status(
            required_path.read_text(encoding="utf-8") if present else ""
        )
        return {
            "required": required,
            "present": present,
            "valid": present and bool(boundary_status["operator_only_boundary_present"]),
            "required_path": str(required_path),
            "matched_paths": [str(required_path)] if present else [],
            "related_paths": related_paths,
            "error": "",
            **boundary_status,
        }
    if path.is_file() and path.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name == required_name or name.endswith(f"/{required_name}")
            )
            top_level_names = [
                name
                for name in names
                if name == required_name or len([part for part in name.split("/") if part]) == 2
            ]
            boundary_text = ""
            if top_level_names:
                try:
                    boundary_text = archive.read(top_level_names[0]).decode("utf-8")
                except (KeyError, UnicodeDecodeError):
                    boundary_text = ""
            boundary_status = _operator_evidence_command_sheet_boundary_status(boundary_text)
        return {
            "required": required,
            "present": bool(top_level_names),
            "valid": bool(top_level_names) and bool(boundary_status["operator_only_boundary_present"]),
            "required_path": f"*/{required_name}",
            "matched_paths": top_level_names,
            "related_paths": names,
            "error": "",
            **boundary_status,
        }
    return {
        "required": required,
        "present": False,
        "valid": False,
        "required_path": required_name,
        "matched_paths": [],
        "related_paths": [],
        "error": "",
        **_operator_evidence_command_sheet_boundary_status(""),
    }



def _launch_evidence_execution_checklist_status(text: str) -> dict[str, object]:
    required_phrases = [
        "Palette web-labeling launch evidence execution checklist",
        "Operator-only checklist",
        "record-zarr-backup-evidence",
        "record-browser-response-security-evidence",
        "record-identity-source-evidence",
        "record-browser-smoke-evidence",
        "record-disposable-zarr-mutation-smoke-evidence",
        "apply-operator-evidence-templates",
        "inspect-handoff --path PACKAGE_PATH --require-shareable",
        "labeler_links_safe_to_share=true",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in text]
    return {
        "checklist_contract_present": not missing_phrases,
        "checklist_required_phrases": required_phrases,
        "checklist_missing_phrases": missing_phrases,
    }



def _inspect_handoff_launch_evidence_execution_checklist(
    path: Path, *, required: bool = True
) -> dict[str, object]:
    required_name = "launch-evidence-execution-checklist.txt"
    if path.is_dir():
        required_path = path / required_name
        related_paths = sorted(str(candidate) for candidate in path.rglob(required_name) if candidate.is_file())
        present = required_path.is_file()
        checklist_status = _launch_evidence_execution_checklist_status(
            required_path.read_text(encoding="utf-8") if present else ""
        )
        return {
            "required": required,
            "present": present,
            "valid": present and bool(checklist_status["checklist_contract_present"]),
            "required_path": str(required_path),
            "matched_paths": [str(required_path)] if present else [],
            "related_paths": related_paths,
            "error": "",
            **checklist_status,
        }
    if path.is_file() and path.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name == required_name or name.endswith(f"/{required_name}")
            )
            top_level_names = [
                name
                for name in names
                if name == required_name or len([part for part in name.split("/") if part]) == 2
            ]
            checklist_text = ""
            if top_level_names:
                try:
                    checklist_text = archive.read(top_level_names[0]).decode("utf-8")
                except (KeyError, UnicodeDecodeError):
                    checklist_text = ""
            checklist_status = _launch_evidence_execution_checklist_status(checklist_text)
        return {
            "required": required,
            "present": bool(top_level_names),
            "valid": bool(top_level_names) and bool(checklist_status["checklist_contract_present"]),
            "required_path": f"*/{required_name}",
            "matched_paths": top_level_names,
            "related_paths": names,
            "error": "",
            **checklist_status,
        }
    return {
        "required": required,
        "present": False,
        "valid": False,
        "required_path": required_name,
        "matched_paths": [],
        "related_paths": [],
        "error": "",
        **_launch_evidence_execution_checklist_status(""),
    }


def _load_operator_evidence_template_from_directory(root: Path, template_path: str) -> tuple[dict[str, object] | None, bool, bool, str]:
    candidates: list[Path] = []
    if template_path:
        raw = Path(template_path)
        candidates.append(raw if raw.is_absolute() else root / raw)
        candidates.append(root / raw.name)
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
            return (loaded if isinstance(loaded, dict) else {}, True, isinstance(loaded, dict), "")
        except (OSError, json.JSONDecodeError) as exc:
            return None, True, False, str(exc)
    return None, False, False, ""



def _load_operator_evidence_template_from_zip(archive, template_path: str) -> tuple[dict[str, object] | None, bool, bool, str]:
    basename = Path(str(template_path or "")).name
    if not basename:
        return None, False, False, ""
    matches = sorted(
        name
        for name in archive.namelist()
        if name == basename or name.endswith(f"/{basename}")
    )
    if not matches:
        return None, False, False, ""
    chosen = min(matches, key=lambda name: (len([part for part in name.split("/") if part]), name))
    try:
        loaded = json.loads(archive.read(chosen).decode("utf-8"))
        return (loaded if isinstance(loaded, dict) else {}, True, isinstance(loaded, dict), "")
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, True, False, str(exc)



def _operator_evidence_commands_public_summary(
    operator_evidence_commands: Mapping[str, object],
) -> dict[str, object]:
    required = bool(operator_evidence_commands.get("required"))
    present = bool(operator_evidence_commands.get("present"))
    valid = bool(operator_evidence_commands.get("valid"))
    boundary_present = bool(operator_evidence_commands.get("operator_only_boundary_present"))
    missing_phrases = [
        str(phrase)
        for phrase in (
            operator_evidence_commands.get("operator_only_boundary_missing_phrases")
            if isinstance(
                operator_evidence_commands.get("operator_only_boundary_missing_phrases"),
                list,
            )
            else []
        )
        if str(phrase)
    ]
    matched_paths = (
        operator_evidence_commands.get("matched_paths")
        if isinstance(operator_evidence_commands.get("matched_paths"), list)
        else []
    )
    related_paths = (
        operator_evidence_commands.get("related_paths")
        if isinstance(operator_evidence_commands.get("related_paths"), list)
        else []
    )
    blocking_reason_id = ""
    if required and not valid:
        blocking_reason_id = (
            "operator_evidence_commands_missing"
            if not present
            else (
                "operator_evidence_commands_boundary_missing"
                if not boundary_present
                else "operator_evidence_commands_invalid"
            )
        )
    return {
        "schema": "palette.web_labeling_operator_evidence_commands_summary.v1",
        "required": required,
        "present": present,
        "valid": valid,
        "operator_only_boundary_present": boundary_present,
        "operator_only_boundary_missing_phrases": missing_phrases,
        "operator_only_boundary_missing_phrase_count": len(missing_phrases),
        "matched_path_count": len(matched_paths),
        "related_path_count": len(related_paths),
        "required_path": str(operator_evidence_commands.get("required_path") or ""),
        "blocking_reason_id": blocking_reason_id,
    }



def _launch_evidence_execution_checklist_public_summary(
    launch_evidence_execution_checklist: Mapping[str, object],
) -> dict[str, object]:
    required = bool(launch_evidence_execution_checklist.get("required"))
    present = bool(launch_evidence_execution_checklist.get("present"))
    valid = bool(launch_evidence_execution_checklist.get("valid"))
    contract_present = bool(
        launch_evidence_execution_checklist.get("checklist_contract_present")
    )
    missing_phrases = [
        str(phrase)
        for phrase in (
            launch_evidence_execution_checklist.get("checklist_missing_phrases")
            if isinstance(
                launch_evidence_execution_checklist.get("checklist_missing_phrases"),
                list,
            )
            else []
        )
        if str(phrase)
    ]
    matched_paths = (
        launch_evidence_execution_checklist.get("matched_paths")
        if isinstance(launch_evidence_execution_checklist.get("matched_paths"), list)
        else []
    )
    related_paths = (
        launch_evidence_execution_checklist.get("related_paths")
        if isinstance(launch_evidence_execution_checklist.get("related_paths"), list)
        else []
    )
    blocking_reason_id = ""
    if required and not valid:
        blocking_reason_id = (
            "launch_evidence_execution_checklist_missing"
            if not present
            else (
                "launch_evidence_execution_checklist_incomplete"
                if not contract_present
                else "launch_evidence_execution_checklist_invalid"
            )
        )
    return {
        "schema": "palette.web_labeling_launch_evidence_execution_checklist_summary.v1",
        "required": required,
        "present": present,
        "valid": valid,
        "checklist_contract_present": contract_present,
        "checklist_missing_phrases": missing_phrases,
        "checklist_missing_phrase_count": len(missing_phrases),
        "matched_path_count": len(matched_paths),
        "related_path_count": len(related_paths),
        "required_path": str(launch_evidence_execution_checklist.get("required_path") or ""),
        "blocking_reason_id": blocking_reason_id,
    }



def _launch_evidence_execution_checklist_inspection_target() -> dict[str, object]:
    return {
        "shareability_launch_evidence_execution_checklist_required": True,
        "shareability_launch_evidence_execution_checklist_file": (
            "launch-evidence-execution-checklist.txt"
        ),
        "shareability_launch_evidence_execution_checklist_field": (
            "launch_evidence_execution_checklist"
        ),
        "shareability_launch_evidence_execution_checklist_summary_field": (
            "launch_evidence_execution_checklist_summary"
        ),
        "shareability_launch_evidence_execution_checklist_top_level_fields": [
            "launch_evidence_execution_checklist_required",
            "launch_evidence_execution_checklist_present",
            "launch_evidence_execution_checklist_valid",
            "launch_evidence_execution_checklist_contract_present",
            "launch_evidence_execution_checklist_missing_phrases",
            "launch_evidence_execution_checklist_blocking_reason_id",
        ],
        "shareability_launch_evidence_execution_checklist_required_phrases": [
            "Palette web-labeling launch evidence execution checklist",
            "Operator-only checklist",
            "record-zarr-backup-evidence",
            "record-browser-response-security-evidence",
            "record-identity-source-evidence",
            "record-browser-smoke-evidence",
            "record-disposable-zarr-mutation-smoke-evidence",
            "apply-operator-evidence-templates",
            "inspect-handoff --path PACKAGE_PATH --require-shareable",
            "labeler_links_safe_to_share=true",
        ],
        "shareability_launch_evidence_execution_checklist_blocking_reason_ids": [
            "launch_evidence_execution_checklist_missing",
            "launch_evidence_execution_checklist_incomplete",
            "launch_evidence_execution_checklist_invalid",
        ],
    }

def _operator_evidence_template_status_impl(
    *,
    gate_id: str,
    template_path: str,
    template: Mapping[str, object] | None,
    present: bool,
    valid: bool,
    error: str = "",
dependencies: Mapping[str, object],
) -> dict[str, object]:

    validation_dependencies = _require_validation_dependencies(dependencies, ('BROWSER_SMOKE_REQUIRED_FIELDS', 'DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS', '_browser_smoke_personalized_route_contract', '_disposable_zarr_smoke_workflow_contract_missing_fields', '_expected_user_query_value_from_url', '_identity_source_personal_queue_status', '_identity_source_row_approved'))
    BROWSER_SMOKE_REQUIRED_FIELDS = validation_dependencies['BROWSER_SMOKE_REQUIRED_FIELDS']
    DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS = validation_dependencies['DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS']
    _browser_smoke_personalized_route_contract = validation_dependencies['_browser_smoke_personalized_route_contract']
    _disposable_zarr_smoke_workflow_contract_missing_fields = validation_dependencies['_disposable_zarr_smoke_workflow_contract_missing_fields']
    _expected_user_query_value_from_url = validation_dependencies['_expected_user_query_value_from_url']
    _identity_source_personal_queue_status = validation_dependencies['_identity_source_personal_queue_status']
    _identity_source_row_approved = validation_dependencies['_identity_source_row_approved']

    if not present:
        return {
            "gate_id": gate_id,
            "template_path": template_path,
            "present": False,
            "valid": False,
            "ready": False,
            "approval_status": "missing",
            "approved_count": 0,
            "required_count": 0,
            "error": error,
        }
    if not valid or template is None:
        return {
            "gate_id": gate_id,
            "template_path": template_path,
            "present": True,
            "valid": False,
            "ready": False,
            "approval_status": "invalid",
            "approved_count": 0,
            "required_count": 0,
            "error": error,
        }

    approval_status = "pending_operator_confirmation"
    approved_count = 0
    required_count = 0
    ready = False
    extra_status: dict[str, object] = {}
    if gate_id == "identity_probe_verification":
        users = template.get("users") if isinstance(template.get("users"), list) else []
        required_count = len([row for row in users if isinstance(row, Mapping)])
        approved_count = sum(
            1
            for row in users
            if isinstance(row, Mapping) and _identity_source_row_approved(row)
        )
        ready = required_count > 0 and approved_count == required_count
        user_statuses = []
        for row in users:
            if not isinstance(row, Mapping):
                continue
            expected_user = str(row.get("expected_user") or "").strip()
            resolved_user = str(row.get("resolved_user") or "").strip()
            resolved_user_matches_expected_user = bool(
                expected_user and resolved_user and resolved_user == expected_user
            )
            personal_queue_status = _identity_source_personal_queue_status(row)
            missing_fields = []
            if not bool(row.get("identity_matches_expected_user")):
                missing_fields.append("identity_matches_expected_user")
            if not resolved_user_matches_expected_user:
                missing_fields.append("resolved_user_exact_match")
            missing_fields.extend(
                str(field)
                for field in personal_queue_status.get("missing_fields", [])
                if field not in missing_fields
            )
            if not str(row.get("operator") or "").strip():
                missing_fields.append("operator")
            if not str(row.get("operator_approved_at_utc") or "").strip():
                missing_fields.append("operator_approved_at_utc")
            user_statuses.append(
                {
                    "expected_user": expected_user,
                    "resolved_user": resolved_user,
                    "identity_matches_expected_user": bool(
                        row.get("identity_matches_expected_user")
                    ),
                    "resolved_user_matches_expected_user": resolved_user_matches_expected_user,
                    "expected_user_dataset_queue_url": personal_queue_status[
                        "expected_user_dataset_queue_url"
                    ],
                    "expected_user_personal_dataset_queue_url": personal_queue_status[
                        "expected_user_personal_dataset_queue_url"
                    ],
                    "preferred_labeler_entry_url": personal_queue_status[
                        "preferred_labeler_entry_url"
                    ],
                    "personalized_labeler_entry_url": personal_queue_status[
                        "personalized_labeler_entry_url"
                    ],
                    "preferred_labeler_entry_url_matches_dataset_queue": personal_queue_status[
                        "preferred_labeler_entry_url_matches_dataset_queue"
                    ],
                    "preferred_labeler_entry_url_matches_personal_dataset_queue": personal_queue_status[
                        "preferred_labeler_entry_url_matches_personal_dataset_queue"
                    ],
                    "personalized_labeler_entry_url_matches_personal_dataset_queue": personal_queue_status[
                        "personalized_labeler_entry_url_matches_personal_dataset_queue"
                    ],
                    "personal_queue_evidence_ready": bool(personal_queue_status["ready"]),
                    "ready": not missing_fields,
                    "missing_fields": missing_fields,
                }
            )
        personal_queue_evidence_ready_users = [
            str(status.get("expected_user") or "")
            for status in user_statuses
            if bool(status.get("personal_queue_evidence_ready"))
        ]
        personal_queue_evidence_missing_users = [
            str(status.get("expected_user") or "")
            for status in user_statuses
            if not bool(status.get("personal_queue_evidence_ready"))
        ]
        personal_queue_evidence_missing_fields_by_user = {
            str(status.get("expected_user") or ""): [
                str(field)
                for field in (
                    status.get("missing_fields")
                    if isinstance(status.get("missing_fields"), list)
                    else []
                )
                if str(field).startswith("expected_user_personal_dataset_queue_url")
                or str(field).startswith("preferred_labeler_entry_url")
                or str(field).startswith("personalized_labeler_entry_url")
            ]
            for status in user_statuses
            if not bool(status.get("personal_queue_evidence_ready"))
        }
        personal_queue_evidence_missing_fields_by_user = {
            user: fields
            for user, fields in personal_queue_evidence_missing_fields_by_user.items()
            if user and fields
        }
        extra_status = {
            "user_statuses": user_statuses,
            "users_missing_required_fields": [
                status
                for status in user_statuses
                if isinstance(status.get("missing_fields"), list) and status.get("missing_fields")
            ],
            "personal_queue_evidence_ready_count": len(personal_queue_evidence_ready_users),
            "personal_queue_evidence_missing_count": len(personal_queue_evidence_missing_users),
            "personal_queue_evidence_ready_users": personal_queue_evidence_ready_users,
            "personal_queue_evidence_missing_users": personal_queue_evidence_missing_users,
            "personal_queue_evidence_missing_fields_by_user": (
                personal_queue_evidence_missing_fields_by_user
            ),
            "all_users_have_personal_queue_evidence": bool(
                required_count > 0
                and len(personal_queue_evidence_ready_users) == required_count
            ),
        }
    elif gate_id == "browser_response_security_headers":
        checks = template.get("checks") if isinstance(template.get("checks"), Mapping) else {}
        capture = template.get("capture") if isinstance(template.get("capture"), Mapping) else {}
        approval = template.get("operator_approval") if isinstance(template.get("operator_approval"), Mapping) else {}
        required_capture_contract = (
            dict(template.get("required_capture_contract"))
            if isinstance(template.get("required_capture_contract"), Mapping)
            else {}
        )
        expected_headers = (
            template.get("expected_headers") if isinstance(template.get("expected_headers"), Mapping) else {}
        )
        captured_headers = (
            template.get("captured_headers") if isinstance(template.get("captured_headers"), Mapping) else {}
        )
        expected_user_capture_query_required = bool(
            template.get("expected_user_capture_query_required")
        )
        capture_url_value = str(capture.get("url") or "").strip()
        authenticated_test_user_value = str(capture.get("authenticated_test_user") or "").strip()
        expected_user_query_value = _expected_user_query_value_from_url(capture_url_value)
        capture_url_has_expected_user_query = bool(expected_user_query_value)
        authenticated_test_user_matches_expected_user = bool(
            expected_user_query_value
            and authenticated_test_user_value == expected_user_query_value
        )
        capture_context_required_fields = (
            [
                "capture.url",
                "capture.url.expected_user_query",
                "capture.authenticated_test_user",
                "capture.authenticated_test_user_matches_expected_user",
            ]
            if expected_user_capture_query_required
            else []
        )
        capture_context_ready_fields = [
            field
            for field, ready_field in {
                "capture.url": bool(capture_url_value),
                "capture.url.expected_user_query": capture_url_has_expected_user_query,
                "capture.authenticated_test_user": bool(authenticated_test_user_value),
                "capture.authenticated_test_user_matches_expected_user": (
                    authenticated_test_user_matches_expected_user
                ),
            }.items()
            if field in capture_context_required_fields and ready_field
        ]
        capture_context_missing_fields = [
            field
            for field in capture_context_required_fields
            if field not in capture_context_ready_fields
        ]
        capture_url_contract_ready = bool(checks.get("capture_url_contract_ready"))
        authenticated_test_user_contract_ready = bool(
            checks.get("authenticated_test_user_contract_ready")
        )
        capture_context_ready = (
            not expected_user_capture_query_required
            or (
                not capture_context_missing_fields
                and capture_url_contract_ready
                and authenticated_test_user_contract_ready
                and authenticated_test_user_matches_expected_user
            )
        )
        required_count = len(checks) + len(expected_headers) + len(capture_context_required_fields)
        approved_checks = sum(1 for value in checks.values() if bool(value))
        captured_header_count = sum(
            1 for key in expected_headers if str(captured_headers.get(key) or "").strip()
        )
        approved_count = (
            approved_checks
            + captured_header_count
            + len(capture_context_ready_fields)
        )
        missing_checks = [str(key) for key, value in checks.items() if not bool(value)]
        missing_headers = [
            str(key)
            for key in expected_headers
            if not str(captured_headers.get(key) or "").strip()
        ]
        approval_status_value = str(approval.get("status") or "")
        ready = (
            approval_status_value == "operator_approved"
            and required_count > 0
            and approved_count == required_count
            and capture_context_ready
        )
        extra_status = {
            "missing_checks": missing_checks,
            "missing_headers": missing_headers,
            "capture_context_required": expected_user_capture_query_required,
            "capture_context_ready": capture_context_ready,
            "required_capture_contract": required_capture_contract,
            "capture_url": capture_url_value,
            "capture_url_has_expected_user_query": capture_url_has_expected_user_query,
            "capture_url_expected_user": expected_user_query_value,
            "authenticated_test_user_present": bool(authenticated_test_user_value),
            "authenticated_test_user_matches_expected_user": (
                authenticated_test_user_matches_expected_user
            ),
            "capture_context_missing_fields": capture_context_missing_fields,
            "expected_header_names": [str(key) for key in expected_headers],
            "captured_header_names": [
                str(key)
                for key in expected_headers
                if str(captured_headers.get(key) or "").strip()
            ],
            "operator_approval_status": approval_status_value,
            "operator_approval_missing": approval_status_value != "operator_approved",
        }
    elif gate_id == "browser_smoke":
        users = template.get("users") if isinstance(template.get("users"), list) else []
        personalized_route_smoke_contract = (
            dict(template.get("personalized_route_smoke_contract"))
            if isinstance(template.get("personalized_route_smoke_contract"), Mapping)
            else {}
        )
        expected_personalized_route_smoke_contract = _browser_smoke_personalized_route_contract()
        personalized_route_smoke_contract_missing_fields = [
            str(key)
            for key, expected_value in expected_personalized_route_smoke_contract.items()
            if personalized_route_smoke_contract.get(key) != expected_value
        ]
        personalized_route_smoke_contract_ready = not personalized_route_smoke_contract_missing_fields
        personalized_route_smoke_contract_operator_action = (
            ""
            if personalized_route_smoke_contract_ready
            else (
                "Regenerate browser-smoke-evidence-template.json from the current launch bundle "
                "before applying approval so personalized /my-datasets, /labeling, and /my-work route "
                "contract metadata is present."
            )
        )
        required_fields = BROWSER_SMOKE_REQUIRED_FIELDS
        required_count = 1 if users else 0
        approved_count = sum(
            1
            for row in users
            if isinstance(row, Mapping)
            and str(row.get("run_status") or "") == "operator_approved"
            and str(row.get("expected_user") or "").strip()
            and str(row.get("resolved_user") or "").strip() == str(row.get("expected_user") or "").strip()
            and bool(str(row.get("operator") or "").strip())
            and bool(str(row.get("operator_approved_at_utc") or "").strip())
            and all(bool(row.get(field)) for field in required_fields)
        )
        ready = approved_count >= 1 and personalized_route_smoke_contract_ready
        user_statuses = []
        for row in users:
            if not isinstance(row, Mapping):
                continue
            expected_user = str(row.get("expected_user") or "").strip()
            resolved_user = str(row.get("resolved_user") or "").strip()
            resolved_user_matches_expected_user = bool(
                expected_user and resolved_user and resolved_user == expected_user
            )
            missing_fields = [field for field in required_fields if not bool(row.get(field))]
            if not resolved_user_matches_expected_user:
                missing_fields.append("resolved_user_exact_match")
            if not str(row.get("operator") or "").strip():
                missing_fields.append("operator")
            if not str(row.get("operator_approved_at_utc") or "").strip():
                missing_fields.append("operator_approved_at_utc")
            user_statuses.append(
                {
                    "expected_user": expected_user,
                    "resolved_user": resolved_user,
                    "personalized_dataset_queue_url": str(
                        row.get("personalized_dataset_queue_url") or ""
                    ),
                    "personalized_work_url": str(
                        row.get("personalized_work_url") or ""
                    ),
                    "wrong_expected_user_personalized_dataset_queue_url": str(
                        row.get("wrong_expected_user_personalized_dataset_queue_url") or ""
                    ),
                    "wrong_expected_user_personalized_work_url": str(
                        row.get("wrong_expected_user_personalized_work_url") or ""
                    ),
                    "resolved_user_matches_expected_user": resolved_user_matches_expected_user,
                    "run_status": str(row.get("run_status") or ""),
                    "ready": (
                        str(row.get("run_status") or "") == "operator_approved"
                        and not missing_fields
                    ),
                    "missing_fields": missing_fields,
                }
            )
        extra_status = {
            "personalized_route_smoke_contract": personalized_route_smoke_contract,
            "actual_personalized_route_smoke_contract": personalized_route_smoke_contract,
            "expected_personalized_route_smoke_contract": expected_personalized_route_smoke_contract,
            "personalized_route_smoke_contract_ready": personalized_route_smoke_contract_ready,
            "personalized_route_smoke_contract_missing_fields": personalized_route_smoke_contract_missing_fields,
            "personalized_route_smoke_contract_operator_action": (
                personalized_route_smoke_contract_operator_action
            ),
            "user_statuses": user_statuses,
            "users_missing_required_fields": [
                status
                for status in user_statuses
                if isinstance(status.get("missing_fields"), list) and status.get("missing_fields")
            ],
        }
    elif gate_id == "disposable_zarr_mutation_smoke":
        workflows = template.get("workflows") if isinstance(template.get("workflows"), list) else []
        required_fields = DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS
        required_count = len([row for row in workflows if isinstance(row, Mapping)])
        approved_count = sum(
            1
            for row in workflows
            if isinstance(row, Mapping)
            and str(row.get("status") or "") == "operator_approved"
            and all(bool(row.get(field)) for field in required_fields)
            and bool(row.get("mutation_event_ids"))
            and not _disposable_zarr_smoke_workflow_contract_missing_fields(row)
        )
        ready = required_count > 0 and approved_count == required_count
        workflow_statuses = []
        for row in workflows:
            if not isinstance(row, Mapping):
                continue
            missing_fields = [field for field in required_fields if not bool(row.get(field))]
            has_mutation_event_ids = bool(row.get("mutation_event_ids"))
            if not has_mutation_event_ids:
                missing_fields.append("mutation_event_ids")
            workflow_contract_missing_fields = (
                _disposable_zarr_smoke_workflow_contract_missing_fields(row)
            )
            missing_fields.extend(workflow_contract_missing_fields)
            workflow_statuses.append(
                {
                    "workflow_kind": str(row.get("workflow_kind") or ""),
                    "status": str(row.get("status") or ""),
                    "ready": (
                        str(row.get("status") or "") == "operator_approved"
                        and not missing_fields
                    ),
                    "missing_fields": missing_fields,
                    "workflow_contract_ready": not workflow_contract_missing_fields,
                    "workflow_contract_missing_fields": workflow_contract_missing_fields,
                    "data_plane_write_target": str(row.get("data_plane_write_target") or ""),
                    "primary_mutation_target_kind": str(
                        row.get("primary_mutation_target_kind") or ""
                    ),
                    "training_zarr_mutation_target_kind": str(
                        row.get("training_zarr_mutation_target_kind") or ""
                    ),
                    "training_zarr_write_mode": str(row.get("training_zarr_write_mode") or ""),
                    "csv_handoff_artifact_role": str(row.get("csv_handoff_artifact_role") or ""),
                    "mutation_event_ids_present": has_mutation_event_ids,
                }
            )
        extra_status = {
            "workflow_statuses": workflow_statuses,
            "workflows_missing_required_fields": [
                status
                for status in workflow_statuses
                if isinstance(status.get("missing_fields"), list) and status.get("missing_fields")
            ],
        }
    elif gate_id == "mutable_zarr_backup_confirmation":
        targets = template.get("targets") if isinstance(template.get("targets"), list) else []
        required_count = len([row for row in targets if isinstance(row, Mapping)])
        if required_count == 0:
            approval_status = "not_applicable"
            ready = True
            extra_status = {
                "target_statuses": [],
                "targets_missing_required_fields": [],
            }
        else:
            approved_count = sum(
                1
                for row in targets
                if isinstance(row, Mapping)
                and str(row.get("status") or "") == "operator_approved"
                and (
                    str(row.get("backup_execution_manifest_path") or "").strip()
                    or str(row.get("backup_manifest_path") or "").strip()
                    or str(row.get("backup_destination") or "").strip()
                )
                and str(row.get("backup_verified_at_utc") or "").strip()
                and str(row.get("restore_test_result") or "").strip()
                and str(row.get("operator_approved_at_utc") or "").strip()
            )
            ready = approved_count == required_count
            target_statuses = []
            for row in targets:
                if not isinstance(row, Mapping):
                    continue
                has_backup_location = bool(
                    str(row.get("backup_execution_manifest_path") or "").strip()
                    or str(row.get("backup_manifest_path") or "").strip()
                    or str(row.get("backup_destination") or "").strip()
                )
                missing_fields = []
                if str(row.get("status") or "") != "operator_approved":
                    missing_fields.append("status")
                if not has_backup_location:
                    missing_fields.append("backup_execution_manifest_path_or_backup_location")
                if not str(row.get("backup_verified_at_utc") or "").strip():
                    missing_fields.append("backup_verified_at_utc")
                if not str(row.get("restore_test_result") or "").strip():
                    missing_fields.append("restore_test_result")
                if not str(row.get("operator_approved_at_utc") or "").strip():
                    missing_fields.append("operator_approved_at_utc")
                target_statuses.append(
                    {
                        "target_index": row.get("target_index"),
                        "zarr_path": str(row.get("zarr_path") or ""),
                        "role": str(row.get("role") or ""),
                        "status": str(row.get("status") or ""),
                        "ready": not missing_fields,
                        "missing_fields": missing_fields,
                        "backup_location_present": has_backup_location,
                    }
                )
            extra_status = {
                "target_statuses": target_statuses,
                "targets_missing_required_fields": [
                    status
                    for status in target_statuses
                    if isinstance(status.get("missing_fields"), list) and status.get("missing_fields")
                ],
            }

    if ready and approval_status != "not_applicable":
        approval_status = "operator_approved"
    return {
        "gate_id": gate_id,
        "template_path": template_path,
        "present": True,
        "valid": True,
        "ready": ready,
        "approval_status": approval_status,
        "approved_count": approved_count,
        "required_count": required_count,
        "error": "",
        "schema": str(template.get("schema") or ""),
        **extra_status,
    }

def _operator_evidence_template_summary_impl(
    payload: Mapping[str, object],
    *,
    load_template,
dependencies: Mapping[str, object],
) -> dict[str, object]:

    validation_dependencies = _require_validation_dependencies(dependencies, ('OPERATOR_EVIDENCE_TEMPLATE_FIELDS', '_identity_personal_queue_evidence_status'))
    OPERATOR_EVIDENCE_TEMPLATE_FIELDS = validation_dependencies['OPERATOR_EVIDENCE_TEMPLATE_FIELDS']
    _identity_personal_queue_evidence_status = validation_dependencies['_identity_personal_queue_evidence_status']

    def _operator_evidence_template_status(**kwargs):
        return _operator_evidence_template_status_impl(
            **kwargs,
            dependencies=dependencies,
        )

    gates = payload.get("gates") if isinstance(payload.get("gates"), list) else []
    gate_status_by_id = {
        str(gate.get("id") or ""): str(gate.get("status") or "")
        for gate in gates
        if isinstance(gate, Mapping)
    }
    statuses: dict[str, dict[str, object]] = {}
    missing_gate_ids: list[str] = []
    pending_gate_ids: list[str] = []
    approved_gate_ids: list[str] = []
    passed_unapproved_gate_ids: list[str] = []
    for gate_id, field in OPERATOR_EVIDENCE_TEMPLATE_FIELDS.items():
        template_path = str(payload.get(field) or "").strip()
        loaded_template, present, valid, error = load_template(template_path)
        status = _operator_evidence_template_status(
            gate_id=gate_id,
            template_path=template_path,
            template=loaded_template if isinstance(loaded_template, Mapping) else None,
            present=present,
            valid=valid,
            error=error,
        )
        status["gate_status"] = gate_status_by_id.get(gate_id, "")
        statuses[gate_id] = status
        if not bool(status.get("present")):
            missing_gate_ids.append(gate_id)
        elif bool(status.get("ready")):
            approved_gate_ids.append(gate_id)
        else:
            pending_gate_ids.append(gate_id)
        if gate_status_by_id.get(gate_id) == "passed" and not bool(status.get("ready")):
            passed_unapproved_gate_ids.append(gate_id)
    identity_status = statuses.get("identity_probe_verification", {})
    identity_ready_count_text = str(
        identity_status.get("personal_queue_evidence_ready_count") or "0"
    ).strip()
    identity_missing_count_text = str(
        identity_status.get("personal_queue_evidence_missing_count") or "0"
    ).strip()
    identity_ready_count = int(identity_ready_count_text) if identity_ready_count_text.isdigit() else 0
    identity_missing_count = int(identity_missing_count_text) if identity_missing_count_text.isdigit() else 0
    identity_ready_users = list(
        identity_status.get("personal_queue_evidence_ready_users")
        if isinstance(identity_status.get("personal_queue_evidence_ready_users"), list)
        else []
    )
    identity_missing_users = list(
        identity_status.get("personal_queue_evidence_missing_users")
        if isinstance(identity_status.get("personal_queue_evidence_missing_users"), list)
        else []
    )
    identity_missing_fields_by_user = dict(
        identity_status.get("personal_queue_evidence_missing_fields_by_user")
        if isinstance(
            identity_status.get("personal_queue_evidence_missing_fields_by_user"),
            Mapping,
        )
        else {}
    )
    identity_all_users_have_personal_queue_evidence = bool(
        identity_status.get("all_users_have_personal_queue_evidence")
    )
    identity_personal_queue_evidence_status = _identity_personal_queue_evidence_status(
        ready_count=identity_ready_count,
        missing_count=identity_missing_count,
        ready_users=identity_ready_users,
        missing_users=identity_missing_users,
        missing_fields_by_user=identity_missing_fields_by_user,
        all_users_have_personal_queue_evidence=identity_all_users_have_personal_queue_evidence,
    )
    return {
        "operator_evidence_template_statuses": statuses,
        "operator_evidence_template_missing_gate_ids": missing_gate_ids,
        "operator_evidence_template_pending_gate_ids": pending_gate_ids,
        "operator_evidence_template_approved_gate_ids": approved_gate_ids,
        "passed_gates_with_unapproved_evidence_templates": passed_unapproved_gate_ids,
        "operator_evidence_template_count": len(statuses),
        "operator_evidence_template_approved_count": len(approved_gate_ids),
        "operator_evidence_template_pending_count": len(pending_gate_ids),
        "operator_evidence_template_missing_count": len(missing_gate_ids),
        "identity_personal_queue_evidence_status": (
            identity_personal_queue_evidence_status
        ),
        "identity_personal_queue_evidence_ready_count": identity_ready_count,
        "identity_personal_queue_evidence_missing_count": identity_missing_count,
        "identity_personal_queue_evidence_ready_users": identity_ready_users,
        "identity_personal_queue_evidence_missing_users": identity_missing_users,
        "identity_personal_queue_evidence_missing_fields_by_user": (
            identity_missing_fields_by_user
        ),
        "identity_all_users_have_personal_queue_evidence": (
            identity_all_users_have_personal_queue_evidence
        ),
    }

def _inspect_handoff_validation_checklist_impl(path: Path, *, dependencies: Mapping[str, object]) -> dict[str, object]:

    validation_dependencies = _require_validation_dependencies(dependencies, ('_implementation_status_artifact_name', '_implementation_status_artifact_summary', '_operator_validation_command_templates', '_operator_validation_visibility_policy', '_validation_checklist_gate_summary'))
    _implementation_status_artifact_name = validation_dependencies['_implementation_status_artifact_name']
    _implementation_status_artifact_summary = validation_dependencies['_implementation_status_artifact_summary']
    _operator_validation_command_templates = validation_dependencies['_operator_validation_command_templates']
    _operator_validation_visibility_policy = validation_dependencies['_operator_validation_visibility_policy']
    _validation_checklist_gate_summary = validation_dependencies['_validation_checklist_gate_summary']

    def _operator_evidence_template_summary(payload, *, load_template):
        return _operator_evidence_template_summary_impl(
            payload,
            load_template=load_template,
            dependencies=dependencies,
        )

    required_name = "validation-checklist.json"
    if path.is_dir():
        required_path = path / required_name
        related_paths = sorted(str(candidate) for candidate in path.rglob(required_name) if candidate.is_file())
        payload: dict[str, object] | None = None
        error = ""
        if required_path.exists():
            try:
                loaded = json.loads(required_path.read_text(encoding="utf-8"))
                payload = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError) as exc:
                error = str(exc)
        implementation_status_name = _implementation_status_artifact_name(payload)
        implementation_status_required_path = path / implementation_status_name
        implementation_status_related_paths = sorted(
            str(candidate)
            for candidate in path.rglob(implementation_status_name)
            if candidate.is_file()
        )
        implementation_status_matched_paths = (
            [str(implementation_status_required_path)]
            if implementation_status_required_path.is_file()
            else []
        )
        summary = _validation_checklist_gate_summary(payload) if payload is not None else {}
        if payload is not None:
            summary = {
                **summary,
                **_operator_evidence_template_summary(
                    payload,
                    load_template=lambda template_path: _load_operator_evidence_template_from_directory(
                        path,
                        template_path,
                    ),
                ),
            }
            summary["operator_validation_command_templates"] = _operator_validation_command_templates(
                [
                    str(gate_id)
                    for gate_id in [
                        *(
                            summary.get("operator_evidence_pending_gate_ids")
                            if isinstance(summary.get("operator_evidence_pending_gate_ids"), list)
                            else []
                        ),
                        *(
                            summary.get("operator_evidence_needs_review_gate_ids")
                            if isinstance(summary.get("operator_evidence_needs_review_gate_ids"), list)
                            else []
                        ),
                        *(
                            summary.get("operator_evidence_template_missing_gate_ids")
                            if isinstance(summary.get("operator_evidence_template_missing_gate_ids"), list)
                            else []
                        ),
                        *(
                            summary.get("operator_evidence_template_pending_gate_ids")
                            if isinstance(summary.get("operator_evidence_template_pending_gate_ids"), list)
                            else []
                        ),
                        *(
                            summary.get("passed_gates_with_unapproved_evidence_templates")
                            if isinstance(
                                summary.get("passed_gates_with_unapproved_evidence_templates"),
                                list,
                            )
                            else []
                        ),
                    ]
                    if str(gate_id).strip()
                ]
            )
        return {
            "required": True,
            "present": required_path.exists(),
            "valid": bool(payload is not None and not error),
            "required_path": str(required_path),
            "matched_paths": [str(required_path)] if required_path.exists() else [],
            "related_paths": related_paths,
            "error": error,
            "operator_validation_visibility_policy": _operator_validation_visibility_policy(),
            **_implementation_status_artifact_summary(
                payload,
                required_path=str(implementation_status_required_path),
                matched_paths=implementation_status_matched_paths,
                related_paths=implementation_status_related_paths,
            ),
            **summary,
        }
    if path.is_file() and path.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name == required_name or name.endswith(f"/{required_name}")
            )
            top_level_names = [
                name
                for name in names
                if name == required_name or len([part for part in name.split("/") if part]) == 2
            ]
            payload = None
            error = ""
            if top_level_names:
                try:
                    loaded = json.loads(archive.read(top_level_names[0]).decode("utf-8"))
                    payload = loaded if isinstance(loaded, dict) else {}
                except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    error = str(exc)
            implementation_status_name = _implementation_status_artifact_name(payload)
            implementation_status_names = sorted(
                name
                for name in archive.namelist()
                if name == implementation_status_name or name.endswith(f"/{implementation_status_name}")
            )
            implementation_status_top_level_names = [
                name
                for name in implementation_status_names
                if name == implementation_status_name
                or len([part for part in name.split("/") if part]) == 2
            ]
            summary = _validation_checklist_gate_summary(payload) if payload is not None else {}
            if payload is not None:
                summary = {
                    **summary,
                    **_operator_evidence_template_summary(
                        payload,
                        load_template=lambda template_path: _load_operator_evidence_template_from_zip(
                            archive,
                            template_path,
                        ),
                    ),
                }
                summary["operator_validation_command_templates"] = _operator_validation_command_templates(
                    [
                        str(gate_id)
                        for gate_id in [
                            *(
                                summary.get("operator_evidence_pending_gate_ids")
                                if isinstance(summary.get("operator_evidence_pending_gate_ids"), list)
                                else []
                            ),
                            *(
                                summary.get("operator_evidence_needs_review_gate_ids")
                                if isinstance(summary.get("operator_evidence_needs_review_gate_ids"), list)
                                else []
                            ),
                            *(
                                summary.get("operator_evidence_template_missing_gate_ids")
                                if isinstance(summary.get("operator_evidence_template_missing_gate_ids"), list)
                                else []
                            ),
                            *(
                                summary.get("operator_evidence_template_pending_gate_ids")
                                if isinstance(summary.get("operator_evidence_template_pending_gate_ids"), list)
                                else []
                            ),
                            *(
                                summary.get("passed_gates_with_unapproved_evidence_templates")
                                if isinstance(
                                    summary.get("passed_gates_with_unapproved_evidence_templates"),
                                    list,
                                )
                                else []
                            ),
                        ]
                        if str(gate_id).strip()
                    ]
                )
        return {
            "required": True,
            "present": bool(top_level_names),
            "valid": bool(payload is not None and not error),
            "required_path": f"*/{required_name}",
            "matched_paths": top_level_names,
            "related_paths": names,
            "error": error,
            "operator_validation_visibility_policy": _operator_validation_visibility_policy(),
            **_implementation_status_artifact_summary(
                payload,
                required_path=f"*/{implementation_status_name}",
                matched_paths=implementation_status_top_level_names,
                related_paths=implementation_status_names,
            ),
            **summary,
        }
    return {
        "required": True,
        "present": False,
        "valid": False,
        "required_path": required_name,
        "matched_paths": [],
        "related_paths": [],
        "error": "",
        "operator_validation_visibility_policy": _operator_validation_visibility_policy(),
        **_implementation_status_artifact_summary(
            None,
            required_path="implementation-status.txt",
            matched_paths=[],
            related_paths=[],
        ),
    }

