"""Read-only labeler-facing JSON routes served through the Flask strangler."""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlparse

from flask import Flask, Response, request

from .admin_dashboard import (
    _assignment_ownership_contract_fields,
    _assignment_ownership_policy,
    _dashboard_operator_validation_fields,
    _dataset_queue_direct_start_policy,
    _known_labeler_status,
    _labeler_route_authorization_policy,
    _labeler_route_authorization_runtime_checklist,
    _labeler_safety_policy,
    _mutation_audit_policy,
    _operator_validation_gate_flat_fields,
    _operator_validation_gate_metadata_fields,
    _operator_validation_public_fields,
    _queue_first_entry_contract_policy,
    _runtime_operator_validation_gate_cli_policy,
    _runtime_operator_validation_mutation_gate,
    _runtime_operator_validation_start_gate,
    _single_owner_assignment_live_contract_fields,
    _single_owner_policy_fields,
    _store_consistency_report,
    _zarr_backup_policy,
)
from .web_app import claimed_route
from .web_auth import (
    DASHBOARD_PATH,
    DATASET_QUEUE_PATH,
    PERSONAL_DATASET_QUEUE_PATH,
    _dashboard_url_for_expected_user,
    _is_admin_user,
    _resolve_user,
)
from .web_auth_errors import _authentication_required_error_details
from .web_authorization_metadata import _labeler_read_authorization_denial_metadata
from .web_diagnostics import _add_payload_contract_compact_fields
from .web_identity import (
    _identity_probe_payload,
    _mark_identity_probe_unknown_labeling_user,
)
from .web_policy import (
    IDENTITY_PROBE_PATH,
    LABELING_HOME_PATH,
    PERSONAL_WORK_PATH,
    _browser_mutation_write_policy,
    _browser_mutation_write_runtime_checklist,
    _browser_response_security_policy,
    _browser_signed_link_policy,
    _browser_task_state_policy,
    _browser_workflow_capabilities,
    _browser_workflow_kinds,
    _browser_workflow_scope_runtime_checklist,
    _operator_validation_visibility_fields,
    _operator_validation_visibility_policy,
    _session_guard_policy,
)
from .web_responses import _format_error, _json_response
from .web_runtimes import _labeler_safe_error_details
from .work_queue import (
    _add_direct_start_contracts_to_work_tasks,
    _add_work_summary_fields,
    _labeler_work_completion_fields,
    _reassignment_session_safety_flat_fields,
)

PersonalApiResponder = Callable[..., tuple[dict[str, object], HTTPStatus]]


def _personal_api_response_payload(
    state: Any,
    *,
    path: str,
    request_adapter: object,
) -> tuple[dict[str, object], HTTPStatus]:
    user, auth_source = _resolve_user(request_adapter, state.config)  # type: ignore[arg-type]
    if user is None:
        return (
            _format_error(
                "authentication_required",
                details=_authentication_required_error_details(auth_source, state.config),
                status=HTTPStatus.UNAUTHORIZED,
            ),
            HTTPStatus.UNAUTHORIZED,
        )

    request_path = str(getattr(request_adapter, "path", path) or path)
    query = parse_qs(urlparse(request_path).query, keep_blank_values=True)

    if path == "/api/me/identity":
        expected_user = str((query.get("expected_user") or [""])[-1]).strip()
        payload = _identity_probe_payload(
            user=user,
            auth_source=auth_source,
            expected_user=expected_user,
            config=state.config,
            store=state.store,
        )
        known_user_status = _known_labeler_status(state.store, user)
        payload["known_user_status"] = known_user_status
        if bool(payload.get("ok")) and not bool(known_user_status.get("is_known_labeler")):
            _mark_identity_probe_unknown_labeling_user(payload)
        return payload, HTTPStatus.OK if bool(payload.get("ok")) else HTTPStatus.FORBIDDEN

    if path not in {"/api/me/tasks", "/api/me/datasets"}:
        return (
            _format_error("not_found", status=HTTPStatus.NOT_FOUND),
            HTTPStatus.NOT_FOUND,
        )

    expected_user = str((query.get("expected_user") or [""])[-1]).strip()
    if expected_user and str(user) != expected_user:
        return (
            _format_error(
                "dashboard_user_mismatch",
                details=(
                    f"This work API request is for {expected_user}, "
                    f"but the browser is authenticated as {user}. "
                    "Stop and contact the operator before labeling."
                ),
                status=HTTPStatus.FORBIDDEN,
                extra=_labeler_read_authorization_denial_metadata(
                    user=user,
                    expected_user=expected_user,
                    route_path=path,
                    response_kind="json",
                ),
            ),
            HTTPStatus.FORBIDDEN,
        )

    known_user_status = _known_labeler_status(state.store, user)
    if not bool(known_user_status.get("is_active_labeling_user")):
        check_report = _store_consistency_report(state.store)
        assignment_ownership_integrity = (
            check_report.get("assignment_ownership_integrity")
            if isinstance(
                check_report.get("assignment_ownership_integrity"),
                Mapping,
            )
            else {}
        )
        single_owner_live_contract_fields = (
            _single_owner_assignment_live_contract_fields(
                state.store,
                integrity=assignment_ownership_integrity,
            )
        )
        labeler_route_authorization_policy = _labeler_route_authorization_policy()
        guarded_user = expected_user or str(user)
        labeler_safety = _labeler_safety_policy()
        expected_user_labeler_landing_url = _dashboard_url_for_expected_user(
            "/",
            guarded_user,
        )
        expected_user_dashboard_url = _dashboard_url_for_expected_user(
            DASHBOARD_PATH,
            guarded_user,
        )
        expected_user_dataset_queue_url = _dashboard_url_for_expected_user(
            DATASET_QUEUE_PATH,
            guarded_user,
        )
        expected_user_personal_work_url = _dashboard_url_for_expected_user(
            PERSONAL_WORK_PATH,
            guarded_user,
        )
        expected_user_personal_dataset_queue_url = _dashboard_url_for_expected_user(
            PERSONAL_DATASET_QUEUE_PATH,
            guarded_user,
        )
        preferred_labeler_entry_url = (
            expected_user_personal_dataset_queue_url
            or expected_user_dataset_queue_url
            or expected_user_labeler_landing_url
        )
        browser_mutation_write_policy = _browser_mutation_write_policy()
        browser_mutation_write_checklist = (
            _browser_mutation_write_runtime_checklist(
                browser_mutation_write_policy
            )
        )
        dataset_queue_direct_start_policy = _dataset_queue_direct_start_policy()
        unknown_user_payload = _format_error(
            "unknown_labeling_user"
            if not bool(known_user_status.get("registry_row_present"))
            else "inactive_labeling_user",
            details=(
                "This browser identity is not active in the labeling_users SQLite table. "
                "Ask the operator to add or activate this user before labeling."
            ),
            status=HTTPStatus.FORBIDDEN,
            extra={
                "known_user_status": known_user_status,
                "expected_user": expected_user,
                "labeler_landing_page_path": "/",
                "dashboard_path": DASHBOARD_PATH,
                "dataset_queue_page_path": DATASET_QUEUE_PATH,
                "personal_work_page_path": PERSONAL_WORK_PATH,
                "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
                "expected_user_labeler_landing_url": expected_user_labeler_landing_url,
                "labeling_home_page_path": LABELING_HOME_PATH,
                "expected_user_labeling_home_url": _dashboard_url_for_expected_user(
                    LABELING_HOME_PATH,
                    guarded_user,
                ),
                "expected_user_dashboard_url": expected_user_dashboard_url,
                "expected_user_dataset_queue_url": expected_user_dataset_queue_url,
                "expected_user_personal_work_url": expected_user_personal_work_url,
                "expected_user_personal_dataset_queue_url": (
                    expected_user_personal_dataset_queue_url
                ),
                "expected_user_identity_probe_url": (
                    _dashboard_url_for_expected_user(
                        IDENTITY_PROBE_PATH,
                        guarded_user,
                    )
                ),
                "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
                "preferred_labeler_entry_url": preferred_labeler_entry_url,
                "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
                "personalized_labeler_entry_url": (
                    expected_user_personal_dataset_queue_url
                ),
                "labeler_landing_link_role": "queue_first_start",
                "personal_dataset_queue_link_role": "preferred_queue",
                "dataset_queue_link_role": "canonical_queue_fallback",
                "canonical_dataset_queue_link_role": "canonical_queue_fallback",
                "dashboard_link_role": "fallback_dashboard",
                "identity_probe_link_role": "identity_check",
                "task_links_role": "convenience_entry_hints",
                "single_owner_policy": _assignment_ownership_policy(),
                **_single_owner_policy_fields(_assignment_ownership_policy()),
                **single_owner_live_contract_fields,
                "preferred_labeler_entry_url_matches_personal_dataset_queue": bool(
                    expected_user_personal_dataset_queue_url
                    and preferred_labeler_entry_url
                    == expected_user_personal_dataset_queue_url
                ),
                "personalized_labeler_entry_url_matches_personal_dataset_queue": bool(
                    expected_user_personal_dataset_queue_url
                ),
                "labeler_safety": labeler_safety,
                "queue_first_entry_contract": _queue_first_entry_contract_policy(
                    labeler_safety=labeler_safety,
                    labeler_landing_page_path="/",
                    labeler_landing_url="/",
                    expected_user_labeler_landing_url=(
                        expected_user_labeler_landing_url
                    ),
                    labeling_home_page_path=LABELING_HOME_PATH,
                    labeling_home_url=LABELING_HOME_PATH,
                    expected_user_labeling_home_url=_dashboard_url_for_expected_user(
                        LABELING_HOME_PATH,
                        guarded_user,
                    ),
                    dataset_queue_page_path=DATASET_QUEUE_PATH,
                    dataset_queue_url=DATASET_QUEUE_PATH,
                    expected_user_dataset_queue_url=(
                        expected_user_dataset_queue_url
                    ),
                    dashboard_url=DASHBOARD_PATH,
                    expected_user_dashboard_url=expected_user_dashboard_url,
                    personal_dataset_queue_page_path=(
                        PERSONAL_DATASET_QUEUE_PATH
                    ),
                    personal_dataset_queue_url=(
                        PERSONAL_DATASET_QUEUE_PATH
                    ),
                    expected_user_personal_dataset_queue_url=(
                        expected_user_personal_dataset_queue_url
                    ),
                    personal_work_page_path=PERSONAL_WORK_PATH,
                    personal_work_url=PERSONAL_WORK_PATH,
                    expected_user_personal_work_url=(
                        expected_user_personal_work_url
                    ),
                ),
                "labeler_route_authorization_policy": labeler_route_authorization_policy,
                "labeler_route_authorization_checklist": (
                    _labeler_route_authorization_runtime_checklist(
                        policy=labeler_route_authorization_policy,
                        user=user,
                        expected_user=expected_user,
                        known_user_status=known_user_status,
                        assignment_ownership_contract=single_owner_live_contract_fields[
                            "assignment_ownership_contract"
                        ],
                    )
                ),
                "browser_mutation_write_policy": browser_mutation_write_policy,
                "browser_mutation_write_checklist": browser_mutation_write_checklist,
                "dataset_queue_direct_start_policy": dataset_queue_direct_start_policy,
            },
        )
        _add_payload_contract_compact_fields(unknown_user_payload)
        return unknown_user_payload, HTTPStatus.FORBIDDEN

    try:
        include_completed = str((query.get("include_completed") or [""])[0]).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        work = state.store.task_summary_for_user(user, include_completed=include_completed)
        work["include_completed"] = include_completed
        work["auth_source"] = auth_source
        work["expected_user"] = expected_user
        work["known_user_status"] = known_user_status
        single_owner_policy = _assignment_ownership_policy()
        work["single_owner_policy"] = single_owner_policy
        work.update(_single_owner_policy_fields(single_owner_policy))
        guarded_user = expected_user or str(user)
        work["labeler_landing_page_path"] = "/"
        work["dashboard_path"] = DASHBOARD_PATH
        work["dataset_queue_page_path"] = DATASET_QUEUE_PATH
        work["personal_work_page_path"] = PERSONAL_WORK_PATH
        work["personal_dataset_queue_page_path"] = PERSONAL_DATASET_QUEUE_PATH
        work["expected_user_labeler_landing_url"] = _dashboard_url_for_expected_user("/", guarded_user)
        work["expected_user_labeling_home_url"] = _dashboard_url_for_expected_user(LABELING_HOME_PATH, guarded_user)
        work["expected_user_dashboard_url"] = _dashboard_url_for_expected_user(DASHBOARD_PATH, guarded_user)
        work["expected_user_dataset_queue_url"] = _dashboard_url_for_expected_user(DATASET_QUEUE_PATH, guarded_user)
        work["expected_user_personal_work_url"] = _dashboard_url_for_expected_user(PERSONAL_WORK_PATH, guarded_user)
        work["expected_user_personal_dataset_queue_url"] = _dashboard_url_for_expected_user(PERSONAL_DATASET_QUEUE_PATH, guarded_user)
        work["expected_user_identity_probe_url"] = _dashboard_url_for_expected_user(IDENTITY_PROBE_PATH, guarded_user)
        work["preferred_labeler_entrypoint"] = "personal_datasets_waiting_queue"
        work["preferred_labeler_entry_url"] = work[
            "expected_user_personal_dataset_queue_url"
        ]
        work["personalized_labeler_entrypoint"] = "personal_datasets_waiting_queue"
        work["personalized_labeler_entry_url"] = work["expected_user_personal_dataset_queue_url"]
        work["labeler_landing_link_role"] = "queue_first_start"
        work["personal_dataset_queue_link_role"] = "preferred_queue"
        work["dataset_queue_link_role"] = "canonical_queue_fallback"
        work["canonical_dataset_queue_link_role"] = "canonical_queue_fallback"
        work["preferred_labeler_entry_url_matches_dataset_queue"] = bool(
            work["preferred_labeler_entry_url"]
            and work["preferred_labeler_entry_url"]
            in {
                work["expected_user_personal_dataset_queue_url"],
                work["expected_user_dataset_queue_url"],
            }
        )
        work["preferred_labeler_entry_url_matches_personal_dataset_queue"] = bool(
            work["preferred_labeler_entry_url"]
            and work["preferred_labeler_entry_url"]
            == work["expected_user_personal_dataset_queue_url"]
        )
        work[
            "personalized_labeler_entry_url_matches_personal_dataset_queue"
        ] = bool(
            work["personalized_labeler_entry_url"]
            and work["personalized_labeler_entry_url"]
            == work["expected_user_personal_dataset_queue_url"]
        )
        work["dashboard_link_role"] = "fallback_dashboard"
        work["identity_probe_link_role"] = "identity_check"
        work["task_links_role"] = "convenience_entry_hints"
        work["is_admin"] = _is_admin_user(user, state.config)
        check_report = _store_consistency_report(state.store)
        _add_work_summary_fields(
            work,
            reassignment_session_safety=check_report.get("reassignment_session_safety", {}),
        )
        work["labeler_safety"] = _labeler_safety_policy()
        work["queue_first_entry_contract"] = _queue_first_entry_contract_policy(
            labeler_safety=work["labeler_safety"],
            labeler_landing_page_path=str(
                work.get("labeler_landing_page_path") or "/"
            ),
            labeler_landing_url=str(work.get("labeler_landing_url") or "/"),
            expected_user_labeler_landing_url=str(
                work.get("expected_user_labeler_landing_url") or ""
            ),
            dataset_queue_page_path=str(
                work.get("dataset_queue_page_path") or DATASET_QUEUE_PATH
            ),
            dataset_queue_url=str(
                work.get("dataset_queue_url") or DATASET_QUEUE_PATH
            ),
            expected_user_dataset_queue_url=str(
                work.get("expected_user_dataset_queue_url") or ""
            ),
            dashboard_url=str(work.get("dashboard_url") or DASHBOARD_PATH),
            expected_user_dashboard_url=str(
                work.get("expected_user_dashboard_url") or ""
            ),
            personal_dataset_queue_page_path=str(
                work.get("personal_dataset_queue_page_path")
                or PERSONAL_DATASET_QUEUE_PATH
            ),
            personal_dataset_queue_url=PERSONAL_DATASET_QUEUE_PATH,
            expected_user_personal_dataset_queue_url=str(
                work.get("expected_user_personal_dataset_queue_url") or ""
            ),
            personal_work_page_path=str(
                work.get("personal_work_page_path") or PERSONAL_WORK_PATH
            ),
            personal_work_url=PERSONAL_WORK_PATH,
            expected_user_personal_work_url=str(
                work.get("expected_user_personal_work_url") or ""
            ),
        )
        labeler_route_authorization_policy = _labeler_route_authorization_policy()
        work["labeler_route_authorization_policy"] = labeler_route_authorization_policy
        work["zarr_backup_policy"] = _zarr_backup_policy()
        work["mutation_audit_policy"] = _mutation_audit_policy()
        browser_mutation_write_policy = _browser_mutation_write_policy()
        work["browser_mutation_write_policy"] = browser_mutation_write_policy
        work["browser_mutation_write_checklist"] = (
            _browser_mutation_write_runtime_checklist(browser_mutation_write_policy)
        )
        work["dataset_queue_direct_start_policy"] = _dataset_queue_direct_start_policy()
        work["runtime_operator_validation_gate_cli_policy"] = (
            _runtime_operator_validation_gate_cli_policy()
        )
        work["single_owner_policy"] = _assignment_ownership_policy()
        work["assignment_ownership_integrity"] = check_report.get(
            "assignment_ownership_integrity",
            {},
        )
        work.update(
            _single_owner_assignment_live_contract_fields(
                state.store,
                integrity=work["assignment_ownership_integrity"]
                if isinstance(
                    work.get("assignment_ownership_integrity"), Mapping
                )
                else None,
            )
        )
        work["labeler_route_authorization_checklist"] = (
            _labeler_route_authorization_runtime_checklist(
                policy=labeler_route_authorization_policy,
                user=user,
                expected_user=expected_user,
                known_user_status=known_user_status,
                assignment_ownership_contract=work[
                    "assignment_ownership_contract"
                ],
            )
        )
        _add_payload_contract_compact_fields(work)
        _add_direct_start_contracts_to_work_tasks(
            work,
            expected_user=guarded_user,
            reassignment_session_safety=check_report.get(
                "reassignment_session_safety", {}
            ),
        )
        work["browser_response_security_policy"] = _browser_response_security_policy()
        work["session_guard_policy"] = _session_guard_policy()
        work["task_state_policy"] = _browser_task_state_policy()
        work["signed_link_policy"] = _browser_signed_link_policy()
        work["browser_workflows"] = _browser_workflow_capabilities()
        work["supported_browser_workflow_kinds"] = _browser_workflow_kinds()
        work["browser_workflow_scope_checklist"] = (
            _browser_workflow_scope_runtime_checklist(
                task_state_policy=work["task_state_policy"],
                browser_workflows=work["browser_workflows"],
            )
        )
        operator_validation_public_fields = _operator_validation_public_fields(
            _dashboard_operator_validation_fields()
        )
        work.update(operator_validation_public_fields)
        work.update(_operator_validation_gate_metadata_fields())
        work.update(_operator_validation_gate_flat_fields(work))
        _add_payload_contract_compact_fields(work)
        work["operator_validation_visibility_policy"] = (
            _operator_validation_visibility_policy()
        )
        work["operator_validation_start_gate"] = (
            _runtime_operator_validation_start_gate(state.config)
        )
        work["operator_validation_mutation_gate"] = (
            _runtime_operator_validation_mutation_gate(state.config)
        )
    except Exception as exc:
        return (
            _format_error(
                "task_query_failed",
                details=_labeler_safe_error_details(exc),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    if path == "/api/me/datasets":
        return (
            {
                "ok": True,
                "user": user,
                "auth_source": auth_source,
                "expected_user": expected_user,
                "labeler_landing_page_path": work["labeler_landing_page_path"],
                "dashboard_path": work["dashboard_path"],
                "dataset_queue_page_path": work["dataset_queue_page_path"],
                "personal_work_page_path": work["personal_work_page_path"],
                "personal_dataset_queue_page_path": work["personal_dataset_queue_page_path"],
                "expected_user_labeler_landing_url": work["expected_user_labeler_landing_url"],
                "expected_user_labeling_home_url": work[
                    "expected_user_labeling_home_url"
                ],
                "expected_user_dashboard_url": work["expected_user_dashboard_url"],
                "expected_user_dataset_queue_url": work["expected_user_dataset_queue_url"],
                "expected_user_personal_work_url": work["expected_user_personal_work_url"],
                "expected_user_personal_dataset_queue_url": work[
                    "expected_user_personal_dataset_queue_url"
                ],
                "expected_user_identity_probe_url": work["expected_user_identity_probe_url"],
                "preferred_labeler_entrypoint": work["preferred_labeler_entrypoint"],
                "preferred_labeler_entry_url": work["preferred_labeler_entry_url"],
                "personalized_labeler_entrypoint": work["personalized_labeler_entrypoint"],
                "personalized_labeler_entry_url": work["personalized_labeler_entry_url"],
                "personalized_launch_readiness": work[
                    "personalized_launch_readiness"
                ],
                "preferred_labeler_entry_url_matches_dataset_queue": work[
                    "preferred_labeler_entry_url_matches_dataset_queue"
                ],
                "preferred_labeler_entry_url_matches_personal_dataset_queue": work[
                    "preferred_labeler_entry_url_matches_personal_dataset_queue"
                ],
                "personalized_labeler_entry_url_matches_personal_dataset_queue": work[
                    "personalized_labeler_entry_url_matches_personal_dataset_queue"
                ],
                "labeler_landing_link_role": work["labeler_landing_link_role"],
                "personal_dataset_queue_link_role": work[
                    "personal_dataset_queue_link_role"
                ],
                "dataset_queue_link_role": work["dataset_queue_link_role"],
                "canonical_dataset_queue_link_role": work[
                    "canonical_dataset_queue_link_role"
                ],
                "dashboard_link_role": work["dashboard_link_role"],
                "identity_probe_link_role": work["identity_probe_link_role"],
                "task_links_role": work["task_links_role"],
                "include_completed": include_completed,
                "known_user_status": work["known_user_status"],
                "single_owner_policy": work["single_owner_policy"],
                **_single_owner_policy_fields(work["single_owner_policy"]),
                "assignment_ownership_integrity": work[
                    "assignment_ownership_integrity"
                ],
                "single_owner_assignment_contract": work[
                    "single_owner_assignment_contract"
                ],
                "assignment_ownership_contract": work[
                    "assignment_ownership_contract"
                ],
                **_assignment_ownership_contract_fields(
                    work["assignment_ownership_contract"]
                ),
                "single_owner_policy_contract_met": work[
                    "single_owner_policy_contract_met"
                ],
                "empty_state": work["empty_state"],
                "progress_summary": work["progress_summary"],
                "dataset_queue_summary": work["dataset_queue_summary"],
                "direct_browser_start_contract_summary": work[
                    "direct_browser_start_contract_summary"
                ],
                "dataset_queue_state": work["dataset_queue_state"],
                "labeler_work_completion": work.get("labeler_work_completion", {}),
                **_labeler_work_completion_fields(
                    work.get("labeler_work_completion")
                    if isinstance(work.get("labeler_work_completion"), Mapping)
                    else None
                ),
                "reassignment_session_safety": work["reassignment_session_safety"],
                **_reassignment_session_safety_flat_fields(
                    work["reassignment_session_safety"]
                ),
                "labeler_start_ready": work["labeler_start_ready"],
                "labeler_start_status": work["labeler_start_status"],
                "labeler_action": work["labeler_action"],
                "labeler_start_message": work["labeler_start_message"],
                "labeler_start_operator_action": work["labeler_start_operator_action"],
                **_operator_validation_public_fields(work),
                **_operator_validation_visibility_fields(),
                **_operator_validation_gate_metadata_fields(),
                **_operator_validation_gate_flat_fields(work),
                "operator_validation_visibility_policy": work[
                    "operator_validation_visibility_policy"
                ],
                "operator_validation_start_gate": work[
                    "operator_validation_start_gate"
                ],
                "operator_validation_mutation_gate": work[
                    "operator_validation_mutation_gate"
                ],
                "dataset_queue": work["dataset_queue"],
                "datasets": work["dataset_queue"],
                "labeler_safety": work["labeler_safety"],
                "queue_first_entry_contract": work["queue_first_entry_contract"],
                "labeler_route_authorization_policy": work["labeler_route_authorization_policy"],
                "labeler_route_authorization_checklist": work[
                    "labeler_route_authorization_checklist"
                ],
                "zarr_backup_policy": work["zarr_backup_policy"],
                "mutation_audit_policy": work["mutation_audit_policy"],
                "browser_mutation_write_policy": work["browser_mutation_write_policy"],
                "browser_mutation_write_checklist": work[
                    "browser_mutation_write_checklist"
                ],
                "dataset_queue_direct_start_policy": work["dataset_queue_direct_start_policy"],
                "runtime_operator_validation_gate_cli_policy": work[
                    "runtime_operator_validation_gate_cli_policy"
                ],
                "browser_response_security_policy": work[
                    "browser_response_security_policy"
                ],
                "session_guard_policy": work["session_guard_policy"],
                "task_state_policy": work["task_state_policy"],
                "signed_link_policy": work["signed_link_policy"],
                "browser_workflows": work["browser_workflows"],
                "supported_browser_workflow_kinds": work[
                    "supported_browser_workflow_kinds"
                ],
                "browser_workflow_scope_checklist": work[
                    "browser_workflow_scope_checklist"
                ],
            },
            HTTPStatus.OK,
        )

    return (
        {
            "ok": True,
            "preferred_labeler_entry_url_matches_dataset_queue": work[
                "preferred_labeler_entry_url_matches_dataset_queue"
            ],
            "preferred_labeler_entry_url_matches_personal_dataset_queue": work[
                "preferred_labeler_entry_url_matches_personal_dataset_queue"
            ],
            "personalized_labeler_entry_url_matches_personal_dataset_queue": work[
                "personalized_labeler_entry_url_matches_personal_dataset_queue"
            ],
            "personalized_launch_readiness": work[
                "personalized_launch_readiness"
            ],
            "work": work,
            "operator_validation_start_gate": work[
                "operator_validation_start_gate"
            ],
            "operator_validation_mutation_gate": work[
                "operator_validation_mutation_gate"
            ],
            "runtime_operator_validation_gate_cli_policy": work[
                "runtime_operator_validation_gate_cli_policy"
            ],
            "single_owner_policy": work["single_owner_policy"],
            **_single_owner_policy_fields(work["single_owner_policy"]),
            "assignment_ownership_integrity": work[
                "assignment_ownership_integrity"
            ],
            "single_owner_assignment_contract": work[
                "single_owner_assignment_contract"
            ],
            "assignment_ownership_contract": work[
                "assignment_ownership_contract"
            ],
            **_assignment_ownership_contract_fields(
                work["assignment_ownership_contract"]
            ),
            "single_owner_policy_contract_met": work[
                "single_owner_policy_contract_met"
            ],
            "reassignment_session_safety": work["reassignment_session_safety"],
            **_reassignment_session_safety_flat_fields(
                work["reassignment_session_safety"]
            ),
        },
        HTTPStatus.OK,
    )

def _json(payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> Response:
    data, response_status, content_type = _json_response(payload, status=status)
    return Response(data, status=int(response_status), content_type=content_type)


def _request_adapter() -> SimpleNamespace:
    path = request.full_path
    if path.endswith("?"):
        path = request.path
    return SimpleNamespace(headers=request.headers, path=path)


def _respond(
    state: Any,
    response_builder: PersonalApiResponder,
    *,
    path: str,
) -> Response:
    payload, status = response_builder(
        state,
        path=path,
        request_adapter=_request_adapter(),
    )
    return _json(payload, status=status)


def register_personal_api_routes(
    app: Flask,
    state: Any,
    response_builder: PersonalApiResponder,
) -> None:
    """Register read-only personal work JSON endpoints on ``app``."""

    @claimed_route(app, "/api/me/identity", methods=["GET"])
    def personal_identity() -> Response:
        return _respond(state, response_builder, path="/api/me/identity")

    @claimed_route(app, "/api/me/tasks", methods=["GET"])
    def personal_tasks() -> Response:
        return _respond(state, response_builder, path="/api/me/tasks")

    @claimed_route(app, "/api/me/datasets", methods=["GET"])
    def personal_datasets() -> Response:
        return _respond(state, response_builder, path="/api/me/datasets")


__all__ = ["_personal_api_response_payload", "register_personal_api_routes"]
