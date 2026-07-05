"""Serve and manage recording-assigned web labeling work."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import hmac
import io
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np

from .assignment_store import (
    LABELER_START_TASK_STATES,
    LABELING_USER_ROLES,
    LABELING_USER_STATUSES,
    LabelingStore,
    default_store_path,
)
from .admin_registry import (
    REGISTRY_PATH_ENV_VAR,
    _admin_compact_task,
    _admin_dataset_export_csv,
    _admin_dataset_export_rows,
    _admin_registry_lookup,
    _admin_registry_path_from_env,
    _admin_registry_summary,
    _admin_registry_warnings_for_recording,
    _admin_task_state_counts,
    _admin_workflow_counts,
    _task_title,
)
from . import web_batch_readiness as _web_batch_readiness
from . import web_operator_evidence_templates as _web_operator_evidence_templates
from .web_operator_evidence_records import (
    BROWSER_RESPONSE_SECURITY_HEADER_CHECK_KEYS,
    BROWSER_SMOKE_REQUIRED_FIELDS,
    DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS,
    _browser_smoke_personalized_route_contract,
    _disposable_zarr_smoke_workflow_contract_missing_fields,
    _identity_source_personal_queue_status,
    _identity_source_row_approved,
    _operator_evidence_truthy,
    _parse_header_evidence_values,
    _record_browser_response_security_evidence,
    _record_browser_smoke_evidence,
    _record_disposable_zarr_mutation_smoke_evidence,
    _record_identity_source_evidence,
)
from . import web_zarr_backup as _web_zarr_backup
from . import web_assignment_freshness as _web_assignment_freshness
from .web_auth import (
    DASHBOARD_PATH,
    DATASET_QUEUE_PATH,
    PERSONAL_DATASET_QUEUE_PATH,
    SIGNED_INVITE_COOKIE_NAME,
    SIGNED_INVITE_SCOPE_PERSONAL_QUEUE,
    _b64url_decode,
    _cookie_value,
    _dashboard_url_for_expected_user,
    _dataset_queue_url_for_base,
    _dataset_queue_url_for_dashboard,
    _expected_user_query_value_from_url,
    _invite_query_token_from_request,
    _invite_token_from_request,
    _is_admin_user,
    _personal_dataset_queue_url_for_base,
    _personal_dataset_queue_url_for_dashboard,
    _resolve_invite_user,
    _resolve_user,
    _signed_task_link_revocation_reason,
    _utc_timestamp,
    _verify_signed_invite_token,
)
from .web_app import create_labeling_app
from .web_admin_api import register_admin_api_routes
from .web_admin_pages import _admin_page_response_payload, register_admin_page_routes
from .web_admin_renderers import (
    _admin_datasets_html,
    _admin_html,
    _admin_recording_html,
    _admin_task_html,
    _admin_user_html,
    _admin_users_html,
)
from .web_auth_errors import _authentication_required_error_details
from .web_authorization_metadata import (
    _add_task_open_personalized_launch_metadata,
    _browser_mutation_failure_metadata,
    _browser_mutation_response_metadata,
    _labeler_authorization_context,
    _labeler_read_authorization_denial_metadata,
    _task_completion_authorization_contract,
    _task_completion_failure_metadata,
    _task_open_authorization_contract,
    _task_open_failure_metadata,
    _task_open_response_metadata,
)
from .web_error_pages import _browser_error_html
from .web_diagnostics import (
    _add_payload_contract_compact_fields,
    _browser_mutation_target_contract_compact_fields,
    _browser_mutation_target_contract_source_from_checklist,
    _direct_browser_start_contract_compact_fields,
    _personalized_launch_readiness_field_names,
    _personalized_launch_readiness_summary,
    _queue_first_entry_contract_flat_fields,
)
from .web_identity import (
    _identity_probe_html,
    _identity_probe_payload,
    _mark_identity_probe_unknown_labeling_user,
)
from .web_personal_api import _personal_api_response_payload, register_personal_api_routes
from .web_personal_pages import _personal_page_response_payload, register_personal_page_routes
from .web_personal_renderers import _dashboard_html, _datasets_html
from .web_post_completion_queue import _post_completion_queue_metadata
from .web_promotion_retry import (
    _admin_promotion_retry_preflight_error,
    _labeler_promotion_retry_failure_metadata,
    _labeler_promotion_retry_operator_support_payload,
    _promotion_success_event_for_retry,
)
from .web_implementation_status import (
    _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT,
    _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS,
    _IMPLEMENTATION_STATUS_DOES_NOT_REPLACE_SENTENCE,
    _IMPLEMENTATION_STATUS_FILE_ADVISORY_SENTENCE,
    _IMPLEMENTATION_STATUS_FLAT_FIELD_COUNT,
    _IMPLEMENTATION_STATUS_FLAT_FIELDS,
    _IMPLEMENTATION_STATUS_INSPECT_FIELDS_ARE_SENTENCE,
    _IMPLEMENTATION_STATUS_MACHINE_READABLE_FIELDS_SENTENCE,
    _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELD_COUNT,
    _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELDS,
    _implementation_status_artifact,
    _implementation_status_artifact_name,
    _implementation_status_artifact_summary,
    _implementation_status_flat_fields_from_artifact,
    _implementation_status_inspection_target_fields,
    _implementation_status_metadata_fields,
)
from .web_launch_bundle_files import (
    _check_directory_zip_output,
    _write_directory_checksums,
    _write_directory_zip,
    _write_launch_bundle_implementation_status,
    _write_launch_bundle_inspect_command,
    _write_launch_bundle_launch_evidence_execution_checklist,
    _write_launch_bundle_operator_evidence_commands,
    _write_launch_bundle_readme,
)
from .web_handoff_files import (
    _safe_user_handoff_dir_name,
    _user_handoff_paths,
    _write_user_handoff_message,
    _write_user_handoff_quickstart,
    _write_user_handoffs_readme,
)
from .web_handoff_package_io import (
    _load_handoff_documents,
    _refresh_handoff_directory_checksums,
    _safe_checksum_relative_path,
    _sha256_bytes,
    _sha256_file,
    _verify_directory_checksums,
    _verify_handoff_checksums,
    _verify_zip_checksums,
)
from . import web_handoff_shareability as _web_handoff_shareability
from . import web_handoff_validation_refresh as _web_handoff_validation_refresh
from .web_handoff_validation import (
    _write_user_handoff_validation_log_impl,
    _write_user_handoff_validation_checklist_impl,
    _inspect_handoff_validation_log,
    _operator_evidence_command_sheet_boundary_status,
    _inspect_handoff_operator_evidence_commands,
    _launch_evidence_execution_checklist_status,
    _inspect_handoff_launch_evidence_execution_checklist,
    _load_operator_evidence_template_from_directory,
    _load_operator_evidence_template_from_zip,
    _operator_evidence_commands_public_summary,
    _launch_evidence_execution_checklist_public_summary,
    _launch_evidence_execution_checklist_inspection_target,
    _operator_evidence_template_status_impl,
    _operator_evidence_template_summary_impl,
    _inspect_handoff_validation_checklist_impl,
)
from .web_handoff_inspection import (
    _handoff_status_from_manifest as _handoff_status_from_manifest_impl,
    _inspect_handoff_package as _inspect_handoff_package_impl,
)
from .web_handoff_bundle import (
    _write_user_handoff_bundle as _write_user_handoff_bundle_impl,
)
from .web_handoff_fields import (
    _safe_share_external_launch_evidence_gap_field_names,
    _operator_validation_gate_flat_fieldnames,
    _handoff_ready_to_send as _handoff_ready_to_send_impl,
    _handoff_browser_response_security_fields as _handoff_browser_response_security_fields_impl,
    _handoff_mutation_audit_fields as _handoff_mutation_audit_fields_impl,
    _handoff_zarr_backup_fields as _handoff_zarr_backup_fields_impl,
    _handoff_task_state_policy_fields as _handoff_task_state_policy_fields_impl,
    _handoff_session_guard_fields as _handoff_session_guard_fields_impl,
    _handoff_labeler_safety_fields as _handoff_labeler_safety_fields_impl,
    _handoff_signed_link_policy_fields as _handoff_signed_link_policy_fields_impl,
    _handoff_labeler_route_authorization_fields as _handoff_labeler_route_authorization_fields_impl,
)
from .web_handoff_roster import (
    _write_user_handoffs_roster_csv as _write_user_handoffs_roster_csv_impl,
)
from . import web_report_renderers as _web_report_renderers
from .web_report_renderers import (
    _IMPLEMENTATION_STATUS_GATE_CONTRACT_SENTENCE,
    _IMPLEMENTATION_STATUS_INSPECT_FIELDS_SENTENCE,
    _IMPLEMENTATION_STATUS_NOT_LAUNCH_APPROVAL_SENTENCE,
    _IMPLEMENTATION_STATUS_SAFE_SHARE_REQUIREMENT_SENTENCE,
    _IMPLEMENTATION_STATUS_STALE_PACKAGE_FAIL_CLOSED_SENTENCE,
    _RUNTIME_ROUTE_CHECKLIST_GATE_CONTRACT_SENTENCE,
    _SHAREABILITY_COMPACT_CONTRACT_SENTENCE,
    _SHAREABILITY_COMPACT_GATE_SENTENCE,
    _SHAREABILITY_COMPACT_SELF_CHECK_SENTENCE,
    _SHAREABILITY_REPAIR_COMMAND_CONTRACTS_SENTENCE,
    _dashboard_url_for_base,
    _handoff_assignment_ownership_fields,
    _handoff_dataset_queue_blocks_labeler_start,
    _handoff_dataset_queue_start_fields,
    _handoff_dataset_queue_state,
    _handoff_entry_artifact_fields,
    _handoff_no_open_task_message,
    _handoff_relative_href,
    _html_escape,
    _safe_share_next_action_summary_text,
    _write_batch_plan_html_report,
    _write_launch_bundle_html_index,
    _write_user_handoff_html_index,
    _write_user_handoffs_html_index as _write_user_handoffs_html_index_render,
)
from . import web_validation_reports as _web_validation_reports
from .web_validation_checklist import (
    OPERATOR_EVIDENCE_VALIDATION_GATE_IDS,
    VALIDATION_GATE_STATUSES,
    _append_validation_log_evidence,
    _recompute_validation_checklist_summary,
    _update_validation_checklist_file,
    _update_validation_checklist_payload,
    _validation_checklist_gate_choices,
    _validation_gate,
    _validation_gate_blocks_invitation_semantics_fields,
)
from .web_session_renderers import (
    _BROWSER_MUTATION_STATUS_JS,
    _IMAGE_CANVAS_VIEWPORT_JS,
    _SESSION_OPERATOR_SUPPORT_CSS,
    _SESSION_OPERATOR_SUPPORT_HTML,
    _SESSION_OPERATOR_SUPPORT_JS,
    _detect_session_html,
    _keypoint_session_html,
    _session_html,
    _session_return_links_html,
    _session_return_url,
    _session_status_banner,
    _subject_mask_session_html,
    _video_detect_session_html,
)
from .web_wsgi_adapter import handle_with_flask_if_claimed
from .template_assets import read_labeling_asset, render_labeling_template
from .web_responses import (
    _decode_uint8_payload,
    _format_error,
    _is_loopback_host,
    _json_response,
    _raw_array_payload,
    _read_json_body,
    _request_has_same_origin,
)
from .web_runtimes import (
    DetectAnalysisPromotionConfig,
    DetectRuntimeSession,
    KeypointRuntimeSession,
    SubjectMaskRuntimeSession,
    VideoDetectRuntimeSession,
    BROWSER_MUTATION_TARGET_SELECTOR_KEYS,
    _advance_keypoint,
    _browser_mutation_target_selector_details,
    _browser_mutation_target_selector_keys,
    _browser_runtime_target_token,
    _detect_analysis_promotion_from_scope,
    _detect_bbox_size_hint_payload,
    _detect_runtime_state,
    _get_detect_runtime,
    _get_keypoint_runtime,
    _get_subject_mask_runtime,
    _get_video_detect_parent_frame,
    _get_video_detect_runtime,
    _next_browser_nav_position,
    _require_browser_mutation_target_token,
    _keypoint_runtime_state,
    _labeler_safe_error_details,
    _redact_labeler_runtime_payload,
    _refresh_keypoint_queue,
    _session_scope,
    _subject_mask_checkpoint_mask,
    _subject_mask_component_completion_guard,
    _subject_mask_current_payload,
    _subject_mask_edit_revision,
    _subject_mask_row_identity,
    _subject_mask_runtime_state,
    _subject_mask_source_rowset_path,
    _subject_mask_target_run_path,
    _video_detect_frame_payload,
    _video_detect_runtime_state,
)
from .web_policy import (
    BROWSER_CLIENT_AUTHORITY,
    BROWSER_MUTATION_AUDIT_PROVENANCE,
    BROWSER_MUTATION_RETRY_POLICY,
    BROWSER_WORKFLOW_SERVER_WRITE_CONTRACT,
    _browser_mutation_target_contract_policy,
    _browser_mutation_write_contract_policy,
    _browser_mutation_write_policy,
    _browser_mutation_write_runtime_checklist,
    _browser_response_security_contract_policy,
    _browser_response_security_policy,
    _browser_signed_link_policy,
    _browser_task_state_contract_policy,
    _browser_task_state_policy,
    _browser_workflow_capabilities,
    _browser_workflow_kinds,
    _browser_workflow_scope_contract_policy,
    _operator_validation_visibility_fields,
    _operator_validation_visibility_policy,
    _session_guard_policy,
    _signed_link_contract_policy,
)
from .work_queue import (
    _add_direct_start_contracts_to_work_tasks,
    _add_work_summary_fields,
    _completion_percent,
    _completion_state,
    _dataset_queue_labeler_start_fields,
    _dataset_queue_state,
    _first_dataset_queue_url,
    _labeler_work_completion_contract,
    _labeler_work_completion_fields,
    _public_reassignment_session_safety_fields,
    _reassignment_session_safety_blocks_recording,
    _reassignment_session_safety_flat_fields,
    _reassignment_session_safety_operator_action,
    _reassignment_session_safety_recording_ids,
    _recording_blocked_by_reassignment_session_safety,
    _service_absolute_url,
    _task_priority_value,
    _work_dataset_queue,
    _work_dataset_queue_summary,
    _work_dataset_queue_task,
    _work_empty_state,
    _work_filter_url,
    _work_progress_summary,
    _work_recording_ids,
)
from .admin_dashboard import (
    _admin_datasets_payload,
    _admin_recording_event_summary,
    _admin_recording_payload,
    _admin_recording_session_summary,
    _admin_summary_payload,
    _admin_user_payload,
    _admin_users_payload,
    _assignment_operator_status_rows,
    _assignment_ownership_contract_fields,
    _assignment_ownership_contract_policy,
    _assignment_ownership_integrity,
    _assignment_ownership_policy,
    _browser_mutation_target_contract_summary,
    _count_recording_values,
    _count_recordings_without_open_tasks,
    _count_recordings_without_open_tasks_by_reason,
    _count_rows_by_field,
    _dashboard_base_url,
    _dashboard_completion_state_counts,
    _dashboard_copy_intent_counts,
    _dashboard_dataset_queue_counts,
    _dashboard_identity_probe_counts,
    _dashboard_invitation_message,
    _dashboard_invite_actions,
    _dashboard_invite_reason_counts,
    _dashboard_operator_validation_fields,
    _dashboard_operator_validation_fields_for_config,
    _dashboard_ready_row_draft_metadata_fields,
    _dashboard_ready_state_counts,
    _dashboard_roster_rows,
    _dashboard_status_report,
    _dataset_queue_direct_start_policy,
    _dataset_queue_direct_start_policy_fields,
    _dataset_queue_start_readiness_from_counts,
    _direct_browser_start_contract_summary,
    _direct_browser_start_contract_summary_fields,
    _handoff_browser_mutation_write_fields,
    _handoff_known_user_status_fields,
    _handoff_operator_recovery_fields,
    _identity_personal_queue_evidence_status,
    _identity_probe_url_for_dashboard,
    _identity_source_policy,
    _known_labeler_status,
    _labeler_landing_url_for_base,
    _labeler_landing_url_for_dashboard,
    _labeler_route_authorization_policy,
    _labeler_route_authorization_runtime_checklist,
    _labeler_route_authorization_runtime_checklist_compact_fields,
    _labeler_route_authorization_runtime_checklist_contract_source_from_checklist,
    _labeler_route_authorization_runtime_checklist_contract_summary,
    _labeler_safety_policy,
    _labeling_home_url_for_base,
    _mutation_audit_policy,
    _no_open_task_reason_for_recording,
    _operator_authorization_policy,
    _operator_recovery_contract_policy,
    _operator_recovery_policy,
    _operator_validation_approval_scope_fields,
    _operator_validation_command_template_fields,
    _operator_validation_command_templates,
    _operator_validation_gate_flat_fields,
    _operator_validation_gate_metadata_fields,
    _operator_validation_invitation_fields,
    _operator_validation_pending_action,
    _operator_validation_public_fields,
    _personal_work_url_for_dashboard,
    _queue_first_entry_contract_policy,
    _recordings_without_open_tasks_actions,
    _redact_admin_recording_task,
    _runtime_operator_validation_gate_cli_policy,
    _runtime_operator_validation_gate_cli_policy_fields,
    _runtime_operator_validation_mutation_gate,
    _runtime_operator_validation_mutation_gate_not_required,
    _runtime_operator_validation_start_gate,
    _runtime_operator_validation_start_gate_not_required,
    _safe_share_checklist_field_values,
    _safe_share_checklist_gate_status_fields,
    _safe_share_checklist_gate_status_fields_from_operator_validation,
    _safe_share_external_launch_evidence_gap_fields,
    _safe_share_external_launch_evidence_gap_todo_fields,
    _safe_share_gate_flat_fields,
    _safe_share_gate_policy,
    _safe_share_launch_blocking_next_action,
    _safe_share_launch_blocking_next_action_command_fields,
    _safe_share_next_action_command_fields,
    _safe_share_next_action_detail_fields,
    _safe_share_next_action_summary_from_fields,
    _server_safety_payload,
    _session_is_expired,
    _shareability_labeler_route_authorization_runtime_checklist_fields,
    _shareability_labeler_route_authorization_runtime_checklist_gate,
    _shareability_labeler_route_authorization_runtime_checklist_required_values,
    _single_owner_assignment_live_contract_fields,
    _single_owner_policy_fields,
    _store_consistency_report,
    _unresolved_failed_promotions,
    _validation_checklist_gate_summary,
    _validation_gate_classification,
    _validation_gate_kind,
    _zarr_backup_policy,
)
from .notification_events import (
    _notification_config_from_values,
    _notification_event_type,
    _notification_exception_result,
    _request_truthy,
)
from .notifications import (
    NOTIFICATION_MODES,
    send_assignment_available_notification,
    send_labeler_added_notification,
)
from .report_io import (
    _csv_export_value,
    _print_json,
    _write_optional_json_report,
    _write_row_export,
)
from .task_generation import (
    _detect_review_status_for_zarr,
    _keypoint_review_status_for_zarr,
    _read_zarr_attrs,
    _registry_path_from_arg,
    _safe_task_id,
    _task_generation_cli_payload,
    _zarr_child_group_exists,
    _zarr_child_path_exists,
    generate_detect_analysis_tasks_from_registry,
    generate_detect_training_tasks_from_registry,
    generate_keypoint_tasks_from_registry,
    generate_subject_mask_component_tasks_from_registry,
)


LINK_SECRET_ENV_VAR = "PALETTE_LABELING_LINK_SECRET"
LINK_NOT_BEFORE_ENV_VAR = "PALETTE_LABELING_LINK_NOT_BEFORE_UTC"
DATASET_QUEUE_PATH = "/datasets"
LABELING_HOME_PATH = "/labeling"
DASHBOARD_PATH = "/work"
PERSONAL_DATASET_QUEUE_PATH = "/my-datasets"
PERSONAL_WORK_PATH = "/my-work"
SIGNED_INVITE_DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60

@dataclass(frozen=True)
class ServerConfig:
    store_path: Path
    host: str
    port: int
    fixed_user: str | None
    auth_header: str
    session_ttl_seconds: int
    trust_auth_header: bool = False
    admin_users: tuple[str, ...] = ()
    link_secret: str | None = None
    link_not_before_utc: str | None = None
    csrf_same_origin: bool = True
    access_log: bool = False
    allow_non_loopback: bool = False
    production: bool = False
    validation_checklist_path: Path | None = None
    require_operator_validation_for_start: bool = False


@dataclass
class ServerState:
    store: LabelingStore
    config: ServerConfig
    keypoint_sessions: dict[str, "KeypointRuntimeSession"] = field(default_factory=dict)
    detect_sessions: dict[str, "DetectRuntimeSession"] = field(default_factory=dict)
    video_detect_sessions: dict[str, "VideoDetectRuntimeSession"] = field(default_factory=dict)
    subject_mask_sessions: dict[str, "SubjectMaskRuntimeSession"] = field(default_factory=dict)








def _drop_runtime_sessions(state: ServerState, session_ids: Sequence[str]) -> None:
    for session_id in session_ids:
        state.keypoint_sessions.pop(str(session_id), None)
        state.detect_sessions.pop(str(session_id), None)
        state.video_detect_sessions.pop(str(session_id), None)
        state.subject_mask_sessions.pop(str(session_id), None)


def _session_closure_support(event: Mapping[str, object] | None) -> dict[str, object] | None:
    if not event:
        return None
    return {
        "event_id": str(event.get("event_id") or ""),
        "event_type": str(event.get("event_type") or ""),
        "event_user": str(event.get("user") or ""),
        "created_at_utc": str(event.get("created_at_utc") or ""),
        "task_id": str(event.get("task_id") or ""),
        "recording_id": str(event.get("recording_id") or ""),
    }


def _session_closure_error_extra(event: Mapping[str, object] | None) -> dict[str, object]:
    support = _session_closure_support(event)
    return {"session_closure_event": support} if support is not None else {}


def _open_session_ids_for_task(store: LabelingStore, task_id: str) -> list[str]:
    return [
        str(session.get("session_id") or "")
        for session in store.list_sessions(include_closed=False, limit=1000)
        if str(session.get("task_id") or "") == str(task_id)
    ]


def _closed_session_response_payload(
    store: LabelingStore,
    session_ids: Sequence[str],
    *,
    fallback_event_type: str = "",
) -> dict[str, object]:
    closed_session_ids = [str(session_id) for session_id in session_ids if str(session_id)]
    closure_events: list[dict[str, object]] = []
    for session_id in closed_session_ids:
        support = _session_closure_support(store.get_session_closure_event(session_id))
        if support is None and fallback_event_type:
            session = store.get_session(session_id)
            if isinstance(session, Mapping):
                support = {
                    "session_id": session_id,
                    "event_id": "",
                    "event_type": str(fallback_event_type),
                    "event_user": str(session.get("user") or ""),
                    "created_at_utc": str(session.get("closed_at_utc") or ""),
                    "task_id": str(session.get("task_id") or ""),
                    "recording_id": str(session.get("recording_id") or ""),
                }
        if support is not None:
            closure_events.append(support)
    return {
        "closed_session_count": len(closed_session_ids),
        "closed_session_ids": closed_session_ids,
        "session_closure_events": closure_events,
    }


def _task_open_preflight_error(
    store: LabelingStore,
    *,
    task_id: str,
    user: str,
) -> tuple[str, str | None, HTTPStatus] | None:
    task = store.get_task(task_id)
    if task is None:
        return "task_not_found", None, HTTPStatus.NOT_FOUND
    if str(task.get("assignee_user") or "") != str(user) or str(task.get("assignment_status") or "") != "active":
        return "not_assigned", "This task is not actively assigned to the current user.", HTTPStatus.FORBIDDEN
    task_state = str(task.get("state") or "")
    if task_state == "complete":
        return "task_complete", "This task is complete and must be reopened by an operator before labeling.", HTTPStatus.CONFLICT
    if task_state not in LABELER_START_TASK_STATES:
        return (
            "task_not_startable",
            f"This task is in state {task_state or 'unknown'} and cannot be opened for labeling.",
            HTTPStatus.CONFLICT,
        )
    mismatched_sessions = store.active_assignment_mismatched_sessions_for_recording(
        str(task.get("recording_id") or ""),
        limit=1,
    )
    if mismatched_sessions:
        return (
            "reassignment_session_safety_failed",
            "Stale previous-owner sessions are still open for this recording. Stop and contact the operator before labeling.",
            HTTPStatus.CONFLICT,
        )
    return None


















def _active_assignee_user_issues(
    store: LabelingStore,
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for row in rows:
        assignee_user = str(row.get("assignee_user") or row.get("user") or "").strip()
        if not assignee_user:
            continue
        status = _known_labeler_status(store, assignee_user)
        if bool(status.get("is_active_labeling_user")):
            continue
        issues.append(
            {
                "code": "inactive_or_unknown_assignee_user",
                "assignee_user": assignee_user,
                "recording_id": str(row.get("recording_id") or ""),
                **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                "assignee_user_status": status,
                "details": (
                    "Assignments can only be created or updated for users with an active "
                    "row in the labeling_users SQLite table. Add or activate the user first."
                ),
            }
        )
    return issues






def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")



def _effective_signed_link_ttl_seconds(ttl_seconds: int) -> int:
    return max(60, int(ttl_seconds))


def _signed_task_link_token_info(
    *,
    task_id: str,
    secret: str,
    ttl_seconds: int,
    expected_user: str | None = None,
) -> dict[str, object]:
    issued_at_unix = int(time.time())
    effective_ttl_seconds = _effective_signed_link_ttl_seconds(ttl_seconds)
    expires_at_unix = issued_at_unix + effective_ttl_seconds
    normalized_expected_user = str(expected_user or "").strip()
    payload = {
        "v": 1,
        "task_id": str(task_id),
        "iat": issued_at_unix,
        "exp": expires_at_unix,
        "nonce": uuid.uuid4().hex,
    }
    if normalized_expected_user:
        payload["expected_user"] = normalized_expected_user
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    signature = hmac.new(str(secret).encode("utf-8"), payload_bytes, hashlib.sha256).digest()
    token = f"{_b64url_encode(payload_bytes)}.{_b64url_encode(signature)}"
    return {
        "token": token,
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "issued_at_utc": datetime.fromtimestamp(issued_at_unix, tz=timezone.utc).isoformat(),
        "expires_at_utc": datetime.fromtimestamp(expires_at_unix, tz=timezone.utc).isoformat(),
        "ttl_seconds": effective_ttl_seconds,
        "expected_user": normalized_expected_user,
    }


def _signed_task_link_token(
    *,
    task_id: str,
    secret: str,
    ttl_seconds: int,
    expected_user: str | None = None,
) -> str:
    return str(
        _signed_task_link_token_info(
            task_id=task_id,
            secret=secret,
            ttl_seconds=ttl_seconds,
            expected_user=expected_user,
        )["token"]
    )


def _signed_invite_token_info(
    *,
    user: str,
    secret: str,
    ttl_seconds: int,
    scope: str = SIGNED_INVITE_SCOPE_PERSONAL_QUEUE,
) -> dict[str, object]:
    normalized_user = str(user or "").strip()
    if not normalized_user:
        raise ValueError("Signed invite tokens require a user.")
    issued_at_unix = int(time.time())
    effective_ttl_seconds = _effective_signed_link_ttl_seconds(ttl_seconds)
    expires_at_unix = issued_at_unix + effective_ttl_seconds
    payload = {
        "v": 1,
        "kind": "labeler_invite",
        "scope": str(scope or SIGNED_INVITE_SCOPE_PERSONAL_QUEUE),
        "user": normalized_user,
        "iat": issued_at_unix,
        "exp": expires_at_unix,
        "nonce": uuid.uuid4().hex,
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    signature = hmac.new(str(secret).encode("utf-8"), payload_bytes, hashlib.sha256).digest()
    token = f"{_b64url_encode(payload_bytes)}.{_b64url_encode(signature)}"
    return {
        "token": token,
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "issued_at_utc": datetime.fromtimestamp(issued_at_unix, tz=timezone.utc).isoformat(),
        "expires_at_utc": datetime.fromtimestamp(expires_at_unix, tz=timezone.utc).isoformat(),
        "ttl_seconds": effective_ttl_seconds,
        "user": normalized_user,
        "scope": payload["scope"],
    }


def _signed_invite_token(
    *,
    user: str,
    secret: str,
    ttl_seconds: int,
    scope: str = SIGNED_INVITE_SCOPE_PERSONAL_QUEUE,
) -> str:
    return str(
        _signed_invite_token_info(
            user=user,
            secret=secret,
            ttl_seconds=ttl_seconds,
            scope=scope,
        )["token"]
    )



def _signed_invite_path(token: str, *, user: str) -> str:
    return (
        f"{PERSONAL_DATASET_QUEUE_PATH}"
        f"?expected_user={quote(str(user), safe='')}"
        f"&invite={quote(str(token), safe='')}"
    )





def _invite_cookie_header_from_query(handler: BaseHTTPRequestHandler, config: ServerConfig) -> str | None:
    token = _invite_query_token_from_request(handler)
    if not token:
        return None
    if not config.link_secret:
        return f"{SIGNED_INVITE_COOKIE_NAME}=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax"
    try:
        payload = _verify_signed_invite_token(token, secret=config.link_secret)
        revocation_reason = _signed_task_link_revocation_reason(
            payload,
            not_before_utc=config.link_not_before_utc,
        )
        if revocation_reason:
            return f"{SIGNED_INVITE_COOKIE_NAME}=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax"
        max_age = max(0, int(payload.get("exp") or 0) - int(time.time()))
    except Exception:
        return f"{SIGNED_INVITE_COOKIE_NAME}=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax"
    return f"{SIGNED_INVITE_COOKIE_NAME}={token}; Max-Age={max_age}; Path=/; HttpOnly; SameSite=Lax"



def _verify_signed_task_link_token(token: str, *, secret: str) -> dict[str, object]:
    parts = str(token or "").split(".", 1)
    if len(parts) != 2:
        raise ValueError("Malformed signed link token.")
    payload_bytes = _b64url_decode(parts[0])
    expected = hmac.new(str(secret).encode("utf-8"), payload_bytes, hashlib.sha256).digest()
    provided = _b64url_decode(parts[1])
    if not hmac.compare_digest(expected, provided):
        raise ValueError("Invalid signed link token.")
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict) or int(payload.get("v") or 0) != 1:
        raise ValueError("Unsupported signed link token.")
    if int(payload.get("exp") or 0) < int(time.time()):
        raise ValueError("Signed link token has expired.")
    task_id = str(payload.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("Signed link token is missing task_id.")
    return payload


def _parse_clip_index(clip_id: object) -> int:
    text = str(clip_id or "").strip()
    if text.startswith("clip_"):
        text = text[len("clip_") :]
    try:
        return int(text)
    except ValueError:
        return -1


def _jsonish(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_jsonish(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonish(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _compact_promotion_result(result: Mapping[str, object]) -> dict[str, object]:
    compact: dict[str, object] = {
        "status": result.get("status"),
        "target_crop_run": result.get("target_crop_run"),
        "action_counts": result.get("action_counts", {}),
    }
    items = result.get("items")
    if isinstance(items, list):
        compact["item_count"] = len(items)
        compact["items"] = [dict(item) for item in items[:3] if isinstance(item, Mapping)]
    if result.get("target_refined_run") is not None:
        compact["target_refined_run"] = result.get("target_refined_run")
    return _jsonish(compact)  # type: ignore[return-value]


def _run_detect_analysis_promotion(
    runtime: VideoDetectRuntimeSession,
    frame_payload: Mapping[str, object],
) -> dict[str, object] | None:
    config = runtime.promotion
    if config is None:
        return None
    from fisheye.tune.detect_training_promotion_backend import (
        ClippedPromotionFrame,
        PromotionOptions,
        promote_clipped_detection_frames,
        promote_detection_frames,
    )

    options = PromotionOptions(
        refined_run=str(frame_payload.get("refined_run_name") or "").strip() or None,
        target_crop_run=config.target_crop_run,
        target_refined_run=config.target_refined_run,
        label_origin=config.label_origin,
        include_negative=bool(config.include_negative),
        allow_unreviewed_negative=bool(config.allow_unreviewed_negative),
        target_size=config.target_size,
    )
    if str(getattr(runtime.review_session, "mode", "")) == "traditional":
        result = promote_detection_frames(
            runtime.review_session.zarr_path,
            config.training_zarr,
            [int(frame_payload.get("source_frame_index"))],
            options=options,
            apply=True,
        )
        return _compact_promotion_result(result)

    video_id = str(frame_payload.get("video_id") or "")
    source = runtime.review_session.videos.get(video_id)
    source_video_path = getattr(source, "source_path", None) or getattr(source, "path", None)
    frame = ClippedPromotionFrame(
        parent_frame_index=int(frame_payload.get("parent_frame_index")),
        clip_local_frame_index=int(frame_payload.get("clip_local_frame_index") or frame_payload.get("source_frame_index")),
        refined_group_path=str(frame_payload.get("refined_group_path") or ""),
        refined_run=str(frame_payload.get("refined_run_name") or ""),
        collection_id=str(getattr(runtime.review_session, "collection_id", "") or ""),
        clip_id=str(frame_payload.get("clip_id") or ""),
        clip_index=_parse_clip_index(frame_payload.get("clip_id")),
        camera_serial=str(frame_payload.get("camera_serial") or ""),
        recording_frame_id=(
            None
            if frame_payload.get("recording_frame_id") is None
            else int(frame_payload.get("recording_frame_id"))  # type: ignore[arg-type]
        ),
        source_video_path=str(source_video_path or ""),
    )
    result = promote_clipped_detection_frames(
        runtime.review_session.zarr_path,
        config.training_zarr,
        [frame],
        options=options,
        apply=True,
    )
    return _compact_promotion_result(result)


def _retry_failed_promotion_event(
    *,
    store: LabelingStore,
    event: Mapping[str, object],
    user: str,
) -> dict[str, object]:
    if str(event.get("event_type") or "") != "promotion_failed":
        raise ValueError("Only promotion_failed events can be retried.")
    scope = event.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("Failed-promotion event is missing task scope.")
    target = event.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("Failed-promotion event is missing target metadata.")
    zarr_path = str(scope.get("zarr_path") or target.get("analysis_zarr") or "").strip()
    if not zarr_path:
        raise ValueError("Failed-promotion event is missing analysis zarr path.")
    promotion_config = _detect_analysis_promotion_from_scope(scope)
    if promotion_config is None:
        training_zarr = str(target.get("training_zarr") or "").strip()
        if not training_zarr:
            raise ValueError("Failed-promotion event is missing training zarr path.")
        promotion_config = DetectAnalysisPromotionConfig(training_zarr=training_zarr)

    from fisheye.tune import video_detect_review_backend as backend_module

    review_session = backend_module.resolve_video_detect_review_session(
        zarr_path,
        collection_id=str(scope.get("collection_id") or "").strip() or None,
        refined_run=str(scope.get("refined_run") or target.get("refined_run") or "").strip() or None,
        recording_frame_index=str(scope.get("recording_frame_index") or "").strip() or None,
        review_proxy_manifest=str(scope.get("review_proxy_manifest") or "").strip() or None,
        editable=False,
        manual_score=float(scope.get("manual_score") or 1.0),
        manual_class_id=int(scope.get("manual_class_id") or 0),
    )
    parent_frame = target.get("parent_frame_index")
    if parent_frame is None:
        parent_frame = target.get("source_frame_index")
    parent_frame_index = int(parent_frame)  # type: ignore[arg-type]
    runtime = VideoDetectRuntimeSession(
        session_id=f"retry:{event.get('event_id')}",
        task_id=str(event.get("task_id") or ""),
        recording_id=str(event.get("recording_id") or ""),
        user=str(user),
        review_session=review_session,
        frame_indices=np.asarray([parent_frame_index], dtype=np.int32),
        editable=False,
        promotion=promotion_config,
    )
    frame_payload = dict(backend_module.load_frame_payload(review_session, parent_frame_index))
    retry_target = {
        **dict(target),
        "retry_of_event_id": str(event.get("event_id") or ""),
    }
    try:
        promotion = _run_detect_analysis_promotion(runtime, frame_payload)
    except Exception as exc:
        error = {"error": "promotion_failed", "details": str(exc), "retry_of_event_id": str(event.get("event_id") or "")}
        store.record_event(
            task_id=str(event.get("task_id") or ""),
            recording_id=str(event.get("recording_id") or ""),
            user=str(user),
            event_type="promotion_failed",
            target=retry_target,
            after=error,
        )
        raise
    promote_training_dataset_id = str(scope.get("promote_training_dataset_id") or "").strip()
    if promote_training_dataset_id:
        _refresh_registry_for_scope(
            store=store,
            task_id=str(event.get("task_id") or ""),
            recording_id=str(event.get("recording_id") or ""),
            user=str(user),
            workflow_kind="detect_training",
            scope=scope,
            zarr_path=promotion_config.training_zarr,
            dataset_id=promote_training_dataset_id,
            zarr_use="training",
        )
    store.record_event(
        task_id=str(event.get("task_id") or ""),
        recording_id=str(event.get("recording_id") or ""),
        user=str(user),
        event_type="promotion_success",
        target=retry_target,
        after=promotion,
    )
    return dict(promotion or {})


def _parse_byte_range(value: str | None, *, file_size: int) -> tuple[int, int] | None:
    if not value:
        return None
    if not value.startswith("bytes="):
        raise ValueError("Only byte ranges are supported.")
    spec = value[len("bytes=") :].split(",", 1)[0].strip()
    if "-" not in spec:
        raise ValueError("Invalid Range header.")
    start_raw, end_raw = spec.split("-", 1)
    if start_raw == "":
        suffix = int(end_raw)
        if suffix <= 0:
            raise ValueError("Invalid suffix byte range.")
        start = max(0, file_size - suffix)
        end = file_size - 1
    else:
        start = int(start_raw)
        end = int(end_raw) if end_raw else file_size - 1
    if start < 0 or end < start or start >= file_size:
        raise ValueError("Unsatisfiable byte range.")
    return start, min(end, file_size - 1)


def _server_config_errors(config: ServerConfig) -> list[str]:
    errors: list[str] = []
    if config.fixed_user and config.trust_auth_header:
        errors.append("choose either --user or --trust-auth-header, not both")
    if not config.fixed_user and not config.trust_auth_header:
        errors.append("serve requires --user for local development or --trust-auth-header behind a trusted proxy")
    if config.production and config.fixed_user:
        errors.append("--production requires proxy/header authentication; do not use --user")
    if config.production and not config.trust_auth_header:
        errors.append("--production requires --trust-auth-header behind a trusted proxy")
    if config.production and not config.admin_users:
        errors.append("--production requires at least one --admin-user")
    if config.require_operator_validation_for_start and config.validation_checklist_path is None:
        errors.append(
            "--require-operator-validation-for-start or --require-operator-validation-for-browser-work requires --validation-checklist"
        )
    if config.validation_checklist_path is not None and not config.validation_checklist_path.is_file():
        errors.append(
            f"--validation-checklist does not exist or is not a file: {config.validation_checklist_path}"
        )
    if config.trust_auth_header and not str(config.auth_header or "").strip():
        errors.append("--trust-auth-header requires a non-empty --auth-header")
    if config.link_not_before_utc:
        try:
            _utc_timestamp(config.link_not_before_utc)
        except Exception as exc:
            errors.append(f"--link-not-before-utc is invalid: {exc}")
    if not _is_loopback_host(config.host) and not config.allow_non_loopback:
        errors.append("--host is non-loopback; pass --allow-non-loopback only when network exposure is intentional")
    return errors


def _refresh_registry_for_scope(
    *,
    store: LabelingStore,
    task_id: str,
    recording_id: str,
    user: str,
    workflow_kind: str,
    scope: Mapping[str, object],
    zarr_path: str | None = None,
    dataset_id: str | None = None,
    zarr_use: str | None = None,
) -> None:
    registry_path = str(scope.get("registry_path") or "").strip()
    resolved_dataset_id = str(dataset_id or scope.get("dataset_id") or "").strip()
    resolved_zarr_path = str(zarr_path or scope.get("zarr_path") or "").strip()
    resolved_zarr_use = str(zarr_use or scope.get("zarr_use") or "").strip() or None
    if not registry_path or not resolved_dataset_id or not resolved_zarr_path:
        return
    try:
        from fisheye.registry.db import Registry

        registry = Registry(Path(registry_path).expanduser())
        try:
            counts: dict[str, int] = {}
            zarr_path_obj = Path(resolved_zarr_path).expanduser()
            if workflow_kind in {"detect_training", "detect_analysis"}:
                counts["detect_quality"] = int(registry.refresh_detect_quality_for_dataset(resolved_dataset_id, zarr_path=zarr_path_obj))
                counts["detect_performance"] = int(
                    registry.refresh_detect_performance_for_dataset(
                        resolved_dataset_id,
                        zarr_path=zarr_path_obj,
                        recording_id=str(recording_id),
                        zarr_use=resolved_zarr_use,
                    )
                )
            elif workflow_kind == "keypoints":
                counts["keypoint_quality"] = int(registry.refresh_keypoint_quality_for_dataset(resolved_dataset_id, zarr_path=zarr_path_obj))
                counts["keypoint_performance"] = int(
                    registry.refresh_keypoint_performance_for_dataset(
                        resolved_dataset_id,
                        zarr_path=zarr_path_obj,
                        recording_id=str(recording_id),
                        zarr_use=resolved_zarr_use,
                    )
                )
            elif workflow_kind == "subject_mask_component":
                counts["subject_mask_component_quality"] = int(
                    registry.refresh_subject_mask_component_quality_for_dataset(
                        resolved_dataset_id,
                        zarr_path=zarr_path_obj,
                        recording_id=str(recording_id),
                        zarr_use=resolved_zarr_use,
                    )
                )
            else:
                return
        finally:
            registry.close()
        store.record_event(
            task_id=task_id,
            recording_id=recording_id,
            user=user,
            event_type="registry_refresh_success",
            target={
                "workflow_kind": workflow_kind,
                "dataset_id": resolved_dataset_id,
                "registry_path": registry_path,
                "zarr_path": resolved_zarr_path,
            },
            after={"counts": counts},
        )
    except Exception as exc:
        store.record_event(
            task_id=task_id,
            recording_id=recording_id,
            user=user,
            event_type="registry_refresh_failed",
            target={
                "workflow_kind": workflow_kind,
                "dataset_id": resolved_dataset_id,
                "registry_path": registry_path,
                "zarr_path": resolved_zarr_path,
            },
            after={"error": "registry_refresh_failed", "details": str(exc)},
        )


def _project_approved_keypoint_review_to_recording_step_status(
    *,
    store: LabelingStore,
    task_id: str,
    recording_id: str,
    user: str,
    scope: Mapping[str, object],
    refined_attrs: Mapping[str, object],
    review_status: Mapping[str, object],
    review_event_id: str,
    zarr_path: str | None = None,
    dataset_id: str | None = None,
    zarr_use: str | None = None,
) -> None:
    review_state = str(review_status.get("state") or "").strip().lower()
    intended_use = str(review_status.get("intended_use") or "").strip().lower()
    if review_state != "approved" or intended_use != "training":
        return

    registry_path = str(scope.get("registry_path") or "").strip()
    resolved_dataset_id = str(dataset_id or scope.get("dataset_id") or "").strip()
    resolved_zarr_path = str(zarr_path or scope.get("zarr_path") or "").strip()
    resolved_zarr_use = str(zarr_use or scope.get("zarr_use") or "").strip() or None
    refined_run = str(
        scope.get("refined_run")
        or refined_attrs.get("palette_run_name")
        or refined_attrs.get("run_name")
        or ""
    ).strip()
    if not registry_path or not resolved_dataset_id or not refined_run:
        return

    pose_schema = refined_attrs.get("pose_schema")
    pose_schema_name = (
        str(pose_schema.get("name") or "")
        if isinstance(pose_schema, Mapping)
        else str(scope.get("pose_schema") or refined_attrs.get("pose_schema_name") or "")
    ).strip()
    skeleton_id = str(
        scope.get("skeleton_id")
        or refined_attrs.get("skeleton_id")
        or (
            pose_schema.get("skeleton_id")
            if isinstance(pose_schema, Mapping)
            else ""
        )
        or ""
    ).strip()
    stage_group = str(scope.get("stage_group") or "refined_keypoints_runs").strip()
    summary_statistics = refined_attrs.get("summary_statistics")
    postprocess = (
        summary_statistics.get("postprocess")
        if isinstance(summary_statistics, Mapping)
        else None
    )
    coverage_pct = None
    if isinstance(postprocess, Mapping):
        raw_coverage = postprocess.get("success_rate_percent")
        if raw_coverage is not None:
            try:
                coverage_pct = float(raw_coverage)
            except (TypeError, ValueError):
                coverage_pct = None
    details = {
        "schema": "palette.web_labeling_keypoints_review_step_projection.v1",
        "pose_schema": pose_schema_name,
        "skeleton_id": skeleton_id,
        "stage_group": stage_group,
        "review_task_id": str(task_id),
        "labeling_store_event_id": str(review_event_id),
        "zarr_use": resolved_zarr_use,
        "zarr_path": resolved_zarr_path,
        "review_source": "web_labeling_keypoint_review_status",
    }
    zarr_mtime_ns = None
    if resolved_zarr_path:
        try:
            zarr_mtime_ns = Path(resolved_zarr_path).expanduser().stat().st_mtime_ns
        except OSError:
            zarr_mtime_ns = None

    try:
        from fisheye.registry.db import Registry
        from fisheye.registry.status_ledger import upsert_recording_step_status

        registry = Registry(Path(registry_path).expanduser())
        try:
            row = upsert_recording_step_status(
                registry,
                dataset_id=resolved_dataset_id,
                recording_id=str(recording_id),
                step_name="keypoints_review",
                status="ok",
                run_name=refined_run,
                method=str(review_status.get("method") or "manual"),
                coverage_pct=coverage_pct,
                review_status_json=dict(review_status),
                details_json=details,
                source="web_labeling_review_completion",
                zarr_mtime_ns=zarr_mtime_ns,
            )
        finally:
            registry.close()
        store.record_event(
            task_id=task_id,
            recording_id=recording_id,
            user=user,
            event_type="registry_step_status_projection_success",
            target={
                "workflow_kind": "keypoints",
                "dataset_id": resolved_dataset_id,
                "registry_path": registry_path,
                "step_name": "keypoints_review",
                "run_name": refined_run,
            },
            after={"recording_step_status": row},
        )
    except Exception as exc:
        store.record_event(
            task_id=task_id,
            recording_id=recording_id,
            user=user,
            event_type="registry_step_status_projection_failed",
            target={
                "workflow_kind": "keypoints",
                "dataset_id": resolved_dataset_id,
                "registry_path": registry_path,
                "step_name": "keypoints_review",
                "run_name": refined_run,
            },
            after={"error": "registry_step_status_projection_failed", "details": str(exc)},
        )




















def _make_handler(state: ServerState):
    flask_app = create_labeling_app(config={"LABELING_SERVER_STATE": state})
    register_admin_api_routes(flask_app, state)
    register_admin_page_routes(flask_app, state, _admin_page_response_payload)
    register_personal_api_routes(flask_app, state, _personal_api_response_payload)
    register_personal_page_routes(flask_app, state, _personal_page_response_payload)

    class LabelingWorkHandler(BaseHTTPRequestHandler):
        server_version = "PaletteLabelingWork/0.1"
        sys_version = ""

        def _handle_flask_if_claimed(self) -> bool:
            return handle_with_flask_if_claimed(self, flask_app)

        def _write_no_store_headers(self) -> None:
            for name, value in BROWSER_RESPONSE_SECURITY_HEADERS.items():
                self.send_header(name, value)
            invite_cookie = _invite_cookie_header_from_query(self, state.config)
            if invite_cookie:
                self.send_header("Set-Cookie", invite_cookie)

        def _write(
            self,
            payload: bytes,
            *,
            status: HTTPStatus = HTTPStatus.OK,
            content_type: str = "text/html; charset=utf-8",
            headers: Mapping[str, str] | None = None,
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self._write_no_store_headers()
            for name, value in (headers or {}).items():
                self.send_header(str(name), str(value))
            self.end_headers()
            self.wfile.write(payload)

        def _write_json(self, payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            data, response_status, content_type = _json_response(payload, status=status)
            self._write(data, status=response_status, content_type=content_type)

        def _write_error(
            self,
            error: str,
            *,
            details: str | None = None,
            status: HTTPStatus = HTTPStatus.BAD_REQUEST,
            html_error: bool = False,
            extra: Mapping[str, object] | None = None,
        ) -> None:
            payload = _format_error(error, details=details, status=status, extra=extra)
            if html_error:
                self._write(_browser_error_html(payload), status=status, content_type="text/html; charset=utf-8")
                return
            self._write_json(payload, status=status)

        def _redirect(self, location: str, *, status: HTTPStatus = HTTPStatus.SEE_OTHER) -> None:
            payload = f"Redirecting to {html.escape(location)}".encode("utf-8")
            self.send_response(int(status))
            self.send_header("Location", location)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self._write_no_store_headers()
            self.end_headers()
            self.wfile.write(payload)

        def _stream_file_range(self, path: Path, *, start: int, length: int) -> None:
            remaining = int(length)
            with path.open("rb") as handle:
                handle.seek(start)
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

        def _current_user(self) -> tuple[str | None, str]:
            return _resolve_user(self, state.config)

        def _require_user(self, *, html_error: bool = False) -> tuple[str | None, str]:
            user, source = self._current_user()
            if not user:
                self._write_error(
                    "authentication_required",
                    details=_authentication_required_error_details(source, state.config),
                    status=HTTPStatus.UNAUTHORIZED,
                    html_error=html_error,
                )
                return None, source
            return user, source

        def _require_admin(self, *, html_error: bool = False) -> str | None:
            user, _source = self._require_user(html_error=html_error)
            if user is None:
                return None
            if not _is_admin_user(user, state.config):
                self._write_error("admin_required", status=HTTPStatus.FORBIDDEN, html_error=html_error)
                return None
            return user

        def _require_active_labeling_user(
            self,
            user: str,
            *,
            html_error: bool = False,
            expected_user: str | None = None,
            route_path: str = "",
        ) -> dict[str, object] | None:
            known_user_status = _known_labeler_status(state.store, user)
            if bool(known_user_status.get("is_active_labeling_user")):
                return known_user_status
            error = (
                "unknown_labeling_user"
                if not bool(known_user_status.get("registry_row_present"))
                else "inactive_labeling_user"
            )
            details = (
                "This browser identity is not present in the labeling user registry. "
                "Ask the operator to add or activate this user before labeling."
                if error == "unknown_labeling_user"
                else "This browser identity is present in the labeling user registry but is inactive. "
                "Ask the operator to reactivate this user before labeling."
            )
            self._write_error(
                error,
                details=details,
                status=HTTPStatus.FORBIDDEN,
                html_error=html_error,
                extra={
                    **_labeler_read_authorization_denial_metadata(
                        user=user,
                        expected_user=expected_user or user,
                        route_path=route_path,
                        response_kind="html" if html_error else "json",
                    ),
                    "known_user_status": known_user_status,
                },
            )
            return None

        def _session_for_user(
            self,
            session_id: str,
            user: str,
            *,
            html_error: bool = False,
            mutation_error: bool = False,
            completion_error: bool = False,
            expected_user: str | None = None,
        ) -> dict[str, object] | None:
            completion_expected_user = expected_user if expected_user is not None else user
            session = state.store.get_session(session_id)
            if session is None:
                extra = {
                    "authorization_context": _labeler_authorization_context(
                        user=user,
                        expected_user=completion_expected_user,
                        session_id=session_id,
                    )
                }
                if mutation_error:
                    mutation_extra = _browser_mutation_failure_metadata(
                        session={"session_id": session_id, "user": user},
                        error="session_not_found",
                        session_lookup_result="not_found",
                        session_owned_by_resolved_user=False,
                        task_reloaded_server_side=False,
                        task_assigned_to_resolved_user=False,
                        assignment_status_active=False,
                        task_open_for_mutation=False,
                        current_session_result="not_checked",
                        reassignment_session_safety_result="not_checked",
                    )
                    mutation_extra["authorization_context"] = extra["authorization_context"]
                    extra.update(mutation_extra)
                if completion_error:
                    extra.update(
                        _task_completion_failure_metadata(
                            user=user,
                            expected_user=completion_expected_user,
                            task=None,
                            session=None,
                            requested_task_id="",
                            error="session_not_found",
                        )
                    )
                self._write_error(
                    "session_not_found",
                    status=HTTPStatus.NOT_FOUND,
                    html_error=html_error,
                    extra=extra,
                )
                return None
            if str(session.get("user") or "") != str(user):
                mismatch_expected_user = (
                    expected_user
                    if expected_user is not None
                    else str(session.get("user") or user)
                )
                extra = {
                    "authorization_context": _labeler_authorization_context(
                        user=user,
                        expected_user=mismatch_expected_user,
                        session=session,
                    )
                }
                if mutation_error:
                    mutation_extra = _browser_mutation_failure_metadata(
                        session=session,
                        error="session_user_mismatch",
                        session_owned_by_resolved_user=False,
                        task_reloaded_server_side=False,
                        task_assigned_to_resolved_user=False,
                        assignment_status_active=False,
                        task_open_for_mutation=False,
                        current_session_result="not_checked",
                        reassignment_session_safety_result="not_checked",
                    )
                    mutation_extra["authorization_context"] = extra["authorization_context"]
                    extra.update(mutation_extra)
                if completion_error:
                    extra.update(
                        _task_completion_failure_metadata(
                            user=user,
                            expected_user=mismatch_expected_user,
                            task=None,
                            session=session,
                            requested_task_id=str(session.get("task_id") or ""),
                            error="session_user_mismatch",
                        )
                    )
                self._write_error(
                    "session_user_mismatch",
                    status=HTTPStatus.FORBIDDEN,
                    html_error=html_error,
                    extra=extra,
                )
                return None
            task = state.store.get_task(str(session.get("task_id") or ""))

            def _guard_extra(
                error: str,
                *,
                task_value: Mapping[str, object] | None = None,
                current_session: Mapping[str, object] | None = None,
                **contract_fields: object,
            ) -> dict[str, object]:
                authorization_context = _labeler_authorization_context(
                    user=user,
                    expected_user=completion_expected_user,
                    session=session,
                    task=task_value,
                    current_session=current_session,
                )
                extra: dict[str, object] = {
                    "authorization_context": authorization_context,
                }
                if mutation_error:
                    mutation_extra = _browser_mutation_failure_metadata(
                        session=session,
                        error=error,
                        **contract_fields,
                    )
                    mutation_extra["authorization_context"] = authorization_context
                    extra.update(mutation_extra)
                if completion_error:
                    requested_task_id = str(
                        (task_value or {}).get("task_id")
                        or session.get("task_id")
                        or ""
                    )
                    extra.update(
                        _task_completion_failure_metadata(
                            user=user,
                            expected_user=completion_expected_user,
                            task=task_value,
                            session=session,
                            requested_task_id=requested_task_id,
                            error=error,
                        )
                    )
                return extra
            if task is None:
                self._write_error(
                    "task_not_found",
                    status=HTTPStatus.NOT_FOUND,
                    html_error=html_error,
                    extra=_guard_extra(
                        "task_not_found",
                        task_reloaded_server_side=False,
                        task_assigned_to_resolved_user=False,
                        assignment_status_active=False,
                        task_open_for_mutation=False,
                        current_session_result="not_checked",
                        reassignment_session_safety_result="not_checked",
                    ),
                )
                return None
            if str(task.get("assignee_user") or "") != str(user) or str(task.get("assignment_status") or "") != "active":
                closure_event = state.store.get_session_closure_event(session_id)
                self._write_error(
                    "not_assigned",
                    details="This task is no longer assigned to the current user.",
                    status=HTTPStatus.FORBIDDEN,
                    html_error=html_error,
                    extra={
                        **_session_closure_error_extra(closure_event),
                        **_guard_extra(
                            "not_assigned",
                            task_value=task,
                            task_assigned_to_resolved_user=False,
                            assignment_status_active=(
                                str(task.get("assignment_status") or "") == "active"
                            ),
                            task_open_for_mutation=False,
                        ),
                    },
                )
                return None
            mismatched_sessions = state.store.active_assignment_mismatched_sessions_for_recording(
                str(task.get("recording_id") or ""),
                limit=10,
            )
            if mismatched_sessions:
                self._write_error(
                    "reassignment_session_safety_failed",
                    details=(
                        "Stale previous-owner sessions are still open for this recording. "
                        "Stop and contact the operator before labeling."
                    ),
                    status=HTTPStatus.CONFLICT,
                    html_error=html_error,
                    extra={
                        "active_session_assignment_mismatch_count": len(mismatched_sessions),
                        **_guard_extra(
                            "reassignment_session_safety_failed",
                            task_value=task,
                            reassignment_session_safety_result="failed",
                        ),
                    },
                )
                return None
            if str(task.get("state") or "") == "complete":
                closure_event = state.store.get_session_closure_event(session_id)
                self._write_error(
                    "task_complete",
                    details="This task is complete and is no longer open for labeling.",
                    status=HTTPStatus.CONFLICT,
                    html_error=html_error,
                    extra={
                        **_session_closure_error_extra(closure_event),
                        **_guard_extra(
                            "task_complete",
                            task_value=task,
                            task_open_for_mutation=False,
                        ),
                    },
                )
                return None
            task_state = str(task.get("state") or "")
            if task_state not in LABELER_START_TASK_STATES:
                closure_event = state.store.get_session_closure_event(session_id)
                self._write_error(
                    "task_not_startable",
                    details=f"This task is in state {task_state or 'unknown'} and is no longer open for labeling.",
                    status=HTTPStatus.CONFLICT,
                    html_error=html_error,
                    extra={
                        **_session_closure_error_extra(closure_event),
                        **_guard_extra(
                            "task_not_startable",
                            task_value=task,
                            task_open_for_mutation=False,
                        ),
                    },
                )
                return None
            if session.get("closed_at_utc"):
                closure_event = state.store.get_session_closure_event(session_id)
                closure_event_type = str((closure_event or {}).get("event_type") or "")
                if closure_event_type == "session_superseded":
                    self._write_error(
                        "session_superseded",
                        details="A newer browser session owns this task. Refresh from the dashboard to continue.",
                        status=HTTPStatus.CONFLICT,
                        html_error=html_error,
                        extra={
                            **_session_closure_error_extra(closure_event),
                            **_guard_extra(
                                "session_superseded",
                                task_value=task,
                                current_session_result="superseded",
                            ),
                        },
                    )
                    return None
                if closure_event_type == "stale_session_closed":
                    self._write_error(
                        "session_expired",
                        details="This labeling session expired and is no longer the active writer for its task.",
                        status=HTTPStatus.CONFLICT,
                        html_error=html_error,
                        extra={
                            **_session_closure_error_extra(closure_event),
                            **_guard_extra(
                                "session_expired",
                                task_value=task,
                                current_session_result="expired",
                            ),
                        },
                    )
                    return None
                self._write_error(
                    "session_closed",
                    details="This labeling session has been closed or superseded by a newer session.",
                    status=HTTPStatus.CONFLICT,
                    html_error=html_error,
                    extra={
                        **_session_closure_error_extra(closure_event),
                        **_guard_extra(
                            "session_closed",
                            task_value=task,
                            current_session_result="closed",
                        ),
                    },
                )
                return None
            if _session_is_expired(session):
                self._write_error(
                    "session_expired",
                    details="This labeling session expired and is no longer the active writer for its task.",
                    status=HTTPStatus.CONFLICT,
                    html_error=html_error,
                    extra={
                        "session_expires_at_utc": str(session.get("expires_at_utc") or ""),
                        **_guard_extra(
                            "session_expired",
                            task_value=task,
                            current_session_result="expired",
                        ),
                    },
                )
                return None
            current = state.store.get_current_task_session(task_id=str(session.get("task_id") or ""))
            if current is None:
                self._write_error(
                    "session_expired",
                    details="This labeling session is no longer the active writer for its task.",
                    status=HTTPStatus.CONFLICT,
                    html_error=html_error,
                    extra=_guard_extra(
                        "session_expired",
                        task_value=task,
                        current_session_result="missing_current_session",
                    ),
                )
                return None
            if str(current.get("session_id") or "") != str(session_id):
                self._write_error(
                    "session_superseded",
                    details="A newer browser session owns this task. Refresh from the dashboard to continue.",
                    status=HTTPStatus.CONFLICT,
                    html_error=html_error,
                    extra=_guard_extra(
                        "session_superseded",
                        task_value=task,
                        current_session=current,
                        current_session_result="superseded",
                    ),
                )
                return None
            return session

        def _operator_validation_mutation_blocked(self, session: Mapping[str, object]) -> bool:
            mutation_gate = _runtime_operator_validation_mutation_gate(state.config)
            if not bool(mutation_gate.get("blocks_browser_mutation")):
                return False
            self._write_json(
                _format_error(
                    "operator_validation_mutation_blocked",
                    details=(
                        str(mutation_gate.get("operator_action") or "")
                        or "Required operator validation evidence is incomplete, so the server did not apply this browser mutation."
                    ),
                    status=HTTPStatus.CONFLICT,
                    extra=_browser_mutation_failure_metadata(
                        session=session,
                        error="operator_validation_mutation_blocked",
                        operator_validation_mutation_gate=mutation_gate,
                    ),
                ),
                status=HTTPStatus.CONFLICT,
            )
            return True

        def _reject_browser_mutation_preflight(
            self,
            session: Mapping[str, object],
            body: Mapping[str, object],
            runtime: object,
        ) -> bool:
            mutation_gate = _runtime_operator_validation_mutation_gate(state.config)
            target_selector_keys = _browser_mutation_target_selector_keys(body)
            if target_selector_keys:
                self._write_json(
                    _format_error(
                        "payload_validation",
                        details=_browser_mutation_target_selector_details(target_selector_keys),
                        status=HTTPStatus.BAD_REQUEST,
                        extra=_browser_mutation_failure_metadata(
                            session=session,
                            error="browser_mutation_target_selector_rejected",
                            operator_validation_mutation_gate=mutation_gate,
                            current_target_token_result="not_checked",
                            browser_supplied_target_selectors_result="rejected",
                        ),
                    ),
                    status=HTTPStatus.BAD_REQUEST,
                )
                return True
            if self._operator_validation_mutation_blocked(session):
                return True
            try:
                _require_browser_mutation_target_token(runtime, body)
            except ValueError as exc:
                self._write_json(
                    _format_error(
                        "payload_validation",
                        details=str(exc),
                        status=HTTPStatus.BAD_REQUEST,
                        extra=_browser_mutation_failure_metadata(
                            session=session,
                            error="target_token_failed",
                            operator_validation_mutation_gate=mutation_gate,
                            current_target_token_result="failed",
                            browser_supplied_target_selectors_result="passed",
                        ),
                    ),
                    status=HTTPStatus.BAD_REQUEST,
                )
                return True
            return False

        def _parse_session_api_path(self, prefix: str) -> tuple[str, str] | None:
            if not prefix.startswith("/api/sessions/"):
                return None
            rest = prefix[len("/api/sessions/") :].strip("/")
            if not rest:
                return None
            parts = rest.split("/", 1)
            session_id = parts[0]
            suffix = "/" + parts[1] if len(parts) > 1 else "/"
            return session_id, suffix

        def _handle_keypoint_get(self, session: Mapping[str, object], suffix: str) -> bool:
            if not suffix.startswith("/keypoints/"):
                return False
            from fisheye.tune import keypoint_review_backend as backend_module

            try:
                runtime = _get_keypoint_runtime(state, session)
            except Exception as exc:
                self._write_json(_format_error("keypoint_session_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                return True

            keypoint_path = suffix[len("/keypoints") :]
            if keypoint_path == "/state":
                self._write_json({"ok": True, "state": _keypoint_runtime_state(runtime, backend_module)})
                return True
            if keypoint_path == "/roi/current":
                try:
                    payload = dict(backend_module.load_roi_payload(runtime.review_session, position=runtime.position))
                    payload["state"] = _keypoint_runtime_state(runtime, backend_module)
                    payload["ok"] = True
                except Exception as exc:
                    self._write_json(_format_error("roi_load_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                    return True
                self._write_json(_redact_labeler_runtime_payload(payload))
                return True
            return False

        def _handle_keypoint_post(self, session: Mapping[str, object], suffix: str, body: Mapping[str, object], user: str) -> bool:
            if not suffix.startswith("/keypoints/"):
                return False
            from fisheye.tune import keypoint_review_backend as backend_module

            try:
                runtime = _get_keypoint_runtime(state, session)
            except Exception as exc:
                self._write_json(_format_error("keypoint_session_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                return True

            keypoint_path = suffix[len("/keypoints") :]
            if keypoint_path == "/nav":
                try:
                    total = int(runtime.review_session.failures.size)
                    runtime.position = _next_browser_nav_position(
                        current_position=runtime.position,
                        total=total,
                        body=body,
                    )
                except (TypeError, ValueError) as exc:
                    self._write_json(_format_error("nav_error", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json({"ok": True, "state": _keypoint_runtime_state(runtime, backend_module)})
                return True

            if keypoint_path == "/save":
                if self._reject_browser_mutation_preflight(session, body, runtime):
                    return True
                if "points" not in body:
                    self._write_json(_format_error("payload_validation", details="Missing points."), status=HTTPStatus.BAD_REQUEST)
                    return True
                try:
                    before = dict(backend_module.load_roi_payload(runtime.review_session, position=runtime.position))
                    result = backend_module.save_roi_correction(
                        runtime.review_session,
                        position=runtime.position,
                        points=body.get("points"),  # type: ignore[arg-type]
                    )
                    target = {
                        "roi_idx": result.get("roi_idx"),
                        "frame_idx": result.get("frame_idx"),
                        "refined_run": str(runtime.review_session.refined_run),
                        "crop_run": str(runtime.review_session.crop_run),
                    }
                    mutation_event = state.store.record_event(
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        event_type="save_keypoints",
                        target=target,
                        before={
                            "roi_idx": before.get("roi_idx"),
                            "frame_idx": before.get("frame_idx"),
                            "points": before.get("points"),
                            "reason": before.get("reason"),
                            "status": before.get("status"),
                        },
                        after={
                            "changed": result.get("changed"),
                            "reason_updated": result.get("reason_updated"),
                            "readback": result.get("readback"),
                        },
                    )
                    _refresh_registry_for_scope(
                        store=state.store,
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        workflow_kind="keypoints",
                        scope=_session_scope(session),
                        zarr_path=str(runtime.review_session.zarr_path),
                        dataset_id=str(session.get("dataset_id") or "") or None,
                        zarr_use=str(session.get("zarr_use") or "") or None,
                    )
                    _advance_keypoint(runtime, advance=bool(body.get("advance", runtime.auto_advance_on_save)))
                except Exception as exc:
                    self._write_json(_format_error("save_error", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json(
                    _redact_labeler_runtime_payload(
                        {
                            "ok": True,
                            "result": result,
                            "mutation": _browser_mutation_response_metadata(
                                workflow_kind="keypoints",
                                session=session,
                                mutation_event=mutation_event,
                                operator_validation_mutation_gate=_runtime_operator_validation_mutation_gate(state.config),
                            ),
                            "state": _keypoint_runtime_state(runtime, backend_module),
                        }
                    )
                )
                return True

            if keypoint_path == "/action":
                if self._reject_browser_mutation_preflight(session, body, runtime):
                    return True
                action = str(body.get("action") or "").strip()
                try:
                    if action == "mark_no_keypoints":
                        result = backend_module.mark_no_keypoints(runtime.review_session, position=runtime.position)
                    elif action == "mark_detection_issue":
                        result = backend_module.mark_detection_issue(runtime.review_session, position=runtime.position)
                    elif action == "clear_failure_label":
                        result = backend_module.clear_failure_label(runtime.review_session, position=runtime.position)
                    else:
                        raise ValueError(f"Unsupported keypoint action: {action}")
                    mutation_event = state.store.record_event(
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        event_type=f"keypoint_{action}",
                        target={"roi_idx": result.get("roi_idx"), "frame_idx": result.get("frame_idx")},
                        after=result,
                    )
                    _refresh_registry_for_scope(
                        store=state.store,
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        workflow_kind="keypoints",
                        scope=_session_scope(session),
                        zarr_path=str(runtime.review_session.zarr_path),
                        dataset_id=str(session.get("dataset_id") or "") or None,
                        zarr_use=str(session.get("zarr_use") or "") or None,
                    )
                    _advance_keypoint(runtime, advance=bool(body.get("advance", runtime.auto_advance_on_save)))
                except Exception as exc:
                    self._write_json(_format_error("keypoint_action_error", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json(
                    _redact_labeler_runtime_payload(
                        {
                            "ok": True,
                            "result": result,
                            "mutation": _browser_mutation_response_metadata(
                                workflow_kind="keypoints",
                                session=session,
                                mutation_event=mutation_event,
                                operator_validation_mutation_gate=_runtime_operator_validation_mutation_gate(state.config),
                            ),
                            "state": _keypoint_runtime_state(runtime, backend_module),
                        }
                    )
                )
                return True

            if keypoint_path == "/review-status":
                if self._reject_browser_mutation_preflight(session, body, runtime):
                    return True
                requested_state = str(body.get("state") or "").strip()
                if not requested_state:
                    self._write_json(_format_error("payload_validation", details="Missing review state."), status=HTTPStatus.BAD_REQUEST)
                    return True
                try:
                    before_status = runtime.review_session.refined.attrs.get("keypoint_review_status")
                    result = backend_module.apply_review_status(
                        runtime.review_session,
                        state=requested_state,
                        method=str(body.get("method") or runtime.review_method or "manual"),
                        intended_use=str(body.get("intended_use") or runtime.review_intended_use or "").strip() or None,
                        reviewer=user,
                        notes=str(body.get("notes") or runtime.review_notes or "").strip() or None,
                    )
                    mutation_event = state.store.record_event(
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        event_type="set_review_status",
                        target={"refined_run": str(runtime.review_session.refined_run)},
                        before={"review_status": dict(before_status) if isinstance(before_status, Mapping) else None},
                        after={"review_status": result.get("review_status")},
                    )
                    _refresh_registry_for_scope(
                        store=state.store,
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        workflow_kind="keypoints",
                        scope=_session_scope(session),
                        zarr_path=str(runtime.review_session.zarr_path),
                        dataset_id=str(session.get("dataset_id") or "") or None,
                        zarr_use=str(session.get("zarr_use") or "") or None,
                    )
                    review_status = result.get("review_status")
                    if isinstance(review_status, Mapping):
                        _project_approved_keypoint_review_to_recording_step_status(
                            store=state.store,
                            task_id=runtime.task_id,
                            recording_id=runtime.recording_id,
                            user=user,
                            scope=_session_scope(session),
                            refined_attrs=runtime.review_session.refined.attrs,
                            review_status=review_status,
                            review_event_id=str(mutation_event.get("event_id") or ""),
                            zarr_path=str(runtime.review_session.zarr_path),
                            dataset_id=str(session.get("dataset_id") or "") or None,
                            zarr_use=str(session.get("zarr_use") or "") or None,
                        )
                except Exception as exc:
                    self._write_json(_format_error("review_status_error", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json(
                    _redact_labeler_runtime_payload(
                        {
                            "ok": True,
                            "result": result,
                            "mutation": _browser_mutation_response_metadata(
                                workflow_kind="keypoints",
                                session=session,
                                mutation_event=mutation_event,
                                operator_validation_mutation_gate=_runtime_operator_validation_mutation_gate(state.config),
                            ),
                            "state": _keypoint_runtime_state(runtime, backend_module),
                        }
                    )
                )
                return True

            return False

        def _handle_detect_get(self, session: Mapping[str, object], suffix: str) -> bool:
            if not suffix.startswith("/detect/"):
                return False
            from fisheye.tune import detect_review_backend as backend_module

            try:
                runtime = _get_detect_runtime(state, session)
            except Exception as exc:
                self._write_json(_format_error("detect_session_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                return True

            detect_path = suffix[len("/detect") :]
            if detect_path == "/state":
                self._write_json({"ok": True, "state": _detect_runtime_state(runtime, backend_module)})
                return True
            if detect_path == "/frame/current":
                try:
                    payload = dict(backend_module.load_frame_payload(runtime.review_session, position=runtime.position))
                    payload["bbox_size_hint_norm"] = _detect_bbox_size_hint_payload(
                        session=session,
                        runtime=runtime,
                    )
                    payload["state"] = _detect_runtime_state(runtime, backend_module)
                    payload["ok"] = True
                except Exception as exc:
                    self._write_json(_format_error("frame_load_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                    return True
                self._write_json(_redact_labeler_runtime_payload(payload))
                return True
            return False

        def _handle_detect_post(self, session: Mapping[str, object], suffix: str, body: Mapping[str, object], user: str) -> bool:
            if not suffix.startswith("/detect/"):
                return False
            from fisheye.tune import detect_review_backend as backend_module

            try:
                runtime = _get_detect_runtime(state, session)
            except Exception as exc:
                self._write_json(_format_error("detect_session_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                return True

            detect_path = suffix[len("/detect") :]
            if detect_path == "/nav":
                try:
                    total = int(runtime.review_session.review_rows.shape[0])
                    runtime.position = _next_browser_nav_position(
                        current_position=runtime.position,
                        total=total,
                        body=body,
                    )
                except (TypeError, ValueError) as exc:
                    self._write_json(_format_error("nav_error", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json({"ok": True, "state": _detect_runtime_state(runtime, backend_module)})
                return True

            if detect_path == "/save":
                if self._reject_browser_mutation_preflight(session, body, runtime):
                    return True
                try:
                    before = dict(backend_module.load_frame_payload(runtime.review_session, position=runtime.position))
                    result = backend_module.apply_manual_edit(
                        runtime.review_session,
                        position=runtime.position,
                        bbox_norm=body.get("bbox_norm"),  # type: ignore[arg-type]
                    )
                    mutation_event = state.store.record_event(
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        event_type="save_detect_bbox",
                        target={
                            "row_idx": result.get("row_idx"),
                            "frame_idx": result.get("frame_idx"),
                            "refined_run": str(runtime.review_session.refined_run_name),
                        },
                        before={
                            "row_idx": before.get("row_idx"),
                            "frame_idx": before.get("frame_idx"),
                            "bbox_norm": before.get("bbox_norm"),
                            "status": before.get("status"),
                        },
                        after={
                            "action": result.get("action"),
                            "bbox_norm": result.get("bbox_norm"),
                            "status": result.get("status"),
                        },
                    )
                    _refresh_registry_for_scope(
                        store=state.store,
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        workflow_kind="detect_training",
                        scope=_session_scope(session),
                        zarr_path=str(runtime.review_session.zarr_path),
                        dataset_id=str(session.get("dataset_id") or "") or None,
                        zarr_use=str(session.get("zarr_use") or "") or None,
                    )
                    if bool(body.get("advance", runtime.auto_advance_on_save)):
                        total = int(runtime.review_session.review_rows.shape[0])
                        if total <= 0:
                            runtime.position = 0
                        elif runtime.position < total - 1:
                            runtime.position += 1
                except Exception as exc:
                    self._write_json(_format_error("save_error", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json(
                    _redact_labeler_runtime_payload(
                        {
                            "ok": True,
                            "result": result,
                            "mutation": _browser_mutation_response_metadata(
                                workflow_kind="detect_training",
                                session=session,
                                mutation_event=mutation_event,
                                operator_validation_mutation_gate=_runtime_operator_validation_mutation_gate(state.config),
                            ),
                            "state": _detect_runtime_state(runtime, backend_module),
                        }
                    )
                )
                return True

            return False

        def _serve_video_detect_media(self, session: Mapping[str, object], suffix: str) -> bool:
            if not suffix.startswith("/detect-analysis/media/"):
                return False
            video_id = suffix[len("/detect-analysis/media/") :].strip("/")
            if not video_id:
                self._write_json(_format_error("missing_video_id", status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                return True
            try:
                runtime = _get_video_detect_runtime(state, session)
            except Exception as exc:
                self._write_json(_format_error("detect_analysis_session_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                return True
            source = runtime.review_session.videos.get(video_id)
            if source is None:
                self._write_json(_format_error("video_not_found", status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                return True
            path = Path(source.path)
            if not path.is_file():
                self._write_json(
                    _format_error(
                        "video_file_not_found",
                        details=_labeler_safe_error_details(path),
                        status=HTTPStatus.NOT_FOUND,
                    ),
                    status=HTTPStatus.NOT_FOUND,
                )
                return True
            file_size = path.stat().st_size
            content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
            try:
                byte_range = _parse_byte_range(self.headers.get("Range"), file_size=file_size)
            except Exception as exc:
                payload = json.dumps(
                    _format_error(
                        "invalid_range",
                        details=_labeler_safe_error_details(exc),
                        status=HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    ),
                    allow_nan=False,
                ).encode("utf-8")
                self.send_response(int(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.send_header("Content-Length", str(len(payload)))
                self._write_no_store_headers()
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)
                return True
            start, end = byte_range if byte_range is not None else (0, file_size - 1)
            length = max(0, end - start + 1)
            self.send_response(int(HTTPStatus.PARTIAL_CONTENT if byte_range is not None else HTTPStatus.OK))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self._write_no_store_headers()
            if byte_range is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()
            if self.command != "HEAD":
                self._stream_file_range(path, start=start, length=length)
            return True

        def _handle_video_detect_get(self, session: Mapping[str, object], suffix: str) -> bool:
            if self._serve_video_detect_media(session, suffix):
                return True
            if not suffix.startswith("/detect-analysis/"):
                return False
            from fisheye.tune import video_detect_review_backend as backend_module

            try:
                runtime = _get_video_detect_runtime(state, session)
            except Exception as exc:
                self._write_json(_format_error("detect_analysis_session_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                return True

            detect_path = suffix[len("/detect-analysis") :]
            if detect_path == "/state":
                self._write_json({"ok": True, "state": _video_detect_runtime_state(runtime, backend_module)})
                return True
            if detect_path == "/frame/current":
                try:
                    payload = _video_detect_frame_payload(runtime, backend_module)
                except Exception as exc:
                    self._write_json(_format_error("frame_load_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                    return True
                self._write_json(_redact_labeler_runtime_payload(payload))
                return True
            return False

        def _handle_video_detect_post(self, session: Mapping[str, object], suffix: str, body: Mapping[str, object], user: str) -> bool:
            if not suffix.startswith("/detect-analysis/"):
                return False
            from fisheye.tune import video_detect_review_backend as backend_module

            try:
                runtime = _get_video_detect_runtime(state, session)
            except Exception as exc:
                self._write_json(_format_error("detect_analysis_session_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                return True

            detect_path = suffix[len("/detect-analysis") :]
            if detect_path == "/nav":
                try:
                    total = int(runtime.frame_indices.shape[0])
                    runtime.position = _next_browser_nav_position(
                        current_position=runtime.position,
                        total=total,
                        body=body,
                    )
                except (TypeError, ValueError) as exc:
                    self._write_json(_format_error("nav_error", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json({"ok": True, "state": _video_detect_runtime_state(runtime, backend_module)})
                return True

            if detect_path == "/save":
                if self._reject_browser_mutation_preflight(session, body, runtime):
                    return True
                if not runtime.editable:
                    self._write_json(
                        _format_error(
                            "read_only",
                            details="detect_analysis task scope did not enable editable=true.",
                            status=HTTPStatus.FORBIDDEN,
                        ),
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return True
                try:
                    parent_frame = _get_video_detect_parent_frame(runtime)
                    before = dict(backend_module.load_frame_payload(runtime.review_session, parent_frame))
                    result = backend_module.apply_manual_edit(
                        runtime.review_session,
                        parent_frame_index=parent_frame,
                        bbox_norm=body.get("bbox_norm"),  # type: ignore[arg-type]
                    )
                    mutation_event = state.store.record_event(
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        event_type="save_detect_analysis_bbox",
                        target={
                            "parent_frame_index": result.get("parent_frame_index"),
                            "source_frame_index": result.get("source_frame_index"),
                            "refined_run": before.get("refined_run_name"),
                            "refined_group_path": before.get("refined_group_path"),
                            "clip_id": before.get("clip_id"),
                            "camera_serial": before.get("camera_serial"),
                        },
                        before={
                            "bbox_norm": before.get("bbox_norm"),
                            "status": before.get("status"),
                        },
                        after={
                            "action": result.get("action"),
                            "bbox_norm": result.get("bbox_norm"),
                            "status": result.get("status"),
                        },
                    )
                    _refresh_registry_for_scope(
                        store=state.store,
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        workflow_kind="detect_analysis",
                        scope=_session_scope(session),
                        zarr_path=str(runtime.review_session.zarr_path),
                        dataset_id=str(session.get("dataset_id") or "") or None,
                        zarr_use=str(session.get("zarr_use") or "") or None,
                    )
                    promotion = None
                    promotion_error = None
                    promotion_event = None
                    if runtime.promotion is not None:
                        promotion_target = {
                            "parent_frame_index": result.get("parent_frame_index"),
                            "source_frame_index": result.get("source_frame_index"),
                            "analysis_zarr": str(runtime.review_session.zarr_path),
                            "training_zarr": runtime.promotion.training_zarr,
                            "refined_run": before.get("refined_run_name"),
                            "refined_group_path": before.get("refined_group_path"),
                            "clip_id": before.get("clip_id"),
                            "camera_serial": before.get("camera_serial"),
                        }
                        try:
                            promotion = _run_detect_analysis_promotion(runtime, before)
                            promotion_scope = _session_scope(session)
                            promote_training_dataset_id = str(promotion_scope.get("promote_training_dataset_id") or "").strip()
                            if promote_training_dataset_id:
                                _refresh_registry_for_scope(
                                    store=state.store,
                                    task_id=runtime.task_id,
                                    recording_id=runtime.recording_id,
                                    user=user,
                                    workflow_kind="detect_training",
                                    scope=promotion_scope,
                                    zarr_path=runtime.promotion.training_zarr,
                                    dataset_id=promote_training_dataset_id,
                                    zarr_use="training",
                                )
                            promotion_event = state.store.record_event(
                                task_id=runtime.task_id,
                                recording_id=runtime.recording_id,
                                user=user,
                                event_type="promotion_success",
                                target=promotion_target,
                                after=promotion,
                            )
                        except Exception as exc:
                            promotion_error = {"error": "promotion_failed", "details": str(exc)}
                            promotion_event = state.store.record_event(
                                task_id=runtime.task_id,
                                recording_id=runtime.recording_id,
                                user=user,
                                event_type="promotion_failed",
                                target=promotion_target,
                                after=promotion_error,
                            )
                    if bool(body.get("advance", runtime.auto_advance_on_save)):
                        total = int(runtime.frame_indices.shape[0])
                        if runtime.position < total - 1:
                            runtime.position += 1
                except Exception as exc:
                    self._write_json(_format_error("save_error", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json(
                    _redact_labeler_runtime_payload(
                        {
                            "ok": True,
                            "result": result,
                            "promotion": promotion,
                            "promotion_error": promotion_error,
                            "mutation": _browser_mutation_response_metadata(
                                workflow_kind="detect_analysis",
                                session=session,
                                mutation_event=mutation_event,
                                promotion_event=promotion_event,
                                operator_validation_mutation_gate=_runtime_operator_validation_mutation_gate(state.config),
                            ),
                            "state": _video_detect_runtime_state(runtime, backend_module),
                        }
                    )
                )
                return True

            return False

        def _handle_subject_mask_get(self, session: Mapping[str, object], suffix: str) -> bool:
            if not suffix.startswith("/subject-mask/"):
                return False
            try:
                runtime = _get_subject_mask_runtime(state, session)
            except Exception as exc:
                self._write_json(_format_error("subject_mask_session_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                return True

            subject_mask_path = suffix[len("/subject-mask") :]
            if subject_mask_path == "/state":
                self._write_json({"ok": True, "state": _subject_mask_runtime_state(runtime, store=state.store)})
                return True
            if subject_mask_path == "/roi/current":
                try:
                    payload = _subject_mask_current_payload(runtime, store=state.store)
                except Exception as exc:
                    self._write_json(_format_error("roi_load_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                    return True
                self._write_json(payload)
                return True
            return False

        def _handle_subject_mask_post(self, session: Mapping[str, object], suffix: str, body: Mapping[str, object], user: str) -> bool:
            if not suffix.startswith("/subject-mask/"):
                return False
            from fisheye.tune import refined_subject_mask_review as review_mod

            try:
                runtime = _get_subject_mask_runtime(state, session)
            except Exception as exc:
                self._write_json(_format_error("subject_mask_session_error", details=_labeler_safe_error_details(exc), status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)
                return True

            subject_mask_path = suffix[len("/subject-mask") :]
            if subject_mask_path == "/nav":
                try:
                    total = int(runtime.roi_indices.shape[0])
                    if body.get("roi_idx") is not None:
                        requested_roi_idx = int(body.get("roi_idx"))  # type: ignore[arg-type]
                        matches = np.flatnonzero(runtime.roi_indices.astype(np.int64) == requested_roi_idx)
                        if matches.size <= 0:
                            raise ValueError(f"ROI {requested_roi_idx} is outside the active task row scope.")
                        runtime.position = int(matches[0])
                    else:
                        runtime.position = _next_browser_nav_position(
                            current_position=runtime.position,
                            total=total,
                            body=body,
                        )
                except (TypeError, ValueError) as exc:
                    self._write_json(_format_error("nav_error", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json({"ok": True, "state": _subject_mask_runtime_state(runtime, store=state.store)})
                return True

            if subject_mask_path == "/save":
                if self._reject_browser_mutation_preflight(session, body, runtime):
                    return True
                mask_payload = body.get("mask")
                if not isinstance(mask_payload, Mapping):
                    self._write_json(_format_error("payload_validation", details="Missing mask payload."), status=HTTPStatus.BAD_REQUEST)
                    return True
                try:
                    total = int(runtime.roi_indices.shape[0])
                    if total <= 0:
                        raise ValueError("Subject-mask task has no ROI rows.")
                    runtime.position = max(0, min(int(runtime.position), total - 1))
                    roi_idx = int(runtime.roi_indices[runtime.position])
                    canonical_mask = (np.asarray(runtime.refined.group["masks_roi"][roi_idx, runtime.comp_idx], dtype=np.uint8) > 0).astype(np.uint8)
                    edited_mask = (_decode_uint8_payload(mask_payload) > 0).astype(np.uint8)
                    if edited_mask.ndim == 3 and edited_mask.shape[-1] == 1:
                        edited_mask = edited_mask[:, :, 0]
                    if tuple(edited_mask.shape) != tuple(canonical_mask.shape):
                        raise ValueError(f"mask shape mismatch: expected {tuple(canonical_mask.shape)}, got {tuple(edited_mask.shape)}")
                    frame_idx: int | None = None
                    if runtime.source.frame_indices is not None:
                        try:
                            frame_idx = int(np.asarray(runtime.source.frame_indices[roi_idx]).item())
                        except Exception:
                            frame_idx = None
                    target_edit_revision = _subject_mask_edit_revision(runtime)
                    row_identity = _subject_mask_row_identity(runtime, roi_idx)
                    checkpoint = state.store.upsert_session_checkpoint(
                        session_id=runtime.session_id,
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        workflow_kind="subject_mask_component",
                        target_run_path=_subject_mask_target_run_path(runtime),
                        target_edit_revision=target_edit_revision,
                        source_rowset_path=_subject_mask_source_rowset_path(runtime),
                        roi_idx=roi_idx,
                        component_name=runtime.component_name,
                        payload={
                            "schema": "palette.web_labeling_subject_mask_checkpoint_payload.v1",
                            "payload_kind": "dense_roi_replacement_mask",
                            "mask": _raw_array_payload(edited_mask),
                        },
                        metadata={
                            "schema": "palette.web_labeling_subject_mask_checkpoint_metadata.v1",
                            "row_identity": row_identity,
                            "component_name": runtime.component_name,
                            "target_run_path": _subject_mask_target_run_path(runtime),
                            "source_rowset_path": _subject_mask_source_rowset_path(runtime),
                        },
                    )
                    result = {
                        "roi_idx": roi_idx,
                        "frame_idx": frame_idx,
                        "component_name": runtime.component_name,
                        "source_run": str(runtime.source.run_name),
                        "refined_run": str(runtime.refined.run_name),
                        "canonical_area_px": int(canonical_mask.sum()),
                        "checkpoint_area_px": int(edited_mask.sum()),
                        "mask_changed_vs_canonical": bool(not np.array_equal(canonical_mask, edited_mask)),
                        "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
                        "target_edit_revision": target_edit_revision,
                        "canonical_zarr_mutated": False,
                    }
                    mutation_event = state.store.record_event(
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        event_type="checkpoint_subject_mask_roi",
                        target={
                            "roi_idx": roi_idx,
                            "frame_idx": frame_idx,
                            "component_name": runtime.component_name,
                            "source_run": str(runtime.source.run_name),
                            "refined_run": str(runtime.refined.run_name),
                            "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
                        },
                        before={"canonical_area_px": int(canonical_mask.sum()), "edit_revision": target_edit_revision},
                        after={
                            "checkpoint_area_px": int(edited_mask.sum()),
                            "mask_changed_vs_canonical": bool(not np.array_equal(canonical_mask, edited_mask)),
                            "canonical_zarr_mutated": False,
                        },
                    )
                    if bool(body.get("advance", runtime.auto_advance_on_save)):
                        if runtime.position < total - 1:
                            runtime.position += 1
                except Exception as exc:
                    self._write_json(_format_error("save_error", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json(
                    _redact_labeler_runtime_payload(
                        {
                            "ok": True,
                            "result": result,
                            "mutation": _browser_mutation_response_metadata(
                                workflow_kind="subject_mask_component",
                                session=session,
                                mutation_event=mutation_event,
                                operator_validation_mutation_gate=_runtime_operator_validation_mutation_gate(state.config),
                            ),
                            "state": _subject_mask_runtime_state(runtime, store=state.store),
                        }
                    )
                )
                return True

            if subject_mask_path == "/apply":
                if self._reject_browser_mutation_preflight(session, body, runtime):
                    return True
                apply_id = str(body.get("apply_id") or "").strip() or str(uuid.uuid4())
                claimed_apply_id: str | None = None
                canonical_write_started = False
                try:
                    already_applied = state.store.get_applied_session_checkpoints_by_apply_id(
                        task_id=runtime.task_id,
                        apply_id=apply_id,
                    )
                    if already_applied:
                        before_values = [
                            int(row.get("edit_revision_before") or 0)
                            for row in already_applied
                            if row.get("edit_revision_before") is not None
                        ]
                        after_values = [
                            int(row.get("edit_revision_after") or 0)
                            for row in already_applied
                            if row.get("edit_revision_after") is not None
                        ]
                        result = {
                            "apply_id": apply_id,
                            "already_applied": True,
                            "applied_checkpoint_count": len(already_applied),
                            "edit_revision_before": before_values[0] if before_values else _subject_mask_edit_revision(runtime),
                            "edit_revision_after": after_values[0] if after_values else _subject_mask_edit_revision(runtime),
                        }
                        mutation_event = state.store.record_event(
                            task_id=runtime.task_id,
                            recording_id=runtime.recording_id,
                            user=user,
                            event_type="apply_subject_mask_session_checkpoints_idempotent_retry",
                            target={"apply_id": apply_id},
                            after=result,
                            )
                    else:
                        checkpoints = state.store.claim_session_checkpoints_for_apply(
                            task_id=runtime.task_id,
                            component_name=runtime.component_name,
                            apply_id=apply_id,
                        )
                        claimed_apply_id = apply_id if checkpoints else None
                        target_path = _subject_mask_target_run_path(runtime)
                        source_rowset_path = _subject_mask_source_rowset_path(runtime)
                        if not checkpoints:
                            result = {
                                "apply_id": apply_id,
                                "already_applied": False,
                                "applied_checkpoint_count": 0,
                                "edit_revision_before": _subject_mask_edit_revision(runtime),
                                "edit_revision_after": _subject_mask_edit_revision(runtime),
                            }
                            mutation_event = state.store.record_event(
                                task_id=runtime.task_id,
                                recording_id=runtime.recording_id,
                                user=user,
                                event_type="apply_subject_mask_session_checkpoints_noop",
                                target={"apply_id": apply_id, "component_name": runtime.component_name},
                                after=result,
                            )
                        else:
                            edit_revision_before = _subject_mask_edit_revision(runtime)
                            checkpoint_ids: list[str] = []
                            applied_rows: list[int] = []
                            edited_stacks: list[np.ndarray] = []
                            before_area_total = 0
                            after_area_total = 0
                            compute_workers_used = 1
                            stale_checkpoint_ids: list[str] = []
                            stale_rows: list[int] = []
                            scoped_row_set = set(int(value) for value in runtime.roi_indices.tolist())
                            for checkpoint in checkpoints:
                                checkpoint_target_path = str(checkpoint.get("target_run_path") or "")
                                if checkpoint_target_path != target_path:
                                    raise ValueError(
                                        f"checkpoint target mismatch: expected {target_path}, got {checkpoint_target_path}"
                                    )
                                checkpoint_source_rowset = str(checkpoint.get("source_rowset_path") or "")
                                if checkpoint_source_rowset and checkpoint_source_rowset != source_rowset_path:
                                    raise ValueError(
                                        f"checkpoint source rowset mismatch: expected {source_rowset_path}, got {checkpoint_source_rowset}"
                                    )
                                checkpoint_revision = int(checkpoint.get("target_edit_revision") or 0)
                                roi_idx = int(checkpoint.get("roi_idx") or 0)
                                if roi_idx not in scoped_row_set:
                                    raise ValueError(f"checkpoint row {roi_idx} is outside the active task row scope.")
                                if checkpoint_revision != edit_revision_before:
                                    stale_checkpoint_ids.append(str(checkpoint.get("checkpoint_id") or ""))
                                    stale_rows.append(roi_idx)
                                    continue
                                metadata = checkpoint.get("metadata")
                                if isinstance(metadata, Mapping):
                                    expected_identity = metadata.get("row_identity")
                                    if isinstance(expected_identity, Mapping):
                                        current_identity = _subject_mask_row_identity(runtime, roi_idx)
                                        for key, expected_value in expected_identity.items():
                                            if key not in current_identity:
                                                continue
                                            if str(current_identity.get(key)) != str(expected_value):
                                                raise ValueError(
                                                    f"checkpoint row identity mismatch for row {roi_idx}, field {key}: "
                                                    f"expected {expected_value}, got {current_identity.get(key)}"
                                                )
                                edited_mask = _subject_mask_checkpoint_mask(checkpoint)
                                current_stack = np.asarray(runtime.refined.group["masks_roi"][roi_idx], dtype=np.uint8)
                                before_mask = (np.asarray(current_stack[runtime.comp_idx], dtype=np.uint8) > 0).astype(np.uint8)
                                if tuple(edited_mask.shape) != tuple(before_mask.shape):
                                    raise ValueError(
                                        f"checkpoint mask shape mismatch for row {roi_idx}: "
                                        f"expected {tuple(before_mask.shape)}, got {tuple(edited_mask.shape)}"
                                    )
                                edited_stack = current_stack.copy()
                                edited_stack[runtime.comp_idx] = edited_mask
                                checkpoint_ids.append(str(checkpoint.get("checkpoint_id") or ""))
                                applied_rows.append(roi_idx)
                                edited_stacks.append(edited_stack)
                                before_area_total += int(before_mask.sum())
                                after_area_total += int(edited_mask.sum())
                            if not edited_stacks:
                                released_stale_checkpoint_count = 0
                                if stale_checkpoint_ids:
                                    released_stale_checkpoint_count = state.store.release_session_checkpoints_apply(
                                        task_id=runtime.task_id,
                                        apply_id=apply_id,
                                    )
                                edit_revision_current = _subject_mask_edit_revision(runtime)
                                result = {
                                    "apply_id": apply_id,
                                    "already_applied": False,
                                    "applied_checkpoint_count": 0,
                                    "requested_checkpoint_count": len(checkpoints),
                                    "stale_checkpoint_count": len(stale_checkpoint_ids),
                                    "stale_rows": stale_rows,
                                    "released_stale_checkpoint_count": int(released_stale_checkpoint_count),
                                    "skipped_checkpoint_count": len(stale_checkpoint_ids),
                                    "component_name": runtime.component_name,
                                    "rows": [],
                                    "edit_revision_before": edit_revision_current,
                                    "edit_revision_after": edit_revision_current,
                                    "before_area_px_total": 0,
                                    "after_area_px_total": 0,
                                    "compute_workers": 0,
                                    "canonical_zarr_mutated": False,
                                }
                                mutation_event = state.store.record_event(
                                    task_id=runtime.task_id,
                                    recording_id=runtime.recording_id,
                                    user=user,
                                    event_type="apply_subject_mask_session_checkpoints_stale_skipped",
                                    target={
                                        "apply_id": apply_id,
                                        "component_name": runtime.component_name,
                                        "refined_run": str(runtime.refined.run_name),
                                        "target_run_path": target_path,
                                    },
                                    after=result,
                                )
                            else:
                                if edited_stacks:
                                    compute_worker_limit_raw = str(
                                        os.environ.get("PALETTE_SUBJECT_MASK_APPLY_COMPUTE_WORKERS", "4")
                                    ).strip()
                                    try:
                                        compute_worker_limit = int(compute_worker_limit_raw)
                                    except ValueError:
                                        compute_worker_limit = 4
                                    compute_workers = max(
                                        1,
                                        min(
                                            max(1, compute_worker_limit),
                                            len(edited_stacks),
                                            max(1, int(os.cpu_count() or 1)),
                                        ),
                                    )
                                    compute_workers_used = int(compute_workers)
                                    canonical_write_started = True
                                    review_mod._apply_refined_subject_roi_rows(  # type: ignore[attr-defined]
                                        source=runtime.source,
                                        refined=runtime.refined,
                                        roi_indices=applied_rows,
                                        edited_masks_batch=np.stack(edited_stacks, axis=0),
                                        component_names=(runtime.component_name,),
                                        update_mode="browser_session_apply",
                                        update_method="palette_web_labeling_session_apply_v1",
                                        update_reason="web_labeling_subject_mask_session_apply",
                                        compute_workers=compute_workers,
                                    )
                                edit_revision_after = int(edit_revision_before) + 1
                                runtime.refined.group.attrs["edit_revision"] = int(edit_revision_after)
                                runtime.refined.group.attrs["edit_revision_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                                runtime.refined.group.attrs["edit_revision_last_apply_id"] = apply_id
                                if "mask_rle" in runtime.refined.group:
                                    runtime.refined.group.attrs["mask_rle_stale_since_edit_revision"] = int(edit_revision_after)
                                updated_count = state.store.mark_session_checkpoints_applied(
                                    checkpoint_ids=checkpoint_ids,
                                    apply_id=apply_id,
                                    edit_revision_before=edit_revision_before,
                                    edit_revision_after=edit_revision_after,
                                )
                                released_stale_checkpoint_count = 0
                                if stale_checkpoint_ids:
                                    released_stale_checkpoint_count = state.store.release_session_checkpoints_apply(
                                        task_id=runtime.task_id,
                                        apply_id=apply_id,
                                    )
                                result = {
                                    "apply_id": apply_id,
                                    "already_applied": False,
                                    "applied_checkpoint_count": int(updated_count),
                                    "requested_checkpoint_count": len(checkpoints),
                                    "stale_checkpoint_count": len(stale_checkpoint_ids),
                                    "stale_rows": stale_rows,
                                    "released_stale_checkpoint_count": int(released_stale_checkpoint_count),
                                    "skipped_checkpoint_count": len(stale_checkpoint_ids),
                                    "component_name": runtime.component_name,
                                    "rows": applied_rows,
                                    "edit_revision_before": edit_revision_before,
                                    "edit_revision_after": edit_revision_after,
                                    "before_area_px_total": before_area_total,
                                    "after_area_px_total": after_area_total,
                                    "compute_workers": int(compute_workers_used),
                                    "canonical_zarr_mutated": True,
                                }
                                mutation_event = state.store.record_event(
                                    task_id=runtime.task_id,
                                    recording_id=runtime.recording_id,
                                    user=user,
                                    event_type="apply_subject_mask_session_checkpoints",
                                    target={
                                        "apply_id": apply_id,
                                        "component_name": runtime.component_name,
                                        "refined_run": str(runtime.refined.run_name),
                                        "target_run_path": target_path,
                                    },
                                    before={
                                        "edit_revision": edit_revision_before,
                                        "area_px_total": before_area_total,
                                    },
                                    after={
                                        "edit_revision": edit_revision_after,
                                        "area_px_total": after_area_total,
                                        "applied_checkpoint_count": int(updated_count),
                                        "compute_workers": int(compute_workers_used),
                                    },
                                )
                                _refresh_registry_for_scope(
                                    store=state.store,
                                    task_id=runtime.task_id,
                                    recording_id=runtime.recording_id,
                                    user=user,
                                    workflow_kind="subject_mask_component",
                                    scope=_session_scope(session),
                                    zarr_path=runtime.zarr_path,
                                    dataset_id=str(session.get("dataset_id") or "") or None,
                                    zarr_use=str(session.get("zarr_use") or "") or None,
                                )
                except Exception as exc:
                    if claimed_apply_id and not canonical_write_started:
                        try:
                            state.store.release_session_checkpoints_apply(
                                task_id=runtime.task_id,
                                apply_id=claimed_apply_id,
                            )
                        except Exception:
                            pass
                    self._write_json(_format_error("apply_error", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json(
                    _redact_labeler_runtime_payload(
                        {
                            "ok": True,
                            "result": result,
                            "mutation": _browser_mutation_response_metadata(
                                workflow_kind="subject_mask_component",
                                session=session,
                                mutation_event=mutation_event,
                                operator_validation_mutation_gate=_runtime_operator_validation_mutation_gate(state.config),
                            ),
                            "state": _subject_mask_runtime_state(runtime, store=state.store),
                        }
                    )
                )
                return True

            if subject_mask_path == "/review-status":
                if self._reject_browser_mutation_preflight(session, body, runtime):
                    return True
                unapplied_count = state.store.count_unapplied_session_checkpoints(
                    task_id=runtime.task_id,
                    component_name=runtime.component_name,
                )
                if int(unapplied_count) > 0:
                    self._write_json(
                        _format_error(
                            "unapplied_session_edits",
                            details=(
                                "Apply saved subject-mask edits to Zarr before changing component review status."
                            ),
                            status=HTTPStatus.CONFLICT,
                            extra={
                                "unapplied_session_edit_count": int(unapplied_count),
                                "required_action": "apply_saved_edits_to_zarr",
                            },
                        ),
                        status=HTTPStatus.CONFLICT,
                    )
                    return True
                requested_state = str(body.get("state") or "").strip()
                if not requested_state:
                    self._write_json(_format_error("payload_validation", details="Missing review state."), status=HTTPStatus.BAD_REQUEST)
                    return True
                if requested_state not in set(review_mod.REVIEW_STATE_CHOICES):
                    self._write_json(_format_error("payload_validation", details=f"Unsupported review state: {requested_state}"), status=HTTPStatus.BAD_REQUEST)
                    return True
                try:
                    before_component_reviews = runtime.refined.group.attrs.get("component_review_statuses")
                    before_run_review = runtime.refined.group.attrs.get("refined_subject_mask_review_status")
                    component_payload, run_payload = review_mod.apply_component_review_status(
                        runtime.refined.parent,
                        str(runtime.refined.run_name),
                        runtime.refined.group,
                        component_name=runtime.component_name,
                        state=requested_state,
                        method=str(body.get("method") or runtime.review_method or "manual"),
                        intended_use=str(body.get("intended_use") or runtime.review_intended_use or "training"),
                        reviewer=user,
                        notes=str(body.get("notes") or runtime.review_notes or "").strip() or None,
                        zarr_path=runtime.zarr_path,
                    )
                    mutation_event = state.store.record_event(
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        event_type="set_review_status",
                        target={
                            "component_name": runtime.component_name,
                            "refined_run": str(runtime.refined.run_name),
                        },
                        before={
                            "component_review_statuses": dict(before_component_reviews) if isinstance(before_component_reviews, Mapping) else None,
                            "run_review_status": dict(before_run_review) if isinstance(before_run_review, Mapping) else None,
                        },
                        after={
                            "component_review_status": component_payload,
                            "run_review_status": run_payload,
                        },
                    )
                    _refresh_registry_for_scope(
                        store=state.store,
                        task_id=runtime.task_id,
                        recording_id=runtime.recording_id,
                        user=user,
                        workflow_kind="subject_mask_component",
                        scope=_session_scope(session),
                        zarr_path=runtime.zarr_path,
                        dataset_id=str(session.get("dataset_id") or "") or None,
                        zarr_use=str(session.get("zarr_use") or "") or None,
                    )
                except Exception as exc:
                    self._write_json(_format_error("review_status_error", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                    return True
                self._write_json(
                    _redact_labeler_runtime_payload(
                        {
                            "ok": True,
                            "result": {
                                "component_review_status": component_payload,
                                "run_review_status": run_payload,
                            },
                            "mutation": _browser_mutation_response_metadata(
                                workflow_kind="subject_mask_component",
                                session=session,
                                mutation_event=mutation_event,
                                operator_validation_mutation_gate=_runtime_operator_validation_mutation_gate(state.config),
                            ),
                            "state": _subject_mask_runtime_state(runtime, store=state.store),
                        }
                    )
                )
                return True

            return False

        def do_GET(self) -> None:  # noqa: N802
            if self._handle_flask_if_claimed():
                return
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "":
                user, _auth_source = self._require_user(html_error=True)
                if user is None:
                    return
                query = parse_qs(parsed.query, keep_blank_values=True)
                expected_user = str((query.get("expected_user") or [""])[-1]).strip()
                if expected_user and str(user) != expected_user:
                    self._write_error(
                        "dashboard_user_mismatch",
                        details=(
                            f"This dashboard link is for {expected_user}, "
                            f"but the browser is authenticated as {user}. "
                            "Stop and contact the operator before labeling."
                        ),
                        status=HTTPStatus.FORBIDDEN,
                        html_error=True,
                        extra=_labeler_read_authorization_denial_metadata(
                            user=user,
                            expected_user=expected_user,
                            route_path=path,
                            response_kind="html",
                        ),
                    )
                    return
                known_user_status = self._require_active_labeling_user(
                    user,
                    html_error=True,
                    expected_user=expected_user or user,
                    route_path=path,
                )
                if known_user_status is None:
                    return
                self._write(
                    _dashboard_html()
                    if path in {DASHBOARD_PATH, PERSONAL_WORK_PATH}
                    else _datasets_html()
                )
                return
            if path.startswith("/t/"):
                token = path[len("/t/") :].strip("/")
                if not token:
                    signed_link_policy = _browser_signed_link_policy()
                    mutation_write_policy = _browser_mutation_write_policy()
                    self._write_error(
                        "missing_token",
                        status=HTTPStatus.NOT_FOUND,
                        html_error=True,
                        extra={
                            "signed_link_policy": signed_link_policy,
                            "signed_link_contract": _signed_link_contract_policy(
                                signed_link_policy
                            ),
                            "browser_mutation_write_policy": mutation_write_policy,
                            "browser_mutation_write_contract": _browser_mutation_write_contract_policy(
                                mutation_write_policy
                            ),
                        },
                    )
                    return
                if not state.config.link_secret:
                    signed_link_policy = _browser_signed_link_policy()
                    mutation_write_policy = _browser_mutation_write_policy()
                    self._write_error(
                        "signed_links_disabled",
                        status=HTTPStatus.NOT_FOUND,
                        html_error=True,
                        extra={
                            "signed_links_enabled": False,
                            "signed_link_policy": signed_link_policy,
                            "signed_link_contract": _signed_link_contract_policy(
                                signed_link_policy
                            ),
                            "browser_mutation_write_policy": mutation_write_policy,
                            "browser_mutation_write_contract": _browser_mutation_write_contract_policy(
                                mutation_write_policy
                            ),
                        },
                    )
                    return
                user, auth_source = self._require_user(html_error=True)
                if user is None:
                    return
                try:
                    payload = _verify_signed_task_link_token(token, secret=state.config.link_secret)
                    revocation_reason = _signed_task_link_revocation_reason(
                        payload,
                        not_before_utc=state.config.link_not_before_utc,
                    )
                    if revocation_reason:
                        expected_user = str(payload.get("expected_user") or "").strip()
                        task = state.store.get_task(str(payload.get("task_id") or ""))
                        operator_validation_start_gate = _runtime_operator_validation_start_gate(
                            state.config
                        )
                        self._write_error(
                            "signed_link_revoked",
                            details=revocation_reason,
                            status=HTTPStatus.FORBIDDEN,
                            html_error=True,
                            extra={
                                "authorization_context": _labeler_authorization_context(
                                    user=user,
                                    expected_user=expected_user,
                                    task=task if task is not None else None,
                                ),
                                "task_open_authorization_contract": _task_open_authorization_contract(
                                    user=user,
                                    expected_user=expected_user,
                                    task=task if task is not None else None,
                                    ready=False,
                                    not_ready_reason="signed_link_revoked",
                                    session_created_server_side=False,
                                    server_authorizes_open=False,
                                    operator_validation_start_gate=operator_validation_start_gate,
                                ),
                            },
                        )
                        return
                    expected_user = str(payload.get("expected_user") or "").strip()
                    operator_validation_start_gate = _runtime_operator_validation_start_gate(
                        state.config
                    )
                    if expected_user and str(user) != expected_user:
                        task = state.store.get_task(str(payload["task_id"]))
                        personal_dataset_queue_url = _dashboard_url_for_expected_user(
                            PERSONAL_DATASET_QUEUE_PATH,
                            expected_user,
                        )
                        browser_mutation_write_policy = _browser_mutation_write_policy()
                        browser_mutation_write_checklist = (
                            _browser_mutation_write_runtime_checklist(
                                browser_mutation_write_policy
                            )
                        )
                        signed_link_mismatch_extra = {
                            "authorization_context": _labeler_authorization_context(
                                user=user,
                                expected_user=expected_user,
                                task=task if task is not None else None,
                            ),
                            "expected_user": expected_user,
                            "expected_user_personal_dataset_queue_url": (
                                personal_dataset_queue_url
                            ),
                            "preferred_labeler_entrypoint": (
                                "personal_datasets_waiting_queue"
                            ),
                            "preferred_labeler_entry_url": personal_dataset_queue_url,
                            "personalized_labeler_entrypoint": (
                                "personal_datasets_waiting_queue"
                            ),
                            "personalized_labeler_entry_url": (
                                personal_dataset_queue_url
                            ),
                            "personal_dataset_queue_link_role": "preferred_queue",
                            "dataset_queue_link_role": "canonical_queue_fallback",
                            "canonical_dataset_queue_link_role": (
                                "canonical_queue_fallback"
                            ),
                            "personalized_labeler_entry_url_matches_personal_dataset_queue": True,
                            "browser_mutation_write_policy": (
                                browser_mutation_write_policy
                            ),
                            "browser_mutation_write_checklist": (
                                browser_mutation_write_checklist
                            ),
                            "dataset_queue_direct_start_policy": (
                                _dataset_queue_direct_start_policy()
                            ),
                            "task_open_authorization_contract": _task_open_authorization_contract(
                                user=user,
                                expected_user=expected_user,
                                task=task if task is not None else None,
                                ready=False,
                                not_ready_reason="signed_link_user_mismatch",
                                session_created_server_side=False,
                                server_authorizes_open=False,
                                operator_validation_start_gate=operator_validation_start_gate,
                            ),
                        }
                        _add_payload_contract_compact_fields(signed_link_mismatch_extra)
                        _add_task_open_personalized_launch_metadata(
                            signed_link_mismatch_extra,
                            authorization_context=signed_link_mismatch_extra[
                                "authorization_context"
                            ],
                        )
                        self._write_error(
                            "signed_link_user_mismatch",
                            details=(
                                f"This signed task link is for {expected_user}, "
                                f"but the browser is authenticated as {user}. "
                                "Stop and contact the operator before labeling."
                            ),
                            status=HTTPStatus.FORBIDDEN,
                            html_error=True,
                            extra=signed_link_mismatch_extra,
                        )
                        return
                    open_error = _task_open_preflight_error(
                        state.store,
                        task_id=str(payload["task_id"]),
                        user=user,
                    )
                    if open_error is not None:
                        error, details, status = open_error
                        task = state.store.get_task(str(payload["task_id"]))
                        self._write_error(
                            error,
                            details=details,
                            status=status,
                            html_error=True,
                            extra={
                                "authorization_context": _labeler_authorization_context(
                                    user=user,
                                    expected_user=expected_user,
                                    task=task if task is not None else None,
                                ),
                                "task_open_authorization_contract": _task_open_authorization_contract(
                                    user=user,
                                    expected_user=expected_user,
                                    task=task if task is not None else None,
                                    ready=False,
                                    not_ready_reason=error,
                                    session_created_server_side=False,
                                    server_authorizes_open=False,
                                    operator_validation_start_gate=operator_validation_start_gate,
                                ),
                            },
                        )
                        return
                    if bool(operator_validation_start_gate.get("blocks_task_open")):
                        task = state.store.get_task(str(payload["task_id"]))
                        self._write_error(
                            "operator_validation_start_blocked",
                            details=(
                                str(operator_validation_start_gate.get("operator_action") or "")
                                or "Required operator validation evidence is incomplete, so the server did not create a labeling session."
                            ),
                            status=HTTPStatus.CONFLICT,
                            html_error=True,
                            extra={
                                "authorization_context": _labeler_authorization_context(
                                    user=user,
                                    expected_user=expected_user,
                                    task=task if task is not None else None,
                                ),
                                "task_open_authorization_contract": _task_open_authorization_contract(
                                    user=user,
                                    expected_user=expected_user,
                                    task=task if task is not None else None,
                                    ready=False,
                                    not_ready_reason="operator_validation_start_blocked",
                                    session_created_server_side=False,
                                    server_authorizes_open=False,
                                    operator_validation_start_gate=operator_validation_start_gate,
                                ),
                            },
                        )
                        return
                    lease = state.store.create_session(
                        task_id=str(payload["task_id"]),
                        user=user,
                        ttl_seconds=state.config.session_ttl_seconds,
                        client_label=f"signed_link:{auth_source}",
                    )
                    _drop_runtime_sessions(state, lease.superseded_session_ids)
                except PermissionError as exc:
                    expected_user = str(payload.get("expected_user") or "").strip()
                    task = state.store.get_task(str(payload.get("task_id") or ""))
                    if "previous-owner sessions" in str(exc):
                        self._write_error(
                            "reassignment_session_safety_failed",
                            details=_labeler_safe_error_details(exc),
                            status=HTTPStatus.CONFLICT,
                            html_error=True,
                            extra={
                                "authorization_context": _labeler_authorization_context(
                                    user=user,
                                    expected_user=expected_user,
                                    task=task if task is not None else None,
                                ),
                                "task_open_authorization_contract": _task_open_authorization_contract(
                                    user=user,
                                    expected_user=expected_user,
                                    task=task if task is not None else None,
                                    ready=False,
                                    not_ready_reason="reassignment_session_safety_failed",
                                    session_created_server_side=False,
                                    server_authorizes_open=False,
                                    reassignment_session_safety_checked_server_side=True,
                                    reassignment_session_safety_passed=False,
                                    operator_validation_start_gate=operator_validation_start_gate,
                                ),
                            },
                        )
                    else:
                        self._write_error(
                            "not_assigned",
                            details=_labeler_safe_error_details(exc),
                            status=HTTPStatus.FORBIDDEN,
                            html_error=True,
                            extra={
                                "authorization_context": _labeler_authorization_context(
                                    user=user,
                                    expected_user=expected_user,
                                    task=task if task is not None else None,
                                ),
                                "task_open_authorization_contract": _task_open_authorization_contract(
                                    user=user,
                                    expected_user=expected_user,
                                    task=task if task is not None else None,
                                    ready=False,
                                    not_ready_reason="not_assigned",
                                    session_created_server_side=False,
                                    server_authorizes_open=False,
                                    operator_validation_start_gate=operator_validation_start_gate,
                                ),
                            },
                        )
                    return
                except Exception as exc:
                    signed_link_policy = _browser_signed_link_policy()
                    mutation_write_policy = _browser_mutation_write_policy()
                    self._write_error(
                        "signed_link_failed",
                        details=_labeler_safe_error_details(exc),
                        status=HTTPStatus.BAD_REQUEST,
                        html_error=True,
                        extra={
                            "signed_link_policy": signed_link_policy,
                            "signed_link_contract": _signed_link_contract_policy(
                                signed_link_policy
                            ),
                            "browser_mutation_write_policy": mutation_write_policy,
                            "browser_mutation_write_contract": _browser_mutation_write_contract_policy(
                                mutation_write_policy
                            ),
                        },
                    )
                    return
                self._redirect(f"/r/{lease.session_id}")
                return
            if path.startswith("/r/"):
                session_id = path[len("/r/") :].strip("/")
                if not session_id:
                    self._write_error("missing_session_id", status=HTTPStatus.NOT_FOUND, html_error=True)
                    return
                user, _auth_source = self._require_user(html_error=True)
                if user is None:
                    return
                session = self._session_for_user(session_id, user, html_error=True)
                if session is None:
                    return
                self._write(_session_html(session))
                return
            session_api = self._parse_session_api_path(path)
            if session_api is not None:
                user, _auth_source = self._require_user()
                if user is None:
                    return
                session_id, suffix = session_api
                session = self._session_for_user(session_id, user)
                if session is None:
                    return
                if self._handle_keypoint_get(session, suffix):
                    return
                if self._handle_detect_get(session, suffix):
                    return
                if self._handle_video_detect_get(session, suffix):
                    return
                if self._handle_subject_mask_get(session, suffix):
                    return
                self._write_json(_format_error("not_found", status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                return
            self._write_json(_format_error("not_found", status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self._handle_flask_if_claimed():
                return
            parsed = urlparse(self.path)
            path = parsed.path
            user, _auth_source = self._require_user()
            if user is None:
                return
            if state.config.csrf_same_origin and not _request_has_same_origin(self):
                self._write_json(
                    _format_error(
                        "same_origin_required",
                        details="Cross-origin mutating requests are rejected.",
                        status=HTTPStatus.FORBIDDEN,
                    ),
                    status=HTTPStatus.FORBIDDEN,
                )
                return
            try:
                body = _read_json_body(self)
            except Exception as exc:
                self._write_json(_format_error("invalid_json", details=_labeler_safe_error_details(exc)), status=HTTPStatus.BAD_REQUEST)
                return

            if not path.startswith("/api/admin/"):
                if self._require_active_labeling_user(
                    user,
                    expected_user=str(body.get("expected_user") or user),
                    route_path=path,
                ) is None:
                    return

            if path == "/api/admin/sessions/cleanup-stale":
                if not _is_admin_user(user, state.config):
                    self._write_json(_format_error("admin_required", status=HTTPStatus.FORBIDDEN), status=HTTPStatus.FORBIDDEN)
                    return
                try:
                    cleaned = state.store.cleanup_stale_sessions(user=user)
                except Exception as exc:
                    self._write_json(_format_error("stale_session_cleanup_failed", details=str(exc)), status=HTTPStatus.BAD_REQUEST)
                    return
                closed_session_ids = [str(session.get("session_id") or "") for session in cleaned]
                if closed_session_ids:
                    _drop_runtime_sessions(state, closed_session_ids)
                closed_session_payload = _closed_session_response_payload(state.store, closed_session_ids)
                self._write_json(
                    {
                        "ok": True,
                        "closed_count": len(cleaned),
                        "sessions": cleaned,
                        **closed_session_payload,
                    }
                )
                return

            if path == "/api/admin/invites/sign":
                if not _is_admin_user(user, state.config):
                    self._write_json(_format_error("admin_required", status=HTTPStatus.FORBIDDEN), status=HTTPStatus.FORBIDDEN)
                    return
                if not state.config.link_secret:
                    self._write_json(
                        _format_error(
                            "signed_invites_disabled",
                            details=(
                                f"Invite signing requires --link-secret or {LINK_SECRET_ENV_VAR} "
                                "when starting the labeling server."
                            ),
                            status=HTTPStatus.CONFLICT,
                        ),
                        status=HTTPStatus.CONFLICT,
                    )
                    return
                invite_user = str(body.get("user") or body.get("assignee_user") or "").strip()
                if not invite_user:
                    self._write_json(_format_error("payload_validation", details="Missing user."), status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    ttl_seconds = int(body.get("ttl_seconds") or SIGNED_INVITE_DEFAULT_TTL_SECONDS)
                except (TypeError, ValueError):
                    self._write_json(_format_error("payload_validation", details="ttl_seconds must be an integer."), status=HTTPStatus.BAD_REQUEST)
                    return
                base_url = str(body.get("base_url") or "").rstrip("/")
                try:
                    token_info = _signed_invite_token_info(
                        user=invite_user,
                        secret=state.config.link_secret,
                        ttl_seconds=ttl_seconds,
                    )
                except Exception as exc:
                    self._write_json(_format_error("invite_signing_failed", details=str(exc)), status=HTTPStatus.BAD_REQUEST)
                    return
                token = str(token_info["token"])
                invite_path = _signed_invite_path(token, user=invite_user)
                known_user_status = _known_labeler_status(state.store, invite_user)
                shareability_warnings = []
                if not base_url:
                    shareability_warnings.append(
                        {
                            "code": "missing_base_url",
                            "details": "Generated invite URL is service-relative because no base_url was supplied.",
                        }
                    )
                if not bool(known_user_status.get("is_active_labeling_user")):
                    shareability_warnings.append(
                        {
                            "code": "inactive_or_unknown_labeling_user",
                            "user": invite_user,
                            "details": "This user is not active in the labeling_users SQLite table; add or activate the user before sharing an invite.",
                        }
                    )
                self._write_json(
                    {
                        "ok": True,
                        "schema": "palette.web_labeling_admin_signed_invite.v1",
                        "operator_user": user,
                        "user": invite_user,
                        "expected_user": invite_user,
                        "scope": token_info["scope"],
                        "issued_at_utc": token_info["issued_at_utc"],
                        "expires_at_utc": token_info["expires_at_utc"],
                        "expires_in_seconds": token_info["ttl_seconds"],
                        "path": invite_path,
                        "url": f"{base_url}{invite_path}" if base_url else invite_path,
                        "url_is_absolute": bool(base_url),
                        "known_user_status": known_user_status,
                        "ready_to_share": bool(base_url) and not shareability_warnings,
                        "shareability_warnings": shareability_warnings,
                    }
                )
                return

            if path == "/api/admin/users":
                if not _is_admin_user(user, state.config):
                    self._write_json(_format_error("admin_required", status=HTTPStatus.FORBIDDEN), status=HTTPStatus.FORBIDDEN)
                    return
                user_id = str(body.get("user_id") or body.get("user") or "").strip()
                if not user_id:
                    self._write_json(_format_error("payload_validation", details="Missing user_id."), status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    labeling_user = state.store.upsert_labeling_user(
                        user_id=user_id,
                        display_name=str(body.get("display_name") or "").strip() or None,
                        email=str(body.get("email") or "").strip() or None,
                        role=str(body.get("role") or "labeler").strip() or "labeler",
                        status=str(body.get("status") or "active").strip() or "active",
                        notes=str(body.get("notes") or "").strip() or None,
                        actor_user=user,
                    )
                except Exception as exc:
                    self._write_json(_format_error("labeling_user_update_failed", details=str(exc)), status=HTTPStatus.BAD_REQUEST)
                    return
                notification_result = None
                if _request_truthy(body.get("notify") or body.get("send_notification")):
                    try:
                        notification_result = send_labeler_added_notification(
                            user=labeling_user,
                            actor_user=user,
                            config=_notification_config_from_values(
                                mode=str(body.get("notification_mode") or "").strip() or None,
                                base_url=str(body.get("notification_base_url") or body.get("base_url") or "").strip() or None,
                            ),
                        )
                    except Exception as exc:
                        notification_result = _notification_exception_result(
                            kind="labeler_added",
                            to_user=str(labeling_user.get("user_id") or user_id),
                            exc=exc,
                        )
                    state.store.record_labeling_user_event(
                        user_id=str(labeling_user.get("user_id") or user_id),
                        actor_user=user,
                        event_type=_notification_event_type(
                            notification_result,
                            prefix="labeling_user_notification",
                        ),
                        after=notification_result,
                    )
                self._write_json(
                    {
                        "ok": True,
                        "schema": "palette.web_labeling_admin_user_update.v1",
                        "operator_user": user,
                        "user": labeling_user,
                        "known_user_status": _known_labeler_status(state.store, str(labeling_user.get("user_id") or user_id)),
                        "notification": notification_result,
                    }
                )
                return

            if path.startswith("/api/admin/users/") and (
                path.endswith("/activate") or path.endswith("/deactivate")
            ):
                if not _is_admin_user(user, state.config):
                    self._write_json(_format_error("admin_required", status=HTTPStatus.FORBIDDEN), status=HTTPStatus.FORBIDDEN)
                    return
                action = "activate" if path.endswith("/activate") else "deactivate"
                suffix = f"/{action}"
                target_user = unquote(path[len("/api/admin/users/") : -len(suffix)].strip("/"))
                if not target_user:
                    self._write_json(_format_error("missing_user"), status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    labeling_user = (
                        state.store.activate_labeling_user(target_user, actor_user=user)
                        if action == "activate"
                        else state.store.deactivate_labeling_user(target_user, actor_user=user)
                    )
                except Exception as exc:
                    self._write_json(_format_error("labeling_user_status_update_failed", details=str(exc)), status=HTTPStatus.BAD_REQUEST)
                    return
                self._write_json(
                    {
                        "ok": True,
                        "schema": "palette.web_labeling_admin_user_status_update.v1",
                        "operator_user": user,
                        "action": action,
                        "user": labeling_user,
                        "known_user_status": _known_labeler_status(state.store, str(labeling_user.get("user_id") or target_user)),
                    }
                )
                return

            if path == "/api/admin/assignments":
                if not _is_admin_user(user, state.config):
                    self._write_json(_format_error("admin_required", status=HTTPStatus.FORBIDDEN), status=HTTPStatus.FORBIDDEN)
                    return
                recording_id = str(body.get("recording_id") or "").strip()
                assignee_user = str(body.get("assignee_user") or body.get("user") or "").strip()
                status = str(body.get("status") or "active").strip() or "active"
                notes = str(body.get("notes") or "").strip() or None
                if not recording_id:
                    self._write_json(_format_error("payload_validation", details="Missing recording_id."), status=HTTPStatus.BAD_REQUEST)
                    return
                if not assignee_user:
                    self._write_json(_format_error("payload_validation", details="Missing assignee_user."), status=HTTPStatus.BAD_REQUEST)
                    return
                assignee_status = _known_labeler_status(state.store, assignee_user)
                if not bool(assignee_status.get("is_active_labeling_user")):
                    self._write_json(
                        _format_error(
                            "inactive_or_unknown_assignee_user",
                            details=(
                                "Assignments can only be created or updated for users with an active "
                                "row in the labeling_users SQLite table. Add or activate the user first."
                            ),
                            status=HTTPStatus.BAD_REQUEST,
                            extra={"assignee_user_status": assignee_status},
                        ),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    transition_result = state.store.assign_recording_with_session_closure(
                        recording_id=recording_id,
                        assignee_user=assignee_user,
                        assigned_by=user,
                        status=status,
                        notes=notes,
                    )
                    assignment = transition_result["assignment"]
                    closed_sessions = list(transition_result.get("closed_sessions") or [])
                    closed_session_ids = [str(session["session_id"]) for session in closed_sessions]
                    _drop_runtime_sessions(state, closed_session_ids)
                except Exception as exc:
                    self._write_json(_format_error("assignment_update_failed", details=str(exc)), status=HTTPStatus.BAD_REQUEST)
                    return
                closed_session_payload = _closed_session_response_payload(state.store, closed_session_ids)
                notification_result = None
                if _request_truthy(body.get("notify") or body.get("send_notification")):
                    try:
                        notification_result = send_assignment_available_notification(
                            user=state.store.get_labeling_user(str(assignment.get("assignee_user") or assignee_user)),
                            assignment=assignment,
                            actor_user=user,
                            config=_notification_config_from_values(
                                mode=str(body.get("notification_mode") or "").strip() or None,
                                base_url=str(body.get("notification_base_url") or body.get("base_url") or "").strip() or None,
                            ),
                        )
                    except Exception as exc:
                        notification_result = _notification_exception_result(
                            kind="assignment_available",
                            to_user=str(assignment.get("assignee_user") or assignee_user),
                            exc=exc,
                        )
                    state.store.record_assignment_event(
                        recording_id=str(assignment.get("recording_id") or recording_id),
                        actor_user=user,
                        event_type=_notification_event_type(
                            notification_result,
                            prefix="assignment_notification",
                        ),
                        after=notification_result,
                    )
                self._write_json(
                    {
                        "ok": True,
                        "assignment": assignment,
                        "previous_assignment": transition_result.get("previous_assignment"),
                        "assignment_transition": transition_result.get("assignment_transition"),
                        "single_owner_policy": _assignment_ownership_policy(),
                        "closed_session_count": transition_result.get("closed_session_count", len(closed_sessions)),
                        "closed_session_ids": transition_result.get("closed_session_ids", []),
                        "closed_sessions": closed_sessions,
                        "session_closure_events": closed_session_payload["session_closure_events"],
                        "notification": notification_result,
                    }
                )
                return

            if path.startswith("/api/admin/recordings/") and path.endswith("/repair-reassignment-sessions"):
                if not _is_admin_user(user, state.config):
                    self._write_json(_format_error("admin_required", status=HTTPStatus.FORBIDDEN), status=HTTPStatus.FORBIDDEN)
                    return
                recording_id = unquote(
                    path[len("/api/admin/recordings/") : -len("/repair-reassignment-sessions")].strip("/")
                )
                if not recording_id:
                    self._write_json(_format_error("missing_recording_id"), status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    check_report_before = _store_consistency_report(state.store)
                    safety_before = (
                        check_report_before.get("reassignment_session_safety")
                        if isinstance(check_report_before.get("reassignment_session_safety"), Mapping)
                        else {}
                    )
                    sessions = state.store.close_assignment_mismatched_sessions_for_recording(
                        recording_id=recording_id,
                        user=user,
                    )
                    closed_session_ids = [
                        str(session.get("session_id") or "")
                        for session in sessions
                        if str(session.get("session_id") or "")
                    ]
                    if closed_session_ids:
                        _drop_runtime_sessions(state, closed_session_ids)
                    check_report_after = _store_consistency_report(state.store)
                except Exception as exc:
                    self._write_json(_format_error("reassignment_session_repair_failed", details=str(exc)), status=HTTPStatus.BAD_REQUEST)
                    return
                closed_session_payload = _closed_session_response_payload(state.store, closed_session_ids)
                self._write_json(
                    {
                        "ok": True,
                        "schema": "palette.web_labeling_reassignment_session_repair_report.v1",
                        "recording_id": recording_id,
                        "operator_user": user,
                        "reassignment_session_safety_before": safety_before,
                        "reassignment_session_safety_after": check_report_after.get(
                            "reassignment_session_safety",
                            {},
                        ),
                        "closed_count": len(closed_session_ids),
                        "closed_session_count": len(closed_session_ids),
                        "closed_session_ids": closed_session_ids,
                        "sessions": sessions,
                        "session_closure_events": closed_session_payload["session_closure_events"],
                    }
                )
                return

            if path.startswith("/api/admin/tasks/") and path.endswith("/repair"):
                if not _is_admin_user(user, state.config):
                    self._write_json(_format_error("admin_required", status=HTTPStatus.FORBIDDEN), status=HTTPStatus.FORBIDDEN)
                    return
                task_id = path[len("/api/admin/tasks/") : -len("/repair")].strip("/")
                reason = str(body.get("reason") or "").strip()
                requested_state = str(body.get("state") or "").strip()
                if not task_id:
                    self._write_json(_format_error("missing_task_id"), status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    before_task = state.store.get_task(task_id)
                    if before_task is None:
                        self._write_json(_format_error("task_not_found", status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                        return
                    previous_state = str(before_task.get("state") or "")
                    target_state = requested_state or ("pending" if previous_state == "complete" else previous_state)
                    if target_state == "complete":
                        self._write_json(
                            _format_error(
                                "payload_validation",
                                details="Use /api/admin/tasks/{task_id}/state for explicit completion; repair opens or preserves task state.",
                            ),
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return
                    closed_sessions = state.store.close_sessions_for_task(
                        task_id=task_id,
                        user=user,
                        event_type="session_closed_by_operator_repair",
                    )
                    closed_session_ids = [str(session.get("session_id") or "") for session in closed_sessions]
                    if closed_session_ids:
                        _drop_runtime_sessions(state, closed_session_ids)
                    updated = (
                        state.store.update_task_state(task_id=task_id, state=target_state, user=user)
                        if target_state != previous_state
                        else state.store.get_task(task_id)
                    )
                    if updated is None:
                        updated = before_task
                    repair_event = state.store.record_event(
                        task_id=task_id,
                        recording_id=str(before_task.get("recording_id") or ""),
                        user=user,
                        event_type="task_operator_repaired",
                        target={
                            "task_id": task_id,
                            "recording_id": str(before_task.get("recording_id") or ""),
                        },
                        before={
                            "state": previous_state,
                            "open_session_count": len(closed_sessions),
                        },
                        after={
                            "state": str(updated.get("state") or ""),
                            "closed_session_ids": closed_session_ids,
                            "reason": reason,
                        },
                    )
                except Exception as exc:
                    self._write_json(_format_error("task_repair_failed", details=str(exc)), status=HTTPStatus.BAD_REQUEST)
                    return
                self._write_json(
                    {
                        "ok": True,
                        "task": updated,
                        "previous_task": before_task,
                        "operator_repair": {
                            "task_id": task_id,
                            "recording_id": str(before_task.get("recording_id") or ""),
                            "previous_state": previous_state,
                            "new_state": str(updated.get("state") or ""),
                            "state_changed": previous_state != str(updated.get("state") or ""),
                            "closed_session_count": len(closed_session_ids),
                            "closed_session_ids": closed_session_ids,
                            "reason": reason,
                            "event_id": repair_event.get("event_id"),
                            "event_type": repair_event.get("event_type"),
                        },
                        "closed_session_count": len(closed_session_ids),
                        "closed_session_ids": closed_session_ids,
                        "session_closure_events": _closed_session_response_payload(
                            state.store,
                            closed_session_ids,
                            fallback_event_type="session_closed_by_operator_repair",
                        )["session_closure_events"],
                    }
                )
                return

            if path.startswith("/api/admin/tasks/") and path.endswith("/state"):
                if not _is_admin_user(user, state.config):
                    self._write_json(_format_error("admin_required", status=HTTPStatus.FORBIDDEN), status=HTTPStatus.FORBIDDEN)
                    return
                task_id = path[len("/api/admin/tasks/") : -len("/state")].strip("/")
                new_state = str(body.get("state") or "").strip()
                if not task_id:
                    self._write_json(_format_error("missing_task_id"), status=HTTPStatus.BAD_REQUEST)
                    return
                if not new_state:
                    self._write_json(_format_error("payload_validation", details="Missing state."), status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    before_task = state.store.get_task(task_id)
                    if before_task is None:
                        self._write_json(_format_error("task_not_found", status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                        return
                    open_sessions = [
                        session
                        for session in state.store.list_sessions(include_closed=False, limit=1000)
                        if str(session.get("task_id") or "") == task_id
                    ]
                    updated = state.store.update_task_state(task_id=task_id, state=new_state, user=user)
                    closed_session_ids = [
                        str(session.get("session_id") or "")
                        for session in open_sessions
                        if str(new_state) == "complete" and str(before_task.get("state") or "") != "complete"
                    ]
                    if closed_session_ids:
                        _drop_runtime_sessions(state, closed_session_ids)
                except Exception as exc:
                    self._write_json(_format_error("task_state_update_failed", details=str(exc)), status=HTTPStatus.BAD_REQUEST)
                    return
                previous_state = str(before_task.get("state") or "")
                current_state = str(updated.get("state") or "")
                self._write_json(
                    {
                        "ok": True,
                        "task": updated,
                        "previous_task": before_task,
                        "task_state_transition": {
                            "task_id": task_id,
                            "recording_id": str(before_task.get("recording_id") or ""),
                            "previous_state": previous_state,
                            "new_state": current_state,
                            "state_changed": previous_state != current_state,
                            "completed": previous_state != "complete" and current_state == "complete",
                            "reopened": previous_state == "complete" and current_state != "complete",
                        },
                        "closed_session_count": len(closed_session_ids),
                        "closed_session_ids": closed_session_ids,
                        "session_closure_events": _closed_session_response_payload(
                            state.store,
                            closed_session_ids,
                        )["session_closure_events"],
                    }
                )
                return

            if path.startswith("/api/admin/events/") and path.endswith("/retry-promotion"):
                if not _is_admin_user(user, state.config):
                    self._write_json(_format_error("admin_required", status=HTTPStatus.FORBIDDEN), status=HTTPStatus.FORBIDDEN)
                    return
                event_id = path[len("/api/admin/events/") : -len("/retry-promotion")].strip("/")
                event = state.store.get_event(event_id)
                if event is None:
                    self._write_json(_format_error("event_not_found", status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                    return
                preflight_error = _admin_promotion_retry_preflight_error(event)
                if preflight_error is not None:
                    payload, status = preflight_error
                    self._write_json(payload, status=status)
                    return
                try:
                    claim = state.store.claim_promotion_retry(
                        failed_event_id=event_id,
                        task_id=str(event["task_id"]),
                        recording_id=str(event["recording_id"]),
                        user=user,
                    )
                    if claim.get("status") == "already_succeeded":
                        self._write_json(
                            {
                                "ok": True,
                                "promotion": {
                                    "status": "already_succeeded",
                                    "event": claim.get("event"),
                                },
                            }
                        )
                        return
                    if claim.get("status") == "in_progress":
                        self._write_json(
                            _format_error(
                                "promotion_retry_in_progress",
                                details="A retry is already running or has not recorded an outcome yet.",
                                status=HTTPStatus.CONFLICT,
                            ),
                            status=HTTPStatus.CONFLICT,
                        )
                        return
                    promotion = _retry_failed_promotion_event(store=state.store, event=event, user=user)
                except Exception as exc:
                    try:
                        state.store.record_event(
                            task_id=str(event["task_id"]),
                            recording_id=str(event["recording_id"]),
                            user=user,
                            event_type="promotion_retry_abandoned",
                            target={"retry_of_event_id": event_id},
                            after={"error": str(exc)},
                        )
                    except Exception:
                        pass
                    self._write_json(_format_error("promotion_retry_failed", details=str(exc)), status=HTTPStatus.BAD_REQUEST)
                    return
                self._write_json({"ok": True, "promotion": promotion})
                return

            if path.startswith("/api/tasks/") and path.endswith("/open"):
                task_id = path[len("/api/tasks/") : -len("/open")].strip("/")
                if not task_id:
                    self._write_json(_format_error("missing_task_id"), status=HTTPStatus.BAD_REQUEST)
                    return
                expected_user = str(body.get("expected_user") or "").strip()
                operator_validation_start_gate = _runtime_operator_validation_start_gate(
                    state.config
                )
                if expected_user and str(user) != expected_user:
                    task = state.store.get_task(task_id)
                    self._write_json(
                        _format_error(
                            "task_open_user_mismatch",
                            details=(
                                f"This task-open request is for {expected_user}, "
                                f"but the browser is authenticated as {user}. "
                                "Stop and contact the operator before labeling."
                            ),
                            status=HTTPStatus.FORBIDDEN,
                            extra=_task_open_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                task=task,
                                error="task_open_user_mismatch",
                                operator_validation_start_gate=operator_validation_start_gate,
                            ),
                        ),
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return
                open_error = _task_open_preflight_error(state.store, task_id=task_id, user=user)
                if open_error is not None:
                    error, details, status = open_error
                    task = state.store.get_task(task_id)
                    self._write_json(
                        _format_error(
                            error,
                            details=details,
                            status=status,
                            extra=_task_open_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                task=task,
                                error=error,
                                operator_validation_start_gate=operator_validation_start_gate,
                            ),
                        ),
                        status=status,
                    )
                    return
                if bool(operator_validation_start_gate.get("blocks_task_open")):
                    task = state.store.get_task(task_id)
                    self._write_json(
                        _format_error(
                            "operator_validation_start_blocked",
                            details=(
                                str(operator_validation_start_gate.get("operator_action") or "")
                                or "Required operator validation evidence is incomplete, so the server did not create a labeling session."
                            ),
                            status=HTTPStatus.CONFLICT,
                            extra=_task_open_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                task=task,
                                error="operator_validation_start_blocked",
                                operator_validation_start_gate=operator_validation_start_gate,
                            ),
                        ),
                        status=HTTPStatus.CONFLICT,
                    )
                    return
                try:
                    lease = state.store.create_session(
                        task_id=task_id,
                        user=user,
                        ttl_seconds=state.config.session_ttl_seconds,
                        client_label=str(body.get("client_label") or "")[:512] or None,
                    )
                    _drop_runtime_sessions(state, lease.superseded_session_ids)
                    closed_session_payload = _closed_session_response_payload(state.store, lease.superseded_session_ids)
                    task = state.store.get_task(task_id)
                except PermissionError as exc:
                    task = state.store.get_task(task_id)
                    if "previous-owner sessions" in str(exc):
                        self._write_json(
                            _format_error(
                                "reassignment_session_safety_failed",
                                details=_labeler_safe_error_details(exc),
                                status=HTTPStatus.CONFLICT,
                                extra=_task_open_failure_metadata(
                                    user=user,
                                    expected_user=expected_user,
                                    task=task,
                                    error="reassignment_session_safety_failed",
                                    operator_validation_start_gate=operator_validation_start_gate,
                                ),
                            ),
                            status=HTTPStatus.CONFLICT,
                        )
                    else:
                        self._write_json(
                            _format_error(
                                "not_assigned",
                                details=_labeler_safe_error_details(exc),
                                status=HTTPStatus.FORBIDDEN,
                                extra=_task_open_failure_metadata(
                                    user=user,
                                    expected_user=expected_user,
                                    task=task,
                                    error="not_assigned",
                                    operator_validation_start_gate=operator_validation_start_gate,
                                ),
                            ),
                            status=HTTPStatus.FORBIDDEN,
                        )
                    return
                except Exception as exc:
                    self._write_json(
                        _format_error(
                            "session_open_failed",
                            details=_labeler_safe_error_details(exc),
                            extra=_task_open_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                task=state.store.get_task(task_id),
                                error="session_open_failed",
                                operator_validation_start_gate=operator_validation_start_gate,
                            ),
                        ),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._write_json(
                    {
                        "ok": True,
                        "session": {
                            "session_id": lease.session_id,
                            "task_id": lease.task_id,
                            "recording_id": lease.recording_id,
                            "user": lease.user,
                            "expires_at_utc": lease.expires_at_utc,
                            "superseded_session_ids": list(lease.superseded_session_ids),
                            "url": f"/r/{lease.session_id}",
                        },
                        **_task_open_response_metadata(
                            user=user,
                            expected_user=expected_user,
                            task=task,
                            session_id=lease.session_id,
                            operator_validation_start_gate=operator_validation_start_gate,
                        ),
                        **closed_session_payload,
                    }
                )
                return

            if path.startswith("/api/tasks/") and path.endswith("/complete"):
                task_id = path[len("/api/tasks/") : -len("/complete")].strip("/")
                expected_user = str(body.get("expected_user") or "").strip()
                if expected_user and str(user) != expected_user:
                    task = state.store.get_task(task_id)
                    self._write_json(
                        _format_error(
                            "task_complete_user_mismatch",
                            details=(
                                f"This task-complete request is for {expected_user}, "
                                f"but the browser is authenticated as {user}. "
                                "Stop and contact the operator before labeling."
                            ),
                            status=HTTPStatus.FORBIDDEN,
                            extra=_task_completion_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                task=task,
                                session=None,
                                requested_task_id=task_id,
                                error="task_complete_user_mismatch",
                            ),
                        ),
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return
                task = state.store.get_task(task_id)
                if task is None:
                    self._write_json(
                        _format_error(
                            "task_not_found",
                            status=HTTPStatus.NOT_FOUND,
                            extra=_task_completion_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                task=None,
                                session=None,
                                requested_task_id=task_id,
                                error="task_not_found",
                            ),
                        ),
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                if str(task.get("assignee_user") or "") != str(user) or str(task.get("assignment_status") or "") != "active":
                    self._write_json(
                        _format_error(
                            "not_assigned",
                            status=HTTPStatus.FORBIDDEN,
                            extra=_task_completion_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                task=task,
                                session=None,
                                requested_task_id=task_id,
                                error="not_assigned",
                            ),
                        ),
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return
                session_id = str(body.get("session_id") or "").strip()
                if not session_id:
                    self._write_json(
                        _format_error(
                            "session_required",
                            details="Completing a task from the browser requires the current guarded session.",
                            status=HTTPStatus.BAD_REQUEST,
                            extra=_task_completion_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                task=task,
                                session=None,
                                requested_task_id=task_id,
                                error="session_required",
                            ),
                        ),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                session = self._session_for_user(
                    session_id,
                    user,
                    completion_error=True,
                    expected_user=expected_user,
                )
                if session is None:
                    return
                if str(session.get("task_id") or "") != task_id:
                    self._write_json(
                        _format_error(
                            "session_task_mismatch",
                            details="This browser session does not belong to the requested task.",
                            status=HTTPStatus.FORBIDDEN,
                            extra=_task_completion_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                task=task,
                                session=session,
                                requested_task_id=task_id,
                                error="session_task_mismatch",
                            ),
                        ),
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return
                unapplied_count = state.store.count_unapplied_session_checkpoints(task_id=task_id)
                if int(unapplied_count) > 0:
                    self._write_json(
                        _format_error(
                            "unapplied_session_edits",
                            details="Apply saved edits to Zarr before completing this task.",
                            status=HTTPStatus.CONFLICT,
                            extra={
                                **_task_completion_failure_metadata(
                                    user=user,
                                    expected_user=expected_user or user,
                                    task=task,
                                    session=session,
                                    requested_task_id=task_id,
                                    error="unapplied_session_edits",
                                ),
                                "unapplied_session_edit_count": int(unapplied_count),
                                "required_action": "apply_saved_edits_to_zarr",
                            },
                        ),
                        status=HTTPStatus.CONFLICT,
                    )
                    return
                if str(session.get("workflow_kind") or "") == "subject_mask_component":
                    try:
                        runtime = _get_subject_mask_runtime(state, session)
                        review_completion_guard = _subject_mask_component_completion_guard(runtime)
                    except Exception as exc:
                        self._write_json(
                            _format_error(
                                "subject_mask_review_status_check_failed",
                                details=_labeler_safe_error_details(exc),
                                status=HTTPStatus.BAD_REQUEST,
                                extra=_task_completion_failure_metadata(
                                    user=user,
                                    expected_user=expected_user or user,
                                    task=task,
                                    session=session,
                                    requested_task_id=task_id,
                                    error="subject_mask_review_status_check_failed",
                                ),
                            ),
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return
                    if not bool(review_completion_guard.get("ready")):
                        self._write_json(
                            _format_error(
                                "component_review_pending",
                                details="Set component review status before completing this subject-mask task.",
                                status=HTTPStatus.CONFLICT,
                                extra={
                                    **_task_completion_failure_metadata(
                                        user=user,
                                        expected_user=expected_user or user,
                                        task=task,
                                        session=session,
                                        requested_task_id=task_id,
                                        error="component_review_pending",
                                    ),
                                    "component_review_completion_guard": review_completion_guard,
                                },
                            ),
                            status=HTTPStatus.CONFLICT,
                        )
                        return
                closed_session_ids = _open_session_ids_for_task(state.store, task_id)
                updated = state.store.update_task_state(task_id=task_id, state="complete", user=user)
                if closed_session_ids:
                    _drop_runtime_sessions(state, closed_session_ids)
                self._write_json(
                    {
                        "ok": True,
                        "task": updated,
                        "task_completion_authorization_contract": (
                            _task_completion_authorization_contract(
                                user=user,
                                expected_user=expected_user or user,
                                task=updated,
                                session=session,
                                requested_task_id=task_id,
                                ready=True,
                                server_authorizes_completion=True,
                            )
                        ),
                        **_closed_session_response_payload(state.store, closed_session_ids),
                        **_post_completion_queue_metadata(
                            state.store,
                            user=user,
                            expected_user=expected_user or user,
                        ),
                    }
                )
                return

            if path.startswith("/api/events/") and path.endswith("/retry-promotion"):
                event_id = path[len("/api/events/") : -len("/retry-promotion")].strip("/")
                expected_user = str(body.get("expected_user") or "").strip()
                if expected_user and str(user) != expected_user:
                    self._write_json(
                        _format_error(
                            "promotion_retry_user_mismatch",
                            details=(
                                f"This promotion-retry request is for {expected_user}, "
                                f"but the browser is authenticated as {user}. "
                                "Stop and contact the operator before labeling."
                            ),
                            status=HTTPStatus.FORBIDDEN,
                            extra=_labeler_promotion_retry_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                event_id=event_id,
                            ),
                        ),
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return
                event = state.store.get_event(event_id)
                if event is None:
                    self._write_json(
                        _format_error(
                            "event_not_found",
                            status=HTTPStatus.NOT_FOUND,
                            extra=_labeler_promotion_retry_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                event_id=event_id,
                            ),
                        ),
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                if str(event.get("assignee_user") or "") != str(user) or str(event.get("assignment_status") or "") != "active":
                    self._write_json(
                        _format_error(
                            "not_assigned",
                            status=HTTPStatus.FORBIDDEN,
                            extra=_labeler_promotion_retry_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                event=event,
                            ),
                        ),
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return
                event_task = state.store.get_task(str(event.get("task_id") or ""))
                if event_task is None:
                    self._write_json(
                        _format_error(
                            "task_not_found",
                            status=HTTPStatus.NOT_FOUND,
                            extra=_labeler_promotion_retry_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                event=event,
                            ),
                        ),
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                already_succeeded_event = _promotion_success_event_for_retry(
                    state.store,
                    event_id,
                    user=user,
                )
                if already_succeeded_event is not None:
                    redacted_event = _redact_labeler_runtime_payload(
                        already_succeeded_event
                    )
                    self._write_json(
                        {
                            "ok": True,
                            "promotion": {
                                "status": "already_succeeded",
                                "event": (
                                    dict(redacted_event)
                                    if isinstance(redacted_event, Mapping)
                                    else {}
                                ),
                            },
                        }
                    )
                    return
                if str(event_task.get("state") or "") == "complete":
                    self._write_json(
                        _format_error(
                            "task_complete",
                            details="This task is complete; ask the operator to reopen it before retrying promotion.",
                            status=HTTPStatus.CONFLICT,
                            extra=_labeler_promotion_retry_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                event=event,
                                event_task=event_task,
                            ),
                        ),
                        status=HTTPStatus.CONFLICT,
                    )
                    return
                session_id = str(body.get("session_id") or "").strip()
                if not session_id:
                    self._write_json(
                        _format_error(
                            "session_required",
                            details="Retrying promotion from the browser requires the current guarded session.",
                            status=HTTPStatus.BAD_REQUEST,
                            extra=_labeler_promotion_retry_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                event=event,
                                event_task=event_task,
                            ),
                        ),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                session = self._session_for_user(session_id, user)
                if session is None:
                    return
                if str(session.get("task_id") or "") != str(event.get("task_id") or ""):
                    self._write_json(
                        _format_error(
                            "session_task_mismatch",
                            details="This browser session does not belong to the promotion event task.",
                            status=HTTPStatus.FORBIDDEN,
                            extra=_labeler_promotion_retry_failure_metadata(
                                user=user,
                                expected_user=expected_user,
                                event=event,
                                event_task=event_task,
                                session=session,
                            ),
                        ),
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return
                self._write_json(
                    _labeler_promotion_retry_operator_support_payload(
                        user=user,
                        expected_user=expected_user,
                        event=event,
                        event_task=event_task,
                        session=session,
                    ),
                    status=HTTPStatus.FORBIDDEN,
                )
                return

            if path.startswith("/api/sessions/") and path.endswith("/complete"):
                session_id = path[len("/api/sessions/") : -len("/complete")].strip("/")
                session = self._session_for_user(
                    session_id,
                    user,
                    completion_error=True,
                    expected_user=user,
                )
                if session is None:
                    return
                task = state.store.get_task(str(session.get("task_id") or ""))
                if task is None:
                    self._write_json(
                        _format_error(
                            "task_not_found",
                            status=HTTPStatus.NOT_FOUND,
                            extra=_task_completion_failure_metadata(
                                user=user,
                                expected_user=user,
                                task=None,
                                session=session,
                                requested_task_id=str(session.get("task_id") or ""),
                                error="task_not_found",
                            ),
                        ),
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                if str(task.get("assignee_user") or "") != str(user) or str(task.get("assignment_status") or "") != "active":
                    self._write_json(
                        _format_error(
                            "not_assigned",
                            status=HTTPStatus.FORBIDDEN,
                            extra=_task_completion_failure_metadata(
                                user=user,
                                expected_user=user,
                                task=task,
                                session=session,
                                requested_task_id=str(task.get("task_id") or ""),
                                error="not_assigned",
                            ),
                        ),
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return
                unapplied_count = state.store.count_unapplied_session_checkpoints(task_id=str(task["task_id"]))
                if int(unapplied_count) > 0:
                    self._write_json(
                        _format_error(
                            "unapplied_session_edits",
                            details="Apply saved edits to Zarr before completing this task.",
                            status=HTTPStatus.CONFLICT,
                            extra={
                                **_task_completion_failure_metadata(
                                    user=user,
                                    expected_user=user,
                                    task=task,
                                    session=session,
                                    requested_task_id=str(task["task_id"]),
                                    error="unapplied_session_edits",
                                ),
                                "unapplied_session_edit_count": int(unapplied_count),
                                "required_action": "apply_saved_edits_to_zarr",
                            },
                        ),
                        status=HTTPStatus.CONFLICT,
                    )
                    return
                if str(session.get("workflow_kind") or "") == "subject_mask_component":
                    try:
                        runtime = _get_subject_mask_runtime(state, session)
                        review_completion_guard = _subject_mask_component_completion_guard(runtime)
                    except Exception as exc:
                        self._write_json(
                            _format_error(
                                "subject_mask_review_status_check_failed",
                                details=_labeler_safe_error_details(exc),
                                status=HTTPStatus.BAD_REQUEST,
                                extra=_task_completion_failure_metadata(
                                    user=user,
                                    expected_user=user,
                                    task=task,
                                    session=session,
                                    requested_task_id=str(task["task_id"]),
                                    error="subject_mask_review_status_check_failed",
                                ),
                            ),
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return
                    if not bool(review_completion_guard.get("ready")):
                        self._write_json(
                            _format_error(
                                "component_review_pending",
                                details="Set component review status before completing this subject-mask task.",
                                status=HTTPStatus.CONFLICT,
                                extra={
                                    **_task_completion_failure_metadata(
                                        user=user,
                                        expected_user=user,
                                        task=task,
                                        session=session,
                                        requested_task_id=str(task["task_id"]),
                                        error="component_review_pending",
                                    ),
                                    "component_review_completion_guard": review_completion_guard,
                                },
                            ),
                            status=HTTPStatus.CONFLICT,
                        )
                        return
                closed_session_ids = _open_session_ids_for_task(state.store, str(task["task_id"]))
                updated = state.store.update_task_state(task_id=str(task["task_id"]), state="complete", user=user)
                if closed_session_ids:
                    _drop_runtime_sessions(state, closed_session_ids)
                self._write_json(
                    {
                        "ok": True,
                        "task": updated,
                        "task_completion_authorization_contract": (
                            _task_completion_authorization_contract(
                                user=user,
                                expected_user=user,
                                task=updated,
                                session=session,
                                requested_task_id=str(task["task_id"]),
                                ready=True,
                                server_authorizes_completion=True,
                            )
                        ),
                        **_closed_session_response_payload(state.store, closed_session_ids),
                        **_post_completion_queue_metadata(
                            state.store,
                            user=user,
                            expected_user=user,
                        ),
                    }
                )
                return

            session_api = self._parse_session_api_path(path)
            if session_api is not None:
                session_id, suffix = session_api
                mutation_error = suffix in {
                    "/keypoints/save",
                    "/keypoints/action",
                    "/keypoints/review-status",
                    "/detect/save",
                    "/detect-analysis/save",
                    "/subject-mask/save",
                    "/subject-mask/apply",
                    "/subject-mask/review-status",
                }
                session = self._session_for_user(
                    session_id,
                    user,
                    mutation_error=mutation_error,
                )
                if session is None:
                    return
                if self._handle_keypoint_post(session, suffix, body, user):
                    return
                if self._handle_detect_post(session, suffix, body, user):
                    return
                if self._handle_video_detect_post(session, suffix, body, user):
                    return
                if self._handle_subject_mask_post(session, suffix, body, user):
                    return
                self._write_json(_format_error("not_found", status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                return

            self._write_json(_format_error("not_found", status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)

        def log_message(self, fmt: str, *args: object) -> None:
            if not state.config.access_log:
                return
            user, source = self._current_user()
            remote = self.client_address[0] if self.client_address else "-"
            message = fmt % args if args else fmt
            print(
                json.dumps(
                    {
                        "event": "palette_labeling_access",
                        "remote": str(remote),
                        "user": str(user or ""),
                        "auth_source": str(source),
                        "request": str(getattr(self, "requestline", "")),
                        "message": str(message),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )

    return LabelingWorkHandler


def serve(config: ServerConfig) -> int:
    config_errors = _server_config_errors(config)
    if config_errors:
        for error in config_errors:
            print(f"configuration_error={error}", file=sys.stderr)
        return 2
    store = LabelingStore(config.store_path)
    store.initialize()
    state = ServerState(store=store, config=config)
    server = ThreadingHTTPServer((config.host, int(config.port)), _make_handler(state))
    url_host = "localhost" if config.host in {"0.0.0.0", "::"} else config.host
    print(f"Palette labeling work UI: http://{url_host}:{config.port}")
    print(f"store={config.store_path}")
    if config.fixed_user:
        user_mode = config.fixed_user
    elif config.trust_auth_header:
        user_mode = "trusted_header:" + config.auth_header
    else:
        user_mode = "disabled"
    print(f"user={user_mode}")
    print(f"admin_users={','.join(config.admin_users) if config.admin_users else '(none)'}")
    print(f"signed_links={'enabled' if config.link_secret else 'disabled'}")
    print(f"link_not_before_utc={config.link_not_before_utc or '(none)'}")
    print(f"csrf_same_origin={'enabled' if config.csrf_same_origin else 'disabled'}")
    print(f"access_log={'enabled' if config.access_log else 'disabled'}")
    print(f"allow_non_loopback={'enabled' if config.allow_non_loopback else 'disabled'}")
    print(f"production={'enabled' if config.production else 'disabled'}")
    safety = _server_safety_payload(config, include_admin_details=False)
    if safety["warnings"]:
        print(f"preflight_warnings={','.join(str(item) for item in safety['warnings'])}")
    else:
        print("preflight_warnings=(none)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()
    return 0


def _write_csv_manifest_template(path: Path, *, fieldnames: list[str], sample: dict[str, object], overwrite: bool) -> None:
    import csv

    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing CSV manifest template: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: _csv_export_value(sample.get(key)) for key in fieldnames})


def _write_manifest_templates_readme(
    path: Path,
    *,
    assignments_path: Path,
    tasks_path: Path,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest template README: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Palette labeling manifest templates",
        "",
        "Files:",
        f"- {assignments_path.name}: recording ownership plan",
        f"- {tasks_path.name}: task definition plan",
        "",
        "Recommended flow:",
        f"1. Edit {assignments_path.name} and {tasks_path.name} in a spreadsheet.",
        "2. Keep recording_id and task_id stable between dry-run and apply.",
        "3. Dry-run assignments:",
        f"   scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite import-assignments --input {assignments_path.name} --assigned-by OPERATOR",
        "4. Apply assignments after review:",
        f"   scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite import-assignments --input {assignments_path.name} --assigned-by OPERATOR --apply",
        "5. Dry-run tasks:",
        f"   scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite import-tasks --input {tasks_path.name} --actor OPERATOR",
        "6. Apply tasks after review:",
        f"   scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite import-tasks --input {tasks_path.name} --actor OPERATOR --apply",
        "",
        "CSV notes:",
        "- Fully blank trailing rows are ignored.",
        "- Partially filled rows fail validation.",
        "- Task scope_json cells must contain valid JSON, for example {\"frames\":[1,2,3]}.",
        "- CSV dry-run and apply output include source_line so spreadsheet rows can be traced.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")




def _parse_scope(value: str | None) -> object:
    if value is None or not str(value).strip():
        return {}
    raw = str(value).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        path = Path(raw).expanduser()
        return json.loads(path.read_text(encoding="utf-8"))


def _parse_csv_dict_rows(
    text: str,
    *,
    required_headers: Sequence[str] = (),
    any_header_groups: Sequence[Sequence[str]] = (),
    source: str = "CSV manifest",
) -> list[dict[str, object]]:
    import csv

    reader = csv.DictReader(text.splitlines())
    fieldnames = [str(field or "").strip() for field in (reader.fieldnames or [])]
    fieldname_set = set(fieldnames)
    missing_headers = [header for header in required_headers if header not in fieldname_set]
    missing_groups = [
        " or ".join(group)
        for group in any_header_groups
        if not any(header in fieldname_set for header in group)
    ]
    if missing_headers or missing_groups:
        missing = missing_headers + missing_groups
        raise ValueError(f"{source} is missing required CSV column(s): {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    for line_number, row in enumerate(reader, start=2):
        if not any(str(value or "").strip() for value in row.values()):
            continue
        parsed = dict(row)
        parsed["_source_line"] = line_number
        rows.append(parsed)
    return rows


def _parse_assignment_manifest(value: str) -> list[dict[str, object]]:
    path = Path(value).expanduser()
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.lower() == ".csv":
        payload = _parse_csv_dict_rows(
            text,
            required_headers=("recording_id",),
            any_header_groups=(("assignee_user", "user"),),
            source="Assignment CSV manifest",
        )
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, Mapping):
        rows = payload.get("assignments", [])
    else:
        rows = payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Assignment manifest must be a CSV file, JSON list, JSONL file, or object with an assignments list.")
    parsed: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Assignment row {idx} must be an object.")
        recording_id = str(row.get("recording_id") or "").strip()
        assignee_user = str(row.get("assignee_user") or row.get("user") or "").strip()
        if not recording_id:
            raise ValueError(f"Assignment row {idx} is missing recording_id.")
        if not assignee_user:
            raise ValueError(f"Assignment row {idx} is missing assignee_user/user.")
        parsed.append(
            {
                "recording_id": recording_id,
                "assignee_user": assignee_user,
                "assigned_by": row.get("assigned_by"),
                "status": str(row.get("status") or "active"),
                "notes": row.get("notes"),
                **({"_source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
            }
        )
    return parsed


def _assignment_rows_for_apply(rows: Sequence[Mapping[str, object]], *, apply: bool) -> list[Mapping[str, object]]:
    if not apply:
        return list(rows)
    latest_by_recording: dict[str, Mapping[str, object]] = {}
    order: list[str] = []
    for row in rows:
        recording_id = str(row.get("recording_id") or "")
        if not recording_id:
            continue
        if recording_id not in latest_by_recording:
            order.append(recording_id)
        latest_by_recording[recording_id] = row
    return [latest_by_recording[recording_id] for recording_id in order]


def _parse_task_manifest(value: str) -> list[dict[str, object]]:
    path = Path(value).expanduser()
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.lower() == ".csv":
        payload = _parse_csv_dict_rows(
            text,
            required_headers=("task_id", "recording_id", "workflow_kind"),
            source="Task CSV manifest",
        )
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, Mapping):
        rows = payload.get("tasks", [])
    else:
        rows = payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Task manifest must be a CSV file, JSON list, JSONL file, or object with a tasks list.")
    parsed: list[dict[str, object]] = []
    seen_task_ids: set[str] = set()
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Task row {idx} must be an object.")
        task_id = str(row.get("task_id") or "").strip()
        recording_id = str(row.get("recording_id") or "").strip()
        workflow_kind = str(row.get("workflow_kind") or "").strip()
        if not task_id:
            raise ValueError(f"Task row {idx} is missing task_id.")
        if not recording_id:
            raise ValueError(f"Task row {idx} is missing recording_id.")
        if not workflow_kind:
            raise ValueError(f"Task row {idx} is missing workflow_kind.")
        if task_id in seen_task_ids:
            raise ValueError(f"Duplicate task_id in task manifest: {task_id}")
        seen_task_ids.add(task_id)
        scope = row.get("scope", row.get("scope_json", {}))
        if isinstance(scope, str):
            scope = json.loads(scope) if scope.strip() else {}
        parsed.append(
            {
                "task_id": task_id,
                "recording_id": recording_id,
                "workflow_kind": workflow_kind,
                "dataset_id": row.get("dataset_id"),
                "zarr_use": row.get("zarr_use"),
                "stage_group": row.get("stage_group"),
                "run_name": row.get("run_name"),
                "component_name": row.get("component_name"),
                "title": row.get("title"),
                "scope": scope if scope is not None else {},
                "state": str(row.get("state") or "pending"),
                "priority": int(row.get("priority") or 0),
                "notes": row.get("notes"),
                **({"_source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
            }
        )
    return parsed


def _link_secret_from_arg(value: str | None) -> str:
    secret = str(value or os.environ.get(LINK_SECRET_ENV_VAR) or "").strip()
    if not secret:
        raise ValueError(f"Signed links require --link-secret or {LINK_SECRET_ENV_VAR}.")
    return secret






ZARR_BACKUP_PATH_KEYS = _web_zarr_backup.ZARR_BACKUP_PATH_KEYS


def _configure_zarr_backup_plan_helpers() -> None:
    _web_zarr_backup.configure_zarr_backup_plan_dependencies(
        {
            "_zarr_backup_policy": _zarr_backup_policy,
        }
    )


def _zarr_backup_contract_policy(
    policy: Mapping[str, object],
    counts: Mapping[str, object],
    files: Mapping[str, object],
) -> dict[str, object]:
    return _web_zarr_backup._zarr_backup_contract_policy(policy, counts, files)


def _iter_zarr_path_values(value: object) -> Iterable[tuple[str, str]]:
    return _web_zarr_backup._iter_zarr_path_values(value)


def _zarr_backup_role_for_key(key: str, task: Mapping[str, object]) -> str:
    return _web_zarr_backup._zarr_backup_role_for_key(key, task)


def _zarr_backup_plan(
    *,
    store: LabelingStore,
    store_path: Path,
    recording_id: str | None = None,
    user: str | None = None,
    include_completed: bool = True,
    include_inactive: bool = False,
) -> dict[str, object]:
    _configure_zarr_backup_plan_helpers()
    return _web_zarr_backup._zarr_backup_plan_impl(
        store=store,
        store_path=store_path,
        recording_id=recording_id,
        user=user,
        include_completed=include_completed,
        include_inactive=include_inactive,
    )







def _safe_backup_slug(value: object, *, fallback: str) -> str:
    return _web_zarr_backup._safe_backup_slug_impl(value, fallback=fallback)


def _backup_target_destination(backup_dir: Path, target: Mapping[str, object], index: int) -> Path:
    return _web_zarr_backup._backup_target_destination_impl(backup_dir, target, index)


def _copy_backup_source(source: Path, destination: Path, *, overwrite: bool) -> None:
    _web_zarr_backup._copy_backup_source_impl(source, destination, overwrite=overwrite)


def _execute_zarr_backup_plan(
    *,
    plan_path: Path,
    backup_dir: Path,
    operator: str,
    output: Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_missing: bool = False,
) -> dict[str, object]:
    _configure_zarr_backup_plan_helpers()
    return _web_zarr_backup._execute_zarr_backup_plan_impl(
        plan_path=plan_path,
        backup_dir=backup_dir,
        operator=operator,
        output=output,
        overwrite=overwrite,
        dry_run=dry_run,
        allow_missing=allow_missing,
    )


def _restore_assignment_conflicts(
    store: LabelingStore,
    recording_ids: Sequence[object],
) -> list[dict[str, object]]:
    return _web_zarr_backup._restore_assignment_conflicts_impl(store, recording_ids)


def _restore_backup_target(
    *,
    source_backup: Path,
    restore_path: Path,
    replace_current: bool,
    generated_at_utc: str,
) -> dict[str, object]:
    return _web_zarr_backup._restore_backup_target_impl(
        source_backup=source_backup,
        restore_path=restore_path,
        replace_current=replace_current,
        generated_at_utc=generated_at_utc,
    )


def _restore_zarr_backup_manifest(
    *,
    store: LabelingStore,
    manifest_path: Path,
    operator: str,
    target_indexes: Sequence[int],
    restore_all: bool = False,
    replace_current: bool = False,
    allow_active_assignment: bool = False,
) -> dict[str, object]:
    return _web_zarr_backup._restore_zarr_backup_manifest_impl(
        store=store,
        manifest_path=manifest_path,
        operator=operator,
        target_indexes=target_indexes,
        restore_all=restore_all,
        replace_current=replace_current,
        allow_active_assignment=allow_active_assignment,
    )


def _record_zarr_backup_evidence(
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
    return _web_zarr_backup._record_zarr_backup_evidence_impl(
        evidence_path=evidence_path,
        execution_manifest_path=execution_manifest_path,
        operator=operator,
        restore_test_result=restore_test_result,
        target_indexes=target_indexes,
        record_all=record_all,
        output=output,
        overwrite=overwrite,
        notes=notes,
    )

























def _assignment_control_plane_report_fields(store: LabelingStore) -> dict[str, object]:
    policy = _assignment_ownership_policy()
    integrity = _assignment_ownership_integrity(
        store.list_assignments(status=None),
        schema_integrity=store.assignment_schema_integrity(),
    )
    single_owner_assignment_contract = store.single_owner_assignment_contract()
    mutation_write_policy = _browser_mutation_write_policy()
    return {
        "single_owner_policy": policy,
        "single_owner_assignment_contract": single_owner_assignment_contract,
        "assignment_ownership_integrity": integrity,
        "assignment_ownership_contract": _assignment_ownership_contract_policy(
            policy,
            integrity,
            store_single_owner_contract=single_owner_assignment_contract,
        ),
        "browser_mutation_write_policy": mutation_write_policy,
        "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(mutation_write_policy),
        "assignment_manifest_artifact_role": "metadata_only_control_plane",
        "assignment_manifest_artifacts_are_label_write_targets": False,
        "assignment_manifest_browser_writes_label_data": False,
        "assignment_manifest_applies_recording_ownership_only": True,
    }




def _labeler_route_authorization_contract_policy(policy: Mapping[str, object]) -> dict[str, object]:
    resolved_browser_user_required = bool(policy.get("resolved_browser_user_required"))
    known_assignment_store_user_required = bool(policy.get("known_assignment_store_user_required"))
    expected_user_must_match_resolved_user = bool(policy.get("expected_user_must_match_resolved_user"))
    personal_work_page_expected_user_guarded = bool(
        policy.get("personal_work_page_expected_user_guarded")
    )
    personal_dataset_queue_page_expected_user_guarded = bool(
        policy.get("personal_dataset_queue_page_expected_user_guarded")
    )
    personal_aliases_route_to_canonical_browser_surfaces = bool(
        policy.get("personal_aliases_route_to_canonical_browser_surfaces")
    )
    task_open_requires_active_assignment = bool(policy.get("task_open_requires_active_assignment"))
    task_open_requires_startable_task_state = bool(policy.get("task_open_requires_startable_task_state"))
    startable_task_states = [str(state) for state in policy.get("startable_task_states") or []]
    signed_links_are_entry_hints_not_authorization = bool(policy.get("signed_links_are_entry_hints_not_authorization"))
    forwarded_signed_links_recheck_runtime_operator_validation_start_gate = bool(
        policy.get("forwarded_signed_links_recheck_runtime_operator_validation_start_gate")
    )
    single_owner_store_proof_required_for_browser_work = bool(
        policy.get("single_owner_store_proof_required_for_browser_work")
    )
    single_owner_store_proof_requires_integrity_ok = bool(
        policy.get("single_owner_store_proof_requires_integrity_ok")
    )
    single_owner_store_proof_requires_zero_duplicate_active_owners = bool(
        policy.get("single_owner_store_proof_requires_zero_duplicate_active_owners")
    )
    single_owner_store_proof_requires_training_zarr_target = bool(
        policy.get("single_owner_store_proof_requires_training_zarr_target")
    )
    single_owner_store_proof_rejects_intermediate_csv_mutation = bool(
        policy.get("single_owner_store_proof_rejects_intermediate_csv_mutation")
    )
    ready = (
        resolved_browser_user_required
        and known_assignment_store_user_required
        and expected_user_must_match_resolved_user
        and personal_work_page_expected_user_guarded
        and personal_dataset_queue_page_expected_user_guarded
        and personal_aliases_route_to_canonical_browser_surfaces
        and task_open_requires_active_assignment
        and task_open_requires_startable_task_state
        and startable_task_states == list(LABELER_START_TASK_STATES)
        and signed_links_are_entry_hints_not_authorization
        and forwarded_signed_links_recheck_runtime_operator_validation_start_gate
        and single_owner_store_proof_required_for_browser_work
        and single_owner_store_proof_requires_integrity_ok
        and single_owner_store_proof_requires_zero_duplicate_active_owners
        and single_owner_store_proof_requires_training_zarr_target
        and single_owner_store_proof_rejects_intermediate_csv_mutation
    )
    return {
        "schema": "palette.web_labeling_labeler_route_authorization_contract.v1",
        "ready": ready,
        "resolved_browser_user_required": resolved_browser_user_required,
        "known_assignment_store_user_required": known_assignment_store_user_required,
        "expected_user_must_match_resolved_user": expected_user_must_match_resolved_user,
        "personal_work_page_expected_user_guarded": personal_work_page_expected_user_guarded,
        "personal_dataset_queue_page_expected_user_guarded": (
            personal_dataset_queue_page_expected_user_guarded
        ),
        "personal_aliases_route_to_canonical_browser_surfaces": (
            personal_aliases_route_to_canonical_browser_surfaces
        ),
        "task_open_requires_active_assignment": task_open_requires_active_assignment,
        "task_open_requires_startable_task_state": task_open_requires_startable_task_state,
        "startable_task_states": startable_task_states,
        "signed_links_are_entry_hints_not_authorization": signed_links_are_entry_hints_not_authorization,
        "forwarded_signed_links_recheck_runtime_operator_validation_start_gate": (
            forwarded_signed_links_recheck_runtime_operator_validation_start_gate
        ),
        "single_owner_store_proof_required_for_browser_work": (
            single_owner_store_proof_required_for_browser_work
        ),
        "single_owner_store_proof_requires_integrity_ok": (
            single_owner_store_proof_requires_integrity_ok
        ),
        "single_owner_store_proof_requires_zero_duplicate_active_owners": (
            single_owner_store_proof_requires_zero_duplicate_active_owners
        ),
        "single_owner_store_proof_requires_training_zarr_target": (
            single_owner_store_proof_requires_training_zarr_target
        ),
        "single_owner_store_proof_rejects_intermediate_csv_mutation": (
            single_owner_store_proof_rejects_intermediate_csv_mutation
        ),
        "copied_links_rechecked_server_side": ready,
        "forwarded_links_are_not_authorization_grants": signed_links_are_entry_hints_not_authorization,
        "personalized_reads_require_resolved_user": bool(policy.get("personal_work_reads_filtered_by_resolved_user")),
        "mutation_requires_assignment_user_match": bool(policy.get("assignment_user_match_required_for_mutation")),
        "authorization_grants": [],
    }


def _session_guard_contract_policy(policy: Mapping[str, object]) -> dict[str, object]:
    requires_current_session = bool(policy.get("requires_current_session"))
    requires_unexpired_session = bool(policy.get("requires_unexpired_session"))
    stale_tab_save_rejected = bool(policy.get("stale_tab_save_rejected"))
    superseded_sessions_rejected = bool(policy.get("superseded_sessions_rejected"))
    non_startable_task_sessions_rejected = bool(policy.get("non_startable_task_sessions_rejected"))
    target_token_required_for_mutation = bool(policy.get("target_token_required_for_mutation"))
    labeler_promotion_retry_requires_current_session = bool(
        policy.get("labeler_promotion_retry_requires_current_session")
    )
    session_closure_event_support = bool(policy.get("session_closure_event_support"))
    ready = (
        requires_current_session
        and requires_unexpired_session
        and stale_tab_save_rejected
        and superseded_sessions_rejected
        and non_startable_task_sessions_rejected
        and target_token_required_for_mutation
        and labeler_promotion_retry_requires_current_session
        and session_closure_event_support
    )
    return {
        "schema": "palette.web_labeling_session_guard_contract.v1",
        "ready": ready,
        "requires_current_session": requires_current_session,
        "requires_unexpired_session": requires_unexpired_session,
        "stale_tab_save_rejected": stale_tab_save_rejected,
        "superseded_sessions_rejected": superseded_sessions_rejected,
        "non_startable_task_sessions_rejected": non_startable_task_sessions_rejected,
        "target_token_required_for_mutation": target_token_required_for_mutation,
        "labeler_promotion_retry_requires_current_session": labeler_promotion_retry_requires_current_session,
        "session_closure_event_support": session_closure_event_support,
        "rejects_after_reassignment": superseded_sessions_rejected and session_closure_event_support,
        "rejects_after_completion_or_reopen": superseded_sessions_rejected,
        "rejects_after_expiration": requires_unexpired_session,
        "rejects_after_target_navigation": target_token_required_for_mutation,
    }


def _browser_payload_redaction_contract_policy(policy: Mapping[str, object]) -> dict[str, object]:
    redaction = (
        policy.get("labeler_api_redaction")
        if isinstance(policy.get("labeler_api_redaction"), Mapping)
        else {}
    )
    browser_receives_task_scope = bool(policy.get("browser_receives_task_scope"))
    browser_receives_raw_zarr_paths = bool(policy.get("browser_receives_raw_zarr_paths"))
    redacts_runtime_state_paths = bool(redaction.get("redacts_runtime_state_paths"))
    redacts_mutation_response_paths = bool(redaction.get("redacts_mutation_response_paths"))
    redacts_error_detail_paths = bool(redaction.get("redacts_error_detail_paths"))
    redacts_path_like_string_values = bool(redaction.get("redacts_path_like_string_values"))
    redacts_user_summary_path_like_string_values = bool(
        redaction.get("redacts_user_summary_path_like_string_values")
    )
    redacts_direct_storage_paths = bool(redaction.get("redacts_direct_storage_paths"))
    admin_diagnostics_unredacted = bool(redaction.get("admin_diagnostics_unredacted"))
    ready = (
        not browser_receives_task_scope
        and not browser_receives_raw_zarr_paths
        and redacts_runtime_state_paths
        and redacts_mutation_response_paths
        and redacts_error_detail_paths
        and redacts_path_like_string_values
        and redacts_user_summary_path_like_string_values
        and redacts_direct_storage_paths
        and admin_diagnostics_unredacted
    )
    return {
        "schema": "palette.web_labeling_browser_payload_redaction_contract.v1",
        "ready": ready,
        "browser_receives_task_scope": browser_receives_task_scope,
        "browser_receives_raw_zarr_paths": browser_receives_raw_zarr_paths,
        "browser_receives_storage_credentials": False,
        "browser_receives_filesystem_write_authority": False,
        "redacts_runtime_state_paths": redacts_runtime_state_paths,
        "redacts_mutation_response_paths": redacts_mutation_response_paths,
        "redacts_error_detail_paths": redacts_error_detail_paths,
        "redacts_path_like_string_values": redacts_path_like_string_values,
        "redacts_user_summary_path_like_string_values": redacts_user_summary_path_like_string_values,
        "redacts_direct_storage_paths": redacts_direct_storage_paths,
        "admin_diagnostics_unredacted": admin_diagnostics_unredacted,
        "labeler_support_text_redacted": redacts_error_detail_paths and redacts_path_like_string_values,
    }




def _identity_probe_link_contract_policy(
    *,
    labeler_safety: Mapping[str, object],
    expected_user_identity_probe_url: str,
    files: Mapping[str, object],
) -> dict[str, object]:
    identity_check_required = bool(labeler_safety.get("dashboard_identity_check_required"))
    expected_user_identity_probe_url_present = bool(str(expected_user_identity_probe_url or "").strip())
    roster_path = str(files.get("handoffs_roster") or files.get("labeler_roster") or "").strip()
    handoffs_index_path = str(files.get("handoffs_index") or files.get("index") or "").strip()
    batch_identity_probe_evidence_present = bool(roster_path or handoffs_index_path)
    ready = identity_check_required and (
        expected_user_identity_probe_url_present or batch_identity_probe_evidence_present
    )
    return {
        "schema": "palette.web_labeling_identity_probe_link_contract.v1",
        "ready": ready,
        "identity_check_required": identity_check_required,
        "expected_user_identity_probe_url": str(expected_user_identity_probe_url or ""),
        "expected_user_identity_probe_url_present": expected_user_identity_probe_url_present,
        "handoffs_roster": roster_path,
        "handoffs_index": handoffs_index_path,
        "batch_identity_probe_evidence_present": batch_identity_probe_evidence_present,
        "operator_verification_still_required": True,
    }


def _configure_operator_evidence_template_helpers() -> None:
    _web_operator_evidence_templates.configure_operator_evidence_template_dependencies(
        {
            "BROWSER_SMOKE_REQUIRED_FIELDS": BROWSER_SMOKE_REQUIRED_FIELDS,
            "DASHBOARD_PATH": DASHBOARD_PATH,
            "DATASET_QUEUE_PATH": DATASET_QUEUE_PATH,
            "DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS": DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS,
            "IDENTITY_PROBE_PATH": IDENTITY_PROBE_PATH,
            "LABELING_HOME_PATH": LABELING_HOME_PATH,
            "PERSONAL_DATASET_QUEUE_PATH": PERSONAL_DATASET_QUEUE_PATH,
            "PERSONAL_WORK_PATH": PERSONAL_WORK_PATH,
            "_browser_response_security_contract_policy": _browser_response_security_contract_policy,
            "_browser_response_security_policy": _browser_response_security_policy,
            "_browser_smoke_personalized_route_contract": _browser_smoke_personalized_route_contract,
            "_browser_workflow_capabilities": _browser_workflow_capabilities,
            "_dashboard_url_for_expected_user": _dashboard_url_for_expected_user,
            "_zarr_backup_policy": _zarr_backup_policy,
        }
    )

def _identity_source_evidence_template(*args: object, **kwargs: object) -> dict[str, object]:
    _configure_operator_evidence_template_helpers()
    return _web_operator_evidence_templates._identity_source_evidence_template_impl(*args, **kwargs)


def _browser_smoke_evidence_template(*args: object, **kwargs: object) -> dict[str, object]:
    _configure_operator_evidence_template_helpers()
    return _web_operator_evidence_templates._browser_smoke_evidence_template_impl(*args, **kwargs)


def _disposable_zarr_mutation_smoke_evidence_template(*args: object, **kwargs: object) -> dict[str, object]:
    _configure_operator_evidence_template_helpers()
    return _web_operator_evidence_templates._disposable_zarr_mutation_smoke_evidence_template_impl(*args, **kwargs)


def _zarr_backup_evidence_template(*args: object, **kwargs: object) -> dict[str, object]:
    _configure_operator_evidence_template_helpers()
    return _web_operator_evidence_templates._zarr_backup_evidence_template_impl(*args, **kwargs)


def _browser_response_security_evidence_template(*args: object, **kwargs: object) -> dict[str, object]:
    _configure_operator_evidence_template_helpers()
    return _web_operator_evidence_templates._browser_response_security_evidence_template_impl(*args, **kwargs)























LABELING_HOME_PATH = "/labeling"
PERSONAL_WORK_PATH = "/my-work"
IDENTITY_PROBE_PATH = "/identity"

BROWSER_TASK_STATE_POLICY: dict[str, object] = {
    "startable_task_states": list(LABELER_START_TASK_STATES),
    "completed_tasks_read_only": True,
    "completed_tasks_open_new_sessions": False,
    "completed_task_open_requests": "reject_task_complete",
    "completed_task_save_requests": "reject_task_complete",
    "non_startable_task_open_requests": "reject_task_not_startable",
    "non_startable_task_save_requests": "reject_task_not_startable",
    "absolute_navigation_out_of_scope": "reject_nav_error",
    "browser_mutation_target_selectors": "server_owned_reject_client_fields",
    "browser_mutation_target_token": "required_current_target_token",
    "task_completion_requires_current_session": True,
    "labeler_promotion_retry_requires_open_task": True,
    "labeler_promotion_retry_requires_current_session": True,
    "labeler_promotion_retry_mutation_enabled": False,
    "labeler_promotion_retry_rejection_error": "operator_support_required",
    "completion_closes_open_sessions": True,
    "reopen_authority": "operator",
    "reopen_required_for_more_labeling": True,
}

ASSIGNMENT_OWNERSHIP_POLICY: dict[str, object] = {
    "assignment_scope": "recording",
    "recording_assignment_key": "recording_id",
    "recording_id_primary_key": True,
    "schema_enforced_recording_primary_key": True,
    "one_current_assignment_row_per_recording": True,
    "one_active_owner": True,
    "multiple_labelers_per_recording_allowed": False,
    "reassignment_replaces_owner": True,
    "stale_sessions_closed_on_reassignment": True,
    "stale_sessions_closed_before_assignment_update": True,
    "reassignment_target_validated_before_session_closure": True,
    "session_closure_and_assignment_update_atomic": True,
    "raw_assignment_change_blocks_open_sessions": True,
    "assignment_manifests_are_control_plane": True,
    "duplicate_manifest_rows_do_not_create_multiple_owners": True,
    "assignment_user_match_required_for_mutation": True,
    "browser_mutation_requires_current_assignment_owner": True,
    "browser_mutation_target_resolved_server_side": True,
    "browser_mutation_target_source": "recording_assignments.active_assignment",
    "labelers_mutate_assigned_training_zarrs": True,
    "labelers_mutate_intermediate_csvs": False,
    "operator_reassignment_helper": "assign_recording_with_session_closure",
}

BROWSER_RESPONSE_SECURITY_HEADERS: dict[str, str] = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

BROWSER_SIGNED_LINK_POLICY: dict[str, object] = {
    "canonical_entrypoint": DASHBOARD_PATH,
    "task_specific_links": "short_lived_convenience_links",
    "default_ttl_seconds": 24 * 60 * 60,
    "authorization_grant": False,
    "requires_authenticated_user": True,
    "requires_active_assignment": True,
    "requires_open_task": True,
    "binds_expected_user_in_new_links": True,
    "expected_user_mismatch_error": "signed_link_user_mismatch",
    "opens_guarded_session": True,
    "session_bound_after_open": True,
    "runtime_operator_validation_start_gate_enforced": True,
    "dashboard_preferred_for_multi_task_work": True,
}

BROWSER_WORKFLOW_CAPABILITIES: tuple[dict[str, object], ...] = (
    {
        "workflow_kind": "keypoints",
        "label": "Keypoint correction",
        "browser_editor": True,
        "server_mutation": True,
        "completion_supported": True,
        "client_authority": dict(BROWSER_CLIENT_AUTHORITY),
        "write_scope": "Correct failed or reviewed keypoints through the guarded session editor.",
        "write_contract": {
            **dict(BROWSER_WORKFLOW_SERVER_WRITE_CONTRACT),
            "primary_mutation_target_kind": "task_scoped_training_zarr",
            "training_zarr_write_mode": "direct",
            "save_method": "POST",
            "save_endpoint": "/api/sessions/{session_id}/keypoints/save",
            "payload_fields": ["points", "advance", "target_token"],
            "required_fields": ["points", "target_token"],
            "response_fields": ["ok", "result", "state"],
            "audit_event": "save_keypoints",
            "audit_provenance": dict(BROWSER_MUTATION_AUDIT_PROVENANCE),
            "retry_policy": dict(BROWSER_MUTATION_RETRY_POLICY),
            "registry_refresh": True,
            "guard": "session_for_user",
        },
        "notes": "The browser submits keypoint edits; the server applies them through Palette review/write tooling.",
    },
    {
        "workflow_kind": "detect_training",
        "label": "Detection training boxes",
        "browser_editor": True,
        "server_mutation": True,
        "completion_supported": True,
        "client_authority": dict(BROWSER_CLIENT_AUTHORITY),
        "write_scope": "Edit training bounding boxes through the guarded session editor.",
        "write_contract": {
            **dict(BROWSER_WORKFLOW_SERVER_WRITE_CONTRACT),
            "primary_mutation_target_kind": "task_scoped_training_zarr",
            "training_zarr_write_mode": "direct",
            "save_method": "POST",
            "save_endpoint": "/api/sessions/{session_id}/detect/save",
            "payload_fields": ["bbox_norm", "advance", "target_token"],
            "required_fields": ["bbox_norm", "target_token"],
            "response_fields": ["ok", "result", "state"],
            "audit_event": "save_detect_bbox",
            "audit_provenance": dict(BROWSER_MUTATION_AUDIT_PROVENANCE),
            "retry_policy": dict(BROWSER_MUTATION_RETRY_POLICY),
            "registry_refresh": True,
            "guard": "session_for_user",
        },
        "notes": "The browser never receives direct zarr write authority.",
    },
    {
        "workflow_kind": "detect_analysis",
        "label": "Analysis detection boxes",
        "browser_editor": True,
        "server_mutation": True,
        "completion_supported": True,
        "client_authority": dict(BROWSER_CLIENT_AUTHORITY),
        "write_scope": "Reviewable by default; editable only when task scope enables analysis-box edits.",
        "write_contract": {
            **dict(BROWSER_WORKFLOW_SERVER_WRITE_CONTRACT),
            "primary_mutation_target_kind": "task_scoped_analysis_zarr",
            "source_mutation_target_kind": "task_scoped_analysis_zarr",
            "promotion_mutation_target_kind": "task_scoped_training_zarr",
            "training_zarr_write_mode": "promotion_when_configured",
            "save_method": "POST",
            "save_endpoint": "/api/sessions/{session_id}/detect-analysis/save",
            "payload_fields": ["bbox_norm", "advance", "target_token"],
            "required_fields": ["bbox_norm", "target_token"],
            "response_fields": ["ok", "result", "state", "promotion"],
            "audit_event": "save_detect_analysis_bbox",
            "audit_provenance": dict(BROWSER_MUTATION_AUDIT_PROVENANCE),
            "retry_policy": {
                **dict(BROWSER_MUTATION_RETRY_POLICY),
                "secondary_side_effects": ["promotion_success", "promotion_failed"],
                "retry_guidance": "Saving the same editable analysis box again should leave the analysis label data in the same state, but may enqueue or record another promotion attempt when promotion is enabled.",
            },
            "secondary_events": ["promotion_success", "promotion_failed"],
            "scope_required": {"editable": True},
            "registry_refresh": True,
            "guard": "session_for_user",
        },
        "notes": "Use task scope to decide whether a detection-analysis task is review-only or mutable.",
    },
    {
        "workflow_kind": "subject_mask_component",
        "label": "Subject mask component masks",
        "browser_editor": True,
        "server_mutation": True,
        "completion_supported": True,
        "client_authority": dict(BROWSER_CLIENT_AUTHORITY),
        "write_scope": "Edit assigned subject-mask components through the guarded session editor.",
        "write_contract": {
            **dict(BROWSER_WORKFLOW_SERVER_WRITE_CONTRACT),
            "primary_mutation_target_kind": "task_scoped_training_zarr",
            "training_zarr_write_mode": "session_checkpoint_then_apply",
            "save_method": "POST",
            "save_endpoint": "/api/sessions/{session_id}/subject-mask/save",
            "save_semantics": "checkpoint_only_no_canonical_zarr_write",
            "apply_method": "POST",
            "apply_endpoint": "/api/sessions/{session_id}/subject-mask/apply",
            "apply_semantics": "coalesce_saved_session_checkpoints_and_write_canonical_zarr_before_assignment_completion",
            "payload_fields": ["mask", "advance", "target_token"],
            "required_fields": ["mask", "target_token"],
            "response_fields": ["ok", "result", "state"],
            "audit_event": "checkpoint_subject_mask_roi",
            "canonical_apply_audit_event": "apply_subject_mask_session_checkpoints",
            "audit_provenance": dict(BROWSER_MUTATION_AUDIT_PROVENANCE),
            "retry_policy": dict(BROWSER_MUTATION_RETRY_POLICY),
            "registry_refresh": "apply_only",
            "guard": "session_for_user",
        },
        "notes": "Subject-mask browser saves checkpoint to the labeling sidecar; explicit apply writes the unified refined subject-mask path while the assignment remains open.",
    },
)


























def _expected_user_guard_contract_policy(
    labeler_safety: Mapping[str, object],
    signed_link_contract: Mapping[str, object],
) -> dict[str, object]:
    guards = (
        labeler_safety.get("expected_user_guards")
        if isinstance(labeler_safety.get("expected_user_guards"), Mapping)
        else {}
    )
    required_guards = {
        "labeler_landing_page": "dashboard_user_mismatch",
        "labeler_me_page": "dashboard_user_mismatch",
        "labeling_home_page": "dashboard_user_mismatch",
        "dashboard": "dashboard_user_mismatch",
        "dataset_queue_page": "dashboard_user_mismatch",
        "personal_work_page": "dashboard_user_mismatch",
        "personal_dataset_queue_page": "dashboard_user_mismatch",
        "personal_work_api": "dashboard_user_mismatch",
        "dataset_queue_api": "dashboard_user_mismatch",
        "task_open_api": "task_open_user_mismatch",
        "task_complete_api": "task_complete_user_mismatch",
        "promotion_retry_api": "promotion_retry_user_mismatch",
        "signed_task_link": "signed_link_user_mismatch",
    }
    missing_guards = [
        key
        for key, expected_value in required_guards.items()
        if str(guards.get(key) or "") != expected_value
    ]
    promotion_retry_labeler_mutation_enabled = bool(
        labeler_safety.get("promotion_retry_labeler_mutation_enabled")
    )
    promotion_retry_labeler_rejection_error = str(
        labeler_safety.get("promotion_retry_labeler_rejection_error") or ""
    )
    signed_links_expected_user_bound = bool(signed_link_contract.get("binds_expected_user_in_new_links"))
    signed_links_recheck_identity = bool(signed_link_contract.get("forwarded_signed_links_recheck_identity"))
    ready = (
        not missing_guards
        and signed_links_expected_user_bound
        and signed_links_recheck_identity
        and not promotion_retry_labeler_mutation_enabled
        and promotion_retry_labeler_rejection_error == "operator_support_required"
    )
    return {
        "schema": "palette.web_labeling_expected_user_guard_contract.v1",
        "ready": ready,
        "required_guards": required_guards,
        "configured_guards": dict(guards),
        "missing_or_mismatched_guards": missing_guards,
        "guarded_labeler_entrypoints": [
            "labeler_landing_page",
            "labeler_me_page",
            "labeling_home_page",
            "dashboard",
            "dataset_queue_page",
            "personal_work_page",
            "personal_dataset_queue_page",
        ],
        "guarded_labeler_apis": [
            "personal_work_api",
            "dataset_queue_api",
            "task_open_api",
            "task_complete_api",
            "promotion_retry_api",
        ],
        "signed_links_expected_user_bound": signed_links_expected_user_bound,
        "signed_links_recheck_identity": signed_links_recheck_identity,
        "promotion_retry_labeler_mutation_enabled": promotion_retry_labeler_mutation_enabled,
        "promotion_retry_labeler_rejection_error": promotion_retry_labeler_rejection_error,
        "promotion_retry_guarded_support_only": (
            not promotion_retry_labeler_mutation_enabled
            and promotion_retry_labeler_rejection_error == "operator_support_required"
        ),
        "forwarded_links_stop_on_expected_user_mismatch": ready,
    }


def _operator_authorization_contract_policy(policy: Mapping[str, object]) -> dict[str, object]:
    admin_routes_require_operator = bool(policy.get("admin_routes_require_operator"))
    admin_route_prefixes = [
        str(prefix)
        for prefix in (
            policy.get("admin_route_prefixes")
            if isinstance(policy.get("admin_route_prefixes"), list)
            else []
        )
        if str(prefix).strip()
    ]
    admin_required_error = str(policy.get("admin_required_error") or "")
    resolved_user_must_be_in_admin_users = bool(policy.get("resolved_user_must_be_in_admin_users"))
    labelers_are_not_operators_by_default = bool(policy.get("labelers_are_not_operators_by_default"))
    operator_authorization_grants_labeler_mutation = bool(
        policy.get("operator_authorization_grants_labeler_mutation")
    )
    operator_boundary_required_for_launch = bool(policy.get("operator_boundary_required_for_launch"))
    production_requires_admin_user = bool(policy.get("production_requires_admin_user"))
    ready = (
        admin_routes_require_operator
        and "/admin" in admin_route_prefixes
        and "/api/admin" in admin_route_prefixes
        and admin_required_error == "admin_required"
        and resolved_user_must_be_in_admin_users
        and labelers_are_not_operators_by_default
        and not operator_authorization_grants_labeler_mutation
        and operator_boundary_required_for_launch
        and production_requires_admin_user
    )
    return {
        "schema": "palette.web_labeling_operator_authorization_contract.v1",
        "ready": ready,
        "admin_routes_require_operator": admin_routes_require_operator,
        "admin_route_prefixes": admin_route_prefixes,
        "admin_required_error": admin_required_error,
        "resolved_user_must_be_in_admin_users": resolved_user_must_be_in_admin_users,
        "labelers_are_not_operators_by_default": labelers_are_not_operators_by_default,
        "operator_authorization_grants_labeler_mutation": operator_authorization_grants_labeler_mutation,
        "operator_boundary_required_for_launch": operator_boundary_required_for_launch,
        "production_requires_admin_user": production_requires_admin_user,
        "runtime_preflight_required": bool(policy.get("runtime_preflight_required")),
        "operator_boundary_known": bool(policy.get("operator_boundary_known")),
        "operator_boundary_ready": bool(policy.get("operator_boundary_ready")),
        "admin_users_configured": bool(policy.get("admin_users_configured")),
    }












def _mutation_audit_contract_policy(policy: Mapping[str, object]) -> dict[str, object]:
    event_store = str(policy.get("event_store") or "")
    append_only = bool(policy.get("append_only"))
    server_records_events = bool(policy.get("server_records_events"))
    browser_records_events_directly = bool(policy.get("browser_records_events_directly"))
    browser_receives_audit_store_write_credentials = bool(
        policy.get("browser_receives_audit_store_write_credentials")
    )
    per_workflow_contracts_include_audit_provenance = bool(
        policy.get("per_workflow_write_contracts_include_audit_provenance")
    )
    required_event_fields = [
        str(field)
        for field in (
            policy.get("required_event_fields")
            if isinstance(policy.get("required_event_fields"), list)
            else []
        )
        if str(field).strip()
    ]
    timestamp_field = str(policy.get("timestamp_field") or "")
    required_fields_present = all(
        field in required_event_fields
        for field in ("event_id", "task_id", "recording_id", "user", "event_type")
    ) and bool(timestamp_field)
    ready = (
        event_store == "labeling_task_events"
        and append_only
        and server_records_events
        and not browser_records_events_directly
        and not browser_receives_audit_store_write_credentials
        and per_workflow_contracts_include_audit_provenance
        and required_fields_present
    )
    return {
        "schema": "palette.web_labeling_mutation_audit_contract.v1",
        "ready": ready,
        "event_store": event_store,
        "append_only": append_only,
        "server_records_events": server_records_events,
        "browser_records_events_directly": browser_records_events_directly,
        "browser_receives_audit_store_write_credentials": browser_receives_audit_store_write_credentials,
        "per_workflow_write_contracts_include_audit_provenance": per_workflow_contracts_include_audit_provenance,
        "required_event_fields": required_event_fields,
        "required_event_fields_present": required_fields_present,
        "timestamp_field": timestamp_field,
        "same_payload_retry_safe": bool(policy.get("same_payload_retry_safe")),
        "duplicate_audit_events_possible": bool(policy.get("duplicate_audit_events_possible")),
        "validation_gate": str(policy.get("validation_gate") or ""),
    }






















def _handoff_store_checks_ok_for_user(
    check_report: Mapping[str, object],
    public_reassignment_session_safety: Mapping[str, object],
) -> bool:
    if bool(check_report.get("ok")):
        return True
    issue_counts = check_report.get("issue_counts") if isinstance(check_report.get("issue_counts"), Mapping) else {}
    issue_codes = {str(code) for code in issue_counts}
    if issue_codes and issue_codes <= {"active_session_assignment_mismatch"}:
        return bool(public_reassignment_session_safety.get("ok", True))
    return False











































def _personal_work_url_for_base(base_url: str | None) -> str:
    if not base_url:
        return ""
    return f"{str(base_url).rstrip('/')}{PERSONAL_WORK_PATH}"




def _identity_probe_url_for_base(base_url: str | None) -> str:
    if not base_url:
        return ""
    return f"{str(base_url).rstrip('/')}{IDENTITY_PROBE_PATH}"


















DEFAULT_OPERATOR_VALIDATION_GATE_IDS = (
    "mutable_zarr_backup_confirmation",
    "browser_response_security_headers",
    "identity_probe_verification",
    "browser_smoke",
    "disposable_zarr_mutation_smoke",
    "operator_recovery_contract",
)

OPERATOR_VALIDATION_GATE_STATUS_VALUES = (
    "unknown",
    "pending",
    "missing_evidence",
    "needs_review",
    "passed",
)

OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES = (
    "status",
    "pending",
    "missing_evidence",
    "needs_review",
    "passed",
)

IDENTITY_PERSONAL_QUEUE_EVIDENCE_STATUS_VALUES = (
    "missing",
    "incomplete",
    "ready",
)







































_DASHBOARD_READY_TO_INVITE_LEGACY_SEMANTICS = "row_readiness_not_safe_share_approval"
_DASHBOARD_READY_ROW_DRAFT_BUNDLE_SCHEMA = "palette.web_labeling_ready_row_draft_bundle.v1"
_DASHBOARD_READY_ROW_DRAFT_BUNDLE_KIND = "ready_row_draft_text"
_DASHBOARD_READY_ROW_STATE_VALUES = ("ready_row_draft", "diagnostic_note")
_DASHBOARD_COPY_INTENT_VALUES = _DASHBOARD_READY_ROW_STATE_VALUES
_DASHBOARD_READY_ROW_DRAFT_LEGACY_SEMANTICS = "draft_text_only_safe_share_required"
_DASHBOARD_READY_ROW_DRAFT_LEGACY_FIELD_NAMES = (
    "ready_invitations",
    "ready_invitations_text",
)
_DASHBOARD_BROWSER_MUTATION_TARGET_REQUIRED_VALUES = {
    "browser_mutation_write_ready": True,
    "browser_mutation_label_mutation_target_kind": "task_scoped_training_zarr",
    "browser_mutation_browser_label_write_target": "training_zarr",
    "browser_mutation_csv_handoff_artifact_role": "metadata_only_control_plane",
    "browser_mutation_csv_handoff_artifacts_are_label_write_targets": False,
    "browser_mutation_handoff_csv_artifacts_are_label_write_targets": False,
    "browser_mutation_intermediate_csv_artifacts_are_label_write_targets": False,
    "browser_mutation_browser_writes_csv_or_handoff_files": False,
    "browser_mutation_browser_writes_handoff_csv": False,
    "browser_mutation_browser_writes_intermediate_csv": False,
    "browser_mutation_browser_has_direct_zarr_write_authority": False,
}
_DASHBOARD_BROWSER_MUTATION_TARGET_FIELDS = tuple(
    _DASHBOARD_BROWSER_MUTATION_TARGET_REQUIRED_VALUES.keys()
)
_DASHBOARD_DIRECT_BROWSER_START_REQUIRED_VALUES = {
    "dataset_queue_direct_start_policy_present": True,
    "dataset_queue_direct_start_enabled": True,
    "dataset_queue_direct_start_method": "POST",
    "dataset_queue_direct_start_same_origin_only": True,
    "dataset_queue_direct_start_exact_route_required": True,
    "dataset_queue_direct_start_endpoint_task_segment_must_match_row_task_id": True,
    "dataset_queue_direct_start_expected_user_guard_required": True,
    "dataset_queue_direct_start_post_body_expected_user_required": True,
    "dataset_queue_direct_start_post_body_expected_user_field": "expected_user",
    "dataset_queue_direct_start_denied_start_returns_task_open_authorization_contract": True,
    "dataset_queue_direct_start_denied_start_support_preserves_task_open_authorization_contract": True,
    "dataset_queue_direct_start_denied_start_support_includes_authorization_context": True,
    "dataset_queue_direct_start_denied_start_contract_reports_no_session_created": True,
    "dataset_queue_direct_start_denied_start_contract_reports_server_authorizes_open_false": True,
    "dataset_queue_direct_start_non_startable_tasks_do_not_advertise_endpoint": True,
    "dataset_queue_direct_start_label_mutation_target_kind": "task_scoped_training_zarr",
    "dataset_queue_direct_start_browser_label_write_target": "training_zarr",
    "dataset_queue_direct_start_csv_handoff_artifact_role": "metadata_only_control_plane",
    "dataset_queue_direct_start_csv_handoff_artifacts_are_label_write_targets": False,
    "dataset_queue_direct_start_handoff_csv_artifacts_are_label_write_targets": False,
    "dataset_queue_direct_start_intermediate_csv_artifacts_are_label_write_targets": False,
    "dataset_queue_direct_start_browser_writes_csv_or_handoff_files": False,
    "dataset_queue_direct_start_browser_writes_handoff_csv": False,
    "dataset_queue_direct_start_browser_writes_intermediate_csv": False,
    "dataset_queue_direct_start_browser_receives_zarr_write_authority": False,
    "dataset_queue_direct_start_browser_has_direct_zarr_write_authority": False,
}
_DASHBOARD_DIRECT_BROWSER_START_FIELDS = tuple(
    _DASHBOARD_DIRECT_BROWSER_START_REQUIRED_VALUES.keys()
)
_DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_FIELD = "labeler_links_safe_to_share"
_DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_VALUE = True
_DASHBOARD_READY_ROW_DRAFT_SHARE_RULE = (
    "Do not share copied ready-row draft text until inspect-handoff "
    f"--require-shareable reports {_DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_FIELD}=true."
)









def _configure_handoff_shareability_helpers() -> None:
    _web_handoff_shareability.configure_handoff_shareability_dependencies(
        {
            "BROWSER_MUTATION_TARGET_SELECTOR_KEYS": BROWSER_MUTATION_TARGET_SELECTOR_KEYS,
            "DASHBOARD_PATH": DASHBOARD_PATH,
            "DATASET_QUEUE_PATH": DATASET_QUEUE_PATH,
            "DEFAULT_OPERATOR_VALIDATION_GATE_IDS": DEFAULT_OPERATOR_VALIDATION_GATE_IDS,
            "IDENTITY_PERSONAL_QUEUE_EVIDENCE_STATUS_VALUES": IDENTITY_PERSONAL_QUEUE_EVIDENCE_STATUS_VALUES,
            "IDENTITY_PROBE_PATH": IDENTITY_PROBE_PATH,
            "LABELING_HOME_PATH": LABELING_HOME_PATH,
            "OPERATOR_VALIDATION_GATE_STATUS_VALUES": OPERATOR_VALIDATION_GATE_STATUS_VALUES,
            "PERSONAL_DATASET_QUEUE_PATH": PERSONAL_DATASET_QUEUE_PATH,
            "PERSONAL_WORK_PATH": PERSONAL_WORK_PATH,
            "_DASHBOARD_BROWSER_MUTATION_TARGET_FIELDS": _DASHBOARD_BROWSER_MUTATION_TARGET_FIELDS,
            "_DASHBOARD_BROWSER_MUTATION_TARGET_REQUIRED_VALUES": _DASHBOARD_BROWSER_MUTATION_TARGET_REQUIRED_VALUES,
            "_DASHBOARD_COPY_INTENT_VALUES": _DASHBOARD_COPY_INTENT_VALUES,
            "_DASHBOARD_DIRECT_BROWSER_START_FIELDS": _DASHBOARD_DIRECT_BROWSER_START_FIELDS,
            "_DASHBOARD_DIRECT_BROWSER_START_REQUIRED_VALUES": _DASHBOARD_DIRECT_BROWSER_START_REQUIRED_VALUES,
            "_DASHBOARD_READY_ROW_DRAFT_BUNDLE_KIND": _DASHBOARD_READY_ROW_DRAFT_BUNDLE_KIND,
            "_DASHBOARD_READY_ROW_DRAFT_BUNDLE_SCHEMA": _DASHBOARD_READY_ROW_DRAFT_BUNDLE_SCHEMA,
            "_DASHBOARD_READY_ROW_DRAFT_LEGACY_FIELD_NAMES": _DASHBOARD_READY_ROW_DRAFT_LEGACY_FIELD_NAMES,
            "_DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_FIELD": _DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_FIELD,
            "_DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_VALUE": _DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_VALUE,
            "_DASHBOARD_READY_ROW_STATE_VALUES": _DASHBOARD_READY_ROW_STATE_VALUES,
            "_browser_response_security_policy": _browser_response_security_policy,
            "_handoff_assignment_ownership_fields": _handoff_assignment_ownership_fields,
            "_handoff_browser_mutation_write_fields": _handoff_browser_mutation_write_fields,
            "_handoff_browser_response_security_fields": _handoff_browser_response_security_fields,
            "_handoff_dataset_queue_blocks_labeler_start": _handoff_dataset_queue_blocks_labeler_start,
            "_handoff_entry_artifact_fields": _handoff_entry_artifact_fields,
            "_handoff_known_user_status_fields": _handoff_known_user_status_fields,
            "_handoff_labeler_route_authorization_fields": _handoff_labeler_route_authorization_fields,
            "_handoff_labeler_safety_fields": _handoff_labeler_safety_fields,
            "_handoff_mutation_audit_fields": _handoff_mutation_audit_fields,
            "_handoff_session_guard_fields": _handoff_session_guard_fields,
            "_handoff_signed_link_policy_fields": _handoff_signed_link_policy_fields,
            "_handoff_task_state_policy_fields": _handoff_task_state_policy_fields,
            "_handoff_zarr_backup_fields": _handoff_zarr_backup_fields,
            "_implementation_status_inspection_target_fields": _implementation_status_inspection_target_fields,
            "_launch_evidence_execution_checklist_inspection_target": _launch_evidence_execution_checklist_inspection_target,
            "_launch_evidence_execution_checklist_status": _launch_evidence_execution_checklist_status,
            "_operator_validation_command_templates": _operator_validation_command_templates,
            "_personalized_launch_readiness_target": _personalized_launch_readiness_target,
            "_safe_share_external_launch_evidence_gap_field_names": _safe_share_external_launch_evidence_gap_field_names,
            "_safe_share_gate_flat_fields": _safe_share_gate_flat_fields,
            "_safe_share_gate_policy": _safe_share_gate_policy,
            "_safe_share_next_action_command_fields": _safe_share_next_action_command_fields,
            "_safe_share_next_action_detail_fields": _safe_share_next_action_detail_fields,
            "_shareability_labeler_route_authorization_runtime_checklist_fields": _shareability_labeler_route_authorization_runtime_checklist_fields,
            "_shareability_labeler_route_authorization_runtime_checklist_required_values": _shareability_labeler_route_authorization_runtime_checklist_required_values,
        }
    )

def _single_owner_package_contract_summary(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._single_owner_package_contract_summary_impl(*args, **kwargs)


def _handoff_sendability_reasons(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._handoff_sendability_reasons_impl(*args, **kwargs)


def _handoff_sendability_actions(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._handoff_sendability_actions_impl(*args, **kwargs)


def _handoff_sendability_summary(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._handoff_sendability_summary_impl(*args, **kwargs)


def _count_handoff_sendability_reasons(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._count_handoff_sendability_reasons_impl(*args, **kwargs)


def _shareability_safe_to_share_requires(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._shareability_safe_to_share_requires_impl(*args, **kwargs)


def _shareability_labeler_route_authorization_runtime_checklist_gate_contract(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._shareability_labeler_route_authorization_runtime_checklist_gate_contract_impl(*args, **kwargs)


def _shareability_compact_contract_fields(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._shareability_compact_contract_fields_impl(*args, **kwargs)


def _shareability_compact_contract_source_fields(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._shareability_compact_contract_source_fields_impl(*args, **kwargs)


def _shareability_compact_contract_safe_to_share_target(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._shareability_compact_contract_safe_to_share_target_impl(*args, **kwargs)


def _shareability_compact_contract_next_action_target(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._shareability_compact_contract_next_action_target_impl(*args, **kwargs)


def _shareability_external_launch_evidence_gap_target(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._shareability_external_launch_evidence_gap_target_impl(*args, **kwargs)


def _shareability_repair_command_detail_fields(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._shareability_repair_command_detail_fields_impl(*args, **kwargs)


def _shareability_repair_command_detail_fields_by_id(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._shareability_repair_command_detail_fields_by_id_impl(*args, **kwargs)


def _shareability_repair_command_contracts(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._shareability_repair_command_contracts_impl(*args, **kwargs)


def _write_launch_bundle_inspection_targets(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._write_launch_bundle_inspection_targets_impl(*args, **kwargs)


def _inspection_failure_actions(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._inspection_failure_actions_impl(*args, **kwargs)


def _inspection_operator_repair_commands(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._inspection_operator_repair_commands_impl(*args, **kwargs)


def _inspection_labeler_entrypoint_summary(*args: object, **kwargs: object) -> object:
    _configure_handoff_shareability_helpers()
    return _web_handoff_shareability._inspection_labeler_entrypoint_summary_impl(*args, **kwargs)




def _configure_dashboard_roster_renderer_helpers() -> None:
    _web_report_renderers.configure_dashboard_roster_renderer_dependencies(
        {
            "_DASHBOARD_READY_ROW_DRAFT_BUNDLE_KIND": _DASHBOARD_READY_ROW_DRAFT_BUNDLE_KIND,
            "_DASHBOARD_READY_ROW_DRAFT_BUNDLE_SCHEMA": _DASHBOARD_READY_ROW_DRAFT_BUNDLE_SCHEMA,
            "_DASHBOARD_READY_ROW_DRAFT_LEGACY_FIELD_NAMES": _DASHBOARD_READY_ROW_DRAFT_LEGACY_FIELD_NAMES,
            "_DASHBOARD_READY_ROW_DRAFT_LEGACY_SEMANTICS": _DASHBOARD_READY_ROW_DRAFT_LEGACY_SEMANTICS,
            "_DASHBOARD_READY_ROW_DRAFT_SHARE_RULE": _DASHBOARD_READY_ROW_DRAFT_SHARE_RULE,
            "_operator_validation_command_templates": _operator_validation_command_templates,
            "_runtime_operator_validation_gate_cli_policy": _runtime_operator_validation_gate_cli_policy,
            "_safe_share_next_action_command_fields": _safe_share_next_action_command_fields,
            "_safe_share_next_action_detail_fields": _safe_share_next_action_detail_fields,
        }
    )


def _dashboard_ready_invitation_bundle(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    _configure_dashboard_roster_renderer_helpers()
    return _web_report_renderers._dashboard_ready_invitation_bundle_impl(rows)


def _dashboard_roster_html(payload: Mapping[str, object]) -> str:
    _configure_dashboard_roster_renderer_helpers()
    return _web_report_renderers._dashboard_roster_html_impl(payload)

















def _handoff_ready_to_send(handoff: Mapping[str, object]) -> bool:
    return _handoff_ready_to_send_impl(
        handoff,
        helpers={
            '_handoff_assignment_ownership_fields': _handoff_assignment_ownership_fields,
            '_handoff_browser_mutation_write_fields': _handoff_browser_mutation_write_fields,
            '_handoff_browser_response_security_fields': _handoff_browser_response_security_fields,
            '_handoff_dataset_queue_blocks_labeler_start': _handoff_dataset_queue_blocks_labeler_start,
            '_handoff_entry_artifact_fields': _handoff_entry_artifact_fields,
            '_handoff_known_user_status_fields': _handoff_known_user_status_fields,
            '_handoff_labeler_route_authorization_fields': _handoff_labeler_route_authorization_fields,
            '_handoff_labeler_safety_fields': _handoff_labeler_safety_fields,
            '_handoff_mutation_audit_fields': _handoff_mutation_audit_fields,
            '_handoff_session_guard_fields': _handoff_session_guard_fields,
            '_handoff_signed_link_policy_fields': _handoff_signed_link_policy_fields,
            '_handoff_task_state_policy_fields': _handoff_task_state_policy_fields,
            '_handoff_zarr_backup_fields': _handoff_zarr_backup_fields
        },
    )


def _handoff_browser_response_security_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    return _handoff_browser_response_security_fields_impl(
        handoff,
        helpers={
            '_browser_response_security_contract_policy': _browser_response_security_contract_policy
        },
    )




def _handoff_mutation_audit_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    return _handoff_mutation_audit_fields_impl(
        handoff,
        helpers={
            '_mutation_audit_contract_policy': _mutation_audit_contract_policy
        },
    )


def _handoff_zarr_backup_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    return _handoff_zarr_backup_fields_impl(
        handoff,
        helpers={
            '_zarr_backup_contract_policy': _zarr_backup_contract_policy
        },
    )


def _handoff_task_state_policy_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    return _handoff_task_state_policy_fields_impl(
        handoff,
        helpers={
            '_browser_mutation_target_contract_policy': _browser_mutation_target_contract_policy,
            '_browser_task_state_contract_policy': _browser_task_state_contract_policy
        },
    )


def _handoff_session_guard_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    return _handoff_session_guard_fields_impl(
        handoff,
        helpers={
            '_session_guard_contract_policy': _session_guard_contract_policy
        },
    )


def _handoff_labeler_safety_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    return _handoff_labeler_safety_fields_impl(
        handoff,
        helpers={
            '_browser_payload_redaction_contract_policy': _browser_payload_redaction_contract_policy,
            '_browser_signed_link_policy': _browser_signed_link_policy,
            '_expected_user_guard_contract_policy': _expected_user_guard_contract_policy,
            '_signed_link_contract_policy': _signed_link_contract_policy
        },
    )


def _handoff_signed_link_policy_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    return _handoff_signed_link_policy_fields_impl(
        handoff,
        helpers={
            '_signed_link_contract_policy': _signed_link_contract_policy
        },
    )


def _handoff_labeler_route_authorization_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    return _handoff_labeler_route_authorization_fields_impl(
        handoff,
        helpers={
            '_labeler_route_authorization_contract_policy': _labeler_route_authorization_contract_policy
        },
    )












def _sum_recordings_without_open_tasks_by_reason(manifests: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for manifest in manifests:
        manifest_counts = manifest.get("counts") if isinstance(manifest.get("counts"), Mapping) else {}
        by_reason = (
            manifest_counts.get("recordings_without_open_tasks_by_reason")
            if isinstance(manifest_counts, Mapping)
            else None
        )
        if isinstance(by_reason, Mapping):
            for reason, value in by_reason.items():
                reason_text = str(reason or "unknown").strip() or "unknown"
                counts[reason_text] = counts.get(reason_text, 0) + int(value or 0)
            continue
        unknown_count = int(manifest_counts.get("recordings_without_open_tasks") or 0) if isinstance(manifest_counts, Mapping) else 0
        if unknown_count:
            counts["unknown"] = counts.get("unknown", 0) + unknown_count
    return dict(sorted(counts.items()))


def _sum_handoff_progress_summary(manifests: Sequence[Mapping[str, object]]) -> dict[str, object]:
    summary = {
        "recording_count": 0,
        "open_task_count": 0,
        "total_task_count": 0,
        "complete_task_count": 0,
        "incomplete_task_count": 0,
        "startable_task_count": 0,
        "non_startable_task_count": 0,
        "waiting_recording_count": 0,
        "complete_recording_count": 0,
        "blocked_recording_count": 0,
        "blocked_recordings_by_reason": {},
    }
    blocked_by_reason: dict[str, int] = {}
    for manifest in manifests:
        progress = manifest.get("progress_summary") if isinstance(manifest.get("progress_summary"), Mapping) else {}
        for key in (
            "recording_count",
            "open_task_count",
            "total_task_count",
            "complete_task_count",
            "incomplete_task_count",
            "startable_task_count",
            "non_startable_task_count",
            "waiting_recording_count",
            "complete_recording_count",
            "blocked_recording_count",
        ):
            summary[key] = int(summary[key]) + int(progress.get(key) or 0)
        by_reason = progress.get("blocked_recordings_by_reason") if isinstance(progress, Mapping) else {}
        if isinstance(by_reason, Mapping):
            for reason, value in by_reason.items():
                reason_text = str(reason or "unknown").strip() or "unknown"
                blocked_by_reason[reason_text] = blocked_by_reason.get(reason_text, 0) + int(value or 0)
    summary["blocked_recordings_by_reason"] = dict(sorted(blocked_by_reason.items()))
    return summary


def _sum_handoff_dataset_queue_summary(manifests: Sequence[Mapping[str, object]]) -> dict[str, object]:
    dataset_count = 0
    waiting_dataset_count = 0
    open_task_count = 0
    non_startable_task_count = 0
    complete_task_count = 0
    task_count = 0
    dataset_ids: list[str] = []
    for manifest in manifests:
        summary = manifest.get("dataset_queue_summary") if isinstance(manifest.get("dataset_queue_summary"), Mapping) else {}
        dataset_count += int(summary.get("dataset_count") or 0)
        waiting_dataset_count += int(summary.get("waiting_dataset_count") or 0)
        open_task_count += int(summary.get("open_task_count") or 0)
        non_startable_task_count += int(summary.get("non_startable_task_count") or 0)
        complete_task_count += int(summary.get("complete_task_count") or 0)
        task_count += int(summary.get("task_count") or 0)
        ids = summary.get("dataset_ids") if isinstance(summary.get("dataset_ids"), list) else []
        dataset_ids.extend(str(dataset_id) for dataset_id in ids if str(dataset_id).strip())
    return {
        "dataset_count": dataset_count,
        "waiting_dataset_count": waiting_dataset_count,
        "open_task_count": open_task_count,
        "non_startable_task_count": non_startable_task_count,
        "complete_task_count": complete_task_count,
        "task_count": task_count,
        "dataset_ids": dataset_ids,
    }




def _count_redacted_summary_fields(value: object) -> int:
    if isinstance(value, Mapping):
        count = 0
        redacted_fields = value.get("redacted_fields")
        if isinstance(redacted_fields, list):
            count += len(redacted_fields)
        for key, item in value.items():
            if key == "redacted_fields":
                continue
            count += _count_redacted_summary_fields(item)
        return count
    if isinstance(value, list):
        return sum(_count_redacted_summary_fields(item) for item in value)
    return 0









def _write_user_handoffs_html_index(index: dict[str, object], output_path: Path) -> None:
    _write_user_handoffs_html_index_render(
        index,
        output_path,
        handoff_ready_to_send=_handoff_ready_to_send,
    )







def _write_user_handoffs_roster_csv(index: dict[str, object], output_path: Path) -> None:
    return _write_user_handoffs_roster_csv_impl(
        index,
        output_path,
        helpers={
            "_browser_mutation_target_contract_summary": _browser_mutation_target_contract_summary,
            "_browser_mutation_write_policy": _browser_mutation_write_policy,
            "_dataset_queue_direct_start_policy_fields": _dataset_queue_direct_start_policy_fields,
            "_direct_browser_start_contract_summary": _direct_browser_start_contract_summary,
            "_direct_browser_start_contract_summary_fields": _direct_browser_start_contract_summary_fields,
            "_handoff_assignment_ownership_fields": _handoff_assignment_ownership_fields,
            "_handoff_browser_mutation_write_fields": _handoff_browser_mutation_write_fields,
            "_handoff_browser_response_security_fields": _handoff_browser_response_security_fields,
            "_handoff_dataset_queue_start_fields": _handoff_dataset_queue_start_fields,
            "_handoff_entry_artifact_fields": _handoff_entry_artifact_fields,
            "_handoff_known_user_status_fields": _handoff_known_user_status_fields,
            "_handoff_labeler_route_authorization_fields": _handoff_labeler_route_authorization_fields,
            "_handoff_labeler_safety_fields": _handoff_labeler_safety_fields,
            "_handoff_mutation_audit_fields": _handoff_mutation_audit_fields,
            "_handoff_operator_recovery_fields": _handoff_operator_recovery_fields,
            "_handoff_ready_to_send": _handoff_ready_to_send,
            "_handoff_session_guard_fields": _handoff_session_guard_fields,
            "_handoff_signed_link_policy_fields": _handoff_signed_link_policy_fields,
            "_handoff_task_state_policy_fields": _handoff_task_state_policy_fields,
            "_handoff_zarr_backup_fields": _handoff_zarr_backup_fields,
            "_labeler_work_completion_fields": _labeler_work_completion_fields,
            "_operator_recovery_policy": _operator_recovery_policy,
            "_operator_validation_command_template_fields": _operator_validation_command_template_fields,
            "_operator_validation_gate_flat_fieldnames": _operator_validation_gate_flat_fieldnames,
            "_operator_validation_gate_flat_fields": _operator_validation_gate_flat_fields,
            "_operator_validation_public_fields": _operator_validation_public_fields,
            "_operator_validation_visibility_fields": _operator_validation_visibility_fields,
            "_personalized_launch_readiness_field_names": _personalized_launch_readiness_field_names,
            "_personalized_launch_readiness_summary": _personalized_launch_readiness_summary,
            "_public_reassignment_session_safety_fields": _public_reassignment_session_safety_fields,
            "_queue_first_entry_contract_flat_fields": _queue_first_entry_contract_flat_fields,
            "_runtime_operator_validation_gate_cli_policy_fields": _runtime_operator_validation_gate_cli_policy_fields,
            "_safe_share_checklist_gate_status_fields_from_operator_validation": _safe_share_checklist_gate_status_fields_from_operator_validation,
            "_safe_share_external_launch_evidence_gap_field_names": _safe_share_external_launch_evidence_gap_field_names,
            "_safe_share_gate_flat_fields": _safe_share_gate_flat_fields,
            "_safe_share_gate_policy": _safe_share_gate_policy
        },
    )


def _configure_validation_report_helpers() -> None:
    _web_validation_reports.configure_validation_report_dependencies(
        {
            "DATASET_QUEUE_PATH": DATASET_QUEUE_PATH,
            "LABELING_HOME_PATH": LABELING_HOME_PATH,
            "PERSONAL_DATASET_QUEUE_PATH": PERSONAL_DATASET_QUEUE_PATH,
            "PERSONAL_WORK_PATH": PERSONAL_WORK_PATH,
            "_IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS": _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS,
            "_IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT": _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT,
            "_IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELDS": _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELDS,
            "_IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELD_COUNT": _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELD_COUNT,
            "_assignment_ownership_contract_policy": _assignment_ownership_contract_policy,
            "_assignment_ownership_policy": _assignment_ownership_policy,
            "_browser_mutation_target_contract_policy": _browser_mutation_target_contract_policy,
            "_browser_mutation_write_contract_policy": _browser_mutation_write_contract_policy,
            "_browser_mutation_write_policy": _browser_mutation_write_policy,
            "_browser_payload_redaction_contract_policy": _browser_payload_redaction_contract_policy,
            "_browser_response_security_contract_policy": _browser_response_security_contract_policy,
            "_browser_response_security_policy": _browser_response_security_policy,
            "_browser_signed_link_policy": _browser_signed_link_policy,
            "_browser_task_state_contract_policy": _browser_task_state_contract_policy,
            "_browser_task_state_policy": _browser_task_state_policy,
            "_browser_workflow_capabilities": _browser_workflow_capabilities,
            "_browser_workflow_scope_contract_policy": _browser_workflow_scope_contract_policy,
            "_dataset_queue_direct_start_policy": _dataset_queue_direct_start_policy,
            "_expected_user_guard_contract_policy": _expected_user_guard_contract_policy,
            "_handoff_dataset_queue_blocks_labeler_start": _handoff_dataset_queue_blocks_labeler_start,
            "_identity_probe_link_contract_policy": _identity_probe_link_contract_policy,
            "_implementation_status_artifact": _implementation_status_artifact,
            "_implementation_status_flat_fields_from_artifact": _implementation_status_flat_fields_from_artifact,
            "_labeler_route_authorization_contract_policy": _labeler_route_authorization_contract_policy,
            "_labeler_route_authorization_policy": _labeler_route_authorization_policy,
            "_labeler_safety_policy": _labeler_safety_policy,
            "_mutation_audit_contract_policy": _mutation_audit_contract_policy,
            "_mutation_audit_policy": _mutation_audit_policy,
            "_operator_authorization_contract_policy": _operator_authorization_contract_policy,
            "_operator_authorization_policy": _operator_authorization_policy,
            "_operator_recovery_contract_policy": _operator_recovery_contract_policy,
            "_operator_recovery_policy": _operator_recovery_policy,
            "_operator_validation_command_templates": _operator_validation_command_templates,
            "_operator_validation_visibility_policy": _operator_validation_visibility_policy,
            "_personal_dataset_queue_url_for_base": _personal_dataset_queue_url_for_base,
            "_personal_work_url_for_base": _personal_work_url_for_base,
            "_queue_first_entry_contract_policy": _queue_first_entry_contract_policy,
            "_runtime_operator_validation_gate_cli_policy": _runtime_operator_validation_gate_cli_policy,
            "_runtime_operator_validation_gate_cli_policy_fields": _runtime_operator_validation_gate_cli_policy_fields,
            "_safe_share_checklist_gate_status_fields": _safe_share_checklist_gate_status_fields,
            "_safe_share_gate_flat_fields": _safe_share_gate_flat_fields,
            "_safe_share_gate_policy": _safe_share_gate_policy,
            "_session_guard_contract_policy": _session_guard_contract_policy,
            "_session_guard_policy": _session_guard_policy,
            "_signed_link_contract_policy": _signed_link_contract_policy,
            "_validation_gate": _validation_gate,
            "_validation_gate_classification": _validation_gate_classification,
            "_zarr_backup_contract_policy": _zarr_backup_contract_policy,
            "_zarr_backup_policy": _zarr_backup_policy,
        }
    )

def _write_web_labeling_validation_log(*args: object, **kwargs: object) -> object:
    _configure_validation_report_helpers()
    return _web_validation_reports._write_web_labeling_validation_log_impl(*args, **kwargs)


def _web_labeling_validation_checklist_payload(*args: object, **kwargs: object) -> object:
    _configure_validation_report_helpers()
    return _web_validation_reports._web_labeling_validation_checklist_payload_impl(*args, **kwargs)


def _write_web_labeling_validation_checklist(*args: object, **kwargs: object) -> object:
    _configure_validation_report_helpers()
    return _web_validation_reports._write_web_labeling_validation_checklist_impl(*args, **kwargs)






def _read_jsonl_objects(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _manifest_relative_path(manifest_path: Path, value: object, fallback_name: str) -> Path:
    text = str(value or "").strip()
    if text:
        path = Path(text)
        return path if path.is_absolute() else manifest_path.parent / path
    return manifest_path.parent / fallback_name


def _configure_handoff_validation_refresh_helpers() -> None:
    _web_handoff_validation_refresh.configure_handoff_validation_refresh_dependencies(
        {
            "DASHBOARD_PATH": DASHBOARD_PATH,
            "DATASET_QUEUE_PATH": DATASET_QUEUE_PATH,
            "LabelingStore": LabelingStore,
            "OPERATOR_EVIDENCE_TEMPLATE_FIELDS": OPERATOR_EVIDENCE_TEMPLATE_FIELDS,
            "_add_payload_contract_compact_fields": _add_payload_contract_compact_fields,
            "_add_work_summary_fields": _add_work_summary_fields,
            "_append_validation_log_evidence": _append_validation_log_evidence,
            "_assignment_ownership_integrity": _assignment_ownership_integrity,
            "_browser_mutation_write_policy": _browser_mutation_write_policy,
            "_browser_mutation_write_runtime_checklist": _browser_mutation_write_runtime_checklist,
            "_count_handoff_sendability_reasons": _count_handoff_sendability_reasons,
            "_dataset_queue_direct_start_policy": _dataset_queue_direct_start_policy,
            "_handoff_dataset_queue_start_fields": _handoff_dataset_queue_start_fields,
            "_handoff_labeler_route_authorization_fields": _handoff_labeler_route_authorization_fields,
            "_handoff_ready_to_send": _handoff_ready_to_send,
            "_handoff_sendability_actions": _handoff_sendability_actions,
            "_handoff_sendability_reasons": _handoff_sendability_reasons,
            "_handoff_sendability_summary": _handoff_sendability_summary,
            "_handoff_status_from_manifest": _handoff_status_from_manifest,
            "_known_labeler_status": _known_labeler_status,
            "_labeler_route_authorization_policy": _labeler_route_authorization_policy,
            "_labeler_route_authorization_runtime_checklist": _labeler_route_authorization_runtime_checklist,
            "_load_operator_evidence_template_from_directory": _load_operator_evidence_template_from_directory,
            "_manifest_relative_path": _manifest_relative_path,
            "_operator_authorization_policy": _operator_authorization_policy,
            "_operator_evidence_template_summary": _operator_evidence_template_summary,
            "_operator_validation_command_template_fields": _operator_validation_command_template_fields,
            "_operator_validation_command_templates": _operator_validation_command_templates,
            "_operator_validation_gate_flat_fields": _operator_validation_gate_flat_fields,
            "_operator_validation_invitation_fields": _operator_validation_invitation_fields,
            "_operator_validation_public_fields": _operator_validation_public_fields,
            "_operator_validation_visibility_policy": _operator_validation_visibility_policy,
            "_public_reassignment_session_safety_fields": _public_reassignment_session_safety_fields,
            "_read_jsonl_objects": _read_jsonl_objects,
            "_reassignment_session_safety_flat_fields": _reassignment_session_safety_flat_fields,
            "_recompute_validation_checklist_summary": _recompute_validation_checklist_summary,
            "_runtime_operator_validation_gate_cli_policy": _runtime_operator_validation_gate_cli_policy,
            "_runtime_operator_validation_gate_cli_policy_fields": _runtime_operator_validation_gate_cli_policy_fields,
            "_safe_share_checklist_gate_status_fields": _safe_share_checklist_gate_status_fields,
            "_safe_share_gate_flat_fields": _safe_share_gate_flat_fields,
            "_safe_share_gate_policy": _safe_share_gate_policy,
            "_single_owner_assignment_live_contract_fields": _single_owner_assignment_live_contract_fields,
            "_update_validation_checklist_payload": _update_validation_checklist_payload,
            "_validation_checklist_gate_summary": _validation_checklist_gate_summary,
            "_work_recording_ids": _work_recording_ids,
            "_write_user_handoff_html_index": _write_user_handoff_html_index,
            "_write_user_handoff_message": _write_user_handoff_message,
            "_write_user_handoff_quickstart": _write_user_handoff_quickstart,
            "_write_user_handoffs_html_index": _write_user_handoffs_html_index,
            "_write_user_handoffs_roster_csv": _write_user_handoffs_roster_csv,
        }
    )

def _refresh_user_handoff_visible_files(*args: object, **kwargs: object) -> object:
    _configure_handoff_validation_refresh_helpers()
    return _web_handoff_validation_refresh._refresh_user_handoff_visible_files_impl(*args, **kwargs)


def _handoff_manifest_paths_for_validation_checklist(*args: object, **kwargs: object) -> object:
    _configure_handoff_validation_refresh_helpers()
    return _web_handoff_validation_refresh._handoff_manifest_paths_for_validation_checklist_impl(*args, **kwargs)


def _refresh_handoff_manifests_from_validation_checklist(*args: object, **kwargs: object) -> object:
    _configure_handoff_validation_refresh_helpers()
    return _web_handoff_validation_refresh._refresh_handoff_manifests_from_validation_checklist_impl(*args, **kwargs)


def _apply_operator_evidence_templates_to_validation_checklist_file(*args: object, **kwargs: object) -> object:
    _configure_handoff_validation_refresh_helpers()
    return _web_handoff_validation_refresh._apply_operator_evidence_templates_to_validation_checklist_file_impl(*args, **kwargs)








def _write_launch_bundle_validation_log(manifest: dict[str, object], output_path: Path) -> None:
    _write_web_labeling_validation_log(manifest, output_path, bundle_label="launch bundle")


def _write_launch_bundle_validation_checklist(manifest: dict[str, object], output_path: Path) -> None:
    _write_web_labeling_validation_checklist(manifest, output_path, bundle_label="launch bundle")


def _write_user_handoffs_validation_log(index: dict[str, object], output_path: Path) -> None:
    _write_web_labeling_validation_log(index, output_path, bundle_label="multi-user handoff bundle")


def _write_user_handoffs_validation_checklist(index: dict[str, object], output_path: Path) -> None:
    _write_web_labeling_validation_checklist(index, output_path, bundle_label="multi-user handoff bundle")


def _write_user_handoff_validation_log(manifest: dict[str, object], output_path: Path) -> None:
    return _write_user_handoff_validation_log_impl(
        manifest,
        output_path,
        dependencies={
            '_write_web_labeling_validation_log': _write_web_labeling_validation_log
        },
    )


def _write_user_handoff_validation_checklist(manifest: dict[str, object], output_path: Path) -> None:
    return _write_user_handoff_validation_checklist_impl(
        manifest,
        output_path,
        dependencies={
            '_write_web_labeling_validation_checklist': _write_web_labeling_validation_checklist
        },
    )






















def _personalized_launch_readiness_target() -> dict[str, object]:
    fields = _personalized_launch_readiness_field_names()
    return {
        "personalized_launch_readiness_schema": (
            "palette.web_labeling_personalized_launch_readiness.v1"
        ),
        "personalized_launch_readiness_field": "personalized_launch_readiness",
        "personalized_launch_readiness_nested_work_field": (
            "work.personalized_launch_readiness"
        ),
        "personalized_launch_readiness_fields": fields,
        "personalized_launch_readiness_field_count": len(fields),
        "personalized_launch_readiness_roster_field_prefix": (
            "personalized_launch_readiness_"
        ),
        "personalized_launch_readiness_required_values": {
            "browser_label_write_target": "training_zarr",
            "browser_writes_csv_or_handoff_files": False,
            "browser_has_direct_zarr_write_authority": False,
        },
    }












def _check_launch_bundle_overwrite_target(
    *,
    output_dir: Path,
    expected_user_dirs: list[Path],
    include_audit_events: bool,
    overwrite: bool,
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Launch bundle output path exists and is not a directory: {output_dir}")
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return
    if not overwrite:
        raise FileExistsError(f"Refusing to write into existing non-empty launch bundle directory: {output_dir}")

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileExistsError(
            "Refusing to overwrite a non-empty directory that does not look like a launch bundle: "
            f"{output_dir}"
        )

    expected_names = {path.name for path in expected_user_dirs}
    handoffs_dir = output_dir / "handoffs"
    stale_dirs = []
    if handoffs_dir.exists():
        stale_dirs = [
            path
            for path in handoffs_dir.iterdir()
            if path.is_dir() and path.name not in expected_names
        ]
    if stale_dirs:
        raise FileExistsError(
            "Refusing to overwrite launch bundle with stale handoff directories: "
            + ", ".join(str(path) for path in stale_dirs)
        )

    audit_dir = output_dir / "audit"
    if audit_dir.exists() and not include_audit_events:
        raise FileExistsError(
            "Refusing to overwrite launch bundle without audit capture while stale audit directory exists: "
            f"{audit_dir}"
        )
    if audit_dir.exists() and include_audit_events:
        allowed_audit_files = {
            "task-events.jsonl",
            "assignment-events.jsonl",
            "task-definition-events.jsonl",
        }
        stale_audit_paths = [
            path
            for path in audit_dir.iterdir()
            if path.name not in allowed_audit_files
        ]
        if stale_audit_paths:
            raise FileExistsError(
                "Refusing to overwrite launch bundle with unexpected audit artifacts: "
                + ", ".join(str(path) for path in stale_audit_paths)
            )


def _parse_handoff_utc(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _handoff_status_from_manifest(
    manifest, now
) -> dict[str, object]:
    return _handoff_status_from_manifest_impl(
        manifest,
        now,
        dependencies={
            'PERSONAL_DATASET_QUEUE_PATH': PERSONAL_DATASET_QUEUE_PATH,
            'PERSONAL_WORK_PATH': PERSONAL_WORK_PATH,
            '_browser_mutation_target_contract_compact_fields': _browser_mutation_target_contract_compact_fields,
            '_browser_mutation_write_policy': _browser_mutation_write_policy,
            '_browser_mutation_write_runtime_checklist': _browser_mutation_write_runtime_checklist,
            '_browser_response_security_policy': _browser_response_security_policy,
            '_browser_signed_link_policy': _browser_signed_link_policy,
            '_browser_task_state_policy': _browser_task_state_policy,
            '_browser_workflow_capabilities': _browser_workflow_capabilities,
            '_dataset_queue_direct_start_policy': _dataset_queue_direct_start_policy,
            '_dataset_queue_direct_start_policy_fields': _dataset_queue_direct_start_policy_fields,
            '_direct_browser_start_contract_compact_fields': _direct_browser_start_contract_compact_fields,
            '_handoff_assignment_ownership_fields': _handoff_assignment_ownership_fields,
            '_handoff_browser_mutation_write_fields': _handoff_browser_mutation_write_fields,
            '_handoff_browser_response_security_fields': _handoff_browser_response_security_fields,
            '_handoff_dataset_queue_blocks_labeler_start': _handoff_dataset_queue_blocks_labeler_start,
            '_handoff_dataset_queue_start_fields': _handoff_dataset_queue_start_fields,
            '_handoff_dataset_queue_state': _handoff_dataset_queue_state,
            '_handoff_entry_artifact_fields': _handoff_entry_artifact_fields,
            '_handoff_known_user_status_fields': _handoff_known_user_status_fields,
            '_handoff_labeler_route_authorization_fields': _handoff_labeler_route_authorization_fields,
            '_handoff_labeler_safety_fields': _handoff_labeler_safety_fields,
            '_handoff_mutation_audit_fields': _handoff_mutation_audit_fields,
            '_handoff_operator_recovery_fields': _handoff_operator_recovery_fields,
            '_handoff_ready_to_send': _handoff_ready_to_send,
            '_handoff_sendability_actions': _handoff_sendability_actions,
            '_handoff_sendability_reasons': _handoff_sendability_reasons,
            '_handoff_session_guard_fields': _handoff_session_guard_fields,
            '_handoff_signed_link_policy_fields': _handoff_signed_link_policy_fields,
            '_handoff_task_state_policy_fields': _handoff_task_state_policy_fields,
            '_handoff_zarr_backup_fields': _handoff_zarr_backup_fields,
            '_labeler_route_authorization_policy': _labeler_route_authorization_policy,
            '_labeler_route_authorization_runtime_checklist': _labeler_route_authorization_runtime_checklist,
            '_labeler_safety_policy': _labeler_safety_policy,
            '_labeler_work_completion_fields': _labeler_work_completion_fields,
            '_mutation_audit_policy': _mutation_audit_policy,
            '_parse_handoff_utc': _parse_handoff_utc,
            '_public_reassignment_session_safety_fields': _public_reassignment_session_safety_fields,
            '_reassignment_session_safety_flat_fields': _reassignment_session_safety_flat_fields,
            '_recordings_without_open_tasks_actions': _recordings_without_open_tasks_actions,
            '_runtime_operator_validation_gate_cli_policy': _runtime_operator_validation_gate_cli_policy,
            '_runtime_operator_validation_gate_cli_policy_fields': _runtime_operator_validation_gate_cli_policy_fields,
            '_session_guard_policy': _session_guard_policy,
            '_zarr_backup_policy': _zarr_backup_policy
        },
    )




def _handoff_dataset_queue_state_counts(handoffs: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return _web_assignment_freshness._handoff_dataset_queue_state_counts_impl(handoffs)


def _handoff_assignment_snapshot_from_work(work: Mapping[str, object], user: str) -> dict[str, object]:
    return _web_assignment_freshness._handoff_assignment_snapshot_from_work_impl(work, user)


def _assignment_snapshot_rows(snapshot: Mapping[str, object], fallback_user: str) -> list[dict[str, object]]:
    return _web_assignment_freshness._assignment_snapshot_rows_impl(snapshot, fallback_user)


def _inspect_handoff_assignment_freshness(
    manifests: Sequence[Mapping[str, object]],
    *,
    store: LabelingStore | None,
    package_kind: str = "",
) -> dict[str, object]:
    return _web_assignment_freshness._inspect_handoff_assignment_freshness_impl(
        manifests,
        store=store,
        package_kind=package_kind,
    )





def _assignment_snapshot_from_assignments(
    assignments: Sequence[Mapping[str, object]],
    *,
    user: str | None = None,
) -> dict[str, object]:
    rows_by_recording: dict[str, dict[str, object]] = {}
    for assignment in assignments:
        recording_id = str(assignment.get("recording_id") or "").strip()
        if not recording_id:
            continue
        rows_by_recording[recording_id] = {
            "recording_id": recording_id,
            "assignee_user": str(assignment.get("assignee_user") or "").strip(),
            "status": str(assignment.get("status") or "active").strip() or "active",
        }
    rows = [rows_by_recording[key] for key in sorted(rows_by_recording)]
    snapshot: dict[str, object] = {
        "schema": "palette.web_labeling_assignment_snapshot.v1",
        "recording_count": len(rows),
        "recording_ids": [str(row["recording_id"]) for row in rows],
        "assignments": rows,
    }
    if user is not None:
        snapshot["user"] = str(user)
    return snapshot






def _audit_row_timestamp(row: dict[str, object]) -> datetime | None:
    for key in ("created_at_utc", "timestamp_utc", "event_time_utc", "assigned_at_utc", "updated_at_utc"):
        parsed = _parse_handoff_utc(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _filter_audit_rows(
    rows: list[dict[str, object]],
    *,
    since_utc: str | None,
    until_utc: str | None,
    limit: int | None,
) -> list[dict[str, object]]:
    since = _parse_handoff_utc(since_utc)
    until = _parse_handoff_utc(until_utc)
    filtered: list[dict[str, object]] = []
    for row in rows:
        timestamp = _audit_row_timestamp(row)
        if since is not None and (timestamp is None or timestamp < since):
            continue
        if until is not None and (timestamp is None or timestamp > until):
            continue
        filtered.append(row)
    if limit is not None and limit >= 0:
        return filtered[:limit]
    return filtered


def _write_jsonl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
























OPERATOR_EVIDENCE_TEMPLATE_FIELDS: dict[str, str] = {
    "identity_probe_verification": "identity_source_evidence_template",
    "browser_response_security_headers": "browser_response_security_evidence_template",
    "browser_smoke": "browser_smoke_evidence_template",
    "disposable_zarr_mutation_smoke": "disposable_zarr_mutation_smoke_evidence_template",
    "mutable_zarr_backup_confirmation": "zarr_backup_evidence_template",
}


def _operator_evidence_template_status(
    *,
    gate_id: str,
    template_path: str,
    template: Mapping[str, object] | None,
    present: bool,
    valid: bool,
    error: str = "",
) -> dict[str, object]:
    return _operator_evidence_template_status_impl(
        gate_id=gate_id,
        template_path=template_path,
        template=template,
        present=present,
        valid=valid,
        error=error,
        dependencies={
            'BROWSER_SMOKE_REQUIRED_FIELDS': BROWSER_SMOKE_REQUIRED_FIELDS,
            'DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS': DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS,
            '_browser_smoke_personalized_route_contract': _browser_smoke_personalized_route_contract,
            '_disposable_zarr_smoke_workflow_contract_missing_fields': _disposable_zarr_smoke_workflow_contract_missing_fields,
            '_expected_user_query_value_from_url': _expected_user_query_value_from_url,
            '_identity_source_personal_queue_status': _identity_source_personal_queue_status,
            '_identity_source_row_approved': _identity_source_row_approved,
        },
    )


def _operator_evidence_template_summary(
    payload: Mapping[str, object],
    *,
    load_template,
) -> dict[str, object]:
    return _operator_evidence_template_summary_impl(
        payload,
        load_template=load_template,
        dependencies={
            'OPERATOR_EVIDENCE_TEMPLATE_FIELDS': OPERATOR_EVIDENCE_TEMPLATE_FIELDS,
            '_identity_personal_queue_evidence_status': _identity_personal_queue_evidence_status,
            'BROWSER_SMOKE_REQUIRED_FIELDS': BROWSER_SMOKE_REQUIRED_FIELDS,
            'DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS': DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS,
            '_browser_smoke_personalized_route_contract': _browser_smoke_personalized_route_contract,
            '_disposable_zarr_smoke_workflow_contract_missing_fields': _disposable_zarr_smoke_workflow_contract_missing_fields,
            '_expected_user_query_value_from_url': _expected_user_query_value_from_url,
            '_identity_source_personal_queue_status': _identity_source_personal_queue_status,
            '_identity_source_row_approved': _identity_source_row_approved,
        },
    )






def _inspect_handoff_validation_checklist(path: Path) -> dict[str, object]:
    return _inspect_handoff_validation_checklist_impl(
        path,
        dependencies={
            '_implementation_status_artifact_name': _implementation_status_artifact_name,
            '_implementation_status_artifact_summary': _implementation_status_artifact_summary,
            '_operator_validation_command_templates': _operator_validation_command_templates,
            '_operator_validation_visibility_policy': _operator_validation_visibility_policy,
            '_validation_checklist_gate_summary': _validation_checklist_gate_summary,
            'OPERATOR_EVIDENCE_TEMPLATE_FIELDS': OPERATOR_EVIDENCE_TEMPLATE_FIELDS,
            '_identity_personal_queue_evidence_status': _identity_personal_queue_evidence_status,
            'BROWSER_SMOKE_REQUIRED_FIELDS': BROWSER_SMOKE_REQUIRED_FIELDS,
            'DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS': DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS,
            '_browser_smoke_personalized_route_contract': _browser_smoke_personalized_route_contract,
            '_disposable_zarr_smoke_workflow_contract_missing_fields': _disposable_zarr_smoke_workflow_contract_missing_fields,
            '_expected_user_query_value_from_url': _expected_user_query_value_from_url,
            '_identity_source_personal_queue_status': _identity_source_personal_queue_status,
            '_identity_source_row_approved': _identity_source_row_approved,
        },
    )






















def _inspect_handoff_package(
    path: Path,
    *,
    store: LabelingStore | None = None,
    require_shareable: bool = False
) -> dict[str, object]:
    return _inspect_handoff_package_impl(
        path,
        store=store,
        require_shareable=require_shareable,
        dependencies={
            '_handoff_status_from_manifest': _handoff_status_from_manifest,
            'DEFAULT_OPERATOR_VALIDATION_GATE_IDS': DEFAULT_OPERATOR_VALIDATION_GATE_IDS,
            'OPERATOR_VALIDATION_GATE_STATUS_VALUES': OPERATOR_VALIDATION_GATE_STATUS_VALUES,
            '_IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS': _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS,
            '_IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT': _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT,
            '_IMPLEMENTATION_STATUS_FLAT_FIELDS': _IMPLEMENTATION_STATUS_FLAT_FIELDS,
            '_IMPLEMENTATION_STATUS_FLAT_FIELD_COUNT': _IMPLEMENTATION_STATUS_FLAT_FIELD_COUNT,
            '_browser_mutation_target_contract_summary': _browser_mutation_target_contract_summary,
            '_count_handoff_sendability_reasons': _count_handoff_sendability_reasons,
            '_direct_browser_start_contract_summary': _direct_browser_start_contract_summary,
            '_handoff_dataset_queue_state_counts': _handoff_dataset_queue_state_counts,
            '_identity_personal_queue_evidence_status': _identity_personal_queue_evidence_status,
            '_implementation_status_artifact': _implementation_status_artifact,
            '_inspect_handoff_assignment_freshness': _inspect_handoff_assignment_freshness,
            '_inspect_handoff_launch_evidence_execution_checklist': _inspect_handoff_launch_evidence_execution_checklist,
            '_inspect_handoff_operator_evidence_commands': _inspect_handoff_operator_evidence_commands,
            '_inspect_handoff_validation_checklist': _inspect_handoff_validation_checklist,
            '_inspect_handoff_validation_log': _inspect_handoff_validation_log,
            '_inspection_failure_actions': _inspection_failure_actions,
            '_inspection_labeler_entrypoint_summary': _inspection_labeler_entrypoint_summary,
            '_inspection_operator_repair_commands': _inspection_operator_repair_commands,
            '_labeler_route_authorization_runtime_checklist_contract_summary': _labeler_route_authorization_runtime_checklist_contract_summary,
            '_launch_evidence_execution_checklist_public_summary': _launch_evidence_execution_checklist_public_summary,
            '_load_handoff_documents': _load_handoff_documents,
            '_operator_evidence_commands_public_summary': _operator_evidence_commands_public_summary,
            '_parse_handoff_utc': _parse_handoff_utc,
            '_recordings_without_open_tasks_actions': _recordings_without_open_tasks_actions,
            '_safe_share_checklist_gate_status_fields': _safe_share_checklist_gate_status_fields,
            '_safe_share_external_launch_evidence_gap_field_names': _safe_share_external_launch_evidence_gap_field_names,
            '_safe_share_gate_flat_fields': _safe_share_gate_flat_fields,
            '_safe_share_gate_policy': _safe_share_gate_policy,
            '_safe_share_next_action_command_fields': _safe_share_next_action_command_fields,
            '_safe_share_next_action_detail_fields': _safe_share_next_action_detail_fields,
            '_shareability_compact_contract_fields': _shareability_compact_contract_fields,
            '_shareability_compact_contract_source_fields': _shareability_compact_contract_source_fields,
            '_shareability_labeler_route_authorization_runtime_checklist_gate': _shareability_labeler_route_authorization_runtime_checklist_gate,
            '_shareability_repair_command_contracts': _shareability_repair_command_contracts,
            '_shareability_repair_command_detail_fields': _shareability_repair_command_detail_fields,
            '_shareability_repair_command_detail_fields_by_id': _shareability_repair_command_detail_fields_by_id,
            '_shareability_safe_to_share_requires': _shareability_safe_to_share_requires,
            '_single_owner_package_contract_summary': _single_owner_package_contract_summary,
            '_sum_recordings_without_open_tasks_by_reason': _sum_recordings_without_open_tasks_by_reason,
            '_verify_handoff_checksums': _verify_handoff_checksums
        },
    )





def _configure_batch_readiness_helpers() -> None:
    _web_batch_readiness.configure_batch_readiness_dependencies(
        {
            "_assignment_ownership_policy": _assignment_ownership_policy,
            "_recordings_without_open_tasks_actions": _recordings_without_open_tasks_actions,
            "_store_consistency_report": _store_consistency_report,
        }
    )


def _batch_readiness_report(store: LabelingStore) -> dict[str, object]:
    _configure_batch_readiness_helpers()
    return _web_batch_readiness._batch_readiness_report_impl(store)


def _apply_readiness_warning_policy(report: dict[str, object], *, warnings_as_errors: bool) -> dict[str, object]:
    readiness_warning_count = int(report.get("readiness_warning_count") or 0)
    store_consistency = report.get("store_consistency") if isinstance(report.get("store_consistency"), dict) else {}
    store_warning_count = int(store_consistency.get("warning_count") or 0)
    total_warning_count = readiness_warning_count + store_warning_count
    readiness_warning_codes = [
        str(row.get("code") or "")
        for row in report.get("readiness_warnings", [])
        if isinstance(row, dict) and str(row.get("code") or "")
    ]
    store_warning_codes = [
        str(row.get("code") or "")
        for row in store_consistency.get("warnings", [])
        if isinstance(row, dict) and str(row.get("code") or "")
    ]
    report["warnings_as_errors"] = bool(warnings_as_errors)
    report["warning_count"] = total_warning_count
    report["blocking_warning_count"] = total_warning_count if warnings_as_errors else 0
    report["blocking_warning_codes"] = sorted(set(readiness_warning_codes + store_warning_codes)) if warnings_as_errors else []
    if warnings_as_errors and total_warning_count:
        report["ok"] = False
    return report


def _write_user_handoff_bundle(
    *,
    store: LabelingStore,
    store_path: Path,
    user: str,
    output_dir: Path,
    secret: str,
    base_url: str | None,
    ttl_seconds: int,
    include_completed: bool,
    overwrite: bool,
    zip_output: Path | None = None,
) -> dict[str, object]:
    return _write_user_handoff_bundle_impl(
        store=store,
        store_path=store_path,
        user=user,
        output_dir=output_dir,
        secret=secret,
        base_url=base_url,
        ttl_seconds=ttl_seconds,
        include_completed=include_completed,
        overwrite=overwrite,
        zip_output=zip_output,
        dependencies={
            'DASHBOARD_PATH': DASHBOARD_PATH,
            'DATASET_QUEUE_PATH': DATASET_QUEUE_PATH,
            'LABELER_START_TASK_STATES': LABELER_START_TASK_STATES,
            'LABELING_HOME_PATH': LABELING_HOME_PATH,
            'PERSONAL_DATASET_QUEUE_PATH': PERSONAL_DATASET_QUEUE_PATH,
            'PERSONAL_WORK_PATH': PERSONAL_WORK_PATH,
            '_add_direct_start_contracts_to_work_tasks': _add_direct_start_contracts_to_work_tasks,
            '_add_payload_contract_compact_fields': _add_payload_contract_compact_fields,
            '_add_work_summary_fields': _add_work_summary_fields,
            '_assignment_ownership_contract_fields': _assignment_ownership_contract_fields,
            '_assignment_ownership_contract_policy': _assignment_ownership_contract_policy,
            '_assignment_ownership_integrity': _assignment_ownership_integrity,
            '_assignment_ownership_policy': _assignment_ownership_policy,
            '_browser_mutation_write_policy': _browser_mutation_write_policy,
            '_browser_mutation_write_runtime_checklist': _browser_mutation_write_runtime_checklist,
            '_browser_response_security_policy': _browser_response_security_policy,
            '_browser_signed_link_policy': _browser_signed_link_policy,
            '_browser_task_state_policy': _browser_task_state_policy,
            '_browser_workflow_capabilities': _browser_workflow_capabilities,
            '_check_directory_zip_output': _check_directory_zip_output,
            '_count_recordings_without_open_tasks': _count_recordings_without_open_tasks,
            '_count_recordings_without_open_tasks_by_reason': _count_recordings_without_open_tasks_by_reason,
            '_count_redacted_summary_fields': _count_redacted_summary_fields,
            '_dashboard_operator_validation_fields': _dashboard_operator_validation_fields,
            '_dashboard_url_for_base': _dashboard_url_for_base,
            '_dashboard_url_for_expected_user': _dashboard_url_for_expected_user,
            '_dataset_queue_direct_start_policy': _dataset_queue_direct_start_policy,
            '_dataset_queue_state': _dataset_queue_state,
            '_dataset_queue_url_for_base': _dataset_queue_url_for_base,
            '_effective_signed_link_ttl_seconds': _effective_signed_link_ttl_seconds,
            '_first_dataset_queue_url': _first_dataset_queue_url,
            '_handoff_assignment_snapshot_from_work': _handoff_assignment_snapshot_from_work,
            '_handoff_browser_mutation_write_fields': _handoff_browser_mutation_write_fields,
            '_handoff_browser_response_security_fields': _handoff_browser_response_security_fields,
            '_handoff_entry_artifact_fields': _handoff_entry_artifact_fields,
            '_handoff_labeler_route_authorization_fields': _handoff_labeler_route_authorization_fields,
            '_handoff_labeler_safety_fields': _handoff_labeler_safety_fields,
            '_handoff_mutation_audit_fields': _handoff_mutation_audit_fields,
            '_handoff_operator_recovery_fields': _handoff_operator_recovery_fields,
            '_handoff_ready_to_send': _handoff_ready_to_send,
            '_handoff_sendability_actions': _handoff_sendability_actions,
            '_handoff_sendability_reasons': _handoff_sendability_reasons,
            '_handoff_sendability_summary': _handoff_sendability_summary,
            '_handoff_session_guard_fields': _handoff_session_guard_fields,
            '_handoff_signed_link_policy_fields': _handoff_signed_link_policy_fields,
            '_handoff_store_checks_ok_for_user': _handoff_store_checks_ok_for_user,
            '_handoff_task_state_policy_fields': _handoff_task_state_policy_fields,
            '_handoff_zarr_backup_fields': _handoff_zarr_backup_fields,
            '_identity_probe_url_for_base': _identity_probe_url_for_base,
            '_known_labeler_status': _known_labeler_status,
            '_labeler_landing_url_for_base': _labeler_landing_url_for_base,
            '_labeler_route_authorization_policy': _labeler_route_authorization_policy,
            '_labeler_route_authorization_runtime_checklist': _labeler_route_authorization_runtime_checklist,
            '_labeler_safety_policy': _labeler_safety_policy,
            '_labeler_work_completion_fields': _labeler_work_completion_fields,
            '_labeling_home_url_for_base': _labeling_home_url_for_base,
            '_mutation_audit_policy': _mutation_audit_policy,
            '_operator_authorization_policy': _operator_authorization_policy,
            '_operator_recovery_policy': _operator_recovery_policy,
            '_operator_validation_command_templates': _operator_validation_command_templates,
            '_operator_validation_gate_flat_fields': _operator_validation_gate_flat_fields,
            '_operator_validation_invitation_fields': _operator_validation_invitation_fields,
            '_operator_validation_public_fields': _operator_validation_public_fields,
            '_operator_validation_visibility_policy': _operator_validation_visibility_policy,
            '_personalized_launch_readiness_summary': _personalized_launch_readiness_summary,
            '_public_reassignment_session_safety_fields': _public_reassignment_session_safety_fields,
            '_queue_first_entry_contract_policy': _queue_first_entry_contract_policy,
            '_reassignment_session_safety_flat_fields': _reassignment_session_safety_flat_fields,
            '_recordings_without_open_tasks_actions': _recordings_without_open_tasks_actions,
            '_runtime_operator_validation_gate_cli_policy': _runtime_operator_validation_gate_cli_policy,
            '_runtime_operator_validation_gate_cli_policy_fields': _runtime_operator_validation_gate_cli_policy_fields,
            '_safe_share_checklist_gate_status_fields_from_operator_validation': _safe_share_checklist_gate_status_fields_from_operator_validation,
            '_safe_share_gate_flat_fields': _safe_share_gate_flat_fields,
            '_safe_share_gate_policy': _safe_share_gate_policy,
            '_session_guard_policy': _session_guard_policy,
            '_signed_task_link_token_info': _signed_task_link_token_info,
            '_store_consistency_report': _store_consistency_report,
            '_user_handoff_paths': _user_handoff_paths,
            '_web_labeling_validation_checklist_payload': _web_labeling_validation_checklist_payload,
            '_work_dataset_queue_summary': _work_dataset_queue_summary,
            '_work_progress_summary': _work_progress_summary,
            '_work_recording_ids': _work_recording_ids,
            '_write_directory_zip': _write_directory_zip,
            '_write_user_handoff_html_index': _write_user_handoff_html_index,
            '_write_user_handoff_message': _write_user_handoff_message,
            '_write_user_handoff_quickstart': _write_user_handoff_quickstart,
            '_write_user_handoff_validation_checklist': _write_user_handoff_validation_checklist,
            '_write_user_handoff_validation_log': _write_user_handoff_validation_log,
            '_zarr_backup_policy': _zarr_backup_policy
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Labeling sidecar SQLite path. Defaults to PALETTE_LABELING_STORE_PATH or ~/.palette/labeling_work.sqlite.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create or migrate the labeling store schema.")

    backup = sub.add_parser("backup-store", help="Write a SQLite-consistent backup of the labeling store.")
    backup.add_argument("--output", required=True, help="Destination SQLite backup path.")
    backup.add_argument("--overwrite", action="store_true", help="Replace an existing backup destination.")

    users_list = sub.add_parser("users-list", help="List SQLite-backed labeling users.")
    users_list.add_argument("--status", choices=LABELING_USER_STATUSES, default=None)
    users_list.add_argument("--role", choices=LABELING_USER_ROLES, default=None)

    users_add = sub.add_parser("users-add", help="Add or update a SQLite-backed labeling user.")
    users_add.add_argument("user_id")
    users_add.add_argument("--display-name", default=None)
    users_add.add_argument("--email", default=None)
    users_add.add_argument("--role", choices=LABELING_USER_ROLES, default="labeler")
    users_add.add_argument("--status", choices=LABELING_USER_STATUSES, default="active")
    users_add.add_argument("--notes", default=None)
    users_add.add_argument("--actor", default=None, help="Operator/user recorded on the user event.")
    users_add.add_argument("--notify", action="store_true", help="Send or queue a labeler-added email notification.")
    users_add.add_argument("--notification-mode", choices=NOTIFICATION_MODES, default=None, help="Override PALETTE_LABELING_NOTIFICATION_MODE for this notification.")
    users_add.add_argument("--notification-base-url", default=None, help="Absolute labeling service URL to include in notification links.")

    users_activate = sub.add_parser("users-activate", help="Mark a labeling user active.")
    users_activate.add_argument("user_id")
    users_activate.add_argument("--actor", default=None, help="Operator/user recorded on the user event.")
    users_activate.add_argument("--notes", default=None)

    users_deactivate = sub.add_parser("users-deactivate", help="Mark a labeling user inactive.")
    users_deactivate.add_argument("user_id")
    users_deactivate.add_argument("--actor", default=None, help="Operator/user recorded on the user event.")
    users_deactivate.add_argument("--notes", default=None)

    zarr_backup_plan = sub.add_parser("zarr-backup-plan", help="Write a read-only operator plan for mutable Zarr backups required by assigned tasks.")
    zarr_backup_plan.add_argument("--recording-id", default=None, help="Restrict to one recording.")
    zarr_backup_plan.add_argument("--user", default=None, help="Restrict to recordings currently assigned to this user.")
    zarr_backup_plan.add_argument("--include-completed", action="store_true", help="Include completed tasks in the backup plan.")
    zarr_backup_plan.add_argument("--include-inactive", action="store_true", help="Include tasks under inactive/paused assignments.")
    zarr_backup_plan.add_argument("--output", default=None, help="Optional JSON output path for archiving the Zarr backup plan.")
    zarr_backup_plan.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    execute_zarr_backup = sub.add_parser(
        "execute-zarr-backup-plan",
        help="Operator-only copy execution for a Zarr backup plan.",
    )
    execute_zarr_backup.add_argument("--plan", required=True, help="Path to zarr-backup-plan.json.")
    execute_zarr_backup.add_argument("--backup-dir", required=True, help="Directory where Zarr backups and manifests are written.")
    execute_zarr_backup.add_argument("--operator", required=True, help="Operator recorded on backup manifests.")
    execute_zarr_backup.add_argument("--output", default=None, help="Optional execution manifest path. Defaults under --backup-dir.")
    execute_zarr_backup.add_argument("--overwrite", action="store_true", help="Allow replacing existing backup destinations and manifest files.")
    execute_zarr_backup.add_argument("--dry-run", action="store_true", help="Plan backup copy destinations without writing backup files.")
    execute_zarr_backup.add_argument("--allow-missing", action="store_true", help="Return success even when some source Zarr paths are missing.")

    restore_zarr_backup = sub.add_parser(
        "restore-zarr-backup",
        help="Operator-only restore from a Zarr backup execution manifest.",
    )
    restore_zarr_backup.add_argument("--manifest", required=True, help="Path to zarr-backup-execution-manifest.json.")
    restore_zarr_backup.add_argument("--target-index", type=int, action="append", default=None, help="Target index to restore. May be repeated.")
    restore_zarr_backup.add_argument("--all-targets", action="store_true", help="Restore every target in the execution manifest.")
    restore_zarr_backup.add_argument("--operator", required=True, help="Operator recorded on the restore report.")
    restore_zarr_backup.add_argument("--replace-current", action="store_true", help="Move any existing Zarr path aside before restoring.")
    restore_zarr_backup.add_argument("--allow-active-assignment", action="store_true", help="Override the default active-assignment restore block.")
    restore_zarr_backup.add_argument("--output", default=None, help="Optional JSON output path for archiving the restore report.")
    restore_zarr_backup.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    record_zarr_backup_evidence = sub.add_parser(
        "record-zarr-backup-evidence",
        help="Record a backup execution manifest into zarr-backup-evidence-template.json.",
    )
    record_zarr_backup_evidence.add_argument("--evidence", required=True, help="Path to zarr-backup-evidence-template.json.")
    record_zarr_backup_evidence.add_argument("--execution-manifest", required=True, help="Path to zarr-backup-execution-manifest.json.")
    record_zarr_backup_evidence.add_argument("--target-index", type=int, action="append", default=None, help="Target index to approve. May be repeated.")
    record_zarr_backup_evidence.add_argument("--all-targets", action="store_true", help="Record every backed-up target in the execution manifest.")
    record_zarr_backup_evidence.add_argument("--restore-test-result", required=True, help="Restore drill result or approved known-good regeneration note.")
    record_zarr_backup_evidence.add_argument("--operator", required=True, help="Operator approving the backup evidence.")
    record_zarr_backup_evidence.add_argument("--notes", default=None)
    record_zarr_backup_evidence.add_argument("--output", default=None, help="Optional output path. Defaults to updating --evidence in place.")
    record_zarr_backup_evidence.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    record_identity_source_evidence = sub.add_parser(
        "record-identity-source-evidence",
        help="Record deployed identity and personal /my-datasets queue evidence into identity-source-evidence-template.json.",
    )
    record_identity_source_evidence.add_argument("--evidence", required=True, help="Path to identity-source-evidence-template.json.")
    record_identity_source_evidence.add_argument("--expected-user", required=True, help="Expected assignment user from the template.")
    record_identity_source_evidence.add_argument("--resolved-user", required=True, help="User reported by the deployed Palette identity probe.")
    record_identity_source_evidence.add_argument("--operator", required=True, help="Operator approving the identity and guarded personal queue evidence.")
    record_identity_source_evidence.add_argument("--authenticated-session-context", default=None, help="Deployment/auth note confirming the identity probe and guarded /my-datasets personal queue URL were checked.")
    record_identity_source_evidence.add_argument("--notes", default=None)
    record_identity_source_evidence.add_argument("--output", default=None, help="Optional output path. Defaults to updating --evidence in place.")
    record_identity_source_evidence.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    record_browser_response_security_evidence = sub.add_parser(
        "record-browser-response-security-evidence",
        help="Record deployed response headers into browser-response-security-evidence-template.json.",
    )
    record_browser_response_security_evidence.add_argument("--evidence", required=True, help="Path to browser-response-security-evidence-template.json.")
    record_browser_response_security_evidence.add_argument("--header", action="append", default=[], help="Captured response header as NAME=VALUE. May be repeated.")
    record_browser_response_security_evidence.add_argument("--operator", required=True, help="Operator approving the response-security evidence.")
    record_browser_response_security_evidence.add_argument("--capture-url", default=None)
    record_browser_response_security_evidence.add_argument("--authenticated-test-user", default=None)
    record_browser_response_security_evidence.add_argument("--capture-note", default=None)
    record_browser_response_security_evidence.add_argument("--proxy-or-deployment", default=None)
    record_browser_response_security_evidence.add_argument("--notes", default=None)
    record_browser_response_security_evidence.add_argument("--output", default=None, help="Optional output path. Defaults to updating --evidence in place.")
    record_browser_response_security_evidence.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    record_browser_smoke_evidence = sub.add_parser(
        "record-browser-smoke-evidence",
        help="Record a representative queue-first browser smoke run into browser-smoke-evidence-template.json.",
    )
    record_browser_smoke_evidence.add_argument("--evidence", required=True, help="Path to browser-smoke-evidence-template.json.")
    record_browser_smoke_evidence.add_argument("--expected-user", required=True)
    record_browser_smoke_evidence.add_argument("--resolved-user", required=True)
    record_browser_smoke_evidence.add_argument("--operator", required=True)
    record_browser_smoke_evidence.add_argument("--browser-only-runtime-verified", action="store_true")
    record_browser_smoke_evidence.add_argument("--no-local-palette-install-verified", action="store_true")
    record_browser_smoke_evidence.add_argument("--no-local-crimson-install-verified", action="store_true")
    record_browser_smoke_evidence.add_argument("--no-local-conda-or-project-dependencies-verified", action="store_true")
    record_browser_smoke_evidence.add_argument("--personalized-dataset-queue-verified", action="store_true")
    record_browser_smoke_evidence.add_argument("--preferred-labeler-entry-url-matches-personal-dataset-queue", action="store_true")
    record_browser_smoke_evidence.add_argument("--personalized-labeler-entry-url-matches-personal-dataset-queue", action="store_true")
    record_browser_smoke_evidence.add_argument("--personalized-work-dashboard-verified", action="store_true")
    record_browser_smoke_evidence.add_argument("--labeler-sees-only-assigned-work", action="store_true")
    record_browser_smoke_evidence.add_argument("--support-text-redacted", action="store_true")
    record_browser_smoke_evidence.add_argument("--expected-user-mismatch-rejected", action="store_true")
    record_browser_smoke_evidence.add_argument("--task-opened", action="store_true")
    record_browser_smoke_evidence.add_argument("--induced-failure-support-detail-redacted", action="store_true")
    record_browser_smoke_evidence.add_argument("--completion-verified", action="store_true")
    record_browser_smoke_evidence.add_argument("--completed-task-read-only-verified", action="store_true")
    record_browser_smoke_evidence.add_argument("--stale-tab-save-rejected", action="store_true")
    record_browser_smoke_evidence.add_argument("--operator-reopen-verified", action="store_true")
    record_browser_smoke_evidence.add_argument("--notes", default=None)
    record_browser_smoke_evidence.add_argument("--output", default=None, help="Optional output path. Defaults to updating --evidence in place.")
    record_browser_smoke_evidence.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    record_disposable_zarr_smoke_evidence = sub.add_parser(
        "record-disposable-zarr-mutation-smoke-evidence",
        help="Record disposable-Zarr mutation smoke evidence for one workflow kind.",
    )
    record_disposable_zarr_smoke_evidence.add_argument("--evidence", required=True, help="Path to disposable-zarr-mutation-smoke-evidence-template.json.")
    record_disposable_zarr_smoke_evidence.add_argument("--workflow-kind", required=True)
    record_disposable_zarr_smoke_evidence.add_argument("--operator", required=True)
    record_disposable_zarr_smoke_evidence.add_argument("--mutation-event-id", action="append", default=[])
    record_disposable_zarr_smoke_evidence.add_argument(
        "--event-lookup-report",
        action="append",
        default=[],
        help="Path to a lookup-event JSON report proving a mutation event ID was resolved by an operator.",
    )
    record_disposable_zarr_smoke_evidence.add_argument("--registry-refresh-event-id", action="append", default=[])
    record_disposable_zarr_smoke_evidence.add_argument("--disposable-recording-id", default=None)
    record_disposable_zarr_smoke_evidence.add_argument("--disposable-task-id", default=None)
    record_disposable_zarr_smoke_evidence.add_argument("--labeler-user", default=None, help="Expected labeler user for lookup-event report context verification.")
    record_disposable_zarr_smoke_evidence.add_argument("--disposable-zarr-or-known-good-source", default=None)
    record_disposable_zarr_smoke_evidence.add_argument("--backup-or-regeneration-verified", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument("--server-write-scope-verified", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument("--task-scoped-training-zarr-write-verified", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument("--browser-no-direct-zarr-write-authority-verified", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument("--handoff-artifacts-metadata-only-verified", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument("--browser-no-csv-or-handoff-write-verified", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument("--client-target-selector-rejection-verified", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument("--audit-event-verified", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument("--operator-event-lookup-verified", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument("--completion-verified", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument("--stale-tab-save-rejected", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument("--bad-mutation-recovery-verified", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument(
        "--bad-mutation-recovery-mode",
        choices=("restore_backup", "regenerate_known_good", "discard_disposable"),
        default=None,
        help="Operator-verified recovery route for a bad disposable mutation.",
    )
    record_disposable_zarr_smoke_evidence.add_argument(
        "--bad-mutation-recovery-report",
        default=None,
        help="Path or note for an archived restore report, regeneration recipe, or disposable discard evidence.",
    )
    record_disposable_zarr_smoke_evidence.add_argument("--restored-or-discarded", action="store_true")
    record_disposable_zarr_smoke_evidence.add_argument("--notes", default=None)
    record_disposable_zarr_smoke_evidence.add_argument("--output", default=None, help="Optional output path. Defaults to updating --evidence in place.")
    record_disposable_zarr_smoke_evidence.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    assign = sub.add_parser("assign", help="Assign one recording to one user.")
    assign.add_argument("--recording-id", required=True)
    assign.add_argument("--user", required=True)
    assign.add_argument("--assigned-by", default=None)
    assign.add_argument("--status", default="active")
    assign.add_argument("--notes", default=None)
    assign.add_argument("--output", default=None, help="Optional JSON output path for archiving the assignment report.")
    assign.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")
    assign.add_argument("--notify", action="store_true", help="Send or queue a dataset-available email notification for the assignee.")
    assign.add_argument("--notification-mode", choices=NOTIFICATION_MODES, default=None, help="Override PALETTE_LABELING_NOTIFICATION_MODE for this notification.")
    assign.add_argument("--notification-base-url", default=None, help="Absolute labeling service URL to include in notification links.")

    import_assignments = sub.add_parser("import-assignments", help="Dry-run or apply a JSON/JSONL recording assignment manifest.")
    import_assignments.add_argument("--input", required=True, help="JSON list, JSONL file, or JSON object with assignments list.")
    import_assignments.add_argument("--assigned-by", default=None, help="Default assigned_by value for rows that omit it.")
    import_assignments.add_argument("--apply", action="store_true", help="Apply the assignment changes. Defaults to dry-run.")
    import_assignments.add_argument("--warnings-as-errors", action="store_true", help="Return nonzero and do not apply when assignment import warnings are present.")
    import_assignments.add_argument("--output", default=None, help="Optional JSON output path for archiving the assignment import report.")
    import_assignments.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")
    import_assignments.add_argument("--notify", action="store_true", help="Send or queue dataset-available email notifications for applied active assignments.")
    import_assignments.add_argument("--notification-mode", choices=NOTIFICATION_MODES, default=None, help="Override PALETTE_LABELING_NOTIFICATION_MODE for these notifications.")
    import_assignments.add_argument("--notification-base-url", default=None, help="Absolute labeling service URL to include in notification links.")

    task = sub.add_parser("add-task", help="Create or update a labeling task.")
    task.add_argument("--task-id", default=None)
    task.add_argument("--recording-id", required=True)
    task.add_argument("--workflow-kind", required=True)
    task.add_argument("--dataset-id", default=None)
    task.add_argument("--zarr-use", default=None)
    task.add_argument("--stage-group", default=None)
    task.add_argument("--run-name", default=None)
    task.add_argument("--component-name", default=None)
    task.add_argument("--title", default=None)
    task.add_argument("--scope-json", default=None, help="Inline JSON object/list or path to a JSON file.")
    task.add_argument("--state", default="pending")
    task.add_argument("--priority", type=int, default=0)
    task.add_argument("--notes", default=None)
    task.add_argument("--actor", default=None, help="Operator/user recorded on task-definition audit events.")
    task.add_argument("--warnings-as-errors", action="store_true", help="Return nonzero and do not create/update when task visibility warnings are present.")
    task.add_argument("--output", default=None, help="Optional JSON output path for archiving the single-task report.")
    task.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    import_tasks = sub.add_parser("import-tasks", help="Dry-run or apply a JSON/JSONL labeling task manifest.")
    import_tasks.add_argument("--input", required=True, help="JSON list, JSONL file, or JSON object with tasks list.")
    import_tasks.add_argument("--actor", default=None, help="Operator/user recorded on task-definition audit events.")
    import_tasks.add_argument("--apply", action="store_true", help="Apply the task changes. Defaults to dry-run.")
    import_tasks.add_argument("--warnings-as-errors", action="store_true", help="Return nonzero and do not apply when task import warnings are present.")
    import_tasks.add_argument("--output", default=None, help="Optional JSON output path for archiving the task import report.")
    import_tasks.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    task_state = sub.add_parser("set-task-state", help="Set task state for operator completion or reopen workflows.")
    task_state.add_argument("--task-id", required=True)
    task_state.add_argument("--state", required=True, help="New task state, for example complete, pending, or in_progress.")
    task_state.add_argument("--user", required=True, help="Operator/user recorded on task-state audit events.")
    task_state.add_argument("--output", default=None, help="Optional JSON output path for archiving the task-state report.")
    task_state.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    list_cmd = sub.add_parser("list", help="List assignments and tasks.")
    list_cmd.add_argument("--user", default=None)
    list_cmd.add_argument("--recording-id", default=None)
    list_cmd.add_argument("--include-completed", action="store_true")
    list_cmd.add_argument("--output", default=None, help="Optional JSON output path for archiving the assignment/task listing report.")
    list_cmd.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    work_summary = sub.add_parser("work-summary", help="Preview the personalized work dashboard payload for a user.")
    work_summary.add_argument("--user", required=True)
    work_summary.add_argument("--include-completed", action="store_true")
    work_summary.add_argument(
        "--operator-launch-approved",
        action="store_true",
        help="Mark this work summary as launch-approved only after required operator validation evidence has been approved elsewhere.",
    )
    work_summary.add_argument(
        "--operator-validation-checklist",
        default=None,
        help="Validation-checklist JSON whose all_validation_complete state controls launch-evidence readiness in this work summary.",
    )
    work_summary.add_argument("--output", default=None, help="Optional output path for archiving the work summary JSON.")
    work_summary.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    dashboard_roster = sub.add_parser(
        "dashboard-roster",
        help=(
            "Export a dashboard-only ready-row draft roster for assigned labelers; "
            "safe-share inspection is still required before sharing links."
        ),
    )
    dashboard_roster.add_argument("--base-url", default=None, help="Browser-visible service base URL, for example https://labeling.example.org.")
    dashboard_roster.add_argument("--user", default=None, help="Restrict to one assigned user.")
    dashboard_roster.add_argument("--include-inactive", action="store_true", help="Include inactive assignments in user discovery.")
    dashboard_roster.add_argument("--include-completed", action="store_true", help="Include completed tasks in per-user dashboard counts.")
    dashboard_roster.add_argument(
        "--operator-launch-approved",
        action="store_true",
        help=(
            "Emit ready-row draft text only after required operator validation "
            "evidence has been approved elsewhere; safe-share inspection is still required."
        ),
    )
    dashboard_roster.add_argument(
        "--operator-validation-checklist",
        default=None,
        help=(
            "Validation-checklist JSON whose all_validation_complete state controls "
            "ready-row draft export; safe-share inspection is still required."
        ),
    )
    dashboard_roster.add_argument("--format", choices=("json", "jsonl", "csv", "html"), default="json")
    dashboard_roster.add_argument("--output", default=None, help="Optional output path for archiving the dashboard roster.")
    dashboard_roster.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    check_store = sub.add_parser("check-store", help="Check sidecar store consistency before sharing links or launching a batch.")
    check_store.add_argument("--output", default=None, help="Optional JSON output path for archiving the store consistency report.")
    check_store.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    list_sessions = sub.add_parser("list-sessions", help="List browser labeling sessions.")
    list_sessions.add_argument("--user", default=None, help="Filter to recordings currently assigned to this user.")
    list_sessions.add_argument("--include-closed", action="store_true")
    list_sessions.add_argument("--expired-only", action="store_true")
    list_sessions.add_argument("--limit", type=int, default=200)
    list_sessions.add_argument("--output", default=None, help="Optional JSON output path for archiving the session listing report.")
    list_sessions.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    cleanup_sessions = sub.add_parser("cleanup-stale-sessions", help="Close expired open browser labeling sessions.")
    cleanup_sessions.add_argument("--user", required=True, help="Operator/user recorded on stale-session audit events.")
    cleanup_sessions.add_argument("--output", default=None, help="Optional JSON output path for archiving the cleanup report.")
    cleanup_sessions.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    repair_reassignment_sessions = sub.add_parser(
        "repair-reassignment-sessions",
        help="Close active sessions whose user no longer matches the current recording assignment.",
    )
    repair_reassignment_sessions.add_argument("--user", required=True, help="Operator/user recorded on repair audit events.")
    repair_reassignment_sessions.add_argument(
        "--recording-id",
        action="append",
        default=[],
        help="Restrict repair to one recording. May be provided multiple times. Defaults to all reassignment-safety mismatches.",
    )
    repair_reassignment_sessions.add_argument("--output", default=None, help="Optional JSON output path for archiving the repair report.")
    repair_reassignment_sessions.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    export_events = sub.add_parser("export-events", help="Export audit events as JSON.")
    export_events.add_argument("--task-id", default=None)
    export_events.add_argument("--recording-id", default=None)
    export_events.add_argument("--event-type", default=None)
    export_events.add_argument("--user", default=None, help="Filter to events for recordings currently assigned to this user.")
    export_events.add_argument("--actor", default=None, help="Filter to events created by this user.")
    export_events.add_argument("--since-utc", default=None, help="Include events at or after this UTC timestamp.")
    export_events.add_argument("--until-utc", default=None, help="Include events at or before this UTC timestamp.")
    export_events.add_argument("--limit", type=int, default=1000)
    export_events.add_argument("--format", choices=("json", "jsonl"), default="json")
    export_events.add_argument("--output", default=None, help="Optional output path for archiving matching events.")
    export_events.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    lookup_event = sub.add_parser("lookup-event", help="Look up one task audit event by event_id.")
    lookup_event.add_argument("--event-id", required=True, help="Audit event ID reported by a labeler or validation smoke.")
    lookup_event.add_argument("--output", default=None, help="Optional JSON output path for archiving the event lookup.")
    lookup_event.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    export_assignment_events = sub.add_parser("export-assignment-events", help="Export recording assignment audit events as JSON.")
    export_assignment_events.add_argument("--recording-id", default=None)
    export_assignment_events.add_argument("--actor", default=None, help="Filter to assignment events created by this operator/user.")
    export_assignment_events.add_argument("--event-type", default=None)
    export_assignment_events.add_argument("--since-utc", default=None, help="Include events at or after this UTC timestamp.")
    export_assignment_events.add_argument("--until-utc", default=None, help="Include events at or before this UTC timestamp.")
    export_assignment_events.add_argument("--limit", type=int, default=1000)
    export_assignment_events.add_argument("--format", choices=("json", "jsonl"), default="json")
    export_assignment_events.add_argument("--output", default=None, help="Optional output path for archiving matching assignment events.")
    export_assignment_events.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    export_task_definition_events = sub.add_parser("export-task-definition-events", help="Export task definition audit events as JSON.")
    export_task_definition_events.add_argument("--task-id", default=None)
    export_task_definition_events.add_argument("--recording-id", default=None)
    export_task_definition_events.add_argument("--actor", default=None, help="Filter to task-definition events created by this operator/user.")
    export_task_definition_events.add_argument("--event-type", default=None)
    export_task_definition_events.add_argument("--since-utc", default=None, help="Include events at or after this UTC timestamp.")
    export_task_definition_events.add_argument("--until-utc", default=None, help="Include events at or before this UTC timestamp.")
    export_task_definition_events.add_argument("--limit", type=int, default=1000)
    export_task_definition_events.add_argument("--format", choices=("json", "jsonl"), default="json")
    export_task_definition_events.add_argument("--output", default=None, help="Optional output path for archiving matching task-definition events.")
    export_task_definition_events.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    export_audit_bundle = sub.add_parser("export-audit-bundle", help="Export task, assignment, and task-definition audit JSONL files plus a manifest.")
    export_audit_bundle.add_argument("--recording-id", default=None)
    export_audit_bundle.add_argument("--user", default=None, help="Bundle current recordings assigned to this user.")
    export_audit_bundle.add_argument("--since-utc", default=None, help="Include events at or after this UTC timestamp.")
    export_audit_bundle.add_argument("--until-utc", default=None, help="Include events at or before this UTC timestamp.")
    export_audit_bundle.add_argument("--limit", type=int, default=10000, help="Maximum events per event family or per selected recording.")
    export_audit_bundle.add_argument("--output-dir", required=True)
    export_audit_bundle.add_argument("--overwrite", action="store_true", help="Allow replacing existing bundle files.")

    sign_link = sub.add_parser("sign-link", help="Create a signed expiring task-open link.")
    sign_link.add_argument("--task-id", required=True)
    sign_link.add_argument("--ttl-seconds", type=int, default=24 * 60 * 60)
    sign_link.add_argument("--link-secret", default=None, help=f"Signing secret. Defaults to {LINK_SECRET_ENV_VAR}.")
    sign_link.add_argument("--base-url", default=None, help="Optional absolute service base URL, for example http://host:8795.")
    sign_link.add_argument("--output", default=None, help="Optional JSON output path for archiving the signed-link report.")
    sign_link.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    sign_links = sub.add_parser("sign-links", help="Export signed expiring links for active assigned tasks.")
    sign_links.add_argument("--recording-id", default=None)
    sign_links.add_argument("--user", default=None, help="Filter to recordings currently assigned to this user.")
    sign_links.add_argument("--include-completed", action="store_true")
    sign_links.add_argument("--ttl-seconds", type=int, default=24 * 60 * 60)
    sign_links.add_argument("--link-secret", default=None, help=f"Signing secret. Defaults to {LINK_SECRET_ENV_VAR}.")
    sign_links.add_argument("--base-url", default=None, help="Optional absolute service base URL, for example http://host:8795.")
    sign_links.add_argument("--format", choices=("json", "jsonl"), default="json")
    sign_links.add_argument("--output", default=None, help="Optional output path for archiving the link manifest.")
    sign_links.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    sign_invite = sub.add_parser("sign-invite", help="Create a signed expiring labeler invite to the personal dataset queue.")
    sign_invite.add_argument("--user", required=True, help="Labeler identity to bind into the invite token.")
    sign_invite.add_argument("--ttl-seconds", type=int, default=SIGNED_INVITE_DEFAULT_TTL_SECONDS)
    sign_invite.add_argument("--link-secret", default=None, help=f"Signing secret. Defaults to {LINK_SECRET_ENV_VAR}.")
    sign_invite.add_argument("--base-url", default=None, help="Optional absolute service base URL, for example http://host:8795.")
    sign_invite.add_argument("--output", default=None, help="Optional JSON output path for archiving the signed-invite report.")
    sign_invite.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    batch_readiness = sub.add_parser("batch-readiness", help="Report whether the current labeling batch is ready to announce.")
    batch_readiness.add_argument("--warnings-as-errors", action="store_true", help="Treat readiness and store-consistency warnings as launch-blocking.")
    batch_readiness.add_argument("--output", default=None, help="Optional JSON output path for archiving the readiness report.")
    batch_readiness.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    export_assignments = sub.add_parser("export-assignments", help="Export the current recording assignment snapshot.")
    export_assignments.add_argument("--recording-id", default=None)
    export_assignments.add_argument("--user", default=None, help="Filter by current assignee user.")
    export_assignments.add_argument("--status", default=None, help="Filter by assignment status, for example active or inactive.")
    export_assignments.add_argument("--format", choices=("json", "jsonl", "csv"), default="json")
    export_assignments.add_argument("--output", default=None, help="Optional output path for archiving the assignment snapshot.")
    export_assignments.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    export_tasks = sub.add_parser("export-tasks", help="Export the current labeling task snapshot.")
    export_tasks.add_argument("--recording-id", default=None)
    export_tasks.add_argument("--user", default=None, help="Filter by current assignee user.")
    export_tasks.add_argument("--workflow-kind", default=None)
    export_tasks.add_argument("--open-only", action="store_true", help="Exclude completed tasks from the snapshot.")
    export_tasks.add_argument("--format", choices=("json", "jsonl", "csv"), default="json")
    export_tasks.add_argument("--output", default=None, help="Optional output path for archiving the task snapshot.")
    export_tasks.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    manifest_templates = sub.add_parser("write-manifest-templates", help="Write starter CSV manifests for spreadsheet-based assignment and task planning.")
    manifest_templates.add_argument("--output-dir", required=True)
    manifest_templates.add_argument("--overwrite", action="store_true", help="Allow replacing existing template files.")

    batch_plan = sub.add_parser("import-batch-plan", help="Dry-run or apply assignment and task manifests together.")
    batch_plan.add_argument("--assignments", required=True, help="CSV/JSON/JSONL assignment manifest path.")
    batch_plan.add_argument("--tasks", required=True, help="CSV/JSON/JSONL task manifest path.")
    batch_plan.add_argument("--assigned-by", default=None, help="Fallback assigned_by value for assignment rows.")
    batch_plan.add_argument("--actor", default=None, help="Operator/user recorded on task-definition audit events.")
    batch_plan.add_argument("--apply", action="store_true", help="Apply assignments first, then tasks. Dry-run is the default.")
    batch_plan.add_argument("--warnings-as-errors", action="store_true", help="Treat batch-plan warnings as blocking validation failures before any apply.")
    batch_plan.add_argument("--output", default=None, help="Optional JSON output path for archiving the combined batch-plan report.")
    batch_plan.add_argument("--html-output", default=None, help="Optional HTML output path for reviewing the combined batch-plan report.")
    batch_plan.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    launch_bundle = sub.add_parser("export-launch-bundle", help="Write assignment/task snapshots, readiness, and all-user handoffs for a batch.")
    launch_bundle.add_argument("--output-dir", required=True)
    launch_bundle.add_argument("--base-url", default=None, help="Optional absolute service base URL, for example https://labeling.example.org.")
    launch_bundle.add_argument("--ttl-seconds", type=int, default=24 * 60 * 60)
    launch_bundle.add_argument("--link-secret", default=None, help=f"Signing secret. Defaults to {LINK_SECRET_ENV_VAR}.")
    launch_bundle.add_argument("--include-completed", action="store_true")
    launch_bundle.add_argument("--include-inactive", action="store_true", help="Include users who only have non-active assignments in handoff generation.")
    launch_bundle.add_argument("--include-audit-events", action="store_true", help="Include task, assignment, and task-definition audit JSONL files in the bundle.")
    launch_bundle.add_argument("--audit-since-utc", default=None, help="Optional lower UTC timestamp bound for included audit events.")
    launch_bundle.add_argument("--audit-until-utc", default=None, help="Optional upper UTC timestamp bound for included audit events.")
    launch_bundle.add_argument("--audit-limit", type=int, default=None, help="Optional maximum rows per audit event file.")
    launch_bundle.add_argument("--dry-run", action="store_true", help="Report the planned launch bundle without writing files or requiring a link secret.")
    launch_bundle.add_argument("--warnings-as-errors", action="store_true", help="Treat readiness warnings as launch-blocking.")
    launch_bundle.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty bundle directory and replacing known files.")
    launch_bundle.add_argument("--zip-output", default=None, help="Optional ZIP path for packaging the generated launch bundle directory.")

    inspect_handoff = sub.add_parser(
        "inspect-handoff",
        help=(
            "Inspect an exported handoff or launch bundle directory/ZIP for freshness. "
            "Assignment freshness is checked against the current store only when global --store is supplied."
        ),
    )
    inspect_handoff.add_argument("--path", required=True, help="Handoff or launch bundle directory/ZIP path to inspect.")
    inspect_handoff.add_argument("--output", default=None, help="Optional JSON output path for archiving the inspection report.")
    inspect_handoff.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")
    inspect_handoff.add_argument(
        "--require-shareable",
        action="store_true",
        help="Return nonzero unless inspection reports labeler links safe to share.",
    )

    refresh_checksums = sub.add_parser(
        "refresh-handoff-checksums",
        help="Refresh an existing checksums.json after operator evidence/checklist updates in a handoff or launch directory.",
    )
    refresh_checksums.add_argument("--path", required=True, help="Handoff or launch bundle directory containing checksums.json.")
    refresh_checksums.add_argument("--operator", required=True)
    refresh_checksums.add_argument("--reason", default="operator evidence update")

    validation_checklist = sub.add_parser("update-validation-checklist", help="Record operator evidence for one validation-checklist gate.")
    validation_checklist.add_argument("--path", required=True, help="validation-checklist.json path to update.")
    validation_checklist.add_argument("--gate", required=True, help="Gate id to update, for example browser_smoke.")
    validation_checklist.add_argument("--status", required=True, choices=sorted(VALIDATION_GATE_STATUSES))
    validation_checklist.add_argument("--evidence", action="append", default=None, help="Evidence note to append. May be repeated.")
    validation_checklist.add_argument("--evidence-file", action="append", default=None, help="Evidence file path to attach to the gate. May be repeated.")
    validation_checklist.add_argument("--operator", default=None, help="Operator/user recording this evidence.")
    validation_checklist.add_argument("--append-log", default=None, help="Append the same evidence to a Markdown validation log and record that log as an evidence file.")
    validation_checklist.add_argument("--output", default=None, help="Optional output path. Defaults to updating --path in place.")
    validation_checklist.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")

    apply_evidence_templates = sub.add_parser(
        "apply-operator-evidence-templates",
        help="Mark validation-checklist gates passed when their linked operator evidence templates are already approved.",
    )
    apply_evidence_templates.add_argument("--path", required=True, help="validation-checklist.json path to update.")
    apply_evidence_templates.add_argument("--operator", default=None, help="Operator/user applying approved evidence templates.")
    apply_evidence_templates.add_argument("--append-log", default=None, help="Append applied evidence entries to a Markdown validation log.")
    apply_evidence_templates.add_argument("--output", default=None, help="Optional output path. Defaults to updating --path in place.")
    apply_evidence_templates.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")
    apply_evidence_templates.add_argument(
        "--skip-handoff-refresh",
        action="store_true",
        help="Only update validation-checklist.json; do not refresh sibling handoff manifests or labeler-facing files.",
    )

    user_handoff = sub.add_parser("export-user-handoff", help="Write work preview, signed links, and safety report for one labeler.")
    user_handoff.add_argument("--user", required=True)
    user_handoff.add_argument("--output-dir", required=True)
    user_handoff.add_argument("--base-url", default=None, help="Optional absolute service base URL, for example https://labeling.example.org.")
    user_handoff.add_argument("--ttl-seconds", type=int, default=24 * 60 * 60)
    user_handoff.add_argument("--link-secret", default=None, help=f"Signing secret. Defaults to {LINK_SECRET_ENV_VAR}.")
    user_handoff.add_argument("--include-completed", action="store_true")
    user_handoff.add_argument("--overwrite", action="store_true", help="Allow replacing existing handoff files.")
    user_handoff.add_argument("--zip-output", default=None, help="Optional ZIP path for packaging the generated handoff directory.")

    user_handoffs = sub.add_parser("export-user-handoffs", help="Write handoff directories for every assigned labeler.")
    user_handoffs.add_argument("--output-dir", required=True)
    user_handoffs.add_argument("--base-url", default=None, help="Optional absolute service base URL, for example https://labeling.example.org.")
    user_handoffs.add_argument("--ttl-seconds", type=int, default=24 * 60 * 60)
    user_handoffs.add_argument("--link-secret", default=None, help=f"Signing secret. Defaults to {LINK_SECRET_ENV_VAR}.")
    user_handoffs.add_argument("--include-completed", action="store_true")
    user_handoffs.add_argument("--include-inactive", action="store_true", help="Include users who only have non-active assignments.")
    user_handoffs.add_argument("--overwrite", action="store_true", help="Allow replacing existing handoff files.")
    user_handoffs.add_argument("--zip-output", default=None, help="Optional ZIP path for packaging the generated batch handoff directory.")

    gen_kp = sub.add_parser("generate-keypoint-tasks", help="Generate assigned keypoint tasks from registry training zarrs.")
    gen_kp.add_argument(
        "--registry",
        nargs="?",
        const="auto",
        default="auto",
        help="Registry SQLite path. Defaults to PALETTE_REGISTRY_PATH/config/default.",
    )
    gen_kp.add_argument("--user", default=None, help="Restrict to recordings assigned to this user.")
    gen_kp.add_argument("--recording-id", default=None, help="Restrict to one assigned recording.")
    gen_kp.add_argument(
        "--review-filter",
        default="needs_review",
        choices=["needs_review", "all", "approved"],
        help="Which keypoint-reviewable training zarrs should receive tasks.",
    )
    gen_kp.add_argument("--priority", type=int, default=0)
    gen_kp.add_argument("--include-all", action="store_true", help="Generated sessions review all ROIs instead of failed ROIs.")
    gen_kp.add_argument("--no-auto-advance-on-save", action="store_true")
    gen_kp.add_argument("--warnings-as-errors", action="store_true", help="Return nonzero when registry rows are skipped during generation.")
    gen_kp.add_argument("--output", default=None, help="Optional JSON output path for archiving the generation report.")
    gen_kp.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    gen_detect = sub.add_parser("generate-detect-training-tasks", help="Generate assigned materialized detection tasks from registry training zarrs.")
    gen_detect.add_argument(
        "--registry",
        nargs="?",
        const="auto",
        default="auto",
        help="Registry SQLite path. Defaults to PALETTE_REGISTRY_PATH/config/default.",
    )
    gen_detect.add_argument("--user", default=None, help="Restrict to recordings assigned to this user.")
    gen_detect.add_argument("--recording-id", default=None, help="Restrict to one assigned recording.")
    gen_detect.add_argument(
        "--review-filter",
        default="needs_review",
        choices=["needs_review", "all", "approved"],
        help="Which detect-reviewable training zarrs should receive tasks.",
    )
    gen_detect.add_argument("--priority", type=int, default=0)
    gen_detect.add_argument("--include-all", action="store_true", help="Generated sessions review all frames instead of missing/filtered frames.")
    gen_detect.add_argument("--no-auto-advance-on-save", action="store_true")
    gen_detect.add_argument("--warnings-as-errors", action="store_true", help="Return nonzero when registry rows are skipped during generation.")
    gen_detect.add_argument("--output", default=None, help="Optional JSON output path for archiving the generation report.")
    gen_detect.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    gen_detect_analysis = sub.add_parser(
        "generate-detect-analysis-tasks",
        help="Generate assigned video-backed detection tasks from registry analysis zarrs.",
    )
    gen_detect_analysis.add_argument(
        "--registry",
        nargs="?",
        const="auto",
        default="auto",
        help="Registry SQLite path. Defaults to PALETTE_REGISTRY_PATH/config/default.",
    )
    gen_detect_analysis.add_argument("--user", default=None, help="Restrict to recordings assigned to this user.")
    gen_detect_analysis.add_argument("--recording-id", default=None, help="Restrict to one assigned recording.")
    gen_detect_analysis.add_argument(
        "--review-filter",
        default="needs_review",
        choices=["needs_review", "all", "approved"],
        help="Which analysis detection zarrs should receive tasks.",
    )
    gen_detect_analysis.add_argument("--priority", type=int, default=0)
    gen_detect_analysis.add_argument("--editable", action="store_true", help="Generated tasks may save edits back to the analysis zarr.")
    gen_detect_analysis.add_argument(
        "--promote-training-zarr",
        default=None,
        help="After each saved analysis edit, promote into this training zarr. Use 'auto' to resolve the assigned recording's training zarr from the registry.",
    )
    gen_detect_analysis.add_argument("--promote-target-crop-run", default=None)
    gen_detect_analysis.add_argument("--promote-target-refined-run", default=None)
    gen_detect_analysis.add_argument("--promote-label-origin", default="palette_labeling_work")
    gen_detect_analysis.add_argument("--promote-no-negative", action="store_true", help="Do not promote clear/no-box saves as negative rows.")
    gen_detect_analysis.add_argument("--promote-allow-unreviewed-negative", action="store_true")
    gen_detect_analysis.add_argument(
        "--promote-target-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Target promoted image size when the training zarr does not define raw_video/images_ds.",
    )
    gen_detect_analysis.add_argument("--no-auto-advance-on-save", action="store_true")
    gen_detect_analysis.add_argument("--warnings-as-errors", action="store_true", help="Return nonzero when registry rows are skipped during generation.")
    gen_detect_analysis.add_argument("--output", default=None, help="Optional JSON output path for archiving the generation report.")
    gen_detect_analysis.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    gen_subject = sub.add_parser(
        "generate-subject-mask-tasks",
        help="Generate assigned subject-mask component tasks from registry component quality rows.",
    )
    gen_subject.add_argument(
        "--registry",
        nargs="?",
        const="auto",
        default="auto",
        help="Registry SQLite path. Defaults to PALETTE_REGISTRY_PATH/config/default.",
    )
    gen_subject.add_argument("--user", default=None, help="Restrict to recordings assigned to this user.")
    gen_subject.add_argument("--recording-id", default=None, help="Restrict to one assigned recording.")
    gen_subject.add_argument(
        "--review-filter",
        default="needs_review",
        choices=["needs_review", "all", "approved"],
        help="Which refined subject-mask components should receive tasks.",
    )
    gen_subject.add_argument(
        "--component",
        dest="components",
        action="append",
        default=None,
        help="Restrict to one component name. May be repeated.",
    )
    gen_subject.add_argument("--priority", type=int, default=0)
    gen_subject.add_argument("--no-auto-advance-on-save", action="store_true")
    gen_subject.add_argument("--warnings-as-errors", action="store_true", help="Return nonzero when registry rows are skipped during generation.")
    gen_subject.add_argument("--output", default=None, help="Optional JSON output path for archiving the generation report.")
    gen_subject.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")

    serve_cmd = sub.add_parser("serve", help="Serve the personalized labeling dashboard.")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8795)
    serve_cmd.add_argument("--user", default=None, help="Fixed local-development user override.")
    serve_cmd.add_argument("--auth-header", default="X-Forwarded-User")
    serve_cmd.add_argument(
        "--trust-auth-header",
        action="store_true",
        help="Trust --auth-header as the authenticated user identity. Use only behind a proxy that strips spoofed inbound auth headers.",
    )
    serve_cmd.add_argument("--session-ttl-seconds", type=int, default=12 * 60 * 60)
    serve_cmd.add_argument("--link-secret", default=None, help=f"Signed-link secret. Defaults to {LINK_SECRET_ENV_VAR}.")
    serve_cmd.add_argument(
        "--link-not-before-utc",
        default=None,
        help=f"Reject signed links issued before this UTC ISO timestamp or Unix epoch. Defaults to {LINK_NOT_BEFORE_ENV_VAR}.",
    )
    serve_cmd.add_argument("--disable-csrf-check", action="store_true", help="Disable same-origin checks on mutating POST requests.")
    serve_cmd.add_argument("--access-log", action="store_true", help="Emit JSON request access logs to stderr.")
    serve_cmd.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Allow binding the service to a non-loopback host. Use only when network exposure is intentional.",
    )
    serve_cmd.add_argument(
        "--production",
        action="store_true",
        help="Enable production launch checks: trusted proxy auth header and at least one admin user are required.",
    )
    serve_cmd.add_argument(
        "--validation-checklist",
        default=None,
        help="Operator validation-checklist.json to enforce before browser Start/Open creates sessions and browser mutation routes write.",
    )
    serve_cmd.add_argument(
        "--require-operator-validation-for-start",
        action="store_true",
        help="Block browser Start/Open and browser mutations until every required gate in --validation-checklist is passed or not_applicable.",
    )
    serve_cmd.add_argument(
        "--require-operator-validation-for-browser-work",
        action="store_true",
        help="Clearer alias for --require-operator-validation-for-start; blocks browser Start/Open and browser mutations until every required gate in --validation-checklist is passed or not_applicable.",
    )
    serve_cmd.add_argument(
        "--admin-user",
        action="append",
        default=None,
        help="User allowed to access /admin and /api/admin/*. May be repeated.",
    )
    preflight_cmd = sub.add_parser("preflight", help="Check labeling server launch safety without starting the HTTP server.")
    preflight_cmd.add_argument("--host", default="127.0.0.1")
    preflight_cmd.add_argument("--port", type=int, default=8795)
    preflight_cmd.add_argument("--user", default=None, help="Fixed local-development user override.")
    preflight_cmd.add_argument("--auth-header", default="X-Forwarded-User")
    preflight_cmd.add_argument(
        "--trust-auth-header",
        action="store_true",
        help="Trust --auth-header as the authenticated user identity. Use only behind a proxy that strips spoofed inbound auth headers.",
    )
    preflight_cmd.add_argument("--session-ttl-seconds", type=int, default=12 * 60 * 60)
    preflight_cmd.add_argument("--link-secret", default=None, help=f"Signed-link secret. Defaults to {LINK_SECRET_ENV_VAR}.")
    preflight_cmd.add_argument(
        "--link-not-before-utc",
        default=None,
        help=f"Reject signed links issued before this UTC ISO timestamp or Unix epoch. Defaults to {LINK_NOT_BEFORE_ENV_VAR}.",
    )
    preflight_cmd.add_argument("--disable-csrf-check", action="store_true", help="Disable same-origin checks on mutating POST requests.")
    preflight_cmd.add_argument("--access-log", action="store_true", help="Emit JSON request access logs to stderr.")
    preflight_cmd.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Allow binding the service to a non-loopback host. Use only when network exposure is intentional.",
    )
    preflight_cmd.add_argument(
        "--production",
        action="store_true",
        help="Enable production launch checks: trusted proxy auth header and at least one admin user are required.",
    )
    preflight_cmd.add_argument(
        "--validation-checklist",
        default=None,
        help="Operator validation-checklist.json to enforce before browser Start/Open creates sessions and browser mutation routes write.",
    )
    preflight_cmd.add_argument(
        "--require-operator-validation-for-start",
        action="store_true",
        help="Block browser Start/Open and browser mutations until every required gate in --validation-checklist is passed or not_applicable.",
    )
    preflight_cmd.add_argument(
        "--require-operator-validation-for-browser-work",
        action="store_true",
        help="Clearer alias for --require-operator-validation-for-start; blocks browser Start/Open and browser mutations until every required gate in --validation-checklist is passed or not_applicable.",
    )
    preflight_cmd.add_argument(
        "--admin-user",
        action="append",
        default=None,
        help="User allowed to access /admin and /api/admin/*. May be repeated.",
    )
    preflight_cmd.add_argument("--output", default=None, help="Optional JSON output path for archiving the server preflight report.")
    preflight_cmd.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output report.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    store_path = Path(args.store).expanduser() if args.store is not None else default_store_path()

    if args.command in {"serve", "preflight"}:
        config = ServerConfig(
            store_path=store_path,
            host=str(args.host),
            port=int(args.port),
            fixed_user=args.user,
            auth_header=str(args.auth_header),
            trust_auth_header=bool(args.trust_auth_header),
            session_ttl_seconds=int(args.session_ttl_seconds),
            admin_users=tuple(str(item) for item in (args.admin_user or [])),
            link_secret=str(args.link_secret or os.environ.get(LINK_SECRET_ENV_VAR) or "").strip() or None,
            link_not_before_utc=str(args.link_not_before_utc or os.environ.get(LINK_NOT_BEFORE_ENV_VAR) or "").strip() or None,
            csrf_same_origin=not bool(args.disable_csrf_check),
            access_log=bool(args.access_log),
            allow_non_loopback=bool(args.allow_non_loopback),
            production=bool(args.production),
            validation_checklist_path=(
                Path(args.validation_checklist).expanduser()
                if args.validation_checklist
                else None
            ),
            require_operator_validation_for_start=bool(
                args.require_operator_validation_for_start
                or args.require_operator_validation_for_browser_work
            ),
        )
        if args.command == "preflight":
            errors = _server_config_errors(config)
            payload = {
                "ok": not errors,
                "errors": errors,
                "error_count": len(errors),
                "preflight": _server_safety_payload(config, include_admin_details=True),
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="server preflight report")
            _print_json(payload)
            return 0 if not errors else 2
        return serve(config)

    if args.command == "inspect-handoff" and args.store is None:
        report = _inspect_handoff_package(
            Path(args.path),
            store=None,
            require_shareable=bool(args.require_shareable),
        )
        if args.output:
            output_path = Path(args.output)
            if output_path.exists() and not bool(args.overwrite):
                raise FileExistsError(f"Refusing to overwrite existing handoff inspection report: {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _print_json(report)
        if bool(args.require_shareable) and not bool(report.get("labeler_links_safe_to_share")):
            return 2
        return 0 if bool(report["ok"]) else 2

    if args.command == "refresh-handoff-checksums":
        payload = _refresh_handoff_directory_checksums(
            path=Path(args.path),
            operator=str(args.operator),
            reason=str(args.reason),
        )
        _print_json(payload)
        return 0

    if args.command == "execute-zarr-backup-plan":
        payload = _execute_zarr_backup_plan(
            plan_path=Path(args.plan),
            backup_dir=Path(args.backup_dir),
            operator=str(args.operator),
            output=Path(args.output) if args.output else None,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            allow_missing=bool(args.allow_missing),
        )
        _print_json(payload)
        return 0 if bool(payload["ok"]) else 2

    if args.command == "record-zarr-backup-evidence":
        target_indexes = args.target_index or []
        if not bool(args.all_targets) and not target_indexes:
            payload = {
                "ok": False,
                "schema": "palette.web_labeling_zarr_backup_evidence_update_report.v1",
                "error": "target_index_or_all_targets_required",
                "details": "Pass --target-index at least once, or pass --all-targets.",
            }
            _write_optional_json_report(
                payload,
                args.output,
                overwrite=bool(args.overwrite),
                description="zarr backup evidence update report",
            )
            _print_json(payload)
            return 2
        payload = _record_zarr_backup_evidence(
            evidence_path=Path(args.evidence),
            execution_manifest_path=Path(args.execution_manifest),
            operator=str(args.operator),
            restore_test_result=str(args.restore_test_result),
            target_indexes=target_indexes,
            record_all=bool(args.all_targets),
            output=Path(args.output) if args.output else None,
            overwrite=bool(args.overwrite),
            notes=args.notes,
        )
        _print_json(payload)
        return 0 if bool(payload["ok"]) else 2

    if args.command == "record-identity-source-evidence":
        payload = _record_identity_source_evidence(
            evidence_path=Path(args.evidence),
            expected_user=str(args.expected_user),
            resolved_user=str(args.resolved_user),
            operator=str(args.operator),
            authenticated_session_context=args.authenticated_session_context,
            notes=args.notes,
            output=Path(args.output) if args.output else None,
            overwrite=bool(args.overwrite),
        )
        _print_json(payload)
        return 0 if bool(payload["ok"]) else 2

    if args.command == "record-browser-response-security-evidence":
        payload = _record_browser_response_security_evidence(
            evidence_path=Path(args.evidence),
            headers=_parse_header_evidence_values(args.header or []),
            operator=str(args.operator),
            capture_url=args.capture_url,
            authenticated_test_user=args.authenticated_test_user,
            capture_note=args.capture_note,
            proxy_or_deployment=args.proxy_or_deployment,
            notes=args.notes,
            output=Path(args.output) if args.output else None,
            overwrite=bool(args.overwrite),
        )
        _print_json(payload)
        return 0 if bool(payload["ok"]) else 2

    if args.command == "record-browser-smoke-evidence":
        payload = _record_browser_smoke_evidence(
            evidence_path=Path(args.evidence),
            expected_user=str(args.expected_user),
            resolved_user=str(args.resolved_user),
            operator=str(args.operator),
            checks={
                "browser_only_runtime_verified": bool(args.browser_only_runtime_verified),
                "no_local_palette_install_verified": bool(args.no_local_palette_install_verified),
                "no_local_crimson_install_verified": bool(args.no_local_crimson_install_verified),
                "no_local_conda_or_project_dependencies_verified": bool(
                    args.no_local_conda_or_project_dependencies_verified
                ),
                "personalized_dataset_queue_verified": bool(
                    args.personalized_dataset_queue_verified
                ),
                "preferred_labeler_entry_url_matches_personal_dataset_queue": bool(
                    args.preferred_labeler_entry_url_matches_personal_dataset_queue
                ),
                "personalized_labeler_entry_url_matches_personal_dataset_queue": bool(
                    args.personalized_labeler_entry_url_matches_personal_dataset_queue
                ),
                "personalized_work_dashboard_verified": bool(
                    args.personalized_work_dashboard_verified
                ),
                "labeler_sees_only_assigned_work": bool(args.labeler_sees_only_assigned_work),
                "support_text_redacted": bool(args.support_text_redacted),
                "expected_user_mismatch_rejected": bool(args.expected_user_mismatch_rejected),
                "task_opened": bool(args.task_opened),
                "induced_failure_support_detail_redacted": bool(args.induced_failure_support_detail_redacted),
                "completion_verified": bool(args.completion_verified),
                "completed_task_read_only_verified": bool(args.completed_task_read_only_verified),
                "stale_tab_save_rejected": bool(args.stale_tab_save_rejected),
                "operator_reopen_verified": bool(args.operator_reopen_verified),
            },
            notes=args.notes,
            output=Path(args.output) if args.output else None,
            overwrite=bool(args.overwrite),
        )
        _print_json(payload)
        return 0 if bool(payload["ok"]) else 2

    if args.command == "record-disposable-zarr-mutation-smoke-evidence":
        payload = _record_disposable_zarr_mutation_smoke_evidence(
            evidence_path=Path(args.evidence),
            workflow_kind=str(args.workflow_kind),
            operator=str(args.operator),
            mutation_event_ids=args.mutation_event_id or [],
            event_lookup_reports=[Path(path) for path in (args.event_lookup_report or [])],
            registry_refresh_event_ids=args.registry_refresh_event_id or [],
            disposable_recording_id=args.disposable_recording_id,
            disposable_task_id=args.disposable_task_id,
            labeler_user=args.labeler_user,
            disposable_zarr_or_known_good_source=args.disposable_zarr_or_known_good_source,
            bad_mutation_recovery_mode=args.bad_mutation_recovery_mode,
            bad_mutation_recovery_report=args.bad_mutation_recovery_report,
            checks={
                "backup_or_regeneration_verified": bool(args.backup_or_regeneration_verified),
                "server_write_scope_verified": bool(args.server_write_scope_verified),
                "task_scoped_training_zarr_write_verified": bool(
                    args.task_scoped_training_zarr_write_verified
                ),
                "browser_no_direct_zarr_write_authority_verified": bool(
                    args.browser_no_direct_zarr_write_authority_verified
                ),
                "handoff_artifacts_metadata_only_verified": bool(
                    args.handoff_artifacts_metadata_only_verified
                ),
                "browser_no_csv_or_handoff_write_verified": bool(
                    args.browser_no_csv_or_handoff_write_verified
                ),
                "client_target_selector_rejection_verified": bool(
                    args.client_target_selector_rejection_verified
                ),
                "audit_event_verified": bool(args.audit_event_verified),
                "operator_event_lookup_verified": bool(args.operator_event_lookup_verified),
                "completion_verified": bool(args.completion_verified),
                "stale_tab_save_rejected": bool(args.stale_tab_save_rejected),
                "bad_mutation_recovery_verified": bool(args.bad_mutation_recovery_verified),
                "restored_or_discarded": bool(args.restored_or_discarded),
            },
            notes=args.notes,
            output=Path(args.output) if args.output else None,
            overwrite=bool(args.overwrite),
        )
        _print_json(payload)
        return 0 if bool(payload["ok"]) else 2

    if args.command == "apply-operator-evidence-templates":
        if not bool(args.skip_handoff_refresh) and store_path.exists():
            with LabelingStore(store_path) as store:
                store.initialize()
                payload = _apply_operator_evidence_templates_to_validation_checklist_file(
                    path=Path(args.path),
                    operator=args.operator,
                    append_log=Path(args.append_log) if args.append_log else None,
                    output=Path(args.output) if args.output else None,
                    overwrite=bool(args.overwrite),
                    refresh_handoffs=True,
                    store=store,
                )
        else:
            payload = _apply_operator_evidence_templates_to_validation_checklist_file(
                path=Path(args.path),
                operator=args.operator,
                append_log=Path(args.append_log) if args.append_log else None,
                output=Path(args.output) if args.output else None,
                overwrite=bool(args.overwrite),
                refresh_handoffs=not bool(args.skip_handoff_refresh),
            )
        _print_json(payload)
        return 0

    with LabelingStore(store_path) as store:
        store.initialize()
        if args.command == "init":
            _print_json({"ok": True, "store": str(store_path)})
            return 0
        if args.command == "backup-store":
            result = store.backup_to(args.output, overwrite=bool(args.overwrite))
            result["ok"] = True
            _print_json(result)
            return 0
        if args.command == "users-list":
            users = store.list_labeling_users(status=args.status, role=args.role)
            payload = {
                "ok": True,
                "schema": "palette.web_labeling_users_list.v1",
                "store_path": str(store_path),
                "filters": {"status": args.status, "role": args.role},
                "roles": list(LABELING_USER_ROLES),
                "statuses": list(LABELING_USER_STATUSES),
                "count": len(users),
                "users": users,
            }
            _print_json(payload)
            return 0
        if args.command == "users-add":
            user_row = store.upsert_labeling_user(
                user_id=args.user_id,
                display_name=args.display_name,
                email=args.email,
                role=args.role,
                status=args.status,
                notes=args.notes,
                actor_user=args.actor,
            )
            notification_result = None
            if bool(args.notify):
                try:
                    notification_result = send_labeler_added_notification(
                        user=user_row,
                        actor_user=args.actor,
                        config=_notification_config_from_values(
                            mode=args.notification_mode,
                            base_url=args.notification_base_url,
                        ),
                    )
                except Exception as exc:
                    notification_result = _notification_exception_result(
                        kind="labeler_added",
                        to_user=str(user_row.get("user_id") or args.user_id),
                        exc=exc,
                    )
                store.record_labeling_user_event(
                    user_id=str(user_row.get("user_id") or args.user_id),
                    actor_user=args.actor,
                    event_type=_notification_event_type(
                        notification_result,
                        prefix="labeling_user_notification",
                    ),
                    after=notification_result,
                )
            payload = {
                "ok": True,
                "schema": "palette.web_labeling_user_update.v1",
                "store_path": str(store_path),
                "user": user_row,
                "known_user_status": _known_labeler_status(store, str(user_row.get("user_id") or args.user_id)),
                "notification": notification_result,
            }
            _print_json(payload)
            return 0
        if args.command == "users-activate":
            user_row = store.activate_labeling_user(
                args.user_id,
                actor_user=args.actor,
                notes=args.notes,
            )
            payload = {
                "ok": True,
                "schema": "palette.web_labeling_user_status_update.v1",
                "store_path": str(store_path),
                "action": "activate",
                "user": user_row,
                "known_user_status": _known_labeler_status(store, str(user_row.get("user_id") or args.user_id)),
            }
            _print_json(payload)
            return 0
        if args.command == "users-deactivate":
            user_row = store.deactivate_labeling_user(
                args.user_id,
                actor_user=args.actor,
                notes=args.notes,
            )
            payload = {
                "ok": True,
                "schema": "palette.web_labeling_user_status_update.v1",
                "store_path": str(store_path),
                "action": "deactivate",
                "user": user_row,
                "known_user_status": _known_labeler_status(store, str(user_row.get("user_id") or args.user_id)),
            }
            _print_json(payload)
            return 0
        if args.command == "zarr-backup-plan":
            payload = _zarr_backup_plan(
                store=store,
                store_path=store_path,
                recording_id=args.recording_id,
                user=args.user,
                include_completed=bool(args.include_completed),
                include_inactive=bool(args.include_inactive),
            )
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="zarr backup plan")
            _print_json(payload)
            return 0
        if args.command == "restore-zarr-backup":
            target_indexes = args.target_index or []
            if not bool(args.all_targets) and not target_indexes:
                payload = {
                    "ok": False,
                    "schema": "palette.web_labeling_zarr_backup_restore_report.v1",
                    "error": "target_index_or_all_targets_required",
                    "details": "Pass --target-index at least once, or pass --all-targets.",
                }
                _write_optional_json_report(
                    payload,
                    args.output,
                    overwrite=bool(args.overwrite),
                    description="zarr backup restore report",
                )
                _print_json(payload)
                return 2
            if args.output is not None and Path(args.output).expanduser().exists() and not bool(args.overwrite):
                raise FileExistsError(f"Output file already exists: {args.output}")
            payload = _restore_zarr_backup_manifest(
                store=store,
                manifest_path=Path(args.manifest),
                operator=str(args.operator),
                target_indexes=target_indexes,
                restore_all=bool(args.all_targets),
                replace_current=bool(args.replace_current),
                allow_active_assignment=bool(args.allow_active_assignment),
            )
            _write_optional_json_report(
                payload,
                args.output,
                overwrite=bool(args.overwrite),
                description="zarr backup restore report",
            )
            _print_json(payload)
            return 0 if bool(payload["ok"]) else 2
        if args.command == "assign":
            if args.output is not None and Path(args.output).expanduser().exists() and not bool(args.overwrite):
                raise FileExistsError(f"Output file already exists: {args.output}")
            assignee_issues = _active_assignee_user_issues(
                store,
                [{"recording_id": args.recording_id, "assignee_user": args.user}],
            )
            if assignee_issues:
                payload = {
                    "ok": False,
                    "schema": "palette.web_labeling_assignment_user_validation.v1",
                    "store_path": str(store_path),
                    "issue_count": len(assignee_issues),
                    "issues": assignee_issues,
                }
                _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="assignment report")
                _print_json(payload)
                return 2
            transition_result = store.assign_recording_with_session_closure(
                recording_id=args.recording_id,
                assignee_user=args.user,
                assigned_by=args.assigned_by,
                status=args.status,
                notes=args.notes,
            )
            notification_result = None
            if bool(args.notify):
                assignment = transition_result["assignment"]
                try:
                    notification_result = send_assignment_available_notification(
                        user=store.get_labeling_user(str(assignment.get("assignee_user") or args.user)),
                        assignment=assignment,
                        actor_user=args.assigned_by,
                        config=_notification_config_from_values(
                            mode=args.notification_mode,
                            base_url=args.notification_base_url,
                        ),
                    )
                except Exception as exc:
                    notification_result = _notification_exception_result(
                        kind="assignment_available",
                        to_user=str(assignment.get("assignee_user") or args.user),
                        exc=exc,
                    )
                store.record_assignment_event(
                    recording_id=str(assignment.get("recording_id") or args.recording_id),
                    actor_user=args.assigned_by,
                    event_type=_notification_event_type(
                        notification_result,
                        prefix="assignment_notification",
                    ),
                    after=notification_result,
                )
            payload = {
                "ok": True,
                "assignment": transition_result["assignment"],
                "previous_assignment": transition_result.get("previous_assignment"),
                "assignment_transition": transition_result.get("assignment_transition"),
                **_assignment_control_plane_report_fields(store),
                "closed_session_count": transition_result.get("closed_session_count", 0),
                "closed_session_ids": transition_result.get("closed_session_ids", []),
                "closed_sessions": transition_result.get("closed_sessions", []),
                "notification": notification_result,
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="assignment report")
            _print_json(payload)
            return 0
        if args.command == "import-assignments":
            rows = _parse_assignment_manifest(args.input)
            assignee_issues = _active_assignee_user_issues(store, rows)
            if assignee_issues:
                payload = {
                    "ok": False,
                    "schema": "palette.web_labeling_assignment_import_user_validation.v1",
                    "store_path": str(store_path),
                    "input": str(args.input),
                    "apply": bool(args.apply),
                    "issue_count": len(assignee_issues),
                    "issues": assignee_issues,
                    "assignment_count": len(rows),
                    **_assignment_control_plane_report_fields(store),
                }
                _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="assignment import report")
                _print_json(payload)
                return 2
            results: list[dict[str, object]] = []
            warnings: list[dict[str, object]] = []
            row_warnings_by_index: dict[int, list[dict[str, object]]] = {}
            rows_by_recording: dict[str, list[dict[str, object]]] = {}
            for row in rows:
                recording_id = str(row.get("recording_id") or "")
                if recording_id:
                    rows_by_recording.setdefault(recording_id, []).append(row)
            for recording_id, duplicate_rows in sorted(rows_by_recording.items()):
                if len(duplicate_rows) <= 1:
                    continue
                warnings.append(
                    {
                        "code": "duplicate_recording_assignment_rows",
                        "recording_id": recording_id,
                        "assignee_users": [row.get("assignee_user") for row in duplicate_rows],
                        "statuses": [str(row.get("status") or "active") for row in duplicate_rows],
                        "source_lines": [
                            int(row["_source_line"])
                            for row in duplicate_rows
                            if row.get("_source_line") is not None
                        ],
                        "details": "Assignment manifest contains multiple rows for one recording; later rows determine the final assignee/status.",
                    }
                )
            for row_index, row in enumerate(rows):
                recording_id = str(row["recording_id"])
                assignee_user = str(row["assignee_user"])
                existing = store.get_assignment(recording_id)
                if existing is not None:
                    previous_user = str(existing.get("assignee_user") or "")
                    if previous_user and previous_user != assignee_user:
                        warning = {
                            "code": "assignment_reassigns_existing_recording",
                            "recording_id": recording_id,
                            "previous_assignee_user": previous_user,
                            "new_assignee_user": assignee_user,
                            "previous_status": existing.get("status"),
                            "new_status": str(row.get("status") or "active"),
                            **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                            "details": "Assignment changes the current recording owner; active sessions for this recording will be closed if applied.",
                        }
                        row_warnings_by_index.setdefault(row_index, []).append(warning)
                        warnings.append(warning)
            blocked_by_warnings = bool(warnings) and bool(args.warnings_as_errors)
            apply_rows = _assignment_rows_for_apply(rows, apply=bool(args.apply) and not blocked_by_warnings)
            apply_row_ids = {id(row) for row in apply_rows} if bool(args.apply) and not blocked_by_warnings else set()
            applied_result_count = 0
            notification_results: list[dict[str, object]] = []
            for row_index, row in enumerate(rows):
                recording_id = str(row["recording_id"])
                assignee_user = str(row["assignee_user"])
                existing = store.get_assignment(recording_id)
                assigned_by = row.get("assigned_by") if row.get("assigned_by") is not None else args.assigned_by
                target = {
                    "recording_id": recording_id,
                    "assignee_user": assignee_user,
                    "assigned_by": assigned_by,
                    "status": str(row.get("status") or "active"),
                    "notes": row.get("notes"),
                }
                existing_target = (
                    {
                        "recording_id": existing.get("recording_id"),
                        "assignee_user": existing.get("assignee_user"),
                        "assigned_by": existing.get("assigned_by"),
                        "status": existing.get("status"),
                        "notes": existing.get("notes"),
                    }
                    if existing is not None
                    else None
                )
                row_warnings = list(row_warnings_by_index.get(row_index, []))
                should_apply_row = bool(args.apply) and not blocked_by_warnings and id(row) in apply_row_ids
                if bool(args.apply) and not blocked_by_warnings and not should_apply_row:
                    row_warnings.insert(
                        0,
                        {
                            "code": "duplicate_assignment_row_skipped_for_apply",
                            "recording_id": recording_id,
                            "assignee_user": assignee_user,
                            **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                            "details": "Duplicate assignment input row was reported but not applied; only the final row for each recording mutates ownership/status.",
                        }
                    )
                notification_result = None
                if should_apply_row:
                    transition_result = store.assign_recording_with_session_closure(
                        recording_id=recording_id,
                        assignee_user=assignee_user,
                        assigned_by=str(assigned_by) if assigned_by is not None else None,
                        status=str(target["status"]),
                        notes=str(target["notes"]) if target["notes"] is not None else None,
                    )
                    assignment = transition_result["assignment"]
                    closed_sessions = list(transition_result.get("closed_sessions") or [])
                    assignment_transition = transition_result.get("assignment_transition")
                    applied_result_count += 1
                    if bool(args.notify):
                        try:
                            notification_result = send_assignment_available_notification(
                                user=store.get_labeling_user(str(assignment.get("assignee_user") or assignee_user)),
                                assignment=assignment,
                                actor_user=str(assigned_by) if assigned_by is not None else None,
                                config=_notification_config_from_values(
                                    mode=args.notification_mode,
                                    base_url=args.notification_base_url,
                                ),
                            )
                        except Exception as exc:
                            notification_result = _notification_exception_result(
                                kind="assignment_available",
                                to_user=str(assignment.get("assignee_user") or assignee_user),
                                exc=exc,
                            )
                        store.record_assignment_event(
                            recording_id=str(assignment.get("recording_id") or recording_id),
                            actor_user=str(assigned_by) if assigned_by is not None else None,
                            event_type=_notification_event_type(
                                notification_result,
                                prefix="assignment_notification",
                            ),
                            after=notification_result,
                        )
                        notification_results.append(notification_result)
                else:
                    assignment = target
                    closed_sessions = []
                    assignment_transition = None
                results.append(
                    {
                        "recording_id": recording_id,
                        "existing": existing,
                        "assignment": assignment,
                        "would_change": existing_target != target,
                        "applied": should_apply_row,
                        "skipped_by_duplicate_apply": bool(args.apply) and not blocked_by_warnings and not should_apply_row,
                        "assignment_transition": assignment_transition,
                        "closed_session_count": len(closed_sessions),
                        "closed_sessions": closed_sessions,
                        "notification": notification_result,
                        "warnings": row_warnings,
                        **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                    }
                )
            warning_codes = sorted(
                {
                    str(warning.get("code") or "")
                    for warning in warnings
                    if str(warning.get("code") or "")
                }
            )
            payload = {
                "ok": not blocked_by_warnings,
                "dry_run": not bool(args.apply) or blocked_by_warnings,
                "input": str(args.input),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "count": len(results),
                "applied_count": applied_result_count,
                "input_row_count": len(rows),
                "deduplicated_apply_count": len(apply_rows) if bool(args.apply) and not blocked_by_warnings else 0,
                "notification_count": len(notification_results),
                "notifications": notification_results,
                "skipped_duplicate_apply_count": sum(
                    1
                    for result in results
                    if bool(result.get("skipped_by_duplicate_apply"))
                ),
                "warning_count": len(warnings),
                "warning_codes": warning_codes,
                "blocking_warning_count": len(warnings) if blocked_by_warnings else 0,
                "blocking_warning_codes": warning_codes if blocked_by_warnings else [],
                "warnings_as_errors": bool(args.warnings_as_errors),
                "blocked_by_warnings": blocked_by_warnings,
                **_assignment_control_plane_report_fields(store),
                "warnings": warnings,
                "assignments": results,
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="assignment import report")
            _print_json(payload)
            return 2 if blocked_by_warnings else 0
        if args.command == "add-task":
            assignment = store.get_assignment(args.recording_id)
            warnings: list[dict[str, object]] = []
            if assignment is None:
                warnings.append(
                    {
                        "code": "task_recording_missing_assignment",
                        "recording_id": args.recording_id,
                        "details": "Task recording has no assignment, so the task will not be visible to a labeler.",
                    }
                )
            elif str(assignment.get("status") or "") != "active":
                warnings.append(
                    {
                        "code": "task_recording_assignment_not_active",
                        "recording_id": args.recording_id,
                        "assignment_status": assignment.get("status"),
                        "assignee_user": assignment.get("assignee_user"),
                        "details": "Task recording assignment is not active, so the task will not be available to the labeler.",
                    }
                )
            warning_codes = sorted(
                {
                    str(warning.get("code") or "")
                    for warning in warnings
                    if str(warning.get("code") or "")
                }
            )
            blocked_by_warnings = bool(warnings) and bool(args.warnings_as_errors)
            task_row = None
            if not blocked_by_warnings:
                task_row = store.upsert_task(
                    task_id=args.task_id,
                    recording_id=args.recording_id,
                    workflow_kind=args.workflow_kind,
                    dataset_id=args.dataset_id,
                    zarr_use=args.zarr_use,
                    stage_group=args.stage_group,
                    run_name=args.run_name,
                    component_name=args.component_name,
                    title=args.title,
                    scope=_parse_scope(args.scope_json),
                    state=args.state,
                    priority=int(args.priority),
                    notes=args.notes,
                    actor_user=args.actor,
                )
            payload = {
                "ok": not blocked_by_warnings,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "applied": not blocked_by_warnings,
                "warning_count": len(warnings),
                "warning_codes": warning_codes,
                "blocking_warning_count": len(warnings) if blocked_by_warnings else 0,
                "blocking_warning_codes": warning_codes if blocked_by_warnings else [],
                "warnings_as_errors": bool(args.warnings_as_errors),
                "blocked_by_warnings": blocked_by_warnings,
                "warnings": warnings,
                "task": task_row,
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="single-task report")
            _print_json(payload)
            return 2 if blocked_by_warnings else 0
        if args.command == "import-tasks":
            rows = _parse_task_manifest(args.input)
            results: list[dict[str, object]] = []
            warnings: list[dict[str, object]] = []
            row_warnings_by_index: dict[int, list[dict[str, object]]] = {}
            assignments_by_recording = {
                str(assignment.get("recording_id") or ""): assignment
                for assignment in store.list_assignments(status=None)
                if str(assignment.get("recording_id") or "")
            }
            logical_rows: dict[str, list[tuple[int, dict[str, object]]]] = {}
            for row_index, row in enumerate(rows):
                recording_id = str(row.get("recording_id") or "")
                assignment = assignments_by_recording.get(recording_id)
                if assignment is None:
                    warning = {
                        "code": "task_recording_missing_assignment",
                        "task_id": row.get("task_id"),
                        "recording_id": recording_id,
                        **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                        "details": "Task recording has no assignment, so the task will not be visible to a labeler.",
                    }
                    row_warnings_by_index.setdefault(row_index, []).append(warning)
                    warnings.append(warning)
                elif str(assignment.get("status") or "") != "active":
                    warning = {
                        "code": "task_recording_assignment_not_active",
                        "task_id": row.get("task_id"),
                        "recording_id": recording_id,
                        "assignment_status": assignment.get("status"),
                        **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                        "details": "Task recording assignment is not active, so the task will not be available to the labeler.",
                    }
                    row_warnings_by_index.setdefault(row_index, []).append(warning)
                    warnings.append(warning)
                logical_key_payload = {
                    "recording_id": recording_id,
                    "workflow_kind": str(row.get("workflow_kind") or ""),
                    "dataset_id": row.get("dataset_id"),
                    "zarr_use": row.get("zarr_use"),
                    "stage_group": row.get("stage_group"),
                    "run_name": row.get("run_name"),
                    "component_name": row.get("component_name"),
                    "scope": row.get("scope") if row.get("scope") is not None else {},
                }
                logical_key = json.dumps(logical_key_payload, sort_keys=True, separators=(",", ":"))
                logical_rows.setdefault(logical_key, []).append((row_index, row))
            for duplicate_rows in logical_rows.values():
                if len(duplicate_rows) <= 1:
                    continue
                warning = {
                    "code": "duplicate_logical_task_scope",
                    "recording_id": duplicate_rows[0][1].get("recording_id"),
                    "workflow_kind": duplicate_rows[0][1].get("workflow_kind"),
                    "task_ids": [row.get("task_id") for _, row in duplicate_rows],
                    "source_lines": [
                        int(row["_source_line"])
                        for _, row in duplicate_rows
                        if row.get("_source_line") is not None
                    ],
                    "details": "Multiple task IDs point at the same recording/workflow/component/run/scope; confirm this duplicate logical work is intentional.",
                }
                warnings.append(warning)
                for row_index, _ in duplicate_rows:
                    row_warnings_by_index.setdefault(row_index, []).append(warning)
            blocked_by_warnings = bool(warnings) and bool(args.warnings_as_errors)
            for row_index, row in enumerate(rows):
                task_id = str(row["task_id"])
                existing = store.get_task(task_id)
                target = {
                    "task_id": task_id,
                    "recording_id": str(row["recording_id"]),
                    "workflow_kind": str(row["workflow_kind"]),
                    "dataset_id": row.get("dataset_id"),
                    "zarr_use": row.get("zarr_use"),
                    "stage_group": row.get("stage_group"),
                    "run_name": row.get("run_name"),
                    "component_name": row.get("component_name"),
                    "title": row.get("title"),
                    "scope": row.get("scope") if row.get("scope") is not None else {},
                    "state": str(row.get("state") or "pending"),
                    "priority": int(row.get("priority") or 0),
                    "notes": row.get("notes"),
                }
                existing_target = (
                    {
                        "task_id": existing.get("task_id"),
                        "recording_id": existing.get("recording_id"),
                        "workflow_kind": existing.get("workflow_kind"),
                        "dataset_id": existing.get("dataset_id"),
                        "zarr_use": existing.get("zarr_use"),
                        "stage_group": existing.get("stage_group"),
                        "run_name": existing.get("run_name"),
                        "component_name": existing.get("component_name"),
                        "title": existing.get("title"),
                        "scope": existing.get("scope") if existing.get("scope") is not None else {},
                        "state": existing.get("state"),
                        "priority": int(existing.get("priority") or 0),
                        "notes": existing.get("notes"),
                    }
                    if existing is not None
                    else None
                )
                if bool(args.apply) and not blocked_by_warnings:
                    task_row = store.upsert_task(
                        task_id=task_id,
                        recording_id=str(target["recording_id"]),
                        workflow_kind=str(target["workflow_kind"]),
                        dataset_id=target.get("dataset_id"),
                        zarr_use=target.get("zarr_use"),
                        stage_group=target.get("stage_group"),
                        run_name=target.get("run_name"),
                        component_name=target.get("component_name"),
                        title=target.get("title"),
                        scope=target.get("scope") if isinstance(target.get("scope"), (Mapping, Sequence)) else {},
                        state=str(target["state"]),
                        priority=int(target["priority"]),
                        notes=target.get("notes"),
                        actor_user=args.actor,
                    )
                else:
                    task_row = target
                results.append(
                    {
                        "task_id": task_id,
                        "existing": existing,
                        "task": task_row,
                        "would_change": existing_target != target,
                        "warnings": row_warnings_by_index.get(row_index, []),
                        **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                    }
                )
            warning_codes = sorted(
                {
                    str(warning.get("code") or "")
                    for warning in warnings
                    if str(warning.get("code") or "")
                }
            )
            payload = {
                "ok": not blocked_by_warnings,
                "dry_run": not bool(args.apply) or blocked_by_warnings,
                "input": str(args.input),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "count": len(results),
                "applied_count": len(results) if bool(args.apply) and not blocked_by_warnings else 0,
                "warning_count": len(warnings),
                "warning_codes": warning_codes,
                "blocking_warning_count": len(warnings) if blocked_by_warnings else 0,
                "blocking_warning_codes": warning_codes if blocked_by_warnings else [],
                "warnings_as_errors": bool(args.warnings_as_errors),
                "blocked_by_warnings": blocked_by_warnings,
                "warnings": warnings,
                "tasks": results,
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="task import report")
            _print_json(payload)
            return 2 if blocked_by_warnings else 0
        if args.command == "set-task-state":
            updated = store.update_task_state(
                task_id=args.task_id,
                state=args.state,
                user=args.user,
            )
            payload = {
                "ok": True,
                "task": updated,
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="task-state report")
            _print_json(payload)
            return 0
        if args.command == "list":
            payload = {
                "ok": True,
                "assignments": store.list_assignments(assignee_user=args.user, status=None),
                "tasks": store.list_tasks(
                    recording_id=args.recording_id,
                    assignee_user=args.user,
                    include_completed=bool(args.include_completed),
                ),
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="assignment/task listing report")
            _print_json(payload)
            return 0
        if args.command == "work-summary":
            work = store.task_summary_for_user(args.user, include_completed=bool(args.include_completed))
            work["include_completed"] = bool(args.include_completed)
            check_report = _store_consistency_report(store)
            _add_work_summary_fields(
                work,
                reassignment_session_safety=check_report.get("reassignment_session_safety", {}),
            )
            known_user_status = _known_labeler_status(store, str(args.user))
            assignment_ownership_integrity = _assignment_ownership_integrity(
                store.list_assignments(assignee_user=str(args.user), status="active"),
                schema_integrity=store.assignment_schema_integrity(),
            )
            work_summary_store_consistency = dict(check_report)
            if isinstance(check_report.get("assignment_ownership_integrity"), Mapping):
                work_summary_store_consistency["global_assignment_ownership_integrity"] = dict(
                    check_report["assignment_ownership_integrity"]
                )
            work_summary_store_consistency["assignment_ownership_integrity"] = assignment_ownership_integrity
            labeler_route_authorization_policy = _labeler_route_authorization_policy()
            operator_validation_fields = _dashboard_operator_validation_fields(
                checklist_path=args.operator_validation_checklist,
                operator_launch_approved=bool(args.operator_launch_approved),
            )
            operator_validation_public_fields = _operator_validation_public_fields(
                operator_validation_fields
            )
            operator_validation_command_templates = _operator_validation_command_templates(
                operator_validation_public_fields.get(
                    "operator_validation_required_missing_evidence_gate_ids"
                )
                if isinstance(
                    operator_validation_public_fields.get(
                        "operator_validation_required_missing_evidence_gate_ids"
                    ),
                    list,
                )
                else None
            )
            safe_share_gate = _safe_share_gate_policy()
            safe_share_fields = _safe_share_gate_flat_fields(safe_share_gate)
            safe_share_checklist_fields = _safe_share_checklist_gate_status_fields_from_operator_validation(
                operator_validation_public_fields,
                safe_share_gate=safe_share_gate,
            )
            work["known_user_status"] = known_user_status
            work["labeler_landing_page_path"] = "/"
            work["labeling_home_page_path"] = LABELING_HOME_PATH
            work["dashboard_path"] = DASHBOARD_PATH
            work["dataset_queue_page_path"] = DATASET_QUEUE_PATH
            work["personal_work_page_path"] = PERSONAL_WORK_PATH
            work["personal_dataset_queue_page_path"] = PERSONAL_DATASET_QUEUE_PATH
            work["expected_user_labeler_landing_url"] = _dashboard_url_for_expected_user(
                "/",
                str(args.user),
            )
            work["expected_user_labeling_home_url"] = _dashboard_url_for_expected_user(
                LABELING_HOME_PATH,
                str(args.user),
            )
            work["expected_user_dashboard_url"] = _dashboard_url_for_expected_user(
                DASHBOARD_PATH,
                str(args.user),
            )
            work["expected_user_dataset_queue_url"] = _dashboard_url_for_expected_user(
                DATASET_QUEUE_PATH,
                str(args.user),
            )
            work["expected_user_personal_work_url"] = _dashboard_url_for_expected_user(
                PERSONAL_WORK_PATH,
                str(args.user),
            )
            work["expected_user_personal_dataset_queue_url"] = _dashboard_url_for_expected_user(
                PERSONAL_DATASET_QUEUE_PATH,
                str(args.user),
            )
            work["expected_user_identity_probe_url"] = _dashboard_url_for_expected_user(
                IDENTITY_PROBE_PATH,
                str(args.user),
            )
            work["preferred_labeler_entrypoint"] = "personal_datasets_waiting_queue"
            work["preferred_labeler_entry_url"] = work[
                "expected_user_personal_dataset_queue_url"
            ]
            work["personalized_labeler_entrypoint"] = "personal_datasets_waiting_queue"
            work["personalized_labeler_entry_url"] = work[
                "expected_user_personal_dataset_queue_url"
            ]
            work["personal_dataset_queue_link_role"] = "preferred_queue"
            work["dataset_queue_link_role"] = "canonical_queue_fallback"
            work["canonical_dataset_queue_link_role"] = "canonical_queue_fallback"
            single_owner_policy = _assignment_ownership_policy()
            work["single_owner_policy"] = single_owner_policy
            work.update(_single_owner_policy_fields(single_owner_policy))
            work["assignment_ownership_integrity"] = assignment_ownership_integrity
            work.update(
                _single_owner_assignment_live_contract_fields(
                    store,
                    integrity=assignment_ownership_integrity,
                )
            )
            labeler_route_authorization_checklist = _labeler_route_authorization_runtime_checklist(
                policy=labeler_route_authorization_policy,
                user=str(args.user),
                expected_user=str(args.user),
                known_user_status=known_user_status,
                assignment_ownership_contract=work["assignment_ownership_contract"],
            )
            work["labeler_route_authorization_policy"] = labeler_route_authorization_policy
            work["labeler_route_authorization_checklist"] = labeler_route_authorization_checklist
            work.update(operator_validation_public_fields)
            operator_validation_gate_fields = _operator_validation_gate_flat_fields(work)
            work.update(operator_validation_gate_fields)
            safe_share_checklist_fields = _safe_share_checklist_gate_status_fields_from_operator_validation(
                work,
                safe_share_gate=safe_share_gate,
            )
            work["operator_validation_command_templates"] = operator_validation_command_templates
            work["operator_validation_visibility_policy"] = (
                _operator_validation_visibility_policy()
            )
            work["safe_share_gate"] = safe_share_gate
            work.update(safe_share_fields)
            work.update(safe_share_checklist_fields)
            work["browser_mutation_write_policy"] = _browser_mutation_write_policy()
            work["browser_mutation_write_checklist"] = _browser_mutation_write_runtime_checklist()
            work["dataset_queue_direct_start_policy"] = _dataset_queue_direct_start_policy()
            work["runtime_operator_validation_gate_cli_policy"] = _runtime_operator_validation_gate_cli_policy()
            work.update(
                _runtime_operator_validation_gate_cli_policy_fields(
                    work["runtime_operator_validation_gate_cli_policy"]
                )
            )
            labeler_safety = _labeler_safety_policy()
            queue_first_entry_contract = _queue_first_entry_contract_policy(
                labeler_safety=labeler_safety,
                labeler_landing_page_path="/",
                labeler_landing_url="/",
                expected_user_labeler_landing_url=work["expected_user_labeler_landing_url"],
                labeling_home_page_path=LABELING_HOME_PATH,
                labeling_home_url=LABELING_HOME_PATH,
                expected_user_labeling_home_url=work["expected_user_labeling_home_url"],
                dataset_queue_page_path=DATASET_QUEUE_PATH,
                dataset_queue_url=DATASET_QUEUE_PATH,
                expected_user_dataset_queue_url=work["expected_user_dataset_queue_url"],
                dashboard_url=DASHBOARD_PATH,
                expected_user_dashboard_url=work["expected_user_dashboard_url"],
                personal_dataset_queue_page_path=PERSONAL_DATASET_QUEUE_PATH,
                personal_dataset_queue_url=PERSONAL_DATASET_QUEUE_PATH,
                expected_user_personal_dataset_queue_url=work[
                    "expected_user_personal_dataset_queue_url"
                ],
                personal_work_page_path=PERSONAL_WORK_PATH,
                personal_work_url=PERSONAL_WORK_PATH,
                expected_user_personal_work_url=work["expected_user_personal_work_url"],
            )
            work["queue_first_entry_contract"] = queue_first_entry_contract
            payload = {
                "ok": True,
                "store_path": str(store_path),
                "include_completed": bool(args.include_completed),
                "known_user_status": known_user_status,
                "single_owner_policy": single_owner_policy,
                **_single_owner_policy_fields(single_owner_policy),
                "assignment_ownership_integrity": assignment_ownership_integrity,
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
                "store_consistency": work_summary_store_consistency,
                "labeler_landing_page_path": work["labeler_landing_page_path"],
                "labeling_home_page_path": work["labeling_home_page_path"],
                "dashboard_path": work["dashboard_path"],
                "dataset_queue_page_path": work["dataset_queue_page_path"],
                "personal_work_page_path": work["personal_work_page_path"],
                "personal_dataset_queue_page_path": work["personal_dataset_queue_page_path"],
                "expected_user_labeler_landing_url": work["expected_user_labeler_landing_url"],
                "expected_user_labeling_home_url": work["expected_user_labeling_home_url"],
                "expected_user_dashboard_url": work["expected_user_dashboard_url"],
                "expected_user_dataset_queue_url": work["expected_user_dataset_queue_url"],
                "expected_user_personal_work_url": work["expected_user_personal_work_url"],
                "expected_user_personal_dataset_queue_url": work[
                    "expected_user_personal_dataset_queue_url"
                ],
                "expected_user_identity_probe_url": work["expected_user_identity_probe_url"],
                "preferred_labeler_entrypoint": work.get(
                    "preferred_labeler_entrypoint",
                    "personal_datasets_waiting_queue",
                ),
                            "preferred_labeler_entry_url": work.get(
                                "preferred_labeler_entry_url",
                                work.get("expected_user_personal_dataset_queue_url", ""),
                            ),
                            "personalized_labeler_entrypoint": work.get(
                                "personalized_labeler_entrypoint",
                                "personal_datasets_waiting_queue",
                            ),
                            "personalized_labeler_entry_url": work.get(
                                "personalized_labeler_entry_url",
                                work.get("expected_user_personal_dataset_queue_url", ""),
                            ),
                            "preferred_labeler_entry_url_matches_dataset_queue": bool(
                                work.get("preferred_labeler_entry_url_matches_dataset_queue")
                            ),
                            "preferred_labeler_entry_url_matches_personal_dataset_queue": bool(
                                work.get(
                                    "preferred_labeler_entry_url_matches_personal_dataset_queue"
                                )
                            ),
                            "personalized_labeler_entry_url_matches_personal_dataset_queue": bool(
                                work.get(
                                    "personalized_labeler_entry_url_matches_personal_dataset_queue"
                                )
                            ),
                            "personal_dataset_queue_link_role": work.get(
                                "personal_dataset_queue_link_role", "preferred_queue"
                            ),
                            "dataset_queue_link_role": work.get(
                                "dataset_queue_link_role", "canonical_queue_fallback"
                            ),
                            "canonical_dataset_queue_link_role": work.get(
                                "canonical_dataset_queue_link_role",
                                "canonical_queue_fallback",
                            ),
                "labeler_safety": labeler_safety,
                "queue_first_entry_contract": queue_first_entry_contract,
                "labeler_route_authorization_policy": labeler_route_authorization_policy,
                "labeler_route_authorization_checklist": labeler_route_authorization_checklist,
                "operator_authorization_policy": _operator_authorization_policy(),
                "operator_recovery_policy": _operator_recovery_policy(),
                "operator_validation_visibility_policy": _operator_validation_visibility_policy(),
                "operator_validation_command_templates": operator_validation_command_templates,
                "operator_validation_checklist_path": str(
                    operator_validation_fields.get("operator_validation_checklist_path") or ""
                ),
                "safe_share_gate": safe_share_gate,
                **safe_share_fields,
                **safe_share_checklist_fields,
                "task_state_policy": _browser_task_state_policy(),
                "zarr_backup_policy": _zarr_backup_policy(),
                "mutation_audit_policy": _mutation_audit_policy(),
                "browser_mutation_write_policy": _browser_mutation_write_policy(),
                "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(),
                "dataset_queue_direct_start_policy": _dataset_queue_direct_start_policy(),
                "runtime_operator_validation_gate_cli_policy": _runtime_operator_validation_gate_cli_policy(),
                **_runtime_operator_validation_gate_cli_policy_fields(
                    work["runtime_operator_validation_gate_cli_policy"]
                ),
                "browser_response_security_policy": _browser_response_security_policy(),
                "session_guard_policy": _session_guard_policy(),
                "signed_link_policy": _browser_signed_link_policy(),
                "browser_workflows": _browser_workflow_capabilities(),
                **operator_validation_public_fields,
                **operator_validation_gate_fields,
                "work": work,
            }
            _add_payload_contract_compact_fields(payload)
            if isinstance(payload.get("personalized_launch_readiness"), Mapping):
                work["personalized_launch_readiness"] = dict(
                    payload["personalized_launch_readiness"]
                )
                payload["work"] = work
            final_safe_share_checklist_fields = _safe_share_checklist_gate_status_fields_from_operator_validation(
                work,
                safe_share_gate=safe_share_gate,
            )
            work.update(final_safe_share_checklist_fields)
            payload.update(final_safe_share_checklist_fields)
            payload["work"] = work
            payload["personalized_launch_readiness"] = _personalized_launch_readiness_summary(
                payload
            )
            work["personalized_launch_readiness"] = dict(
                payload["personalized_launch_readiness"]
            )
            payload["work"] = work
            if args.output:
                output_path = Path(args.output)
                if output_path.exists() and not bool(args.overwrite):
                    raise FileExistsError(f"Refusing to overwrite existing work summary export: {output_path}")
                output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                summary_safe_share_fields = dict(safe_share_checklist_fields)
                nested_work_payload = payload.get("work") if isinstance(payload.get("work"), Mapping) else {}
                if int(nested_work_payload.get("safe_share_launch_blocking_next_action_count") or 0) > int(
                    summary_safe_share_fields.get("safe_share_launch_blocking_next_action_count") or 0
                ):
                    summary_safe_share_fields = dict(nested_work_payload)
                _print_json(
                    {
                        "ok": True,
                        "store_path": str(store_path),
                        "output_path": str(output_path),
                        "include_completed": bool(args.include_completed),
                        "recording_count": len(work.get("recordings", [])),
                        "task_count": int(work.get("task_count", 0)),
                        "known_user": bool(known_user_status.get("is_known_labeler")),
                        "labeler_landing_page_path": str(
                            work.get("labeler_landing_page_path") or ""
                        ),
                        "dashboard_path": str(work.get("dashboard_path") or ""),
                        "dataset_queue_page_path": str(
                            work.get("dataset_queue_page_path") or ""
                        ),
                        "personal_work_page_path": str(
                            work.get("personal_work_page_path") or ""
                        ),
                        "personal_dataset_queue_page_path": str(
                            work.get("personal_dataset_queue_page_path") or ""
                        ),
                        "expected_user_labeler_landing_url": str(
                            work.get("expected_user_labeler_landing_url") or ""
                        ),
                        "expected_user_labeling_home_url": str(
                            work.get("expected_user_labeling_home_url") or ""
                        ),
                        "expected_user_dashboard_url": str(
                            work.get("expected_user_dashboard_url") or ""
                        ),
                        "expected_user_dataset_queue_url": str(
                            work.get("expected_user_dataset_queue_url") or ""
                        ),
                        "expected_user_personal_work_url": str(
                            work.get("expected_user_personal_work_url") or ""
                        ),
                        "expected_user_personal_dataset_queue_url": str(
                            work.get("expected_user_personal_dataset_queue_url") or ""
                        ),
                        "expected_user_identity_probe_url": str(
                            work.get("expected_user_identity_probe_url") or ""
                        ),
                        "preferred_labeler_entrypoint": str(
                            work.get("preferred_labeler_entrypoint") or ""
                        ),
                        "preferred_labeler_entry_url": str(
                            work.get("preferred_labeler_entry_url") or ""
                        ),
                        "personalized_labeler_entrypoint": str(
                            work.get("personalized_labeler_entrypoint") or ""
                        ),
                        "personalized_labeler_entry_url": str(
                            work.get("personalized_labeler_entry_url") or ""
                        ),
                        "active_assignment_count": int(
                            known_user_status.get("active_assignment_count") or 0
                        ),
                        "assignment_ownership_ok": bool(
                            assignment_ownership_integrity.get("ok")
                        ),
                        "assignment_schema_enforced_recording_primary_key": bool(
                            assignment_ownership_integrity.get(
                                "schema_enforced_recording_primary_key"
                            )
                        ),
                        "assignment_duplicate_active_owner_count": int(
                            assignment_ownership_integrity.get(
                                "duplicate_active_owner_count"
                            )
                            or 0
                        ),
                        **_single_owner_policy_fields(
                            payload["single_owner_policy"]
                            if isinstance(payload.get("single_owner_policy"), Mapping)
                            else None
                        ),
                        **_assignment_ownership_contract_fields(
                            work["assignment_ownership_contract"]
                        ),
                        "single_owner_policy_contract_met": bool(
                            work.get("single_owner_policy_contract_met")
                        ),
                        "store_consistency_ok": bool(check_report.get("ok")),
                        "reassignment_session_safety_ok": bool(
                            (
                                check_report.get("reassignment_session_safety")
                                if isinstance(check_report.get("reassignment_session_safety"), Mapping)
                                else {}
                            ).get("ok", True)
                        ),
                        "reassignment_session_safety_mismatch_count": int(
                            (
                                check_report.get("reassignment_session_safety")
                                if isinstance(check_report.get("reassignment_session_safety"), Mapping)
                                else {}
                            ).get("active_session_assignment_mismatch_count")
                            or 0
                        ),
                        "reassignment_session_safety_blocks_labeler_mutation": bool(
                            (
                                check_report.get("reassignment_session_safety")
                                if isinstance(check_report.get("reassignment_session_safety"), Mapping)
                                else {}
                            ).get("blocks_labeler_mutation")
                        ),
                        "store_consistency_issue_count": int(
                            check_report.get("issue_count") or 0
                        ),
                        "store_consistency_issue_codes": [
                            str(issue.get("code") or "")
                            for issue in (
                                check_report.get("issues")
                                if isinstance(check_report.get("issues"), list)
                                else []
                            )
                            if isinstance(issue, Mapping) and str(issue.get("code") or "")
                        ],
                        "store_consistency_warning_count": int(
                            check_report.get("warning_count") or 0
                        ),
                        "store_consistency_warning_codes": [
                            str(warning.get("code") or "")
                            for warning in (
                                check_report.get("warnings")
                                if isinstance(check_report.get("warnings"), list)
                                else []
                            )
                            if isinstance(warning, Mapping) and str(warning.get("code") or "")
                        ],
                        "store_consistency_blocking_warning_count": int(
                            check_report.get("blocking_warning_count") or 0
                        ),
                        "store_consistency_blocking_warning_codes": check_report.get(
                            "blocking_warning_codes", []
                        ),
                        "labeler_start_ready": bool(work.get("labeler_start_ready")),
                        "labeler_start_status": str(work.get("labeler_start_status") or ""),
                        "labeler_action": str(work.get("labeler_action") or ""),
                        "labeler_start_message": str(work.get("labeler_start_message") or ""),
                        "labeler_start_operator_action": str(
                            work.get("labeler_start_operator_action") or ""
                        ),
                        "waiting_dataset_count": int(
                            (
                                work.get("dataset_queue_summary")
                                if isinstance(work.get("dataset_queue_summary"), Mapping)
                                else {}
                            ).get("waiting_dataset_count")
                            or 0
                        ),
                        "dataset_open_task_count": int(
                            (
                                work.get("dataset_queue_summary")
                                if isinstance(work.get("dataset_queue_summary"), Mapping)
                                else {}
                            ).get("open_task_count")
                            or 0
                        ),
                        "recordings_without_open_tasks": _count_recordings_without_open_tasks(work),
                        "recordings_without_open_tasks_by_reason": _count_recordings_without_open_tasks_by_reason(work),
                        "recordings_without_open_tasks_actions": _recordings_without_open_tasks_actions(
                            _count_recordings_without_open_tasks_by_reason(work)
                        ),
                        "labeler_route_authorization_ready": bool(
                            labeler_route_authorization_checklist.get("ready")
                        ),
                        "labeler_route_authorization_active_assignment_required": bool(
                            labeler_route_authorization_checklist.get("active_assignment_required")
                        ),
                        "labeler_route_authorization_active_assignment_count": int(
                            labeler_route_authorization_checklist.get("active_assignment_count") or 0
                        ),
                        "labeler_route_authorization_has_active_assignment": bool(
                            labeler_route_authorization_checklist.get("has_active_assignment")
                        ),
                        "operator_admin_routes_require_operator": bool(
                            payload["operator_authorization_policy"].get(
                                "admin_routes_require_operator"
                            )
                        ),
                        "operator_boundary_ready": bool(
                            payload["operator_authorization_policy"].get(
                                "operator_boundary_ready"
                            )
                        ),
                        "operator_recovery_task_reopen_operator_only": bool(
                            payload["operator_recovery_policy"].get(
                                "task_reopen_operator_only"
                            )
                        ),
                        "operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update": bool(
                            payload["operator_recovery_policy"].get(
                                "reassignment_closes_previous_owner_sessions_before_assignment_update"
                            )
                        ),
                        "operator_recovery_reassignment_target_validated_before_session_closure": bool(
                            payload["operator_recovery_policy"].get(
                                "reassignment_target_validated_before_session_closure"
                            )
                        ),
                        "operator_recovery_session_closure_and_assignment_update_atomic": bool(
                            payload["operator_recovery_policy"].get(
                                "session_closure_and_assignment_update_atomic"
                            )
                        ),
                        "operator_recovery_failed_promotion_retry_operator_only": bool(
                            payload["operator_recovery_policy"].get(
                                "failed_promotion_retry_operator_only"
                            )
                        ),
                        "operator_validation_all_complete": bool(
                            payload.get("operator_validation_all_complete")
                        ),
                        "operator_validation_declared_all_complete": bool(
                            payload.get("operator_validation_declared_all_complete")
                        ),
                        "operator_validation_status": str(
                            payload.get("operator_validation_status") or ""
                        ),
                        "operator_validation_gate_count": int(
                            payload.get("operator_validation_gate_count") or 0
                        ),
                        "operator_validation_pending_gate_ids": [
                            str(gate_id)
                            for gate_id in (
                                payload.get("operator_validation_pending_gate_ids")
                                if isinstance(
                                    payload.get("operator_validation_pending_gate_ids"),
                                    list,
                                )
                                else []
                            )
                        ],
                        "operator_validation_required_missing_evidence_gate_ids": [
                            str(gate_id)
                            for gate_id in (
                                payload.get(
                                    "operator_validation_required_missing_evidence_gate_ids"
                                )
                                if isinstance(
                                    payload.get(
                                        "operator_validation_required_missing_evidence_gate_ids"
                                    ),
                                    list,
                                )
                                else []
                            )
                        ],
                        "operator_validation_required_pending_gate_count": int(
                            payload.get("operator_validation_required_pending_gate_count") or 0
                        ),
                        "operator_validation_needs_review_gate_count": int(
                            payload.get("operator_validation_needs_review_gate_count") or 0
                        ),
                        "operator_validation_required_missing_evidence_gate_count": int(
                            payload.get(
                                "operator_validation_required_missing_evidence_gate_count"
                            )
                            or 0
                        ),
                        "operator_validation_gate_status_values": list(
                            OPERATOR_VALIDATION_GATE_STATUS_VALUES
                        ),
                        "operator_validation_gate_ids": list(
                            DEFAULT_OPERATOR_VALIDATION_GATE_IDS
                        ),
                        "operator_validation_gate_flat_field_suffixes": [
                            "status",
                            "pending",
                            "missing_evidence",
                            "needs_review",
                            "passed",
                        ],
                        **_operator_validation_gate_flat_fields(payload),
                        "operator_validation_source": str(
                            payload.get("operator_validation_source") or ""
                        ),
                        "operator_validation_operator_action": str(
                            payload.get("operator_validation_operator_action") or ""
                        ),
                        "operator_validation_external_evidence_required": bool(
                            payload.get("operator_validation_external_evidence_required")
                        ),
                        "operator_validation_external_evidence_required_gate_ids": [
                            str(gate_id)
                            for gate_id in (
                                payload.get(
                                    "operator_validation_external_evidence_required_gate_ids"
                                )
                                if isinstance(
                                    payload.get(
                                        "operator_validation_external_evidence_required_gate_ids"
                                    ),
                                    list,
                                )
                                else []
                            )
                        ],
                        "operator_validation_external_evidence_required_gate_count": int(
                            payload.get(
                                "operator_validation_external_evidence_required_gate_count"
                            )
                            or 0
                        ),
                        "safe_share_external_launch_evidence_gap_count": int(
                            summary_safe_share_fields.get(
                                "safe_share_external_launch_evidence_gap_count"
                            )
                            or 0
                        ),
                        "safe_share_external_launch_evidence_gap_gate_ids": [
                            str(gate_id)
                            for gate_id in (
                                summary_safe_share_fields.get(
                                    "safe_share_external_launch_evidence_gap_gate_ids"
                                )
                                if isinstance(
                                    summary_safe_share_fields.get(
                                        "safe_share_external_launch_evidence_gap_gate_ids"
                                    ),
                                    list,
                                )
                                else []
                            )
                        ],
                        "safe_share_launch_blocking_next_action_count": int(
                            summary_safe_share_fields.get(
                                "safe_share_launch_blocking_next_action_count"
                            )
                            or 0
                        ),
                        "safe_share_launch_blocking_next_action_detail_fields": [
                            str(field)
                            for field in (
                                summary_safe_share_fields.get(
                                    "safe_share_launch_blocking_next_action_detail_fields"
                                )
                                if isinstance(
                                    summary_safe_share_fields.get(
                                        "safe_share_launch_blocking_next_action_detail_fields"
                                    ),
                                    list,
                                )
                                else []
                            )
                        ],
                        "safe_share_launch_blocking_next_action_command_fields": [
                            str(field)
                            for field in (
                                summary_safe_share_fields.get(
                                    "safe_share_launch_blocking_next_action_command_fields"
                                )
                                if isinstance(
                                    summary_safe_share_fields.get(
                                        "safe_share_launch_blocking_next_action_command_fields"
                                    ),
                                    list,
                                )
                                else []
                            )
                        ],
                        "safe_share_next_action_summary": str(
                            summary_safe_share_fields.get(
                                "safe_share_next_action_summary"
                            )
                            or ""
                        ),
                        "safe_share_launch_blocking_next_action_ids": [
                            str(action.get("id") or "")
                            for action in (
                                summary_safe_share_fields.get(
                                    "safe_share_launch_blocking_next_actions"
                                )
                                if isinstance(
                                    summary_safe_share_fields.get(
                                        "safe_share_launch_blocking_next_actions"
                                    ),
                                    list,
                                )
                                else []
                            )
                            if isinstance(action, Mapping)
                            and str(action.get("id") or "")
                        ],
                        "safe_share_launch_blocking_next_actions": [
                            dict(action)
                            for action in (
                                summary_safe_share_fields.get(
                                    "safe_share_launch_blocking_next_actions"
                                )
                                if isinstance(
                                    summary_safe_share_fields.get(
                                        "safe_share_launch_blocking_next_actions"
                                    ),
                                    list,
                                )
                                else []
                            )
                            if isinstance(action, Mapping)
                        ],
                        "operator_validation_external_evidence_template_fields_by_gate_id": dict(
                            payload.get(
                                "operator_validation_external_evidence_template_fields_by_gate_id"
                            )
                            if isinstance(
                                payload.get(
                                    "operator_validation_external_evidence_template_fields_by_gate_id"
                                ),
                                Mapping,
                            )
                            else {}
                        ),
                        "operator_validation_external_evidence_template_paths_by_gate_id": dict(
                            payload.get(
                                "operator_validation_external_evidence_template_paths_by_gate_id"
                            )
                            if isinstance(
                                payload.get(
                                    "operator_validation_external_evidence_template_paths_by_gate_id"
                                ),
                                Mapping,
                            )
                            else {}
                        ),
                        "operator_validation_checklist_only_required_gate_ids": [
                            str(gate_id)
                            for gate_id in (
                                payload.get(
                                    "operator_validation_checklist_only_required_gate_ids"
                                )
                                if isinstance(
                                    payload.get(
                                        "operator_validation_checklist_only_required_gate_ids"
                                    ),
                                    list,
                                )
                                else []
                            )
                        ],
                        "operator_validation_checklist_only_required_gate_count": int(
                            payload.get(
                                "operator_validation_checklist_only_required_gate_count"
                            )
                            or 0
                        ),
                        "operator_validation_command_template_command_count": int(
                            operator_validation_command_templates.get("command_count") or 0
                        ),
                        "operator_validation_command_template_command_ids": [
                            str(command_id)
                            for command_id in (
                                operator_validation_command_templates.get("command_ids")
                                if isinstance(
                                    operator_validation_command_templates.get("command_ids"),
                                    list,
                                )
                                else []
                            )
                        ],
                        "operator_validation_command_template_template_backed_gate_ids": [
                            str(gate_id)
                            for gate_id in (
                                operator_validation_command_templates.get(
                                    "template_backed_gate_ids"
                                )
                                if isinstance(
                                    operator_validation_command_templates.get(
                                        "template_backed_gate_ids"
                                    ),
                                    list,
                                )
                                else []
                            )
                        ],
                        "operator_validation_command_template_validation_checklist_gate_ids": [
                            str(gate_id)
                            for gate_id in (
                                operator_validation_command_templates.get(
                                    "validation_checklist_gate_ids"
                                )
                                if isinstance(
                                    operator_validation_command_templates.get(
                                        "validation_checklist_gate_ids"
                                    ),
                                    list,
                                )
                                else []
                            )
                        ],
                        "operator_validation_command_template_apply_required_gate_ids": [
                            str(gate_id)
                            for gate_id in (
                                operator_validation_command_templates.get(
                                    "apply_required_gate_ids"
                                )
                                if isinstance(
                                    operator_validation_command_templates.get(
                                        "apply_required_gate_ids"
                                    ),
                                    list,
                                )
                                else []
                            )
                        ],
                        "operator_validation_command_template_evidence_template_fields_by_gate_id": dict(
                            operator_validation_command_templates.get(
                                "evidence_template_fields_by_gate_id"
                            )
                            if isinstance(
                                operator_validation_command_templates.get(
                                    "evidence_template_fields_by_gate_id"
                                ),
                                Mapping,
                            )
                            else {}
                        ),
                        "operator_validation_command_template_evidence_template_paths_by_gate_id": dict(
                            operator_validation_command_templates.get(
                                "evidence_template_paths_by_gate_id"
                            )
                            if isinstance(
                                operator_validation_command_templates.get(
                                    "evidence_template_paths_by_gate_id"
                                ),
                                Mapping,
                            )
                            else {}
                        ),
                        "operator_validation_command_template_launch_evidence_collection_plan_schema": str(
                            operator_validation_command_templates.get(
                                "launch_evidence_collection_plan_schema"
                            )
                            or ""
                        ),
                        "operator_validation_command_template_launch_evidence_collection_step_count": int(
                            operator_validation_command_templates.get(
                                "launch_evidence_collection_step_count"
                            )
                            or 0
                        ),
                        "operator_validation_command_template_launch_evidence_collection_gate_ids": [
                            str(gate_id)
                            for gate_id in (
                                operator_validation_command_templates.get(
                                    "launch_evidence_collection_gate_ids"
                                )
                                if isinstance(
                                    operator_validation_command_templates.get(
                                        "launch_evidence_collection_gate_ids"
                                    ),
                                    list,
                                )
                                else []
                            )
                        ],
                        "operator_validation_command_template_launch_evidence_collection_record_command_ids": [
                            str(command_id)
                            for command_id in (
                                operator_validation_command_templates.get(
                                    "launch_evidence_collection_record_command_ids"
                                )
                                if isinstance(
                                    operator_validation_command_templates.get(
                                        "launch_evidence_collection_record_command_ids"
                                    ),
                                    list,
                                )
                                else []
                            )
                        ],
                        "operator_validation_command_template_launch_evidence_collection_operator_only": bool(
                            operator_validation_command_templates.get(
                                "launch_evidence_collection_operator_only",
                                True,
                            )
                        ),
                        "operator_validation_command_template_launch_evidence_collection_required_final_field": str(
                            operator_validation_command_templates.get(
                                "launch_evidence_collection_required_final_field"
                            )
                            or ""
                        ),
                        "operator_validation_command_template_launch_evidence_collection_required_final_value": bool(
                            operator_validation_command_templates.get(
                                "launch_evidence_collection_required_final_value"
                            )
                        ),
                        "operator_validation_command_template_launch_evidence_collection_final_inspection_command": str(
                            operator_validation_command_templates.get(
                                "launch_evidence_collection_final_inspection_command"
                            )
                            or ""
                        ),
                        "operator_validation_command_template_commands_are_operator_only": bool(
                            operator_validation_command_templates.get(
                                "commands_are_operator_only",
                                True,
                            )
                        ),
                        "operator_validation_command_template_commands_are_labeler_instructions": bool(
                            operator_validation_command_templates.get(
                                "commands_are_labeler_instructions"
                            )
                        ),
                        "operator_validation_command_template_labelers_must_not_run_commands": bool(
                            operator_validation_command_templates.get(
                                "labelers_must_not_run_commands",
                                True,
                            )
                        ),
                        "operator_validation_operator_only_fields": [
                            str(field_name)
                            for field_name in (
                                payload["operator_validation_visibility_policy"].get(
                                    "operator_only_fields"
                                )
                                if isinstance(
                                    payload["operator_validation_visibility_policy"].get(
                                        "operator_only_fields"
                                    ),
                                    list,
                                )
                                else []
                            )
                        ],
                        "operator_validation_labeler_visible_payloads_include_operator_only_fields": bool(
                            payload["operator_validation_visibility_policy"].get(
                                "labeler_visible_payloads_include_operator_only_fields"
                            )
                        ),
                        "operator_validation_per_user_payloads_use_public_fields_only": bool(
                            payload["operator_validation_visibility_policy"].get(
                                "per_user_payloads_use_public_fields_only"
                            )
                        ),
                        "operator_validation_top_level_operator_reports_may_include_operator_only_fields": bool(
                            payload["operator_validation_visibility_policy"].get(
                                "top_level_operator_reports_may_include_operator_only_fields"
                            )
                        ),
                        "safe_share_gate": payload["safe_share_gate"],
                        **_safe_share_gate_flat_fields(
                            payload["safe_share_gate"]
                            if isinstance(payload.get("safe_share_gate"), Mapping)
                            else None
                        ),
                        **_safe_share_checklist_field_values(payload),
                        "runtime_operator_validation_gate_cli_policy": payload[
                            "runtime_operator_validation_gate_cli_policy"
                        ],
                        **_runtime_operator_validation_gate_cli_policy_fields(
                            payload["runtime_operator_validation_gate_cli_policy"]
                            if isinstance(
                                payload.get("runtime_operator_validation_gate_cli_policy"),
                                Mapping,
                            )
                            else None
                        ),
                        **_browser_mutation_target_contract_compact_fields(
                            payload["browser_mutation_write_checklist"]
                            if isinstance(
                                payload.get("browser_mutation_write_checklist"),
                                Mapping,
                            )
                            else None,
                            user=str(args.user),
                        ),
                        "browser_mutation_write_ready": bool(
                            payload["browser_mutation_write_checklist"].get("ready")
                        ),
                        "browser_mutation_label_mutation_target_kind": str(
                            payload["browser_mutation_write_checklist"].get(
                                "label_mutation_target_kind"
                            )
                            or ""
                        ),
                        "browser_mutation_browser_label_write_target": str(
                            payload["browser_mutation_write_checklist"].get(
                                "browser_label_write_target"
                            )
                            or ""
                        ),
                        "browser_mutation_csv_handoff_artifact_role": str(
                            payload["browser_mutation_write_checklist"].get(
                                "csv_handoff_artifact_role"
                            )
                            or ""
                        ),
                        "browser_mutation_csv_handoff_artifacts_are_label_write_targets": bool(
                            payload["browser_mutation_write_checklist"].get(
                                "csv_handoff_artifacts_are_label_write_targets"
                            )
                        ),
                        "browser_mutation_handoff_csv_artifacts_are_label_write_targets": bool(
                            payload["browser_mutation_write_checklist"].get(
                                "handoff_csv_artifacts_are_label_write_targets"
                            )
                        ),
                        "browser_mutation_intermediate_csv_artifacts_are_label_write_targets": bool(
                            payload["browser_mutation_write_checklist"].get(
                                "intermediate_csv_artifacts_are_label_write_targets"
                            )
                        ),
                        "browser_mutation_browser_writes_csv_or_handoff_files": bool(
                            payload["browser_mutation_write_checklist"].get(
                                "browser_writes_csv_or_handoff_files"
                            )
                        ),
                        "browser_mutation_browser_writes_handoff_csv": bool(
                            payload["browser_mutation_write_checklist"].get(
                                "browser_writes_handoff_csv"
                            )
                        ),
                        "browser_mutation_browser_writes_intermediate_csv": bool(
                            payload["browser_mutation_write_checklist"].get(
                                "browser_writes_intermediate_csv"
                            )
                        ),
                        "browser_mutation_browser_has_direct_zarr_write_authority": bool(
                            payload["browser_mutation_write_checklist"].get(
                                "browser_has_direct_zarr_write_authority"
                            )
                        ),
                        **_direct_browser_start_contract_compact_fields(
                            payload["dataset_queue_direct_start_policy"]
                            if isinstance(
                                payload.get("dataset_queue_direct_start_policy"),
                                Mapping,
                            )
                            else None,
                            user=str(args.user),
                        ),
                        "dataset_queue_direct_start_enabled": bool(
                            payload["dataset_queue_direct_start_policy"].get("enabled")
                        ),
                        "dataset_queue_direct_start_post_body_expected_user_required": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "post_body_expected_user_required"
                            )
                        ),
                        "dataset_queue_direct_start_post_body_expected_user_field": str(
                            payload["dataset_queue_direct_start_policy"].get(
                                "post_body_expected_user_field"
                            )
                            or ""
                        ),
                        "dataset_queue_direct_start_denied_start_returns_task_open_authorization_contract": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "denied_start_returns_task_open_authorization_contract"
                            )
                        ),
                        "dataset_queue_direct_start_denied_start_support_preserves_task_open_authorization_contract": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "denied_start_support_preserves_task_open_authorization_contract"
                            )
                        ),
                        "dataset_queue_direct_start_denied_start_support_includes_authorization_context": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "denied_start_support_includes_authorization_context"
                            )
                        ),
                        "dataset_queue_direct_start_denied_start_contract_reports_no_session_created": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "denied_start_contract_reports_no_session_created"
                            )
                        ),
                        "dataset_queue_direct_start_denied_start_contract_reports_server_authorizes_open_false": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "denied_start_contract_reports_server_authorizes_open_false"
                            )
                        ),
                        "dataset_queue_direct_start_label_mutation_target_kind": str(
                            payload["dataset_queue_direct_start_policy"].get(
                                "label_mutation_target_kind"
                            )
                            or ""
                        ),
                        "dataset_queue_direct_start_browser_label_write_target": str(
                            payload["dataset_queue_direct_start_policy"].get(
                                "browser_label_write_target"
                            )
                            or ""
                        ),
                        "dataset_queue_direct_start_csv_handoff_artifact_role": str(
                            payload["dataset_queue_direct_start_policy"].get(
                                "csv_handoff_artifact_role"
                            )
                            or ""
                        ),
                        "dataset_queue_direct_start_csv_handoff_artifacts_are_label_write_targets": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "csv_handoff_artifacts_are_label_write_targets"
                            )
                        ),
                        "dataset_queue_direct_start_handoff_csv_artifacts_are_label_write_targets": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "handoff_csv_artifacts_are_label_write_targets"
                            )
                        ),
                        "dataset_queue_direct_start_intermediate_csv_artifacts_are_label_write_targets": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "intermediate_csv_artifacts_are_label_write_targets"
                            )
                        ),
                        "dataset_queue_direct_start_browser_writes_csv_or_handoff_files": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "browser_writes_csv_or_handoff_files"
                            )
                        ),
                        "dataset_queue_direct_start_browser_writes_handoff_csv": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "browser_writes_handoff_csv"
                            )
                        ),
                        "dataset_queue_direct_start_browser_writes_intermediate_csv": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "browser_writes_intermediate_csv"
                            )
                        ),
                        "dataset_queue_direct_start_browser_has_direct_zarr_write_authority": bool(
                            payload["dataset_queue_direct_start_policy"].get(
                                "browser_has_direct_zarr_write_authority"
                            )
                        ),
                        "labeler_browser_only": bool(
                            payload["labeler_safety"].get("browser_only")
                        ),
                        "labeler_requires_local_palette_installation": bool(
                            payload["labeler_safety"].get("requires_local_palette_installation")
                        ),
                        "labeler_requires_local_crimson_installation": bool(
                            payload["labeler_safety"].get("requires_local_crimson_installation")
                        ),
                        "zarr_backup_copy_before_labeling": bool(
                            payload["zarr_backup_policy"].get("copy_before_labeling")
                        ),
                        "zarr_backup_labelers_do_not_receive_backup_paths": bool(
                            payload["zarr_backup_policy"].get("labelers_do_not_receive_backup_paths")
                        ),
                        "mutation_audit_server_records_events": bool(
                            payload["mutation_audit_policy"].get("server_records_events")
                        ),
                        "browser_response_security_clickjacking_protection": bool(
                            payload["browser_response_security_policy"].get("clickjacking_protection")
                        ),
                        "session_guard_stale_tab_save_rejected": bool(
                            payload["session_guard_policy"].get("stale_tab_save_rejected")
                        ),
                        "task_state_completed_tasks_read_only": bool(
                            payload["task_state_policy"].get("completed_tasks_read_only")
                        ),
                        "task_state_requires_current_target_token": bool(
                            payload["task_state_policy"].get("requires_current_target_token")
                            or payload["task_state_policy"].get("browser_mutation_target_token")
                            == "required_current_target_token"
                        ),
                        "signed_link_authorization_grant": bool(
                            payload["signed_link_policy"].get("authorization_grant")
                        ),
                        "signed_link_binds_expected_user": bool(
                            payload["signed_link_policy"].get("binds_expected_user_in_new_links")
                        ),
                        "signed_link_expected_user_required_on_open": bool(
                            payload["signed_link_policy"].get(
                                "expected_user_required_on_open",
                                payload["signed_link_policy"].get(
                                    "binds_expected_user_in_new_links"
                                ),
                            )
                        ),
                        "signed_link_runtime_operator_validation_start_gate_enforced": bool(
                            payload["signed_link_policy"].get(
                                "runtime_operator_validation_start_gate_enforced",
                                True,
                            )
                        ),
                        "signed_link_operator_validation_start_gate_checked_before_session_create": bool(
                            payload["signed_link_policy"].get(
                                "runtime_operator_validation_start_gate_enforced",
                                True,
                            )
                        ),
                    }
                )
                return 0
            _print_json(payload)
            return 0
        if args.command == "dashboard-roster":
            base_url = str(args.base_url or "").rstrip("/")
            labeler_landing_url = _labeler_landing_url_for_base(base_url)
            dashboard_url = _dashboard_url_for_base(base_url)
            dataset_queue_url = _dataset_queue_url_for_base(base_url)
            labeling_home_url = _labeling_home_url_for_base(base_url)
            personal_work_url = _personal_work_url_for_base(base_url)
            personal_dataset_queue_url = _personal_dataset_queue_url_for_base(base_url)
            operator_validation_fields = _dashboard_operator_validation_fields(
                checklist_path=args.operator_validation_checklist,
                operator_launch_approved=bool(args.operator_launch_approved),
            )
            operator_validation_public_fields = _operator_validation_public_fields(
                operator_validation_fields
            )
            operator_validation_command_templates = _operator_validation_command_templates(
                operator_validation_public_fields.get(
                    "operator_validation_required_missing_evidence_gate_ids"
                )
                if isinstance(
                    operator_validation_public_fields.get(
                        "operator_validation_required_missing_evidence_gate_ids"
                    ),
                    list,
                )
                else None
            )
            rows = _dashboard_roster_rows(
                store,
                dashboard_url=dashboard_url,
                user=args.user,
                include_inactive=bool(args.include_inactive),
                include_completed=bool(args.include_completed),
                require_dashboard_url=True,
                operator_launch_approved=bool(args.operator_launch_approved),
                operator_validation_fields=operator_validation_fields,
            )
            invite_reason_counts = _dashboard_invite_reason_counts(rows)
            dashboard_warning_counts = dict(invite_reason_counts)
            dataset_queue_blocked_start_users = [
                str(row.get("user") or "")
                for row in rows
                if bool(row.get("dataset_queue_blocks_labeler_start"))
            ]
            if dataset_queue_blocked_start_users:
                dashboard_warning_counts["dataset_queue_blocks_labeler_start"] = len(dataset_queue_blocked_start_users)
            warnings = [
                {
                    "code": reason,
                    "count": count,
                    "details": "One or more dashboard roster users are not ready-row draft candidates.",
                }
                for reason, count in sorted(dashboard_warning_counts.items())
            ]
            ready_invitation_bundle = _dashboard_ready_invitation_bundle(rows)
            runtime_gate_cli_policy = _runtime_operator_validation_gate_cli_policy()
            top_level_safe_share_gate = _safe_share_gate_policy()
            top_level_safe_share_checklist_fields = (
                _safe_share_checklist_gate_status_fields_from_operator_validation(
                    operator_validation_public_fields,
                    safe_share_gate=top_level_safe_share_gate,
                )
            )
            payload = {
                "ok": not dashboard_warning_counts,
                "store_path": str(store_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "base_url": base_url or None,
                "labeler_landing_page_path": "/",
                "labeler_landing_url": labeler_landing_url,
                "dashboard_path": DASHBOARD_PATH,
                "dashboard_url": dashboard_url,
                "dataset_queue_page_path": DATASET_QUEUE_PATH,
                "dataset_queue_url": dataset_queue_url,
                "labeling_home_page_path": LABELING_HOME_PATH,
                "labeling_home_url": labeling_home_url,
                "personal_work_page_path": PERSONAL_WORK_PATH,
                "personal_work_url": personal_work_url,
                "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
                "personal_dataset_queue_url": personal_dataset_queue_url,
                "labeler_safety": _labeler_safety_policy(),
                "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
                "task_state_policy": _browser_task_state_policy(),
                "signed_link_policy": _browser_signed_link_policy(),
                "operator_recovery_policy": _operator_recovery_policy(),
                "operator_recovery_contract": _operator_recovery_contract_policy(_operator_recovery_policy()),
                "browser_workflows": _browser_workflow_capabilities(),
                "operator_validation_visibility_policy": _operator_validation_visibility_policy(),
                "operator_validation_command_templates": operator_validation_command_templates,
                "runtime_operator_validation_gate_cli_policy": runtime_gate_cli_policy,
                **_runtime_operator_validation_gate_cli_policy_fields(
                    runtime_gate_cli_policy
                ),
                "safe_share_gate": top_level_safe_share_gate,
                **_safe_share_gate_flat_fields(top_level_safe_share_gate),
                **top_level_safe_share_checklist_fields,
                **operator_validation_fields,
                **_operator_validation_gate_metadata_fields(),
                **_operator_validation_gate_flat_fields(operator_validation_fields),
                "invite_actions": _dashboard_invite_actions(invite_reason_counts),
                "include_inactive": bool(args.include_inactive),
                "include_completed": bool(args.include_completed),
                "filters": {"user": args.user},
                "counts": {
                    "users": len(rows),
                    "ready_to_invite": sum(1 for row in rows if bool(row.get("ready_to_invite"))),
                    "not_ready_to_invite": sum(1 for row in rows if not bool(row.get("ready_to_invite"))),
                    "ready_to_invite_users": [
                        str(row.get("user") or "")
                        for row in rows
                        if bool(row.get("ready_to_invite"))
                    ],
                    "not_ready_to_invite_users": [
                        str(row.get("user") or "")
                        for row in rows
                        if not bool(row.get("ready_to_invite"))
                    ],
                    "ready_row_draft_count": sum(
                        1 for row in rows if str(row.get("copy_intent") or "") == "ready_row_draft"
                    ),
                    "diagnostic_note_count": sum(
                        1 for row in rows if str(row.get("copy_intent") or "") == "diagnostic_note"
                    ),
                    "ready_row_draft_users": [
                        str(row.get("user") or "")
                        for row in rows
                        if str(row.get("copy_intent") or "") == "ready_row_draft"
                    ],
                    "diagnostic_note_users": [
                        str(row.get("user") or "")
                        for row in rows
                        if str(row.get("copy_intent") or "") == "diagnostic_note"
                    ],
                    "recordings": sum(int(row.get("recordings") or 0) for row in rows),
                    "open_tasks": sum(int(row.get("open_tasks") or 0) for row in rows),
                    "total_tasks": sum(int(row.get("total_tasks") or 0) for row in rows),
                    "complete_tasks": sum(int(row.get("complete_tasks") or 0) for row in rows),
                    "completion_percent": _completion_percent(
                        sum(int(row.get("complete_tasks") or 0) for row in rows),
                        sum(int(row.get("total_tasks") or 0) for row in rows),
                    ),
                    "completion_states": _dashboard_completion_state_counts(rows),
                    "recordings_without_open_tasks": sum(int(row.get("recordings_without_open_tasks") or 0) for row in rows),
                    "invite_reasons": invite_reason_counts,
                    "dashboard_warnings": dashboard_warning_counts,
                    "copy_intents": _dashboard_copy_intent_counts(rows),
                    "ready_states": _dashboard_ready_state_counts(rows),
                    **_dashboard_identity_probe_counts(rows),
                    **_dashboard_dataset_queue_counts(rows),
                },
                "warning_count": len(warnings),
                "warning_codes": [str(warning["code"]) for warning in warnings],
                "warnings": warnings,
                "ready_to_invite_legacy_semantics": _DASHBOARD_READY_TO_INVITE_LEGACY_SEMANTICS,
                "ready_row_state_values": list(_DASHBOARD_READY_ROW_STATE_VALUES),
                "copy_intent_values": list(_DASHBOARD_COPY_INTENT_VALUES),
                "ready_invitations": ready_invitation_bundle["messages"],
                "ready_invitations_text": ready_invitation_bundle["text"],
                "ready_row_draft_bundle_schema": ready_invitation_bundle["schema"],
                "ready_row_draft_bundle_kind": ready_invitation_bundle["kind"],
                "ready_invitations_legacy_semantics": ready_invitation_bundle["legacy_semantics"],
                "ready_invitations_legacy_field_names": ready_invitation_bundle["legacy_field_names"],
                "ready_row_drafts": ready_invitation_bundle["messages"],
                "ready_row_draft_text": ready_invitation_bundle["text"],
                "ready_row_draft_share_rule": ready_invitation_bundle["share_rule"],
                "users": rows,
            }
            dataset_queue_start_readiness = _dataset_queue_start_readiness_from_counts(payload["counts"])
            payload["dataset_queue_start_readiness"] = dataset_queue_start_readiness
            payload["counts"]["dataset_queue_start_readiness"] = dataset_queue_start_readiness
            payload["status_report"] = _dashboard_status_report(
                rows,
                payload["counts"],
                warnings,
                operator_validation_fields=operator_validation_fields,
            )
            if str(args.format) == "html":
                text = _dashboard_roster_html(payload)
                summary = {
                    "ok": bool(payload["ok"]),
                    "count": len(rows),
                    "format": "html",
                    "output": args.output,
                    "filters": payload.get("filters", {}),
                    "warning_count": len(warnings),
                    "warning_codes": payload["warning_codes"],
                    "dataset_queue_start_readiness": payload["dataset_queue_start_readiness"],
                    "browser_mutation_target_contract_all_users_met": bool(
                        payload["counts"].get(
                            "browser_mutation_target_contract_all_users_met"
                        )
                    ),
                    "browser_mutation_target_contract_not_met_users": payload["counts"].get(
                        "browser_mutation_target_contract_not_met_users", []
                    ),
                    "browser_mutation_target_contract_not_met_user_count": int(
                        payload["counts"].get(
                            "browser_mutation_target_contract_not_met_user_count"
                        )
                        or 0
                    ),
                    "browser_mutation_target_total_mismatch_count": int(
                        payload["counts"].get(
                            "browser_mutation_target_total_mismatch_count"
                        )
                        or 0
                    ),
                    "direct_browser_start_contract_all_users_met": bool(
                        payload["counts"].get(
                            "direct_browser_start_contract_all_users_met"
                        )
                    ),
                    "direct_browser_start_contract_not_met_users": payload["counts"].get(
                        "direct_browser_start_contract_not_met_users", []
                    ),
                    "direct_browser_start_contract_not_met_user_count": int(
                        payload["counts"].get(
                            "direct_browser_start_contract_not_met_user_count"
                        )
                        or 0
                    ),
                    "direct_browser_start_total_mismatch_count": int(
                        payload["counts"].get("direct_browser_start_total_mismatch_count")
                        or 0
                    ),
                    "single_owner_policy_contract_all_users_met": bool(
                        payload["counts"].get(
                            "single_owner_policy_contract_all_users_met"
                        )
                    ),
                    "single_owner_policy_contract_not_met_users": payload["counts"].get(
                        "single_owner_policy_contract_not_met_users", []
                    ),
                    "single_owner_policy_contract_not_met_user_count": int(
                        payload["counts"].get(
                            "single_owner_policy_contract_not_met_user_count"
                        )
                        or 0
                    ),
                    "labeler_route_authorization_runtime_checklist_gate_all_users_met": bool(
                        payload["counts"].get(
                            "labeler_route_authorization_runtime_checklist_gate_all_users_met"
                        )
                    ),
                    "labeler_route_authorization_runtime_checklist_not_met_users": payload[
                        "counts"
                    ].get(
                        "labeler_route_authorization_runtime_checklist_not_met_users",
                        [],
                    ),
                    "labeler_route_authorization_runtime_checklist_not_met_user_count": int(
                        payload["counts"].get(
                            "labeler_route_authorization_runtime_checklist_not_met_user_count"
                        )
                        or 0
                    ),
                    "labeler_route_authorization_runtime_checklist_total_mismatch_count": int(
                        payload["counts"].get(
                            "labeler_route_authorization_runtime_checklist_total_mismatch_count"
                        )
                        or 0
                    ),
                }
                if args.output:
                    output_path = Path(args.output)
                    if output_path.exists() and not bool(args.overwrite):
                        raise FileExistsError(f"Refusing to overwrite existing dashboard roster HTML: {output_path}")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(text, encoding="utf-8")
                    _print_json(summary)
                else:
                    print(text, end="")
                return 0 if bool(payload["ok"]) else 2
            summary = _write_row_export(
                payload=payload,
                rows=rows,
                output=args.output,
                output_format=str(args.format),
                overwrite=bool(args.overwrite),
            )
            if args.output:
                _print_json({
                    **summary,
                    "ok": bool(payload["ok"]),
                    "warning_count": len(warnings),
                    "warning_codes": payload["warning_codes"],
                    "dataset_queue_start_readiness": payload["dataset_queue_start_readiness"],
                    "browser_mutation_target_contract_all_users_met": bool(
                        payload["counts"].get(
                            "browser_mutation_target_contract_all_users_met"
                        )
                    ),
                    "browser_mutation_target_contract_not_met_users": payload["counts"].get(
                        "browser_mutation_target_contract_not_met_users", []
                    ),
                    "browser_mutation_target_contract_not_met_user_count": int(
                        payload["counts"].get(
                            "browser_mutation_target_contract_not_met_user_count"
                        )
                        or 0
                    ),
                    "browser_mutation_target_total_mismatch_count": int(
                        payload["counts"].get(
                            "browser_mutation_target_total_mismatch_count"
                        )
                        or 0
                    ),
                    "direct_browser_start_contract_all_users_met": bool(
                        payload["counts"].get(
                            "direct_browser_start_contract_all_users_met"
                        )
                    ),
                    "direct_browser_start_contract_not_met_users": payload["counts"].get(
                        "direct_browser_start_contract_not_met_users", []
                    ),
                    "direct_browser_start_contract_not_met_user_count": int(
                        payload["counts"].get(
                            "direct_browser_start_contract_not_met_user_count"
                        )
                        or 0
                    ),
                    "direct_browser_start_total_mismatch_count": int(
                        payload["counts"].get("direct_browser_start_total_mismatch_count")
                        or 0
                    ),
                    "single_owner_policy_contract_all_users_met": bool(
                        payload["counts"].get(
                            "single_owner_policy_contract_all_users_met"
                        )
                    ),
                    "single_owner_policy_contract_not_met_users": payload["counts"].get(
                        "single_owner_policy_contract_not_met_users", []
                    ),
                    "single_owner_policy_contract_not_met_user_count": int(
                        payload["counts"].get(
                            "single_owner_policy_contract_not_met_user_count"
                        )
                        or 0
                    ),
                    "labeler_route_authorization_runtime_checklist_gate_all_users_met": bool(
                        payload["counts"].get(
                            "labeler_route_authorization_runtime_checklist_gate_all_users_met"
                        )
                    ),
                    "labeler_route_authorization_runtime_checklist_not_met_users": payload[
                        "counts"
                    ].get(
                        "labeler_route_authorization_runtime_checklist_not_met_users",
                        [],
                    ),
                    "labeler_route_authorization_runtime_checklist_not_met_user_count": int(
                        payload["counts"].get(
                            "labeler_route_authorization_runtime_checklist_not_met_user_count"
                        )
                        or 0
                    ),
                    "labeler_route_authorization_runtime_checklist_total_mismatch_count": int(
                        payload["counts"].get(
                            "labeler_route_authorization_runtime_checklist_total_mismatch_count"
                        )
                        or 0
                    ),
                    **operator_validation_public_fields,
                    **_operator_validation_gate_metadata_fields(),
                    **_operator_validation_gate_flat_fields(operator_validation_public_fields),
                    **_safe_share_checklist_field_values(payload),
                    "runtime_operator_validation_gate_cli_policy": (
                        payload["runtime_operator_validation_gate_cli_policy"]
                    ),
                    **_runtime_operator_validation_gate_cli_policy_fields(
                        payload["runtime_operator_validation_gate_cli_policy"]
                        if isinstance(
                            payload.get("runtime_operator_validation_gate_cli_policy"),
                            Mapping,
                        )
                        else None
                    ),
                })
            return 0 if bool(payload["ok"]) else 2
        if args.command == "check-store":
            report = _store_consistency_report(store)
            payload = {
                "store_path": str(store_path),
                **report,
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="store consistency report")
            _print_json(payload)
            return 0 if bool(report["ok"]) else 2
        if args.command == "list-sessions":
            sessions = store.list_sessions(
                assignee_user=args.user,
                include_closed=bool(args.include_closed),
                expired_only=bool(args.expired_only),
                limit=int(args.limit),
            )
            payload = {
                "ok": True,
                "store_path": str(args.store),
                "filters": {
                    "user": args.user,
                    "include_closed": bool(args.include_closed),
                    "expired_only": bool(args.expired_only),
                    "limit": int(args.limit),
                },
                "count": len(sessions),
                "sessions": sessions,
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="session listing report")
            _print_json(payload)
            return 0
        if args.command == "cleanup-stale-sessions":
            if args.output is not None and Path(args.output).expanduser().exists() and not bool(args.overwrite):
                raise FileExistsError(f"Output file already exists: {args.output}")
            sessions = store.cleanup_stale_sessions(user=args.user)
            closed_session_ids = [str(session.get("session_id") or "") for session in sessions]
            payload = {
                "ok": True,
                "store_path": str(args.store),
                "closed_count": len(sessions),
                **_closed_session_response_payload(store, closed_session_ids),
                "sessions": sessions,
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="stale-session cleanup report")
            _print_json(payload)
            return 0
        if args.command == "repair-reassignment-sessions":
            if args.output is not None and Path(args.output).expanduser().exists() and not bool(args.overwrite):
                raise FileExistsError(f"Output file already exists: {args.output}")
            check_report_before = _store_consistency_report(store)
            safety_before = (
                check_report_before.get("reassignment_session_safety")
                if isinstance(check_report_before.get("reassignment_session_safety"), Mapping)
                else {}
            )
            requested_recording_ids = [
                str(recording_id).strip()
                for recording_id in (args.recording_id or [])
                if str(recording_id).strip()
            ]
            recording_ids = requested_recording_ids or [
                str(recording_id)
                for recording_id in (
                    safety_before.get("active_session_assignment_mismatch_recording_ids")
                    if isinstance(
                        safety_before.get("active_session_assignment_mismatch_recording_ids"),
                        list,
                    )
                    else []
                )
                if str(recording_id).strip()
            ]
            repaired: list[dict[str, object]] = []
            closed_session_ids: list[str] = []
            for recording_id in recording_ids:
                sessions = store.close_assignment_mismatched_sessions_for_recording(
                    recording_id=recording_id,
                    user=args.user,
                )
                session_ids = [str(session.get("session_id") or "") for session in sessions]
                closed_session_ids.extend(session_id for session_id in session_ids if session_id)
                repaired.append(
                    {
                        "recording_id": recording_id,
                        "closed_count": len(sessions),
                        "closed_session_ids": session_ids,
                        "sessions": sessions,
                    }
                )
            check_report_after = _store_consistency_report(store)
            payload = {
                "ok": True,
                "schema": "palette.web_labeling_reassignment_session_repair_report.v1",
                "store_path": str(args.store),
                "operator_user": str(args.user),
                "recording_ids": recording_ids,
                "requested_recording_ids": requested_recording_ids,
                "reassignment_session_safety_before": safety_before,
                "reassignment_session_safety_after": check_report_after.get(
                    "reassignment_session_safety",
                    {},
                ),
                "closed_count": len(closed_session_ids),
                **_closed_session_response_payload(store, closed_session_ids),
                "repairs": repaired,
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="reassignment-session repair report")
            _print_json(payload)
            return 0
        if args.command == "lookup-event":
            event_id = str(args.event_id or "").strip()
            event = store.get_event(event_id)
            if event is None:
                payload = {
                    "ok": False,
                    "schema": "palette.web_labeling_audit_event_lookup.v1",
                    "store_path": str(store_path),
                    "event_id": event_id,
                    "error": "event_not_found",
                    "operator_action": "Confirm the labeler copied the full audit event ID from the browser save status or support reference.",
                }
                _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="audit event lookup report")
                _print_json(payload)
                return 2
            payload = {
                "ok": True,
                "schema": "palette.web_labeling_audit_event_lookup.v1",
                "store_path": str(store_path),
                "event_id": event_id,
                "event": event,
                "retry_promotion_route": (
                    f"/api/admin/events/{event_id}/retry-promotion"
                    if str(event.get("event_type") or "") == "promotion_failed"
                    else ""
                ),
                "operator_action": (
                    "Use this audit event to reconcile a labeler-provided save reference with the assigned task, recording, user, workflow, target, and mutation outcome."
                ),
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="audit event lookup report")
            _print_json(payload)
            return 0
        if args.command == "export-events":
            events = store.list_events(
                task_id=args.task_id,
                recording_id=args.recording_id,
                event_type=args.event_type,
                assignee_user=args.user,
                actor_user=args.actor,
                since_utc=args.since_utc,
                until_utc=args.until_utc,
                limit=int(args.limit),
            )
            payload = {
                "ok": True,
                "store_path": str(args.store),
                "filters": {
                    "task_id": args.task_id,
                    "recording_id": args.recording_id,
                    "event_type": args.event_type,
                    "user": args.user,
                    "actor": args.actor,
                    "since_utc": args.since_utc,
                    "until_utc": args.until_utc,
                    "limit": int(args.limit),
                },
                "count": len(events),
                "events": events,
            }
            if args.output:
                output_path = Path(args.output)
                if output_path.exists() and not bool(args.overwrite):
                    raise FileExistsError(f"Refusing to overwrite existing audit export: {output_path}")
                if args.format == "jsonl":
                    output_text = "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
                else:
                    output_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
                output_path.write_text(output_text)
                _print_json(
                    {
                        "ok": True,
                        "store_path": str(args.store),
                        "output_path": str(output_path),
                        "format": args.format,
                        "count": len(events),
                        "filters": payload["filters"],
                    }
                )
                return 0
            _print_json(payload)
            return 0
        if args.command == "export-assignment-events":
            events = store.list_assignment_events(
                recording_id=args.recording_id,
                actor_user=args.actor,
                event_type=args.event_type,
                since_utc=args.since_utc,
                until_utc=args.until_utc,
                limit=int(args.limit),
            )
            payload = {
                "ok": True,
                "store_path": str(args.store),
                "filters": {
                    "recording_id": args.recording_id,
                    "actor": args.actor,
                    "event_type": args.event_type,
                    "since_utc": args.since_utc,
                    "until_utc": args.until_utc,
                    "limit": int(args.limit),
                },
                "count": len(events),
                "events": events,
            }
            if args.output:
                output_path = Path(args.output)
                if output_path.exists() and not bool(args.overwrite):
                    raise FileExistsError(f"Refusing to overwrite existing assignment audit export: {output_path}")
                if args.format == "jsonl":
                    output_text = "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
                else:
                    output_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
                output_path.write_text(output_text)
                _print_json(
                    {
                        "ok": True,
                        "store_path": str(args.store),
                        "output_path": str(output_path),
                        "format": args.format,
                        "count": len(events),
                        "filters": payload["filters"],
                    }
                )
                return 0
            _print_json(payload)
            return 0
        if args.command == "export-task-definition-events":
            events = store.list_task_definition_events(
                task_id=args.task_id,
                recording_id=args.recording_id,
                actor_user=args.actor,
                event_type=args.event_type,
                since_utc=args.since_utc,
                until_utc=args.until_utc,
                limit=int(args.limit),
            )
            payload = {
                "ok": True,
                "store_path": str(args.store),
                "filters": {
                    "task_id": args.task_id,
                    "recording_id": args.recording_id,
                    "actor": args.actor,
                    "event_type": args.event_type,
                    "since_utc": args.since_utc,
                    "until_utc": args.until_utc,
                    "limit": int(args.limit),
                },
                "count": len(events),
                "events": events,
            }
            if args.output:
                output_path = Path(args.output)
                if output_path.exists() and not bool(args.overwrite):
                    raise FileExistsError(f"Refusing to overwrite existing task-definition audit export: {output_path}")
                if args.format == "jsonl":
                    output_text = "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
                else:
                    output_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
                output_path.write_text(output_text)
                _print_json(
                    {
                        "ok": True,
                        "store_path": str(args.store),
                        "output_path": str(output_path),
                        "format": args.format,
                        "count": len(events),
                        "filters": payload["filters"],
                    }
                )
                return 0
            _print_json(payload)
            return 0
        if args.command == "export-audit-bundle":
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            bundle_paths = {
                "task_events": output_dir / "task-events.jsonl",
                "assignment_events": output_dir / "assignment-events.jsonl",
                "task_definition_events": output_dir / "task-definition-events.jsonl",
                "manifest": output_dir / "manifest.json",
            }
            existing_paths = [path for path in bundle_paths.values() if path.exists()]
            if existing_paths and not bool(args.overwrite):
                raise FileExistsError(
                    "Refusing to overwrite existing audit bundle files: "
                    + ", ".join(str(path) for path in existing_paths)
                )

            if args.recording_id:
                recording_ids: list[str | None] = [str(args.recording_id)]
            elif args.user:
                recording_ids = [
                    str(assignment["recording_id"])
                    for assignment in store.list_assignments(assignee_user=args.user, status=None)
                ]
            else:
                recording_ids = [None]

            task_events: list[dict[str, object]] = []
            assignment_events: list[dict[str, object]] = []
            task_definition_events: list[dict[str, object]] = []
            for recording_id in recording_ids:
                task_events.extend(
                    store.list_events(
                        recording_id=recording_id,
                        assignee_user=args.user if recording_id is None else None,
                        since_utc=args.since_utc,
                        until_utc=args.until_utc,
                        limit=int(args.limit),
                    )
                )
                assignment_events.extend(
                    store.list_assignment_events(
                        recording_id=recording_id,
                        since_utc=args.since_utc,
                        until_utc=args.until_utc,
                        limit=int(args.limit),
                    )
                )
                task_definition_events.extend(
                    store.list_task_definition_events(
                        recording_id=recording_id,
                        since_utc=args.since_utc,
                        until_utc=args.until_utc,
                        limit=int(args.limit),
                    )
                )

            bundle_paths["task_events"].write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in task_events),
                encoding="utf-8",
            )
            bundle_paths["assignment_events"].write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in assignment_events),
                encoding="utf-8",
            )
            bundle_paths["task_definition_events"].write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in task_definition_events),
                encoding="utf-8",
            )
            manifest = {
                "ok": True,
                "store_path": str(store_path),
                "output_dir": str(output_dir),
                "filters": {
                    "recording_id": args.recording_id,
                    "user": args.user,
                    "since_utc": args.since_utc,
                    "until_utc": args.until_utc,
                    "limit": int(args.limit),
                },
                "recording_ids": [recording_id for recording_id in recording_ids if recording_id is not None],
                "files": {key: str(path) for key, path in bundle_paths.items()},
                "counts": {
                    "task_events": len(task_events),
                    "assignment_events": len(assignment_events),
                    "task_definition_events": len(task_definition_events),
                },
            }
            bundle_paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _print_json(manifest)
            return 0
        if args.command == "sign-invite":
            secret = _link_secret_from_arg(args.link_secret)
            user = str(args.user or "").strip()
            token_info = _signed_invite_token_info(
                user=user,
                secret=secret,
                ttl_seconds=int(args.ttl_seconds),
            )
            token = str(token_info["token"])
            path = _signed_invite_path(token, user=user)
            base_url = str(args.base_url or "").rstrip("/")
            url_is_absolute = bool(base_url)
            known_user_status = _known_labeler_status(store, user)
            shareability_warnings = [] if url_is_absolute else [
                {
                    "code": "missing_base_url",
                    "details": "Generated invite URL is service-relative. Provide --base-url before sharing with a labeler.",
                }
            ]
            if not bool(known_user_status.get("is_active_labeling_user")):
                shareability_warnings.append(
                    {
                        "code": "inactive_or_unknown_labeling_user",
                        "user": user,
                        "details": "This user is not active in the labeling_users SQLite table; add or activate the user before sharing an invite.",
                    }
                )
            payload = {
                "ok": True,
                "store_path": str(args.store),
                "user": user,
                "expected_user": user,
                "scope": token_info["scope"],
                "base_url": base_url or None,
                "issued_at_utc": token_info["issued_at_utc"],
                "expires_at_utc": token_info["expires_at_utc"],
                "expires_in_seconds": token_info["ttl_seconds"],
                "path": path,
                "url": f"{base_url}{path}" if base_url else path,
                "url_is_absolute": url_is_absolute,
                "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
                "preferred_labeler_entry_url": f"{base_url}{path}" if base_url else path,
                "known_user_status": known_user_status,
                "ready_to_share": url_is_absolute and not shareability_warnings,
                "shareability_warnings": shareability_warnings,
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="signed-invite report")
            _print_json(payload)
            return 0
        if args.command == "sign-link":
            task = store.get_task(args.task_id)
            if task is None:
                raise KeyError(f"Unknown task_id: {args.task_id}")
            secret = _link_secret_from_arg(args.link_secret)
            token_info = _signed_task_link_token_info(
                task_id=str(args.task_id),
                secret=secret,
                ttl_seconds=int(args.ttl_seconds),
                expected_user=str(task.get("assignee_user") or ""),
            )
            token = str(token_info["token"])
            path = f"/t/{token}"
            base_url = str(args.base_url or "").rstrip("/")
            url_is_absolute = bool(base_url)
            shareability_warnings = [] if url_is_absolute else [
                {
                    "code": "missing_base_url",
                    "details": "Generated URL is service-relative. Provide --base-url before sharing with a labeler.",
                }
            ]
            if str(task.get("assignment_status") or "") != "active":
                shareability_warnings.append(
                    {
                        "code": "task_assignment_not_active",
                        "assignment_status": task.get("assignment_status"),
                        "assignee_user": task.get("assignee_user"),
                        "recording_id": task.get("recording_id"),
                        "details": "Task recording is not actively assigned, so the signed link will not open a labeling session.",
                    }
                )
            if str(task.get("state") or "") == "complete":
                shareability_warnings.append(
                    {
                        "code": "task_completed",
                        "task_id": str(args.task_id),
                        "recording_id": task.get("recording_id"),
                        "details": "Task is complete, so the signed link will not open a new labeling session unless the task is reopened.",
                    }
                )
            elif str(task.get("state") or "") not in LABELER_START_TASK_STATES:
                shareability_warnings.append(
                    {
                        "code": "task_not_startable",
                        "task_id": str(args.task_id),
                        "recording_id": task.get("recording_id"),
                        "state": task.get("state"),
                        "startable_task_states": list(LABELER_START_TASK_STATES),
                        "details": "Task is not in a startable labeling state, so the signed link will not open a new labeling session.",
                    }
                )
            payload = {
                "ok": True,
                "task_id": str(args.task_id),
                "recording_id": task.get("recording_id"),
                "assignee_user": task.get("assignee_user"),
                "expected_user": token_info.get("expected_user") or task.get("assignee_user"),
                "assignment_status": task.get("assignment_status"),
                "state": task.get("state"),
                "startable_task_states": list(LABELER_START_TASK_STATES),
                "base_url": base_url or None,
                "issued_at_utc": token_info["issued_at_utc"],
                "expires_at_utc": token_info["expires_at_utc"],
                "expires_in_seconds": token_info["ttl_seconds"],
                "path": path,
                "url": f"{base_url}{path}" if base_url else path,
                "url_is_absolute": url_is_absolute,
                "task_launchable": not any(
                    str(warning.get("code") or "").startswith("task_")
                    for warning in shareability_warnings
                ),
                "ready_to_share": url_is_absolute and not shareability_warnings,
                "shareability_warnings": shareability_warnings,
            }
            _write_optional_json_report(payload, args.output, overwrite=bool(args.overwrite), description="signed-link report")
            _print_json(payload)
            return 0
        if args.command == "sign-links":
            secret = _link_secret_from_arg(args.link_secret)
            base_url = str(args.base_url or "").rstrip("/")
            url_is_absolute = bool(base_url)
            shareability_warnings = [] if url_is_absolute else [
                {
                    "code": "missing_base_url",
                    "details": "Generated URLs are service-relative. Provide --base-url before sharing with labelers.",
                }
            ]
            tasks = [
                task
                for task in store.list_tasks(
                    recording_id=args.recording_id,
                    assignee_user=args.user,
                    include_completed=bool(args.include_completed),
                )
                if str(task.get("assignment_status") or "") == "active"
            ]
            links: list[dict[str, object]] = []
            for task in tasks:
                task_id = str(task["task_id"])
                row_warnings = list(shareability_warnings)
                if str(task.get("state") or "") == "complete":
                    row_warnings.append(
                        {
                            "code": "task_completed",
                            "task_id": task_id,
                            "recording_id": task.get("recording_id"),
                            "details": "Task is complete, so the signed link will not open a new labeling session unless the task is reopened.",
                        }
                    )
                elif str(task.get("state") or "") not in LABELER_START_TASK_STATES:
                    row_warnings.append(
                        {
                            "code": "task_not_startable",
                            "task_id": task_id,
                            "recording_id": task.get("recording_id"),
                            "state": task.get("state"),
                            "startable_task_states": list(LABELER_START_TASK_STATES),
                            "details": "Task is not in a startable labeling state, so the signed link will not open a new labeling session.",
                        }
                    )
                token_info = _signed_task_link_token_info(
                    task_id=task_id,
                    secret=secret,
                    ttl_seconds=int(args.ttl_seconds),
                    expected_user=str(task.get("assignee_user") or ""),
                )
                token = str(token_info["token"])
                path = f"/t/{token}"
                links.append(
                    {
                        "task_id": task_id,
                        "recording_id": task.get("recording_id"),
                        "assignee_user": task.get("assignee_user"),
                        "expected_user": token_info.get("expected_user") or task.get("assignee_user"),
                        "workflow_kind": task.get("workflow_kind"),
                        "state": task.get("state"),
                        "startable_task_states": list(LABELER_START_TASK_STATES),
                        "title": task.get("title"),
                        "issued_at_utc": token_info["issued_at_utc"],
                        "expires_at_utc": token_info["expires_at_utc"],
                        "expires_in_seconds": token_info["ttl_seconds"],
                        "path": path,
                        "url": f"{base_url}{path}" if base_url else path,
                        "url_is_absolute": url_is_absolute,
                        "task_launchable": not any(
                            str(warning.get("code") or "").startswith("task_")
                            for warning in row_warnings
                        ),
                        "ready_to_share": url_is_absolute and not row_warnings,
                        "shareability_warnings": row_warnings,
                    }
                )
            payload = {
                "ok": True,
                "store_path": str(args.store),
                "base_url": base_url or None,
                "startable_task_states": list(LABELER_START_TASK_STATES),
                "ready_to_share": bool(links) and all(bool(link.get("ready_to_share")) for link in links),
                "ready_to_share_count": sum(1 for link in links if bool(link.get("ready_to_share"))),
                "not_ready_to_share_count": sum(1 for link in links if not bool(link.get("ready_to_share"))),
                "shareability_warnings": shareability_warnings,
                "filters": {
                    "recording_id": args.recording_id,
                    "user": args.user,
                    "include_completed": bool(args.include_completed),
                    "active_assignments_only": True,
                },
                "count": len(links),
                "links": links,
            }
            if args.output:
                output_path = Path(args.output)
                if output_path.exists() and not bool(args.overwrite):
                    raise FileExistsError(f"Refusing to overwrite existing signed-link export: {output_path}")
                if args.format == "jsonl":
                    output_text = "".join(json.dumps(link, sort_keys=True) + "\n" for link in links)
                else:
                    output_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
                output_path.write_text(output_text)
                _print_json(
                    {
                        "ok": True,
                        "store_path": str(args.store),
                        "output_path": str(output_path),
                        "format": args.format,
                        "count": len(links),
                        "ready_to_share": bool(links) and all(bool(link.get("ready_to_share")) for link in links),
                        "ready_to_share_count": payload["ready_to_share_count"],
                        "not_ready_to_share_count": payload["not_ready_to_share_count"],
                        "shareability_warnings": shareability_warnings,
                        "filters": payload["filters"],
                    }
                )
                return 0
            _print_json(payload)
            return 0
        if args.command == "batch-readiness":
            report = {
                "store_path": str(store_path),
                **_apply_readiness_warning_policy(
                    _batch_readiness_report(store),
                    warnings_as_errors=bool(args.warnings_as_errors),
                ),
            }
            if args.output:
                output_path = Path(args.output)
                if output_path.exists() and not bool(args.overwrite):
                    raise FileExistsError(f"Refusing to overwrite existing readiness report: {output_path}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            _print_json(report)
            return 0 if bool(report["ok"]) else 2
        if args.command == "export-assignments":
            rows = store.list_assignments(status=args.status)
            if args.recording_id:
                rows = [row for row in rows if str(row.get("recording_id") or "") == str(args.recording_id)]
            if args.user:
                rows = [row for row in rows if str(row.get("assignee_user") or "") == str(args.user)]
            payload = {
                "ok": True,
                "store_path": str(store_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "filters": {
                    "recording_id": args.recording_id,
                    "user": args.user,
                    "status": args.status,
                },
                "single_owner_policy": _assignment_ownership_policy(),
                "count": len(rows),
                "assignments": rows,
            }
            summary = _write_row_export(
                payload=payload,
                rows=rows,
                output=args.output,
                output_format=str(args.format),
                overwrite=bool(args.overwrite),
            )
            if args.output:
                _print_json({**summary, "store_path": str(store_path)})
            return 0
        if args.command == "export-tasks":
            rows = store.list_tasks(include_completed=not bool(args.open_only))
            if args.recording_id:
                rows = [row for row in rows if str(row.get("recording_id") or "") == str(args.recording_id)]
            if args.user:
                rows = [row for row in rows if str(row.get("assignee_user") or "") == str(args.user)]
            if args.workflow_kind:
                rows = [row for row in rows if str(row.get("workflow_kind") or "") == str(args.workflow_kind)]
            payload = {
                "ok": True,
                "store_path": str(store_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "filters": {
                    "recording_id": args.recording_id,
                    "user": args.user,
                    "workflow_kind": args.workflow_kind,
                    "open_only": bool(args.open_only),
                },
                "count": len(rows),
                "tasks": rows,
            }
            summary = _write_row_export(
                payload=payload,
                rows=rows,
                output=args.output,
                output_format=str(args.format),
                overwrite=bool(args.overwrite),
            )
            if args.output:
                _print_json({**summary, "store_path": str(store_path)})
            return 0
        if args.command == "write-manifest-templates":
            output_dir = Path(args.output_dir)
            assignment_path = output_dir / "assignments-template.csv"
            task_path = output_dir / "tasks-template.csv"
            readme_path = output_dir / "manifest-templates-readme.txt"
            existing_template_paths = [
                path
                for path in (assignment_path, task_path, readme_path)
                if path.exists()
            ]
            if existing_template_paths and not bool(args.overwrite):
                raise FileExistsError(
                    "Refusing to overwrite existing manifest template files: "
                    + ", ".join(str(path) for path in existing_template_paths)
                )
            _write_csv_manifest_template(
                assignment_path,
                fieldnames=["recording_id", "assignee_user", "status", "notes", "assigned_by"],
                sample={
                    "recording_id": "recording-a",
                    "assignee_user": "alice",
                    "status": "active",
                    "notes": "Review keypoints first.",
                    "assigned_by": "operator",
                },
                overwrite=bool(args.overwrite),
            )
            _write_csv_manifest_template(
                task_path,
                fieldnames=[
                    "task_id",
                    "recording_id",
                    "workflow_kind",
                    "title",
                    "scope_json",
                    "priority",
                    "notes",
                    "dataset_id",
                    "zarr_use",
                    "stage_group",
                    "run_name",
                    "component_name",
                    "state",
                ],
                sample={
                    "task_id": "recording-a-keypoints-review",
                    "recording_id": "recording-a",
                    "workflow_kind": "keypoints",
                    "title": "Review keypoints",
                    "scope_json": {"frames": [1, 2, 3]},
                    "priority": 5,
                    "notes": "First pass",
                    "state": "pending",
                },
                overwrite=bool(args.overwrite),
            )
            _write_manifest_templates_readme(
                readme_path,
                assignments_path=assignment_path,
                tasks_path=task_path,
                overwrite=bool(args.overwrite),
            )
            _print_json(
                {
                    "ok": True,
                    "output_dir": str(output_dir),
                    "files": {
                        "assignments_template": str(assignment_path),
                        "tasks_template": str(task_path),
                        "readme": str(readme_path),
                    },
                }
            )
            return 0
        if args.command == "import-batch-plan":
            assignment_rows = _parse_assignment_manifest(args.assignments)
            task_rows = _parse_task_manifest(args.tasks)
            report_paths = [
                path
                for path in (
                    Path(args.output) if args.output else None,
                    Path(args.html_output) if args.html_output else None,
                )
                if path is not None
            ]
            existing_report_paths = [path for path in report_paths if path.exists()]
            if existing_report_paths and not bool(args.overwrite):
                raise FileExistsError(
                    "Refusing to overwrite existing batch-plan report files: "
                    + ", ".join(str(path) for path in existing_report_paths)
                )
            existing_assignment_ids = {
                str(row.get("recording_id") or "")
                for row in store.list_assignments(status=None)
                if str(row.get("recording_id") or "")
            }
            existing_assignment_by_recording = {
                str(row.get("recording_id") or ""): row
                for row in store.list_assignments(status=None)
                if str(row.get("recording_id") or "")
            }
            existing_assignment_status_by_recording = {
                recording_id: str(row.get("status") or "")
                for recording_id, row in existing_assignment_by_recording.items()
            }
            planned_assignment_status_by_recording = dict(existing_assignment_status_by_recording)
            planned_assignment_status_by_recording.update(
                {
                    str(row.get("recording_id") or ""): str(row.get("status") or "active")
                    for row in assignment_rows
                    if str(row.get("recording_id") or "")
                }
            )
            planned_assignment_ids = existing_assignment_ids | {
                str(row.get("recording_id") or "")
                for row in assignment_rows
                if str(row.get("recording_id") or "")
            }
            issues: list[dict[str, object]] = []
            warnings: list[dict[str, object]] = []
            issues.extend(_active_assignee_user_issues(store, assignment_rows))
            assignment_rows_by_recording: dict[str, list[dict[str, object]]] = {}
            for row in assignment_rows:
                recording_id = str(row.get("recording_id") or "")
                if recording_id:
                    assignment_rows_by_recording.setdefault(recording_id, []).append(row)
            for recording_id, duplicate_rows in sorted(assignment_rows_by_recording.items()):
                if len(duplicate_rows) <= 1:
                    continue
                warnings.append(
                    {
                        "code": "duplicate_recording_assignment_rows",
                        "recording_id": recording_id,
                        "assignee_users": [row.get("assignee_user") for row in duplicate_rows],
                        "statuses": [str(row.get("status") or "active") for row in duplicate_rows],
                        "source_lines": [
                            int(row["_source_line"])
                            for row in duplicate_rows
                            if row.get("_source_line") is not None
                        ],
                        "details": "Assignment manifest contains multiple rows for one recording; later rows determine the final assignee/status, so confirm one owner before launch.",
                    }
                )
            for row in assignment_rows:
                recording_id = str(row.get("recording_id") or "")
                existing_assignment = existing_assignment_by_recording.get(recording_id)
                if not existing_assignment:
                    continue
                previous_user = str(existing_assignment.get("assignee_user") or "")
                next_user = str(row.get("assignee_user") or "")
                if previous_user and next_user and previous_user != next_user:
                    warnings.append(
                        {
                            "code": "assignment_reassigns_existing_recording",
                            "recording_id": recording_id,
                            "previous_assignee_user": previous_user,
                            "new_assignee_user": next_user,
                            "previous_status": existing_assignment.get("status"),
                            "new_status": row.get("status") or "active",
                            **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                            "details": "Assignment manifest changes the current recording owner; active sessions for this recording will be closed if applied.",
                        }
                    )
            workflows_by_recording: dict[str, set[str]] = {}
            task_rows_by_logical_key: dict[str, list[dict[str, object]]] = {}
            for row in task_rows:
                recording_id = str(row.get("recording_id") or "")
                workflows_by_recording.setdefault(recording_id, set()).add(str(row.get("workflow_kind") or ""))
                logical_key_payload = {
                    "recording_id": recording_id,
                    "workflow_kind": str(row.get("workflow_kind") or ""),
                    "dataset_id": row.get("dataset_id"),
                    "zarr_use": row.get("zarr_use"),
                    "stage_group": row.get("stage_group"),
                    "run_name": row.get("run_name"),
                    "component_name": row.get("component_name"),
                    "scope": row.get("scope") if row.get("scope") is not None else {},
                }
                logical_key = json.dumps(logical_key_payload, sort_keys=True, separators=(",", ":"))
                task_rows_by_logical_key.setdefault(logical_key, []).append(row)
                if recording_id not in planned_assignment_ids:
                    issues.append(
                        {
                            "code": "task_recording_missing_assignment_after_plan",
                            "task_id": row.get("task_id"),
                            "recording_id": recording_id,
                            **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                            "details": "Task recording is not assigned in the imported assignment manifest or existing store.",
                        }
                    )
                    continue
                if str(planned_assignment_status_by_recording.get(recording_id) or "") != "active":
                    warnings.append(
                        {
                            "code": "task_recording_assignment_not_active_after_plan",
                            "task_id": row.get("task_id"),
                            "recording_id": recording_id,
                            "assignment_status": planned_assignment_status_by_recording.get(recording_id),
                            **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                            "details": "Task recording assignment is not active after the imported plan, so the task will not be available to the labeler.",
                        }
                    )
            source_lines_by_recording: dict[str, list[int]] = {}
            for row in task_rows:
                recording_id = str(row.get("recording_id") or "")
                if row.get("_source_line") is not None:
                    source_lines_by_recording.setdefault(recording_id, []).append(int(row["_source_line"]))
            for duplicate_rows in task_rows_by_logical_key.values():
                if len(duplicate_rows) <= 1:
                    continue
                warnings.append(
                    {
                        "code": "duplicate_logical_task_scope",
                        "recording_id": duplicate_rows[0].get("recording_id"),
                        "workflow_kind": duplicate_rows[0].get("workflow_kind"),
                        "task_ids": [row.get("task_id") for row in duplicate_rows],
                        "source_lines": [
                            int(row["_source_line"])
                            for row in duplicate_rows
                            if row.get("_source_line") is not None
                        ],
                        "details": "Multiple task IDs point at the same recording/workflow/component/run/scope; confirm this duplicate logical work is intentional.",
                    }
                )
            for recording_id, workflow_kinds in sorted(workflows_by_recording.items()):
                normalized_workflows = sorted(workflow for workflow in workflow_kinds if workflow)
                if len(normalized_workflows) > 1:
                    warnings.append(
                        {
                            "code": "recording_has_multiple_workflow_kinds",
                            "recording_id": recording_id,
                            "workflow_kinds": normalized_workflows,
                            "source_lines": source_lines_by_recording.get(recording_id, []),
                            "details": "Recording has tasks for multiple workflow kinds; confirm this is intentional labeler workload.",
                        }
                    )
            if issues or (warnings and bool(args.warnings_as_errors)):
                issue_codes = sorted(
                    {
                        str(issue.get("code") or "")
                        for issue in issues
                        if str(issue.get("code") or "")
                    }
                )
                warning_codes = sorted(
                    {
                        str(warning.get("code") or "")
                        for warning in warnings
                        if str(warning.get("code") or "")
                    }
                )
                payload = {
                    "ok": False,
                    "dry_run": not bool(args.apply),
                    "assignments_input": str(args.assignments),
                    "tasks_input": str(args.tasks),
                    "issue_count": len(issues),
                    "warning_count": len(warnings),
                    "issue_codes": issue_codes,
                    "warning_codes": warning_codes,
                    "blocking_warning_count": len(warnings) if bool(args.warnings_as_errors) else 0,
                    "blocking_warning_codes": warning_codes if bool(args.warnings_as_errors) else [],
                    "issues": issues,
                    "warnings": warnings,
                    "warnings_as_errors": bool(args.warnings_as_errors),
                    "assignment_count": len(assignment_rows),
                    "task_count": len(task_rows),
                    **_assignment_control_plane_report_fields(store),
                }
                if args.output:
                    output_path = Path(args.output)
                    if output_path.exists() and not bool(args.overwrite):
                        raise FileExistsError(f"Refusing to overwrite existing batch-plan report: {output_path}")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                if args.html_output:
                    html_output_path = Path(args.html_output)
                    html_output_path.parent.mkdir(parents=True, exist_ok=True)
                    _write_batch_plan_html_report(payload, html_output_path)
                _print_json(payload)
                return 2

            assignment_results: list[dict[str, object]] = []
            assignment_apply_rows = _assignment_rows_for_apply(assignment_rows, apply=bool(args.apply))
            assignment_apply_row_ids = {id(row) for row in assignment_apply_rows} if bool(args.apply) else set()
            applied_assignment_count = 0
            for row in assignment_rows:
                should_apply_assignment = bool(args.apply) and id(row) in assignment_apply_row_ids
                recording_id = str(row["recording_id"])
                assignee_user = str(row["assignee_user"])
                existing = store.get_assignment(recording_id)
                assigned_by = row.get("assigned_by") if row.get("assigned_by") is not None else args.assigned_by
                target = {
                    "recording_id": recording_id,
                    "assignee_user": assignee_user,
                    "assigned_by": assigned_by,
                    "status": str(row.get("status") or "active"),
                    "notes": row.get("notes"),
                }
                existing_target = (
                    {
                        "recording_id": existing.get("recording_id"),
                        "assignee_user": existing.get("assignee_user"),
                        "assigned_by": existing.get("assigned_by"),
                        "status": existing.get("status"),
                        "notes": existing.get("notes"),
                    }
                    if existing is not None
                    else None
                )
                row_warnings: list[dict[str, object]] = []
                if bool(args.apply) and not should_apply_assignment:
                    row_warnings.append(
                        {
                            "code": "duplicate_assignment_row_skipped_for_apply",
                            "recording_id": recording_id,
                            "assignee_user": assignee_user,
                            **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                            "details": "Duplicate assignment input row was reported but not applied; only the final row for each recording mutates ownership/status.",
                        }
                    )
                if should_apply_assignment:
                    transition_result = store.assign_recording_with_session_closure(
                        recording_id=recording_id,
                        assignee_user=assignee_user,
                        assigned_by=str(assigned_by) if assigned_by is not None else None,
                        status=str(target["status"]),
                        notes=str(target["notes"]) if target["notes"] is not None else None,
                    )
                    assignment = transition_result["assignment"]
                    closed_sessions = list(transition_result.get("closed_sessions") or [])
                    assignment_transition = transition_result.get("assignment_transition")
                    applied_assignment_count += 1
                else:
                    assignment = target
                    closed_sessions = []
                    assignment_transition = None
                assignment_results.append(
                    {
                        "recording_id": recording_id,
                        "existing": existing,
                        "assignment": assignment,
                        "would_change": existing_target != target,
                        "applied": should_apply_assignment,
                        "skipped_by_duplicate_apply": bool(args.apply) and not should_apply_assignment,
                        "assignment_transition": assignment_transition,
                        "warnings": row_warnings,
                        "closed_session_count": len(closed_sessions),
                        "closed_sessions": closed_sessions,
                        **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                    }
                )

            task_results: list[dict[str, object]] = []
            for row in task_rows:
                task_id = str(row["task_id"])
                existing = store.get_task(task_id)
                target = {
                    "task_id": task_id,
                    "recording_id": str(row["recording_id"]),
                    "workflow_kind": str(row["workflow_kind"]),
                    "dataset_id": row.get("dataset_id"),
                    "zarr_use": row.get("zarr_use"),
                    "stage_group": row.get("stage_group"),
                    "run_name": row.get("run_name"),
                    "component_name": row.get("component_name"),
                    "title": row.get("title"),
                    "scope": row.get("scope") if row.get("scope") is not None else {},
                    "state": str(row.get("state") or "pending"),
                    "priority": int(row.get("priority") or 0),
                    "notes": row.get("notes"),
                }
                existing_target = (
                    {
                        "task_id": existing.get("task_id"),
                        "recording_id": existing.get("recording_id"),
                        "workflow_kind": existing.get("workflow_kind"),
                        "dataset_id": existing.get("dataset_id"),
                        "zarr_use": existing.get("zarr_use"),
                        "stage_group": existing.get("stage_group"),
                        "run_name": existing.get("run_name"),
                        "component_name": existing.get("component_name"),
                        "title": existing.get("title"),
                        "scope": existing.get("scope") if existing.get("scope") is not None else {},
                        "state": existing.get("state"),
                        "priority": int(existing.get("priority") or 0),
                        "notes": existing.get("notes"),
                    }
                    if existing is not None
                    else None
                )
                if bool(args.apply):
                    task_row = store.upsert_task(
                        task_id=task_id,
                        recording_id=str(target["recording_id"]),
                        workflow_kind=str(target["workflow_kind"]),
                        dataset_id=target.get("dataset_id"),
                        zarr_use=target.get("zarr_use"),
                        stage_group=target.get("stage_group"),
                        run_name=target.get("run_name"),
                        component_name=target.get("component_name"),
                        title=target.get("title"),
                        scope=target.get("scope") if isinstance(target.get("scope"), (Mapping, Sequence)) else {},
                        state=str(target["state"]),
                        priority=int(target["priority"]),
                        notes=target.get("notes"),
                        actor_user=args.actor,
                    )
                else:
                    task_row = target
                task_results.append(
                    {
                        "task_id": task_id,
                        "existing": existing,
                        "task": task_row,
                        "would_change": existing_target != target,
                        **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                    }
                )
            payload = {
                "ok": True,
                "dry_run": not bool(args.apply),
                "assignments_input": str(args.assignments),
                "tasks_input": str(args.tasks),
                "assignment_count": len(assignment_results),
                "assignment_input_row_count": len(assignment_rows),
                "deduplicated_assignment_apply_count": len(assignment_apply_rows) if bool(args.apply) else 0,
                "skipped_duplicate_assignment_apply_count": sum(
                    1
                    for result in assignment_results
                    if bool(result.get("skipped_by_duplicate_apply"))
                ),
                "task_count": len(task_results),
                "applied_assignment_count": applied_assignment_count,
                "applied_task_count": len(task_results) if bool(args.apply) else 0,
                "issue_count": 0,
                "warning_count": len(warnings),
                "issue_codes": [],
                "warning_codes": sorted(
                    {
                        str(warning.get("code") or "")
                        for warning in warnings
                        if str(warning.get("code") or "")
                    }
                ),
                "blocking_warning_count": 0,
                "blocking_warning_codes": [],
                **_assignment_control_plane_report_fields(store),
                "issues": [],
                "warnings": warnings,
                "warnings_as_errors": bool(args.warnings_as_errors),
                "assignments": assignment_results,
                "tasks": task_results,
            }
            if args.output:
                output_path = Path(args.output)
                if output_path.exists() and not bool(args.overwrite):
                    raise FileExistsError(f"Refusing to overwrite existing batch-plan report: {output_path}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            if args.html_output:
                html_output_path = Path(args.html_output)
                html_output_path.parent.mkdir(parents=True, exist_ok=True)
                _write_batch_plan_html_report(payload, html_output_path)
            _print_json(payload)
            return 0
        if args.command == "export-launch-bundle":
            dry_run = bool(args.dry_run)
            secret = "" if dry_run else _link_secret_from_arg(args.link_secret)
            output_dir = Path(args.output_dir)
            zip_output = Path(args.zip_output) if args.zip_output else None
            if zip_output is not None and not dry_run:
                _check_directory_zip_output(output_dir, zip_output, overwrite=bool(args.overwrite))

            generated_at_utc = datetime.now(timezone.utc).isoformat()
            assignments = store.list_assignments(status=None)
            tasks = store.list_tasks(include_completed=True)
            readiness = {
                "store_path": str(store_path),
                **_apply_readiness_warning_policy(
                    _batch_readiness_report(store),
                    warnings_as_errors=bool(args.warnings_as_errors),
                ),
            }
            assignment_ownership_integrity = readiness.get("assignment_ownership_integrity")
            if not isinstance(assignment_ownership_integrity, Mapping):
                assignment_ownership_integrity = _assignment_ownership_integrity(
                    assignments,
                    schema_integrity=store.assignment_schema_integrity(),
                )
            handoffs_dir = output_dir / "handoffs"
            handoff_assignments = store.list_assignments(status=None if bool(args.include_inactive) else "active")
            users = sorted(
                {
                    str(assignment.get("assignee_user") or "").strip()
                    for assignment in handoff_assignments
                    if str(assignment.get("assignee_user") or "").strip()
                }
            )
            used_dir_names: set[str] = set()
            user_dirs = [
                (user, handoffs_dir / _safe_user_handoff_dir_name(user, used_dir_names))
                for user in users
            ]
            planned_files = {
                "assignments": str(output_dir / "assignments.json"),
                "tasks": str(output_dir / "tasks.json"),
                "zarr_backup_plan": str(output_dir / "zarr-backup-plan.json"),
                "zarr_backup_evidence_template": str(output_dir / "zarr-backup-evidence-template.json"),
                "browser_response_security_evidence_template": str(output_dir / "browser-response-security-evidence-template.json"),
                "identity_source_evidence_template": str(output_dir / "identity-source-evidence-template.json"),
                "browser_smoke_evidence_template": str(output_dir / "browser-smoke-evidence-template.json"),
                "disposable_zarr_mutation_smoke_evidence_template": str(output_dir / "disposable-zarr-mutation-smoke-evidence-template.json"),
                "readiness": str(output_dir / "batch-readiness.json"),
                "handoffs_index": str(handoffs_dir / "index.json"),
                "handoffs_html_index": str(handoffs_dir / "index.html"),
                "handoffs_roster": str(handoffs_dir / "labeler-roster.csv"),
                "html_index": str(output_dir / "index.html"),
                "readme": str(output_dir / "launch-readme.txt"),
                "implementation_status": str(output_dir / "implementation-status.txt"),
                "validation_log": str(output_dir / "validation-log-template.md"),
                "validation_checklist": str(output_dir / "validation-checklist.json"),
                "operator_evidence_commands": str(output_dir / "operator-evidence-commands.txt"),
                "launch_evidence_execution_checklist": str(output_dir / "launch-evidence-execution-checklist.txt"),
                "inspect_command": str(output_dir / "inspect-command.txt"),
                "inspection_targets": str(output_dir / "inspection-targets.json"),
                "checksums": str(output_dir / "checksums.json"),
                "manifest": str(output_dir / "manifest.json"),
                **({"bundle_zip": str(zip_output)} if zip_output is not None else {}),
            }
            if dry_run:
                output_exists = output_dir.exists()
                output_non_empty = output_exists and output_dir.is_dir() and any(output_dir.iterdir())
                zarr_backup_plan_payload = _zarr_backup_plan(
                    store=store,
                    store_path=store_path,
                    include_completed=True,
                    include_inactive=bool(args.include_inactive),
                )
                dry_run_warning_count = int(readiness.get("warning_count") or 0) + int(
                    zarr_backup_plan_payload.get("warning_count") or 0
                )
                dry_run_warning_codes = sorted(
                    {
                        *[
                            str(code)
                            for code in readiness.get("warning_codes", [])
                            if str(code).strip()
                        ],
                        *[
                            str(code)
                            for code in zarr_backup_plan_payload.get(
                                "warning_codes",
                                [],
                            )
                            if str(code).strip()
                        ],
                    }
                )
                dry_run_counts = {
                    "assignments": len(assignments),
                    "tasks": len(tasks),
                    "users": len(users),
                    "zarr_backup_targets": zarr_backup_plan_payload["counts"]["zarr_targets"],
                    "zarr_backup_required_targets": zarr_backup_plan_payload["counts"]["backup_required_targets"],
                    "zarr_backup_targets_by_role": zarr_backup_plan_payload["counts"].get("zarr_targets_by_role", {}),
                    "zarr_backup_required_targets_by_role": zarr_backup_plan_payload["counts"].get(
                        "backup_required_targets_by_role", {}
                    ),
                    "zarr_backup_missing_path_tasks": zarr_backup_plan_payload["counts"]["tasks_missing_zarr_path"],
                    "readiness_issues": int(readiness["readiness_issue_count"]),
                    "readiness_warnings": int(readiness["readiness_warning_count"]),
                    "assignment_ownership_active_assignments": int(
                        assignment_ownership_integrity.get("active_assignment_count") or 0
                    ),
                    "assignment_ownership_unique_active_recordings": int(
                        assignment_ownership_integrity.get("unique_active_recording_count") or 0
                    ),
                    "assignment_ownership_duplicate_active_owners": int(
                        assignment_ownership_integrity.get("duplicate_active_owner_count") or 0
                    ),
                }
                dry_run_validation_manifest = {
                    "dry_run": True,
                    "ok": bool(readiness["ok"]),
                    "readiness_ok": bool(readiness["ok"]),
                    "store_path": str(store_path),
                    "base_url": str(args.base_url or "").rstrip("/") or None,
                    "labeler_landing_page_path": "/",
                    "labeler_landing_url": _labeler_landing_url_for_base(str(args.base_url or "").rstrip("/")),
                    "dashboard_url": _dashboard_url_for_base(str(args.base_url or "").rstrip("/")),
                    "labeling_home_page_path": LABELING_HOME_PATH,
                    "labeling_home_url": _labeling_home_url_for_base(str(args.base_url or "").rstrip("/")),
                    "dataset_queue_page_path": DATASET_QUEUE_PATH,
                    "dataset_queue_url": _dataset_queue_url_for_base(str(args.base_url or "").rstrip("/")),
                    "single_owner_policy": _assignment_ownership_policy(),
                    "assignment_ownership_integrity": assignment_ownership_integrity,
                    "generated_at_utc": generated_at_utc,
                    "files": planned_files,
                    **_implementation_status_metadata_fields(
                        checklist_declared_path=str(planned_files.get("implementation_status") or ""),
                    ),
                    "counts": dry_run_counts,
                }
                payload = {
                    "ok": bool(readiness["ok"]),
                    "dry_run": True,
                    "store_path": str(store_path),
                    "output_dir": str(output_dir),
                    "base_url": str(args.base_url or "").rstrip("/") or None,
                    "labeler_landing_page_path": "/",
                    "labeler_landing_url": _labeler_landing_url_for_base(str(args.base_url or "").rstrip("/")),
                    "dashboard_path": DASHBOARD_PATH,
                    "dashboard_url": _dashboard_url_for_base(str(args.base_url or "").rstrip("/")),
                    "dataset_queue_page_path": DATASET_QUEUE_PATH,
                    "dataset_queue_url": _dataset_queue_url_for_base(str(args.base_url or "").rstrip("/")),
                    "labeling_home_page_path": LABELING_HOME_PATH,
                    "labeling_home_url": _labeling_home_url_for_base(str(args.base_url or "").rstrip("/")),
                    "labeler_safety": _labeler_safety_policy(),
                    "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
                    "operator_authorization_policy": _operator_authorization_policy(),
                    "operator_recovery_policy": _operator_recovery_policy(),
                    "zarr_backup_policy": _zarr_backup_policy(),
                    "mutation_audit_policy": _mutation_audit_policy(),
                    "browser_mutation_write_policy": _browser_mutation_write_policy(),
                    "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(),
                    "dataset_queue_direct_start_policy": _dataset_queue_direct_start_policy(),
                    "runtime_operator_validation_gate_cli_policy": _runtime_operator_validation_gate_cli_policy(),
                    "browser_response_security_policy": _browser_response_security_policy(),
                    "session_guard_policy": _session_guard_policy(),
                    "task_state_policy": _browser_task_state_policy(),
                    "signed_link_policy": _browser_signed_link_policy(),
                    "browser_workflows": _browser_workflow_capabilities(),
                    "single_owner_policy": _assignment_ownership_policy(),
                    "assignment_ownership_integrity": assignment_ownership_integrity,
                    **_implementation_status_metadata_fields(
                        checklist_declared_path=str(planned_files.get("implementation_status") or ""),
                    ),
                    "include_completed": bool(args.include_completed),
                    "include_inactive": bool(args.include_inactive),
                    "include_audit_events": bool(args.include_audit_events),
                    "warnings_as_errors": bool(args.warnings_as_errors),
                    "warning_count": dry_run_warning_count,
                    "warning_codes": dry_run_warning_codes,
                    "blocking_warning_count": (
                        dry_run_warning_count if bool(args.warnings_as_errors) else 0
                    ),
                    "blocking_warning_codes": (
                        dry_run_warning_codes if bool(args.warnings_as_errors) else []
                    ),
                    "audit_filters": {
                        "since_utc": args.audit_since_utc,
                        "until_utc": args.audit_until_utc,
                        "limit": args.audit_limit,
                    },
                    "counts": dry_run_counts,
                    "zarr_backup_plan_summary": {
                        "schema": zarr_backup_plan_payload["schema"],
                        "policy": zarr_backup_plan_payload["policy"],
                        "counts": zarr_backup_plan_payload["counts"],
                        "warning_count": zarr_backup_plan_payload["warning_count"],
                        "warning_codes": zarr_backup_plan_payload["warning_codes"],
                    },
                    "output_state": {
                        "exists": output_exists,
                        "non_empty": output_non_empty,
                        "zip_exists": bool(zip_output is not None and zip_output.exists()),
                        "overwrite_requested": bool(args.overwrite),
                    },
                    "planned_files": planned_files,
                    "validation_checklist": _web_labeling_validation_checklist_payload(
                        dry_run_validation_manifest,
                        bundle_label="launch bundle dry-run",
                    ),
                    "handoff_users": [
                        {"user": user, "output_dir": str(user_dir)}
                        for user, user_dir in user_dirs
                    ],
                    "readiness": readiness,
                }
                _print_json(payload)
                return 0 if bool(payload["ok"]) else 2
            _check_launch_bundle_overwrite_target(
                output_dir=output_dir,
                expected_user_dirs=[user_dir for _, user_dir in user_dirs],
                include_audit_events=bool(args.include_audit_events),
                overwrite=bool(args.overwrite),
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            assignments_payload = {
                "ok": True,
                "store_path": str(store_path),
                "generated_at_utc": generated_at_utc,
                "filters": {},
                "single_owner_policy": _assignment_ownership_policy(),
                "assignment_snapshot": _assignment_snapshot_from_assignments(assignments),
                "assignment_ownership_integrity": assignment_ownership_integrity,
                "count": len(assignments),
                "assignments": assignments,
            }
            tasks_payload = {
                "ok": True,
                "store_path": str(store_path),
                "generated_at_utc": generated_at_utc,
                "filters": {"include_completed": True},
                "count": len(tasks),
                "tasks": tasks,
            }
            zarr_backup_plan_payload = _zarr_backup_plan(
                store=store,
                store_path=store_path,
                include_completed=True,
                include_inactive=bool(args.include_inactive),
            )
            zarr_backup_evidence_template_payload = _zarr_backup_evidence_template(
                zarr_backup_plan_payload
            )
            browser_response_security_evidence_template_payload = (
                _browser_response_security_evidence_template(
                    base_url=str(args.base_url or "").rstrip("/") or None
                )
            )
            identity_source_evidence_template_payload = _identity_source_evidence_template(
                base_url=str(args.base_url or "").rstrip("/") or None,
                users=users,
            )
            browser_smoke_evidence_template_payload = _browser_smoke_evidence_template(
                base_url=str(args.base_url or "").rstrip("/") or None,
                users=users,
            )
            disposable_zarr_mutation_smoke_evidence_template_payload = (
                _disposable_zarr_mutation_smoke_evidence_template(zarr_backup_plan_payload)
            )
            (output_dir / "assignments.json").write_text(
                json.dumps(assignments_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "tasks.json").write_text(
                json.dumps(tasks_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "zarr-backup-plan.json").write_text(
                json.dumps(zarr_backup_plan_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "zarr-backup-evidence-template.json").write_text(
                json.dumps(zarr_backup_evidence_template_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "browser-response-security-evidence-template.json").write_text(
                json.dumps(
                    browser_response_security_evidence_template_payload,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "identity-source-evidence-template.json").write_text(
                json.dumps(identity_source_evidence_template_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "browser-smoke-evidence-template.json").write_text(
                json.dumps(browser_smoke_evidence_template_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "disposable-zarr-mutation-smoke-evidence-template.json").write_text(
                json.dumps(
                    disposable_zarr_mutation_smoke_evidence_template_payload,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "batch-readiness.json").write_text(
                json.dumps(readiness, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            manifests = [
                _write_user_handoff_bundle(
                    store=store,
                    store_path=store_path,
                    user=user,
                    output_dir=user_dir,
                    secret=secret,
                    base_url=args.base_url,
                    ttl_seconds=int(args.ttl_seconds),
                    include_completed=bool(args.include_completed),
                    overwrite=bool(args.overwrite),
                )
                for user, user_dir in user_dirs
            ]
            handoff_sendability = _handoff_sendability_summary(manifests)
            handoffs_index_path = handoffs_dir / "index.json"
            handoffs_html_index_path = handoffs_dir / "index.html"
            handoffs_readme_path = handoffs_dir / "handoff-readme.txt"
            handoffs_roster_path = handoffs_dir / "labeler-roster.csv"
            handoffs_index = {
                "ok": all(bool(manifest["ok"]) for manifest in manifests)
                and int(handoff_sendability["not_ready_to_send_count"]) == 0,
                "store_checks_ok": all(bool(manifest["ok"]) for manifest in manifests),
                "store_path": str(store_path),
                "output_dir": str(handoffs_dir),
                "include_completed": bool(args.include_completed),
                "include_inactive": bool(args.include_inactive),
                "base_url": str(args.base_url or "").rstrip("/") or None,
                "labeler_landing_page_path": "/",
                "labeler_landing_url": _labeler_landing_url_for_base(str(args.base_url or "").rstrip("/")),
                "dashboard_path": DASHBOARD_PATH,
                "dashboard_url": _dashboard_url_for_base(str(args.base_url or "").rstrip("/")),
                "dataset_queue_page_path": DATASET_QUEUE_PATH,
                "dataset_queue_url": _dataset_queue_url_for_base(str(args.base_url or "").rstrip("/")),
                "labeling_home_page_path": LABELING_HOME_PATH,
                "labeling_home_url": _labeling_home_url_for_base(str(args.base_url or "").rstrip("/")),
                "personal_work_page_path": PERSONAL_WORK_PATH,
                "personal_work_url": _personal_work_url_for_base(str(args.base_url or "").rstrip("/")),
                "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
                "personal_dataset_queue_url": _personal_dataset_queue_url_for_base(
                    str(args.base_url or "").rstrip("/")
                ),
                "labeler_safety": _labeler_safety_policy(),
                "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
                "operator_authorization_policy": _operator_authorization_policy(),
                "operator_recovery_policy": _operator_recovery_policy(),
                "operator_validation_visibility_policy": _operator_validation_visibility_policy(),
                "operator_validation_command_templates": _operator_validation_command_templates(),
                "zarr_backup_policy": _zarr_backup_policy(),
                "mutation_audit_policy": _mutation_audit_policy(),
                "browser_mutation_write_policy": _browser_mutation_write_policy(),
                "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(),
                "dataset_queue_direct_start_policy": _dataset_queue_direct_start_policy(),
                "runtime_operator_validation_gate_cli_policy": _runtime_operator_validation_gate_cli_policy(),
                "safe_share_gate": _safe_share_gate_policy(),
                **_safe_share_gate_flat_fields(),
                "browser_response_security_policy": _browser_response_security_policy(),
                "session_guard_policy": _session_guard_policy(),
                "task_state_policy": _browser_task_state_policy(),
                "signed_link_policy": _browser_signed_link_policy(),
                "browser_workflows": _browser_workflow_capabilities(),
                "single_owner_policy": _assignment_ownership_policy(),
                "assignment_snapshot": _assignment_snapshot_from_assignments(assignments),
                "assignment_ownership_integrity": assignment_ownership_integrity,
                "ttl_seconds": _effective_signed_link_ttl_seconds(int(args.ttl_seconds)),
                "generated_at_utc": generated_at_utc,
                "progress_summary": _sum_handoff_progress_summary(manifests),
                "dataset_queue_summary": _sum_handoff_dataset_queue_summary(manifests),
                "files": {
                    "index": str(handoffs_index_path),
                    "html_index": str(handoffs_html_index_path),
                    "readme": str(handoffs_readme_path),
                    "labeler_roster": str(handoffs_roster_path),
                },
                "counts": {
                    "users": len(manifests),
                    "handoffs": len(manifests),
                    "failed_store_checks": sum(1 for manifest in manifests if not bool(manifest["ok"])),
                    "ready_to_send": handoff_sendability["ready_to_send_count"],
                    "not_ready_to_send": handoff_sendability["not_ready_to_send_count"],
                    "sendability_reasons": _count_handoff_sendability_reasons(handoff_sendability["warnings"]),
                    "waiting_datasets": sum(
                        int((manifest.get("counts") or {}).get("waiting_datasets") or 0)
                        for manifest in manifests
                        if isinstance(manifest.get("counts"), Mapping)
                    ),
                    "dataset_open_tasks": sum(
                        int((manifest.get("counts") or {}).get("dataset_open_tasks") or 0)
                        for manifest in manifests
                        if isinstance(manifest.get("counts"), Mapping)
                    ),
                    "dataset_queue_states": _handoff_dataset_queue_state_counts(manifests),
                    "dataset_queue_blocked_start_users": [
                        str(manifest.get("user") or "")
                        for manifest in manifests
                        if _handoff_dataset_queue_blocks_labeler_start(manifest)
                    ],
                    "recordings_without_open_tasks": sum(
                        int((manifest.get("counts") or {}).get("recordings_without_open_tasks") or 0)
                        for manifest in manifests
                        if isinstance(manifest.get("counts"), Mapping)
                    ),
                    "recordings_without_open_tasks_by_reason": _sum_recordings_without_open_tasks_by_reason(manifests),
                    "recordings_without_open_tasks_actions": _recordings_without_open_tasks_actions(
                        _sum_recordings_without_open_tasks_by_reason(manifests)
                    ),
                    "redacted_summary_fields": sum(
                        int((manifest.get("counts") or {}).get("redacted_summary_fields") or 0)
                        for manifest in manifests
                        if isinstance(manifest.get("counts"), Mapping)
                    ),
                },
                "sendability_warnings": handoff_sendability["warnings"],
                "sendability_actions": _handoff_sendability_actions(
                    _count_handoff_sendability_reasons(handoff_sendability["warnings"]).keys()
                ),
                "handoffs": [
                    {
                        "user": manifest["user"],
                        "output_dir": manifest["output_dir"],
                        "index_html": manifest["files"]["html_index"],
                        "message": manifest["files"]["message"],
                        "manifest": manifest["files"]["manifest"],
                        "ready_to_send": _handoff_ready_to_send(manifest),
                        "operator_validation_visibility_policy": manifest.get(
                            "operator_validation_visibility_policy",
                            _operator_validation_visibility_policy(),
                        ),
                        **_operator_validation_public_fields(manifest),
                        **_operator_validation_gate_flat_fields(manifest),
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
                        **_operator_validation_command_template_fields(manifest),
                        "sendability_reasons": _handoff_sendability_reasons(manifest),
                        "sendability_actions": _handoff_sendability_actions(_handoff_sendability_reasons(manifest)),
                        "dashboard_url": str(manifest.get("dashboard_url") or ""),
                        "labeler_landing_url": str(manifest.get("labeler_landing_url") or ""),
                        "expected_user_labeler_landing_url": str(manifest.get("expected_user_labeler_landing_url") or ""),
                        "expected_user_dashboard_url": str(manifest.get("expected_user_dashboard_url") or ""),
                        "expected_user_dataset_queue_url": str(manifest.get("expected_user_dataset_queue_url") or ""),
                        "expected_user_personal_work_url": str(
                            manifest.get("expected_user_personal_work_url") or ""
                        ),
                        "expected_user_personal_dataset_queue_url": str(
                            manifest.get("expected_user_personal_dataset_queue_url") or ""
                        ),
                        "personalized_labeler_entrypoint": str(
                            manifest.get("personalized_labeler_entrypoint") or ""
                        ),
                        "personalized_labeler_entry_url": str(
                            manifest.get("personalized_labeler_entry_url") or ""
                        ),
                        "queue_first_entry_contract": manifest.get(
                            "queue_first_entry_contract", {}
                        ),
                        "personalized_launch_readiness": manifest.get(
                            "personalized_launch_readiness",
                            _personalized_launch_readiness_summary(manifest),
                        ),
                        "expected_user_identity_probe_url": str(manifest.get("expected_user_identity_probe_url") or ""),
                        "labeler_safety": manifest.get("labeler_safety", _labeler_safety_policy()),
                        "labeler_route_authorization_policy": manifest.get("labeler_route_authorization_policy", _labeler_route_authorization_policy()),
                        "operator_authorization_policy": manifest.get(
                            "operator_authorization_policy", _operator_authorization_policy()
                        ),
                        "operator_recovery_policy": manifest.get("operator_recovery_policy", _operator_recovery_policy()),
                        "zarr_backup_policy": manifest.get("zarr_backup_policy", _zarr_backup_policy()),
                        **_handoff_zarr_backup_fields(manifest),
                        "mutation_audit_policy": manifest.get("mutation_audit_policy", _mutation_audit_policy()),
                        **_handoff_mutation_audit_fields(manifest),
                        "browser_mutation_write_policy": manifest.get(
                            "browser_mutation_write_policy", _browser_mutation_write_policy()
                        ),
                        "browser_mutation_write_checklist": manifest.get(
                            "browser_mutation_write_checklist",
                            _browser_mutation_write_runtime_checklist(
                                manifest.get("browser_mutation_write_policy")
                                if isinstance(manifest.get("browser_mutation_write_policy"), Mapping)
                                else None
                            ),
                        ),
                        "dataset_queue_direct_start_policy": manifest.get(
                            "dataset_queue_direct_start_policy", _dataset_queue_direct_start_policy()
                        ),
                        **_dataset_queue_direct_start_policy_fields(
                            manifest.get("dataset_queue_direct_start_policy")
                            if isinstance(manifest.get("dataset_queue_direct_start_policy"), Mapping)
                            else None
                        ),
                        "runtime_operator_validation_gate_cli_policy": manifest.get(
                            "runtime_operator_validation_gate_cli_policy",
                            _runtime_operator_validation_gate_cli_policy(),
                        ),
                        **_runtime_operator_validation_gate_cli_policy_fields(
                            manifest.get("runtime_operator_validation_gate_cli_policy")
                            if isinstance(
                                manifest.get("runtime_operator_validation_gate_cli_policy"),
                                Mapping,
                            )
                            else None
                        ),
                        "direct_browser_start_contract_summary": manifest.get(
                            "direct_browser_start_contract_summary", {}
                        ),
                        **_direct_browser_start_contract_summary_fields(
                            manifest.get("direct_browser_start_contract_summary")
                            if isinstance(manifest.get("direct_browser_start_contract_summary"), Mapping)
                            else None
                        ),
                        **_handoff_browser_mutation_write_fields(manifest),
                        **_browser_mutation_target_contract_compact_fields(
                            manifest.get("browser_mutation_write_checklist")
                            if isinstance(manifest.get("browser_mutation_write_checklist"), Mapping)
                            else _browser_mutation_write_runtime_checklist(
                                manifest.get("browser_mutation_write_policy")
                                if isinstance(manifest.get("browser_mutation_write_policy"), Mapping)
                                else None
                            ),
                            user=str(manifest.get("user") or ""),
                        ),
                        **_direct_browser_start_contract_compact_fields(
                            manifest.get("dataset_queue_direct_start_policy")
                            if isinstance(manifest.get("dataset_queue_direct_start_policy"), Mapping)
                            else None,
                            user=str(manifest.get("user") or ""),
                        ),
                        "single_owner_policy_contract_met": (
                            bool(manifest.get("single_owner_policy_contract_met"))
                            if "single_owner_policy_contract_met" in manifest
                            else bool(
                                _handoff_assignment_ownership_fields(manifest).get(
                                    "assignment_ownership_contract_ready"
                                )
                            )
                            and int(
                                _handoff_assignment_ownership_fields(manifest).get(
                                    "assignment_duplicate_active_owner_count"
                                )
                                or 0
                            )
                            == 0
                        ),
                        **_handoff_operator_recovery_fields(manifest),
                        "browser_response_security_policy": manifest.get(
                            "browser_response_security_policy", _browser_response_security_policy()
                        ),
                        **_handoff_browser_response_security_fields(manifest),
                        "session_guard_policy": manifest.get("session_guard_policy", _session_guard_policy()),
                        **_handoff_session_guard_fields(manifest),
                        "task_state_policy": manifest.get("task_state_policy", _browser_task_state_policy()),
                        **_handoff_task_state_policy_fields(manifest),
                        "signed_link_policy": manifest.get("signed_link_policy", _browser_signed_link_policy()),
                        **_handoff_signed_link_policy_fields(manifest),
                        "browser_workflows": manifest.get("browser_workflows", _browser_workflow_capabilities()),
                        "progress_summary": manifest.get("progress_summary", {}),
                        "dataset_queue_summary": manifest.get("dataset_queue_summary", {}),
                        "dataset_queue_state": manifest.get("dataset_queue_state", {}),
                        "dataset_queue_state_code": str(
                            (manifest.get("dataset_queue_state") or {}).get("code") or ""
                        )
                        if isinstance(manifest.get("dataset_queue_state"), Mapping)
                        else "",
                        "dataset_queue_state_title": str(
                            (manifest.get("dataset_queue_state") or {}).get("title") or ""
                        )
                        if isinstance(manifest.get("dataset_queue_state"), Mapping)
                        else "",
                        "labeler_work_completion": manifest.get("labeler_work_completion", {}),
                        **_labeler_work_completion_fields(
                            manifest.get("labeler_work_completion")
                            if isinstance(manifest.get("labeler_work_completion"), Mapping)
                            else None
                        ),
                        "dataset_queue_blocks_labeler_start": _handoff_dataset_queue_blocks_labeler_start(manifest),
                        **_handoff_known_user_status_fields(manifest),
                        **_handoff_assignment_ownership_fields(manifest),
                        **_handoff_entry_artifact_fields(manifest),
                        **_handoff_dataset_queue_start_fields(manifest),
                        "reassignment_session_safety": manifest.get(
                            "reassignment_session_safety",
                            {},
                        ),
                        **_reassignment_session_safety_flat_fields(
                            manifest.get("reassignment_session_safety")
                            if isinstance(manifest.get("reassignment_session_safety"), Mapping)
                            else None
                        ),
                        **_handoff_labeler_safety_fields(manifest),
                        **_handoff_labeler_route_authorization_fields(manifest),
                        "dataset_queue_preview_url": manifest.get("dataset_queue_preview_url", ""),
                        "canonical_dataset_queue_preview_url": manifest.get(
                            "canonical_dataset_queue_preview_url",
                            manifest.get("expected_user_dataset_queue_url", ""),
                        ),
                        "recordings_without_open_tasks_actions": (manifest.get("counts") or {}).get(
                            "recordings_without_open_tasks_actions", []
                        )
                        if isinstance(manifest.get("counts"), Mapping)
                        else [],
                        "links_expire_at_utc": manifest.get("links_expire_at_utc"),
                        "files": manifest["files"],
                        "ok": manifest["ok"],
                        "counts": manifest["counts"],
                    }
                    for manifest in manifests
                ],
            }
            handoffs_index.update(
                _operator_validation_invitation_fields(
                    _web_labeling_validation_checklist_payload(
                        handoffs_index,
                        bundle_label="multi-user handoff bundle",
                    )
                )
            )
            handoffs_safe_share_gate = (
                handoffs_index.get("safe_share_gate")
                if isinstance(handoffs_index.get("safe_share_gate"), Mapping)
                else _safe_share_gate_policy()
            )
            handoffs_index["safe_share_gate"] = handoffs_safe_share_gate
            handoffs_index.update(_safe_share_gate_flat_fields(handoffs_safe_share_gate))
            handoffs_index.update(
                _safe_share_checklist_gate_status_fields_from_operator_validation(
                    handoffs_index,
                    safe_share_gate=handoffs_safe_share_gate,
                )
            )
            handoffs_dir.mkdir(parents=True, exist_ok=True)
            handoffs_index_path.write_text(
                json.dumps(handoffs_index, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_user_handoffs_html_index(handoffs_index, handoffs_html_index_path)
            _write_user_handoffs_readme(handoffs_index, handoffs_readme_path)
            _write_user_handoffs_roster_csv(handoffs_index, handoffs_roster_path)

            audit_files: dict[str, str] = {}
            audit_counts: dict[str, int] = {}
            if bool(args.include_audit_events):
                audit_dir = output_dir / "audit"
                task_events = _filter_audit_rows(
                    store.list_events(),
                    since_utc=args.audit_since_utc,
                    until_utc=args.audit_until_utc,
                    limit=args.audit_limit,
                )
                assignment_events = _filter_audit_rows(
                    store.list_assignment_events(),
                    since_utc=args.audit_since_utc,
                    until_utc=args.audit_until_utc,
                    limit=args.audit_limit,
                )
                task_definition_events = _filter_audit_rows(
                    store.list_task_definition_events(),
                    since_utc=args.audit_since_utc,
                    until_utc=args.audit_until_utc,
                    limit=args.audit_limit,
                )
                audit_paths = {
                    "audit_task_events": audit_dir / "task-events.jsonl",
                    "audit_assignment_events": audit_dir / "assignment-events.jsonl",
                    "audit_task_definition_events": audit_dir / "task-definition-events.jsonl",
                }
                _write_jsonl_rows(audit_paths["audit_task_events"], task_events)
                _write_jsonl_rows(audit_paths["audit_assignment_events"], assignment_events)
                _write_jsonl_rows(audit_paths["audit_task_definition_events"], task_definition_events)
                audit_files = {key: str(path) for key, path in audit_paths.items()}
                audit_counts = {
                    "audit_task_events": len(task_events),
                    "audit_assignment_events": len(assignment_events),
                    "audit_task_definition_events": len(task_definition_events),
                }

            launch_safe_share_gate = (
                handoffs_index.get("safe_share_gate")
                if isinstance(handoffs_index.get("safe_share_gate"), Mapping)
                else _safe_share_gate_policy()
            )
            launch_safe_share_fields = _safe_share_gate_flat_fields(launch_safe_share_gate)
            launch_safe_share_checklist_fields = (
                _safe_share_checklist_gate_status_fields_from_operator_validation(
                    handoffs_index,
                    safe_share_gate=launch_safe_share_gate,
                )
            )
            manifest = {
                "ok": bool(readiness["ok"]) and bool(handoffs_index["ok"]),
                "store_path": str(store_path),
                "output_dir": str(output_dir),
                "base_url": str(args.base_url or "").rstrip("/") or None,
                "labeler_landing_page_path": "/",
                "labeler_landing_url": _labeler_landing_url_for_base(str(args.base_url or "").rstrip("/")),
                "dashboard_path": DASHBOARD_PATH,
                "dashboard_url": _dashboard_url_for_base(str(args.base_url or "").rstrip("/")),
                "dataset_queue_page_path": DATASET_QUEUE_PATH,
                "dataset_queue_url": _dataset_queue_url_for_base(str(args.base_url or "").rstrip("/")),
                "labeling_home_page_path": LABELING_HOME_PATH,
                "labeling_home_url": _labeling_home_url_for_base(str(args.base_url or "").rstrip("/")),
                "personal_work_page_path": PERSONAL_WORK_PATH,
                "personal_work_url": _personal_work_url_for_base(str(args.base_url or "").rstrip("/")),
                "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
                "personal_dataset_queue_url": _personal_dataset_queue_url_for_base(
                    str(args.base_url or "").rstrip("/")
                ),
                "labeler_safety": _labeler_safety_policy(),
                "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
                "operator_authorization_policy": _operator_authorization_policy(),
                "operator_recovery_policy": _operator_recovery_policy(),
                "operator_validation_visibility_policy": _operator_validation_visibility_policy(),
                "operator_validation_command_templates": _operator_validation_command_templates(),
                "safe_share_gate": launch_safe_share_gate,
                **launch_safe_share_fields,
                **launch_safe_share_checklist_fields,
                "zarr_backup_policy": _zarr_backup_policy(),
                "mutation_audit_policy": _mutation_audit_policy(),
                "browser_mutation_write_policy": _browser_mutation_write_policy(),
                "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(),
                "browser_response_security_policy": _browser_response_security_policy(),
                "session_guard_policy": _session_guard_policy(),
                "task_state_policy": _browser_task_state_policy(),
                "signed_link_policy": _browser_signed_link_policy(),
                "browser_workflows": _browser_workflow_capabilities(),
                "single_owner_policy": _assignment_ownership_policy(),
                "assignment_snapshot": _assignment_snapshot_from_assignments(assignments),
                "assignment_ownership_integrity": assignment_ownership_integrity,
                "generated_at_utc": generated_at_utc,
                "readiness_ok": bool(readiness["ok"]),
                "handoffs_ok": bool(handoffs_index["ok"]),
                "handoff_store_checks_ok": bool(handoffs_index.get("store_checks_ok")),
                "include_audit_events": bool(args.include_audit_events),
                "warnings_as_errors": bool(args.warnings_as_errors),
                "audit_filters": {
                    "since_utc": args.audit_since_utc,
                    "until_utc": args.audit_until_utc,
                    "limit": args.audit_limit,
                },
                "files": {
                    "assignments": str(output_dir / "assignments.json"),
                    "tasks": str(output_dir / "tasks.json"),
                    "zarr_backup_plan": str(output_dir / "zarr-backup-plan.json"),
                    "zarr_backup_evidence_template": str(output_dir / "zarr-backup-evidence-template.json"),
                    "browser_response_security_evidence_template": str(output_dir / "browser-response-security-evidence-template.json"),
                    "identity_source_evidence_template": str(output_dir / "identity-source-evidence-template.json"),
                    "browser_smoke_evidence_template": str(output_dir / "browser-smoke-evidence-template.json"),
                    "disposable_zarr_mutation_smoke_evidence_template": str(output_dir / "disposable-zarr-mutation-smoke-evidence-template.json"),
                    "readiness": str(output_dir / "batch-readiness.json"),
                    "handoffs_index": str(handoffs_index_path),
                "handoffs_html_index": str(handoffs_html_index_path),
                "handoffs_roster": str(handoffs_roster_path),
                "html_index": str(output_dir / "index.html"),
                "readme": str(output_dir / "launch-readme.txt"),
                "implementation_status": str(output_dir / "implementation-status.txt"),
                "validation_log": str(output_dir / "validation-log-template.md"),
                "validation_checklist": str(output_dir / "validation-checklist.json"),
                "operator_evidence_commands": str(output_dir / "operator-evidence-commands.txt"),
                "launch_evidence_execution_checklist": str(output_dir / "launch-evidence-execution-checklist.txt"),
                "inspect_command": str(output_dir / "inspect-command.txt"),
                    "inspection_targets": str(output_dir / "inspection-targets.json"),
                    "checksums": str(output_dir / "checksums.json"),
                    **audit_files,
                    **({"bundle_zip": str(zip_output)} if zip_output is not None else {}),
                },
                **_implementation_status_metadata_fields(
                    checklist_declared_path=str(output_dir / "implementation-status.txt"),
                ),
                "counts": {
                    "assignments": len(assignments),
                    "tasks": len(tasks),
                    "zarr_backup_targets": zarr_backup_plan_payload["counts"]["zarr_targets"],
                    "zarr_backup_required_targets": zarr_backup_plan_payload["counts"]["backup_required_targets"],
                    "zarr_backup_targets_by_role": zarr_backup_plan_payload["counts"].get("zarr_targets_by_role", {}),
                    "zarr_backup_required_targets_by_role": zarr_backup_plan_payload["counts"].get(
                        "backup_required_targets_by_role", {}
                    ),
                    "zarr_backup_missing_path_tasks": zarr_backup_plan_payload["counts"]["tasks_missing_zarr_path"],
                    "users": len(manifests),
                    "handoffs": len(manifests),
                    "handoffs_ready_to_send": handoffs_index["counts"]["ready_to_send"],
                    "handoffs_not_ready_to_send": handoffs_index["counts"]["not_ready_to_send"],
                    "handoff_store_checks_ok": bool(handoffs_index.get("store_checks_ok")),
                    "handoff_sendability_reasons": handoffs_index["counts"].get("sendability_reasons", {}),
                    "handoff_waiting_datasets": handoffs_index["counts"].get("waiting_datasets", 0),
                    "handoff_dataset_open_tasks": handoffs_index["counts"].get("dataset_open_tasks", 0),
                    "dataset_queue_states": handoffs_index["counts"].get("dataset_queue_states", {}),
                    "dataset_queue_blocked_start_users": handoffs_index["counts"].get("dataset_queue_blocked_start_users", []),
                    "handoff_dataset_queue_states": handoffs_index["counts"].get("dataset_queue_states", {}),
                    "handoff_dataset_queue_blocked_start_users": handoffs_index["counts"].get("dataset_queue_blocked_start_users", []),
                    "handoff_recordings_without_open_tasks": handoffs_index["counts"].get("recordings_without_open_tasks", 0),
                    "handoff_recordings_without_open_tasks_by_reason": handoffs_index["counts"].get("recordings_without_open_tasks_by_reason", {}),
                    "handoff_recordings_without_open_tasks_actions": handoffs_index["counts"].get("recordings_without_open_tasks_actions", []),
                    "handoff_redacted_summary_fields": handoffs_index["counts"].get("redacted_summary_fields", 0),
                    "readiness_issues": int(readiness["readiness_issue_count"]),
                    "readiness_warnings": int(readiness["readiness_warning_count"]),
                    "assignment_ownership_active_assignments": int(
                        assignment_ownership_integrity.get("active_assignment_count") or 0
                    ),
                    "assignment_ownership_unique_active_recordings": int(
                        assignment_ownership_integrity.get("unique_active_recording_count") or 0
                    ),
                    "assignment_ownership_duplicate_active_owners": int(
                        assignment_ownership_integrity.get("duplicate_active_owner_count") or 0
                    ),
                    "readiness_active_recordings_without_open_tasks_by_reason": (
                        readiness.get("counts", {}).get("active_recordings_without_open_tasks_by_reason", {})
                        if isinstance(readiness.get("counts"), Mapping)
                        else {}
                    ),
                    "readiness_active_recordings_without_open_tasks_actions": (
                        readiness.get("counts", {}).get("active_recordings_without_open_tasks_actions", [])
                        if isinstance(readiness.get("counts"), Mapping)
                        else []
                    ),
                    **audit_counts,
                },
                "handoff_sendability_warnings": handoffs_index.get("sendability_warnings", []),
                "handoff_sendability_actions": handoffs_index.get("sendability_actions", []),
            }
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_launch_bundle_readme(manifest, output_dir / "launch-readme.txt")
            _write_launch_bundle_implementation_status(
                manifest,
                output_dir / "implementation-status.txt",
            )
            _write_launch_bundle_validation_log(manifest, output_dir / "validation-log-template.md")
            _write_launch_bundle_validation_checklist(manifest, output_dir / "validation-checklist.json")
            _write_launch_bundle_operator_evidence_commands(
                manifest,
                output_dir / "operator-evidence-commands.txt",
            )
            _write_launch_bundle_launch_evidence_execution_checklist(
                manifest,
                output_dir / "launch-evidence-execution-checklist.txt",
            )
            _write_launch_bundle_inspect_command(
                store_path=store_path,
                output_dir=output_dir,
                zip_output=zip_output,
                output_path=output_dir / "inspect-command.txt",
            )
            _write_launch_bundle_inspection_targets(
                store_path=store_path,
                output_dir=output_dir,
                zip_output=zip_output,
                output_path=output_dir / "inspection-targets.json",
            )
            _write_launch_bundle_html_index(manifest, output_dir / "index.html")
            _write_directory_checksums(output_dir, output_dir / "checksums.json")
            if zip_output is not None:
                _write_directory_zip(output_dir, zip_output, overwrite=bool(args.overwrite))
            _print_json(manifest)
            return 0 if bool(manifest["ok"]) else 2
        if args.command == "inspect-handoff":
            report = _inspect_handoff_package(
                Path(args.path),
                store=store,
                require_shareable=bool(args.require_shareable),
            )
            if args.output:
                output_path = Path(args.output)
                if output_path.exists() and not bool(args.overwrite):
                    raise FileExistsError(f"Refusing to overwrite existing handoff inspection report: {output_path}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            _print_json(report)
            if bool(args.require_shareable) and not bool(report.get("labeler_links_safe_to_share")):
                return 2
            return 0 if bool(report["ok"]) else 2
        if args.command == "update-validation-checklist":
            payload = _update_validation_checklist_file(
                path=Path(args.path),
                gate_id=str(args.gate),
                status=str(args.status),
                evidence=args.evidence or [],
                evidence_files=args.evidence_file or [],
                operator=args.operator,
                append_log=Path(args.append_log) if args.append_log else None,
                output=Path(args.output) if args.output else None,
                overwrite=bool(args.overwrite),
            )
            _print_json(payload)
            return 0
        if args.command == "export-user-handoff":
            secret = _link_secret_from_arg(args.link_secret)
            manifest = _write_user_handoff_bundle(
                store=store,
                store_path=store_path,
                user=str(args.user),
                output_dir=Path(args.output_dir),
                secret=secret,
                base_url=args.base_url,
                ttl_seconds=int(args.ttl_seconds),
                include_completed=bool(args.include_completed),
                overwrite=bool(args.overwrite),
                zip_output=Path(args.zip_output) if args.zip_output else None,
            )
            _print_json(manifest)
            return 0 if bool(manifest["ok"]) else 2
        if args.command == "export-user-handoffs":
            secret = _link_secret_from_arg(args.link_secret)
            output_dir = Path(args.output_dir)
            assignments = store.list_assignments(status=None if bool(args.include_inactive) else "active")
            users = sorted(
                {
                    str(assignment.get("assignee_user") or "").strip()
                    for assignment in assignments
                    if str(assignment.get("assignee_user") or "").strip()
                }
            )
            used_dir_names: set[str] = set()
            user_dirs = [
                (user, output_dir / _safe_user_handoff_dir_name(user, used_dir_names))
                for user in users
            ]
            index_path = output_dir / "index.json"
            html_index_path = output_dir / "index.html"
            readme_path = output_dir / "handoff-readme.txt"
            roster_path = output_dir / "labeler-roster.csv"
            validation_log_path = output_dir / "validation-log-template.md"
            validation_checklist_path = output_dir / "validation-checklist.json"
            zip_output = Path(args.zip_output) if args.zip_output else None
            if zip_output is not None:
                _check_directory_zip_output(output_dir, zip_output, overwrite=bool(args.overwrite))
            existing_paths = [
                path
                for path in (index_path, html_index_path, readme_path, roster_path, validation_log_path, validation_checklist_path)
                if path.exists()
            ]
            if zip_output is not None and zip_output.exists():
                existing_paths.append(zip_output)
            for _, user_dir in user_dirs:
                existing_paths.extend(path for path in _user_handoff_paths(user_dir).values() if path.exists())
            if existing_paths and not bool(args.overwrite):
                raise FileExistsError(
                    "Refusing to overwrite existing user handoff files: "
                    + ", ".join(str(path) for path in existing_paths)
                )

            output_dir.mkdir(parents=True, exist_ok=True)
            assignment_ownership_integrity = _assignment_ownership_integrity(
                assignments,
                schema_integrity=store.assignment_schema_integrity(),
            )
            manifests = [
                _write_user_handoff_bundle(
                    store=store,
                    store_path=store_path,
                    user=user,
                    output_dir=user_dir,
                    secret=secret,
                    base_url=args.base_url,
                    ttl_seconds=int(args.ttl_seconds),
                    include_completed=bool(args.include_completed),
                    overwrite=bool(args.overwrite),
                )
                for user, user_dir in user_dirs
            ]
            handoff_sendability = _handoff_sendability_summary(manifests)
            index = {
                "ok": all(bool(manifest["ok"]) for manifest in manifests)
                and int(handoff_sendability["not_ready_to_send_count"]) == 0,
                "store_checks_ok": all(bool(manifest["ok"]) for manifest in manifests),
                "store_path": str(store_path),
                "output_dir": str(output_dir),
                "include_completed": bool(args.include_completed),
                "include_inactive": bool(args.include_inactive),
                "base_url": str(args.base_url or "").rstrip("/") or None,
                "labeler_landing_page_path": "/",
                "labeler_landing_url": _labeler_landing_url_for_base(str(args.base_url or "").rstrip("/")),
                "dashboard_path": DASHBOARD_PATH,
                "dashboard_url": _dashboard_url_for_base(str(args.base_url or "").rstrip("/")),
                "dataset_queue_page_path": DATASET_QUEUE_PATH,
                "dataset_queue_url": _dataset_queue_url_for_base(str(args.base_url or "").rstrip("/")),
                "personal_work_page_path": PERSONAL_WORK_PATH,
                "personal_work_url": _personal_work_url_for_base(str(args.base_url or "").rstrip("/")),
                "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
                "personal_dataset_queue_url": _personal_dataset_queue_url_for_base(
                    str(args.base_url or "").rstrip("/")
                ),
                "labeler_safety": _labeler_safety_policy(),
                "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
                "operator_authorization_policy": _operator_authorization_policy(),
                "operator_recovery_policy": _operator_recovery_policy(),
                "operator_validation_visibility_policy": _operator_validation_visibility_policy(),
                "operator_validation_command_templates": _operator_validation_command_templates(),
                "safe_share_gate": _safe_share_gate_policy(),
                **_safe_share_gate_flat_fields(),
                "zarr_backup_policy": _zarr_backup_policy(),
                "mutation_audit_policy": _mutation_audit_policy(),
                "browser_mutation_write_policy": _browser_mutation_write_policy(),
                "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(),
                "dataset_queue_direct_start_policy": _dataset_queue_direct_start_policy(),
                "runtime_operator_validation_gate_cli_policy": _runtime_operator_validation_gate_cli_policy(),
                "browser_response_security_policy": _browser_response_security_policy(),
                "session_guard_policy": _session_guard_policy(),
                "task_state_policy": _browser_task_state_policy(),
                "signed_link_policy": _browser_signed_link_policy(),
                "browser_workflows": _browser_workflow_capabilities(),
                "single_owner_policy": _assignment_ownership_policy(),
                "assignment_snapshot": _assignment_snapshot_from_assignments(assignments),
                "assignment_ownership_integrity": assignment_ownership_integrity,
                "ttl_seconds": _effective_signed_link_ttl_seconds(int(args.ttl_seconds)),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "progress_summary": _sum_handoff_progress_summary(manifests),
                "dataset_queue_summary": _sum_handoff_dataset_queue_summary(manifests),
                "files": {
                    "index": str(index_path),
                    "html_index": str(html_index_path),
                    "readme": str(readme_path),
                    "labeler_roster": str(roster_path),
                    "validation_log": str(validation_log_path),
                    "validation_checklist": str(validation_checklist_path),
                    **({"bundle_zip": str(zip_output)} if zip_output is not None else {}),
                },
                "counts": {
                    "users": len(manifests),
                    "handoffs": len(manifests),
                    "failed_store_checks": sum(1 for manifest in manifests if not bool(manifest["ok"])),
                    "ready_to_send": handoff_sendability["ready_to_send_count"],
                    "not_ready_to_send": handoff_sendability["not_ready_to_send_count"],
                    "sendability_reasons": _count_handoff_sendability_reasons(handoff_sendability["warnings"]),
                    "waiting_datasets": sum(
                        int((manifest.get("counts") or {}).get("waiting_datasets") or 0)
                        for manifest in manifests
                        if isinstance(manifest.get("counts"), Mapping)
                    ),
                    "dataset_open_tasks": sum(
                        int((manifest.get("counts") or {}).get("dataset_open_tasks") or 0)
                        for manifest in manifests
                        if isinstance(manifest.get("counts"), Mapping)
                    ),
                    "dataset_queue_states": _handoff_dataset_queue_state_counts(manifests),
                    "dataset_queue_blocked_start_users": [
                        str(manifest.get("user") or "")
                        for manifest in manifests
                        if _handoff_dataset_queue_blocks_labeler_start(manifest)
                    ],
                    "recordings_without_open_tasks": sum(
                        int((manifest.get("counts") or {}).get("recordings_without_open_tasks") or 0)
                        for manifest in manifests
                        if isinstance(manifest.get("counts"), Mapping)
                    ),
                    "recordings_without_open_tasks_by_reason": _sum_recordings_without_open_tasks_by_reason(manifests),
                    "recordings_without_open_tasks_actions": _recordings_without_open_tasks_actions(
                        _sum_recordings_without_open_tasks_by_reason(manifests)
                    ),
                    "redacted_summary_fields": sum(
                        int((manifest.get("counts") or {}).get("redacted_summary_fields") or 0)
                        for manifest in manifests
                        if isinstance(manifest.get("counts"), Mapping)
                    ),
                },
                "sendability_warnings": handoff_sendability["warnings"],
                "sendability_actions": _handoff_sendability_actions(
                    _count_handoff_sendability_reasons(handoff_sendability["warnings"]).keys()
                ),
                "handoffs": [
                    {
                        "user": manifest["user"],
                        "output_dir": manifest["output_dir"],
                        "index_html": manifest["files"]["html_index"],
                        "message": manifest["files"]["message"],
                        "manifest": manifest["files"]["manifest"],
                        "ready_to_send": _handoff_ready_to_send(manifest),
                        "operator_validation_visibility_policy": manifest.get(
                            "operator_validation_visibility_policy",
                            _operator_validation_visibility_policy(),
                        ),
                        **_operator_validation_public_fields(manifest),
                        **_operator_validation_gate_flat_fields(manifest),
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
                        **_operator_validation_command_template_fields(manifest),
                        "sendability_reasons": _handoff_sendability_reasons(manifest),
                        "sendability_actions": _handoff_sendability_actions(_handoff_sendability_reasons(manifest)),
                        "dashboard_url": str(manifest.get("dashboard_url") or ""),
                        "labeler_landing_url": str(manifest.get("labeler_landing_url") or ""),
                        "expected_user_labeler_landing_url": str(manifest.get("expected_user_labeler_landing_url") or ""),
                        "expected_user_dashboard_url": str(manifest.get("expected_user_dashboard_url") or ""),
                        "expected_user_dataset_queue_url": str(manifest.get("expected_user_dataset_queue_url") or ""),
                        "expected_user_personal_work_url": str(
                            manifest.get("expected_user_personal_work_url") or ""
                        ),
                        "expected_user_personal_dataset_queue_url": str(
                            manifest.get("expected_user_personal_dataset_queue_url") or ""
                        ),
                        "personalized_labeler_entrypoint": str(
                            manifest.get("personalized_labeler_entrypoint") or ""
                        ),
                        "personalized_labeler_entry_url": str(
                            manifest.get("personalized_labeler_entry_url") or ""
                        ),
                        "queue_first_entry_contract": manifest.get(
                            "queue_first_entry_contract", {}
                        ),
                        "personalized_launch_readiness": manifest.get(
                            "personalized_launch_readiness",
                            _personalized_launch_readiness_summary(manifest),
                        ),
                        "expected_user_identity_probe_url": str(manifest.get("expected_user_identity_probe_url") or ""),
                        "reassignment_session_safety": manifest.get("reassignment_session_safety", {}),
                        **_reassignment_session_safety_flat_fields(
                            manifest.get("reassignment_session_safety")
                            if isinstance(manifest.get("reassignment_session_safety"), Mapping)
                            else None
                        ),
                        "labeler_safety": manifest.get("labeler_safety", _labeler_safety_policy()),
                        **_handoff_labeler_safety_fields(manifest),
                        "labeler_route_authorization_policy": manifest.get("labeler_route_authorization_policy", _labeler_route_authorization_policy()),
                        **_handoff_labeler_route_authorization_fields(manifest),
                        "browser_mutation_write_policy": manifest.get(
                            "browser_mutation_write_policy", _browser_mutation_write_policy()
                        ),
                        "browser_mutation_write_checklist": manifest.get(
                            "browser_mutation_write_checklist",
                            _browser_mutation_write_runtime_checklist(
                                manifest.get("browser_mutation_write_policy")
                                if isinstance(manifest.get("browser_mutation_write_policy"), Mapping)
                                else None
                            ),
                        ),
                        "dataset_queue_direct_start_policy": manifest.get(
                            "dataset_queue_direct_start_policy", _dataset_queue_direct_start_policy()
                        ),
                        **_dataset_queue_direct_start_policy_fields(
                            manifest.get("dataset_queue_direct_start_policy")
                            if isinstance(manifest.get("dataset_queue_direct_start_policy"), Mapping)
                            else None
                        ),
                        "runtime_operator_validation_gate_cli_policy": manifest.get(
                            "runtime_operator_validation_gate_cli_policy",
                            _runtime_operator_validation_gate_cli_policy(),
                        ),
                        **_runtime_operator_validation_gate_cli_policy_fields(
                            manifest.get("runtime_operator_validation_gate_cli_policy")
                            if isinstance(
                                manifest.get("runtime_operator_validation_gate_cli_policy"),
                                Mapping,
                            )
                            else None
                        ),
                        "direct_browser_start_contract_summary": manifest.get(
                            "direct_browser_start_contract_summary", {}
                        ),
                        **_direct_browser_start_contract_summary_fields(
                            manifest.get("direct_browser_start_contract_summary")
                            if isinstance(manifest.get("direct_browser_start_contract_summary"), Mapping)
                            else None
                        ),
                        **_handoff_browser_mutation_write_fields(manifest),
                        **_browser_mutation_target_contract_compact_fields(
                            manifest.get("browser_mutation_write_checklist")
                            if isinstance(manifest.get("browser_mutation_write_checklist"), Mapping)
                            else _browser_mutation_write_runtime_checklist(
                                manifest.get("browser_mutation_write_policy")
                                if isinstance(manifest.get("browser_mutation_write_policy"), Mapping)
                                else None
                            ),
                            user=str(manifest.get("user") or ""),
                        ),
                        **_direct_browser_start_contract_compact_fields(
                            manifest.get("dataset_queue_direct_start_policy")
                            if isinstance(manifest.get("dataset_queue_direct_start_policy"), Mapping)
                            else None,
                            user=str(manifest.get("user") or ""),
                        ),
                        "single_owner_policy_contract_met": (
                            bool(manifest.get("single_owner_policy_contract_met"))
                            if "single_owner_policy_contract_met" in manifest
                            else bool(
                                _handoff_assignment_ownership_fields(manifest).get(
                                    "assignment_ownership_contract_ready"
                                )
                            )
                            and int(
                                _handoff_assignment_ownership_fields(manifest).get(
                                    "assignment_duplicate_active_owner_count"
                                )
                                or 0
                            )
                            == 0
                        ),
                        **_handoff_operator_recovery_fields(manifest),
                        "zarr_backup_policy": manifest.get("zarr_backup_policy", _zarr_backup_policy()),
                        **_handoff_zarr_backup_fields(manifest),
                        "mutation_audit_policy": manifest.get("mutation_audit_policy", _mutation_audit_policy()),
                        **_handoff_mutation_audit_fields(manifest),
                        "browser_response_security_policy": manifest.get(
                            "browser_response_security_policy", _browser_response_security_policy()
                        ),
                        **_handoff_browser_response_security_fields(manifest),
                        "session_guard_policy": manifest.get("session_guard_policy", _session_guard_policy()),
                        **_handoff_session_guard_fields(manifest),
                        "task_state_policy": manifest.get("task_state_policy", _browser_task_state_policy()),
                        **_handoff_task_state_policy_fields(manifest),
                        "signed_link_policy": manifest.get("signed_link_policy", _browser_signed_link_policy()),
                        **_handoff_signed_link_policy_fields(manifest),
                        "browser_workflows": manifest.get("browser_workflows", _browser_workflow_capabilities()),
                        "progress_summary": manifest.get("progress_summary", {}),
                        "dataset_queue_summary": manifest.get("dataset_queue_summary", {}),
                        "dataset_queue_state": manifest.get("dataset_queue_state", {}),
                        "dataset_queue_state_code": str(
                            (manifest.get("dataset_queue_state") or {}).get("code") or ""
                        )
                        if isinstance(manifest.get("dataset_queue_state"), Mapping)
                        else "",
                        "dataset_queue_state_title": str(
                            (manifest.get("dataset_queue_state") or {}).get("title") or ""
                        )
                        if isinstance(manifest.get("dataset_queue_state"), Mapping)
                        else "",
                        "labeler_work_completion": manifest.get("labeler_work_completion", {}),
                        **_labeler_work_completion_fields(
                            manifest.get("labeler_work_completion")
                            if isinstance(manifest.get("labeler_work_completion"), Mapping)
                            else None
                        ),
                        "dataset_queue_blocks_labeler_start": _handoff_dataset_queue_blocks_labeler_start(manifest),
                        **_handoff_known_user_status_fields(manifest),
                        **_handoff_assignment_ownership_fields(manifest),
                        **_handoff_entry_artifact_fields(manifest),
                        **_handoff_dataset_queue_start_fields(manifest),
                        "dataset_queue_preview_url": manifest.get("dataset_queue_preview_url", ""),
                        "canonical_dataset_queue_preview_url": manifest.get(
                            "canonical_dataset_queue_preview_url",
                            manifest.get("expected_user_dataset_queue_url", ""),
                        ),
                        "recordings_without_open_tasks_actions": (manifest.get("counts") or {}).get(
                            "recordings_without_open_tasks_actions", []
                        )
                        if isinstance(manifest.get("counts"), Mapping)
                        else [],
                        "links_expire_at_utc": manifest.get("links_expire_at_utc"),
                        "files": manifest["files"],
                        "ok": manifest["ok"],
                        "counts": manifest["counts"],
                    }
                    for manifest in manifests
                ],
            }
            index.update(
                _operator_validation_invitation_fields(
                    _web_labeling_validation_checklist_payload(
                        index,
                        bundle_label="multi-user handoff bundle",
                    )
                )
            )
            index_safe_share_gate = (
                index.get("safe_share_gate")
                if isinstance(index.get("safe_share_gate"), Mapping)
                else _safe_share_gate_policy()
            )
            index["safe_share_gate"] = index_safe_share_gate
            index.update(_safe_share_gate_flat_fields(index_safe_share_gate))
            index.update(
                _safe_share_checklist_gate_status_fields_from_operator_validation(
                    index,
                    safe_share_gate=index_safe_share_gate,
                )
            )
            index_path.write_text(
                json.dumps(index, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_user_handoffs_html_index(index, html_index_path)
            _write_user_handoffs_readme(index, readme_path)
            _write_user_handoffs_roster_csv(index, roster_path)
            _write_user_handoffs_validation_log(index, validation_log_path)
            _write_user_handoffs_validation_checklist(index, validation_checklist_path)
            if zip_output is not None:
                _write_directory_zip(output_dir, zip_output, overwrite=bool(args.overwrite))
            _print_json(index)
            return (
                0
                if bool(index["ok"])
                and int(index["counts"].get("not_ready_to_send") or 0) == 0
                else 2
            )
        if args.command == "generate-keypoint-tasks":
            payload = generate_keypoint_tasks_from_registry(
                store=store,
                registry_path=_registry_path_from_arg(args.registry),
                assignee_user=args.user,
                recording_id=args.recording_id,
                review_filter=str(args.review_filter),
                priority=int(args.priority),
                include_all=bool(args.include_all),
                auto_advance_on_save=not bool(args.no_auto_advance_on_save),
            )
            result = _task_generation_cli_payload(payload, warnings_as_errors=bool(args.warnings_as_errors))
            _write_optional_json_report(result, args.output, overwrite=bool(args.overwrite), description="task-generation report")
            _print_json(result)
            return 2 if bool(result["failed_by_warnings"]) else 0
        if args.command == "generate-detect-training-tasks":
            payload = generate_detect_training_tasks_from_registry(
                store=store,
                registry_path=_registry_path_from_arg(args.registry),
                assignee_user=args.user,
                recording_id=args.recording_id,
                review_filter=str(args.review_filter),
                priority=int(args.priority),
                include_all=bool(args.include_all),
                auto_advance_on_save=not bool(args.no_auto_advance_on_save),
            )
            result = _task_generation_cli_payload(payload, warnings_as_errors=bool(args.warnings_as_errors))
            _write_optional_json_report(result, args.output, overwrite=bool(args.overwrite), description="task-generation report")
            _print_json(result)
            return 2 if bool(result["failed_by_warnings"]) else 0
        if args.command == "generate-detect-analysis-tasks":
            payload = generate_detect_analysis_tasks_from_registry(
                store=store,
                registry_path=_registry_path_from_arg(args.registry),
                assignee_user=args.user,
                recording_id=args.recording_id,
                review_filter=str(args.review_filter),
                priority=int(args.priority),
                editable=bool(args.editable),
                promote_training_zarr=args.promote_training_zarr,
                promote_target_crop_run=args.promote_target_crop_run,
                promote_target_refined_run=args.promote_target_refined_run,
                promote_label_origin=str(args.promote_label_origin),
                promote_include_negative=not bool(args.promote_no_negative),
                promote_allow_unreviewed_negative=bool(args.promote_allow_unreviewed_negative),
                promote_target_size=tuple(args.promote_target_size) if args.promote_target_size else None,
                auto_advance_on_save=not bool(args.no_auto_advance_on_save),
            )
            result = _task_generation_cli_payload(payload, warnings_as_errors=bool(args.warnings_as_errors))
            _write_optional_json_report(result, args.output, overwrite=bool(args.overwrite), description="task-generation report")
            _print_json(result)
            return 2 if bool(result["failed_by_warnings"]) else 0
        if args.command == "generate-subject-mask-tasks":
            payload = generate_subject_mask_component_tasks_from_registry(
                store=store,
                registry_path=_registry_path_from_arg(args.registry),
                assignee_user=args.user,
                recording_id=args.recording_id,
                review_filter=str(args.review_filter),
                priority=int(args.priority),
                component_names=args.components,
                auto_advance_on_save=not bool(args.no_auto_advance_on_save),
            )
            result = _task_generation_cli_payload(payload, warnings_as_errors=bool(args.warnings_as_errors))
            _write_optional_json_report(result, args.output, overwrite=bool(args.overwrite), description="task-generation report")
            _print_json(result)
            return 2 if bool(result["failed_by_warnings"]) else 0
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
