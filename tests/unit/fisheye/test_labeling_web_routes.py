from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from types import ModuleType, SimpleNamespace

import numpy as np

from fisheye.labeling import web as labeling_web
from fisheye.labeling.assignment_store import LabelingStore


@contextmanager
def _running_server(
    store: LabelingStore,
    *,
    user: str | None,
    admin_users: tuple[str, ...] = (),
    link_secret: str | None = None,
    link_not_before_utc: str | None = None,
    validation_checklist_path=None,
    require_operator_validation_for_start: bool = False,
    configure_state=None,
):
    config = labeling_web.ServerConfig(
        store_path=store.path,
        host="127.0.0.1",
        port=0,
        fixed_user=user,
        auth_header="X-Forwarded-User",
        session_ttl_seconds=600,
        admin_users=admin_users,
        link_secret=link_secret,
        link_not_before_utc=link_not_before_utc,
        validation_checklist_path=validation_checklist_path,
        require_operator_validation_for_start=require_operator_validation_for_start,
    )
    state = labeling_web.ServerState(store=store, config=config)
    if configure_state is not None:
        configure_state(state)
    server = ThreadingHTTPServer(("127.0.0.1", 0), labeling_web._make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _json_request(base_url: str, path: str, *, method: str = "GET", payload: dict[str, object] | None = None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(f"{base_url}{path}", data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _text_request(base_url: str, path: str):
    request = urllib.request.Request(f"{base_url}{path}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _headers_request(base_url: str, path: str):
    request = urllib.request.Request(f"{base_url}{path}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
            return response.status, dict(response.headers)
    except urllib.error.HTTPError as exc:
        exc.read()
        return exc.code, dict(exc.headers)


def _assert_browser_response_security_headers(headers: dict[str, str]) -> None:
    assert headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert headers["Pragma"] == "no-cache"
    assert headers["Expires"] == "0"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["Content-Security-Policy"] == (
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'"
    )
    assert headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"


def _fake_module(monkeypatch, name: str, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent_name, _, child_name = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None and child_name:
        monkeypatch.setattr(parent, child_name, module, raising=False)
    return module


def _assert_mutation_event_provenance(event, *, task_id: str, recording_id: str, user: str, event_type: str):
    assert event["event_id"]
    assert event["task_id"] == task_id
    assert event["recording_id"] == recording_id
    assert event["user"] == user
    assert event["event_type"] == event_type
    assert event["created_at_utc"]
    assert isinstance(event["target"], dict)
    assert isinstance(event["before"], dict)
    assert isinstance(event["after"], dict)


def _assert_save_response_mutation_contract(
    payload,
    event,
    *,
    workflow_kind: str,
    task_id: str,
    recording_id: str,
    event_type: str,
) -> None:
    mutation = payload["mutation"]
    assert mutation["schema"] == "palette.web_labeling_browser_mutation_response.v1"
    assert mutation["workflow_kind"] == workflow_kind
    assert mutation["task_id"] == task_id
    assert mutation["recording_id"] == recording_id
    assert mutation["audit_event_store"] == "labeling_task_events"
    assert mutation["audit_event_id"] == event["event_id"]
    assert mutation["audit_event_type"] == event_type
    assert len(mutation["audit_events"]) == 1
    assert mutation["audit_events"][0]["event_id"] == event["event_id"]
    assert mutation["audit_events"][0]["event_type"] == event_type
    assert mutation["audit_events"][0]["task_id"] == task_id
    assert mutation["audit_events"][0]["recording_id"] == recording_id
    assert mutation["audit_events"][0]["created_at_utc"] == event["created_at_utc"]
    assert mutation["assignment_authorization_checked_server_side"] is True
    assert mutation["assignment_authorization_result"] == "passed"
    assert mutation["active_assignment_checked_server_side"] is True
    assert mutation["active_assignment_present"] is True
    assert mutation["task_assigned_to_resolved_user_checked_server_side"] is True
    assert mutation["task_assigned_to_resolved_user"] is True
    assert mutation["session_checked_server_side"] is True
    assert mutation["session_ownership_checked_server_side"] is True
    assert mutation["session_user_matches_resolved_user"] is True
    assert mutation["current_session_checked_server_side"] is True
    assert mutation["current_session_required"] is True
    assert mutation["reassignment_session_safety_checked_server_side"] is True
    assert mutation["reassignment_session_safety_passed"] is True
    assert mutation["current_target_token_checked_server_side"] is True
    assert mutation["target_token_required_for_mutation"] is True
    assert mutation["server_authorizes_mutation"] is True
    assert mutation["data_plane_write_target"] == "server_owned_assigned_task_zarr_scope"
    assert mutation["mutable_label_data_plane"] == "task_scoped_training_zarr"
    assert mutation["label_mutation_target_kind"] == "task_scoped_training_zarr"
    assert mutation["browser_label_write_target"] == "training_zarr"
    assert mutation["server_mutates_task_scoped_zarr_targets"] is True
    assert mutation["training_zarr_mutations_are_server_owned"] is True
    assert mutation["handoff_artifacts_are_metadata_only"] is True
    assert mutation["csv_handoff_artifact_role"] == "metadata_only_control_plane"
    assert mutation["csv_handoff_artifacts_are_label_write_targets"] is False
    assert mutation["handoff_csv_artifacts_are_label_write_targets"] is False
    assert mutation["intermediate_csv_artifacts_are_label_write_targets"] is False
    assert mutation["browser_writes_csv_or_handoff_files"] is False
    assert mutation["browser_writes_handoff_csv"] is False
    assert mutation["browser_writes_intermediate_csv"] is False
    assert mutation["browser_receives_zarr_write_authority"] is False
    assert mutation["browser_has_direct_zarr_write_authority"] is False
    contract = mutation["mutation_authorization_contract"]
    assert contract["schema"] == "palette.web_labeling_mutation_authorization_contract.v1"
    assert contract["ready"] is True
    assert contract["session_lookup_result"] == "passed"
    assert contract["session_owned_by_resolved_user"] is True
    assert contract["task_reloaded_server_side"] is True
    assert contract["task_assigned_to_resolved_user"] is True
    assert contract["assignment_status_active"] is True
    assert contract["task_open_for_mutation"] is True
    assert contract["current_session_result"] == "passed"
    assert contract["reassignment_session_safety_result"] == "passed"
    assert contract["current_target_token_result"] == "passed"
    assert contract["browser_supplied_zarr_or_csv_target_allowed"] is False
    assert contract["browser_supplied_target_selectors_allowed"] is False
    assert contract["client_authorizes_mutation"] is False
    assert contract["server_authorizes_mutation"] is True


def test_dashboard_page_describes_browser_only_assigned_work(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-paused", assignee_user="alice", status="paused")
        with _running_server(store, user="alice") as base_url:
            status, html = _text_request(base_url, "/")
            labeling_status, labeling_html = _text_request(base_url, "/labeling?expected_user=alice")
            work_status, work_html = _text_request(base_url, "/work")
            expected_status, expected_html = _text_request(base_url, "/work?expected_user=alice")
            datasets_page_status, datasets_page_html = _text_request(base_url, "/datasets")
            my_work_status, my_work_html = _text_request(base_url, "/my-work?expected_user=alice")
            my_datasets_status, my_datasets_html = _text_request(
                base_url,
                "/my-datasets?expected_user=alice",
            )
            header_status, headers = _headers_request(base_url, "/datasets?expected_user=alice")
            my_datasets_header_status, my_datasets_headers = _headers_request(
                base_url,
                "/my-datasets?expected_user=alice",
            )
            my_work_header_status, my_work_headers = _headers_request(
                base_url,
                "/my-work?expected_user=alice",
            )
            labeling_header_status, labeling_headers = _headers_request(
                base_url,
                "/labeling?expected_user=alice",
            )
            me_header_status, me_headers = _headers_request(base_url, "/me")
            identity_header_status, identity_headers = _headers_request(
                base_url,
                "/identity?expected_user=alice",
            )
            identity_api_header_status, identity_api_headers = _headers_request(
                base_url,
                "/api/me/identity?expected_user=alice",
            )
            expected_datasets_page_status, expected_datasets_page_html = _text_request(
                base_url,
                "/datasets?expected_user=alice",
            )
            me_status, me_html = _text_request(base_url, "/me")
            identity_status, identity_html = _text_request(base_url, "/identity?expected_user=alice")
            identity_api_status, identity_api_payload = _json_request(base_url, "/api/me/identity?expected_user=alice")

        assert status == 200
        assert labeling_status == 200
        assert work_status == 200
        assert expected_status == 200
        assert datasets_page_status == 200
        assert my_work_status == 200
        assert my_datasets_status == 200
        assert header_status == 200
        assert my_datasets_header_status == 200
        assert my_work_header_status == 200
        assert labeling_header_status == 200
        assert me_header_status == 200
        assert identity_header_status == 200
        assert identity_api_header_status == 200
        _assert_browser_response_security_headers(headers)
        _assert_browser_response_security_headers(my_datasets_headers)
        _assert_browser_response_security_headers(my_work_headers)
        _assert_browser_response_security_headers(labeling_headers)
        _assert_browser_response_security_headers(me_headers)
        _assert_browser_response_security_headers(identity_headers)
        _assert_browser_response_security_headers(identity_api_headers)
        assert expected_datasets_page_status == 200
        assert me_status == 200
        assert identity_status == 200
        assert identity_api_status == 200
        assert identity_api_payload["ok"] is True
        assert identity_api_payload["identity"]["resolved_user"] == "alice"
        assert identity_api_payload["identity"]["expected_user"] == "alice"
        assert identity_api_payload["identity"]["matches_expected_user"] is True
        assert identity_api_payload["identity"]["expected_user_labeler_landing_url"] == "/?expected_user=alice"
        assert identity_api_payload["identity"]["expected_user_labeling_home_url"] == (
            "/labeling?expected_user=alice"
        )
        assert identity_api_payload["identity"]["expected_user_dashboard_url"] == "/work?expected_user=alice"
        assert identity_api_payload["identity"]["expected_user_dataset_queue_url"] == "/datasets?expected_user=alice"
        assert identity_api_payload["identity"]["queue_first_entry_contract"]["ready"] is True
        assert identity_api_payload["identity"]["queue_first_entry_contract"]["labeling_home_ready"] is True
        assert identity_api_payload["identity"]["queue_first_entry_contract"]["expected_user_labeling_home_url"] == (
            "/labeling?expected_user=alice"
        )
        assert identity_api_payload["identity"]["queue_first_entry_contract"][
            "personalized_labeler_entry_url"
        ] == "/my-datasets?expected_user=alice"
        assert identity_api_payload["identity"]["queue_first_entry_contract"][
            "personalized_labeler_entry_url_matches_personal_dataset_queue"
        ] is True
        assert identity_api_payload["personalized_launch_readiness"] == (
            identity_api_payload["identity"]["personalized_launch_readiness"]
        )
        assert identity_api_payload["identity"]["personalized_launch_readiness"][
            "schema"
        ] == "palette.web_labeling_personalized_launch_readiness.v1"
        assert identity_api_payload["identity"]["personalized_launch_readiness"][
            "personalized_labeler_entry_url"
        ] == "/my-datasets?expected_user=alice"
        assert identity_api_payload["identity"]["personalized_launch_readiness"][
            "browser_label_write_target"
        ] == "training_zarr"
        assert identity_api_payload["identity"]["personalized_launch_readiness"][
            "browser_writes_csv_or_handoff_files"
        ] is False
        assert identity_api_payload["identity"]["personalized_launch_readiness"][
            "browser_has_direct_zarr_write_authority"
        ] is False
        assert identity_api_payload["identity"]["single_owner_policy"]["assignment_scope"] == "recording"
        assert identity_api_payload["identity"]["single_owner_policy_one_active_owner"] is True
        assert identity_api_payload["identity"][
            "single_owner_policy_browser_mutation_requires_current_assignment_owner"
        ] is True
        assert identity_api_payload["identity"]["single_owner_assignment_contract"]["schema"] == (
            "palette.web_labeling_assignment_single_owner_contract.v1"
        )
        assert identity_api_payload["identity"]["single_owner_assignment_contract"][
            "browser_mutation_target_resolved_server_side"
        ] is True
        assert identity_api_payload["identity"]["single_owner_assignment_contract"][
            "browser_mutation_target_source"
        ] == "recording_assignments.active_assignment"
        assert identity_api_payload["identity"]["single_owner_assignment_contract"][
            "labelers_mutate_assigned_training_zarrs"
        ] is True
        assert identity_api_payload["identity"]["single_owner_assignment_contract"][
            "labelers_mutate_intermediate_csvs"
        ] is False
        assert identity_api_payload["identity"][
            "assignment_ownership_contract_store_single_owner_assignment_contract_present"
        ] is True
        assert identity_api_payload["identity"][
            "assignment_ownership_contract_store_single_owner_assignment_contract_ready"
        ] is True
        assert identity_api_payload["identity"][
            "assignment_ownership_contract_store_single_owner_assignment_contract_met"
        ] is True
        assert identity_api_payload["identity"]["assignment_ownership_integrity"]["ok"] is True
        assert identity_api_payload["identity"][
            "assignment_ownership_contract_assignment_ownership_integrity_ok"
        ] is True
        assert identity_api_payload["identity"][
            "assignment_ownership_contract_duplicate_active_owner_count"
        ] == 0
        assert identity_api_payload["identity"][
            "assignment_ownership_contract_browser_mutation_target_resolved_server_side"
        ] is True
        assert identity_api_payload["identity"]["browser_label_write_target"] == "training_zarr"
        assert identity_api_payload["identity"]["browser_mutation_target_contract_met"] is True
        assert identity_api_payload["identity"]["browser_mutation_target_mismatch_count"] == 0
        assert identity_api_payload["identity"]["label_mutation_target_kind"] == "task_scoped_training_zarr"
        assert identity_api_payload["identity"]["csv_handoff_artifact_role"] == "metadata_only_control_plane"
        assert identity_api_payload["identity"]["csv_handoff_artifacts_are_label_write_targets"] is False
        assert identity_api_payload["identity"]["handoff_csv_artifacts_are_label_write_targets"] is False
        assert identity_api_payload["identity"][
            "intermediate_csv_artifacts_are_label_write_targets"
        ] is False
        assert identity_api_payload["identity"]["browser_writes_csv_or_handoff_files"] is False
        assert identity_api_payload["identity"]["browser_has_direct_zarr_write_authority"] is False
        assert identity_api_payload["identity"]["direct_browser_start_contract_met"] is True
        assert identity_api_payload["identity"]["direct_browser_start_mismatch_count"] == 0
        assert identity_api_payload["identity"][
            "dataset_queue_direct_start_post_body_expected_user_required"
        ] is True
        assert identity_api_payload["identity"][
            "dataset_queue_direct_start_browser_label_write_target"
        ] == "training_zarr"
        assert identity_api_payload["identity"]["handoff_artifacts_are_metadata_only"] is True
        assert "Palette labeling identity check" in identity_html
        assert "Handoff CSV label target" in identity_html
        assert "Intermediate CSV label target" in identity_html
        assert "Launch readiness schema" in identity_html
        assert "palette.web_labeling_personalized_launch_readiness.v1" in identity_html
        assert "Launch personal queue URL" in identity_html
        assert "/my-datasets?expected_user=alice" in identity_html
        assert "Launch browser target" in identity_html
        assert "training_zarr" in identity_html
        assert "Launch writes CSV/handoff" in identity_html
        assert "Launch direct Zarr authority" in identity_html
        assert "Identity matches expected user" in identity_html
        assert "Open your datasets-waiting landing page" in identity_html
        assert "/?expected_user=alice" in identity_html
        assert "queue_first_entry_contract" in identity_html
        assert "/my-datasets?expected_user=alice" in identity_html
        assert "Open your personalized dataset queue" in identity_html
        assert "/datasets?expected_user=alice" in identity_html
        assert "Browser label target" in identity_html
        assert "training_zarr" in identity_html
        assert "CSV, HTML, JSON, and handoff files are metadata only" in identity_html
        assert "Each recording has one active assigned owner" in identity_html
        assert 'id="identity-support"' in identity_html
        assert "Copy identity details" in identity_html
        assert "copyIdentitySupport" in identity_html
        assert "resolved_user" in identity_html
        assert "Datasets waiting for completion" in html
        assert "Datasets waiting for completion" in labeling_html
        assert "Datasets waiting for completion" in me_html
        assert "Open full work dashboard" in html
        assert "/api/me/datasets" in html
        assert "Work waiting for completion" in work_html
        assert "Work waiting for completion" in expected_html
        assert "Work waiting for completion" in my_work_html
        assert "Datasets waiting for completion" in datasets_page_html
        assert "Datasets waiting for completion" in my_datasets_html
        assert "operatorValidationGateSupportLines" in work_html
        assert "OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES" in work_html
        assert "operator_validation_gate_status_values" in work_html
        assert "operator_validation_gate_ids" in work_html
        assert "operator_validation_gate_flat_field_suffixes" in work_html
        assert "operator_validation_external_evidence_required" in work_html
        assert "operator_validation_external_evidence_required_gate_ids" in work_html
        assert "operator_validation_external_evidence_template_paths_by_gate_id" in work_html
        assert "operator_validation_checklist_only_required_gate_ids" in work_html
        assert "operator_validation_command_template_template_backed_gate_ids" in work_html
        assert "operator_validation_command_template_apply_required_gate_ids" in work_html
        assert "operator_validation_command_template_evidence_template_fields_by_gate_id" in work_html
        assert "operator_validation_command_template_evidence_template_paths_by_gate_id" in work_html
        assert "safe_share_external_launch_evidence_gap_gate_ids" in work_html
        assert "safe_share_external_launch_evidence_gap_summary" in work_html
        assert "safe_share_external_launch_evidence_gap_todos" in work_html
        assert "safe_share_external_launch_evidence_gap_todo_fields" in work_html
        assert "safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id" in work_html
        assert "personalizedLaunchReadinessSupportLines" in work_html
        assert "personalized_launch_readiness_schema" in work_html
        assert "personalized_launch_readiness_personalized_labeler_entry_url" in work_html
        assert "personalized_launch_readiness_external_launch_evidence_gap_todos" in work_html
        assert "personalized_launch_readiness_browser_label_write_target" in work_html
        assert "personalized_launch_readiness_browser_writes_csv_or_handoff_files" in work_html
        assert "personalized_launch_readiness_browser_has_direct_zarr_write_authority" in work_html
        assert "operator_support_expected_user_personal_dataset_queue_url" in work_html
        assert "operator_support_personalized_labeler_entry_url" in work_html
        assert "operator_support_browser_label_write_target" in work_html
        assert "operator_support_browser_writes_csv_or_handoff_files" in work_html
        assert "operator_support_browser_has_direct_zarr_write_authority" in work_html
        assert "operator_support_csv_handoff_artifact_role" in work_html
        assert "browser_smoke" in work_html
        assert "missing_evidence" in work_html
        assert "${prefix}_${suffix}" in work_html
        assert "Datasets waiting for completion" in expected_datasets_page_html
        assert "Open landing page" in datasets_page_html
        assert 'id="landing-link"' in datasets_page_html
        assert "Copy start link" in datasets_page_html
        assert "copyLandingLink" in datasets_page_html
        assert "expected_user_labeler_landing_url" in datasets_page_html
        assert "Open full work dashboard" in datasets_page_html
        assert "Open identity check" in datasets_page_html
        assert 'id="identity-link"' in datasets_page_html
        assert "expected_user_identity_probe_url" in datasets_page_html
        assert "setGuardedEntryLinks" in datasets_page_html
        assert "expectedUserText" in datasets_page_html
        assert "task-row" in datasets_page_html
        assert "task.expected_user_work_url" in datasets_page_html
        assert "task.notes" in datasets_page_html
        assert "taskIdText" in datasets_page_html
        assert "supportDetailsText" in datasets_page_html
        assert "task.operator_support" in datasets_page_html
        assert "dataset.operator_support" in datasets_page_html
        assert "recording.operator_support" in datasets_page_html
        assert "Copy dataset support" in datasets_page_html
        assert "Copy recording support" in datasets_page_html
        assert "Copy task support details" in datasets_page_html
        assert "data-support-details" in datasets_page_html
        assert "progress.complete_task_count" in datasets_page_html
        assert "blocked/no-open recordings" in datasets_page_html
        assert 'id="queue-state"' in datasets_page_html
        assert "Queue state:" in datasets_page_html
        assert "Labeler start" in datasets_page_html
        assert "Assigned datasets waiting for browser labeling" in datasets_page_html
        assert "All assigned work complete" in datasets_page_html
        assert "Assigned work is blocked" in datasets_page_html
        assert "server-side assigned task/training-Zarr writers" in datasets_page_html
        assert "CSV, HTML, JSON, handoff, and roster files are metadata only" in datasets_page_html
        assert "queue_state_blocks_labeler_start" in datasets_page_html
        assert "Do not start new labeling from this queue until the operator resolves this state" in datasets_page_html
        assert "blocked-recordings" in datasets_page_html
        assert "blocked_recordings_by_reason" in datasets_page_html
        assert "blockedSupport" in datasets_page_html
        assert "Copy blocked details" in datasets_page_html
        assert "backup-policy" in datasets_page_html
        assert "zarr_backup_policy" in datasets_page_html
        assert "Copy backup policy" in datasets_page_html
        assert "labelers_do_not_receive_backup_paths" in datasets_page_html
        assert "audit-policy" in datasets_page_html
        assert "mutation_audit_policy" in datasets_page_html
        assert "Copy audit policy" in datasets_page_html
        assert "labeling_task_events" in datasets_page_html
        assert "session-guard-policy" in datasets_page_html
        assert "session_guard_policy" in datasets_page_html
        assert "Copy session guard policy" in datasets_page_html
        assert "stale_tab_save_rejected" in datasets_page_html
        assert "/api/me/datasets" in datasets_page_html
        assert "expectedUserGuardParam" in datasets_page_html
        assert "expected_user: expectedUserGuardParam || \"\"" in datasets_page_html
        assert "dataset_queue_load_failed" in datasets_page_html
        assert "Check your identity before opening work" in datasets_page_html
        assert "No local Palette or Crimson installation is needed" in datasets_page_html
        assert "assigned task/training Zarr writers" in datasets_page_html
        assert "CSV, HTML, JSON, and handoff files are metadata only" in datasets_page_html
        assert "Each recording has one active assigned owner" in datasets_page_html
        assert "single_owner_policy_browser_mutation_target_resolved_server_side" in datasets_page_html
        assert "single_owner_policy_labelers_mutate_assigned_training_zarrs" in datasets_page_html
        assert "single_owner_policy_labelers_mutate_intermediate_csvs" in datasets_page_html
        assert (
            "assignment_ownership_contract_store_single_owner_assignment_contract_present"
            in datasets_page_html
        )
        assert (
            "assignment_ownership_contract_store_single_owner_assignment_contract_met"
            in datasets_page_html
        )
        assert "labeler_route_authorization_single_owner_store_contract_required" in datasets_page_html
        assert "labeler_route_authorization_single_owner_store_proof_ready" in datasets_page_html
        assert "labeler_route_authorization_assignment_ownership_integrity_ok" in datasets_page_html
        assert "labeler_route_authorization_duplicate_active_owner_count" in datasets_page_html
        assert "labeler_route_authorization_browser_mutation_target_resolved_server_side" in datasets_page_html
        assert "labeler_route_authorization_labelers_mutate_assigned_training_zarrs" in datasets_page_html
        assert "labeler_route_authorization_labelers_mutate_intermediate_csvs" in datasets_page_html
        assert "What to send the operator" in datasets_page_html
        assert "Copy support details" in datasets_page_html
        assert "task_open_authorization_contract_schema" in datasets_page_html
        assert "task_open_authorization_contract_ready" in datasets_page_html
        assert "task_open_authorization_contract_not_ready_reason" in datasets_page_html
        assert "authorization_return_personal_dataset_queue_url" in datasets_page_html
        assert "authorization_return_personal_dataset_queue_expected_user_guarded" in datasets_page_html
        assert "authorization_return_personal_work_url" in datasets_page_html
        assert "authorization_return_personal_work_expected_user_guarded" in datasets_page_html
        assert "task_open_expected_user_guard_checked_server_side" in datasets_page_html
        assert "task_open_session_created_server_side" in datasets_page_html
        assert "task_open_server_authorizes_open" in datasets_page_html
        assert "task_open_operator_validation_start_gate_required" in datasets_page_html
        assert "task_open_operator_validation_start_gate_ready" in datasets_page_html
        assert "task_open_operator_validation_start_gate_blocks_task_open" in datasets_page_html
        assert "task_open_operator_validation_start_gate_not_ready_reason" in datasets_page_html
        assert "task_open_operator_validation_pending_gate_ids" in datasets_page_html
        assert "operator_validation_mutation_gate_required" in datasets_page_html
        assert "operator_validation_mutation_gate_ready" in datasets_page_html
        assert "operator_validation_mutation_gate_blocks_browser_mutation" in datasets_page_html
        assert "operator_validation_mutation_gate_not_ready_reason" in datasets_page_html
        assert "operator_validation_mutation_gate_pending_gate_ids" in datasets_page_html
        assert "operator_validation_mutation_gate_required_missing_evidence_gate_ids" in datasets_page_html
        assert "runtime_operator_validation_gate_cli_policy_preferred_require_flag" in datasets_page_html
        assert "runtime_operator_validation_gate_cli_policy_protects_browser_mutations" in datasets_page_html
        assert "runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write" in datasets_page_html
        assert "task_open_browser_label_write_target" in datasets_page_html
        assert "task_open_browser_writes_csv_or_handoff_files" in datasets_page_html
        assert "task_open_browser_has_direct_zarr_write_authority" in datasets_page_html
        assert "single_owner_policy_assignment_scope" in datasets_page_html
        assert "single_owner_policy_browser_mutation_requires_current_assignment_owner" in datasets_page_html
        assert "labeler_work_completion_status" in datasets_page_html
        assert "labeler_work_completion_completed" in datasets_page_html
        assert "labeler_work_completion_has_waiting_work" in datasets_page_html
        assert "labeler_work_completion_ready_for_more_labeling" in datasets_page_html
        assert "labeler_work_completion_operator_action_required" in datasets_page_html
        assert "direct_start_policy_post_body_expected_user_required" in datasets_page_html
        assert "direct_start_policy_post_body_expected_user_field" in datasets_page_html
        assert "direct_start_policy_denied_start_returns_task_open_authorization_contract" in datasets_page_html
        assert "direct_start_policy_denied_start_support_preserves_task_open_authorization_contract" in datasets_page_html
        assert "direct_start_policy_denied_start_support_includes_authorization_context" in datasets_page_html
        assert "direct_start_policy_denied_start_contract_reports_no_session_created" in datasets_page_html
        assert "direct_start_policy_denied_start_contract_reports_server_authorizes_open_false" in datasets_page_html
        assert "identity_personal_queue_evidence_status" in datasets_page_html
        assert "operator_validation_identity_personal_queue_evidence_status_values" in datasets_page_html
        assert "operatorValidationGateSupportLines" in datasets_page_html
        assert "OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES" in datasets_page_html
        assert "operator_validation_gate_status_values" in datasets_page_html
        assert "operator_validation_gate_ids" in datasets_page_html
        assert "operator_validation_gate_flat_field_suffixes" in datasets_page_html
        assert "operator_validation_external_evidence_required" in datasets_page_html
        assert "operator_validation_external_evidence_required_gate_ids" in datasets_page_html
        assert "operator_validation_external_evidence_template_paths_by_gate_id" in datasets_page_html
        assert "operator_validation_checklist_only_required_gate_ids" in datasets_page_html
        assert "operator_validation_command_template_template_backed_gate_ids" in datasets_page_html
        assert "operator_validation_command_template_apply_required_gate_ids" in datasets_page_html
        assert "operator_validation_command_template_evidence_template_fields_by_gate_id" in datasets_page_html
        assert "operator_validation_command_template_evidence_template_paths_by_gate_id" in datasets_page_html
        assert "safe_share_external_launch_evidence_gap_gate_ids" in datasets_page_html
        assert "safe_share_external_launch_evidence_gap_summary" in datasets_page_html
        assert "safe_share_external_launch_evidence_gap_todos" in datasets_page_html
        assert "safe_share_external_launch_evidence_gap_todo_fields" in datasets_page_html
        assert "safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id" in datasets_page_html
        assert "personalizedLaunchReadinessSupportLines" in datasets_page_html
        assert "personalized_launch_readiness_schema" in datasets_page_html
        assert "personalized_launch_readiness_personalized_labeler_entry_url" in datasets_page_html
        assert "personalized_launch_readiness_external_launch_evidence_gap_todos" in datasets_page_html
        assert "personalized_launch_readiness_browser_label_write_target" in datasets_page_html
        assert "personalized_launch_readiness_browser_writes_csv_or_handoff_files" in datasets_page_html
        assert "personalized_launch_readiness_browser_has_direct_zarr_write_authority" in datasets_page_html
        assert "operator_support_expected_user_personal_dataset_queue_url" in datasets_page_html
        assert "operator_support_personalized_labeler_entry_url" in datasets_page_html
        assert "operator_support_browser_label_write_target" in datasets_page_html
        assert "operator_support_browser_writes_csv_or_handoff_files" in datasets_page_html
        assert "operator_support_browser_has_direct_zarr_write_authority" in datasets_page_html
        assert "operator_support_csv_handoff_artifact_role" in datasets_page_html
        assert "mutable_zarr_backup_confirmation" in datasets_page_html
        assert "browser_response_security_headers" in datasets_page_html
        assert "identity_probe_verification" in datasets_page_html
        assert "browser_smoke" in datasets_page_html
        assert "disposable_zarr_mutation_smoke" in datasets_page_html
        assert "operator_recovery_contract" in datasets_page_html
        assert "${prefix}_${suffix}" in datasets_page_html
        assert "status" in datasets_page_html
        assert "pending" in datasets_page_html
        assert "missing_evidence" in datasets_page_html
        assert "needs_review" in datasets_page_html
        assert "passed" in datasets_page_html
        assert "All assigned dataset work is complete" in datasets_page_html
        assert "Assigned recordings need operator action before more labeling" in datasets_page_html
        assert "Copy queue state" in datasets_page_html
        assert "queue_state_code" in datasets_page_html
        assert "empty_state_code" in datasets_page_html
        assert "blocked_recordings_by_reason" in datasets_page_html
        assert "copyDatasetSupport" in datasets_page_html
        assert "Work waiting for completion" not in datasets_page_html
        assert "Datasets waiting for completion" in work_html
        assert "datasets waiting" in work_html
        assert "Loading personalized dataset queue" in work_html
        assert "dataset_id" in work_html
        assert "recording_id" in work_html
        assert "expectedUserGuardParam" in work_html
        assert "new URLSearchParams(window.location.search)" in work_html
        assert 'params.set("expected_user", expectedUserGuardParam)' in work_html
        assert 'task_id: initialWorkQuery.get("task_id")' in work_html
        assert "activeLinkFilters.task_id" in work_html
        assert "waiting recordings" in work_html
        assert "blocked / no-open recordings" in work_html
        assert "Show completed tasks" in work_html
        assert "include_completed=1" in work_html
        assert "Complete; ask the operator to reopen this task" in work_html
        assert "Higher-priority tasks are shown first" in work_html
        assert "priority=" in work_html
        assert "task-specific notes appear under the relevant task" in work_html
        assert "personalized to the user shown at the top" in work_html
        assert "Supported browser workflows" in work_html
        assert "keypoints" in work_html
        assert "detect_training" in work_html
        assert "detect_analysis" in work_html
        assert "subject_mask_component" in work_html
        assert "task-notes" in work_html
        assert "Refresh work" in work_html
        assert "Refreshing assigned work" in work_html
        assert "Open landing page" in work_html
        assert 'id="landing-link"' in work_html
        assert "Copy start link" in work_html
        assert "copyDashboardLandingLink" in work_html
        assert "expected_user_labeler_landing_url" in work_html
        assert "guardedWorkPath" in work_html
        assert "do not need a local Palette or Crimson installation" in work_html
        assert "Do not edit zarr files directly" in work_html
        assert "Do not forward links or handoff files" in work_html
        assert "no browser-labeling tasks have been generated yet" in work_html
        assert "All tasks for this recording are complete" in work_html
        assert "admin recovery view after repair" in work_html
        assert "retryPromotion(" not in work_html
        assert "What to send the operator" in work_html
        assert "dashboard_load_failed" in work_html
        assert "task_open_failed" in work_html
        assert "Copy support details" in work_html
        assert "single_owner_policy_browser_mutation_target_resolved_server_side" in work_html
        assert "single_owner_policy_labelers_mutate_assigned_training_zarrs" in work_html
        assert "single_owner_policy_labelers_mutate_intermediate_csvs" in work_html
        assert (
            "assignment_ownership_contract_store_single_owner_assignment_contract_present"
            in work_html
        )
        assert (
            "assignment_ownership_contract_store_single_owner_assignment_contract_met"
            in work_html
        )
        assert "labeler_route_authorization_single_owner_store_contract_required" in work_html
        assert "labeler_route_authorization_single_owner_store_proof_ready" in work_html
        assert "labeler_route_authorization_assignment_ownership_integrity_ok" in work_html
        assert "labeler_route_authorization_duplicate_active_owner_count" in work_html
        assert "labeler_route_authorization_browser_mutation_target_resolved_server_side" in work_html
        assert "labeler_route_authorization_labelers_mutate_assigned_training_zarrs" in work_html
        assert "labeler_route_authorization_labelers_mutate_intermediate_csvs" in work_html
        assert "task_open_authorization_contract_schema" in work_html
        assert "task_open_authorization_contract_ready" in work_html
        assert "task_open_authorization_contract_not_ready_reason" in work_html
        assert "authorization_return_personal_dataset_queue_url" in work_html
        assert "authorization_return_personal_dataset_queue_expected_user_guarded" in work_html
        assert "authorization_return_personal_work_url" in work_html
        assert "authorization_return_personal_work_expected_user_guarded" in work_html
        assert "task_open_expected_user_guard_checked_server_side" in work_html
        assert "task_open_session_created_server_side" in work_html
        assert "task_open_server_authorizes_open" in work_html
        assert "task_open_operator_validation_start_gate_required" in work_html
        assert "task_open_operator_validation_start_gate_ready" in work_html
        assert "task_open_operator_validation_start_gate_blocks_task_open" in work_html
        assert "task_open_operator_validation_start_gate_not_ready_reason" in work_html
        assert "task_open_operator_validation_pending_gate_ids" in work_html
        assert "operator_validation_start_gate_required" in work_html
        assert "operator_validation_start_gate_blocks_task_open" in work_html
        assert "operator_validation_mutation_gate_required" in work_html
        assert "operator_validation_mutation_gate_ready" in work_html
        assert "operator_validation_mutation_gate_blocks_browser_mutation" in work_html
        assert "operator_validation_mutation_gate_not_ready_reason" in work_html
        assert "operator_validation_mutation_gate_pending_gate_ids" in work_html
        assert "operator_validation_mutation_gate_required_missing_evidence_gate_ids" in work_html
        assert "runtime_operator_validation_gate_cli_policy_preferred_require_flag" in work_html
        assert "runtime_operator_validation_gate_cli_policy_protects_browser_mutations" in work_html
        assert "runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write" in work_html
        assert "Start is waiting for operator validation" in work_html
        assert "task_open_browser_label_write_target" in work_html
        assert "task_open_browser_writes_csv_or_handoff_files" in work_html
        assert "task_open_browser_has_direct_zarr_write_authority" in work_html
        assert "single_owner_policy_assignment_scope" in work_html
        assert "single_owner_policy_browser_mutation_requires_current_assignment_owner" in work_html
        assert "labeler_work_completion_status" in work_html
        assert "labeler_work_completion_completed" in work_html
        assert "labeler_work_completion_has_waiting_work" in work_html
        assert "labeler_work_completion_ready_for_more_labeling" in work_html
        assert "labeler_work_completion_operator_action_required" in work_html
        assert "direct_start_policy_post_body_expected_user_required" in work_html
        assert "direct_start_policy_post_body_expected_user_field" in work_html
        assert "direct_start_policy_denied_start_returns_task_open_authorization_contract" in work_html
        assert "direct_start_policy_denied_start_support_preserves_task_open_authorization_contract" in work_html
        assert "direct_start_policy_denied_start_support_includes_authorization_context" in work_html
        assert "direct_start_policy_denied_start_contract_reports_no_session_created" in work_html
        assert "direct_start_policy_denied_start_contract_reports_server_authorizes_open_false" in work_html
        assert "work_filter_no_matches" in work_html
        assert "filterSupport" in work_html
        assert "training=" not in work_html
    finally:
        store.close()


def test_personal_work_payload_includes_no_assignment_empty_state(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-paused", assignee_user="alice", status="paused")
        with _running_server(store, user="alice") as base_url:
            status, payload = _json_request(base_url, "/api/me/tasks")
            header_status, headers = _headers_request(base_url, "/api/me/tasks")
            datasets_header_status, datasets_headers = _headers_request(
                base_url,
                "/api/me/datasets",
            )
            completed_status, completed_payload = _json_request(base_url, "/api/me/tasks?include_completed=1")
            guarded_status, guarded_payload = _json_request(base_url, "/api/me/tasks?expected_user=alice")
            mismatch_status, mismatch_payload = _json_request(base_url, "/api/me/tasks?expected_user=bob")
            datasets_status, datasets_payload = _json_request(base_url, "/api/me/datasets")
            guarded_datasets_status, guarded_datasets_payload = _json_request(
                base_url,
                "/api/me/datasets?expected_user=alice",
            )
            mismatch_datasets_status, mismatch_datasets_payload = _json_request(
                base_url,
                "/api/me/datasets?expected_user=bob",
            )

        assert status == 200
        assert header_status == 200
        assert datasets_header_status == 200
        _assert_browser_response_security_headers(headers)
        _assert_browser_response_security_headers(datasets_headers)
        assert payload["ok"] is True
        assert payload["preferred_labeler_entry_url_matches_dataset_queue"] is True
        assert payload["preferred_labeler_entry_url_matches_personal_dataset_queue"] is True
        assert payload["personalized_labeler_entry_url_matches_personal_dataset_queue"] is True
        assert payload["work"]["preferred_labeler_entry_url_matches_dataset_queue"] is True
        assert payload["work"]["preferred_labeler_entry_url_matches_personal_dataset_queue"] is True
        assert payload["work"][
            "personalized_labeler_entry_url_matches_personal_dataset_queue"
        ] is True
        assert datasets_status == 200
        assert datasets_payload["preferred_labeler_entry_url_matches_dataset_queue"] is True
        assert datasets_payload[
            "preferred_labeler_entry_url_matches_personal_dataset_queue"
        ] is True
        assert datasets_payload[
            "personalized_labeler_entry_url_matches_personal_dataset_queue"
        ] is True
        assert payload["personalized_launch_readiness"]["schema"] == (
            "palette.web_labeling_personalized_launch_readiness.v1"
        )
        assert payload["personalized_launch_readiness"]["field_count"] == len(
            payload["personalized_launch_readiness"]["fields"]
        )
        assert "browser_label_write_target" in payload[
            "personalized_launch_readiness"
        ]["fields"]
        assert payload["personalized_launch_readiness"][
            "personalized_labeler_entry_url"
        ] == "/my-datasets?expected_user=alice"
        assert payload["personalized_launch_readiness"] == payload["work"][
            "personalized_launch_readiness"
        ]
        assert payload["personalized_launch_readiness"][
            "browser_label_write_target"
        ] == "training_zarr"
        assert payload["personalized_launch_readiness"][
            "browser_writes_csv_or_handoff_files"
        ] is False
        assert payload["personalized_launch_readiness"][
            "browser_has_direct_zarr_write_authority"
        ] is False
        assert payload["personalized_launch_readiness"][
            "external_launch_evidence_gap_action_required"
        ] is True
        assert "browser_smoke" in payload["personalized_launch_readiness"][
            "external_launch_evidence_gap_gate_ids"
        ]
        assert payload["work"]["personalized_launch_readiness"][
            "external_launch_evidence_gap_gate_ids"
        ] == payload["personalized_launch_readiness"][
            "external_launch_evidence_gap_gate_ids"
        ]
        assert datasets_payload["personalized_launch_readiness"][
            "personalized_labeler_entry_url"
        ] == "/my-datasets?expected_user=alice"
        assert datasets_payload["personalized_launch_readiness"]["schema"] == (
            "palette.web_labeling_personalized_launch_readiness.v1"
        )
        assert datasets_payload["personalized_launch_readiness"]["field_count"] == len(
            datasets_payload["personalized_launch_readiness"]["fields"]
        )
        assert datasets_payload["personalized_launch_readiness"][
            "browser_label_write_target"
        ] == "training_zarr"
        assert datasets_payload["personalized_launch_readiness"][
            "browser_writes_csv_or_handoff_files"
        ] is False
        assert datasets_payload["personalized_launch_readiness"][
            "browser_has_direct_zarr_write_authority"
        ] is False
        assert datasets_payload["personalized_launch_readiness"][
            "external_launch_evidence_gap_action_required"
        ] is True
        assert "browser_smoke" in datasets_payload["personalized_launch_readiness"][
            "external_launch_evidence_gap_gate_ids"
        ]
        assert guarded_payload["personalized_launch_readiness"][
            "personalized_labeler_entry_url"
        ] == "/my-datasets?expected_user=alice"
        assert guarded_datasets_payload["personalized_launch_readiness"][
            "personalized_labeler_entry_url"
        ] == "/my-datasets?expected_user=alice"
        assert payload["work"]["queue_first_entry_contract"]["ready"] is True
        assert payload["work"]["queue_first_entry_contract"][
            "personalized_labeler_entry_url"
        ] == "/my-datasets?expected_user=alice"
        assert payload["work"]["queue_first_entry_contract"][
            "personalized_labeler_entry_url_matches_personal_dataset_queue"
        ] is True
        assert datasets_payload["queue_first_entry_contract"] == payload["work"][
            "queue_first_entry_contract"
        ]
        assert payload["work"]["identity_personal_queue_evidence_status"] == "missing"
        assert datasets_payload["identity_personal_queue_evidence_status"] == "missing"
        assert payload["work"]["identity_all_users_have_personal_queue_evidence"] is False
        assert datasets_payload["identity_all_users_have_personal_queue_evidence"] is False
        assert payload["work"][
            "operator_validation_identity_personal_queue_evidence_status_values"
        ] == list(labeling_web.IDENTITY_PERSONAL_QUEUE_EVIDENCE_STATUS_VALUES)
        assert datasets_payload[
            "operator_validation_identity_personal_queue_evidence_status_values"
        ] == list(labeling_web.IDENTITY_PERSONAL_QUEUE_EVIDENCE_STATUS_VALUES)
        assert payload["work"]["operator_validation_visibility_policy"][
            "operator_validation_gate_status_values"
        ] == list(labeling_web.OPERATOR_VALIDATION_GATE_STATUS_VALUES)
        assert datasets_payload["operator_validation_visibility_policy"][
            "operator_validation_gate_status_values"
        ] == list(labeling_web.OPERATOR_VALIDATION_GATE_STATUS_VALUES)
        assert payload["work"]["operator_validation_gate_status_values"] == list(
            labeling_web.OPERATOR_VALIDATION_GATE_STATUS_VALUES
        )
        assert datasets_payload["operator_validation_gate_status_values"] == list(
            labeling_web.OPERATOR_VALIDATION_GATE_STATUS_VALUES
        )
        assert payload["work"]["operator_validation_mutation_gate"]["schema"] == (
            "palette.web_labeling_runtime_operator_validation_mutation_gate.v1"
        )
        assert payload["work"]["operator_validation_mutation_gate"][
            "required_for_browser_mutation"
        ] is False
        assert payload["work"]["operator_validation_mutation_gate"]["ready"] is True
        assert payload["work"]["operator_validation_mutation_gate"][
            "blocks_browser_mutation"
        ] is False
        assert "validation_checklist_path" not in payload["work"][
            "operator_validation_mutation_gate"
        ]
        assert payload["operator_validation_mutation_gate"] == payload["work"][
            "operator_validation_mutation_gate"
        ]
        assert datasets_payload["operator_validation_mutation_gate"] == payload["work"][
            "operator_validation_mutation_gate"
        ]
        assert payload["runtime_operator_validation_gate_cli_policy"] == payload["work"][
            "runtime_operator_validation_gate_cli_policy"
        ]
        assert datasets_payload["runtime_operator_validation_gate_cli_policy"] == payload[
            "work"
        ]["runtime_operator_validation_gate_cli_policy"]
        assert payload["work"]["runtime_operator_validation_gate_cli_policy"][
            "preferred_require_flag"
        ] == "--require-operator-validation-for-browser-work"
        assert payload["work"]["runtime_operator_validation_gate_cli_policy"][
            "legacy_require_flag"
        ] == "--require-operator-validation-for-start"
        assert payload["work"]["runtime_operator_validation_gate_cli_policy"][
            "protects_browser_mutations"
        ] is True
        assert payload["work"]["operator_validation_gate_ids"] == list(
            labeling_web.DEFAULT_OPERATOR_VALIDATION_GATE_IDS
        )
        assert datasets_payload["operator_validation_gate_ids"] == list(
            labeling_web.DEFAULT_OPERATOR_VALIDATION_GATE_IDS
        )
        assert payload["work"]["operator_validation_gate_flat_field_suffixes"] == [
            "status",
            "pending",
            "missing_evidence",
            "needs_review",
            "passed",
        ]
        assert datasets_payload["operator_validation_gate_flat_field_suffixes"] == [
            "status",
            "pending",
            "missing_evidence",
            "needs_review",
            "passed",
        ]
        for gate_id in (
            "mutable_zarr_backup_confirmation",
            "browser_response_security_headers",
            "identity_probe_verification",
            "browser_smoke",
            "disposable_zarr_mutation_smoke",
            "operator_recovery_contract",
        ):
            assert payload["work"][f"operator_validation_gate_{gate_id}_status"] == (
                "missing_evidence"
            )
            assert datasets_payload[f"operator_validation_gate_{gate_id}_status"] == (
                "missing_evidence"
            )
            assert payload["work"][f"operator_validation_gate_{gate_id}_pending"] is True
            assert datasets_payload[f"operator_validation_gate_{gate_id}_pending"] is True
            assert (
                payload["work"][f"operator_validation_gate_{gate_id}_missing_evidence"]
                is True
            )
            assert (
                datasets_payload[f"operator_validation_gate_{gate_id}_missing_evidence"]
                is True
            )
            assert payload["work"][f"operator_validation_gate_{gate_id}_passed"] is False
            assert datasets_payload[f"operator_validation_gate_{gate_id}_passed"] is False
        assert payload["work"]["empty_state"] == {
            "code": "no_active_assignments",
            "is_empty": True,
            "message": "No active labeling recordings are assigned to you right now. If you expected work, ask the operator to check your recording assignment.",
            "operator_action": "Assign at least one recording to this user or confirm the user is signing in with the expected identity.",
        }
        assert payload["work"]["known_user_status"]["is_known_labeler"] is True
        assert payload["work"]["known_user_status"]["assignment_count"] == 1
        assert payload["work"]["known_user_status"]["active_assignment_count"] == 0
        assert payload["work"]["known_user_status"]["assignment_status_counts"] == {"paused": 1}
        assert payload["single_owner_policy"] == payload["work"]["single_owner_policy"]
        assert datasets_payload["single_owner_policy"] == payload["work"]["single_owner_policy"]
        assert payload["work"]["single_owner_policy"]["assignment_scope"] == "recording"
        assert payload["work"]["single_owner_policy"]["recording_assignment_key"] == "recording_id"
        assert payload["work"]["single_owner_policy"]["one_active_owner"] is True
        assert payload["work"]["single_owner_policy"]["multiple_labelers_per_recording_allowed"] is False
        for route_row in (payload["work"], datasets_payload):
            route_checklist = route_row["labeler_route_authorization_checklist"]
            assert route_checklist["ready"] is False
            assert route_checklist["has_active_assignment"] is False
            assert route_checklist["single_owner_store_contract_required"] is True
            assert route_checklist["single_owner_store_contract_present"] is True
            assert route_checklist["single_owner_store_contract_ready"] is True
            assert route_checklist["single_owner_store_contract_met"] is True
            assert route_checklist["single_owner_store_proof_ready"] is True
            assert route_checklist["assignment_ownership_integrity_ok"] is True
            assert route_checklist["duplicate_active_owner_count"] == 0
        for row in (payload, payload["work"], datasets_payload):
            assert row["single_owner_policy_assignment_scope"] == "recording"
            assert row["single_owner_policy_recording_assignment_key"] == "recording_id"
            assert row["single_owner_policy_one_active_owner"] is True
            assert row["single_owner_policy_multiple_labelers_per_recording_allowed"] is False
            assert row["single_owner_policy_browser_mutation_requires_current_assignment_owner"] is True
            assert row["single_owner_assignment_contract"]["schema"] == (
                "palette.web_labeling_assignment_single_owner_contract.v1"
            )
            assert row["single_owner_assignment_contract"][
                "browser_mutation_target_resolved_server_side"
            ] is True
            assert row["single_owner_assignment_contract"][
                "labelers_mutate_assigned_training_zarrs"
            ] is True
            assert row["single_owner_assignment_contract"][
                "labelers_mutate_intermediate_csvs"
            ] is False
            assert row[
                "assignment_ownership_contract_store_single_owner_assignment_contract_present"
            ] is True
            assert row[
                "assignment_ownership_contract_store_single_owner_assignment_contract_ready"
            ] is True
            assert row[
                "assignment_ownership_contract_store_single_owner_assignment_contract_met"
            ] is True
            assert row["assignment_ownership_integrity"]["ok"] is True
            assert row[
                "assignment_ownership_contract_assignment_ownership_integrity_ok"
            ] is True
            assert row["assignment_ownership_contract_duplicate_active_owner_count"] == 0
            assert row[
                "assignment_ownership_contract_browser_mutation_target_resolved_server_side"
            ] is True
        assert payload["work"]["dataset_queue"] == []
        assert payload["work"]["labeler_safety"]["labeler_landing_page_path"] == "/"
        assert payload["work"]["labeler_safety"]["queue_first_landing_paths"] == [
            "/",
            "/me",
            "/labeling",
            "/datasets",
            "/my-datasets",
        ]
        assert payload["work"]["labeler_safety"]["dataset_queue_page_path"] == "/datasets"
        assert payload["work"]["labeler_safety"]["work_filter_query_keys"] == [
            "expected_user",
            "dataset_id",
            "recording_id",
            "task_id",
            "workflow",
        ]
        assert payload["work"]["labeler_safety"]["expected_user_guards"]["dataset_queue_page"] == "dashboard_user_mismatch"
        assert payload["work"]["labeler_safety"]["expected_user_guards"]["labeler_landing_page"] == "dashboard_user_mismatch"
        assert payload["work"]["labeler_safety"]["expected_user_guards"]["labeler_me_page"] == "dashboard_user_mismatch"
        assert payload["work"]["zarr_backup_policy"]["validation_gate"] == "mutable_zarr_backup_confirmation"
        assert payload["work"]["zarr_backup_policy"]["labelers_do_not_receive_backup_paths"] is True
        assert payload["work"]["mutation_audit_policy"]["event_store"] == "labeling_task_events"
        assert payload["work"]["mutation_audit_policy"]["server_records_events"] is True
        assert payload["work"]["mutation_audit_policy"]["browser_records_events_directly"] is False
        assert payload["work"]["browser_mutation_write_policy"]["authoritative_label_state"] == "assigned_task_zarr_scope"
        assert payload["work"]["browser_mutation_write_policy"]["handoff_artifacts_are_metadata_only"] is True
        assert payload["work"]["browser_mutation_write_policy"]["browser_writes_csv_or_handoff_files"] is False
        assert payload["work"]["labeler_route_authorization_policy"]["expected_user_must_match_resolved_user"] is True
        assert payload["work"]["labeler_route_authorization_policy"]["known_assignment_store_user_required"] is True
        assert payload["work"]["labeler_route_authorization_policy"]["task_open_requires_active_assignment"] is True
        assert payload["work"]["labeler_route_authorization_policy"]["signed_links_are_entry_hints_not_authorization"] is True
        assert payload["work"]["labeler_route_authorization_policy"][
            "single_owner_store_proof_required_for_browser_work"
        ] is True
        assert payload["work"]["labeler_route_authorization_policy"][
            "single_owner_store_proof_requires_zero_duplicate_active_owners"
        ] is True
        assert payload["work"]["labeler_route_authorization_policy"][
            "single_owner_store_proof_requires_training_zarr_target"
        ] is True
        assert payload["work"]["labeler_route_authorization_policy"][
            "single_owner_store_proof_rejects_intermediate_csv_mutation"
        ] is True
        route_checklist = payload["work"]["labeler_route_authorization_checklist"]
        assert route_checklist["schema"] == (
            "palette.web_labeling_labeler_route_authorization_runtime_checklist.v1"
        )
        assert route_checklist["ready"] is False
        assert route_checklist["resolved_user"] == "alice"
        assert route_checklist["known_assignment_store_user"] is True
        assert route_checklist["active_assignment_required"] is True
        assert route_checklist["active_assignment_count"] == 0
        assert route_checklist["has_active_assignment"] is False
        assert route_checklist["expected_user_matches_resolved_user"] is True
        assert route_checklist["personal_work_reads_filtered_by_resolved_user"] is True
        assert route_checklist["dataset_queue_reads_filtered_by_resolved_user"] is True
        assert route_checklist["task_open_requires_active_assignment"] is True
        assert route_checklist["task_open_requires_task_assigned_to_resolved_user"] is True
        assert route_checklist["task_open_rejects_completed_tasks"] is True
        assert route_checklist["mutation_requires_current_session"] is True
        assert route_checklist["mutation_requires_task_assigned_to_resolved_user"] is True
        assert route_checklist["mutation_requires_browser_supported_workflow"] is True
        assert route_checklist["mutation_requires_current_target_token"] is True
        assert route_checklist["mutation_rejects_client_target_selectors"] is True
        assert route_checklist["labeler_visible_scope"] == "assigned_recordings_for_resolved_user"
        assert route_checklist["data_plane_mutation_scope"] == "current_guarded_session_task_target"
        assert payload["work"]["session_guard_policy"]["stale_tab_save_rejected"] is True
        assert payload["work"]["session_guard_policy"]["target_token_required_for_mutation"] is True
        assert payload["work"]["session_guard_policy"]["labeler_promotion_retry_requires_current_session"] is True
        assert payload["work"]["session_guard_policy"]["session_closure_event_support"] is True
        assert payload["work"]["task_state_policy"]["completed_tasks_read_only"] is True
        assert payload["work"]["task_state_policy"]["browser_mutation_target_selectors"] == (
            "server_owned_reject_client_fields"
        )
        assert payload["work"]["task_state_policy"]["browser_mutation_target_token"] == (
            "required_current_target_token"
        )
        scope_checklist = payload["work"]["browser_workflow_scope_checklist"]
        assert scope_checklist["schema"] == (
            "palette.web_labeling_browser_workflow_scope_runtime_checklist.v1"
        )
        assert scope_checklist["ready"] is True
        assert scope_checklist["supported_browser_workflow_kinds"] == [
            "keypoints",
            "detect_training",
            "detect_analysis",
            "subject_mask_component",
        ]
        assert scope_checklist["absolute_navigation_out_of_scope"] == "reject_nav_error"
        assert scope_checklist["absolute_navigation_out_of_scope_rejects"] is True
        assert scope_checklist["browser_mutation_targets_server_owned"] is True
        assert scope_checklist["current_target_token_required"] is True
        assert scope_checklist["mutable_workflows_require_target_token"] is True
        assert scope_checklist["workflows_missing_target_token"] == []
        assert scope_checklist["mutable_workflows_session_guarded"] is True
        assert scope_checklist["workflows_missing_session_guard"] == []
        assert scope_checklist["browser_direct_target_selection_rejected"] is True
        assert {
            "target_zarr",
            "csv_path",
            "data_plane_write_target",
            "browser_label_write_target",
        }.issubset(set(scope_checklist["target_selector_fields_rejected"]))
        assert scope_checklist["target_indices_components_labels_frames_inside_task_scope"] is True
        assert payload["work"]["labeler_landing_page_path"] == "/"
        assert payload["work"]["expected_user_labeler_landing_url"] == "/?expected_user=alice"
        assert payload["work"]["expected_user_labeling_home_url"] == "/labeling?expected_user=alice"
        assert payload["work"]["expected_user_dashboard_url"] == "/work?expected_user=alice"
        assert payload["work"]["expected_user_dataset_queue_url"] == "/datasets?expected_user=alice"
        assert payload["work"]["expected_user_identity_probe_url"] == "/identity?expected_user=alice"
        assert guarded_status == 200
        assert guarded_payload["ok"] is True
        assert guarded_payload["work"]["expected_user"] == "alice"
        assert guarded_payload["work"]["expected_user_labeler_landing_url"] == "/?expected_user=alice"
        assert guarded_payload["work"]["expected_user_labeling_home_url"] == (
            "/labeling?expected_user=alice"
        )
        assert guarded_payload["work"]["expected_user_dataset_queue_url"] == "/datasets?expected_user=alice"
        assert mismatch_status == 403
        assert mismatch_payload["ok"] is False
        assert mismatch_payload["error"] == "dashboard_user_mismatch"
        mismatch_contract = mismatch_payload["labeler_read_authorization_contract"]
        assert mismatch_contract["schema"] == (
            "palette.web_labeling_labeler_read_authorization_contract.v1"
        )
        assert mismatch_contract["ready"] is False
        assert mismatch_contract["not_ready_reason"] == "dashboard_user_mismatch"
        assert mismatch_contract["route_path"] == "/api/me/tasks"
        assert mismatch_contract["response_kind"] == "json"
        assert mismatch_contract["resolved_user"] == "alice"
        assert mismatch_contract["expected_user"] == "bob"
        assert mismatch_contract["expected_user_guard_checked_server_side"] is True
        assert mismatch_contract["expected_user_guard_present"] is True
        assert mismatch_contract["expected_user_matches_resolved_user"] is False
        assert mismatch_contract["returns_assigned_work_payload"] is False
        assert mismatch_contract["server_authorizes_read"] is False
        assert mismatch_contract["server_authorizes_task_open"] is False
        assert mismatch_contract["server_authorizes_mutation"] is False
        assert mismatch_contract["browser_writes_csv_or_handoff_files"] is False
        assert mismatch_contract["browser_has_direct_zarr_write_authority"] is False
        assert mismatch_payload["authorization_context"]["expected_user"] == "bob"
        assert mismatch_payload["authorization_context"]["resolved_user"] == "alice"
        def assert_mismatch_readiness(source):
            readiness = source["personalized_launch_readiness"]
            assert readiness["schema"] == (
                "palette.web_labeling_personalized_launch_readiness.v1"
            )
            assert readiness["personalized_labeler_entry_url"] == (
                "/my-datasets?expected_user=bob"
            )
            assert readiness["browser_label_write_target"] == "training_zarr"
            assert readiness["browser_writes_csv_or_handoff_files"] is False
            assert readiness["browser_has_direct_zarr_write_authority"] is False

        assert_mismatch_readiness(mismatch_payload)
        assert payload["work"]["dataset_queue_summary"] == {
            "dataset_count": 0,
            "waiting_dataset_count": 0,
            "open_task_count": 0,
            "complete_task_count": 0,
            "non_startable_task_count": 0,
            "task_count": 0,
            "dataset_ids": [],
        }
        assert payload["work"]["dataset_queue_state"]["schema"] == "palette.web_labeling_dataset_queue_state.v1"
        assert payload["work"]["dataset_queue_state"]["code"] == "no_active_assignments"
        assert payload["work"]["dataset_queue_state"]["is_empty"] is True
        assert payload["work"]["dataset_queue_state"]["blocks_labeler_start"] is True
        assert payload["work"]["dataset_queue_state"]["empty_state_code"] == "no_active_assignments"
        assert datasets_status == 200
        assert datasets_payload["ok"] is True
        assert datasets_payload["labeler_landing_page_path"] == "/"
        assert datasets_payload["expected_user_labeler_landing_url"] == "/?expected_user=alice"
        assert datasets_payload["expected_user_labeling_home_url"] == "/labeling?expected_user=alice"
        assert datasets_payload["expected_user_dashboard_url"] == "/work?expected_user=alice"
        assert datasets_payload["expected_user_dataset_queue_url"] == "/datasets?expected_user=alice"
        assert datasets_payload["expected_user_identity_probe_url"] == "/identity?expected_user=alice"
        assert datasets_payload["known_user_status"] == payload["work"]["known_user_status"]
        assert datasets_payload["labeler_safety"]["dataset_queue_page_path"] == "/datasets"
        assert datasets_payload["labeler_safety"]["expected_user_guards"]["dataset_queue_page"] == "dashboard_user_mismatch"
        assert datasets_payload["zarr_backup_policy"] == payload["work"]["zarr_backup_policy"]
        assert datasets_payload["mutation_audit_policy"] == payload["work"]["mutation_audit_policy"]
        assert datasets_payload["browser_mutation_write_policy"] == payload["work"]["browser_mutation_write_policy"]
        assert datasets_payload["labeler_route_authorization_policy"] == payload["work"]["labeler_route_authorization_policy"]
        assert datasets_payload["labeler_route_authorization_checklist"] == route_checklist
        assert datasets_payload["session_guard_policy"] == payload["work"]["session_guard_policy"]
        assert datasets_payload["task_state_policy"] == payload["work"]["task_state_policy"]
        assert datasets_payload["supported_browser_workflow_kinds"] == payload["work"][
            "supported_browser_workflow_kinds"
        ]
        assert datasets_payload["browser_workflow_scope_checklist"] == scope_checklist
        assert datasets_payload["datasets"] == []
        assert datasets_payload["dataset_queue"] == []
        assert datasets_payload["dataset_queue_summary"] == payload["work"]["dataset_queue_summary"]
        assert datasets_payload["dataset_queue_state"] == payload["work"]["dataset_queue_state"]
        assert guarded_datasets_status == 200
        assert guarded_datasets_payload["ok"] is True
        assert guarded_datasets_payload["expected_user"] == "alice"
        assert guarded_datasets_payload["expected_user_labeler_landing_url"] == "/?expected_user=alice"
        assert guarded_datasets_payload["expected_user_labeling_home_url"] == (
            "/labeling?expected_user=alice"
        )
        assert guarded_datasets_payload["expected_user_dataset_queue_url"] == "/datasets?expected_user=alice"
        assert guarded_datasets_payload["dataset_queue"] == []
        assert mismatch_datasets_status == 403
        assert mismatch_datasets_payload["ok"] is False
        assert mismatch_datasets_payload["error"] == "dashboard_user_mismatch"
        mismatch_datasets_contract = mismatch_datasets_payload[
            "labeler_read_authorization_contract"
        ]
        assert mismatch_datasets_contract["route_path"] == "/api/me/datasets"
        assert mismatch_datasets_contract["response_kind"] == "json"
        assert mismatch_datasets_contract["expected_user_matches_resolved_user"] is False
        assert mismatch_datasets_contract["returns_dataset_queue_payload"] is False
        assert mismatch_datasets_contract["server_authorizes_read"] is False
        assert mismatch_datasets_contract["server_authorizes_mutation"] is False
        assert mismatch_datasets_contract["browser_writes_csv_or_handoff_files"] is False
        assert_mismatch_readiness(mismatch_datasets_payload)
        assert mismatch_datasets_contract["browser_has_direct_zarr_write_authority"] is False
    finally:
        store.close()


def test_personal_work_api_rejects_unknown_labeling_user(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        with _running_server(store, user="alice") as base_url:
            status, payload = _json_request(base_url, "/api/me/tasks")
            datasets_status, datasets_payload = _json_request(base_url, "/api/me/datasets")
            identity_status, identity_payload = _json_request(base_url, "/api/me/identity?expected_user=alice")
            identity_html_status, identity_html = _text_request(base_url, "/identity?expected_user=alice")

        assert status == 403
        assert payload["ok"] is False
        assert payload["error"] == "unknown_labeling_user"
        assert payload["known_user_status"]["is_known_labeler"] is False
        assert payload["labeler_route_authorization_policy"]["known_assignment_store_user_required"] is True
        assert payload["labeler_route_authorization_checklist"]["ready"] is False
        assert payload["labeler_route_authorization_checklist"][
            "known_assignment_store_user"
        ] is False
        assert payload["labeler_route_authorization_checklist"][
            "active_assignment_required"
        ] is True
        assert payload["labeler_route_authorization_checklist"]["has_active_assignment"] is False
        assert payload["expected_user_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert payload["preferred_labeler_entrypoint"] == "personal_datasets_waiting_queue"
        assert payload["preferred_labeler_entry_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert payload["personal_dataset_queue_link_role"] == "preferred_queue"
        assert payload["dataset_queue_link_role"] == "canonical_queue_fallback"
        assert payload["queue_first_entry_contract"]["personalized_labeler_entry_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert payload["queue_first_entry_contract"]["personalized_labeler_entry_url_matches_personal_dataset_queue"] is True
        def assert_unknown_user_readiness(source):
            readiness = source["personalized_launch_readiness"]
            assert readiness["schema"] == (
                "palette.web_labeling_personalized_launch_readiness.v1"
            )
            assert readiness["personalized_labeler_entry_url"] == (
                "/my-datasets?expected_user=alice"
            )
            assert readiness["browser_label_write_target"] == "training_zarr"
            assert readiness["browser_writes_csv_or_handoff_files"] is False
            assert readiness["browser_has_direct_zarr_write_authority"] is False

        assert_unknown_user_readiness(payload)
        assert payload["single_owner_policy"]["assignment_scope"] == "recording"
        assert payload["single_owner_policy"]["one_active_owner"] is True
        assert payload["single_owner_policy"]["multiple_labelers_per_recording_allowed"] is False
        assert payload["single_owner_policy_browser_mutation_requires_current_assignment_owner"] is True
        assert datasets_status == 403
        assert datasets_payload["error"] == "unknown_labeling_user"
        assert datasets_payload["known_user_status"]["is_known_labeler"] is False
        assert datasets_payload["single_owner_policy"] == payload["single_owner_policy"]
        assert datasets_payload["single_owner_policy_assignment_scope"] == "recording"
        assert datasets_payload["single_owner_policy_one_active_owner"] is True
        assert datasets_payload["single_owner_policy_multiple_labelers_per_recording_allowed"] is False
        assert datasets_payload["expected_user_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert datasets_payload["preferred_labeler_entry_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert datasets_payload["queue_first_entry_contract"] == payload[
            "queue_first_entry_contract"
        ]
        assert_unknown_user_readiness(datasets_payload)
        assert identity_status == 403
        assert identity_payload["ok"] is False
        assert identity_payload["error"] == "unknown_labeling_user"
        assert identity_payload["identity"]["resolved_user"] == "alice"
        assert identity_payload["known_user_status"]["is_known_labeler"] is False
        assert identity_payload["personalized_launch_readiness"] == (
            identity_payload["identity"]["personalized_launch_readiness"]
        )
        assert_unknown_user_readiness(identity_payload)
        assert identity_html_status == 403
        assert "Unknown labeling user: stop before labeling" in identity_html
        assert "Known labeler" in identity_html
    finally:
        store.close()


def test_dataset_queue_summary_counts_unspecified_dataset_group(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            title="Task without dataset id",
        )

        with _running_server(store, user="alice") as base_url:
            status, payload = _json_request(base_url, "/api/me/datasets")

        assert status == 200
        assert payload["ok"] is True
        assert payload["dataset_queue_summary"] == {
            "dataset_count": 1,
            "waiting_dataset_count": 1,
            "open_task_count": 1,
            "complete_task_count": 0,
            "non_startable_task_count": 0,
            "task_count": 1,
            "dataset_ids": [],
        }
        assert payload["dataset_queue"][0]["dataset_id"] == ""
        assert payload["dataset_queue"][0]["dataset_label"] == "Unspecified dataset"
        assert payload["dataset_queue"][0]["expected_user_work_url"] == "/work?expected_user=alice"
        assert payload["dataset_queue"][0]["recordings"][0]["tasks"][0]["expected_user_work_url"] == (
            "/work?expected_user=alice&recording_id=rec-a&task_id=task-a&workflow=keypoints"
        )
    finally:
        store.close()


def test_dashboard_page_requires_authenticated_user(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        with _running_server(store, user=None) as base_url:
            status, body = _text_request(base_url, "/")
            work_status, work_body = _text_request(base_url, "/work")

        assert status == 401
        assert work_status == 401
        assert "Palette labeling access problem" in body
        assert "Palette labeling access problem" in work_body
        assert "What to send the operator" in body
        assert "Copy support details" in body
        assert 'href="/"' in body
        assert "Return to your labeling landing page" in body
        assert 'href="/work"' in body
        assert "authentication_required" in body
        assert "authentication_required" in work_body
        assert "Work waiting for completion" not in body
        assert "Work waiting for completion" not in work_body
    finally:
        store.close()


def test_me_tasks_returns_only_current_assignee_work(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-done", assignee_user="alice", notes="Finished recording")
        store.assign_recording(recording_id="rec-empty", assignee_user="alice", notes="Waiting for task generation")
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            title="Alice task",
            dataset_id="dataset-a",
            zarr_use="training",
            scope={"zarr_path": "/secret/alice-training.zarr", "target_frames": [1, 2]},
        )
        store.upsert_task(
            task_id="task-done",
            recording_id="rec-done",
            workflow_kind="keypoints",
            title="Completed Alice task",
            dataset_id="dataset-done",
            zarr_use="training",
            state="complete",
        )
        store.upsert_task(
            task_id="task-b",
            recording_id="rec-b",
            workflow_kind="detect_training",
            title="Bob task",
            dataset_id="dataset-b",
            zarr_use="training",
        )

        with _running_server(store, user="alice") as base_url:
            status, payload = _json_request(base_url, "/api/me/tasks")
            datasets_status, datasets_payload = _json_request(base_url, "/api/me/datasets")
            completed_status, completed_payload = _json_request(base_url, "/api/me/tasks?include_completed=1")

        assert status == 200
        assert payload["ok"] is True
        assert payload["work"]["include_completed"] is False
        assert payload["work"]["user"] == "alice"
        assert [row["recording_id"] for row in payload["work"]["recordings"]] == ["rec-a", "rec-done", "rec-empty"]
        task = payload["work"]["recordings"][0]["tasks"][0]
        assert task["task_id"] == "task-a"
        assert task["dataset_id"] == "dataset-a"
        assert "scope" not in task
        assert task["redacted_fields"] == ["scope"]
        raw_task_support = task["operator_support"]
        assert raw_task_support["expected_user_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert raw_task_support["personalized_labeler_entry_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert raw_task_support["personal_dataset_queue_link_role"] == "preferred_queue"
        assert raw_task_support["canonical_dataset_queue_link_role"] == (
            "canonical_queue_fallback"
        )
        assert raw_task_support["browser_label_write_target"] == "training_zarr"
        assert raw_task_support["browser_writes_csv_or_handoff_files"] is False
        assert raw_task_support["browser_has_direct_zarr_write_authority"] is False
        assert raw_task_support["csv_handoff_artifact_role"] == (
            "metadata_only_control_plane"
        )
        assert payload["work"]["dataset_queue_summary"] == {
            "dataset_count": 1,
            "waiting_dataset_count": 1,
            "open_task_count": 1,
            "complete_task_count": 0,
            "non_startable_task_count": 0,
            "task_count": 1,
            "dataset_ids": ["dataset-a"],
        }
        assert payload["work"]["dataset_queue"][0]["dataset_id"] == "dataset-a"
        assert payload["work"]["dataset_queue_state"]["code"] == "has_open_dataset_work"
        assert payload["work"]["dataset_queue_state"]["has_open_dataset_work"] is True
        assert payload["work"]["dataset_queue_state"]["blocks_labeler_start"] is False
        assert payload["work"]["dataset_queue_state"]["counts"]["waiting_dataset_count"] == 1
        assert payload["work"]["dataset_queue"][0]["dataset_label"] == "dataset-a"
        expected_row_support_contract = {
            "expected_user_personal_dataset_queue_url": "/my-datasets?expected_user=alice",
            "personalized_labeler_entry_url": "/my-datasets?expected_user=alice",
            "personal_dataset_queue_link_role": "preferred_queue",
            "canonical_dataset_queue_link_role": "canonical_queue_fallback",
            "browser_label_write_target": "training_zarr",
            "browser_writes_csv_or_handoff_files": False,
            "browser_has_direct_zarr_write_authority": False,
            "csv_handoff_artifact_role": "metadata_only_control_plane",
        }
        dataset_support = payload["work"]["dataset_queue"][0]["operator_support"]
        assert {
            key: dataset_support[key]
            for key in expected_row_support_contract
        } == expected_row_support_contract
        assert {
            key: dataset_support[key]
            for key in (
                "user",
                "dataset_id",
                "dataset_label",
                "expected_user_work_url",
                "task_count",
                "open_task_count",
                "recording_count",
                "workflow_counts",
            )
        } == {
            "user": "alice",
            "dataset_id": "dataset-a",
            "dataset_label": "dataset-a",
            "expected_user_work_url": "/work?expected_user=alice&dataset_id=dataset-a",
            "task_count": 1,
            "open_task_count": 1,
            "recording_count": 1,
            "workflow_counts": {"keypoints": 1},
        }
        assert payload["work"]["dataset_queue"][0]["work_url"] == "/work?dataset_id=dataset-a"
        assert payload["work"]["dataset_queue"][0]["expected_user_work_url"] == "/work?expected_user=alice&dataset_id=dataset-a"
        assert payload["work"]["dataset_queue"][0]["open_task_count"] == 1
        assert payload["work"]["dataset_queue"][0]["recordings"][0]["recording_id"] == "rec-a"
        recording_support = payload["work"]["dataset_queue"][0]["recordings"][0][
            "operator_support"
        ]
        assert {
            key: recording_support[key]
            for key in expected_row_support_contract
        } == expected_row_support_contract
        assert {
            key: recording_support[key]
            for key in (
                "user",
                "dataset_id",
                "recording_id",
                "expected_user_work_url",
                "task_count",
                "open_task_count",
                "workflow_counts",
            )
        } == {
            "user": "alice",
            "dataset_id": "dataset-a",
            "recording_id": "rec-a",
            "expected_user_work_url": "/work?expected_user=alice&dataset_id=dataset-a&recording_id=rec-a",
            "task_count": 1,
            "open_task_count": 1,
            "workflow_counts": {"keypoints": 1},
        }
        assert payload["work"]["dataset_queue"][0]["recordings"][0]["work_url"] == "/work?dataset_id=dataset-a&recording_id=rec-a"
        assert payload["work"]["dataset_queue"][0]["recordings"][0]["expected_user_work_url"] == (
            "/work?expected_user=alice&dataset_id=dataset-a&recording_id=rec-a"
        )
        assert payload["work"]["dataset_queue"][0]["recordings"][0]["tasks"][0]["task_id"] == "task-a"
        task_support = payload["work"]["dataset_queue"][0]["recordings"][0]["tasks"][0][
            "operator_support"
        ]
        assert {
            key: task_support[key]
            for key in expected_row_support_contract
        } == expected_row_support_contract
        assert {
            key: task_support[key]
            for key in (
                "user",
                "dataset_id",
                "recording_id",
                "task_id",
                "workflow_kind",
                "state",
                "zarr_use",
                "stage_group",
                "run_name",
                "component_name",
                "expected_user_work_url",
            )
        } == {
            "user": "alice",
            "dataset_id": "dataset-a",
            "recording_id": "rec-a",
            "task_id": "task-a",
            "workflow_kind": "keypoints",
            "state": "pending",
            "zarr_use": "training",
            "stage_group": "",
            "run_name": "",
            "component_name": "",
            "expected_user_work_url": "/work?expected_user=alice&dataset_id=dataset-a&recording_id=rec-a&task_id=task-a&workflow=keypoints",
        }
        assert payload["work"]["dataset_queue"][0]["recordings"][0]["tasks"][0]["work_url"] == (
            "/work?dataset_id=dataset-a&recording_id=rec-a&task_id=task-a&workflow=keypoints"
        )
        assert payload["work"]["dataset_queue"][0]["recordings"][0]["tasks"][0]["expected_user_work_url"] == (
            "/work?expected_user=alice&dataset_id=dataset-a&recording_id=rec-a&task_id=task-a&workflow=keypoints"
        )
        assert datasets_status == 200
        assert datasets_payload["ok"] is True
        assert datasets_payload["user"] == "alice"
        assert datasets_payload["datasets"] == payload["work"]["dataset_queue"]
        assert datasets_payload["dataset_queue_summary"] == payload["work"]["dataset_queue_summary"]
        assert datasets_payload["dataset_queue_state"] == payload["work"]["dataset_queue_state"]
        assert datasets_payload["labeler_safety"]["browser_receives_task_scope"] is False
        assert datasets_payload["labeler_safety"]["browser_receives_raw_zarr_paths"] is False
        assert datasets_payload["labeler_safety"]["labeler_failed_promotion_retry_action"] == "operator_support_only"
        assert datasets_payload["labeler_safety"]["labeler_promotion_retry_requires_current_session"] is True
        assert datasets_payload["labeler_safety"]["operator_failed_promotion_retry_route"] == (
            "/api/admin/events/{event_id}/retry-promotion"
        )
        assert datasets_payload["labeler_safety"]["labeler_api_redaction"]["redacts_task_scope"] is True
        assert datasets_payload["labeler_safety"]["labeler_api_redaction"]["redacts_raw_zarr_paths"] is True
        assert datasets_payload["labeler_safety"]["labeler_api_redaction"]["redacts_runtime_state_paths"] is True
        assert datasets_payload["labeler_safety"]["labeler_api_redaction"]["redacts_mutation_response_paths"] is True
        assert datasets_payload["labeler_safety"]["labeler_api_redaction"]["redacts_error_detail_paths"] is True
        assert datasets_payload["labeler_safety"]["labeler_api_redaction"]["redacts_path_like_string_values"] is True
        assert datasets_payload["labeler_safety"]["labeler_api_redaction"]["redacts_user_summary_path_like_string_values"] is True
        assert datasets_payload["labeler_safety"]["labeler_api_redaction"]["admin_diagnostics_unredacted"] is True
        assert datasets_payload["labeler_safety"]["labeler_api_redaction"]["operator_support_redacted"] is True
        assert datasets_payload["zarr_backup_policy"]["validation_gate"] == "mutable_zarr_backup_confirmation"
        assert datasets_payload["zarr_backup_policy"]["labelers_do_not_receive_backup_paths"] is True
        assert datasets_payload["mutation_audit_policy"]["event_store"] == "labeling_task_events"
        assert datasets_payload["mutation_audit_policy"]["append_only"] is True
        assert datasets_payload["labeler_route_authorization_checklist"]["ready"] is True
        assert datasets_payload["labeler_route_authorization_checklist"]["has_active_assignment"] is True
        assert datasets_payload["labeler_route_authorization_checklist"]["active_assignment_count"] == 3
        assert datasets_payload["labeler_route_authorization_checklist"][
            "single_owner_store_contract_required"
        ] is True
        assert datasets_payload["labeler_route_authorization_checklist"][
            "single_owner_store_contract_present"
        ] is True
        assert datasets_payload["labeler_route_authorization_checklist"][
            "single_owner_store_contract_ready"
        ] is True
        assert datasets_payload["labeler_route_authorization_checklist"][
            "single_owner_store_contract_met"
        ] is True
        assert datasets_payload["labeler_route_authorization_checklist"][
            "single_owner_store_proof_ready"
        ] is True
        assert datasets_payload["labeler_route_authorization_checklist"][
            "assignment_ownership_integrity_ok"
        ] is True
        assert datasets_payload["labeler_route_authorization_checklist"][
            "duplicate_active_owner_count"
        ] == 0
        assert datasets_payload["labeler_route_authorization_checklist"][
            "browser_mutation_target_resolved_server_side"
        ] is True
        assert datasets_payload["labeler_route_authorization_checklist"][
            "labelers_mutate_assigned_training_zarrs"
        ] is True
        assert datasets_payload["labeler_route_authorization_checklist"][
            "labelers_mutate_intermediate_csvs"
        ] is False
        assert datasets_payload["labeler_route_authorization_checklist"][
            "mutation_requires_current_target_token"
        ] is True
        assert datasets_payload["task_state_policy"]["completed_tasks_read_only"] is True
        assert datasets_payload["task_state_policy"]["browser_mutation_target_token"] == (
            "required_current_target_token"
        )
        assert datasets_payload["browser_workflow_scope_checklist"]["ready"] is True
        assert datasets_payload["browser_workflow_scope_checklist"][
            "target_indices_components_labels_frames_inside_task_scope"
        ] is True
        assert datasets_payload["browser_workflow_scope_checklist"][
            "browser_direct_target_selection_rejected"
        ] is True
        assert datasets_payload["browser_workflow_scope_checklist"][
            "workflows_missing_target_token"
        ] == []
        assert payload["work"]["recordings"][1]["assignment_notes"] == "Finished recording"
        assert payload["work"]["recordings"][1]["tasks"] == []
        assert payload["work"]["recordings"][1]["total_task_count"] == 1
        assert payload["work"]["recordings"][1]["complete_task_count"] == 1
        assert payload["work"]["recordings"][1]["no_open_task_reason"] == "all_tasks_complete"
        assert "All tasks for this recording are complete" in payload["work"]["recordings"][1]["no_open_task_message"]
        assert payload["work"]["recordings"][2]["assignment_notes"] == "Waiting for task generation"
        assert payload["work"]["recordings"][2]["tasks"] == []
        assert payload["work"]["recordings"][2]["total_task_count"] == 0
        assert payload["work"]["recordings"][2]["no_open_task_reason"] == "tasks_not_generated"
        assert "no browser-labeling tasks have been generated yet" in payload["work"]["recordings"][2]["no_open_task_message"]
        assert payload["work"]["progress_summary"] == {
            "recording_count": 3,
            "open_task_count": 1,
            "startable_task_count": 1,
            "non_startable_task_count": 0,
            "total_task_count": 2,
            "complete_task_count": 1,
            "incomplete_task_count": 1,
            "waiting_recording_count": 1,
            "complete_recording_count": 1,
            "blocked_recording_count": 1,
            "waiting_recordings": ["rec-a"],
            "complete_recordings": ["rec-done"],
            "blocked_recordings": ["rec-empty"],
            "blocked_recordings_by_reason": {"tasks_not_generated": 1},
        }
        assert completed_status == 200
        assert completed_payload["ok"] is True
        assert completed_payload["work"]["include_completed"] is True
        assert completed_payload["work"]["recordings"][1]["tasks"][0]["task_id"] == "task-done"
        assert completed_payload["work"]["recordings"][1]["tasks"][0]["state"] == "complete"
        assert {row["dataset_id"] for row in completed_payload["work"]["dataset_queue"]} == {"dataset-a"}
        assert "scope" not in completed_payload["work"]["recordings"][1]["tasks"][0]
        assert "/secret/alice-training.zarr" not in json.dumps(payload)
        assert "/secret/alice-training.zarr" not in json.dumps(datasets_payload)
        assert "/secret/alice-training.zarr" not in json.dumps(completed_payload)
    finally:
        store.close()


def test_dashboard_expected_user_query_blocks_wrong_authenticated_user(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        with _running_server(store, user="bob") as base_url:
            status, body = _text_request(base_url, "/work?expected_user=alice")
            datasets_status, datasets_body = _text_request(base_url, "/datasets?expected_user=alice")
            my_work_status, my_work_body = _text_request(base_url, "/my-work?expected_user=alice")
            my_datasets_status, my_datasets_body = _text_request(
                base_url,
                "/my-datasets?expected_user=alice",
            )
            me_status, me_body = _text_request(base_url, "/me?expected_user=alice")
            identity_status, identity_body = _text_request(base_url, "/identity?expected_user=alice")
            identity_api_status, identity_api_payload = _json_request(base_url, "/api/me/identity?expected_user=alice")

        assert status == 403
        assert "Palette labeling access problem" in body
        assert "dashboard_user_mismatch" in body
        assert "labeler_read_authorization_contract" in body
        assert "server_authorizes_read" in body
        assert datasets_status == 403
        assert "Palette labeling access problem" in datasets_body
        assert "dashboard_user_mismatch" in datasets_body
        assert "labeler_read_authorization_contract" in datasets_body
        assert my_work_status == 403
        assert "dashboard_user_mismatch" in my_work_body
        assert "labeler_read_authorization_contract" in my_work_body
        assert my_datasets_status == 403
        assert "dashboard_user_mismatch" in my_datasets_body
        assert "labeler_read_authorization_contract" in my_datasets_body
        assert me_status == 403
        assert "dashboard_user_mismatch" in me_body
        assert "labeler_read_authorization_contract" in me_body
        assert "This dashboard link is for alice" in body
        assert "authenticated as bob" in body
        assert "Work waiting for completion" not in body
        assert identity_status == 403
        assert "Palette labeling identity check" in identity_body
        assert "Identity mismatch: stop before labeling" in identity_body
        assert "This identity probe is for alice" in identity_body
        assert identity_api_status == 403
        assert identity_api_payload["ok"] is False
        assert identity_api_payload["error"] == "identity_user_mismatch"
        assert identity_api_payload["identity"]["resolved_user"] == "bob"
        assert identity_api_payload["identity"]["expected_user_labeler_landing_url"] == "/?expected_user=alice"
        assert identity_api_payload["identity"]["expected_user"] == "alice"
        assert identity_api_payload["identity"]["matches_expected_user"] is False
        assert identity_api_payload["identity"]["expected_user_dataset_queue_url"] == "/datasets?expected_user=alice"
    finally:
        store.close()


def test_open_task_route_requires_current_assignment(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.assign_recording(recording_id="rec-complete", assignee_user="bob")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="keypoints")
        store.upsert_task(
            task_id="task-complete",
            recording_id="rec-complete",
            workflow_kind="keypoints",
            state="complete",
        )

        with _running_server(store, user="bob") as base_url:
            status, payload = _json_request(base_url, "/api/tasks/task-a/open", method="POST", payload={})
            mismatch_status, mismatch_payload = _json_request(
                base_url,
                "/api/tasks/task-b/open",
                method="POST",
                payload={"expected_user": "alice"},
            )
            complete_status, complete_payload = _json_request(
                base_url,
                "/api/tasks/task-complete/open",
                method="POST",
                payload={},
            )

        assert status == 403
        assert payload["ok"] is False
        assert payload["error"] == "not_assigned"
        assert payload["authorization_context"]["resolved_user"] == "bob"
        assert payload["authorization_context"]["task_id"] == "task-a"
        assert payload["authorization_context"]["assignee_user"] == "alice"
        assert payload["authorization_context"]["return_expected_user"] == "alice"
        assert payload["authorization_context"]["return_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert payload["authorization_context"][
            "return_personal_dataset_queue_expected_user_guarded"
        ] is True
        assert payload["authorization_context"]["return_personal_work_url"] == (
            "/my-work?expected_user=alice"
        )
        assert payload["authorization_context"]["return_personal_work_expected_user_guarded"] is True
        assert payload["task_open_authorization_contract"]["ready"] is False
        assert payload["task_open_authorization_contract"]["not_ready_reason"] == "not_assigned"
        assert payload["task_open_authorization_contract"][
            "expected_user_guard_checked_server_side"
        ] is True
        assert payload["task_open_authorization_contract"]["expected_user_guard_present"] is False
        assert payload["task_open_authorization_contract"][
            "expected_user_matches_resolved_user"
        ] is True
        assert payload["task_open_authorization_contract"]["active_assignment_present"] is False
        assert payload["task_open_authorization_contract"][
            "task_assigned_to_resolved_user"
        ] is False
        assert payload["task_open_authorization_contract"]["task_state_startable"] is True
        assert payload["task_open_authorization_contract"][
            "reassignment_session_safety_checked_server_side"
        ] is False
        assert payload["task_open_authorization_contract"]["session_created_server_side"] is False
        assert payload["task_open_authorization_contract"]["server_authorizes_open"] is False
        assert payload["task_open_authorization_contract"]["browser_label_write_target"] == (
            "training_zarr"
        )
        assert payload["task_open_authorization_contract"][
            "browser_writes_csv_or_handoff_files"
        ] is False
        assert payload["task_open_authorization_contract"][
            "browser_has_direct_zarr_write_authority"
        ] is False
        assert mismatch_status == 403
        assert mismatch_payload["ok"] is False
        assert mismatch_payload["error"] == "task_open_user_mismatch"
        assert mismatch_payload["authorization_context"]["resolved_user"] == "bob"
        assert mismatch_payload["authorization_context"]["expected_user"] == "alice"
        assert mismatch_payload["authorization_context"]["task_id"] == "task-b"
        assert mismatch_payload["authorization_context"]["return_expected_user"] == "alice"
        assert mismatch_payload["authorization_context"]["return_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert mismatch_payload["authorization_context"][
            "return_personal_dataset_queue_expected_user_guarded"
        ] is True
        assert mismatch_payload["authorization_context"]["return_personal_work_url"] == (
            "/my-work?expected_user=alice"
        )
        assert mismatch_payload["authorization_context"][
            "return_personal_work_expected_user_guarded"
        ] is True
        assert mismatch_payload["task_open_authorization_contract"]["ready"] is False
        assert mismatch_payload["task_open_authorization_contract"]["not_ready_reason"] == (
            "task_open_user_mismatch"
        )
        assert mismatch_payload["task_open_authorization_contract"][
            "expected_user_guard_checked_server_side"
        ] is True
        assert mismatch_payload["task_open_authorization_contract"][
            "expected_user_guard_present"
        ] is True
        assert mismatch_payload["task_open_authorization_contract"][
            "expected_user_matches_resolved_user"
        ] is False
        assert mismatch_payload["task_open_authorization_contract"][
            "active_assignment_present"
        ] is True
        assert mismatch_payload["task_open_authorization_contract"][
            "reassignment_session_safety_checked_server_side"
        ] is False
        assert mismatch_payload["task_open_authorization_contract"][
            "session_created_server_side"
        ] is False
        assert mismatch_payload["expected_user_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert mismatch_payload["personalized_labeler_entry_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert mismatch_payload["dataset_queue_direct_start_policy"][
            "browser_label_write_target"
        ] == "training_zarr"
        assert mismatch_payload["dataset_queue_direct_start_policy"][
            "browser_writes_csv_or_handoff_files"
        ] is False
        assert mismatch_payload["dataset_queue_direct_start_policy"][
            "browser_has_direct_zarr_write_authority"
        ] is False
        mismatch_readiness = mismatch_payload["personalized_launch_readiness"]
        assert mismatch_readiness["schema"] == (
            "palette.web_labeling_personalized_launch_readiness.v1"
        )
        assert mismatch_readiness["personalized_labeler_entry_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert mismatch_readiness["browser_label_write_target"] == "training_zarr"
        assert mismatch_readiness["browser_writes_csv_or_handoff_files"] is False
        assert mismatch_readiness["browser_has_direct_zarr_write_authority"] is False
        assert complete_status == 409
        assert complete_payload["ok"] is False
        assert complete_payload["error"] == "task_complete"
        assert "reopened by an operator" in complete_payload["details"]
        assert complete_payload["authorization_context"]["task_id"] == "task-complete"
        assert complete_payload["authorization_context"]["task_state"] == "complete"
        assert complete_payload["authorization_context"]["return_expected_user"] == "bob"
        assert complete_payload["authorization_context"]["return_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=bob"
        )
        assert complete_payload["authorization_context"][
            "return_personal_dataset_queue_expected_user_guarded"
        ] is True
        assert complete_payload["authorization_context"]["return_personal_work_url"] == (
            "/my-work?expected_user=bob"
        )
        assert complete_payload["authorization_context"][
            "return_personal_work_expected_user_guarded"
        ] is True
        assert complete_payload["task_open_authorization_contract"]["ready"] is False
        assert complete_payload["task_open_authorization_contract"]["not_ready_reason"] == (
            "task_complete"
        )
        assert complete_payload["task_open_authorization_contract"][
            "active_assignment_present"
        ] is True
        assert complete_payload["task_open_authorization_contract"][
            "task_assigned_to_resolved_user"
        ] is True
        assert complete_payload["task_open_authorization_contract"][
            "task_state_startable"
        ] is False
        assert complete_payload["task_open_authorization_contract"][
            "session_created_server_side"
        ] is False
        assert complete_payload["task_open_authorization_contract"][
            "server_authorizes_open"
        ] is False
    finally:
        store.close()


def test_open_task_route_blocks_until_operator_validation_complete(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    checklist_path = tmp_path / "validation-checklist.json"
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        checklist_path.write_text(
            json.dumps(
                {
                    "schema": "palette.web_labeling_validation_checklist.v1",
                    "all_validation_complete": False,
                    "ready_for_operator_validation": True,
                    "gates": [
                        {
                            "id": "browser_smoke",
                            "title": "Browser smoke",
                            "status": "pending_operator_evidence",
                            "required": True,
                            "blocks_invitation": True,
                            "evidence": [],
                            "evidence_files": [],
                        }
                    ],
                }
            )
        )

        with _running_server(
            store,
            user="alice",
            admin_users=("alice",),
            validation_checklist_path=checklist_path,
            require_operator_validation_for_start=True,
        ) as base_url:
            queue_status, queue_payload = _json_request(
                base_url,
                "/api/me/datasets?expected_user=alice",
            )
            work_status, work_payload = _json_request(
                base_url,
                "/api/me/tasks?expected_user=alice",
            )
            blocked_status, blocked_payload = _json_request(
                base_url,
                "/api/tasks/task-a/open",
                method="POST",
                payload={"expected_user": "alice"},
            )
            pending_admin_status, pending_admin_payload = _json_request(
                base_url,
                "/api/admin/summary",
            )
            checklist_path.write_text(
                json.dumps(
                    {
                        "schema": "palette.web_labeling_validation_checklist.v1",
                        "all_validation_complete": True,
                        "ready_for_operator_validation": True,
                        "gates": [
                            {
                                "id": "browser_smoke",
                                "title": "Browser smoke",
                                "status": "passed",
                                "required": True,
                                "blocks_invitation": True,
                                "evidence": ["operator-approved browser smoke"],
                                "evidence_files": ["browser-smoke-evidence-template.json"],
                            }
                        ],
                    }
                )
            )
            passed_status, passed_payload = _json_request(
                base_url,
                "/api/tasks/task-a/open",
                method="POST",
                payload={"expected_user": "alice"},
            )
            passed_admin_status, passed_admin_payload = _json_request(
                base_url,
                "/api/admin/summary",
            )

        assert queue_status == 200
        assert queue_payload["operator_validation_start_gate"][
            "required_for_browser_start"
        ] is True
        assert queue_payload["operator_validation_start_gate"]["ready"] is False
        assert queue_payload["operator_validation_start_gate"][
            "blocks_task_open"
        ] is True
        assert queue_payload["operator_validation_start_gate"][
            "operator_validation_pending_gate_ids"
        ] == ["browser_smoke"]
        assert "validation_checklist_path" not in queue_payload[
            "operator_validation_start_gate"
        ]
        assert work_status == 200
        assert work_payload["operator_validation_start_gate"][
            "required_for_browser_start"
        ] is True
        assert work_payload["operator_validation_start_gate"]["ready"] is False
        assert work_payload["operator_validation_start_gate"]["blocks_task_open"] is True
        assert work_payload["work"]["operator_validation_start_gate"] == work_payload[
            "operator_validation_start_gate"
        ]
        assert "validation_checklist_path" not in work_payload[
            "operator_validation_start_gate"
        ]
        blocked_contract = blocked_payload["task_open_authorization_contract"]
        blocked_gate = blocked_contract["operator_validation_start_gate"]
        assert blocked_status == 409
        assert blocked_payload["ok"] is False
        assert blocked_payload["error"] == "operator_validation_start_blocked"
        assert "session" not in blocked_payload
        assert blocked_contract["ready"] is False
        assert blocked_contract["not_ready_reason"] == "operator_validation_start_blocked"
        assert blocked_contract["active_assignment_present"] is True
        assert blocked_contract["task_state_startable"] is True
        assert blocked_contract["session_created_server_side"] is False
        assert blocked_contract["server_authorizes_open"] is False
        assert blocked_contract["operator_validation_start_gate_required"] is True
        assert blocked_contract["operator_validation_start_gate_ready"] is False
        assert blocked_contract["operator_validation_start_gate_blocks_task_open"] is True
        assert blocked_contract["operator_validation_start_gate_not_ready_reason"] == (
            "operator_validation_pending_operator_evidence"
        )
        assert blocked_gate["required_for_browser_start"] is True
        assert blocked_gate["validation_checklist_configured"] is True
        assert blocked_gate["blocks_task_open"] is True
        assert blocked_gate["operator_validation_status"] == "pending_operator_evidence"
        assert blocked_gate["operator_validation_pending_gate_ids"] == ["browser_smoke"]
        assert blocked_gate["operator_validation_required_missing_evidence_gate_ids"] == [
            "browser_smoke"
        ]
        assert "validation_checklist_path" not in blocked_gate
        assert pending_admin_status == 200
        pending_admin = pending_admin_payload["admin"]
        assert pending_admin["operator_validation"]["operator_validation_source"] == (
            "validation_checklist"
        )
        assert pending_admin["operator_validation"]["operator_validation_pending_gate_ids"] == [
            "browser_smoke"
        ]
        assert pending_admin["operator_validation"][
            "safe_share_checklist_gate_evidence_complete"
        ] is False
        assert pending_admin["operator_validation"][
            "safe_share_launch_blocking_pending_gate_ids"
        ] == ["browser_smoke"]
        assert pending_admin["safe_share_checklist_gate_evidence_complete"] is False
        assert pending_admin["safe_share_launch_blocking_pending_gate_ids"] == [
            "browser_smoke"
        ]
        assert pending_admin[
            "safe_share_external_launch_evidence_gap_action_required"
        ] is True
        assert pending_admin[
            "safe_share_external_launch_evidence_gap_gate_ids"
        ] == pending_admin["safe_share_launch_blocking_unsatisfied_gate_ids"]
        assert pending_admin[
            "safe_share_external_launch_evidence_gap_statuses"
        ]["browser_smoke"] == "pending_operator_evidence"
        assert pending_admin["operator_validation"][
            "safe_share_external_launch_evidence_gap_gate_ids"
        ] == pending_admin["safe_share_external_launch_evidence_gap_gate_ids"]
        assert pending_admin["preflight"][
            "safe_share_external_launch_evidence_gap_action_required"
        ] is True
        assert "identity_probe_verification" in pending_admin[
            "safe_share_launch_blocking_missing_gate_ids"
        ]
        assert "browser_smoke" in pending_admin["preflight"][
            "safe_share_launch_blocking_pending_gate_ids"
        ]

        passed_contract = passed_payload["task_open_authorization_contract"]
        passed_gate = passed_contract["operator_validation_start_gate"]
        assert passed_status == 200
        assert passed_payload["ok"] is True
        assert passed_contract["ready"] is True
        assert passed_contract["server_authorizes_open"] is True
        assert passed_contract["operator_validation_start_gate_required"] is True
        assert passed_contract["operator_validation_start_gate_ready"] is True
        assert passed_contract["operator_validation_start_gate_blocks_task_open"] is False
        assert passed_gate["operator_validation_status"] == "passed"
        assert passed_gate["operator_validation_all_complete"] is True
        assert passed_gate["operator_validation_pending_gate_ids"] == []
        assert passed_admin_status == 200
        passed_admin = passed_admin_payload["admin"]
        assert passed_admin["operator_validation"]["operator_validation_status"] == "passed"
        assert passed_admin["safe_share_checklist_gate_evidence_complete"] is False
        assert "browser_smoke" in passed_admin[
            "safe_share_launch_blocking_satisfied_gate_ids"
        ]
        assert passed_admin[
            "safe_share_external_launch_evidence_gap_action_required"
        ] is True
        assert "browser_smoke" not in passed_admin[
            "safe_share_external_launch_evidence_gap_gate_ids"
        ]
        assert "identity_probe_verification" in passed_admin[
            "safe_share_launch_blocking_missing_gate_ids"
        ]
        assert "identity_probe_verification" in passed_admin[
            "safe_share_external_launch_evidence_gap_gate_ids"
        ]
        assert "identity_probe_verification" in passed_admin["preflight"][
            "safe_share_launch_blocking_missing_gate_ids"
        ]
    finally:
        store.close()


def test_open_task_route_reports_superseded_session_closure(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")

        with _running_server(store, user="alice") as base_url:
            first_status, first_payload = _json_request(
                base_url,
                "/api/tasks/task-a/open",
                method="POST",
                payload={"expected_user": "alice"},
            )
            second_status, second_payload = _json_request(
                base_url,
                "/api/tasks/task-a/open",
                method="POST",
                payload={"expected_user": "alice"},
            )
            stale_status, stale_payload = _json_request(
                base_url,
                f"/api/sessions/{first_payload['session']['session_id']}/keypoints/state",
            )
            stale_save_status, stale_save_payload = _json_request(
                base_url,
                f"/api/sessions/{first_payload['session']['session_id']}/keypoints/save",
                method="POST",
                payload={"points": [[1.0, 2.0]]},
            )
            missing_session_save_status, missing_session_save_payload = _json_request(
                base_url,
                "/api/sessions/missing-session/keypoints/save",
                method="POST",
                payload={"points": [[1.0, 2.0]]},
            )
            with _running_server(store, user="bob") as bob_base_url:
                wrong_user_save_status, wrong_user_save_payload = _json_request(
                    bob_base_url,
                    f"/api/sessions/{first_payload['session']['session_id']}/keypoints/save",
                    method="POST",
                    payload={"points": [[1.0, 2.0]]},
                )

        first_session_id = first_payload["session"]["session_id"]
        first_session = store.get_session(first_session_id)

        assert first_status == 200
        assert first_payload["ok"] is True
        assert first_payload["task_open_authorization_contract"]["schema"] == (
            "palette.web_labeling_task_open_authorization_contract.v1"
        )
        assert first_payload["task_open_authorization_contract"]["ready"] is True
        assert first_payload["task_open_authorization_contract"][
            "expected_user_guard_checked_server_side"
        ] is True
        assert first_payload["task_open_authorization_contract"]["expected_user_guard_present"] is True
        assert first_payload["task_open_authorization_contract"][
            "expected_user_matches_resolved_user"
        ] is True
        assert first_payload["task_open_authorization_contract"]["active_assignment_present"] is True
        assert first_payload["task_open_authorization_contract"][
            "task_assigned_to_resolved_user"
        ] is True
        assert first_payload["task_open_authorization_contract"]["assignment_status_active"] is True
        assert first_payload["task_open_authorization_contract"]["task_state_startable"] is True
        assert first_payload["task_open_authorization_contract"][
            "reassignment_session_safety_passed"
        ] is True
        assert first_payload["task_open_authorization_contract"]["session_created_server_side"] is True
        assert first_payload["task_open_authorization_contract"]["client_authorizes_open"] is False
        assert first_payload["task_open_authorization_contract"]["server_authorizes_open"] is True
        assert first_payload["task_open_authorization_contract"]["data_plane_write_target"] == (
            "server_owned_assigned_task_zarr_scope"
        )
        assert first_payload["task_open_authorization_contract"]["label_mutation_target_kind"] == (
            "task_scoped_training_zarr"
        )
        assert first_payload["task_open_authorization_contract"]["browser_label_write_target"] == (
            "training_zarr"
        )
        assert first_payload["task_open_authorization_contract"][
            "browser_writes_csv_or_handoff_files"
        ] is False
        assert first_payload["task_open_authorization_contract"][
            "browser_has_direct_zarr_write_authority"
        ] is False
        assert first_payload["authorization_context"]["resolved_user"] == "alice"
        assert first_payload["authorization_context"]["expected_user"] == "alice"
        assert first_payload["authorization_context"]["task_id"] == "task-a"
        assert first_payload["authorization_context"]["session_id"] == first_session_id
        assert first_payload["closed_session_count"] == 0
        assert first_payload["closed_session_ids"] == []
        assert second_status == 200
        assert second_payload["ok"] is True
        assert second_payload["session"]["superseded_session_ids"] == [first_session_id]
        assert second_payload["closed_session_count"] == 1
        assert second_payload["closed_session_ids"] == [first_session_id]
        assert second_payload["session_closure_events"][0]["event_type"] == "session_superseded"
        assert second_payload["session_closure_events"][0]["task_id"] == "task-a"
        assert second_payload["session_closure_events"][0]["recording_id"] == "rec-a"
        assert stale_status == 409
        assert stale_payload["ok"] is False
        assert stale_payload["error"] == "session_superseded"
        assert stale_payload["authorization_context"]["session_id"] == first_session_id
        assert stale_payload["authorization_context"]["session_task_id"] == "task-a"
        assert stale_payload["authorization_context"]["resolved_user"] == "alice"
        assert stale_payload["session_closure_event"]["event_type"] == "session_superseded"
        assert stale_payload["session_closure_event"]["task_id"] == "task-a"
        assert stale_payload["session_closure_event"]["recording_id"] == "rec-a"
        assert stale_save_status == 409
        assert stale_save_payload["ok"] is False
        assert stale_save_payload["error"] == "session_superseded"
        stale_save_contract = stale_save_payload["mutation_authorization_contract"]
        assert stale_save_contract["ready"] is False
        assert stale_save_contract["not_ready_reason"] == "session_superseded"
        assert stale_save_contract["current_session_result"] == "superseded"
        assert stale_save_contract["current_target_token_result"] == "not_checked"
        assert stale_save_contract["server_authorizes_mutation"] is False
        assert stale_save_payload["browser_label_write_target"] == "training_zarr"
        assert stale_save_payload["browser_writes_csv_or_handoff_files"] is False
        assert stale_save_payload["browser_has_direct_zarr_write_authority"] is False
        assert missing_session_save_status == 404
        assert missing_session_save_payload["ok"] is False
        assert missing_session_save_payload["error"] == "session_not_found"
        missing_session_contract = missing_session_save_payload[
            "mutation_authorization_contract"
        ]
        assert missing_session_contract["ready"] is False
        assert missing_session_contract["not_ready_reason"] == "session_not_found"
        assert missing_session_contract["session_lookup_result"] == "not_found"
        assert missing_session_contract["session_owned_by_resolved_user"] is False
        assert missing_session_contract["task_reloaded_server_side"] is False
        assert missing_session_contract["server_authorizes_mutation"] is False
        assert missing_session_save_payload["browser_label_write_target"] == "training_zarr"
        assert missing_session_save_payload["browser_writes_csv_or_handoff_files"] is False
        assert missing_session_save_payload["browser_has_direct_zarr_write_authority"] is False
        assert wrong_user_save_status == 403
        assert wrong_user_save_payload["ok"] is False
        assert wrong_user_save_payload["error"] == "session_user_mismatch"
        wrong_user_contract = wrong_user_save_payload["mutation_authorization_contract"]
        assert wrong_user_contract["ready"] is False
        assert wrong_user_contract["not_ready_reason"] == "session_user_mismatch"
        assert wrong_user_contract["session_lookup_result"] == "passed"
        assert wrong_user_contract["session_owned_by_resolved_user"] is False
        assert wrong_user_contract["task_reloaded_server_side"] is False
        assert wrong_user_contract["current_target_token_result"] == "not_checked"
        assert wrong_user_contract["server_authorizes_mutation"] is False
        assert wrong_user_save_payload["browser_label_write_target"] == "training_zarr"
        assert wrong_user_save_payload["browser_writes_csv_or_handoff_files"] is False
        assert wrong_user_save_payload["browser_has_direct_zarr_write_authority"] is False
        assert first_session is not None
        assert first_session["closed_at_utc"]
    finally:
        store.close()


def test_direct_task_complete_route_honors_expected_user_guard(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-b", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        other_lease = store.create_session(task_id="task-b", user="alice", ttl_seconds=600)

        with _running_server(store, user="alice") as base_url:
            mismatch_status, mismatch_payload = _json_request(
                base_url,
                "/api/tasks/task-a/complete",
                method="POST",
                payload={"expected_user": "bob"},
            )
            missing_session_status, missing_session_payload = _json_request(
                base_url,
                "/api/tasks/task-a/complete",
                method="POST",
                payload={"expected_user": "alice"},
            )
            missing_session_id_status, missing_session_id_payload = _json_request(
                base_url,
                "/api/tasks/task-a/complete",
                method="POST",
                payload={"expected_user": "alice", "session_id": "missing-session"},
            )
            wrong_session_status, wrong_session_payload = _json_request(
                base_url,
                "/api/tasks/task-a/complete",
                method="POST",
                payload={"expected_user": "alice", "session_id": other_lease.session_id},
            )
            complete_status, complete_payload = _json_request(
                base_url,
                "/api/tasks/task-a/complete",
                method="POST",
                payload={"expected_user": "alice", "session_id": lease.session_id},
            )

        assert mismatch_status == 403
        assert mismatch_payload["ok"] is False
        assert mismatch_payload["error"] == "task_complete_user_mismatch"
        mismatch_contract = mismatch_payload["task_completion_authorization_contract"]
        assert mismatch_contract["ready"] is False
        assert mismatch_contract["not_ready_reason"] == "task_complete_user_mismatch"
        assert mismatch_contract["expected_user_guard_present"] is True
        assert mismatch_contract["expected_user_matches_resolved_user"] is False
        assert mismatch_contract["server_authorizes_completion"] is False
        assert mismatch_contract["browser_writes_csv_or_handoff_files"] is False
        assert mismatch_contract["browser_has_direct_zarr_write_authority"] is False

        def assert_completion_denial_readiness(source, expected_user):
            expected_queue_url = f"/my-datasets?expected_user={expected_user}"
            assert source["authorization_context"]["expected_user"] == expected_user
            assert source["authorization_context"]["return_personal_dataset_queue_url"] == (
                expected_queue_url
            )
            assert source["expected_user_personal_dataset_queue_url"] == expected_queue_url
            assert source["dataset_queue_direct_start_policy"][
                "browser_label_write_target"
            ] == "training_zarr"
            assert source["dataset_queue_direct_start_policy"][
                "browser_writes_csv_or_handoff_files"
            ] is False
            assert source["dataset_queue_direct_start_policy"][
                "browser_has_direct_zarr_write_authority"
            ] is False
            readiness = source["personalized_launch_readiness"]
            assert readiness["schema"] == (
                "palette.web_labeling_personalized_launch_readiness.v1"
            )
            assert readiness["personalized_labeler_entry_url"] == expected_queue_url
            assert readiness["browser_label_write_target"] == "training_zarr"
            assert readiness["browser_writes_csv_or_handoff_files"] is False
            assert readiness["browser_has_direct_zarr_write_authority"] is False

        assert_completion_denial_readiness(mismatch_payload, "bob")
        assert mismatch_payload["authorization_context"]["expected_user"] == "bob"
        assert mismatch_payload["authorization_context"]["return_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=bob"
        )
        assert mismatch_payload["expected_user_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=bob"
        )
        assert mismatch_payload["dataset_queue_direct_start_policy"][
            "browser_label_write_target"
        ] == "training_zarr"
        assert mismatch_payload["dataset_queue_direct_start_policy"][
            "browser_writes_csv_or_handoff_files"
        ] is False
        assert mismatch_payload["dataset_queue_direct_start_policy"][
            "browser_has_direct_zarr_write_authority"
        ] is False
        mismatch_readiness = mismatch_payload["personalized_launch_readiness"]
        assert mismatch_readiness["schema"] == (
            "palette.web_labeling_personalized_launch_readiness.v1"
        )
        assert mismatch_readiness["personalized_labeler_entry_url"] == (
            "/my-datasets?expected_user=bob"
        )
        assert mismatch_readiness["browser_label_write_target"] == "training_zarr"
        assert mismatch_readiness["browser_writes_csv_or_handoff_files"] is False
        assert mismatch_readiness["browser_has_direct_zarr_write_authority"] is False
        assert missing_session_status == 400
        assert missing_session_payload["ok"] is False
        assert missing_session_payload["error"] == "session_required"
        missing_session_contract = missing_session_payload[
            "task_completion_authorization_contract"
        ]
        assert missing_session_contract["ready"] is False
        assert missing_session_contract["not_ready_reason"] == "session_required"
        assert missing_session_contract["current_session_present"] is False
        assert missing_session_contract["server_authorizes_completion"] is False
        assert missing_session_contract["browser_writes_csv_or_handoff_files"] is False
        assert missing_session_contract["browser_has_direct_zarr_write_authority"] is False
        assert_completion_denial_readiness(missing_session_payload, "alice")
        assert missing_session_id_status == 404
        assert missing_session_id_payload["ok"] is False
        assert missing_session_id_payload["error"] == "session_not_found"
        missing_session_id_contract = missing_session_id_payload[
            "task_completion_authorization_contract"
        ]
        assert missing_session_id_contract["ready"] is False
        assert missing_session_id_contract["not_ready_reason"] == "session_not_found"
        assert missing_session_id_contract["current_session_present"] is False
        assert missing_session_id_contract["server_authorizes_completion"] is False
        assert missing_session_id_contract["browser_writes_csv_or_handoff_files"] is False
        assert missing_session_id_contract["browser_has_direct_zarr_write_authority"] is False
        assert wrong_session_status == 403
        assert wrong_session_payload["ok"] is False
        assert wrong_session_payload["error"] == "session_task_mismatch"
        wrong_session_contract = wrong_session_payload[
            "task_completion_authorization_contract"
        ]
        assert wrong_session_contract["ready"] is False
        assert wrong_session_contract["not_ready_reason"] == "session_task_mismatch"
        assert wrong_session_contract["current_session_present"] is True
        assert wrong_session_contract["session_task_matches_requested_task"] is False
        assert wrong_session_contract["server_authorizes_completion"] is False
        assert wrong_session_contract["browser_writes_csv_or_handoff_files"] is False
        assert wrong_session_contract["browser_has_direct_zarr_write_authority"] is False
        assert complete_status == 200
        assert complete_payload["ok"] is True
        assert complete_payload["task"]["state"] == "complete"
        completion_contract = complete_payload["task_completion_authorization_contract"]
        assert completion_contract["ready"] is True
        assert completion_contract["expected_user_matches_resolved_user"] is True
        assert completion_contract["active_assignment_present"] is True
        assert completion_contract["current_session_present"] is True
        assert completion_contract["session_task_matches_requested_task"] is True
        assert completion_contract["task_completion_state_mutation_target"] == (
            "labeling_task_store"
        )
        assert completion_contract["client_authorizes_completion"] is False
        assert completion_contract["server_authorizes_completion"] is True
        assert completion_contract["browser_writes_csv_or_handoff_files"] is False
        assert completion_contract["browser_has_direct_zarr_write_authority"] is False
        assert complete_payload["closed_session_count"] == 1
        assert complete_payload["closed_session_ids"] == [lease.session_id]
        assert complete_payload["session_closure_events"][0]["event_type"] == "session_closed_by_task_completion"
        assert complete_payload["session_closure_events"][0]["task_id"] == "task-a"
        assert complete_payload["session_closure_events"][0]["recording_id"] == "rec-a"
        post_queue = complete_payload["post_completion_queue"]
        assert post_queue["schema"] == "palette.web_labeling_post_completion_queue.v1"
        assert post_queue["resolved_user"] == "alice"
        assert post_queue["expected_user"] == "alice"
        assert post_queue["expected_user_guard_checked_server_side"] is True
        assert post_queue["expected_user_matches_resolved_user"] is True
        assert post_queue["next_labeler_action"] == "open_dataset_queue"
        assert post_queue["next_labeler_url"] == "/my-datasets?expected_user=alice"
        assert post_queue["next_labeler_url_role"] == "preferred_queue"
        assert post_queue["return_expected_user"] == "alice"
        assert post_queue["return_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert post_queue["return_personal_dataset_queue_expected_user_guarded"] is True
        assert post_queue["return_personal_work_url"] == "/my-work?expected_user=alice"
        assert post_queue["return_personal_work_expected_user_guarded"] is True
        assert post_queue["labeler_work_completion"]["status"] == "waiting"
        assert post_queue["labeler_work_completion"]["has_waiting_work"] is True
        assert post_queue["labeler_work_completion"]["ready_for_more_labeling"] is True
        assert post_queue["browser_label_write_target"] == "training_zarr"
        assert post_queue["browser_writes_csv_or_handoff_files"] is False
        assert post_queue["browser_writes_handoff_csv"] is False
        assert post_queue["browser_writes_intermediate_csv"] is False
        assert post_queue["browser_has_direct_zarr_write_authority"] is False
        assert complete_payload["post_completion_next_labeler_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert complete_payload["post_completion_return_expected_user"] == "alice"
        assert complete_payload["post_completion_return_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert (
            complete_payload["post_completion_return_personal_dataset_queue_expected_user_guarded"]
            is True
        )
        assert complete_payload["post_completion_return_personal_work_url"] == (
            "/my-work?expected_user=alice"
        )
        assert complete_payload["post_completion_return_personal_work_expected_user_guarded"] is True
        assert complete_payload["labeler_work_completion_status"] == "waiting"
    finally:
        store.close()


def test_labeler_promotion_retry_route_honors_expected_user_guard(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-complete", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_analysis")
        store.upsert_task(task_id="task-other", recording_id="rec-a", workflow_kind="detect_analysis")
        store.upsert_task(
            task_id="task-complete",
            recording_id="rec-complete",
            workflow_kind="detect_analysis",
            state="complete",
        )
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        wrong_lease = store.create_session(task_id="task-other", user="alice", ttl_seconds=600)
        event = store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="promotion_failed",
            target={"target_zarr": "/tmp/fake-analysis.zarr"},
            after={"error": "promotion target missing"},
        )
        completed_event = store.record_event(
            task_id="task-complete",
            recording_id="rec-complete",
            user="alice",
            event_type="promotion_failed",
            target={"target_zarr": "/tmp/fake-analysis.zarr"},
            after={"error": "promotion target missing"},
        )
        succeeded_event = store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="promotion_failed",
            target={"target_zarr": "/tmp/fake-analysis.zarr"},
            after={"error": "promotion target missing"},
        )
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="promotion_success",
            target={
                "retry_of_event_id": succeeded_event["event_id"],
                "analysis_zarr": "/tmp/fake-analysis.zarr",
                "training_zarr": "/tmp/fake-training.zarr",
            },
            after={"training_zarr_path": "/tmp/fake-training.zarr", "rows": 1},
        )

        with _running_server(store, user="alice") as base_url:
            mismatch_status, mismatch_payload = _json_request(
                base_url,
                f"/api/events/{event['event_id']}/retry-promotion",
                method="POST",
                payload={"expected_user": "bob"},
            )
            complete_status, complete_payload = _json_request(
                base_url,
                f"/api/events/{completed_event['event_id']}/retry-promotion",
                method="POST",
                payload={"expected_user": "alice"},
            )
            missing_session_status, missing_session_payload = _json_request(
                base_url,
                f"/api/events/{event['event_id']}/retry-promotion",
                method="POST",
                payload={"expected_user": "alice"},
            )
            wrong_session_status, wrong_session_payload = _json_request(
                base_url,
                f"/api/events/{event['event_id']}/retry-promotion",
                method="POST",
                payload={"expected_user": "alice", "session_id": wrong_lease.session_id},
            )
            already_status, already_payload = _json_request(
                base_url,
                f"/api/events/{succeeded_event['event_id']}/retry-promotion",
                method="POST",
                payload={"expected_user": "alice", "session_id": lease.session_id},
            )

        assert mismatch_status == 403
        assert mismatch_payload["ok"] is False
        assert mismatch_payload["error"] == "promotion_retry_user_mismatch"

        def assert_promotion_retry_denial_readiness(source, expected_user):
            expected_queue_url = f"/my-datasets?expected_user={expected_user}"
            assert source["labeler_failed_promotion_retry_action"] == "operator_support_only"
            assert source["promotion_retry_attempted"] is False
            assert source["promotion_retry_claimed"] is False
            assert source["browser_label_write_target"] == "training_zarr"
            assert source["browser_writes_csv_or_handoff_files"] is False
            assert source["browser_writes_handoff_csv"] is False
            assert source["browser_writes_intermediate_csv"] is False
            assert source["browser_has_direct_zarr_write_authority"] is False
            assert source["authorization_context"]["expected_user"] == expected_user
            assert source["return_personal_dataset_queue_url"] == expected_queue_url
            assert source["expected_user_personal_dataset_queue_url"] == expected_queue_url
            assert source["dataset_queue_direct_start_policy"][
                "browser_label_write_target"
            ] == "training_zarr"
            assert source["dataset_queue_direct_start_policy"][
                "browser_writes_csv_or_handoff_files"
            ] is False
            assert source["dataset_queue_direct_start_policy"][
                "browser_has_direct_zarr_write_authority"
            ] is False
            readiness = source["personalized_launch_readiness"]
            assert readiness["schema"] == (
                "palette.web_labeling_personalized_launch_readiness.v1"
            )
            assert readiness["personalized_labeler_entry_url"] == expected_queue_url
            assert readiness["browser_label_write_target"] == "training_zarr"
            assert readiness["browser_writes_csv_or_handoff_files"] is False
            assert readiness["browser_has_direct_zarr_write_authority"] is False

        assert_promotion_retry_denial_readiness(mismatch_payload, "bob")
        assert complete_status == 409
        assert complete_payload["ok"] is False
        assert complete_payload["error"] == "task_complete"
        assert "reopen" in complete_payload["details"]
        assert_promotion_retry_denial_readiness(complete_payload, "alice")
        assert missing_session_status == 400
        assert missing_session_payload["ok"] is False
        assert missing_session_payload["error"] == "session_required"
        assert_promotion_retry_denial_readiness(missing_session_payload, "alice")
        assert wrong_session_status == 403
        assert wrong_session_payload["ok"] is False
        assert wrong_session_payload["error"] == "session_task_mismatch"
        assert_promotion_retry_denial_readiness(wrong_session_payload, "alice")
        assert already_status == 200
        assert already_payload["ok"] is True
        assert already_payload["promotion"]["status"] == "already_succeeded"
        assert "analysis_zarr" not in already_payload["promotion"]["event"]["target"]
        assert "training_zarr" not in already_payload["promotion"]["event"]["target"]
        assert "training_zarr_path" not in already_payload["promotion"]["event"]["after"]
        assert "/tmp/fake-analysis.zarr" not in json.dumps(already_payload)
        assert "/tmp/fake-training.zarr" not in json.dumps(already_payload)
    finally:
        store.close()


def test_labeler_failed_promotion_summary_redacts_path_like_error_values(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_analysis")
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="promotion_failed",
            target={
                "source_frame_index": 12,
                "analysis_zarr": "/tmp/fake-analysis.zarr",
                "safe_note": "visible support detail",
            },
            after={
                "error": "promotion target missing at /tmp/fake-analysis.zarr",
                "details": "training output fake-training.zarr was unavailable",
                "source_frame_index": 12,
            },
        )

        with _running_server(store, user="alice") as base_url:
            status, payload = _json_request(base_url, "/api/me/tasks")
            page_status, page_html = _text_request(base_url, "/work")

        event = payload["work"]["failed_promotions"][0]

        assert status == 200
        assert payload["ok"] is True
        assert event["target"]["source_frame_index"] == 12
        assert event["target"]["safe_note"] == "visible support detail"
        assert "analysis_zarr" not in event["target"]
        assert "[redacted_path]" in event["after"]["error"]
        assert "[redacted_zarr_path]" in event["after"]["details"]
        assert "/tmp/fake-analysis.zarr" not in json.dumps(payload)
        assert "fake-training.zarr" not in json.dumps(payload)
        assert page_status == 200
        assert "/tmp/fake-analysis.zarr" not in page_html
        assert "fake-training.zarr" not in page_html
    finally:
        store.close()


def test_labeler_runtime_redaction_removes_zarr_and_path_fields():
    payload = {
        "ok": True,
        "zarr_path": "/tmp/analysis.zarr",
        "nested": {
            "training_zarr": "/tmp/training.zarr",
            "source_path": "/tmp/video.mp4",
            "refined_group_path": "refined_detect_runs/run-a",
            "target_zarr": "/tmp/target.zarr",
            "kept": "value",
        },
        "items": [
            {
                "analysis_zarr_path": "/tmp/analysis.zarr",
                "frame_idx": 3,
                "media_url": "/api/sessions/session-a/detect-analysis/media/source",
                "message": "Loaded /tmp/analysis.zarr and relative-training.zarr successfully.",
            }
        ],
    }

    redacted = labeling_web._redact_labeler_runtime_payload(payload)

    assert redacted == {
        "ok": True,
        "nested": {"kept": "value"},
        "items": [
            {
                "frame_idx": 3,
                "media_url": "/api/sessions/session-a/detect-analysis/media/source",
                "message": "Loaded [redacted_path] and [redacted_zarr_path] successfully.",
            }
        ],
    }
    details = labeling_web._labeler_safe_error_details(
        "Could not open /tmp/analysis.zarr or /tmp/source/video.mp4 for reading."
    )
    assert details is not None
    assert "/tmp/analysis.zarr" not in details
    assert "/tmp/source/video.mp4" not in details
    assert "[browser_path_redacted]" in details


def test_session_complete_route_closes_session_and_marks_task_complete(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="keypoints")
        wrong_owner_lease = store.create_session(task_id="task-b", user="bob", ttl_seconds=600)
        store.assign_recording(recording_id="rec-stale", assignee_user="alice")
        store.upsert_task(task_id="task-stale", recording_id="rec-stale", workflow_kind="keypoints")
        stale_assignment_lease = store.create_session(
            task_id="task-stale",
            user="alice",
            ttl_seconds=600,
        )
        store.assign_recording(
            recording_id="rec-stale",
            assignee_user="bob",
            allow_stale_open_sessions=True,
        )

        with _running_server(store, user="alice") as base_url:
            missing_status, missing_payload = _json_request(
                base_url,
                "/api/sessions/missing-session/complete",
                method="POST",
                payload={},
            )
            wrong_owner_status, wrong_owner_payload = _json_request(
                base_url,
                f"/api/sessions/{wrong_owner_lease.session_id}/complete",
                method="POST",
                payload={},
            )
            stale_assignment_status, stale_assignment_payload = _json_request(
                base_url,
                f"/api/sessions/{stale_assignment_lease.session_id}/complete",
                method="POST",
                payload={},
            )
            status, payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/complete",
                method="POST",
                payload={},
            )

        closed_session = store.get_session(lease.session_id)
        completed_events = store.list_events(task_id="task-a", event_type="task_completed")
        close_events = store.list_events(task_id="task-a", event_type="session_closed_by_task_completion")

        assert missing_status == 404
        assert missing_payload["ok"] is False
        assert missing_payload["error"] == "session_not_found"
        missing_contract = missing_payload["task_completion_authorization_contract"]
        assert missing_contract["ready"] is False
        assert missing_contract["not_ready_reason"] == "session_not_found"
        assert missing_contract["current_session_present"] is False
        assert missing_contract["server_authorizes_completion"] is False
        assert missing_contract["browser_writes_csv_or_handoff_files"] is False
        assert missing_contract["browser_has_direct_zarr_write_authority"] is False

        def assert_session_completion_denial_readiness(source):
            assert source["authorization_context"]["expected_user"] == "alice"
            assert source["authorization_context"]["return_personal_dataset_queue_url"] == (
                "/my-datasets?expected_user=alice"
            )
            assert source["expected_user_personal_dataset_queue_url"] == (
                "/my-datasets?expected_user=alice"
            )
            assert source["dataset_queue_direct_start_policy"][
                "browser_label_write_target"
            ] == "training_zarr"
            assert source["dataset_queue_direct_start_policy"][
                "browser_writes_csv_or_handoff_files"
            ] is False
            assert source["dataset_queue_direct_start_policy"][
                "browser_has_direct_zarr_write_authority"
            ] is False
            readiness = source["personalized_launch_readiness"]
            assert readiness["schema"] == (
                "palette.web_labeling_personalized_launch_readiness.v1"
            )
            assert readiness["personalized_labeler_entry_url"] == (
                "/my-datasets?expected_user=alice"
            )
            assert readiness["browser_label_write_target"] == "training_zarr"
            assert readiness["browser_writes_csv_or_handoff_files"] is False
            assert readiness["browser_has_direct_zarr_write_authority"] is False

        assert_session_completion_denial_readiness(missing_payload)
        assert wrong_owner_status == 403
        assert wrong_owner_payload["ok"] is False
        assert wrong_owner_payload["error"] == "session_user_mismatch"
        wrong_owner_contract = wrong_owner_payload["task_completion_authorization_contract"]
        assert wrong_owner_contract["ready"] is False
        assert wrong_owner_contract["not_ready_reason"] == "session_user_mismatch"
        assert wrong_owner_contract["current_session_present"] is True
        assert wrong_owner_contract["session_owned_by_resolved_user"] is False
        assert wrong_owner_contract["server_authorizes_completion"] is False
        assert wrong_owner_contract["browser_writes_csv_or_handoff_files"] is False
        assert wrong_owner_contract["browser_has_direct_zarr_write_authority"] is False
        assert_session_completion_denial_readiness(wrong_owner_payload)
        assert stale_assignment_status == 403
        assert stale_assignment_payload["ok"] is False
        assert stale_assignment_payload["error"] == "not_assigned"
        stale_assignment_contract = stale_assignment_payload[
            "task_completion_authorization_contract"
        ]
        assert stale_assignment_contract["ready"] is False
        assert stale_assignment_contract["not_ready_reason"] == "not_assigned"
        assert stale_assignment_contract["server_authorizes_completion"] is False
        assert stale_assignment_contract["browser_writes_csv_or_handoff_files"] is False
        assert stale_assignment_contract["browser_has_direct_zarr_write_authority"] is False
        assert_session_completion_denial_readiness(stale_assignment_payload)
        assert status == 200
        assert payload["ok"] is True
        assert payload["task"]["state"] == "complete"
        contract = payload["task_completion_authorization_contract"]
        assert contract["schema"] == "palette.web_labeling_task_completion_authorization_contract.v1"
        assert contract["ready"] is True
        assert contract["expected_user_matches_resolved_user"] is True
        assert contract["active_assignment_present"] is True
        assert contract["current_session_present"] is True
        assert contract["session_owned_by_resolved_user"] is True
        assert contract["session_task_matches_requested_task"] is True
        assert contract["task_completion_state_mutation_target"] == "labeling_task_store"
        assert contract["client_authorizes_completion"] is False
        assert contract["server_authorizes_completion"] is True
        assert contract["browser_writes_csv_or_handoff_files"] is False
        assert contract["browser_writes_handoff_csv"] is False
        assert contract["browser_writes_intermediate_csv"] is False
        assert contract["browser_has_direct_zarr_write_authority"] is False
        assert payload["closed_session_count"] == 1
        assert payload["closed_session_ids"] == [lease.session_id]
        assert payload["session_closure_events"][0]["event_type"] == "session_closed_by_task_completion"
        assert payload["session_closure_events"][0]["task_id"] == "task-a"
        assert payload["session_closure_events"][0]["recording_id"] == "rec-a"
        post_queue = payload["post_completion_queue"]
        assert post_queue["schema"] == "palette.web_labeling_post_completion_queue.v1"
        assert post_queue["resolved_user"] == "alice"
        assert post_queue["expected_user"] == "alice"
        assert post_queue["expected_user_guard_checked_server_side"] is True
        assert post_queue["expected_user_matches_resolved_user"] is True
        assert post_queue["next_labeler_action"] == "complete"
        assert post_queue["next_labeler_url"] == "/my-datasets?expected_user=alice"
        assert post_queue["return_expected_user"] == "alice"
        assert post_queue["return_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert post_queue["return_personal_dataset_queue_expected_user_guarded"] is True
        assert post_queue["return_personal_work_url"] == "/my-work?expected_user=alice"
        assert post_queue["return_personal_work_expected_user_guarded"] is True
        assert post_queue["labeler_work_completion"]["status"] == "complete"
        assert post_queue["labeler_work_completion"]["completed"] is True
        assert post_queue["labeler_work_completion"]["has_waiting_work"] is False
        assert post_queue["labeler_work_completion"]["ready_for_more_labeling"] is False
        assert post_queue["dataset_queue_state"]["code"] == "all_assigned_work_complete"
        assert post_queue["browser_label_write_target"] == "training_zarr"
        assert post_queue["browser_writes_csv_or_handoff_files"] is False
        assert post_queue["browser_has_direct_zarr_write_authority"] is False
        assert payload["post_completion_next_labeler_action"] == "complete"
        assert payload["post_completion_next_labeler_url"] == "/my-datasets?expected_user=alice"
        assert payload["post_completion_return_expected_user"] == "alice"
        assert payload["post_completion_return_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert payload["post_completion_return_personal_dataset_queue_expected_user_guarded"] is True
        assert payload["post_completion_return_personal_work_url"] == (
            "/my-work?expected_user=alice"
        )
        assert payload["post_completion_return_personal_work_expected_user_guarded"] is True
        assert payload["labeler_work_completion_status"] == "complete"
        assert payload["labeler_work_completion_completed"] is True
        assert closed_session is not None
        assert closed_session["closed_at_utc"]
        assert len(completed_events) == 1
        assert completed_events[0]["user"] == "alice"
        assert len(close_events) == 1
    finally:
        store.close()


def test_stale_session_api_errors_include_closure_event_support(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        store.assign_recording_with_session_closure(
            recording_id="rec-a",
            assignee_user="bob",
            assigned_by="operator",
        )

        with _running_server(store, user="alice") as base_url:
            status, payload = _json_request(base_url, f"/api/sessions/{lease.session_id}/keypoints/state")

        assert status == 403
        assert payload["ok"] is False
        assert payload["error"] == "not_assigned"
        assert payload["authorization_context"]["resolved_user"] == "alice"
        assert payload["authorization_context"]["session_task_id"] == "task-a"
        assert payload["authorization_context"]["assignee_user"] == "bob"
        assert payload["session_closure_event"]["event_type"] == "session_closed_by_assignment_change"
        assert payload["session_closure_event"]["task_id"] == "task-a"
        assert payload["session_closure_event"]["recording_id"] == "rec-a"
    finally:
        store.close()


def test_completed_session_api_errors_include_closure_event_support(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        store.update_task_state(task_id="task-a", state="complete", user="operator")

        with _running_server(store, user="alice") as base_url:
            status, payload = _json_request(base_url, f"/api/sessions/{lease.session_id}/keypoints/state")
            save_status, save_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/save",
                method="POST",
                payload={"points": [[1.0, 2.0]], "target_token": "stale-token"},
            )

        assert status == 409
        assert payload["ok"] is False
        assert payload["error"] == "task_complete"
        assert payload["authorization_context"]["resolved_user"] == "alice"
        assert payload["authorization_context"]["session_task_id"] == "task-a"
        assert payload["authorization_context"]["task_state"] == "complete"
        assert payload["session_closure_event"]["event_type"] == "session_closed_by_task_completion"
        assert payload["session_closure_event"]["task_id"] == "task-a"
        assert payload["session_closure_event"]["recording_id"] == "rec-a"
        assert save_status == 409
        assert save_payload["ok"] is False
        assert save_payload["error"] == "task_complete"
        assert save_payload["authorization_context"]["resolved_user"] == "alice"
        assert save_payload["authorization_context"]["task_state"] == "complete"
        save_contract = save_payload["mutation_authorization_contract"]
        assert save_contract["schema"] == (
            "palette.web_labeling_mutation_authorization_contract.v1"
        )
        assert save_contract["ready"] is False
        assert save_contract["not_ready_reason"] == "task_complete"
        assert save_contract["task_reloaded_server_side"] is True
        assert save_contract["task_assigned_to_resolved_user"] is True
        assert save_contract["assignment_status_active"] is True
        assert save_contract["task_open_for_mutation"] is False
        assert save_contract["current_target_token_result"] == "not_checked"
        assert save_contract["server_authorizes_mutation"] is False
        assert save_payload["browser_label_write_target"] == "training_zarr"
        assert save_payload["browser_writes_csv_or_handoff_files"] is False
        assert save_payload["browser_has_direct_zarr_write_authority"] is False
        assert save_payload["session_closure_event"]["event_type"] == (
            "session_closed_by_task_completion"
        )
    finally:
        store.close()


def test_expired_open_session_reports_expired_before_superseded(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        expired_lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        expired_at = "2000-01-01T00:00:00+00:00"
        store.conn.execute(
            "UPDATE labeling_sessions SET expires_at_utc = ? WHERE session_id = ?;",
            (expired_at, expired_lease.session_id),
        )
        store.conn.commit()
        current_lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        with _running_server(store, user="alice") as base_url:
            status, payload = _json_request(base_url, f"/api/sessions/{expired_lease.session_id}/keypoints/state")

        expired_session = store.get_session(expired_lease.session_id)

        assert current_lease.session_id != expired_lease.session_id
        assert status == 409
        assert payload["ok"] is False
        assert payload["error"] == "session_expired"
        assert payload["session_expires_at_utc"] == expired_at
        assert payload["authorization_context"]["resolved_user"] == "alice"
        assert payload["authorization_context"]["session_id"] == expired_lease.session_id
        assert payload["authorization_context"]["session_expires_at_utc"] == expired_at
        assert expired_session is not None
        assert not expired_session["closed_at_utc"]
    finally:
        store.close()


def test_browser_session_route_errors_are_human_readable(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        with _running_server(store, user="bob") as base_url:
            status, html = _text_request(base_url, f"/r/{lease.session_id}")

        assert status == 403
        assert "Palette labeling access problem" in html
        assert "session_user_mismatch" in html
        assert "What to send the operator" in html
        assert "Copy support details" in html
        assert "error=session_user_mismatch" in html
        assert "return_expected_user=alice" in html
        assert "return_personal_dataset_queue_url=/my-datasets?expected_user=alice" in html
        assert "return_personal_dataset_queue_expected_user_guarded=True" in html
        assert "return_personal_work_url=/my-work?expected_user=alice" in html
        assert "return_personal_work_expected_user_guarded=True" in html
        assert "Return to your personalized dataset queue" in html
        assert 'href="/my-datasets?expected_user=alice"' in html
        assert "Return to your personalized work dashboard" in html
        assert 'href="/my-work?expected_user=alice"' in html
    finally:
        store.close()


def test_signed_link_browser_errors_are_human_readable(tmp_path, monkeypatch):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-complete", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.upsert_task(
            task_id="task-complete",
            recording_id="rec-complete",
            workflow_kind="keypoints",
            state="complete",
        )
        token = labeling_web._signed_task_link_token(task_id="task-a", secret="test-secret", ttl_seconds=600)
        bound_token = labeling_web._signed_task_link_token(
            task_id="task-a",
            secret="test-secret",
            ttl_seconds=600,
            expected_user="alice",
        )
        complete_token = labeling_web._signed_task_link_token(
            task_id="task-complete",
            secret="test-secret",
            ttl_seconds=600,
            expected_user="alice",
        )

        with _running_server(store, user="alice") as base_url:
            missing_token_status, missing_token_html = _text_request(base_url, "/t/")
            disabled_status, disabled_html = _text_request(base_url, f"/t/{bound_token}")
        with _running_server(store, user="bob", link_secret="test-secret") as base_url:
            status, html = _text_request(base_url, f"/t/{token}")
            bound_status, bound_html = _text_request(base_url, f"/t/{bound_token}")
        with _running_server(store, user="alice", link_secret="test-secret") as base_url:
            complete_status, complete_html = _text_request(base_url, f"/t/{complete_token}")
            invalid_status, invalid_html = _text_request(base_url, "/t/not-a-valid-token")
        with _running_server(
            store,
            user="alice",
            link_secret="test-secret",
            link_not_before_utc="2999-01-01T00:00:00+00:00",
        ) as base_url:
            revoked_status, revoked_html = _text_request(base_url, f"/t/{bound_token}")
        with monkeypatch.context() as patch_context:
            def fail_create_session(*args, **kwargs):
                raise PermissionError(
                    "Stale previous-owner sessions are still open for this recording."
                )

            patch_context.setattr(store, "create_session", fail_create_session)
            with _running_server(store, user="alice", link_secret="test-secret") as base_url:
                session_failure_status, session_failure_html = _text_request(
                    base_url,
                    f"/t/{bound_token}",
                )
        store.update_task_state(task_id="task-complete", state="pending", user="operator")
        with _running_server(store, user="alice", link_secret="test-secret") as base_url:
            reopened_status, reopened_html = _text_request(base_url, f"/t/{complete_token}")

        assert missing_token_status == 404
        assert "Palette labeling access problem" in missing_token_html
        assert "missing_token" in missing_token_html
        assert "signed_link_policy" in missing_token_html
        assert "signed_link_contract" in missing_token_html
        assert "&quot;signed_links_are_entry_hints_not_authorization&quot;: true" in missing_token_html
        assert "browser_mutation_write_policy" in missing_token_html
        assert "browser_mutation_write_contract" in missing_token_html
        assert "&quot;browser_label_write_target&quot;: &quot;training_zarr&quot;" in missing_token_html
        assert "&quot;browser_has_direct_zarr_write_authority&quot;: false" in missing_token_html
        assert disabled_status == 404
        assert "Palette labeling access problem" in disabled_html
        assert "signed_links_disabled" in disabled_html
        assert "signed_links_enabled=False" in disabled_html
        assert "signed_link_policy" in disabled_html
        assert "signed_link_contract" in disabled_html
        assert "&quot;signed_links_are_entry_hints_not_authorization&quot;: true" in disabled_html
        assert "browser_mutation_write_policy" in disabled_html
        assert "browser_mutation_write_contract" in disabled_html
        assert "&quot;browser_label_write_target&quot;: &quot;training_zarr&quot;" in disabled_html
        assert "&quot;browser_has_direct_zarr_write_authority&quot;: false" in disabled_html
        assert status == 403
        assert "Palette labeling access problem" in html
        assert "not_assigned" in html
        assert "What to send the operator" in html
        assert "Copy support details" in html
        assert "error=not_assigned" in html
        assert "return_expected_user=alice" in html
        assert "return_personal_dataset_queue_url=/my-datasets?expected_user=alice" in html
        assert "return_personal_dataset_queue_expected_user_guarded=True" in html
        assert "return_personal_work_url=/my-work?expected_user=alice" in html
        assert "return_personal_work_expected_user_guarded=True" in html
        assert "authorization_context=" in html
        assert "&quot;task_id&quot;: &quot;task-a&quot;" in html
        assert "task_open_authorization_contract" in html
        assert "&quot;not_ready_reason&quot;: &quot;not_assigned&quot;" in html
        assert "&quot;server_authorizes_open&quot;: false" in html
        assert "Return to your personalized dataset queue" in html
        assert 'href="/my-datasets?expected_user=alice"' in html
        assert "Return to your personalized work dashboard" in html
        assert 'href="/my-work?expected_user=alice"' in html
        assert bound_status == 403
        assert "Palette labeling access problem" in bound_html
        assert "signed_link_user_mismatch" in bound_html
        assert "error=signed_link_user_mismatch" in bound_html
        assert "return_expected_user=alice" in bound_html
        assert "return_personal_dataset_queue_url=/my-datasets?expected_user=alice" in bound_html
        assert "return_personal_dataset_queue_expected_user_guarded=True" in bound_html
        assert "return_personal_work_url=/my-work?expected_user=alice" in bound_html
        assert "return_personal_work_expected_user_guarded=True" in bound_html
        assert "task_open_authorization_contract" in bound_html
        assert "&quot;not_ready_reason&quot;: &quot;signed_link_user_mismatch&quot;" in bound_html
        assert "&quot;server_authorizes_open&quot;: false" in bound_html
        assert "personalized_launch_readiness" in bound_html
        assert "palette.web_labeling_personalized_launch_readiness.v1" in bound_html
        assert "&quot;personalized_labeler_entry_url&quot;: &quot;/my-datasets?expected_user=alice&quot;" in bound_html
        assert "&quot;browser_label_write_target&quot;: &quot;training_zarr&quot;" in bound_html
        assert "&quot;browser_writes_csv_or_handoff_files&quot;: false" in bound_html
        assert "&quot;browser_has_direct_zarr_write_authority&quot;: false" in bound_html
        assert 'href="/my-datasets?expected_user=alice"' in bound_html
        assert 'href="/my-work?expected_user=alice"' in bound_html
        assert complete_status == 409
        assert "Palette labeling access problem" in complete_html
        assert "task_complete" in complete_html
        assert "reopened by an operator" in complete_html
        assert "authorization_context=" in complete_html
        assert "task_open_authorization_contract" in complete_html
        assert "&quot;not_ready_reason&quot;: &quot;task_complete&quot;" in complete_html
        assert "&quot;server_authorizes_open&quot;: false" in complete_html
        assert "&quot;task_state&quot;: &quot;complete&quot;" in complete_html
        assert "return_expected_user=alice" in complete_html
        assert "return_personal_dataset_queue_url=/my-datasets?expected_user=alice" in complete_html
        assert "return_personal_dataset_queue_expected_user_guarded=True" in complete_html
        assert "return_personal_work_url=/my-work?expected_user=alice" in complete_html
        assert "return_personal_work_expected_user_guarded=True" in complete_html
        assert 'href="/my-datasets?expected_user=alice"' in complete_html
        assert 'href="/my-work?expected_user=alice"' in complete_html
        assert invalid_status == 400
        assert "Palette labeling access problem" in invalid_html
        assert "signed_link_failed" in invalid_html
        assert "signed_link_policy" in invalid_html
        assert "signed_link_contract" in invalid_html
        assert "&quot;authorization_grant&quot;: false" in invalid_html
        assert "&quot;signed_links_are_entry_hints_not_authorization&quot;: true" in invalid_html
        assert "browser_mutation_write_policy" in invalid_html
        assert "browser_mutation_write_contract" in invalid_html
        assert "&quot;browser_label_write_target&quot;: &quot;training_zarr&quot;" in invalid_html
        assert "&quot;browser_has_direct_zarr_write_authority&quot;: false" in invalid_html
        assert revoked_status == 403
        assert "Palette labeling access problem" in revoked_html
        assert "signed_link_revoked" in revoked_html
        assert "task_open_authorization_contract" in revoked_html
        assert "&quot;not_ready_reason&quot;: &quot;signed_link_revoked&quot;" in revoked_html
        assert "&quot;server_authorizes_open&quot;: false" in revoked_html
        assert "return_expected_user=alice" in revoked_html
        assert session_failure_status == 409
        assert "Palette labeling access problem" in session_failure_html
        assert "reassignment_session_safety_failed" in session_failure_html
        assert "task_open_authorization_contract" in session_failure_html
        assert "&quot;not_ready_reason&quot;: &quot;reassignment_session_safety_failed&quot;" in session_failure_html
        assert "&quot;server_authorizes_open&quot;: false" in session_failure_html
        assert "&quot;reassignment_session_safety_passed&quot;: false" in session_failure_html
        assert reopened_status == 200
        assert "Palette labeling access problem" not in reopened_html
        assert "Palette labeling access problem" not in reopened_html
        assert "task-complete" in reopened_html
        assert "rec-complete" in reopened_html
    finally:
        store.close()


def test_admin_browser_route_errors_are_human_readable(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        with _running_server(store, user="alice") as base_url:
            status, html = _text_request(base_url, "/admin")

        assert status == 403
        assert "Palette labeling access problem" in html
        assert "admin_required" in html
        assert "What to send the operator" in html
        assert "Copy support details" in html
    finally:
        store.close()


def test_admin_assignment_route_updates_owner_and_closes_old_sessions(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_training")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        with _running_server(store, user="admin", admin_users=("admin",)) as base_url:
            status, payload = _json_request(
                base_url,
                "/api/admin/assignments",
                method="POST",
                payload={
                    "recording_id": "rec-a",
                    "assignee_user": "bob",
                    "status": "active",
                    "notes": "Please finish detection review.",
                },
            )

        closed_session = store.get_session(lease.session_id)
        assignment = store.get_assignment("rec-a")

        assert status == 200
        assert payload["ok"] is True
        assert payload["assignment"]["assignee_user"] == "bob"
        assert payload["previous_assignment"]["assignee_user"] == "alice"
        expected_transition = {
            "recording_id": "rec-a",
            "previous_assignee_user": "alice",
            "previous_status": "active",
            "new_assignee_user": "bob",
            "new_status": "active",
            "owner_changed": True,
            "status_changed": False,
            "changed_owner_or_status": True,
        }
        for key, value in expected_transition.items():
            assert payload["assignment_transition"][key] == value
        assert payload["single_owner_policy"]["recording_id_primary_key"] is True
        assert payload["single_owner_policy"]["one_active_owner"] is True
        assert payload["single_owner_policy"]["stale_sessions_closed_on_reassignment"] is True
        assert payload["closed_session_count"] == 1
        assert payload["closed_session_ids"] == [lease.session_id]
        assert payload["session_closure_events"][0]["event_type"] == "session_closed_by_assignment_change"
        assert payload["session_closure_events"][0]["task_id"] == "task-a"
        assert payload["session_closure_events"][0]["recording_id"] == "rec-a"
        assert closed_session is not None
        assert closed_session["closed_at_utc"]
        assert assignment is not None
        assert assignment["notes"] == "Please finish detection review."
    finally:
        store.close()


def test_admin_cleanup_stale_sessions_reports_closure_events(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        store.conn.execute(
            "UPDATE labeling_sessions SET expires_at_utc = ? WHERE session_id = ?;",
            ("2000-01-01T00:00:00+00:00", lease.session_id),
        )
        store.conn.commit()

        with _running_server(store, user="admin", admin_users=("admin",)) as base_url:
            status, payload = _json_request(
                base_url,
                "/api/admin/sessions/cleanup-stale",
                method="POST",
                payload={},
            )
            closure_status, closure_payload = _json_request(
                base_url,
                f"/api/admin/sessions/{lease.session_id}/closure",
            )
        with _running_server(store, user="alice") as base_url:
            expired_status, expired_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/state",
            )

        closed_session = store.get_session(lease.session_id)

        assert status == 200
        assert payload["ok"] is True
        assert payload["closed_count"] == 1
        assert payload["closed_session_count"] == 1
        assert payload["closed_session_ids"] == [lease.session_id]
        assert payload["session_closure_events"][0]["event_type"] == "stale_session_closed"
        assert payload["session_closure_events"][0]["task_id"] == "task-a"
        assert payload["session_closure_events"][0]["recording_id"] == "rec-a"
        assert closure_status == 200
        assert closure_payload["ok"] is True
        assert closure_payload["session_id"] == lease.session_id
        assert closure_payload["session"]["session_id"] == lease.session_id
        assert closure_payload["session"]["task_id"] == "task-a"
        assert closure_payload["has_closure_event"] is True
        assert closure_payload["session_closure_event"]["event_type"] == "stale_session_closed"
        assert closure_payload["session_closure_event"]["task_id"] == "task-a"
        assert closure_payload["session_closure_event"]["recording_id"] == "rec-a"
        assert expired_status == 409
        assert expired_payload["ok"] is False
        assert expired_payload["error"] == "session_expired"
        assert expired_payload["session_closure_event"]["event_type"] == "stale_session_closed"
        assert expired_payload["session_closure_event"]["task_id"] == "task-a"
        assert expired_payload["session_closure_event"]["recording_id"] == "rec-a"
        assert closed_session is not None
        assert closed_session["closed_at_utc"]
    finally:
        store.close()


def test_unsupported_workflow_session_page_is_operator_actionable(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="unsupported_workflow")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        with _running_server(store, user="alice") as base_url:
            status, html = _text_request(base_url, f"/r/{lease.session_id}")

        assert status == 200
        assert "No browser editor is configured for this workflow" in html
        assert "ask the operator to inspect this task definition" in html
        assert 'href="/"' in html
        assert 'href="/work"' in html
        assert "next implementation phase" not in html
    finally:
        store.close()


def test_admin_summary_includes_dashboard_user_readiness(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-done", assignee_user="bob")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            scope={"zarr_path": "/secret/admin-recording.zarr"},
        )
        store.upsert_task(task_id="task-done", recording_id="rec-done", workflow_kind="detect_analysis", state="complete")
        store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_keypoints",
            target={"frame_idx": 12},
            before={"points": []},
            after={"changed": True},
        )

        with _running_server(store, user="admin", admin_users=("admin",)) as base_url:
            status, payload = _json_request(base_url, "/api/admin/summary")
            user_api_status, user_api_payload = _json_request(base_url, "/api/admin/users/alice")
            recording_api_status, recording_api_payload = _json_request(base_url, "/api/admin/recordings/rec-a")
            html_status, html = _text_request(base_url, "/admin")
            user_status, user_html = _text_request(base_url, "/admin/users/alice")
            recording_status, recording_html = _text_request(base_url, "/admin/recordings/rec-a")

        assert status == 200
        assert payload["ok"] is True
        admin = payload["admin"]
        assert admin["active_session_user_counts"] == {"alice": 1}
        assert admin["active_session_workflow_counts"] == {"keypoints": 1}
        assert admin["stale_session_user_counts"] == {}
        assert admin["stale_session_workflow_counts"] == {}
        assert admin["assignment_work_state_counts"] == {
            "active_work": 1,
            "complete": 1,
        }
        assignment_rows = {row["recording_id"]: row for row in admin["assignment_operator_rows"]}
        assert assignment_rows["rec-a"]["work_state"] == "active_work"
        assert assignment_rows["rec-a"]["open_tasks"] == 1
        assert assignment_rows["rec-a"]["complete_tasks"] == 0
        assert assignment_rows["rec-done"]["work_state"] == "complete"
        assert assignment_rows["rec-done"]["open_tasks"] == 0
        assert assignment_rows["rec-done"]["complete_tasks"] == 1
        assert admin["recent_audit_event_count"] == 2
        assert admin["recent_audit_event_user_counts"] == {"alice": 2}
        assert admin["recent_audit_event_type_counts"] == {
            "save_keypoints": 1,
            "session_opened": 1,
        }
        assert admin["recent_audit_event_workflow_counts"] == {"keypoints": 2}
        assert admin["recent_audit_events"][0]["user"] == "alice"
        assert admin["recent_audit_events"][0]["event_type"] == "save_keypoints"
        assert admin["dashboard_path"] == "/work"
        assert admin["dataset_queue_page_path"] == "/datasets"
        expected_single_owner_policy = {
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
        }
        for key, value in expected_single_owner_policy.items():
            assert admin["single_owner_policy"][key] == value
        assert admin["assignment_ownership_integrity"]["ok"] is True
        assert admin["assignment_ownership_integrity"]["active_assignment_count"] == 2
        assert admin["assignment_ownership_integrity"]["unique_active_recording_count"] == 2
        assert admin["assignment_ownership_integrity"]["duplicate_active_owner_count"] == 0
        assert admin["assignment_ownership_integrity"]["duplicate_active_owners"] == []
        assert admin["labeler_safety"]["dashboard_identity_check_required"] is True
        assert admin["labeler_safety"]["dataset_queue_page_path"] == "/datasets"
        assert admin["identity_source_policy"]["assignment_user_source"] == "fixed_user"
        assert admin["identity_source_policy"]["assignment_user_match_required"] is True
        assert admin["identity_source_policy"]["dashboard_expected_user_guard_supported"] is True
        assert admin["identity_source_policy"]["dataset_queue_page_expected_user_guard_supported"] is True
        assert admin["identity_source_policy"]["personal_work_expected_user_guard_supported"] is True
        assert admin["identity_source_policy"]["dataset_queue_expected_user_guard_supported"] is True
        assert admin["identity_source_policy"]["task_open_expected_user_guard_supported"] is True
        assert admin["identity_source_policy"]["task_complete_expected_user_guard_supported"] is True
        assert admin["identity_source_policy"]["promotion_retry_expected_user_guard_supported"] is True
        assert admin["identity_source_policy"]["promotion_retry_current_session_required"] is True
        assert admin["identity_source_policy"]["promotion_retry_dashboard_action"] == "operator_support_only"
        assert admin["identity_source_policy"]["signed_task_link_expected_user_binding_supported"] is True
        expected_user_guards = {
            "labeler_landing_page": "dashboard_user_mismatch",
            "labeler_me_page": "dashboard_user_mismatch",
            "labeling_home_page": "dashboard_user_mismatch",
            "dashboard": "dashboard_user_mismatch",
            "dataset_queue_page": "dashboard_user_mismatch",
            "personal_work_api": "dashboard_user_mismatch",
            "dataset_queue_api": "dashboard_user_mismatch",
            "task_open_api": "task_open_user_mismatch",
            "task_complete_api": "task_complete_user_mismatch",
            "promotion_retry_api": "promotion_retry_user_mismatch",
            "signed_task_link": "signed_link_user_mismatch",
        }
        for key, value in expected_user_guards.items():
            assert admin["identity_source_policy"]["expected_user_guards"][key] == value
        assert admin["identity_source_policy"]["labeler_landing_page_path"] == "/"
        assert admin["identity_source_policy"]["labeling_home_page_path"] == "/labeling"
        assert admin["identity_source_policy"]["queue_first_landing_paths"] == [
            "/",
            "/me",
            "/labeling",
            "/datasets",
            "/my-datasets",
        ]
        assert admin["identity_source_policy"]["queue_first_landing_expected_user_guard_supported"] is True
        assert admin["identity_source_policy"]["expected_user_probe_supported"] is True
        assert admin["identity_source_policy"]["identity_probe_path"] == "/identity"
        assert admin["identity_source_policy"]["identity_probe_api_path"] == "/api/me/identity"
        assert admin["identity_source_policy"]["signed_links_are_not_identity"] is True
        assert admin["identity_source_policy"]["operator_verification_required_before_invite"] is True
        assert admin["preflight"]["identity_source_policy"] == admin["identity_source_policy"]
        expected_operator_authorization_policy = {
            "admin_routes_require_operator": True,
            "admin_route_prefixes": ["/admin", "/api/admin"],
            "operator_user_source": "server_config_admin_users",
            "configuration_flag": "--admin-user",
            "admin_required_error": "admin_required",
            "resolved_user_must_be_in_admin_users": True,
            "labelers_are_not_operators_by_default": True,
            "operator_boundary_known": True,
            "runtime_preflight_required": False,
            "operator_recovery_routes": [
                "/admin",
                "/admin/users/{user}",
                "/admin/recordings/{recording_id}",
                "/admin/tasks/{task_id}",
                "/api/admin/assignments",
                "/api/admin/tasks/{task_id}/state",
                "/api/admin/tasks/{task_id}/repair",
                "/api/admin/sessions/cleanup-stale",
                "/api/admin/sessions/{session_id}/closure",
                "/api/admin/events/{event_id}/retry-promotion",
            ],
            "operator_authorization_grants_labeler_mutation": False,
            "operator_boundary_required_for_launch": True,
            "production_requires_admin_user": True,
            "admin_user_count": 1,
            "admin_users_configured": True,
            "operator_boundary_ready": True,
            "admin_users": ["admin"],
        }
        for source in (
            admin["operator_authorization_policy"],
            admin["preflight"]["operator_authorization_policy"],
        ):
            for key, value in expected_operator_authorization_policy.items():
                if key == "operator_recovery_routes":
                    for route in value:
                        assert route in source[key]
                else:
                    assert source[key] == value
        assert admin["operator_validation"]["operator_validation_gate_status_values"] == list(
            labeling_web.OPERATOR_VALIDATION_GATE_STATUS_VALUES
        )
        assert admin["operator_validation"]["operator_validation_gate_ids"] == list(
            labeling_web.DEFAULT_OPERATOR_VALIDATION_GATE_IDS
        )
        assert admin["operator_validation"]["operator_validation_gate_flat_field_suffixes"] == [
            "status",
            "pending",
            "missing_evidence",
            "needs_review",
            "passed",
        ]
        assert admin["preflight"]["operator_validation"]["operator_validation_gate_ids"] == list(
            labeling_web.DEFAULT_OPERATOR_VALIDATION_GATE_IDS
        )
        for gate_id in labeling_web.DEFAULT_OPERATOR_VALIDATION_GATE_IDS:
            assert admin["operator_validation"][
                f"operator_validation_gate_{gate_id}_status"
            ] == "missing_evidence"
            assert admin["operator_validation"][
                f"operator_validation_gate_{gate_id}_pending"
            ] is True
            assert admin["preflight"]["operator_validation"][
                f"operator_validation_gate_{gate_id}_status"
            ] == "missing_evidence"
        expected_operator_recovery_policy = {
            "schema": "palette.web_labeling_operator_recovery_policy.v1",
            "operator_only": True,
            "admin_routes_require_operator": True,
            "assignment_reassign_route": "/api/admin/assignments",
            "task_detail_route": "/api/admin/tasks/{task_id}",
            "task_state_route": "/api/admin/tasks/{task_id}/state",
            "task_repair_route": "/api/admin/tasks/{task_id}/repair",
            "session_cleanup_route": "/api/admin/sessions/cleanup-stale",
            "session_closure_route": "/api/admin/sessions/{session_id}/closure",
            "failed_promotion_retry_route": "/api/admin/events/{event_id}/retry-promotion",
            "reassignment_closes_previous_owner_sessions": True,
            "task_reopen_operator_only": True,
            "completion_closes_open_sessions": True,
            "failed_promotion_retry_operator_only": True,
            "labeler_failed_promotion_retry_action": "operator_support_only",
            "session_closure_events_operator_inspectable": True,
            "operator_repair_closes_or_supersedes_sessions": True,
            "operator_repair_records_audit_event": True,
            "rollback_requires_backup_plan": True,
            "restore_pauses_or_unassigns_recording_before_write": True,
            "labelers_receive_recovery_write_authority": False,
            "browser_recovery_mutations_direct": False,
            "validation_gate": "operator_recovery_contract",
        }
        for source in (
            admin["operator_recovery_policy"],
            admin["preflight"]["operator_recovery_policy"],
        ):
            for key, value in expected_operator_recovery_policy.items():
                assert source[key] == value
        assert admin["operator_recovery_contract"]["ready"] is True
        assert admin["operator_recovery_contract"]["task_reopen_operator_only"] is True
        assert admin["operator_recovery_contract"]["failed_promotion_retry_operator_only"] is True
        assert admin["operator_recovery_contract"]["task_repair_route"] == "/api/admin/tasks/{task_id}/repair"
        assert admin["operator_recovery_contract"]["operator_repair_records_audit_event"] is True
        assert admin["operator_recovery_contract"]["rollback_requires_backup_plan"] is True
        assert admin["preflight"]["operator_recovery_contract"] == admin["operator_recovery_contract"]
        assert admin["zarr_backup_policy"]["schema"] == "palette.web_labeling_zarr_backup_policy.v1"
        assert admin["zarr_backup_policy"]["validation_gate"] == "mutable_zarr_backup_confirmation"
        assert admin["zarr_backup_policy"]["labelers_do_not_receive_backup_paths"] is True
        assert admin["preflight"]["zarr_backup_policy"] == admin["zarr_backup_policy"]
        assert admin["mutation_audit_policy"]["event_store"] == "labeling_task_events"
        assert admin["mutation_audit_policy"]["append_only"] is True
        assert admin["mutation_audit_policy"]["server_records_events"] is True
        assert admin["preflight"]["mutation_audit_policy"] == admin["mutation_audit_policy"]
        assert admin["browser_response_security_policy"]["headers"]["Cache-Control"] == (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        assert admin["browser_response_security_policy"]["headers"]["X-Frame-Options"] == "DENY"
        assert admin["preflight"]["browser_response_security_policy"] == admin["browser_response_security_policy"]
        assert admin["session_guard_policy"]["requires_current_session"] is True
        assert admin["session_guard_policy"]["stale_tab_save_rejected"] is True
        assert admin["session_guard_policy"]["target_token_required_for_mutation"] is True
        assert admin["session_guard_policy"]["labeler_promotion_retry_requires_current_session"] is True
        assert admin["session_guard_policy"]["session_closure_event_support"] is True
        assert admin["preflight"]["session_guard_policy"] == admin["session_guard_policy"]
        assert "Operator boundary" in html
        assert "operator_authorization_policy" in html
        assert "admin_routes_require_operator" in html
        assert "operator_boundary_ready" in html
        assert "Operator recovery policy" in html
        assert "operator_recovery_contract" in html
        assert "Zarr backup policy" in html
        assert "mutable_zarr_backup_confirmation" in html
        assert "Mutation audit policy" in html
        assert "labeling_task_events" in html
        assert "Browser response security policy" in html
        assert "proxy preserves headers" in html
        assert "Session guard policy" in html
        assert "multiple labelers per recording allowed" in html
        assert "browser mutation requires current owner" in html
        assert "raw_assignment_change_blocks_open_sessions" in html
        assert admin["preflight"]["single_owner_policy"] == admin["single_owner_policy"]
        expected_task_state_policy = {
            "completed_tasks_read_only": True,
            "completed_tasks_open_new_sessions": False,
            "completed_task_open_requests": "reject_task_complete",
            "completed_task_save_requests": "reject_task_complete",
            "absolute_navigation_out_of_scope": "reject_nav_error",
            "browser_mutation_target_selectors": "server_owned_reject_client_fields",
            "browser_mutation_target_token": "required_current_target_token",
            "task_completion_requires_current_session": True,
            "labeler_promotion_retry_requires_open_task": True,
            "labeler_promotion_retry_requires_current_session": True,
            "completion_closes_open_sessions": True,
            "reopen_authority": "operator",
            "reopen_required_for_more_labeling": True,
        }
        for key, value in expected_task_state_policy.items():
            assert admin["task_state_policy"][key] == value
        assert admin["preflight"]["task_state_policy"] == admin["task_state_policy"]
        assert admin["signed_link_policy"] == {
            "canonical_entrypoint": "/work",
            "task_specific_links": "short_lived_convenience_links",
            "default_ttl_seconds": 86400,
            "authorization_grant": False,
            "requires_authenticated_user": True,
            "requires_active_assignment": True,
            "requires_open_task": True,
            "binds_expected_user_in_new_links": True,
            "expected_user_mismatch_error": "signed_link_user_mismatch",
            "opens_guarded_session": True,
            "session_bound_after_open": True,
            "dashboard_preferred_for_multi_task_work": True,
            "runtime_operator_validation_start_gate_enforced": True,
        }
        assert admin["preflight"]["signed_link_policy"] == admin["signed_link_policy"]
        browser_workflows = {row["workflow_kind"]: row for row in admin["browser_workflows"]}
        assert set(browser_workflows) == {
            "keypoints",
            "detect_training",
            "detect_analysis",
            "subject_mask_component",
        }
        expected_client_authority = {
            "mutation_executor": "server",
            "browser_can_submit_edits": True,
            "browser_can_write_zarr": False,
            "browser_can_write_filesystem": False,
            "browser_receives_write_credentials": False,
            "browser_receives_direct_zarr_handles": False,
        }
        expected_retry_policy = {
            "data_write_semantics": "replace_target_payload",
            "same_payload_retry_safe": True,
            "audit_semantics": "append_only",
            "duplicate_audit_events_possible": True,
            "client_idempotency_key_supported": False,
        }
        for workflow in browser_workflows.values():
            assert workflow["client_authority"] == expected_client_authority
            retry_policy = workflow["write_contract"]["retry_policy"]
            for key, value in expected_retry_policy.items():
                assert retry_policy[key] == value
            assert retry_policy["retry_guidance"]
            assert workflow["write_contract"]["audit_provenance"] == {
                "event_store": "labeling_task_events",
                "required_event_fields": [
                    "event_id",
                    "task_id",
                    "recording_id",
                    "user",
                    "event_type",
                    "created_at_utc",
                    "target",
                    "before",
                    "after",
                ],
                "timestamp_field": "created_at_utc",
                "identity_fields": ["task_id", "recording_id", "user"],
                "mutation_summary_fields": ["target", "before", "after"],
            }
        assert browser_workflows["keypoints"]["server_mutation"] is True
        expected_keypoint_write_contract = {
            "save_method": "POST",
            "save_endpoint": "/api/sessions/{session_id}/keypoints/save",
            "payload_fields": ["points", "advance", "target_token"],
            "required_fields": ["points", "target_token"],
            "response_fields": ["ok", "result", "state"],
            "audit_event": "save_keypoints",
            "audit_provenance": {
                "event_store": "labeling_task_events",
                "required_event_fields": [
                    "event_id",
                    "task_id",
                    "recording_id",
                    "user",
                    "event_type",
                    "created_at_utc",
                    "target",
                    "before",
                    "after",
                ],
                "timestamp_field": "created_at_utc",
                "identity_fields": ["task_id", "recording_id", "user"],
                "mutation_summary_fields": ["target", "before", "after"],
            },
            "retry_policy": {
                "data_write_semantics": "replace_target_payload",
                "same_payload_retry_safe": True,
                "audit_semantics": "append_only",
                "duplicate_audit_events_possible": True,
                "client_idempotency_key_supported": False,
                "retry_guidance": "If the browser loses the response after submitting, reopening the task and saving the same target payload again should leave the label data in the same state, but records another audit event.",
            },
            "registry_refresh": True,
            "guard": "session_for_user",
        }
        for key, value in expected_keypoint_write_contract.items():
            assert browser_workflows["keypoints"]["write_contract"][key] == value
        assert browser_workflows["detect_training"]["write_contract"]["save_endpoint"] == (
            "/api/sessions/{session_id}/detect/save"
        )
        assert browser_workflows["detect_training"]["write_contract"]["audit_event"] == "save_detect_bbox"
        assert browser_workflows["detect_analysis"]["write_scope"] == (
            "Reviewable by default; editable only when task scope enables analysis-box edits."
        )
        assert browser_workflows["detect_analysis"]["write_contract"]["save_endpoint"] == (
            "/api/sessions/{session_id}/detect-analysis/save"
        )
        assert browser_workflows["detect_analysis"]["write_contract"]["audit_event"] == "save_detect_analysis_bbox"
        assert browser_workflows["detect_analysis"]["write_contract"]["scope_required"] == {"editable": True}
        assert browser_workflows["detect_analysis"]["write_contract"]["retry_policy"]["secondary_side_effects"] == [
            "promotion_success",
            "promotion_failed",
        ]
        assert browser_workflows["subject_mask_component"]["write_contract"]["save_endpoint"] == (
            "/api/sessions/{session_id}/subject-mask/save"
        )
        assert browser_workflows["subject_mask_component"]["write_contract"]["audit_event"] == "save_subject_mask_roi"
        assert admin["preflight"]["browser_workflows"] == admin["browser_workflows"]
        assert admin["safe_share_gate"]["schema"] == "palette.web_labeling_safe_share_gate.v1"
        assert admin["safe_share_gate_id"] == "labeler_links_safe_to_share"
        assert admin["safe_share_ready_to_send_is_sufficient"] is False
        assert admin["safe_share_required_inspection_field"] == "labeler_links_safe_to_share"
        assert admin["safe_share_checklist_gate_evidence_complete"] is False
        assert admin["operator_validation"]["safe_share_checklist_gate_evidence_complete"] is False
        assert "browser_smoke" in admin[
            "safe_share_launch_blocking_missing_evidence_gate_ids"
        ]
        assert "browser_smoke" in admin["operator_validation"][
            "safe_share_launch_blocking_missing_evidence_gate_ids"
        ]
        assert "operator_recovery_contract" in admin[
            "safe_share_launch_blocking_unsatisfied_gate_ids"
        ]
        assert admin["preflight"]["safe_share_gate_id"] == "labeler_links_safe_to_share"
        assert admin["preflight"]["safe_share_ready_to_send_is_sufficient"] is False
        assert admin["preflight"]["safe_share_checklist_gate_evidence_complete"] is False
        assert "disposable_zarr_mutation_smoke" in admin["preflight"][
            "safe_share_launch_blocking_missing_evidence_gate_ids"
        ]
        assert (
            "Approve required operator validation evidence before copying ready-row draft text or sharing links."
            in admin["dashboard_invite_actions"]
        )
        assert (
            "Generate, import, or reopen browser-labeling tasks for this user's active recordings."
            in admin["dashboard_invite_actions"]
        )
        assert admin["dashboard_user_counts"]["ready_to_invite"] == 0
        assert admin["dashboard_user_counts"]["not_ready_to_invite"] == 2
        assert admin["dashboard_user_counts"]["ready_to_invite_users"] == []
        assert admin["dashboard_user_counts"]["not_ready_to_invite_users"] == ["alice", "bob"]
        assert admin["dashboard_user_counts"]["ready_to_invite_legacy_semantics"] == (
            "row_readiness_not_safe_share_approval"
        )
        assert admin["dashboard_user_counts"]["ready_row_state_values"] == [
            "ready_row_draft",
            "diagnostic_note",
        ]
        assert admin["dashboard_user_counts"]["copy_intent_values"] == [
            "ready_row_draft",
            "diagnostic_note",
        ]
        assert admin["dashboard_user_counts"]["ready_row_draft_count"] == 0
        assert admin["dashboard_user_counts"]["diagnostic_note_count"] == 2
        assert admin["dashboard_user_counts"]["ready_row_draft_users"] == []
        assert admin["dashboard_user_counts"]["diagnostic_note_users"] == ["alice", "bob"]
        assert admin["dashboard_user_counts"]["open_tasks"] == 1
        assert admin["dashboard_user_counts"]["total_tasks"] == 2
        assert admin["dashboard_user_counts"]["complete_tasks"] == 1
        assert admin["dashboard_user_counts"]["completion_percent"] == 50.0
        assert admin["dashboard_user_counts"]["completion_states"] == {
            "complete": 1,
            "not_started": 1,
        }
        assert admin["dashboard_user_counts"]["invite_reasons"] == {
            "no_open_tasks": 1,
            "operator_validation_pending": 1,
        }
        assert admin["dashboard_user_counts"]["copy_intents"] == {
            "diagnostic_note": 2,
        }
        assert admin["dashboard_user_counts"]["ready_states"] == {
            "not_ready_to_invite": 2,
        }
        assert admin["dashboard_user_counts"]["identity_probe_required"] == 2
        assert admin["dashboard_user_counts"]["identity_probe_available"] == 2
        assert admin["dashboard_user_counts"]["identity_probe_missing"] == 0
        assert admin["dashboard_user_counts"]["identity_probe_missing_users"] == []
        assert admin["dashboard_user_counts"]["waiting_datasets"] == 1
        assert admin["dashboard_user_counts"]["dataset_open_tasks"] == 1
        assert admin["dashboard_user_counts"]["users_with_waiting_datasets"] == 1
        assert admin["dashboard_user_counts"]["dataset_queue_states"] == {
            "all_assigned_work_complete": 1,
            "has_open_dataset_work": 1,
        }
        assert admin["dashboard_user_counts"]["dataset_queue_blocked_start_users"] == ["bob"]
        assert admin["dashboard_user_counts"]["dataset_queue_preview_users"] == ["alice", "bob"]
        assert admin["dashboard_user_counts"]["personalized_dataset_queue_preview_users"] == [
            "alice",
            "bob",
        ]
        assert admin["dashboard_user_counts"]["canonical_dataset_queue_preview_users"] == [
            "alice",
            "bob",
        ]
        assert admin["dashboard_user_counts"][
            "missing_personalized_dataset_queue_preview_users"
        ] == []
        assert admin["dashboard_user_counts"][
            "all_users_have_personalized_dataset_queue_preview"
        ] is True
        assert "preferred_personal_queue_match_users" in admin["dashboard_user_counts"]
        assert "missing_preferred_personal_queue_match_users" in admin["dashboard_user_counts"]
        assert "all_users_have_preferred_personal_queue_match" in admin["dashboard_user_counts"]
        assert "personalized_personal_queue_match_users" in admin["dashboard_user_counts"]
        assert "missing_personalized_personal_queue_match_users" in admin["dashboard_user_counts"]
        assert "all_users_have_personalized_personal_queue_match" in admin["dashboard_user_counts"]
        assert admin["dashboard_user_counts"]["dataset_queue_preferred_entrypoint_counts"] == {
            "personal_datasets_waiting_queue": 2
        }
        assert admin["dashboard_user_counts"]["dataset_queue_link_role_counts"] == {
            "canonical_queue_fallback": 2,
        }
        expected_dataset_queue_start_readiness = {
            "schema": "palette.web_labeling_dataset_queue_start_readiness.v1",
            "gate_id": "dataset_queue_start_readiness",
            "status": "needs_review",
            "ready": False,
            "dataset_queue_states": {
                "all_assigned_work_complete": 1,
                "has_open_dataset_work": 1,
            },
            "dataset_queue_blocked_start_users": ["bob"],
            "blocked_start_user_count": 1,
            "operator_action": "Resolve blocked dataset queue states before inviting labelers; generate or reopen work if labeling should continue, or treat completed assignments as finished.",
        }
        for key, value in expected_dataset_queue_start_readiness.items():
            assert admin["dataset_queue_start_readiness"][key] == value
        dashboard_rows = {row["user"]: row for row in admin["dashboard_users"]}
        assert dashboard_rows["alice"]["ready_to_invite"] is False
        assert dashboard_rows["alice"]["ready_to_invite_legacy_semantics"] == (
            "row_readiness_not_safe_share_approval"
        )
        assert dashboard_rows["alice"]["ready_state"] == "not_ready_to_invite"
        assert dashboard_rows["alice"]["ready_row_state"] == "diagnostic_note"
        assert dashboard_rows["alice"]["ready_row_state_values"] == [
            "ready_row_draft",
            "diagnostic_note",
        ]
        assert dashboard_rows["alice"]["copy_intent_values"] == [
            "ready_row_draft",
            "diagnostic_note",
        ]
        assert dashboard_rows["alice"]["copy_intent"] == "diagnostic_note"
        assert dashboard_rows["alice"]["safe_share_gate"]["schema"] == "palette.web_labeling_safe_share_gate.v1"
        assert dashboard_rows["alice"]["safe_share_gate_id"] == "labeler_links_safe_to_share"
        assert dashboard_rows["alice"]["safe_share_ready_to_send_is_sufficient"] is False
        assert dashboard_rows["alice"]["safe_share_required_inspection_field"] == "labeler_links_safe_to_share"
        assert "disposable_zarr_mutation_smoke" in dashboard_rows["alice"][
            "safe_share_launch_blocking_evidence_gate_ids"
        ]
        assert dashboard_rows["alice"]["safe_share_checklist_gate_evidence_complete"] is False
        assert dashboard_rows["alice"]["safe_share_launch_blocking_missing_evidence_gate_ids"] == (
            dashboard_rows["alice"]["safe_share_launch_blocking_evidence_gate_ids"]
        )
        assert dashboard_rows["alice"]["labeler_landing_url"] == "/"
        assert dashboard_rows["alice"]["expected_user_labeler_landing_url"] == "/?expected_user=alice"
        assert dashboard_rows["alice"]["expected_user_identity_probe_url"] == "/identity?expected_user=alice"
        assert dashboard_rows["alice"]["expected_user_dataset_queue_url"] == "/datasets?expected_user=alice"
        assert dashboard_rows["alice"]["identity_probe_required"] is True
        assert dashboard_rows["alice"]["identity_probe_available"] is True
        assert dashboard_rows["alice"]["labeler_safety"]["dashboard_identity_check_required"] is True
        assert dashboard_rows["alice"]["labeler_safety"]["expected_user_guards"]["task_open_api"] == "task_open_user_mismatch"
        assert dashboard_rows["alice"]["labeler_safety"]["expected_user_guards"]["promotion_retry_api"] == "promotion_retry_user_mismatch"
        assert dashboard_rows["alice"]["labeler_safety"]["labeler_failed_promotion_retry_action"] == "operator_support_only"
        assert dashboard_rows["alice"]["labeler_safety"]["expected_user_guards"]["signed_task_link"] == "signed_link_user_mismatch"
        assert dashboard_rows["alice"]["dashboard_identity_check_required"] is True
        assert dashboard_rows["alice"]["browser_only"] is True
        assert dashboard_rows["alice"]["open_tasks"] == 1
        assert dashboard_rows["alice"]["waiting_datasets"] == 1
        assert dashboard_rows["alice"]["dataset_open_tasks"] == 1
        assert dashboard_rows["alice"]["dataset_queue_summary"]["waiting_dataset_count"] == 1
        assert dashboard_rows["alice"]["dataset_queue_state"]["code"] == "has_open_dataset_work"
        assert dashboard_rows["alice"]["dataset_queue_state_code"] == "has_open_dataset_work"
        assert dashboard_rows["alice"]["dataset_queue_blocks_labeler_start"] is False
        assert dashboard_rows["alice"]["expected_user_personal_dataset_queue_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert dashboard_rows["alice"]["dataset_queue_preview_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert dashboard_rows["alice"]["canonical_dataset_queue_preview_url"] == (
            "/datasets?expected_user=alice"
        )
        assert dashboard_rows["alice"]["preferred_labeler_entry_url"] == (
            "/my-datasets?expected_user=alice"
        )
        assert dashboard_rows["alice"]["personal_dataset_queue_link_role"] == (
            "preferred_queue"
        )
        assert dashboard_rows["alice"]["dataset_queue_link_role"] == (
            "canonical_queue_fallback"
        )
        assert dashboard_rows["alice"]["total_tasks"] == 1
        assert dashboard_rows["alice"]["complete_tasks"] == 0
        assert dashboard_rows["alice"]["completion_percent"] == 0.0
        assert dashboard_rows["alice"]["completion_state"] == "not_started"
        assert "operator_validation_pending" in dashboard_rows["alice"]["invitation_message"]
        assert "dashboard: /work?expected_user=alice" in dashboard_rows["alice"]["invitation_message"]
        assert "The operator must resolve this before sending" in dashboard_rows["alice"]["invitation_message"]
        assert "identity check: /identity?expected_user=alice" in dashboard_rows["alice"][
            "invitation_message"
        ]
        assert dashboard_rows["bob"]["ready_to_invite"] is False
        assert dashboard_rows["bob"]["ready_to_invite_legacy_semantics"] == (
            "row_readiness_not_safe_share_approval"
        )
        assert dashboard_rows["bob"]["ready_state"] == "not_ready_to_invite"
        assert dashboard_rows["bob"]["ready_row_state"] == "diagnostic_note"
        assert dashboard_rows["bob"]["ready_row_draft_bundle_schema"] == (
            "palette.web_labeling_ready_row_draft_bundle.v1"
        )
        assert dashboard_rows["bob"]["ready_row_state_values"] == [
            "ready_row_draft",
            "diagnostic_note",
        ]
        assert dashboard_rows["bob"]["copy_intent_values"] == [
            "ready_row_draft",
            "diagnostic_note",
        ]
        assert dashboard_rows["bob"]["ready_row_draft_requires_safe_share_inspection"] is True
        assert dashboard_rows["bob"]["copy_label"] == "Copy not-ready note"
        assert dashboard_rows["bob"]["copy_intent"] == "diagnostic_note"
        assert dashboard_rows["bob"]["invite_reasons"] == ["no_open_tasks"]
        assert dashboard_rows["bob"]["total_tasks"] == 1
        assert dashboard_rows["bob"]["complete_tasks"] == 1
        assert dashboard_rows["bob"]["completion_percent"] == 100.0
        assert dashboard_rows["bob"]["completion_state"] == "complete"
        assert dashboard_rows["bob"]["dataset_queue_state"]["code"] == "all_assigned_work_complete"
        assert dashboard_rows["bob"]["dataset_queue_state_code"] == "all_assigned_work_complete"
        assert dashboard_rows["bob"]["dataset_queue_blocks_labeler_start"] is True
        assert dashboard_rows["bob"]["invite_actions"] == [
            "Generate, import, or reopen browser-labeling tasks for this user's active recordings."
        ]
        assert dashboard_rows["bob"]["recordings_without_open_tasks_actions"] == [
            "Reopen a completed task only if more labeling work is required; otherwise treat the recording as finished."
        ]
        assert html_status == 200
        assert "Dashboard users" in html
        assert "Task-specific signed links" in html
        assert "Identity source" in html
        assert "Identity must match assignment user" in html
        assert "Identity probe" in html
        assert "Dataset queue page" in html
        assert "Supported browser workflows" in html
        assert "Identity probes" in html
        assert "Completion" in html
        assert "Completion states" in html
        assert "Queue start readiness" in html
        assert "blocked users" in html
        assert user_api_payload["admin_user"]["dataset_queue_state"]["code"] == "has_open_dataset_work"
        assert "Dataset queue state:" in user_html
        assert "has_open_dataset_work" in user_html
        assert "Active users" in html
        assert "Active workflows" in html
        assert "Stale users" in html
        assert "Stale workflows" in html
        assert "Audit summary" in html
        assert "Recent task events" in html
        assert "Assignment work states" in html
        assert "Task state controls" in html
        assert "/admin/tasks/" in html
        assert "Ready states" in html
        assert "Invite reasons" in html
        assert "Copy intents" in html
        assert "Invite actions" in html
        assert "<textarea readonly>" in html
        assert "Copy ready-row draft" in html
        assert "Copy not-ready note" in html
        assert "/admin/users/" in html
        assert "/admin/recordings/" in html
        assert "copyInvitation" in html
        assert "dashboardHref" in html
        assert "row.expected_user_labeler_landing_url" in html
        assert "Start here:" in html
        assert "Sign in as" in html
        assert "Confirm the dashboard shows you as" in html
        assert user_api_status == 200
        assert user_api_payload["ok"] is True
        assert user_api_payload["admin_user"]["user"] == "alice"
        assert user_api_payload["admin_user"]["dashboard_user"]["ready_to_invite"] is False
        assert user_api_payload["admin_user"]["dashboard_user"]["ready_state"] == "not_ready_to_invite"
        assert user_api_payload["admin_user"]["dashboard_user"]["copy_intent"] == "diagnostic_note"
        assert user_api_payload["admin_user"]["work"]["progress_summary"]["waiting_recording_count"] == 1
        assert user_api_payload["admin_user"]["expected_user_labeler_landing_url"] == "/?expected_user=alice"
        assert user_api_payload["admin_user"]["expected_user_dashboard_url"] == "/work?expected_user=alice"
        assert user_api_payload["admin_user"]["expected_user_dataset_queue_url"] == "/datasets?expected_user=alice"
        assert user_api_payload["admin_user"]["expected_user_identity_probe_url"] == "/identity?expected_user=alice"
        assert user_status == 200
        assert "Admin user: alice" in user_html
        assert "Expected-user landing page" in user_html
        assert "Expected-user dashboard" in user_html
        assert "Expected-user dataset queue" in user_html
        assert "Identity probe" in user_html
        assert "/identity?expected_user=alice" in user_html
        assert "/?expected_user=alice" in user_html
        assert "/datasets?expected_user=alice" in user_html
        assert "/work?expected_user=alice" in user_html
        assert "Copy intent" in user_html
        assert "diagnostic_note" in user_html
        assert "Copy not-ready note" in user_html
        assert "copyUserInvitation" in user_html
        assert "waiting recordings" in user_html
        assert "blocked/no-open recordings" in user_html
        assert "rec-a" in user_html
        assert "/admin/recordings/rec-a" in user_html
        assert "/admin/tasks/task-a" in user_html
        assert "Expected-user dataset queue" in recording_html
        assert "Expected-user landing page" in recording_html
        assert "/?expected_user=alice" in recording_html
        assert "/datasets?expected_user=alice" in recording_html
        assert recording_api_status == 200
        assert recording_api_payload["ok"] is True
        admin_recording = recording_api_payload["admin_recording"]
        assert admin_recording["recording_id"] == "rec-a"
        assert admin_recording["assignment"]["assignee_user"] == "alice"
        assert admin_recording["expected_user_labeler_landing_url"] == "/?expected_user=alice"
        assert admin_recording["expected_user_dashboard_url"] == "/work?expected_user=alice"
        assert admin_recording["expected_user_dataset_queue_url"] == "/datasets?expected_user=alice"
        assert admin_recording["single_owner_policy"]["one_active_owner"] is True
        assert admin_recording["single_owner_policy"]["reassignment_replaces_owner"] is True
        assert admin_recording["task_counts"]["open_tasks"] == 1
        assert admin_recording["task_counts"]["complete_tasks"] == 0
        assert admin_recording["active_session_count"] == 1
        assert admin_recording["recent_event_count"] == 2
        assert admin_recording["tasks"][0]["task_id"] == "task-a"
        assert "scope" not in admin_recording["tasks"][0]
        assert admin_recording["tasks"][0]["redacted_fields"] == ["scope"]
        assert "/secret/admin-recording.zarr" not in json.dumps(admin_recording)
        assert recording_status == 200
        assert "Admin recording: rec-a" in recording_html
        assert "Recording ownership is exclusive" in recording_html
        assert "/admin/users/alice" in recording_html
        assert "/work?expected_user=alice" in recording_html
        assert "/admin/tasks/task-a" in recording_html
        assert "save_keypoints" in recording_html
        assert "/secret/admin-recording.zarr" not in recording_html
    finally:
        store.close()


def test_admin_task_detail_includes_recent_audit_events(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_analysis")
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="promotion_failed",
            target={"source_frame_index": 12},
            after={"error": "promotion target missing"},
        )

        with _running_server(store, user="admin", admin_users=("admin",)) as base_url:
            status, html = _text_request(base_url, "/admin/tasks/task-a")

        assert status == 200
        assert "Admin task detail" in html
        assert "Operator task state" in html
        assert "task-state-form" in html
        assert "task_reopened" in html
        assert "Recent audit events" in html
        assert "promotion_failed" in html
        assert "promotion target missing" in html
    finally:
        store.close()


def test_admin_task_state_route_completes_and_reopens_task(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        with _running_server(store, user="admin", admin_users=("admin",)) as base_url:
            complete_status, complete_payload = _json_request(
                base_url,
                "/api/admin/tasks/task-a/state",
                method="POST",
                payload={"state": "complete"},
            )
            reopen_status, reopen_payload = _json_request(
                base_url,
                "/api/admin/tasks/task-a/state",
                method="POST",
                payload={"state": "pending"},
            )
        with _running_server(store, user="alice") as base_url:
            reopened_open_status, reopened_open_payload = _json_request(
                base_url,
                "/api/tasks/task-a/open",
                method="POST",
                payload={"expected_user": "alice"},
            )

        closed_session = store.get_session(lease.session_id)
        completed_events = store.list_events(task_id="task-a", event_type="task_completed")
        reopened_events = store.list_events(task_id="task-a", event_type="task_reopened")

        assert complete_status == 200
        assert complete_payload["ok"] is True
        assert complete_payload["previous_task"]["state"] == "pending"
        assert complete_payload["task"]["state"] == "complete"
        assert complete_payload["task_state_transition"] == {
            "task_id": "task-a",
            "recording_id": "rec-a",
            "previous_state": "pending",
            "new_state": "complete",
            "state_changed": True,
            "completed": True,
            "reopened": False,
        }
        assert complete_payload["closed_session_count"] == 1
        assert complete_payload["closed_session_ids"] == [lease.session_id]
        assert complete_payload["session_closure_events"][0]["event_type"] == "session_closed_by_task_completion"
        assert complete_payload["session_closure_events"][0]["task_id"] == "task-a"
        assert complete_payload["session_closure_events"][0]["recording_id"] == "rec-a"
        assert closed_session is not None
        assert closed_session["closed_at_utc"]
        assert reopen_status == 200
        assert reopen_payload["ok"] is True
        assert reopen_payload["previous_task"]["state"] == "complete"
        assert reopen_payload["task"]["state"] == "pending"
        assert reopen_payload["task_state_transition"]["reopened"] is True
        assert reopen_payload["closed_session_count"] == 0
        assert len(completed_events) == 1
        assert len(reopened_events) == 1
        assert reopened_events[0]["user"] == "admin"
        assert reopened_open_status == 200
        assert reopened_open_payload["ok"] is True
        reopened_contract = reopened_open_payload["task_open_authorization_contract"]
        assert reopened_contract["ready"] is True
        assert reopened_contract["expected_user_guard_checked_server_side"] is True
        assert reopened_contract["expected_user_matches_resolved_user"] is True
        assert reopened_contract["active_assignment_present"] is True
        assert reopened_contract["task_assigned_to_resolved_user"] is True
        assert reopened_contract["assignment_status_active"] is True
        assert reopened_contract["task_state_startable"] is True
        assert reopened_contract["session_created_server_side"] is True
        assert reopened_contract["client_authorizes_open"] is False
        assert reopened_contract["server_authorizes_open"] is True
        assert reopened_contract["browser_label_write_target"] == "training_zarr"
        assert reopened_contract["browser_writes_csv_or_handoff_files"] is False
        assert reopened_contract["browser_has_direct_zarr_write_authority"] is False
        assert reopened_open_payload["authorization_context"]["resolved_user"] == "alice"
        assert reopened_open_payload["authorization_context"]["expected_user"] == "alice"
        assert reopened_open_payload["session"]["task_id"] == "task-a"
        assert reopened_open_payload["session"]["user"] == "alice"
    finally:
        store.close()


def test_admin_task_repair_route_reopens_task_and_closes_sessions(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        store.conn.execute(
            "UPDATE labeling_tasks SET state = 'complete' WHERE task_id = ?",
            ("task-a",),
        )
        store.conn.commit()

        with _running_server(store, user="alice", admin_users=("admin",)) as base_url:
            non_admin_status, non_admin_payload = _json_request(
                base_url,
                "/api/admin/tasks/task-a/repair",
                method="POST",
                payload={"reason": "non-admin attempt"},
            )
        with _running_server(store, user="admin", admin_users=("admin",)) as base_url:
            status, payload = _json_request(
                base_url,
                "/api/admin/tasks/task-a/repair",
                method="POST",
                payload={"reason": "admin recovery view after repair"},
            )

        closed_session = store.get_session(lease.session_id)
        repair_events = store.list_events(task_id="task-a", event_type="task_operator_repaired")
        reopened_events = store.list_events(task_id="task-a", event_type="task_reopened")

        assert non_admin_status == 403
        assert non_admin_payload["error"] == "admin_required"
        assert status == 200
        assert payload["ok"] is True
        assert payload["previous_task"]["state"] == "complete"
        assert payload["task"]["state"] == "pending"
        assert payload["operator_repair"]["task_id"] == "task-a"
        assert payload["operator_repair"]["recording_id"] == "rec-a"
        assert payload["operator_repair"]["previous_state"] == "complete"
        assert payload["operator_repair"]["new_state"] == "pending"
        assert payload["operator_repair"]["state_changed"] is True
        assert payload["operator_repair"]["closed_session_count"] == 1
        assert payload["operator_repair"]["closed_session_ids"] == [lease.session_id]
        assert payload["operator_repair"]["reason"] == "admin recovery view after repair"
        assert payload["operator_repair"]["event_type"] == "task_operator_repaired"
        assert payload["closed_session_count"] == 1
        assert payload["closed_session_ids"] == [lease.session_id]
        assert payload["session_closure_events"][0]["event_type"] == "session_closed_by_operator_repair"
        assert payload["session_closure_events"][0]["task_id"] == "task-a"
        assert payload["session_closure_events"][0]["recording_id"] == "rec-a"
        assert closed_session is not None
        assert closed_session["closed_at_utc"]
        assert len(repair_events) == 1
        assert repair_events[0]["user"] == "admin"
        assert repair_events[0]["after"]["reason"] == "admin recovery view after repair"
        assert len(reopened_events) == 1
    finally:
        store.close()


def test_keypoint_state_route_uses_cached_runtime_without_real_zarr(tmp_path, monkeypatch):
    _fake_module(
        monkeypatch,
        "fisheye.tune.keypoint_review_backend",
        review_session_summary=lambda session: {"summary": "ok", "zarr_path": "/tmp/fake.zarr"},
    )
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        def configure(state):
            review_session = SimpleNamespace(
                failures=np.asarray([0], dtype=np.int32),
                frame_indices=np.asarray([42], dtype=np.int32),
                refined=SimpleNamespace(attrs={}),
                zarr_path="/tmp/fake.zarr",
                refined_run="refined-a",
                crop_run="crop-a",
                keypoint_labels=("snout",),
            )
            state.keypoint_sessions[lease.session_id] = labeling_web.KeypointRuntimeSession(
                session_id=lease.session_id,
                task_id="task-a",
                recording_id="rec-a",
                user="alice",
                review_session=review_session,
            )

        with _running_server(store, user="alice", configure_state=configure) as base_url:
            page_status, page_html = _text_request(base_url, f"/r/{lease.session_id}")
            status, payload = _json_request(base_url, f"/api/sessions/{lease.session_id}/keypoints/state")

        assert page_status == 200
        assert "What to send the operator" in page_html
        assert "session_request_failed" in page_html
        assert "task_complete_failed" in page_html
        assert "postCompletionQueueUrl" in page_html
        assert "handleTaskCompletionSuccess(data)" in page_html
        assert "post_completion_queue" in page_html
        assert "expected_user_personal_dataset_queue_url" in page_html
        assert "setMutationSupportReference(result, mutation" in page_html
        assert "browser_label_write_target=" in page_html
        assert "browser_writes_csv_or_handoff_files=" in page_html
        assert "browser_has_direct_zarr_write_authority=" in page_html
        assert "return_personal_dataset_queue_url=" in page_html
        assert "return_personal_dataset_queue_expected_user_guarded=" in page_html
        assert "return_personal_work_url=" in page_html
        assert "return_personal_work_expected_user_guarded=" in page_html
        assert "window.location.href = nextUrl" in page_html
        assert "Copy support details" in page_html
        assert "Personalized dataset queue" in page_html
        assert 'href="/my-datasets?expected_user=alice"' in page_html
        assert 'data-session-return="dataset-queue"' in page_html
        assert "Personalized work dashboard" in page_html
        assert 'href="/my-work?expected_user=alice"' in page_html
        assert 'data-session-return="work-dashboard"' in page_html
        assert "sessionReturnHref" in page_html
        assert "session_closure_event" in page_html
        assert status == 200
        assert payload["ok"] is True
        assert payload["state"]["task_id"] == "task-a"
        assert payload["state"]["total"] == 1
        assert payload["state"]["current"]["frame_idx"] == 42
        assert "zarr_path" not in payload["state"]
        assert "zarr_path" not in payload["state"]["summary"]
        assert "/tmp/fake.zarr" not in json.dumps(payload)
    finally:
        store.close()


def test_detect_state_route_uses_cached_runtime_without_real_zarr(tmp_path, monkeypatch):
    _fake_module(
        monkeypatch,
        "fisheye.tune.detect_review_backend",
        review_session_summary=lambda session: {"summary": "ok"},
        load_frame_payload=lambda session, position=0: {"row_idx": 0, "frame_idx": 33},
    )
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_training")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        def configure(state):
            review_session = SimpleNamespace(
                review_rows=np.zeros((1, 2), dtype=np.float32),
                zarr_path="/tmp/fake.zarr",
                refined_run_name="refined-detect",
                width=640,
                height=480,
            )
            state.detect_sessions[lease.session_id] = labeling_web.DetectRuntimeSession(
                session_id=lease.session_id,
                task_id="task-a",
                recording_id="rec-a",
                user="alice",
                review_session=review_session,
            )

        with _running_server(store, user="alice", configure_state=configure) as base_url:
            status, payload = _json_request(base_url, f"/api/sessions/{lease.session_id}/detect/state")

        assert status == 200
        assert payload["ok"] is True
        assert payload["state"]["total"] == 1
        assert payload["state"]["current"]["frame_idx"] == 33
        assert "zarr_path" not in payload["state"]
        assert "/tmp/fake.zarr" not in json.dumps(payload)
    finally:
        store.close()


def test_detect_analysis_state_route_uses_cached_runtime_without_real_zarr(tmp_path, monkeypatch):
    _fake_module(
        monkeypatch,
        "fisheye.tune.video_detect_review_backend",
        review_session_summary=lambda session: {"summary": "ok"},
        video_sources_payload=lambda session: [{"video_id": "source", "path": "/tmp/source.mp4"}],
    )
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_analysis")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        def configure(state):
            review_session = SimpleNamespace(
                zarr_path="/tmp/fake.zarr",
                mode="traditional",
                collection_id="collection-a",
            )
            state.video_detect_sessions[lease.session_id] = labeling_web.VideoDetectRuntimeSession(
                session_id=lease.session_id,
                task_id="task-a",
                recording_id="rec-a",
                user="alice",
                review_session=review_session,
                frame_indices=np.asarray([17], dtype=np.int32),
                promotion=labeling_web.DetectAnalysisPromotionConfig(training_zarr="/tmp/training.zarr"),
            )

        with _running_server(store, user="alice", configure_state=configure) as base_url:
            status, payload = _json_request(base_url, f"/api/sessions/{lease.session_id}/detect-analysis/state")

        assert status == 200
        assert payload["ok"] is True
        assert payload["state"]["parent_frame_index"] == 17
        assert payload["state"]["videos"][0]["media_url"].endswith("/detect-analysis/media/source")
        assert "zarr_path" not in payload["state"]
        assert "training_zarr" not in payload["state"]["promotion"]
        assert "path" not in payload["state"]["videos"][0]
        assert "/tmp/fake.zarr" not in json.dumps(payload)
        assert "/tmp/training.zarr" not in json.dumps(payload)
    finally:
        store.close()


def test_subject_mask_state_route_uses_cached_runtime_without_real_zarr(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="subject_mask_component",
            component_name="body",
        )
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        def configure(state):
            source = SimpleNamespace(frame_indices=np.asarray([88], dtype=np.int32), run_name="subject-a")
            refined = SimpleNamespace(
                group=SimpleNamespace(attrs={}),
                run_name="refined-subject-a",
                component_names=("body",),
            )
            state.subject_mask_sessions[lease.session_id] = labeling_web.SubjectMaskRuntimeSession(
                session_id=lease.session_id,
                task_id="task-a",
                recording_id="rec-a",
                user="alice",
                zarr_path="/tmp/fake.zarr",
                root=SimpleNamespace(),
                source=source,
                refined=refined,
                roi_images=SimpleNamespace(),
                component_name="body",
                comp_idx=0,
                roi_indices=np.asarray([0], dtype=np.int32),
            )

        with _running_server(store, user="alice", configure_state=configure) as base_url:
            status, payload = _json_request(base_url, f"/api/sessions/{lease.session_id}/subject-mask/state")

        assert status == 200
        assert payload["ok"] is True
        assert payload["state"]["component_name"] == "body"
        assert payload["state"]["current"]["frame_idx"] == 88
        assert "zarr_path" not in payload["state"]
        assert "/tmp/fake.zarr" not in json.dumps(payload)
    finally:
        store.close()


def test_keypoint_nav_and_save_routes_record_audit_event_without_real_zarr(tmp_path, monkeypatch):
    _fake_module(
        monkeypatch,
        "fisheye.tune.keypoint_review_backend",
        review_session_summary=lambda session: {"summary": "ok"},
        load_roi_payload=lambda session, position=0: {
            "roi_idx": int(position),
            "frame_idx": int(session.frame_indices[int(position)]),
            "points": [[1.0, 2.0]],
            "reason": "needs_review",
            "status": "pending",
        },
        save_roi_correction=lambda session, position=0, points=None: {
            "roi_idx": int(position),
            "frame_idx": int(session.frame_indices[int(position)]),
            "changed": True,
            "reason_updated": True,
            "readback": {"points": points, "source_path": "/tmp/keypoint-source.json"},
            "zarr_path": "/tmp/fake.zarr",
        },
    )
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        def configure(state):
            review_session = SimpleNamespace(
                failures=np.asarray([0, 1], dtype=np.int32),
                frame_indices=np.asarray([42, 43], dtype=np.int32),
                refined=SimpleNamespace(attrs={}),
                zarr_path="/tmp/fake.zarr",
                refined_run="refined-a",
                crop_run="crop-a",
                keypoint_labels=("snout",),
            )
            state.keypoint_sessions[lease.session_id] = labeling_web.KeypointRuntimeSession(
                session_id=lease.session_id,
                task_id="task-a",
                recording_id="rec-a",
                user="alice",
                review_session=review_session,
            )

        with _running_server(store, user="alice", configure_state=configure) as base_url:
            nav_status, nav_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/nav",
                method="POST",
                payload={"delta": 1},
            )
            nav_oob_status, nav_oob_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/nav",
                method="POST",
                payload={"position": 2},
            )
            selector_status, selector_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/save",
                method="POST",
                payload={
                    "position": 0,
                    "target_zarr": "/tmp/should-not-be-used.zarr",
                    "csv_path": "/tmp/should-not-be-used.csv",
                    "data_plane_write_target": "handoff_csv",
                    "points": [[9.0, 9.0]],
                },
            )
            missing_token_status, missing_token_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/save",
                method="POST",
                payload={"points": [[9.0, 9.0]]},
            )
            reposition_status, reposition_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/nav",
                method="POST",
                payload={"position": 0},
            )
            stale_token_status, stale_token_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/save",
                method="POST",
                payload={
                    "points": [[9.0, 9.0]],
                    "target_token": nav_payload["state"]["target_token"],
                },
            )
            restore_status, restore_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/nav",
                method="POST",
                payload={"position": 1},
            )
            save_status, save_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/save",
                method="POST",
                payload={
                    "points": [[5.0, 6.0]],
                    "target_token": restore_payload["state"]["target_token"],
                },
            )

        events = store.list_events(event_type="save_keypoints", limit=10)

        assert nav_status == 200
        assert nav_payload["state"]["position"] == 1
        assert nav_oob_status == 400
        assert nav_oob_payload["error"] == "nav_error"
        assert "outside task scope" in nav_oob_payload["details"]
        assert selector_status == 400
        assert selector_payload["error"] == "payload_validation"
        assert "server-owned" in selector_payload["details"]
        assert "position" in selector_payload["details"]
        assert "target_zarr" in selector_payload["details"]
        assert "csv_path" in selector_payload["details"]
        assert "data_plane_write_target" in selector_payload["details"]
        selector_contract = selector_payload["mutation_authorization_contract"]
        assert selector_contract["ready"] is False
        assert selector_contract["not_ready_reason"] == "browser_mutation_target_selector_rejected"
        assert selector_contract["browser_supplied_zarr_or_csv_target_allowed"] is False
        assert selector_contract["browser_supplied_target_selectors_allowed"] is False
        assert selector_contract["browser_supplied_target_selectors_result"] == "rejected"
        assert selector_contract["current_target_token_result"] == "not_checked"
        assert selector_contract["server_authorizes_mutation"] is False
        assert selector_payload["browser_label_write_target"] == "training_zarr"
        assert selector_payload["browser_writes_csv_or_handoff_files"] is False
        assert selector_payload["browser_has_direct_zarr_write_authority"] is False
        assert missing_token_status == 400
        assert missing_token_payload["error"] == "payload_validation"
        assert "target_token" in missing_token_payload["details"]
        missing_token_contract = missing_token_payload["mutation_authorization_contract"]
        assert missing_token_contract["ready"] is False
        assert missing_token_contract["not_ready_reason"] == "target_token_failed"
        assert missing_token_contract["browser_supplied_target_selectors_result"] == "passed"
        assert missing_token_contract["current_target_token_result"] == "failed"
        assert missing_token_contract["server_authorizes_mutation"] is False
        assert missing_token_payload["browser_label_write_target"] == "training_zarr"
        assert missing_token_payload["browser_writes_csv_or_handoff_files"] is False
        assert missing_token_payload["browser_has_direct_zarr_write_authority"] is False
        assert reposition_status == 200
        assert reposition_payload["state"]["position"] == 0
        assert stale_token_status == 400
        assert stale_token_payload["error"] == "payload_validation"
        assert "target_token" in stale_token_payload["details"]
        stale_token_contract = stale_token_payload["mutation_authorization_contract"]
        assert stale_token_contract["ready"] is False
        assert stale_token_contract["not_ready_reason"] == "target_token_failed"
        assert stale_token_contract["browser_supplied_target_selectors_result"] == "passed"
        assert stale_token_contract["current_target_token_result"] == "failed"
        assert stale_token_contract["server_authorizes_mutation"] is False
        assert stale_token_payload["browser_label_write_target"] == "training_zarr"
        assert stale_token_payload["browser_writes_csv_or_handoff_files"] is False
        assert stale_token_payload["browser_has_direct_zarr_write_authority"] is False
        assert restore_status == 200
        assert restore_payload["state"]["position"] == 1
        assert save_status == 200
        assert save_payload["ok"] is True
        assert save_payload["result"]["changed"] is True
        assert "zarr_path" not in save_payload["result"]
        assert "source_path" not in save_payload["result"]["readback"]
        assert "/tmp/fake.zarr" not in json.dumps(save_payload)
        assert "/tmp/keypoint-source.json" not in json.dumps(save_payload)
        assert len(events) == 1
        _assert_mutation_event_provenance(
            events[0],
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_keypoints",
        )
        _assert_save_response_mutation_contract(
            save_payload,
            events[0],
            workflow_kind="keypoints",
            task_id="task-a",
            recording_id="rec-a",
            event_type="save_keypoints",
        )
        assert events[0]["target"]["frame_idx"] == 43
    finally:
        store.close()


def test_session_save_blocks_when_operator_validation_gate_pending(tmp_path, monkeypatch):
    _fake_module(
        monkeypatch,
        "fisheye.tune.keypoint_review_backend",
        review_session_summary=lambda session: {"summary": "ok"},
        load_roi_payload=lambda session, position=0: {
            "roi_idx": int(position),
            "frame_idx": int(session.frame_indices[int(position)]),
            "points": [[1.0, 2.0]],
        },
        save_roi_correction=lambda session, position=0, points=None: {
            "roi_idx": int(position),
            "frame_idx": int(session.frame_indices[int(position)]),
            "changed": True,
            "readback": {"points": points},
        },
    )
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    checklist_path = tmp_path / "validation-checklist.json"
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        checklist_path.write_text(
            json.dumps(
                {
                    "schema": "palette.web_labeling_validation_checklist.v1",
                    "all_validation_complete": False,
                    "ready_for_operator_validation": True,
                    "gates": [
                        {
                            "id": "browser_smoke",
                            "title": "Browser smoke",
                            "status": "pending_operator_evidence",
                            "required": True,
                            "blocks_invitation": True,
                            "evidence": [],
                            "evidence_files": [],
                        }
                    ],
                }
            )
        )

        def configure(state):
            review_session = SimpleNamespace(
                failures=np.asarray([0], dtype=np.int32),
                frame_indices=np.asarray([42], dtype=np.int32),
                refined=SimpleNamespace(attrs={}),
                zarr_path="/tmp/fake.zarr",
                refined_run="refined-a",
                crop_run="crop-a",
                keypoint_labels=("snout",),
            )
            state.keypoint_sessions[lease.session_id] = labeling_web.KeypointRuntimeSession(
                session_id=lease.session_id,
                task_id="task-a",
                recording_id="rec-a",
                user="alice",
                review_session=review_session,
            )

        with _running_server(
            store,
            user="alice",
            validation_checklist_path=checklist_path,
            require_operator_validation_for_start=True,
            configure_state=configure,
        ) as base_url:
            state_status, state_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/state",
            )
            save_status, save_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/save",
                method="POST",
                payload={
                    "points": [[5.0, 6.0]],
                    "target_token": state_payload["state"]["target_token"],
                },
            )

        events = store.list_events(event_type="save_keypoints", limit=10)
        contract = save_payload["mutation_authorization_contract"]
        gate = contract["operator_validation_mutation_gate"]

        assert state_status == 200
        assert save_status == 409
        assert save_payload["ok"] is False
        assert save_payload["error"] == "operator_validation_mutation_blocked"
        assert contract["ready"] is False
        assert contract["not_ready_reason"] == "operator_validation_mutation_blocked"
        assert contract["current_target_token_result"] == "not_checked"
        assert contract["server_authorizes_mutation"] is False
        assert contract["operator_validation_mutation_gate_required"] is True
        assert contract["operator_validation_mutation_gate_ready"] is False
        assert contract["operator_validation_mutation_gate_blocks_browser_mutation"] is True
        assert gate["required_for_browser_mutation"] is True
        assert gate["blocks_browser_mutation"] is True
        assert gate["operator_validation_pending_gate_ids"] == ["browser_smoke"]
        assert gate["operator_validation_required_missing_evidence_gate_ids"] == [
            "browser_smoke"
        ]
        assert "validation_checklist_path" not in gate
        assert save_payload["operator_validation_mutation_gate"] == gate
        assert save_payload["operator_validation_mutation_gate_checked_server_side"] is True
        assert save_payload["operator_validation_mutation_gate_required"] is True
        assert save_payload["operator_validation_mutation_gate_ready"] is False
        assert save_payload["operator_validation_mutation_gate_blocks_browser_mutation"] is True
        assert save_payload["operator_validation_mutation_gate_not_ready_reason"] == gate[
            "not_ready_reason"
        ]
        assert save_payload["operator_validation_mutation_gate_not_ready_reason"]
        assert save_payload["data_plane_write_target"] == (
            "server_owned_assigned_task_zarr_scope"
        )
        assert save_payload["label_mutation_target_kind"] == "task_scoped_training_zarr"
        assert save_payload["mutable_label_data_plane"] == "task_scoped_training_zarr"
        assert save_payload["browser_label_write_target"] == "training_zarr"
        assert save_payload["server_mutates_task_scoped_zarr_targets"] is True
        assert save_payload["training_zarr_mutations_are_server_owned"] is True
        assert save_payload["promotion_training_zarr_requires_task_scope"] is True
        assert save_payload["handoff_artifacts_are_metadata_only"] is True
        assert save_payload["csv_handoff_artifact_role"] == "metadata_only_control_plane"
        assert save_payload["csv_handoff_artifacts_are_label_write_targets"] is False
        assert save_payload["handoff_csv_artifacts_are_label_write_targets"] is False
        assert save_payload["intermediate_csv_artifacts_are_label_write_targets"] is False
        assert save_payload["browser_writes_csv_or_handoff_files"] is False
        assert save_payload["browser_writes_handoff_csv"] is False
        assert save_payload["browser_writes_intermediate_csv"] is False
        assert save_payload["browser_receives_zarr_write_authority"] is False
        assert save_payload["browser_has_direct_zarr_write_authority"] is False
        assert events == []
    finally:
        store.close()


def test_session_save_rechecks_current_assignment_after_reassignment(tmp_path, monkeypatch):
    _fake_module(
        monkeypatch,
        "fisheye.tune.keypoint_review_backend",
        review_session_summary=lambda session: {"summary": "ok"},
        load_roi_payload=lambda session, position=0: {
            "roi_idx": int(position),
            "frame_idx": int(session.frame_indices[int(position)]),
            "points": [[1.0, 2.0]],
        },
        save_roi_correction=lambda session, position=0, points=None: {
            "roi_idx": int(position),
            "frame_idx": int(session.frame_indices[int(position)]),
            "changed": True,
            "readback": {"points": points},
        },
    )
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        store.assign_recording(recording_id="rec-a", assignee_user="bob", allow_stale_open_sessions=True)

        def configure(state):
            review_session = SimpleNamespace(
                failures=np.asarray([0], dtype=np.int32),
                frame_indices=np.asarray([42], dtype=np.int32),
                refined=SimpleNamespace(attrs={}),
                zarr_path="/tmp/fake.zarr",
                refined_run="refined-a",
                crop_run="crop-a",
                keypoint_labels=("snout",),
            )
            state.keypoint_sessions[lease.session_id] = labeling_web.KeypointRuntimeSession(
                session_id=lease.session_id,
                task_id="task-a",
                recording_id="rec-a",
                user="alice",
                review_session=review_session,
            )

        with _running_server(store, user="alice", configure_state=configure) as base_url:
            status, payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/keypoints/save",
                method="POST",
                payload={"points": [[5.0, 6.0]]},
            )

        events = store.list_events(event_type="save_keypoints", limit=10)

        assert status == 403
        assert payload["ok"] is False
        assert payload["error"] == "not_assigned"
        assert events == []
    finally:
        store.close()


def test_detect_nav_and_save_routes_record_audit_event_without_real_zarr(tmp_path, monkeypatch):
    _fake_module(
        monkeypatch,
        "fisheye.tune.detect_review_backend",
        review_session_summary=lambda session: {"summary": "ok"},
        load_frame_payload=lambda session, position=0: {
            "row_idx": int(position),
            "frame_idx": int(session.frame_indices[int(position)]),
            "bbox_norm": [0.1, 0.2, 0.3, 0.4],
            "status": "pending",
        },
        apply_manual_edit=lambda session, position=0, bbox_norm=None: {
            "row_idx": int(position),
            "frame_idx": int(session.frame_indices[int(position)]),
            "action": "update",
            "bbox_norm": bbox_norm,
            "status": "reviewed",
            "target_zarr": "/tmp/fake.zarr",
            "source_path": "/tmp/detect-source.json",
        },
    )
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_training")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        def configure(state):
            review_session = SimpleNamespace(
                review_rows=np.zeros((2, 2), dtype=np.float32),
                frame_indices=np.asarray([33, 34], dtype=np.int32),
                zarr_path="/tmp/fake.zarr",
                refined_run_name="refined-detect",
                width=640,
                height=480,
            )
            state.detect_sessions[lease.session_id] = labeling_web.DetectRuntimeSession(
                session_id=lease.session_id,
                task_id="task-a",
                recording_id="rec-a",
                user="alice",
                review_session=review_session,
            )

        with _running_server(store, user="alice", configure_state=configure) as base_url:
            nav_status, nav_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/detect/nav",
                method="POST",
                payload={"position": 1},
            )
            nav_oob_status, nav_oob_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/detect/nav",
                method="POST",
                payload={"position": 2},
            )
            selector_status, selector_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/detect/save",
                method="POST",
                payload={
                    "row_idx": 0,
                    "target_zarr": "/tmp/should-not-be-used.zarr",
                    "csv_path": "/tmp/should-not-be-used.csv",
                    "data_plane_write_target": "handoff_csv",
                    "bbox_norm": [0.5, 0.6, 0.7, 0.8],
                },
            )
            save_status, save_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/detect/save",
                method="POST",
                payload={
                    "bbox_norm": [0.2, 0.3, 0.4, 0.5],
                    "target_token": nav_payload["state"]["target_token"],
                },
            )

        events = store.list_events(event_type="save_detect_bbox", limit=10)

        assert nav_status == 200
        assert nav_payload["state"]["position"] == 1
        assert nav_oob_status == 400
        assert nav_oob_payload["error"] == "nav_error"
        assert "outside task scope" in nav_oob_payload["details"]
        assert selector_status == 400
        assert selector_payload["error"] == "payload_validation"
        assert "server-owned" in selector_payload["details"]
        assert "row_idx" in selector_payload["details"]
        assert "target_zarr" in selector_payload["details"]
        assert "csv_path" in selector_payload["details"]
        assert "data_plane_write_target" in selector_payload["details"]
        assert save_status == 200
        assert save_payload["ok"] is True
        assert save_payload["result"]["status"] == "reviewed"
        assert "target_zarr" not in save_payload["result"]
        assert "source_path" not in save_payload["result"]
        assert "/tmp/fake.zarr" not in json.dumps(save_payload)
        assert "/tmp/detect-source.json" not in json.dumps(save_payload)
        assert len(events) == 1
        _assert_mutation_event_provenance(
            events[0],
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_detect_bbox",
        )
        _assert_save_response_mutation_contract(
            save_payload,
            events[0],
            workflow_kind="detect_training",
            task_id="task-a",
            recording_id="rec-a",
            event_type="save_detect_bbox",
        )
        assert events[0]["target"]["frame_idx"] == 34
    finally:
        store.close()


def test_detect_analysis_nav_and_save_routes_record_audit_event_without_real_zarr(tmp_path, monkeypatch):
    _fake_module(
        monkeypatch,
        "fisheye.tune.video_detect_review_backend",
        review_session_summary=lambda session: {"summary": "ok"},
        video_sources_payload=lambda session: [{"video_id": "source", "path": "/tmp/source.mp4"}],
        load_frame_payload=lambda session, parent_frame: {
            "parent_frame_index": int(parent_frame),
            "source_frame_index": int(parent_frame) + 100,
            "bbox_norm": [0.1, 0.2, 0.3, 0.4],
            "status": "pending",
            "video_id": "source",
            "refined_run_name": "analysis-refined",
            "refined_group_path": "refined_detect_runs/analysis-refined",
            "clip_id": "clip-a",
            "camera_serial": "camera-a",
        },
        apply_manual_edit=lambda session, parent_frame_index=0, bbox_norm=None: {
            "parent_frame_index": int(parent_frame_index),
            "source_frame_index": int(parent_frame_index) + 100,
            "action": "update",
            "bbox_norm": bbox_norm,
            "status": "reviewed",
            "analysis_zarr": "/tmp/fake-analysis.zarr",
            "refined_group_path": "refined_detect_runs/analysis-refined",
        },
    )
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_analysis")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        def configure(state):
            review_session = SimpleNamespace(
                zarr_path="/tmp/fake-analysis.zarr",
                mode="traditional",
                collection_id="collection-a",
            )
            state.video_detect_sessions[lease.session_id] = labeling_web.VideoDetectRuntimeSession(
                session_id=lease.session_id,
                task_id="task-a",
                recording_id="rec-a",
                user="alice",
                review_session=review_session,
                frame_indices=np.asarray([17, 18], dtype=np.int32),
                editable=True,
            )

        with _running_server(store, user="alice", configure_state=configure) as base_url:
            nav_status, nav_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/detect-analysis/nav",
                method="POST",
                payload={"position": 1},
            )
            nav_oob_status, nav_oob_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/detect-analysis/nav",
                method="POST",
                payload={"position": 2},
            )
            selector_status, selector_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/detect-analysis/save",
                method="POST",
                payload={
                    "parent_frame_index": 17,
                    "analysis_zarr": "/tmp/should-not-be-used-analysis.zarr",
                    "target_csv": "/tmp/should-not-be-used.csv",
                    "browser_label_write_target": "handoff_csv",
                    "bbox_norm": [0.5, 0.6, 0.7, 0.8],
                },
            )
            save_status, save_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/detect-analysis/save",
                method="POST",
                payload={
                    "bbox_norm": [0.2, 0.3, 0.4, 0.5],
                    "target_token": nav_payload["state"]["target_token"],
                },
            )

        events = store.list_events(event_type="save_detect_analysis_bbox", limit=10)

        assert nav_status == 200
        assert nav_payload["state"]["position"] == 1
        assert nav_oob_status == 400
        assert nav_oob_payload["error"] == "nav_error"
        assert "outside task scope" in nav_oob_payload["details"]
        assert selector_status == 400
        assert selector_payload["error"] == "payload_validation"
        assert "server-owned" in selector_payload["details"]
        assert "parent_frame_index" in selector_payload["details"]
        assert "analysis_zarr" in selector_payload["details"]
        assert "target_csv" in selector_payload["details"]
        assert "browser_label_write_target" in selector_payload["details"]
        assert save_status == 200
        assert save_payload["ok"] is True
        assert save_payload["result"]["source_frame_index"] == 118
        assert "analysis_zarr" not in save_payload["result"]
        assert "refined_group_path" not in save_payload["result"]
        assert "/tmp/fake-analysis.zarr" not in json.dumps(save_payload)
        assert len(events) == 1
        _assert_mutation_event_provenance(
            events[0],
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_detect_analysis_bbox",
        )
        _assert_save_response_mutation_contract(
            save_payload,
            events[0],
            workflow_kind="detect_analysis",
            task_id="task-a",
            recording_id="rec-a",
            event_type="save_detect_analysis_bbox",
        )
        assert events[0]["target"]["source_frame_index"] == 118
    finally:
        store.close()


def test_subject_mask_nav_and_save_routes_record_audit_event_without_real_zarr(tmp_path, monkeypatch):
    def save_refined_subject_roi(*, source, refined, roi_idx, edited_masks, component_names):
        refined.group["masks_roi"][roi_idx] = edited_masks

    _fake_module(
        monkeypatch,
        "fisheye.tune.refined_subject_mask_review",
        save_refined_subject_roi=save_refined_subject_roi,
    )

    class FakeGroup(dict):
        def __init__(self, *args, attrs=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.attrs = attrs or {}

    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="subject_mask_component",
            component_name="body",
        )
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        def configure(state):
            masks = np.zeros((2, 1, 3, 3), dtype=np.uint8)
            masks[1, 0, 0, 0] = 1
            group = FakeGroup({"masks_roi": masks}, attrs={})
            source = SimpleNamespace(frame_indices=np.asarray([88, 89], dtype=np.int32), run_name="subject-a")
            refined = SimpleNamespace(
                group=group,
                parent=SimpleNamespace(),
                run_name="refined-subject-a",
                component_names=("body",),
            )
            state.subject_mask_sessions[lease.session_id] = labeling_web.SubjectMaskRuntimeSession(
                session_id=lease.session_id,
                task_id="task-a",
                recording_id="rec-a",
                user="alice",
                zarr_path="/tmp/fake-subject.zarr",
                root=SimpleNamespace(),
                source=source,
                refined=refined,
                roi_images=SimpleNamespace(),
                component_name="body",
                comp_idx=0,
                roi_indices=np.asarray([0, 1], dtype=np.int32),
            )

        edited_mask = np.ones((3, 3), dtype=np.uint8)
        with _running_server(store, user="alice", configure_state=configure) as base_url:
            nav_status, nav_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/subject-mask/nav",
                method="POST",
                payload={"position": 1},
            )
            nav_oob_status, nav_oob_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/subject-mask/nav",
                method="POST",
                payload={"position": 2},
            )
            selector_status, selector_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/subject-mask/save",
                method="POST",
                payload={
                    "roi_idx": 0,
                    "training_zarr": "/tmp/should-not-be-used-training.zarr",
                    "intermediate_csv": "/tmp/should-not-be-used.csv",
                    "write_target": "csv",
                    "mask": labeling_web._raw_array_payload(edited_mask),
                },
            )
            save_status, save_payload = _json_request(
                base_url,
                f"/api/sessions/{lease.session_id}/subject-mask/save",
                method="POST",
                payload={
                    "mask": labeling_web._raw_array_payload(edited_mask),
                    "target_token": nav_payload["state"]["target_token"],
                },
            )

        events = store.list_events(event_type="save_subject_mask_roi", limit=10)

        assert nav_status == 200
        assert nav_payload["state"]["position"] == 1
        assert nav_oob_status == 400
        assert nav_oob_payload["error"] == "nav_error"
        assert "outside task scope" in nav_oob_payload["details"]
        assert selector_status == 400
        assert selector_payload["error"] == "payload_validation"
        assert "server-owned" in selector_payload["details"]
        assert "roi_idx" in selector_payload["details"]
        assert "training_zarr" in selector_payload["details"]
        assert "intermediate_csv" in selector_payload["details"]
        assert "write_target" in selector_payload["details"]
        assert save_status == 200
        assert save_payload["ok"] is True
        assert save_payload["result"]["frame_idx"] == 89
        assert save_payload["result"]["after_area_px"] == 9
        assert "/tmp/fake-subject.zarr" not in json.dumps(save_payload)
        assert len(events) == 1
        _assert_mutation_event_provenance(
            events[0],
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_subject_mask_roi",
        )
        _assert_save_response_mutation_contract(
            save_payload,
            events[0],
            workflow_kind="subject_mask_component",
            task_id="task-a",
            recording_id="rec-a",
            event_type="save_subject_mask_roi",
        )
        assert events[0]["target"]["component_name"] == "body"
    finally:
        store.close()
